"""Link TXT exports to decoded BIN streams by counter and both ECG fields.

This does not assign physical semantics to undocumented configuration fields.
It requires complete-record comparisons, rather than timestamp proximity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, irfft, next_fast_len


def read_txt_ecg(source) -> np.ndarray:
    # TXT has eleven header fields and a trailing empty twelfth data field.
    # Explicitly disable inferred row indexes so COUNT remains column zero.
    frame = pd.read_csv(source, sep='\t', decimal=',', usecols=[0, 1, 2], index_col=False)
    if list(frame.columns) != ['COUNT', 'EMG1mV', 'EMG2mV']:
        raise ValueError('Unexpected TXT ECG column names')
    data = frame.to_numpy(dtype=float)
    if not np.isfinite(data).all() or np.any(data[:, 0] != np.floor(data[:, 0])):
        raise ValueError('Nonfinite ECG or noninteger counter')
    if np.any((data[:, 0] < 0) | (data[:, 0] > 255)):
        raise ValueError('Counter outside byte range')
    return data


def fit_full(raw: np.ndarray, exported: np.ndarray) -> dict:
    raw = raw.astype(float)
    centered = raw - raw.mean()
    slope = float(centered @ (exported - exported.mean()) / (centered @ centered))
    offset = float(exported.mean() - slope * raw.mean())
    error = exported - (slope * raw + offset)
    return {'slope_mV_per_integer': slope, 'offset_mV': offset,
            'max_absolute_residual_mV': float(np.max(np.abs(error))),
            'rms_residual_mV': float(np.sqrt(np.mean(error**2))),
            'pearson_correlation': float(np.corrcoef(raw, exported)[0, 1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    out = experiment / 'derived/scorpion_bin_review'
    structure_path = out / 'binary_structure.local.json'
    structure = json.loads(structure_path.read_text(encoding='utf-8'))
    inventory = json.loads((experiment / 'metadata/inventory.local.json').read_text(encoding='utf-8'))
    exports = []
    for entry in inventory['files']:
        if entry.get('signal_input_candidate'):
            path = experiment / entry['raw_relative_path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'TXT hash differs from inventory: {path.name}')
            data = read_txt_ecg(path)
            if len(data) != entry['text_qc']['rows']:
                raise ValueError('TXT row count differs from inventory')
            exports.append((entry, data))
    matches = []
    diagnostics = []
    template_length = 1024
    for record in structure['files']:
        if not record.get('data_packets'):
            continue
        with np.load(out / f'{record["file_id"]}_decoded.local.npz') as a:
            decoded = {key: a[key] for key in a.files}
        x = decoded['ecg1_signed24'].astype(float)
        transform_size = next_fast_len(len(x) + template_length - 1)
        transformed = rfft(x, transform_size)
        sums = np.r_[0., np.cumsum(x)]
        squares = np.r_[0., np.cumsum(x*x)]
        window_sum = sums[template_length:] - sums[:-template_length]
        window_power = (squares[template_length:] - squares[:-template_length]
                        - window_sum**2 / template_length)
        window_power = np.maximum(window_power, 1e-20)
        for entry, exported in exports:
            template = exported[:template_length, 1]
            centered = template - template.mean()
            correlation = irfft(transformed * rfft(centered[::-1], transform_size), transform_size)
            scores = correlation[template_length-1:len(x)] / np.sqrt(window_power * (centered @ centered))
            valid = decoded['count'][:len(scores)] == int(exported[0, 0])
            scores[~valid] = 0
            candidates = np.argpartition(np.abs(scores), -8)[-8:]
            candidates = candidates[np.argsort(-np.abs(scores[candidates]))]
            diagnostic = {'bin_id': record['file_id'], 'record_id': entry['record_id'],
                          'highest_prefix_abs_correlation': float(abs(scores[candidates[0]]))}
            diagnostics.append(diagnostic)
            for start in candidates:
                if abs(scores[start]) < .98 or start + len(exported) > len(x):
                    continue
                end = int(start + len(exported))
                count_equal = bool(np.array_equal(decoded['count'][start:end], exported[:, 0]))
                if not count_equal:
                    continue
                ecg_fits = {str(ch): fit_full(decoded[f'ecg{ch}_signed24'][start:end], exported[:, ch])
                            for ch in (1, 2)}
                for ch in (1, 2):
                    residual = exported[:, ch] - decoded[f'ecg{ch}_signed24'][start:end] * .001
                    ecg_fits[str(ch)]['fixed_scale_0_001_max_absolute_residual_mV'] = float(np.max(np.abs(residual)))
                if max(fit['fixed_scale_0_001_max_absolute_residual_mV'] for fit in ecg_fits.values()) > .0050000002:
                    continue
                states, occurrences = np.unique(decoded['settings_state'][start:end], return_counts=True)
                controls = []
                for state, count in zip(states, occurrences):
                    control = next((v for v in record['control_variants'] if v['state_id'] == int(state)), None)
                    controls.append({'state_id': int(state), 'data_packets': int(count), 'control': control})
                match = {'record_id': entry['record_id'], 'txt_sha256': entry['sha256'],
                         'bin_id': record['file_id'], 'bin_name': record['source_name'],
                         'bin_sha256': record['sha256'], 'start_data_index': int(start),
                         'end_data_index_exclusive': end, 'matched_rows': len(exported),
                         'first_byte_offset': int(decoded['sample_byte_offsets'][start]),
                         'last_packet_byte_offset': int(decoded['sample_byte_offsets'][end-1]),
                         'counter_equal_all_rows': count_equal, 'ecg_full_record_fits': ecg_fits,
                         'settings_states_in_record': controls,
                         'match_status': 'whole_record_counter_and_two_ECG_fields_match_at_fixed_0_001_scale_within_export_rounding',
                         'settings_semantics_status': 'unconfirmed_device_protocol'}
                matches.append(match)
                print(entry['record_id'], record['file_id'], int(start), len(exported),
                      [v['control']['u16_at_2'] if v['control'] else None for v in controls], flush=True)
                break
        del decoded, x, transformed, sums, squares, window_sum, window_power
    matched_ids = {m['record_id'] for m in matches}
    result = {'status': 'local_empirical_link_not_device_protocol',
              'generated_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'binary_structure_sha256': hashlib.sha256(structure_path.read_bytes()).hexdigest(),
              'criteria': {'prefix_samples': template_length, 'prefix_abs_correlation_min': .98,
                           'all_rows_counter_equal': True,
                           'TXT_raw_files_read_directly_and_sha256_verified': True,
                           'ECG_empirical_fixed_scale_mV_per_BIN_integer': .001,
                           'two_ECG_all_rows_fixed_scale_max_residual_mV': .0050000002},
              'matches': matches,
              'unmatched_record_ids': [e['record_id'] for e, _ in exports if e['record_id'] not in matched_ids],
              'prefix_diagnostics': diagnostics}
    (out / 'txt_binary_matches.local.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'matching_source.local.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
