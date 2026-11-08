"""Пересчёт бокового BEI «Скорпиона» по индивидуальным однородным моделям."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scorpion_apparent_resistivity import ROOT, SUBJECTS, read, read_impedance, sha

sys.path.insert(0, str(ROOT / 'Colab Notebooks/scorpion_v2'))
from scorpion_qc import PARAMS, bei_filter, pair_match
from lateral_pilot import PHASE, candidate_windows

STATUS = 'exploratory_hypothesis_not_validated'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cycle_data(pulse, raw, peaks, bounds, fs):
    """Полные интервалы QRS–QRS и неизменённый размах после центрирования."""
    start, end = bounds
    require(0 <= start < end <= len(raw) == len(pulse), 'Некорректное окно вдоха')
    require(np.all(np.diff(peaks) > 0), 'Метки QRS должны строго возрастать')
    # Отмечаем только длительные повторы глобального экстремума BEI.
    plateaus = []
    for extreme in np.unique([np.min(raw), np.max(raw)]):
        mask = raw == extreme
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        plateaus.extend((int(a), int(b)) for a,b in zip(edges[::2], edges[1::2]) if b-a >= .1*fs)
    retained, excluded = [], []
    inner_peaks = peaks[(peaks >= start) & (peaks < end)]
    for left, right in zip(inner_peaks[:-1], inner_peaks[1:]):
        left, right = int(left), int(right)
        rr = (right - left) / fs
        reasons = []
        if not .45 <= rr <= 1.8:
            reasons.append('RR_outside_0.45_to_1.8_s')
        if any(left < b and right >= a for a, b in plateaus):
            reasons.append('raw_BEI_extreme_plateau_at_least_0.1_s')
        values = pulse[left:right+1]
        if not np.isfinite(values).all():
            reasons.append('nonfinite_pulse')
        cycle = dict(start_sample=left, stop_sample=right, rr_s=rr)
        if reasons:
            excluded.append(dict(**cycle, reasons=reasons))
            continue
        waveform = np.interp(PHASE * (right-left), np.arange(len(values)), values)
        cycle.update(centered_Z_ohm=(waveform - np.median(waveform)).tolist(),
                     native_peak_to_peak_Z_ohm=float(np.ptp(values[:-1])))
        retained.append(cycle)
    return retained, excluded, plateaus


def prepare(input_path, baseline_path, out):
    input_path, baseline_path = Path(input_path).resolve(), Path(baseline_path).resolve()
    base, inputs = read(baseline_path), read(input_path)
    require(inputs['schema'] == 'scorpion_uniform_rho_input_v1' and inputs['frequency_hz'] == 100000 and
            base['schema'] == 'scorpion_uniform_rho_results_v1' and base['input_sha256'] == sha(input_path),
            'Базовый расчёт относится к другому плану')
    require(tuple(s['subject_alias'] for s in inputs['subjects']) == SUBJECTS, 'Нужны три индивидуальные модели')
    baseline = {a['record_id']: a for a in base['answers']}
    records = []
    protected = {}

    def protect(path, expected=None):
        actual = sha(path)
        require(expected is None or actual == expected, f'Изменился вход {Path(path).name}')
        protected[str(path)] = actual
        return actual

    protect(input_path, base['input_sha256'])
    baseline_sha = protect(baseline_path)
    filter_sha = protect(Path(sys.modules['scorpion_qc'].__file__))
    for subject in inputs['subjects']:
        alias = subject['subject_alias']
        georg = alias == 'georg'
        model_input = Path(subject['model_directory']) / 'electrical_inputs.local.json'
        protect(model_input, subject['input_snapshot_sha256'])
        model = read(model_input)
        summary_path = Path(model['summary_source']['path'])
        protect(summary_path, subject['summary_sha256'])
        summary = read(summary_path)
        require(summary['code_sha256' if georg else 'filter_source_sha256'] == filter_sha,
                'Изменился код выделения пульсовой полосы')
        if georg:
            ann_path = summary_path.parents[2] / 'annotations/scorpion_review/annotations_candidates.local.json'
            protect(ann_path, protect(summary_path.parent / 'annotations_candidates.local.json'))
        else:
            ann_path = summary_path.parents[3] / 'annotations/session_1140/candidate_annotations.local.json'
            protect(ann_path, summary['annotations_sha256'])
        annotations = read(ann_path)['records']
        for o in subject['observations']:
            rid, fs, channel = o['record_id'], o['fs_hz'], o['channel']
            r, = [r for r in summary['records'] if r['record_id'] == rid and r.get('fs_hz_scenario',r.get('fs')) == fs]
            ann, = [a for a in annotations if a['record_id'] == rid and a.get('fs_hz_scenario',a.get('fs_hz')) == fs]
            require(r['source_sha256'] == ann['source_sha256'] == o['source_sha256'], f'Чужая запись {rid}')
            require(ann['status'] == 'pending_manual_review' and not o['manual_annotation_accepted'],
                    f'Изменился статус разметки {rid}')
            reference = r['selected_ecg_channel_candidate' if georg else 'ecg_reference']
            ann_reference = ann['selected_ecg_channel_candidate' if georg else 'ecg_reference_channel_candidate']
            require(reference == ann_reference and reference in (1,2), f'Нет опорного канала ЭКГ {rid}')
            n = r['rows' if georg else 'n']
            windows = candidate_windows(ann['breathing'], fs, n) if georg else ann['inner_windows_samples_half_open']
            require(windows.get('inhale') == o['window_samples_half_open'], f'Окно вдоха не совпадает {rid}')
            lateral = (ann['channel_mapping_constraints']['physical_channel_roles'][str(channel)].startswith('lateral')
                       if georg else r['physical_channel_roles']['lateral'] == channel)
            require(lateral, f'Выбран не боковой канал {rid}')
            cache = summary_path.parent / (r['key'] + '.npz' if georg else rid + '.local.npz')
            protect(cache, r.get('arrays_sha256'))
            protect(o['source_path'], o['source_sha256'])
            raw = read_impedance(o['source_path'], channel)
            pulse_key = 'bei_pulse_band_' if georg else 'pulse_'
            peak_key = 'qrs_samples_' if georg else 'qrs_'
            with np.load(cache) as arrays:
                require(np.array_equal(raw, arrays[f'{"bei_raw_" if georg else "raw_"}{channel}']), f'Кэш BEI отличается от TXT {rid}')
                pulse = arrays[f'{pulse_key}{channel}']
                peak_sets = [arrays[f'{peak_key}{ch}'] for ch in (1,2)]
            if georg:
                for ch, peaks in enumerate(peak_sets, 1):
                    require(np.array_equal(peaks, ann['qrs_samples_by_channel'][str(ch)]), f'Изменились метки QRS {rid}')
            start, end = o['window_samples_half_open']
            require(len(raw) == n and np.isfinite(raw).all() and abs(np.median(raw[start:end])-o['Z_ohm']) <= 1e-8,
                    f'Не воспроизведён базовый уровень {rid}')
            require(np.isfinite(pulse).all(), f'Нечисловой фильтрованный BEI {rid}')
            reproduced = bei_filter(raw, fs)[2]
            filter_error = float(np.max(np.abs(reproduced - pulse)))
            require(filter_error <= 1e-9, f'Не воспроизведена фильтрация BEI {rid}')
            cycles, rejected, plateaus = cycle_data(pulse, raw, peak_sets[reference-1], (start,end), fs)
            require(len(cycles) >= 3, f'Менее трёх пригодных циклов {rid}')
            in_window = [p[(p>=start)&(p<end)] for p in peak_sets]
            match = pair_match(*in_window, fs)
            a = baseline[rid]
            require(a['subject_alias'] == alias and a['case_id'] == o['case_id'] and a['observed_Z_ohm'] == o['Z_ohm'] and
                    abs(a['residual_ohm']) <= a['tolerance_ohm'], f'Чужой базовый результат {rid}')
            rho0, slope = a['rho_apparent_ohm_m'], a['dZ_dlogrho_ohm']
            require(np.isfinite([rho0,slope]).all() and min(rho0,slope)>0, f'Некорректная чувствительность {rid}')
            waves = np.asarray([c['centered_Z_ohm'] for c in cycles])
            factor = rho0 / slope
            amplitudes = np.ptp(waves,axis=1)
            med, q25, q75 = np.quantile(waves * factor,[.5,.25,.75],axis=0)
            quantiles = np.quantile(amplitudes * factor,[.25,.5,.75]).tolist()
            records.append(dict(**o, subject_alias=alias, rho0_ohm_m=rho0, dZ_dlogrho_ohm=slope,
                qrs_status=ann['status'], ecg_reference_channel=reference, in_window_QRS_pair_match=match,
                source_flags=r.get('flags',[]), filter_reproduction_max_error_ohm=filter_error,
                n_cycles=len(cycles), n_excluded=len(rejected), excluded_cycles=rejected,
                raw_BEI_extreme_plateaus=plateaus, cycles=cycles, max_abs_delta_Z_ohm=float(np.max(np.abs(waves))),
                delta_Z_cycle_median_ohm=float(np.median(amplitudes)), delta_rho_cycle_q25_median_q75_ohm_m=quantiles,
                delta_rho_median_wave_span_ohm_m=float(np.ptp(med)),
                median_wave_ohm_m=med.tolist(), q25_wave_ohm_m=q25.tolist(), q75_wave_ohm_m=q75.tolist()))
            print(rid, o['case_id'], 'cycles', len(cycles), 'excluded', len(rejected),
                  f'QRS F1 {match["F1"]:.2f} delta_rho {quantiles[1]:.8g} Ohm m', flush=True)
    require(len(records) == len(baseline) == 15 and len({r['record_id'] for r in records}) == 15,
            'Нужны все 15 записей базового ряда')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='scorpion_uniform_pulse_v1', status=STATUS, input_sha256=sha(input_path),
        baseline_path=str(baseline_path), baseline_sha256=baseline_sha,
        records=records, excluded_observations=inputs['excluded_observations'], phase_rr=PHASE.tolist(),
        signal='lateral BEI in ohm at 100 kHz', pulse_filter_hz=PARAMS['bei_pulse_band_hz'],
        protected_inputs=protected, breathing_annotation_accepted=False,
        producer_sha256=sha(__file__))
    path = out / 'pulse.local.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    print(prepare(a.input,a.baseline,a.out),flush=True)
