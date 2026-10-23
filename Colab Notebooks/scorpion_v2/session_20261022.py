"""11.40/33.40: separate participant processing of the 22 October session.

Raw files, annotations and derived arrays stay in the ignored experiment folder.
Breath boundaries and QRS events are candidates, never promoted by this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from audit_scorpion_session import load_txt, read, sha, write
from scorpion_qc import bei_filter, ecg_filter, qrs_candidates, peak_metrics, mains_notch, zero_phase
from lateral_pilot import cycle_ensemble

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'Experiments/scorpion_20261022'
FS = 1000
STATUS = 'exploratory_hypothesis_not_validated'
ROLES = {'nik': {'ttrkg': 1, 'lateral': 2},
         'yaroslav': {'ttrkg': 2, 'lateral': 1}}
# Three ambiguity bands: reaching inspiration hold, leaving it, reaching
# expiration hold. Reconstructed by viewing each participant's own slow BEI.
BOUNDARIES_S = {
    'N001': [[16, 19], [34, 36], [50, 53]],
    'N002': [[14, 17], [28, 30], [47, 51]],
    'N003': [[16, 19], [29, 31], [46, 49]],
    'N005': [[16, 19], [31, 33], [48, 51]],
    'N006': [[14, 18], [27, 29], [47, 51]],
    'N007': [[18, 22], [33, 35], [49, 53]],
    'N008': [[16, 20], [33, 35], [51, 55]],
    'Y001': [[18, 22], [36, 38], [54, 58]],
    'Y002': [[24, 28], [39, 41], [59, 63]],
    'Y003': [[16, 19], [29, 31], [47, 51]],
    'Y005': [[16, 20], [34, 36], [49, 53]],
    'Y006': [[16, 20], [33, 35], [50, 54]],
    'Y007': [[16, 21], [35, 37], [51, 55]],
}
# Stable within-record state windows, away from disconnection/reconnection.
# Every row is an explicit physical-state hypothesis from the raw traces.
SWITCH_WINDOWS_S = {
    'nik': [('both', 5, 13), ('ttrkg_only', 24, 41),
            ('both', 55, 62), ('lateral_only', 77, 100), ('both', 112, 116)],
    'yaroslav': [('both', 5, 11), ('ttrkg_only', 21, 32),
                 ('both', 41, 44), ('lateral_only', 55, 65), ('both', 78, 89)],
}
SUBJECT_NAMES = {'nik': 'Ник', 'yaroslav': 'Ярослав'}


def active_roles(kind):
    if kind == 'ttrkg_only':
        return ['ttrkg']
    if kind == 'lateral_only':
        return ['lateral']
    if kind == 'switching':
        return []  # State windows have their own active ports.
    if kind not in ('both', 'reciprocity'):
        raise ValueError('Unknown record kind')
    return ['ttrkg', 'lateral']


def inner_windows(bands_s, n, fs=FS, margin_s=2., edge_s=5.):
    bands = np.asarray(bands_s, dtype=float)
    if bands.shape != (3, 2) or not np.isfinite(bands).all():
        raise ValueError('Three finite ambiguity bands required')
    if np.any(np.diff(bands.ravel()) <= 0) or bands[0, 0] < 0 or bands[-1, 1] >= n / fs:
        raise ValueError('Breathing bands must be ordered inside this recording')
    ends = {'inhale': (bands[0, 1] + margin_s, bands[1, 0] - margin_s),
            'exhale': (bands[2, 1] + margin_s, n / fs - edge_s)}
    return {state: [int(round(max(edge_s, a) * fs)),
                    int(round(min(n / fs - edge_s, b) * fs))]
            for state, (a, b) in ends.items() if b > a}


def choose_four(peaks, start, end, fs=FS):
    """Choose central four consecutive intervals, with no amplitude selection."""
    peaks = np.asarray(peaks, dtype=int)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError('QRS events must be strictly ordered')
    p = peaks[(peaks >= start) & (peaks < end)]
    candidates = [p[i:i + 5] for i in range(len(p) - 4)
                  if np.all((np.diff(p[i:i + 5]) / fs >= .45) &
                            (np.diff(p[i:i + 5]) / fs <= 1.8))]
    return (min(candidates, key=lambda x: (abs((x[0] + x[-1]) / 2 - (start + end) / 2), x[0]))
            if candidates else None)


def suppress_close_candidates(peaks, ecg, fs=FS):
    """Keep stronger ECG extrema within a 450 ms analysis dead time.

    The original output remains available. Removing weak close extrema is
    candidate QC, not manual R annotation; rates above 133/min are excluded.
    """
    selected = []
    for p in sorted(np.asarray(peaks, dtype=int), key=lambda x: (-abs(ecg[x]), x)):
        if all(abs(p - q) >= .45 * fs for q in selected):
            selected.append(int(p))
    return np.array(sorted(selected), dtype=int)


def confirmed_sampling(inventory, matches):
    """Require every TXT row linked to the same working control-field scenario."""
    by_id = {m['record_id']: m for m in matches['matches']}
    for r in inventory['files']:
        if not r['signal_input_candidate']:
            continue
        m = by_id[r['record_id']]
        if (m['txt_sha256'] != r['sha256'] or not m['counter_equal_all_rows'] or
                m['matched_rows'] != r['text_qc']['rows']):
            raise ValueError('Incomplete BIN/TXT match')
        states = m['settings_states_in_record']
        if sum(s['data_packets'] for s in states) != m['matched_rows']:
            raise ValueError('Settings do not cover the whole recording')
        for s in states:
            c = s['control']
            if c['u16_at_2'] != 1000 or c['groups_from_14_u32_u16'] != [[100000000, 7000]]:
                raise ValueError('Unexpected source settings; do not reuse 1000 Hz silently')
    return {'working_fs_hz': FS, 'manufacturer_field_semantics_confirmed': False,
            'basis': 'Совпадение COUNT и обоих ЭКГ-полей на всей длине каждого TXT; '
                     'одинаковые поля настроек 1000 и [100000000, 7000]. '
                     'Рабочая трактовка периода 1 мс согласуется с ЭКГ; единицы '
                     'двоичного протокола производителем не подтверждены.'}


def author_context():
    return dict(date='2026-10-23', source='Ответы автора в текущем диалоге',
        probing_frequency_hz=100000, current_per_source_microampere_approx=7000,
        current_amplitude_or_rms='не установлено',
        ordinary_breathing_order=['спокойное дыхание', 'задержка на глубоком вдохе',
                                  'усиленное дыхание', 'задержка на глубоком выдохе'],
        channel_assignment_constant_within_subject=True,
        assembly_meanings='Размер в мм относится к продольной боковой сборке с общим центром; '
                          '1×1 и 2×2 — варианты ТТРКГ.',
        arm_circumference_mm={'nik': {'value': 370, 'status': 'сообщено автором'},
                             'yaroslav': {'value': 370, 'status': 'приблизительно как у Ника',
                                          'uncertainty_mm': None}},
        arm_contacts='2×2: контакты сверху и снизу; 1×1: только верхние. '
                     'Токовая и измерительная группы разделены примерно на 100 мм; '
                     'измерительная группа примерно в 20 мм от плеча.',
        lateral_region='Правая боковая поверхность в той же области рёбер; '
                       'индивидуальные координаты центра в КТ ещё не установлены.',
        disconnection='Разъём вынут из прибора; электроды остаются на теле.',
        switching_breathing='По уточнению автора 23.10.2026 — глубокий вдох на протяжении всей пробы попеременного отключения.',
        reciprocity='У Ника менялись токовая и измерительная пары ТТРКГ.',
        no_georg_anatomy_or_resistivity_transfer=True)


def process(experiment=EXPERIMENT):
    experiment = Path(experiment)
    inventory_path = experiment / 'metadata/inventory.local.json'
    matches_path = experiment / 'derived/scorpion_bin_review/txt_binary_matches.local.json'
    inventory, matches = read(inventory_path), read(matches_path)
    identities=read(experiment/'record_identity.json')
    expected={r['record_id']:(r['sha256'],r['raw_relative_path']) for r in identities['records']}
    actual={r['record_id']:(r['sha256'],r['raw_relative_path']) for r in inventory['files'] if r['signal_input_candidate']}
    if expected!=actual:raise ValueError('Fixed recording identities changed; do not transfer boundary bands')
    sampling = confirmed_sampling(inventory, matches)
    ann_dir = experiment / 'annotations/session_1140'
    out = experiment / 'derived/exploratory/session_3340'
    ann_dir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    author_path = experiment / 'metadata/author_confirmations_20261023.local.json'
    write(author_path, author_context())
    annotations, records, levels, switch_rows = [], [], [], []
    for source in inventory['files']:
        if not source['signal_input_candidate']:
            continue
        rid, subject = source['record_id'], source['subject_id']
        short = rid.replace('SCV2-', '').replace('-', '')
        path = experiment / source['raw_relative_path']
        if sha(path) != source['sha256']:
            raise ValueError('Changed raw source: ' + rid)
        frame = load_txt(path)
        kind = source['montage_context']['record_kind']
        context = dict(source['montage_context'])
        # Single-lateral filenames omit the disconnected TTRKG variant;
        # these are the 140/2×2 control block, not a new inferred montage.
        if kind == 'lateral_only':
            context['ttrkg_variant'] = '2x2'
        if kind == 'ttrkg_only':
            context['lateral_size_mm'] = 140
        roles = ROLES[subject]
        reference = roles['lateral'] if kind == 'lateral_only' else roles['ttrkg']
        arrays = {}
        ecg_qc = {}
        for channel in (1, 2):
            z = frame[f'BEI{channel}ohm_100kHz'].to_numpy(float)
            ecg = frame[f'EMG{channel}mV'].to_numpy(float)
            _, ef = ecg_filter(ecg, FS)
            ef50 = zero_phase(signal.butter(4,[.5,40],btype='bandpass',fs=FS,output='sos'),
                              mains_notch(ecg,FS,[50]),FS,3)
            slow, residual, pulse, other, activity = bei_filter(z, FS)
            original_peaks = qrs_candidates(ef, FS) if kind != 'switching' else np.array([], dtype=int)
            peaks = suppress_close_candidates(original_peaks, ef)
            arrays.update({f'raw_{channel}': z, f'slow_{channel}': slow,
                           f'pulse_{channel}': pulse, f'ecg_raw_{channel}': ecg,
                           f'ecg_{channel}': ef, f'ecg_notch50_{channel}': ef50, f'qrs_{channel}': peaks,
                           f'qrs_unpruned_{channel}': original_peaks})
            ecg_qc[str(channel)] = {**peak_metrics(peaks, FS, len(frame)),
                'unpruned_candidate_count': len(original_peaks),
                'notch50_vs_50_100_150_rms_difference_mV': float(np.sqrt(np.mean((ef50-ef)**2))),
                'raw_q05_q95_span_mV': float(np.ptp(np.quantile(ecg, [.05, .95])))}
        windows = inner_windows(BOUNDARIES_S[short], len(frame)) if kind != 'switching' else {}
        annotation = dict(record_id=rid, source_sha256=source['sha256'], owner='11.40',
            status='pending_manual_review', author='Codex: ретроспективная кандидатная реконструкция',
            evidence='Порядок обычных дыхательных проб подтверждён автором. '
                     'Границы определены по медленной составляющей собственного BEI; '
                     'зумированный просмотр обеих сборок, 23.10.2026.',
            confidence='средняя; переходы неоднозначны', fs_hz=FS,
            boundary_uncertainty_ranges_s=BOUNDARIES_S.get(short),
            inner_windows_samples_half_open=windows,
            ecg_reference_channel_candidate=reference if kind != 'switching' else None,
            ecg_events_status='автоматические кандидаты комплексов QRS, ручное принятие отсутствует',
            switching_windows_s=SWITCH_WINDOWS_S[subject] if kind == 'switching' else None)
        annotations.append(annotation)
        record = dict(record_id=rid, subject=subject, source_name=source['source_name'],
            source_relative_path=source['raw_relative_path'], source_sha256=source['sha256'],
            n=len(frame), fs=FS, context=context, physical_channel_roles=roles,
            roles_evidence='Вывод по записям одной подключённой сборки и ослаблению ЭКГ '
                           'при отключении; постоянство внутри участника подтверждено автором.',
            active_roles=active_roles(kind), ecg_reference=reference,
            ecg_qc=ecg_qc, windows=windows, ensembles={}, four_cycles=None)
        if kind == 'switching':
            for state, start_s, end_s in SWITCH_WINDOWS_S[subject]:
                a, b = int(start_s * FS), int(end_s * FS)
                if not 0 <= a < b <= len(frame):
                    raise ValueError('Switching interval exceeds source')
                row = dict(record_id=rid, subject=subject, state=state, start_s=start_s, end_s=end_s)
                for role, channel in roles.items():
                    raw = arrays[f'raw_{channel}'][a:b]
                    row[role + '_median_ohm'] = float(np.median(raw))
                    row[role + '_q05_q95_ohm'] = np.quantile(raw, [.05, .95]).tolist()
                switch_rows.append(row)
        else:
            peaks = arrays[f'qrs_{reference}']
            for state, (a, b) in windows.items():
                record['ensembles'][state] = {}
                for role in record['active_roles']:
                    channel = roles[role]
                    raw, slow = arrays[f'raw_{channel}'][a:b], arrays[f'slow_{channel}'][a:b]
                    en = cycle_ensemble(arrays[f'pulse_{channel}'], peaks, a, b, FS)
                    record['ensembles'][state][role] = en
                    levels.append(dict(record_id=rid, subject=subject, role=role, kind=kind,
                        size_mm=context['lateral_size_mm'], variant=context['ttrkg_variant'],
                        state=state, start_s=a / FS, end_s=b / FS,
                        baseline_ohm=float(np.median(raw)),
                        raw_q05_q95_ohm=np.quantile(raw, [.05, .95]).tolist(),
                        slow_drift_ohm=float(np.median(slow[len(slow)//2:]) -
                                             np.median(slow[:len(slow)//2])),
                        n_candidate_cycles=en['n_cycles'],
                        median_curve_ptp_mohm=en.get('median_curve_ptp_mohm'),
                        pulse_rms_mohm=float(np.sqrt(np.mean(arrays[f'pulse_{channel}'][a:b]**2))*1000)))
            if 'inhale' in windows:
                events = choose_four(peaks, *windows['inhale'])
                if events is not None:
                    record['four_cycles'] = dict(events_samples=events.tolist(),
                        start_s=float(events[0]/FS), end_s=float(events[-1]/FS),
                        rr_s=(np.diff(events)/FS).tolist(),
                        status='четыре кандидатных цикла; QRS не приняты вручную')
        npz_path = out / (rid + '.local.npz')
        np.savez_compressed(npz_path, **arrays)
        record['arrays_sha256'] = sha(npz_path)
        records.append(record)
    ann_path = ann_dir / 'candidate_annotations.local.json'
    write(ann_path, dict(owner='11.40', status='pending_manual_review', records=annotations))
    result = dict(status=STATUS, owner='33.40', sampling=sampling,
        producer_sha256=sha(__file__), inventory_sha256=sha(inventory_path),
        matches_sha256=sha(matches_path), annotations_sha256=sha(ann_path),
        author_context_sha256=sha(author_path),
        filter_source_sha256=sha(Path(__file__).with_name('scorpion_qc.py')),
        ensemble_source_sha256=sha(Path(__file__).with_name('lateral_pilot.py')),
        records=records, levels=levels, switching=switch_rows,
        missing_montages=['1×4', '80 мм', '120 мм', 'кресты', 'ТЕПС', 'АЕПС'])
    write(out / 'summary.local.json', result)
    pd.DataFrame(levels).to_csv(out / 'levels.local.csv', index=False, encoding='utf-8-sig')
    print('Processed', len(records), 'records;', len(levels), 'active-assembly state summaries')
    print('Four-cycle candidate windows:', sum(r['four_cycles'] is not None for r in records))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    process(parser.parse_args().experiment)
