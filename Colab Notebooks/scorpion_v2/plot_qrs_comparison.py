"""Plot four candidate QRS complexes from each of two recorded ECG fields."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from match_scorpion_bin import read_txt_ecg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    review = experiment / 'derived/scorpion_review'
    out = review / 'qrs_comparison'
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((review / 'summary.local.json').read_text(encoding='utf-8'))
    rates = json.loads((experiment / 'derived/scorpion_bin_review/working_sampling_rates.local.json').read_text(encoding='utf-8'))
    rates_by_id = {r['record_id']: r for r in rates['records']}
    selections = []
    for record_id, fs, color, label in [
        ('SCV2-G-004', 1000, '#2563eb', '1000 Гц — 140 горизонт 1×4 (004)'),
        ('SCV2-G-005', 400, '#dc6803', '400 Гц — 140 верт 1×4 (005)'),
    ]:
        if rates_by_id[record_id]['working_sample_rate_hz'] != fs:
            raise ValueError('Requested timebase differs from the working BIN evidence')
        stats = next(r for r in summary['records'] if r['record_id'] == record_id and r['fs_hz_scenario'] == fs)
        source = experiment / stats['source_relative_path']
        if hashlib.sha256(source.read_bytes()).hexdigest() != stats['source_sha256']:
            raise ValueError('Primary signal hash changed')
        cache = review / f'{stats["key"]}.npz'
        with np.load(cache) as a:
            ecg = a['ecg_filtered_1'].copy()
            candidates = a['qrs_samples_1'].astype(int)
            raw = a['ecg_raw_1'].copy()
        if not np.array_equal(read_txt_ecg(source)[:, 1], raw):
            raise ValueError('Cached ECG differs from the primary TXT')
        chosen = candidates[candidates >= 10 * fs][:4]
        if len(chosen) != 4:
            raise ValueError('Four candidate events are required')
        following_events = candidates[candidates > chosen[-1]]
        if not len(following_events):
            raise ValueError('A following event is needed to bound the comparison window')
        following = following_events[0]
        selections.append({'record_id': record_id, 'fs_hz': fs, 'color': color, 'label': label,
                           'ecg': ecg, 'all_candidates': candidates, 'chosen': chosen,
                           'next_event_relative_s': (following - chosen[0]) / fs,
                           'last_event_relative_s': (chosen[-1] - chosen[0]) / fs,
                           'source_sha256': stats['source_sha256'],
                           'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()})
    start_s = -.25
    end_s = min(max(s['last_event_relative_s'] for s in selections) + .35,
                min(s['next_event_relative_s'] for s in selections) - .12)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=False)
    provenance = []
    for s in selections:
        fs = s['fs_hz']; anchor = int(s['chosen'][0])
        lo = anchor + int(np.floor(start_s * fs))
        hi = anchor + int(np.floor(end_s * fs)) + 1
        if not 0 <= lo < hi <= len(s['ecg']):
            raise ValueError('Selected window lies outside the record')
        if np.count_nonzero((s['all_candidates'] >= lo) & (s['all_candidates'] < hi)) != 4:
            raise ValueError('Window does not contain exactly four candidate events')
        t = (np.arange(lo, hi) - anchor) / fs
        ax.plot(t, s['ecg'][lo:hi], color=s['color'], linewidth=1.7, label=s['label'])
        ax.scatter((s['chosen'] - anchor) / fs, s['ecg'][s['chosen']],
                   s=29, color=s['color'], edgecolor='white', linewidth=.7, zorder=4)
        provenance.append({'record_id': s['record_id'], 'fs_hz': fs, 'field': 'EMG1mV',
                           'source_sha256': s['source_sha256'], 'cache_sha256': s['cache_sha256'],
                           'window_sample_range_half_open': [lo, hi],
                           'window_original_time_s': [lo / fs, (hi - 1) / fs],
                           'qrs_candidate_samples': s['chosen'].tolist(),
                           'qrs_candidate_original_time_s': (s['chosen'] / fs).tolist(),
                           'RR_candidate_intervals_s': (np.diff(s['chosen']) / fs).tolist()})
    ax.set(xlabel='Время относительно первого выбранного QRS, с', ylabel='ЭКГ, мВ',
           xlim=(start_s, end_s))
    ax.axhline(0, color='#94a3b8', linewidth=.6)
    ax.grid(True, color='#e2e8f0', linewidth=.7)
    fig.suptitle('ЭКГ1: по четыре QRS-комплекса при 400 и 1000 Гц', y=.965, fontsize=15)
    fig.legend(loc='upper center', bbox_to_anchor=(.53, .91), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.23, top=.80, left=.075, right=.985)
    fig.text(.075, .065, '50/100/150 Гц подавлены; полоса 0,5–40 Гц. Амплитуда и интервалы не нормированы.\n'
             'Маркеры — кандидаты QRS. Записи раздельные; совпадение первого маркера задано при построении.',
             fontsize=10, color='#475569')
    basename = out / 'ECG1_four_QRS_400_vs_1000Hz'
    fig.savefig(basename.with_suffix('.png'), dpi=160, facecolor='white')
    fig.savefig(basename.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'status': 'visual_comparison_existing_pending_manual_review_candidates',
              'filters': summary['filters'], 'analysis_code_sha256': summary['code_sha256'],
              'timebase_status': 'working_BIN_supported_period_units_not_documented',
              'alignment': 'first_QRS_candidate_at_zero_no_time_warping',
              'amplitude_normalization': False, 'records': provenance}
    basename.with_suffix('.local.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(basename.with_suffix('.png'))


if __name__ == '__main__':
    main()
