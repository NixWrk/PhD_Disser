"""Four consecutive existing QRS intervals; no new breathing or ECG annotation."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lateral_pilot import EXPERIMENT, candidate_windows, read, save, sha
from scorpion_protocol import channel_assignment
from scorpion_comparisons import COLORS

OUT = EXPERIMENT / 'derived/exploratory/four_cycles_20261013'
AUTHOR_HINT = EXPERIMENT / 'annotations/scorpion_review/four_cycle_display_author_20261013.local.json'
MAIN = {4: 140, 7: 120, 8: 100, 9: 80, 10: 60}
GROUPS = ('lateral', 'cross', 'ttrkg_variants', 'ttrkg', 'tepc')
TTRKG_VARIANTS = (2, 4, 3)  # Confirmed 1x1 / 1x4 / 2x2, same lateral 140 mm.


def choose_four(peaks, start, end, fs):
    """Central four consecutive valid RR intervals, bounded by five events."""
    peaks = np.asarray(peaks, dtype=int)
    if np.any(np.diff(peaks) <= 0):
        raise ValueError('QRS events must be strictly ordered')
    inside = peaks[(peaks >= start) & (peaks < end)]
    candidates = [inside[i:i + 5] for i in range(len(inside) - 4)
                  if np.all((np.diff(inside[i:i + 5]) / fs >= .45) &
                            (np.diff(inside[i:i + 5]) / fs <= 1.8))]
    if not candidates:
        return None
    midpoint = (start + end) / 2
    # Selection depends on the breathing window and ECG, never on pulse amplitude.
    return min(candidates, key=lambda p: (abs((p[0] + p[-1]) / 2 - midpoint), p[0]))


def lead_name(number, role, assignment):
    if role == 'ttrkg':
        config = assignment['ttrkg']
        variant = config['variant'].replace('x', '×')
        if 7 <= number <= 10:
            variant = '1×1? / 1×4?'
        elif config['status'] != 'author_confirmed':
            variant += '?'
        return f'{number:03d} · ТТРКГ {variant}'
    if role == 'tepc':
        return f'{number:03d} · ТЕПС-{number - 10}'
    if role == 'aepc':
        return f'{number:03d} · АЕПС'
    size = 140 if number <= 6 else MAIN.get(number, 60)
    axis = 'горизонтальная' if role == 'lateral_horizontal' else 'вертикальная'
    return f'{number:03d} · {size} мм · {axis}'


def run(tepc6='author_approximate_window', cross60='four_seconds'):
    """Only presentation windows change; strict 11.20 annotations stay immutable."""
    if tepc6 not in ('author_approximate_window', 'strict_available') or cross60 not in ('four_seconds', 'omit'):
        raise ValueError('Unknown explicit presentation policy')
    OUT.mkdir(parents=True, exist_ok=True)
    review = EXPERIMENT / 'derived/scorpion_review'
    summary_path = review / 'summary.local.json'
    annotations_path = EXPERIMENT / 'annotations/scorpion_review/annotations_candidates.local.json'
    rates_path = EXPERIMENT / 'derived/scorpion_bin_review/working_sampling_rates.local.json'
    summary, annotations, rates = read(summary_path), read(annotations_path), read(rates_path)
    hint = read(AUTHOR_HINT)
    if summary['code_sha256'] != sha(Path(__file__).with_name('scorpion_qc.py')):
        raise ValueError('Signal-processing producer changed')
    rates_by_id = {r['record_id']: r for r in rates['records']}
    segments, provenance, excluded = [], [], []
    for record in summary['records']:
        rid, fs = record['record_id'], record['fs_hz_scenario']
        number = int(rid[-3:])
        if fs != (400 if number in (5, 6) else 1000):
            continue
        if rid in rates_by_id and rates_by_id[rid]['working_sample_rate_hz'] != fs:
            raise ValueError('Timebase differs from existing BIN evidence')
        annotation = next(a for a in annotations['records']
                          if (a['record_id'], a['fs_hz_scenario']) == (rid, fs))
        source = EXPERIMENT / record['source_relative_path']
        if sha(source) != record['source_sha256'] or annotation['source_sha256'] != record['source_sha256']:
            raise ValueError('Signal or annotation source changed: ' + rid)
        cache = review / (record['key'] + '.npz')
        windows = candidate_windows(annotation['breathing'], fs, record['rows'])
        if 'inhale' not in windows:
            excluded.append(dict(record_id=rid, reason='no_reconstructed_deep_inspiration_hold'))
            continue
        strict_start, strict_end = windows['inhale']
        start, end = strict_start, strict_end
        ecg_channel = record['selected_ecg_channel_candidate']
        mode = 'four_candidate_cycles'
        with np.load(cache) as arrays:
            if ecg_channel is None:
                if number != 11:
                    raise ValueError('Unexpected record without selected ECG')
                if cross60 == 'omit':
                    excluded.append(dict(record_id=rid, reason='unreliable_ECG_no_selected_reference'))
                    continue
                mode = 'four_seconds_unreliable_ecg'
                start = int(round((strict_start + strict_end - 4 * fs) / 2))
                end = start + int(4 * fs)
                events = np.array([], dtype=int)
            else:
                peaks = arrays[f'qrs_samples_{ecg_channel}'].astype(int)
                if not np.array_equal(peaks, annotation['qrs_samples_by_channel'][str(ecg_channel)]):
                    raise ValueError('Cache and annotation QRS differ: ' + rid)
                events = choose_four(peaks, start, end, fs)
                if number == 16:
                    if tepc6 == 'author_approximate_window':
                        if hint['record_id'] != rid or hint['source_sha256'] != record['source_sha256']:
                            raise ValueError('Author display interval refers to another source')
                        start, end = (int(round(v * fs)) for v in hint['approximate_inhale_interval_s'])
                        events = choose_four(peaks, start, end, fs)
                        mode = 'four_candidate_cycles_author_approximate_window'
                    else:
                        events = peaks[(peaks >= start) & (peaks < end)]
                        mode = 'available_candidate_cycles_only'
                if events is None or len(events) < 2:
                    raise ValueError('Insufficient existing candidate events: ' + rid)
                start, end = int(events[0]), int(events[-1])
            indices = np.arange(start, end + 1, dtype=int)
            assignment = channel_assignment(rid)
            if len(indices) < 2 or indices[-1] >= record['rows']:
                raise ValueError('Presentation window outside source')
            t = (indices - start) / fs
            item = dict(record_id=rid, number=number, fs_hz=fs, mode=mode,
                        n_cycles=len(events) - 1 if len(events) else None,
                        window_samples_inclusive=[start, end],
                        window_original_time_s=[start / fs, end / fs],
                        strict_inhale_window_samples_half_open=[strict_start, strict_end],
                        qrs_candidate_samples=events.tolist(),
                        qrs_relative_time_s=((events - start) / fs).tolist(),
                        rr_s=(np.diff(events) / fs).tolist(),
                        ecg_reference_channel=ecg_channel,
                        annotation_status=annotation['status'],
                        breathing_status=annotation['breathing']['status'],
                        breathing_author_accepted=annotation['breathing']['author_accepted'],
                        flags=record['flags'],
                        montage_assignment_status=assignment['status'],
                        time_s=t.tolist(), leads=[], ecg=[])
            for channel in (1, 2):
                role = assignment['physical_channel_roles'][str(channel)]
                item['leads'].append(dict(channel=channel, role=role,
                    label=lead_name(number, role, assignment),
                    pulse_mohm=(1000 * arrays[f'bei_pulse_band_{channel}'][indices]).tolist()))
                if ecg_channel is None or channel == ecg_channel:
                    item['ecg'].append(dict(channel=channel,
                        label=lead_name(number, role, assignment),
                        wave_mV=arrays[f'ecg_filtered_{channel}'][indices].tolist()))
            if not np.isfinite(t).all() or any(not np.isfinite(x['pulse_mohm']).all() for x in item['leads']):
                raise ValueError('Nonfinite samples')
            segments.append(item)
            provenance.append(dict(record_id=rid, source_relative_path=record['source_relative_path'],
                source_sha256=record['source_sha256'], cache_path=cache.relative_to(EXPERIMENT).as_posix(),
                cache_sha256=sha(cache)))
    save(OUT / 'segments.local.json', segments)
    save(OUT / 'contract.local.json', dict(status='exploratory_hypothesis_not_validated',
        producer_sha256=sha(__file__), protocol_sha256=sha(Path(__file__).with_name('scorpion_protocol.py')),
        summary_sha256=sha(summary_path), annotations_sha256=sha(annotations_path),
        sampling_rates_sha256=sha(rates_path), filters=summary['filters'],
        author_display_hint_sha256=sha(AUTHOR_HINT), author_display_hint=hint,
        policies=dict(tepc6=tepc6, cross60=cross60), source_artifacts=provenance, excluded=excluded,
        selection_rule='Central four consecutive 0.45–1.8 s existing QRS intervals, five boundary events; no pulse-amplitude selection',
        alignment='First selected QRS at zero; no amplitude, sign or time normalization; no averaging',
        pulse_source='Existing whole-record 0.7–12 Hz BEI band, multiplied by 1000 for milliohms',
        annotations_modified=False, clinical_validation=False,
        outputs={'segments.local.json': sha(OUT / 'segments.local.json')}))
    verify.cache_clear()
    return segments


@lru_cache(maxsize=1)
def verify():
    contract = read(OUT / 'contract.local.json')
    checks = [(Path(__file__), contract['producer_sha256']),
              (Path(__file__).with_name('scorpion_protocol.py'), contract['protocol_sha256']),
              (EXPERIMENT / 'derived/scorpion_review/summary.local.json', contract['summary_sha256']),
              (EXPERIMENT / 'annotations/scorpion_review/annotations_candidates.local.json', contract['annotations_sha256']),
              (EXPERIMENT / 'derived/scorpion_bin_review/working_sampling_rates.local.json', contract['sampling_rates_sha256'])]
    checks.append((AUTHOR_HINT, contract['author_display_hint_sha256']))
    checks.extend((OUT / name, digest) for name, digest in contract['outputs'].items())
    for item in contract['source_artifacts']:
        checks.extend([(EXPERIMENT / item['source_relative_path'], item['source_sha256']),
                       (EXPERIMENT / item['cache_path'], item['cache_sha256'])])
    for path, digest in checks:
        if sha(path) != digest:
            raise ValueError('Four-cycle input changed: ' + str(path.name))
    return contract


def selected_leads(group, item):
    if group not in GROUPS:
        raise ValueError('Unknown physical assembly group')
    return [lead for lead in item['leads'] if
            (group == 'lateral' and item['number'] in MAIN and lead['role'] == 'lateral_horizontal') or
            (group == 'cross' and item['number'] in (6, 11)) or
            (group == 'ttrkg_variants' and item['number'] in TTRKG_VARIANTS and lead['role'] == 'ttrkg') or
            (group == 'ttrkg' and lead['role'] == 'ttrkg') or
            (group == 'tepc' and lead['role'] == 'tepc')]


def figure(group):
    verify()
    segments = [s for s in read(OUT / 'segments.local.json') if selected_leads(group, s)]
    if group == 'ttrkg_variants':
        segments.sort(key=lambda s: TTRKG_VARIANTS.index(s['number']))
    cross = group == 'cross'
    titles = ('Импеданс ТЕПС-2–7: пульсовая составляющая', 'Опорная ЭКГ для границ циклов') if group == 'tepc' else \
             ('ТТРКГ 1×1, 1×4 и 2×2: пульсовая составляющая', 'Опорная ЭКГ для границ циклов') if group == 'ttrkg_variants' else None
    fig = make_subplots(rows=2, cols=2 if cross else 1, shared_xaxes=True,
        row_heights=[.64, .36], vertical_spacing=.11,
        subplot_titles=('Крест 140 мм · четыре кандидатных цикла',
                        'Крест 60 мм · 4 с; ЭКГ ненадёжна' if any(s['number'] == 11 for s in segments) else
                        'Крест 60 мм · график отложен: ЭКГ ненадёжна',
                        'ЭКГ выбранного отведения', 'ЭКГ обеих осей') if cross else titles)
    for i, item in enumerate(segments):
        col = 1 if not cross or item['number'] == 6 else 2
        leads = selected_leads(group, item)
        label = leads[0]['label']
        if item['mode'] == 'four_candidate_cycles_author_approximate_window':
            label += ' · вдох ≈22–30 с по автору'
        elif item['mode'] == 'available_candidate_cycles_only':
            label += f' · {item["n_cycles"]} цикл'
        for j, lead in enumerate(leads):
            color = ('#176aaf' if j == 0 else '#d97706') if cross else COLORS[i % len(COLORS)]
            legend_label = lead['label'] if cross else label
            fig.add_trace(go.Scatter(x=item['time_s'], y=lead['pulse_mohm'],
                name=legend_label, legendgroup=legend_label, line=dict(color=color, width=1.6),
                hovertemplate='%{x:.3f} с<br>%{y:.3f} мОм<extra>%{fullData.name}</extra>'), row=1, col=col)
        for j, ecg in enumerate(item['ecg']):
            color = ('#176aaf' if j == 0 else '#d97706') if cross else COLORS[i % len(COLORS)]
            legend_label = label if not cross else ecg['label']
            fig.add_trace(go.Scatter(x=item['time_s'], y=ecg['wave_mV'], name='ЭКГ · ' + ecg['label'],
                legendgroup=legend_label, showlegend=False, line=dict(color=color, width=1),
                hovertemplate='%{x:.3f} с<br>%{y:.4f} мВ<extra>%{fullData.name}</extra>'), row=2, col=col)
            if item['qrs_relative_time_s']:
                indices = np.array(item['qrs_candidate_samples']) - item['window_samples_inclusive'][0]
                fig.add_trace(go.Scatter(x=item['qrs_relative_time_s'],
                    y=np.array(ecg['wave_mV'])[indices].tolist(), mode='markers',
                    marker=dict(color=color, size=6), showlegend=False, legendgroup=legend_label,
                    name='Граница кандидатного цикла'), row=2, col=col)
        if cross and item['number'] == 6:
            for event in item['qrs_relative_time_s']:
                fig.add_vline(x=event, line=dict(color='#94a3b8', width=.8, dash='dot'), row=1, col=col)
        if item['mode'] == 'four_candidate_cycles_author_approximate_window':
            lo, hi = item['strict_inhale_window_samples_half_open']
            anchor = item['window_samples_inclusive'][0]
            for a, b in [(0, (lo - anchor) / item['fs_hz']),
                         ((hi - anchor) / item['fs_hz'], item['time_s'][-1])]:
                a, b = max(0, a), min(item['time_s'][-1], b)
                if b > a:
                    for row in (1, 2):
                        fig.add_vrect(x0=a, x1=b, fillcolor='#f59e0b', opacity=.09,
                                      line_width=0, row=row, col=col)
    max_time = max(s['time_s'][-1] for s in segments)
    limit = 1.07 * max(abs(v) for s in segments for lead in selected_leads(group, s) for v in lead['pulse_mohm'])
    for col in range(1, 3 if cross else 2):
        fig.update_yaxes(title_text='Пульсовая составляющая, мОм', range=[-limit, limit], zeroline=True, row=1, col=col)
        fig.update_yaxes(title_text='ЭКГ, мВ', row=2, col=col)
        fig.update_xaxes(range=[0, max_time], row=1, col=col)
        fig.update_xaxes(title_text='Время от начала показанного участка, с' if cross and col == 2 else
                         'Время от первого выбранного QRS, с', range=[0, max_time], row=2, col=col)
    fig.update_layout(template='plotly_white', height=800 if not cross else 760,
        hovermode='closest', legend=dict(orientation='h', y=-.12, yanchor='top', x=0),
        margin=dict(t=60, b=180 if group == 'ttrkg' else 130, l=80, r=35),
        uirevision='four-cycles-' + group)
    if group == 'lateral':
        # Keep physical units; the second view hides 60 mm and changes only axes.
        small = [s for s in segments if s['number'] != 10]
        small_limit = 1.07 * max(abs(v) for s in small for p in selected_leads(group, s) for v in p['pulse_mohm'])
        fig.update_layout(updatemenus=[dict(type='buttons', direction='right', x=0, y=1.14,
            buttons=[dict(label='Все пять сборок', method='update',
                          args=[{'visible': [True] * len(fig.data)}, {'yaxis.range': [-limit, limit]}]),
                     dict(label='140–80 мм: подробный вид', method='update',
                          args=[{'visible': [not str(t.name).startswith('010') and
                                             t.legendgroup != '010 · 60 мм · горизонтальная' for t in fig.data]},
                                {'yaxis.range': [-small_limit, small_limit]}])])])
        fig.update_layout(margin=dict(t=95))
    elif cross:
        local_limits = {}
        for number in (6, 11):
            values = [abs(v) for s in segments if s['number'] == number for p in selected_leads(group, s) for v in p['pulse_mohm']]
            local_limits[number] = 1.07 * max(values) if values else limit
        individual = {'yaxis.range': [-local_limits[6], local_limits[6]],
                      'yaxis2.range': [-local_limits[11], local_limits[11]]}
        common = {'yaxis.range': [-limit, limit], 'yaxis2.range': [-limit, limit]}
        fig.update_layout(**{'yaxis': {'range': individual['yaxis.range']},
                             'yaxis2': {'range': individual['yaxis2.range']}},
            updatemenus=[dict(type='buttons', direction='right', x=0, y=1.16,
                buttons=[dict(label='Масштаб каждого креста', method='relayout', args=[individual]),
                         dict(label='Общая шкала 140 и 60 мм', method='relayout', args=[common])])])
        fig.update_layout(margin=dict(t=115))
    return fig


def window_table():
    verify()
    rows = []
    for item in read(OUT / 'segments.local.json'):
        lo, hi = item['window_original_time_s']
        rows.append({'Запись': f'{item["number"]:03d}',
            'Сборки': '; '.join(lead['label'].split(' · ', 1)[1] for lead in item['leads']),
            'Участок исходной записи, с': f'{lo:.3f}–{hi:.3f}',
            'Кандидатные циклы': item['n_cycles'] if item['n_cycles'] is not None else 'Не определены',
            'R–R-интервалы, с': ' / '.join(f'{v:.3f}' for v in item['rr_s']) or 'ЭКГ ненадёжна',
            'Условия показа': 'Вдох ≈22–30 с по автору; оранжевые зоны вне 27–29 с' if item['mode'].endswith('author_approximate_window') else
                'Четыре секунды без привязки к QRS' if item['n_cycles'] is None else
                'Внутри участка задержки с отступами'})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tepc6', choices=['author_approximate_window', 'strict_available'], default='author_approximate_window')
    parser.add_argument('--cross60', choices=['four_seconds', 'omit'], default='four_seconds')
    args = parser.parse_args()
    result = run(args.tepc6, args.cross60)
    print(window_table().to_string(index=False))
