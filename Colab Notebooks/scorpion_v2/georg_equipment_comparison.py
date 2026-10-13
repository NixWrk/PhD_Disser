"""Georg: reported-unit equipment comparisons and fixed-literature CT controls.

Reuses accepted experiment-2 and candidate Scorpion annotations. This module
does not annotate signals, fit gains/materials/positions, or solve FEM.
"""
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt

from lateral_pilot import ROOT, EXPERIMENT, read, save, sha
import scorpion_comparisons as SC
import four_cycle_comparisons as P4
import build_georg_paired_preflight_report as FEM

OUT = EXPERIMENT / 'derived/exploratory/equipment_comparison_20261013'
SIZES = (60, 80, 100, 120, 140)
MAIN = {140: 4, 120: 7, 100: 8, 80: 9, 60: 10}
MODE = {'inhale': 'задержка_вдох', 'exhale': 'задержка_выдох'}
STATE = {'inhale': 'Глубокий вдох', 'exhale': 'Глубокий выдох'}
DEVICE = {'mgtu': 'Реокардиомонитор МГТУ', 'scorpion': 'Scorpion v2'}
ROLE = {'lateral': 'Боковая сборка', 'ttrkg': 'ТТРКГ'}


def config_path():
    return Path(os.environ.get('KALMYKOV_EXP02_CONFIG',
                ROOT / 'Colab Notebooks/config/exp02_paths.local.json'))


def choose_legacy_four(cycles, start, stop):
    """Four contiguous conservative-QC cycles, nearest the hold midpoint."""
    candidates = []
    for i in range(len(cycles) - 3):
        block = cycles[i:i + 4]
        if not all(c['eligible_conservative'] and .45 <= c['rr_s'] <= 1.8 for c in block):
            continue
        if block[0]['start_r_s'] < start or block[-1]['stop_r_s'] > stop:
            continue
        if not all(abs(a['stop_r_s'] - b['start_r_s']) < 1e-9
                   for a, b in zip(block, block[1:])):
            continue
        events = [c['start_r_s'] for c in block] + [block[-1]['stop_r_s']]
        candidates.append(events)
    if not candidates:
        raise ValueError('No four consecutive conservative-QC cycles')
    return min(candidates, key=lambda p: (abs((p[0] + p[-1] - start - stop) / 2), p[0]))


def run():
    SC.verify(); P4.verify()
    plan, forward, current = FEM.verify()
    config_file = config_path(); config = read(config_file)
    derived = Path(config['derived_root']) / 'exp02'
    manifest_file = derived / 'qc/10.01_record_manifest.candidate.json'
    qc_file = derived / 'qc/12.01_signal_qc.candidate.json'
    manifest, qc = read(manifest_file), read(qc_file)
    qc_by_sha = {r['input_sha256']: r for r in qc['records']}
    sources = [config_file, manifest_file, qc_file,
               SC.OUT / 'contract.local.json', SC.OUT / 'measured_levels.local.csv',
               P4.OUT / 'contract.local.json', P4.OUT / 'segments.local.json',
               FEM.OUT / 'plan.local.json', FEM.RUN / 'forward_results.local.json']
    levels, segments = [], []
    for record in manifest['records']:
        if record['subject_id'] != 'exp02_georg' or record.get('side_size_mm') not in SIZES:
            continue
        if not record['include']:
            raise ValueError('Excluded experiment-2 record requested')
        digest = record['input_sha256']; size = record['side_size_mm']
        raw = Path(config['data_root']) / record['relative_path']
        breathing_file = derived / 'annotations/breathing' / (digest[:16] + '.json')
        ecg_file = derived / 'annotations/ecg' / (digest[:16] + '.json')
        breathing, ecg = read(breathing_file), read(ecg_file)
        if sha(raw) != digest or any(a['input']['sha256'] != digest for a in (breathing, ecg)):
            raise ValueError('Legacy source/annotation mismatch')
        if breathing['qc']['status'] != 'accepted' or ecg['qc']['status'] != 'accepted':
            raise ValueError('Experiment-2 annotation has not been accepted')
        if ecg['upstream_breathing']['sidecar_sha256'] != sha(breathing_file):
            raise ValueError('ECG classification refers to another breathing annotation')
        q = qc_by_sha[digest]
        data = pd.read_csv(raw, encoding='utf-8-sig')
        time = data['TIME_s'].to_numpy(float)
        fs = 1 / np.median(np.diff(time))
        if np.any(np.diff(time) <= 0) or not np.isclose(fs, 200) or \
                np.max(np.abs(np.diff(time) * fs - 1)) > 1e-6:
            raise ValueError('Legacy timebase is not uniform 200 Hz')
        # Identity in reported units; additional common filtering is explicit.
        band = butter(4, [.7, 12], btype='bandpass', fs=fs, output='sos')
        for role, channel in [('lateral', 2), ('ttrkg', 1)]:
            allowed = bool(q['component_use'][f'BASE{channel}'])
            base = data[f'BASE_{channel}_Ω'].to_numpy(float)
            for state, mode in MODE.items():
                a, b = breathing['accepted_modes'][mode]
                mask = (time >= a + .5) & (time <= b - .5)
                values = base[mask]
                cached = q['static_levels'][f'BASE{channel}_{state}']
                if len(values) != cached['sample_count'] or \
                        not np.isclose(np.median(values), cached['median'], rtol=0, atol=1e-10):
                    raise ValueError('Legacy BASE summary does not reproduce accepted-window QC')
                levels.append(dict(device='mgtu', role=role, record_id=record['record_id'],
                    size_mm=size, state=state, usable=allowed,
                    median_ohm=float(np.median(values)) if allowed else None,
                    q25_ohm=float(np.quantile(values, .25)) if allowed else None,
                    q75_ohm=float(np.quantile(values, .75)) if allowed else None,
                    start_s=a + .5, stop_s=b - .5, n_samples=len(values),
                    variant='unknown', annotation_status='accepted',
                    exclusion=None if allowed else 'TTRKG_component_excluded_by_experiment_2_QC'))
            if not q['component_use'][f'RHEO{channel}']:
                continue
            a, b = breathing['accepted_modes'][MODE['inhale']]
            cycles = [c for c in q['cycles'] if c['mode'] == MODE['inhale']]
            events = choose_legacy_four(cycles, a + .5, b - .5)
            if not all(np.any(np.isclose(ecg['rpeaks_s'], p, atol=1e-9)) for p in events):
                raise ValueError('QC cycles are not bounded by accepted ECG events')
            signal = data[f'RHEO_{channel}_mΩ'].to_numpy(float)
            filtered = sosfiltfilt(band, signal)
            indices = np.flatnonzero((time >= events[0] - 1e-9) & (time <= events[-1] + 1e-9))
            segments.append(dict(device='mgtu', role=role, record_id=record['record_id'],
                size_mm=size, variant='unknown', fs_hz=float(fs),
                window_original_time_s=[events[0], events[-1]],
                event_time_relative_s=(np.array(events) - events[0]).tolist(),
                time_s=(time[indices] - events[0]).tolist(),
                pulse_mohm=filtered[indices].tolist(), reported_rheo_mohm=signal[indices].tolist(),
                ecg_mV=(1000 * data['ECG_V'].to_numpy(float)[indices]).tolist(),
                ecg_status='accepted', signal_qc_status=qc['status']))
        sources.extend([raw, breathing_file, ecg_file])
    measured = pd.read_csv(SC.OUT / 'measured_levels.local.csv')
    for r in measured.to_dict('records'):
        number = int(r['record_id'][-3:])
        if number not in (2, 3, 4, 7, 8, 9, 10):
            continue
        role = 'lateral' if r['role'] == 'lateral_horizontal' else 'ttrkg'
        levels.append(dict(device='scorpion', role=role, record_id=r['record_id'],
            size_mm={2: 140, 3: 140, 4: 140, 7: 120, 8: 100, 9: 80, 10: 60}[number],
            state=r['state'], usable=True, median_ohm=r['baseline_raw_median_ohm'],
            q25_ohm=r['raw_q25_ohm'], q75_ohm=r['raw_q75_ohm'],
            start_s=r['start_s'], stop_s=r['end_s'], n_samples=None,
            variant={2: '1×1', 3: '2×2', 4: '1×4'}.get(number, '1×1? / 1×4?'),
            annotation_status='pending_manual_review', exclusion=None))
    for segment in read(P4.OUT / 'segments.local.json'):
        if segment['number'] not in MAIN.values():
            continue
        size = next(s for s, n in MAIN.items() if n == segment['number'])
        for lead in segment['leads']:
            role = 'lateral' if lead['role'] == 'lateral_horizontal' else 'ttrkg'
            segments.append(dict(device='scorpion', role=role, record_id=segment['record_id'],
                size_mm=size, variant='1×4' if size == 140 else '1×1? / 1×4?',
                fs_hz=segment['fs_hz'], window_original_time_s=segment['window_original_time_s'],
                event_time_relative_s=segment['qrs_relative_time_s'], time_s=segment['time_s'],
                pulse_mohm=lead['pulse_mohm'], ecg_mV=segment['ecg'][0]['wave_mV'],
                ecg_status='pending_manual_review', signal_qc_status='pending_manual_review'))
    reference_file = FEM.PIPE / plan['reference_materials_file']
    registry_file = FEM.PIPE / 'configs/tissue_literature_v2.json'
    reference = read(reference_file)
    if reference['frequency_hz'] != 100000 or reference['registry_sha256'] != sha(registry_file):
        raise ValueError('FEM material registry/frequency mismatch')
    model_rows = []
    for case in forward['cases']:
        if case['family'] != 'lateral':
            continue
        if not case['contact_qc_passed'] or not np.allclose(case['rho_ohm_m'], reference['rho_ohm_m']):
            raise ValueError('Fixed-literature finite-contact control unavailable')
        for role, index in [('lateral', (3, 2)), ('ttrkg', (1, 0))]:
            model_rows.append(dict(record_id=case['record_id'], role=role,
                variant=case['variant'].replace('x', '×'),
                model_ohm=float(np.array(case['port_transfer_ohm'])[index]),
                integration_difference_ohm=float(np.array(case['quadrature_difference_ohm'])[index]),
                own_source_quadrature_passed=case['own_source_quadrature_passed']))
    sources.extend([reference_file, registry_file,
                    FEM.PIPE / 'output/georg_current_model.local.json',
                    FEM.PIPE / current['model_directory'] / 'contacts.local.json'])
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'levels.local.json', levels); save(OUT / 'segments.local.json', segments)
    save(OUT / 'fem.local.json', model_rows)
    save(OUT / 'contract.local.json', dict(
        schema='georg_equipment_comparison_v1', status='exploratory_hypothesis_not_validated',
        producer_sha256=sha(__file__), source_artifacts=[dict(path=str(p), sha256=sha(p)) for p in sources],
        outputs={n: sha(OUT / n) for n in ['levels.local.json', 'segments.local.json', 'fem.local.json']},
        author_confirmation_date='2026-10-13', probing_frequency_hz={'mgtu': 100000, 'scorpion': 100000},
        legacy_ttrkg_variant='unknown_author_does_not_recall',
        ct_placement_policy='author_adopts_illustrative_lateral_region_and_cylindrical_arms_not_fitted_pose',
        ct_lateral_yz_mm=[0, -210], position_is_independently_measured=False,
        literature_rho_ohm_m=reference['rho_ohm_m'],
        amplitude_sign_gain_policy='instrument_reported_units_identity_transform',
        pulse_filter='whole-record 0.7–12 Hz order-4 Butterworth forward/backward; legacy input is already instrument RHEO',
        pulse_selection='four existing contiguous conservative-QC legacy cycles; existing Scorpion four-cycle windows',
        annotation_acceptance_changed=False, recordings_are_simultaneous=False,
        equipment_only_effect_identifiable=False, fem_mesh_convergence_passed=False,
        quantitative_fem_inversion_allowed=False, static_ct_predicts_pulse_waveform=False))
    verify.cache_clear()
    return levels, segments


@lru_cache(maxsize=1)
def verify():
    c = read(OUT / 'contract.local.json')
    for p, digest in [(Path(__file__), c['producer_sha256'])] + \
            [(Path(x['path']), x['sha256']) for x in c['source_artifacts']] + \
            [(OUT / n, h) for n, h in c['outputs'].items()]:
        if sha(p) != digest:
            raise ValueError('Equipment comparison input changed: ' + p.name)
    SC.verify(); P4.verify(); FEM.verify()
    return c


def level_frame():
    verify()
    return pd.DataFrame(read(OUT / 'levels.local.json'))


def level_figure(role):
    frame = level_frame(); frame = frame[(frame.role == role) & frame.usable]
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(STATE.values()))
    for column, state in enumerate(STATE, 1):
        for device, color in [('mgtu', '#176aaf'), ('scorpion', '#d97706')]:
            f = frame[(frame.device == device) & (frame.state == state)]
            if device == 'scorpion':
                f = f[f.record_id.isin(['SCV2-G-' + f'{n:03d}' for n in MAIN.values()])]
            f = f.sort_values('size_mm')
            fig.add_trace(go.Scatter(x=f.size_mm, y=f.median_ohm,
                mode='markers' if role == 'ttrkg' else 'lines+markers',
                name=DEVICE[device], legendgroup=device, showlegend=column == 1,
                line=dict(color=color), error_y=dict(type='data', symmetric=False,
                array=f.q75_ohm - f.median_ohm, arrayminus=f.median_ohm - f.q25_ohm),
                customdata=f[['record_id', 'variant', 'start_s', 'stop_s']].values,
                hovertemplate='%{customdata[0]} · ТТРКГ %{customdata[1]}<br>L=%{x} мм; %{y:.3f} Ом<br>Окно %{customdata[2]:.2f}–%{customdata[3]:.2f} с<extra>%{fullData.name}</extra>'), row=1, col=column)
        repeats = frame[(frame.device == 'scorpion') & (frame.state == state) &
                        frame.record_id.isin(['SCV2-G-002', 'SCV2-G-003'])]
        for r in repeats.to_dict('records'):
            fig.add_trace(go.Scatter(x=[140], y=[r['median_ohm']], mode='markers',
                marker=dict(size=10, symbol='diamond-open' if r['variant'] == '1×1' else 'square-open', color='#d97706'),
                name='Scorpion · одновременно ТТРКГ ' + r['variant'],
                legendgroup=r['record_id'], showlegend=column == 1), row=1, col=column)
    fig.update_xaxes(title_text='Размер одновременно подключённой боковой сборки L, мм' if role == 'ttrkg' else 'Размер боковой сборки L, мм', tickvals=list(SIZES))
    fig.update_yaxes(title_text='Медиана импеданса, Ом')
    fig.update_yaxes(matches='y', row=1, col=2)
    fig.update_layout(title=ROLE[role] + ': уровни на двух приборах', template='plotly_white',
                      height=470, legend=dict(orientation='h', y=-.3), margin=dict(b=130))
    return fig


def comparison_table():
    frame = level_frame(); rows = []
    for role in ROLE:
        for size in SIZES:
            for state in STATE:
                old = frame[(frame.device == 'mgtu') & (frame.role == role) &
                            (frame.size_mm == size) & (frame.state == state)].iloc[0]
                new = frame[(frame.device == 'scorpion') & (frame.role == role) &
                            (frame.record_id == f'SCV2-G-{MAIN[size]:03d}') & (frame.state == state)].iloc[0]
                rows.append({'Сборка': ROLE[role], 'Боковая L, мм': size, 'Дыхание': STATE[state],
                    'МГТУ, Ом': old.median_ohm, 'Scorpion, Ом': new.median_ohm,
                    'Scorpion − МГТУ, Ом': new.median_ohm - old.median_ohm,
                    'Scorpion / МГТУ': new.median_ohm / old.median_ohm,
                    'QC МГТУ': 'Допустимо по компоненту' if old.usable else 'ТТРКГ исключена',
                    'Вариант ТТРКГ Scorpion': new.variant})
    return pd.DataFrame(rows)


def contrast_figure():
    f = level_frame(); f = f[f.usable]
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(ROLE.values()))
    for column, role in enumerate(ROLE, 1):
        for device, color in [('mgtu', '#176aaf'), ('scorpion', '#d97706')]:
            v = f[(f.role == role) & (f.device == device)]
            if device == 'scorpion':
                v = v[v.record_id.isin([f'SCV2-G-{n:03d}' for n in MAIN.values()])]
            p = v.pivot(index='size_mm', columns='state', values='median_ohm').sort_index()
            fig.add_trace(go.Scatter(x=p.index, y=p.inhale - p.exhale, mode='lines+markers',
                name=DEVICE[device], legendgroup=device, showlegend=column == 1,
                line=dict(color=color)), row=1, col=column)
        fig.add_hline(y=0, line_color='#64748b', line_dash='dot', row=1, col=column)
    fig.update_xaxes(title_text='Размер боковой сборки L, мм', tickvals=list(SIZES))
    fig.update_yaxes(title_text='Медиана на вдохе − медиана на выдохе, Ом')
    fig.update_layout(title='Парный дыхательный контраст в каждой сессии', template='plotly_white', height=450,
                      legend=dict(orientation='h', y=-.25))
    return fig


def pulse_figure(role):
    verify(); segments = [s for s in read(OUT / 'segments.local.json') if s['role'] == role]
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
        subplot_titles=[DEVICE['mgtu'], DEVICE['scorpion'], 'ЭКГ МГТУ', 'Опорная ЭКГ Scorpion'],
        row_heights=[.65, .35], vertical_spacing=.13)
    for s in segments:
        col = 1 if s['device'] == 'mgtu' else 2
        color = SC.COLORS[SIZES.index(s['size_mm'])]
        label = f"{s['size_mm']} мм · {DEVICE[s['device']]}"
        group = f"{s['device']}_{s['size_mm']}"
        events = s['event_time_relative_s']
        pulse = np.array(s['pulse_mohm']); t = np.array(s['time_s'])
        fig.add_trace(go.Scatter(x=t, y=pulse, mode='lines', name=label, legendgroup=group,
            line=dict(color=color), hovertemplate='%{x:.3f} с; %{y:.2f} мОм<extra>%{fullData.name}</extra>'), row=1, col=col)
        fig.add_trace(go.Scatter(x=events, y=np.interp(events, t, pulse), mode='markers',
            marker=dict(color=color, symbol='x', size=7), name='Границы четырёх циклов',
            legendgroup=group, showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=t, y=s['ecg_mV'], mode='lines', name=label + ' · ЭКГ',
            legendgroup=group, showlegend=False, line=dict(color=color, width=1)), row=2, col=col)
    fig.update_xaxes(title_text='Время от первого выбранного R-зубца, с', row=2)
    fig.update_yaxes(title_text='Пульсовая составляющая, мОм', row=1)
    fig.update_yaxes(matches='y', row=1, col=2)
    fig.update_yaxes(title_text='ЭКГ, мВ', row=2)
    fig.update_layout(title=ROLE[role] + ': четыре цикла на глубоком вдохе, полоса 0,7–12 Гц',
        template='plotly_white', height=700, legend=dict(orientation='h', y=-.2, groupclick='togglegroup'), margin=dict(b=145))
    if role == 'lateral':
        visible = [t.legendgroup != 'scorpion_60' for t in fig.data]
        fig.update_layout(updatemenus=[dict(type='buttons', direction='right', x=0, y=1.2,
            buttons=[dict(label='Все размеры', method='update',
                          args=[{'visible': [True] * len(fig.data)}, {'yaxis.autorange': True, 'yaxis2.autorange': True}]),
                     dict(label='Скрыть Scorpion 60 мм', method='update',
                          args=[{'visible': visible}, {'yaxis.autorange': True, 'yaxis2.autorange': True}])])], margin=dict(t=140))
    return fig


def pulse_table():
    verify(); rows = []
    for s in read(OUT / 'segments.local.json'):
        t, pulse = np.array(s['time_s']), np.array(s['pulse_mohm'])
        events = s['event_time_relative_s']
        amplitudes = [float(np.ptp(pulse[(t >= a - 1e-9) & (t < b - 1e-9)]))
                      for a, b in zip(events, events[1:])]
        rows.append({'Прибор': DEVICE[s['device']], 'Сборка': ROLE[s['role']],
            'Боковая L, мм': s['size_mm'], 'Начало, с': s['window_original_time_s'][0],
            'Конец, с': s['window_original_time_s'][1], 'Циклов': len(events) - 1,
            'Медиана размаха четырёх циклов, мОм': float(np.median(amplitudes)),
            'Вариант ТТРКГ': 'Неизвестен' if s['variant'] == 'unknown' else s['variant'],
            'ЭКГ': 'Принятая разметка' if s['ecg_status'] == 'accepted' else 'Кандидатная разметка'})
    return pd.DataFrame(rows)


def fem_table():
    frame = level_frame(); rows = []
    for m in read(OUT / 'fem.local.json'):
        obs = frame[(frame.device == 'scorpion') & (frame.role == m['role']) & (frame.record_id == m['record_id'])]
        for r in obs.to_dict('records'):
            rows.append({'Сборка': ROLE[m['role']], 'Запись': r['record_id'][-3:],
                'Боковая L, мм': r['size_mm'], 'ТТРКГ в расчёте': m['variant'], 'Дыхание': STATE[r['state']],
                'Scorpion, Ом': r['median_ohm'], 'КТ/FEM при литературных ρ, Ом': m['model_ohm'],
                'FEM − Scorpion, Ом': m['model_ohm'] - r['median_ohm'],
                'Расхождение относительно Scorpion, %': 100 * (m['model_ohm'] / r['median_ohm'] - 1),
                'Разность правил интегрирования, Ом': m['integration_difference_ohm']})
    return pd.DataFrame(rows)


def fem_figure(role):
    f = fem_table(); f = f[f['Сборка'] == ROLE[role]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Измеренный и расчётный уровни', 'Расхождение FEM − Scorpion'])
    low = float(min(f['Scorpion, Ом'].min(), f['КТ/FEM при литературных ρ, Ом'].min())) - 5
    high = float(max(f['Scorpion, Ом'].max(), f['КТ/FEM при литературных ρ, Ом'].max())) + 5
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode='lines', name='Равенство уровней',
                            line=dict(color='#64748b', dash='dot')), row=1, col=1)
    for state, symbol in [('Глубокий вдох', 'circle'), ('Глубокий выдох', 'circle-open')]:
        for variant, color in zip(['1×1', '1×4', '2×2'], SC.COLORS):
            v = f[(f['Дыхание'] == state) & (f['ТТРКГ в расчёте'] == variant)]
            label = state + ' · ТТРКГ ' + variant
            fig.add_trace(go.Scatter(x=v['Scorpion, Ом'], y=v['КТ/FEM при литературных ρ, Ом'],
                mode='markers', name=label, legendgroup=label, marker=dict(color=color, symbol=symbol, size=9),
                customdata=v[['Запись', 'Боковая L, мм']].values,
                hovertemplate='Запись %{customdata[0]}, боковая %{customdata[1]} мм<br>Измерено %{x:.3f} Ом<br>FEM %{y:.3f} Ом<extra>%{fullData.name}</extra>'), row=1, col=1)
            x = [f"{r['Запись']} · {r['Боковая L, мм']} мм · {variant}" for r in v.to_dict('records')]
            fig.add_trace(go.Scatter(x=x, y=v['FEM − Scorpion, Ом'], mode='markers', name=label,
                legendgroup=label, showlegend=False, marker=dict(color=color, symbol=symbol, size=9)), row=1, col=2)
    fig.add_hline(y=0, line_dash='dot', line_color='#64748b', row=1, col=2)
    fig.update_xaxes(title_text='Scorpion, Ом', row=1, col=1)
    fig.update_xaxes(range=[low, high], row=1, col=1)
    fig.update_yaxes(title_text='КТ/FEM при литературных ρ, Ом', row=1, col=1)
    fig.update_yaxes(range=[low, high], row=1, col=1)
    fig.update_xaxes(title_text='Монтаж: запись · боковая сборка · ТТРКГ', tickangle=-60, row=1, col=2)
    fig.update_yaxes(title_text='FEM − Scorpion, Ом', row=1, col=2)
    fig.update_layout(title=ROLE[role] + ': КТ глубокого вдоха, фиксированные литературные ρ',
        template='plotly_white', height=750, margin=dict(b=230), legend=dict(orientation='h', y=-.45))
    return fig


if __name__ == '__main__':
    a, b = run()
    print(f'Equipment comparison: {len(a)} hold summaries, {len(b)} four-cycle segments')
