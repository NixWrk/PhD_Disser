"""Offline signal review and candidate annotations for Scorpion v2.

Primary inputs are never changed. All subject-linked outputs stay in the
experiment's ignored derived/ and annotations/ directories. No physiological
acceptance, tissue inversion or cardiac-function estimates are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy import ndimage, signal


STATUS = 'pending_manual_review'
COLORS = ['#2563eb', '#d97706']
MODES = ['calm_breathing', 'inspiration_hold', 'enhanced_breathing', 'expiration_hold']
MODE_NAMES = ['Спокойное дыхание', 'Задержка на вдохе', 'Усиленное дыхание', 'Задержка на выдохе']
PARAMS = {
    'ecg_notch_hz': [50.0, 100.0, 150.0], 'ecg_notch_Q': 30.0,
    'ecg_notch_comparison_hz': [50.0],
    'ecg_display_band_hz': [0.5, 40.0], 'ecg_band_order': 4,
    'ecg_detector_band_hz': [5.0, 25.0], 'ecg_detector_order': 3,
    'ecg_energy_window_s': 0.08, 'ecg_threshold_window_s': 10.0,
    'ecg_threshold_MAD_multiplier': 3.0, 'qrs_refractory_s': 0.30,
    'ecg_padding_s': 3.0, 'ecg_edge_exclusion_s': 1.0,
    'bei_slow_lowpass_hz': 0.5, 'bei_slow_order': 4,
    'bei_pulse_band_hz': [0.7, 12.0], 'bei_pulse_order': 4,
    'bei_padding_s': 5.0, 'bei_edge_advisory_s': 5.0,
    'bei_sensitivity_bands_hz': [[0.5, 12.0], [0.7, 12.0], [1.0, 12.0], [0.7, 20.0]],
    'resp_activity_derivative_rms_window_s': 4.0,
    'cross_ecg_match_tolerance_s': 0.08,
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def zero_phase(sos, x, fs, padding_s):
    return signal.sosfiltfilt(sos, x, padlen=min(int(padding_s * fs), len(x) - 1))


def mains_notch(x, fs, frequencies):
    if any(not 0 < hz < fs / 2 for hz in frequencies):
        raise ValueError('Notch frequencies must lie below the Nyquist frequency')
    sections = [signal.tf2sos(*signal.iirnotch(hz, PARAMS['ecg_notch_Q'], fs=fs))
                for hz in frequencies]
    return zero_phase(np.vstack(sections), x, fs, PARAMS['ecg_padding_s'])


def ecg_filter(x, fs):
    notch = mains_notch(x, fs, PARAMS['ecg_notch_hz'])
    sos = signal.butter(4, PARAMS['ecg_display_band_hz'], btype='bandpass', fs=fs, output='sos')
    return notch, zero_phase(sos, notch, fs, PARAMS['ecg_padding_s'])


def bei_filter(z, fs):
    slow_sos = signal.butter(4, PARAMS['bei_slow_lowpass_hz'], fs=fs, output='sos')
    pulse_sos = signal.butter(4, PARAMS['bei_pulse_band_hz'], btype='bandpass', fs=fs, output='sos')
    slow = zero_phase(slow_sos, z, fs, PARAMS['bei_padding_s'])
    residual = z - slow
    pulse = zero_phase(pulse_sos, z, fs, PARAMS['bei_padding_s'])
    other = residual - pulse
    activity = np.sqrt(ndimage.uniform_filter1d(np.gradient(slow, 1 / fs)**2, size=int(4 * fs)))
    return slow, residual, pulse, other, activity


def qrs_candidates(ecg, fs):
    """Energy detector; dominant QRS extrema are not automatically R waves."""
    qrs = zero_phase(signal.butter(3, [5, 25], btype='bandpass', fs=fs, output='sos'), ecg, fs, 3)
    energy = ndimage.uniform_filter1d(qrs * qrs, size=max(1, int(.08 * fs)))
    stride = max(1, int(fs / 50))
    e50 = energy[::stride]
    window = int(10 * fs / stride) | 1
    median = ndimage.median_filter(e50, size=window)
    mad = ndimage.median_filter(np.abs(e50 - median), size=window)
    threshold = np.interp(np.arange(len(energy)), np.arange(len(e50)) * stride, median + 3 * mad)
    peaks, _ = signal.find_peaks(energy, height=threshold, distance=int(.30 * fs),
                                 prominence=max(float(np.median(energy)) * .5, 1e-12))
    refined = []
    for p in peaks:
        left, right = max(0, p - int(.09 * fs)), min(len(ecg), p + int(.09 * fs) + 1)
        local = ecg[left:right]
        r = left + int(np.argmax(np.abs(local - np.median(local))))
        if fs <= r < len(ecg) - fs:
            refined.append(r)
    # Keep the stronger dominant extremum when refinement produces close events.
    retained = []
    for r in sorted(set(refined)):
        if retained and r - retained[-1] < .30 * fs:
            if abs(ecg[r]) > abs(ecg[retained[-1]]):
                retained[-1] = r
        else:
            retained.append(r)
    return np.asarray(retained, dtype=int)


def peak_metrics(peaks, fs, n):
    rr = np.diff(peaks) / fs
    gaps = np.diff(np.r_[0, peaks, n - 1]) / fs
    return {
        'candidate_count': len(peaks),
        'median_RR_candidate_s': float(np.median(rr)) if len(rr) else None,
        'RR_candidate_cv': float(np.std(rr) / np.mean(rr)) if len(rr) else None,
        'max_candidate_gap_s_including_record_edges': float(np.max(gaps)),
        'short_RR_fraction_below_0_45_s': float(np.mean(rr < .45)) if len(rr) else None,
        'long_RR_fraction_above_1_8_s': float(np.mean(rr > 1.8)) if len(rr) else None,
        'events_are_confirmed_R_waves': False,
    }


def pair_match(p1, p2, fs):
    i = j = hits = 0
    while i < len(p1) and j < len(p2):
        delta = p1[i] - p2[j]
        if abs(delta) <= .08 * fs:
            hits += 1; i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return {'matched_pairs': hits, 'F1': 2 * hits / (len(p1) + len(p2)) if len(p1) + len(p2) else 0.0}


def longest_run(mask, fs):
    positions = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return float(np.max(positions[1::2] - positions[::2]) / fs) if len(positions) else 0.0


def channel_qc(raw, filtered, fs):
    f, p = signal.welch(raw, fs=fs, nperseg=min(len(raw), int(8 * fs)))
    fn, pn = signal.welch(filtered, fs=fs, nperseg=min(len(raw), int(8 * fs)))
    line = (f >= 49) & (f <= 51)
    power_before = float(np.sum(p[line]))
    power_after = float(np.sum(pn[line]))
    eligible = f >= 1
    return {
        'raw_min': float(raw.min()), 'raw_median': float(np.median(raw)), 'raw_max': float(raw.max()),
        'raw_p01': float(np.quantile(raw, .01)), 'raw_p99': float(np.quantile(raw, .99)),
        'raw_unique_levels': int(len(np.unique(raw))),
        'raw_longest_constant_run_s': longest_run(np.r_[False, np.diff(raw) == 0], fs),
        'raw_longest_extreme_run_s': max(longest_run(raw == raw.min(), fs), longest_run(raw == raw.max(), fs)),
        'raw_psd_peak_above_1Hz': float(f[eligible][np.argmax(p[eligible])]),
        'raw_49_51Hz_power_density_sum': power_before,
        'notch_49_51Hz_power_density_sum': power_after,
        'notch_49_51Hz_reduction_dB': float(10 * np.log10(max(power_before, 1e-30) / max(power_after, 1e-30))),
    }, f, p, pn


def analyze_record(experiment, info, fs, out):
    raw_path = experiment / info['raw_relative_path']
    if sha256_file(raw_path) != info['sha256']:
        raise ValueError('Primary input hash mismatch: ' + info['record_id'])
    frame = pd.read_csv(raw_path, sep='\t', decimal=',', usecols=range(11))
    raw = frame.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite input')
    counts = raw[:, 0].astype(int)
    if not np.all(np.diff(counts) % 256 == 1):
        raise ValueError('Unexpected COUNT discontinuity')
    rid = info['record_id']; key = rid + '_fs' + str(int(fs))
    arrays = {'time_s': np.arange(len(frame)) / fs, 'raw_columns': frame.columns.to_numpy(dtype=str), 'raw': raw}
    stats = {'record_id': rid, 'key': key, 'fs_hz_scenario': fs,
             'timebase_status': 'author_default_unverified' if fs == 1000 else 'alternative_400Hz_not_confirmed',
             'rows': len(frame), 'duration_s_N_over_fs': len(frame) / fs,
             'montage_label': Path(info['archive_member']).stem,
             'source_relative_path': info['raw_relative_path'], 'source_sha256': info['sha256'],
             'status': STATUS, 'channels': {}, 'physiological_acceptance': False}
    peak_sets = []
    for ch in (1, 2):
        ecg_raw = frame[f'EMG{ch}mV'].to_numpy()
        z = frame[f'BEI{ch}ohm_100kHz'].to_numpy()
        notch, ecg = ecg_filter(ecg_raw, fs)
        notch50 = mains_notch(ecg_raw, fs, PARAMS['ecg_notch_comparison_hz'])
        ecg50 = zero_phase(signal.butter(4, PARAMS['ecg_display_band_hz'],
                                        btype='bandpass', fs=fs, output='sos'),
                          notch50, fs, PARAMS['ecg_padding_s'])
        slow, residual, pulse, other, activity = bei_filter(z, fs)
        peaks = qrs_candidates(ecg, fs); peak_sets.append(peaks)
        qc, freq, psd_raw, psd_notch = channel_qc(ecg_raw, notch, fs)
        _, psd50 = signal.welch(notch50, fs=fs, nperseg=min(len(ecg_raw), int(8 * fs)))
        qc['mains_notch_comparison'] = {}
        for hz in PARAMS['ecg_notch_hz']:
            band = np.abs(freq - hz) <= 1
            neighbors = (np.abs(freq - hz) >= 3) & (np.abs(freq - hz) <= 6)
            df = freq[1] - freq[0]
            before = float(np.sum(psd_raw[band]) * df)
            after50 = float(np.sum(psd50[band]) * df)
            after = float(np.sum(psd_notch[band]) * df)
            qc['mains_notch_comparison'][str(int(hz))] = {
                'band_hz': [hz - 1, hz + 1], 'raw_band_power_mV2': before,
                'after_50Hz_only_band_power_mV2': after50,
                'after_50_100_150Hz_band_power_mV2': after,
                'reduction_dB_against_raw': float(10 * np.log10(max(before, 1e-30) / max(after, 1e-30))),
                'raw_peak_to_neighbor_median_ratio': float(np.max(psd_raw[band]) / max(np.median(psd_raw[neighbors]), 1e-30)),
            }
        interior_ecg = slice(int(5 * fs), -int(5 * fs))
        difference = ecg[interior_ecg] - ecg50[interior_ecg]
        rms50 = np.sqrt(np.mean(ecg50[interior_ecg]**2))
        qc['ECG_final_band_relative_rms_change_vs_50Hz_only'] = float(np.sqrt(np.mean(difference**2)) / max(rms50, 1e-30))
        peaks50 = qrs_candidates(ecg50, fs)
        qc['QRS_comparison_vs_50Hz_only'] = pair_match(peaks50, peaks, fs)
        qc['QRS_count_50Hz_only'] = len(peaks50)
        qc.update(peak_metrics(peaks, fs, len(frame)))
        qc['BEI_raw_median_ohm'] = float(np.median(z))
        qc['BEI_raw_p01_p99_span_ohm'] = float(np.quantile(z, .99) - np.quantile(z, .01))
        qc['BEI_longest_extreme_run_s'] = max(longest_run(z == z.min(), fs), longest_run(z == z.max(), fs))
        interior = slice(int(5 * fs), -int(5 * fs))
        qc['BEI_pulse_band_rms_mohm'] = float(1000 * np.sqrt(np.mean(pulse[interior]**2)))
        qc['BEI_band_rms_sensitivity_mohm'] = {}
        for low, high in PARAMS['bei_sensitivity_bands_hz']:
            variant = zero_phase(signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos'), z, fs, 5)
            qc['BEI_band_rms_sensitivity_mohm'][f'{low:g}-{high:g}'] = float(1000 * np.sqrt(np.mean(variant[interior]**2)))
        qc['BEI_reconstruction_max_error_ohm'] = float(np.max(np.abs(z - (slow + pulse + other))))
        if qc['BEI_reconstruction_max_error_ohm'] > 1e-10:
            raise ValueError('BEI components do not reconstruct raw input')
        stats['channels'][str(ch)] = qc
        for label, value in [('ecg_raw', ecg_raw), ('ecg_notch', notch), ('ecg_filtered', ecg),
                             ('ecg_notch_50Hz_only', notch50), ('ecg_filtered_50Hz_only', ecg50),
                             ('bei_raw', z), ('bei_slow', slow), ('bei_residual', residual),
                             ('bei_pulse_band', pulse), ('bei_other', other), ('resp_activity', activity),
                             ('qrs_samples', peaks), ('qrs_samples_50Hz_only', peaks50), ('psd_frequency_hz', freq),
                             ('psd_ecg_raw', psd_raw), ('psd_ecg_notch', psd_notch), ('psd_ecg_notch_50Hz_only', psd50)]:
            arrays[f'{label}_{ch}'] = value
    stats['qrs_pair_match'] = pair_match(*peak_sets, fs)
    qualities = []
    for ch in (1, 2):
        qc = stats['channels'][str(ch)]
        cv = qc['RR_candidate_cv']
        short = qc['short_RR_fraction_below_0_45_s']
        score = (cv if cv is not None else 10) + 2 * (short if short is not None else 1)
        quality = (qc['candidate_count'] >= 10 and cv is not None and cv < .35
                   and qc['max_candidate_gap_s_including_record_edges'] < 3)
        qualities.append((score, ch, quality))
    viable = [q for q in qualities if q[2]]
    stats['selected_ecg_channel_candidate'] = min(viable)[1] if viable else None
    stats['flags'] = []
    if stats['selected_ecg_channel_candidate'] is None:
        stats['flags'].append('qrs_detection_unreliable_both_channels')
    if rid in ('SCV2-G-005', 'SCV2-G-006'):
        stats['flags'].append('sampling_rate_1000_vs_400_unresolved')
    if rid == 'SCV2-G-011':
        stats['flags'].append('large_high_frequency_ecg_interference_both_channels')
    if rid == 'SCV2-G-010':
        stats['flags'].append('large_high_frequency_ecg_interference_channel_1')
    if stats['qrs_pair_match']['F1'] < .65:
        stats['flags'].append('qrs_cross_channel_disagreement')
    if min(c['BEI_raw_median_ohm'] for c in stats['channels'].values()) < 5:
        stats['flags'].append('low_BEI_level_observed_channel_role_unknown_not_inactive_proof')
    np.savez_compressed(out / (key + '.npz'), **arrays)
    return stats


def compact_trace(x, y, limit=3500):
    """Min/max envelope preserves extrema for the full-record raw preview."""
    if len(x) <= limit:
        return x, y
    size = int(np.ceil(len(x) / (limit / 2)))
    positions = [0, len(x) - 1]
    for start in range(0, len(x), size):
        block = y[start:start + size]
        positions.extend([start + int(np.argmin(block)), start + int(np.argmax(block))])
    indices = np.asarray(sorted(set(positions)))
    return x[indices], y[indices]


def breath_regions(annotation, fs, n):
    if not annotation or not annotation.get('boundaries_samples'):
        if annotation and annotation.get('single_mode'):
            return [(0, n / fs, MODE_NAMES[0])]
        return []
    edges = [0, *annotation['boundaries_samples'], n]
    return [(edges[i] / fs, edges[i + 1] / fs, MODE_NAMES[i]) for i in range(4)]


def render_record(stats, out, annotation):
    key = stats['key']; fs = stats['fs_hz_scenario']; rid = stats['record_id']
    with np.load(out / (key + '.npz')) as stored:
        a = {k: stored[k] for k in stored.files}
    t = a['time_s']; n = len(t)
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=.045,
                        subplot_titles=['BEI 1: исходный и медленная составляющая, Ом',
                                        'BEI 2: исходный и медленная составляющая, Ом',
                                        'BEI: полоса 0,7–12 Гц, мОм — кандидат пульсовой составляющей',
                                        'ЭКГ: режекторы 50/100/150 Гц + 0,5–40 Гц; QRS-кандидаты, мВ',
                                        'Активность медленного BEI: RMS производной за 4 с, Ом/с'])
    ecg_trace_ids = {}; marker_ids = {}
    for ch in (1, 2):
        color = COLORS[ch - 1]
        for label, dash in [('bei_raw', 'solid'), ('bei_slow', 'dash')]:
            tx, yy = compact_trace(t, a[f'{label}_{ch}'])
            fig.add_trace(go.Scatter(x=tx.tolist(), y=yy.tolist(), name=f'BEI{ch} '+('сырой' if label == 'bei_raw' else 'НЧ 0,5 Гц'),
                                    line={'color': color if label == 'bei_slow' else '#94a3b8', 'width': 1, 'dash': dash}), row=ch, col=1)
        stride = int(fs / 100)
        pulse = signal.resample_poly(a[f'bei_pulse_band_{ch}'], 1, stride)
        fig.add_trace(go.Scattergl(x=(np.arange(len(pulse))*stride/fs).tolist(), y=(1000*pulse).tolist(),
                                  name=f'BEI{ch} 0,7–12 Гц', line={'color':color, 'width':1}), row=3, col=1)
        stride = max(1, int(fs / 200))
        ecg = signal.resample_poly(a[f'ecg_filtered_{ch}'], 1, stride)
        ecg_trace_ids[ch] = len(fig.data)
        fig.add_trace(go.Scattergl(x=(np.arange(len(ecg))*stride/fs).tolist(), y=ecg.tolist(), name=f'ЭКГ{ch} фильтр',
                                  line={'color':color, 'width':1}), row=4, col=1)
        for label, title in [('ecg_raw','сырая'), ('ecg_notch_50Hz_only','только режектор 50 Гц'),
                             ('ecg_notch','режекторы 50/100/150 Гц'), ('ecg_filtered_50Hz_only','50 Гц + 0,5–40 Гц, сравнение')]:
            tx, yy = compact_trace(t, a[f'{label}_{ch}'])
            fig.add_trace(go.Scattergl(x=tx.tolist(), y=yy.tolist(), name=f'ЭКГ{ch} {title}',
                                      visible='legendonly', line={'color':color,'width':.7}),row=4,col=1)
        peaks = a[f'qrs_samples_{ch}'].astype(int)
        marker_ids[ch] = len(fig.data)
        fig.add_trace(go.Scatter(x=(peaks/fs).tolist(), y=a[f'ecg_filtered_{ch}'][peaks].tolist(),mode='markers',
                                name=f'QRS{ch} кандидаты',marker={'color':color,'size':6,'symbol':'x'}),row=4,col=1)
        tx, yy = compact_trace(t, a[f'resp_activity_{ch}'])
        fig.add_trace(go.Scatter(x=tx.tolist(),y=yy.tolist(),name=f'Активность BEI{ch}',line={'color':color}),row=5,col=1)
    regions = breath_regions(annotation, fs, n)
    for i, (start, end, name) in enumerate(regions):
        for row in (1, 2, 3, 4, 5):
            fig.add_vrect(x0=start, x1=end, fillcolor=['#dbeafe','#dcfce7','#ffedd5','#f3e8ff'][i],
                          opacity=.25, line_width=0, layer='below', row=row,col=1)
    if annotation:
        for left, right in annotation.get('boundary_uncertainty_ranges_samples', []):
            for row in (1,2,3,4,5):
                fig.add_vrect(x0=left/fs,x1=right/fs,fillcolor='#94a3b8',opacity=.18,
                              line_width=0,layer='below',row=row,col=1)
    fig.update_layout(height=1150,margin={'t':70,'b':50,'l':70,'r':30},template='plotly_white',
                      title=f'{rid} · {stats["montage_label"]} · fs={fs:g} Гц, сценарий · кандидатная разметка',
                      uirevision=key,legend={'orientation':'h','y':-.07},hovermode='x unified')
    fig.update_xaxes(matches='x',range=[0,n/fs]);fig.update_xaxes(title_text='Условное время от начала TXT, с',row=5,col=1)
    graph = fig.to_html(full_html=False,include_plotlyjs=False,div_id='signals',config={'responsive':True,'scrollZoom':True})
    spectrum=make_subplots(rows=1,cols=2,subplot_titles=['ЭКГ1: спектр до/после режектора','ЭКГ2: спектр до/после режектора'])
    for ch in (1,2):
        for field,title,color in [('psd_ecg_raw','исходный','#94a3b8'),
                                  ('psd_ecg_notch_50Hz_only','после 50 Гц','#8b5cf6'),
                                  ('psd_ecg_notch','после 50/100/150 Гц',COLORS[ch-1])]:
            spectrum.add_trace(go.Scatter(x=a[f'psd_frequency_hz_{ch}'].tolist(),
                               y=np.maximum(a[f'{field}_{ch}'],1e-20).tolist(),name=f'ЭКГ{ch}: '+title,
                               line={'color':color}),row=1,col=ch)
    spectrum.update_layout(height=340,template='plotly_white',margin={'t':60,'b':45})
    spectrum.update_xaxes(title_text='Гц, выбранный временной сценарий',range=[0,min(fs/2,160)])
    spectrum.update_yaxes(type='log',title_text='мВ²/Гц')
    graph+=spectrum.to_html(full_html=False,include_plotlyjs=False,div_id='spectrum',config={'responsive':True})
    boundaries = annotation.get('boundaries_samples', []) if annotation else []
    times = [b/fs for b in boundaries]
    editor = {
        'record_id':rid, 'source_sha256':stats['source_sha256'], 'fs_hz_scenario':fs,
        'sampling_rate_confirmed':False, 'status':STATUS,
        'event_kind':'QRS_dominant_extremum_candidate_R_not_confirmed',
        'qrs_samples_by_channel':{str(ch):a[f'qrs_samples_{ch}'].astype(int).tolist() for ch in (1,2)},
        'breathing_candidate':annotation or {},
        'physical_channel_roles':{'1':'unknown','2':'unknown'},
        'channel_mapping_constraints':annotation.get('channel_mapping_constraints', {}) if annotation else {},
        'channel_role_scenarios':annotation.get('channel_role_scenarios', {}) if annotation else {},
        'filters':PARAMS,'reviewer':'','comments':'',
    }
    if rid in ('SCV2-G-006','SCV2-G-011'):
        editor['physical_channel_roles']={'1':'lateral_direction_unknown','2':'lateral_direction_unknown'}
    payload=json.dumps(editor,ensure_ascii=False).replace('</','<\\/')
    controls=''.join(f'<label>{title}<input id="b{i}" type="number" step="0.1" value="{times[i] if len(times)>i else ""}"></label>'
                     for i,title in enumerate(['Конец спокойного','Конец задержки на вдохе','Конец усиленного']))
    observed=html.escape(annotation.get('observed_features','Не размечено') if annotation else 'Не размечено')
    hypothesis=html.escape(json.dumps(annotation.get('channel_role_hypothesis',{}) if annotation else {},ensure_ascii=False))
    scenarios=html.escape(json.dumps(annotation.get('channel_role_scenarios',{}) if annotation else {},ensure_ascii=False))
    role_options='<option value="unknown">Не установлено</option><option value="ttrkg">ТТРКГ</option><option value="lateral_horizontal">Боковая горизонтальная</option><option value="lateral_vertical">Боковая вертикальная</option><option value="lateral_direction_unknown">Боковая, направление неизвестно</option><option value="TEPC">ТЕПС</option><option value="AEPC">АЕПС</option>'
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Просмотр Scorpion v2</title>
<style>body{font:15px system-ui;margin:22px;color:#172033}a{color:#2459b5}.note{background:#fff7dd;padding:12px;border-radius:8px}.controls{display:flex;gap:12px;flex-wrap:wrap}label{display:grid;gap:4px}input,select,button{font:inherit;padding:7px;border:1px solid #b5bfce;border-radius:6px}textarea{width:95%;height:60px}h2{font-size:19px}.table{overflow:auto}table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ddd}</style>
<script src="plotly.min.js"></script><a href="index.html">← Все записи</a>
<p class="note">Кандидатная разметка. Частота отсчётов не подтверждена по файлу. Полосы BEI разделяют частоты, а не ткани. Края ±5 с для BEI и ±1 с для ЭКГ требуют отдельного внимания. Подтверждённых R-зубцов и физиологических заключений здесь нет.</p>'''
    page+=graph+f'<h2>Кандидат дыхательной разметки</h2><p>{observed}</p><div class="controls">{controls}</div>'
    if annotation and annotation.get('channel_role_scenarios'):
        page+=f'<p>По подтверждению автора горизонтальная сборка и ТТРКГ сохраняли свои каналы во всём горизонтальном ряду. В кресте канал ТТРКГ подключён к вертикальной сборке, горизонтальный сохранён. Числовое назначение не установлено; два общих для серии сценария: {scenarios}. Прежняя гипотеза смены каналов отвергнута. Выбор ниже остаётся кандидатом и требует согласования со всей серией.</p>'
    else:
        page+=f'<p>Гипотеза по согласованности записей: {hypothesis}. Она не подтверждает подключение.</p>'
    page+=f'<p>Назначение каналов для ручного уточнения:</p><div class="controls"><label>Канал 1<select id="role1">{role_options}</select></label><label>Канал 2<select id="role2">{role_options}</select></label></div>'
    page+='''<h2>Проверка событий ЭКГ</h2><p>Увеличьте участок ЭКГ. В режиме редактирования щелчок по кривой добавляет или удаляет ближайший QRS-кандидат. События не становятся принятыми автоматически.</p><div class="controls"><label>Канал событий<select id="eventCh"><option value="1">ЭКГ1</option><option value="2">ЭКГ2</option></select></label><label>Режим<select id="editMode"><option value="view">Просмотр</option><option value="add">Добавить событие</option><option value="remove">Удалить событие рядом</option></select></label><label>Автор проверки<input id="reviewer"></label><button id="export">Скачать исправленную кандидатную разметку JSON</button></div><p><textarea id="comments" placeholder="Основания границ, неоднозначные участки, сведения о подключении"></textarea></p>'''
    page+='<script>const annotation='+payload+';const ecgIds='+json.dumps(ecg_trace_ids)+';const markerIds='+json.dumps(marker_ids)+';const fs='+str(fs)+';const n='+str(n)+';'
    page+='''for(let ch of [1,2])document.getElementById('role'+ch).value=annotation.physical_channel_roles[ch];
for(let i of [0,1,2])document.getElementById('b'+i).onchange=()=>{let edges=[0,1,2].map(j=>Number(document.getElementById('b'+j).value));if(!(edges[0]>0&&edges[0]<edges[1]&&edges[1]<edges[2]&&edges[2]<n/fs))return;let bounds=[0,...edges,n/fs],shapes=[];for(let stage=0;stage<4;stage++)for(let row=1;row<=5;row++)shapes.push({type:'rect',xref:'x'+(row===1?'':row),yref:'y'+(row===1?'':row)+' domain',x0:bounds[stage],x1:bounds[stage+1],y0:0,y1:1,fillcolor:['#dbeafe','#dcfce7','#ffedd5','#f3e8ff'][stage],opacity:.25,line:{width:0},layer:'below'});Plotly.relayout(document.getElementById('signals'),{shapes});};
const graph=document.getElementById('signals');graph.on('plotly_click',e=>{let mode=document.getElementById('editMode').value;if(mode==='view')return;let ch=document.getElementById('eventCh').value;let pt=e.points.find(p=>p.curveNumber===ecgIds[ch]||p.curveNumber===markerIds[ch]);if(!pt)return;let sample=Math.max(0,Math.min(n-1,Math.round(pt.x*fs)));let peaks=annotation.qrs_samples_by_channel[ch];let ys=graph.data[markerIds[ch]].y.slice();if(mode==='add'){if(peaks.some(p=>Math.abs(p-sample)<.03*fs))return;peaks.push(sample);ys.push(pt.y);}else{if(!peaks.length)return;let distances=peaks.map(p=>Math.abs(p-sample));let idx=distances.indexOf(Math.min(...distances));if(distances[idx]>.2*fs)return;peaks.splice(idx,1);ys.splice(idx,1);}let ordered=peaks.map((p,i)=>[p,ys[i]]).sort((a,b)=>a[0]-b[0]);annotation.qrs_samples_by_channel[ch]=ordered.map(p=>p[0]);Plotly.restyle(graph,{x:[ordered.map(p=>p[0]/fs)],y:[ordered.map(p=>p[1])]},[markerIds[ch]]);});
document.getElementById('export').onclick=()=>{
 let values=[0,1,2].map(i=>document.getElementById('b'+i).value);
 if(values.some(Boolean)&&!values.every(Boolean)){alert('Заполните все три границы либо оставьте все пустыми');return;}
 let edges=values.every(Boolean)?values.map(Number):[];
 if(edges.length&&!(edges[0]>0&&edges[0]<edges[1]&&edges[1]<edges[2]&&edges[2]<n/fs)){alert('Границы должны возрастать и лежать внутри записи');return;}
 let result=JSON.parse(JSON.stringify(annotation));
 let samples=edges.map(t=>Math.round(t*fs));
 let changed=JSON.stringify(samples)!==JSON.stringify(result.breathing_candidate.boundaries_samples||[]);
 result.reviewer=document.getElementById('reviewer').value;
 if(changed){
  result.prior_breathing_candidate=JSON.parse(JSON.stringify(result.breathing_candidate));
  result.breathing_candidate.boundary_uncertainty_ranges_samples=[];
  result.breathing_candidate.confidence='unrated_after_browser_edit';
  result.breathing_candidate.method='candidate_boundary_edit_in_browser';
  result.breathing_candidate.annotator=result.reviewer||'unspecified';
  result.breathing_candidate.decision_date=new Date().toISOString().slice(0,10);
  delete result.breathing_candidate.single_mode;
 }
 result.breathing_candidate.boundaries_samples=samples;
 result.breathing_candidate.edited_with_timebase_hz_scenario=fs;
 result.breathing_candidate.status='pending_manual_review';
 result.breathing_candidate.author_accepted=false;
 result.comments=document.getElementById('comments').value;
 result.physical_channel_roles={'1':document.getElementById('role1').value,'2':document.getElementById('role2').value};
 result.edited_in_browser=true;
 let blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
 let link=document.createElement('a');link.href=URL.createObjectURL(blob);
 link.download=result.record_id+'_fs'+fs+'_review_candidate.json';link.click();URL.revokeObjectURL(link.href);
};</script></html>'''
    (out/(key+'.html')).write_text(page,encoding='utf-8')
    # Static overview used to inspect every record, with axes in physical units.
    fig,axes=plt.subplots(4,1,figsize=(14,9),sharex=True)
    for ch in (1,2):
        color=COLORS[ch-1]
        for axis,values in zip(axes,[a[f'bei_slow_{ch}']-np.median(a[f'bei_slow_{ch}']),
                                    1000*a[f'bei_pulse_band_{ch}'],a[f'ecg_filtered_{ch}'],a[f'resp_activity_{ch}']]):
            tx,yy=compact_trace(t,values);axis.plot(tx,yy,color=color,lw=.7,label=f'канал {ch}')
    for axis,title in zip(axes,['Медленный BEI − медиана, Ом','BEI 0,7–12 Гц, мОм','ЭКГ после фильтров, мВ','RMS производной BEI, Ом/с']):
        axis.set_ylabel(title);axis.grid(alpha=.25);axis.legend(loc='upper right')
        for i,(start,end,_) in enumerate(regions):axis.axvspan(start,end,color=['#dbeafe','#dcfce7','#ffedd5','#f3e8ff'][i],alpha=.35)
    axes[-1].set_xlabel(f'Время, с; сценарий fs={fs:g} Гц');fig.suptitle(rid+' · '+stats['montage_label']);fig.tight_layout()
    fig.savefig(out/(key+'.png'),dpi=120);plt.close(fig)


def render_index(summary,out,annotations):
    (out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    rows=[]
    for stats in summary['records']:
        rid=stats['record_id']; ch=stats['selected_ecg_channel_candidate']
        q=stats['channels'][str(ch)] if ch else None
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in [
            f'<a href="{stats["key"]}.html">{rid}</a>',html.escape(stats['montage_label']),
            str(stats['fs_hz_scenario']),f'{stats["duration_s_N_over_fs"]:.3f}',
            f'ЭКГ{ch}: {q["candidate_count"]}' if ch else 'Надёжный кандидат канала не выбран',
            f'{stats["qrs_pair_match"]["F1"]:.2f}',html.escape(', '.join(stats['flags']) or 'Автоматических флагов нет'),
            html.escape(annotations.get(rid,{}).get('confidence','не размечено'))])+'</tr>')
    text='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Scorpion v2 — просмотр и разметка</title><style>body{font:15px system-ui;margin:28px;color:#172033;max-width:1500px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #dbe1e8;text-align:left}th{background:#f3f6fa}a{color:#2459b5}.note{background:#fff4ce;padding:16px;border-radius:8px}li{margin:8px 0}code{background:#f3f6fa;padding:2px 5px}</style>
<h1>Scorpion v2 — просмотр сигналов и кандидатная разметка</h1><p>Георг · BMSTU · 05.10.2026 · 18 TXT. BIN — настройки программы, в анализ сигналов не включены.</p>
<p class="note">Все результаты остаются кандидатными. Автор подтвердил постоянное распределение ролей каналов в горизонтальном ряду; в кресте канал ТТРКГ заменён вертикальной сборкой, горизонтальный сохранён. Числовые номера ролей ещё не установлены. Прежняя гипотеза смены каналов отвергнута. Для 005 и 006 показаны сценарии 1000 и 400 Гц. Частотное разложение BEI не разделяет ткани. QRS-кандидаты не являются принятой R-разметкой.</p>
<h2>Фильтры и сохранённые данные</h2><ul><li>ЭКГ: режекторы 50, 100 и 150 Гц, Q=30 каждый, затем Баттерворт 4 порядка 0,5–40 Гц; детектор QRS использует полосу 5–25 Гц. Прежний вариант только с 50 Гц сохранён для сравнения. <a href="harmonics_review.html">Проверка гармоник и влияния на QRS</a>.</li><li>BEI: медленная составляющая НЧ 0,5 Гц; кандидат пульсовой составляющей 0,7–12 Гц; дополнительный остаток сохраняется. Сумма медленной, полосовой и дополнительной составляющих воспроизводит исходный BEI.</li><li>Фильтрация вперёд и назад, без введения временного сдвига от однопроходного фильтра. Это не устраняет неизвестные задержки прибора. BEI: дополнение 5 с и предостережение для краёв ±5 с; ЭКГ: дополнение 3 с и исключение краёв ±1 с из детектора.</li><li>NPZ содержит все исходные столбцы, полную временную сетку и вычисленные составляющие. В HTML используются min/max-превью сырых кривых и отображение отфильтрованных сигналов после антиалиасного прореживания; расчёты выполняются на полной сетке.</li></ul>
<p>Общее описание автора: спокойное дыхание → задержка на вдохе → усиленное дыхание → задержка на выдохе. Цветные зоны — реконструкция по форме сигнала и этому порядку; неоднозначные границы требуют проверки автора. В каждой записи можно увеличить сигналы, исправить границы, отредактировать QRS и скачать кандидатный JSON.</p>
<p>Время от начала каждого TXT условное, абсолютная синхронизация разных файлов не назначена. Флаги — основания для просмотра, а не доказанные аппаратные или физиологические диагнозы. <a href="sampling_review.html">Отдельная проверка исходных сигналов, временной шкалы и повторов 140 мм</a>.</p>
<table><thead><tr><th>Запись</th><th>Метка</th><th>fs, Гц, сценарий</th><th>N/fs, с</th><th>QRS-кандидаты выбранного поля</th><th>Совпадение двух ЭКГ, F1</th><th>Флаги</th><th>Дыхание: уверенность</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<h2>Происхождение</h2><p>Полные параметры, версии библиотек, хэши входов и кода, метрики обоих каналов и статус каждой записи: <a href="summary.local.json">summary.local.json</a>. Разметка дыхания и событий: <a href="annotations_candidates.local.json">annotations_candidates.local.json</a>.</p>
<p>Определения цифровых фильтров: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html">SciPy iirnotch</a>, <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html">SciPy sosfiltfilt</a>. Границы полос и критерии отбора кандидатов — рабочие решения текущего разбора, не клинически валидированный протокол.</p></html>'''
    (out/'index.html').write_text(text,encoding='utf-8')


def sampling_review(summary, out, inventory):
    selected=[]
    for rid in ['SCV2-G-004','SCV2-G-005','SCV2-G-006','SCV2-G-007']:
        fs=400 if rid in ('SCV2-G-005','SCV2-G-006') else 1000
        selected.append(next(r for r in summary['records'] if r['record_id']==rid and r['fs_hz_scenario']==fs))
    fig,axes=plt.subplots(4,3,figsize=(18,11)); evidence=[]
    for row,stats in enumerate(selected):
        ch=2 if stats['record_id']=='SCV2-G-007' else 1
        with np.load(out/(stats['key']+'.npz')) as a:
            x=a[f'ecg_raw_{ch}'];y=a[f'ecg_filtered_{ch}'];peaks=a[f'qrs_samples_{ch}']
        f,p=signal.welch(x,fs=1.0,nperseg=min(len(x),16000))
        mask=(f>=.03)&(f<=.2); dominant=float(f[mask][np.argmax(p[mask])])
        spacing=float(np.median(np.diff(peaks))) if len(peaks)>1 else None
        evidence.append({'record_id':stats['record_id'],'field':f'EMG{ch}',
                         'dominant_raw_spectral_line_cycles_per_sample':dominant,
                         'line_period_samples':1/dominant,
                         'fs_hz_if_line_is_50Hz':50/dominant,
                         'candidate_qrs_median_spacing_samples':spacing,
                         'inference_status':'conditional_line_identity_not_verified'})
        start=5000
        axes[row,0].plot(np.arange(400),x[start:start+400]-np.median(x[start:start+400]),lw=.8)
        axes[row,0].set_title(stats['record_id']+f' EMG{ch}, исходный − медиана');axes[row,0].set_ylabel('мВ')
        axes[row,1].plot(np.arange(2500),y[start:start+2500],lw=.8)
        within=peaks[(peaks>=start)&(peaks<start+2500)]
        axes[row,1].scatter(within-start,y[within],s=14,color='#d97706')
        axes[row,1].set_title(f'ЭКГ после фильтров при сценарии {stats["fs_hz_scenario"]} Гц')
        axes[row,2].semilogy(f[mask],p[mask]);axes[row,2].axvline(.05,color='#16a34a',ls='--');axes[row,2].axvline(.125,color='#dc2626',ls='--')
        axes[row,2].set_title(f'Сильная линия: {dominant:.3f} цикла/отсчёт')
        for axis in axes[row]:axis.grid(alpha=.3)
        for axis in axes[row,:2]:axis.set_xlabel('Номер отсчёта в показанном окне')
        axes[row,2].set_xlabel('Циклов/отсчёт — не требует знания fs')
    fig.tight_layout();fig.savefig(out/'sampling_review.png',dpi=110);plt.close(fig)
    signals=[r for r in inventory['files'] if r.get('signal_input_candidate')]
    exact_duplicates=[]
    for item in signals:
        same=[other['record_id'] for other in signals if other['sha256']==item['sha256']]
        if len(same)>1 and same not in exact_duplicates:exact_duplicates.append(same)
    facts={'status':STATUS,'evidence':evidence,'exact_file_duplicate_groups':exact_duplicates,
           'vertical_140_label_count':sum('140 верт' in f['archive_member'] for f in signals),
           'cross_140_label_count':sum('140 крест' in f['archive_member'] for f in signals),
           'search_scope':'uploaded_archive_and_matching_file_names_in_workspace; archive_extraction_is_same_copy_not_independent_record',
           'timebase_not_accepted':True}
    write_json(out/'sampling_review.local.json',facts)
    table=''.join(f'<tr><td>{e["record_id"]}</td><td>{e["line_period_samples"]:.2f}</td><td>{e["fs_hz_if_line_is_50Hz"]:.1f}</td><td>{e["candidate_qrs_median_spacing_samples"]:.0f}</td></tr>' for e in evidence)
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка частоты и повторов</title><style>body{font:16px system-ui;max-width:1500px;margin:25px;color:#172033}td,th{border:1px solid #ddd;padding:9px}table{border-collapse:collapse}img{width:100%}</style><a href="index.html">← Все записи</a><h1>140 мм: исходные сигналы, временная шкала и повторы</h1>
<p>В загрузке по одному TXT с метками «140 верт 1х4» и «140 крест». Точных файловых дублей среди 18 TXT нет. Извлечённые файлы и те же члены исходного ZIP являются одной копией данных. Три записи «140 горизонт» используют 1×1, 2×2 и 1×4 на руках; это соседние конфигурации, а не повторы вертикального или перекрёстного монтажа. У тестовой записи 140 мм ориентация не установлена.</p>
<p>В 005 и 006 сильная линия исходной ЭКГ имеет период около 8 отсчётов; у соседних 004 и 007 — около 20. <strong>Если это одна и та же сетевая помеха 50 Гц</strong>, получаются соответственно 400 и 1000 Гц. Отнесение линии к сети остаётся гипотезой. Дополнительные признаки: сравнимые дыхательные стадии и более согласованные QRS-кандидаты при сценарии 400 Гц. Они поддерживают гипотезу, но не заменяют настройки прибора.</p>
<table><tr><th>Запись</th><th>Период линии, отсчётов</th><th>fs, если линия 50 Гц</th><th>Промежуток QRS-кандидатов, отсчётов</th></tr>'''+table+'''</table><p>При 1000 Гц длительности 005/006 равны 32,425/32,897 с; при 400 Гц — 81,063/82,243 с. Обе шкалы сохранены раздельно. Границы дыхания записаны в отсчётах.</p><img src="sampling_review.png"><p>Сырые кривые показаны после вычитания медианы только для просмотра; первичные уровни сохранены. Фильтрованные кривые вычислены в явно указанном сценарии. Маркеры — QRS-кандидаты. Полные числа и происхождение: <a href="sampling_review.local.json">локальный JSON</a>.</p></html>'''
    (out/'sampling_review.html').write_text(page,encoding='utf-8')


def render_harmonics_review(summary, out):
    rows = []
    comparison = []
    for record in summary['records']:
        for ch in (1, 2):
            qc = record['channels'][str(ch)]
            lines = qc['mains_notch_comparison']
            peak = ' / '.join(f'{lines[str(hz)]["raw_peak_to_neighbor_median_ratio"]:.1f}' for hz in (50, 100, 150))
            rejection = ' / '.join(f'{lines[str(hz)]["reduction_dB_against_raw"]:.1f}' for hz in (50, 100, 150))
            change = 100 * qc['ECG_final_band_relative_rms_change_vs_50Hz_only']
            count_before = qc['QRS_count_50Hz_only']; count_after = qc['candidate_count']
            agreement = qc['QRS_comparison_vs_50Hz_only']['F1']
            rows.append(f'<tr><td><a href="{record["key"]}.html">{record["record_id"]}</a></td>'
                        f'<td>{record["fs_hz_scenario"]}</td><td>{ch}</td><td>{peak}</td>'
                        f'<td>{rejection}</td><td>{change:.3f}</td><td>{count_before} → {count_after}</td>'
                        f'<td>{agreement:.3f}</td></tr>')
            comparison.append({'record_id': record['record_id'], 'fs_hz_scenario': record['fs_hz_scenario'],
                               'ecg_channel': ch, 'source_sha256': record['source_sha256'],
                               'mains_bands': lines, 'relative_rms_change_final_band': change / 100,
                               'qrs_count_before': count_before, 'qrs_count_after': count_after,
                               'qrs_match': qc['QRS_comparison_vs_50Hz_only']})
    write_json(out / 'harmonics_review.local.json', {'status': STATUS, 'filters': PARAMS,
               'code_sha256': summary['code_sha256'], 'records': comparison})
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Гармоники сети Scorpion v2</title>
<style>body{font:15px system-ui;margin:24px;color:#172033}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}a{color:#2459b5}.table{overflow:auto}</style>
<a href="index.html">← Все записи</a><h1>Режекторы ЭКГ 50, 100 и 150 Гц</h1>
<p>По предложению автора добавлены 100 и 150 Гц, Q=30 для каждого режектора. Все три частоты ниже Найквиста в обоих рассматриваемых сценариях, 400 и 1000 Гц. Исходные поля сохранены.</p>
<p>В NPZ и на страницах записей сохранены варианты «только 50 Гц» и «50/100/150 Гц» до полосового фильтра и после одинаковой полосы 0,5–40 Гц. Спектр показывает все три кривые: исходную и два варианта режекции. Полоса уже подавляет высокие частоты; добавление режекторов не гарантирует улучшение выделения QRS.</p>
<p>Пик/фон — максимум исходной спектральной плотности в полосе ±1 Гц вокруг указанной частоты, делённый на медиану в соседних полосах от 3 до 6 Гц. Это наблюдение спектра, не доказательство сетевого происхождения. Подавление вычислено по интегральной мощности в той же полосе ±1 Гц. Если исходной линии нет, число характеризует фильтр и фон.</p>
<p>ΔRMS — RMS разности конечных кривых, делённый на RMS варианта только с 50 Гц, на внутренней части записи без первых/последних 5 с. F1 сравнивает кандидатные события в пределах 80 мс; это не проверка относительно истинных R-зубцов. Спорные временные шкалы сохранены раздельно.</p>
<div class="table"><table><tr><th>Запись</th><th>fs, Гц</th><th>ЭКГ</th><th>Пик/фон 50 / 100 / 150</th><th>Подавление, дБ 50 / 100 / 150</th><th>ΔRMS, %</th><th>QRS: 50 → 50/100/150</th><th>F1 кандидатов</th></tr>'''
    page += ''.join(rows) + '</table></div><p><a href="harmonics_review.local.json">Полные числа и происхождение</a></p></html>'
    (out / 'harmonics_review.html').write_text(page, encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args(); experiment=args.experiment.resolve()
    out=experiment/'derived/scorpion_review';out.mkdir(parents=True,exist_ok=True)
    ann_dir=experiment/'annotations/scorpion_review';ann_dir.mkdir(parents=True,exist_ok=True)
    inv_path=experiment/'metadata/inventory.local.json'
    inventory=json.loads(inv_path.read_text(encoding='utf-8'))
    annotation_source=ann_dir/'breathing_review.local.json'
    annotations=json.loads(annotation_source.read_text(encoding='utf-8'))['records'] if annotation_source.exists() else {}
    if args.render_only:
        summary=json.loads((out/'summary.local.json').read_text(encoding='utf-8'))
    else:
        records=[]
        for info in inventory['files']:
            if not info.get('signal_input_candidate'):continue
            for fs in ([1000,400] if info['record_id'] in ('SCV2-G-005','SCV2-G-006') else [1000]):
                records.append(analyze_record(experiment,info,fs,out))
                print(info['record_id'],fs,'processed',flush=True)
        summary={'experiment_id':inventory['experiment_id'],'status':STATUS,
                 'analysis_utc':datetime.now(timezone.utc).isoformat(),
                 'source_archive_sha256':inventory['source_archive_sha256'],
                 'inventory_sha256':sha256_file(inv_path),'code_sha256':sha256_file(__file__),
                 'channel_mapping_constraints': inventory.get('channel_mapping_constraints', {}),
                 'lateral_nominal_geometry': inventory.get('lateral_nominal_geometry', {}),
                 'arm_electrode_group_wiring': inventory.get('arm_electrode_group_wiring', {}),
                 'TEPC_AEPC_reference': inventory.get('TEPC_AEPC_reference', {}),
                 'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
                 'filters':PARAMS,'primary_inputs_unchanged':True,'records':records}
        write_json(out/'summary.local.json',summary)
        (out/'analysis_source.local.py').write_bytes(Path(__file__).read_bytes())
    candidates=[]
    for stats in summary['records']:
        with np.load(out/(stats['key']+'.npz')) as a:
            events={str(ch):a[f'qrs_samples_{ch}'].astype(int).tolist() for ch in (1,2)}
        candidates.append({'record_id':stats['record_id'],'fs_hz_scenario':stats['fs_hz_scenario'],
                           'source_sha256':stats['source_sha256'],'status':STATUS,
                           'event_kind':'QRS_dominant_extremum_candidate_R_not_confirmed',
                           'qrs_samples_by_channel':events,'selected_ecg_channel_candidate':stats['selected_ecg_channel_candidate'],
                           'breathing':annotations.get(stats['record_id'],{}),
                           'physical_channel_mapping_status':'relative_roles_author_confirmed_numeric_assignment_unresolved' if annotations.get(stats['record_id'],{}).get('channel_mapping_constraints') else 'numeric_assignment_unresolved',
                           'channel_role_scenarios':annotations.get(stats['record_id'],{}).get('channel_role_scenarios',{})})
        render_record(stats,out,annotations.get(stats['record_id']))
    artifact={'owner':'11.20','status':STATUS,'code_sha256':summary['code_sha256'],
              'filters':PARAMS,'breathing_source_sha256':sha256_file(annotation_source) if annotation_source.exists() else None,
              'records':candidates}
    write_json(out/'annotations_candidates.local.json',artifact)
    write_json(ann_dir/'annotations_candidates.local.json',artifact)
    render_index(summary,out,annotations)
    render_harmonics_review(summary,out)
    sampling_review(summary,out,inventory)
    print('Report:',out/'index.html')


if __name__=='__main__':main()
