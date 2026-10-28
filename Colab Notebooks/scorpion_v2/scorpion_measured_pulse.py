"""Reuse existing Scorpion QRS/inhale artifacts and fixed-centre FEM results.

No annotation, filtering, FEM solve, fitted gain, sign reversal or CT cycle closure.
The inverse is deliberately unavailable for nonmonotone three-volume operators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEM = ROOT / 'Experiments/scorpion_20261022/derived/exploratory/fem_smart_20261026'
NAMES = {'georg': 'Георгий', 'nik': 'Ник', 'yaroslav': 'Ярослав'}
PROFILE_NAMES = {'control': 'Эффективное сердце 4,65', 'blood': 'Сердце как кровь 1,35'}
CASE = 'L140_TT2x2'
CT = ROOT / 'Colab Notebooks/cardiac_4d_segmentation/heart_rr_analysis'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError('Input fingerprint mismatch: ' + str(path))


def cycle_metrics(pulse, events, fs, inhale, origin=0):
    """Five existing boundaries; endpoints included, no newly detected events."""
    pulse = np.asarray(pulse, float)
    events = np.asarray(events, int)
    if len(events) != 5 or np.any(np.diff(events) <= 0):
        raise ValueError('Exactly five ordered QRS boundaries are required')
    if not (inhale[0] <= events[0] < events[-1] < inhale[1]):
        raise ValueError('Four cycles must be inside the existing inhale window')
    rr = np.diff(events) / fs
    if np.any((rr < .45) | (rr > 1.8)) or not np.isfinite(pulse).all():
        raise ValueError('Invalid RR or nonfinite pulse')
    if events[0] < origin or events[-1] - origin >= len(pulse):
        raise ValueError('Selected cycles exceed cached signal')
    grid = np.linspace(0, 1, 201)
    rows, waves = [], []
    for i, (a, b) in enumerate(zip(events[:-1], events[1:])):
        y = pulse[a-origin:b-origin+1]
        phase = np.linspace(0, 1, len(y))
        # Reference subtraction is a contrast, not an instrument offset fit.
        delta = y - y[0]
        waves.append(np.interp(grid, phase, delta))
        lo, hi = int(np.argmin(delta)), int(np.argmax(delta))
        rows.append(dict(cycle=i+1, rr_s=float(rr[i]),
            start_s=float(a/fs), end_s=float(b/fs), ptp_ohm=float(np.ptp(y)),
            minimum_from_R_ohm=float(delta[lo]), maximum_from_R_ohm=float(delta[hi]),
            minimum_phase_rr=float(phase[lo]), maximum_phase_rr=float(phase[hi])))
    return dict(cycles=rows, phase_rr=grid.tolist(), delta_Z_from_R_ohm=waves,
                median_delta_Z_from_R_ohm=np.median(waves, axis=0).tolist())


def volume_operator(rows, v0, q):
    """Actual three-point response, secant and both one-sided finite slopes."""
    points = []
    for factor, name in [(0.95, 'sphere_v095'), (1., 'sphere_v100'), (1.05, 'sphere_v105')]:
        matches = [r for r in rows if r['geometry'] == name and int(r['rule']) == q]
        if len(matches) != 1:
            raise ValueError('Missing/duplicate volume operator')
        r = matches[0]
        points.append(dict(volume_ml=factor*v0, Z_ohm=r['transfers']['ttrkg']['own_ohm']))
    z = np.array([p['Z_ohm'] for p in points])
    v = np.array([p['volume_ml'] for p in points])
    slopes = np.diff(z) / np.diff(v)
    monotone = bool(np.all(slopes > 0) or np.all(slopes < 0))
    return dict(rule=q, points=points, secant_ohm_per_ml=float((z[-1]-z[0])/(v[-1]-v[0])),
        half_slopes_ohm_per_ml=slopes.tolist(), monotone_at_three_points=monotone,
        inverse_unique_at_three_points=monotone,
        tested_volume_span_ml=float(v[-1]-v[0]), tested_Z_span_ohm=float(np.ptp(z)),
        step_convergence_proven=False)


def amplitude_diagnostic(ptp, operator):
    k = operator['secant_ohm_per_ml']
    monotone = operator['monotone_at_three_points']
    equivalent = float(ptp / abs(k)) if monotone and k != 0 else None
    return dict(measured_ptp_ohm=float(ptp),
        measured_over_tested_model_span=float(ptp/operator['tested_Z_span_ohm']),
        equivalent_volume_span_ml=equivalent,
        inverse_status='secant_extrapolation_only' if equivalent is not None else 'blocked_nonmonotone',
        pulse_span_exceeds_sampled_model_span=bool(ptp > operator['tested_Z_span_ohm']),
        equivalent_span_exceeds_tested_volume_span=(equivalent > operator['tested_volume_span_ml'])
        if equivalent is not None else None,
        stroke_volume_estimated=False)


def ct_cycle(frame, operators):
    """Within-trigger-interval changes only. Never join or close CT cycles."""
    f = frame.sort_values('time_after_cycle_r_ms')
    t = f.time_after_cycle_r_ms.to_numpy(float)
    if len(f) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('CT phases must be ordered within their own trigger interval')
    columns = ['whole_heart_ml', 'heart_atrium_left_ml', 'heart_ventricle_left_ml',
               'heart_atrium_right_ml', 'heart_ventricle_right_ml']
    volumes = {c: f[c].to_numpy(float) for c in columns}
    volumes['ventricles_ml'] = volumes['heart_ventricle_left_ml'] + volumes['heart_ventricle_right_ml']
    volumes['chambers_ml'] = sum(volumes[c] for c in columns[1:])
    whole = volumes['whole_heart_ml']
    dw = np.diff(whole)
    comparisons = []
    for key in ['heart_ventricle_left_ml', 'heart_ventricle_right_ml', 'ventricles_ml', 'chambers_ml']:
        delta = np.diff(volumes[key])
        nonzero = (dw != 0) & (delta != 0)
        corr = float(np.corrcoef(dw, delta)[0, 1]) if len(dw) >= 3 and np.std(dw) > 0 and np.std(delta) > 0 else None
        comparisons.append(dict(target=key, n_pairs=len(dw),
            same_sign_pairs=int(np.sum((np.sign(dw) == np.sign(delta)) & nonzero)),
            nonzero_pairs=int(np.sum(nonzero)), delta_correlation=corr,
            rmse_identity_delta_ml=float(np.sqrt(np.mean((dw-delta)**2))),
            target_observed_span_ml=float(np.ptp(volumes[key]))))
    return dict(cycle_index=int(f.cycle_index.iloc[0]), phase_ids=f.phase_id.tolist(),
        time_after_R_ms=t.tolist(), rr_ms=float(f.rr_interval_ms.iloc[0]),
        phase_percent_within_cycle=f.phase_percent_within_cycle.tolist(),
        volumes_ml={k:v.tolist() for k,v in volumes.items()},
        whole_observed_span_ml=float(np.ptp(whole)), comparisons=comparisons,
        temporal_resolution_ms=f.temporal_resolution_ms.tolist(),
        volume_only_projection=[dict(profile=name,
            predicted_delta_Z_from_first_frame_ohm=(op['secant_ohm_per_ml']*(whole-whole[0])).tolist(),
            predicted_span_ohm=float(abs(op['secant_ohm_per_ml'])*np.ptp(whole)),
            extrapolates_tested_volume_span=bool(np.ptp(whole)>op['tested_volume_span_ml']),
            CT_states_outside_tested_absolute_volume=int(np.sum((whole<op['points'][0]['volume_ml'])|(whole>op['points'][-1]['volume_ml']))),
            CT_shape_transfer_validated=False,
            monotone_at_three_points=op['monotone_at_three_points'],
            is_dynamic_FEM=False, filter_matched=False)
            for name, op in operators.items()],
        closed_cycle=False, ED_ES_selected=False, independent_segmentation_validation=False)


def phase_sign_diagnostic(signal_metrics, ct_cycles, profiles):
    """Nik-only descriptive RR-fraction hypothesis, no delay or gain search.

    Same fraction of a separate cardiac interval is NOT a synchronization claim.
    Filtered pulse versus unfiltered CT projection remains an explicit limitation.
    """
    result=[]
    phase=np.asarray(signal_metrics['phase_rr'])
    for ct in ct_cycles:
        x=np.asarray(ct['phase_percent_within_cycle'])/100
        if np.any((x<0)|(x>1)):
            raise ValueError('CT phase outside its own trigger interval')
        measured=np.array([np.interp(x,phase,w) for w in signal_metrics['delta_Z_from_R_ohm']])
        measured=measured-measured[:,0,None]
        for profile in profiles:
            predicted=profile['fine']['secant_ohm_per_ml']*np.array(ct['volumes_ml']['whole_heart_ml'])
            predicted=predicted-predicted[0]
            dp=np.diff(predicted)
            counts=[]
            for w in measured:
                dz=np.diff(w)
                valid=(dp!=0)&(dz!=0)
                counts.append(dict(same_sign_pairs=int(np.sum((np.sign(dp)==np.sign(dz))&valid)),
                    nonzero_pairs=int(np.sum(valid))))
            result.append(dict(ct_cycle_index=ct['cycle_index'],profile=profile['id'],phase_rr=x.tolist(),
                measured_delta_Z_from_first_CT_phase_ohm=measured.tolist(),
                projected_delta_Z_from_first_CT_phase_ohm=predicted.tolist(),sign_counts_by_measured_cycle=counts,
                phase_alignment='same_RR_fraction_across_sessions_unvalidated', filter_matched=False,
                delay_fitted=False, gain_fitted=False))
    return result


def collect():
    pins = {}
    def pin(p, expected=None):
        p = Path(p)
        if expected is not None:
            checked(p, expected)
        pins[p.relative_to(ROOT).as_posix()] = sha(p)
        return p

    model_path = pin(HERE/'_assets/fem_smart_20261026.summary.json')
    models = read(model_path)
    if not models['final_inputs_complete'] or models['source_relative_phase'] != 'not_applicable_time_division':
        raise ValueError('Require completed time-division FEM reader inputs')
    for s in models['sources']:
        pin(FEM/s['relative_path'], s['sha256'])
    pin(HERE/'build_scorpion_fem_smart_report.py', models['generator']['sha256'])
    model_by_name = {s['id']: s for s in models['subjects']}

    # Georg: validate the selected record independently of unrelated late-record
    # protocol revisions. No wholesale acceptance of the archived display contract.
    expg = ROOT/'Experiments/scorpion_v2_georg'
    gp = expg/'derived/exploratory/four_cycles_20261013'
    contract = read(pin(gp/'contract.local.json'))
    segments_path = pin(gp/'segments.local.json', contract['outputs']['segments.local.json'])
    item = next(s for s in read(segments_path) if s['number'] == 3)
    src = next(s for s in contract['source_artifacts'] if s['record_id'] == item['record_id'])
    pin(expg/src['source_relative_path'], src['source_sha256'])
    cache = pin(expg/src['cache_path'], src['cache_sha256'])
    annotations = read(pin(expg/'annotations/scorpion_review/annotations_candidates.local.json',
                           contract['annotations_sha256']))
    ann = next(a for a in annotations['records'] if a['record_id']==item['record_id'] and a['fs_hz_scenario']==1000)
    pin(HERE/'scorpion_qc.py', read(expg/'derived/scorpion_review/summary.local.json')['code_sha256'])
    pin(expg/'derived/scorpion_review/summary.local.json', contract['summary_sha256'])
    pin(HERE/'scorpion_protocol.py')
    rates = read(pin(expg/'derived/scorpion_bin_review/working_sampling_rates.local.json', contract['sampling_rates_sha256']))
    if next(r for r in rates['records'] if r['record_id']==item['record_id'])['working_sample_rate_hz'] != 1000:
        raise ValueError('Georg timebase differs from existing BIN evidence')
    from scorpion_protocol import channel_assignment
    assignment = channel_assignment(item['record_id'])
    if assignment['physical_channel_roles']['1'] != 'ttrkg' or assignment['ttrkg']['variant'] != '2x2':
        raise ValueError('Georg primary physical montage changed')
    with np.load(cache) as a:
        events = np.array(item['qrs_candidate_samples'])
        if not set(events).issubset(set(ann['qrs_samples_by_channel']['1'])):
            raise ValueError('Display selection no longer belongs to existing annotation')
        if not np.array_equal(a['qrs_samples_1'], ann['qrs_samples_by_channel']['1']):
            raise ValueError('Georg cache/annotation disagreement')
        start, end = events[[0, -1]]
        pulse = a['bei_pulse_band_1'][start:end+1]
        stored = next(l['pulse_mohm'] for l in item['leads'] if l['role']=='ttrkg')
        if not np.array_equal(1000*pulse, stored):
            raise ValueError('Georg display pulse differs from original filtered cache')
        signals = {'georg': dict(record_id=item['record_id'], events_samples=events.tolist(),
            fs_hz=1000, inhale_window=item['strict_inhale_window_samples_half_open'],
            pulse_ohm=pulse, ecg_mV=a['ecg_filtered_1'][start:end+1], origin=int(start),
            physical_ttrkg_channel=1, ecg_channel=1, flags=item['flags'],
            annotation_status=item['annotation_status'],
            protocol_full_digest_changed_since_display=sha(HERE/'scorpion_protocol.py')!=contract['protocol_sha256'],
            protocol_binding_scope='SCV2-G-003 only; current author-confirmed mapping rechecked')}

    exp = ROOT/'Experiments/scorpion_20261022'
    session = exp/'derived/exploratory/session_3340'
    summary = read(pin(session/'summary.local.json'))
    session_annotations = read(exp/'annotations/session_1140/candidate_annotations.local.json')
    for name, key in [('session_20261022.py','producer_sha256'), ('scorpion_qc.py','filter_source_sha256'),
                      ('lateral_pilot.py','ensemble_source_sha256')]:
        pin(HERE/name, summary[key])
    for rel, key in [('annotations/session_1140/candidate_annotations.local.json','annotations_sha256'),
                     ('metadata/inventory.local.json','inventory_sha256'),
                     ('derived/scorpion_bin_review/txt_binary_matches.local.json','matches_sha256'),
                     ('metadata/author_confirmations_20261023.local.json','author_context_sha256')]:
        pin(exp/rel, summary[key])
    for subject, rid in [('nik','SCV2-N-001'), ('yaroslav','SCV2-Y-001')]:
        r = next(r for r in summary['records'] if r['record_id']==rid)
        ra = next(a for a in session_annotations['records'] if a['record_id']==rid)
        if ra['inner_windows_samples_half_open'] != r['windows'] or ra['source_sha256']!=r['source_sha256']:
            raise ValueError('Session annotation/summary source or windows disagree')
        if r['context']['ttrkg_variant']!='2x2' or r['context']['lateral_size_mm']!=140:
            raise ValueError('Signal does not match primary FEM montage')
        pin(exp/r['source_relative_path'], r['source_sha256'])
        cache = pin(session/(rid+'.local.npz'), r['arrays_sha256'])
        with np.load(cache) as a:
            events = np.array(r['four_cycles']['events_samples'])
            ec, ch = r['ecg_reference'], r['physical_channel_roles']['ttrkg']
            if not set(events).issubset(set(a[f'qrs_{ec}'])):
                raise ValueError('Selected events are absent in the original QRS artifact')
            start, end = events[[0,-1]]
            signals[subject] = dict(record_id=rid, events_samples=events.tolist(), fs_hz=r['fs'],
                inhale_window=r['windows']['inhale'], pulse_ohm=a[f'pulse_{ch}'][start:end+1],
                ecg_mV=a[f'ecg_{ec}'][start:end+1], origin=int(start),
                physical_ttrkg_channel=ch, ecg_channel=ec, flags=[], annotation_status=ra['status'])

    ct = pd.read_csv(pin(CT/'chamber_volumes_by_rr.csv'))
    ct_summary = read(pin(CT/'rr_and_subtraction_summary.json'))
    pin(CT.parent/'analyze_subtraction_and_rr.py')
    ct_by_id = {s['subject']:s for s in ct_summary['subjects']}
    out = []
    for subject, sig in signals.items():
        model = model_by_name[NAMES[subject]]
        v0 = model['heart']['volume_reference_ml']
        profiles = []
        for name, rows, rho in [('control', model['heart']['rows'], model['baseline_materials_inhale_ohm_m']),
                                ('blood',model['material_range']['heart']['rows'],[2.5,7.,48.09767352557847,1.35,23.040605218284004])]:
            op = {q:volume_operator(rows,v0,q) for q in [1024,4096]}
            profiles.append(dict(id=name, rho_ohm_m=rho, coarse=op[1024], fine=op[4096],
                delta_q_secant_ohm_per_ml=op[4096]['secant_ohm_per_ml']-op[1024]['secant_ohm_per_ml'],
                q_sign_consistent=bool(np.sign(op[4096]['secant_ohm_per_ml'])==np.sign(op[1024]['secant_ohm_per_ml'])),
                q_is_not_mesh_convergence=True))
        metrics = cycle_metrics(sig['pulse_ohm'],sig['events_samples'],sig['fs_hz'],sig['inhale_window'],sig['origin'])
        for p in profiles:
            p['cycle_diagnostics'] = [amplitude_diagnostic(c['ptp_ohm'],p['fine']) for c in metrics['cycles']]
        ct_id = {'georg':'georg','nik':'nix'}.get(subject)
        ct_result = dict(available=False, reason='No dynamic CT available; author confirmation 2026-10-28')
        if ct_id:
            frame = ct[ct.subject==ct_id]
            if frame.empty:
                raise ValueError('Known dynamic CT subject missing')
            ct_result = dict(available=True, ct_subject=ct_id,
                cycle_merge_assessment=ct_by_id[ct_id]['cycle_merge_assessment'],
                cycles=[ct_cycle(f,{p['id']:p['fine'] for p in profiles}) for _,f in frame.groupby('cycle_index')],
                signal_CT_simultaneous=False, segmentation_status='automatic_candidate_not_independently_validated',
                ct_scope='Saved TS 2.18 total/heart and heartchambers_highres volume table; no masks resegmented')
            ct_result['phase_sign_diagnostic'] = phase_sign_diagnostic(metrics,ct_result['cycles'],profiles) if subject=='nik' else []
            ct_result['phase_sign_scope'] = ('Nik only: provisional same-RR-fraction hypothesis' if subject=='nik'
                else 'not_run: incompatible 491/1159 ms trigger intervals; no uniform phase alignment')
        out.append(dict(id=subject, name=NAMES[subject], case_id=CASE, volume_reference_ml=v0,
            signal=dict(**{k:v for k,v in sig.items() if k not in ['pulse_ohm','ecg_mV']},
                time_s=(np.arange(len(sig['pulse_ohm']))/sig['fs_hz']).tolist(),
                pulse_mohm=(1000*sig['pulse_ohm']).tolist(), ecg_mV=sig['ecg_mV'].tolist(),
                metrics={**metrics,'delta_Z_from_R_ohm':np.array(metrics['delta_Z_from_R_ohm']).tolist()}),
            profiles=profiles, dynamic_CT=ct_result))
    return dict(schema='scorpion_measured_pulse_volume_v1', subjects=out,
        status='exploratory_hypothesis_not_validated', annotation_status='pending_manual_review',
        pulse_band_hz=[.7,12], pulse_filter='cached whole-record fourth-order Butterworth, zero phase',
        source_excitation='time_division_own_drive', annotations_modified=False, FEM_was_run=False,
        tissue_pulse_correction_applied=False, pulse_is_isolated_cardiac_component=False,
        stroke_volume_estimated=False, clinical_validation=False, reader_tables_default_open=False,
        sources=[dict(relative_path=p,sha256=h) for p,h in sorted(pins.items())],
        analysis=dict(relative_path=Path(__file__).relative_to(ROOT).as_posix(),sha256=sha(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'_assets/measured_pulse_volume_20261028.summary.json')
    args=p.parse_args()
    result=collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print('Three subjects / 12 existing ECG intervals; 4 separate CT trigger intervals; no new FEM')


if __name__=='__main__':
    main()
