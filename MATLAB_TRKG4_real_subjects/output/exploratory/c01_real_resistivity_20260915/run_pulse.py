"""Conditional pulse inversion using measured RR-normalized ensembles, native scale."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from run_inversion import ROOT, OUT, field, load_models, save


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(derived):
    source = derived / 'exp02/exploratory/33.03_pulse_ensembles.provisional.json'
    a = read(source)
    static_source = derived / 'exp02/exploratory/33.06_side_arrays_exploratory.json'
    static = read(static_source)
    assert sha(static_source)==read(OUT/'contract.json')['observations_sha256']
    g, models = load_models()
    sizes = g['sizes_mm']
    baseline = pd.read_csv(OUT/'baseline_independent.csv')
    bounds = np.asarray(g['bounds_ohm_m'])
    summary, waves, qc = [], [], []
    for state, mode in [('inhale','задержка_вдох'), ('exhale','задержка_выдох')]:
        es = [e for e in a['ensembles'] if e['subject_id']=='exp02_nik' and e['mode']==mode]
        es.sort(key=lambda e:e['size_mm'])
        assert [e['size_mm'] for e in es] == sizes
        phase = np.array(es[0]['phase_rr'])
        for e in es:
            np.testing.assert_array_equal(e['phase_rr'], phase)
            assert e['signal_unit']=='mOhm_native_file_scale'
            p = next(p for p in static['input_provenance'] if p['record_id']==e['record_id'])
            assert p['size_mm']==e['size_mm'] and p['subject_id']==e['subject_id']
            assert p['source_csv_sha256']==e['input_provenance']['input_sha256']
            for folder, key in [('breathing','breathing_sidecar_sha256'),('ecg','ecg_sidecar_sha256')]:
                current = derived / 'exp02/annotations' / folder / (e['record_id']+'.json')
                assert sha(current)==e['input_provenance'][key], (state,e['size_mm'],folder)
            qc.append(dict(state=state,size_mm=e['size_mm'],n_cycles=e['primary']['n_cycles'],
                           source_and_annotation_identity_matches=True))
        y = np.array([e['primary']['median_mohm'] for e in es])/1000
        assert y.shape==(9,200) and np.isfinite(y).all()
        for name in ['reference','variable_transverse','uniform_transverse','planar_nominal','reference_cem']:
            if name=='reference_cem':
                if state!='inhale':
                    continue
                cdir = ROOT/'output/exploratory/c01_baseline_models_20260914'
                c = read(cdir/'cem_fit_summary.json')['best']
                rho = np.array([c['rho1'],c['rho2']])
                ct = pd.read_csv(cdir/'cem_experimental_predictions.csv')
                np.testing.assert_array_equal(ct.L_mm,sizes)
                jlog = ct[['dZ_dlogrho1','dZ_dlogrho2']].to_numpy()
                jac = jlog / rho
            else:
                b = baseline.query('model == @name and state == @state').iloc[0]
                rho = np.array([b.rho1,b.rho2])
                z0, jac = field(models[name],rho)
                jlog = jac * rho
            frac, _, rank, sv = np.linalg.lstsq(jlog,y,rcond=None)
            assert rank==2
            delta = rho[:,None] * frac
            prediction = jac @ delta
            residual = prediction-y
            # Restricted pulse inverse quantifies the effect of omitting tissue 1.
            only2 = np.linalg.lstsq(jlog[:,1:2],y,rcond=None)[0]
            restricted_residual = jlog[:,1:2] @ only2 - y
            remainder = None
            if name!='reference_cem':
                rr = rho[:,None]+delta
                assert (rr>0).all()
                xx = np.log(rr[1]/rr[0])
                assert xx.min()>=models[name].x[0] and xx.max()<=models[name].x[-1]
                nonlinear = (rr[0,:,None]*models[name](xx)).T-z0[:,None]
                remainder = float(np.linalg.norm(nonlinear-prediction)/np.linalg.norm(prediction))
            summary.append(dict(model=name,state=state,rho1=float(rho[0]),rho2=float(rho[1]),
                delta_rho1_ptp_ohm_m=float(np.ptp(delta[0])),delta_rho2_ptp_ohm_m=float(np.ptp(delta[1])),
                delta_rho1_min_ohm_m=float(delta[0].min()),delta_rho1_max_ohm_m=float(delta[0].max()),
                delta_rho2_min_ohm_m=float(delta[1].min()),delta_rho2_max_ohm_m=float(delta[1].max()),
                delta_rho1_ptp_percent=float(100*np.ptp(frac[0])),delta_rho2_ptp_percent=float(100*np.ptp(frac[1])),
                input_rms_mohm=float(1000*np.sqrt(np.mean(y*y))),
                residual_rms_mohm=float(1000*np.sqrt(np.mean(residual*residual))),
                relative_residual_percent=float(100*np.linalg.norm(residual)/np.linalg.norm(y)),
                rho1_omitted_relative_residual_percent=float(100*np.linalg.norm(restricted_residual)/np.linalg.norm(y)),
                rho1_omitted_delta_rho2_ptp_ohm_m=float(rho[1]*np.ptp(only2)),
                condition_relative_parameters=float(sv[0]/sv[-1]),
                nonlinear_remainder_relative_to_linear=remainder,
                samples_outside_baseline_rho2_box=int(np.count_nonzero((rho[1]+delta[1]<bounds[1,0])|(rho[1]+delta[1]>bounds[1,1]))),
                native_gain_assumed=1,physical_sign_validated=False))
            waves.extend(dict(model=name,state=state,phase_rr=float(t),delta_rho1_ohm_m=float(d1),delta_rho2_ohm_m=float(d2))
                         for t,d1,d2 in zip(phase,delta[0],delta[1]))
    pd.DataFrame(summary).to_csv(OUT/'pulse_summary.csv',index=False)
    pd.DataFrame(waves).to_csv(OUT/'pulse_waveforms.csv',index=False)
    save('pulse_contract.json',dict(status='conditional_native_scale_not_calibrated_tissue_amplitude',
        source_sha256=sha(source),static_artifact_sha256=sha(static_source),script_sha256=sha(Path(__file__)),
        native_signal_semantics=a['signal_semantics'],upstream_status=a['upstream'],
        assumptions=['Per-cycle median removed upstream; pointwise median RR ensembles retained.',
                     'Each state uses its own fitted baseline rho1/rho2 with fixed C01 geometry.',
                     'Native numerical mOhm divided by 1000; unknown physical gain assumed one and unknown sign preserved.',
                     'Equal measurement weights; sequential sizes treated as repeatable physiological cycles.',
                     'No pulse geometry/contact/independent organ source variation modeled.',
                     'Full-rank two-column linear inverse with positive reconstructed total rho is conditional, not physiologically validated.',
                     'Reported amplitude is max minus min of recovered waveform, not max excursion and not respiratory contrast.',
                     'Inspiration baseline at prior upper bound undermines interpretation; counts outside box retained.',
                     'Nonlinear remainder is an internal interpolation-model check, not a new independent FEM calculation.',
                     'CEM pulse derivative control uses existing CT/CEM inspiration point only; no CEM expiration fit available.',
                     'CEM control retains prior baseline observations (inhale 90 mm +0.007 ohm, 130 mm -0.002 ohm vs current).',
                     'Restricted delta-rho1=0 pulse fit is separately profiled; its error is not assessed by rho1 amplitude alone.'],
        provenance_qc=qc))
    print(pd.DataFrame(summary)[['model','state','delta_rho1_ptp_ohm_m','delta_rho2_ptp_ohm_m',
         'delta_rho1_ptp_percent','delta_rho2_ptp_percent','relative_residual_percent',
         'nonlinear_remainder_relative_to_linear']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--derived-root',type=Path,required=True)
    main(p.parse_args().derived_root)
