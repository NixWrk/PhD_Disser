"""Apply the selected CT-derived lateral model to accepted experimental intervals."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from c01_baseline_analysis import load_models
from geometry_validation_tests import recover_bounded

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'output/exploratory/lateral_real_application_20260915'
SOURCE = ROOT / 'output/exploratory/transverse_consistency_20260914'
MODELS = ['variable_transverse', 'reference', 'planar_nominal']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def field(sp, rho):
    x = np.log(rho[1]/rho[0])
    f, d = sp(x), sp(x, 1)
    return rho[0]*f, np.column_stack([f-d, d/np.exp(x)])


def verify_inputs(out):
    contract = read(out/'input_contract.json')
    for name, digest in contract['prepared_sha256'].items():
        assert sha(out/name)==digest, 'Changed experimental input: '+name
    for name, digest in contract['geometry_sha256'].items():
        assert sha(ROOT/name)==digest, 'Changed geometry input: '+name
    assert contract['subject_id']=='exp02_nik' and contract['candidate']=='C01'
    return contract


def prepare(derived, out):
    out.mkdir(parents=True, exist_ok=True)
    g, _ = load_models()
    static_path = derived/'exp02/exploratory/33.06_side_arrays_exploratory.json'
    pulse_path = derived/'exp02/exploratory/33.03_pulse_ensembles.provisional.json'
    static, pulses = read(static_path), read(pulse_path)
    obs = pd.DataFrame(static['static_observations']).query('subject_id == "exp02_nik"').sort_values('size_mm')
    assert obs.breathing_annotation_status.eq('accepted').all()
    np.testing.assert_array_equal(obs.size_mm, g['sizes_mm'])
    obs = obs.rename(columns={'size_mm':'L_mm', 'z_inhale_ohm':'Z_inhale_hold_ohm', 'z_exhale_ohm':'Z_exhale_hold_ohm'})
    obs[['L_mm','Z_inhale_hold_ohm','Z_exhale_hold_ohm']].to_csv(out/'observations.csv', index=False)
    rows, provenance = [], []
    for state, mode in [('inhale','задержка_вдох'), ('exhale','задержка_выдох')]:
        es = sorted([e for e in pulses['ensembles'] if e['subject_id']=='exp02_nik' and e['mode']==mode], key=lambda e:e['size_mm'])
        assert [e['size_mm'] for e in es]==g['sizes_mm']
        phase = np.asarray(es[0]['phase_rr'])
        for e in es:
            np.testing.assert_array_equal(e['phase_rr'], phase)
            assert e['signal_unit']=='mOhm_native_file_scale'
            p = next(p for p in static['input_provenance'] if p['record_id']==e['record_id'])
            assert p['subject_id']==e['subject_id'] and p['size_mm']==e['size_mm']
            assert p['source_csv_sha256']==e['input_provenance']['input_sha256']
            for folder, key in [('breathing','breathing_sidecar_sha256'), ('ecg','ecg_sidecar_sha256')]:
                digest = sha(derived/'exp02/annotations'/folder/(e['record_id']+'.json'))
                assert digest==e['input_provenance'][key]
                if folder=='breathing':
                    assert digest==p[key] and p['breathing_qc_status']=='accepted'
            y = np.asarray(e['primary']['median_mohm'], dtype=float)/1000
            assert y.shape==(200,) and np.isfinite(y).all()
            rows.extend(dict(state=state,L_mm=e['size_mm'],phase_rr=float(t),delta_Z_ohm=float(z)) for t,z in zip(phase,y))
            provenance.append(dict(state=state,L_mm=e['size_mm'],n_cycles=e['primary']['n_cycles'],
                input_sha256=e['input_provenance']['input_sha256'],annotation_sha256=e['input_provenance']))
    pd.DataFrame(rows).to_csv(out/'pulse_inputs.csv', index=False)
    paths = [SOURCE/n for n in ['geometry.json','geometry_masks.mat','input_contract.json','prepared_fem.mat','contacts.csv','cem_baseline.csv']]
    save(out/'input_contract.json', dict(status='exploratory_hypothesis_not_validated',subject_id='exp02_nik',
        experiment='2: MGTU rheocardiomonitor',candidate='C01',centre_mm=g['centre_mm'],sizes_mm=g['sizes_mm'],
        bounds_ohm_m=g['bounds_ohm_m'],source_static_sha256=sha(static_path),source_pulse_sha256=sha(pulse_path),
        source_upstream_status=pulses['upstream'],source_legacy_signal_semantics=pulses['signal_semantics'],
        signal_interpretation='Author decision: recorded physical units and sign accepted; no additional gain/offset/sign fitted.',
        decision_source='Colab Notebooks/AGENTS.md: Scientific results, experiments 2-3 measured output units.',
        source_code_sha256=sha(Path(__file__)),geometry_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        prepared_sha256={n:sha(out/n) for n in ['observations.csv','pulse_inputs.csv']},provenance=provenance,
        assumptions=['Exact CT and fixed C01 montage accepted.', 'Inspiration geometry reused for expiration; h unchanged.',
                     'Two scalar resistivities, isotropic real conductivity.', 'Nine sequential sizes treated as repeatable states.',
                     'Literature-motivated search bounds reused for both states, not personal physiological norms.',
                     'No pulse geometry/contact change; upstream exploratory QC status preserved.']))
    return verify_inputs(out)


def point_fits(out):
    contract = verify_inputs(out)
    _, models = load_models()
    obs = pd.read_csv(out/'observations.csv')
    bounds = np.asarray(contract['bounds_ohm_m'])
    fits, predictions = [], []
    for name in MODELS:
        for state in ['inhale','exhale']:
            y = obs[f'Z_{state}_hold_ohm'].to_numpy()
            fit, pred, jlog = recover_bounded(models[name], y, bounds, grid_size=1601)
            rho = np.array([fit['rho1'], fit['rho2']])
            z, jac = field(models[name],rho)
            np.testing.assert_allclose(z,pred)
            np.testing.assert_allclose(jac*rho,jlog)
            for k in range(2):
                dr = np.zeros(2); dr[k] = rho[k]*1e-5
                fd = (field(models[name],rho+dr)[0]-field(models[name],rho-dr)[0])/(2*dr[k])
                np.testing.assert_allclose(jac[:,k],fd,rtol=1e-5,atol=1e-8)
            fits.append(dict(model=name,electrodes='PEM',state=state,**fit))
            predictions.extend(dict(model=name,electrodes='PEM',state=state,L_mm=float(L),observed_ohm=float(a),predicted_ohm=float(b),
                dZ_dlogrho1=float(J[0]),dZ_dlogrho2=float(J[1])) for L,a,b,J in zip(obs.L_mm,y,pred,jlog))
    pd.DataFrame(fits).to_csv(out/'pem_fits.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'pem_predictions.csv',index=False)
    sp = models['variable_transverse']
    yin,yex = [obs[f'Z_{s}_hold_ohm'].to_numpy() for s in ['inhale','exhale']]
    def residual(x):
        r = np.exp(x)
        return np.r_[field(sp,r[[0,1]])[0]-yin,field(sp,r[[0,2]])[0]-yex]
    lo,hi = np.log(bounds[[0,1,1],0]),np.log(bounds[[0,1,1],1])
    runs = [least_squares(residual,lo+(hi-lo)*a,bounds=(lo,hi),ftol=1e-12,xtol=1e-12,gtol=1e-12) for a in [.2,.5,.8]]
    assert all(r.success for r in runs)
    best = min(runs,key=lambda r:r.cost)
    save(out/'pem_shared.json',dict(rho=np.exp(best.x).tolist(),rmse_ohm=float(np.sqrt(np.mean(best.fun**2))),
        scenario='shared_rho1',model='variable_transverse',order=['rho1','rho2_inhale','rho2_exhale']))
    return pd.DataFrame(fits)


def combine(out):
    verify_inputs(out)
    fits = pd.read_csv(out/'pem_fits.csv')
    preds = pd.read_csv(out/'pem_predictions.csv')
    more, pp = [], []
    for model in ['variable_transverse','reference']:
        for state in ['inhale','exhale']:
            r = read(out/f'cem_{model}_{state}.json')
            assert r['complete'] and r['best']['exitflag']>0
            p = pd.read_csv(out/f'cem_{model}_{state}_predictions.csv')
            b = r['best']; rho = b['rho']
            more.append(dict(model=model,electrodes='CEM',state=state,rho1=rho[0],rho2=rho[1],train_rmse=b['rmse_ohm'],
                condition_log_parameters=b['condition_log_parameters'],rho1_at_bound=b['at_bound'][0],rho2_at_bound=b['at_bound'][1]))
            p['model'],p['electrodes'],p['state'] = model,'CEM',state
            pp.append(p)
    fits = pd.concat([fits,pd.DataFrame(more)],ignore_index=True)
    preds = pd.concat([preds,*pp],ignore_index=True)
    fits.to_csv(out/'baseline_fits.csv',index=False)
    preds.to_csv(out/'baseline_predictions.csv',index=False)
    return fits,preds


def pulse_fits(out, fits, preds):
    contract = verify_inputs(out)
    inputs = pd.read_csv(out/'pulse_inputs.csv')
    bounds = np.asarray(contract['bounds_ohm_m'])
    summary,waves = [],[]
    for b in fits.itertuples():
        source = inputs[inputs.state.eq(b.state)]
        table = source.pivot(index='L_mm',columns='phase_rr',values='delta_Z_ohm').sort_index()
        p = preds[(preds.model==b.model)&(preds.state==b.state)&(preds.electrodes==b.electrodes)].sort_values('L_mm')
        np.testing.assert_array_equal(table.index,p.L_mm)
        rho = np.array([b.rho1,b.rho2])
        jlog = p[['dZ_dlogrho1','dZ_dlogrho2']].to_numpy()
        y = table.to_numpy()
        frac,_,rank,sv = np.linalg.lstsq(jlog,y,rcond=None)
        assert rank==2
        delta = rho[:,None]*frac
        assert np.all(rho[:,None]+delta>0)
        prediction = jlog@frac
        only2 = np.linalg.lstsq(jlog[:,1:2],y,rcond=None)[0]
        summary.append(dict(model=b.model,electrodes=b.electrodes,state=b.state,
            delta_rho1_ptp=float(np.ptp(delta[0])),delta_rho2_ptp=float(np.ptp(delta[1])),
            delta_rho1_percent=float(100*np.ptp(frac[0])),delta_rho2_percent=float(100*np.ptp(frac[1])),
            relative_residual_percent=float(100*np.linalg.norm(prediction-y)/np.linalg.norm(y)),
            condition_log=float(sv[0]/sv[1]),rho1_omitted_delta_rho2_ptp=float(rho[1]*np.ptp(only2)),
            rho1_omitted_residual_percent=float(100*np.linalg.norm(jlog[:,1:2]@only2-y)/np.linalg.norm(y)),
            rho2_outside_baseline_box_count=int(np.count_nonzero((rho[1]+delta[1]<bounds[1,0])|(rho[1]+delta[1]>bounds[1,1])))))
        waves.extend(dict(model=b.model,electrodes=b.electrodes,state=b.state,phase_rr=float(t),delta_rho1=float(a),delta_rho2=float(c))
                     for t,a,c in zip(table.columns,delta[0],delta[1]))
    pd.DataFrame(summary).to_csv(out/'pulse_fits.csv',index=False)
    pd.DataFrame(waves).to_csv(out/'pulse_waveforms.csv',index=False)
    return pd.DataFrame(summary)


def analyze(out):
    point_fits(out)
    fits,preds = combine(out)
    pulse = pulse_fits(out,fits,preds)
    save(out/'analysis_manifest.json',dict(status='exploratory_hypothesis_not_validated',source_sha256=sha(Path(__file__)),
        input_contract_sha256=sha(out/'input_contract.json'),
        outputs_sha256={p.name:sha(p) for p in out.glob('*.csv')},
        cem_reports_sha256={p.name:sha(p) for p in out.glob('cem_*.json')},
        statement='New direct CEM fits; PEM library reused. Recorded units accepted by author. No independent tissue truth.'))
    print(fits[['model','electrodes','state','rho1','rho2','train_rmse']].to_string(index=False))
    print(pulse.to_string(index=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','analyze'])
    parser.add_argument('--derived-root',type=Path)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.stage=='prepare':
        if args.derived_root is None:
            parser.error('--derived-root is required for prepare')
        prepare(args.derived_root,args.out)
        print(point_fits(args.out)[['model','state','rho1','rho2']].to_string(index=False))
    else:
        analyze(args.out)
