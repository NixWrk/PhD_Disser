"""Conditional experimental inversion at fixed C01; no source artifacts changed."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from c01_baseline_analysis import load_models
from geometry_validation_tests import recover_bounded

OUT = Path(__file__).resolve().parent


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def field(sp, rho):
    q = np.log(rho[1] / rho[0])
    f, d = sp(q), sp(q, 1)
    return rho[0] * f, np.column_stack([f-d, d / np.exp(q)])


def main(derived):
    g, models = load_models()
    source = ROOT / 'data/nik/experimental/nik_breath_hold_levels.csv'
    legacy = pd.read_csv(source).query('is_independent == 1').sort_values('L_mm')
    current_source = derived / 'exp02/exploratory/33.06_side_arrays_exploratory.json'
    current = json.loads(current_source.read_text(encoding='utf-8'))
    obs = pd.DataFrame(current['static_observations']).query('subject_id == "exp02_nik"').sort_values('size_mm')
    assert obs.breathing_annotation_status.eq('accepted').all()
    obs = obs.rename(columns={'size_mm':'L_mm','z_inhale_ohm':'Z_inhale_hold_ohm','z_exhale_ohm':'Z_exhale_hold_ohm'})
    for p in current['input_provenance']:
        if p['subject_id']=='exp02_nik':
            annotation = derived/'exp02/annotations/breathing'/(p['record_id']+'.json')
            assert hashlib.sha256(annotation.read_bytes()).hexdigest()==p['breathing_sidecar_sha256']
    compare = obs[['L_mm','Z_inhale_hold_ohm','Z_exhale_hold_ohm']].merge(
        legacy[['L_mm','Z_inhale_hold_ohm','Z_exhale_hold_ohm']],on='L_mm',suffixes=('_current','_legacy'))
    for state in ['inhale','exhale']:
        compare[f'difference_{state}_ohm'] = compare[f'Z_{state}_hold_ohm_current']-compare[f'Z_{state}_hold_ohm_legacy']
    compare.to_csv(OUT/'observations_version_comparison.csv',index=False)
    np.testing.assert_array_equal(obs.L_mm, g['sizes_mm'])
    bounds = np.asarray(g['bounds_ohm_m'])
    names = ['reference', 'variable_transverse', 'uniform_transverse', 'planar_nominal']
    fits, predictions, joint = [], [], []
    working = {}
    for name in names:
        sp = models[name]
        for state in ['inhale', 'exhale']:
            y = obs[f'Z_{state}_hold_ohm'].to_numpy()
            fit, pred, jlog = recover_bounded(sp, y, bounds, grid_size=1601)
            rho = np.array([fit['rho1'], fit['rho2']])
            z, jac = field(sp, rho)
            np.testing.assert_allclose(z, pred)
            np.testing.assert_allclose(jac * rho, jlog)
            # Central differences verify physical-unit derivatives, independently of optimizer.
            for k in range(2):
                dr = np.zeros(2); dr[k] = rho[k] * 1e-5
                fd = (field(sp, rho+dr)[0]-field(sp, rho-dr)[0])/(2*dr[k])
                np.testing.assert_allclose(jac[:, k], fd, rtol=1e-5, atol=1e-8)
            grad = 2 * jac.T @ (z-y) / len(y)
            fits.append(dict(model=name, state=state, **fit,
                             mse_gradient_rho1=float(grad[0]), mse_gradient_rho2=float(grad[1])))
            predictions.extend(dict(model=name, state=state, L_mm=float(l),
                                    observed_ohm=float(a), predicted_ohm=float(b))
                               for l, a, b in zip(obs.L_mm, y, z))
            working[name, state] = rho
        yin = obs.Z_inhale_hold_ohm.to_numpy()
        yex = obs.Z_exhale_hold_ohm.to_numpy()
        def residual(x):
            r = np.exp(x)
            return np.r_[field(sp, r[[0,1]])[0]-yin, field(sp, r[[0,2]])[0]-yex]
        lo = np.log([bounds[0,0], bounds[1,0], bounds[1,0]])
        hi = np.log([bounds[0,1], bounds[1,1], bounds[1,1]])
        runs = [least_squares(residual, lo+(hi-lo)*a, bounds=(lo,hi),
                             ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
                for a in [.2,.5,.8]]
        assert all(r.success for r in runs)
        opt = min(runs, key=lambda r: r.cost)
        rho = np.exp(opt.x)
        zin = field(sp, rho[[0,1]])[0]; zex = field(sp, rho[[0,2]])[0]
        joint.append(dict(model=name, rho1_shared=float(rho[0]), rho2_inhale=float(rho[1]),
                          rho2_exhale=float(rho[2]), rmse_all_ohm=float(np.sqrt(np.mean(opt.fun**2))),
                          rmse_breath_difference_ohm=float(np.sqrt(np.mean(((zin-zex)-(yin-yex))**2))),
                          rho2_inhale_at_upper=bool(np.isclose(rho[1],bounds[1,1])),
                          rho2_exhale_at_lower=bool(np.isclose(rho[2],bounds[1,0])),
                          rho2_exhale_at_upper=bool(np.isclose(rho[2],bounds[1,1]))))
    pd.DataFrame(fits).to_csv(OUT / 'baseline_independent.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'baseline_predictions.csv', index=False)
    pd.DataFrame(joint).to_csv(OUT / 'baseline_shared_rho1.csv', index=False)
    save('contract.json', dict(status='conditional_calculation_not_validated', candidate=g['candidate'],
        centre_mm=g['centre_mm'], sizes_mm=g['sizes_mm'], bounds_ohm_m=bounds.tolist(),
        observations_source='exp02/exploratory/33.06_side_arrays_exploratory.json:static_observations:exp02_nik',
        observations_sha256=hashlib.sha256(current_source.read_bytes()).hexdigest(),
        legacy_observations_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assumptions=['Exact fixed CT and C01 montage from 20.16.',
                     'Inspiration CT reused for expiration: geometry and h unchanged.',
                     'Same prior box used for both states as a comparison scenario; not an established expiration norm.',
                     'Two effective scalar tissues; no measured gain, offset or contact parameter fitted.',
                     'PEM mesh-node electrodes for CT models; nominal points for planar_nominal.',
                     'Equal weights in ohms; covariance unknown.',
                     'Independent rho1 per state is main scenario; constant rho1 tested separately.']))
    print(pd.DataFrame(fits)[['model','state','rho1','rho2','train_rmse','rho2_at_bound','mse_gradient_rho2']].to_string(index=False))
    print(pd.DataFrame(joint).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--derived-root',type=Path,required=True)
    main(parser.parse_args().derived_root)
