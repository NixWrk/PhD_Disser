"""Conditional two-layer lateral diagnostics for the three-size October row.

Every h is a different fixed-thickness fit; no CT thickness is inferred here.
TTRKG never enters this model. Outputs live in ignored exploratory directories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from audit_scorpion_session import read, sha, write
from lateral_pilot import DiskModel
from session_20261022 import EXPERIMENT, ROOT, STATUS

SIZES = [60, 100, 140]
H_GRID_MM = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150]
REGISTRY=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
PROFILE=read(REGISTRY)['screening_profiles']['100000']['compartments']
RHO_BOUNDS=np.array([PROFILE[k]['rho_bounds_ohm_m'] for k in ['soft','lung_unspecified']])


def fit_fixed_h(model, measured, rows=None):
    """Equal-weight residuals in ohms; positive rho via log coordinates."""
    rows = np.arange(len(measured)) if rows is None else np.asarray(rows)
    def residual(x):
        return model.evaluate(np.exp(x))[0][rows] - measured[rows]
    def jacobian(x):
        return model.evaluate(np.exp(x))[1][rows]
    fits = [least_squares(residual, np.log(start), jac=jacobian,
                bounds=(np.log(RHO_BOUNDS[:,0]),np.log(RHO_BOUNDS[:,1])), max_nfev=400,
                ftol=1e-11, xtol=1e-11, gtol=1e-11)
            for initial in ([2, 10], [10, 5], [3, 4])
            for start in [np.clip(initial,RHO_BOUNDS[:,0]+1e-7,RHO_BOUNDS[:,1]-1e-7)]]
    best = min(fits, key=lambda f: (np.linalg.norm(f.fun), tuple(f.x)))
    rho = np.exp(best.x)
    z, j = model.evaluate(rho)
    s = np.linalg.svd(j[rows], compute_uv=False)
    return dict(rho_ohm_m=rho.tolist(), predicted_ohm=z.tolist(),
        residual_ohm=(z-measured).tolist(),
        rmse_ohm=float(np.sqrt(np.mean((z[rows]-measured[rows])**2))),
        log_rho_jacobian=j.tolist(), singular_values_ohm=s.tolist(),
        condition_log_rho=float(s[0]/s[-1]) if s[-1] > 0 else None,
        boundary_parameters=[int(i+1) for i,x in enumerate(rho)
                             if min(abs(np.log(x/RHO_BOUNDS[i,0])),
                                    abs(np.log(x/RHO_BOUNDS[i,1]))) < .001],
        success=bool(best.success), termination_code=int(best.status),
        nfev=int(best.nfev), starts_rmse_ohm=[float(np.sqrt(np.mean(f.fun**2))) for f in fits])


def prepare(experiment=EXPERIMENT):
    out = Path(experiment) / 'derived/exploratory/session_3340'
    summary_file = out / 'summary.local.json'
    summary = read(summary_file)
    if summary['producer_sha256'] != sha(Path(__file__).with_name('session_20261022.py')):
        raise ValueError('Signal producer changed; recompute candidates first')
    models = {h: DiskModel(SIZES, h, diameter_mm=5, radial=4, angular=16, terms=256)
              for h in H_GRID_MM}
    profiles, controls = [], []
    for subject in ('nik', 'yaroslav'):
        main = [r for r in summary['records'] if r['subject'] == subject and
                r['context']['record_kind'] == 'both' and r['context']['ttrkg_variant'] == '2x2']
        main = sorted(main, key=lambda r: r['context']['lateral_size_mm'])
        if [r['context']['lateral_size_mm'] for r in main] != SIZES:
            raise ValueError('Need exactly three independent sizes under the same TTRKG variant')
        if len({r['source_sha256'] for r in main})!=3:
            raise ValueError('Byte-identical recordings cannot provide independent sizes')
        for state in ('inhale', 'exhale'):
            inputs = [next(v for v in summary['levels'] if v['record_id'] == r['record_id'] and
                           v['state'] == state and v['role'] == 'lateral') for r in main]
            measured = np.array([v['baseline_ohm'] for v in inputs])
            en = [r['ensembles'][state]['lateral'] for r in main]
            observed_pulse = (np.array([v['median_ohm'] for v in en])
                              if all(v['n_cycles'] >= 4 for v in en) else None)
            for h, model in models.items():
                fit = fit_fixed_h(model, measured)
                # The withheld baseline also refits rho to the other two sizes.
                left_out = []
                for k, size in enumerate(SIZES):
                    f = fit_fixed_h(model, measured, [i for i in range(3) if i != k])
                    left_out.append(dict(withheld_size_mm=size,
                        predicted_ohm=f['predicted_ohm'][k], observed_ohm=float(measured[k]),
                        error_ohm=f['residual_ohm'][k],
                        boundary_parameters=f['boundary_parameters'],
                        condition_log_rho=f['condition_log_rho']))
                pulse = None
                if observed_pulse is not None:
                    jac = np.array(fit['log_rho_jacobian'])
                    dlogrho = np.linalg.lstsq(jac, observed_pulse, rcond=1e-12)[0]
                    pred = jac @ dlogrho
                    tests = []
                    for k, size in enumerate(SIZES):
                        train = [i for i in range(3) if i != k]
                        d = np.linalg.lstsq(jac[train], observed_pulse[train], rcond=1e-12)[0]
                        p = jac[k] @ d
                        tests.append(dict(withheld_size_mm=size,
                            rmse_mohm=float(np.sqrt(np.mean((p-observed_pulse[k])**2))*1000),
                            predicted_mohm=(1000*p).tolist()))
                    pulse = dict(phase_rr=en[0]['phase_rr'], observed_mohm=(1000*observed_pulse).tolist(),
                        fitted_mohm=(1000*pred).tolist(), dlogrho=dlogrho.tolist(),
                        maximum_absolute_dlogrho=float(np.max(np.abs(dlogrho))),
                        joint_rmse_mohm=float(np.sqrt(np.mean((pred-observed_pulse)**2))*1000),
                        left_out=tests,
                        linearization_smallness_demonstrated=bool(np.max(np.abs(dlogrho)) < .1))
                profiles.append(dict(subject=subject, state=state, h_mm=h,
                    record_ids=[r['record_id'] for r in main], sizes_mm=SIZES,
                    observed_ohm=measured.tolist(), **fit, left_out_baseline=left_out,
                    pulse=pulse, pulse_missing_reason=None if pulse else
                    'Не во всех трёх размерах есть хотя бы четыре полных кандидатных цикла.'))
            best = min([p for p in profiles if p['subject']==subject and p['state']==state],
                       key=lambda p:p['rmse_ohm'])
            # Independent denser disk quadrature and doubled image series.
            refined = DiskModel(SIZES, best['h_mm'], diameter_mm=5, radial=6, angular=24, terms=512)
            ref_z, ref_j = refined.evaluate(best['rho_ohm_m'])
            controls.append(dict(subject=subject,state=state,h_mm=best['h_mm'],
                quadrature_comparison='4×16/256 образов против 6×24/512 образов',
                baseline_difference_ohm=(ref_z-np.array(best['predicted_ohm'])).tolist(),
                max_jacobian_difference_ohm=float(np.max(np.abs(ref_j-np.array(best['log_rho_jacobian']))))))
    result = dict(status=STATUS, producer_sha256=sha(__file__),
        summary_sha256=sha(summary_file), model_source_sha256=sha(Path(__file__).with_name('lateral_pilot.py')),
        h_grid_mm=H_GRID_MM, h_grid_is_CT_bound=False, rho_bounds_ohm_m=RHO_BOUNDS.tolist(),
        registry_sha256=sha(REGISTRY),registry_id=read(REGISTRY)['registry_id'],
        profile='100000',rho_bounds_are_literature_informed_working_ranges=True,
        rho_bounds_are_individual_norms=False, finite_disk_diameter_mm=5,
        contact_diameter_is_measured=False, contact_impedance_is_modelled=False,
        ttrkg_included=False, shared_subject_resistivities=False,
        numerical_controls=controls, profiles=profiles)
    write(out / 'two_layer.local.json', result)
    for c in controls:
        best = min([p for p in profiles if p['subject']==c['subject'] and p['state']==c['state']],
                   key=lambda p:p['rmse_ohm'])
        print(c['subject'], c['state'], 'h',best['h_mm'], 'rho',best['rho_ohm_m'],
              'RMSE',round(best['rmse_ohm'],4),'cond',round(best['condition_log_rho'],2),
              'boundary',best['boundary_parameters'])
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=EXPERIMENT)
    prepare(p.parse_args().experiment)
