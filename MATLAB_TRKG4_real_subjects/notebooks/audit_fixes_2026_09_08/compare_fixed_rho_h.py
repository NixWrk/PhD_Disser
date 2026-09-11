"""One-pose, fixed-rho planar 2L/FEM comparison; h fitting is not validation."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H_RANGE = (0.0, 200.0)  # mm; numerical comparison range, not an anatomical prior
TOL_OHM = 1e-10


def planar_z(L_mm, rho1, rho2, h_mm, terms):
    """Z=V(M)-V(N) per unit current A->B; [A,M,N,B]=[-L/2,-L/4,L/4,L/2].

    Z=rho1/pi * [D(0)+2 sum(k**n D(2nh))], D(t)=1/hypot(L/4,t)-1/hypot(3L/4,t).
    D is positive and decreasing, so |tail| <= 2rho1/pi D(2(N+1)h) q**(N+1)/(1-q).
    Returned bound covers truncation only, not floating-point rounding.
    """
    L = np.atleast_1d(np.asarray(L_mm, dtype=float)) * 1e-3
    if (not np.all(np.isfinite(L)) or np.any(L <= 0)
            or not np.all(np.isfinite([rho1, rho2, h_mm]))
            or min(rho1, rho2) <= 0 or h_mm < 0 or terms < 1):
        raise ValueError("Require finite L>0 mm, rho1,rho2>0 ohm_m, h>=0 mm, terms>=1")
    k = (rho2 - rho1) / (rho2 + rho1)
    if not abs(k) < 1:
        raise ValueError("Resistivity contrast is not representable with |k|<1")
    a, b = L / 4, 3 * L / 4
    n = np.arange(1, terms + 1)
    def gap(t):
        u, v = np.hypot(a, t), np.hypot(b, t)
        return (b*b - a*a) / (u*v*(u+v))  # avoids far-image cancellation
    z = rho1 / np.pi * (gap(0) + 2 * np.sum(k**n[:, None] * gap(2*n[:, None]*h_mm*1e-3), axis=0))
    bound = 2*rho1/np.pi * gap(2*(terms+1)*h_mm*1e-3) * abs(k)**(terms+1)/(1-abs(k))
    return z, bound


def math_checks():
    L = np.array([50., 90., 140.])
    errors = dict(homogeneous=0., h_zero=0., h_large=0., series_32_64=0., independent_scalar_series=0.)
    max_bounds = [0., 0.]
    for r1, r2 in [(4., 20.), (20., 4.)]:  # both signs of reflection coefficient
        for h in [0., 0.25, 27., 200.]:
            z32, b32 = planar_z(L, r1, r2, h, 32)
            z64, b64 = planar_z(L, r1, r2, h, 64)
            delta = np.abs(z32-z64)
            assert np.all(delta <= b32+b64+TOL_OHM)
            errors['series_32_64'] = max(errors['series_32_64'], float(max(delta)))
            max_bounds = [max(max_bounds[i], float(max(b))) for i, b in enumerate([b32, b64])]
            k = (r2-r1)/(r2+r1)
            reference = np.array([r1/np.pi * (1/a-1/b + 2*math.fsum(
                k**n * (1/math.hypot(a, 2*n*h*1e-3)-1/math.hypot(b, 2*n*h*1e-3))
                for n in range(1, 65))) for a, b in zip(L*1e-3/4, 3*L*1e-3/4)])
            independent_error = float(max(np.abs(reference-z64)))
            assert independent_error < TOL_OHM
            errors['independent_scalar_series'] = max(errors['independent_scalar_series'], independent_error)
        for name, h, top, bottom, expected_rho in [
                ('homogeneous', 27., r1, r1, r1), ('h_zero', 0., r1, r2, r2),
                ('h_large', 1e8, r1, r2, r1)]:
            expected = 8*expected_rho/(3*np.pi*(L*1e-3))
            err = float(max(np.abs(planar_z(L, top, bottom, h, 256)[0]-expected)))
            assert err < TOL_OHM, (name, err)
            errors[name] = max(errors[name], err)
    return dict(status='passed', max_abs_errors_ohm=errors, series_terms=[32, 64],
                max_truncation_bounds_ohm=max_bounds, limit_test_h_mm=1e8)


def fit_h(model, target, keep):
    """Scan every 1 mm, refine all detected local minima, explicitly test endpoints."""
    grid = np.linspace(*H_RANGE, 201)
    def loss(h):
        return float(np.mean((model(float(h))[keep]-target[keep])**2))
    values = np.array([loss(h) for h in grid])
    candidates = [(values[i], float(grid[i])) for i in [0, int(np.argmin(values)), 200]]
    if np.ptp(values) > 1e-20:
        for i in range(201):
            if values[i] <= min(values[max(i-1, 0)], values[min(i+1, 200)]):
                opt = minimize_scalar(loss, bounds=(grid[max(i-1, 0)], grid[min(i+1, 200)]), method='bounded',
                                      options={'xatol': 1e-8})
                if not opt.success:
                    raise RuntimeError(opt.message)
                candidates.append((float(opt.fun), float(opt.x)))
    mse, h = min(candidates)
    return dict(h_mm=h, rmse_train_ohm=float(np.sqrt(mse)),
                boundary_lower=h <= 1e-6, boundary_upper=h >= 200-1e-6,
                objective_flat_on_grid=bool(np.ptp(values) <= 1e-20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    tag = parser.parse_args().tag
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', tag):
        parser.error('tag must contain only letters, digits, underscore or hyphen')
    checks = math_checks()
    tables, hashes = {}, {}
    for kind in ['comparison', 'fit_summary']:
        path = ROOT / 'output' / f'nik_trkg4_inverse_inhale_{kind}_{tag}.csv'
        raw = path.read_bytes()
        tables[kind] = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    if len(tables['fit_summary']) != 1 or len(tables['comparison']) != 9:
        raise ValueError('Require exactly one summary row and nine FEM sizes')
    summary = tables['fit_summary'][0]
    rho1, rho2, hgeom = [float(summary[k]) for k in ['rho_soft_ohm_m', 'rho_lungs_ohm_m', 'h_centre_mm']]
    rows = sorted(tables['comparison'], key=lambda row: float(row['L_mm']))
    L, target = np.array([[float(row['L_mm']), float(row['Z_FEM_ohm'])] for row in rows]).T
    if len(np.unique(L)) != 9 or not np.all(np.isfinite(target)):
        raise ValueError('Require nine distinct sizes and finite signed FEM impedances')
    terms = 32
    planar_z(L, rho1, rho2, hgeom, terms)  # validate inputs before optimizing
    while max(planar_z(L, rho1, rho2, 0., terms)[1]) > TOL_OHM:
        terms *= 2
        if terms > 32768:
            raise RuntimeError('Series convergence budget exceeded')
    @lru_cache(maxsize=4096)
    def model(h):
        return planar_z(L, rho1, rho2, h, 2*terms)[0]
    keep = np.ones(9, dtype=bool)
    fit = fit_h(model, target, keep)
    loo, predictions = [], []
    for i in range(9):
        mask = keep.copy()
        mask[i] = False
        fold = fit_h(model, target, mask)
        pred = float(model(fold['h_mm'])[i])
        loo.append(dict(L_mm=float(L[i]), **fold, prediction_ohm=pred, error_ohm=pred-float(target[i])))
        predictions.append(pred)
    probe_h = sorted(set([0., 0.25, 200., hgeom, fit['h_mm']] + [f['h_mm'] for f in loo]))
    step_delta = 0.
    for h in probe_h:
        zn, bn = planar_z(L, rho1, rho2, h, terms)
        z2n, b2n = planar_z(L, rho1, rho2, h, 2*terms)
        delta = float(max(np.abs(zn-z2n)))
        assert np.all(np.abs(zn-z2n) <= bn+b2n+TOL_OHM)
        step_delta = max(step_delta, delta)
    recovery_h = [0., 0.25, 37.25, 200.]
    recovered = [fit_h(model, model(h), keep) for h in recovery_h]
    recovery_error = max(abs(f['h_mm']-h) for f, h in zip(recovered, recovery_h))
    assert rho1 == rho2 or recovery_error < 1e-4
    assert rho1 == rho2 or (recovered[0]['boundary_lower'] and recovered[-1]['boundary_upper'])
    checks['known_h_recovery_test_mm'] = recovery_h
    checks['known_h_recovery_max_error_mm'] = None if rho1 == rho2 else recovery_error
    checks['endpoint_flags_passed'] = None if rho1 == rho2 else True
    geom_z, fit_z = model(hgeom), model(fit['h_mm'])
    def rmse(z):
        return float(np.sqrt(np.mean((np.asarray(z)-target)**2)))
    result = dict(tag=tag, status='single_pose_fixed_rho_model_comparison_not_validation',
        input_hashes=hashes, input_pair_contract='not_validated_here',
        legacy_debug_only=tag == 'disjoint_restartable_manifest90',
        units=dict(L='mm', h='mm', rho='ohm_m', Z='ohm'),
        rho1_ohm_m=rho1, rho2_ohm_m=rho2, nominal_electrodes_over_L=[-.5, -.25, .25, .5],
        contact_models=dict(planar='point_electrodes', FEM='saved_CEM_reference'),
        h_geom_mm=hgeom, h_geom_status='nearest_vertex_not_soft', h_range_mm=list(H_RANGE),
        h_eff=fit, h_eff_status='curve_fit_parameter_without_anatomical_interpretation',
        fit_method='1mm_grid_all_detected_local_minima_refined_including_endpoints',
        rmse_geom_ohm=rmse(geom_z), rmse_fit_ohm=rmse(fit_z), rmse_loo_ohm=rmse(predictions),
        loo_h_range_mm=[min(f['h_mm'] for f in loo), max(f['h_mm'] for f in loo)], loo=loo,
        curves=dict(L_mm=L.tolist(), FEM_ohm=target.tolist(), geom_ohm=geom_z.tolist(), fit_ohm=fit_z.tolist()),
        series_check=dict(N=terms, production_N=2*terms, max_N_2N_delta_ohm=step_delta,
                          uniform_production_truncation_bound_ohm=float(max(planar_z(L, rho1, rho2, 0., 2*terms)[1])),
                          checked_h_mm=probe_h, bound_excludes_roundoff=True), math_checks=checks,
        interpretation='Fit and leave-one-size-out measure approximation of nine FEM values at one pose; '
                       'they do not validate anatomy, FEM, experimental calibration or transfer to another pose. '
                       'Residuals include point-versus-finite electrode and projected-versus-nominal geometry differences; '
                       'they cannot be attributed only to curvature or a third tissue layer.')
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != sha for path, sha in hashes.items()):
        raise RuntimeError('Inputs changed during comparison; rerun after FEM artifacts are complete')
    output = HERE / f'compare_fixed_rho_h_{tag}.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=output.name, h_eff_mm=fit['h_mm'],
                         rmse_ohm=[rmse(geom_z), rmse(fit_z), rmse(predictions)], math_checks=checks)))


if __name__ == '__main__':
    main()
