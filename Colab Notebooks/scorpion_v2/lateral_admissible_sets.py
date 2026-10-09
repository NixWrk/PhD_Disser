"""Additional bounded families and the independent 140-mm cross observation."""
from pathlib import Path
import argparse

import numpy as np
from scipy.optimize import brentq, least_squares
from threadpoolctl import threadpool_limits

from lateral_pilot import ROOT, EXPERIMENT, DiskModel, prepare, read, save, sha
from lateral_identifiability import KernelBank, SIZES, IDS, H_BOUNDS, vector, log_jacobian

OUT = EXPERIMENT / 'derived/exploratory/33.21_review_20261009'


def profile(bank, y, bounds, indices=None):
    indices = np.arange(len(SIZES)) if indices is None else np.asarray(indices)
    f = bank.f[:, indices, :]
    lower = np.maximum(bounds[0, 0], bounds[1, 0] / bank.q)
    upper = np.minimum(bounds[0, 1], bounds[1, 1] / bank.q)
    valid = lower <= upper
    a = np.sum(f * np.asarray(y)[None, :, None], axis=1) / np.sum(f*f, axis=1)
    a = np.clip(a, lower[None], upper[None])
    rss = np.sum((f*a[:, None, :] - np.asarray(y)[None, :, None])**2, axis=1)
    rss[:, ~valid] = np.inf
    j = np.argmin(rss, axis=1)
    i = np.arange(len(bank.h))
    best = np.unravel_index(np.argmin(rss), rss.shape)
    return dict(scale=a, rss=rss, lower=lower, upper=upper,
                best=best, h=bank.h, rho1=a[i, j], rho2=a[i, j]*bank.q[j],
                h_rss=rss[i, j])


def bounded_fit(y, bank, bounds, indices=None):
    indices = np.arange(len(SIZES)) if indices is None else np.asarray(indices)
    p = profile(bank, y, bounds, indices)
    i, j = p['best']
    seed = np.array([p['scale'][i, j], p['scale'][i, j]*bank.q[j], bank.h[i]])
    lo, hi = np.log(np.r_[bounds[:, 0], H_BOUNDS[0]]), np.log(np.r_[bounds[:, 1], H_BOUNDS[1]])
    result = least_squares(lambda x: vector(np.exp(x))[indices]-y, np.clip(np.log(seed), lo, hi),
        jac=lambda x: log_jacobian(np.exp(x))[indices], bounds=(lo, hi),
        max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not result.success:
        raise ValueError('Bounded diagnostic fit did not converge')
    values = np.exp(result.x)
    z = vector(values)
    return dict(parameters=values.tolist(), rss=float(result.fun@result.fun),
                residual_percent=float(100*np.linalg.norm(result.fun)/np.linalg.norm(y)),
                prediction_ohm=z.tolist(), observations_ohm=np.asarray(y).tolist(),
                boundary_hits=(np.minimum(result.x-lo, hi-result.x)<1e-4).tolist(),
                bounds_ohm_m=bounds.tolist())


def cloud(bank, y, bounds, minimum, allowance=.1):
    f = bank.f
    a = np.sum(f*np.asarray(y)[None, :, None], axis=1)/np.sum(f*f, axis=1)
    rss = np.sum((f*a[:, None, :]-np.asarray(y)[None, :, None])**2, axis=1)
    width = np.sqrt(np.maximum((1+allowance)*minimum-rss, 0)/np.sum(f*f, axis=1))
    lower = np.maximum(bounds[0, 0], bounds[1, 0]/bank.q)[None]
    upper = np.minimum(bounds[0, 1], bounds[1, 1]/bank.q)[None]
    start, end = np.maximum(a-width, lower), np.minimum(a+width, upper)
    i, j = np.where((rss <= (1+allowance)*minimum) & (start <= end))
    rows = []
    for fraction in np.linspace(0, 1, 9):
        rho1 = start[i, j]+fraction*(end[i, j]-start[i, j])
        error = rss[i, j]+np.sum(f[i, :, j]**2, axis=1)*(rho1-a[i, j])**2
        rows.append(np.c_[rho1, rho1*bank.q[j], bank.h[i], error])
    return np.concatenate(rows) if rows else np.empty((0, 4))


def single_assembly(bank, y, bounds, index, fixed_rho1, fixed_rho2):
    # A scalar observation leaves a sampled relation, not a unique pair.
    scale = y/bank.f[:, index, :]
    rho2 = scale*bank.q[None]
    mask = ((scale >= bounds[0, 0]) & (scale <= bounds[0, 1]) &
            (rho2 >= bounds[1, 0]) & (rho2 <= bounds[1, 1]))
    envelopes = []
    for a in (scale, rho2):
        envelopes.extend([np.min(np.where(mask, a, np.inf), axis=1),
                          np.max(np.where(mask, a, -np.inf), axis=1)])
    envelopes = np.array(envelopes)
    envelopes[~np.isfinite(envelopes)] = np.nan
    conditional = np.full((2, 3, len(bank.h)), np.nan)
    max_error = 0.
    for ih, h in enumerate(bank.h):
        model = DiskModel([SIZES[index]], h, radial=2, angular=8, terms=2048)
        for variable, fixed_values in [(0, fixed_rho2), (1, fixed_rho1)]:
            for j, fixed in enumerate(fixed_values):
                def residual(x):
                    rho = [x, fixed] if variable == 0 else [fixed, x]
                    return model.evaluate(rho)[0][0]-y
                lo, hi = bounds[variable]
                if residual(lo)*residual(hi) <= 0:
                    value = brentq(residual, lo, hi, xtol=1e-11)
                    conditional[variable, j, ih] = value
                    max_error = max(max_error, abs(residual(value)))
    return envelopes, conditional, max_error


def run(out=OUT):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    observations, ensembles, contract = prepare(EXPERIMENT)
    registry = ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    literature = read(registry)
    compartments = literature['screening_profiles']['100000']['compartments']
    bounds = np.array([compartments[k]['rho_bounds_ohm_m'] for k in ('soft', 'lung_unspecified')])
    fixed_rho1 = [bounds[0, 0], literature['tissues']['Muscle']['gabriel_at_frequency']['100000']['rho_ohm_m'], bounds[0, 1]]
    fixed_rho2 = [bounds[1, 0], literature['tissues']['Lung (Inflated)']['gabriel_at_frequency']['100000']['rho_ohm_m'], bounds[1, 1]]
    q = np.unique(np.r_[np.geomspace(.001, 1000, 481), bounds[1, 0]/bounds[0, 1], bounds[1, 1]/bounds[0, 0]])
    results, comparisons = [], []
    observations.to_csv(out/'observations.local.csv', index=False)
    with threadpool_limits(3):
        bank = KernelBank(q=q)
        for channel in (1, 2):
            for state in ('inhale', 'exhale'):
                selected = observations.query('channel==@channel and state==@state').set_index('record_id')
                y = selected.loc[IDS].baseline_raw_median_ohm.to_numpy()
                fitted = bounded_fit(y, bank, bounds)
                p = profile(bank, y, bounds)
                points = cloud(bank, y, bounds, fitted['rss'])
                key = f'channel_{channel}_{state}'
                bands, lines = [], []
                errors = []
                for index in range(5):
                    band, line, error = single_assembly(bank, y[index], bounds, index, fixed_rho1, fixed_rho2)
                    bands.append(band); lines.append(line); errors.append(error)
                np.savez_compressed(out/(key+'.npz'), cloud=points, h=bank.h,
                    rho1=p['rho1'], rho2=p['rho2'], h_rss=p['h_rss'],
                    assembly_bands=np.array(bands), conditional_lines=np.array(lines),
                    fixed_rho1=fixed_rho1, fixed_rho2=fixed_rho2)
                results.append(dict(key=key, channel=channel, state=state, fit=fitted,
                    cloud_count=len(points), conditional_scalar_max_error_ohm=max(errors)))
                for record in ('004', '006'):
                    replacement = y.copy()
                    replacement[-1] = selected.loc['SCV2-G-'+record].baseline_raw_median_ohm
                    for area, limits in [('technical', np.array([[.1, 100.], [.1, 100.]])), ('literature', bounds)]:
                        candidate = bounded_fit(replacement, bank, limits)
                        # One unchanged four-size reference predicts both 140 observations.
                        four = bounded_fit(y[:4], bank, limits, np.arange(4))
                        comparisons.append(dict(channel=channel, state=state, area=area,
                            record_140=record, fit=candidate, four_size_prediction_140_ohm=four['prediction_ohm'][-1],
                            held_out_140_error_ohm=float(four['prediction_ohm'][-1]-replacement[-1])))
                print(key, 'literature fit', fitted['parameters'], 'residual', fitted['residual_percent'], flush=True)
    save(out/'results.local.json', results)
    save(out/'cross_comparison.local.json', comparisons)
    contract.update(owner='33.21 additional diagnostics', code_sha256=sha(__file__),
        literature_bounds_ohm_m=bounds.tolist(), literature_registry_sha256=sha(registry),
        literature_bounds_are_population_physiological_limits=False,
        physiological_interpretation='separate literature screening scenario; not a replacement for the technical inverse',
        fixed_rho1_ohm_m=fixed_rho1, fixed_rho2_ohm_m=fixed_rho2,
        h_grid_mm=bank.h.tolist(), q_grid=q.tolist(), scalar_observation_identifies_two_resistivities=False,
        cross_140_record='006 at 400 Hz, same numerical channel hypothesis as the entire series',
        cross_replacement_is_accepted=False)
    save(out/'input_contract.local.json', contract)
    save(out/'manifest.local.json', dict(code_sha256=sha(__file__),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.local.json'}))
    return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=OUT)
    run(parser.parse_args().out)
