"""33.21: solution sets and stability of finite-contact two-layer inversion.

Literature ranges do not constrain the inverse. Stored 11.20 annotations retain
their candidate status. Within-record bootstrap is explicitly conditional.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from lateral_pilot import (DiskModel, EXPERIMENT, PRIMARY, ROOT, STATUS,
                           prepare, read, save, sha)

BOUNDS = np.array([[.1, 100.], [.1, 100.]])
H_BOUNDS = np.array([.5, 100.])
SIZES = np.array(sorted(PRIMARY.values()))
IDS = ['SCV2-G-' + next(k for k, v in PRIMARY.items() if v == size)
       for size in SIZES]


class KernelBank:
    """Finite disks, densely sampled h and contrast, exact scale profiling.

At fixed q=rho2/rho1 and h, Z=rho1*f(q,h). The minimizing positive rho1
is a bounded one-dimensional quadratic minimizer, not an extra approximation.
"""

    def __init__(self, h=None, q=None):
        self.h = np.unique(np.r_[np.geomspace(*H_BOUNDS, 101),
                                 np.arange(1., 61.), np.arange(65., 101., 5.)]
                           if h is None else h)
        self.q = np.geomspace(.001, 1000., 241) if q is None else np.asarray(q)
        self.k = (self.q - 1) / (self.q + 1)
        powers = self.k[None, :] ** np.arange(1, 2049)[:, None]
        self.f = np.empty((len(self.h), len(SIZES), len(self.q)))
        for i, thickness in enumerate(self.h):
            model = DiskModel(SIZES, float(thickness), radial=2, angular=8,
                              terms=2048)
            self.f[i] = (model.base[:, None] + 2 * model.images @ powers) / np.pi

    def profile(self, y, indices=None):
        indices = np.arange(len(SIZES)) if indices is None else np.asarray(indices)
        y = np.asarray(y)
        f = self.f[:, indices, :]
        lower = np.maximum(BOUNDS[0, 0], BOUNDS[1, 0] / self.q)
        upper = np.minimum(BOUNDS[0, 1], BOUNDS[1, 1] / self.q)
        scale = np.sum(f * y[None, :, None], axis=1) / np.sum(f * f, axis=1)
        scale = np.clip(scale, lower[None], upper[None])
        residual = scale[:, None, :] * f - y[None, :, None]
        sse = np.sum(residual ** 2, axis=1)
        j = np.argmin(sse, axis=1)
        i = np.arange(len(self.h))
        return dict(sse=sse, scale=scale,
                    h_profile_sse=sse[i, j],
                    h_profile_rho1=scale[i, j],
                    h_profile_rho2=scale[i, j] * self.q[j],
                    best_index=np.unravel_index(np.argmin(sse), sse.shape))

    def seed(self, y, indices=None):
        p = self.profile(y, indices)
        i, j = p['best_index']
        return np.array([p['scale'][i, j], p['scale'][i, j] * self.q[j], self.h[i]])

    def resistance_profiles(self, y):
        """Profile each rho over the other rho and h on a declared log grid."""
        rho=np.geomspace(.1,100.,61)
        i,j=np.indices((61,61))
        ratio_index=120+2*(j-i)
        z=self.f.transpose(0,2,1)[:,ratio_index,:]*rho[None,:,None,None]
        sse=np.sum((z-np.asarray(y)[None,None,None,:])**2,axis=-1)
        return dict(profile_rho_axis=rho,
                    profile_rho1_sse=np.min(sse,axis=(0,2)),
                    profile_rho2_sse=np.min(sse,axis=(0,1)),
                    rho1_rho2_profile_sse=np.min(sse,axis=0))

    def cloud(self, y, minimum, indices=None):
        p = self.profile(y, indices)
        indices = np.arange(len(SIZES)) if indices is None else np.asarray(indices)
        # Include every scale satisfying SSE <= 1.10*minimum at every (h,q).
        # SSE(a)=SSE(a*)+sum(f^2)*(a-a*)^2; evaluate bounded endpoints too.
        f = self.f[:, indices, :]
        unconstrained = np.sum(f * np.asarray(y)[None, :, None], axis=1) / np.sum(f*f, axis=1)
        unconstrained_sse = np.sum((unconstrained[:, None] * f - np.asarray(y)[None, :, None])**2, axis=1)
        allowance = 1.10 * minimum - unconstrained_sse
        width = np.sqrt(np.maximum(allowance, 0) / np.sum(f*f, axis=1))
        lower = np.maximum(BOUNDS[0, 0], BOUNDS[1, 0] / self.q)[None]
        upper = np.minimum(BOUNDS[0, 1], BOUNDS[1, 1] / self.q)[None]
        start = np.maximum(unconstrained - width, lower)
        end = np.minimum(unconstrained + width, upper)
        valid = (allowance >= 0) & (end >= start)
        i, j = np.where(valid)
        values = []
        for fraction in np.linspace(0, 1, 9):
            r1 = start[i, j] + fraction * (end[i, j] - start[i, j])
            sse = unconstrained_sse[i, j] + np.sum(f[i, :, j]**2, axis=1) * (r1-unconstrained[i,j])**2
            values.append(np.c_[r1, r1*self.q[j], self.h[i], sse])
        result = np.concatenate(values) if values else np.empty((0,4))
        return result, p


@lru_cache(maxsize=24)
def model_at(h):
    return DiskModel(SIZES, float(h), radial=2, angular=8, terms=2048)


def vector(parameters):
    return model_at(float(parameters[2])).evaluate(np.asarray(parameters[:2]))[0]


def log_jacobian(parameters):
    model = model_at(float(parameters[2]))
    _, j = model.evaluate(np.asarray(parameters[:2]))
    step = 1e-4
    plus = np.asarray(parameters, float).copy()
    minus = plus.copy()
    plus[2] *= np.exp(step)
    minus[2] *= np.exp(-step)
    return np.column_stack([j, (vector(plus)-vector(minus))/(2*step)])


def fit(y, bank, indices=None, seed=None):
    indices = np.arange(len(SIZES)) if indices is None else np.asarray(indices)
    seed = bank.seed(y, indices) if seed is None else np.asarray(seed)
    bounds = np.log(np.r_[BOUNDS[:, 0], H_BOUNDS[0]]), np.log(np.r_[BOUNDS[:, 1], H_BOUNDS[1]])
    solution = least_squares(lambda x: vector(np.exp(x))[indices] - y,
                             np.log(seed),
                             jac=lambda x: log_jacobian(np.exp(x))[indices],
                             bounds=bounds, max_nfev=100,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not solution.success:
        raise RuntimeError('Continuous two-layer fit did not converge')
    parameters = np.exp(solution.x)
    prediction = vector(parameters)
    j = log_jacobian(parameters)[indices]
    singular = np.linalg.svd(j, compute_uv=False)
    rank = int(np.linalg.matrix_rank(j))
    distance = np.minimum(solution.x-bounds[0], bounds[1]-solution.x)
    return dict(parameters=parameters.tolist(), sse=float(solution.fun @ solution.fun),
                relative_residual_percent=float(100*np.linalg.norm(solution.fun)/np.linalg.norm(y)),
                rmse_ohm=float(np.sqrt(np.mean(solution.fun**2))),
                observed_ohm=np.asarray(y).tolist(), predicted_all_sizes_ohm=prediction.tolist(),
                rank=rank, singular_values_ohm=singular.tolist(),
                condition_log_parameters=float(singular[0]/singular[-1]) if rank==3 else None,
                boundary_hits=(distance<1e-4).tolist(), selected_indices=indices.tolist(),
                jacobian_log_parameters_ohm=j.tolist(), nfev=solution.nfev)


def bootstrap_baselines(experiment, ensembles, draws=120, seed=20261008):
    rng = np.random.default_rng(seed)
    results = {}
    for rid in IDS:
        fs = ensembles[rid]['fs_hz_scenario']
        review = read(experiment/'derived/scorpion_review/summary.local.json')
        record = next(r for r in review['records'] if r['record_id']==rid and r['fs_hz_scenario']==fs)
        with np.load(experiment/'derived/scorpion_review'/(record['key']+'.npz')) as data:
            for state, window in ensembles[rid]['windows'].items():
                left, right = window['sample_interval_half_open']
                block = int(round(fs))
                for channel in (1,2):
                    raw = data[f'bei_raw_{channel}'][left:right]
                    blocks = [raw[i:i+block] for i in range(0,len(raw)-block+1,block)]
                    if len(blocks)<3:
                        raise ValueError('At least three complete one-second blocks required')
                    samples = np.vstack(blocks)
                    baselines = np.array([np.median(samples[rng.integers(0,len(blocks),len(blocks))])
                                          for _ in range(draws)])
                    results[rid,state,channel] = dict(samples=baselines,
                        p05_p95=np.quantile(baselines,[.05,.95]).tolist(),
                        complete_blocks=len(blocks), omitted_tail_samples=len(raw)%block)
    return results


def verify_numerics(bank, cases):
    rows = []
    for p in ([.1,100,.5],[100,.1,.5],[4,16,30],
              *[case['fit']['parameters'] for case in cases]):
        rho1,rho2,h = p
        direct = DiskModel(SIZES,h,radial=4,angular=16,terms=4096).evaluate([rho1,rho2])[0]
        fast = vector(p)
        error = float(np.max(abs(direct-fast)))
        if error > .001:
            raise ValueError('Finite disk verification exceeds 0.001 ohm')
        rows.append(dict(parameters=p, max_abs_error_ohm=error))
    truth=np.array([4.,16.,30.])
    recovered=fit(vector(truth),bank)
    error=np.max(abs(np.array(recovered['parameters'])/truth-1))
    if error>1e-4:
        raise ValueError('Synthetic three-parameter recovery failed')
    return dict(status='internal_numerical_verification',checks=rows,
                maximum_abs_difference_ohm=max(r['max_abs_error_ohm'] for r in rows),
                synthetic_truth=truth.tolist(),synthetic_relative_parameter_error=float(error))


def run(experiment=EXPERIMENT, out=None):
    experiment=Path(experiment)
    out=Path(out) if out else experiment/'derived/exploratory/33.21_identifiability'
    out.mkdir(parents=True,exist_ok=True)
    observations,ensembles,contract=prepare(experiment)
    bootstrap=bootstrap_baselines(experiment,ensembles)
    with threadpool_limits(3):
        bank=KernelBank()
        cases=[]
        for channel in (1,2):
            for state in ('inhale','exhale'):
                table=observations.query('channel==@channel and state==@state').set_index('record_id').loc[IDS]
                y=table.baseline_raw_median_ohm.to_numpy()
                fitted=fit(y,bank)
                cloud,profile=bank.cloud(y,fitted['sse'])
                resistance_profiles=bank.resistance_profiles(y)
                key=f'channel_{channel}_{state}'
                np.savez_compressed(out/(key+'.npz'),cloud=cloud,h=bank.h,q=bank.q,
                                    sse_h_q=profile['sse'],scale_h_q=profile['scale'],
                                    h_profile_sse=profile['h_profile_sse'],
                                    h_profile_rho1=profile['h_profile_rho1'],h_profile_rho2=profile['h_profile_rho2'],
                                    **resistance_profiles)
                leave_out=[]
                for omitted in range(len(SIZES)):
                    indices=np.delete(np.arange(len(SIZES)),omitted)
                    row=fit(y[indices],bank,indices)
                    row['omitted_size_mm']=int(SIZES[omitted])
                    row['held_out_residual_ohm']=row['predicted_all_sizes_ohm'][omitted]-float(y[omitted])
                    leave_out.append(row)
                resamples=np.array([bootstrap[rid,state,channel]['samples'] for rid in IDS]).T
                estimates=[]
                for resampled in resamples:
                    estimates.append(bank.seed(resampled).tolist())
                fits=np.array(estimates)
                tolerance_sets={}
                for allowance in (.01,.05,.10):
                    selected=cloud[cloud[:,3]<=fitted['sse']*(1+allowance)*(1+1e-12)]
                    allp=np.vstack([selected[:,:3],np.array(fitted['parameters'])[None]])
                    tolerance_sets[str(allowance)]=dict(count=int(len(selected)),
                        parameter_min=allp.min(axis=0).tolist(),parameter_max=allp.max(axis=0).tolist())
                case=dict(key=key,channel=channel,state=state,fit=fitted,
                          leave_one_out=leave_out,near_minimum_sets=tolerance_sets,
                          bootstrap=dict(draws=len(fits),seed=20261008,block_s=1,
                              p05_p95_parameters=np.quantile(fits,[.05,.95],axis=0).tolist(),
                              parameter_samples=fits.tolist(),grid_estimates_only=True,
                              observation_p05_p95=[bootstrap[rid,state,channel]['p05_p95'] for rid in IDS],
                              scope='within-record variability at fixed candidate windows; not full uncertainty or confidence interval'),
                          cloud_file=key+'.npz')
                cases.append(case)
                print(key,'rho1 rho2 h',np.round(fitted['parameters'],6),
                      'residual %',round(fitted['relative_residual_percent'],3),
                      'cond',round(fitted['condition_log_parameters'],2),flush=True)
        verification=verify_numerics(bank,cases)
    registry=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    contract.update(owner='33.21',code_sha256=sha(Path(__file__)),
                    previous_pilot_code_sha256=sha(Path(__file__).with_name('lateral_pilot.py')),
                    search_bounds_rho_ohm_m=BOUNDS.tolist(),search_bounds_h_mm=H_BOUNDS.tolist(),
                    literature_ranges_are_inverse_bounds=False,literature_registry_sha256=sha(registry),
                    grid_h_mm=bank.h.tolist(),grid_ratio=bank.q.tolist(),
                    numerical_model=dict(radial=2,angular=8,image_terms=2048,diameter_mm=5),
                    near_minimum_sse_relative_allowances=[.01,.05,.10],
                    tolerance_sets_are_confidence_regions=False,
                    ct_fem_operator_used=False,cardiac_function_estimation_performed=False,
                    applicability_verdict='conditional diagnostics; full uncertainty and individual CT validation remain unresolved')
    save(out/'input_contract.local.json',contract)
    save(out/'results.local.json',cases)
    save(out/'verification.local.json',verification)
    from build_lateral_identifiability_report import build
    build(out,observations,cases,contract,verification)
    save(out/'manifest.local.json',dict(status=STATUS,owner='33.21',
        code_sha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.local.json'}))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=EXPERIMENT)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    run(args.experiment,args.out)
