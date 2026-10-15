"""Algebra of a completed inverse evaluation; no new FEM or altered objective.

Separates pair means/contrasts, local material sensitivities and explicitly
unmeasured two-source scenarios. A bounded linear contrast projection is a
local diagnostic, not a nonlinear fit or an impossibility proof.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def analyse(row, observations, bounds):
    row = row.get('best', row)
    if row.get('status') in ('invalid_geometry', 'forward_calculation_running'):
        raise ValueError('A completed FEM evaluation is required')
    obs = observations[observations.state.isin(['inhale', 'exhale'])].reset_index(drop=True)
    predicted = np.asarray(row['predicted_Z_ohm'], float).reshape(-1)
    target = np.asarray(row['target_Z_ohm'], float).reshape(-1)
    J = np.asarray(row['jacobian_logrho'], float)
    if (len(obs) != 24 or predicted.shape != (24,) or target.shape != (24,) or J.shape != (24, 6) or
            not np.all(np.isfinite(np.r_[predicted, target, J.ravel()])) or
            not np.allclose(obs.Z_ohm, target, rtol=0, atol=5e-13)):
        raise ValueError('Expected the unchanged 24 levels and finite material Jacobian')
    pairs = []
    for (record, axis), block in obs.groupby(['record_id', 'axis'], sort=False):
        if len(block) != 2 or set(block.state) != {'inhale', 'exhale'}:
            raise ValueError('Every lead must contain exactly one level per phase')
        i = int(block.index[block.state == 'inhale'][0])
        e = int(block.index[block.state == 'exhale'][0])
        pairs.append((i, e, record, axis))
    if len(pairs) != 12:
        raise ValueError('All twelve respiratory pairs are required')
    ii = np.array([p[0] for p in pairs]); ee = np.array([p[1] for p in pairs])
    error = predicted-target
    means_error = (error[ii]+error[ee])/2
    contrast_error = error[ee]-error[ii]
    sse = float(error@error)
    mean_sse = float(2*means_error@means_error)
    contrast_sse = float(.5*contrast_error@contrast_error)
    if not np.isclose(sse, mean_sse+contrast_sse, rtol=1e-12, atol=1e-12):
        raise ValueError('Pair-error decomposition failed')
    measured_delta = target[ee]-target[ii]
    model_delta = predicted[ee]-predicted[ii]
    params = np.asarray(row.get('parameters', np.log(np.r_[row.get('rho_inhale_ohm_m', []),
                                                         row.get('rho_exhale_ohm_m', [])])), float).reshape(-1)[:6]
    bound = np.log(np.asarray(bounds, float))
    if params.shape != (6,) or bound.shape != (3, 2):
        raise ValueError('Six log-resistivities and three bound pairs required')
    lower, upper = np.tile(bound[:, 0], 2), np.tile(bound[:, 1], 2)
    if np.any(params < lower-1e-8) or np.any(params > upper+1e-8):
        raise ValueError('Completed parameters are outside the declared bounds')
    params = np.clip(params, lower, upper)  # rounding only at original bounds
    A = J[ee]-J[ii]
    projection = lsq_linear(A, measured_delta-model_delta,
                           bounds=(lower-params, upper-params), tol=1e-10)
    linear_delta = model_delta+A@projection.x
    sensitivity = []
    for state, columns in [('inhale', slice(0, 3)), ('exhale', slice(3, 6))]:
        matrix = J[obs.state == state, columns]
        norms = np.linalg.norm(matrix, axis=0)
        cosine = matrix.T@matrix/np.maximum(np.outer(norms, norms), np.finfo(float).tiny)
        sv = np.linalg.svd(matrix, compute_uv=False)
        sensitivity.append(dict(state=state, norms=norms.tolist(), singular_values=sv.tolist(),
            cosine=cosine.tolist(), condition_number=float(sv[0]/sv[-1]) if sv[-1] else None,
            derivative_units='Ohm per unit log-resistivity; local at saved pose and materials'))
    scenarios = []
    own = row.get('own_transfer_ohm', row.get('own_drive_transfer_ohm'))
    counter = row.get('counter_transfer_ohm', row.get('counter_drive_transfer_ohm'))
    if own is not None and counter is not None:
        own, counter = np.asarray(own).reshape(-1), np.asarray(counter).reshape(-1)
        if own.shape != (24,) or counter.shape != (24,) or not np.all(np.isfinite(np.r_[own,counter])):
            raise ValueError('Invalid source transfer vectors')
        for name, values in [('own_drive',own), ('in_phase',own+counter),
                             ('opposite_phase',own-counter), ('quadrature_magnitude',np.hypot(own,counter))]:
            delta = values[ee]-values[ii]
            scenarios.append(dict(name=name, RMSE_ohm=float(np.sqrt(np.mean((values-target)**2))),
                contrast_ohm=delta.tolist(), wrong_signs=int(np.count_nonzero(np.sign(delta)!=np.sign(measured_delta))),
                equal_source_amplitudes_assumed=name!='own_drive', experimentally_confirmed=False,
                resistivities_or_pose_refitted=False))
    return dict(schema='georg_completed_point_residual_diagnostics_v1',
        observation_count=24, pair_count=12, quantitative_result_accepted=False,
        independently_validated=False, original_inverse_objective_unchanged=True,
        SSE_ohm2=sse, mean_SSE_ohm2=mean_sse, contrast_SSE_ohm2=contrast_sse,
        contrast_SSE_fraction=contrast_sse/sse if sse else 0,
        RMSE_ohm=float(np.sqrt(sse/24)), gradient_MSE_per_logrho=(2*J.T@error/24).tolist(),
        lower_bound_parameters=np.isclose(params,lower,rtol=0,atol=1e-4).tolist(),
        upper_bound_parameters=np.isclose(params,upper,rtol=0,atol=1e-4).tolist(),
        pairs=[dict(record_id=p[2],axis=p[3],measured_contrast_ohm=float(measured_delta[k]),
            model_contrast_ohm=float(model_delta[k]),mean_error_ohm=float(means_error[k]),
            pair_SSE_fraction=float((error[p[0]]**2+error[p[1]]**2)/sse) if sse else 0)
            for k,p in enumerate(pairs)], local_sensitivity=sensitivity, source_scenarios=scenarios,
        bounded_linear_contrast_diagnostic=dict(logrho_step=projection.x.tolist(),
            linear_contrast_ohm=linear_delta.tolist(), solver_success=bool(projection.success),
            largest_absolute_log_step=float(np.max(np.abs(projection.x))),
            contrast_RMSE_ohm=float(np.sqrt(np.mean((linear_delta-measured_delta)**2))),
            local_linearization_validity_verified=False, mean_levels_constrained=False,
            nonlinear_feasibility_or_impossibility_proven=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'observations', 'inputs', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    row, inputs = read(args.candidate), read(args.inputs)
    candidate = row.get('best', row)
    if candidate['identity']['observations_sha256'] != sha(args.observations):
        raise ValueError('Observation identity changed')
    result = analyse(row, pd.read_csv(args.observations), inputs['search_bounds_ohm_m'])
    result['sources'] = [dict(path=str(p.resolve()),sha256=sha(p)) for p in
                         (args.candidate,args.observations,args.inputs,Path(__file__))]
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(dict(RMSE_ohm=result['RMSE_ohm'],contrast_SSE_fraction=result['contrast_SSE_fraction']))


if __name__ == '__main__':main()
