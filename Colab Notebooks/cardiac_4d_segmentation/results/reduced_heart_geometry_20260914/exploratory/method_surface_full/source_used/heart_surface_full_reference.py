"""Full-quadrature sphere reference on the three frozen comparison examples.

No area-CDF subset enters fitting. G3a is analytic, G5 is linear LS, G3b/c
use bounded local LS with three starts and max_nfev per start. This removes
subset selection error ONLY, not surface-discretization or anatomical error.
The automatic target phase remains a computational comparator, not a manual
reference. No FEM or population inference. Execute from a captured source_used
directory to preserve code provenance; input/config files are never copied.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from heart_reduced_geometry import _ellipsoid, sampled_dice
from heart_construction_variants import surface_quadrature
from heart_additional_sphere_methods import (
    fit_surface_sphere_fixed_volume, fit_algebraic_sphere, sphere_surface_metrics,
)
from run_heart_method_comparison import get_mask, moments, read_json
from heart_ct_overlays import select_examples
from run_reduced_heart_geometry import digest, serial, write_json


METHODS = ('G3a', 'G3b', 'G3c', 'G5')


def fit_full_surface_free_center(points, weights, center, bounds, *, max_nfev=600):
    """Same geometric objective/domain/starts as project G3b, explicit budget.

    Three starts: volume centroid, surface quadrature centroid, bbox midpoint.
    Fixed radius bounds (eps, bbox diagonal), center bounded by surface bbox.
    Local least-squares convergence never certifies a global optimum.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    center = np.asarray(center, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or bounds.shape != (2, 3)
            or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1])
            or center.shape != (3,) or not np.isfinite(center).all()):
        raise ValueError('Invalid full surface, weights, center or bounds')
    if isinstance(max_nfev, bool) or int(max_nfev) != max_nfev or max_nfev <= 0:
        raise ValueError('max_nfev must be a positive integer')
    weights = weights/weights.sum()
    sw = np.sqrt(weights)
    lower = np.r_[bounds[0], np.finfo(float).eps]
    upper = np.r_[bounds[1], np.linalg.norm(bounds[1]-bounds[0])]

    def residual(p):
        return sw*(np.linalg.norm(points-p[:3], axis=1)-p[3])

    def jacobian(p):
        delta = p[:3]-points
        distance = np.maximum(np.linalg.norm(delta, axis=1), np.finfo(float).eps)
        return sw[:, None]*np.column_stack((delta/distance[:, None], -np.ones(len(points))))

    trials = []
    for start in (center, weights@points, bounds.mean(axis=0)):
        start = np.clip(start, lower[:3]+1e-7, upper[:3]-1e-7)
        r0 = float(weights@np.linalg.norm(points-start, axis=1))
        result = least_squares(residual, np.r_[start, r0], jac=jacobian,
            bounds=(lower, upper), ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=max_nfev)
        trials.append({'start_mm': start.tolist(), 'parameters_mm': result.x.tolist(),
            'success': bool(result.success), 'nfev': int(result.nfev),
            'njev': int(result.njev), 'RMS_mm': float(np.linalg.norm(result.fun)),
            'active_bounds': result.active_mask.tolist(), 'message': str(result.message)})
    converged = [t for t in trials if t['success']]
    best = min(converged or trials, key=lambda t: t['RMS_mm'])
    p = np.asarray(best['parameters_mm'])
    return _ellipsoid('sphere_surface_free_center', p[:3], np.full(3, p[3]), np.eye(3), True,
        radius_mm=float(p[3]), preserves_volume=False,
        status='local_solver_converged' if converged else 'solver_not_converged',
        surface_RMS_mm=best['RMS_mm'],
        criterion={'name': 'area_weighted_squared_radial_distance',
                   'value': best['RMS_mm']**2, 'units': 'mm2'},
        optimization={'success': bool(converged), 'trials': trials,
            'global_optimum_certified': False, 'max_nfev_per_start': max_nfev,
            'center_bounds_mm': bounds.tolist(), 'radius_bounds_mm': [lower[3], upper[3]],
            'all_surface_points_used': True})


def full_sphere_metrics(mask, affine, m, candidate, points, weights):
    """Common Dice/volume/center plus exact sphere distance, full area weights.

    Radial distance equals Euclidean distance to a sphere; use the explicit
    formula instead of invoking the general ellipsoid distance root solver.
    """
    candidate = dict(candidate, sampling_family='ellipsoid')
    overlap = sampled_dice(mask, affine, candidate, target_moments=m)
    surface = sphere_surface_metrics(points, weights, candidate['center_mm'], candidate['radius_mm'])
    total_weight = float(np.sum(weights))
    mean = 0.
    for start in range(0, len(points), 65536):
        d = np.abs(np.linalg.norm(points[start:start+65536]-candidate['center_mm'], axis=1)
                   -candidate['radius_mm'])
        mean += float(weights[start:start+65536]@d)/total_weight
    return {'dice': overlap['dice_voxel_center_approx'], 'overlap': overlap,
        'volume_error_percent': 100*(candidate['volume_mm3']/m['volume_mm3']-1),
        'center_shift_mm': float(np.linalg.norm(candidate['center_mm']-m['centroid_mm'])),
        'target_surface_to_candidate_RMS_mm': surface['surface_RMS_mm'],
        'target_surface_to_candidate_mean_mm': mean,
        'surface_quadrature_point_count': len(points),
        'algebraic_RMS_mm2': surface['algebraic_RMS_mm2']}


def compare_to_subset(reference_rows, subset_results):
    subset = {(r['case_id'], r['method_id']): r for r in subset_results['rows']}
    comparisons = []
    for full in reference_rows:
        key = (full['case_id'], full['method_id'])
        low = subset.get(key)
        if low is None or full.get('metrics') is None or low.get('metrics') is None:
            comparisons.append({'case_id': key[0], 'method_id': key[1], 'status': 'missing_metrics'})
            continue
        fp, lp = full['parameters'], low['parameters']
        fm, lm = full['metrics'], low['metrics']
        if full['target_mask_sha256'] != low['target_mask_sha256']:
            raise ValueError('Subset/reference target mask mismatch')
        # Covariance tensor of a homogeneous sphere: r^2 I / 5.
        tensor_full = np.eye(3)*fp['radius_mm']**2/5
        tensor_low = np.eye(3)*lp['radius_mm']**2/5
        comparisons.append({'case_id': key[0], 'method_id': key[1], 'status': 'computed',
            'volume_change_mm3': fp['volume_mm3']-lp['volume_mm3'],
            'volume_relative_change_percent': 100*(fp['volume_mm3']-lp['volume_mm3'])/fp['volume_mm3'],
            'radius_change_mm': fp['radius_mm']-lp['radius_mm'],
            'center_change_vector_mm': (np.asarray(fp['center_mm'])-lp['center_mm']).tolist(),
            'center_change_norm_mm': float(np.linalg.norm(np.asarray(fp['center_mm'])-lp['center_mm'])),
            'shape_tensor_frobenius_change_mm2': float(np.linalg.norm(tensor_full-tensor_low)),
            'shape_tensor_relative_change_percent': 100*float(np.linalg.norm(tensor_full-tensor_low)/np.linalg.norm(tensor_full)),
            'full_surface_RMS_subset_fit_mm': lm['target_surface_to_candidate_RMS_mm'],
            'full_surface_RMS_full_fit_mm': fm['target_surface_to_candidate_RMS_mm'],
            'full_surface_RMS_change_mm': fm['target_surface_to_candidate_RMS_mm']-lm['target_surface_to_candidate_RMS_mm'],
            'dice_subset_fit': lm['dice'], 'dice_full_fit': fm['dice'],
            'dice_change_percentage_points': 100*(fm['dice']-lm['dice'])})
    return comparisons


def run(args):
    config = read_json(args.config)
    root = Path(config['derived_root'])
    baseline = Path(args.baseline)/'geometry_comparison.jsonl'
    rows = [json.loads(line) for line in baseline.read_text(encoding='utf-8').splitlines()]
    selected = select_examples(rows)
    out = Path(args.output)
    final = Path(args.subset_comparison)
    subset_results = read_json(final/'results.json')
    if read_json(final/'status.json')['status'] != 'completed':
        raise ValueError('Coherent final comparison must complete before full reference')
    if any((out/name).exists() for name in ('protocol.json', 'results.json', 'status.json')):
        raise ValueError('New output required; no implicit resume')
    if out.exists() and any(p.name not in ('source_used', 'source_manifest.json') for p in out.iterdir()):
        raise ValueError('Unexpected output contents')
    out.mkdir(parents=True, exist_ok=True)
    code_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    protocol = {'schema': 'heart_full_surface_reference_v1', 'baseline_sha256': digest(baseline),
        'subset_comparison_results_sha256': digest(final/'results.json'),
        'methods': list(METHODS), 'selection': 'same_three_examples',
        'fit_surface': 'all_area_weighted_surface_quadrature_points',
        'metric_surface': 'all_area_weighted_surface_quadrature_points',
        'max_nfev_per_start_G3b_G3c': args.max_nfev, 'n_starts_G3b_G3c': 3,
        'code_sha256': code_hashes, 'FEM_executed': False}
    write_json(out/'protocol.json', protocol)
    write_json(out/'status.json', {'status': 'running', 'requested': 12, 'completed': 0})
    started = time.monotonic()
    completed = []
    memory_checks = []
    for row, ref in selected:
        mask, affine = get_mask(root, row)
        m = moments(row)
        # Marching-cubes triangle and quadrature arrays will be temporary.
        points, weights, surface = surface_quadrature(mask, affine)
        n = len(points)
        # Conservative working-array estimate, not measured peak RSS: full
        # points/weights, residuals, Jacobian, QR/SVD workspace, trial arrays.
        estimate = int(mask.nbytes+points.nbytes+weights.nbytes+n*8*40)
        memory = {'case_id': f"{row['subject']}_{row['phase_id']}_{row['region']}",
            'mask_bytes': mask.nbytes, 'surface_point_count': n,
            'points_and_weights_bytes': points.nbytes+weights.nbytes,
            'estimated_solver_working_bytes': estimate, 'measured_peak_RSS': False,
            'estimate_includes_surface_extraction_peak': False}
        memory_checks.append(memory)
        write_json(out/'memory_checks.json', memory_checks)
        print('MEMORY_ESTIMATE', json.dumps(memory), flush=True)
        if estimate > args.max_working_mb*1024**2:
            raise MemoryError('Working estimate exceeds explicit memory guard')
        center = m['centroid_mm']; bounds = np.asarray(surface['bounds_mm'])
        w = weights/weights.sum()
        starts = np.stack((center, w@points, bounds.mean(axis=0)))
        common = {'case_id': memory['case_id'], 'subject': row['subject'],
            'phase_id': row['phase_id'], 'region': row['region'], 'cycle_index': row['cycle_index'],
            'reference_phase_id': ref['phase_id'], 'is_reference_phase': row['is_reference_phase'],
            'target_mask_sha256': row['provenance']['sha256'],
            'target_touches_CT_boundary': row['provenance']['touches_source_image_boundary'],
            'target_volume_ml': m['volume_ml'], 'surface': surface,
            'anatomical_reference_independent': False,
            'fit_sampling': {'method': 'full_area_quadrature_no_subset', 'point_count': n},
            'evaluation_sampling': {'method': 'full_area_quadrature_no_subset', 'full_point_count': n}}
        for method in METHODS:
            begin = time.monotonic()
            record = dict(common, method_id=method)
            try:
                if method == 'G3a':
                    radius = float(w@np.linalg.norm(points-center, axis=1))
                    candidate = _ellipsoid('sphere_surface_fixed_center', center,
                        np.full(3, radius), np.eye(3), True, radius_mm=radius,
                        preserves_volume=False, status='analytic_radius_solved',
                        criterion={'name': 'area_weighted_squared_radial_distance', 'units': 'mm2'},
                        diagnostics={'analytic_optimum_for_fixed_center': True})
                elif method == 'G3b':
                    candidate = fit_full_surface_free_center(points, weights, center, bounds, max_nfev=args.max_nfev)
                elif method == 'G3c':
                    candidate = fit_surface_sphere_fixed_volume(points, weights, m['volume_mm3'],
                        bounds, starts, max_nfev=args.max_nfev)
                else:
                    candidate = fit_algebraic_sphere(points, weights)
                metrics = full_sphere_metrics(mask, affine, m, candidate, points, weights)
                if method == 'G3a':
                    candidate['criterion']['value'] = metrics['target_surface_to_candidate_RMS_mm']**2
                record.update(status=candidate['status'], parameters=serial(candidate), metrics=serial(metrics),
                    surface_metric_rule='full_target_surface_area_quadrature_not_fit_subset',
                    metric_implementation='analytic_sphere_distance_equals_absolute_radial_residual')
            except (ValueError, RuntimeError, FloatingPointError, MemoryError) as exc:
                record.update(status='failed', reason=str(exc), metrics=None)
            record['elapsed_seconds'] = time.monotonic()-begin
            write_json(out/f"{common['case_id']}__{method}.json", record)
            completed.append(record)
            write_json(out/'status.json', {'status': 'running', 'requested': 12, 'completed': len(completed)})
            print(common['case_id'], method, record['status'], round(record['elapsed_seconds'], 2), flush=True)
    comparisons = compare_to_subset(completed, subset_results)
    result = {'schema': 'heart_method_comparison_v1', 'status': 'exploratory_hypothesis_not_validated',
        'scope': 'three_examples_full_quadrature_sphere_reference', 'protocol': protocol, 'rows': completed,
        'FEM_executed_by_this_runner': False, 'T0_is_automatic_target_phase_mask': True,
        'I_is_transformed_individual_reference_shape': True}
    write_json(out/'results.json', result)
    audit = {'schema': 'heart_full_surface_reference_audit_v1',
        'status': 'exploratory_hypothesis_not_validated', 'comparisons_to_final4096': comparisons,
        'difference_sign': 'full fit minus subset fit', 'relative_denominator': 'full fit',
        'memory_checks': memory_checks, 'elapsed_seconds': time.monotonic()-started,
        'report_ru': [
            'Подгонка всех четырёх сфер использовала полный набор поверхностной квадратуры. Ошибка выбора подмножества area-CDF устранена; поверхность маски и её квадратура не изменялись.',
            'Г3а решён аналитически при фиксированном центре; Г5 — линейным методом наименьших квадратов. Г3б/Г3в проверены из трёх начальных центров с ограничениями; сходимость отдельных стартов не является сертификатом глобального оптимума.',
            'Параметры и метрики сопоставлены с новым согласованным расчётом method_comparison_final, а не с пилотом 1024. У обеих сторон итоговая RMS вычислена по всей поверхности.',
            'Тензор формы — ковариация однородной сферы r²I/5. Он зависит только от радиуса и не даёт независимой информации о несферичности.',
            'Эти результаты не удостоверяют точность автоматической сегментации, сходимость самой поверхностной квадратуры, электрическую эквивалентность или популяционную применимость.'],
        'global_geometric_optimum_certified': False, 'surface_quadrature_convergence_certified': False,
        'population_validation': False, 'results_sha256': digest(out/'results.json')}
    write_json(out/'full_reference_audit.json', audit)
    write_json(out/'status.json', {'status': 'completed', 'requested': 12, 'completed': len(completed),
        'failures': sum(r.get('metrics') is None for r in completed)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--subset-comparison', required=True)
    parser.add_argument('--max-nfev', type=int, default=600)
    parser.add_argument('--max-working-mb', type=float, default=2048.)
    run(parser.parse_args())
