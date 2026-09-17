"""Project sphere methods Г3в, Г4а/б and Г5; NumPy/SciPy geometry only.

Coordinates/affines are physical mm; volumes are mm^3. Surface weights must
represent quadrature areas, not vertex multiplicities (normalization is local).
No equivalence to MATLAB or RFBR implementations is asserted. The automatic
target-phase mask and transformed individual reference shape are separate
anatomical representations; neither is an independently validated reference.

Returned dictionaries follow the ellipsoid parameter layout of the geometry
core but use NEW kind identifiers. The current core's sampled_dice dispatch
does not yet recognize them; use sampled_sphere_iou here or integrate dispatch
explicitly. No existing core code, masks or input arrays are modified.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares

from heart_reduced_geometry import _affine, _ellipsoid, _lookup, mask_moments


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _surface(points_mm, area_weights):
    points = np.asarray(points_mm, dtype=float)
    weights = np.asarray(area_weights, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
            or weights.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError('Expected at least four finite 3D points and nonnegative area weights')
    weights = weights / weights.max()
    return points, weights / weights.sum()


def _domain(bounds, starts, dimensions):
    bounds = np.asarray(bounds, dtype=float)
    starts = np.asarray(starts, dtype=float)
    if (bounds.shape != (2, dimensions) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1])):
        raise ValueError('Bounds must have shape (2, dimensions) with strictly ordered limits')
    if (starts.ndim != 2 or starts.shape[1] != dimensions or len(starts) == 0
            or not np.isfinite(starts).all()
            or np.any(starts < bounds[0]) or np.any(starts > bounds[1])):
        raise ValueError('Explicit starts must be finite and inside the bounds; no silent clipping')
    return bounds, starts


def _sphere(kind, center, radius, preserves_volume, criterion, constraints, diagnostics, status):
    result = _ellipsoid(kind, center, np.full(3, radius), np.eye(3), True,
        radius_mm=float(radius), preserves_volume=preserves_volume,
        criterion=criterion, constraints=constraints, diagnostics=diagnostics,
        status=status, evidence_status='exploratory_hypothesis_not_validated')
    return result


def fit_surface_sphere_fixed_volume(points_mm, area_weights, volume_mm3,
                                    center_bounds_mm, starts_mm, *, max_nfev=200):
    """Г3в: minimize sum(w*(distance-r)^2), r=cbrt(3*V/(4*pi)).

    Bounded local least squares from EACH explicit center start. Bounds are a
    numerical search domain supplied by the caller, not a physiological prior.
    Center coordinates are internally normalized by the fixed radius.
    """
    points, weights = _surface(points_mm, area_weights)
    volume = _positive(volume_mm3, 'volume_mm3')
    bounds, starts = _domain(center_bounds_mm, starts_mm, 3)
    max_nfev = _integer(max_nfev, 'max_nfev')
    radius = float(np.cbrt(3 * volume / (4 * np.pi)))
    origin = weights @ points
    points_n = (points - origin) / radius
    bounds_n = (bounds - origin) / radius
    sw = np.sqrt(weights)

    def residual(center):
        return sw * (np.linalg.norm(points_n - center, axis=1) - 1)

    def jacobian(center):
        delta = center - points_n
        d = np.linalg.norm(delta, axis=1)
        # A zero subgradient at coincident points; report this nonsmooth case.
        return sw[:, None] * delta / np.maximum(d[:, None], np.finfo(float).eps)

    trials = []
    for start in starts:
        fit = least_squares(residual, (start-origin)/radius, jac=jacobian,
            bounds=bounds_n, ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev)
        trials.append({'start_mm': start.tolist(), 'center_mm': (origin+radius*fit.x).tolist(),
            'objective_mm2': float(fit.fun @ fit.fun * radius**2),
            'success': bool(fit.success), 'solver_message': str(fit.message),
            'nfev': int(fit.nfev), 'active_bounds': fit.active_mask.tolist(),
            'coincident_surface_points': int(np.count_nonzero(
                np.linalg.norm(points_n-fit.x, axis=1) <= 1e-12))})
    successful = [t for t in trials if t['success']]
    best = min(successful or trials, key=lambda t: t['objective_mm2'])
    objective = best['objective_mm2']
    return _sphere('sphere_surface_fixed_volume', best['center_mm'], radius, True,
        {'name': 'area_weighted_squared_radial_distance', 'value': objective, 'units': 'mm2'},
        {'fixed_volume_mm3': volume, 'center_bounds_mm': bounds.tolist()},
        {'trials': trials, 'global_optimum_certified': False,
         'surface_RMS_mm': math.sqrt(objective),
         'surface_quadrature_convergence_verified': False,
         'start_solution_span_mm': np.ptp([t['center_mm'] for t in trials], axis=0).tolist()},
        'local_solver_converged' if successful else 'solver_not_converged')


def fit_algebraic_sphere(points_mm, area_weights):
    """Г5: weighted LS for ||x||^2 + a.x + b = 0, quadratic part I.

    Solve a centered, isotropically scaled LINEAR problem by SVD least squares.
    This is not radial-distance fitting. Objective units are mm^4; algebraic
    RMS has units mm^2 and must never be labeled surface distance. Rank-deficient
    data are rejected because center/radius are not identifiable. Full-rank
    success certifies only the linear LS solution, not anatomical validity.
    """
    points, weights = _surface(points_mm, area_weights)
    origin = weights @ points
    delta = points - origin
    scale = math.sqrt(float(weights @ np.einsum('ij,ij->i', delta, delta)))
    _positive(scale, 'point spread')
    q = delta / scale
    design = np.column_stack((2*q, np.ones(len(q))))
    rhs = np.einsum('ij,ij->i', q, q)
    sw = np.sqrt(weights)
    beta, _, rank, singular = np.linalg.lstsq(sw[:, None]*design, sw*rhs, rcond=None)
    if rank != 4:
        raise ValueError('Rank-deficient surface: algebraic sphere center is not identifiable')
    radius_squared = float(beta[3] + beta[:3] @ beta[:3])
    if not np.isfinite(radius_squared) or radius_squared <= 0:
        raise ValueError('Algebraic solution has nonpositive or nonfinite radius squared')
    center = origin + scale*beta[:3]
    radius = scale*math.sqrt(radius_squared)
    residual = (rhs-design@beta)*scale**2
    objective = float(weights @ residual**2)
    return _sphere('sphere_algebraic_unit_quadratic', center, radius, False,
        {'name': 'area_weighted_squared_implicit_residual_unit_quadratic',
         'value': objective, 'units': 'mm4'},
        {'quadratic_matrix': np.eye(3).tolist(), 'center_and_radius': 'free'},
        {'rank': int(rank), 'scaled_design_condition_number': float(singular[0]/singular[-1]),
         'normalization_origin_mm': origin.tolist(), 'normalization_scale_mm': scale,
         'algebraic_RMS_mm2': math.sqrt(objective),
         'surface_RMS_mm': float(np.sqrt(weights @ (np.linalg.norm(points-center, axis=1)-radius)**2)),
         'linear_LS_global_optimum': True, 'surface_quadrature_convergence_verified': False},
        'linear_least_squares_solved')


class _MaskSampler:
    """Validated mask and inverse affine, shared across candidate evaluations."""
    def __init__(self, mask, affine, subdivisions, chunk_size, max_samples):
        self.mask = np.asarray(mask)
        self.affine = _affine(affine)
        self.moments = mask_moments(self.mask, self.affine)
        self.inverse = np.linalg.inv(self.affine)
        self.subdivisions = _integer(subdivisions, 'subdivisions')
        self.chunk_size = _integer(chunk_size, 'chunk_size')
        self.max_samples = _integer(max_samples, 'max_samples_per_evaluation')

    def physical(self, center_mm, radius_mm, spacing_mm, evaluation_bounds_mm=None):
        center = np.asarray(center_mm, dtype=float)
        radius = _positive(radius_mm, 'radius_mm')
        spacing = _positive(spacing_mm, 'spacing_mm')
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Expected finite center (3,)')
        from itertools import product
        bbox = self.moments['bbox_index_centers']
        corners = np.array(list(product(*zip(bbox[0]-.5, bbox[1]+.5))))
        corners = corners @ self.affine[:3, :3].T + self.affine[:3, 3]
        lo = np.minimum(corners.min(0), center-radius)
        hi = np.maximum(corners.max(0), center+radius)
        if evaluation_bounds_mm is not None:
            bounds = np.asarray(evaluation_bounds_mm, dtype=float)
            if (bounds.shape != (2,3) or not np.isfinite(bounds).all()
                    or np.any(bounds[0] > lo) or np.any(bounds[1] < hi)):
                raise ValueError('evaluation_bounds_mm must include full target AND sphere; clipping forbidden')
            lo, hi = bounds
        origin = self.affine[:3, 3]
        low_f = np.floor((lo-origin)/spacing)
        high_f = np.ceil((hi-origin)/spacing)
        if (not np.isfinite([low_f, high_f]).all()
                or np.max(np.abs([low_f, high_f])) >= np.iinfo(np.int64).max/4):
            raise ValueError('Sampling bounds exceed integer capacity')
        lower, upper = low_f.astype(np.int64), high_f.astype(np.int64)
        shape = tuple(int(v) for v in upper-lower+1)
        total = math.prod(shape)
        if total > self.max_samples:
            raise ValueError(f'Full candidate grid needs {total} samples, limit {self.max_samples}; cropping forbidden')
        nt = nc = ni = 0
        for start in range(0, total, self.chunk_size):
            flat = np.arange(start, min(total, start+self.chunk_size), dtype=np.int64)
            lattice = np.column_stack(np.unravel_index(flat, shape))+lower
            relative = lattice*spacing
            index = relative @ self.inverse[:3, :3].T
            target = _lookup(self.mask, np.floor(index+.5).astype(np.int64))
            delta = relative-(center-origin)
            inside = np.einsum('ij,ij->i', delta, delta) <= radius**2*(1+1e-12)
            nt += int(np.count_nonzero(target)); nc += int(np.count_nonzero(inside))
            ni += int(np.count_nonzero(inside & target))
        if not nt:
            raise ValueError('Target missed by physical sampling grid; reduce spacing_mm')
        return {'iou': ni/(nt+nc-ni), 'target_sample_count': nt, 'candidate_sample_count': nc,
            'intersection_sample_count': ni, 'union_sample_count': nt+nc-ni,
            'candidate_sampled_volume_mm3': nc*spacing**3,
            'candidate_continuous_volume_mm3': 4*np.pi/3*radius**3,
            'grid_shape': list(shape), 'grid_sample_count': total,
            'grid_index_bounds_inclusive': np.stack((lower,upper)).tolist(),
            'grid_origin_mm': origin.tolist(), 'spacing_mm': spacing,
            'sampling': 'world_axis_uniform_lattice_anchored_at_affine_origin_full_union_bbox',
            'candidate_clipped_to_image_extent': False, 'exact_continuous_iou': False}


    def evaluate(self, center_mm, radius_mm, subdivisions=None):
        center = np.asarray(center_mm, dtype=float)
        radius = _positive(radius_mm, 'radius_mm')
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Expected finite center_mm of shape (3,)')
        sub = self.subdivisions if subdivisions is None else _integer(subdivisions, 'subdivisions')
        center_index = self.inverse[:3, :3] @ (center-self.affine[:3, 3])
        extent = radius*np.linalg.norm(self.inverse[:3, :3], axis=1)
        target_bounds = self.moments['bbox_index_centers']
        lo = np.minimum(target_bounds[0]-.5, center_index-extent)
        hi = np.maximum(target_bounds[1]+.5, center_index+extent)
        # Include entire continuous candidate bbox, even beyond the image.
        lower_f = np.floor((lo+.5)*sub)
        upper_f = np.ceil((hi+.5)*sub)
        if (not np.isfinite([lower_f, upper_f]).all()
                or np.max(np.abs([lower_f, upper_f])) >= np.iinfo(np.int64).max/4):
            raise ValueError('Sampling bounds exceed integer capacity')
        lower, upper = lower_f.astype(np.int64), upper_f.astype(np.int64)
        shape = tuple(int(v) for v in upper-lower+1)
        total = math.prod(shape)
        if total > self.max_samples:
            raise ValueError(f'Full candidate grid needs {total} samples, limit {self.max_samples}; '
                             'increase the explicit budget or change sampling, never crop the sphere')
        candidate_count = intersection = 0
        for start in range(0, total, self.chunk_size):
            flat = np.arange(start, min(total, start+self.chunk_size), dtype=np.int64)
            lattice = np.column_stack(np.unravel_index(flat, shape)) + lower
            index = (lattice+.5)/sub-.5
            # Integer floor division locates the half-open parent voxel exactly.
            target = _lookup(self.mask, lattice//sub)
            physical_delta = (index-center_index) @ self.affine[:3, :3].T
            inside = np.einsum('ij,ij->i', physical_delta, physical_delta) <= radius**2*(1+1e-12)
            candidate_count += int(np.count_nonzero(inside))
            intersection += int(np.count_nonzero(inside & target))
        target_count = self.moments['voxel_count']*sub**3
        union = target_count+candidate_count-intersection
        sample_volume = self.moments['voxel_volume_mm3']/sub**3
        return {'iou': intersection/union, 'intersection_sample_count': intersection,
            'candidate_sample_count': candidate_count, 'target_sample_count': target_count,
            'union_sample_count': union, 'candidate_sampled_volume_mm3': candidate_count*sample_volume,
            'candidate_continuous_volume_mm3': 4*np.pi/3*radius**3,
            'grid_subindex_bounds_inclusive': np.stack((lower, upper)).tolist(),
            'grid_shape': list(shape), 'grid_sample_count': total, 'subdivisions': sub,
            'sampling_basis_mm': (self.affine[:3, :3]/sub).tolist(),
            'sampling': 'uniform_subvoxel_centers_on_full_extended_target_affine_lattice',
            'candidate_clipped_to_image_extent': False, 'exact_continuous_iou': False}


def sampled_sphere_iou(mask, affine, center_mm, radius_mm, *, subdivisions=1,
                       chunk_size=65536, max_samples_per_evaluation=20_000_000):
    """Sample IoU on the complete union bbox, including sphere outside the CT.

    Target = union of half-open affine voxel cells; background outside the mask
    array is false. Subdivision refines integration, not the anatomy. Sampling
    is approximate and may change with resolution/lattice orientation/phase.
    A resource limit raises rather than returning a silently clipped score.
    """
    return _MaskSampler(mask, affine, subdivisions, chunk_size,
                        max_samples_per_evaluation).evaluate(center_mm, radius_mm)


def _fit_iou(mask, affine, bounds, starts, steps, fixed_volume, *, subdivisions,
             check_subdivisions, max_sweeps_per_step, chunk_size, max_samples_per_evaluation,
             evaluation_spacing_mm, verification_spacing_mm, evaluation_bounds_mm, max_evaluations):
    fixed_radius = None if fixed_volume is None else float(np.cbrt(3*fixed_volume/(4*np.pi)))
    bounds, starts = _domain(bounds, starts, 4 if fixed_radius is None else 3)
    steps = np.asarray(steps, dtype=float)
    if (steps.ndim != 1 or len(steps) == 0 or not np.isfinite(steps).all()
            or np.any(steps <= 0) or np.any(np.diff(steps) >= 0)):
        raise ValueError('search_steps_mm must be an explicit strictly descending positive sequence')
    max_sweeps = _integer(max_sweeps_per_step, 'max_sweeps_per_step')
    sampler = _MaskSampler(mask, affine, subdivisions, chunk_size, max_samples_per_evaluation)
    if check_subdivisions is not None:
        check_subdivisions = _integer(check_subdivisions, 'check_subdivisions')
        if evaluation_spacing_mm is None and verification_spacing_mm is None and check_subdivisions <= subdivisions:
            raise ValueError('check_subdivisions must exceed optimization subdivisions')

    if evaluation_spacing_mm is not None:
        _positive(evaluation_spacing_mm, 'evaluation_spacing_mm')
    if verification_spacing_mm is not None:
        _positive(verification_spacing_mm, 'verification_spacing_mm')
    max_evaluations = _integer(max_evaluations, 'max_evaluations')
    if max_evaluations < len(starts):
        raise ValueError('max_evaluations must allow all initial starts')
    cache = {}

    def score(p, verification=False):
        radius = fixed_radius if fixed_radius is not None else p[3]
        spacing = verification_spacing_mm if verification else evaluation_spacing_mm
        if spacing is not None:
            return sampler.physical(p[:3], radius, spacing, evaluation_bounds_mm)
        if evaluation_bounds_mm is not None:
            # Validate the explicit physical envelope even for native sampling.
            bounds_check = np.asarray(evaluation_bounds_mm, dtype=float)
            from itertools import product
            bb = sampler.moments['bbox_index_centers']
            corners = np.array(list(product(*zip(bb[0]-.5, bb[1]+.5))))
            corners = corners@sampler.affine[:3,:3].T+sampler.affine[:3,3]
            lo = np.minimum(corners.min(0), p[:3]-radius)
            hi = np.maximum(corners.max(0), p[:3]+radius)
            if (bounds_check.shape != (2,3) or not np.isfinite(bounds_check).all()
                    or np.any(bounds_check[0] > lo) or np.any(bounds_check[1] < hi)):
                raise ValueError('evaluation_bounds_mm would clip full target or candidate')
        return sampler.evaluate(p[:3], radius, check_subdivisions if verification else subdivisions)

    for start in starts:
        key = tuple(start)
        if key not in cache:
            cache[key] = score(start)
    trials = []
    for start in starts:
        p = start.copy()
        current = cache[tuple(p)]
        initial_iou = current['iou']
        history = []
        budget_exhausted = False
        for step in steps:
            exhausted = True
            for sweep in range(max_sweeps):
                best_p, best = p, current
                for axis in range(len(p)):
                    for sign in (-1, 1):
                        trial = p.copy()
                        trial[axis] = np.clip(p[axis]+sign*step, bounds[0, axis], bounds[1, axis])
                        if np.array_equal(trial, p):
                            continue
                        key = tuple(trial)
                        if key not in cache:
                            if len(cache) >= max_evaluations:
                                budget_exhausted = True
                                break
                            cache[key] = score(trial)
                        value = cache[key]
                        if value['iou'] > best['iou']:
                            best_p, best = trial, value
                    if budget_exhausted:
                        break
                unchanged = np.array_equal(best_p, p)
                p, current = best_p, best
                if budget_exhausted:
                    break
                if unchanged:
                    exhausted = False
                    break
            history.append({'step_mm': float(step), 'sweeps': sweep+1,
                'sweep_budget_exhausted': exhausted, 'evaluation_budget_exhausted': budget_exhausted,
                'iou': current['iou']})
            if budget_exhausted:
                break
        trials.append({'start_parameters_mm': start.tolist(), 'parameters_mm': p.tolist(),
            'initial_iou': initial_iou, 'iou': current['iou'],
            'step_history': history, 'last_step_stationary':
                len(history) == len(steps) and not history[-1]['sweep_budget_exhausted']})
    winner = int(np.argmax([t['iou'] for t in trials]))
    best = trials[winner]
    p = np.asarray(best['parameters_mm'])
    radius = fixed_radius if fixed_radius is not None else p[3]
    sampling = cache[tuple(p)]
    refined = None
    if check_subdivisions is not None or verification_spacing_mm is not None:
        checked = [score(np.asarray(t['parameters_mm']), True)['iou'] for t in trials]
        refined = {'spacing_mm': verification_spacing_mm,
            'subdivisions': check_subdivisions if verification_spacing_mm is None else None,
            'trial_final_iou': checked, 'evaluation_count': len(trials),
            'selected_iou': checked[winner], 'selected_iou_change': checked[winner]-best['iou'],
            'selected_still_best_among_trial_endpoints': checked[winner] >= max(checked),
            'reoptimized': False, 'continuous_convergence_certified': False}
    return _sphere('sphere_iou_fixed_volume' if fixed_radius is not None else 'sphere_iou_free_volume',
        p[:3], radius, fixed_radius is not None,
        {'name': 'sampled_iou_maximized', 'value': best['iou'], 'units': 'dimensionless'},
        {'parameter_order': ['x_mm', 'y_mm', 'z_mm'] + ([] if fixed_radius is not None else ['radius_mm']),
         'parameter_bounds_mm': bounds.tolist(), 'fixed_volume_mm3': fixed_volume},
        {'trials': trials, 'selected_trial': winner, 'sampling': sampling,
         'optimization_evaluations': len(cache), 'max_evaluations': max_evaluations,
         'evaluation_spacing_mm': evaluation_spacing_mm,
         'evaluation_bounds_mm': None if evaluation_bounds_mm is None else np.asarray(evaluation_bounds_mm).tolist(),
         'max_samples_per_evaluation': max_samples_per_evaluation,
         'refined_sampling_check': refined, 'search_steps_mm': steps.tolist(),
         'start_solution_span_mm': np.ptp([t['parameters_mm'] for t in trials], axis=0).tolist(),
         'start_iou_range': [min(t['iou'] for t in trials), max(t['iou'] for t in trials)],
         'active_bounds': np.any(np.isclose(p, bounds, rtol=0, atol=steps[-1]*1e-9), axis=0).tolist(),
         'global_optimum_certified': False, 'continuous_iou_convergence_verified': False,
         'limitations': ['discontinuous sampled objective; optimum may be nonunique',
             'axis-direction local search with strict improvement; no traversal of equal-score plateaus',
             'start points, their order, search steps, budget and sampling lattice affect the solution',
             'small final step is not an error bound on center or radius',
             'refinement checks endpoint scores only; repeat optimization to assess optimum stability']},
        'discrete_directional_search_stationary' if best['last_step_stationary'] else 'search_budget_exhausted')


def fit_iou_sphere_fixed_volume(mask, affine, volume_mm3, center_bounds_mm, starts_mm,
                                search_steps_mm, *, subdivisions=1, check_subdivisions=2,
                                max_sweeps_per_step=100, chunk_size=65536,
                                max_samples_per_evaluation=20_000_000, evaluation_spacing_mm=2.0,
                                verification_spacing_mm=1.0, evaluation_bounds_mm=None,
                                max_evaluations=256):
    """Г4а: fixed continuous V, explicit (n,3) center starts/bounds in mm.

    Each step is used in all three physical coordinate directions. Defaults
    optimize on a 2-mm physical grid and check endpoints at 1 mm. To check native
    sampling set verification_spacing_mm=None, check_subdivisions=1. No verification
    reoptimization is performed. max_evaluations bounds TOTAL coarse evaluations;
    verification adds at most len(starts) evaluations, each under the sample cap.
    Set check_subdivisions explicitly when subdivisions is changed (or None to
    skip, recorded as unverified). Bounds/starts/steps are never inferred.
    """
    return _fit_iou(mask, affine, center_bounds_mm, starts_mm, search_steps_mm,
        _positive(volume_mm3, 'volume_mm3'), subdivisions=subdivisions,
        check_subdivisions=check_subdivisions, max_sweeps_per_step=max_sweeps_per_step,
        chunk_size=chunk_size, max_samples_per_evaluation=max_samples_per_evaluation,
        evaluation_spacing_mm=evaluation_spacing_mm, verification_spacing_mm=verification_spacing_mm,
        evaluation_bounds_mm=evaluation_bounds_mm, max_evaluations=max_evaluations)


def fit_iou_sphere_free_volume(mask, affine, center_bounds_mm, radius_bounds_mm,
                               starts_mm, search_steps_mm, *, subdivisions=1,
                               check_subdivisions=2, max_sweeps_per_step=100,
                               chunk_size=65536, max_samples_per_evaluation=20_000_000, evaluation_spacing_mm=2.0,
                                verification_spacing_mm=1.0, evaluation_bounds_mm=None,
                                max_evaluations=256):
    """Г4б: free radius, explicit starts of shape (n,4): x,y,z,r in mm.

    center_bounds_mm is (2,3); radius_bounds_mm is (min,max), min>0.
    Steps move coordinates OR radius by the same physical distance, not volume.
    Continuous volume is returned separately from sampled sphere volume.
    """
    cb = np.asarray(center_bounds_mm, dtype=float)
    rb = np.asarray(radius_bounds_mm, dtype=float)
    if cb.shape != (2, 3) or rb.shape != (2,) or not np.isfinite(rb).all() or rb[0] <= 0:
        raise ValueError('Expected center bounds (2,3) and finite positive radius bounds (2,)')
    return _fit_iou(mask, affine, np.column_stack((cb, rb)), starts_mm, search_steps_mm,
        None, subdivisions=subdivisions, check_subdivisions=check_subdivisions,
        max_sweeps_per_step=max_sweeps_per_step, chunk_size=chunk_size,
        max_samples_per_evaluation=max_samples_per_evaluation,
        evaluation_spacing_mm=evaluation_spacing_mm, verification_spacing_mm=verification_spacing_mm,
        evaluation_bounds_mm=evaluation_bounds_mm, max_evaluations=max_evaluations)


def area_systematic_sample(points_mm, area_weights, count=2048):
    """Deterministic area-CDF midpoint sampling (input order matters).

    Returns (points, normalized_weights, diagnostics). Repeated point selections
    carry multiplicity via weights. This approximates full quadrature, not a
    random sample or accuracy certificate; repeat at 4096 then score the full
    surface separately. No claim of spatially uniform point spacing.
    """
    points, weights = _surface(points_mm, area_weights)
    count = _integer(count, 'count')
    cdf = np.cumsum(weights); cdf[-1] = 1.0
    selected = np.searchsorted(cdf, (np.arange(count)+.5)/count)
    unique, multiplicity = np.unique(selected, return_counts=True)
    return points[unique].copy(), multiplicity/count, {
        'method': 'deterministic_area_cdf_midpoints_in_input_order', 'requested_count': count,
        'unique_count': len(unique), 'full_point_count': len(points),
        'quadrature_convergence_verified': False}


def sphere_surface_metrics(points_mm, area_weights, center_mm, radius_mm, *, chunk_size=65536):
    """Full-set area-weighted radial RMS (mm) and algebraic RMS (mm^2)."""
    points, weights = _surface(points_mm, area_weights)
    center = np.asarray(center_mm, dtype=float)
    radius = _positive(radius_mm, 'radius_mm')
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Expected finite center (3,)')
    chunk_size = _integer(chunk_size, 'chunk_size')
    radial = algebraic = 0.0
    for start in range(0, len(points), chunk_size):
        d = np.linalg.norm(points[start:start+chunk_size]-center, axis=1)
        w = weights[start:start+chunk_size]
        radial += float(w @ (d-radius)**2)
        algebraic += float(w @ ((d-radius)*(d+radius))**2)
    return {'surface_RMS_mm': math.sqrt(radial), 'algebraic_RMS_mm2': math.sqrt(algebraic),
            'point_count': len(points), 'surface_quadrature_convergence_verified': False}
