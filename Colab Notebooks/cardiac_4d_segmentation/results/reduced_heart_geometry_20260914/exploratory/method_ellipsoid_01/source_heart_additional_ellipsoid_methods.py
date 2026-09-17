"""Project ellipsoids L3a--L3d (Л3а--Л3г) and L4a--L4b (Л4а--Л4б).

Pure NumPy/SciPy geometry in physical millimetres; no I/O or FEM. These are
project definitions, NOT reproductions of RFBR/MATLAB algorithms. L3 minimizes
area-weighted squared *nearest Euclidean* distances from the target surface to
the ellipsoid (one-sided, not a symmetric surface metric). L4 maximizes sampled
IoU on an explicitly bounded physical lattice. Neither local convergence nor
synthetic tests establish anatomical accuracy or a global optimum.

``initial`` uses the existing center_mm / semi_axes_mm / axes_matrix dictionary.
Axes are columns. Output parameters use distinct ``ellipsoid_project_L...``
kinds: the legacy sampled_dice kind whitelist needs an explicit runner adapter;
do NOT rename these as moment ellipsoids. Numerical domains, all starts and
budgets are returned. Failure never returns an initial moment fit as a result.

Surface API accepts the full quadrature; systematic area-proportional selection
reduces it to max_fit_points. Validation points are caller supplied and NEVER
used in optimization. Independent denser quadrature/grid checks are still
required before interpreting small between-method differences.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from heart_reduced_geometry import _affine, _ellipsoid, mask_moments

METHODS = {
    "Л3а": ("L3a", False, False), "Л3б": ("L3b", True, False),
    "Л3в": ("L3c", False, True), "Л3г": ("L3d", True, True),
    "Л4а": ("L4a", True, True), "Л4б": ("L4b", True, False),
}
METHODS.update({v[0]: v for v in list(METHODS.values())})
LIMITATIONS = [
    "exploratory_hypothesis_not_validated",
    "bounded_multistart_local_search_not_global_optimum_certificate",
    "numerical_bounds_are_not_physiological_priors",
    "surface_or_grid_convergence_not_established_by_optimizer_success",
    "principal_axis_orientation_is_ambiguous_for_repeated_semi_axes",
]


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _geometry(center, radii, rotation):
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    if (center.shape != (3,) or radii.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(center).all() or not np.isfinite(radii).all()
            or not np.isfinite(rotation).all() or np.any(radii <= 0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0)):
        raise ValueError("Expected finite center, positive semi-axes and a proper rotation")
    if radii.min() / radii.max() < 1e-8:
        raise ValueError("Axis ratio below 1e-8 is outside the numerical distance domain")
    return center, radii, rotation


def _points_weights(points, weights):
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or weights.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Expected nonempty finite Nx3 points and strictly positive weights")
    weights = weights / weights.max()
    return points, weights / weights.sum()


def ellipsoid_surface_distance(points, center_mm, semi_axes_mm, axes_matrix=None,
                               *, signed=False, return_closest=False):
    """True closest-point distance, in mm, including interior/medial-axis points.

    In principal coordinates q_i=a_i²*y_i/(a_i²+lambda). The global nearest
    solution has lambda >= -min(a_i²), since I+lambda*diag(1/a_i²) is PSD.
    Solve the monotone secular equation with 80 bracketed arithmetic/geometric bisections in
    s=lambda+min(a_i²). At its singular endpoint, explicitly fill the remaining
    shortest-axis component; this handles the centre and repeated shortest axes.
    No radial or algebraic residual is substituted. Computation is normalized
    by max(a), with a surface-equation check on the returned closest points.
    At nonunique projections one deterministic shortest-axis direction is used.
    """
    center, a, rotation = _geometry(center_mm, semi_axes_mm, axes_matrix)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 physical points")
    if not len(points):
        d = np.empty(0)
        return (d, points.copy()) if return_closest else d
    scale = float(a.max())
    a = a / scale
    y = ((points - center) @ rotation) / scale
    a2 = a*a
    amin2 = float(a2.min())
    shortest = a2 == amin2
    delta = a2 - amin2
    q = np.zeros_like(y)
    other = ~shortest
    if np.any(other):
        q[:, other] = a2[other] * y[:, other] / delta[other]
    endpoint_norm2 = np.sum((q / a)**2, axis=1)
    short_norm = np.hypot.reduce(y[:, shortest], axis=1)
    hard = (short_norm == 0) & (endpoint_norm2 <= 1)
    if np.any(hard):
        j = int(np.flatnonzero(shortest)[0])
        q[hard, j] = math.sqrt(amin2) * np.sqrt(np.maximum(0., 1-endpoint_norm2[hard]))
    regular = ~hard
    if np.any(regular):
        yr = y[regular]
        # The shortest-axis term alone supplies a lower bound. This avoids loss
        # of a very small positive root near a singular endpoint.
        lower = math.sqrt(amin2) * short_norm[regular]
        upper = amin2 + np.linalg.norm(yr * a, axis=1)
        for _ in range(80):
            # Geometric bisection resolves roots arbitrarily close to the
            # singular endpoint; arithmetic bisection first lifts a zero bound.
            middle = np.where(lower > 0, np.sqrt(lower)*np.sqrt(upper),
                              (lower + upper)/2)
            den = delta + middle[:, None]
            secular = np.sum((a * yr / den)**2, axis=1)
            above = secular > 1
            lower = np.where(above, middle, lower)
            upper = np.where(above, upper, middle)
        root = (lower + upper) / 2
        q[regular] = a2 * yr / (delta + root[:, None])
    if not np.allclose(np.sum((q/a)**2, axis=1), 1., atol=2e-8, rtol=0):
        raise FloatingPointError("Closest-point secular solve failed its surface check")
    d = np.linalg.norm(y-q, axis=1) * scale
    if signed:
        d *= np.where(np.sum((y/a)**2, axis=1) < 1, -1., 1.)
    if return_closest:
        return d, center + (q * scale) @ rotation.T
    return d


def area_subsample(points, weights, max_points=1024):
    """Deterministic midpoint strata on cumulative area, aggregated duplicates.

    This is a numerical quadrature approximation depending on input order, not
    random data or anatomy selection. Returned indices make it reproducible.
    Increasing max_points and independent denser evaluation must be checked.
    """
    points, weights = _points_weights(points, weights)
    max_points = _positive_int(max_points, "max_points")
    if len(points) <= max_points:
        index = np.arange(len(points))
        return points.copy(), weights.copy(), index
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.
    selected = np.searchsorted(cumulative, (np.arange(max_points)+.5)/max_points)
    index, counts = np.unique(selected, return_counts=True)
    return points[index], counts/max_points, index


class _ParameterSpace:
    """Bounded dimensionless centre, log semi-axes, relative rotation vector."""
    def __init__(self, initial, free_center, fixed_volume, volume, center_limit_mm,
                 axis_bounds_mm):
        self.center, a, self.frame = _geometry(initial["center_mm"],
            initial["semi_axes_mm"], initial["axes_matrix"])
        self.center = self.center.copy()
        self.frame = self.frame.copy()
        self.free_center, self.fixed_volume = free_center, fixed_volume
        if fixed_volume:
            if volume is None or not np.isfinite(volume) or volume <= 0:
                raise ValueError("Fixed-volume methods require target_volume_mm3 > 0")
            a = a * np.cbrt(volume / (4*np.pi/3*np.prod(a)))
        self.radius = float(np.cbrt(np.prod(a)))
        self.volume = float(4*np.pi/3 * self.radius**3)
        self.center_limit = np.broadcast_to(self.radius if center_limit_mm is None
                                            else center_limit_mm, (3,)).astype(float)
        if not np.isfinite(self.center_limit).all() or np.any(self.center_limit <= 0):
            raise ValueError("center_limit_mm must be positive (in initial-axis coordinates)")
        ab = np.array([a.min()*.5, a.max()*1.75] if axis_bounds_mm is None
                      else axis_bounds_mm, dtype=float)
        if (ab.shape != (2,) or not np.isfinite(ab).all() or not 0 < ab[0] < ab[1]
                or np.any(a < ab[0]) or np.any(a > ab[1])):
            raise ValueError("axis_bounds_mm must be positive bounds containing the scaled initial axes")
        self.axis_bounds = ab
        self.log_bounds = np.log(ab/self.radius)
        self.offset = 3 if free_center else 0
        self.nlog = 2 if fixed_volume else 3
        logs = np.log(a/self.radius)
        self.x0 = np.r_[np.zeros(self.offset), logs[:self.nlog], np.zeros(3)]
        self.lower = np.r_[-self.center_limit/self.radius if free_center else [],
                          np.repeat(self.log_bounds[0], self.nlog), [-np.pi]*3]
        self.upper = np.r_[self.center_limit/self.radius if free_center else [],
                          np.repeat(self.log_bounds[1], self.nlog), [np.pi]*3]
        self.constraints = ()
        if fixed_volume:
            row = np.zeros(len(self.x0))
            row[self.offset:self.offset+2] = -1
            self.constraints = (LinearConstraint(row, *self.log_bounds),)

    def logs(self, x):
        logs = x[self.offset:self.offset+self.nlog]
        return np.r_[logs, -logs.sum()] if self.fixed_volume else logs

    def feasible(self, x, tol=1e-8):
        logs = self.logs(x)
        return (np.all(x >= self.lower-tol) and np.all(x <= self.upper+tol)
                and np.all(logs >= self.log_bounds[0]-tol)
                and np.all(logs <= self.log_bounds[1]+tol))

    def decode(self, x):
        center = self.center.copy()
        if self.free_center:
            center += self.frame @ (x[:3]*self.radius)
        radii = self.radius*np.exp(self.logs(x))
        rotation = self.frame @ Rotation.from_rotvec(x[-3:]).as_matrix()
        return center, radii, rotation

    def starts(self, count):
        rng = np.random.default_rng(20260915)
        starts = [self.x0.copy()]
        for _ in range(count-1):
            for _ in range(100):
                x = self.x0.copy()
                if self.free_center:
                    x[:3] += rng.uniform(-.15, .15, 3)
                x[self.offset:self.offset+self.nlog] += rng.uniform(-.15, .15, self.nlog)
                x[-3:] += rng.uniform(-.25, .25, 3)
                if self.feasible(x):
                    starts.append(x)
                    break
            else:
                starts.append(self.x0.copy())
        return starts

    def metadata(self):
        return {"parameterization": "centre_in_initial_frame_over_radius_log_axes_relative_rotvec",
                "fixed_volume_exact_parameterization": self.fixed_volume,
                "fixed_center": not self.free_center,
                "center_origin_mm": self.center.tolist(), "center_frame": self.frame.tolist(),
                "center_halfwidth_mm": self.center_limit.tolist() if self.free_center else [0.]*3,
                "semi_axis_bounds_mm": self.axis_bounds.tolist(),
                "rotation_vector_component_bounds_rad": [-math.pi, math.pi],
                "fixed_volume_mm3": self.volume if self.fixed_volume else None}


def _parameters(space, x, method):
    center, radii, rotation = space.decode(x)
    order = np.argsort(radii)[::-1]
    radii, rotation = radii[order], rotation[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    degenerate = any(abs(radii[i]-radii[j]) <= 1e-6*radii.max()
                     for i, j in ((0, 1), (0, 2), (1, 2)))
    return _ellipsoid("ellipsoid_project_"+method, center, radii, rotation, degenerate,
        method_id=method, preserves_volume=space.fixed_volume,
        sampling_family="ellipsoid", source="project_definition_not_RFBR_or_MATLAB_identity")


class _BudgetExceeded(Exception):
    pass


def _search(objective, space, *, n_starts, max_nfev, algorithm):
    n_starts = _positive_int(n_starts, "n_starts")
    max_nfev = _positive_int(max_nfev, "max_nfev")
    if max_nfev < n_starts:
        raise ValueError("max_nfev must allow at least one call per start")
    trials, eligible = [], []
    for i, start in enumerate(space.starts(n_starts)):
        budget = max_nfev//n_starts + (i < max_nfev % n_starts)
        count = 0
        best_value = float("inf")
        def counted(x):
            nonlocal count, best_value
            if count >= budget:
                raise _BudgetExceeded
            count += 1
            value = float(objective(x))
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite objective")
            best_value = min(best_value, value)
            return value
        try:
            if algorithm == "SLSQP":
                fit = minimize(counted, start, method=algorithm,
                    bounds=Bounds(space.lower, space.upper), constraints=space.constraints,
                    options={"ftol": 1e-11, "maxiter": budget})
            else:
                fit = minimize(counted, start, method="Powell",
                    bounds=Bounds(space.lower, space.upper),
                    options={"ftol": 1e-7, "xtol": 1e-4, "maxfev": budget})
            success = bool(fit.success and space.feasible(fit.x))
            message = str(fit.message)
            if success:
                eligible.append((float(fit.fun), fit.x.copy(), i))
        except (_BudgetExceeded, FloatingPointError) as exc:
            success = False
            message = "evaluation_budget_exhausted" if isinstance(exc, _BudgetExceeded) else str(exc)
        trials.append({"start": start.tolist(), "success": success, "nfev": count,
                       "budget": budget, "message": message,
                       "best_visited_objective": best_value if np.isfinite(best_value) else None})
    best = min(eligible, key=lambda item: item[0]) if eligible else None
    return best, {"algorithm": algorithm, "success": best is not None,
        "global_optimum_certified": False, "max_nfev_total": max_nfev,
        "nfev_total": sum(t["nfev"] for t in trials), "trials": trials,
        "selected_start": None if best is None else best[2], "domain": space.metadata()}


def _result(method):
    return {"method_id": method, "status": "failed", "parameters": None,
            "optimization": None, "fit_metrics": None, "validation_metrics": None,
            "sampling": {}, "limitations": LIMITATIONS.copy()}


def _surface_metrics(points, weights, parameters):
    p, w = _points_weights(points, weights)
    d = ellipsoid_surface_distance(p, parameters["center_mm"],
        parameters["semi_axes_mm"], parameters["axes_matrix"])
    return {"surface_RMS_mm": float(np.sqrt(w @ (d*d))),
            "surface_mean_distance_mm": float(w @ d),
            "max_sampled_distance_mm": float(d.max()), "point_count": len(p),
            "criterion": "target_to_ellipsoid_true_euclidean_area_weighted_squared_distance"}


def fit_surface_ellipsoid(points, weights, initial, *, method="Л3а",
        target_volume_mm3=None, max_nfev=600, n_starts=3, max_fit_points=1024,
        validation_points=None, validation_weights=None, center_limit_mm=None,
        axis_bounds_mm=None):
    """Fit L3a/b/c/d; max_nfev is TOTAL objective calls, including differences.

    Fixed centre means initial['center_mm']; caller must supply the mask's
    volume centroid. Fixed volume always requires explicit target_volume_mm3.
    Return status='failed' if no start converged within its allocated budget.
    Separate validation points are evaluated in full in bounded-size chunks.
    """
    spec = METHODS.get(method)
    if spec is None or not spec[0].startswith("L3"):
        raise ValueError("Expected Л3а/Л3б/Л3в/Л3г or L3a/L3b/L3c/L3d")
    name, free_center, fixed_volume = spec
    if (validation_points is None) != (validation_weights is None):
        raise ValueError("Provide validation_points and validation_weights together")
    if validation_points is not None:
        validation_points, validation_weights = _points_weights(validation_points, validation_weights)
    full_points, full_weights = _points_weights(points, weights)
    p, w, indices = area_subsample(full_points, full_weights, max_fit_points)
    space = _ParameterSpace(initial, free_center, fixed_volume, target_volume_mm3,
                            center_limit_mm, axis_bounds_mm)
    result = _result(name)
    result["sampling"] = {"input_point_count": len(full_points), "fit_point_count": len(p),
        "fit_indices": indices.tolist(), "fit_weights_normalized": w.tolist(),
        "rule": "cumulative_area_midpoint_strata_with_duplicate_aggregation",
        "max_fit_points": max_fit_points, "validation_used_in_optimization": False}
    def objective(x):
        d = ellipsoid_surface_distance(p, *space.decode(x)) / space.radius
        return w @ (d*d)
    best, result["optimization"] = _search(objective, space, n_starts=n_starts,
                                           max_nfev=max_nfev, algorithm="SLSQP")
    if best is not None:
        parameters = _parameters(space, best[1], name)
        result.update(status="completed", parameters=parameters,
                      fit_metrics=_surface_metrics(p, w, parameters))
        if validation_points is not None:
            if validation_weights is None:
                raise ValueError("validation_weights required with validation_points")
            vp, vw = _points_weights(validation_points, validation_weights)
            sums, largest = 0., 0.
            mean = 0.
            for offset in range(0, len(vp), 4096):
                d = ellipsoid_surface_distance(vp[offset:offset+4096],
                    parameters["center_mm"], parameters["semi_axes_mm"], parameters["axes_matrix"])
                weights_chunk = vw[offset:offset+4096]
                sums += float(weights_chunk @ (d*d))
                mean += float(weights_chunk @ d)
                largest = max(largest, float(d.max()))
            result["validation_metrics"] = {"surface_RMS_mm": math.sqrt(sums),
                "surface_mean_distance_mm": mean, "max_sampled_distance_mm": largest,
                "point_count": len(vp), "source": "caller_supplied_not_used_in_fit",
                "anatomical_independence_claimed": False}
        else:
            result["limitations"].append("independent_dense_surface_evaluation_not_supplied")
    return result


class _GridTooLarge(Exception):
    pass


def _lattice(mask, affine, moments, origin, frame, lower, upper, step, max_points):
    """Uniform mm lattice; mask occupancy is its union of full affine cells."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid_step_mm must be positive")
    max_points = _positive_int(max_points, "max_grid_points")
    corners_index = np.array(list(product(*zip(moments["bbox_index_centers"][0]-.5,
                                               moments["bbox_index_centers"][1]+.5))))
    corners = corners_index @ affine[:3, :3].T + affine[:3, 3]
    corners_local = (corners-origin) @ frame
    lo = np.floor(np.minimum(lower, corners_local.min(0))/step).astype(np.int64)-1
    hi = np.ceil(np.maximum(upper, corners_local.max(0))/step).astype(np.int64)+1
    shape = tuple(int(v) for v in hi-lo+1)
    count = math.prod(shape)
    if count > max_points:
        raise _GridTooLarge(f"whole_candidate_lattice_requires_{count}_points_limit_{max_points}")
    index = np.column_stack(np.unravel_index(np.arange(count), shape))+lo
    points = origin + (index*step) @ frame.T
    inv = np.linalg.inv(affine)
    voxel = np.floor(points @ inv[:3, :3].T + inv[:3, 3]+.5).astype(np.int64)
    inside = np.all((voxel >= 0) & (voxel < np.array(mask.shape)), axis=1)
    target = np.zeros(count, dtype=bool)
    target[inside] = mask[tuple(voxel[inside].T)]
    meta = {"criterion": "physical_uniform_lattice_IoU_not_continuous_IoU",
        "grid_step_mm": float(step), "point_count": count, "origin_mm": origin.tolist(),
        "frame": frame.tolist(), "index_bounds": [lo.tolist(), hi.tolist()],
        "target_occupancy": "nearest_cell_floor_index_plus_half_full_affine",
        "whole_candidate_counted": True, "native_voxel_dice": False}
    return points, target, meta


def _overlap(points, target, center, radii, rotation):
    candidate = np.sum((((points-center) @ rotation)/radii)**2, axis=1) <= 1
    nc, nt = int(candidate.sum()), int(target.sum())
    ni = int(np.count_nonzero(candidate & target))
    union = nt+nc-ni
    return {"IoU_sampled": float(ni/union) if union else 0.,
            "Dice_sampled": float(2*ni/(nt+nc)) if nt+nc else 0.,
            "target_count": nt, "candidate_count": nc, "intersection_count": ni,
            "union_count": union}


def evaluate_overlap_ellipsoid(mask, affine, parameters, *, grid_step_mm=2.,
                              max_grid_points=500000):
    """Independent bounded lattice evaluation over full candidate + target.

    This is not native sampled_dice. Grid origin/frame are fixed at zero/identity
    in physical space, so this check also changes grid phase from the fitting
    lattice. Frame/phase dependence at finite spacing is a numerical error.
    """
    affine = _affine(affine)
    mask = np.asarray(mask)
    moments = mask_moments(mask, affine)
    c, a, r = _geometry(parameters["center_mm"], parameters["semi_axes_mm"],
                         parameters["axes_matrix"])
    extent = np.linalg.norm(r*a, axis=1)
    try:
        p, target, meta = _lattice(mask, affine, moments, np.zeros(3), np.eye(3),
            c-extent, c+extent, grid_step_mm, max_grid_points)
    except _GridTooLarge as exc:
        return {"status": "unsupported", "reason": str(exc)}
    return {"status": "completed", "metrics": _overlap(p, target, c, a, r), "sampling": meta}


def fit_overlap_ellipsoid(mask, affine, initial, *, method="Л4а",
        target_volume_mm3=None, grid_step_mm=2., max_nfev=600, n_starts=3,
        max_grid_points=500000, center_limit_mm=None, axis_bounds_mm=None,
        validation_grid_step_mm=None):
    """L4a/b: multistart Powell on bounded sampled IoU; no smooth surrogate.

    One fixed lattice covers the target AND EVERY candidate within the numerical
    domain, including material outside the source image. Fixed volume is exact
    continuously (two log-axes, third is minus their sum), not a sampled-count
    constraint. Default 2 mm is an exploratory setting, not proven resolution.
    No lattice allocation above max_grid_points is attempted. Powell may report
    local convergence on grid plateaus; validation/refinement remains necessary.
    """
    spec = METHODS.get(method)
    if spec is None or not spec[0].startswith("L4"):
        raise ValueError("Expected Л4а/Л4б or L4a/L4b")
    name, free_center, fixed_volume = spec
    affine = _affine(affine)
    mask = np.asarray(mask)
    moments = mask_moments(mask, affine)
    if target_volume_mm3 is None:
        target_volume_mm3 = moments["volume_mm3"]
    if fixed_volume and not np.isclose(target_volume_mm3, moments["volume_mm3"], rtol=1e-10):
        raise ValueError("L4a target volume must match the full-cell mask volume")
    space = _ParameterSpace(initial, free_center, fixed_volume, target_volume_mm3,
                            center_limit_mm, axis_bounds_mm)
    result = _result(name)
    extent = space.center_limit + space.axis_bounds[1]
    try:
        p, target, result["sampling"] = _lattice(mask, affine, moments, space.center,
            space.frame, -extent, extent, grid_step_mm, max_grid_points)
    except _GridTooLarge as exc:
        result.update(status="unsupported", reason=str(exc))
        return result
    if not np.any(target):
        result.update(status="unsupported", reason="target_not_resolved_on_fit_lattice")
        return result
    result["limitations"].append("discontinuous_IoU_plateaus_can_cause_premature_local_convergence")
    def objective(x):
        # Powell handles box bounds. For exact volume the third log-axis has an
        # additional linear bound; infeasible points have value > any valid 1-IoU.
        logs = space.logs(x)
        violation = np.maximum(space.log_bounds[0]-logs, 0)+np.maximum(logs-space.log_bounds[1], 0)
        if np.any(violation > 0):
            return 1+float(violation @ violation)
        return 1-_overlap(p, target, *space.decode(x))["IoU_sampled"]
    best, result["optimization"] = _search(objective, space, n_starts=n_starts,
                                          max_nfev=max_nfev, algorithm="Powell")
    if best is not None:
        parameters = _parameters(space, best[1], name)
        result.update(status="completed", parameters=parameters,
                      fit_metrics=_overlap(p, target, *space.decode(best[1])))
        if validation_grid_step_mm is not None:
            result["validation_metrics"] = evaluate_overlap_ellipsoid(mask, affine,
                parameters, grid_step_mm=validation_grid_step_mm, max_grid_points=max_grid_points)
        else:
            result["limitations"].append("independent_dense_overlap_evaluation_not_supplied")
    return result
