"""Utilities for explicitly non-validated exploratory calculations.

The functions in this module never promote candidate annotations or scenario
parameters to accepted inputs.  They provide reproducible calculations for
method development while the strict notebook contracts remain blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from two_layer_model import evaluate, geometry_from_size
from ttrkg_analysis import ensemble_waveform, matrix_diagnostics


EXPLORATORY_STATUS = "exploratory_hypothesis_not_validated"


@dataclass(frozen=True)
class SelectedAnnotation:
    values: object
    source_field: str
    qc_status: str | None
    accepted: bool


def select_modes(annotation: dict) -> SelectedAnnotation:
    """Select accepted modes when available, otherwise candidate modes."""
    qc_status = annotation.get("qc", {}).get("status")
    accepted = annotation.get("accepted_modes")
    if qc_status == "accepted" and accepted:
        values = accepted
        source = "accepted_modes"
        is_accepted = True
    else:
        values = annotation.get("candidate_modes")
        source = "candidate_modes"
        is_accepted = False
    if not values:
        raise ValueError("No accepted or candidate breathing modes are available")
    return SelectedAnnotation(values, source, qc_status, is_accepted)


def normalized_modes(annotation: dict) -> tuple[dict[str, list[float]], SelectedAnnotation]:
    """Return a mode-to-interval mapping for both exp02 and exp03 schemas."""
    selected = select_modes(annotation)
    raw = selected.values
    if isinstance(raw, dict):
        result = {str(name): [float(bounds[0]), float(bounds[1])] for name, bounds in raw.items()}
    elif isinstance(raw, list):
        result = {
            str(item["mode"]): [float(item["start_s"]), float(item["stop_s"])]
            for item in raw
        }
    else:
        raise ValueError("Unsupported breathing annotation format")
    if any(stop <= start for start, stop in result.values()):
        raise ValueError("Breathing intervals must have positive duration")
    return result, selected


def select_rpeaks(annotation: dict, *, prefer_diagnostic: bool = False) -> SelectedAnnotation:
    """Select accepted R peaks or an explicitly candidate-only list."""
    qc_status = annotation.get("qc", {}).get("status")
    accepted = annotation.get("rpeaks_s")
    if qc_status == "accepted" and accepted:
        return SelectedAnnotation(
            [float(value) for value in accepted],
            "rpeaks_s",
            qc_status,
            True,
        )
    diagnostic = annotation.get("diagnostic_lower_threshold", {}).get("candidate_rpeaks_s")
    if prefer_diagnostic and diagnostic:
        return SelectedAnnotation(
            [float(value) for value in diagnostic],
            "diagnostic_lower_threshold.candidate_rpeaks_s",
            qc_status,
            False,
        )
    candidates = annotation.get("candidate_rpeaks_s")
    if not candidates:
        raise ValueError("No accepted or candidate R peaks are available")
    return SelectedAnnotation(
        [float(value) for value in candidates],
        "candidate_rpeaks_s",
        qc_status,
        False,
    )


def interval_median(time_s, values, interval_s, *, margin_s: float = 0.5) -> float:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_s.shape != values.shape or len(time_s) < 2:
        raise ValueError("Time and values must be one-dimensional and aligned")
    left, right = map(float, interval_s)
    left += float(margin_s)
    right -= float(margin_s)
    if right <= left:
        raise ValueError("Interval is too short after applying the margin")
    mask = (time_s >= left) & (time_s <= right)
    if mask.sum() < 3:
        raise ValueError("Interval contains too few samples")
    selected = values[mask]
    if not np.isfinite(selected).all():
        raise ValueError("Interval contains non-finite samples")
    return float(np.median(selected))


def _static_prediction(theta_log, sizes_m, h_m):
    rho1, rho2_inhale, rho2_exhale = np.exp(np.asarray(theta_log, dtype=float))
    predicted = []
    jacobian_log = []
    for size_m in np.asarray(sizes_m, dtype=float):
        a, b = geometry_from_size(float(size_m))
        inhale = evaluate(rho1, rho2_inhale, h_m, a, b)
        exhale = evaluate(rho1, rho2_exhale, h_m, a, b)
        predicted.extend([inhale.z, exhale.z])
        jacobian_log.extend([
            [inhale.d_rho1 * rho1, inhale.d_rho2 * rho2_inhale, 0.0],
            [exhale.d_rho1 * rho1, 0.0, exhale.d_rho2 * rho2_exhale],
        ])
    return np.asarray(predicted), np.asarray(jacobian_log)


def fit_static_at_h(sizes_m, z_inhale_ohm, z_exhale_ohm, h_m):
    """Fit three resistivities for a fixed scenario value of ``h``."""
    sizes_m = np.asarray(sizes_m, dtype=float)
    z_inhale_ohm = np.asarray(z_inhale_ohm, dtype=float)
    z_exhale_ohm = np.asarray(z_exhale_ohm, dtype=float)
    if sizes_m.ndim != 1 or len(sizes_m) < 3:
        raise ValueError("At least three assembly sizes are required")
    if z_inhale_ohm.shape != sizes_m.shape or z_exhale_ohm.shape != sizes_m.shape:
        raise ValueError("Static observations must match assembly sizes")
    observed = np.column_stack([z_inhale_ohm, z_exhale_ohm]).ravel()
    if not np.isfinite(observed).all() or np.any(observed <= 0):
        raise ValueError("This exploratory observation model requires positive finite impedance")

    def residual(theta_log):
        predicted, _ = _static_prediction(theta_log, sizes_m, float(h_m))
        return predicted - observed

    seeds = [
        np.log([rho1, rho2_inhale, rho2_exhale])
        for rho1 in (0.5, 2.0, 10.0, 50.0)
        for rho2_inhale, rho2_exhale in ((0.5, 0.7), (5.0, 3.0), (30.0, 20.0), (100.0, 60.0))
    ]
    lower = np.log([0.01, 0.01, 0.01])
    upper = np.log([1000.0, 1000.0, 1000.0])
    candidates = [
        least_squares(residual, seed, bounds=(lower, upper), method="trf")
        for seed in seeds
    ]
    solution = min(candidates, key=lambda item: float(np.sum(item.fun**2)))
    predicted, jacobian_log = _static_prediction(solution.x, sizes_m, float(h_m))
    diagnostics = matrix_diagnostics(jacobian_log)
    rho1, rho2_inhale, rho2_exhale = np.exp(solution.x)
    return {
        "h_m": float(h_m),
        "rho1_ohm_m": float(rho1),
        "rho2_inhale_ohm_m": float(rho2_inhale),
        "rho2_exhale_ohm_m": float(rho2_exhale),
        "residual_sum_squares_ohm2": float(np.sum((predicted - observed) ** 2)),
        "residual_rms_ohm": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "predicted_inhale_ohm": predicted[0::2].tolist(),
        "predicted_exhale_ohm": predicted[1::2].tolist(),
        "jacobian_log_rank": int(diagnostics["rank"]),
        "jacobian_log_condition": float(diagnostics["condition"]),
        "parameter_bound_hit": bool(
            np.any(np.isclose(solution.x, lower, atol=1e-5))
            or np.any(np.isclose(solution.x, upper, atol=1e-5))
        ),
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
    }


def profile_static_h(sizes_m, z_inhale_ohm, z_exhale_ohm, h_grid_m):
    """Profile the static fit over an explicitly supplied computational h grid."""
    rows = [
        fit_static_at_h(sizes_m, z_inhale_ohm, z_exhale_ohm, float(h_m))
        for h_m in np.asarray(h_grid_m, dtype=float)
    ]
    best_index = int(np.argmin([row["residual_sum_squares_ohm2"] for row in rows]))
    return {"profile": rows, "best_index": best_index, "best": rows[best_index]}



def _static_prediction_four(theta_log, sizes_m):
    """Predict both respiratory states for log(h, rho1, rho2-in, rho2-ex)."""
    h_m, rho1, rho2_inhale, rho2_exhale = np.exp(np.asarray(theta_log, dtype=float))
    predicted = []
    jacobian_log = []
    for size_m in np.asarray(sizes_m, dtype=float):
        a, b = geometry_from_size(float(size_m))
        inhale = evaluate(rho1, rho2_inhale, h_m, a, b)
        exhale = evaluate(rho1, rho2_exhale, h_m, a, b)
        predicted.extend([inhale.z, exhale.z])
        jacobian_log.extend([
            [inhale.d_h * h_m, inhale.d_rho1 * rho1, inhale.d_rho2 * rho2_inhale, 0.0],
            [exhale.d_h * h_m, exhale.d_rho1 * rho1, 0.0, exhale.d_rho2 * rho2_exhale],
        ])
    return np.asarray(predicted), np.asarray(jacobian_log)


def _bounded_seed_vectors(lower_log, upper_log):
    midpoint = 0.5 * (lower_log + upper_log)
    span = upper_log - lower_log
    seeds = [midpoint]
    for fractions in product((0.2, 0.8), repeat=len(lower_log)):
        seeds.append(lower_log + np.asarray(fractions, dtype=float) * span)
    return seeds


def _validate_static_observations(sizes_m, z_inhale_ohm, z_exhale_ohm):
    sizes = np.asarray(sizes_m, dtype=float)
    inhale = np.asarray(z_inhale_ohm, dtype=float)
    exhale = np.asarray(z_exhale_ohm, dtype=float)
    if sizes.ndim != 1 or len(sizes) < 3:
        raise ValueError("At least three assembly sizes are required")
    if inhale.shape != sizes.shape or exhale.shape != sizes.shape:
        raise ValueError("Static observations must match assembly sizes")
    values = np.concatenate([sizes, inhale, exhale])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Sizes and static observations must be positive and finite")
    return sizes, inhale, exhale, np.column_stack([inhale, exhale]).ravel()


def _validated_positive_bounds(*bounds):
    pairs = np.asarray(bounds, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Each parameter bound must contain two values")
    if not np.isfinite(pairs).all() or np.any(pairs <= 0) or np.any(pairs[:, 1] <= pairs[:, 0]):
        raise ValueError("Parameter bounds must be positive, finite and strictly increasing")
    return pairs


def fit_static_bounded(
    sizes_m,
    z_inhale_ohm,
    z_exhale_ohm,
    *,
    h_bounds_m,
    rho1_bounds_ohm_m,
    rho2_bounds_ohm_m,
):
    """Fit common h, rho1, rho2-in and rho2-ex inside explicit bounds."""
    sizes, _, _, observed = _validate_static_observations(
        sizes_m, z_inhale_ohm, z_exhale_ohm
    )
    bounds = _validated_positive_bounds(
        h_bounds_m, rho1_bounds_ohm_m, rho2_bounds_ohm_m, rho2_bounds_ohm_m
    )
    lower_log = np.log(bounds[:, 0])
    upper_log = np.log(bounds[:, 1])

    def residual(theta_log):
        try:
            predicted, _ = _static_prediction_four(theta_log, sizes)
        except RuntimeError:
            return np.full_like(observed, 1.0e6)
        return predicted - observed

    candidates = [
        least_squares(
            residual,
            seed,
            bounds=(lower_log, upper_log),
            method="trf",
            x_scale="jac",
            max_nfev=3000,
        )
        for seed in _bounded_seed_vectors(lower_log, upper_log)
    ]
    finite = [candidate for candidate in candidates if np.isfinite(candidate.fun).all()]
    if not finite:
        raise RuntimeError("No finite bounded static solution was found")
    finite.sort(key=lambda item: float(np.sum(item.fun**2)))
    solution = finite[0]
    predicted, jacobian_log = _static_prediction_four(solution.x, sizes)
    diagnostics = matrix_diagnostics(jacobian_log)
    h_m, rho1, rho2_inhale, rho2_exhale = np.exp(solution.x)

    unique = []
    span = upper_log - lower_log
    for candidate in finite:
        normalized = (candidate.x - lower_log) / span
        if any(np.linalg.norm(normalized - row["normalized"]) < 1.0e-4 for row in unique):
            continue
        values = np.exp(candidate.x)
        unique.append({
            "normalized": normalized,
            "residual_sum_squares_ohm2": float(np.sum(candidate.fun**2)),
            "h_m": float(values[0]),
            "rho1_ohm_m": float(values[1]),
            "rho2_inhale_ohm_m": float(values[2]),
            "rho2_exhale_ohm_m": float(values[3]),
            "optimizer_success": bool(candidate.success),
        })
    for row in unique:
        row.pop("normalized")

    bound_hit = np.isclose(solution.x, lower_log, atol=1.0e-5) | np.isclose(
        solution.x, upper_log, atol=1.0e-5
    )
    residual_values = predicted - observed
    return {
        "h_m": float(h_m),
        "rho1_ohm_m": float(rho1),
        "rho2_inhale_ohm_m": float(rho2_inhale),
        "rho2_exhale_ohm_m": float(rho2_exhale),
        "rho2_inhale_not_less_than_exhale": bool(rho2_inhale >= rho2_exhale),
        "residual_sum_squares_ohm2": float(np.sum(residual_values**2)),
        "residual_rms_ohm": float(np.sqrt(np.mean(residual_values**2))),
        "relative_residual_rms": float(np.sqrt(np.mean((residual_values / observed) ** 2))),
        "predicted_inhale_ohm": predicted[0::2].tolist(),
        "predicted_exhale_ohm": predicted[1::2].tolist(),
        "jacobian_log_rank": int(diagnostics["rank"]),
        "jacobian_log_condition": float(diagnostics["condition"]),
        "parameter_bound_hits": {
            "h": bool(bound_hit[0]),
            "rho1": bool(bound_hit[1]),
            "rho2_inhale": bool(bound_hit[2]),
            "rho2_exhale": bool(bound_hit[3]),
        },
        "optimizer_success": bool(solution.success),
        "optimizer_message": str(solution.message),
        "multi_start_count": len(candidates),
        "unique_candidate_solutions": unique,
        "bounds": {
            "h_m": list(map(float, h_bounds_m)),
            "rho1_ohm_m": list(map(float, rho1_bounds_ohm_m)),
            "rho2_ohm_m": list(map(float, rho2_bounds_ohm_m)),
        },
    }


def fit_static_at_h_bounded(
    sizes_m,
    z_inhale_ohm,
    z_exhale_ohm,
    h_m,
    *,
    rho1_bounds_ohm_m,
    rho2_bounds_ohm_m,
):
    """Fit three common resistivities at fixed h inside explicit bounds."""
    sizes, _, _, observed = _validate_static_observations(
        sizes_m, z_inhale_ohm, z_exhale_ohm
    )
    h_m = float(h_m)
    if not np.isfinite(h_m) or h_m <= 0:
        raise ValueError("h must be positive and finite")
    bounds = _validated_positive_bounds(
        rho1_bounds_ohm_m, rho2_bounds_ohm_m, rho2_bounds_ohm_m
    )
    lower_log = np.log(bounds[:, 0])
    upper_log = np.log(bounds[:, 1])

    def residual(theta_log):
        try:
            predicted, _ = _static_prediction(theta_log, sizes, h_m)
        except RuntimeError:
            return np.full_like(observed, 1.0e6)
        return predicted - observed

    candidates = [
        least_squares(
            residual,
            seed,
            bounds=(lower_log, upper_log),
            method="trf",
            x_scale="jac",
            max_nfev=3000,
        )
        for seed in _bounded_seed_vectors(lower_log, upper_log)
    ]
    finite = [candidate for candidate in candidates if np.isfinite(candidate.fun).all()]
    if not finite:
        raise RuntimeError("No finite bounded static solution was found at fixed h")
    solution = min(finite, key=lambda item: float(np.sum(item.fun**2)))
    predicted, jacobian_log = _static_prediction(solution.x, sizes, h_m)
    rho1, rho2_inhale, rho2_exhale = np.exp(solution.x)
    bound_hit = np.isclose(solution.x, lower_log, atol=1.0e-5) | np.isclose(
        solution.x, upper_log, atol=1.0e-5
    )
    residual_values = predicted - observed
    return {
        "h_m": h_m,
        "rho1_ohm_m": float(rho1),
        "rho2_inhale_ohm_m": float(rho2_inhale),
        "rho2_exhale_ohm_m": float(rho2_exhale),
        "rho2_inhale_not_less_than_exhale": bool(rho2_inhale >= rho2_exhale),
        "residual_sum_squares_ohm2": float(np.sum(residual_values**2)),
        "residual_rms_ohm": float(np.sqrt(np.mean(residual_values**2))),
        "relative_residual_rms": float(np.sqrt(np.mean((residual_values / observed) ** 2))),
        "predicted_inhale_ohm": predicted[0::2].tolist(),
        "predicted_exhale_ohm": predicted[1::2].tolist(),
        "jacobian_log_condition_fixed_h": float(matrix_diagnostics(jacobian_log)["condition"]),
        "parameter_bound_hits": {
            "rho1": bool(bound_hit[0]),
            "rho2_inhale": bool(bound_hit[1]),
            "rho2_exhale": bool(bound_hit[2]),
        },
        "optimizer_success": bool(solution.success),
    }


def profile_static_h_bounded(
    sizes_m,
    z_inhale_ohm,
    z_exhale_ohm,
    h_grid_m,
    *,
    rho1_bounds_ohm_m,
    rho2_bounds_ohm_m,
):
    """Profile a bounded three-resistivity fit over an explicit h grid."""
    h_grid = np.asarray(h_grid_m, dtype=float)
    if h_grid.ndim != 1 or len(h_grid) < 2 or not np.all(np.diff(h_grid) > 0):
        raise ValueError("h grid must be one-dimensional and strictly increasing")
    rows = [
        fit_static_at_h_bounded(
            sizes_m,
            z_inhale_ohm,
            z_exhale_ohm,
            float(h_m),
            rho1_bounds_ohm_m=rho1_bounds_ohm_m,
            rho2_bounds_ohm_m=rho2_bounds_ohm_m,
        )
        for h_m in h_grid
    ]
    best_index = int(np.argmin([row["residual_sum_squares_ohm2"] for row in rows]))
    return {
        "profile": rows,
        "best_index": best_index,
        "best": rows[best_index],
        "best_h_grid_boundary": bool(best_index in {0, len(rows) - 1}),
    }



def invert_rho2_at_observed_z(
    size_m,
    observed_z_ohm,
    h_m,
    rho1_ohm_m,
    *,
    rho2_bounds_ohm_m=(0.01, 1000.0),
    max_abs_contrast=0.99,
    initial_rho2_ohm_m=None,
    z_rtol=1e-9,
    z_atol_ohm=1e-8,
):
    """Invert one observed impedance to rho2 on the positive monotone branch."""
    size_m = float(size_m)
    observed_z_ohm = float(observed_z_ohm)
    h_m = float(h_m)
    rho1 = float(rho1_ohm_m)
    lower, upper = map(float, rho2_bounds_ohm_m)
    values = (size_m, observed_z_ohm, h_m, rho1, lower, upper, max_abs_contrast)
    if not all(np.isfinite(values)) or min(size_m, observed_z_ohm, h_m, rho1, lower) <= 0:
        raise ValueError("Size, impedance, h, rho1 and rho2 bounds must be positive and finite")
    if upper <= lower:
        raise ValueError("rho2 bounds must be strictly increasing")
    if not 0 < max_abs_contrast < 1:
        raise ValueError("max_abs_contrast must lie strictly between zero and one")

    contrast_ratio = (1.0 + max_abs_contrast) / (1.0 - max_abs_contrast)
    effective_lower = max(lower, rho1 / contrast_ratio)
    effective_upper = min(upper, rho1 * contrast_ratio)
    if effective_lower >= effective_upper:
        return {
            "feasible": False,
            "reason": "empty_rho2_interval_after_contrast_limit",
            "rho2_ohm_m": None,
        }

    a, b = geometry_from_size(size_m)
    log_lower, log_upper = np.log([effective_lower, effective_upper])
    tolerance = z_atol_ohm + z_rtol * abs(observed_z_ohm)

    def at_log_rho2(log_rho2):
        rho2 = float(np.exp(log_rho2))
        result = evaluate(rho1, rho2, h_m, a, b)
        return result, float(result.z - observed_z_ohm)

    if initial_rho2_ohm_m is not None and np.isfinite(initial_rho2_ohm_m):
        log_rho2 = float(np.clip(np.log(initial_rho2_ohm_m), log_lower, log_upper))
        for _ in range(8):
            try:
                result, residual = at_log_rho2(log_rho2)
            except RuntimeError:
                break
            if abs(residual) <= tolerance:
                return {
                    "feasible": True,
                    "reason": None,
                    "rho2_ohm_m": float(np.exp(log_rho2)),
                    "predicted_z_ohm": float(result.z),
                    "z_residual_ohm": residual,
                    "d_z_d_rho2": float(result.d_rho2),
                    "solver": "newton",
                    "effective_rho2_bounds_ohm_m": [effective_lower, effective_upper],
                }
            derivative_log = float(result.d_rho2 * np.exp(log_rho2))
            if not np.isfinite(derivative_log) or derivative_log <= 0:
                break
            next_log = log_rho2 - residual / derivative_log
            if not np.isfinite(next_log) or not log_lower < next_log < log_upper:
                break
            log_rho2 = float(next_log)

    try:
        lower_result, lower_residual = at_log_rho2(log_lower)
        upper_result, upper_residual = at_log_rho2(log_upper)
    except RuntimeError:
        return {"feasible": False, "reason": "forward_series_nonconvergence", "rho2_ohm_m": None}

    if abs(lower_residual) <= tolerance:
        root_log = log_lower
        root_result = lower_result
    elif abs(upper_residual) <= tolerance:
        root_log = log_upper
        root_result = upper_result
    elif np.signbit(lower_residual) == np.signbit(upper_residual):
        return {
            "feasible": False,
            "reason": "observed_z_outside_invertible_range",
            "rho2_ohm_m": None,
            "invertible_z_range_ohm": sorted([float(lower_result.z), float(upper_result.z)]),
            "effective_rho2_bounds_ohm_m": [effective_lower, effective_upper],
        }
    else:
        root_log = brentq(
            lambda value: at_log_rho2(value)[1],
            log_lower,
            log_upper,
            xtol=1e-12,
            rtol=1e-12,
        )
        root_result, _ = at_log_rho2(root_log)

    rho2 = float(np.exp(root_log))
    residual = float(root_result.z - observed_z_ohm)
    return {
        "feasible": bool(abs(residual) <= tolerance),
        "reason": None if abs(residual) <= tolerance else "root_tolerance_not_reached",
        "rho2_ohm_m": rho2,
        "predicted_z_ohm": float(root_result.z),
        "z_residual_ohm": residual,
        "d_z_d_rho2": float(root_result.d_rho2),
        "solver": "brentq",
        "effective_rho2_bounds_ohm_m": [effective_lower, effective_upper],
    }


def rho2_consistency_at_h_rho1(
    sizes_m,
    z_inhale_ohm,
    z_exhale_ohm,
    h_m,
    rho1_ohm_m,
    *,
    rho2_bounds_ohm_m=(0.01, 1000.0),
    max_abs_contrast=0.99,
    initial_rho2=None,
):
    """Exactly invert each Z and measure between-assembly rho2 consistency."""
    sizes = np.asarray(sizes_m, dtype=float)
    inhale = np.asarray(z_inhale_ohm, dtype=float)
    exhale = np.asarray(z_exhale_ohm, dtype=float)
    if sizes.ndim != 1 or len(sizes) < 3:
        raise ValueError("At least three assembly sizes are required")
    if inhale.shape != sizes.shape or exhale.shape != sizes.shape:
        raise ValueError("Static observations must match assembly sizes")
    if not np.isfinite(np.concatenate([sizes, inhale, exhale])).all():
        raise ValueError("Sizes and observations must be finite")
    if np.any(sizes <= 0) or np.any(inhale <= 0) or np.any(exhale <= 0):
        raise ValueError("Sizes and observations must be positive")

    initial_rho2 = initial_rho2 or {}
    recovered = {}
    residuals = []
    for state, observed in (("inhale", inhale), ("exhale", exhale)):
        state_initial = initial_rho2.get(state)
        if state_initial is not None and len(state_initial) != len(sizes):
            raise ValueError("Initial rho2 values must match assembly sizes")
        state_values = []
        state_details = []
        for index, (size_m, z_ohm) in enumerate(zip(sizes, observed)):
            initial = None if state_initial is None else state_initial[index]
            inverse = invert_rho2_at_observed_z(
                size_m,
                z_ohm,
                h_m,
                rho1_ohm_m,
                rho2_bounds_ohm_m=rho2_bounds_ohm_m,
                max_abs_contrast=max_abs_contrast,
                initial_rho2_ohm_m=initial,
            )
            state_details.append(inverse)
            if not inverse["feasible"]:
                return {
                    "h_m": float(h_m),
                    "rho1_ohm_m": float(rho1_ohm_m),
                    "feasible": False,
                    "reason": inverse["reason"],
                    "failed_state": state,
                    "failed_index": int(index),
                }
            state_values.append(inverse["rho2_ohm_m"])
            residuals.append(inverse["z_residual_ohm"])
        recovered[state] = np.asarray(state_values, dtype=float)

    log_deviations = []
    state_summary = {}
    for state, values in recovered.items():
        log_values = np.log(values)
        log_center = float(log_values.mean())
        log_deviations.extend(log_values - log_center)
        state_summary[state] = {
            "rho2_geometric_mean_ohm_m": float(np.exp(log_center)),
            "rho2_arithmetic_mean_ohm_m": float(values.mean()),
            "rho2_sd_ohm_m": float(values.std(ddof=1)),
            "rho2_min_ohm_m": float(values.min()),
            "rho2_max_ohm_m": float(values.max()),
            "rho2_max_to_min_ratio": float(values.max() / values.min()),
        }
    log_deviations = np.asarray(log_deviations, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    spread = float(np.sqrt(np.mean(log_deviations**2)))
    return {
        "h_m": float(h_m),
        "rho1_ohm_m": float(rho1_ohm_m),
        "feasible": True,
        "reason": None,
        "rho2_inhale_ohm_m": recovered["inhale"].tolist(),
        "rho2_exhale_ohm_m": recovered["exhale"].tolist(),
        "rho2_log_spread_rms": spread,
        "rho2_multiplicative_spread": float(np.exp(spread)),
        "z_reconstruction_rms_ohm": float(np.sqrt(np.mean(residuals**2))),
        "inhale": state_summary["inhale"],
        "exhale": state_summary["exhale"],
    }


def profile_rho2_consistency(
    sizes_m,
    z_inhale_ohm,
    z_exhale_ohm,
    h_grid_m,
    rho1_grid_ohm_m,
    *,
    rho2_bounds_ohm_m=(0.01, 1000.0),
    max_abs_contrast=0.99,
    refine_rho1=True,
):
    """Profile exact-Z inversions over a two-dimensional (h, rho1) grid."""
    h_grid = np.asarray(h_grid_m, dtype=float)
    rho1_grid = np.asarray(rho1_grid_ohm_m, dtype=float)
    if h_grid.ndim != 1 or rho1_grid.ndim != 1 or len(h_grid) < 2 or len(rho1_grid) < 2:
        raise ValueError("h and rho1 grids must be one-dimensional and contain at least two points")
    if not np.all(np.diff(h_grid) > 0) or not np.all(np.diff(rho1_grid) > 0):
        raise ValueError("h and rho1 grids must be strictly increasing")
    if np.any(h_grid <= 0) or np.any(rho1_grid <= 0):
        raise ValueError("h and rho1 grids must be positive")

    details = [[None for _ in rho1_grid] for _ in h_grid]
    metric = np.full((len(h_grid), len(rho1_grid)), np.inf, dtype=float)
    surface = []
    for h_index, h_m in enumerate(h_grid):
        initial = None
        for rho1_index, rho1 in enumerate(rho1_grid):
            result = rho2_consistency_at_h_rho1(
                sizes_m,
                z_inhale_ohm,
                z_exhale_ohm,
                float(h_m),
                float(rho1),
                rho2_bounds_ohm_m=rho2_bounds_ohm_m,
                max_abs_contrast=max_abs_contrast,
                initial_rho2=initial,
            )
            details[h_index][rho1_index] = result
            row = {
                "h_m": float(h_m),
                "rho1_ohm_m": float(rho1),
                "feasible": bool(result["feasible"]),
                "reason": result.get("reason"),
            }
            if result["feasible"]:
                initial = {
                    "inhale": result["rho2_inhale_ohm_m"],
                    "exhale": result["rho2_exhale_ohm_m"],
                }
                metric[h_index, rho1_index] = result["rho2_log_spread_rms"]
                row.update({
                    "rho2_log_spread_rms": result["rho2_log_spread_rms"],
                    "rho2_multiplicative_spread": result["rho2_multiplicative_spread"],
                    "z_reconstruction_rms_ohm": result["z_reconstruction_rms_ohm"],
                })
            surface.append(row)

    profile = []
    candidates_by_h = []
    for h_index, h_m in enumerate(h_grid):
        finite = np.flatnonzero(np.isfinite(metric[h_index]))
        if not len(finite):
            profile.append({"h_m": float(h_m), "feasible": False, "reason": "no_feasible_rho1"})
            candidates_by_h.append({"h_m": float(h_m), "candidates": []})
            continue

        candidate_indices = []
        for rho1_index in finite:
            left_value = metric[h_index, rho1_index - 1] if rho1_index > 0 else np.inf
            right_value = (
                metric[h_index, rho1_index + 1]
                if rho1_index + 1 < len(rho1_grid)
                else np.inf
            )
            value = metric[h_index, rho1_index]
            if value <= left_value and value <= right_value:
                candidate_indices.append(int(rho1_index))
        if not candidate_indices:
            candidate_indices = [int(finite[np.argmin(metric[h_index, finite])])]

        candidates = []
        for rho1_index in candidate_indices:
            candidate = dict(details[h_index][rho1_index])
            candidate["rho1_refined"] = False
            candidate["rho1_grid_seed_ohm_m"] = float(rho1_grid[rho1_index])
            can_refine = (
                refine_rho1
                and rho1_index > 0
                and rho1_index + 1 < len(rho1_grid)
                and np.isfinite(metric[h_index, rho1_index - 1])
                and np.isfinite(metric[h_index, rho1_index + 1])
            )
            if can_refine:
                def objective(log_rho1):
                    trial = rho2_consistency_at_h_rho1(
                        sizes_m,
                        z_inhale_ohm,
                        z_exhale_ohm,
                        float(h_m),
                        float(np.exp(log_rho1)),
                        rho2_bounds_ohm_m=rho2_bounds_ohm_m,
                        max_abs_contrast=max_abs_contrast,
                    )
                    return trial["rho2_log_spread_rms"] if trial["feasible"] else 1.0e6

                refined = minimize_scalar(
                    objective,
                    bounds=(
                        float(np.log(rho1_grid[rho1_index - 1])),
                        float(np.log(rho1_grid[rho1_index + 1])),
                    ),
                    method="bounded",
                    options={"xatol": 1e-8},
                )
                refined_result = rho2_consistency_at_h_rho1(
                    sizes_m,
                    z_inhale_ohm,
                    z_exhale_ohm,
                    float(h_m),
                    float(np.exp(refined.x)),
                    rho2_bounds_ohm_m=rho2_bounds_ohm_m,
                    max_abs_contrast=max_abs_contrast,
                )
                if refined.success and refined_result["feasible"]:
                    candidate = dict(refined_result)
                    candidate["rho1_refined"] = True
                    candidate["rho1_grid_seed_ohm_m"] = float(rho1_grid[rho1_index])
            candidate["rho1_grid_boundary"] = bool(
                rho1_index in {0, len(rho1_grid) - 1}
            )
            candidates.append(candidate)
        candidates.sort(key=lambda row: row["rho2_log_spread_rms"])
        profile.append(candidates[0])
        candidates_by_h.append({"h_m": float(h_m), "candidates": candidates})

    feasible_profile_indices = [index for index, row in enumerate(profile) if row["feasible"]]
    if not feasible_profile_indices:
        return {
            "h_grid_m": h_grid,
            "rho1_grid_ohm_m": rho1_grid,
            "surface": surface,
            "profile": profile,
            "rho1_candidates_by_h": candidates_by_h,
            "best": None,
            "local_minima": [],
        }

    best_h_index = min(
        feasible_profile_indices,
        key=lambda index: profile[index]["rho2_log_spread_rms"],
    )
    local_minima = []
    for h_index in feasible_profile_indices:
        value = profile[h_index]["rho2_log_spread_rms"]
        left_value = (
            profile[h_index - 1]["rho2_log_spread_rms"]
            if h_index > 0 and profile[h_index - 1]["feasible"]
            else np.inf
        )
        right_value = (
            profile[h_index + 1]["rho2_log_spread_rms"]
            if h_index + 1 < len(profile) and profile[h_index + 1]["feasible"]
            else np.inf
        )
        if value <= left_value and value <= right_value:
            for candidate in candidates_by_h[h_index]["candidates"]:
                local = dict(candidate)
                local["h_grid_boundary"] = bool(h_index in {0, len(h_grid) - 1})
                local_minima.append(local)
    local_minima.sort(key=lambda row: row["rho2_log_spread_rms"])
    return {
        "h_grid_m": h_grid,
        "rho1_grid_ohm_m": rho1_grid,
        "surface": surface,
        "profile": profile,
        "rho1_candidates_by_h": candidates_by_h,
        "best": profile[best_h_index],
        "best_h_index": int(best_h_index),
        "local_minima": local_minima,
    }

def pulse_ensemble(time_s, signal_input, rpeaks_s, interval_s, config: dict, *, sign=1.0, gain=1.0):
    """Create an ensemble under an explicit, unvalidated instrument-scale scenario."""
    scale = float(config["unit_scale_to_ohm"]) * float(sign) * float(gain)
    grid = np.arange(
        -float(config["pre_s"]),
        float(config["post_s"]) + 0.5 * float(config["grid_step_s"]),
        float(config["grid_step_s"]),
    )
    left, right = map(float, interval_s)
    margin = float(config.get("hold_margin_s", 0.0))
    result = ensemble_waveform(
        time_s,
        np.asarray(signal_input, dtype=float) * scale,
        rpeaks_s,
        [left + margin, right - margin],
        grid,
        config["baseline_s"],
    )
    return {"grid_s": grid, **result}


def dynamic_two_layer_inverse(sizes_m, waveforms_ohm, static_fit: dict, state: str):
    """Solve the signed linearized two-layer inverse on an aligned beat grid."""
    sizes_m = np.asarray(sizes_m, dtype=float)
    waveforms = np.asarray(waveforms_ohm, dtype=float)
    if waveforms.ndim != 2 or waveforms.shape[0] != len(sizes_m):
        raise ValueError("Waveforms must have shape size-by-time")
    rho1 = float(static_fit["rho1_ohm_m"])
    rho2_key = f"rho2_{state}_ohm_m"
    if rho2_key not in static_fit:
        raise KeyError(rho2_key)
    rho2 = float(static_fit[rho2_key])
    h_m = float(static_fit["h_m"])
    rows = []
    for size_m in sizes_m:
        a, b = geometry_from_size(float(size_m))
        result = evaluate(rho1, rho2, h_m, a, b)
        rows.append([result.d_rho1, result.d_rho2])
    operator = np.asarray(rows)
    diagnostics = matrix_diagnostics(operator)
    solution, _, _, _ = np.linalg.lstsq(operator, waveforms, rcond=None)
    predicted = operator @ solution
    residual = waveforms - predicted
    return {
        "operator_ohm_per_ohm_m": operator,
        "operator_rank": int(diagnostics["rank"]),
        "operator_condition": float(diagnostics["condition"]),
        "delta_rho1_ohm_m": solution[0],
        "delta_rho2_ohm_m": solution[1],
        "fractional_rho1": solution[0] / rho1,
        "fractional_rho2": solution[1] / rho2,
        "predicted_waveforms_ohm": predicted,
        "residual_waveforms_ohm": residual,
        "residual_rms_ohm": float(np.sqrt(np.mean(residual**2))),
    }


def aggregate_waveforms(waveforms):
    values = np.asarray(waveforms, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("At least one finite waveform is required")
    return {
        "mean": values.mean(axis=0),
        "between_record_sd": (
            values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1])
        ),
        "n_records": int(values.shape[0]),
    }


def residual_scenario(ttrkg_fractional, tissue_fractional, sensitivities):
    """Apply a dimensionless algebraic sensitivity scenario."""
    measured = np.asarray(ttrkg_fractional, dtype=float)
    tissues = np.asarray(tissue_fractional, dtype=float)
    sensitivity = np.asarray(sensitivities, dtype=float)
    if tissues.shape[0] != 2 or sensitivity.shape != (2,):
        raise ValueError("Expected two tissue signals and two sensitivities")
    predicted = sensitivity @ tissues
    if predicted.shape != measured.shape:
        raise ValueError("TTRKG and tissue signals must share a time grid")
    residual = measured - predicted
    return {
        "predicted_tissue_fractional": predicted,
        "residual_fractional": residual,
        "measured_peak_to_peak": float(np.ptp(measured)),
        "predicted_peak_to_peak": float(np.ptp(predicted)),
        "residual_peak_to_peak": float(np.ptp(residual)),
    }


def source_design_diagnostics(ttrkg_row, side_row=None):
    """Diagnose rank for soft-tissue, lung and heart source amplitudes."""
    rows = [np.asarray(ttrkg_row, dtype=float)]
    if side_row is not None:
        rows.append(np.asarray(side_row, dtype=float))
    matrix = np.vstack(rows)
    if matrix.shape[1] != 3:
        raise ValueError("Each observation row must contain three source coefficients")
    diagnostics = matrix_diagnostics(matrix)
    return {
        "matrix": matrix,
        "rank": int(diagnostics["rank"]),
        "nullity": int(matrix.shape[1] - diagnostics["rank"]),
        "condition": float(diagnostics["condition"]),
        "singular_values": diagnostics["singular_values"],
        "nullspace": diagnostics["nullspace"],
    }


def equivalent_volume_scenarios(residual_fractional, sensitivity_per_ml, edv_grid_ml):
    """Map a signal excursion to sphere/SV/EF under deliberately strong hypotheses."""
    residual = np.asarray(residual_fractional, dtype=float)
    if residual.ndim != 1 or len(residual) < 2 or not np.isfinite(residual).all():
        raise ValueError("Residual waveform must be a finite one-dimensional vector")
    amplitude = float(np.ptp(residual))
    rows = []
    for gain in np.asarray(sensitivity_per_ml, dtype=float):
        if not np.isfinite(gain) or gain == 0:
            raise ValueError("Volume sensitivity scenarios must be finite and non-zero")
        excursion_ml = amplitude / abs(float(gain))
        for edv_ml in np.asarray(edv_grid_ml, dtype=float):
            if not np.isfinite(edv_ml) or edv_ml <= 0:
                raise ValueError("EDV scenarios must be positive and finite")
            esv_ml = float(edv_ml - excursion_ml)
            physically_admissible = bool(0.0 <= esv_ml <= edv_ml)
            radius_ed_mm = (3.0 * edv_ml * 1e-6 / (4.0 * pi)) ** (1.0 / 3.0) * 1000.0
            radius_es_mm = (
                (3.0 * esv_ml * 1e-6 / (4.0 * pi)) ** (1.0 / 3.0) * 1000.0
                if physically_admissible
                else None
            )
            rows.append({
                "volume_sensitivity_fraction_per_ml": float(gain),
                "edv_scenario_ml": float(edv_ml),
                "equivalent_volume_excursion_ml": float(excursion_ml),
                "sv_scenario_ml": float(excursion_ml),
                "esv_scenario_ml": esv_ml,
                "ef_scenario_fraction": float(excursion_ml / edv_ml),
                "equivalent_sphere_radius_ed_mm": float(radius_ed_mm),
                "equivalent_sphere_radius_es_mm": None if radius_es_mm is None else float(radius_es_mm),
                "physically_admissible": physically_admissible,
            })
    return {"residual_peak_to_peak": amplitude, "scenarios": rows}


__all__ = [
    "EXPLORATORY_STATUS",
    "SelectedAnnotation",
    "aggregate_waveforms",
    "dynamic_two_layer_inverse",
    "equivalent_volume_scenarios",
    "fit_static_at_h",
    "fit_static_at_h_bounded",
    "fit_static_bounded",
    "profile_static_h_bounded",
    "interval_median",
    "invert_rho2_at_observed_z",
    "profile_rho2_consistency",
    "rho2_consistency_at_h_rho1",
    "normalized_modes",
    "profile_static_h",
    "pulse_ensemble",
    "residual_scenario",
    "select_modes",
    "select_rpeaks",
    "source_design_diagnostics",
]
