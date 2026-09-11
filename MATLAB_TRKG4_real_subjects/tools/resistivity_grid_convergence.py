"""Pure numerical audits for Cartesian whole-compartment rho-grid convergence.

The module does not read FEM files or run a solver. It receives signed
whole-compartment derivatives S=dZ/d(rho), interpolates S on a regular Cartesian
rho grid, derives B=abs(S)*widths and Q, and audits individual values and
assembly-pair differences. The exact metric criteria names are sealed in the
refinement plan. Results are empirical numerical checks of the supplied sampled
model, not physical-accuracy, measurement-noise, or physiological claims.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


N_COMPARTMENTS = 3


# Exact sealed-plan names. bounds_atol is an operational boundary option.
DEFAULT_CRITERIA = {
    "derivative_rtol": 0.05,
    "derivative_grid_atol_fraction": 0.005,
    "Q_atol": 0.005,
    "pair_rtol": 0.05,
    "pair_S_scale_atol_fraction": 0.0001,
    "pair_Q_atol": 0.0001,
    "min_B_refinement_rtol": 0.01,
    "min_Q_refinement_atol": 0.0005,
    "preferred_B_relative_band": 0.0001,
    "preferred_Q_absolute_band": 0.0001,
    "bounds_atol": 0.0,
}


# First-draft aliases remain accepted. Results always expose canonical names.
_CRITERIA_ALIASES = {
    "derivative_grid_fraction": "derivative_grid_atol_fraction",
    "q_atol": "Q_atol",
    "pair_s_rtol": "pair_rtol",
    "pair_s_grid_fraction": "pair_S_scale_atol_fraction",
    "pair_q_rtol": "pair_rtol",
    "pair_q_floor": "pair_Q_atol",
}


def _finite_real(value, name, *, ndim=None):
    """Return float64 after strict real, finite, nonempty validation."""

    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as float64") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _nonnegative_scalar(value, name):
    """Validate a finite nonnegative, non-boolean scalar."""

    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    number = float(array)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return number


def _normalise_criteria(criteria):
    """Return canonical criteria, accepting known first-draft aliases."""

    if criteria is None:
        supplied = {}
    elif isinstance(criteria, Mapping):
        supplied = dict(criteria)
    else:
        raise ValueError("criteria must be a mapping or None")

    normalised = dict(DEFAULT_CRITERIA)
    seen = set()
    for key, value in supplied.items():
        canonical = _CRITERIA_ALIASES.get(key, key)
        if canonical not in DEFAULT_CRITERIA:
            raise ValueError(f"unknown criteria key: {key!r}")
        if canonical in seen:
            raise ValueError(
                f"criteria specify more than one alias for {canonical!r}"
            )
        seen.add(canonical)
        normalised[canonical] = _nonnegative_scalar(value, canonical)
    return normalised


def _validate_axes(axes):
    """Validate three strictly increasing, distinct one-dimensional axes."""

    if isinstance(axes, (str, bytes)):
        raise ValueError("axes must be a sequence of three numeric axes")
    try:
        axes = tuple(axes)
    except TypeError as exc:
        raise ValueError("axes must be a sequence of three numeric axes") from exc
    if len(axes) != N_COMPARTMENTS:
        raise ValueError(f"axes must contain exactly {N_COMPARTMENTS} axes")

    result = []
    for index, axis in enumerate(axes):
        array = _finite_real(axis, f"axes[{index}]", ndim=1)
        if array.size < 2:
            raise ValueError(f"axes[{index}] must contain at least two nodes")
        if np.any(np.diff(array) <= 0):
            raise ValueError(
                f"axes[{index}] must be strictly increasing and distinct"
            )
        result.append(array.copy())
    return tuple(result)


def _validate_widths(widths):
    widths = _finite_real(widths, "widths", ndim=1)
    if widths.shape != (N_COMPARTMENTS,):
        raise ValueError(f"widths must have shape ({N_COMPARTMENTS},)")
    if np.any(widths <= 0):
        raise ValueError("widths must be strictly positive")
    return widths.copy()


def _validate_grid(axes, grid_values):
    grid_values = _finite_real(grid_values, "grid_values")
    expected_prefix = tuple(axis.size for axis in axes)
    if grid_values.ndim != 5 or grid_values.shape[:3] != expected_prefix:
        raise ValueError(
            "grid_values must have shape "
            f"{expected_prefix} + (nassemblies, {N_COMPARTMENTS})"
        )
    if grid_values.shape[-1] != N_COMPARTMENTS:
        raise ValueError(
            f"grid_values last axis must have length {N_COMPARTMENTS}"
        )
    if grid_values.shape[3] < 1:
        raise ValueError("grid_values must contain at least one assembly")
    return grid_values.copy()


def _validate_check_inputs(axes, grid_values, check_points, direct_values, widths):
    axes = _validate_axes(axes)
    grid_values = _validate_grid(axes, grid_values)
    check_points = _finite_real(check_points, "check_points", ndim=2)
    if check_points.shape[1] != N_COMPARTMENTS:
        raise ValueError(
            f"check_points must have shape (nchecks, {N_COMPARTMENTS})"
        )
    direct_values = _finite_real(direct_values, "direct_values", ndim=3)
    expected_direct = (
        check_points.shape[0],
        grid_values.shape[3],
        N_COMPARTMENTS,
    )
    if direct_values.shape != expected_direct:
        raise ValueError(f"direct_values must have shape {expected_direct}")
    widths = _validate_widths(widths)
    return axes, grid_values, check_points.copy(), direct_values.copy(), widths


def _check_and_clip_points(axes, check_points, bounds_atol):
    """Check inclusive bounds and clip only points within configured slack."""

    lower = np.array([axis[0] for axis in axes], dtype=np.float64)
    upper = np.array([axis[-1] for axis in axes], dtype=np.float64)
    below = check_points < (lower - bounds_atol)
    above = check_points > (upper + bounds_atol)
    invalid = below | above
    if np.any(invalid):
        row, coordinate = np.argwhere(invalid)[0]
        value = check_points[row, coordinate]
        raise ValueError(
            "check_points must lie within all axis bounds; "
            f"point {row}, coordinate {coordinate} has value {value}"
        )
    return np.clip(check_points, lower, upper)


def derive_B_Q(signed_S, widths):
    """Derive B=abs(S)*widths and normalised Q.

    signed_S may have any leading dimensions but must end in three
    compartments. Q is undefined when all three B values are zero; its entries
    are then NaN and q_defined is false. Inputs are not modified.
    """

    signed_S = _finite_real(signed_S, "signed_S")
    if signed_S.ndim == 0 or signed_S.shape[-1] != N_COMPARTMENTS:
        raise ValueError(
            f"signed_S must have a final axis of length {N_COMPARTMENTS}"
        )
    widths = _validate_widths(widths)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        B = np.abs(signed_S) * widths
        total = np.sum(B, axis=-1, keepdims=True)
        q_defined = total[..., 0] > 0
        Q = np.full_like(B, np.nan)
        np.divide(
            B,
            total,
            out=Q,
            where=np.broadcast_to(q_defined[..., None], B.shape),
        )
    return {"B": B, "Q": Q, "q_defined": q_defined}


def _pair_differences(values, pairs):
    """Return values[:, pair[0], :] - values[:, pair[1], :]."""

    return values[:, pairs[:, 0], :] - values[:, pairs[:, 1], :]


def _all_finite_or_empty(array):
    return bool(array.size == 0 or np.all(np.isfinite(array)))


def _pair_sign_pass(interpolated_delta, direct_delta, unresolved, valid):
    """Relax only sign agreement for valid pairs marked unresolved."""

    same_sign = np.sign(interpolated_delta) == np.sign(direct_delta)
    return valid & (unresolved | same_sign)


def audit_grid(axes, grid_values, check_points, direct_values, widths, criteria=None):
    """Audit individual interpolation and paired assembly differences.

    Input shapes are axes=(three one-dimensional axes),
    grid_values=(nx,ny,nz,nassemblies,3), check_points=(nchecks,3), and
    direct_values=(nchecks,nassemblies,3). widths is a positive three-vector.

    The canonical criteria names are the exact sealed-plan names in
    DEFAULT_CRITERIA. Individual derivative tolerance is
    derivative_rtol*abs(direct) plus
    derivative_grid_atol_fraction*max_grid_abs_S. Pair-S tolerance is
    pair_rtol*abs(direct_delta) plus
    pair_S_scale_atol_fraction*same_state_max_abs_S. Pair-Q tolerance is
    pair_rtol*abs(direct_delta_Q) plus pair_Q_atol.

    Pair interpolation pass always means error <= tolerance. Pair unresolved
    means only that direct preference is inside its numerical band. It relaxes
    only robust sign agreement; it never bypasses the interpolation error
    check. Therefore pair_pass is interpolation_pass AND robust_sign_pass.
    This prevents interpolation from inventing a large false advantage for an
    essentially identical pair. Metrics are empirical numerical checks.
    """

    criteria = _normalise_criteria(criteria)
    axes, grid_values, check_points, direct_values, widths = _validate_check_inputs(
        axes, grid_values, check_points, direct_values, widths
    )
    used_points = _check_and_clip_points(
        axes, check_points, criteria["bounds_atol"]
    )

    interpolator = RegularGridInterpolator(
        axes, grid_values, method="linear", bounds_error=True
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        interpolated_S = np.asarray(interpolator(used_points), dtype=np.float64)
    if interpolated_S.shape != direct_values.shape or not np.all(
        np.isfinite(interpolated_S)
    ):
        raise ValueError("interpolation produced an incomplete or nonfinite result")

    grid_max_abs_S = np.max(np.abs(grid_values), axis=(0, 1, 2))
    grid_min_S = np.min(grid_values, axis=(0, 1, 2))
    grid_max_S = np.max(grid_values, axis=(0, 1, 2))

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        derivative_error = np.abs(interpolated_S - direct_values)
        derivative_tolerance = (
            criteria["derivative_rtol"] * np.abs(direct_values)
            + criteria["derivative_grid_atol_fraction"] * grid_max_abs_S[None, ...]
        )
    derivative_pass = derivative_error <= derivative_tolerance
    sign_reversal = (interpolated_S * direct_values) < 0

    interpolated_scores = derive_B_Q(interpolated_S, widths)
    direct_scores = derive_B_Q(direct_values, widths)
    interpolated_Q = interpolated_scores["Q"]
    direct_Q = direct_scores["Q"]
    q_defined = interpolated_scores["q_defined"] & direct_scores["q_defined"]
    valid_q = np.broadcast_to(q_defined[..., None], interpolated_Q.shape)
    Q_absolute_error = np.full_like(interpolated_Q, np.nan)
    Q_absolute_error[valid_q] = np.abs(
        interpolated_Q[valid_q] - direct_Q[valid_q]
    )
    Q_tolerance = criteria["Q_atol"]
    Q_pass = valid_q & (Q_absolute_error <= Q_tolerance)

    pairs = np.asarray(
        list(combinations(range(grid_values.shape[3]), 2)), dtype=int
    )
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=int)

    pair_direct_delta_abs_S = _pair_differences(
        np.abs(direct_values), pairs
    )
    pair_interpolated_delta_abs_S = _pair_differences(
        np.abs(interpolated_S), pairs
    )
    pair_S_error = np.abs(pair_interpolated_delta_abs_S - pair_direct_delta_abs_S)
    same_state_max_abs_S = np.max(np.abs(direct_values), axis=1)
    pair_S_tolerance = (
        criteria["pair_rtol"] * np.abs(pair_direct_delta_abs_S)
        + criteria["pair_S_scale_atol_fraction"] * same_state_max_abs_S[:, None, :]
    )
    pair_S_interpolation_pass = pair_S_error <= pair_S_tolerance
    pair_S_unresolved = np.abs(pair_direct_delta_abs_S) <= pair_S_tolerance
    pair_S_robust_sign_pass = _pair_sign_pass(
        pair_interpolated_delta_abs_S,
        pair_direct_delta_abs_S,
        pair_S_unresolved,
        np.ones(pair_S_unresolved.shape, dtype=bool),
    )
    pair_S_pass = pair_S_interpolation_pass & pair_S_robust_sign_pass
    pair_S_preference_reversal = (
        pair_interpolated_delta_abs_S * pair_direct_delta_abs_S < 0
    )

    pair_direct_delta_Q = _pair_differences(direct_Q, pairs)
    pair_interpolated_delta_Q = _pair_differences(interpolated_Q, pairs)
    if pairs.shape[0]:
        pair_Q_members_defined = q_defined[:, pairs]
        pair_Q_valid = np.broadcast_to(
            np.all(pair_Q_members_defined, axis=2)[..., None],
            pair_direct_delta_Q.shape,
        )
    else:
        pair_Q_valid = np.empty(pair_direct_delta_Q.shape, dtype=bool)
    pair_Q_error = np.full_like(pair_direct_delta_Q, np.nan)
    pair_Q_error[pair_Q_valid] = np.abs(
        pair_interpolated_delta_Q[pair_Q_valid]
        - pair_direct_delta_Q[pair_Q_valid]
    )
    pair_Q_tolerance = (
        criteria["pair_rtol"] * np.abs(pair_direct_delta_Q)
        + criteria["pair_Q_atol"]
    )
    pair_Q_interpolation_pass = pair_Q_valid & (
        pair_Q_error <= pair_Q_tolerance
    )
    pair_Q_unresolved = pair_Q_valid & (
        np.abs(pair_direct_delta_Q) <= pair_Q_tolerance
    )
    pair_Q_robust_sign_pass = _pair_sign_pass(
        pair_interpolated_delta_Q,
        pair_direct_delta_Q,
        pair_Q_unresolved,
        pair_Q_valid,
    )
    pair_Q_pass = pair_Q_interpolation_pass & pair_Q_robust_sign_pass
    pair_Q_preference_reversal = (
        pair_interpolated_delta_Q * pair_direct_delta_Q < 0
    ) & pair_Q_valid

    finite_fields = {
        "interpolated_S": interpolated_S,
        "derivative_error": derivative_error,
        "derivative_tolerance": derivative_tolerance,
        "grid_max_abs_S": grid_max_abs_S,
        "pair_S_error": pair_S_error,
        "pair_S_tolerance": pair_S_tolerance,
        "pair_Q_tolerance": pair_Q_tolerance,
    }
    if not all(_all_finite_or_empty(array) for array in finite_fields.values()):
        raise ValueError("audit arithmetic produced a nonfinite finite-valued field")

    return {
        "axes": tuple(axis.copy() for axis in axes),
        "check_points": check_points,
        "check_points_used": used_points,
        "grid_shape": grid_values.shape,
        "nassemblies": grid_values.shape[3],
        "criteria": criteria,
        "grid_max_abs_S": grid_max_abs_S,
        "grid_sign_reversal": (grid_min_S < 0) & (grid_max_S > 0),
        "interpolated_S": interpolated_S,
        "direct_S": direct_values,
        "derivative_error": derivative_error,
        "derivative_tolerance": derivative_tolerance,
        "derivative_pass": derivative_pass,
        "sign_reversal": sign_reversal,
        "interpolated_B": interpolated_scores["B"],
        "direct_B": direct_scores["B"],
        "interpolated_Q": interpolated_Q,
        "direct_Q": direct_Q,
        "Q_absolute_error": Q_absolute_error,
        "Q_tolerance": Q_tolerance,
        "Q_defined": q_defined,
        "Q_pass": Q_pass,
        "pairs": pairs,
        "same_state_max_abs_S": same_state_max_abs_S,
        "pair_direct_delta_abs_S": pair_direct_delta_abs_S,
        "pair_interpolated_delta_abs_S": pair_interpolated_delta_abs_S,
        "pair_S_error": pair_S_error,
        "pair_S_tolerance": pair_S_tolerance,
        "pair_S_interpolation_pass": pair_S_interpolation_pass,
        "pair_S_accuracy_pass": pair_S_interpolation_pass,
        "pair_S_unresolved": pair_S_unresolved,
        "pair_S_robust_sign_pass": pair_S_robust_sign_pass,
        "pair_S_pass": pair_S_pass,
        "pair_S_preference_reversal": pair_S_preference_reversal,
        "pair_direct_delta_Q": pair_direct_delta_Q,
        "pair_interpolated_delta_Q": pair_interpolated_delta_Q,
        "pair_Q_error": pair_Q_error,
        "pair_Q_tolerance": pair_Q_tolerance,
        "pair_Q_valid": pair_Q_valid,
        "pair_Q_interpolation_pass": pair_Q_interpolation_pass,
        "pair_Q_accuracy_pass": pair_Q_interpolation_pass,
        "pair_Q_unresolved": pair_Q_unresolved,
        "pair_Q_robust_sign_pass": pair_Q_robust_sign_pass,
        "pair_Q_pass": pair_Q_pass,
        "pair_Q_preference_reversal": pair_Q_preference_reversal,
        "all_derivative_pass": bool(np.all(derivative_pass)),
        "all_Q_pass": bool(np.all(Q_pass)),
        "all_pair_S_interpolation_pass": bool(np.all(pair_S_interpolation_pass)),
        "all_pair_Q_interpolation_pass": bool(np.all(pair_Q_interpolation_pass)),
        "all_pair_S_accuracy_pass": bool(np.all(pair_S_interpolation_pass)),
        "all_pair_Q_accuracy_pass": bool(np.all(pair_Q_interpolation_pass)),
        "all_pair_S_robust_sign_pass": bool(np.all(pair_S_robust_sign_pass)),
        "all_pair_Q_robust_sign_pass": bool(np.all(pair_Q_robust_sign_pass)),
        "all_pair_S_pass": bool(np.all(pair_S_pass)),
        "all_pair_Q_pass": bool(np.all(pair_Q_pass)),
        "any_sign_reversal": bool(np.any(sign_reversal)),
        "any_pair_S_preference_reversal": bool(np.any(pair_S_preference_reversal)),
        "any_pair_Q_preference_reversal": bool(np.any(pair_Q_preference_reversal)),
    }


def _nested_indices(axis_size, level):
    """Return equally indexed nested subset indices for one requested level."""

    if type(level) is not int or level < 2:
        raise ValueError("nested levels must be integers >= 2")
    if level > axis_size:
        raise ValueError(
            f"requested level {level} exceeds axis size {axis_size}"
        )
    interval_count = level - 1
    if (axis_size - 1) % interval_count:
        raise ValueError(
            "axis size cannot provide an equally indexed nested subset "
            f"of level {level}"
        )
    step = (axis_size - 1) // interval_count
    return np.arange(0, axis_size, step, dtype=int)


def _preferred_sets_for(score, metric, criteria):
    """Return preferred assembly index sets per compartment for one score."""

    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or score.shape[1] != N_COMPARTMENTS:
        raise ValueError("score must have shape (nassemblies, 3)")
    result = []
    for compartment in range(N_COMPARTMENTS):
        values = score[:, compartment]
        finite = np.isfinite(values)
        if not np.any(finite):
            result.append(tuple())
            continue
        best = np.max(values[finite])
        if metric == "B":
            band = criteria["preferred_B_relative_band"] * abs(best)
        else:
            band = criteria["preferred_Q_absolute_band"]
        result.append(tuple(np.flatnonzero(values >= best - band).tolist()))
    return tuple(result)


def _scorecard_from_states(values, widths, criteria):
    """Return conservative minima and preferred sets from arbitrary states."""

    values = _finite_real(values, "scorecard_values", ndim=3)
    if values.shape[1] < 1 or values.shape[2] != N_COMPARTMENTS:
        raise ValueError(
            "scorecard_values must have shape (nstates, nassemblies, 3)"
        )
    scores = derive_B_Q(values, widths)
    min_B = np.min(scores["B"], axis=0)
    q_complete = np.all(scores["q_defined"], axis=0)
    min_Q = np.full_like(min_B, np.nan)
    for assembly in range(values.shape[1]):
        if q_complete[assembly]:
            min_Q[assembly] = np.min(scores["Q"][:, assembly, :], axis=0)
    return {
        "min_B": min_B,
        "min_Q": min_Q,
        "q_complete": q_complete,
        "preferred_B": _preferred_sets_for(min_B, "B", criteria),
        "preferred_Q": _preferred_sets_for(min_Q, "Q", criteria),
    }


def _compare_scorecards(left, right, left_label, right_label, criteria):
    """Compare scorecards with fixed refinement and exact-set criteria."""

    delta_B = right["min_B"] - left["min_B"]
    B_tolerance = criteria["min_B_refinement_rtol"] * np.maximum(
        np.abs(left["min_B"]), np.abs(right["min_B"])
    )
    B_pass = np.abs(delta_B) <= B_tolerance

    delta_Q = right["min_Q"] - left["min_Q"]
    Q_tolerance = np.full_like(delta_Q, criteria["min_Q_refinement_atol"])
    Q_valid = np.isfinite(left["min_Q"]) & np.isfinite(right["min_Q"])
    Q_pass = Q_valid & (np.abs(delta_Q) <= Q_tolerance)

    equal_B = np.array(
        [left["preferred_B"][c] == right["preferred_B"][c] for c in range(3)],
        dtype=bool,
    )
    equal_Q = np.array(
        [left["preferred_Q"][c] == right["preferred_Q"][c] for c in range(3)],
        dtype=bool,
    )
    intersection_B = tuple(
        tuple(sorted(set(left["preferred_B"][c]) & set(right["preferred_B"][c])))
        for c in range(3)
    )
    intersection_Q = tuple(
        tuple(sorted(set(left["preferred_Q"][c]) & set(right["preferred_Q"][c])))
        for c in range(3)
    )
    stable_B = tuple(
        left["preferred_B"][c] if equal_B[c] else tuple()
        for c in range(3)
    )
    stable_Q = tuple(
        left["preferred_Q"][c] if equal_Q[c] else tuple()
        for c in range(3)
    )

    return {
        "left": left_label,
        "right": right_label,
        "delta_min_B": delta_B,
        "min_B_tolerance": B_tolerance,
        "min_B_pass": B_pass,
        "delta_min_Q": delta_Q,
        "min_Q_tolerance": Q_tolerance,
        "min_Q_pass": Q_pass,
        "preferred_B_equal": equal_B,
        "preferred_Q_equal": equal_Q,
        "preferred_set_stable_B": equal_B,
        "preferred_set_stable_Q": equal_Q,
        "preferred_set_intersection_B": intersection_B,
        "preferred_set_intersection_Q": intersection_Q,
        "stable_preferred_set_B": stable_B,
        "stable_preferred_set_Q": stable_Q,
        "min_B_refinement_pass": bool(np.all(B_pass)),
        "min_Q_refinement_pass": bool(np.all(Q_pass)),
        "preferred_B_stable": bool(np.all(equal_B)),
        "preferred_Q_stable": bool(np.all(equal_Q)),
    }


def compare_nested_scorecards(
    axes,
    grid_values,
    widths,
    *,
    levels=(3, 5),
    extra_values=None,
    criteria=None,
):
    """Compare 3-vs-5 and optionally 5-vs-(5+checks) scorecards.

    grid_values is one fine Cartesian grid. Each requested level is formed by
    equally indexed nested nodes on every axis. With default levels, a 5-node
    grid supplies levels 3 and 5; a 9-node grid supplies level 3 at indices
    (0,4,8) and level 5 at (0,2,4,6,8).

    extra_values is direct signed S data with shape
    (nextra,nassemblies,3). When supplied, the second comparison is the level-5
    grid against the union of level-5 states and extra_values.

    Refinement passes use min_B_refinement_rtol=0.01 and
    min_Q_refinement_atol=0.0005 by default. Preferred-set stability means
    exact equality of complete sets per compartment. Intersections are
    additional diagnostics only and are never called stable when sets differ.
    No universal winner is selected.
    """

    criteria = _normalise_criteria(criteria)
    axes = _validate_axes(axes)
    grid_values = _validate_grid(axes, grid_values)
    widths = _validate_widths(widths)

    if isinstance(levels, (str, bytes)):
        raise ValueError("levels must be a sequence of at least two integers")
    try:
        levels = tuple(levels)
    except TypeError as exc:
        raise ValueError("levels must be a sequence of at least two integers") from exc
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ValueError("levels must contain at least two distinct integers")
    if any(type(level) is not int or level < 2 for level in levels):
        raise ValueError("levels must contain integers >= 2")

    indices_by_level = {
        level: tuple(_nested_indices(axis.size, level) for axis in axes)
        for level in levels
    }

    cards = {}
    level_results = []
    for level in levels:
        i0, i1, i2 = indices_by_level[level]
        subset = grid_values[
            np.ix_(
                i0,
                i1,
                i2,
                np.arange(grid_values.shape[3]),
                np.arange(N_COMPARTMENTS),
            )
        ]
        card = _scorecard_from_states(
            subset.reshape(-1, grid_values.shape[3], N_COMPARTMENTS),
            widths,
            criteria,
        )
        cards[level] = card
        level_results.append(
            {
                "level": level,
                "indices": tuple(
                    tuple(int(x) for x in index)
                    for index in indices_by_level[level]
                ),
                **card,
            }
        )

    if levels[0] == 3 and levels[1] == 5:
        first_label = "3_vs_5"
    else:
        first_label = f"{levels[0]}_vs_{levels[1]}"
    refinements = {
        first_label: _compare_scorecards(
            cards[levels[0]],
            cards[levels[1]],
            str(levels[0]),
            str(levels[1]),
            criteria,
        )
    }

    extra_card = None
    if extra_values is not None:
        extra_values = _finite_real(extra_values, "extra_values", ndim=3)
        expected = (grid_values.shape[3], N_COMPARTMENTS)
        if extra_values.shape[1:] != expected:
            raise ValueError(
                "extra_values must have shape (nextra, nassemblies, 3)"
            )
        i0, i1, i2 = indices_by_level[levels[1]]
        level5_values = grid_values[
            np.ix_(
                i0,
                i1,
                i2,
                np.arange(grid_values.shape[3]),
                np.arange(N_COMPARTMENTS),
            )
        ].reshape(-1, grid_values.shape[3], N_COMPARTMENTS)
        extra_card = _scorecard_from_states(
            np.concatenate([level5_values, extra_values], axis=0),
            widths,
            criteria,
        )
        refinements["5_vs_5_plus_checks"] = _compare_scorecards(
            cards[levels[1]],
            extra_card,
            "5",
            "5+checks",
            criteria,
        )

    min_B = np.stack([cards[level]["min_B"] for level in levels], axis=0)
    min_Q = np.stack([cards[level]["min_Q"] for level in levels], axis=0)

    def exact_stable(preferred):
        return tuple(
            preferred[0][c]
            if all(level[c] == preferred[0][c] for level in preferred[1:])
            else tuple()
            for c in range(N_COMPARTMENTS)
        )

    def all_intersection(preferred):
        result = []
        for c in range(N_COMPARTMENTS):
            common = set(preferred[0][c])
            for level in preferred[1:]:
                common &= set(level[c])
            result.append(tuple(sorted(common)))
        return tuple(result)

    preferred_B = tuple(cards[level]["preferred_B"] for level in levels)
    preferred_Q = tuple(cards[level]["preferred_Q"] for level in levels)
    result = {
        "levels": levels,
        "criteria": criteria,
        "grid_shape": grid_values.shape,
        "subset_indices": {
            level: tuple(
                tuple(int(x) for x in index)
                for index in indices_by_level[level]
            )
            for level in levels
        },
        "level_results": tuple(level_results),
        "scorecards": {level: cards[level] for level in levels},
        "min_B": min_B,
        "min_Q": min_Q,
        "delta_min_B_last_minus_first": min_B[-1] - min_B[0],
        "delta_min_Q_last_minus_first": min_Q[-1] - min_Q[0],
        "preferred_set_B": preferred_B,
        "preferred_set_Q": preferred_Q,
        "stable_preferred_set_B": exact_stable(preferred_B),
        "stable_preferred_set_Q": exact_stable(preferred_Q),
        "preferred_set_intersection_B": all_intersection(preferred_B),
        "preferred_set_intersection_Q": all_intersection(preferred_Q),
        "refinement_comparisons": refinements,
        "extra_scorecard": extra_card,
    }
    if extra_card is not None:
        result["min_B_5_plus_checks"] = extra_card["min_B"]
        result["min_Q_5_plus_checks"] = extra_card["min_Q"]
        result["preferred_set_B_5_plus_checks"] = extra_card["preferred_B"]
        result["preferred_set_Q_5_plus_checks"] = extra_card["preferred_Q"]
    return result


nested_scorecard_comparison = compare_nested_scorecards


__all__ = [
    "DEFAULT_CRITERIA",
    "N_COMPARTMENTS",
    "audit_grid",
    "compare_nested_scorecards",
    "derive_B_Q",
    "nested_scorecard_comparison",
]

