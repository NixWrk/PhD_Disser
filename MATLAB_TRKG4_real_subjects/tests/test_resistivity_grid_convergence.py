"""Manufactured numerical checks for rho-grid convergence.

No FEM files, solver calls, medical data, or physical/noise assumptions are
used. These tests exercise interpolation, pairwise error and sign handling,
sealed criteria names, input validation, and nested conservative scorecards.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "resistivity_grid_convergence.py"
)
_SPEC = importlib.util.spec_from_file_location("resistivity_grid_convergence", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
grid = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(grid)


WIDTHS = np.array([2.0, 3.0, 5.0])


SEALED_CRITERIA = {
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
}


def _multilinear(points):
    """Return a positive multilinear field with shape (..., assemblies, 3)."""

    x, y, z = (points[..., i] for i in range(3))
    values = np.empty(points.shape[:-1] + (2, 3), dtype=float)
    values[..., 0, 0] = 4.0 + 0.7 * x + 0.2 * y + 0.1 * z + 0.05 * x * y
    values[..., 0, 1] = 2.0 + 0.1 * x + 0.6 * y + 0.2 * z + 0.03 * x * z
    values[..., 0, 2] = 1.0 + 0.2 * x + 0.1 * y + 0.8 * z + 0.04 * y * z
    values[..., 1, 0] = 3.0 + 0.4 * x + 0.1 * y + 0.2 * z + 0.02 * y * z
    values[..., 1, 1] = 2.5 + 0.3 * x + 0.5 * y + 0.1 * z + 0.02 * x * y
    values[..., 1, 2] = 1.5 + 0.1 * x + 0.2 * y + 0.7 * z + 0.03 * x * z
    return values


def _case(level=3):
    axes = tuple(np.linspace(0.0, 2.0, level) for _ in range(3))
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = _multilinear(mesh)
    points = np.array(
        [[0.25, 0.75, 1.25], [1.4, 0.35, 1.8], [0.9, 1.6, 0.2]],
        dtype=float,
    )
    direct = _multilinear(points)
    return axes, values, points, direct


def test_exact_sealed_criteria_names_are_exposed_and_used():
    result = grid.audit_grid(*_case(), WIDTHS, SEALED_CRITERIA)
    assert set(SEALED_CRITERIA).issubset(result["criteria"])
    assert result["criteria"]["derivative_grid_atol_fraction"] == 0.005
    assert result["criteria"]["pair_S_scale_atol_fraction"] == 0.0001


def test_multilinear_derivatives_are_recovered_and_q_is_derived_after_s_interpolation():
    axes, values, points, direct = _case(3)
    result = grid.audit_grid(axes, values, points, direct, WIDTHS, SEALED_CRITERIA)

    np.testing.assert_allclose(result["interpolated_S"], direct, atol=1e-12)
    np.testing.assert_allclose(result["derivative_error"], 0.0, atol=1e-12)
    assert result["all_derivative_pass"]
    assert result["all_Q_pass"]
    assert result["all_pair_S_pass"]
    assert result["all_pair_Q_pass"]

    expected_B = np.abs(direct) * WIDTHS
    expected_Q = expected_B / expected_B.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(result["direct_B"], expected_B)
    np.testing.assert_allclose(result["direct_Q"], expected_Q)
    np.testing.assert_allclose(result["interpolated_Q"], expected_Q)
    assert result["pairs"].shape == (1, 2)


def test_correlated_common_grid_error_is_rejected_for_s_but_cancels_in_pair_difference():
    axes, values, points, _ = _case(3)
    interpolated = _multilinear(points)
    direct = interpolated + np.array([0.8, 0.4, 0.2])
    result = grid.audit_grid(
        axes, values, points, direct, WIDTHS, SEALED_CRITERIA
    )

    assert not result["all_derivative_pass"]
    assert result["all_pair_S_interpolation_pass"]
    np.testing.assert_allclose(result["pair_S_error"], 0.0, atol=1e-12)
    assert not result["any_sign_reversal"]


def test_direct_equal_pair_with_artificial_interpolated_difference_fails_pair_pass():
    axes = (np.array([0.0, 1.0]),) * 3
    values = np.ones((2, 2, 2, 2, 3), dtype=float)
    values[1, 1, 1, 1, 0] = 4.0
    points = np.array([[0.5, 0.5, 0.5]])
    direct = np.ones((1, 2, 3), dtype=float)
    result = grid.audit_grid(
        axes, values, points, direct, WIDTHS, SEALED_CRITERIA
    )

    assert result["pair_S_unresolved"][0, 0, 0]
    assert not result["pair_S_interpolation_pass"][0, 0, 0]
    assert not result["pair_S_pass"][0, 0, 0]
    assert not result["all_pair_S_pass"]
    assert result["pair_S_robust_sign_pass"][0, 0, 0]


def test_unresolved_pair_skips_only_sign_requirement_when_error_is_small():
    axes = (np.array([0.0, 1.0]),) * 3
    values = np.ones((2, 2, 2, 2, 3), dtype=float)
    points = np.array([[0.5, 0.5, 0.5]])
    direct = np.ones((1, 2, 3), dtype=float)
    result = grid.audit_grid(axes, values, points, direct, WIDTHS, SEALED_CRITERIA)

    assert result["pair_S_unresolved"].all()
    assert result["pair_S_interpolation_pass"].all()
    assert result["pair_S_robust_sign_pass"].all()
    assert result["pair_S_pass"].all()


def test_sign_and_pair_preference_reversal_are_reported():
    axes = (np.array([0.0, 1.0]),) * 3
    values = np.zeros((2, 2, 2, 2, 3), dtype=float)
    values[..., 0, :] = [2.0, 1.0, 1.0]
    values[..., 1, :] = [1.0, 2.0, 1.0]
    points = np.array([[0.5, 0.5, 0.5]])
    direct = np.array([[[-0.1, 1.0, 1.0], [1.5, 1.0, 1.0]]])
    result = grid.audit_grid(axes, values, points, direct, WIDTHS, SEALED_CRITERIA)

    assert result["sign_reversal"][0, 0, 0]
    assert result["any_sign_reversal"]
    assert result["pair_S_preference_reversal"][0, 0, 0]
    assert result["any_pair_S_preference_reversal"]
    assert result["Q_defined"].all()


def test_tiny_pair_difference_is_unresolved_but_error_is_still_required():
    axes = (np.array([0.0, 1.0]),) * 3
    values = np.ones((2, 2, 2, 2, 3), dtype=float)
    values[..., 0, :] = [2.0, 1.0, 1.0]
    values[..., 1, :] = [2.0, 1.0, 1.0]
    points = np.array([[0.5, 0.5, 0.5]])
    direct = values[0, 0, 0][None, ...].copy()
    direct[0, 1, 0] += 1e-12
    result = grid.audit_grid(
        axes, values, points, direct, WIDTHS, SEALED_CRITERIA
    )

    assert result["pair_S_unresolved"][0, 0, 0]
    assert result["pair_Q_unresolved"][0, 0, 0]
    assert result["pair_S_pass"].all()
    assert result["pair_Q_pass"].all()
    assert not result["pair_S_preference_reversal"].any()


@pytest.mark.parametrize("kind", ["duplicate_axis", "nonfinite_grid"])
def test_invalid_axes_and_nonfinite_grid_are_rejected(kind):
    axes, values, points, direct = _case(3)
    if kind == "duplicate_axis":
        axes = list(axes)
        axes[0] = axes[0].copy()
        axes[0][1] = axes[0][0]
        match = "strictly increasing"
    else:
        values = values.copy()
        values[0, 0, 0, 0, 0] = np.nan
        match = "finite"
    with pytest.raises(ValueError, match=match):
        grid.audit_grid(axes, values, points, direct, WIDTHS, SEALED_CRITERIA)


def test_shape_bounds_and_width_validation_is_explicit():
    axes, values, points, direct = _case(3)
    with pytest.raises(ValueError, match="shape"):
        grid.audit_grid(axes, values[:-1], points, direct, WIDTHS, SEALED_CRITERIA)
    with pytest.raises(ValueError, match="bounds"):
        grid.audit_grid(
            axes,
            values,
            np.array([[2.1, 0.5, 0.5]]),
            direct[:1],
            WIDTHS,
            SEALED_CRITERIA,
        )
    with pytest.raises(ValueError, match="widths"):
        grid.audit_grid(
            axes, values, points, direct, [1.0, 0.0, 1.0], SEALED_CRITERIA
        )
    with pytest.raises(ValueError, match="direct_values"):
        grid.audit_grid(
            axes, values, points, direct[:, :, :2], WIDTHS, SEALED_CRITERIA
        )


def test_bounds_atol_clips_only_a_small_boundary_roundoff():
    axes, values, points, direct = _case(3)
    near_boundary = np.array([[-1e-10, 0.5, 0.5]])
    direct_one = direct[:1]
    with pytest.raises(ValueError, match="bounds"):
        grid.audit_grid(
            axes, values, near_boundary, direct_one, WIDTHS, SEALED_CRITERIA
        )
    result = grid.audit_grid(
        axes,
        values,
        near_boundary,
        direct_one,
        WIDTHS,
        {**SEALED_CRITERIA, "bounds_atol": 1e-9},
    )
    assert result["check_points_used"][0, 0] == 0.0


def test_nested_level_3_vs_5_scorecards_use_sealed_refinement_and_exact_stability():
    axes = tuple(np.linspace(0.0, 4.0, 5) for _ in range(3))
    values = np.empty((5, 5, 5, 2, 3), dtype=float)
    values[..., 0, :] = [10.0, 1.0, 1.0]
    values[..., 1, :] = [5.0, 4.0, 1.0]
    result = grid.compare_nested_scorecards(
        axes, values, WIDTHS, criteria=SEALED_CRITERIA
    )

    assert result["levels"] == (3, 5)
    assert result["subset_indices"][3][0] == (0, 2, 4)
    assert result["subset_indices"][5][0] == (0, 1, 2, 3, 4)
    np.testing.assert_allclose(result["min_B"][0], result["min_B"][1])
    np.testing.assert_allclose(result["min_Q"][0], result["min_Q"][1])
    assert result["stable_preferred_set_B"] == ((0,), (1,), (0, 1))
    assert result["stable_preferred_set_Q"] == ((0,), (1,), (1,))
    assert result["refinement_comparisons"]["3_vs_5"]["preferred_B_stable"]
    assert result["refinement_comparisons"]["3_vs_5"]["preferred_Q_stable"]
    assert "universal_winner" not in result


def test_preferred_set_difference_is_not_called_stable_even_with_nonempty_intersection():
    axes = tuple(np.linspace(0.0, 4.0, 5) for _ in range(3))
    values = np.empty((5, 5, 5, 3, 3), dtype=float)
    values[..., 0, :] = [10.0, 1.0, 1.0]
    values[..., 1, :] = [9.999, 1.0, 1.0]
    values[..., 2, :] = [1.0, 4.0, 1.0]
    extra = np.array(
        [[[1.0, 1.0, 1.0], [9.0, 1.0, 1.0], [1.0, 4.0, 1.0]]]
    )
    result = grid.compare_nested_scorecards(
        axes, values, WIDTHS, extra_values=extra, criteria=SEALED_CRITERIA
    )
    comparison = result["refinement_comparisons"]["5_vs_5_plus_checks"]
    assert not comparison["preferred_set_stable_B"][0]
    assert comparison["preferred_set_intersection_B"][0] == (1,)
    assert comparison["stable_preferred_set_B"][0] == ()


def test_nested_scorecards_accept_extra_direct_checks_and_compare_5_vs_5_plus_checks():
    axes = tuple(np.linspace(0.0, 4.0, 5) for _ in range(3))
    values = np.empty((5, 5, 5, 2, 3), dtype=float)
    values[..., 0, :] = [10.0, 1.0, 1.0]
    values[..., 1, :] = [5.0, 4.0, 1.0]
    extra = np.array([[[10.0, 1.0, 1.0], [5.0, 4.0, 1.0]]])
    result = grid.compare_nested_scorecards(
        axes, values, WIDTHS, extra_values=extra, criteria=SEALED_CRITERIA
    )
    assert "5_vs_5_plus_checks" in result["refinement_comparisons"]
    comparison = result["refinement_comparisons"]["5_vs_5_plus_checks"]
    assert comparison["min_B_refinement_pass"]
    assert comparison["min_Q_refinement_pass"]
    assert comparison["preferred_B_stable"]
    assert comparison["preferred_Q_stable"]
    assert result["extra_scorecard"] is not None


def test_nested_scorecards_reject_non_nested_axis_sizes_and_unknown_criteria():
    axes, values, _, _ = _case(4)
    with pytest.raises(ValueError, match="nested subset"):
        grid.compare_nested_scorecards(
            axes, values, WIDTHS, levels=(3, 5), criteria=SEALED_CRITERIA
        )
    axes, values, points, direct = _case(3)
    with pytest.raises(ValueError, match="unknown criteria"):
        grid.audit_grid(
            axes,
            values,
            points,
            direct,
            WIDTHS,
            {**SEALED_CRITERIA, "made_up_floor": 1.0},
        )
