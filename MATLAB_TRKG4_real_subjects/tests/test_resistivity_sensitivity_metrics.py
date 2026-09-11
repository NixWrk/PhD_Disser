"""Analytic resistor checks and edge cases; no FEM or experimental data."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools" / "resistivity_sensitivity_metrics.py"
_SPEC = importlib.util.spec_from_file_location("resistivity_sensitivity_metrics", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metrics)


@pytest.mark.parametrize("topology", ["series", "parallel"])
def test_resistors_chain_rule_and_centered_resistivity_perturbations(topology):
    # R_i = rho_i * L_i/A_i; geometry [1/m], rho [Ohm*m], Z [Ohm].
    geometry = np.array([0.3, 1.2, 2.0])
    rho = np.array([[0.5, 2.0, 4.0], [1.5, 0.7, 6.0]])
    sigma = 1 / rho

    def impedance(resistivity):
        resistance = resistivity * geometry
        if topology == "series":
            return resistance.sum(axis=-1)
        return 1 / (1 / resistance).sum(axis=-1)

    if topology == "series":
        derivative_sigma = -geometry / sigma**2
        expected = np.broadcast_to(geometry, rho.shape)
    else:
        z = impedance(rho)[..., None]
        derivative_sigma = -z**2 / geometry
        expected = z**2 / (geometry * rho**2)

    recovered_rho, derivative_rho = metrics.conductivity_to_resistivity(
        sigma, derivative_sigma
    )
    np.testing.assert_allclose(recovered_rho, rho)
    np.testing.assert_allclose(derivative_rho, expected, rtol=1e-13)
    for compartment in range(rho.shape[-1]):
        perturbation = np.zeros_like(rho)
        step = rho[:, compartment] * 1e-5
        perturbation[:, compartment] = step
        centered = (impedance(rho + perturbation) - impedance(rho - perturbation)) / (2 * step)
        np.testing.assert_allclose(derivative_rho[:, compartment], centered, rtol=1e-8)


def test_parallel_finite_change_is_not_the_local_range_score():
    rho = np.array([1.0, 2.0])
    delta = np.array([0.8, 0.1])
    z = 1 / np.sum(1 / rho)  # Two parallel resistors with L/A = 1 / m.
    _, derivative = metrics.conductivity_to_resistivity(1 / rho, [-z**2, -z**2])
    linear_score = metrics.range_scaled_metrics(derivative, delta)["signed_response"].sum()
    finite_change = 1 / np.sum(1 / (rho + delta)) - z
    assert abs(linear_score - finite_change) > 0.01


def test_conductivity_broadcast_signs_and_no_input_mutation():
    sigma = np.array([0.5, 2.0])
    derivative = np.array([[-8.0, 3.0], [4.0, 0.0]])
    rho, converted = metrics.conductivity_to_resistivity(sigma, derivative)
    np.testing.assert_allclose(rho, [[2, 0.5], [2, 0.5]])
    np.testing.assert_allclose(converted, [[2, -12], [-1, 0]])
    np.testing.assert_array_equal(sigma, [0.5, 2])
    np.testing.assert_array_equal(derivative, [[-8, 3], [4, 0]])
    scalar_rho, scalar_derivative = metrics.conductivity_to_resistivity(2, -3)
    assert scalar_rho == 0.5 and scalar_derivative == 12


def test_range_scores_preserve_sign_and_ignore_cross_compartment_cancellation():
    result = metrics.range_scaled_metrics([2, -1, 0], [3, 6, 4])
    np.testing.assert_array_equal(result["signed_response"], [6, -6, 0])
    assert result["signed_response"].sum() == 0
    np.testing.assert_array_equal(result["absolute_response"], [6, 6, 0])
    np.testing.assert_allclose(result["selectivity"], [0.5, 0.5, 0])
    np.testing.assert_array_equal(result["nuisance_sum"], [6, 6, 12])
    np.testing.assert_allclose(result["nuisance_to_target"], [1, 1, np.nan])
    np.testing.assert_array_equal(result["selectivity_undefined"], [False] * 3)
    np.testing.assert_array_equal(result["nuisance_to_target_undefined"], [False, False, True])


def test_all_zero_and_single_target_cases():
    result = metrics.range_scaled_metrics([[0, 0, 0], [0, -2, 0]], [1, 1, 1])
    np.testing.assert_array_equal(result["nuisance_sum"], [[0, 0, 0], [2, 0, 2]])
    np.testing.assert_allclose(result["selectivity"], [[np.nan] * 3, [0, 1, 0]])
    np.testing.assert_allclose(result["nuisance_to_target"], [[np.nan] * 3, [np.nan, 0, np.nan]])
    np.testing.assert_array_equal(result["selectivity_undefined"], [[True] * 3, [False] * 3])
    np.testing.assert_array_equal(
        result["nuisance_to_target_undefined"], [[True] * 3, [True, False, True]]
    )
    single = metrics.range_scaled_metrics([2], [3])
    np.testing.assert_array_equal(single["selectivity"], [1])
    np.testing.assert_array_equal(single["nuisance_sum"], [0])
    np.testing.assert_array_equal(single["nuisance_to_target"], [0])


@pytest.mark.parametrize("factor", [1e-100, 0.1, 7.0, 1e100])
def test_selectivity_and_ratio_common_scale_invariance(factor):
    derivative = np.array([[2.0, -3.0, 0.5], [0.0, 1e-14, -1.0]])
    scale = np.array([3.0, 2.0, 1.0])
    baseline = metrics.range_scaled_metrics(derivative, scale)
    for result in (
        metrics.range_scaled_metrics(derivative, scale * factor),
        metrics.range_scaled_metrics(derivative * factor, scale),
        metrics.range_scaled_metrics(-derivative * factor, scale),
    ):
        for key in ("selectivity", "nuisance_to_target"):
            np.testing.assert_allclose(result[key], baseline[key], rtol=1e-14)
        for key in ("selectivity_undefined", "nuisance_to_target_undefined"):
            np.testing.assert_array_equal(result[key], baseline[key])


def test_near_zero_guard_is_explicit_and_tiny_nuisance_is_retained():
    result = metrics.range_scaled_metrics([1, 1e-20, 0], [1, 1, 1])
    assert result["nuisance_sum"][0] == 1e-20
    assert result["nuisance_to_target"][0] == 1e-20
    assert np.isfinite(result["selectivity"][1])
    np.testing.assert_array_equal(result["nuisance_to_target_undefined"], [False, True, True])
    exact_only = metrics.range_scaled_metrics([1, 1e-20, 0], [1, 1, 1], target_rtol=0)
    np.testing.assert_allclose(exact_only["nuisance_to_target"][:2], [1e-20, 1e20])
    assert np.isnan(exact_only["nuisance_to_target"][2])
    boundary = metrics.range_scaled_metrics([1, 3], [1, 1], target_rtol=0.25)
    np.testing.assert_array_equal(boundary["nuisance_to_target_undefined"], [True, False])


def test_leading_axes_broadcast_with_fixed_compartment_axis():
    derivative = np.arange(1.0, 19.0).reshape(2, 3, 3)
    scales = np.array([[[2, 3, 4]], [[5, 6, 7]]])
    original = derivative.copy()
    result = metrics.range_scaled_metrics(derivative, scales)
    for value in result.values():
        assert value.shape == (2, 3, 3)
    for state in range(2):
        for assembly in range(3):
            expected = derivative[state, assembly] * scales[state, 0]
            np.testing.assert_allclose(result["signed_response"][state, assembly], expected)
            np.testing.assert_allclose(result["selectivity"][state, assembly], expected / expected.sum())
    np.testing.assert_array_equal(derivative, original)
    exact = metrics.range_scaled_metrics(derivative, np.broadcast_to(scales, derivative.shape))
    np.testing.assert_allclose(exact["signed_response"], result["signed_response"])


def test_state_summary_zero_contact_reversal_and_sampled_absolute_extrema():
    values = np.array([[-2, -2, 0, 0, 1, -3], [3, 0, 4, 0, 2, -1]])
    result = metrics.summarize_over_states(values)
    np.testing.assert_array_equal(result["min"], [-2, -2, 0, 0, 1, -3])
    np.testing.assert_array_equal(result["max"], [3, 0, 4, 0, 2, -1])
    np.testing.assert_array_equal(result["sign_changed"], [True, True, True, False, False, False])
    np.testing.assert_array_equal(result["min_abs"], [2, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(result["max_abs"], [3, 2, 4, 0, 2, 3])
    one_state = metrics.summarize_over_states(values[:1, None, :])
    assert one_state["min"].shape == (1, 6)
    assert not one_state["sign_changed"].any()
    scalar = metrics.summarize_over_states([-2, 3])
    assert scalar["sign_changed"] and scalar["min_abs"] == 2


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [-np.inf], [1 + 0j], ["1"], [True], [None]])
def test_invalid_numeric_inputs(bad):
    for call in (
        lambda: metrics.conductivity_to_resistivity(bad, [1]),
        lambda: metrics.conductivity_to_resistivity([1], bad),
        lambda: metrics.range_scaled_metrics(bad, [1]),
        lambda: metrics.range_scaled_metrics([1], bad),
        lambda: metrics.summarize_over_states(bad),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [[0], [-1], [-0.0]])
def test_conductivity_and_range_scales_must_be_positive(bad):
    with pytest.raises(ValueError):
        metrics.conductivity_to_resistivity(bad, [1])
    with pytest.raises(ValueError):
        metrics.range_scaled_metrics([1], bad)


@pytest.mark.parametrize("derivative,scale", [
    (1, [1]), ([1], 1), ([1, 2, 3], [1]),
    (np.ones((2, 3)), np.ones((4, 3))), (np.empty((0, 3)), [1, 1, 1]),
])
def test_invalid_range_shapes(derivative, scale):
    with pytest.raises(ValueError):
        metrics.range_scaled_metrics(derivative, scale)


def test_invalid_conversion_and_state_shapes():
    with pytest.raises(ValueError):
        metrics.conductivity_to_resistivity([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        metrics.summarize_over_states(1)
    with pytest.raises(ValueError):
        metrics.summarize_over_states(np.empty((2, 0)))


@pytest.mark.parametrize("bad", [-1, 1, np.nan, np.inf, [0], True])
def test_invalid_ratio_tolerance(bad):
    with pytest.raises(ValueError):
        metrics.range_scaled_metrics([1], [1], target_rtol=bad)


def test_unsafe_arithmetic_does_not_silently_produce_nonfinite_or_zero_scores():
    for call in (
        lambda: metrics.conductivity_to_resistivity(1e-320, 1),
        lambda: metrics.conductivity_to_resistivity(1e200, 1e200),
        lambda: metrics.range_scaled_metrics([1e308], [2]),
        lambda: metrics.range_scaled_metrics([1e308, 1e308], [1, 1]),
        lambda: metrics.range_scaled_metrics([1e-200], [1e-200]),
        lambda: metrics.range_scaled_metrics([1e-310, 1], [1, 1], target_rtol=0),
    ):
        with pytest.raises(FloatingPointError):
            call()
