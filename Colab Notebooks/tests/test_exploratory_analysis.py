import numpy as np

from exploratory_analysis import (
    equivalent_volume_scenarios,
    fit_static_bounded,
    invert_rho2_at_observed_z,
    profile_static_h_bounded,
    profile_rho2_consistency,
    rho2_consistency_at_h_rho1,
    normalized_modes,
    residual_scenario,
    select_rpeaks,
    source_design_diagnostics,
)


def test_candidate_annotations_are_not_promoted():
    modes, selected = normalized_modes({
        "candidate_modes": {"hold": [1.0, 2.0]},
        "accepted_modes": None,
        "qc": {"status": "pending_manual_review"},
    })
    peaks = select_rpeaks({
        "candidate_rpeaks_s": [1.0, 2.0],
        "rpeaks_s": None,
        "qc": {"status": "pending_manual_review"},
    })
    assert modes == {"hold": [1.0, 2.0]}
    assert selected.accepted is False
    assert peaks.accepted is False


def test_residual_scenario_preserves_sign():
    measured = np.array([0.0, 2.0, -1.0])
    tissues = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    result = residual_scenario(measured, tissues, [0.5, -0.25])
    np.testing.assert_allclose(result["residual_fractional"], [-0.5, 2.25, -0.25])


def test_one_or_two_channels_do_not_identify_three_sources():
    one = source_design_diagnostics([0.2, -0.3, 1.0])
    two = source_design_diagnostics([0.2, -0.3, 1.0], [0.7, 0.1, 0.0])
    assert one["rank"] == 1 and one["nullity"] == 2
    assert two["rank"] == 2 and two["nullity"] == 1


def test_volume_scenario_flags_nonphysical_combinations():
    result = equivalent_volume_scenarios([0.0, 0.01], [1e-4], [50.0, 200.0])
    assert result["scenarios"][0]["physically_admissible"] is False
    assert result["scenarios"][1]["physically_admissible"] is True


def test_exact_z_inversion_recovers_rho2():
    from two_layer_model import evaluate, geometry_from_size

    size_m, h_m, rho1, rho2 = 0.09, 0.015, 5.0, 22.0
    a, b = geometry_from_size(size_m)
    observed = evaluate(rho1, rho2, h_m, a, b).z
    result = invert_rho2_at_observed_z(size_m, observed, h_m, rho1)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["rho2_ohm_m"], rho2, rtol=1e-8)
    assert abs(result["z_residual_ohm"]) < 1e-8


def test_rho2_consistency_is_zero_for_exact_common_layers():
    from two_layer_model import evaluate, geometry_from_size

    sizes = np.array([0.05, 0.07, 0.09, 0.12])
    h_m, rho1, rho2_in, rho2_ex = 0.015, 5.0, 22.0, 17.0

    def curve(rho2):
        values = []
        for size in sizes:
            a, b = geometry_from_size(size)
            values.append(evaluate(rho1, rho2, h_m, a, b).z)
        return np.asarray(values)

    result = rho2_consistency_at_h_rho1(
        sizes, curve(rho2_in), curve(rho2_ex), h_m, rho1
    )
    assert result["feasible"] is True
    assert result["rho2_log_spread_rms"] < 1e-9
    assert result["z_reconstruction_rms_ohm"] < 1e-8
    np.testing.assert_allclose(result["rho2_inhale_ohm_m"], rho2_in, rtol=1e-8)
    np.testing.assert_allclose(result["rho2_exhale_ohm_m"], rho2_ex, rtol=1e-8)


def test_profile_keeps_surface_and_multiple_local_candidates():
    from two_layer_model import evaluate, geometry_from_size

    sizes = np.array([0.05, 0.07, 0.09, 0.12])
    truth = {"h": 0.015, "rho1": 5.0, "rho2_in": 22.0, "rho2_ex": 17.0}

    def curve(rho2):
        return np.asarray([
            evaluate(truth["rho1"], rho2, truth["h"], *geometry_from_size(size)).z
            for size in sizes
        ])

    result = profile_rho2_consistency(
        sizes,
        curve(truth["rho2_in"]),
        curve(truth["rho2_ex"]),
        [0.01, 0.015, 0.02],
        [2.0, 5.0, 10.0],
    )
    assert len(result["surface"]) == 9
    assert result["best"] is not None
    np.testing.assert_allclose(result["best"]["h_m"], truth["h"])
    np.testing.assert_allclose(result["best"]["rho1_ohm_m"], truth["rho1"])
    assert result["best"]["rho2_log_spread_rms"] < 1e-9
    assert len(result["local_minima"]) >= 1

def test_bounded_four_parameter_fit_recovers_exact_synthetic_solution():
    from two_layer_model import evaluate, geometry_from_size

    sizes = np.array([0.05, 0.07, 0.09, 0.12, 0.14])
    truth = {"h": 0.02, "rho1": 5.0, "rho2_in": 22.0, "rho2_ex": 15.0}

    def curve(rho2):
        return np.asarray([
            evaluate(truth["rho1"], rho2, truth["h"], *geometry_from_size(size)).z
            for size in sizes
        ])

    result = fit_static_bounded(
        sizes,
        curve(truth["rho2_in"]),
        curve(truth["rho2_ex"]),
        h_bounds_m=(0.01, 0.04),
        rho1_bounds_ohm_m=(3.0, 15.0),
        rho2_bounds_ohm_m=(10.0, 25.0),
    )
    np.testing.assert_allclose(result["h_m"], truth["h"], rtol=1e-6)
    np.testing.assert_allclose(result["rho1_ohm_m"], truth["rho1"], rtol=1e-6)
    np.testing.assert_allclose(result["rho2_inhale_ohm_m"], truth["rho2_in"], rtol=1e-6)
    np.testing.assert_allclose(result["rho2_exhale_ohm_m"], truth["rho2_ex"], rtol=1e-6)
    assert result["residual_rms_ohm"] < 1e-8
    assert not any(result["parameter_bound_hits"].values())


def test_bounded_h_profile_uses_one_common_rho2_per_state():
    from two_layer_model import evaluate, geometry_from_size

    sizes = np.array([0.05, 0.07, 0.09, 0.12, 0.14])
    h_m, rho1, rho2_in, rho2_ex = 0.02, 5.0, 22.0, 15.0

    def curve(rho2):
        return np.asarray([
            evaluate(rho1, rho2, h_m, *geometry_from_size(size)).z
            for size in sizes
        ])

    result = profile_static_h_bounded(
        sizes,
        curve(rho2_in),
        curve(rho2_ex),
        [0.015, 0.02, 0.025],
        rho1_bounds_ohm_m=(3.0, 15.0),
        rho2_bounds_ohm_m=(10.0, 25.0),
    )
    np.testing.assert_allclose(result["best"]["h_m"], h_m)
    np.testing.assert_allclose(result["best"]["rho1_ohm_m"], rho1, rtol=1e-6)
    np.testing.assert_allclose(result["best"]["rho2_inhale_ohm_m"], rho2_in, rtol=1e-6)
    np.testing.assert_allclose(result["best"]["rho2_exhale_ohm_m"], rho2_ex, rtol=1e-6)
    assert result["best_h_grid_boundary"] is False

