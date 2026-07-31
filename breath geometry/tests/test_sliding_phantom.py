import numpy as np
import pytest

from breathgeom.measure.registration import (
    displacement_round_trip_metrics,
    mask_metrics,
    warp_mask,
)
from breathgeom.measure.sliding_phantom import (
    SlidingPhantomParams,
    evaluate_sliding_interface,
    make_sliding_phantom,
    make_sliding_phantom_pair,
    region_field_error,
)


def test_sliding_phantom_preserves_normal_contact_and_tangential_slip() -> None:
    phantom = make_sliding_phantom()
    metrics = evaluate_sliding_interface(phantom)

    assert phantom.transform_direction == "fixed-expiration_to_moving-inspiration"
    assert phantom.coordinate_basis == "local-zero-origin-RAS+_mm"
    assert metrics.interface_voxel_count > 500
    assert metrics.normal_mismatch_p95_mm < 1e-6
    assert metrics.tangential_slip_median_mm == pytest.approx(4.0, abs=0.05)
    assert metrics.lung_jacobian_p01 > 0.1
    assert metrics.body_jacobian_p01 > 0.1
    assert metrics.lung_nonpositive_jacobian_fraction == 0.0
    assert metrics.body_nonpositive_jacobian_fraction == 0.0


def test_tangential_field_vanishes_smoothly_towards_rotation_axis() -> None:
    params = SlidingPhantomParams()
    phantom = make_sliding_phantom(params)
    indices = np.indices(phantom.image.shape, dtype=np.float64)
    centre = (np.asarray(phantom.image.shape, dtype=np.float64) - 1.0) / 2.0
    coordinates = np.moveaxis(indices, 0, -1)
    coordinates = (
        coordinates - centre
    ) * np.asarray(phantom.spacing_mm, dtype=np.float64)
    cylindrical_level = np.sqrt(
        np.square(coordinates[..., 0] / phantom.lung_radii_mm[0])
        + np.square(coordinates[..., 1] / phantom.lung_radii_mm[1])
    )
    tangential_component = np.linalg.norm(
        phantom.lung_displacement_mm
        - np.sum(
            phantom.lung_displacement_mm * phantom.interface_normal,
            axis=-1,
        )[..., None]
        * phantom.interface_normal,
        axis=-1,
    )
    near_axis = phantom.lung_mask & (cylindrical_level < 0.1)

    assert np.any(near_axis)
    expected_upper_bound = (
        params.tangential_slip_mm
        * 0.1
        / params.tangential_full_slip_radius_fraction
    )
    assert (
        float(np.max(tangential_component[near_axis]))
        <= expected_upper_bound + 1e-6
    )


def test_historical_tangential_ramp_remains_reproducible() -> None:
    historical = make_sliding_phantom(
        SlidingPhantomParams(tangential_ramp_mode="ellipsoidal_v1")
    )
    axis_safe = make_sliding_phantom()

    assert not np.array_equal(
        historical.lung_displacement_mm,
        axis_safe.lung_displacement_mm,
    )
    assert float(np.max(historical.tangential_ramp)) == pytest.approx(1.0)
    assert float(np.max(axis_safe.tangential_ramp)) == pytest.approx(1.0)
    near_axis = axis_safe.lung_mask & (axis_safe.tangential_ramp < 0.1)
    historical_tangent = np.linalg.norm(
        historical.lung_displacement_mm
        - np.sum(
            historical.lung_displacement_mm * historical.interface_normal,
            axis=-1,
        )[..., None]
        * historical.interface_normal,
        axis=-1,
    )
    axis_safe_tangent = np.linalg.norm(
        axis_safe.lung_displacement_mm
        - np.sum(
            axis_safe.lung_displacement_mm * axis_safe.interface_normal,
            axis=-1,
        )[..., None]
        * axis_safe.interface_normal,
        axis=-1,
    )
    assert float(np.max(historical_tangent[near_axis])) > 3.5
    assert float(np.max(axis_safe_tangent[near_axis])) < 0.4


def test_sliding_phantom_rejects_unknown_tangential_ramp() -> None:
    with pytest.raises(ValueError, match="ramp mode"):
        SlidingPhantomParams(tangential_ramp_mode="unknown")


def test_longitudinal_pattern_is_smooth_and_tangential() -> None:
    phantom = make_sliding_phantom(
        SlidingPhantomParams(tangential_pattern="longitudinal_projection")
    )
    metrics = evaluate_sliding_interface(phantom)
    normal_projection = np.sum(
        phantom.interface_tangent * phantom.interface_normal,
        axis=-1,
    )

    assert np.all(np.isfinite(phantom.lung_displacement_mm))
    assert float(np.max(np.abs(normal_projection))) < 1e-6
    assert metrics.tangential_slip_median_mm > 2.0
    assert metrics.tangential_slip_median_mm < 4.0
    assert metrics.normal_mismatch_p95_mm < 1e-6
    assert metrics.lung_nonpositive_jacobian_fraction == 0.0


def test_sliding_phantom_rejects_unknown_tangential_pattern() -> None:
    with pytest.raises(ValueError, match="pattern"):
        SlidingPhantomParams(tangential_pattern="unknown")


def test_sliding_metrics_detect_wrong_body_normal_motion() -> None:
    phantom = make_sliding_phantom()
    zero_body = np.zeros_like(phantom.body_displacement_mm)
    metrics = evaluate_sliding_interface(
        phantom,
        body_displacement_mm=zero_body,
    )

    assert metrics.normal_mismatch_p95_mm > 2.5
    assert metrics.tangential_slip_median_mm == pytest.approx(4.0, abs=0.05)


def test_sliding_metrics_detect_artificially_glued_interface() -> None:
    phantom = make_sliding_phantom()
    metrics = evaluate_sliding_interface(
        phantom,
        body_displacement_mm=phantom.lung_displacement_mm,
    )

    assert metrics.normal_mismatch_p95_mm < 1e-6
    assert metrics.tangential_slip_median_mm == pytest.approx(0.0, abs=1e-6)


def test_region_field_error_uses_physical_vector_components() -> None:
    phantom = make_sliding_phantom(
        SlidingPhantomParams(
            shape=(48, 48, 48),
            lung_radii_mm=(10.0, 10.0, 14.0),
            body_radii_mm=(21.0, 20.0, 22.0),
        )
    )
    shifted = phantom.lung_displacement_mm.copy()
    shifted[..., 0] += 1.0
    error = region_field_error(
        shifted,
        phantom.lung_displacement_mm,
        phantom.lung_mask,
    )

    assert error.voxel_count == int(np.count_nonzero(phantom.lung_mask))
    assert error.mean_mm == pytest.approx(1.0)
    assert error.p95_mm == pytest.approx(1.0)


def test_sliding_phantom_rejects_lung_outside_body() -> None:
    with pytest.raises(ValueError, match="inside"):
        SlidingPhantomParams(
            lung_radii_mm=(20.0, 20.0, 20.0),
            body_radii_mm=(19.0, 25.0, 25.0),
        )


def test_sliding_phantom_texture_is_seeded_and_fields_are_seed_invariant() -> None:
    first = make_sliding_phantom(SlidingPhantomParams(texture_seed=11))
    repeated = make_sliding_phantom(SlidingPhantomParams(texture_seed=11))
    changed = make_sliding_phantom(SlidingPhantomParams(texture_seed=12))

    assert np.array_equal(first.image, repeated.image)
    assert not np.array_equal(first.image, changed.image)
    assert np.array_equal(
        first.lung_displacement_mm,
        changed.lung_displacement_mm,
    )
    assert np.array_equal(
        first.body_displacement_mm,
        changed.body_displacement_mm,
    )


def test_sliding_pair_synthesis_preserves_hidden_transform_direction() -> None:
    pair = make_sliding_phantom_pair()
    phantom = pair.phantom
    body_wall = phantom.body_mask & ~phantom.lung_mask
    lung_round_trip = displacement_round_trip_metrics(
        phantom.lung_displacement_mm,
        pair.lung_moving_to_fixed_mm,
        pair.spacing_mm,
        valid_domain=phantom.lung_mask,
    )
    body_round_trip = displacement_round_trip_metrics(
        phantom.body_displacement_mm,
        pair.body_moving_to_fixed_mm,
        pair.spacing_mm,
        valid_domain=body_wall,
    )

    assert pair.transform_direction == "fixed-expiration_to_moving-inspiration"
    assert lung_round_trip.p95_mm < max(pair.spacing_mm)
    assert body_round_trip.p95_mm < max(pair.spacing_mm)
    assert int(pair.moving_lung_mask.sum()) > int(pair.fixed_lung_mask.sum())


def test_sliding_pair_masks_warp_back_to_fixed_phase() -> None:
    pair = make_sliding_phantom_pair()
    lung_warped = warp_mask(
        pair.moving_lung_mask,
        pair.phantom.lung_displacement_mm,
        pair.spacing_mm,
    )
    body_warped = warp_mask(
        pair.moving_body_mask,
        pair.phantom.body_displacement_mm,
        pair.spacing_mm,
    )
    lung_metrics = mask_metrics(
        pair.fixed_lung_mask,
        lung_warped,
        pair.spacing_mm,
    )
    body_metrics = mask_metrics(
        pair.fixed_body_mask,
        body_warped,
        pair.spacing_mm,
    )

    assert lung_metrics.dice > 0.99
    assert body_metrics.dice > 0.93
