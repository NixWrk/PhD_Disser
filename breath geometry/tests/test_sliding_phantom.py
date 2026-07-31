import numpy as np
import pytest

from breathgeom.measure.sliding_phantom import (
    SlidingPhantomParams,
    evaluate_sliding_interface,
    make_sliding_phantom,
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
