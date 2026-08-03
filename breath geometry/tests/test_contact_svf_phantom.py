from pathlib import Path

import numpy as np
import pytest

from breathgeom.measure.contact_svf_phantom import (
    ContactSVFPhantomParams,
    analytic_target_signed_distance,
    load_contact_svf_suite,
    make_contact_svf_phantom_pair,
    sample_fixed_interface_points,
)
from breathgeom.measure.registration import (
    displacement_round_trip_metrics,
    jacobian_metrics,
    transform_points,
)
from breathgeom.measure.sliding_phantom import evaluate_sliding_interface

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_PATH = (
    REPO_ROOT / "configs/piecewise_svf_j12_contact_development_suite_v2.json"
)


@pytest.fixture(scope="module")
def development():
    return load_contact_svf_suite(DEVELOPMENT_PATH)


def test_frozen_contact_suite_schema_and_split(development) -> None:
    assert development.suite_version == "piecewise-svf-j1.2-contact-development-v2"
    assert development.generator_version == "contact-svf-phantom-v1"
    assert development.transform_direction == "fixed-expiration_to_moving-inspiration"
    assert len(development.cases) == 3
    assert [case.params.motion_kind for case in development.cases] == [
        "axisymmetric_affine_rotation",
        "axisymmetric_longitudinal_twist",
        "axisymmetric_affine_rotation",
    ]


def test_contact_params_reject_non_axisymmetric_interface(development) -> None:
    source = development.cases[0].params
    values = vars(source) | {"lung_radii_mm": (15.0, 14.0, 20.0)}
    with pytest.raises(ValueError, match="axisymmetric lung"):
        ContactSVFPhantomParams(**values)


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_regional_truth_maps_to_one_analytic_surface(
    development,
    case_index: int,
) -> None:
    params = development.cases[case_index].params
    generated = make_contact_svf_phantom_pair(params)
    points = sample_fixed_interface_points(params)
    lung_points = transform_points(
        points,
        generated.pair.phantom.lung_displacement_mm,
        generated.pair.spacing_mm,
    )
    body_points = transform_points(
        points,
        generated.pair.phantom.body_displacement_mm,
        generated.pair.spacing_mm,
    )
    lung_signed = analytic_target_signed_distance(
        lung_points,
        center_mm=generated.center_mm,
        target_radii_mm=generated.target_lung_radii_mm,
    )
    body_signed = analytic_target_signed_distance(
        body_points,
        center_mm=generated.center_mm,
        target_radii_mm=generated.target_lung_radii_mm,
    )

    assert float(np.percentile(np.abs(lung_signed), 95)) <= 0.05
    assert float(np.percentile(np.abs(body_signed), 95)) <= 0.05
    assert float(np.mean(np.abs(lung_signed) <= 0.05)) >= 0.99
    assert float(np.mean(np.abs(body_signed) <= 0.05)) >= 0.99


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_contact_truth_is_invertible_and_has_slip(
    development,
    case_index: int,
) -> None:
    generated = make_contact_svf_phantom_pair(development.cases[case_index].params)
    pair = generated.pair
    phantom = pair.phantom
    lung_round_trip = displacement_round_trip_metrics(
        phantom.lung_displacement_mm,
        pair.lung_moving_to_fixed_mm,
        pair.spacing_mm,
        valid_domain=phantom.lung_mask,
    )
    body_wall = phantom.body_mask & ~phantom.lung_mask
    body_round_trip = displacement_round_trip_metrics(
        phantom.body_displacement_mm,
        pair.body_moving_to_fixed_mm,
        pair.spacing_mm,
        valid_domain=body_wall,
    )
    interface = evaluate_sliding_interface(phantom)
    lung_jacobian = jacobian_metrics(
        phantom.lung_displacement_mm,
        pair.spacing_mm,
        valid_domain=phantom.lung_mask,
    )
    body_jacobian = jacobian_metrics(
        phantom.body_displacement_mm,
        pair.spacing_mm,
        valid_domain=body_wall,
    )

    assert lung_round_trip.p95_mm <= 0.05
    assert body_round_trip.p95_mm <= 0.05
    assert interface.tangential_slip_median_mm >= 0.5
    assert lung_jacobian.p01 >= 0.2
    assert body_jacobian.p01 >= 0.2
    assert lung_jacobian.nonpositive_fraction == 0.0
    assert body_jacobian.nonpositive_fraction == 0.0
