import numpy as np
import pytest

from breathgeom.measure.piecewise_svf import exponentiate_piecewise_svf
from breathgeom.measure.registration import (
    displacement_round_trip_metrics,
    exponentiate_stationary_velocity,
)


def test_piecewise_svf_keeps_lung_and_body_translations_separate() -> None:
    shape = (20, 18, 16)
    lung_velocity = np.zeros(shape + (3,), dtype=np.float32)
    body_velocity = np.zeros_like(lung_velocity)
    lung_velocity[..., 0] = 1.5
    lung_velocity[..., 2] = 2.0
    body_velocity[..., 0] = 1.5
    body_velocity[..., 1] = -3.0

    transform = exponentiate_piecewise_svf(
        lung_velocity,
        body_velocity,
        (1.0, 1.0, 1.0),
        squaring_steps=7,
    )

    assert transform.lung_displacement_mm == pytest.approx(lung_velocity, abs=1e-6)
    assert transform.body_displacement_mm == pytest.approx(body_velocity, abs=1e-6)
    assert not np.allclose(
        transform.lung_displacement_mm,
        transform.body_displacement_mm,
    )
    assert transform.transform_direction == "fixed-expiration_to_moving-inspiration"


def test_piecewise_svf_inverse_velocity_has_small_interior_round_trip() -> None:
    shape = (30, 24, 20)
    spacing = (1.0, 1.0, 1.0)
    velocity = np.zeros(shape + (3,), dtype=np.float32)
    coordinate = np.arange(shape[0], dtype=np.float32)[:, None, None] - 15.0
    velocity[..., 0] = 0.02 * coordinate
    forward = exponentiate_stationary_velocity(
        velocity,
        spacing,
        squaring_steps=8,
    )
    inverse = exponentiate_stationary_velocity(
        -velocity,
        spacing,
        squaring_steps=8,
    )
    interior = np.zeros(shape, dtype=bool)
    interior[4:-4, 4:-4, 4:-4] = True

    round_trip = displacement_round_trip_metrics(
        forward,
        inverse,
        spacing,
        valid_domain=interior,
    )

    assert round_trip.p95_mm < 5e-4
