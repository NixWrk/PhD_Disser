"""Regional stationary-velocity representation for sliding registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from breathgeom.measure.registration import (
    VectorArray,
    exponentiate_stationary_velocity,
)


@dataclass(frozen=True)
class PiecewiseSVFTransform:
    """Two regional fixed-to-moving transforms without cross-interface blending."""

    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    squaring_steps: int
    outside_mode: str
    transform_direction: str = "fixed-expiration_to_moving-inspiration"


def exponentiate_piecewise_svf(
    lung_velocity_mm: VectorArray,
    body_velocity_mm: VectorArray,
    spacing: tuple[float, float, float],
    *,
    squaring_steps: int = 8,
    outside_mode: Literal["zero", "nearest"] = "nearest",
    transform_direction: str = "fixed-expiration_to_moving-inspiration",
) -> PiecewiseSVFTransform:
    """Exponentiate lung and body velocities independently on a shared grid."""
    if lung_velocity_mm.shape != body_velocity_mm.shape:
        raise ValueError("lung and body velocity fields must share one grid")
    if not transform_direction:
        raise ValueError("transform_direction must not be empty")
    lung = exponentiate_stationary_velocity(
        lung_velocity_mm,
        spacing,
        squaring_steps=squaring_steps,
        outside_mode=outside_mode,
    )
    body = exponentiate_stationary_velocity(
        body_velocity_mm,
        spacing,
        squaring_steps=squaring_steps,
        outside_mode=outside_mode,
    )
    return PiecewiseSVFTransform(
        lung_displacement_mm=lung,
        body_displacement_mm=body,
        squaring_steps=squaring_steps,
        outside_mode=outside_mode,
        transform_direction=transform_direction,
    )


__all__ = [
    "PiecewiseSVFTransform",
    "exponentiate_piecewise_svf",
]
