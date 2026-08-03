"""Deterministic multi-region phantom for pleural sliding validation.

The phantom deliberately stores separate lung and body-wall displacement fields.
Their normal components agree at the pleural interface, while the lung has an
additional tangential component.  A single globally smooth DVF cannot represent
that jump without spreading the slip into one of the adjacent regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.registration import (
    invert_displacement,
    jacobian_metrics,
    warp_image,
    warp_mask,
)
from breathgeom.measure.wall import IntArray as WallIntArray

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int16]
VectorArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class SlidingPhantomParams:
    """Physical dimensions and known expiration-to-inspiration motion."""

    shape: tuple[int, int, int] = (72, 72, 72)
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lung_radii_mm: tuple[float, float, float] = (16.0, 16.0, 22.0)
    body_radii_mm: tuple[float, float, float] = (31.0, 29.0, 33.0)
    normal_motion_mm: float = 3.0
    tangential_slip_mm: float = 4.0
    tangential_pattern: str = "azimuthal"
    tangential_ramp_mode: str = "cylindrical_axis_safe"
    tangential_full_slip_radius_fraction: float = 0.75
    interface_band_mm: float = 1.0
    jacobian_margin_mm: float = 2.0
    texture_seed: int = 20260731
    texture_correlation_mm: float = 1.2
    lung_texture_std_hu: float = 20.0
    body_texture_std_hu: float = 6.0

    def __post_init__(self) -> None:
        if any(value < 16 for value in self.shape):
            raise ValueError("phantom shape must be at least 16 voxels on every axis")
        if any(value <= 0 for value in self.spacing_mm):
            raise ValueError("spacing must be positive")
        if any(value <= 0 for value in self.lung_radii_mm + self.body_radii_mm):
            raise ValueError("phantom radii must be positive")
        if any(
            lung >= body
            for lung, body in zip(
                self.lung_radii_mm, self.body_radii_mm, strict=True
            )
        ):
            raise ValueError("lung must fit strictly inside the body")
        half_extent = (
            (np.asarray(self.shape, dtype=np.float64) - 1.0)
            * np.asarray(self.spacing_mm, dtype=np.float64)
            / 2.0
        )
        if any(
            radius >= extent
            for radius, extent in zip(
                self.body_radii_mm, half_extent.tolist(), strict=True
            )
        ):
            raise ValueError("body ellipsoid must fit inside the phantom grid")
        if self.normal_motion_mm < 0 or self.tangential_slip_mm < 0:
            raise ValueError("motion amplitudes must be non-negative")
        if self.tangential_pattern not in {"azimuthal", "longitudinal_projection"}:
            raise ValueError("unknown tangential pattern")
        if self.tangential_ramp_mode not in {
            "ellipsoidal_v1",
            "cylindrical_axis_safe",
        }:
            raise ValueError("unknown tangential ramp mode")
        if not 0 < self.tangential_full_slip_radius_fraction <= 1:
            raise ValueError("full-slip radius fraction must lie in (0, 1]")
        if self.interface_band_mm <= 0 or self.jacobian_margin_mm <= 0:
            raise ValueError("validation margins must be positive")
        if self.texture_correlation_mm <= 0:
            raise ValueError("texture correlation length must be positive")
        if self.lung_texture_std_hu < 0 or self.body_texture_std_hu < 0:
            raise ValueError("texture amplitudes must be non-negative")


@dataclass(frozen=True)
class SlidingPhantom:
    """Known two-region motion in local zero-origin RAS+ millimetres."""

    image: IntArray
    body_mask: BoolArray
    lung_mask: BoolArray
    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    interface_normal: FloatArray
    interface_tangent: FloatArray
    tangential_ramp: FloatArray
    lung_level_set: FloatArray
    body_level_set: FloatArray
    spacing_mm: tuple[float, float, float]
    interface_band_mm: float
    jacobian_margin_mm: float
    lung_radii_mm: tuple[float, float, float]
    body_radii_mm: tuple[float, float, float]
    coordinate_basis: str = "local-zero-origin-RAS+_mm"
    transform_direction: str = "fixed-expiration_to_moving-inspiration"


@dataclass(frozen=True)
class SlidingPhantomPair:
    """Synthetic expiration/inspiration pair with hidden regional transforms."""

    phantom: SlidingPhantom
    fixed_image: IntArray
    moving_image: IntArray
    fixed_lung_mask: BoolArray
    moving_lung_mask: BoolArray
    fixed_body_mask: BoolArray
    moving_body_mask: BoolArray
    lung_moving_to_fixed_mm: VectorArray
    body_moving_to_fixed_mm: VectorArray

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        return self.phantom.spacing_mm

    @property
    def transform_direction(self) -> str:
        return self.phantom.transform_direction


@dataclass(frozen=True)
class SlidingInterfaceMetrics:
    """Independent checks for a piecewise-smooth sliding representation."""

    interface_voxel_count: int
    normal_mismatch_mean_mm: float
    normal_mismatch_p95_mm: float
    tangential_slip_median_mm: float
    tangential_slip_p95_mm: float
    lung_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    body_jacobian_p01: float
    body_nonpositive_jacobian_fraction: float


@dataclass(frozen=True)
class RegionFieldError:
    """Endpoint error of one estimated field against known phantom truth."""

    voxel_count: int
    mean_mm: float
    p95_mm: float
    maximum_mm: float


def _physical_coordinates(
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> FloatArray:
    indices = np.indices(shape, dtype=np.float64)
    centre = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    coordinates = np.moveaxis(indices, 0, -1)
    return (coordinates - centre) * spacing


def _ellipsoid_level(coordinates_mm: FloatArray, radii_mm: tuple[float, ...]) -> FloatArray:
    radii = np.asarray(radii_mm, dtype=np.float64)
    return cast(
        FloatArray,
        np.sqrt(np.sum(np.square(coordinates_mm / radii), axis=-1)),
    )


def _ellipsoid_normal(
    coordinates_mm: FloatArray,
    radii_mm: tuple[float, ...],
) -> FloatArray:
    radii = np.asarray(radii_mm, dtype=np.float64)
    gradient = coordinates_mm / np.square(radii)
    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    return cast(
        FloatArray,
        np.divide(
            gradient,
            norm,
            out=np.zeros_like(gradient),
            where=norm > np.finfo(np.float64).eps,
        ),
    )


def _azimuthal_tangent(
    coordinates_mm: FloatArray,
    normal: FloatArray,
) -> FloatArray:
    rotation = np.stack(
        (
            -coordinates_mm[..., 1],
            coordinates_mm[..., 0],
            np.zeros(coordinates_mm.shape[:-1], dtype=np.float64),
        ),
        axis=-1,
    )
    rotation -= np.sum(rotation * normal, axis=-1, keepdims=True) * normal
    norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
    return cast(
        FloatArray,
        np.divide(
            rotation,
            norm,
            out=np.zeros_like(rotation),
            where=norm > np.finfo(np.float64).eps,
        ),
    )


def make_sliding_phantom(
    params: SlidingPhantomParams | None = None,
) -> SlidingPhantom:
    """Create deterministic anatomy and exact piecewise expiration-to-inspiration fields."""
    params = params or SlidingPhantomParams()
    coordinates = _physical_coordinates(params.shape, params.spacing_mm)
    lung_level = _ellipsoid_level(coordinates, params.lung_radii_mm)
    body_level = _ellipsoid_level(coordinates, params.body_radii_mm)
    lung_mask = lung_level <= 1.0
    body_mask = body_level <= 1.0
    normal = _ellipsoid_normal(coordinates, params.lung_radii_mm)
    # The normal amplitude is constant throughout a narrow two-sided interface
    # band.  It decays only deeper into the body wall.  Lung motion ramps from
    # zero at the centre to avoid an orientation singularity.
    lung_ramp = np.clip(lung_level / 0.75, 0.0, 1.0)
    if params.tangential_pattern == "longitudinal_projection":
        longitudinal = np.zeros_like(coordinates)
        longitudinal[..., 2] = 1.0
        projected = longitudinal - (
            np.sum(longitudinal * normal, axis=-1, keepdims=True) * normal
        )
        projected_norm = np.linalg.norm(projected, axis=-1, keepdims=True)
        tangent = np.divide(
            projected,
            projected_norm,
            out=np.zeros_like(projected),
            where=projected_norm > np.finfo(np.float64).eps,
        )
        # Keep the unnormalised projection amplitude.  It is smooth and
        # naturally vanishes where the superior direction is normal to the
        # pleura, unlike a normalised tangent at those poles.
        tangential_ramp = lung_ramp * projected_norm[..., 0]
    else:
        tangent = _azimuthal_tangent(coordinates, normal)
        if params.tangential_ramp_mode == "ellipsoidal_v1":
            # Historical phantom v2.0 behaviour.  This is retained only so that
            # the superseded S1.0 report remains exactly reproducible.
            tangential_ramp = lung_ramp
        else:
            # An azimuthal unit tangent is undefined on its rotation axis.
            # Scaling by cylindrical distance makes the resulting vector field
            # continuous there, including at the superior/inferior poles.
            cylindrical_level = np.sqrt(
                np.square(coordinates[..., 0] / params.lung_radii_mm[0])
                + np.square(coordinates[..., 1] / params.lung_radii_mm[1])
            )
            tangential_ramp = np.clip(
                cylindrical_level / params.tangential_full_slip_radius_fraction,
                0.0,
                1.0,
            )
    interface_hold = params.interface_band_mm / min(params.lung_radii_mm) + 0.02
    body_excess = np.maximum(lung_level - 1.0 - interface_hold, 0.0)
    body_normal_amplitude = params.normal_motion_mm * np.exp(-body_excess / 0.8)

    lung_field = (
        params.normal_motion_mm * lung_ramp[..., None] * normal
        + params.tangential_slip_mm * tangential_ramp[..., None] * tangent
    )
    body_field = body_normal_amplitude[..., None] * normal
    lung_field = np.where(body_mask[..., None], lung_field, 0.0).astype(np.float32)
    body_field = np.where(body_mask[..., None], body_field, 0.0).astype(np.float32)

    texture = (
        12.0 * np.sin(coordinates[..., 0] / 3.5)
        + 9.0 * np.cos(coordinates[..., 1] / 4.5)
        + 7.0 * np.sin(coordinates[..., 2] / 5.5)
    )
    random = np.random.default_rng(params.texture_seed).normal(size=params.shape)
    correlation_voxels = tuple(
        params.texture_correlation_mm / value for value in params.spacing_mm
    )
    correlated = ndimage.gaussian_filter(
        random,
        sigma=correlation_voxels,
        mode="reflect",
    )
    correlated -= float(np.mean(correlated))
    correlated_std = float(np.std(correlated))
    if correlated_std <= np.finfo(np.float64).eps:
        raise ValueError("phantom texture has zero variance")
    correlated /= correlated_std
    image_float = np.full(params.shape, -1000.0, dtype=np.float64)
    image_float[body_mask] = (
        35.0
        + texture[body_mask]
        + params.body_texture_std_hu * correlated[body_mask]
    )
    fat_shell = body_mask & (body_level >= 0.82)
    image_float[fat_shell] = (
        -100.0
        + 0.5 * texture[fat_shell]
        + params.body_texture_std_hu * correlated[fat_shell]
    )
    image_float[lung_mask] = (
        -820.0
        + 1.5 * texture[lung_mask]
        + params.lung_texture_std_hu * correlated[lung_mask]
    )
    image = np.rint(image_float).astype(np.int16)

    return SlidingPhantom(
        image=image,
        body_mask=np.ascontiguousarray(body_mask),
        lung_mask=np.ascontiguousarray(lung_mask),
        lung_displacement_mm=np.ascontiguousarray(lung_field),
        body_displacement_mm=np.ascontiguousarray(body_field),
        interface_normal=np.ascontiguousarray(normal),
        interface_tangent=np.ascontiguousarray(tangent),
        tangential_ramp=np.ascontiguousarray(tangential_ramp),
        lung_level_set=np.ascontiguousarray(lung_level),
        body_level_set=np.ascontiguousarray(body_level),
        spacing_mm=params.spacing_mm,
        interface_band_mm=params.interface_band_mm,
        jacobian_margin_mm=params.jacobian_margin_mm,
        lung_radii_mm=params.lung_radii_mm,
        body_radii_mm=params.body_radii_mm,
    )


def _nearest_field_extension(
    field_mm: VectorArray,
    valid_domain: BoolArray,
) -> VectorArray:
    """Extend a regional field for numerical inversion only."""
    nearest = ndimage.distance_transform_edt(
        ~valid_domain,
        return_distances=False,
        return_indices=True,
    )
    extended = field_mm[tuple(nearest)]
    return np.ascontiguousarray(extended.astype(np.float32))


def make_sliding_phantom_pair(
    params: SlidingPhantomParams | None = None,
) -> SlidingPhantomPair:
    """Generate phase images without exposing truth to a registration candidate.

    The analytical fields map fixed expiration coordinates to moving
    inspiration coordinates.  They are inverted only to synthesize the moving
    phase.  A candidate later receives the phase images and masks, not either
    field.
    """
    phantom = make_sliding_phantom(params)
    lung_inverse = invert_displacement(
        phantom.lung_displacement_mm,
        phantom.spacing_mm,
    )
    body_extension = _nearest_field_extension(
        phantom.body_displacement_mm,
        phantom.body_mask,
    )
    body_inverse = invert_displacement(
        body_extension,
        phantom.spacing_mm,
    )
    moving_lung = warp_mask(
        phantom.lung_mask,
        lung_inverse,
        phantom.spacing_mm,
    )
    moving_body = warp_mask(
        phantom.body_mask,
        body_inverse,
        phantom.spacing_mm,
    )
    lung_values = warp_image(
        cast(WallIntArray, phantom.image),
        lung_inverse,
        phantom.spacing_mm,
    )
    body_values = warp_image(
        cast(WallIntArray, phantom.image),
        body_inverse,
        phantom.spacing_mm,
    )
    moving = np.full(phantom.image.shape, -1000, dtype=np.int16)
    moving_wall = moving_body & ~moving_lung
    moving[moving_wall] = body_values[moving_wall]
    moving[moving_lung] = lung_values[moving_lung]

    return SlidingPhantomPair(
        phantom=phantom,
        fixed_image=phantom.image,
        moving_image=np.ascontiguousarray(moving),
        fixed_lung_mask=phantom.lung_mask,
        moving_lung_mask=np.ascontiguousarray(moving_lung),
        fixed_body_mask=phantom.body_mask,
        moving_body_mask=np.ascontiguousarray(moving_body),
        lung_moving_to_fixed_mm=lung_inverse,
        body_moving_to_fixed_mm=body_inverse,
    )


def evaluate_sliding_interface(
    phantom: SlidingPhantom,
    *,
    lung_displacement_mm: VectorArray | None = None,
    body_displacement_mm: VectorArray | None = None,
) -> SlidingInterfaceMetrics:
    """Evaluate normal agreement, retained slip, and regional Jacobians."""
    lung_field = (
        phantom.lung_displacement_mm
        if lung_displacement_mm is None
        else lung_displacement_mm
    )
    body_field = (
        phantom.body_displacement_mm
        if body_displacement_mm is None
        else body_displacement_mm
    )
    expected_shape = phantom.image.shape + (3,)
    if lung_field.shape != expected_shape or body_field.shape != expected_shape:
        raise ValueError(f"fields must have shape {expected_shape}")

    level_margin = phantom.interface_band_mm / min(phantom.lung_radii_mm)
    interface = phantom.body_mask & (
        np.abs(phantom.lung_level_set - 1.0) <= level_margin
    )
    if not np.any(interface):
        raise ValueError("phantom interface band is empty")

    difference = lung_field - body_field
    signed_normal_difference = np.sum(
        difference * phantom.interface_normal,
        axis=-1,
    )
    tangential_difference = (
        difference
        - signed_normal_difference[..., None] * phantom.interface_normal
    )
    normal_mismatch = np.abs(signed_normal_difference[interface])
    tangential_slip = np.linalg.norm(tangential_difference[interface], axis=-1)

    lung_level_margin = phantom.jacobian_margin_mm / min(phantom.lung_radii_mm)
    body_level_margin = phantom.jacobian_margin_mm / min(phantom.body_radii_mm)
    lung_core = phantom.lung_mask & (
        phantom.lung_level_set <= 1.0 - lung_level_margin
    )
    body_core = (
        phantom.body_mask
        & (phantom.lung_level_set >= 1.0 + lung_level_margin)
        & (phantom.body_level_set <= 1.0 - body_level_margin)
    )
    lung_jacobian = jacobian_metrics(
        lung_field,
        phantom.spacing_mm,
        valid_domain=lung_core,
    )
    body_jacobian = jacobian_metrics(
        body_field,
        phantom.spacing_mm,
        valid_domain=body_core,
    )
    return SlidingInterfaceMetrics(
        interface_voxel_count=int(np.count_nonzero(interface)),
        normal_mismatch_mean_mm=float(np.mean(normal_mismatch)),
        normal_mismatch_p95_mm=float(np.percentile(normal_mismatch, 95)),
        tangential_slip_median_mm=float(np.median(tangential_slip)),
        tangential_slip_p95_mm=float(np.percentile(tangential_slip, 95)),
        lung_jacobian_p01=lung_jacobian.p01,
        lung_nonpositive_jacobian_fraction=lung_jacobian.nonpositive_fraction,
        body_jacobian_p01=body_jacobian.p01,
        body_nonpositive_jacobian_fraction=body_jacobian.nonpositive_fraction,
    )


def region_field_error(
    estimated_mm: VectorArray,
    reference_mm: VectorArray,
    valid_domain: BoolArray,
) -> RegionFieldError:
    """Compute endpoint error without treating voxels as statistical subjects."""
    if estimated_mm.shape != reference_mm.shape:
        raise ValueError("estimated and reference fields must have equal shapes")
    if estimated_mm.shape[:-1] != valid_domain.shape or estimated_mm.shape[-1] != 3:
        raise ValueError("valid_domain must match a three-component field")
    if not np.any(valid_domain):
        raise ValueError("valid_domain is empty")
    error = np.linalg.norm(estimated_mm - reference_mm, axis=-1)[valid_domain]
    return RegionFieldError(
        voxel_count=len(error),
        mean_mm=float(np.mean(error)),
        p95_mm=float(np.percentile(error, 95)),
        maximum_mm=float(np.max(error)),
    )


__all__ = [
    "RegionFieldError",
    "SlidingInterfaceMetrics",
    "SlidingPhantom",
    "SlidingPhantomPair",
    "SlidingPhantomParams",
    "evaluate_sliding_interface",
    "make_sliding_phantom",
    "make_sliding_phantom_pair",
    "region_field_error",
]
