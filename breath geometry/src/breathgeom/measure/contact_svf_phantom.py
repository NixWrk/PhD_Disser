"""Contact-valid two-region phantom generated from regional stationary velocities."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.piecewise_svf import exponentiate_piecewise_svf
from breathgeom.measure.registration import VectorArray, warp_image
from breathgeom.measure.sliding_phantom import (
    BoolArray,
    FloatArray,
    SlidingPhantom,
    SlidingPhantomPair,
)
from breathgeom.measure.wall import IntArray as WallIntArray

MotionKind = Literal[
    "axisymmetric_affine_rotation",
    "axisymmetric_longitudinal_twist",
]
OutsideMode = Literal["zero", "nearest"]
FloatVectorArray = npt.NDArray[np.float64]
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ContactSVFPhantomParams:
    """Physical and texture parameters for one contact-valid phase pair."""

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    lung_radii_mm: tuple[float, float, float]
    body_radii_mm: tuple[float, float, float]
    motion_kind: MotionKind
    common_log_scales: tuple[float, float, float]
    lung_motion_rad: float
    body_motion_rad: float
    squaring_steps: int
    outside_mode: OutsideMode
    interface_band_mm: float
    jacobian_margin_mm: float
    texture_seed: int
    texture_correlation_mm: float
    lung_texture_std_hu: float
    body_texture_std_hu: float

    def __post_init__(self) -> None:
        numeric = (
            self.spacing_mm
            + self.lung_radii_mm
            + self.body_radii_mm
            + self.common_log_scales
            + (
                self.lung_motion_rad,
                self.body_motion_rad,
                self.interface_band_mm,
                self.jacobian_margin_mm,
                self.texture_correlation_mm,
                self.lung_texture_std_hu,
                self.body_texture_std_hu,
            )
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("all physical parameters must be finite")
        if any(value < 24 for value in self.shape):
            raise ValueError("phantom shape must be at least 24 voxels on every axis")
        if any(value <= 0 for value in self.spacing_mm):
            raise ValueError("spacing must be positive")
        if any(value <= 0 for value in self.lung_radii_mm + self.body_radii_mm):
            raise ValueError("phantom radii must be positive")
        if not math.isclose(
            self.lung_radii_mm[0],
            self.lung_radii_mm[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("contact construction requires axisymmetric lung radii")
        if not math.isclose(
            self.body_radii_mm[0],
            self.body_radii_mm[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("contact construction requires axisymmetric body radii")
        if not math.isclose(
            self.common_log_scales[0],
            self.common_log_scales[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("contact construction requires equal X/Y log scales")
        if any(
            lung >= body
            for lung, body in zip(
                self.lung_radii_mm,
                self.body_radii_mm,
                strict=True,
            )
        ):
            raise ValueError("lung must fit strictly inside the body")
        if self.lung_motion_rad == self.body_motion_rad:
            raise ValueError("regional motions must differ to create sliding")
        if not 1 <= self.squaring_steps <= 12:
            raise ValueError("squaring_steps must lie in [1, 12]")
        if self.outside_mode not in {"zero", "nearest"}:
            raise ValueError("outside_mode must be zero or nearest")
        if self.interface_band_mm <= 0 or self.jacobian_margin_mm <= 0:
            raise ValueError("validation margins must be positive")
        if self.texture_correlation_mm <= 0:
            raise ValueError("texture correlation length must be positive")
        if self.lung_texture_std_hu < 0 or self.body_texture_std_hu < 0:
            raise ValueError("texture amplitudes must be non-negative")
        half_extent = (
            (np.asarray(self.shape, dtype=np.float64) - 1.0)
            * np.asarray(self.spacing_mm, dtype=np.float64)
            / 2.0
        )
        target_body = np.asarray(self.body_radii_mm) * np.exp(
            np.asarray(self.common_log_scales)
        )
        required_margin = 2.0 * max(self.spacing_mm)
        if np.any(target_body + required_margin >= half_extent):
            raise ValueError("moving body must fit inside FOV with a two-voxel margin")


@dataclass(frozen=True)
class ContactSVFSuiteCase:
    """One frozen contact-valid generator case."""

    case_id: str
    description: str
    params: ContactSVFPhantomParams


@dataclass(frozen=True)
class ContactSVFSuite:
    """Versioned development or held-out contact-valid suite."""

    suite_version: str
    generator_version: str
    transform_direction: str
    coordinate_basis: str
    cases: tuple[ContactSVFSuiteCase, ...]


@dataclass(frozen=True)
class ContactSVFPhantomPair:
    """Generated images plus analytic and numerical hidden truth."""

    pair: SlidingPhantomPair
    lung_velocity_mm: VectorArray
    body_velocity_mm: VectorArray
    analytic_lung_displacement_mm: FloatVectorArray
    analytic_body_displacement_mm: FloatVectorArray
    target_lung_radii_mm: tuple[float, float, float]
    center_mm: tuple[float, float, float]


def _tuple_values(
    payload: dict[str, Any],
    key: str,
    length: int,
    caster: type[int] | type[float],
) -> tuple[Any, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must be a JSON array of length {length}")
    return tuple(caster(item) for item in value)


def load_contact_svf_suite(path: Path) -> ContactSVFSuite:
    """Load a suite only when the caller is permitted to open that split."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "suite_version",
        "generator_version",
        "transform_direction",
        "coordinate_basis",
        "cases",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"suite must contain exactly {sorted(required)}")
    header_keys = required - {"cases"}
    if any(not isinstance(payload[key], str) or not payload[key] for key in header_keys):
        raise ValueError("suite header values must be non-empty strings")
    if payload["transform_direction"] != "fixed-expiration_to_moving-inspiration":
        raise ValueError("unsupported transform direction")
    if payload["coordinate_basis"] != "local-zero-origin-RAS+_mm":
        raise ValueError("unsupported coordinate basis")
    case_payloads = payload["cases"]
    if not isinstance(case_payloads, list) or not case_payloads:
        raise ValueError("cases must be a non-empty JSON array")
    allowed_params = {item.name for item in fields(ContactSVFPhantomParams)}
    identifiers: set[str] = set()
    cases: list[ContactSVFSuiteCase] = []
    for item in case_payloads:
        if not isinstance(item, dict) or set(item) != {
            "case_id",
            "description",
            "params",
        }:
            raise ValueError("each case must contain case_id, description and params")
        case_id = item["case_id"]
        description = item["description"]
        params_payload = item["params"]
        if (
            not isinstance(case_id, str)
            or not CASE_ID_PATTERN.fullmatch(case_id)
            or case_id in identifiers
        ):
            raise ValueError(f"invalid or duplicate case_id: {case_id!r}")
        if not isinstance(description, str) or not description:
            raise ValueError(f"{case_id}: description must be non-empty")
        if not isinstance(params_payload, dict) or set(params_payload) != allowed_params:
            missing = allowed_params - set(params_payload or {})
            extra = set(params_payload or {}) - allowed_params
            raise ValueError(
                f"{case_id}: invalid params, missing={sorted(missing)}, extra={sorted(extra)}"
            )
        values = cast(dict[str, Any], params_payload).copy()
        values["shape"] = _tuple_values(values, "shape", 3, int)
        for key in (
            "spacing_mm",
            "lung_radii_mm",
            "body_radii_mm",
            "common_log_scales",
        ):
            values[key] = _tuple_values(values, key, 3, float)
        cases.append(
            ContactSVFSuiteCase(
                case_id=case_id,
                description=description,
                params=ContactSVFPhantomParams(**values),
            )
        )
        identifiers.add(case_id)
    return ContactSVFSuite(
        suite_version=payload["suite_version"],
        generator_version=payload["generator_version"],
        transform_direction=payload["transform_direction"],
        coordinate_basis=payload["coordinate_basis"],
        cases=tuple(cases),
    )


def _relative_grid(
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> tuple[FloatVectorArray, npt.NDArray[np.float64]]:
    indices = np.indices(shape, dtype=np.float64)
    center_index = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    relative = (np.moveaxis(indices, 0, -1) - center_index) * spacing
    center = center_index * spacing
    return np.ascontiguousarray(relative), center


def _ellipsoid_level(
    relative_mm: FloatVectorArray,
    radii_mm: tuple[float, float, float],
) -> FloatArray:
    radii = np.asarray(radii_mm, dtype=np.float64)
    return cast(
        FloatArray,
        np.sqrt(np.sum(np.square(relative_mm / radii), axis=-1)),
    )


def _ellipsoid_normal(
    relative_mm: FloatVectorArray,
    radii_mm: tuple[float, float, float],
) -> FloatVectorArray:
    radii = np.asarray(radii_mm, dtype=np.float64)
    gradient = relative_mm / np.square(radii)
    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    return cast(
        FloatVectorArray,
        np.divide(
            gradient,
            norm,
            out=np.zeros_like(gradient),
            where=norm > np.finfo(np.float64).eps,
        ),
    )


def _azimuthal_tangent(relative_mm: FloatVectorArray) -> FloatVectorArray:
    tangent = np.zeros_like(relative_mm)
    tangent[..., 0] = -relative_mm[..., 1]
    tangent[..., 1] = relative_mm[..., 0]
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    return cast(
        FloatVectorArray,
        np.divide(
            tangent,
            norm,
            out=np.zeros_like(tangent),
            where=norm > np.finfo(np.float64).eps,
        ),
    )


def _motion_fields(
    params: ContactSVFPhantomParams,
    relative_mm: FloatVectorArray,
    motion_rad: float,
) -> tuple[VectorArray, FloatVectorArray]:
    scales = np.asarray(params.common_log_scales, dtype=np.float64)
    x = relative_mm[..., 0]
    y = relative_mm[..., 1]
    z = relative_mm[..., 2]
    if params.motion_kind == "axisymmetric_affine_rotation":
        angular_velocity = np.full(x.shape, motion_rad, dtype=np.float64)
        final_angle = angular_velocity
    else:
        angular_velocity = motion_rad * z / params.lung_radii_mm[2]
        if abs(scales[2]) < 1e-12:
            final_angle = angular_velocity
        else:
            final_angle = angular_velocity * math.expm1(scales[2]) / scales[2]
    velocity = np.empty(relative_mm.shape, dtype=np.float32)
    velocity[..., 0] = (scales[0] * x - angular_velocity * y).astype(np.float32)
    velocity[..., 1] = (angular_velocity * x + scales[1] * y).astype(np.float32)
    velocity[..., 2] = (scales[2] * z).astype(np.float32)
    exp_xy = math.exp(scales[0])
    cosine = np.cos(final_angle)
    sine = np.sin(final_angle)
    transformed = np.empty_like(relative_mm)
    transformed[..., 0] = exp_xy * (cosine * x - sine * y)
    transformed[..., 1] = exp_xy * (sine * x + cosine * y)
    transformed[..., 2] = math.exp(scales[2]) * z
    return np.ascontiguousarray(velocity), np.ascontiguousarray(
        transformed - relative_mm
    )


def _fixed_image(
    params: ContactSVFPhantomParams,
    relative_mm: FloatVectorArray,
    lung_mask: BoolArray,
    body_mask: BoolArray,
    body_level: FloatArray,
) -> npt.NDArray[np.int16]:
    texture = (
        12.0 * np.sin(relative_mm[..., 0] / 3.5)
        + 9.0 * np.cos(relative_mm[..., 1] / 4.5)
        + 7.0 * np.sin(relative_mm[..., 2] / 5.5)
    )
    random = np.random.default_rng(params.texture_seed).normal(size=params.shape)
    sigma = tuple(
        params.texture_correlation_mm / spacing for spacing in params.spacing_mm
    )
    correlated = ndimage.gaussian_filter(random, sigma=sigma, mode="reflect")
    correlated -= float(np.mean(correlated))
    standard_deviation = float(np.std(correlated))
    if standard_deviation <= np.finfo(np.float64).eps:
        raise ValueError("phantom texture has zero variance")
    correlated /= standard_deviation
    image = np.full(params.shape, -1000.0, dtype=np.float64)
    image[body_mask] = (
        35.0
        + texture[body_mask]
        + params.body_texture_std_hu * correlated[body_mask]
    )
    fat_shell = body_mask & (body_level >= 0.82)
    image[fat_shell] = (
        -100.0
        + 0.5 * texture[fat_shell]
        + params.body_texture_std_hu * correlated[fat_shell]
    )
    image[lung_mask] = (
        -820.0
        + 1.5 * texture[lung_mask]
        + params.lung_texture_std_hu * correlated[lung_mask]
    )
    return np.ascontiguousarray(np.rint(image).astype(np.int16))


def make_contact_svf_phantom_pair(
    params: ContactSVFPhantomParams,
) -> ContactSVFPhantomPair:
    """Create a pair whose regional flows share one analytic target interface."""
    relative, center = _relative_grid(params.shape, params.spacing_mm)
    lung_level = _ellipsoid_level(relative, params.lung_radii_mm)
    body_level = _ellipsoid_level(relative, params.body_radii_mm)
    fixed_lung = np.ascontiguousarray(lung_level <= 1.0)
    fixed_body = np.ascontiguousarray(body_level <= 1.0)
    fixed_image = _fixed_image(
        params,
        relative,
        fixed_lung,
        fixed_body,
        body_level,
    )
    lung_velocity, analytic_lung = _motion_fields(
        params,
        relative,
        params.lung_motion_rad,
    )
    body_velocity, analytic_body = _motion_fields(
        params,
        relative,
        params.body_motion_rad,
    )
    forward = exponentiate_piecewise_svf(
        lung_velocity,
        body_velocity,
        params.spacing_mm,
        squaring_steps=params.squaring_steps,
        outside_mode=params.outside_mode,
    )
    inverse = exponentiate_piecewise_svf(
        -lung_velocity,
        -body_velocity,
        params.spacing_mm,
        squaring_steps=params.squaring_steps,
        outside_mode=params.outside_mode,
        transform_direction="moving-inspiration_to_fixed-expiration",
    )
    target_lung_radii = cast(
        tuple[float, float, float],
        tuple(
            float(radius * math.exp(scale))
            for radius, scale in zip(
                params.lung_radii_mm,
                params.common_log_scales,
                strict=True,
            )
        ),
    )
    moving_lung = np.ascontiguousarray(
        _ellipsoid_level(relative, target_lung_radii) <= 1.0
    )
    target_body_radii = cast(
        tuple[float, float, float],
        tuple(
            float(radius * math.exp(scale))
            for radius, scale in zip(
                params.body_radii_mm,
                params.common_log_scales,
                strict=True,
            )
        ),
    )
    moving_body = np.ascontiguousarray(
        _ellipsoid_level(relative, target_body_radii) <= 1.0
    )
    lung_values = warp_image(
        cast(WallIntArray, fixed_image),
        inverse.lung_displacement_mm,
        params.spacing_mm,
    )
    body_values = warp_image(
        cast(WallIntArray, fixed_image),
        inverse.body_displacement_mm,
        params.spacing_mm,
    )
    moving_image = np.full(params.shape, -1000, dtype=np.int16)
    moving_wall = moving_body & ~moving_lung
    moving_image[moving_wall] = body_values[moving_wall]
    moving_image[moving_lung] = lung_values[moving_lung]
    normal = _ellipsoid_normal(relative, params.lung_radii_mm)
    tangent = _azimuthal_tangent(relative)
    cylindrical_level = np.sqrt(
        np.square(relative[..., 0] / params.lung_radii_mm[0])
        + np.square(relative[..., 1] / params.lung_radii_mm[1])
    )
    phantom = SlidingPhantom(
        image=fixed_image,
        body_mask=fixed_body,
        lung_mask=fixed_lung,
        lung_displacement_mm=forward.lung_displacement_mm,
        body_displacement_mm=forward.body_displacement_mm,
        interface_normal=np.ascontiguousarray(normal),
        interface_tangent=np.ascontiguousarray(tangent),
        tangential_ramp=np.ascontiguousarray(np.clip(cylindrical_level, 0.0, 1.0)),
        lung_level_set=np.ascontiguousarray(lung_level),
        body_level_set=np.ascontiguousarray(body_level),
        spacing_mm=params.spacing_mm,
        interface_band_mm=params.interface_band_mm,
        jacobian_margin_mm=params.jacobian_margin_mm,
        lung_radii_mm=params.lung_radii_mm,
        body_radii_mm=params.body_radii_mm,
    )
    pair = SlidingPhantomPair(
        phantom=phantom,
        fixed_image=fixed_image,
        moving_image=np.ascontiguousarray(moving_image),
        fixed_lung_mask=fixed_lung,
        moving_lung_mask=moving_lung,
        fixed_body_mask=fixed_body,
        moving_body_mask=moving_body,
        lung_moving_to_fixed_mm=inverse.lung_displacement_mm,
        body_moving_to_fixed_mm=inverse.body_displacement_mm,
    )
    return ContactSVFPhantomPair(
        pair=pair,
        lung_velocity_mm=lung_velocity,
        body_velocity_mm=body_velocity,
        analytic_lung_displacement_mm=analytic_lung,
        analytic_body_displacement_mm=analytic_body,
        target_lung_radii_mm=target_lung_radii,
        center_mm=cast(tuple[float, float, float], tuple(center.tolist())),
    )


def sample_fixed_interface_points(
    params: ContactSVFPhantomParams,
    *,
    polar_count: int = 49,
    azimuth_count: int = 128,
) -> FloatArray:
    """Sample deterministic off-pole points on the analytic fixed interface."""
    if polar_count < 3 or azimuth_count < 4:
        raise ValueError("surface sample counts are too small")
    theta = (np.arange(polar_count, dtype=np.float64) + 0.5) * math.pi / polar_count
    phi = np.arange(azimuth_count, dtype=np.float64) * 2.0 * math.pi / azimuth_count
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    radii = np.asarray(params.lung_radii_mm, dtype=np.float64)
    relative = np.stack(
        (
            radii[0] * np.sin(theta_grid) * np.cos(phi_grid),
            radii[1] * np.sin(theta_grid) * np.sin(phi_grid),
            radii[2] * np.cos(theta_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)
    center = (
        (np.asarray(params.shape, dtype=np.float64) - 1.0)
        * np.asarray(params.spacing_mm, dtype=np.float64)
        / 2.0
    )
    return np.ascontiguousarray(relative + center)


def analytic_target_signed_distance(
    points_mm: FloatArray,
    *,
    center_mm: tuple[float, float, float],
    target_radii_mm: tuple[float, float, float],
) -> FloatArray:
    """First-order physical signed distance; exact zero on the target ellipsoid."""
    relative = points_mm - np.asarray(center_mm, dtype=np.float64)
    radii = np.asarray(target_radii_mm, dtype=np.float64)
    rho = np.sqrt(np.sum(np.square(relative / radii), axis=1))
    gradient = relative / np.square(radii)
    gradient = np.divide(
        gradient,
        rho[:, None],
        out=np.zeros_like(gradient),
        where=rho[:, None] > np.finfo(np.float64).eps,
    )
    norm = np.linalg.norm(gradient, axis=1)
    return cast(
        FloatArray,
        np.divide(
            rho - 1.0,
            norm,
            out=np.zeros_like(rho),
            where=norm > np.finfo(np.float64).eps,
        ),
    )


__all__ = [
    "ContactSVFPhantomPair",
    "ContactSVFPhantomParams",
    "ContactSVFSuite",
    "ContactSVFSuiteCase",
    "analytic_target_signed_distance",
    "load_contact_svf_suite",
    "make_contact_svf_phantom_pair",
    "sample_fixed_interface_points",
]
