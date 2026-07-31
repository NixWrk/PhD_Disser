"""Frozen curved-interface representation gate for piecewise-SVF J1.1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from breathgeom.measure.piecewise_svf import exponentiate_piecewise_svf
from breathgeom.measure.registration import (
    BoolArray,
    FloatArray,
    VectorArray,
    displacement_round_trip_metrics,
    jacobian_metrics,
    transform_points,
)

CaseRole = Literal[
    "positive",
    "negative_global_glued",
    "negative_posthoc_fixed_normal",
]
CaseKind = Literal[
    "axisymmetric_affine_rotation",
    "axisymmetric_longitudinal_twist",
    "global_glued_from_case",
    "fixed_normal_projected_euler_rotation",
]
MotionKind = Literal[
    "axisymmetric_affine_rotation",
    "axisymmetric_longitudinal_twist",
]
OutsideMode = Literal["zero", "nearest"]
CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
FloatVectorArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class J11Interface:
    """Analytic ellipsoidal interface and sampling resolution."""

    radii_mm: tuple[float, float, float]
    outer_body_radii_mm: tuple[float, float, float]
    polar_sample_count: int
    azimuth_sample_count: int


@dataclass(frozen=True)
class J11Thresholds:
    """Frozen region, contact, coverage and control thresholds."""

    endpoint_p95_max_mm: float
    round_trip_p95_max_mm: float
    jacobian_p01_abs_error_max: float
    nonpositive_jacobian_fraction_max: float
    target_surface_abs_p95_max_mm: float
    target_surface_coverage_min: float
    interregional_surface_p95_max_mm: float
    interregional_surface_coverage_min: float
    gap_fraction_max: float
    collision_fraction_max: float
    tangential_slip_error_max_mm: float
    fixed_normal_control_p95_max_mm: float


@dataclass(frozen=True)
class J11Case:
    """One positive representation case or signature-bearing negative control."""

    case_id: str
    role: CaseRole
    kind: CaseKind
    common_log_scales: tuple[float, float, float] | None = None
    lung_motion: float | None = None
    body_motion: float | None = None
    source_case_id: str | None = None
    rotation_rule: str | None = None


@dataclass(frozen=True)
class J11Config:
    """Strictly parsed frozen J1.1 challenge."""

    protocol_version: str
    transform_direction: str
    coordinate_system: str
    field_units: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    squaring_steps: int
    outside_mode: OutsideMode
    evaluation_margin_voxels: tuple[int, int, int]
    interface: J11Interface
    cases: tuple[J11Case, ...]
    thresholds: J11Thresholds


@dataclass(frozen=True)
class J11Record:
    """One case-level row with independent region and advected-surface QC."""

    protocol_version: str
    case_id: str
    role: str
    kind: str
    transform_direction: str
    coordinate_system: str
    lung_voxel_count: int
    body_voxel_count: int
    lung_endpoint_p95_mm: float
    body_endpoint_p95_mm: float
    lung_round_trip_p95_mm: float
    body_round_trip_p95_mm: float
    expected_jacobian: float
    lung_jacobian_p01: float
    body_jacobian_p01: float
    lung_jacobian_p01_abs_error: float
    body_jacobian_p01_abs_error: float
    lung_nonpositive_jacobian_fraction: float
    body_nonpositive_jacobian_fraction: float
    fixed_normal_mismatch_p95_mm: float
    lung_target_surface_abs_p95_mm: float
    body_target_surface_abs_p95_mm: float
    target_surface_coverage_min: float
    interregional_surface_p95_mm: float
    interregional_surface_coverage_min: float
    gap_fraction: float
    collision_fraction: float
    tangential_slip_truth_median_mm: float
    tangential_slip_observed_median_mm: float
    tangential_slip_error_mm: float
    endpoint_pass: bool
    round_trip_pass: bool
    jacobian_accuracy_pass: bool
    topology_pass: bool
    contact_pass: bool
    slip_pass: bool
    candidate_full_gate_pass: bool
    candidate_gate_reasons: tuple[str, ...]
    expected_signature_pass: bool
    expected_signature_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | int | float | bool]:
        """Flatten tuple fields for a portable batch summary."""
        values = asdict(self)
        values["candidate_gate_reasons"] = ";".join(self.candidate_gate_reasons)
        values["expected_signature_reasons"] = ";".join(
            self.expected_signature_reasons
        )
        return cast(dict[str, str | int | float | bool], values)


@dataclass(frozen=True)
class J11Run:
    """Fields, advected surfaces and signed contact residuals for one case."""

    case: J11Case
    record: J11Record
    lung_velocity_mm: VectorArray
    body_velocity_mm: VectorArray
    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    truth_lung_displacement_mm: FloatVectorArray
    truth_body_displacement_mm: FloatVectorArray
    fixed_surface_points_mm: FloatArray
    advected_lung_surface_points_mm: FloatArray
    advected_body_surface_points_mm: FloatArray
    truth_lung_surface_points_mm: FloatArray
    truth_body_surface_points_mm: FloatArray
    lung_target_signed_distance_mm: FloatArray
    body_target_signed_distance_mm: FloatArray


def _triple(
    value: object,
    *,
    key: str,
    caster: type[int] | type[float],
) -> tuple[Any, Any, Any]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must be a JSON array of length 3")
    return cast(tuple[Any, Any, Any], tuple(caster(item) for item in value))


def _finite_float(payload: dict[str, Any], key: str) -> float:
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _bounded_float(
    payload: dict[str, Any],
    key: str,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool,
) -> float:
    value = _finite_float(payload, key)
    lower_ok = value >= lower if lower_inclusive else value > lower
    if not lower_ok or value > upper:
        opening = "[" if lower_inclusive else "("
        raise ValueError(f"{key} must lie in {opening}{lower}, {upper}]")
    return value


def _load_interface(payload: object) -> J11Interface:
    expected = {
        "radii_mm",
        "outer_body_radii_mm",
        "polar_sample_count",
        "azimuth_sample_count",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("interface contains missing or unknown keys")
    values = cast(dict[str, Any], payload)
    radii = cast(
        tuple[float, float, float],
        _triple(values["radii_mm"], key="radii_mm", caster=float),
    )
    outer = cast(
        tuple[float, float, float],
        _triple(
            values["outer_body_radii_mm"],
            key="outer_body_radii_mm",
            caster=float,
        ),
    )
    if any(not math.isfinite(value) or value <= 0 for value in radii + outer):
        raise ValueError("interface radii must be finite and positive")
    if any(outer_value <= inner for outer_value, inner in zip(outer, radii, strict=True)):
        raise ValueError("outer body radii must exceed interface radii")
    polar = int(values["polar_sample_count"])
    azimuth = int(values["azimuth_sample_count"])
    if polar < 8 or azimuth < 16:
        raise ValueError("surface sampling is too sparse")
    return J11Interface(
        radii_mm=radii,
        outer_body_radii_mm=outer,
        polar_sample_count=polar,
        azimuth_sample_count=azimuth,
    )


def _load_thresholds(payload: object) -> J11Thresholds:
    expected = {
        "endpoint_p95_max_mm",
        "round_trip_p95_max_mm",
        "jacobian_p01_abs_error_max",
        "nonpositive_jacobian_fraction_max",
        "target_surface_abs_p95_max_mm",
        "target_surface_coverage_min",
        "interregional_surface_p95_max_mm",
        "interregional_surface_coverage_min",
        "gap_fraction_max",
        "collision_fraction_max",
        "tangential_slip_error_max_mm",
        "fixed_normal_control_p95_max_mm",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("thresholds contain missing or unknown keys")
    values = cast(dict[str, Any], payload)
    return J11Thresholds(
        endpoint_p95_max_mm=_bounded_float(
            values,
            "endpoint_p95_max_mm",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
        round_trip_p95_max_mm=_bounded_float(
            values,
            "round_trip_p95_max_mm",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
        jacobian_p01_abs_error_max=_bounded_float(
            values,
            "jacobian_p01_abs_error_max",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        ),
        nonpositive_jacobian_fraction_max=_bounded_float(
            values,
            "nonpositive_jacobian_fraction_max",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
        ),
        target_surface_abs_p95_max_mm=_bounded_float(
            values,
            "target_surface_abs_p95_max_mm",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
        target_surface_coverage_min=_bounded_float(
            values,
            "target_surface_coverage_min",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        ),
        interregional_surface_p95_max_mm=_bounded_float(
            values,
            "interregional_surface_p95_max_mm",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
        interregional_surface_coverage_min=_bounded_float(
            values,
            "interregional_surface_coverage_min",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        ),
        gap_fraction_max=_bounded_float(
            values,
            "gap_fraction_max",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
        ),
        collision_fraction_max=_bounded_float(
            values,
            "collision_fraction_max",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
        ),
        tangential_slip_error_max_mm=_bounded_float(
            values,
            "tangential_slip_error_max_mm",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
        fixed_normal_control_p95_max_mm=_bounded_float(
            values,
            "fixed_normal_control_p95_max_mm",
            lower=0.0,
            upper=10.0,
            lower_inclusive=False,
        ),
    )


def _load_case(payload: object) -> J11Case:
    if not isinstance(payload, dict):
        raise ValueError("each case must be a JSON object")
    values = cast(dict[str, Any], payload)
    case_id = values.get("case_id")
    role = values.get("role")
    kind = values.get("kind")
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"invalid case_id: {case_id!r}")
    positive_kinds = {
        "axisymmetric_affine_rotation",
        "axisymmetric_longitudinal_twist",
    }
    if role == "positive" and kind in positive_kinds:
        motion_keys = (
            {"lung_rotation_rad", "body_rotation_rad"}
            if kind == "axisymmetric_affine_rotation"
            else {
                "lung_twist_rad_at_positive_pole",
                "body_twist_rad_at_positive_pole",
            }
        )
        expected = {"case_id", "role", "kind", "common_log_scales"} | motion_keys
        if set(values) != expected:
            raise ValueError(f"{case_id}: positive case contains invalid keys")
        lung_key, body_key = tuple(sorted(motion_keys))
        if lung_key.startswith("body"):
            lung_key, body_key = body_key, lung_key
        return J11Case(
            case_id=case_id,
            role="positive",
            kind=cast(CaseKind, kind),
            common_log_scales=cast(
                tuple[float, float, float],
                _triple(
                    values["common_log_scales"],
                    key="common_log_scales",
                    caster=float,
                ),
            ),
            lung_motion=_finite_float(values, lung_key),
            body_motion=_finite_float(values, body_key),
        )
    if role == "negative_global_glued" and kind == "global_glued_from_case":
        expected = {
            "case_id",
            "role",
            "kind",
            "source_case_id",
            "rotation_rule",
        }
        if set(values) != expected:
            raise ValueError(f"{case_id}: global control contains invalid keys")
        source = values["source_case_id"]
        if not isinstance(source, str) or not CASE_ID_PATTERN.fullmatch(source):
            raise ValueError(f"{case_id}: invalid source_case_id")
        if values["rotation_rule"] != "mean":
            raise ValueError(f"{case_id}: only the frozen mean rule is supported")
        return J11Case(
            case_id=case_id,
            role="negative_global_glued",
            kind="global_glued_from_case",
            source_case_id=source,
            rotation_rule="mean",
        )
    if (
        role == "negative_posthoc_fixed_normal"
        and kind == "fixed_normal_projected_euler_rotation"
    ):
        expected = {"case_id", "role", "kind", "rotation_rad"}
        if set(values) != expected:
            raise ValueError(f"{case_id}: posthoc control contains invalid keys")
        return J11Case(
            case_id=case_id,
            role="negative_posthoc_fixed_normal",
            kind="fixed_normal_projected_euler_rotation",
            lung_motion=_finite_float(values, "rotation_rad"),
            body_motion=0.0,
            common_log_scales=(0.0, 0.0, 0.0),
        )
    raise ValueError(f"{case_id}: unsupported role/kind combination")


def load_j11_config(path: Path) -> J11Config:
    """Load the exact J1.1 schema and validate control references."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_version",
        "transform_direction",
        "coordinate_system",
        "field_units",
        "shape",
        "spacing_mm",
        "squaring_steps",
        "outside_mode",
        "evaluation_margin_voxels",
        "interface",
        "cases",
        "thresholds",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("J1.1 config contains missing or unknown top-level keys")
    for key in (
        "protocol_version",
        "transform_direction",
        "coordinate_system",
        "field_units",
    ):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"{key} must be a non-empty string")
    shape = cast(
        tuple[int, int, int],
        _triple(payload["shape"], key="shape", caster=int),
    )
    spacing = cast(
        tuple[float, float, float],
        _triple(payload["spacing_mm"], key="spacing_mm", caster=float),
    )
    margin = cast(
        tuple[int, int, int],
        _triple(
            payload["evaluation_margin_voxels"],
            key="evaluation_margin_voxels",
            caster=int,
        ),
    )
    if any(value <= 0 for value in shape):
        raise ValueError("shape values must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("spacing values must be finite and positive")
    if any(value < 1 for value in margin):
        raise ValueError("evaluation margins must be positive")
    if any(2 * border + 3 > size for border, size in zip(margin, shape, strict=True)):
        raise ValueError("evaluation margin leaves no derivative-safe interior")
    squaring_steps = int(payload["squaring_steps"])
    if squaring_steps < 0:
        raise ValueError("squaring_steps must be non-negative")
    outside_mode = payload["outside_mode"]
    if outside_mode not in {"zero", "nearest"}:
        raise ValueError("outside_mode must be 'zero' or 'nearest'")
    cases_payload = payload["cases"]
    if not isinstance(cases_payload, list) or not cases_payload:
        raise ValueError("cases must be a non-empty JSON array")
    cases = tuple(_load_case(item) for item in cases_payload)
    cases_by_id = {case.case_id: case for case in cases}
    if len(cases_by_id) != len(cases):
        raise ValueError("case_id values must be unique")
    for case in cases:
        if case.role == "negative_global_glued":
            source = cases_by_id.get(case.source_case_id or "")
            if (
                source is None
                or source.role != "positive"
                or source.kind != "axisymmetric_affine_rotation"
            ):
                raise ValueError(
                    f"{case.case_id}: source must be a positive affine rotation"
                )
    return J11Config(
        protocol_version=payload["protocol_version"],
        transform_direction=payload["transform_direction"],
        coordinate_system=payload["coordinate_system"],
        field_units=payload["field_units"],
        shape=shape,
        spacing_mm=spacing,
        squaring_steps=squaring_steps,
        outside_mode=cast(OutsideMode, outside_mode),
        evaluation_margin_voxels=margin,
        interface=_load_interface(payload["interface"]),
        cases=cases,
        thresholds=_load_thresholds(payload["thresholds"]),
    )


def _grid_and_center(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> tuple[FloatVectorArray, npt.NDArray[np.float64]]:
    axes = [
        np.arange(size, dtype=np.float64) * step
        for size, step in zip(shape, spacing, strict=True)
    ]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    center = np.asarray(
        [(size - 1) * step / 2.0 for size, step in zip(shape, spacing, strict=True)],
        dtype=np.float64,
    )
    return coordinates - center, center


def _surface_points_and_normals(
    interface: J11Interface,
    center: npt.NDArray[np.float64],
) -> tuple[FloatArray, FloatArray]:
    theta = (
        np.arange(interface.polar_sample_count, dtype=np.float64) + 0.5
    ) * math.pi / interface.polar_sample_count
    phi = (
        np.arange(interface.azimuth_sample_count, dtype=np.float64)
        * 2.0
        * math.pi
        / interface.azimuth_sample_count
    )
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    radii = np.asarray(interface.radii_mm, dtype=np.float64)
    relative = np.stack(
        (
            radii[0] * np.sin(theta_grid) * np.cos(phi_grid),
            radii[1] * np.sin(theta_grid) * np.sin(phi_grid),
            radii[2] * np.cos(theta_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)
    gradient = relative / np.square(radii)
    normals = gradient / np.linalg.norm(gradient, axis=1, keepdims=True)
    return relative + center, normals


def _domains(
    config: J11Config,
    relative_grid_mm: FloatVectorArray,
) -> tuple[BoolArray, BoolArray]:
    radii = np.asarray(config.interface.radii_mm, dtype=np.float64)
    outer = np.asarray(config.interface.outer_body_radii_mm, dtype=np.float64)
    lung = np.sum(np.square(relative_grid_mm / radii), axis=-1) <= 1.0
    outer_body = np.sum(np.square(relative_grid_mm / outer), axis=-1) <= 1.0
    interior = np.zeros(config.shape, dtype=bool)
    slices = tuple(
        slice(border, size - border)
        for border, size in zip(
            config.evaluation_margin_voxels,
            config.shape,
            strict=True,
        )
    )
    interior[slices] = True
    return lung & interior, outer_body & ~lung & interior


def _motion_fields(
    *,
    kind: MotionKind,
    log_scales: tuple[float, float, float],
    motion: float,
    relative_grid_mm: FloatVectorArray,
    positive_pole_mm: float,
) -> tuple[VectorArray, FloatVectorArray, float]:
    scales = np.asarray(log_scales, dtype=np.float64)
    if abs(scales[0] - scales[1]) > 1e-12:
        raise ValueError("axisymmetric motion requires equal X/Y log scales")
    x = relative_grid_mm[..., 0]
    y = relative_grid_mm[..., 1]
    z = relative_grid_mm[..., 2]
    if kind == "axisymmetric_affine_rotation":
        angular_velocity = np.full(x.shape, motion, dtype=np.float64)
        final_angle = angular_velocity
    else:
        angular_velocity = motion * z / positive_pole_mm
        if abs(scales[2]) < 1e-12:
            final_angle = angular_velocity
        else:
            final_angle = angular_velocity * math.expm1(scales[2]) / scales[2]
    velocity = np.zeros(relative_grid_mm.shape, dtype=np.float32)
    velocity[..., 0] = (scales[0] * x - angular_velocity * y).astype(np.float32)
    velocity[..., 1] = (angular_velocity * x + scales[1] * y).astype(np.float32)
    velocity[..., 2] = (scales[2] * z).astype(np.float32)
    exp_xy = math.exp(scales[0])
    cosine = np.cos(final_angle)
    sine = np.sin(final_angle)
    transformed_x = exp_xy * (cosine * x - sine * y)
    transformed_y = exp_xy * (sine * x + cosine * y)
    transformed_z = math.exp(scales[2]) * z
    truth = np.stack(
        (
            transformed_x - x,
            transformed_y - y,
            transformed_z - z,
        ),
        axis=-1,
    )
    return (
        velocity,
        truth,
        float(math.exp(2.0 * scales[0] + scales[2])),
    )


def _analytic_surface(
    *,
    kind: MotionKind,
    log_scales: tuple[float, float, float],
    motion: float,
    fixed_points_mm: FloatArray,
    center: npt.NDArray[np.float64],
    positive_pole_mm: float,
) -> FloatArray:
    relative = fixed_points_mm - center
    scales = np.asarray(log_scales, dtype=np.float64)
    if kind == "axisymmetric_affine_rotation":
        angle = np.full(len(relative), motion, dtype=np.float64)
    elif abs(scales[2]) < 1e-12:
        angle = motion * relative[:, 2] / positive_pole_mm
    else:
        angle = (
            motion
            * relative[:, 2]
            / positive_pole_mm
            * math.expm1(scales[2])
            / scales[2]
        )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    exp_xy = math.exp(scales[0])
    transformed = np.empty_like(relative)
    transformed[:, 0] = exp_xy * (
        cosine * relative[:, 0] - sine * relative[:, 1]
    )
    transformed[:, 1] = exp_xy * (
        sine * relative[:, 0] + cosine * relative[:, 1]
    )
    transformed[:, 2] = math.exp(scales[2]) * relative[:, 2]
    return transformed + center


def _projected_euler_field(
    relative_grid_mm: FloatVectorArray,
    radii_mm: tuple[float, float, float],
    rotation_rad: float,
) -> VectorArray:
    radii = np.asarray(radii_mm, dtype=np.float64)
    gradient = relative_grid_mm / np.square(radii)
    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    normals = np.divide(
        gradient,
        norm,
        out=np.zeros_like(gradient),
        where=norm > 0,
    )
    rotation = np.zeros(relative_grid_mm.shape, dtype=np.float64)
    rotation[..., 0] = -rotation_rad * relative_grid_mm[..., 1]
    rotation[..., 1] = rotation_rad * relative_grid_mm[..., 0]
    normal_component = np.sum(rotation * normals, axis=-1, keepdims=True)
    return np.ascontiguousarray((rotation - normal_component * normals).astype(np.float32))


def _implicit_signed_distance(
    points_mm: FloatArray,
    *,
    center: npt.NDArray[np.float64],
    target_radii_mm: npt.NDArray[np.float64],
) -> FloatArray:
    relative = points_mm - center
    rho = np.sqrt(np.sum(np.square(relative / target_radii_mm), axis=1))
    gradient = relative / np.square(target_radii_mm)
    gradient /= rho[:, None]
    gradient_norm = np.linalg.norm(gradient, axis=1)
    return cast(FloatArray, (rho - 1.0) / gradient_norm)


def _symmetric_surface_metrics(
    lung_points_mm: FloatArray,
    body_points_mm: FloatArray,
    tolerance_mm: float,
) -> tuple[float, float]:
    lung_to_body = cKDTree(body_points_mm).query(lung_points_mm, k=1)[0]
    body_to_lung = cKDTree(lung_points_mm).query(body_points_mm, k=1)[0]
    p95 = max(
        float(np.percentile(lung_to_body, 95)),
        float(np.percentile(body_to_lung, 95)),
    )
    coverage = min(
        float(np.mean(lung_to_body <= tolerance_mm)),
        float(np.mean(body_to_lung <= tolerance_mm)),
    )
    return p95, coverage


def _full_gate_reasons(
    config: J11Config,
    *,
    lung_endpoint_p95: float,
    body_endpoint_p95: float,
    lung_round_trip_p95: float,
    body_round_trip_p95: float,
    lung_jacobian_error: float,
    body_jacobian_error: float,
    lung_folding: float,
    body_folding: float,
    lung_target_p95: float,
    body_target_p95: float,
    target_coverage: float,
    interregional_p95: float,
    interregional_coverage: float,
    gap_fraction: float,
    collision_fraction: float,
    slip_error: float,
) -> tuple[str, ...]:
    threshold = config.thresholds
    reasons: list[str] = []
    if max(lung_endpoint_p95, body_endpoint_p95) > threshold.endpoint_p95_max_mm:
        reasons.append("endpoint_p95")
    if (
        max(lung_round_trip_p95, body_round_trip_p95)
        > threshold.round_trip_p95_max_mm
    ):
        reasons.append("round_trip_p95")
    if (
        max(lung_jacobian_error, body_jacobian_error)
        > threshold.jacobian_p01_abs_error_max
    ):
        reasons.append("jacobian_p01")
    if (
        max(lung_folding, body_folding)
        > threshold.nonpositive_jacobian_fraction_max
    ):
        reasons.append("folding")
    if max(lung_target_p95, body_target_p95) > threshold.target_surface_abs_p95_max_mm:
        reasons.append("target_surface_p95")
    if target_coverage < threshold.target_surface_coverage_min:
        reasons.append("target_surface_coverage")
    if interregional_p95 > threshold.interregional_surface_p95_max_mm:
        reasons.append("interregional_surface_p95")
    if interregional_coverage < threshold.interregional_surface_coverage_min:
        reasons.append("interregional_surface_coverage")
    if gap_fraction > threshold.gap_fraction_max:
        reasons.append("gap")
    if collision_fraction > threshold.collision_fraction_max:
        reasons.append("collision")
    if slip_error > threshold.tangential_slip_error_max_mm:
        reasons.append("tangential_slip")
    return tuple(reasons)


def _evaluate_case(
    *,
    config: J11Config,
    case: J11Case,
    lung_domain: BoolArray,
    body_domain: BoolArray,
    fixed_surface_points_mm: FloatArray,
    fixed_normals: FloatArray,
    center: npt.NDArray[np.float64],
    log_scales: tuple[float, float, float],
    expected_jacobian: float,
    lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    inverse_lung_displacement_mm: VectorArray,
    inverse_body_displacement_mm: VectorArray,
    truth_lung_displacement_mm: FloatVectorArray,
    truth_body_displacement_mm: FloatVectorArray,
    truth_lung_surface_points_mm: FloatArray,
    truth_body_surface_points_mm: FloatArray,
) -> tuple[J11Record, dict[str, FloatArray]]:
    lung_error = np.linalg.norm(
        lung_displacement_mm.astype(np.float64) - truth_lung_displacement_mm,
        axis=-1,
    )[lung_domain]
    body_error = np.linalg.norm(
        body_displacement_mm.astype(np.float64) - truth_body_displacement_mm,
        axis=-1,
    )[body_domain]
    lung_endpoint_p95 = float(np.percentile(lung_error, 95))
    body_endpoint_p95 = float(np.percentile(body_error, 95))
    lung_round_trip = displacement_round_trip_metrics(
        lung_displacement_mm,
        inverse_lung_displacement_mm,
        config.spacing_mm,
        valid_domain=lung_domain,
    )
    body_round_trip = displacement_round_trip_metrics(
        body_displacement_mm,
        inverse_body_displacement_mm,
        config.spacing_mm,
        valid_domain=body_domain,
    )
    lung_jacobian = jacobian_metrics(
        lung_displacement_mm,
        config.spacing_mm,
        valid_domain=lung_domain,
    )
    body_jacobian = jacobian_metrics(
        body_displacement_mm,
        config.spacing_mm,
        valid_domain=body_domain,
    )
    lung_jacobian_error = abs(lung_jacobian.p01 - expected_jacobian)
    body_jacobian_error = abs(body_jacobian.p01 - expected_jacobian)
    advected_lung = transform_points(
        fixed_surface_points_mm,
        lung_displacement_mm,
        config.spacing_mm,
    )
    advected_body = transform_points(
        fixed_surface_points_mm,
        body_displacement_mm,
        config.spacing_mm,
    )
    target_radii = (
        np.asarray(config.interface.radii_mm, dtype=np.float64)
        * np.exp(np.asarray(log_scales, dtype=np.float64))
    )
    lung_signed = _implicit_signed_distance(
        advected_lung,
        center=center,
        target_radii_mm=target_radii,
    )
    body_signed = _implicit_signed_distance(
        advected_body,
        center=center,
        target_radii_mm=target_radii,
    )
    tolerance = config.thresholds.target_surface_abs_p95_max_mm
    lung_target_p95 = float(np.percentile(np.abs(lung_signed), 95))
    body_target_p95 = float(np.percentile(np.abs(body_signed), 95))
    target_coverage = min(
        float(np.mean(np.abs(lung_signed) <= tolerance)),
        float(np.mean(np.abs(body_signed) <= tolerance)),
    )
    interregional_p95, interregional_coverage = _symmetric_surface_metrics(
        advected_lung,
        advected_body,
        config.thresholds.interregional_surface_p95_max_mm,
    )
    all_signed = np.concatenate((lung_signed, body_signed))
    gap_fraction = float(np.mean(all_signed < -tolerance))
    collision_fraction = float(np.mean(all_signed > tolerance))
    observed_delta = advected_lung - advected_body
    truth_delta = truth_lung_surface_points_mm - truth_body_surface_points_mm
    observed_slip = np.linalg.norm(observed_delta, axis=1)
    truth_slip = np.linalg.norm(truth_delta, axis=1)
    observed_slip_median = float(np.median(observed_slip))
    truth_slip_median = float(np.median(truth_slip))
    slip_error = abs(observed_slip_median - truth_slip_median)
    lung_surface_displacement = advected_lung - fixed_surface_points_mm
    body_surface_displacement = advected_body - fixed_surface_points_mm
    fixed_normal = np.sum(
        (lung_surface_displacement - body_surface_displacement) * fixed_normals,
        axis=1,
    )
    fixed_normal_p95 = float(np.percentile(np.abs(fixed_normal), 95))
    reasons = _full_gate_reasons(
        config,
        lung_endpoint_p95=lung_endpoint_p95,
        body_endpoint_p95=body_endpoint_p95,
        lung_round_trip_p95=lung_round_trip.p95_mm,
        body_round_trip_p95=body_round_trip.p95_mm,
        lung_jacobian_error=lung_jacobian_error,
        body_jacobian_error=body_jacobian_error,
        lung_folding=lung_jacobian.nonpositive_fraction,
        body_folding=body_jacobian.nonpositive_fraction,
        lung_target_p95=lung_target_p95,
        body_target_p95=body_target_p95,
        target_coverage=target_coverage,
        interregional_p95=interregional_p95,
        interregional_coverage=interregional_coverage,
        gap_fraction=gap_fraction,
        collision_fraction=collision_fraction,
        slip_error=slip_error,
    )
    endpoint_pass = "endpoint_p95" not in reasons
    round_trip_pass = "round_trip_p95" not in reasons
    jacobian_accuracy_pass = "jacobian_p01" not in reasons
    topology_pass = "folding" not in reasons
    contact_reasons = {
        "target_surface_p95",
        "target_surface_coverage",
        "interregional_surface_p95",
        "interregional_surface_coverage",
        "gap",
        "collision",
    }
    contact_pass = not bool(contact_reasons.intersection(reasons))
    slip_pass = "tangential_slip" not in reasons
    full_pass = not reasons
    signature_reasons: list[str] = []
    if case.role == "positive":
        if not full_pass:
            signature_reasons.append("positive_candidate_failed")
    elif case.role == "negative_global_glued":
        if not topology_pass:
            signature_reasons.append("global_control_topology_failed")
        if not contact_pass:
            signature_reasons.append("global_control_contact_failed")
        if slip_pass:
            signature_reasons.append("global_control_did_not_lose_slip")
    else:
        if fixed_normal_p95 > config.thresholds.fixed_normal_control_p95_max_mm:
            signature_reasons.append("fixed_normal_screen_failed")
        if not topology_pass:
            signature_reasons.append("posthoc_control_topology_failed")
        if contact_pass:
            signature_reasons.append("posthoc_control_contact_unexpectedly_passed")
        geometric_failure = (
            gap_fraction > config.thresholds.gap_fraction_max
            or collision_fraction > config.thresholds.collision_fraction_max
            or target_coverage < config.thresholds.target_surface_coverage_min
        )
        if not geometric_failure:
            signature_reasons.append("posthoc_geometric_failure_not_detected")
    record = J11Record(
        protocol_version=config.protocol_version,
        case_id=case.case_id,
        role=case.role,
        kind=case.kind,
        transform_direction=config.transform_direction,
        coordinate_system=config.coordinate_system,
        lung_voxel_count=int(np.count_nonzero(lung_domain)),
        body_voxel_count=int(np.count_nonzero(body_domain)),
        lung_endpoint_p95_mm=lung_endpoint_p95,
        body_endpoint_p95_mm=body_endpoint_p95,
        lung_round_trip_p95_mm=lung_round_trip.p95_mm,
        body_round_trip_p95_mm=body_round_trip.p95_mm,
        expected_jacobian=expected_jacobian,
        lung_jacobian_p01=lung_jacobian.p01,
        body_jacobian_p01=body_jacobian.p01,
        lung_jacobian_p01_abs_error=lung_jacobian_error,
        body_jacobian_p01_abs_error=body_jacobian_error,
        lung_nonpositive_jacobian_fraction=lung_jacobian.nonpositive_fraction,
        body_nonpositive_jacobian_fraction=body_jacobian.nonpositive_fraction,
        fixed_normal_mismatch_p95_mm=fixed_normal_p95,
        lung_target_surface_abs_p95_mm=lung_target_p95,
        body_target_surface_abs_p95_mm=body_target_p95,
        target_surface_coverage_min=target_coverage,
        interregional_surface_p95_mm=interregional_p95,
        interregional_surface_coverage_min=interregional_coverage,
        gap_fraction=gap_fraction,
        collision_fraction=collision_fraction,
        tangential_slip_truth_median_mm=truth_slip_median,
        tangential_slip_observed_median_mm=observed_slip_median,
        tangential_slip_error_mm=slip_error,
        endpoint_pass=endpoint_pass,
        round_trip_pass=round_trip_pass,
        jacobian_accuracy_pass=jacobian_accuracy_pass,
        topology_pass=topology_pass,
        contact_pass=contact_pass,
        slip_pass=slip_pass,
        candidate_full_gate_pass=full_pass,
        candidate_gate_reasons=reasons,
        expected_signature_pass=not signature_reasons,
        expected_signature_reasons=tuple(signature_reasons),
    )
    arrays = {
        "advected_lung": advected_lung,
        "advected_body": advected_body,
        "lung_signed": lung_signed,
        "body_signed": body_signed,
    }
    return record, arrays


def run_j11_gate(config: J11Config) -> tuple[J11Run, ...]:
    """Run positive curved sliding and signature-bearing negative controls."""
    relative_grid, center = _grid_and_center(config.shape, config.spacing_mm)
    lung_domain, body_domain = _domains(config, relative_grid)
    fixed_surface, fixed_normals = _surface_points_and_normals(
        config.interface,
        center,
    )
    cases_by_id = {case.case_id: case for case in config.cases}
    runs: list[J11Run] = []
    for case in config.cases:
        truth_case = case
        if case.role == "negative_global_glued":
            truth_case = cases_by_id[case.source_case_id or ""]
        if case.role == "negative_posthoc_fixed_normal":
            truth_kind: MotionKind = "axisymmetric_affine_rotation"
            log_scales = (0.0, 0.0, 0.0)
        else:
            truth_kind = cast(MotionKind, truth_case.kind)
            if truth_case.common_log_scales is None:
                raise ValueError(f"{case.case_id}: truth scales are missing")
            log_scales = truth_case.common_log_scales
        lung_truth_motion = truth_case.lung_motion
        body_truth_motion = truth_case.body_motion
        if lung_truth_motion is None or body_truth_motion is None:
            raise ValueError(f"{case.case_id}: truth motion parameters are missing")
        lung_truth_velocity, lung_truth, expected_jacobian = _motion_fields(
            kind=truth_kind,
            log_scales=log_scales,
            motion=lung_truth_motion,
            relative_grid_mm=relative_grid,
            positive_pole_mm=config.interface.radii_mm[2],
        )
        body_truth_velocity, body_truth, body_expected_jacobian = _motion_fields(
            kind=truth_kind,
            log_scales=log_scales,
            motion=body_truth_motion,
            relative_grid_mm=relative_grid,
            positive_pole_mm=config.interface.radii_mm[2],
        )
        if abs(body_expected_jacobian - expected_jacobian) > 1e-12:
            raise ValueError(f"{case.case_id}: regional truth Jacobians differ")
        truth_lung_surface = _analytic_surface(
            kind=truth_kind,
            log_scales=log_scales,
            motion=lung_truth_motion,
            fixed_points_mm=fixed_surface,
            center=center,
            positive_pole_mm=config.interface.radii_mm[2],
        )
        truth_body_surface = _analytic_surface(
            kind=truth_kind,
            log_scales=log_scales,
            motion=body_truth_motion,
            fixed_points_mm=fixed_surface,
            center=center,
            positive_pole_mm=config.interface.radii_mm[2],
        )
        if case.role == "positive":
            lung_velocity = lung_truth_velocity
            body_velocity = body_truth_velocity
            forward = exponentiate_piecewise_svf(
                lung_velocity,
                body_velocity,
                config.spacing_mm,
                squaring_steps=config.squaring_steps,
                outside_mode=config.outside_mode,
                transform_direction=config.transform_direction,
            )
            inverse = exponentiate_piecewise_svf(
                -lung_velocity,
                -body_velocity,
                config.spacing_mm,
                squaring_steps=config.squaring_steps,
                outside_mode=config.outside_mode,
                transform_direction="moving-inspiration_to_fixed-expiration",
            )
            lung_displacement = forward.lung_displacement_mm
            body_displacement = forward.body_displacement_mm
            inverse_lung = inverse.lung_displacement_mm
            inverse_body = inverse.body_displacement_mm
        elif case.role == "negative_global_glued":
            average_motion = 0.5 * (lung_truth_motion + body_truth_motion)
            lung_velocity, _, _ = _motion_fields(
                kind=truth_kind,
                log_scales=log_scales,
                motion=average_motion,
                relative_grid_mm=relative_grid,
                positive_pole_mm=config.interface.radii_mm[2],
            )
            body_velocity = lung_velocity.copy()
            forward = exponentiate_piecewise_svf(
                lung_velocity,
                body_velocity,
                config.spacing_mm,
                squaring_steps=config.squaring_steps,
                outside_mode=config.outside_mode,
                transform_direction=config.transform_direction,
            )
            inverse = exponentiate_piecewise_svf(
                -lung_velocity,
                -body_velocity,
                config.spacing_mm,
                squaring_steps=config.squaring_steps,
                outside_mode=config.outside_mode,
                transform_direction="moving-inspiration_to_fixed-expiration",
            )
            lung_displacement = forward.lung_displacement_mm
            body_displacement = forward.body_displacement_mm
            inverse_lung = inverse.lung_displacement_mm
            inverse_body = inverse.body_displacement_mm
        else:
            lung_velocity = _projected_euler_field(
                relative_grid,
                config.interface.radii_mm,
                lung_truth_motion,
            )
            body_velocity = np.zeros_like(lung_velocity)
            lung_displacement = lung_velocity.copy()
            body_displacement = body_velocity.copy()
            inverse_lung = -lung_velocity
            inverse_body = body_velocity.copy()
        record, arrays = _evaluate_case(
            config=config,
            case=case,
            lung_domain=lung_domain,
            body_domain=body_domain,
            fixed_surface_points_mm=fixed_surface,
            fixed_normals=fixed_normals,
            center=center,
            log_scales=log_scales,
            expected_jacobian=expected_jacobian,
            lung_displacement_mm=lung_displacement,
            body_displacement_mm=body_displacement,
            inverse_lung_displacement_mm=inverse_lung,
            inverse_body_displacement_mm=inverse_body,
            truth_lung_displacement_mm=lung_truth,
            truth_body_displacement_mm=body_truth,
            truth_lung_surface_points_mm=truth_lung_surface,
            truth_body_surface_points_mm=truth_body_surface,
        )
        runs.append(
            J11Run(
                case=case,
                record=record,
                lung_velocity_mm=lung_velocity,
                body_velocity_mm=body_velocity,
                lung_displacement_mm=lung_displacement,
                body_displacement_mm=body_displacement,
                truth_lung_displacement_mm=lung_truth,
                truth_body_displacement_mm=body_truth,
                fixed_surface_points_mm=fixed_surface,
                advected_lung_surface_points_mm=arrays["advected_lung"],
                advected_body_surface_points_mm=arrays["advected_body"],
                truth_lung_surface_points_mm=truth_lung_surface,
                truth_body_surface_points_mm=truth_body_surface,
                lung_target_signed_distance_mm=arrays["lung_signed"],
                body_target_signed_distance_mm=arrays["body_signed"],
            )
        )
    return tuple(runs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _code_version(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_j11_gate(
    runs: tuple[J11Run, ...],
    output_dir: Path,
    *,
    config: J11Config,
    config_path: Path,
    repo_root: Path,
) -> Path:
    """Write local synthetic fields, surface residuals and full provenance."""
    if not runs:
        raise ValueError("cannot write an empty J1.1 gate")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run.record.as_csv_row() for run in runs]
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    field_hashes: dict[str, str] = {}
    record_hashes: dict[str, str] = {}
    for run in runs:
        field_path = output_dir / f"{run.case.case_id}.npz"
        np.savez_compressed(
            field_path,
            usage=np.asarray("synthetic_representation_diagnostic_ONLY"),
            lung_velocity_mm=run.lung_velocity_mm,
            body_velocity_mm=run.body_velocity_mm,
            lung_displacement_mm=run.lung_displacement_mm,
            body_displacement_mm=run.body_displacement_mm,
            truth_lung_displacement_mm=run.truth_lung_displacement_mm,
            truth_body_displacement_mm=run.truth_body_displacement_mm,
            fixed_surface_points_mm=run.fixed_surface_points_mm,
            advected_lung_surface_points_mm=(
                run.advected_lung_surface_points_mm
            ),
            advected_body_surface_points_mm=(
                run.advected_body_surface_points_mm
            ),
            truth_lung_surface_points_mm=run.truth_lung_surface_points_mm,
            truth_body_surface_points_mm=run.truth_body_surface_points_mm,
            lung_target_signed_distance_mm=(
                run.lung_target_signed_distance_mm
            ),
            body_target_signed_distance_mm=(
                run.body_target_signed_distance_mm
            ),
            spacing_mm=np.asarray(config.spacing_mm),
            transform_direction=np.asarray(config.transform_direction),
            coordinate_system=np.asarray(config.coordinate_system),
        )
        field_hashes[field_path.name] = _sha256(field_path)
        record_path = output_dir / f"{run.case.case_id}.json"
        record_path.write_text(
            json.dumps(
                asdict(run.record),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        record_hashes[record_path.name] = _sha256(record_path)
    manifest = {
        "artifact_type": "piecewise_svf_j11_representation_gate",
        "usage": "synthetic_representation_diagnostic_ONLY",
        "protocol_version": config.protocol_version,
        "code_version": _code_version(repo_root),
        "config": {
            "path": config_path.name,
            "sha256": _sha256(config_path),
        },
        "transform_direction": config.transform_direction,
        "coordinate_system": config.coordinate_system,
        "field_units": config.field_units,
        "thresholds": asdict(config.thresholds),
        "case_count": len(runs),
        "positive_case_count": sum(
            run.case.role == "positive" for run in runs
        ),
        "positive_pass_count": sum(
            run.case.role == "positive"
            and run.record.candidate_full_gate_pass
            for run in runs
        ),
        "negative_control_count": sum(
            run.case.role != "positive" for run in runs
        ),
        "negative_control_recognized_count": sum(
            run.case.role != "positive"
            and run.record.expected_signature_pass
            for run in runs
        ),
        "all_expected_signatures_pass": all(
            run.record.expected_signature_pass for run in runs
        ),
        "summary_sha256": _sha256(summary_path),
        "field_sha256": field_hashes,
        "record_sha256": record_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "J11Case",
    "J11Config",
    "J11Interface",
    "J11Record",
    "J11Run",
    "J11Thresholds",
    "load_j11_config",
    "run_j11_gate",
    "write_j11_gate",
]
