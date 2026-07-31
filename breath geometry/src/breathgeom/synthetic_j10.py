"""Frozen numerical-invariant batch for the J1.0 piecewise-SVF foundation."""

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

from breathgeom.measure.piecewise_svf import exponentiate_piecewise_svf
from breathgeom.measure.registration import (
    BoolArray,
    VectorArray,
    displacement_round_trip_metrics,
    jacobian_metrics,
)

CaseKind = Literal[
    "zero",
    "planar_sliding_translation",
    "diagonal_affine",
    "opposed_axial_rotation",
]
OutsideMode = Literal["zero", "nearest"]
FloatVectorArray = npt.NDArray[np.float64]
CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


@dataclass(frozen=True)
class J10Thresholds:
    """Frozen pass thresholds for numerical, not anatomical, accuracy."""

    endpoint_p95_max_mm: float
    round_trip_p95_max_mm: float
    jacobian_abs_error_max: float
    nonpositive_jacobian_fraction_max: float
    normal_mismatch_p95_max_mm: float
    tangential_slip_error_max_mm: float


@dataclass(frozen=True)
class J10Case:
    """One analytic stationary-velocity invariant."""

    case_id: str
    kind: CaseKind
    interface_normal_axis: int | None = None
    lung_vector: tuple[float, float, float] | None = None
    body_vector: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class J10Config:
    """Strictly parsed frozen J1.0 protocol."""

    protocol_version: str
    transform_direction: str
    coordinate_system: str
    field_units: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    squaring_steps: int
    outside_mode: OutsideMode
    evaluation_margin_voxels: tuple[int, int, int]
    cases: tuple[J10Case, ...]
    thresholds: J10Thresholds


@dataclass(frozen=True)
class J10Record:
    """One region-level row of the frozen numerical gate."""

    protocol_version: str
    case_id: str
    case_kind: str
    region: str
    transform_direction: str
    coordinate_system: str
    field_units: str
    evaluation_voxel_count: int
    endpoint_mean_mm: float
    endpoint_p95_mm: float
    endpoint_maximum_mm: float
    round_trip_mean_mm: float
    round_trip_p95_mm: float
    round_trip_maximum_mm: float
    expected_jacobian: float
    jacobian_minimum: float
    jacobian_p01: float
    jacobian_p01_abs_error: float
    nonpositive_jacobian_fraction: float
    normal_mismatch_p95_mm: float | None
    tangential_slip_truth_mm: float | None
    tangential_slip_observed_mm: float | None
    tangential_slip_error_mm: float | None
    gate_pass: bool
    gate_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | int | float | bool | None]:
        """Flatten tuple reasons for a portable subject-like table."""
        values = asdict(self)
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return cast(dict[str, str | int | float | bool | None], values)


@dataclass(frozen=True)
class J10Run:
    """Computed fields and independent region-level QC for one case."""

    case: J10Case
    records: tuple[J10Record, J10Record]
    lung_velocity_mm: VectorArray
    body_velocity_mm: VectorArray
    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    truth_lung_displacement_mm: FloatVectorArray
    truth_body_displacement_mm: FloatVectorArray


def _triple(
    value: object,
    *,
    key: str,
    caster: type[int] | type[float],
) -> tuple[Any, Any, Any]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must be a JSON array of length 3")
    return cast(tuple[Any, Any, Any], tuple(caster(item) for item in value))


def _positive_float(payload: dict[str, Any], key: str, *, zero_ok: bool) -> float:
    value = float(payload[key])
    if not math.isfinite(value) or value < 0 or (not zero_ok and value == 0):
        qualifier = "non-negative" if zero_ok else "positive"
        raise ValueError(f"{key} must be a finite {qualifier} number")
    return value


def _load_thresholds(payload: object) -> J10Thresholds:
    expected = {
        "endpoint_p95_max_mm",
        "round_trip_p95_max_mm",
        "jacobian_abs_error_max",
        "nonpositive_jacobian_fraction_max",
        "normal_mismatch_p95_max_mm",
        "tangential_slip_error_max_mm",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("thresholds contain missing or unknown keys")
    values = cast(dict[str, Any], payload)
    return J10Thresholds(
        endpoint_p95_max_mm=_positive_float(
            values,
            "endpoint_p95_max_mm",
            zero_ok=False,
        ),
        round_trip_p95_max_mm=_positive_float(
            values,
            "round_trip_p95_max_mm",
            zero_ok=False,
        ),
        jacobian_abs_error_max=_positive_float(
            values,
            "jacobian_abs_error_max",
            zero_ok=False,
        ),
        nonpositive_jacobian_fraction_max=_positive_float(
            values,
            "nonpositive_jacobian_fraction_max",
            zero_ok=True,
        ),
        normal_mismatch_p95_max_mm=_positive_float(
            values,
            "normal_mismatch_p95_max_mm",
            zero_ok=False,
        ),
        tangential_slip_error_max_mm=_positive_float(
            values,
            "tangential_slip_error_max_mm",
            zero_ok=False,
        ),
    )


def _load_case(payload: object) -> J10Case:
    if not isinstance(payload, dict):
        raise ValueError("each case must be a JSON object")
    values = cast(dict[str, Any], payload)
    case_id = values.get("case_id")
    kind = values.get("kind")
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"invalid case_id: {case_id!r}")
    if kind == "zero":
        if set(values) != {"case_id", "kind"}:
            raise ValueError(f"{case_id}: zero case contains unknown keys")
        return J10Case(case_id=case_id, kind="zero")
    if kind == "planar_sliding_translation":
        expected = {
            "case_id",
            "kind",
            "interface_normal_axis",
            "lung_velocity_mm",
            "body_velocity_mm",
        }
        if set(values) != expected:
            raise ValueError(f"{case_id}: translation case contains invalid keys")
        normal_axis = int(values["interface_normal_axis"])
        if normal_axis not in {0, 1, 2}:
            raise ValueError(f"{case_id}: interface_normal_axis must be 0, 1 or 2")
        return J10Case(
            case_id=case_id,
            kind="planar_sliding_translation",
            interface_normal_axis=normal_axis,
            lung_vector=cast(
                tuple[float, float, float],
                _triple(
                    values["lung_velocity_mm"],
                    key="lung_velocity_mm",
                    caster=float,
                ),
            ),
            body_vector=cast(
                tuple[float, float, float],
                _triple(
                    values["body_velocity_mm"],
                    key="body_velocity_mm",
                    caster=float,
                ),
            ),
        )
    if kind == "diagonal_affine":
        expected = {"case_id", "kind", "lung_rates", "body_rates"}
        if set(values) != expected:
            raise ValueError(f"{case_id}: affine case contains invalid keys")
        return J10Case(
            case_id=case_id,
            kind="diagonal_affine",
            lung_vector=cast(
                tuple[float, float, float],
                _triple(values["lung_rates"], key="lung_rates", caster=float),
            ),
            body_vector=cast(
                tuple[float, float, float],
                _triple(values["body_rates"], key="body_rates", caster=float),
            ),
        )
    if kind == "opposed_axial_rotation":
        expected = {"case_id", "kind", "lung_angle_rad", "body_angle_rad"}
        if set(values) != expected:
            raise ValueError(f"{case_id}: rotation case contains invalid keys")
        lung_angle = float(values["lung_angle_rad"])
        body_angle = float(values["body_angle_rad"])
        if not math.isfinite(lung_angle) or not math.isfinite(body_angle):
            raise ValueError(f"{case_id}: rotation angles must be finite")
        return J10Case(
            case_id=case_id,
            kind="opposed_axial_rotation",
            lung_vector=(lung_angle, 0.0, 0.0),
            body_vector=(body_angle, 0.0, 0.0),
        )
    raise ValueError(f"{case_id}: unsupported case kind {kind!r}")


def load_j10_config(path: Path) -> J10Config:
    """Load the exact frozen schema and reject ambiguous coordinate metadata."""
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
        "cases",
        "thresholds",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("J1.0 config contains missing or unknown top-level keys")
    protocol_version = payload["protocol_version"]
    transform_direction = payload["transform_direction"]
    coordinate_system = payload["coordinate_system"]
    field_units = payload["field_units"]
    for key, value in (
        ("protocol_version", protocol_version),
        ("transform_direction", transform_direction),
        ("coordinate_system", coordinate_system),
        ("field_units", field_units),
    ):
        if not isinstance(value, str) or not value:
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
        raise ValueError("spacing_mm values must be finite and positive")
    if any(value < 1 for value in margin):
        raise ValueError("evaluation margins must be at least one voxel")
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
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case_id values must be unique")
    return J10Config(
        protocol_version=protocol_version,
        transform_direction=transform_direction,
        coordinate_system=coordinate_system,
        field_units=field_units,
        shape=shape,
        spacing_mm=spacing,
        squaring_steps=squaring_steps,
        outside_mode=cast(OutsideMode, outside_mode),
        evaluation_margin_voxels=margin,
        cases=cases,
        thresholds=_load_thresholds(payload["thresholds"]),
    )


def _physical_grid(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> FloatVectorArray:
    axes = [
        np.arange(size, dtype=np.float64) * step
        for size, step in zip(shape, spacing, strict=True)
    ]
    coordinates = np.meshgrid(*axes, indexing="ij")
    return np.stack(coordinates, axis=-1)


def _analytic_fields(
    case: J10Case,
    grid_mm: FloatVectorArray,
    *,
    region: Literal["lung", "body"],
) -> tuple[VectorArray, FloatVectorArray, float]:
    vector = case.lung_vector if region == "lung" else case.body_vector
    velocity = np.zeros(grid_mm.shape, dtype=np.float32)
    truth = np.zeros(grid_mm.shape, dtype=np.float64)
    if case.kind == "zero":
        return velocity, truth, 1.0
    if vector is None:
        raise ValueError(f"{case.case_id}: missing {region} parameters")
    if case.kind == "planar_sliding_translation":
        velocity[...] = np.asarray(vector, dtype=np.float32)
        truth[...] = np.asarray(vector, dtype=np.float64)
        return velocity, truth, 1.0
    center = 0.5 * (grid_mm[0, 0, 0] + grid_mm[-1, -1, -1])
    centered = grid_mm - center
    if case.kind == "diagonal_affine":
        rates = np.asarray(vector, dtype=np.float64)
        velocity[...] = (centered * rates).astype(np.float32)
        truth[...] = centered * np.expm1(rates)
        return velocity, truth, float(np.exp(np.sum(rates)))
    angle = vector[0]
    x = centered[..., 0]
    y = centered[..., 1]
    velocity[..., 0] = (-angle * y).astype(np.float32)
    velocity[..., 1] = (angle * x).astype(np.float32)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    truth[..., 0] = (cosine - 1.0) * x - sine * y
    truth[..., 1] = sine * x + (cosine - 1.0) * y
    return velocity, truth, 1.0


def _evaluation_domain(config: J10Config) -> BoolArray:
    domain = np.zeros(config.shape, dtype=bool)
    slices = tuple(
        slice(border, size - border)
        for border, size in zip(
            config.evaluation_margin_voxels,
            config.shape,
            strict=True,
        )
    )
    domain[slices] = True
    return domain


def _interface_metrics(
    case: J10Case,
    lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    truth_lung_displacement_mm: FloatVectorArray,
    truth_body_displacement_mm: FloatVectorArray,
    domain: BoolArray,
) -> tuple[float | None, float | None, float | None, float | None]:
    if case.kind != "planar_sliding_translation":
        return None, None, None, None
    if case.interface_normal_axis is None:
        raise ValueError(f"{case.case_id}: missing interface normal")
    axis = case.interface_normal_axis
    observed_delta = (
        lung_displacement_mm.astype(np.float64)
        - body_displacement_mm.astype(np.float64)
    )
    truth_delta = truth_lung_displacement_mm - truth_body_displacement_mm
    normal = np.abs(observed_delta[..., axis][domain])
    observed_delta[..., axis] = 0.0
    truth_delta[..., axis] = 0.0
    observed_slip = np.linalg.norm(observed_delta, axis=-1)[domain]
    truth_slip = np.linalg.norm(truth_delta, axis=-1)[domain]
    normal_p95 = float(np.percentile(normal, 95))
    observed_median = float(np.median(observed_slip))
    truth_median = float(np.median(truth_slip))
    return (
        normal_p95,
        truth_median,
        observed_median,
        abs(observed_median - truth_median),
    )


def _evaluate_region(
    *,
    config: J10Config,
    case: J10Case,
    region: Literal["lung", "body"],
    displacement_mm: VectorArray,
    inverse_mm: VectorArray,
    truth_displacement_mm: FloatVectorArray,
    expected_jacobian: float,
    domain: BoolArray,
    interface_metrics: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ],
) -> J10Record:
    endpoint = np.linalg.norm(
        displacement_mm.astype(np.float64) - truth_displacement_mm,
        axis=-1,
    )[domain]
    round_trip = displacement_round_trip_metrics(
        displacement_mm,
        inverse_mm,
        config.spacing_mm,
        valid_domain=domain,
    )
    jacobian = jacobian_metrics(
        displacement_mm,
        config.spacing_mm,
        valid_domain=domain,
    )
    jacobian_error = abs(jacobian.p01 - expected_jacobian)
    normal, slip_truth, slip_observed, slip_error = interface_metrics
    reasons: list[str] = []
    endpoint_p95 = float(np.percentile(endpoint, 95))
    if endpoint_p95 > config.thresholds.endpoint_p95_max_mm:
        reasons.append("endpoint_p95")
    if round_trip.p95_mm > config.thresholds.round_trip_p95_max_mm:
        reasons.append("round_trip_p95")
    if jacobian_error > config.thresholds.jacobian_abs_error_max:
        reasons.append("jacobian_p01")
    if (
        jacobian.nonpositive_fraction
        > config.thresholds.nonpositive_jacobian_fraction_max
    ):
        reasons.append("folding")
    if (
        normal is not None
        and normal > config.thresholds.normal_mismatch_p95_max_mm
    ):
        reasons.append("normal_mismatch_p95")
    if (
        slip_error is not None
        and slip_error > config.thresholds.tangential_slip_error_max_mm
    ):
        reasons.append("tangential_slip")
    return J10Record(
        protocol_version=config.protocol_version,
        case_id=case.case_id,
        case_kind=case.kind,
        region=region,
        transform_direction=config.transform_direction,
        coordinate_system=config.coordinate_system,
        field_units=config.field_units,
        evaluation_voxel_count=int(np.count_nonzero(domain)),
        endpoint_mean_mm=float(np.mean(endpoint)),
        endpoint_p95_mm=endpoint_p95,
        endpoint_maximum_mm=float(np.max(endpoint)),
        round_trip_mean_mm=round_trip.mean_mm,
        round_trip_p95_mm=round_trip.p95_mm,
        round_trip_maximum_mm=round_trip.maximum_mm,
        expected_jacobian=expected_jacobian,
        jacobian_minimum=jacobian.minimum,
        jacobian_p01=jacobian.p01,
        jacobian_p01_abs_error=jacobian_error,
        nonpositive_jacobian_fraction=jacobian.nonpositive_fraction,
        normal_mismatch_p95_mm=normal,
        tangential_slip_truth_mm=slip_truth,
        tangential_slip_observed_mm=slip_observed,
        tangential_slip_error_mm=slip_error,
        gate_pass=not reasons,
        gate_reasons=tuple(reasons),
    )


def run_j10_gate(config: J10Config) -> tuple[J10Run, ...]:
    """Run all frozen analytic cases without image data or hidden tuning."""
    grid = _physical_grid(config.shape, config.spacing_mm)
    domain = _evaluation_domain(config)
    runs: list[J10Run] = []
    for case in config.cases:
        lung_velocity, lung_truth, lung_jacobian = _analytic_fields(
            case,
            grid,
            region="lung",
        )
        body_velocity, body_truth, body_jacobian = _analytic_fields(
            case,
            grid,
            region="body",
        )
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
            transform_direction=(
                "moving-inspiration_to_fixed-expiration"
            ),
        )
        interface = _interface_metrics(
            case,
            forward.lung_displacement_mm,
            forward.body_displacement_mm,
            lung_truth,
            body_truth,
            domain,
        )
        lung_record = _evaluate_region(
            config=config,
            case=case,
            region="lung",
            displacement_mm=forward.lung_displacement_mm,
            inverse_mm=inverse.lung_displacement_mm,
            truth_displacement_mm=lung_truth,
            expected_jacobian=lung_jacobian,
            domain=domain,
            interface_metrics=interface,
        )
        body_record = _evaluate_region(
            config=config,
            case=case,
            region="body",
            displacement_mm=forward.body_displacement_mm,
            inverse_mm=inverse.body_displacement_mm,
            truth_displacement_mm=body_truth,
            expected_jacobian=body_jacobian,
            domain=domain,
            interface_metrics=interface,
        )
        runs.append(
            J10Run(
                case=case,
                records=(lung_record, body_record),
                lung_velocity_mm=lung_velocity,
                body_velocity_mm=body_velocity,
                lung_displacement_mm=forward.lung_displacement_mm,
                body_displacement_mm=forward.body_displacement_mm,
                truth_lung_displacement_mm=lung_truth,
                truth_body_displacement_mm=body_truth,
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


def write_j10_gate(
    runs: tuple[J10Run, ...],
    output_dir: Path,
    *,
    config: J10Config,
    config_path: Path,
    repo_root: Path,
) -> Path:
    """Write reproducible batch artifacts before notebook visualisation."""
    if not runs:
        raise ValueError("cannot write an empty J1.0 gate")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.as_csv_row() for run in runs for record in run.records]
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
            usage=np.asarray("synthetic_numeric_diagnostic_ONLY"),
            lung_velocity_mm=run.lung_velocity_mm,
            body_velocity_mm=run.body_velocity_mm,
            lung_displacement_mm=run.lung_displacement_mm,
            body_displacement_mm=run.body_displacement_mm,
            truth_lung_displacement_mm=run.truth_lung_displacement_mm,
            truth_body_displacement_mm=run.truth_body_displacement_mm,
            spacing_mm=np.asarray(config.spacing_mm),
            transform_direction=np.asarray(config.transform_direction),
            coordinate_system=np.asarray(config.coordinate_system),
        )
        field_hashes[field_path.name] = _sha256(field_path)
        record_path = output_dir / f"{run.case.case_id}.json"
        record_path.write_text(
            json.dumps(
                {
                    "case_id": run.case.case_id,
                    "case_kind": run.case.kind,
                    "gate_pass": all(record.gate_pass for record in run.records),
                    "records": [asdict(record) for record in run.records],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        record_hashes[record_path.name] = _sha256(record_path)
    records = [record for run in runs for record in run.records]
    manifest = {
        "artifact_type": "piecewise_svf_j10_numeric_gate",
        "usage": "synthetic_numeric_diagnostic_ONLY",
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
        "case_pass_count": sum(
            all(record.gate_pass for record in run.records) for run in runs
        ),
        "region_row_count": len(records),
        "region_pass_count": sum(record.gate_pass for record in records),
        "all_pass": all(record.gate_pass for record in records),
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
    "J10Case",
    "J10Config",
    "J10Record",
    "J10Run",
    "J10Thresholds",
    "load_j10_config",
    "run_j10_gate",
    "write_j10_gate",
]
