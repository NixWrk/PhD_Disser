"""Frozen pre-expert S1.1 evaluation on real paired respiratory CT.

This module deliberately keeps development diagnostics separate from the
independent registration gate.  Its outputs can decide whether the frozen
expert benchmark may be opened, but they are never measurement fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from breathgeom.benchmark import PairData, load_pair_data
from breathgeom.io.pairs import RespiratoryPair
from breathgeom.measure.registration import (
    acquisition_fov_mask,
    fov_aware_mask_metrics,
    jacobian_metrics,
    landmark_tre,
    warp_mask,
)
from breathgeom.measure.sliding_registration import (
    SlidingS1Params,
    SlidingS1Result,
    register_sliding_s1,
)
from breathgeom.measure.wall import IntArray as WallIntArray

FloatArray = npt.NDArray[np.float64]
DEVELOPMENT_FIELD_DIR = "diagnostic_fields_NOT_FOR_MEASUREMENT"
DEVELOPMENT_FIELD_USAGE = "real_development_diagnostic_ONLY"
DEVELOPMENT_MARKER = "NOT_FOR_MEASUREMENT.txt"


@dataclass(frozen=True)
class RealS1Thresholds:
    """Frozen numerical limits checked before the expert benchmark is opened."""

    lung_fov_dice_min: float
    lung_fov_surface_p95_max_mm: float
    lung_surface_coverage_min: float
    common_fov_fraction_min: float
    keypoint_tre_after_mean_max_mm: float
    keypoint_tre_after_p95_max_mm: float
    keypoint_tre_improvement_fraction_min: float
    lung_jacobian_p01_min: float
    body_jacobian_p01_min: float
    nonpositive_jacobian_fraction_max: float
    interface_normal_mismatch_p95_max_mm: float
    coefficient_abs_fraction_max: float
    tangential_objective_relative_improvement_min: float
    lung_displacement_p99_max_mm: float
    body_displacement_p99_max_mm: float
    interface_core_voxel_count_min: int

    def __post_init__(self) -> None:
        fractions = (
            self.lung_fov_dice_min,
            self.lung_surface_coverage_min,
            self.common_fov_fraction_min,
            self.keypoint_tre_improvement_fraction_min,
            self.nonpositive_jacobian_fraction_max,
            self.coefficient_abs_fraction_max,
            self.tangential_objective_relative_improvement_min,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("fraction thresholds must lie in [0, 1]")
        positive = (
            self.lung_fov_surface_p95_max_mm,
            self.keypoint_tre_after_mean_max_mm,
            self.keypoint_tre_after_p95_max_mm,
            self.lung_jacobian_p01_min,
            self.body_jacobian_p01_min,
            self.interface_normal_mismatch_p95_max_mm,
            self.lung_displacement_p99_max_mm,
            self.body_displacement_p99_max_mm,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("positive real-development thresholds must be positive")
        if self.interface_core_voxel_count_min <= 0:
            raise ValueError("interface_core_voxel_count_min must be positive")


@dataclass(frozen=True)
class RealS1Protocol:
    """Exact development selection and the gate frozen before its first run."""

    gate_version: str
    dataset_id: str
    subject_ids: tuple[str, ...]
    expected_s1_version: str
    expected_transform_direction: str
    require_all_subjects: bool
    require_absent_expert_landmarks: bool
    require_image_derived_keypoints: bool
    fov_boundary_margin_mm: float
    thresholds: RealS1Thresholds

    def __post_init__(self) -> None:
        if not self.gate_version or not self.dataset_id or not self.expected_s1_version:
            raise ValueError("real-development identifiers must not be empty")
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("subject_ids must be non-empty and unique")
        if self.fov_boundary_margin_mm < 0:
            raise ValueError("fov_boundary_margin_mm must be non-negative")


@dataclass(frozen=True)
class RealS1Record:
    """One subject-level real-development result."""

    gate_version: str
    dataset_id: str
    subject_id: str
    s1_version: str
    coordinate_basis: str
    transform_direction: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    measurement_eligible: bool
    expert_landmarks_loaded: bool
    keypoint_count: int
    fixed_lung_outside_threshold_body_voxels: int
    moving_lung_outside_threshold_body_voxels: int
    lung_fov_dice_before: float
    lung_fov_dice_after: float
    lung_fov_surface_p95_before_mm: float
    lung_fov_surface_p95_after_mm: float
    lung_fov_fixed_surface_coverage_after: float
    lung_fov_moving_surface_coverage_after: float
    common_fov_fraction_after: float
    body_fov_dice_after: float
    body_fov_surface_p95_after_mm: float
    keypoint_tre_before_mean_mm: float
    keypoint_tre_before_p95_mm: float
    keypoint_tre_after_mean_mm: float
    keypoint_tre_after_p95_mm: float
    keypoint_tre_improvement_fraction: float
    lung_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    lung_jacobian_voxel_count: int
    body_jacobian_p01: float
    body_nonpositive_jacobian_fraction: float
    body_jacobian_voxel_count: int
    interface_core_voxel_count: int
    interface_normal_mismatch_p95_mm: float
    interface_tangential_slip_median_mm: float
    interface_tangential_slip_p95_mm: float
    lung_displacement_p99_mm: float
    body_displacement_p99_mm: float
    tangential_mode_names: tuple[str, ...]
    tangential_coefficients_mm: tuple[float, ...]
    coefficient_abs_fraction_max: float
    tangential_objective_initial: float
    tangential_objective_final: float
    tangential_objective_relative_improvement: float
    tangential_valid_voxel_count: int
    tangential_function_evaluations: int
    tangential_optimizer_success: bool
    lung_runtime_s: float
    body_runtime_s: float
    torch_version: str
    convexadam_version: str
    device: str
    gate_pass: bool
    gate_reasons: tuple[str, ...]

    def csv_row(self) -> dict[str, str | float | int | bool]:
        values = asdict(self)
        values["shape"] = "x".join(str(value) for value in self.shape)
        values["spacing_mm"] = "x".join(f"{value:g}" for value in self.spacing_mm)
        values["tangential_mode_names"] = ";".join(self.tangential_mode_names)
        values["tangential_coefficients_mm"] = ";".join(
            f"{value:.9g}" for value in self.tangential_coefficients_mm
        )
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return cast(dict[str, str | float | int | bool], values)


@dataclass(frozen=True)
class RealS1Run:
    """In-memory development run; fields remain diagnostic only."""

    record: RealS1Record
    result: SlidingS1Result


def _exact_dataclass(payload: dict[str, Any], target: type[Any], label: str) -> Any:
    expected = {field.name for field in fields(target)}
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return target(**payload)


def load_real_s1_protocol(path: Path) -> RealS1Protocol:
    """Load the frozen development protocol and reject schema drift."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("real-development protocol must be a JSON object")
    expected = {
        "gate_version",
        "dataset_id",
        "subject_ids",
        "expected_s1_version",
        "expected_transform_direction",
        "require_all_subjects",
        "require_absent_expert_landmarks",
        "require_image_derived_keypoints",
        "fov_boundary_margin_mm",
        "thresholds",
    }
    if set(payload) != expected:
        raise ValueError(
            "real-development protocol keys differ; "
            f"missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    subjects = payload["subject_ids"]
    threshold_payload = payload["thresholds"]
    if not isinstance(subjects, list) or not all(
        isinstance(subject, str) for subject in subjects
    ):
        raise ValueError("subject_ids must be a string array")
    if not isinstance(threshold_payload, dict):
        raise ValueError("thresholds must be a JSON object")
    thresholds = cast(
        RealS1Thresholds,
        _exact_dataclass(
            cast(dict[str, Any], threshold_payload),
            RealS1Thresholds,
            "real-development thresholds",
        ),
    )
    return RealS1Protocol(
        gate_version=str(payload["gate_version"]),
        dataset_id=str(payload["dataset_id"]),
        subject_ids=tuple(subjects),
        expected_s1_version=str(payload["expected_s1_version"]),
        expected_transform_direction=str(payload["expected_transform_direction"]),
        require_all_subjects=bool(payload["require_all_subjects"]),
        require_absent_expert_landmarks=bool(
            payload["require_absent_expert_landmarks"]
        ),
        require_image_derived_keypoints=bool(
            payload["require_image_derived_keypoints"]
        ),
        fov_boundary_margin_mm=float(payload["fov_boundary_margin_mm"]),
        thresholds=thresholds,
    )


def select_real_development_pairs(
    rows: tuple[RespiratoryPair, ...],
    protocol: RealS1Protocol,
) -> tuple[RespiratoryPair, ...]:
    """Return the exact frozen selection, failing before any image is loaded."""
    matching = {
        row.subject_id: row
        for row in rows
        if row.dataset_id == protocol.dataset_id
        and row.subject_id in protocol.subject_ids
    }
    missing = [subject for subject in protocol.subject_ids if subject not in matching]
    if missing:
        raise ValueError(f"frozen development subjects missing from manifest: {missing}")
    selected = tuple(matching[subject] for subject in protocol.subject_ids)
    for row in selected:
        label = f"{row.dataset_id}/{row.subject_id}"
        if not row.complete:
            raise ValueError(f"{label}: pair is incomplete")
        if row.fixed_phase != "expiration" or row.moving_phase != "inspiration":
            raise ValueError(f"{label}: expected fixed expiration and moving inspiration")
        has_any_expert_locator = bool(
            row.fixed_expert_landmarks or row.moving_expert_landmarks
        )
        if protocol.require_absent_expert_landmarks and has_any_expert_locator:
            raise ValueError(f"{label}: expert landmark locator is forbidden")
        has_both_keypoint_locators = bool(row.fixed_keypoints and row.moving_keypoints)
        if (
            protocol.require_image_derived_keypoints
            and not has_both_keypoint_locators
        ):
            raise ValueError(f"{label}: image-derived keypoints are required")
    return selected


def load_real_development_pair(pair: RespiratoryPair) -> PairData:
    """Load a pair after forcibly stripping expert locators."""
    if pair.fixed_expert_landmarks or pair.moving_expert_landmarks:
        raise ValueError("expert landmark locators are forbidden in real development")
    safe_pair = replace(
        pair,
        fixed_expert_landmarks="",
        moving_expert_landmarks="",
    )
    data = load_pair_data(safe_pair)
    if (
        data.fixed_expert_points_mm is not None
        or data.moving_expert_points_mm is not None
    ):
        raise RuntimeError("expert landmarks were unexpectedly loaded")
    return data


def _percentile(values: FloatArray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def _finite_limit(
    reasons: list[str],
    value: float,
    *,
    reason: str,
    maximum: float | None = None,
    minimum: float | None = None,
) -> None:
    if not np.isfinite(value):
        reasons.append(f"{reason}_nonfinite")
    elif (maximum is not None and value > maximum) or (
        minimum is not None and value < minimum
    ):
        reasons.append(reason)


def evaluate_real_s1(
    data: PairData,
    result: SlidingS1Result,
    *,
    params: SlidingS1Params,
    protocol: RealS1Protocol,
) -> RealS1Record:
    """Evaluate only predeclared real-development diagnostics."""
    if params.version != protocol.expected_s1_version:
        raise ValueError(
            f"S1 version {params.version!r} differs from "
            f"{protocol.expected_s1_version!r}"
        )
    if params.tangential is None:
        raise ValueError("real S1.1 development requires the tangential model")
    if result.transform_direction != protocol.expected_transform_direction:
        raise ValueError("S1 transform direction differs from the frozen protocol")
    if (
        data.fixed_expert_points_mm is not None
        or data.moving_expert_points_mm is not None
    ):
        raise ValueError("expert landmarks must not be loaded in real development")
    if data.fixed_keypoints_mm is None or data.moving_keypoints_mm is None:
        raise ValueError("image-derived keypoints are required by the frozen protocol")

    fixed_fov = acquisition_fov_mask(data.fixed_body_mask)
    moving_fov = acquisition_fov_mask(data.moving_body_mask)
    common_fov_before = fixed_fov & moving_fov
    warped_lung = warp_mask(
        data.moving_lung_mask,
        result.lung_displacement_mm,
        data.spacing,
    )
    warped_lung_fov = warp_mask(
        moving_fov,
        result.lung_displacement_mm,
        data.spacing,
    )
    common_lung_fov_after = fixed_fov & warped_lung_fov
    lung_before = fov_aware_mask_metrics(
        data.fixed_lung_mask,
        data.moving_lung_mask,
        data.spacing,
        valid_domain=common_fov_before,
        boundary_margin_mm=protocol.fov_boundary_margin_mm,
    )
    lung_after = fov_aware_mask_metrics(
        data.fixed_lung_mask,
        warped_lung,
        data.spacing,
        valid_domain=common_lung_fov_after,
        boundary_margin_mm=protocol.fov_boundary_margin_mm,
    )

    warped_body = warp_mask(
        data.moving_body_mask,
        result.body_displacement_mm,
        data.spacing,
    )
    warped_body_fov = warp_mask(
        moving_fov,
        result.body_displacement_mm,
        data.spacing,
    )
    common_body_fov_after = fixed_fov & warped_body_fov
    body_after = fov_aware_mask_metrics(
        data.fixed_body_mask,
        warped_body,
        data.spacing,
        valid_domain=common_body_fov_after,
        boundary_margin_mm=protocol.fov_boundary_margin_mm,
    )

    lung_jacobian_domain = data.fixed_lung_mask & common_lung_fov_after
    body_jacobian_domain = (
        data.fixed_body_mask & ~data.fixed_lung_mask & common_body_fov_after
    )
    lung_jacobian = jacobian_metrics(
        result.lung_displacement_mm,
        data.spacing,
        valid_domain=lung_jacobian_domain,
    )
    body_jacobian = jacobian_metrics(
        result.body_displacement_mm,
        data.spacing,
        valid_domain=body_jacobian_domain,
    )

    keypoint_before = landmark_tre(
        data.fixed_keypoints_mm,
        data.moving_keypoints_mm,
    )
    keypoint_after = landmark_tre(
        data.fixed_keypoints_mm,
        data.moving_keypoints_mm,
        result.lung_displacement_mm,
        data.spacing,
    )
    keypoint_before_mean = float(np.mean(keypoint_before))
    keypoint_after_mean = float(np.mean(keypoint_after))
    keypoint_improvement = (
        0.0
        if keypoint_before_mean <= np.finfo(np.float64).eps
        else 1.0 - keypoint_after_mean / keypoint_before_mean
    )

    interface_core = result.coupling_weight >= 1.0 - 1e-8
    difference = (
        result.lung_displacement_mm.astype(np.float64)
        - result.body_displacement_mm.astype(np.float64)
    )
    normal_component = np.abs(
        np.sum(difference * result.interface_normal, axis=-1)
    )
    tangential = difference - (
        np.sum(difference * result.interface_normal, axis=-1)[..., None]
        * result.interface_normal
    )
    tangential_magnitude = np.linalg.norm(tangential, axis=-1)

    lung_magnitude = np.linalg.norm(result.lung_displacement_mm, axis=-1)
    body_magnitude = np.linalg.norm(result.body_displacement_mm, axis=-1)
    coefficient_fraction = (
        max(abs(value) for value in result.tangential_coefficients_mm)
        / params.tangential.coefficient_bound_mm
    )
    objective_improvement = (
        0.0
        if abs(result.tangential_objective_initial) <= np.finfo(np.float64).eps
        else (
            result.tangential_objective_initial
            - result.tangential_objective_final
        )
        / abs(result.tangential_objective_initial)
    )

    thresholds = protocol.thresholds
    reasons: list[str] = []
    _finite_limit(
        reasons,
        lung_after.dice,
        reason="lung_fov_dice",
        minimum=thresholds.lung_fov_dice_min,
    )
    _finite_limit(
        reasons,
        lung_after.surface_p95_mm,
        reason="lung_fov_surface_p95",
        maximum=thresholds.lung_fov_surface_p95_max_mm,
    )
    for label, coverage in (
        ("lung_fixed_surface_coverage", lung_after.first_surface_coverage),
        ("lung_moving_surface_coverage", lung_after.second_surface_coverage),
    ):
        _finite_limit(
            reasons,
            coverage,
            reason=label,
            minimum=thresholds.lung_surface_coverage_min,
        )
    _finite_limit(
        reasons,
        lung_after.valid_voxel_fraction,
        reason="common_fov_fraction",
        minimum=thresholds.common_fov_fraction_min,
    )
    _finite_limit(
        reasons,
        keypoint_after_mean,
        reason="keypoint_tre_mean",
        maximum=thresholds.keypoint_tre_after_mean_max_mm,
    )
    _finite_limit(
        reasons,
        _percentile(keypoint_after, 95),
        reason="keypoint_tre_p95",
        maximum=thresholds.keypoint_tre_after_p95_max_mm,
    )
    _finite_limit(
        reasons,
        keypoint_improvement,
        reason="keypoint_tre_improvement",
        minimum=thresholds.keypoint_tre_improvement_fraction_min,
    )
    _finite_limit(
        reasons,
        lung_jacobian.p01,
        reason="lung_jacobian_p01",
        minimum=thresholds.lung_jacobian_p01_min,
    )
    _finite_limit(
        reasons,
        body_jacobian.p01,
        reason="body_jacobian_p01",
        minimum=thresholds.body_jacobian_p01_min,
    )
    if (
        lung_jacobian.nonpositive_fraction
        > thresholds.nonpositive_jacobian_fraction_max
    ):
        reasons.append("lung_folding")
    if (
        body_jacobian.nonpositive_fraction
        > thresholds.nonpositive_jacobian_fraction_max
    ):
        reasons.append("body_folding")
    normal_p95 = _percentile(normal_component[interface_core], 95)
    _finite_limit(
        reasons,
        normal_p95,
        reason="interface_normal_mismatch_p95",
        maximum=thresholds.interface_normal_mismatch_p95_max_mm,
    )
    if int(interface_core.sum()) < thresholds.interface_core_voxel_count_min:
        reasons.append("interface_core_voxel_count")
    _finite_limit(
        reasons,
        coefficient_fraction,
        reason="tangential_coefficient_bound",
        maximum=thresholds.coefficient_abs_fraction_max,
    )
    _finite_limit(
        reasons,
        objective_improvement,
        reason="tangential_objective_improvement",
        minimum=thresholds.tangential_objective_relative_improvement_min,
    )
    if not result.tangential_optimizer_success:
        reasons.append("tangential_optimizer")
    if result.tangential_valid_voxel_count < params.tangential.minimum_valid_voxels:
        reasons.append("tangential_valid_voxel_count")
    lung_p99 = _percentile(lung_magnitude[data.fixed_lung_mask], 99)
    body_p99 = _percentile(
        body_magnitude[data.fixed_body_mask & ~data.fixed_lung_mask],
        99,
    )
    _finite_limit(
        reasons,
        lung_p99,
        reason="lung_displacement_p99",
        maximum=thresholds.lung_displacement_p99_max_mm,
    )
    _finite_limit(
        reasons,
        body_p99,
        reason="body_displacement_p99",
        maximum=thresholds.body_displacement_p99_max_mm,
    )

    return RealS1Record(
        gate_version=protocol.gate_version,
        dataset_id=data.pair.dataset_id,
        subject_id=data.pair.subject_id,
        s1_version=params.version,
        coordinate_basis="local-zero-origin-RAS+_mm",
        transform_direction=result.transform_direction,
        shape=cast(tuple[int, int, int], data.fixed_ras.shape),
        spacing_mm=data.spacing,
        measurement_eligible=False,
        expert_landmarks_loaded=False,
        keypoint_count=len(keypoint_after),
        fixed_lung_outside_threshold_body_voxels=int(
            np.count_nonzero(data.fixed_lung_mask & ~data.fixed_body_mask)
        ),
        moving_lung_outside_threshold_body_voxels=int(
            np.count_nonzero(data.moving_lung_mask & ~data.moving_body_mask)
        ),
        lung_fov_dice_before=lung_before.dice,
        lung_fov_dice_after=lung_after.dice,
        lung_fov_surface_p95_before_mm=lung_before.surface_p95_mm,
        lung_fov_surface_p95_after_mm=lung_after.surface_p95_mm,
        lung_fov_fixed_surface_coverage_after=lung_after.first_surface_coverage,
        lung_fov_moving_surface_coverage_after=lung_after.second_surface_coverage,
        common_fov_fraction_after=lung_after.valid_voxel_fraction,
        body_fov_dice_after=body_after.dice,
        body_fov_surface_p95_after_mm=body_after.surface_p95_mm,
        keypoint_tre_before_mean_mm=keypoint_before_mean,
        keypoint_tre_before_p95_mm=_percentile(keypoint_before, 95),
        keypoint_tre_after_mean_mm=keypoint_after_mean,
        keypoint_tre_after_p95_mm=_percentile(keypoint_after, 95),
        keypoint_tre_improvement_fraction=keypoint_improvement,
        lung_jacobian_p01=lung_jacobian.p01,
        lung_nonpositive_jacobian_fraction=lung_jacobian.nonpositive_fraction,
        lung_jacobian_voxel_count=lung_jacobian.voxel_count,
        body_jacobian_p01=body_jacobian.p01,
        body_nonpositive_jacobian_fraction=body_jacobian.nonpositive_fraction,
        body_jacobian_voxel_count=body_jacobian.voxel_count,
        interface_core_voxel_count=int(interface_core.sum()),
        interface_normal_mismatch_p95_mm=normal_p95,
        interface_tangential_slip_median_mm=_percentile(
            tangential_magnitude[interface_core], 50
        ),
        interface_tangential_slip_p95_mm=_percentile(
            tangential_magnitude[interface_core], 95
        ),
        lung_displacement_p99_mm=lung_p99,
        body_displacement_p99_mm=body_p99,
        tangential_mode_names=result.tangential_mode_names,
        tangential_coefficients_mm=result.tangential_coefficients_mm,
        coefficient_abs_fraction_max=coefficient_fraction,
        tangential_objective_initial=result.tangential_objective_initial,
        tangential_objective_final=result.tangential_objective_final,
        tangential_objective_relative_improvement=objective_improvement,
        tangential_valid_voxel_count=result.tangential_valid_voxel_count,
        tangential_function_evaluations=result.tangential_function_evaluations,
        tangential_optimizer_success=result.tangential_optimizer_success,
        lung_runtime_s=result.lung_runtime.elapsed_s,
        body_runtime_s=result.body_elapsed_s,
        torch_version=result.lung_runtime.torch_version,
        convexadam_version=result.lung_runtime.convexadam_version,
        device=result.lung_runtime.device,
        gate_pass=not reasons,
        gate_reasons=tuple(reasons),
    )


def run_real_s1_pair(
    data: PairData,
    *,
    params: SlidingS1Params,
    protocol: RealS1Protocol,
    registration_python: Path,
    repo_root: Path,
    temporary_root: Path | None = None,
) -> RealS1Run:
    """Run S1.1 with a contract-only body closure and evaluate frozen QC."""
    fixed_registration_body = np.ascontiguousarray(
        data.fixed_body_mask | data.fixed_lung_mask
    )
    moving_registration_body = np.ascontiguousarray(
        data.moving_body_mask | data.moving_lung_mask
    )
    result = register_sliding_s1(
        cast(WallIntArray, data.fixed_ras),
        cast(WallIntArray, data.moving_ras),
        data.spacing,
        data.fixed_lung_mask,
        data.moving_lung_mask,
        fixed_registration_body,
        moving_registration_body,
        registration_python=registration_python,
        repo_root=repo_root,
        params=params,
        temporary_root=temporary_root,
    )
    record = evaluate_real_s1(
        data,
        result,
        params=params,
        protocol=protocol,
    )
    return RealS1Run(record=record, result=result)


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


def write_real_s1_batch(
    runs: tuple[RealS1Run, ...],
    output_dir: Path,
    *,
    failures: tuple[str, ...],
    pair_manifest_path: Path,
    s1_config_path: Path,
    gate_config_path: Path,
    protocol: RealS1Protocol,
    repo_root: Path,
) -> Path:
    """Write batch-first diagnostic artifacts with explicit non-measurement status."""
    output_dir.mkdir(parents=True, exist_ok=True)
    field_dir = output_dir / DEVELOPMENT_FIELD_DIR
    field_dir.mkdir(parents=True, exist_ok=True)
    (field_dir / DEVELOPMENT_MARKER).write_text(
        "These fields use real development data but have no independent expert gate.\n"
        "They must not be used for measurements, model training, FEM, or anatomy claims.\n",
        encoding="utf-8",
    )
    field_hashes: dict[str, str] = {}
    for run in runs:
        stem = f"{run.record.dataset_id}__{run.record.subject_id}"
        field_path = field_dir / f"{stem}.npz"
        np.savez_compressed(
            field_path,
            usage=np.asarray(DEVELOPMENT_FIELD_USAGE),
            measurement_eligible=np.asarray(False),
            gate_pass=np.asarray(run.record.gate_pass),
            gate_reasons=np.asarray(run.record.gate_reasons, dtype=np.str_),
            lung_displacement_mm=run.result.lung_displacement_mm,
            body_displacement_mm=run.result.body_displacement_mm,
            initial_lung_displacement_mm=run.result.initial_lung_displacement_mm,
            spacing_mm=np.asarray(run.record.spacing_mm, dtype=np.float64),
            transform_direction=np.asarray(run.record.transform_direction),
            coordinate_basis=np.asarray(run.record.coordinate_basis),
            s1_version=np.asarray(run.record.s1_version),
        )
        relative_field = field_path.relative_to(output_dir).as_posix()
        field_hashes[relative_field] = _sha256(field_path)
        payload = {
            "record": asdict(run.record),
            "provenance": {
                "field_file": relative_field,
                "field_sha256": field_hashes[relative_field],
                "field_disposition": DEVELOPMENT_FIELD_USAGE,
                "measurement_eligible": False,
            },
        }
        (output_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    summary_path = output_dir / "summary.csv"
    if runs:
        rows = [run.record.csv_row() for run in runs]
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary_sha256: str | None = _sha256(summary_path)
    else:
        summary_sha256 = None

    passed_subjects = [run.record.subject_id for run in runs if run.record.gate_pass]
    expected_subjects = list(protocol.subject_ids)
    all_pass = (
        not failures
        and len(runs) == len(expected_subjects)
        and set(passed_subjects) == set(expected_subjects)
    )
    manifest = {
        "artifact_type": "real_s1_preexpert_development_batch",
        "usage": DEVELOPMENT_FIELD_USAGE,
        "measurement_eligible": False,
        "gate_version": protocol.gate_version,
        "s1_version": protocol.expected_s1_version,
        "code_version": _code_version(repo_root),
        "selection": {
            "dataset_id": protocol.dataset_id,
            "subject_ids": expected_subjects,
            "expert_landmarks_used": False,
            "image_derived_keypoints_used": True,
        },
        "pair_manifest": {
            "path": pair_manifest_path.name,
            "sha256": _sha256(pair_manifest_path),
        },
        "s1_config": {
            "path": s1_config_path.name,
            "sha256": _sha256(s1_config_path),
        },
        "gate_config": {
            "path": gate_config_path.name,
            "sha256": _sha256(gate_config_path),
        },
        "subject_count_expected": len(expected_subjects),
        "subject_count_completed": len(runs),
        "pass_count": len(passed_subjects),
        "all_pass": all_pass,
        "failures": list(failures),
        "field_sha256": field_hashes,
        "summary_sha256": summary_sha256,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "DEVELOPMENT_FIELD_DIR",
    "DEVELOPMENT_FIELD_USAGE",
    "RealS1Protocol",
    "RealS1Record",
    "RealS1Run",
    "RealS1Thresholds",
    "evaluate_real_s1",
    "load_real_development_pair",
    "load_real_s1_protocol",
    "run_real_s1_pair",
    "select_real_development_pairs",
    "write_real_s1_batch",
]
