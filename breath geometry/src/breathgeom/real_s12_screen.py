"""Reproducible post-hoc screen of simple S1.2 repair classes.

The screen consumes quarantined non-expert S1.1 fields.  It is deliberately
separate from registration and cannot produce measurement-eligible fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import ndimage

from breathgeom.benchmark import PairData
from breathgeom.measure.registration import (
    VectorArray,
    acquisition_fov_mask,
    fov_aware_mask_metrics,
    jacobian_metrics,
    landmark_tre,
    warp_mask,
)
from breathgeom.measure.sliding_registration import (
    couple_normal_components,
    lung_interface_normal,
)
from breathgeom.real_s1 import RealS1Protocol

SCREEN_USAGE = "posthoc_s12_heuristic_screen_ONLY"
EXPECTED_FIELD_USAGE = "real_development_diagnostic_ONLY"


@dataclass(frozen=True)
class S12HeuristicVariant:
    """One predeclared field-repair and contact-coupling variant."""

    name: str
    lung_smoothing_sigma_mm: float
    lung_scale: float
    couple_normal: bool
    normal_smoothing_mm: float
    contact_core_mm: float
    contact_taper_mm: float
    shared_normal_source: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("S1.2 screen variant name must not be empty")
        if self.lung_smoothing_sigma_mm < 0:
            raise ValueError("lung smoothing sigma must be non-negative")
        if not 0 < self.lung_scale <= 1:
            raise ValueError("lung scale must lie in (0, 1]")
        if min(
            self.normal_smoothing_mm,
            self.contact_core_mm,
            self.contact_taper_mm,
        ) <= 0:
            raise ValueError("normal and contact distances must be positive")
        if self.shared_normal_source not in {"body", "mean"}:
            raise ValueError("shared_normal_source must be 'body' or 'mean'")


@dataclass(frozen=True)
class S12HeuristicScreen:
    """Frozen post-hoc model-class screen."""

    screen_version: str
    input_artifact_type: str
    input_usage: str
    input_s1_version: str
    expected_completed_subject_ids: tuple[str, ...]
    expected_missing_subject_ids: tuple[str, ...]
    expert_landmarks_used: bool
    measurement_eligible: bool
    variants: tuple[S12HeuristicVariant, ...]

    def __post_init__(self) -> None:
        identifiers = (
            self.screen_version,
            self.input_artifact_type,
            self.input_usage,
            self.input_s1_version,
        )
        if any(not value for value in identifiers):
            raise ValueError("S1.2 screen identifiers must not be empty")
        if not self.expected_completed_subject_ids:
            raise ValueError("S1.2 screen requires completed input subjects")
        all_subjects = (
            self.expected_completed_subject_ids + self.expected_missing_subject_ids
        )
        if len(set(all_subjects)) != len(all_subjects):
            raise ValueError("S1.2 screen subject sets must be disjoint and unique")
        if self.expert_landmarks_used:
            raise ValueError("expert landmarks are forbidden in the S1.2 screen")
        if self.measurement_eligible:
            raise ValueError("S1.2 screen cannot be measurement eligible")
        names = tuple(variant.name for variant in self.variants)
        if not names or len(set(names)) != len(names):
            raise ValueError("S1.2 screen variants must be non-empty and unique")


@dataclass(frozen=True)
class S12HeuristicRecord:
    """Subject-level metrics for one frozen heuristic variant."""

    dataset_id: str
    subject_id: str
    screen_version: str
    variant: str
    transform_direction: str
    coordinate_basis: str
    expert_landmarks_loaded: bool
    measurement_eligible: bool
    lung_smoothing_sigma_mm: float
    lung_scale: float
    couple_normal: bool
    normal_smoothing_mm: float
    contact_core_mm: float
    contact_taper_mm: float
    shared_normal_source: str
    keypoint_count: int
    lung_fov_dice: float
    lung_fov_surface_p95_mm: float
    lung_fixed_surface_coverage: float
    lung_moving_surface_coverage: float
    common_fov_fraction: float
    keypoint_tre_mean_mm: float
    keypoint_tre_p95_mm: float
    keypoint_tre_improvement_fraction: float
    lung_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    body_jacobian_p01: float
    body_nonpositive_jacobian_fraction: float
    interface_core_voxel_count: int
    interface_normal_mismatch_p95_mm: float
    interface_tangential_slip_median_mm: float
    interface_tangential_slip_p95_mm: float
    lung_displacement_p99_mm: float
    body_displacement_p99_mm: float
    correspondence_pass: bool
    topology_pass: bool
    contact_pass: bool
    screen_criteria_pass: bool
    screen_reasons: tuple[str, ...]
    elapsed_s: float

    def csv_row(self) -> dict[str, str | float | int | bool]:
        values = asdict(self)
        values["screen_reasons"] = ";".join(self.screen_reasons)
        return cast(dict[str, str | float | int | bool], values)


def _exact_keys(payload: dict[str, Any], target: type[Any], label: str) -> None:
    expected = {item.name for item in fields(target)}
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def load_s12_heuristic_screen(path: Path) -> S12HeuristicScreen:
    """Load the frozen screen and reject schema drift."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S1.2 heuristic screen must be a JSON object")
    typed = cast(dict[str, Any], payload)
    _exact_keys(typed, S12HeuristicScreen, "S1.2 screen")
    completed = typed["expected_completed_subject_ids"]
    missing = typed["expected_missing_subject_ids"]
    variants = typed["variants"]
    if not isinstance(completed, list) or not all(
        isinstance(value, str) for value in completed
    ):
        raise ValueError("expected_completed_subject_ids must be a string array")
    if not isinstance(missing, list) or not all(
        isinstance(value, str) for value in missing
    ):
        raise ValueError("expected_missing_subject_ids must be a string array")
    if not isinstance(variants, list) or not all(
        isinstance(value, dict) for value in variants
    ):
        raise ValueError("variants must be an object array")
    parsed_variants: list[S12HeuristicVariant] = []
    for index, value in enumerate(variants):
        variant = cast(dict[str, Any], value)
        _exact_keys(variant, S12HeuristicVariant, f"S1.2 variant {index}")
        parsed_variants.append(S12HeuristicVariant(**variant))
    return S12HeuristicScreen(
        screen_version=str(typed["screen_version"]),
        input_artifact_type=str(typed["input_artifact_type"]),
        input_usage=str(typed["input_usage"]),
        input_s1_version=str(typed["input_s1_version"]),
        expected_completed_subject_ids=tuple(completed),
        expected_missing_subject_ids=tuple(missing),
        expert_landmarks_used=bool(typed["expert_landmarks_used"]),
        measurement_eligible=bool(typed["measurement_eligible"]),
        variants=tuple(parsed_variants),
    )


def smooth_scale_field(
    displacement_mm: VectorArray,
    spacing: tuple[float, float, float],
    *,
    smoothing_sigma_mm: float,
    scale: float,
) -> VectorArray:
    """Apply a physical Gaussian filter and global magnitude scale."""
    if displacement_mm.ndim != 4 or displacement_mm.shape[-1] != 3:
        raise ValueError("displacement field must have shape X x Y x Z x 3")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if smoothing_sigma_mm < 0:
        raise ValueError("smoothing sigma must be non-negative")
    if not 0 < scale <= 1:
        raise ValueError("scale must lie in (0, 1]")
    if smoothing_sigma_mm == 0:
        return np.ascontiguousarray((displacement_mm * scale).astype(np.float32))
    sigma = tuple(smoothing_sigma_mm / value for value in spacing)
    output = np.empty_like(displacement_mm, dtype=np.float32)
    for component in range(3):
        ndimage.gaussian_filter(
            displacement_mm[..., component],
            sigma=sigma,
            mode="nearest",
            output=output[..., component],
        )
    output *= np.float32(scale)
    return np.ascontiguousarray(output)


def build_s12_variant_fields(
    initial_lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    fixed_lung_mask: np.ndarray,
    spacing: tuple[float, float, float],
    variant: S12HeuristicVariant,
    *,
    repaired_lung_displacement_mm: VectorArray | None = None,
) -> tuple[VectorArray, VectorArray, np.ndarray, np.ndarray]:
    """Build one frozen repair/coupling variant and its contact core."""
    repaired = repaired_lung_displacement_mm
    if repaired is None:
        repaired = smooth_scale_field(
            initial_lung_displacement_mm,
            spacing,
            smoothing_sigma_mm=variant.lung_smoothing_sigma_mm,
            scale=variant.lung_scale,
        )
    if repaired.shape != initial_lung_displacement_mm.shape:
        raise ValueError("cached repaired lung field has an unexpected shape")
    if variant.couple_normal:
        lung, body, normal, weight = couple_normal_components(
            repaired,
            body_displacement_mm,
            fixed_lung_mask,
            spacing,
            normal_smoothing_mm=variant.normal_smoothing_mm,
            contact_core_mm=variant.contact_core_mm,
            contact_taper_mm=variant.contact_taper_mm,
            shared_normal_source=variant.shared_normal_source,
        )
        core = weight >= 1.0 - 1e-8
        return lung, body, normal, np.ascontiguousarray(core)
    normal, signed_distance = lung_interface_normal(
        fixed_lung_mask,
        spacing,
        smoothing_mm=variant.normal_smoothing_mm,
    )
    core = np.abs(signed_distance) <= variant.contact_core_mm
    return (
        repaired,
        np.ascontiguousarray(body_displacement_mm),
        normal,
        np.ascontiguousarray(core),
    )


def _p(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def evaluate_s12_variant(
    data: PairData,
    lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    interface_normal: np.ndarray,
    interface_core: np.ndarray,
    *,
    variant: S12HeuristicVariant,
    screen: S12HeuristicScreen,
    protocol: RealS1Protocol,
    transform_direction: str,
    elapsed_s: float,
) -> S12HeuristicRecord:
    """Evaluate correspondence, regional topology and contact independently."""
    if (
        data.fixed_expert_points_mm is not None
        or data.moving_expert_points_mm is not None
    ):
        raise ValueError("expert landmarks are forbidden in the S1.2 screen")
    if data.fixed_keypoints_mm is None or data.moving_keypoints_mm is None:
        raise ValueError("image-derived keypoints are required in the S1.2 screen")
    if transform_direction != protocol.expected_transform_direction:
        raise ValueError("screen field has an unexpected transform direction")

    fixed_fov = acquisition_fov_mask(data.fixed_body_mask)
    moving_fov = acquisition_fov_mask(data.moving_body_mask)
    warped_lung = warp_mask(
        data.moving_lung_mask,
        lung_displacement_mm,
        data.spacing,
    )
    common_lung_fov = fixed_fov & warp_mask(
        moving_fov,
        lung_displacement_mm,
        data.spacing,
    )
    lung_metrics = fov_aware_mask_metrics(
        data.fixed_lung_mask,
        warped_lung,
        data.spacing,
        valid_domain=common_lung_fov,
        boundary_margin_mm=protocol.fov_boundary_margin_mm,
    )
    common_body_fov = fixed_fov & warp_mask(
        moving_fov,
        body_displacement_mm,
        data.spacing,
    )
    lung_topology = jacobian_metrics(
        lung_displacement_mm,
        data.spacing,
        valid_domain=data.fixed_lung_mask & common_lung_fov,
    )
    body_topology = jacobian_metrics(
        body_displacement_mm,
        data.spacing,
        valid_domain=(
            data.fixed_body_mask & ~data.fixed_lung_mask & common_body_fov
        ),
    )
    keypoints_before = landmark_tre(
        data.fixed_keypoints_mm,
        data.moving_keypoints_mm,
    )
    keypoints_after = landmark_tre(
        data.fixed_keypoints_mm,
        data.moving_keypoints_mm,
        lung_displacement_mm,
        data.spacing,
    )
    before_mean = float(np.mean(keypoints_before))
    after_mean = float(np.mean(keypoints_after))
    improvement = (
        0.0
        if before_mean <= np.finfo(np.float64).eps
        else 1.0 - after_mean / before_mean
    )
    difference = (
        lung_displacement_mm.astype(np.float64)
        - body_displacement_mm.astype(np.float64)
    )
    signed_normal = np.sum(difference * interface_normal, axis=-1)
    normal_mismatch = np.abs(signed_normal)
    tangential = difference - signed_normal[..., None] * interface_normal
    tangential_magnitude = np.linalg.norm(tangential, axis=-1)
    lung_magnitude = np.linalg.norm(lung_displacement_mm, axis=-1)
    body_magnitude = np.linalg.norm(body_displacement_mm, axis=-1)

    thresholds = protocol.thresholds
    correspondence_reasons: list[str] = []
    if lung_metrics.dice < thresholds.lung_fov_dice_min:
        correspondence_reasons.append("lung_fov_dice")
    if lung_metrics.surface_p95_mm > thresholds.lung_fov_surface_p95_max_mm:
        correspondence_reasons.append("lung_fov_surface_p95")
    if (
        lung_metrics.first_surface_coverage < thresholds.lung_surface_coverage_min
    ):
        correspondence_reasons.append("lung_fixed_surface_coverage")
    if (
        lung_metrics.second_surface_coverage < thresholds.lung_surface_coverage_min
    ):
        correspondence_reasons.append("lung_moving_surface_coverage")
    if lung_metrics.valid_voxel_fraction < thresholds.common_fov_fraction_min:
        correspondence_reasons.append("common_fov_fraction")
    if after_mean > thresholds.keypoint_tre_after_mean_max_mm:
        correspondence_reasons.append("keypoint_tre_mean")
    after_p95 = _p(keypoints_after, 95)
    if after_p95 > thresholds.keypoint_tre_after_p95_max_mm:
        correspondence_reasons.append("keypoint_tre_p95")
    if improvement < thresholds.keypoint_tre_improvement_fraction_min:
        correspondence_reasons.append("keypoint_tre_improvement")

    topology_reasons: list[str] = []
    if lung_topology.p01 < thresholds.lung_jacobian_p01_min:
        topology_reasons.append("lung_jacobian_p01")
    if body_topology.p01 < thresholds.body_jacobian_p01_min:
        topology_reasons.append("body_jacobian_p01")
    if (
        lung_topology.nonpositive_fraction
        > thresholds.nonpositive_jacobian_fraction_max
    ):
        topology_reasons.append("lung_folding")
    if (
        body_topology.nonpositive_fraction
        > thresholds.nonpositive_jacobian_fraction_max
    ):
        topology_reasons.append("body_folding")

    contact_reasons: list[str] = []
    normal_p95 = _p(normal_mismatch[interface_core], 95)
    if (
        not np.isfinite(normal_p95)
        or normal_p95 > thresholds.interface_normal_mismatch_p95_max_mm
    ):
        contact_reasons.append("interface_normal_mismatch_p95")
    core_count = int(np.count_nonzero(interface_core))
    if core_count < thresholds.interface_core_voxel_count_min:
        contact_reasons.append("interface_core_voxel_count")

    magnitude_reasons: list[str] = []
    lung_p99 = _p(lung_magnitude[data.fixed_lung_mask], 99)
    body_p99 = _p(
        body_magnitude[data.fixed_body_mask & ~data.fixed_lung_mask],
        99,
    )
    if lung_p99 > thresholds.lung_displacement_p99_max_mm:
        magnitude_reasons.append("lung_displacement_p99")
    if body_p99 > thresholds.body_displacement_p99_max_mm:
        magnitude_reasons.append("body_displacement_p99")

    reasons = (
        correspondence_reasons
        + topology_reasons
        + contact_reasons
        + magnitude_reasons
    )
    return S12HeuristicRecord(
        dataset_id=data.pair.dataset_id,
        subject_id=data.pair.subject_id,
        screen_version=screen.screen_version,
        variant=variant.name,
        transform_direction=transform_direction,
        coordinate_basis="local-zero-origin-RAS+_mm",
        expert_landmarks_loaded=False,
        measurement_eligible=False,
        lung_smoothing_sigma_mm=variant.lung_smoothing_sigma_mm,
        lung_scale=variant.lung_scale,
        couple_normal=variant.couple_normal,
        normal_smoothing_mm=variant.normal_smoothing_mm,
        contact_core_mm=variant.contact_core_mm,
        contact_taper_mm=variant.contact_taper_mm,
        shared_normal_source=variant.shared_normal_source,
        keypoint_count=len(keypoints_after),
        lung_fov_dice=lung_metrics.dice,
        lung_fov_surface_p95_mm=lung_metrics.surface_p95_mm,
        lung_fixed_surface_coverage=lung_metrics.first_surface_coverage,
        lung_moving_surface_coverage=lung_metrics.second_surface_coverage,
        common_fov_fraction=lung_metrics.valid_voxel_fraction,
        keypoint_tre_mean_mm=after_mean,
        keypoint_tre_p95_mm=after_p95,
        keypoint_tre_improvement_fraction=improvement,
        lung_jacobian_p01=lung_topology.p01,
        lung_nonpositive_jacobian_fraction=lung_topology.nonpositive_fraction,
        body_jacobian_p01=body_topology.p01,
        body_nonpositive_jacobian_fraction=body_topology.nonpositive_fraction,
        interface_core_voxel_count=core_count,
        interface_normal_mismatch_p95_mm=normal_p95,
        interface_tangential_slip_median_mm=_p(
            tangential_magnitude[interface_core],
            50,
        ),
        interface_tangential_slip_p95_mm=_p(
            tangential_magnitude[interface_core],
            95,
        ),
        lung_displacement_p99_mm=lung_p99,
        body_displacement_p99_mm=body_p99,
        correspondence_pass=not correspondence_reasons,
        topology_pass=not topology_reasons,
        contact_pass=not contact_reasons,
        screen_criteria_pass=not reasons,
        screen_reasons=tuple(reasons),
        elapsed_s=elapsed_s,
    )


def screen_real_s1_fields(
    data: PairData,
    field_path: Path,
    *,
    screen: S12HeuristicScreen,
    protocol: RealS1Protocol,
) -> tuple[S12HeuristicRecord, ...]:
    """Evaluate all frozen variants for one stored non-expert S1.1 field."""
    with np.load(field_path) as artifact:
        usage = str(artifact["usage"])
        if usage != screen.input_usage or usage != EXPECTED_FIELD_USAGE:
            raise ValueError(f"{field_path}: unexpected field usage {usage!r}")
        if bool(artifact["measurement_eligible"]):
            raise ValueError(f"{field_path}: development field cannot be measurement eligible")
        s1_version = str(artifact["s1_version"])
        if s1_version != screen.input_s1_version:
            raise ValueError(f"{field_path}: unexpected S1 version {s1_version!r}")
        transform_direction = str(artifact["transform_direction"])
        stored_spacing = tuple(float(value) for value in artifact["spacing_mm"])
        if not np.allclose(stored_spacing, data.spacing, rtol=0.0, atol=1e-8):
            raise ValueError(f"{field_path}: field/data spacing mismatch")
        initial = np.ascontiguousarray(
            artifact["initial_lung_displacement_mm"].astype(np.float32)
        )
        body = np.ascontiguousarray(
            artifact["body_displacement_mm"].astype(np.float32)
        )

    repaired_cache: dict[tuple[float, float], VectorArray] = {}
    records: list[S12HeuristicRecord] = []
    for variant in screen.variants:
        started = time.perf_counter()
        cache_key = (variant.lung_smoothing_sigma_mm, variant.lung_scale)
        repaired = repaired_cache.get(cache_key)
        if repaired is None:
            repaired = smooth_scale_field(
                initial,
                data.spacing,
                smoothing_sigma_mm=variant.lung_smoothing_sigma_mm,
                scale=variant.lung_scale,
            )
            repaired_cache[cache_key] = repaired
        lung, variant_body, normal, core = build_s12_variant_fields(
            initial,
            body,
            data.fixed_lung_mask,
            data.spacing,
            variant,
            repaired_lung_displacement_mm=repaired,
        )
        records.append(
            evaluate_s12_variant(
                data,
                lung,
                variant_body,
                normal,
                core,
                variant=variant,
                screen=screen,
                protocol=protocol,
                transform_direction=transform_direction,
                elapsed_s=time.perf_counter() - started,
            )
        )
    return tuple(records)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _code_version(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_s12_heuristic_screen(
    records: tuple[S12HeuristicRecord, ...],
    output_dir: Path,
    *,
    failures: tuple[str, ...],
    input_batch_manifest_path: Path,
    input_field_sha256: dict[str, str],
    screen_config_path: Path,
    gate_config_path: Path,
    screen: S12HeuristicScreen,
    repo_root: Path,
    batch_elapsed_s: float,
) -> Path:
    """Write subject-level rows and provenance without writing derived fields."""
    if not records:
        raise ValueError("cannot write an empty S1.2 heuristic screen")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows = [record.csv_row() for record in records]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    input_manifest: Any = json.loads(
        input_batch_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(input_manifest, dict):
        raise ValueError("input batch manifest must be a JSON object")
    variant_pass_counts = {
        variant.name: sum(
            record.screen_criteria_pass
            for record in records
            if record.variant == variant.name
        )
        for variant in screen.variants
    }
    coupled_viable_variants = [
        variant.name
        for variant in screen.variants
        if variant.couple_normal
        and variant_pass_counts[variant.name]
        == len(screen.expected_completed_subject_ids)
    ]
    manifest = {
        "artifact_type": "s12_heuristic_model_class_screen",
        "usage": SCREEN_USAGE,
        "measurement_eligible": False,
        "expert_landmarks_used": False,
        "code_version": _code_version(repo_root),
        "screen_version": screen.screen_version,
        "input_batch_manifest": {
            "path": input_batch_manifest_path.name,
            "sha256": _sha256(input_batch_manifest_path),
            "code_version": input_manifest.get("code_version"),
            "all_pass": input_manifest.get("all_pass"),
        },
        "screen_config": {
            "path": screen_config_path.name,
            "sha256": _sha256(screen_config_path),
        },
        "gate_config": {
            "path": gate_config_path.name,
            "sha256": _sha256(gate_config_path),
        },
        "expected_completed_subject_ids": list(
            screen.expected_completed_subject_ids
        ),
        "expected_missing_subject_ids": list(screen.expected_missing_subject_ids),
        "verified_input_field_sha256": input_field_sha256,
        "variant_names": [variant.name for variant in screen.variants],
        "subject_count": len({record.subject_id for record in records}),
        "record_count": len(records),
        "variant_pass_counts": variant_pass_counts,
        "coupled_viable_variants": coupled_viable_variants,
        "simple_heuristic_class_viable": bool(coupled_viable_variants),
        "failures": list(failures),
        "batch_elapsed_s": batch_elapsed_s,
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "SCREEN_USAGE",
    "S12HeuristicRecord",
    "S12HeuristicScreen",
    "S12HeuristicVariant",
    "build_s12_variant_fields",
    "evaluate_s12_variant",
    "load_s12_heuristic_screen",
    "screen_real_s1_fields",
    "smooth_scale_field",
    "write_s12_heuristic_screen",
]
