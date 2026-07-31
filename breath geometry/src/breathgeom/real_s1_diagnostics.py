"""Post-hoc failure decomposition for a completed real S1.1 development batch."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from breathgeom.benchmark import PairData
from breathgeom.measure.registration import (
    VectorArray,
    acquisition_fov_mask,
    fov_aware_mask_metrics,
    jacobian_metrics,
    landmark_tre,
    warp_mask,
)
from breathgeom.measure.sliding_registration import lung_interface_normal

DIAGNOSTIC_USAGE = "posthoc_real_s1_failure_diagnosis_ONLY"
EXPECTED_FIELD_USAGE = "real_development_diagnostic_ONLY"


@dataclass(frozen=True)
class RealS1VariantRecord:
    """Metrics for one stored or deterministically derived lung-field variant."""

    dataset_id: str
    subject_id: str
    variant: str
    transform_direction: str
    keypoint_count: int
    lung_fov_dice: float
    lung_fov_surface_p95_mm: float
    keypoint_tre_mean_mm: float
    keypoint_tre_p95_mm: float
    lung_jacobian_p01: float
    lung_nonpositive_jacobian_fraction: float
    lung_displacement_p99_mm: float


def evaluate_lung_field_variant(
    data: PairData,
    displacement_mm: VectorArray,
    *,
    variant: str,
    fov_boundary_margin_mm: float,
    transform_direction: str = "fixed-expiration_to_moving-inspiration",
) -> RealS1VariantRecord:
    """Evaluate one diagnostic variant using no expert landmarks."""
    if (
        data.fixed_expert_points_mm is not None
        or data.moving_expert_points_mm is not None
    ):
        raise ValueError("expert landmarks are forbidden in S1.1 failure diagnosis")
    if data.fixed_keypoints_mm is None or data.moving_keypoints_mm is None:
        raise ValueError("image-derived keypoints are required for failure diagnosis")
    fixed_fov = acquisition_fov_mask(data.fixed_body_mask)
    moving_fov = acquisition_fov_mask(data.moving_body_mask)
    warped = warp_mask(data.moving_lung_mask, displacement_mm, data.spacing)
    common_fov = fixed_fov & warp_mask(moving_fov, displacement_mm, data.spacing)
    lung = fov_aware_mask_metrics(
        data.fixed_lung_mask,
        warped,
        data.spacing,
        valid_domain=common_fov,
        boundary_margin_mm=fov_boundary_margin_mm,
    )
    keypoints = landmark_tre(
        data.fixed_keypoints_mm,
        data.moving_keypoints_mm,
        displacement_mm,
        data.spacing,
    )
    topology = jacobian_metrics(
        displacement_mm,
        data.spacing,
        valid_domain=data.fixed_lung_mask & common_fov,
    )
    magnitude = np.linalg.norm(displacement_mm, axis=-1)
    return RealS1VariantRecord(
        dataset_id=data.pair.dataset_id,
        subject_id=data.pair.subject_id,
        variant=variant,
        transform_direction=transform_direction,
        keypoint_count=len(keypoints),
        lung_fov_dice=lung.dice,
        lung_fov_surface_p95_mm=lung.surface_p95_mm,
        keypoint_tre_mean_mm=float(np.mean(keypoints)),
        keypoint_tre_p95_mm=float(np.percentile(keypoints, 95)),
        lung_jacobian_p01=topology.p01,
        lung_nonpositive_jacobian_fraction=topology.nonpositive_fraction,
        lung_displacement_p99_mm=float(
            np.percentile(magnitude[data.fixed_lung_mask], 99)
        ),
    )


def diagnose_real_s1_fields(
    data: PairData,
    field_path: Path,
    *,
    fov_boundary_margin_mm: float,
    normal_smoothing_mm: float,
) -> tuple[RealS1VariantRecord, ...]:
    """Compare raw ConvexAdam, its normal projection, and final S1.1."""
    with np.load(field_path) as artifact:
        usage = str(artifact["usage"])
        if usage != EXPECTED_FIELD_USAGE:
            raise ValueError(f"{field_path}: unexpected field usage {usage!r}")
        if bool(artifact["measurement_eligible"]):
            raise ValueError(f"{field_path}: development field cannot be measurement eligible")
        transform_direction = str(artifact["transform_direction"])
        initial = np.ascontiguousarray(
            artifact["initial_lung_displacement_mm"].astype(np.float32)
        )
        final = np.ascontiguousarray(
            artifact["lung_displacement_mm"].astype(np.float32)
        )
    normal, _ = lung_interface_normal(
        data.fixed_lung_mask,
        data.spacing,
        smoothing_mm=normal_smoothing_mm,
    )
    normal_scalar = np.sum(initial * normal, axis=-1)
    normal_only = np.ascontiguousarray(
        (normal_scalar[..., None] * normal).astype(np.float32)
    )
    fields = (
        ("initial_convexadam", initial),
        ("normal_projection", normal_only),
        ("final_s1.1", final),
    )
    return tuple(
        evaluate_lung_field_variant(
            data,
            field,
            variant=name,
            fov_boundary_margin_mm=fov_boundary_margin_mm,
            transform_direction=transform_direction,
        )
        for name, field in fields
    )


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


def write_real_s1_diagnostics(
    records: tuple[RealS1VariantRecord, ...],
    output_dir: Path,
    *,
    failures: tuple[str, ...],
    input_batch_manifest_path: Path,
    gate_config_path: Path,
    repo_root: Path,
) -> Path:
    """Write a separate post-hoc artifact without mutating the frozen batch."""
    if not records:
        raise ValueError("cannot write empty real S1.1 diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "variant_summary.csv"
    rows = [asdict(record) for record in records]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    input_manifest: Any = json.loads(
        input_batch_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(input_manifest, dict):
        raise ValueError("input batch manifest must be a JSON object")
    manifest = {
        "artifact_type": "real_s1_posthoc_failure_diagnostics",
        "usage": DIAGNOSTIC_USAGE,
        "measurement_eligible": False,
        "expert_landmarks_used": False,
        "code_version": _code_version(repo_root),
        "input_batch_manifest": {
            "path": input_batch_manifest_path.name,
            "sha256": _sha256(input_batch_manifest_path),
            "code_version": input_manifest.get("code_version"),
            "all_pass": input_manifest.get("all_pass"),
        },
        "gate_config": {
            "path": gate_config_path.name,
            "sha256": _sha256(gate_config_path),
        },
        "variants": [
            "initial_convexadam",
            "normal_projection",
            "final_s1.1",
        ],
        "subject_count": len({record.subject_id for record in records}),
        "record_count": len(records),
        "failures": list(failures),
        "summary_sha256": _sha256(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "DIAGNOSTIC_USAGE",
    "RealS1VariantRecord",
    "diagnose_real_s1_fields",
    "evaluate_lung_field_variant",
    "write_real_s1_diagnostics",
]
