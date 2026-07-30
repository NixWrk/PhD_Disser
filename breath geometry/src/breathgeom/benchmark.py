"""Subject-level paired respiratory registration benchmark.

The benchmark keeps independent expert landmarks separate from automatically
derived keypoints.  Only the former may satisfy the landmark part of the phase
comparison gate.  Dense transforms always map fixed expiration coordinates to
moving inspiration coordinates in the project's zero-origin RAS+ basis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from breathgeom.io.dirlab import (
    load_copdgene_landmarks,
    load_copdgene_locator,
)
from breathgeom.io.pairs import RespiratoryPair
from breathgeom.measure.elastix_registration import ElastixParams, register_elastix
from breathgeom.measure.registration import (
    DeformableResult,
    landmark_tre,
    mask_metrics,
    warp_mask,
)
from breathgeom.measure.segmentation import body_mask_3d, lung_mask_3d
from breathgeom.measure.wall import IntArray as WallIntArray
from breathgeom.measure.wall import load_ras

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
CTArray = npt.NDArray[np.int16]


@dataclass(frozen=True)
class PairData:
    """Two phases and their evaluation data on a shared local RAS+ grid."""

    pair: RespiratoryPair
    fixed_ras: CTArray
    moving_ras: CTArray
    spacing: tuple[float, float, float]
    fixed_lung_mask: BoolArray
    moving_lung_mask: BoolArray
    fixed_body_mask: BoolArray
    moving_body_mask: BoolArray
    fixed_expert_points_mm: FloatArray | None
    moving_expert_points_mm: FloatArray | None
    fixed_keypoints_mm: FloatArray | None
    moving_keypoints_mm: FloatArray | None


@dataclass(frozen=True)
class RegistrationGate:
    """Predeclared acceptance limits for phase-wise anatomical comparisons."""

    expert_tre_mean_max_mm: float = 2.0
    expert_tre_p95_max_mm: float = 5.0
    lung_dice_min: float = 0.90
    lung_surface_p95_max_mm: float = 5.0
    jacobian_p01_min: float = 0.10
    nonpositive_jacobian_fraction_max: float = 0.0


@dataclass(frozen=True)
class RegistrationRecord:
    """One de-identified benchmark row; the subject is the statistical unit."""

    dataset_id: str
    subject_id: str
    method: str
    coordinate_basis: str
    transform_direction: str
    elapsed_s: float
    lung_dice_before: float
    lung_dice_after: float
    lung_surface_p95_before_mm: float
    lung_surface_p95_after_mm: float
    body_dice_before: float
    body_dice_after: float
    body_surface_p95_before_mm: float
    body_surface_p95_after_mm: float
    expert_count: int
    expert_tre_before_mean_mm: float | None
    expert_tre_after_mean_mm: float | None
    expert_tre_after_median_mm: float | None
    expert_tre_after_p95_mm: float | None
    keypoint_count: int
    keypoint_tre_before_mean_mm: float | None
    keypoint_tre_after_mean_mm: float | None
    jacobian_min: float
    jacobian_p01: float
    nonpositive_jacobian_fraction: float
    gate_pass: bool
    gate_reasons: tuple[str, ...]

    def csv_row(self) -> dict[str, object]:
        values = asdict(self)
        values["gate_reasons"] = ";".join(self.gate_reasons)
        return values


@dataclass(frozen=True)
class BenchmarkRun:
    record: RegistrationRecord
    result: DeformableResult
    params: ElastixParams
    spacing: tuple[float, float, float]


def _same_grid(
    fixed: npt.NDArray[np.generic],
    moving: npt.NDArray[np.generic],
    fixed_spacing: tuple[float, float, float],
    moving_spacing: tuple[float, float, float],
) -> None:
    if fixed.shape != moving.shape:
        raise ValueError(f"phase grids differ: {fixed.shape} versus {moving.shape}")
    if not np.allclose(fixed_spacing, moving_spacing, atol=1e-6):
        raise ValueError(f"phase spacings differ: {fixed_spacing} versus {moving_spacing}")


def _read_csv_points(path: str, image_path: str) -> FloatArray:
    points = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path}: expected an N x 3 point table")

    # Learn2Reg CSV coordinates are voxel indices in the stored NIfTI.  Map
    # them through world RAS into the canonical array, then remove the NIfTI
    # origin because all project arrays use a local zero-origin physical grid.
    source = cast(nib.Nifti1Image, nib.load(image_path))
    canonical = cast(
        nib.Nifti1Image,
        nib.as_closest_canonical(source),  # type: ignore[no-untyped-call]
    )
    source_to_canonical = np.linalg.inv(canonical.affine) @ source.affine
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    canonical_voxels = (source_to_canonical @ homogeneous.T).T[:, :3]
    spacing = np.asarray(
        canonical.header.get_zooms()[:3],  # type: ignore[no-untyped-call]
        dtype=np.float64,
    )
    converted: FloatArray = canonical_voxels * spacing
    return converted


def _optional_points(path: str, image_path: str) -> FloatArray | None:
    return _read_csv_points(path, image_path) if path else None


def load_pair_data(pair: RespiratoryPair) -> PairData:
    """Load one manifest row without modifying any source file."""
    if not pair.complete:
        raise ValueError(f"{pair.dataset_id}/{pair.subject_id} is incomplete: {pair.missing}")

    fixed: CTArray
    moving: CTArray
    fixed_expert: FloatArray | None
    moving_expert: FloatArray | None
    fixed_keypoints: FloatArray | None
    moving_keypoints: FloatArray | None

    if pair.dataset_id == "dirlab_copdgene":
        # The larger inspiration image is the more reliable orientation anchor;
        # reuse it verbatim for expiration to prohibit phase-dependent flips.
        moving, moving_spacing, orientation = load_copdgene_locator(
            pair.moving_image, pair.subject_id
        )
        fixed, fixed_spacing, _ = load_copdgene_locator(
            pair.fixed_image, pair.subject_id, orientation=orientation
        )
        _same_grid(fixed, moving, fixed_spacing, moving_spacing)
        fixed_expert = load_copdgene_landmarks(
            pair.fixed_expert_landmarks, pair.subject_id, orientation
        )
        moving_expert = load_copdgene_landmarks(
            pair.moving_expert_landmarks, pair.subject_id, orientation
        )
        fixed_keypoints = None
        moving_keypoints = None
        spacing = fixed_spacing
        fixed_lung = lung_mask_3d(cast(WallIntArray, fixed))
        moving_lung = lung_mask_3d(cast(WallIntArray, moving))
    elif pair.dataset_id == "learn2reg_lungct":
        fixed_values, fixed_spacing = load_ras(Path(pair.fixed_image))
        moving_values, moving_spacing = load_ras(Path(pair.moving_image))
        fixed = cast(CTArray, fixed_values)
        moving = cast(CTArray, moving_values)
        _same_grid(fixed, moving, fixed_spacing, moving_spacing)
        fixed_mask_values, fixed_mask_spacing = load_ras(Path(pair.fixed_mask))
        moving_mask_values, moving_mask_spacing = load_ras(Path(pair.moving_mask))
        _same_grid(fixed_mask_values, moving_mask_values, fixed_mask_spacing, moving_mask_spacing)
        _same_grid(fixed, fixed_mask_values, fixed_spacing, fixed_mask_spacing)
        spacing = fixed_spacing
        fixed_lung = fixed_mask_values > 0
        moving_lung = moving_mask_values > 0
        fixed_expert = _optional_points(pair.fixed_expert_landmarks, pair.fixed_image)
        moving_expert = _optional_points(pair.moving_expert_landmarks, pair.moving_image)
        fixed_keypoints = _optional_points(pair.fixed_keypoints, pair.fixed_image)
        moving_keypoints = _optional_points(pair.moving_keypoints, pair.moving_image)
    else:
        raise ValueError(f"unsupported respiratory dataset: {pair.dataset_id}")

    return PairData(
        pair=pair,
        fixed_ras=fixed,
        moving_ras=moving,
        spacing=spacing,
        fixed_lung_mask=np.ascontiguousarray(fixed_lung),
        moving_lung_mask=np.ascontiguousarray(moving_lung),
        fixed_body_mask=body_mask_3d(cast(WallIntArray, fixed)),
        moving_body_mask=body_mask_3d(cast(WallIntArray, moving)),
        fixed_expert_points_mm=fixed_expert,
        moving_expert_points_mm=moving_expert,
        fixed_keypoints_mm=fixed_keypoints,
        moving_keypoints_mm=moving_keypoints,
    )


def _tre_summary(
    fixed: FloatArray | None,
    moving: FloatArray | None,
    result: DeformableResult,
    spacing: tuple[float, float, float],
) -> tuple[int, float | None, float | None, float | None, float | None]:
    if fixed is None or moving is None:
        return 0, None, None, None, None
    before = landmark_tre(fixed, moving)
    after = landmark_tre(fixed, moving, result.displacement_mm, spacing)
    return (
        len(after),
        float(np.mean(before)),
        float(np.mean(after)),
        float(np.median(after)),
        float(np.percentile(after, 95)),
    )


def evaluate_registration(
    data: PairData,
    result: DeformableResult,
    *,
    gate: RegistrationGate | None = None,
    method: str = "itk-elastix",
) -> RegistrationRecord:
    """Compute independent QC and the predeclared acceptance decision."""
    gate = gate or RegistrationGate()
    lung_before = mask_metrics(data.fixed_lung_mask, data.moving_lung_mask, data.spacing)
    lung_after = mask_metrics(data.fixed_lung_mask, result.warped_moving_mask, data.spacing)
    body_before = mask_metrics(data.fixed_body_mask, data.moving_body_mask, data.spacing)
    warped_body = warp_mask(data.moving_body_mask, result.displacement_mm, data.spacing)
    body_after = mask_metrics(data.fixed_body_mask, warped_body, data.spacing)
    expert = _tre_summary(
        data.fixed_expert_points_mm,
        data.moving_expert_points_mm,
        result,
        data.spacing,
    )
    keypoint = _tre_summary(
        data.fixed_keypoints_mm,
        data.moving_keypoints_mm,
        result,
        data.spacing,
    )

    reasons: list[str] = []
    if expert[0] == 0:
        reasons.append("no_independent_expert_landmarks")
    else:
        if expert[2] is None or expert[2] > gate.expert_tre_mean_max_mm:
            reasons.append("expert_tre_mean")
        if expert[4] is None or expert[4] > gate.expert_tre_p95_max_mm:
            reasons.append("expert_tre_p95")
    if lung_after.dice < gate.lung_dice_min:
        reasons.append("lung_dice")
    if lung_after.surface_p95_mm > gate.lung_surface_p95_max_mm:
        reasons.append("lung_surface_p95")
    if result.jacobian_p01 < gate.jacobian_p01_min:
        reasons.append("jacobian_p01")
    if result.nonpositive_jacobian_fraction > gate.nonpositive_jacobian_fraction_max:
        reasons.append("folding")

    return RegistrationRecord(
        dataset_id=data.pair.dataset_id,
        subject_id=data.pair.subject_id,
        method=method,
        coordinate_basis="local-zero-origin-RAS+_mm",
        transform_direction="fixed-expiration_to_moving-inspiration",
        elapsed_s=result.elapsed_s,
        lung_dice_before=lung_before.dice,
        lung_dice_after=lung_after.dice,
        lung_surface_p95_before_mm=lung_before.surface_p95_mm,
        lung_surface_p95_after_mm=lung_after.surface_p95_mm,
        body_dice_before=body_before.dice,
        body_dice_after=body_after.dice,
        body_surface_p95_before_mm=body_before.surface_p95_mm,
        body_surface_p95_after_mm=body_after.surface_p95_mm,
        expert_count=expert[0],
        expert_tre_before_mean_mm=expert[1],
        expert_tre_after_mean_mm=expert[2],
        expert_tre_after_median_mm=expert[3],
        expert_tre_after_p95_mm=expert[4],
        keypoint_count=keypoint[0],
        keypoint_tre_before_mean_mm=keypoint[1],
        keypoint_tre_after_mean_mm=keypoint[2],
        jacobian_min=result.jacobian_min,
        jacobian_p01=result.jacobian_p01,
        nonpositive_jacobian_fraction=result.nonpositive_jacobian_fraction,
        gate_pass=not reasons,
        gate_reasons=tuple(reasons),
    )


def run_registration_benchmark(
    data: PairData,
    *,
    params: ElastixParams | None = None,
    gate: RegistrationGate | None = None,
) -> BenchmarkRun:
    params = params or ElastixParams()
    result = register_elastix(
        cast(WallIntArray, data.fixed_ras),
        cast(WallIntArray, data.moving_ras),
        data.spacing,
        data.fixed_lung_mask,
        data.moving_lung_mask,
        params=params,
    )
    return BenchmarkRun(
        record=evaluate_registration(
            data,
            result,
            gate=gate,
            method=f"itk-elastix-{'-'.join(params.stages)}",
        ),
        result=result,
        params=params,
        spacing=data.spacing,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_benchmark_run(
    output_dir: Path,
    run: BenchmarkRun,
    *,
    pair_manifest: Path,
    code_version: str,
    save_field: bool = False,
) -> tuple[Path, Path | None]:
    """Write de-identified QC and optionally the dense field outside Git."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run.record.dataset_id}__{run.record.subject_id}"
    field_path: Path | None = None
    if save_field:
        field_path = output_dir / f"{stem}.field.npz"
        np.savez_compressed(
            field_path,
            displacement_mm=run.result.displacement_mm,
            grid_shape=np.asarray(run.result.displacement_mm.shape[:3], dtype=np.int32),
            spacing_mm=np.asarray(run.spacing, dtype=np.float64),
            direction=np.asarray("fixed-expiration_to_moving-inspiration"),
            coordinate_basis=np.asarray("local-zero-origin-RAS+_mm"),
        )

    json_path = output_dir / f"{stem}.json"
    payload: dict[str, Any] = {
        "record": asdict(run.record),
        "parameters": asdict(run.params),
        "provenance": {
            "pair_manifest_sha256": _sha256(pair_manifest),
            "code_version": code_version,
            "field_file": field_path.name if field_path is not None else None,
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return json_path, field_path


def write_benchmark_csv(path: Path, records: list[RegistrationRecord]) -> None:
    """Write one row per subject, never one row per landmark or voxel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("cannot write an empty benchmark")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].csv_row()))
        writer.writeheader()
        writer.writerows(record.csv_row() for record in records)


__all__ = [
    "BenchmarkRun",
    "PairData",
    "RegistrationGate",
    "RegistrationRecord",
    "evaluate_registration",
    "load_pair_data",
    "run_registration_benchmark",
    "write_benchmark_csv",
    "write_benchmark_run",
]
