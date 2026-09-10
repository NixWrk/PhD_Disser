#!/usr/bin/env python
"""Prepare a reproducible four-chamber blood-pool review package.

The input is an explicit, ordered JSON manifest.  For every phase it names
one CT image, the four raw ``heartchambers_highres`` masks and, optionally, a
raw ``total/heart`` mask.  No file is discovered by globbing and no spatial,
morphological or temporal correction is applied.

The four raw chamber masks and their Boolean union are the primary candidate
initialisation.  A four-label NIfTI map is written only when the raw masks are
pairwise disjoint.  If they overlap, the package contains the individual raw
copies and an overlap report, but no priority-based label assignment.  The
intersection with ``total/heart`` is written only as a QC variant; it is never
used as the final blood mask.

This is research software for method development, not a clinical validation
or clinical segmentation tool.  Output directories must be supplied
explicitly and should be outside the repository because reports and masks may
remain derived medical data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

CHAMBERS: tuple[tuple[str, int, str], ...] = (
    ("left_atrium_blood", 1, "left_atrium_blood"),
    ("left_ventricle_blood", 2, "left_ventricle_blood"),
    ("right_atrium_blood", 3, "right_atrium_blood"),
    ("right_ventricle_blood", 4, "right_ventricle_blood"),
)
CHAMBER_IDS = tuple(item[0] for item in CHAMBERS)
CHAMBER_LABELS = {item[0]: item[1] for item in CHAMBERS}
CHAMBER_MANIFEST_KEYS = {item[0]: item[2] for item in CHAMBERS}
CSV_FIELDS = (
    "subject_id",
    "phase_id",
    "order_index",
    "structure",
    "source_kind",
    "voxel_count",
    "volume_ml",
    "is_empty",
    "connected_components_26",
    "pairwise_overlap_voxels",
    "pairwise_overlap_volume_ml",
    "voxels_outside_total_heart",
    "outside_total_heart_volume_ml",
    "containment_fraction_in_total_heart",
    "previous_phase_id",
    "previous_volume_ml",
    "adjacent_signed_volume_change_ml",
    "adjacent_absolute_volume_change_ml",
    "adjacent_relative_volume_change_fraction",
    "status",
)
QUEUE_FIELDS = (
    "rank",
    "subject_id",
    "phase_id",
    "order_index",
    "structure",
    "priority_score",
    "flags",
    "reason",
    "source_status",
    "requires_manual_review",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_filename(path: Path, root: Path | None = None) -> str:
    """Return a portable provenance label without leaking a local path."""

    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def _assert_external_output_dir(output_dir: Path) -> Path:
    """Reject accidental writes below this repository's working tree."""

    resolved = output_dir.resolve()
    code_dir = Path(__file__).resolve().parent
    for ancestor in (code_dir, *code_dir.parents):
        if (ancestor / ".git").exists():
            try:
                resolved.relative_to(ancestor)
            except ValueError:
                return resolved
            raise ValueError(
                "output-dir must be outside the repository because review masks "
                "and reports may contain derived medical data"
            )
    return resolved


def _as_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a non-empty path string")
    path = Path(value)
    if path.suffix.lower() not in {".nii", ".gz"} or not (
        path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")
    ):
        raise ValueError(f"{context}: expected .nii or .nii.gz, got {path.name!r}")
    return path


def _resolve_input(manifest_dir: Path, value: Any, context: str) -> Path:
    path = _as_path(value, context)
    resolved = (path if path.is_absolute() else manifest_dir / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context}: file not found: {resolved}")
    return resolved


def _affine_is_orthogonal(affine: np.ndarray, atol: float = 1e-4) -> bool:
    axes = np.asarray(affine[:3, :3], dtype=np.float64)
    lengths = np.linalg.norm(axes, axis=0)
    if np.any(lengths <= 0):
        return False
    normalized = axes / lengths
    return bool(np.allclose(normalized.T @ normalized, np.eye(3), atol=atol))


def _coded_forms_are_valid(image: nib.Nifti1Image, context: str) -> dict[str, int]:
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(float(np.linalg.det(affine[:3, :3]))) <= 0:
        raise ValueError(f"{context}: invalid or singular affine")
    if not _affine_is_orthogonal(affine):
        raise ValueError(f"{context}: sheared voxel axes are unsupported")
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    qcode = int(qform_code)
    scode = int(sform_code)
    if qcode == 0 and scode == 0:
        raise ValueError(f"{context}: both qform and sform codes are unset")
    if qcode > 0 and not np.allclose(qform, affine, atol=1e-4):
        raise ValueError(f"{context}: qform differs from selected affine")
    if scode > 0 and not np.allclose(sform, affine, atol=1e-4):
        raise ValueError(f"{context}: sform differs from selected affine")
    return {"qform_code": qcode, "sform_code": scode}


def _spatial_unit(image: nib.Nifti1Image, context: str, geometry_mm_confirmed: bool) -> str:
    unit = image.header.get_xyzt_units()[0] or "unknown"
    if unit != "mm" and not geometry_mm_confirmed:
        raise ValueError(
            f"{context}: spatial unit is {unit!r}; use the explicit "
            "geometry_mm_confirmed manifest flag only after checking DICOM provenance"
        )
    return str(unit)


def _load_image(
    path: Path,
    context: str,
    *,
    kind: str,
    geometry_mm_confirmed: bool,
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{context}: expected a 3D NIfTI, got shape {image.shape}")
    forms = _coded_forms_are_valid(image, context)
    unit = _spatial_unit(image, context, geometry_mm_confirmed)
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{context}: non-finite voxel values")
    if kind == "mask":
        if np.issubdtype(data.dtype, np.integer):
            if int(data.min()) < 0 or int(data.max()) > 1:
                raise ValueError(f"{context}: expected a binary 0/1 mask")
        elif not np.all((data == 0) | (data == 1)):
            raise ValueError(f"{context}: expected a binary 0/1 mask")
        array = np.asarray(data == 1, dtype=bool)
    elif kind == "ct":
        array = np.asarray(data)
    else:  # pragma: no cover - private API guard
        raise ValueError(f"{context}: unsupported image kind {kind!r}")
    metadata = {
        "source_filename": path.name,
        "sha256": sha256_file(path),
        "shape": list(image.shape),
        "orientation": list(nib.aff2axcodes(image.affine)),
        "voxel_spacing_mm": [float(value) for value in nib.affines.voxel_sizes(image.affine)],
        "spatial_unit_in_header": unit,
        "qform_code": forms["qform_code"],
        "sform_code": forms["sform_code"],
        "absolute_path_omitted": True,
    }
    return image, array, metadata


def assert_same_grid(
    reference: nib.Nifti1Image,
    candidate: nib.Nifti1Image,
    context: str,
) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference.affine, candidate.affine, atol=1e-4
    ):
        raise ValueError(f"{context}: images are not on the same CT grid")
    ref_unit = reference.header.get_xyzt_units()[0] or "unknown"
    cand_unit = candidate.header.get_xyzt_units()[0] or "unknown"
    if ref_unit != cand_unit:
        raise ValueError(f"{context}: spatial units differ ({ref_unit!r} != {cand_unit!r})")


def _voxel_volume_ml(image: nib.Nifti1Image) -> float:
    return abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    return int(ndimage.label(mask, structure=structure)[1])


def _mask_stats(mask: np.ndarray, image: nib.Nifti1Image) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    voxel_volume_ml = _voxel_volume_ml(image)
    return {
        "voxel_count": count,
        "voxel_volume_ml": voxel_volume_ml,
        "volume_ml": count * voxel_volume_ml,
        "is_empty": count == 0,
        "connected_components_26": _component_count(mask),
    }


def _save_mask(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> dict[str, Any]:
    """Write a derived binary mask on the original grid, never resampling it."""

    image = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    image.header.set_xyzt_units("mm")
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(path))
    stats = _mask_stats(np.asarray(mask, dtype=bool), reference)
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        **stats,
        "derived_medical_data": True,
    }


def _save_labelmap(path: Path, labels: np.ndarray, reference: nib.Nifti1Image) -> dict[str, Any]:
    image = nib.Nifti1Image(np.asarray(labels, dtype=np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    image.header.set_xyzt_units("mm")
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(path))
    counts = {str(value): int(np.count_nonzero(labels == value)) for value in range(5)}
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "label_values": counts,
        "derived_medical_data": True,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _embedded_manifest_provenance(
    provenance: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{context}: missing segmentation_provenance object")
    backend = provenance.get("backend") or provenance.get("software_name")
    version = provenance.get("backend_version") or provenance.get("software_version")
    if not backend or not version:
        raise ValueError(f"{context}: segmentation provenance needs backend and version")
    source_filename = provenance.get("source_filename")
    exact_sha = provenance.get("sha256")
    if not isinstance(source_filename, str) or not source_filename.strip():
        raise ValueError(f"{context}: embedded provenance needs source_filename")
    if not isinstance(exact_sha, str) or len(exact_sha) != 64:
        raise ValueError(f"{context}: embedded provenance.sha256 must be a SHA-256 hex string")
    output = {
        "source_filename": Path(source_filename).name,
        "sha256": exact_sha,
        "backend": backend,
        "backend_version": version,
        "models": provenance.get("models"),
        "run_id": provenance.get("run_id"),
        "input_manifest_sha256": exact_sha,
        "absolute_path_omitted": True,
    }
    return output


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"manifest: schema_version must be {SCHEMA_VERSION}")
    if not manifest.get("analysis_plan_locked_before_review", False):
        raise ValueError("manifest: analysis_plan_locked_before_review must be true")
    subject_id = manifest.get("subject_id")
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("manifest: subject_id must be a non-empty pseudonym")
    phases = manifest.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("manifest: phases must be a non-empty ordered list")
    seen: set[str] = set()
    for index, phase in enumerate(phases):
        context = f"manifest phases[{index}]"
        if not isinstance(phase, Mapping):
            raise ValueError(f"{context}: expected an object")
        phase_id = phase.get("phase_id")
        if not isinstance(phase_id, str) or not phase_id.strip():
            raise ValueError(f"{context}: phase_id must be a non-empty string")
        if phase_id in seen:
            raise ValueError(f"manifest: duplicate phase_id {phase_id!r}")
        seen.add(phase_id)
        _as_path(phase.get("ct_path"), f"{context} ct_path")
        masks = phase.get("chamber_masks")
        if not isinstance(masks, Mapping):
            raise ValueError(f"{context}: chamber_masks object is required")
        for chamber_id in CHAMBER_IDS:
            if chamber_id not in masks:
                raise ValueError(f"{context}: missing chamber_masks[{chamber_id!r}]")
            _as_path(masks[chamber_id], f"{context} chamber_masks[{chamber_id!r}]")
        total_path = phase.get("total_heart_path")
        if total_path is not None:
            _as_path(total_path, f"{context} total_heart_path")
    thresholds = manifest.get("review_priority_thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("manifest: review_priority_thresholds must be an object")
    for key, default in {
        "outside_total_fraction_alert": 0.05,
        "adjacent_relative_change_alert": 0.25,
        "components_alert": 1,
    }.items():
        value = thresholds.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"manifest: threshold {key!r} must be a finite non-negative number")


def _pairwise_overlap_stats(
    chamber_masks: Mapping[str, np.ndarray],
    image: nib.Nifti1Image,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    pairwise: dict[str, dict[str, Any]] = {}
    per_chamber_voxels = {chamber_id: 0 for chamber_id in CHAMBER_IDS}
    for index, left_id in enumerate(CHAMBER_IDS):
        for right_id in CHAMBER_IDS[index + 1 :]:
            overlap = chamber_masks[left_id] & chamber_masks[right_id]
            count = int(np.count_nonzero(overlap))
            pair_key = f"{left_id}__{right_id}"
            pairwise[pair_key] = {
                "chamber_a": left_id,
                "chamber_b": right_id,
                "overlap_voxels": count,
                "overlap_volume_ml": count * _voxel_volume_ml(image),
                "fraction_of_a": (
                    count / int(np.count_nonzero(chamber_masks[left_id]))
                    if np.count_nonzero(chamber_masks[left_id])
                    else None
                ),
                "fraction_of_b": (
                    count / int(np.count_nonzero(chamber_masks[right_id]))
                    if np.count_nonzero(chamber_masks[right_id])
                    else None
                ),
            }
            per_chamber_voxels[left_id] += count
            per_chamber_voxels[right_id] += count
    return pairwise, per_chamber_voxels


def _boundary_hu_qc(
    mask: np.ndarray,
    ct: np.ndarray,
    spacing_mm: Sequence[float],
) -> dict[str, Any]:
    """Return descriptive one-voxel shell statistics; never alters a mask."""

    if mask.shape != ct.shape:
        raise ValueError("CT and mask shape mismatch for boundary QC")
    structure = np.ones((3, 3, 3), dtype=bool)
    inner = mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)
    outer = ndimage.binary_dilation(mask, structure=structure) & ~mask
    inner_values = np.asarray(ct[inner], dtype=np.float64)
    outer_values = np.asarray(ct[outer], dtype=np.float64)
    inner_values = inner_values[np.isfinite(inner_values)]
    outer_values = outer_values[np.isfinite(outer_values)]
    if not len(inner_values) or not len(outer_values):
        return {
            "enabled": True,
            "status": "insufficient_finite_shell_voxels",
            "inside_boundary_voxels": int(len(inner_values)),
            "outside_boundary_voxels": int(len(outer_values)),
            "inside_median_hu": None,
            "outside_median_hu": None,
            "absolute_median_hu_difference": None,
            "interpretation": "QC ranking only; not an automated boundary or truth decision",
        }
    inside_median = float(np.median(inner_values))
    outside_median = float(np.median(outer_values))
    return {
        "enabled": True,
        "status": "descriptive_shell_statistics",
        "inside_boundary_voxels": int(len(inner_values)),
        "outside_boundary_voxels": int(len(outer_values)),
        "inside_median_hu": inside_median,
        "outside_median_hu": outside_median,
        "absolute_median_hu_difference": abs(inside_median - outside_median),
        "interpretation": "QC ranking only; not an automated boundary or truth decision",
        "voxel_spacing_mm": [float(value) for value in spacing_mm],
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def _phase_paths(
    phase: Mapping[str, Any],
    manifest_dir: Path,
    context: str,
) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {
        "ct": _resolve_input(manifest_dir, phase["ct_path"], f"{context} ct_path"),
        "total": None,
    }
    if phase.get("total_heart_path") is not None:
        paths["total"] = _resolve_input(
            manifest_dir, phase["total_heart_path"], f"{context} total_heart_path"
        )
    masks = phase["chamber_masks"]
    for chamber_id in CHAMBER_IDS:
        paths[chamber_id] = _resolve_input(
            manifest_dir,
            masks[chamber_id],
            f"{context} chamber_masks[{chamber_id!r}]",
        )
    return paths


def _make_labelmap(chamber_masks: Mapping[str, np.ndarray]) -> np.ndarray:
    overlap = np.zeros_like(next(iter(chamber_masks.values())), dtype=bool)
    labels = np.zeros_like(overlap, dtype=np.uint8)
    for chamber_id in CHAMBER_IDS:
        mask = chamber_masks[chamber_id]
        if np.any(labels & mask):
            raise ValueError("cannot create four-label map from overlapping chamber masks")
        labels[mask] = CHAMBER_LABELS[chamber_id]
        overlap |= mask
    return labels


def _write_raw_copy(
    path: Path,
    mask: np.ndarray,
    reference: nib.Nifti1Image,
) -> dict[str, Any]:
    return _save_mask(path, mask, reference)


def _phase_row(
    *,
    subject_id: str,
    phase_id: str,
    order_index: int,
    structure: str,
    source_kind: str,
    stats: Mapping[str, Any],
    overlap_voxels: int | None,
    outside_total: int | None,
    total_voxels: int | None,
    previous_phase_id: str | None,
    previous_volume_ml: float | None,
    previous_empty: bool | None = None,
    status: str,
) -> dict[str, Any]:
    current_volume = stats.get("volume_ml")
    signed_change = (
        current_volume - previous_volume_ml
        if current_volume is not None and previous_volume_ml is not None
        else None
    )
    absolute_change = abs(signed_change) if signed_change is not None else None
    relative_change = (
        _safe_ratio(signed_change, previous_volume_ml)
        if signed_change is not None and previous_volume_ml not in (None, 0)
        else None
    )
    outside_ml = None
    containment = None
    if outside_total is not None:
        outside_ml = outside_total * float(stats["voxel_volume_ml"])
        if total_voxels is not None:
            containment = _safe_ratio(stats["voxel_count"] - outside_total, stats["voxel_count"])
    return {
        "subject_id": subject_id,
        "phase_id": phase_id,
        "order_index": order_index,
        "structure": structure,
        "source_kind": source_kind,
        "voxel_count": stats.get("voxel_count"),
        "volume_ml": current_volume,
        "is_empty": stats.get("is_empty"),
        "connected_components_26": stats.get("connected_components_26"),
        "pairwise_overlap_voxels": overlap_voxels,
        "pairwise_overlap_volume_ml": (
            overlap_voxels * (float(current_volume) / float(stats["voxel_count"]))
            if overlap_voxels is not None and stats.get("voxel_count")
            else (0.0 if overlap_voxels == 0 else None)
        ),
        "voxels_outside_total_heart": outside_total,
        "outside_total_heart_volume_ml": outside_ml,
        "containment_fraction_in_total_heart": containment,
        "previous_phase_id": previous_phase_id,
        "previous_volume_ml": previous_volume_ml,
        "adjacent_signed_volume_change_ml": signed_change,
        "adjacent_absolute_volume_change_ml": absolute_change,
        "adjacent_relative_volume_change_fraction": relative_change,
        "status": status,
    }


def _source_status(
    *,
    overlap_voxels: int,
    empty: bool,
    components: int,
    outside_total: int | None,
    total_present: bool,
) -> str:
    flags: list[str] = []
    if overlap_voxels:
        flags.append("overlap_blocking_labelmap")
    if empty:
        flags.append("empty_mask")
    if components > 1:
        flags.append("multiple_components")
    if total_present and outside_total:
        flags.append("outside_total_heart")
    return "qc_flags:" + ",".join(flags) if flags else "raw_candidate_requires_manual_review"


def _priority_queue(
    phase_reports: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    outside_limit = float(thresholds.get("outside_total_fraction_alert", 0.05))
    jump_limit = float(thresholds.get("adjacent_relative_change_alert", 0.25))
    component_limit = int(thresholds.get("components_alert", 1))
    items: list[dict[str, Any]] = []
    for phase in phase_reports:
        for structure, details in phase["chambers"].items():
            flags: list[str] = []
            scores: list[float] = []
            reasons: list[str] = []
            if details["stats"]["is_empty"]:
                flags.append("empty_mask")
                scores.append(100.0)
                reasons.append("raw chamber mask is empty")
            if details["stats"]["connected_components_26"] > component_limit:
                flags.append("multiple_components")
                scores.append(50.0 + min(30.0, 5.0 * details["stats"]["connected_components_26"]))
                reasons.append("raw chamber mask has multiple 26-connected components")
            if details["overlap_voxels"]:
                flags.append("pairwise_overlap")
                scores.append(100.0)
                reasons.append("pairwise overlap blocks unambiguous four-label export")
            if details["outside_total_fraction"] is not None and details["outside_total_fraction"] > outside_limit:
                flags.append("outside_total_heart")
                scores.append(70.0)
                reasons.append("part of chamber lies outside optional total/heart mask")
            if details["adjacent_relative_change_fraction"] is not None and abs(details["adjacent_relative_change_fraction"]) > jump_limit:
                flags.append("adjacent_volume_jump")
                scores.append(60.0)
                reasons.append("adjacent phase volume change exceeds review threshold")
            items.append(
                {
                    "rank": None,
                    "subject_id": phase["subject_id"],
                    "phase_id": phase["phase_id"],
                    "order_index": phase["order_index"],
                    "structure": structure,
                    "priority_score": max(scores, default=0.0),
                    "flags": ";".join(flags),
                    "reason": "; ".join(reasons) or "raw candidate has no automatic QC flag; manual review remains required",
                    "source_status": details["status"],
                    "requires_manual_review": True,
                }
            )
        union = phase["union"]
        union_flags: list[str] = []
        union_scores: list[float] = []
        union_reasons: list[str] = []
        if union["overlap_voxels"]:
            union_flags.append("pairwise_overlap")
            union_scores.append(100.0)
            union_reasons.append("union contains overlapping chamber assignments")
        if union["outside_total_fraction"] is not None and union["outside_total_fraction"] > outside_limit:
            union_flags.append("outside_total_heart")
            union_scores.append(70.0)
            union_reasons.append("blood union extends outside optional total/heart mask")
        items.append(
            {
                "rank": None,
                "subject_id": phase["subject_id"],
                "phase_id": phase["phase_id"],
                "order_index": phase["order_index"],
                "structure": "blood_pool_four_chambers_union",
                "priority_score": max(union_scores, default=0.0),
                "flags": ";".join(union_flags),
                "reason": "; ".join(union_reasons) or "raw union is the primary initialization; manual review remains required",
                "source_status": union["status"],
                "requires_manual_review": True,
            }
        )
    items.sort(key=lambda row: (-float(row["priority_score"]), int(row["order_index"]), str(row["structure"])))
    for rank, row in enumerate(items, start=1):
        row["rank"] = rank
    return items


def _load_segmentation_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"segmentation manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"segmentation manifest is not valid JSON: {path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("segmentation manifest must contain a JSON object")
    backend = manifest.get("backend") or manifest.get("software_name")
    version = manifest.get("backend_version") or manifest.get("software_version")
    if not backend or not version:
        raise ValueError("segmentation manifest needs backend and backend_version")
    return {
        "source_filename": path.name,
        "sha256": sha256_file(path),
        "backend": backend,
        "backend_version": version,
        "models": manifest.get("models"),
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "absolute_path_omitted": True,
    }


def prepare_review(
    manifest_path: Path,
    output_dir: Path,
    *,
    segmentation_manifest_path: Path | None = None,
    geometry_mm_confirmed: bool = False,
    boundary_hu_qc: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare the package and return its portable report.

    ``dry_run`` validates and inventories all input files but does not create
    the output directory or write any medical-derived masks/reports.
    """

    manifest_path = manifest_path.resolve()
    output_dir = _assert_external_output_dir(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"review manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("review manifest must contain a JSON object")
    _validate_manifest(manifest)
    manifest_dir = manifest_path.parent
    geometry_override = bool(geometry_mm_confirmed or manifest.get("geometry_mm_confirmed", False))
    subject_id = str(manifest["subject_id"])
    threshold_config = {
        "outside_total_fraction_alert": float(
            manifest.get("review_priority_thresholds", {}).get("outside_total_fraction_alert", 0.05)
        ),
        "adjacent_relative_change_alert": float(
            manifest.get("review_priority_thresholds", {}).get("adjacent_relative_change_alert", 0.25)
        ),
        "components_alert": int(
            manifest.get("review_priority_thresholds", {}).get("components_alert", 1)
        ),
    }

    segmentation_provenance = None
    if segmentation_manifest_path is not None:
        segmentation_provenance = _load_segmentation_manifest(segmentation_manifest_path.resolve())
    elif manifest.get("segmentation_provenance") is not None:
        segmentation_provenance = _embedded_manifest_provenance(
            manifest["segmentation_provenance"], "review manifest segmentation_provenance"
        )
    else:
        raise ValueError(
            "provide --segmentation-manifest or a segmentation_provenance object with backend/version"
        )

    phase_reports: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    previous_volumes: dict[str, float] = {}
    previous_phase_id: str | None = None
    for order_index, phase in enumerate(manifest["phases"]):
        phase_id = str(phase["phase_id"])
        context = f"phase {phase_id}"
        paths = _phase_paths(phase, manifest_dir, context)
        ct_image, ct_data, ct_metadata = _load_image(
            paths["ct"], f"{context} CT", kind="ct", geometry_mm_confirmed=geometry_override
        )
        ct_metadata["source_relative_path"] = _relative_or_filename(paths["ct"], manifest_dir)
        chamber_masks: dict[str, np.ndarray] = {}
        chamber_metadata: dict[str, dict[str, Any]] = {}
        for chamber_id in CHAMBER_IDS:
            image, mask, metadata = _load_image(
                paths[chamber_id], f"{context} {chamber_id}", kind="mask", geometry_mm_confirmed=geometry_override
            )
            assert_same_grid(ct_image, image, f"{context} {chamber_id}")
            chamber_masks[chamber_id] = mask
            metadata["source_relative_path"] = _relative_or_filename(paths[chamber_id], manifest_dir)
            chamber_metadata[chamber_id] = metadata
        total_image = total_mask = total_metadata = None
        if paths["total"] is not None:
            total_image, total_mask, total_metadata = _load_image(
                paths["total"], f"{context} total/heart", kind="mask", geometry_mm_confirmed=geometry_override
            )
            assert_same_grid(ct_image, total_image, f"{context} total/heart")
            total_metadata["source_relative_path"] = _relative_or_filename(paths["total"], manifest_dir)

        union = np.logical_or.reduce([chamber_masks[chamber_id] for chamber_id in CHAMBER_IDS])
        pairwise, overlap_by_chamber = _pairwise_overlap_stats(chamber_masks, ct_image)
        total_voxels = int(np.count_nonzero(total_mask)) if total_mask is not None else None
        outside_union = int(np.count_nonzero(union & ~total_mask)) if total_mask is not None else None
        union_stats = _mask_stats(union, ct_image)
        union_status = _source_status(
            overlap_voxels=int(np.count_nonzero(np.logical_or.reduce([
                chamber_masks[a] & chamber_masks[b]
                for index, a in enumerate(CHAMBER_IDS)
                for b in CHAMBER_IDS[index + 1 :]
            ]))),
            empty=bool(union_stats["is_empty"]),
            components=int(union_stats["connected_components_26"]),
            outside_total=outside_union,
            total_present=total_mask is not None,
        )
        phase_output = output_dir / phase_id
        if phase_output.exists() and not dry_run and not overwrite:
            raise FileExistsError(f"{phase_output}: refusing to overwrite existing review phase")
        if not dry_run:
            phase_output.mkdir(parents=True, exist_ok=True)
        output_products: dict[str, Any] = {}
        for chamber_id in CHAMBER_IDS:
            if not dry_run:
                output_products[chamber_id] = _write_raw_copy(
                    phase_output / f"raw_{chamber_id}.nii.gz",
                    chamber_masks[chamber_id],
                    ct_image,
                )
            else:
                output_products[chamber_id] = {"file": f"raw_{chamber_id}.nii.gz", "dry_run": True}
        if not dry_run:
            output_products["blood_pool_four_chambers_union"] = _save_mask(
                phase_output / "raw_blood_pool_four_chambers_union.nii.gz", union, ct_image
            )
        else:
            output_products["blood_pool_four_chambers_union"] = {
                "file": "raw_blood_pool_four_chambers_union.nii.gz",
                "dry_run": True,
            }

        overlap_voxels_total = int(sum(value["overlap_voxels"] for value in pairwise.values()))
        labelmap_status = "not_written_overlap_requires_manual_disambiguation"
        if overlap_voxels_total == 0:
            labels = _make_labelmap(chamber_masks)
            labelmap_status = "written_pairwise_disjoint_raw_masks"
            if not dry_run:
                output_products["blood_chambers_labelmap_raw"] = _save_labelmap(
                    phase_output / "raw_blood_chambers_labelmap.nii.gz", labels, ct_image
                )
            else:
                output_products["blood_chambers_labelmap_raw"] = {
                    "file": "raw_blood_chambers_labelmap.nii.gz",
                    "dry_run": True,
                }
        if total_mask is not None:
            inside_total = union & total_mask
            outside_total_mask = union & ~total_mask
            if not dry_run:
                output_products["blood_inside_total_heart_qc"] = _save_mask(
                    phase_output / "qc_blood_inside_total_heart.nii.gz", inside_total, ct_image
                )
                output_products["blood_outside_total_heart_qc"] = _save_mask(
                    phase_output / "qc_blood_outside_total_heart.nii.gz", outside_total_mask, ct_image
                )
            else:
                output_products["blood_inside_total_heart_qc"] = {
                    "file": "qc_blood_inside_total_heart.nii.gz",
                    "dry_run": True,
                }
                output_products["blood_outside_total_heart_qc"] = {
                    "file": "qc_blood_outside_total_heart.nii.gz",
                    "dry_run": True,
                }

        chamber_reports: dict[str, Any] = {}
        for chamber_id in CHAMBER_IDS:
            stats = _mask_stats(chamber_masks[chamber_id], ct_image)
            chamber_overlap = int(overlap_by_chamber[chamber_id])
            outside = int(np.count_nonzero(chamber_masks[chamber_id] & ~total_mask)) if total_mask is not None else None
            previous_volume = previous_volumes.get(chamber_id)
            relative_change = (
                _safe_ratio(stats["volume_ml"] - previous_volume, previous_volume)
                if previous_volume not in (None, 0)
                else None
            )
            status = _source_status(
                overlap_voxels=chamber_overlap,
                empty=bool(stats["is_empty"]),
                components=int(stats["connected_components_26"]),
                outside_total=outside,
                total_present=total_mask is not None,
            )
            boundary_qc = None
            if boundary_hu_qc:
                boundary_qc = _boundary_hu_qc(
                    chamber_masks[chamber_id],
                    ct_data,
                    nib.affines.voxel_sizes(ct_image.affine),
                )
            chamber_reports[chamber_id] = {
                "source": chamber_metadata[chamber_id],
                "stats": stats,
                "overlap_voxels": chamber_overlap,
                "overlap_volume_ml": chamber_overlap * _voxel_volume_ml(ct_image),
                "outside_total_heart_voxels": outside,
                "outside_total_fraction": _safe_ratio(outside, stats["voxel_count"]) if outside is not None else None,
                "containment_fraction_in_total_heart": _safe_ratio(
                    stats["voxel_count"] - outside, stats["voxel_count"]
                ) if outside is not None else None,
                "previous_phase_id": previous_phase_id,
                "previous_volume_ml": previous_volume,
                "adjacent_signed_volume_change_ml": (
                    stats["volume_ml"] - previous_volume if previous_volume is not None else None
                ),
                "adjacent_absolute_volume_change_ml": (
                    abs(stats["volume_ml"] - previous_volume) if previous_volume is not None else None
                ),
                "adjacent_relative_change_fraction": relative_change,
                "status": status,
                "boundary_hu_qc": boundary_qc,
            }
            phase_rows.append(
                _phase_row(
                    subject_id=subject_id,
                    phase_id=phase_id,
                    order_index=order_index,
                    structure=chamber_id,
                    source_kind="raw_heartchambers_highres_chamber",
                    stats=stats,
                    overlap_voxels=chamber_overlap,
                    outside_total=outside,
                    total_voxels=total_voxels,
                    previous_phase_id=previous_phase_id,
                    previous_volume_ml=previous_volume,
                    status=status,
                )
            )
            previous_volumes[chamber_id] = float(stats["volume_ml"])

        union_previous = None
        if phase_reports:
            union_previous = float(phase_reports[-1]["union"]["stats"]["volume_ml"])
        union_relative = (
            _safe_ratio(union_stats["volume_ml"] - union_previous, union_previous)
            if union_previous not in (None, 0)
            else None
        )
        phase_rows.append(
            _phase_row(
                subject_id=subject_id,
                phase_id=phase_id,
                order_index=order_index,
                structure="blood_pool_four_chambers_union",
                source_kind="raw_heartchambers_highres_blood_union",
                stats=union_stats,
                overlap_voxels=overlap_voxels_total,
                outside_total=outside_union,
                total_voxels=total_voxels,
                previous_phase_id=previous_phase_id,
                previous_volume_ml=union_previous,
                status=union_status,
            )
        )
        phase_reports.append(
            {
                "subject_id": subject_id,
                "phase_id": phase_id,
                "order_index": order_index,
                "source": {
                    "ct": ct_metadata,
                    "total_heart": total_metadata,
                    "chambers": chamber_metadata,
                    "absolute_paths_omitted": True,
                },
                "geometry": {
                    "shape": list(ct_image.shape),
                    "orientation": list(nib.aff2axcodes(ct_image.affine)),
                    "voxel_spacing_mm": [float(value) for value in nib.affines.voxel_sizes(ct_image.affine)],
                    "spatial_unit_in_header": ct_metadata["spatial_unit_in_header"],
                    "geometry_mm_confirmed_override": geometry_override,
                    "qform_code": ct_metadata["qform_code"],
                    "sform_code": ct_metadata["sform_code"],
                },
                "chambers": chamber_reports,
                "pairwise_overlaps": pairwise,
                "union": {
                    "stats": union_stats,
                    "overlap_voxels": overlap_voxels_total,
                    "outside_total_heart_voxels": outside_union,
                    "outside_total_fraction": _safe_ratio(outside_union, union_stats["voxel_count"]) if outside_union is not None else None,
                    "containment_fraction_in_total_heart": _safe_ratio(
                        union_stats["voxel_count"] - outside_union, union_stats["voxel_count"]
                    ) if outside_union is not None else None,
                    "previous_phase_id": previous_phase_id,
                    "previous_volume_ml": union_previous,
                    "adjacent_signed_volume_change_ml": (
                        union_stats["volume_ml"] - union_previous if union_previous is not None else None
                    ),
                    "adjacent_absolute_volume_change_ml": (
                        abs(union_stats["volume_ml"] - union_previous) if union_previous is not None else None
                    ),
                    "adjacent_relative_change_fraction": union_relative,
                    "status": union_status,
                },
                "labelmap": {
                    "status": labelmap_status,
                    "label_values": CHAMBER_LABELS,
                    "disjointness_required": True,
                },
                "products": output_products,
            }
        )
        previous_phase_id = phase_id

    queue = _priority_queue(phase_reports, threshold_config)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "automatic_blood_candidates_pending_manual_review",
        "clinical_validation": False,
        "analysis_scope": "blood_pool_four_chambers_per_phase_only; physiology_and_volume_curve_interpretation_deferred",
        "subject_id": subject_id,
        "input_manifest": {
            "source_filename": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "absolute_path_omitted": True,
        },
        "segmentation_provenance": segmentation_provenance,
        "definitions": {
            "primary_candidate": "raw independent LA/LV/RA/RV heartchambers_highres masks and their Boolean union",
            "fixed_labelmap": "LA=1, LV=2, RA=3, RV=4; written only if raw masks are pairwise disjoint",
            "overlap_policy": "never silently resolve; priority-based labels and morphology are forbidden",
            "total_intersection": "blood union AND total/heart, emitted only as QC variant and never promoted to truth",
            "containment": "reported relative to optional total/heart mask; missing total is explicit null, not assumed containment",
            "connected_components": "26-connectivity on the original voxel grid",
            "adjacent_phase_change": "manifest order; descriptive QC only, not physiology or gating",
            "boundary_hu_qc": "one-voxel CT shell statistics when enabled; QC prioritization only, no automated truth decision",
        },
        "review_priority_thresholds": threshold_config,
        "review_queue_policy": "all phase/chamber items are listed; rank is a deterministic QC prioritization, not an acceptance rule",
        "phases": phase_reports,
        "phase_measurements": len(phase_rows),
        "review_queue": queue,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "scipy": getattr(ndimage, "__name__", "unknown").split(".")[0],
            "dry_run": dry_run,
            "boundary_hu_qc_enabled": boundary_hu_qc,
        },
        "privacy": {
            "absolute_paths_omitted": True,
            "source_images_copied": False,
            "derived_masks_may_contain_medical_data": True,
            "do_not_commit_without_explicit_review": True,
        },
        "limitations": [
            "Raw TotalSegmentator masks are candidates, not independent manual references.",
            "No morphology, temporal smoothing, interpolation, or overlap arbitration is applied.",
            "A total/heart mask does not define the blood truth; its intersection is a QC comparison only.",
            "The package does not assess physiological phase landmarks or explain chamber volume curves.",
            "Clinical validity is not established.",
        ],
    }
    if not dry_run:
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(f"output-dir is not a directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "blood_phase_measurements.csv", phase_rows, CSV_FIELDS)
        _write_csv(output_dir / "blood_review_queue.csv", queue, QUEUE_FIELDS)
        (output_dir / "blood_review_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Explicit ordered blood review manifest")
    parser.add_argument(
        "--segmentation-manifest",
        type=Path,
        help="Exact TotalSegmentator run manifest; its hash/backend/version are recorded",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="External derived-data output directory")
    parser.add_argument(
        "--geometry-mm-confirmed",
        action="store_true",
        help="Explicitly accept legacy unknown spatial units after independent DICOM provenance check",
    )
    parser.add_argument(
        "--boundary-hu-qc",
        action="store_true",
        help="Compute descriptive one-voxel CT shell statistics for QC prioritization only",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and inventory inputs without writing outputs")
    args = parser.parse_args(argv)
    report = prepare_review(
        args.manifest,
        args.output_dir,
        segmentation_manifest_path=args.segmentation_manifest,
        geometry_mm_confirmed=args.geometry_mm_confirmed,
        boundary_hu_qc=args.boundary_hu_qc,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "subject_id": report["subject_id"],
                "phases": len(report["phases"]),
                "phase_measurements": report["phase_measurements"],
                "review_queue_items": len(report["review_queue"]),
                "dry_run": bool(args.dry_run),
                "output_files": [] if args.dry_run else [
                    "blood_phase_measurements.csv",
                    "blood_review_queue.csv",
                    "blood_review_report.json",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
