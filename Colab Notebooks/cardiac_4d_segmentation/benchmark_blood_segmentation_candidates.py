#!/usr/bin/env python
"""Benchmark explicitly named four-chamber blood-mask candidates.

The manifest names a CT phase, one or more LA/LV/RA/RV candidates, and an
optional independent manual reference for every phase/structure.  This
harness evaluates existing files only: it never registers, resamples,
smooths, morphologically edits, arbitrates overlaps, or ranks candidates.

With a blinded independent reference it reports volume error, Dice, surface
Dice at a locked tolerance, ASSD and HD95.  Without references it reports
technical QC only and sets ``no_accuracy_inference``.  Reports are research
artifacts, not clinical validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
CHAMBERS = (
    "left_atrium_blood",
    "left_ventricle_blood",
    "right_atrium_blood",
    "right_ventricle_blood",
)
METHOD_CATEGORIES = {
    "raw": "raw_automatic",
    "raw_automatic": "raw_automatic",
    "automatic": "raw_automatic",
    "assisted": "assisted",
    "semi_automatic": "assisted",
    "alternative": "alternative_algorithm",
    "alternative_algorithm": "alternative_algorithm",
}
METRIC_NAMES = (
    "signed_volume_error_ml",
    "absolute_volume_error_ml",
    "signed_volume_error_percent",
    "absolute_volume_error_percent",
    "dice",
    "surface_dice",
    "assd_mm",
    "hd95_mm",
)
ACCURACY_FIELDS = (
    "subject_id", "phase_id", "structure", "candidate_set_id", "candidate_id",
    "method_category", "software_name", "software_version", "candidate_mask_sha256",
    "reference_id", "reference_kind", "reference_mask_sha256",
    "candidate_independent_from_reference", "reference_independent_from_candidates",
    "reference_blinded_to_candidates", "accuracy_eligible", "surface_tolerance_mm",
    "candidate_volume_ml", "reference_volume_ml", *METRIC_NAMES,
    "candidate_empty", "reference_empty", "manual_interaction_time_s", "manual_clicks", "status",
)
TECHNICAL_FIELDS = (
    "record_type", "subject_id", "phase_id", "structure", "candidate_set_id", "candidate_id",
    "method_category", "software_name", "software_version", "mask_sha256", "voxel_count",
    "volume_ml", "is_empty", "connected_components_26", "pairwise_overlap_voxels",
    "pairwise_overlap_volume_ml", "outside_total_heart_voxels", "outside_total_fraction",
    "containment_fraction_in_total_heart", "previous_phase_id", "previous_volume_ml",
    "adjacent_signed_volume_change_ml", "adjacent_absolute_volume_change_ml",
    "adjacent_relative_volume_change_fraction", "boundary_inside_median_hu",
    "boundary_outside_median_hu", "boundary_absolute_median_hu_difference", "status",
)
BOOTSTRAP_FIELDS = (
    "analysis_type", "structure", "candidate_set_id", "candidate_id", "metric", "estimand",
    "estimate", "interval_lower", "interval_upper", "confidence_level", "bootstrap_iterations",
    "bootstrap_seed", "cluster_unit", "unique_subjects", "observations", "inference_scope", "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_external_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    code_dir = Path(__file__).resolve().parent
    for ancestor in (code_dir, *code_dir.parents):
        if (ancestor / ".git").exists():
            try:
                resolved.relative_to(ancestor)
            except ValueError:
                return resolved
            raise ValueError("output-dir must be outside the repository for derived medical reports")
    return resolved


def _resolve_path(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a non-empty path")
    path = Path(value)
    if not (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
        raise ValueError(f"{context}: expected .nii or .nii.gz")
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context}: file not found: {resolved}")
    return resolved


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _check_forms(image: nib.Nifti1Image, context: str) -> dict[str, int]:
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(float(np.linalg.det(affine[:3, :3]))) <= 0:
        raise ValueError(f"{context}: invalid or singular affine")
    axes = affine[:3, :3]
    lengths = np.linalg.norm(axes, axis=0)
    if np.any(lengths <= 0) or not np.allclose((axes / lengths).T @ (axes / lengths), np.eye(3), atol=1e-4):
        raise ValueError(f"{context}: sheared voxel axes are unsupported")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    qcode, scode = int(qcode), int(scode)
    if qcode == 0 and scode == 0:
        raise ValueError(f"{context}: both qform and sform codes are unset")
    if qcode > 0 and not np.allclose(qform, affine, atol=1e-4):
        raise ValueError(f"{context}: qform differs from selected affine")
    if scode > 0 and not np.allclose(sform, affine, atol=1e-4):
        raise ValueError(f"{context}: sform differs from selected affine")
    return {"qform_code": qcode, "sform_code": scode}


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    return int(ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))[1])


def _voxel_volume_ml(image: nib.Nifti1Image) -> float:
    return abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0


def _mask_stats(mask: np.ndarray, image: nib.Nifti1Image) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    return {
        "voxel_count": count,
        "volume_ml": count * _voxel_volume_ml(image),
        "is_empty": count == 0,
        "connected_components_26": _component_count(mask),
    }


def _load_nifti(
    path: Path,
    context: str,
    *,
    kind: str,
    label_value: int | float | None,
    geometry_mm_confirmed: bool,
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{context}: expected a 3D NIfTI, got {image.shape}")
    forms = _check_forms(image, context)
    unit = image.header.get_xyzt_units()[0] or "unknown"
    if unit != "mm" and not geometry_mm_confirmed:
        raise ValueError(f"{context}: spatial unit is {unit!r}; explicit geometry-mm-confirmed is required")
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{context}: non-finite voxel values")
    if kind == "ct":
        values = np.asarray(data)
        counts = volume = empty = components = None
    elif kind == "mask":
        if label_value is None:
            if np.issubdtype(data.dtype, np.integer):
                if int(data.min()) < 0 or int(data.max()) > 1:
                    raise ValueError(f"{context}: expected binary 0/1 mask or explicit label_value")
            elif not np.all((data == 0) | (data == 1)):
                raise ValueError(f"{context}: expected binary 0/1 mask or explicit label_value")
            values = np.asarray(data == 1, dtype=bool)
        else:
            values = np.asarray(data == label_value, dtype=bool)
        stats = _mask_stats(values, image)
        counts, volume = stats["voxel_count"], stats["volume_ml"]
        empty, components = stats["is_empty"], stats["connected_components_26"]
    else:
        raise ValueError(f"{context}: unsupported image kind {kind!r}")
    metadata = {
        "source_filename": path.name,
        "sha256": sha256_file(path),
        "shape": [int(v) for v in image.shape],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "voxel_spacing_mm": [float(v) for v in nib.affines.voxel_sizes(image.affine)],
        "spatial_unit_in_header": str(unit),
        "geometry_mm_confirmed_override": bool(geometry_mm_confirmed),
        "qform_code": forms["qform_code"],
        "sform_code": forms["sform_code"],
        "voxel_count": counts,
        "volume_ml": volume,
        "is_empty": empty,
        "connected_components_26": components,
        "absolute_path_omitted": True,
    }
    return image, values, metadata


def _assert_same_grid(left: nib.Nifti1Image, right: nib.Nifti1Image, context: str) -> None:
    if left.shape != right.shape or not np.allclose(left.affine, right.affine, atol=1e-4):
        raise ValueError(f"{context}: grid/affine mismatch; automatic resampling is forbidden")
    lu = left.header.get_xyzt_units()[0] or "unknown"
    ru = right.header.get_xyzt_units()[0] or "unknown"
    if lu != ru:
        raise ValueError(f"{context}: spatial units differ ({lu!r} != {ru!r})")


def _boundary_hu_qc(mask: np.ndarray, ct: np.ndarray) -> dict[str, Any]:
    structure = np.ones((3, 3, 3), dtype=bool)
    inside = mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)
    outside = ndimage.binary_dilation(mask, structure=structure) & ~mask
    inside_values = np.asarray(ct[inside], dtype=np.float64)
    outside_values = np.asarray(ct[outside], dtype=np.float64)
    inside_values = inside_values[np.isfinite(inside_values)]
    outside_values = outside_values[np.isfinite(outside_values)]
    if not len(inside_values) or not len(outside_values):
        return {"inside_median_hu": None, "outside_median_hu": None, "absolute_median_hu_difference": None, "status": "insufficient_shell_voxels", "interpretation": "QC only"}
    inside_median = float(np.median(inside_values))
    outside_median = float(np.median(outside_values))
    return {
        "inside_median_hu": inside_median,
        "outside_median_hu": outside_median,
        "absolute_median_hu_difference": abs(inside_median - outside_median),
        "inside_boundary_voxels": int(len(inside_values)),
        "outside_boundary_voxels": int(len(outside_values)),
        "status": "descriptive_shell_statistics",
        "interpretation": "QC only; not an automated boundary or truth decision",
    }


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _method_category(value: Any, context: str) -> str:
    if not isinstance(value, str) or value not in METHOD_CATEGORIES:
        raise ValueError(f"{context}: unsupported method_category {value!r}")
    return METHOD_CATEGORIES[value]


def _nonnegative_optional(value: Any, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{context}: expected a finite non-negative number or null")


def _validate_candidate(spec: Mapping[str, Any], context: str) -> None:
    required = ("candidate_id", "candidate_set_id", "path", "method_category", "software_name", "software_version")
    for key in required:
        if not isinstance(spec.get(key), str) or not str(spec[key]).strip():
            raise ValueError(f"{context}: missing non-empty {key!r}")
    _method_category(spec["method_category"], context)
    if not isinstance(spec.get("independent_from_reference"), bool):
        raise ValueError(f"{context}: independent_from_reference must be boolean")
    _nonnegative_optional(spec.get("manual_interaction_time_s"), context + " interaction time")
    _nonnegative_optional(spec.get("manual_clicks"), context + " clicks")
    if spec.get("provenance") is not None and not isinstance(spec.get("provenance"), Mapping):
        raise ValueError(f"{context}: provenance must be an object or null")


def _validate_reference(spec: Mapping[str, Any], context: str) -> None:
    for key in ("reference_id", "path", "reference_kind"):
        if not isinstance(spec.get(key), str) or not str(spec[key]).strip():
            raise ValueError(f"{context}: missing non-empty {key!r}")
    if spec.get("independent_from_candidates") is not True:
        raise ValueError(f"{context}: independent_from_candidates must be true")
    if spec.get("blinded_to_candidates") is not True:
        raise ValueError(f"{context}: blinded_to_candidates must be true")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"manifest: schema_version must be {SCHEMA_VERSION}")
    if manifest.get("analysis_plan_locked_before_evaluation") is not True:
        raise ValueError("manifest: analysis_plan_locked_before_evaluation must be true")
    tolerance = manifest.get("surface_tolerance_mm")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("manifest: surface_tolerance_mm must be positive and finite")
    if manifest.get("surface_tolerance_locked_before_evaluation") is not True:
        raise ValueError("manifest: surface_tolerance_locked_before_evaluation must be true")
    scope = manifest.get("inference_scope")
    if scope not in {"pilot_descriptive_only", "external_validation"}:
        raise ValueError("manifest: inference_scope must be pilot_descriptive_only or external_validation")
    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("manifest: bootstrap object is required")
    if int(bootstrap.get("iterations", 0)) <= 0:
        raise ValueError("manifest.bootstrap.iterations must be positive")
    confidence = float(bootstrap.get("confidence_level", 0))
    if not 0 < confidence < 1:
        raise ValueError("manifest.bootstrap.confidence_level must be in (0, 1)")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest.samples must be a non-empty list")
    seen_sample_ids: set[str] = set()
    seen_structures: set[tuple[str, str, str]] = set()
    seen_phase_keys: set[tuple[str, str]] = set()
    active_phase_key: tuple[str, str] | None = None
    seen_phase_keys: set[tuple[str, str]] = set()
    active_phase_key: tuple[str, str] | None = None
    for index, sample in enumerate(samples):
        context = f"manifest samples[{index}]"
        if not isinstance(sample, Mapping):
            raise ValueError(f"{context}: expected an object")
        for key in ("sample_id", "subject_id", "phase_id", "structure", "ct_path"):
            if not isinstance(sample.get(key), str) or not str(sample[key]).strip():
                raise ValueError(f"{context}: missing non-empty {key!r}")
        sample_id = str(sample["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"manifest: duplicate sample_id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        structure = str(sample["structure"])
        if structure not in CHAMBERS:
            raise ValueError(f"{context}: unsupported structure {structure!r}")
        structural_key = (str(sample["subject_id"]), str(sample["phase_id"]), structure)
        if structural_key in seen_structures:
            raise ValueError(f"manifest: duplicate subject/phase/structure {structural_key!r}")
        seen_structures.add(structural_key)
        phase_key = (str(sample["subject_id"]), str(sample["phase_id"]))
        if phase_key in seen_phase_keys and phase_key != active_phase_key:
            raise ValueError("manifest: records for one subject/phase must be contiguous")
        seen_phase_keys.add(phase_key)
        active_phase_key = phase_key
        phase_key = (str(sample["subject_id"]), str(sample["phase_id"]))
        if phase_key in seen_phase_keys and phase_key != active_phase_key:
            raise ValueError("manifest: records for one subject/phase must be contiguous")
        seen_phase_keys.add(phase_key)
        active_phase_key = phase_key
        candidates = sample.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{context}: candidates must be a non-empty list")
        ids: set[str] = set()
        sets: set[str] = set()
        for candidate_index, candidate in enumerate(candidates):
            candidate_context = f"{context} candidate[{candidate_index}]"
            if not isinstance(candidate, Mapping):
                raise ValueError(f"{candidate_context}: expected an object")
            _validate_candidate(candidate, candidate_context)
            candidate_id = str(candidate["candidate_id"])
            candidate_set_id = str(candidate["candidate_set_id"])
            if candidate_id in ids:
                raise ValueError(f"{context}: duplicate candidate_id {candidate_id!r}")
            if candidate_set_id in sets:
                raise ValueError(f"{context}: candidate_set_id repeated for this structure")
            ids.add(candidate_id)
            sets.add(candidate_set_id)
        reference = sample.get("manual_reference")
        if reference is not None:
            if not isinstance(reference, Mapping):
                raise ValueError(f"{context}: manual_reference must be an object or null")
            _validate_reference(reference, context + " manual_reference")


def _provenance(spec: Mapping[str, Any], manifest_dir: Path, context: str) -> dict[str, Any]:
    source = spec.get("provenance") or {}
    if not isinstance(source, Mapping):
        raise ValueError(f"{context}: provenance must be an object")
    result: dict[str, Any] = {
        "software_name": spec["software_name"],
        "software_version": spec["software_version"],
        "model_or_method": spec.get("model_or_method") or source.get("model_or_method"),
        "run_id": spec.get("run_id") or source.get("run_id"),
        "provenance_manifest": None,
    }
    path_value = spec.get("provenance_manifest_path") or source.get("manifest_path")
    if path_value is not None:
        path = _resolve_path(manifest_dir, path_value, context + " provenance_manifest_path")
        result["provenance_manifest"] = {
            "source_filename": path.name,
            "source_relative_path": _portable_path(path, manifest_dir),
            "sha256": sha256_file(path),
        }
    return result


def _load_sample(
    sample: Mapping[str, Any], manifest_dir: Path, geometry_mm_confirmed: bool, boundary_hu_qc: bool
) -> dict[str, Any]:
    subject_id, phase_id, structure = str(sample["subject_id"]), str(sample["phase_id"]), str(sample["structure"])
    context = f"{subject_id} {phase_id} {structure}"
    ct_path = _resolve_path(manifest_dir, sample["ct_path"], context + " ct_path")
    ct_image, ct_data, ct_meta = _load_nifti(ct_path, context + " CT", kind="ct", label_value=None, geometry_mm_confirmed=geometry_mm_confirmed)
    ct_meta["source_relative_path"] = _portable_path(ct_path, manifest_dir)
    total = None
    if sample.get("total_heart_path") is not None:
        total_path = _resolve_path(manifest_dir, sample["total_heart_path"], context + " total_heart_path")
        total_image, total_mask, total_meta = _load_nifti(total_path, context + " total/heart", kind="mask", label_value=sample.get("total_heart_label_value"), geometry_mm_confirmed=geometry_mm_confirmed)
        _assert_same_grid(ct_image, total_image, context + " total/heart")
        total_meta["source_relative_path"] = _portable_path(total_path, manifest_dir)
        total = {"image": total_image, "mask": total_mask, "metadata": total_meta}
    candidates = []
    for candidate in sample["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        path = _resolve_path(manifest_dir, candidate["path"], context + " candidate " + candidate_id)
        image, mask, metadata = _load_nifti(path, context + " candidate " + candidate_id, kind="mask", label_value=candidate.get("label_value"), geometry_mm_confirmed=geometry_mm_confirmed)
        _assert_same_grid(ct_image, image, context + " candidate " + candidate_id)
        metadata["source_relative_path"] = _portable_path(path, manifest_dir)
        candidates.append({
            "spec": candidate,
            "mask": mask,
            "image": image,
            "metadata": metadata,
            "mask_sha256": metadata["sha256"],
            "method_category": _method_category(candidate["method_category"], context + " candidate " + candidate_id),
            "provenance": _provenance(candidate, manifest_dir, context + " candidate " + candidate_id),
            "subject_id": subject_id,
            "phase_id": phase_id,
            "structure": structure,
            "total_mask": total["mask"] if total is not None else None,
            "stats": _mask_stats(mask, image),
            "boundary": _boundary_hu_qc(mask, ct_data) if boundary_hu_qc else None,
        })
    reference = None
    if sample.get("manual_reference") is not None:
        reference_spec = sample["manual_reference"]
        reference_path = _resolve_path(manifest_dir, reference_spec["path"], context + " manual_reference")
        reference_image, reference_mask, reference_meta = _load_nifti(reference_path, context + " manual_reference", kind="mask", label_value=reference_spec.get("label_value"), geometry_mm_confirmed=geometry_mm_confirmed)
        _assert_same_grid(ct_image, reference_image, context + " manual_reference")
        if not np.any(reference_mask):
            raise ValueError(f"{context}: manual reference must be non-empty")
        reference_meta["source_relative_path"] = _portable_path(reference_path, manifest_dir)
        reference = {"spec": reference_spec, "mask": reference_mask, "image": reference_image, "metadata": reference_meta}
    return {"sample": sample, "subject_id": subject_id, "phase_id": phase_id, "structure": structure, "ct": {"image": ct_image, "data": ct_data, "metadata": ct_meta}, "total": total, "candidates": candidates, "reference": reference}


@lru_cache(maxsize=1)
def _metric_backend():
    path = Path(__file__).with_name("validate_manual_cardiac_masks.py")
    spec = importlib.util.spec_from_file_location("_cardiac_validation_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load metric backend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compare_masks(candidate: np.ndarray, reference: np.ndarray, affine: np.ndarray, tolerance_mm: float) -> tuple[dict[str, Any], str]:
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    candidate_count = int(np.count_nonzero(candidate))
    reference_count = int(np.count_nonzero(reference))
    if not reference_count:
        raise ValueError("manual reference must be non-empty")
    voxel_volume_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000.0
    candidate_volume_ml = candidate_count * voxel_volume_ml
    reference_volume_ml = reference_count * voxel_volume_ml
    signed_ml = candidate_volume_ml - reference_volume_ml
    metrics: dict[str, Any] = {
        "candidate_volume_ml": candidate_volume_ml,
        "reference_volume_ml": reference_volume_ml,
        "signed_volume_error_ml": signed_ml,
        "absolute_volume_error_ml": abs(signed_ml),
        "signed_volume_error_percent": signed_ml / reference_volume_ml * 100.0,
        "absolute_volume_error_percent": abs(signed_ml / reference_volume_ml * 100.0),
        "dice": 2.0 * int(np.count_nonzero(candidate & reference)) / (candidate_count + reference_count),
        "surface_dice": None,
        "assd_mm": None,
        "hd95_mm": None,
    }
    if not candidate_count:
        return metrics, "candidate_empty_surface_metrics_undefined"
    backend = _metric_backend()
    try:
        surface = backend.surface_metrics(
            candidate,
            reference,
            tuple(float(v) for v in nib.affines.voxel_sizes(affine)),
            tolerance_mm,
        )
    except ValueError as error:
        return metrics, f"surface_metrics_undefined:{str(error)}"
    metrics.update({"surface_dice": surface["surface_dice"], "assd_mm": surface["assd_mm"], "hd95_mm": surface["hd95_mm"]})
    return metrics, "metrics_computed_compatible_with_validate_manual_cardiac_masks"


def _pairwise_overlaps(masks: Mapping[str, np.ndarray], image: nib.Nifti1Image) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    voxel_volume_ml = _voxel_volume_ml(image)
    pairwise: dict[str, dict[str, Any]] = {}
    by_chamber = {chamber: 0 for chamber in CHAMBERS}
    for index, left in enumerate(CHAMBERS):
        for right in CHAMBERS[index + 1 :]:
            count = int(np.count_nonzero(masks[left] & masks[right]))
            key = f"{left}__{right}"
            pairwise[key] = {
                "chamber_a": left,
                "chamber_b": right,
                "overlap_voxels": count,
                "overlap_volume_ml": count * voxel_volume_ml,
                "fraction_of_a": _ratio(count, np.count_nonzero(masks[left])),
                "fraction_of_b": _ratio(count, np.count_nonzero(masks[right])),
            }
            by_chamber[left] += count
            by_chamber[right] += count
    return pairwise, by_chamber


def _technical_row(item: Mapping[str, Any], *, record_type: str = "candidate", overlap_voxels: int | None = None, overlap_volume_ml: float | None = None, outside: int | None = None, previous: Mapping[str, Any] | None = None, boundary: Mapping[str, Any] | None = None, status: str) -> dict[str, Any]:
    spec, stats = item["spec"], item["stats"]
    previous = previous or {}
    boundary = boundary or {}
    signed_change = stats["volume_ml"] - previous["volume_ml"] if previous.get("volume_ml") is not None else None
    return {
        "record_type": record_type,
        "subject_id": item["subject_id"],
        "phase_id": item["phase_id"],
        "structure": item["structure"],
        "candidate_set_id": spec.get("candidate_set_id"),
        "candidate_id": spec.get("candidate_id"),
        "method_category": item.get("method_category"),
        "software_name": spec.get("software_name"),
        "software_version": spec.get("software_version"),
        "mask_sha256": item.get("mask_sha256"),
        "voxel_count": stats["voxel_count"],
        "volume_ml": stats["volume_ml"],
        "is_empty": stats["is_empty"],
        "connected_components_26": stats["connected_components_26"],
        "pairwise_overlap_voxels": overlap_voxels,
        "pairwise_overlap_volume_ml": overlap_volume_ml,
        "outside_total_heart_voxels": outside,
        "outside_total_fraction": _ratio(outside, stats["voxel_count"]),
        "containment_fraction_in_total_heart": _ratio(stats["voxel_count"] - outside, stats["voxel_count"]) if outside is not None else None,
        "previous_phase_id": previous.get("phase_id"),
        "previous_volume_ml": previous.get("volume_ml"),
        "adjacent_signed_volume_change_ml": signed_change,
        "adjacent_absolute_volume_change_ml": abs(signed_change) if signed_change is not None else None,
        "adjacent_relative_volume_change_fraction": _ratio(signed_change, previous.get("volume_ml")),
        "boundary_inside_median_hu": boundary.get("inside_median_hu"),
        "boundary_outside_median_hu": boundary.get("outside_median_hu"),
        "boundary_absolute_median_hu_difference": boundary.get("absolute_median_hu_difference"),
        "status": status,
    }


def _finalize_group(group: Mapping[str, Mapping[str, Any]], technical_rows: list[dict[str, Any]], technical_groups: list[dict[str, Any]], previous_volumes: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if not group:
        return {}
    first = next(iter(group.values()))
    present = {item["structure"]: item for item in group.values()}
    complete = all(chamber in present for chamber in CHAMBERS)
    pairwise: dict[str, dict[str, Any]] = {}
    overlap_by_chamber = {chamber: 0 for chamber in CHAMBERS}
    union = None
    outside_union = None
    total_mask = first.get("total_mask")
    if complete:
        masks = {chamber: present[chamber]["mask"] for chamber in CHAMBERS}
        pairwise, overlap_by_chamber = _pairwise_overlaps(masks, first["image"])
        union = np.logical_or.reduce(list(masks.values()))
        if total_mask is not None:
            outside_union = int(np.count_nonzero(union & ~total_mask))
    union_stats = _mask_stats(union, first["image"]) if union is not None else None
    status = "complete_four_chamber_set" if complete else "incomplete_four_chamber_set"
    if complete and any(v["overlap_voxels"] for v in pairwise.values()):
        status = "pairwise_overlap_requires_manual_review"
    if outside_union:
        status += ";outside_total_heart_qc"
    chamber_reports: dict[str, Any] = {}
    for chamber, item in present.items():
        outside = int(np.count_nonzero(item["mask"] & ~total_mask)) if total_mask is not None else None
        previous = previous_volumes.get((item["subject_id"], str(item["spec"]["candidate_set_id"]), chamber))
        chamber_reports[chamber] = {
            "candidate_id": item["spec"]["candidate_id"],
            "method_category": item["method_category"],
            "software_name": item["spec"]["software_name"],
            "software_version": item["spec"]["software_version"],
            "mask_sha256": item["mask_sha256"],
            "stats": item["stats"],
            "pairwise_overlap_voxels": overlap_by_chamber[chamber] if complete else None,
            "outside_total_heart_voxels": outside,
            "outside_total_fraction": _ratio(outside, item["stats"]["voxel_count"]),
            "containment_fraction_in_total_heart": _ratio(item["stats"]["voxel_count"] - outside, item["stats"]["voxel_count"]) if outside is not None else None,
            "previous_phase_id": previous.get("phase_id") if previous else None,
            "previous_volume_ml": previous.get("volume_ml") if previous else None,
            "adjacent_relative_volume_change_fraction": _ratio(item["stats"]["volume_ml"] - previous["volume_ml"], previous["volume_ml"]) if previous and previous.get("volume_ml") else None,
            "boundary_hu_qc": item.get("boundary"),
        }
    technical_groups.append({
        "subject_id": first["subject_id"],
        "phase_id": first["phase_id"],
        "candidate_set_id": first["spec"]["candidate_set_id"],
        "structures_present": sorted(present),
        "complete_four_chamber_set": complete,
        "chambers": chamber_reports,
        "pairwise_overlaps": pairwise,
        "union": {
            "stats": union_stats,
            "outside_total_heart_voxels": outside_union,
            "outside_total_fraction": _ratio(outside_union, union_stats["voxel_count"]) if union_stats else None,
            "status": status,
        },
        "interpretation": "technical QC only; no accuracy or physiological inference",
    })
    for chamber, item in present.items():
        outside = int(np.count_nonzero(item["mask"] & ~total_mask)) if total_mask is not None else None
        chamber_status = "technical_qc"
        if item["stats"]["is_empty"]:
            chamber_status += ";empty_mask"
        if item["stats"]["connected_components_26"] > 1:
            chamber_status += ";multiple_components"
        if overlap_by_chamber[chamber]:
            chamber_status += ";pairwise_overlap"
        previous = previous_volumes.get((item["subject_id"], item["spec"]["candidate_set_id"], chamber))
        technical_rows.append(_technical_row(item, overlap_voxels=overlap_by_chamber[chamber] if complete else None, overlap_volume_ml=overlap_by_chamber[chamber] * _voxel_volume_ml(item["image"]) if complete else None, outside=outside, previous=previous, boundary=item.get("boundary"), status=chamber_status))
    if union is not None:
        union_item = {**first, "structure": "blood_pool_four_chambers_union", "spec": {"candidate_set_id": first["spec"]["candidate_set_id"], "candidate_id": first["spec"]["candidate_set_id"] + "__union"}, "method_category": "derived_union", "mask_sha256": None, "stats": union_stats}
        previous = previous_volumes.get((first["subject_id"], first["spec"]["candidate_set_id"], "blood_pool_four_chambers_union"))
        technical_rows.append(_technical_row(union_item, record_type="set_union", overlap_voxels=sum(v["overlap_voxels"] for v in pairwise.values()), overlap_volume_ml=sum(v["overlap_volume_ml"] for v in pairwise.values()), outside=outside_union, previous=previous, status=status))
    for key, value in pairwise.items():
        technical_rows.append({
            "record_type": "pairwise_overlap", "subject_id": first["subject_id"], "phase_id": first["phase_id"], "structure": key, "candidate_set_id": first["spec"]["candidate_set_id"], "candidate_id": None, "method_category": "derived_qc", "software_name": None, "software_version": None, "mask_sha256": None, "voxel_count": None, "volume_ml": None, "is_empty": None, "connected_components_26": None, "pairwise_overlap_voxels": value["overlap_voxels"], "pairwise_overlap_volume_ml": value["overlap_volume_ml"], "outside_total_heart_voxels": None, "outside_total_fraction": None, "containment_fraction_in_total_heart": None, "previous_phase_id": None, "previous_volume_ml": None, "adjacent_signed_volume_change_ml": None, "adjacent_absolute_volume_change_ml": None, "adjacent_relative_volume_change_fraction": None, "boundary_inside_median_hu": None, "boundary_outside_median_hu": None, "boundary_absolute_median_hu_difference": None, "status": "technical_pairwise_overlap_qc",
        })
    return group


def _bootstrap_accuracy(rows: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, Any], inference_scope: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        if row.get("accuracy_eligible") is not True:
            continue
        key = (str(row["structure"]), str(row["candidate_set_id"]), str(row["candidate_id"]))
        subject = str(row["subject_id"])
        subject_values = groups.setdefault(key, {}).setdefault(subject, {})
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if value is not None and np.isfinite(float(value)):
                subject_values.setdefault(metric, []).append(float(value))
    iterations = int(bootstrap["iterations"])
    seed = int(bootstrap["seed"])
    confidence = float(bootstrap["confidence_level"])
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    output: list[dict[str, Any]] = []
    for (structure, candidate_set_id, candidate_id), subjects in sorted(groups.items()):
        for metric in METRIC_NAMES:
            means = [float(np.mean(subjects[subject][metric])) for subject in sorted(subjects) if metric in subjects[subject]]
            estimate = float(np.mean(means)) if means else None
            lower = upper = None
            if len(means) >= 2:
                values = np.asarray(means, dtype=np.float64)
                indices = rng.integers(0, len(values), size=(iterations, len(values)))
                draws = np.mean(values[indices], axis=1)
                lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0]).tolist()
                status = "pilot_cluster_bootstrap_interval_not_population_ci" if len(means) <= 3 or inference_scope == "pilot_descriptive_only" else "cluster_bootstrap_interval"
            else:
                status = "insufficient_subject_clusters_for_interval"
            output.append({
                "analysis_type": "candidate_accuracy",
                "structure": structure,
                "candidate_set_id": candidate_set_id,
                "candidate_id": candidate_id,
                "metric": metric,
                "estimand": "mean_of_subject_means",
                "estimate": estimate,
                "interval_lower": lower,
                "interval_upper": upper,
                "confidence_level": confidence,
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
                "cluster_unit": "subject_id",
                "unique_subjects": len(means),
                "observations": sum(len(subjects[subject].get(metric, [])) for subject in subjects),
                "inference_scope": inference_scope,
                "status": status,
            })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def benchmark(
    manifest_path: Path,
    output_dir: Path,
    *,
    geometry_mm_confirmed: bool = False,
    boundary_hu_qc: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    output_dir = _assert_external_output_dir(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("benchmark manifest must contain a JSON object")
    _validate_manifest(manifest)
    manifest_dir = manifest_path.parent
    geometry_override = bool(geometry_mm_confirmed or manifest.get("geometry_mm_confirmed", False))
    boundary_enabled = bool(boundary_hu_qc or manifest.get("boundary_hu_qc", False))
    tolerance_mm = float(manifest["surface_tolerance_mm"])
    subjects = {str(sample["subject_id"]) for sample in manifest["samples"]}
    inference_scope = str(manifest["inference_scope"])
    if len(subjects) <= 3 and inference_scope != "pilot_descriptive_only":
        raise ValueError("three or fewer subjects require inference_scope=pilot_descriptive_only")
    accuracy_rows: list[dict[str, Any]] = []
    technical_rows: list[dict[str, Any]] = []
    technical_groups: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    previous_volumes: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    group_buffers: dict[str, dict[str, dict[str, Any]]] = {}
    active_phase: tuple[str, str] | None = None
    references_seen = 0
    no_references_seen = 0

    def flush_groups() -> None:
        nonlocal group_buffers
        for group in group_buffers.values():
            _finalize_group(group, technical_rows, technical_groups, previous_volumes)
            for item in group.values():
                key = (item["subject_id"], str(item["spec"]["candidate_set_id"]), item["structure"])
                previous_volumes[key] = {
                    "phase_id": item["phase_id"],
                    "volume_ml": item["stats"]["volume_ml"],
                }
            if all(chamber in group for chamber in CHAMBERS):
                first = next(iter(group.values()))
                masks = {chamber: group[chamber]["mask"] for chamber in CHAMBERS}
                union = np.logical_or.reduce(list(masks.values()))
                previous_volumes[(first["subject_id"], str(first["spec"]["candidate_set_id"]), "blood_pool_four_chambers_union")] = {
                    "phase_id": first["phase_id"],
                    "volume_ml": _mask_stats(union, first["image"])["volume_ml"],
                }
        group_buffers = {}

    for sample in manifest["samples"]:
        loaded = _load_sample(sample, manifest_dir, geometry_override, boundary_enabled)
        sample_id = str(sample["sample_id"])
        phase_key = (loaded["subject_id"], loaded["phase_id"])
        if active_phase is not None and phase_key != active_phase:
            flush_groups()
        active_phase = phase_key
        reference = loaded["reference"]
        if reference is None:
            no_references_seen += 1
        else:
            references_seen += 1
        candidate_summary = []
        for item in loaded["candidates"]:
            spec = item["spec"]
            candidate_id = str(spec["candidate_id"])
            candidate_set_id = str(spec["candidate_set_id"])
            candidate_summary.append({
                "candidate_id": candidate_id,
                "candidate_set_id": candidate_set_id,
                "method_category": item["method_category"],
                "software_name": spec["software_name"],
                "software_version": spec["software_version"],
                "model_or_method": item["provenance"]["model_or_method"],
                "mask_sha256": item["metadata"]["sha256"],
                "source_relative_path": item["metadata"]["source_relative_path"],
                "provenance_manifest": item["provenance"]["provenance_manifest"],
            })
            group_buffers.setdefault(candidate_set_id, {})[loaded["structure"]] = item
            if reference is None:
                continue
            metrics, metric_status = _compare_masks(item["mask"], reference["mask"], item["image"].affine, tolerance_mm)
            candidate_independent = bool(spec["independent_from_reference"])
            reference_independent = bool(reference["spec"]["independent_from_candidates"])
            reference_blinded = bool(reference["spec"]["blinded_to_candidates"])
            eligible = candidate_independent and reference_independent and reference_blinded
            status = metric_status if eligible else "non_independent_comparison_not_accuracy"
            if not eligible:
                warnings.append(f"{sample_id}/{candidate_id}: excluded from accuracy bootstrap by independence flags")
            accuracy_rows.append({
                "subject_id": loaded["subject_id"],
                "phase_id": loaded["phase_id"],
                "structure": loaded["structure"],
                "candidate_set_id": candidate_set_id,
                "candidate_id": candidate_id,
                "method_category": item["method_category"],
                "software_name": spec["software_name"],
                "software_version": spec["software_version"],
                "candidate_mask_sha256": item["metadata"]["sha256"],
                "reference_id": reference["spec"]["reference_id"],
                "reference_kind": reference["spec"]["reference_kind"],
                "reference_mask_sha256": reference["metadata"]["sha256"],
                "candidate_independent_from_reference": candidate_independent,
                "reference_independent_from_candidates": reference_independent,
                "reference_blinded_to_candidates": reference_blinded,
                "accuracy_eligible": eligible,
                "surface_tolerance_mm": tolerance_mm,
                **metrics,
                "candidate_empty": item["stats"]["is_empty"],
                "reference_empty": reference["metadata"]["is_empty"],
                "manual_interaction_time_s": spec.get("manual_interaction_time_s"),
                "manual_clicks": spec.get("manual_clicks"),
                "status": status,
            })
        sample_summaries.append({
            "sample_id": sample_id,
            "subject_id": loaded["subject_id"],
            "phase_id": loaded["phase_id"],
            "structure": loaded["structure"],
            "candidate_count": len(loaded["candidates"]),
            "has_manual_reference": reference is not None,
            "ct": loaded["ct"]["metadata"],
            "total_heart": loaded["total"]["metadata"] if loaded["total"] is not None else None,
            "manual_reference": reference["metadata"] if reference is not None else None,
            "candidates": candidate_summary,
            "status": "accuracy_candidate_compared" if reference is not None else "technical_qc_only",
        })
    flush_groups()
    if references_seen and no_references_seen:
        analysis_mode, status = "mixed_accuracy_and_technical_only", "mixed_accuracy_and_technical_only"
    elif references_seen:
        analysis_mode, status = "accuracy", "accuracy_metrics_computed_pending_interpretation"
    else:
        analysis_mode, status = "technical_only", "no_accuracy_inference"
        warnings.append("No independent manual reference was supplied; accuracy metrics are unavailable")
    bootstrap_rows = _bootstrap_accuracy(accuracy_rows, manifest["bootstrap"], inference_scope) if accuracy_rows else []
    if not bootstrap_rows:
        warnings.append("No eligible accuracy rows were available for subject-cluster bootstrap")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": status,
        "clinical_validation": False,
        "analysis_mode": analysis_mode,
        "analysis_plan_locked_before_evaluation": True,
        "input_manifest": {"source_filename": manifest_path.name, "sha256": sha256_file(manifest_path), "absolute_path_omitted": True},
        "surface_tolerance_mm": tolerance_mm,
        "definitions": {
            "accuracy": "independent manual reference on the same original CT grid; candidate minus reference for volume error",
            "surface_metrics": "surface Dice, ASSD and HD95 use the locked tolerance/affine-compatible metric backend",
            "technical_only": "geometry, empty/components, chamber overlap, optional total/heart containment, optional CT shell QC and adjacent volume changes",
            "candidate_set_id": "groups one LA/LV/RA/RV family for phase-level technical QC",
            "efficiency": "interaction time and clicks are reported only when observed; no ranking by unobserved data",
            "forbidden_operations": "no morphology, registration, resampling, temporal smoothing or overlap arbitration",
        },
        "inference": {
            "scope": inference_scope,
            "cluster_unit": "subject_id",
            "unique_subjects": len(subjects),
            "population_generalization_allowed": False if len(subjects) <= 3 else inference_scope == "external_validation",
            "three_subject_warning": "three subjects are a pilot; any interval is descriptive and not a population confidence interval",
            "tta_or_ensemble_uncertainty": "QC prioritization only unless calibrated against manual reference",
        },
        "geometry": {"same_original_grid_required": True, "spatial_units_required": "mm unless explicit geometry-mm-confirmed override", "geometry_mm_confirmed_override": geometry_override},
        "samples": sample_summaries,
        "accuracy_rows": len(accuracy_rows),
        "technical_rows": len(technical_rows),
        "technical_groups": technical_groups,
        "bootstrap": bootstrap_rows,
        "review": {"boundary_hu_qc_enabled": boundary_enabled, "status": "QC prioritization only; no automated truth decision"},
        "warnings": warnings,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "nibabel": nib.__version__, "dry_run": dry_run},
        "privacy": {"absolute_paths_omitted": True, "source_masks_copied": False, "reports_may_be_derived_medical_data": True, "do_not_commit_without_explicit_review": True},
    }
    if not dry_run:
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(f"output-dir is not a directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            existing = [output_dir / name for name in ("benchmark_accuracy_metrics.csv", "benchmark_technical_metrics.csv", "benchmark_subject_bootstrap.csv", "benchmark_report.json") if (output_dir / name).exists()]
            if existing:
                raise FileExistsError(f"output-dir contains benchmark outputs; use --overwrite: {existing[0]}")
        _write_csv(output_dir / "benchmark_accuracy_metrics.csv", accuracy_rows, ACCURACY_FIELDS)
        _write_csv(output_dir / "benchmark_technical_metrics.csv", technical_rows, TECHNICAL_FIELDS)
        _write_csv(output_dir / "benchmark_subject_bootstrap.csv", bootstrap_rows, BOOTSTRAP_FIELDS)
        (output_dir / "benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="External derived-data output directory")
    parser.add_argument("--geometry-mm-confirmed", action="store_true", help="Accept legacy unknown units only after DICOM review")
    parser.add_argument("--boundary-hu-qc", action="store_true", help="Compute descriptive CT shell statistics for QC only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = benchmark(
        args.manifest,
        args.output_dir,
        geometry_mm_confirmed=args.geometry_mm_confirmed,
        boundary_hu_qc=args.boundary_hu_qc,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "status": report["status"],
        "analysis_mode": report["analysis_mode"],
        "samples": len(report["samples"]),
        "accuracy_rows": report["accuracy_rows"],
        "technical_rows": report["technical_rows"],
        "bootstrap_rows": len(report["bootstrap"]),
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
