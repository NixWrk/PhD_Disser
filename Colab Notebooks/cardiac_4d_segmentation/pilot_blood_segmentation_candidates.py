#!/usr/bin/env python
"""Generate and QC a locked 12-phase four-chamber blood-mask pilot.

This is an isolated research harness for method development. It accepts one
CT phase and four raw ``heartchambers_highres`` masks per manifest record and
can generate four explicitly named candidate families:

* ``raw`` — the raw TotalSegmentator chamber mask;
* ``conservative_cleanup`` — removal-only component filtering: with 26-
  connectivity, remove only components below the locked physical-volume
  threshold whose minimum physical distance from the largest component is
  above the locked distance threshold; no voxel is ever added;
* ``nninteractive_prompt_from_ts_geometry`` — nnInteractive with a positive
  point selected from raw-mask geometry, without an initial label;
* ``nninteractive_ts_initial_refinement`` — nnInteractive with the raw mask
  supplied as its initial-label interaction.

The last two modes are optional and require an explicit ``--run-nninteractive``
flag. The harness never downloads model weights, resamples, registers,
arbitrates overlaps, or modifies a source mask. Without an independent manual
reference it reports technical QC and candidate agreement only, with status
``no_accuracy_inference``. All derived files are written to a user-supplied
directory outside the repository. This is not clinical validation software.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
DEFAULT_PHASE_COUNT = 12
CHAMBERS = (
    "left_atrium_blood",
    "left_ventricle_blood",
    "right_atrium_blood",
    "right_ventricle_blood",
)
METHOD_ORDER = (
    "raw",
    "conservative_cleanup",
    "nninteractive_prompt_from_ts_geometry",
    "nninteractive_ts_initial_refinement",
)
NN_METHODS = METHOD_ORDER[2:]
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
BOOTSTRAP_FIELDS = (
    "analysis_type",
    "structure",
    "candidate_id",
    "metric",
    "estimand",
    "estimate",
    "interval_lower",
    "interval_upper",
    "confidence_level",
    "bootstrap_iterations",
    "bootstrap_seed",
    "cluster_unit",
    "unique_subjects",
    "observations",
    "inference_scope",
    "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_external_output_dir(output_dir: Path) -> Path:
    """Reject derived output below the repository containing this script."""

    resolved = output_dir.resolve()
    code_dir = Path(__file__).resolve().parent
    for ancestor in (code_dir, *code_dir.parents):
        if (ancestor / ".git").exists():
            try:
                resolved.relative_to(ancestor)
            except ValueError:
                return resolved
            raise ValueError(
                "output-dir must be outside the repository; medical-derived "
                "outputs are not written to Git"
            )
    return resolved


def _portable_path(path: Path, root: Path) -> str:
    """Return a relative path when possible and never emit an absolute path."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _safe_component(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a non-empty identifier")
    value = value.strip()
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{context}: path separators are not allowed in identifiers")
    return value


def _resolve_input_path(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a non-empty path")
    path = Path(value)
    name = path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError(f"{context}: expected .nii or .nii.gz")
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context}: file not found: {resolved}")
    return resolved


def _check_forms(image: nib.Nifti1Image, context: str) -> dict[str, int]:
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(float(np.linalg.det(affine[:3, :3]))) <= 0:
        raise ValueError(f"{context}: invalid or singular affine")
    axes = affine[:3, :3]
    lengths = np.linalg.norm(axes, axis=0)
    if np.any(lengths <= 0):
        raise ValueError(f"{context}: invalid voxel spacing")
    normalized = axes / lengths
    if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-4):
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


def _voxel_volume_ml(image: nib.Nifti1Image) -> float:
    return abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0


def _component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity only
    return int(ndimage.label(mask, structure=structure)[1])


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
    spec: Mapping[str, Any] | None,
    geometry_mm_confirmed: bool,
    compute_mask_stats: bool = True,
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{context}: expected a 3D NIfTI, got {image.shape}")
    forms = _check_forms(image, context)
    unit = image.header.get_xyzt_units()[0] or "unknown"
    local_geometry_override = bool((spec or {}).get("geometry_mm_confirmed", False))
    if unit != "mm" and not (geometry_mm_confirmed or local_geometry_override):
        raise ValueError(
            f"{context}: spatial unit is {unit!r}; explicit geometry_mm_confirmed "
            "is required after checking the source export"
        )
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{context}: non-finite voxel values")
    if kind == "ct":
        values = np.asarray(data)
        stats = {
            "voxel_count": None,
            "volume_ml": None,
            "is_empty": None,
            "connected_components_26": None,
        }
    elif kind == "mask":
        label_value = (spec or {}).get("label_value")
        if label_value is None:
            if np.issubdtype(data.dtype, np.integer):
                if int(data.min()) < 0 or int(data.max()) > 1:
                    raise ValueError(f"{context}: non-binary mask; declare label_value")
            elif not np.all((data == 0) | (data == 1)):
                raise ValueError(f"{context}: non-binary mask; declare label_value")
            values = np.asarray(data == 1, dtype=bool)
        else:
            if not isinstance(label_value, (int, float)) or not math.isfinite(float(label_value)):
                raise ValueError(f"{context}: label_value must be a finite number")
            values = np.asarray(data == label_value, dtype=bool)
        stats = _mask_stats(values, image) if compute_mask_stats else {
            "voxel_count": None,
            "volume_ml": None,
            "is_empty": None,
            "connected_components_26": None,
        }
    else:
        raise ValueError(f"{context}: unsupported image kind {kind!r}")
    metadata = {
        "source_filename": path.name,
        "sha256": sha256_file(path),
        "shape": [int(v) for v in image.shape],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "voxel_spacing_mm": [float(v) for v in nib.affines.voxel_sizes(image.affine)],
        "spatial_unit_in_header": str(unit),
        "geometry_mm_confirmed_override": bool(geometry_mm_confirmed or local_geometry_override),
        "qform_code": forms["qform_code"],
        "sform_code": forms["sform_code"],
        **stats,
        "absolute_path_omitted": True,
    }
    return image, values, metadata


def _assert_same_grid(left: nib.Nifti1Image, right: nib.Nifti1Image, context: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{context}: grid shape mismatch; automatic resampling is forbidden")
    if not np.allclose(left.affine, right.affine, atol=1e-4):
        raise ValueError(f"{context}: affine mismatch; automatic resampling is forbidden")
    left_unit = left.header.get_xyzt_units()[0] or "unknown"
    right_unit = right.header.get_xyzt_units()[0] or "unknown"
    if left_unit != right_unit:
        raise ValueError(f"{context}: spatial units differ ({left_unit!r} != {right_unit!r})")


def _mask_spec(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"path": value}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a path or object with path")
    if not isinstance(value.get("path"), str) or not value["path"].strip():
        raise ValueError(f"{context}: missing path")
    return dict(value)


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    if manifest.get("analysis_plan_locked_before_evaluation") is not True:
        raise ValueError("analysis_plan_locked_before_evaluation must be true")
    if manifest.get("surface_tolerance_locked_before_evaluation") is not True:
        raise ValueError("surface_tolerance_locked_before_evaluation must be true")
    tolerance = manifest.get("surface_tolerance_mm")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("surface_tolerance_mm must be a positive finite number")
    if not isinstance(manifest.get("geometry_mm_confirmed", False), bool):
        raise ValueError("geometry_mm_confirmed must be boolean")
    expected = manifest.get("required_phase_count", DEFAULT_PHASE_COUNT)
    if not isinstance(expected, int) or expected < 1:
        raise ValueError("required_phase_count must be a positive integer")
    phases = manifest.get("phases")
    if not isinstance(phases, list) or len(phases) != expected:
        observed = len(phases) if isinstance(phases, list) else "non-list"
        raise ValueError(f"pilot requires exactly {expected} phase records; got {observed}")
    selection = manifest.get("phase_selection")
    if selection is not None:
        if not isinstance(selection, Mapping) or selection.get("locked_before_evaluation") is not True:
            raise ValueError("phase_selection.locked_before_evaluation must be true")
        selected_ids = selection.get("phase_ids")
        phase_ids = [phase.get("phase_id") if isinstance(phase, Mapping) else None for phase in phases]
        if selected_ids != phase_ids:
            raise ValueError("phase_selection.phase_ids must equal manifest phase order")
    seen: set[tuple[str, str]] = set()
    any_reference = False
    for index, phase in enumerate(phases):
        context = f"phases[{index}]"
        if not isinstance(phase, Mapping):
            raise ValueError(f"{context}: expected an object")
        subject = _safe_component(phase.get("subject_id"), f"{context}.subject_id")
        phase_id = _safe_component(phase.get("phase_id"), f"{context}.phase_id")
        key = (subject, phase_id)
        if key in seen:
            raise ValueError(f"{context}: duplicate subject_id/phase_id")
        seen.add(key)
        if not isinstance(phase.get("ct_path"), str) or not phase["ct_path"].strip():
            raise ValueError(f"{context}: missing ct_path")
        raw_masks = phase.get("raw_masks")
        if not isinstance(raw_masks, Mapping) or set(raw_masks) != set(CHAMBERS):
            raise ValueError(f"{context}.raw_masks must contain exactly LA/LV/RA/RV chamber keys")
        for chamber in CHAMBERS:
            _mask_spec(raw_masks[chamber], f"{context}.raw_masks.{chamber}")
        reference = phase.get("manual_reference")
        if reference is not None:
            if not isinstance(reference, Mapping):
                raise ValueError(f"{context}.manual_reference: expected an object or null")
            for chamber, value in reference.items():
                if chamber not in CHAMBERS:
                    raise ValueError(f"{context}.manual_reference: unknown chamber {chamber!r}")
                spec = _mask_spec(value, f"{context}.manual_reference.{chamber}")
                if not isinstance(spec.get("reference_id"), str) or not spec["reference_id"].strip():
                    raise ValueError(f"{context}.manual_reference.{chamber}: reference_id is required")
                if spec.get("independent_from_candidates") is not True:
                    raise ValueError(f"{context}.manual_reference.{chamber}: independent_from_candidates must be true")
                if spec.get("blinded_to_candidates") is not True:
                    raise ValueError(f"{context}.manual_reference.{chamber}: blinded_to_candidates must be true")
                any_reference = True
    bootstrap = manifest.get("bootstrap", {})
    if not isinstance(bootstrap, Mapping):
        raise ValueError("bootstrap must be an object")
    for key in ("iterations", "seed"):
        if key in bootstrap and (not isinstance(bootstrap[key], int) or bootstrap[key] < 1):
            raise ValueError(f"bootstrap.{key} must be a positive integer")
    confidence = bootstrap.get("confidence_level", 0.95)
    if not isinstance(confidence, (int, float)) or not 0 < float(confidence) < 1:
        raise ValueError("bootstrap.confidence_level must be in (0, 1)")
    nn_config = manifest.get("nninteractive", {})
    if not isinstance(nn_config, Mapping):
        raise ValueError("nninteractive must be an object")
    if not isinstance(nn_config.get("enabled", False), bool):
        raise ValueError("nninteractive.enabled must be boolean")
    if nn_config.get("enabled") and (not isinstance(nn_config.get("model_dir"), str) or not nn_config["model_dir"].strip()):
        raise ValueError("nninteractive.model_dir is required when nninteractive.enabled=true")
    if not isinstance(manifest.get("qc_png", False), bool):
        raise ValueError("qc_png must be boolean")
    cleanup = manifest.get("conservative_cleanup", {})
    if not isinstance(cleanup, Mapping):
        raise ValueError("conservative_cleanup must be an object")
    for key in ("small_component_threshold_mm3", "minimum_distance_mm"):
        value = cleanup.get(key, 50.0 if key.endswith("mm3") else 1.0)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"conservative_cleanup.{key} must be a positive finite number")
    if any_reference and manifest.get("inference_scope", "pilot_descriptive_only") != "pilot_descriptive_only":
        raise ValueError("this pilot must use inference_scope=pilot_descriptive_only")


def _phase_input(
    phase: Mapping[str, Any],
    manifest_dir: Path,
    geometry_mm_confirmed: bool,
    compute_mask_stats: bool = True,
) -> dict[str, Any]:
    subject = _safe_component(phase["subject_id"], "subject_id")
    phase_id = _safe_component(phase["phase_id"], "phase_id")
    ct_path = _resolve_input_path(manifest_dir, phase["ct_path"], f"{subject}/{phase_id}/ct_path")
    ct_image, ct, ct_meta = _load_nifti(
        ct_path,
        f"{subject}/{phase_id}/CT",
        kind="ct",
        spec=None,
        geometry_mm_confirmed=geometry_mm_confirmed,
    )
    ct_meta["source_path"] = _portable_path(ct_path, manifest_dir)
    raw_images: dict[str, nib.Nifti1Image] = {}
    raw_masks: dict[str, np.ndarray] = {}
    raw_meta: dict[str, dict[str, Any]] = {}
    for chamber in CHAMBERS:
        spec = _mask_spec(phase["raw_masks"][chamber], f"{subject}/{phase_id}/{chamber}")
        path = _resolve_input_path(manifest_dir, spec["path"], f"{subject}/{phase_id}/{chamber}")
        image, mask, metadata = _load_nifti(
            path,
            f"{subject}/{phase_id}/{chamber}",
            kind="mask",
            spec=spec,
            geometry_mm_confirmed=geometry_mm_confirmed,
            compute_mask_stats=compute_mask_stats,
        )
        _assert_same_grid(ct_image, image, f"{subject}/{phase_id}/{chamber}")
        metadata["source_path"] = _portable_path(path, manifest_dir)
        raw_images[chamber] = image
        raw_masks[chamber] = mask
        raw_meta[chamber] = metadata
    total_image = total_mask = total_meta = None
    if phase.get("total_heart_path"):
        total_spec = _mask_spec(phase["total_heart_path"], f"{subject}/{phase_id}/total_heart_path")
        total_path = _resolve_input_path(manifest_dir, total_spec["path"], f"{subject}/{phase_id}/total_heart_path")
        total_image, total_mask, total_meta = _load_nifti(
            total_path,
            f"{subject}/{phase_id}/total_heart_qc_only",
            kind="mask",
            spec=total_spec,
            geometry_mm_confirmed=geometry_mm_confirmed,
            compute_mask_stats=compute_mask_stats,
        )
        _assert_same_grid(ct_image, total_image, f"{subject}/{phase_id}/total_heart_qc_only")
        total_meta["source_path"] = _portable_path(total_path, manifest_dir)
        total_meta["role"] = "QC only; never used to clip or define a candidate"
    references: dict[str, dict[str, Any]] = {}
    for chamber, value in (phase.get("manual_reference") or {}).items():
        spec = _mask_spec(value, f"{subject}/{phase_id}/manual_reference.{chamber}")
        path = _resolve_input_path(manifest_dir, spec["path"], f"{subject}/{phase_id}/manual_reference.{chamber}")
        image, mask, metadata = _load_nifti(
            path,
            f"{subject}/{phase_id}/manual_reference.{chamber}",
            kind="mask",
            spec=spec,
            geometry_mm_confirmed=geometry_mm_confirmed,
            compute_mask_stats=compute_mask_stats,
        )
        _assert_same_grid(ct_image, image, f"{subject}/{phase_id}/manual_reference.{chamber}")
        metadata["source_path"] = _portable_path(path, manifest_dir)
        references[chamber] = {"spec": spec, "image": image, "mask": mask, "metadata": metadata}
    return {
        "subject_id": subject,
        "phase_id": phase_id,
        "ct_path": ct_path,
        "ct_image": ct_image,
        "ct": np.asarray(ct),
        "ct_meta": ct_meta,
        "raw_images": raw_images,
        "raw_masks": raw_masks,
        "raw_meta": raw_meta,
        "total_image": total_image,
        "total_mask": total_mask,
        "total_meta": total_meta,
        "references": references,
    }


def _centroid_voxel(mask: np.ndarray) -> list[int] | None:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None
    center = coordinates.mean(axis=0)
    distances = np.sum((coordinates - center) ** 2, axis=1)
    return [int(value) for value in coordinates[int(np.argmin(distances))]]


def _conservative_cleanup(
    mask: np.ndarray,
    image: nib.Nifti1Image,
    *,
    small_component_threshold_mm3: float = 50.0,
    minimum_distance_mm: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only distant tiny 26-connected components; never add voxels.

    The rule is intentionally narrower than morphology: a component is removed
    only when its physical volume is below ``small_component_threshold_mm3``
    and its minimum physical distance from the largest component exceeds
    ``minimum_distance_mm``. Holes, boundaries, overlaps and temporal changes
    are left untouched.
    """

    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 3)
    coordinates = np.argwhere(mask)
    cleaned = mask.copy()
    details: dict[str, Any] = {
        "connectivity": 26,
        "small_component_threshold_mm3": float(small_component_threshold_mm3),
        "minimum_distance_mm": float(minimum_distance_mm),
        "components_before": 0,
        "removed_components": [],
        "removed_voxels": 0,
        "changed_voxels": 0,
        "operation": "component_filter_only; no morphology/hole_fill/clipping",
    }
    if not len(coordinates):
        count = 0
        details["components_after"] = 0
        details["fast_path"] = "empty_mask_no_edt"
        return cleaned, details

    # Label only the non-zero ROI.  The previous implementation labelled the
    # complete CT grid and then built a full-grid EDT, which is needlessly
    # expensive for 512x512x256 phases.  A crop cannot change connectivity or
    # distances between foreground voxel centres because all foreground voxels
    # are retained and only a constant index offset is introduced.
    bbox_start = coordinates.min(axis=0).astype(int)
    bbox_stop = coordinates.max(axis=0).astype(int) + 1
    roi_slices = tuple(slice(int(start), int(stop)) for start, stop in zip(bbox_start, bbox_stop))
    roi_mask = mask[roi_slices]
    labels, count = ndimage.label(roi_mask, structure=structure)
    details["components_before"] = int(count)
    details["roi_bbox_start_voxel"] = [int(value) for value in bbox_start]
    details["roi_bbox_stop_voxel_exclusive"] = [int(value) for value in bbox_stop]
    details["roi_shape"] = [int(value) for value in roi_mask.shape]
    if count == 0:
        # This is unreachable for a non-empty binary ROI, but keeps the
        # invariant explicit if ndimage behaviour changes.
        details["components_after"] = 0
        details["fast_path"] = "empty_roi_no_edt"
        return cleaned, details
    if count == 1:
        details["components_after"] = 1
        details["fast_path"] = "single_component_no_distance_query"
        return cleaned, details
    sizes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1)
    voxel_volume_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))
    small_labels = [
        component_label
        for component_label in range(1, count + 1)
        if component_label != largest_label
        and float(sizes[component_label]) * voxel_volume_mm3 < float(small_component_threshold_mm3)
    ]
    details["distance_method"] = "cKDTree_physical_voxel_centers"
    details["distance_query_components"] = int(len(small_labels))
    if not small_labels:
        details["components_after"] = int(count)
        details["fast_path"] = "no_small_components_no_distance_query"
        return cleaned, details

    # Query only small components.  cKDTree over physical voxel centres is
    # equivalent to the old EDT+sampling measure for an orthogonal affine,
    # while avoiding an array-sized distance transform and never querying
    # components that cannot satisfy the removal-volume criterion.
    from scipy.spatial import cKDTree

    largest_local = np.argwhere(labels == largest_label)
    largest_global = largest_local + bbox_start
    largest_physical = nib.affines.apply_affine(image.affine, largest_global)
    largest_tree = cKDTree(largest_physical)
    for component_label in small_labels:
        component_local = np.argwhere(labels == component_label)
        component_global = component_local + bbox_start
        component_physical = nib.affines.apply_affine(image.affine, component_global)
        distances, _ = largest_tree.query(component_physical, k=1)
        component_volume_mm3 = float(sizes[component_label]) * voxel_volume_mm3
        minimum_distance = float(np.min(np.asarray(distances)))
        if component_volume_mm3 < float(small_component_threshold_mm3) and minimum_distance > float(minimum_distance_mm):
            cleaned[tuple(component_global.T)] = False
            details["removed_components"].append({
                "label": component_label,
                "voxel_count": int(sizes[component_label]),
                "volume_mm3": component_volume_mm3,
                "minimum_distance_mm": minimum_distance,
            })
            details["removed_voxels"] += int(sizes[component_label])
    details["changed_voxels"] = int(np.count_nonzero(mask != cleaned))
    details["components_after"] = _component_count(cleaned)
    return cleaned, details


def _method_definition(method_id: str, manifest: Mapping[str, Any], nn_config: Mapping[str, Any]) -> dict[str, Any]:
    raw_source = manifest.get("raw_source") or {}
    raw_version = str(raw_source.get("software_version", "unspecified"))
    if method_id == "raw":
        return {
            "candidate_id": method_id,
            "candidate_set_id": "pilot_raw_totalsegmentator",
            "method_category": "raw_automatic",
            "software_name": str(raw_source.get("software_name", "TotalSegmentator")),
            "software_version": raw_version,
            "model_or_method": str(raw_source.get("model_or_method", "heartchambers_highres")),
            "description": "raw per-chamber mask; primary initialization",
            "depends_on_raw_mask": True,
            "uses_initial_label": False,
            "independent_from_manual_reference": True,
        }
    if method_id == "conservative_cleanup":
        cleanup = manifest.get("conservative_cleanup") or {}
        threshold = float(cleanup.get("small_component_threshold_mm3", 50.0))
        distance = float(cleanup.get("minimum_distance_mm", 1.0))
        return {
            "candidate_id": method_id,
            "candidate_set_id": "pilot_conservative_component_filter",
            "method_category": "conservative_cleanup_no_morphology",
            "software_name": "pilot_harness",
            "software_version": str(REPORT_SCHEMA_VERSION),
            "model_or_method": (
                "remove_components_below_" + str(threshold) + "mm3_if_distance_gt_" + str(distance) + "mm_26connectivity"
            ),
            "description": "remove only distant tiny components; never add voxels; no morphology or hole filling",
            "cleanup_parameters": {
                "connectivity": 26,
                "small_component_threshold_mm3": threshold,
                "minimum_distance_mm": distance,
            },
            "depends_on_raw_mask": True,
            "uses_initial_label": False,
            "independent_from_manual_reference": True,
        }
    model_id = str(nn_config.get("model_id", "nnInteractive_v1.0"))
    version = str(nn_config.get("software_version", "unspecified"))
    if method_id == "nninteractive_prompt_from_ts_geometry":
        return {
            "candidate_id": method_id,
            "candidate_set_id": "pilot_nninteractive_prompt_geometry",
            "method_category": "assisted_prompt_from_geometry",
            "software_name": "nnInteractive",
            "software_version": version,
            "model_or_method": model_id,
            "description": "one positive point from raw-mask geometry; no initial label",
            "depends_on_raw_mask": True,
            "uses_initial_label": False,
            "independent_from_manual_reference": True,
        }
    if method_id == "nninteractive_ts_initial_refinement":
        return {
            "candidate_id": method_id,
            "candidate_set_id": "pilot_nninteractive_ts_initial_refinement",
            "method_category": "assisted_initial_label_refinement",
            "software_name": "nnInteractive",
            "software_version": version,
            "model_or_method": model_id,
            "description": "raw mask supplied as initial label; no manual reference is supplied",
            "depends_on_raw_mask": True,
            "uses_initial_label": True,
            "independent_from_manual_reference": True,
        }
    raise ValueError(f"unknown candidate method {method_id!r}")


def _candidate_output_path(output_dir: Path, subject: str, phase_id: str, method_id: str, chamber: str) -> Path:
    return output_dir / _safe_component(subject, "subject_id") / _safe_component(phase_id, "phase_id") / method_id / f"{chamber}.nii.gz"


def _save_binary_mask(path: Path, mask: np.ndarray, reference_image: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_image.header.copy()
    output = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), reference_image.affine, header=header)
    output.set_data_dtype(np.uint8)
    qcode = int(reference_image.header["qform_code"])
    scode = int(reference_image.header["sform_code"])
    output.set_qform(reference_image.affine, code=qcode or 1)
    output.set_sform(reference_image.affine, code=scode or 1)
    units = reference_image.header.get_xyzt_units()
    output.header.set_xyzt_units(xyz=units[0], t=units[1])
    nib.save(output, str(path))


class NNInteractiveRunner:
    """Small adapter around the local nnInteractive 2.5.x direct API."""

    def __init__(self, config: Mapping[str, Any], manifest_dir: Path):
        import torch

        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

        raw_model_dir = Path(str(config["model_dir"]))
        self.model_dir = (raw_model_dir if raw_model_dir.is_absolute() else manifest_dir / raw_model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"nnInteractive model directory not found: {self.model_dir}")
        self.config = dict(config)
        device_name = str(config.get("device", "cuda"))
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("nnInteractive requested CUDA but torch.cuda.is_available() is false")
        self._session_class = nnInteractiveInferenceSession
        start = time.perf_counter()
        self.session = nnInteractiveInferenceSession(
            device=self.device,
            use_torch_compile=bool(config.get("use_torch_compile", False)),
            verbose=bool(config.get("verbose", False)),
            torch_n_threads=int(config.get("torch_n_threads", 8)),
            do_autozoom=bool(config.get("do_autozoom", True)),
            interactions_storage=str(config.get("interactions_storage", "auto")),
            enable_undo=bool(config.get("enable_undo", False)),
        )
        init_kwargs: dict[str, Any] = {
            "model_training_output_dir": str(self.model_dir),
            "checkpoint_name": str(config.get("checkpoint_name", "checkpoint_final.pth")),
        }
        if config.get("use_fold") is not None:
            init_kwargs["use_fold"] = config["use_fold"]
        self.session.initialize_from_trained_model_folder(**init_kwargs)
        self.initialization_seconds = time.perf_counter() - start
        self.model_provenance = _model_provenance(self.model_dir, manifest_dir)

    def predict(
        self,
        image: np.ndarray,
        spacing_mm: Sequence[float],
        *,
        mode: str,
        initial_seg: np.ndarray | None,
        positive_points: Sequence[Sequence[int]],
        negative_points: Sequence[Sequence[int]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if mode == "prompt_from_ts_geometry" and initial_seg is not None:
            raise ValueError("prompt-from-geometry mode must not receive an initial label")
        if mode == "ts_initial_refinement" and initial_seg is None:
            raise ValueError("TS initial-refinement mode requires an initial label")
        image = np.asarray(image, dtype=np.float32)
        target = np.zeros(image.shape, dtype=np.uint8)
        started = time.perf_counter()
        self.session.set_image(image[None, ...], {"spacing": tuple(float(v) for v in spacing_mm)})
        self.session.set_target_buffer(target)
        override = bool(self.config.get("override_capability_checks", False))
        if mode == "prompt_from_ts_geometry":
            if len(positive_points) == 0:
                raise ValueError("prompt-from-geometry mode requires at least one positive point")
            all_points: list[tuple[Sequence[int], bool]] = [(point, True) for point in positive_points]
            all_points.extend((point, False) for point in negative_points)
            for index, (point, include) in enumerate(all_points):
                self.session.add_point_interaction(
                    tuple(int(v) for v in point),
                    include_interaction=include,
                    run_prediction=index == len(all_points) - 1,
                    override_capability_checks=override,
                )
        else:
            self.session.add_initial_seg_interaction(
                np.asarray(initial_seg, dtype=np.uint8),
                run_prediction=True,
                override_capability_checks=override,
            )
        prediction = np.asarray(target).copy()
        details = {
            "elapsed_seconds": time.perf_counter() - started,
            "output_dtype": str(prediction.dtype),
            "output_shape": [int(v) for v in prediction.shape],
            "output_labels": [int(v) for v in np.unique(prediction)],
            "positive_voxels": int(np.count_nonzero(prediction)),
        }
        return np.asarray(prediction != 0, dtype=bool), details

    def close(self) -> None:
        executor = getattr(self.session, "executor", None)
        if executor is not None and hasattr(executor, "shutdown"):
            executor.shutdown(wait=True)


def _model_provenance(model_dir: Path, manifest_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if model_dir.is_dir():
        for path in sorted(model_dir.rglob("*")):
            if not path.is_file() or path.name.endswith(".lock"):
                continue
            files.append({
                "path": _portable_path(path, manifest_dir),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            })
    return {
        "model_dir": _portable_path(model_dir, manifest_dir),
        "available": model_dir.is_dir(),
        "files": files,
        "absolute_path_omitted": True,
    }


@lru_cache(maxsize=1)
def _metric_compare_function():
    """Reuse the locked surface metric implementation from the manual validator."""

    module_path = Path(__file__).with_name("validate_manual_cardiac_masks.py")
    spec = importlib.util.spec_from_file_location("_pilot_manual_metric_backend", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load validate_manual_cardiac_masks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_masks


def _safe_compare(
    candidate: np.ndarray,
    reference: np.ndarray,
    affine: np.ndarray,
    tolerance_mm: float,
) -> tuple[dict[str, Any], str]:
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    candidate_count = int(np.count_nonzero(candidate))
    reference_count = int(np.count_nonzero(reference))
    voxel_volume_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000.0
    candidate_volume = candidate_count * voxel_volume_ml
    reference_volume = reference_count * voxel_volume_ml
    base: dict[str, Any] = {
        "candidate_volume_ml": candidate_volume,
        "reference_volume_ml": reference_volume,
        "signed_volume_error_ml": None,
        "absolute_volume_error_ml": None,
        "signed_volume_error_percent": None,
        "absolute_volume_error_percent": None,
        "dice": None,
        "surface_dice": None,
        "assd_mm": None,
        "hd95_mm": None,
    }
    if reference_volume > 0:
        signed = candidate_volume - reference_volume
        base.update({
            "signed_volume_error_ml": signed,
            "absolute_volume_error_ml": abs(signed),
            "signed_volume_error_percent": 100.0 * signed / reference_volume,
            "absolute_volume_error_percent": 100.0 * abs(signed) / reference_volume,
        })
    denominator = candidate_count + reference_count
    if denominator > 0:
        base["dice"] = 2.0 * int(np.count_nonzero(candidate & reference)) / denominator
    if candidate_count == 0 and reference_count == 0:
        return base, "undefined_both_empty"
    if reference_count == 0:
        return base, "reference_empty_surface_metrics_undefined"
    try:
        base.update(_metric_compare_function()(candidate, reference, np.asarray(affine), tolerance_mm))
        return base, "ok"
    except (ValueError, RuntimeError) as exc:
        return base, f"surface_metrics_error:{type(exc).__name__}"


def _technical_row(
    *,
    phase: Mapping[str, Any],
    chamber: str,
    method: Mapping[str, Any],
    record: Mapping[str, Any],
    image: nib.Nifti1Image,
    total_mask: np.ndarray | None,
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mask = np.asarray(record["mask"], dtype=bool)
    stats = _mask_stats(mask, image)
    row: dict[str, Any] = {
        "record_type": "candidate_technical_qc",
        "subject_id": phase["subject_id"],
        "phase_id": phase["phase_id"],
        "structure": chamber,
        "candidate_id": method["candidate_id"],
        "candidate_set_id": method["candidate_set_id"],
        "method_category": method["method_category"],
        "software_name": method["software_name"],
        "software_version": method["software_version"],
        "mask_path": record.get("mask_path"),
        "mask_sha256": record.get("output_sha256"),
        **stats,
        "outside_total_heart_voxels": None,
        "outside_total_fraction": None,
        "containment_fraction_in_total_heart": None,
        "previous_phase_id": None,
        "previous_volume_ml": None,
        "adjacent_signed_volume_change_ml": None,
        "adjacent_absolute_volume_change_ml": None,
        "adjacent_relative_volume_change_fraction": None,
        "status": record.get("status"),
    }
    if total_mask is not None:
        outside = int(np.count_nonzero(mask & ~total_mask))
        row["outside_total_heart_voxels"] = outside
        if stats["voxel_count"]:
            row["outside_total_fraction"] = outside / stats["voxel_count"]
            row["containment_fraction_in_total_heart"] = 1.0 - outside / stats["voxel_count"]
    if previous is not None:
        previous_volume = float(previous["volume_ml"])
        change = float(stats["volume_ml"]) - previous_volume
        row.update({
            "previous_phase_id": previous["phase_id"],
            "previous_volume_ml": previous_volume,
            "adjacent_signed_volume_change_ml": change,
            "adjacent_absolute_volume_change_ml": abs(change),
            "adjacent_relative_volume_change_fraction": (change / previous_volume if previous_volume else None),
        })
    return row


def _overlap_summary(masks: Mapping[str, np.ndarray], image: nib.Nifti1Image) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    names = list(masks)
    voxel_volume = _voxel_volume_ml(image)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            count = int(np.count_nonzero(masks[left_name] & masks[right_name]))
            key = f"{left_name}__{right_name}"
            summary[key] = {
                "overlap_voxels": count,
                "overlap_volume_ml": count * voxel_volume,
                "status": "descriptive_QC; no overlap arbitration",
            }
    return summary


def _accuracy_row(
    *,
    phase: Mapping[str, Any],
    chamber: str,
    method: Mapping[str, Any],
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
    tolerance_mm: float,
) -> dict[str, Any]:
    metrics, metric_status = _safe_compare(
        record["mask"],
        reference["mask"],
        phase["ct_image"].affine,
        tolerance_mm,
    )
    spec = reference["spec"]
    return {
        "record_type": "manual_reference_accuracy",
        "subject_id": phase["subject_id"],
        "phase_id": phase["phase_id"],
        "structure": chamber,
        "candidate_id": method["candidate_id"],
        "candidate_set_id": method["candidate_set_id"],
        "method_category": method["method_category"],
        "software_name": method["software_name"],
        "software_version": method["software_version"],
        "candidate_mask_path": record.get("mask_path"),
        "candidate_mask_sha256": record.get("output_sha256"),
        "reference_id": spec["reference_id"],
        "reference_kind": spec.get("reference_kind", "manual_independent"),
        "reference_source_path": reference["metadata"].get("source_path"),
        "reference_mask_sha256": reference["metadata"]["sha256"],
        "candidate_independent_from_reference": method["independent_from_manual_reference"],
        "reference_independent_from_candidates": spec["independent_from_candidates"],
        "reference_blinded_to_candidates": spec["blinded_to_candidates"],
        "expert_id": spec.get("expert_id"),
        "session_id": spec.get("session_id"),
        "surface_tolerance_mm": tolerance_mm,
        "accuracy_eligible": bool(method["independent_from_manual_reference"]),
        "manual_interaction_time_s": spec.get("manual_interaction_time_s"),
        "manual_clicks": spec.get("manual_clicks"),
        **metrics,
        "candidate_empty": bool(np.count_nonzero(record["mask"]) == 0),
        "reference_empty": bool(np.count_nonzero(reference["mask"]) == 0),
        "status": metric_status,
    }


def _agreement_row(
    *,
    phase: Mapping[str, Any],
    chamber: str,
    left_method: Mapping[str, Any],
    left_record: Mapping[str, Any],
    right_method: Mapping[str, Any],
    right_record: Mapping[str, Any],
    tolerance_mm: float,
) -> dict[str, Any]:
    metrics, metric_status = _safe_compare(
        left_record["mask"],
        right_record["mask"],
        phase["ct_image"].affine,
        tolerance_mm,
    )
    return {
        "record_type": "candidate_agreement_only",
        "comparison_scope": "agreement_only; no manual truth inference",
        "subject_id": phase["subject_id"],
        "phase_id": phase["phase_id"],
        "structure": chamber,
        "candidate_a_id": left_method["candidate_id"],
        "candidate_b_id": right_method["candidate_id"],
        "candidate_a_set_id": left_method["candidate_set_id"],
        "candidate_b_set_id": right_method["candidate_set_id"],
        "candidate_a_method_category": left_method["method_category"],
        "candidate_b_method_category": right_method["method_category"],
        "candidate_a_mask_path": left_record.get("mask_path"),
        "candidate_b_mask_path": right_record.get("mask_path"),
        "candidate_a_mask_sha256": left_record.get("output_sha256"),
        "candidate_b_mask_sha256": right_record.get("output_sha256"),
        "surface_tolerance_mm": tolerance_mm,
        "candidate_a_volume_ml": metrics["candidate_volume_ml"],
        "candidate_b_volume_ml": metrics["reference_volume_ml"],
        "signed_volume_difference_ml": metrics["signed_volume_error_ml"],
        "absolute_volume_difference_ml": metrics["absolute_volume_error_ml"],
        "dice": metrics["dice"],
        "surface_dice": metrics["surface_dice"],
        "assd_mm": metrics["assd_mm"],
        "hd95_mm": metrics["hd95_mm"],
        "status": metric_status,
    }


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    inference_scope: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    iterations = int(config.get("iterations", 2000))
    seed = int(config.get("seed", 20260902))
    confidence = float(config.get("confidence_level", 0.95))
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            key = (str(row["structure"]), str(row["candidate_id"]), metric)
            grouped.setdefault(key, {}).setdefault(str(row["subject_id"]), []).append(float(value))
    output: list[dict[str, Any]] = []
    for (structure, candidate_id, metric), by_subject in sorted(grouped.items()):
        subject_ids = sorted(by_subject)
        subject_values = np.asarray([np.mean(by_subject[subject]) for subject in subject_ids], dtype=float)
        estimate = float(np.mean(subject_values))
        sampled = rng.integers(0, len(subject_values), size=(iterations, len(subject_values)))
        bootstrap_estimates = subject_values[sampled].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        lower, upper = np.quantile(bootstrap_estimates, [alpha, 1.0 - alpha])
        output.append({
            "analysis_type": "subject_cluster_bootstrap",
            "structure": structure,
            "candidate_id": candidate_id,
            "metric": metric,
            "estimand": "mean of subject-level phase means",
            "estimate": estimate,
            "interval_lower": float(lower),
            "interval_upper": float(upper),
            "confidence_level": confidence,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "cluster_unit": "subject",
            "unique_subjects": len(subject_ids),
            "observations": int(sum(len(values) for values in by_subject.values())),
            "inference_scope": inference_scope,
            "status": (
                "pilot_cluster_bootstrap_interval_not_population_ci"
                if len(subject_ids) <= 3
                else "cluster_bootstrap_descriptive_interval"
            ),
        })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred_fields: Sequence[str] = ()) -> None:
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_qc_png(
    path: Path,
    ct: np.ndarray,
    raw_masks: Mapping[str, np.ndarray],
    candidate_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write optional external-only central-slice visual QC."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage import measure
    except ImportError as exc:
        return {"status": f"unavailable:{type(exc).__name__}"}
    raw_union = np.logical_or.reduce(list(raw_masks.values()))
    coordinates = np.where(raw_union)
    if not coordinates[0].size:
        return {"status": "empty_raw_union"}
    centers = [int(round((int(values.min()) + int(values.max())) / 2)) for values in coordinates]
    plane_names = ("sagittal", "coronal", "axial")
    colors = {
        "raw": "cyan",
        "conservative_cleanup": "yellow",
        "nninteractive_prompt_from_ts_geometry": "magenta",
        "nninteractive_ts_initial_refinement": "lime",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (name, index) in enumerate(zip(plane_names, centers)):
        ct_slice = np.take(ct, index, axis=axis).T
        axes[axis].imshow(ct_slice, cmap="gray")
        for method_id, mask in candidate_masks.items():
            mask_slice = np.take(mask, index, axis=axis).T
            for contour in measure.find_contours(mask_slice.astype(float), 0.5):
                axes[axis].plot(contour[:, 1], contour[:, 0], color=colors.get(method_id, "white"), linewidth=0.8)
        axes[axis].set_title(f"{name}; index={index}")
        axes[axis].axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {"status": "written", "path": path.name, "sha256": sha256_file(path)}


def _environment_metadata() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nibabel": version("nibabel"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "skimage": version("scikit-image"),
        "nninteractive": version("nninteractive"),
    }


def _error_status(exc: BaseException) -> str:
    return f"failed:{type(exc).__name__}"


def run_pilot(
    manifest_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = True,
    run_nninteractive: bool = False,
    nninteractive_runner: Any | None = None,
    qc_png_override: bool | None = None,
) -> dict[str, Any]:
    """Validate the manifest and optionally generate external pilot outputs."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest root must be an object")
    _validate_manifest(manifest)
    manifest_dir = manifest_path.parent
    output_dir = _assert_external_output_dir(Path(output_dir))
    tolerance_mm = float(manifest["surface_tolerance_mm"])
    nn_config = dict(manifest.get("nninteractive", {}))
    qc_png_enabled = bool(manifest.get("qc_png", False) if qc_png_override is None else qc_png_override)
    owned_runner = False
    runner = nninteractive_runner
    if run_nninteractive and not dry_run and nn_config.get("enabled", False) and runner is None:
        runner = NNInteractiveRunner(nn_config, manifest_dir)
        owned_runner = True
    previous_volumes: dict[tuple[str, str, str], dict[str, Any]] = {}
    technical_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    phases_report: list[dict[str, Any]] = []
    qc_png_rows: list[dict[str, Any]] = []
    generation_failures = 0
    try:
        for phase_index, phase_spec in enumerate(manifest["phases"]):
            phase = _phase_input(
                phase_spec,
                manifest_dir,
                bool(manifest.get("geometry_mm_confirmed", False)),
                compute_mask_stats=not dry_run,
            )
            subject = phase["subject_id"]
            phase_id = phase["phase_id"]
            candidate_records: dict[str, dict[str, dict[str, Any]]] = {chamber: {} for chamber in CHAMBERS}
            prompt_report: dict[str, Any] = {}
            for chamber in CHAMBERS:
                raw_mask = phase["raw_masks"][chamber]
                point = None if dry_run else _centroid_voxel(raw_mask)
                prompt_report[chamber] = {
                    "selection_policy": (
                        "not computed in dry-run validation mode"
                        if dry_run
                        else "nearest raw-mask voxel to voxel centroid"
                    ),
                    "positive_points": [point] if point is not None else [],
                    "negative_points": [],
                    "source_mask_sha256": phase["raw_meta"][chamber]["sha256"],
                    "coordinate_space": "NIfTI array voxel indices; no resampling or reorientation",
                }
                for method_id in METHOD_ORDER:
                    method = _method_definition(method_id, manifest, nn_config)
                    record: dict[str, Any] = {
                        "status": "not_run",
                        "mask": None,
                        "mask_path": None,
                        "output_sha256": None,
                        "runtime": None,
                        "prompt": prompt_report[chamber] if method_id == "nninteractive_prompt_from_ts_geometry" else None,
                    }
                    try:
                        if dry_run and method_id in {"raw", "conservative_cleanup"}:
                            # Dry-run is validation-only: do not perform
                            # cleanup, component metrics, or candidate work.
                            record["status"] = "not_run_dry_run"
                        elif method_id == "raw":
                            record["mask"] = np.asarray(raw_mask, dtype=bool).copy()
                            record["status"] = "would_generate" if dry_run else "generated"
                        elif method_id == "conservative_cleanup":
                            cleanup_config = manifest.get("conservative_cleanup") or {}
                            record["mask"], record["cleanup_details"] = _conservative_cleanup(
                                raw_mask,
                                phase["ct_image"],
                                small_component_threshold_mm3=float(cleanup_config.get("small_component_threshold_mm3", 50.0)),
                                minimum_distance_mm=float(cleanup_config.get("minimum_distance_mm", 1.0)),
                            )
                            record["cleanup_changed_voxels"] = record["cleanup_details"]["changed_voxels"]
                            record["status"] = "would_generate" if dry_run else "generated"
                        elif not run_nninteractive:
                            record["status"] = "not_run_requires_explicit_flag"
                        elif not nn_config.get("enabled", False):
                            record["status"] = "not_run_manifest_disabled"
                        elif dry_run:
                            record["status"] = "not_run_dry_run"
                        elif runner is None:
                            record["status"] = "not_run_runner_unavailable"
                        elif point is None:
                            record["status"] = "failed_empty_raw_mask_no_prompt"
                        else:
                            mode = (
                                "prompt_from_ts_geometry"
                                if method_id == "nninteractive_prompt_from_ts_geometry"
                                else "ts_initial_refinement"
                            )
                            started = time.perf_counter()
                            prediction, runtime = runner.predict(
                                phase["ct"],
                                phase["ct_meta"]["voxel_spacing_mm"],
                                mode=mode,
                                initial_seg=(raw_mask if method_id == "nninteractive_ts_initial_refinement" else None),
                                positive_points=([point] if method_id == "nninteractive_prompt_from_ts_geometry" else []),
                                negative_points=[],
                            )
                            prediction = np.asarray(prediction, dtype=bool)
                            if prediction.shape != raw_mask.shape:
                                raise ValueError(f"prediction shape {prediction.shape} differs from CT/raw shape {raw_mask.shape}")
                            record["mask"] = prediction
                            record["runtime"] = {**dict(runtime or {}), "harness_elapsed_seconds": time.perf_counter() - started}
                            record["status"] = "generated"
                    except Exception as exc:  # preserve an explicit failed-candidate record
                        record["status"] = _error_status(exc)
                        record["error_type"] = type(exc).__name__
                        record["mask"] = None
                        generation_failures += 1
                    if record["mask"] is not None:
                        if not dry_run:
                            output_path = _candidate_output_path(output_dir, subject, phase_id, method_id, chamber)
                            _save_binary_mask(output_path, record["mask"], phase["ct_image"])
                            record["mask_path"] = output_path.relative_to(output_dir).as_posix()
                            record["output_sha256"] = sha256_file(output_path)
                        else:
                            record["mask_path"] = _candidate_output_path(output_dir, subject, phase_id, method_id, chamber).relative_to(output_dir).as_posix()
                        record["source_mask_sha256"] = phase["raw_meta"][chamber]["sha256"]
                    candidate_records[chamber][method_id] = record

            phase_technical_rows: list[dict[str, Any]] = []
            phase_accuracy_rows: list[dict[str, Any]] = []
            phase_agreement_rows: list[dict[str, Any]] = []
            phase_pairwise_overlaps: dict[str, Any] = {}
            for chamber in CHAMBERS:
                for method_id, record in candidate_records[chamber].items():
                    if record["mask"] is None:
                        continue
                    method = _method_definition(method_id, manifest, nn_config)
                    key = (subject, chamber, method_id)
                    previous = previous_volumes.get(key)
                    row = _technical_row(
                        phase=phase,
                        chamber=chamber,
                        method=method,
                        record=record,
                        image=phase["ct_image"],
                        total_mask=phase["total_mask"],
                        previous=previous,
                    )
                    row["phase_index"] = phase_index
                    phase_technical_rows.append(row)
                    previous_volumes[key] = {"phase_id": phase_id, "volume_ml": row["volume_ml"]}

                for method_id in METHOD_ORDER:
                    available = {
                        chamber_name: candidate_records[chamber_name][method_id]["mask"]
                        for chamber_name in CHAMBERS
                        if candidate_records[chamber_name][method_id]["mask"] is not None
                    }
                    if available:
                        phase_pairwise_overlaps[method_id] = _overlap_summary(available, phase["ct_image"])

                reference = phase["references"].get(chamber)
                if reference is not None and not dry_run:
                    for method_id, record in candidate_records[chamber].items():
                        if record["mask"] is None:
                            continue
                        method = _method_definition(method_id, manifest, nn_config)
                        phase_accuracy_rows.append(_accuracy_row(
                            phase=phase,
                            chamber=chamber,
                            method=method,
                            record=record,
                            reference=reference,
                            tolerance_mm=tolerance_mm,
                        ))
                if not dry_run:
                    available_methods = [
                        method_id for method_id in METHOD_ORDER
                        if candidate_records[chamber][method_id]["mask"] is not None
                    ]
                    for left_index, left_id in enumerate(available_methods):
                        for right_id in available_methods[left_index + 1 :]:
                            phase_agreement_rows.append(_agreement_row(
                                phase=phase,
                                chamber=chamber,
                                left_method=_method_definition(left_id, manifest, nn_config),
                                left_record=candidate_records[chamber][left_id],
                                right_method=_method_definition(right_id, manifest, nn_config),
                                right_record=candidate_records[chamber][right_id],
                                tolerance_mm=tolerance_mm,
                            ))
            technical_rows.extend(phase_technical_rows)
            accuracy_rows.extend(phase_accuracy_rows)
            agreement_rows.extend(phase_agreement_rows)

            if qc_png_enabled and not dry_run:
                candidate_union_masks: dict[str, np.ndarray] = {}
                for method_id in METHOD_ORDER:
                    available = [
                        candidate_records[chamber][method_id]["mask"]
                        for chamber in CHAMBERS
                        if candidate_records[chamber][method_id]["mask"] is not None
                    ]
                    if available:
                        candidate_union_masks[method_id] = np.logical_or.reduce(available)
                qc_path = output_dir / "qc_png" / subject / f"{phase_id}.png"
                qc_result = _write_qc_png(qc_path, phase["ct"], phase["raw_masks"], candidate_union_masks)
                if qc_result.get("status") == "written":
                    qc_result["path"] = qc_path.relative_to(output_dir).as_posix()
                qc_result.update({"subject_id": subject, "phase_id": phase_id})
                qc_png_rows.append(qc_result)

            serialized_candidates: dict[str, list[dict[str, Any]]] = {}
            for chamber in CHAMBERS:
                serialized_candidates[chamber] = []
                for method_id in METHOD_ORDER:
                    method = _method_definition(method_id, manifest, nn_config)
                    record = candidate_records[chamber][method_id]
                    serialized = {
                        **method,
                        **{key: value for key, value in record.items() if key != "mask"},
                    }
                    serialized["input_raw_mask_sha256"] = phase["raw_meta"][chamber]["sha256"]
                    serialized_candidates[chamber].append(serialized)
            phases_report.append({
                "phase_index": phase_index,
                "subject_id": subject,
                "phase_id": phase_id,
                "ct": phase["ct_meta"],
                "raw_mask_provenance": phase["raw_meta"],
                "total_heart_qc_only": phase["total_meta"],
                "manual_reference_provenance": {
                    chamber: reference["metadata"]
                    for chamber, reference in phase["references"].items()
                },
                "prompts": prompt_report,
                "candidates": serialized_candidates,
                "pairwise_chamber_overlaps": phase_pairwise_overlaps,
                "technical_qc": phase_technical_rows,
                "agreement_rows": phase_agreement_rows,
                "accuracy_rows": phase_accuracy_rows,
                "qc_png": next(
                    (
                        row for row in reversed(qc_png_rows)
                        if row.get("subject_id") == subject and row.get("phase_id") == phase_id
                    ),
                    None,
                ),
            })
    finally:
        if owned_runner and runner is not None:
            runner.close()

    unique_subjects = sorted({str(phase["subject_id"]) for phase in phases_report})
    inference_scope = str(manifest.get("inference_scope", "pilot_descriptive_only"))
    if len(unique_subjects) <= 3:
        inference_scope = "pilot_descriptive_only"
    bootstrap = [] if dry_run else _cluster_bootstrap(accuracy_rows, manifest.get("bootstrap", {}), inference_scope)
    if dry_run:
        status = "dry_run_no_accuracy_inference"
        analysis_mode = "dry_run"
        comparison_status = "not_evaluated"
    elif accuracy_rows:
        status = "accuracy_metrics_computed_pending_interpretation"
        analysis_mode = "accuracy"
        comparison_status = "manual_reference_accuracy_plus_candidate_agreement"
    else:
        status = "no_accuracy_inference" if not generation_failures else "generation_failures_no_accuracy_inference"
        analysis_mode = "technical_only"
        comparison_status = "agreement_only"

    model_provenance = None
    if isinstance(nn_config.get("model_dir"), str) and nn_config["model_dir"].strip():
        raw_model_dir = Path(nn_config["model_dir"])
        model_dir = (raw_model_dir if raw_model_dir.is_absolute() else manifest_dir / raw_model_dir).resolve()
        model_provenance = getattr(runner, "model_provenance", None) or _model_provenance(model_dir, manifest_dir)
    report: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": status,
        "analysis_mode": analysis_mode,
        "comparison_status": comparison_status,
        "accuracy_inference_allowed": bool(accuracy_rows) and not dry_run,
        "accuracy_statement": (
            "Independent manual-reference metrics are computed, but this is a descriptive pilot; "
            "intervals are clustered by subject and are not population confidence intervals."
            if accuracy_rows
            else "No independent manual reference was evaluated; candidate agreement and technical QC do not infer accuracy."
        ),
        "inference_scope": inference_scope,
        "required_phase_count": int(manifest.get("required_phase_count", DEFAULT_PHASE_COUNT)),
        "phase_count_observed": len(phases_report),
        "unique_subjects": unique_subjects,
        "analysis_plan": {
            "analysis_plan_locked_before_evaluation": True,
            "surface_tolerance_locked_before_evaluation": True,
            "surface_tolerance_mm": tolerance_mm,
            "phase_selection": manifest.get("phase_selection"),
        },
        "input_manifest": {
            "source_path": _portable_path(manifest_path, manifest_dir),
            "sha256": sha256_file(manifest_path),
            "content_hash": _json_hash(manifest),
            "absolute_path_omitted": True,
        },
        "environment": _environment_metadata(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "nninteractive": {
            "requested": bool(run_nninteractive),
            "executed": bool(runner is not None and run_nninteractive and not dry_run),
            "config": {key: value for key, value in nn_config.items() if key != "model_dir"},
            "model_provenance": model_provenance,
        },
        "candidate_definitions": {
            method_id: _method_definition(method_id, manifest, nn_config)
            for method_id in METHOD_ORDER
        },
        "constraints": {
            "no_morphology": True,
            "no_temporal_smoothing": True,
            "no_registration": True,
            "no_resampling": True,
            "total_heart_is_qc_only": True,
            "overlaps_are_never_arbitrated": True,
            "ct_intensity_is_not_truth": True,
            "ranking_performed": False,
        },
        "qc_png_enabled": qc_png_enabled,
        "phases": phases_report,
        "technical_rows": technical_rows,
        "accuracy_rows": accuracy_rows,
        "agreement_rows": agreement_rows,
        "bootstrap": bootstrap,
        "qc_png_rows": qc_png_rows,
        "output_directory_name": output_dir.name,
        "output_directory_absolute_path_omitted": True,
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "pilot_technical_qc.csv", technical_rows)
        _write_csv(output_dir / "pilot_accuracy_metrics.csv", accuracy_rows)
        _write_csv(output_dir / "pilot_candidate_agreement.csv", agreement_rows)
        _write_csv(output_dir / "pilot_subject_cluster_bootstrap.csv", bootstrap, BOOTSTRAP_FIELDS)
        (output_dir / "pilot_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="write generated candidates and reports")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without writing outputs (default)")
    parser.add_argument("--run-nninteractive", action="store_true", help="explicitly run local nnInteractive")
    parser.add_argument("--qc-png", action="store_true", help="write optional external orthogonal-slice QC PNGs")
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("choose at most one of --execute and --dry-run")
    report = run_pilot(
        args.manifest,
        args.output_dir,
        dry_run=not args.execute,
        run_nninteractive=bool(args.run_nninteractive),
        qc_png_override=(True if args.qc_png else None),
    )
    print(json.dumps({
        "status": report["status"],
        "analysis_mode": report["analysis_mode"],
        "phase_count": report["phase_count_observed"],
        "subjects": report["unique_subjects"],
        "accuracy_rows": len(report["accuracy_rows"]),
        "agreement_rows": len(report["agreement_rows"]),
        "output_directory_name": report["output_directory_name"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
