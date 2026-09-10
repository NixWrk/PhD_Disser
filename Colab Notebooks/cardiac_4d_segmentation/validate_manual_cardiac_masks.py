#!/usr/bin/env python
"""Validate cardiac CT masks against independent manual NIfTI references.

The input is a versioned JSON manifest.  Absolute paths are used only while
reading local files and are omitted from all reports.  STL is intentionally not
accepted as a quantitative reference because it does not encode units, a voxel
grid, or an anatomical coordinate system.

This is research software for method development, not a clinical validation
tool.  A run is rejected unless the surface tolerance and analysis plan were
locked before the results were inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage, special
from skimage import measure


SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

STRUCTURES = {
    "whole_heart",
    "blood_pool_four_chambers",
    "left_atrium_blood",
    "left_ventricle_blood",
    "right_atrium_blood",
    "right_ventricle_blood",
    "heart_myocardium_highres_target",
    "complete_four_chamber_myocardium",
    "geometric_residual_whole_heart_minus_blood_pool",
}

CANDIDATE_SOURCE_KINDS = {
    "totalsegmentator_total_heart_raw",
    "totalsegmentator_total_heart_postprocessed",
    "totalsegmentator_heartchambers_highres_chamber_raw",
    "totalsegmentator_heartchambers_highres_blood_union",
    "totalsegmentator_heartchambers_highres_heart_myocardium_raw",
    "totalsegmentator_heartchambers_highres_heart_myocardium_postprocessed",
    "geometric_residual_total_heart_minus_highres_blood",
    "other_algorithm",
}

SOURCE_STRUCTURE_RULES = {
    "totalsegmentator_total_heart_raw": {"whole_heart"},
    "totalsegmentator_total_heart_postprocessed": {"whole_heart"},
    "totalsegmentator_heartchambers_highres_chamber_raw": {
        "left_atrium_blood",
        "left_ventricle_blood",
        "right_atrium_blood",
        "right_ventricle_blood",
    },
    "totalsegmentator_heartchambers_highres_blood_union": {
        "blood_pool_four_chambers"
    },
    "totalsegmentator_heartchambers_highres_heart_myocardium_raw": {
        "heart_myocardium_highres_target"
    },
    "totalsegmentator_heartchambers_highres_heart_myocardium_postprocessed": {
        "heart_myocardium_highres_target"
    },
    "geometric_residual_total_heart_minus_highres_blood": {
        "geometric_residual_whole_heart_minus_blood_pool"
    },
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

VALIDATION_FIELDS = (
    "sample_id",
    "subject_id",
    "phase_id",
    "structure",
    "structure_definition_version",
    "candidate_id",
    "candidate_source_kind",
    "reference_id",
    "reference_kind",
    "is_primary_reference",
    "surface_tolerance_mm",
    "candidate_volume_ml",
    "reference_volume_ml",
    *METRIC_NAMES,
)

REPRODUCIBILITY_FIELDS = (
    "sample_id",
    "subject_id",
    "phase_id",
    "structure",
    "pair_type",
    "annotation_a",
    "annotation_b",
    "expert_a",
    "expert_b",
    "session_a",
    "session_b",
    "surface_tolerance_mm",
    "candidate_volume_ml",
    "reference_volume_ml",
    *METRIC_NAMES,
)

BOOTSTRAP_FIELDS = (
    "analysis_type",
    "structure",
    "candidate_id",
    "pair_type",
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
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: missing required field {key!r}")
    return mapping[key]


def resolve_input_path(manifest_dir: Path, raw: str, context: str) -> Path:
    path = Path(raw)
    if path.suffix.lower() == ".stl":
        raise ValueError(
            f"{context}: STL cannot be a quantitative reference; export a NIfTI "
            "mask on the original CT grid"
        )
    resolved = path if path.is_absolute() else manifest_dir / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context}: mask not found: {resolved}")
    name = resolved.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError(f"{context}: expected .nii or .nii.gz, got {resolved.name}")
    return resolved


def _affine_is_orthogonal(affine: np.ndarray, atol: float = 1e-4) -> bool:
    axes = np.asarray(affine[:3, :3], dtype=np.float64)
    lengths = np.linalg.norm(axes, axis=0)
    if np.any(lengths <= 0):
        return False
    unit_axes = axes / lengths
    gram = unit_axes.T @ unit_axes
    return bool(np.allclose(gram, np.eye(3), atol=atol))


def load_binary_mask(
    path: Path,
    spec: dict[str, Any],
    context: str,
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Any]]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{context}: expected a 3D NIfTI, got shape {image.shape}")
    if not np.all(np.isfinite(image.affine)) or abs(
        float(np.linalg.det(image.affine[:3, :3]))
    ) <= 0:
        raise ValueError(f"{context}: invalid or singular affine")
    if not _affine_is_orthogonal(image.affine):
        raise ValueError(
            f"{context}: sheared voxel axes are unsupported for surface metrics"
        )
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    if int(qform_code) == 0 and int(sform_code) == 0:
        raise ValueError(f"{context}: both qform and sform codes are unset")
    if int(qform_code) > 0 and not np.allclose(qform, image.affine, atol=1e-4):
        raise ValueError(f"{context}: qform differs from selected affine")
    if int(sform_code) > 0 and not np.allclose(sform, image.affine, atol=1e-4):
        raise ValueError(f"{context}: sform differs from selected affine")

    spatial_unit = image.header.get_xyzt_units()[0]
    if spatial_unit != "mm" and not bool(spec.get("geometry_mm_confirmed", False)):
        raise ValueError(
            f"{context}: NIfTI spatial unit is {spatial_unit or 'unknown'!r}; "
            "set geometry_mm_confirmed=true only after checking the source export"
        )

    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{context}: mask contains non-finite values")
    label_value = spec.get("label_value")
    if label_value is None:
        if np.issubdtype(data.dtype, np.integer):
            minimum = int(data.min())
            maximum = int(data.max())
            if minimum < 0 or maximum > 1:
                raise ValueError(
                    f"{context}: non-binary integer mask; declare label_value"
                )
        elif not np.all((data == 0) | (data == 1)):
            raise ValueError(
                f"{context}: non-binary floating mask; declare label_value"
            )
        mask = data == 1
    else:
        mask = data == label_value
    mask = np.asarray(mask, dtype=bool)
    voxel_count = int(np.count_nonzero(mask))
    if voxel_count == 0:
        raise ValueError(f"{context}: empty mask")

    zooms = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    voxel_volume_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0
    metadata = {
        "source_filename": path.name,
        "sha256": sha256_file(path),
        "shape": [int(value) for value in image.shape],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "voxel_spacing_mm": list(zooms),
        "voxel_volume_ml": voxel_volume_ml,
        "voxel_count": voxel_count,
        "volume_ml": voxel_count * voxel_volume_ml,
        "spatial_unit_in_header": spatial_unit or "unknown",
        "geometry_mm_confirmed_override": bool(spec.get("geometry_mm_confirmed", False)),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "absolute_path_omitted": True,
    }
    return image, mask, metadata


def assert_same_grid(
    left_image: nib.Nifti1Image,
    right_image: nib.Nifti1Image,
    context: str,
    affine_atol: float = 1e-4,
) -> None:
    if left_image.shape != right_image.shape:
        raise ValueError(
            f"{context}: grid shape mismatch {left_image.shape} != {right_image.shape}; "
            "automatic resampling is forbidden"
        )
    if not np.allclose(left_image.affine, right_image.affine, atol=affine_atol):
        raise ValueError(
            f"{context}: affine mismatch; export both masks on the original CT grid. "
            "Automatic resampling is forbidden"
        )


def dice_score(left: np.ndarray, right: np.ndarray) -> float:
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    denominator = left_count + right_count
    if denominator == 0:
        raise ValueError("Dice is undefined for two empty masks")
    intersection = int(np.count_nonzero(left & right))
    return 2.0 * intersection / denominator


def _crop_union(
    left: np.ndarray,
    right: np.ndarray,
    margin: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    union = left | right
    coordinates = np.where(union)
    if not coordinates[0].size:
        raise ValueError("Surface metrics are undefined for empty masks")
    slices = []
    for axis, values in enumerate(coordinates):
        start = max(int(values.min()) - margin, 0)
        stop = min(int(values.max()) + margin + 1, union.shape[axis])
        slices.append(slice(start, stop))
    selection = tuple(slices)
    return left[selection], right[selection]


def _cell_codes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8, copy=False), 1)
    shape = tuple(value - 1 for value in padded.shape)
    codes = np.zeros(shape, dtype=np.uint8)
    bit = 0
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                corner = padded[
                    i : i + shape[0],
                    j : j + shape[1],
                    k : k + shape[2],
                ]
                codes |= corner << bit
                bit += 1
    return codes


@lru_cache(maxsize=32)
def _surfel_area_lookup(spacing_mm: tuple[float, float, float]) -> np.ndarray:
    spacing = tuple(float(value) for value in spacing_mm)
    lookup = np.zeros(256, dtype=np.float64)
    for code in range(1, 255):
        block = np.zeros((2, 2, 2), dtype=np.uint8)
        bit = 0
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    block[i, j, k] = (code >> bit) & 1
                    bit += 1
        vertices, faces, _, _ = measure.marching_cubes(
            block,
            level=0.5,
            spacing=spacing,
            allow_degenerate=False,
        )
        lookup[code] = float(measure.mesh_surface_area(vertices, faces))
    return lookup


def _weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    if not len(values) or float(np.sum(weights)) <= 0:
        raise ValueError("weighted percentile requires positive surface area")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    threshold = percentile / 100.0 * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)
    return float(sorted_values[index])


def surface_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    spacing_mm: Sequence[float],
    tolerance_mm: float,
) -> dict[str, float]:
    if not np.isfinite(float(tolerance_mm)) or tolerance_mm <= 0:
        raise ValueError("surface tolerance must be positive and finite")
    try:
        spacing = tuple(float(value) for value in spacing_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing_mm must contain three finite positive values") from exc
    if len(spacing) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing
    ):
        raise ValueError("spacing_mm must contain three finite positive values")
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if candidate.ndim != 3 or reference.ndim != 3:
        raise ValueError("surface metrics require two 3D masks")
    if candidate.shape != reference.shape:
        raise ValueError(
            f"surface metrics require equal mask shapes, got {candidate.shape} != {reference.shape}"
        )
    candidate, reference = _crop_union(candidate, reference, margin=1)
    candidate_codes = _cell_codes(candidate)
    reference_codes = _cell_codes(reference)
    candidate_surface = (candidate_codes != 0) & (candidate_codes != 255)
    reference_surface = (reference_codes != 0) & (reference_codes != 255)
    if not np.any(candidate_surface) or not np.any(reference_surface):
        raise ValueError("surface metrics require two non-empty surfaces")

    lookup = _surfel_area_lookup(spacing)
    candidate_weights = lookup[candidate_codes[candidate_surface]]
    reference_weights = lookup[reference_codes[reference_surface]]
    distances_to_reference = ndimage.distance_transform_edt(
        ~reference_surface, sampling=spacing
    )[candidate_surface]
    distances_to_candidate = ndimage.distance_transform_edt(
        ~candidate_surface, sampling=spacing
    )[reference_surface]

    candidate_area = float(np.sum(candidate_weights))
    reference_area = float(np.sum(reference_weights))
    total_area = candidate_area + reference_area
    within_tolerance_area = float(
        np.sum(candidate_weights[distances_to_reference <= tolerance_mm])
        + np.sum(reference_weights[distances_to_candidate <= tolerance_mm])
    )
    total_distance_area = float(
        np.sum(candidate_weights * distances_to_reference)
        + np.sum(reference_weights * distances_to_candidate)
    )
    hd95 = max(
        _weighted_percentile(distances_to_reference, candidate_weights, 95.0),
        _weighted_percentile(distances_to_candidate, reference_weights, 95.0),
    )
    return {
        "surface_dice": within_tolerance_area / total_area,
        "assd_mm": total_distance_area / total_area,
        "hd95_mm": hd95,
        "candidate_surface_area_mm2": candidate_area,
        "reference_surface_area_mm2": reference_area,
    }


def compare_masks(
    candidate: np.ndarray,
    reference: np.ndarray,
    affine: np.ndarray,
    tolerance_mm: float,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if candidate.ndim != 3 or reference.ndim != 3:
        raise ValueError("mask comparison requires two 3D masks")
    if candidate.shape != reference.shape:
        raise ValueError(
            f"mask comparison requires equal shapes, got {candidate.shape} != {reference.shape}"
        )
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("mask comparison requires a finite 4x4 affine")
    determinant = float(np.linalg.det(affine[:3, :3]))
    if abs(determinant) <= 0:
        raise ValueError("mask comparison requires a non-singular affine")
    voxel_volume_ml = abs(determinant) / 1000.0
    candidate_volume_ml = int(np.count_nonzero(candidate)) * voxel_volume_ml
    reference_volume_ml = int(np.count_nonzero(reference)) * voxel_volume_ml
    if reference_volume_ml <= 0:
        raise ValueError("reference mask must be non-empty")
    signed_error_ml = candidate_volume_ml - reference_volume_ml
    surfaces = surface_metrics(
        candidate,
        reference,
        tuple(float(value) for value in nib.affines.voxel_sizes(affine)),
        tolerance_mm,
    )
    return {
        "candidate_volume_ml": candidate_volume_ml,
        "reference_volume_ml": reference_volume_ml,
        "signed_volume_error_ml": signed_error_ml,
        "absolute_volume_error_ml": abs(signed_error_ml),
        "signed_volume_error_percent": 100.0 * signed_error_ml / reference_volume_ml,
        "absolute_volume_error_percent": 100.0 * abs(signed_error_ml) / reference_volume_ml,
        "dice": dice_score(candidate, reference),
        "surface_dice": surfaces["surface_dice"],
        "assd_mm": surfaces["assd_mm"],
        "hd95_mm": surfaces["hd95_mm"],
    }


def _union_bbox(masks: Sequence[np.ndarray], margin: int = 1) -> tuple[slice, ...]:
    union = np.logical_or.reduce(masks)
    coordinates = np.where(union)
    if not coordinates[0].size:
        raise ValueError("STAPLE annotations are all empty")
    slices = []
    for axis, values in enumerate(coordinates):
        start = max(int(values.min()) - margin, 0)
        stop = min(int(values.max()) + margin + 1, union.shape[axis])
        slices.append(slice(start, stop))
    return tuple(slices)


def staple_consensus(
    masks: Sequence[np.ndarray],
    threshold: float = 0.5,
    max_iterations: int = 100,
    convergence: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a binary STAPLE consensus using a cropped EM calculation.

    The posterior is evaluated only in a bounding box around the union of the
    annotations.  Voxels outside that box are known to be all-zero observations;
    their contribution to the prior and specificity M-steps is included
    analytically so cropping cannot change the fitted foreground prevalence.
    """
    if len(masks) < 3:
        raise ValueError(
            "STAPLE requires at least three independent annotations in this protocol; "
            "use expert adjudication for two annotations"
        )
    if not 0 < threshold < 1:
        raise ValueError("STAPLE threshold must be in (0, 1)")
    if int(max_iterations) < 1:
        raise ValueError("STAPLE max_iterations must be positive")
    if not np.isfinite(float(convergence)) or convergence <= 0:
        raise ValueError("STAPLE convergence must be positive and finite")

    normalized_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    shape = normalized_masks[0].shape
    if len(shape) != 3 or any(mask.ndim != 3 for mask in normalized_masks):
        raise ValueError("STAPLE annotations must be 3D masks")
    if any(mask.shape != shape for mask in normalized_masks):
        raise ValueError("STAPLE annotations have different shapes")
    selection = _union_bbox(normalized_masks, margin=1)
    observations = np.stack(
        [mask[selection] for mask in normalized_masks], axis=0
    ).astype(bool, copy=False)
    raters = observations.shape[0]
    total_voxels = int(np.prod(shape, dtype=np.int64))
    cropped_voxels = int(np.prod(observations.shape[1:], dtype=np.int64))
    outside_voxels = total_voxels - cropped_voxels
    if outside_voxels < 0:
        raise ValueError("STAPLE crop is larger than the annotation grid")

    sensitivity = np.full(raters, 0.999, dtype=np.float64)
    specificity = np.full(raters, 0.999, dtype=np.float64)
    observed_foreground = float(np.count_nonzero(observations))
    prior = float(
        np.clip(observed_foreground / (raters * total_voxels), 1e-6, 1 - 1e-6)
    )
    epsilon = 1e-6
    converged = False

    posterior = np.mean(observations, axis=0, dtype=np.float32)
    for iteration in range(1, int(max_iterations) + 1):
        old = np.concatenate([sensitivity, specificity, [prior]])
        log_odds = np.full(
            observations.shape[1:],
            math.log(prior / (1.0 - prior)),
            dtype=np.float32,
        )
        for index in range(raters):
            positive = math.log(
                sensitivity[index] / (1.0 - specificity[index])
            )
            negative = math.log(
                (1.0 - sensitivity[index]) / specificity[index]
            )
            log_odds += np.where(observations[index], positive, negative).astype(
                np.float32
            )
        posterior = special.expit(log_odds).astype(np.float32)

        inside_positive_sum = float(np.sum(posterior, dtype=np.float64))
        inside_negative_sum = float(
            np.sum(1.0 - posterior, dtype=np.float64)
        )
        outside_posterior = 0.0
        if outside_voxels:
            outside_log_odds = math.log(prior / (1.0 - prior))
            for index in range(raters):
                outside_log_odds += math.log(
                    (1.0 - sensitivity[index]) / specificity[index]
                )
            outside_posterior = float(special.expit(outside_log_odds))
        positive_sum = max(
            inside_positive_sum + outside_voxels * outside_posterior,
            epsilon,
        )
        negative_probability = 1.0 - posterior
        negative_sum = max(
            inside_negative_sum + outside_voxels * (1.0 - outside_posterior),
            epsilon,
        )
        for index in range(raters):
            observation = observations[index]
            sensitivity[index] = np.clip(
                float(np.sum(posterior[observation], dtype=np.float64))
                / positive_sum,
                epsilon,
                1.0 - epsilon,
            )
            specificity[index] = np.clip(
                (
                    float(
                        np.sum(
                            negative_probability[~observation],
                            dtype=np.float64,
                        )
                    )
                    + outside_voxels * (1.0 - outside_posterior)
                )
                / negative_sum,
                epsilon,
                1.0 - epsilon,
            )
        prior = float(
            np.clip(
                (inside_positive_sum + outside_voxels * outside_posterior)
                / total_voxels,
                epsilon,
                1.0 - epsilon,
            )
        )
        updated = np.concatenate([sensitivity, specificity, [prior]])
        if float(np.max(np.abs(updated - old))) < convergence:
            converged = True
            break

    consensus = np.zeros(shape, dtype=bool)
    consensus[selection] = posterior >= threshold
    if not np.any(consensus):
        raise ValueError("STAPLE produced an empty consensus")
    return consensus, {
        "method": "staple",
        "threshold": threshold,
        "max_iterations": int(max_iterations),
        "convergence_tolerance": float(convergence),
        "iterations": iteration,
        "converged": converged,
        "estimated_sensitivity": sensitivity.tolist(),
        "estimated_specificity": specificity.tolist(),
        "estimated_foreground_prior": prior,
        "total_voxels": total_voxels,
        "cropped_voxels": cropped_voxels,
        "outside_all_zero_voxels": outside_voxels,
        "interpretation": "statistical consensus of manual annotations, not ground truth",
    }


def _validate_manifest_header(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version={SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    if manifest.get("analysis_plan_locked_before_evaluation") is not True:
        raise ValueError(
            "analysis_plan_locked_before_evaluation must be true before calculation"
        )
    tolerance = float(require(manifest, "surface_tolerance_mm", "manifest"))
    if tolerance <= 0:
        raise ValueError("surface_tolerance_mm must be positive")
    if manifest.get("surface_tolerance_locked_before_evaluation") is not True:
        raise ValueError(
            "surface_tolerance_locked_before_evaluation must be true before calculation"
        )
    rationale = str(manifest.get("surface_tolerance_rationale", "")).strip()
    if not rationale:
        raise ValueError("surface_tolerance_rationale must be recorded")
    bootstrap = require(manifest, "bootstrap", "manifest")
    if bootstrap.get("cluster_unit") != "subject_id":
        raise ValueError("bootstrap.cluster_unit must be subject_id")
    if bootstrap.get("estimand") != "mean_of_subject_means":
        raise ValueError("bootstrap.estimand must be mean_of_subject_means")
    if int(bootstrap.get("iterations", 0)) < 100:
        raise ValueError("bootstrap.iterations must be at least 100")
    require(bootstrap, "seed", "manifest.bootstrap")
    try:
        int(bootstrap["seed"])
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest.bootstrap.seed must be an integer") from exc
    confidence = float(bootstrap.get("confidence_level", 0))
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("bootstrap.confidence_level must be finite and in (0, 1)")
    if manifest.get("inference_scope") != "pilot_descriptive_only":
        raise ValueError(
            "Current three-subject protocol requires inference_scope=pilot_descriptive_only"
        )


def _validate_candidate(candidate: dict[str, Any], structure: str, context: str) -> None:
    source_kind = require(candidate, "source_kind", context)
    if source_kind not in CANDIDATE_SOURCE_KINDS:
        raise ValueError(f"{context}: unsupported source_kind {source_kind!r}")
    allowed = SOURCE_STRUCTURE_RULES.get(source_kind)
    if allowed is not None and structure not in allowed:
        raise ValueError(
            f"{context}: {source_kind!r} cannot be evaluated as {structure!r}"
        )
    for key in (
        "candidate_id",
        "path",
        "software_name",
        "software_version",
        "model_task",
        "model_class",
        "postprocessing",
    ):
        require(candidate, key, context)


def _candidate_provenance_for_report(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "candidate_id",
        "source_kind",
        "software_name",
        "software_version",
        "model_task",
        "model_class",
        "postprocessing",
        "label_value",
        "geometry_mm_confirmed",
    )
    return {key: candidate[key] for key in allowed if key in candidate}


def _validate_annotation(annotation: dict[str, Any], context: str) -> None:
    for key in (
        "annotation_id",
        "path",
        "expert_id",
        "session_id",
        "mask_version",
        "source_project_sha256",
        "annotation_mode",
    ):
        require(annotation, key, context)
    source_hash = str(annotation["source_project_sha256"])
    if len(source_hash) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in source_hash):
        raise ValueError(f"{context}: source_project_sha256 must be a SHA-256 hex digest")
    if source_hash == "0" * 64:
        raise ValueError(
            f"{context}: replace the all-zero source_project_sha256 placeholder"
        )
    if annotation.get("independent_from_candidate") is not True:
        raise ValueError(
            f"{context}: accuracy references must be independent_from_candidate=true"
        )
    if annotation.get("blinded_to_candidate_during_initial_annotation") is not True:
        raise ValueError(
            f"{context}: initial reference annotation must be blinded to the candidate"
        )


def _load_sample_masks(
    sample: dict[str, Any],
    manifest_dir: Path,
) -> tuple[
    dict[str, tuple[nib.Nifti1Image, np.ndarray, dict[str, Any], dict[str, Any]]],
    dict[str, tuple[nib.Nifti1Image, np.ndarray, dict[str, Any], dict[str, Any]]],
]:
    context = f"sample {sample.get('sample_id', '<unknown>')}"
    structure = require(sample, "structure", context)
    if structure not in STRUCTURES:
        raise ValueError(f"{context}: unsupported structure {structure!r}")
    require(sample, "subject_id", context)
    require(sample, "phase_id", context)
    require(sample, "structure_definition_version", context)

    candidates = {}
    for index, spec in enumerate(require(sample, "candidates", context)):
        item_context = f"{context} candidate[{index}]"
        _validate_candidate(spec, structure, item_context)
        item_id = str(spec["candidate_id"])
        if item_id in candidates:
            raise ValueError(f"{context}: duplicate candidate_id {item_id!r}")
        path = resolve_input_path(manifest_dir, str(spec["path"]), item_context)
        candidates[item_id] = (*load_binary_mask(path, spec, item_context), spec)

    annotations = {}
    for index, spec in enumerate(require(sample, "manual_annotations", context)):
        item_context = f"{context} manual_annotations[{index}]"
        _validate_annotation(spec, item_context)
        item_id = str(spec["annotation_id"])
        if item_id in annotations:
            raise ValueError(f"{context}: duplicate annotation_id {item_id!r}")
        path = resolve_input_path(manifest_dir, str(spec["path"]), item_context)
        annotations[item_id] = (*load_binary_mask(path, spec, item_context), spec)
    if not candidates:
        raise ValueError(f"{context}: at least one candidate is required")
    if not annotations:
        raise ValueError(f"{context}: at least one manual annotation is required")

    images = [item[0] for item in [*candidates.values(), *annotations.values()]]
    reference_image = images[0]
    for index, image in enumerate(images[1:], start=1):
        assert_same_grid(reference_image, image, f"{context} grid[{index}]")
    return candidates, annotations


def _primary_reference(
    sample: dict[str, Any],
    annotations: dict[str, tuple[nib.Nifti1Image, np.ndarray, dict[str, Any], dict[str, Any]]],
    manifest_dir: Path,
    warnings: list[str],
) -> tuple[str, str, nib.Nifti1Image, np.ndarray, dict[str, Any], dict[str, Any] | None]:
    context = f"sample {sample['sample_id']} primary_reference"
    primary = require(sample, "primary_reference", context)
    method = require(primary, "method", context)
    first_image = next(iter(annotations.values()))[0]
    if method == "single":
        annotation_id = str(require(primary, "annotation_id", context))
        if annotation_id not in annotations:
            raise ValueError(f"{context}: unknown annotation_id {annotation_id!r}")
        image, mask, metadata, _ = annotations[annotation_id]
        return annotation_id, "single_manual", image, mask, metadata, None
    if method == "adjudicated":
        spec = require(primary, "annotation", context)
        _validate_annotation(spec, f"{context} adjudicated annotation")
        path = resolve_input_path(
            manifest_dir,
            str(spec["path"]),
            f"{context} adjudicated annotation",
        )
        image, mask, metadata = load_binary_mask(
            path, spec, f"{context} adjudicated annotation"
        )
        assert_same_grid(first_image, image, context)
        return str(spec["annotation_id"]), "adjudicated_manual", image, mask, metadata, None
    if method == "staple":
        ids = [str(value) for value in require(primary, "annotation_ids", context)]
        if len(ids) < 3:
            raise ValueError(f"{context}: list at least three annotation_ids")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{context}: annotation_ids contain duplicates")
        missing = [value for value in ids if value not in annotations]
        if missing:
            raise ValueError(f"{context}: unknown annotation_ids {missing}")
        masks = [annotations[value][1] for value in ids]
        consensus, staple_info = staple_consensus(
            masks,
            threshold=float(primary.get("threshold", 0.5)),
            max_iterations=int(primary.get("max_iterations", 100)),
            convergence=float(primary.get("convergence", 1e-5)),
        )
        if not staple_info["converged"]:
            warnings.append(f"{sample['sample_id']}: STAPLE did not converge")
        voxel_ml = abs(float(np.linalg.det(first_image.affine[:3, :3]))) / 1000.0
        metadata = {
            "source_filename": None,
            "sha256": None,
            "shape": list(first_image.shape),
            "orientation": list(nib.aff2axcodes(first_image.affine)),
            "voxel_spacing_mm": list(nib.affines.voxel_sizes(first_image.affine)),
            "voxel_count": int(np.count_nonzero(consensus)),
            "volume_ml": int(np.count_nonzero(consensus)) * voxel_ml,
            "absolute_path_omitted": True,
            "derived_from_annotation_ids": ids,
        }
        return (
            f"staple_{sample['sample_id']}",
            "staple_consensus",
            first_image,
            consensus,
            metadata,
            staple_info,
        )
    raise ValueError(f"{context}: unsupported method {method!r}")


def _metric_row(
    sample: dict[str, Any],
    candidate_id: str,
    candidate_spec: dict[str, Any],
    reference_id: str,
    reference_kind: str,
    is_primary: bool,
    metrics: dict[str, float],
    tolerance_mm: float,
) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "subject_id": sample["subject_id"],
        "phase_id": sample["phase_id"],
        "structure": sample["structure"],
        "structure_definition_version": sample["structure_definition_version"],
        "candidate_id": candidate_id,
        "candidate_source_kind": candidate_spec["source_kind"],
        "reference_id": reference_id,
        "reference_kind": reference_kind,
        "is_primary_reference": is_primary,
        "surface_tolerance_mm": tolerance_mm,
        **metrics,
    }


def _pair_type(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    if left["expert_id"] == right["expert_id"]:
        if left["session_id"] != right["session_id"]:
            return "intraexpert"
        return None
    return "interexpert"


def cluster_bootstrap_summary(
    rows: Sequence[dict[str, Any]],
    bootstrap: dict[str, Any],
    inference_scope: str,
) -> list[dict[str, Any]]:
    primary_rows = [row for row in rows if row["is_primary_reference"]]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in primary_rows:
        groups.setdefault((row["structure"], row["candidate_id"]), []).append(row)

    iterations = int(bootstrap["iterations"])
    seed = int(bootstrap["seed"])
    confidence = float(bootstrap["confidence_level"])
    rng = np.random.default_rng(seed)
    output = []
    alpha = 1.0 - confidence
    for (structure, candidate_id), group_rows in sorted(groups.items()):
        subject_ids = sorted({str(row["subject_id"]) for row in group_rows})
        for metric in METRIC_NAMES:
            subject_means = []
            observations = 0
            for subject_id in subject_ids:
                values = [
                    float(row[metric])
                    for row in group_rows
                    if str(row["subject_id"]) == subject_id
                    and row.get(metric) is not None
                    and np.isfinite(float(row[metric]))
                ]
                if values:
                    subject_means.append(float(np.mean(values)))
                    observations += len(values)
            estimate = float(np.mean(subject_means)) if subject_means else None
            lower = upper = None
            if len(subject_means) >= 2:
                values_array = np.asarray(subject_means, dtype=np.float64)
                indices = rng.integers(
                    0, len(values_array), size=(iterations, len(values_array))
                )
                draws = np.mean(values_array[indices], axis=1)
                lower, upper = np.quantile(
                    draws, [alpha / 2.0, 1.0 - alpha / 2.0]
                ).tolist()
                status = "pilot_cluster_bootstrap_interval_not_population_ci"
            else:
                status = "insufficient_subject_clusters_for_interval"
            output.append(
                {
                    "structure": structure,
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
                    "unique_subjects": len(subject_means),
                    "observations": observations,
                    "inference_scope": inference_scope,
                    "status": status,
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_validation(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_header(manifest)
    tolerance_mm = float(manifest["surface_tolerance_mm"])
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    reproducibility_rows: list[dict[str, Any]] = []
    samples_report = []
    warnings: list[str] = []
    samples = require(manifest, "samples", "manifest")
    if not samples:
        raise ValueError("manifest.samples must not be empty")

    seen_samples: set[str] = set()
    for sample in samples:
        sample_id = str(require(sample, "sample_id", "sample"))
        if sample_id in seen_samples:
            raise ValueError(f"duplicate sample_id {sample_id!r}")
        seen_samples.add(sample_id)
        candidates, annotations = _load_sample_masks(sample, manifest_path.parent)
        (
            primary_id,
            primary_kind,
            primary_image,
            primary_mask,
            primary_metadata,
            consensus_info,
        ) = _primary_reference(sample, annotations, manifest_path.parent, warnings)

        sample_validation_start = len(validation_rows)
        for candidate_id, (candidate_image, candidate_mask, _, candidate_spec) in candidates.items():
            for annotation_id, (annotation_image, annotation_mask, _, _) in annotations.items():
                metrics = compare_masks(
                    candidate_mask,
                    annotation_mask,
                    candidate_image.affine,
                    tolerance_mm,
                )
                validation_rows.append(
                    _metric_row(
                        sample,
                        candidate_id,
                        candidate_spec,
                        annotation_id,
                        "individual_manual",
                        primary_kind == "single_manual" and annotation_id == primary_id,
                        metrics,
                        tolerance_mm,
                    )
                )
            if primary_kind != "single_manual":
                assert_same_grid(candidate_image, primary_image, f"{sample_id} primary grid")
                metrics = compare_masks(
                    candidate_mask,
                    primary_mask,
                    candidate_image.affine,
                    tolerance_mm,
                )
                validation_rows.append(
                    _metric_row(
                        sample,
                        candidate_id,
                        candidate_spec,
                        primary_id,
                        primary_kind,
                        True,
                        metrics,
                        tolerance_mm,
                    )
                )

        annotation_items = list(annotations.items())
        for left_index in range(len(annotation_items)):
            left_id, (left_image, left_mask, _, left_spec) = annotation_items[left_index]
            for right_index in range(left_index + 1, len(annotation_items)):
                right_id, (right_image, right_mask, _, right_spec) = annotation_items[right_index]
                pair_type = _pair_type(left_spec, right_spec)
                if pair_type is None:
                    continue
                assert_same_grid(left_image, right_image, f"{sample_id} reproducibility")
                metrics = compare_masks(
                    left_mask,
                    right_mask,
                    left_image.affine,
                    tolerance_mm,
                )
                reproducibility_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": sample["subject_id"],
                        "phase_id": sample["phase_id"],
                        "structure": sample["structure"],
                        "pair_type": pair_type,
                        "annotation_a": left_id,
                        "annotation_b": right_id,
                        "expert_a": left_spec["expert_id"],
                        "expert_b": right_spec["expert_id"],
                        "session_a": left_spec["session_id"],
                        "session_b": right_spec["session_id"],
                        "surface_tolerance_mm": tolerance_mm,
                        **metrics,
                    }
                )

        samples_report.append(
            {
                "sample_id": sample_id,
                "subject_id": sample["subject_id"],
                "phase_id": sample["phase_id"],
                "structure": sample["structure"],
                "structure_definition_version": sample["structure_definition_version"],
                "candidate_masks": {
                    item_id: {
                        "metadata": value[2],
                        "provenance": _candidate_provenance_for_report(value[3]),
                    }
                    for item_id, value in candidates.items()
                },
                "manual_annotations": {
                    item_id: {
                        "metadata": value[2],
                        "expert_id": value[3]["expert_id"],
                        "session_id": value[3]["session_id"],
                        "mask_version": value[3]["mask_version"],
                        "source_project_sha256": value[3]["source_project_sha256"],
                    }
                    for item_id, value in annotations.items()
                },
                "primary_reference": {
                    "reference_id": primary_id,
                    "reference_kind": primary_kind,
                    "metadata": primary_metadata,
                    "consensus": consensus_info,
                },
                "validation_comparisons": len(validation_rows) - sample_validation_start,
                "status": "method_development_not_clinically_validated",
            }
        )

    validation_bootstrap_rows = cluster_bootstrap_summary(
        validation_rows,
        manifest["bootstrap"],
        manifest["inference_scope"],
    )
    for row in validation_bootstrap_rows:
        row["analysis_type"] = "candidate_accuracy"
        row["pair_type"] = None

    reproducibility_bootstrap_input = [
        {
            **row,
            "candidate_id": row["pair_type"],
            "is_primary_reference": True,
        }
        for row in reproducibility_rows
    ]
    reproducibility_bootstrap_rows = cluster_bootstrap_summary(
        reproducibility_bootstrap_input,
        manifest["bootstrap"],
        manifest["inference_scope"],
    )
    for row in reproducibility_bootstrap_rows:
        row["analysis_type"] = "manual_reproducibility"
        row["pair_type"] = row["candidate_id"]
        row["candidate_id"] = None
    bootstrap_rows = validation_bootstrap_rows + reproducibility_bootstrap_rows
    _write_csv(output_dir / "validation_metrics.csv", validation_rows, VALIDATION_FIELDS)
    _write_csv(
        output_dir / "reproducibility_metrics.csv",
        reproducibility_rows,
        REPRODUCIBILITY_FIELDS,
    )
    _write_csv(
        output_dir / "cluster_bootstrap_summary.csv",
        bootstrap_rows,
        BOOTSTRAP_FIELDS,
    )

    try:
        scipy_version = importlib.metadata.version("scipy")
        skimage_version = importlib.metadata.version("scikit-image")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = skimage_version = "unknown"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "method_development_not_clinically_validated",
        "manifest": {
            "source_filename": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "absolute_path_omitted": True,
            "analysis_plan_locked_before_evaluation": True,
            "surface_tolerance_mm": tolerance_mm,
            "surface_tolerance_rationale": manifest["surface_tolerance_rationale"],
        },
        "definitions": {
            "volume_error": "candidate minus independent manual reference; both signed and absolute values are reported",
            "dice": "voxel Dice on the common original CT grid",
            "surface_dice": "symmetric surface-area-weighted fraction within the prespecified tolerance",
            "assd_mm": "symmetric surface-area-weighted mean surface distance",
            "hd95_mm": "maximum of the two directed surface-area-weighted 95th percentiles",
            "staple": "statistical consensus of independent annotations, not anatomical ground truth",
            "cluster_bootstrap": "subjects are resampled for candidate accuracy and manual reproducibility; phase measurements are averaged within subject before the cohort mean",
        },
        "inference": {
            "scope": manifest["inference_scope"],
            "cluster_unit": "subject_id",
            "population_generalization_allowed": False,
            "three_subject_warning": "pilot intervals must not be presented as population confidence intervals",
            "tta_or_ensemble_uncertainty": "allowed only for QC prioritization unless calibrated against manual references",
        },
        "samples": samples_report,
        "validation_comparisons": len(validation_rows),
        "reproducibility_comparisons": len(reproducibility_rows),
        "bootstrap_summaries": len(bootstrap_rows),
        "warnings": warnings,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "scipy": scipy_version,
            "scikit_image": skimage_version,
        },
        "privacy": {
            "absolute_paths_omitted": True,
            "source_images_and_masks_copied": False,
            "reports_may_still_be_derived_medical_data": True,
            "do_not_commit_without_explicit_review": True,
        },
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_validation(args.manifest, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "validation_comparisons": report["validation_comparisons"],
                "reproducibility_comparisons": report[
                    "reproducibility_comparisons"
                ],
                "output_files": [
                    "validation_metrics.csv",
                    "reproducibility_metrics.csv",
                    "cluster_bootstrap_summary.csv",
                    "validation_report.json",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
