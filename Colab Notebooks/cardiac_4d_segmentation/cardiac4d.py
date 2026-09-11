#!/usr/bin/env python
"""Reproducible, local-only utilities for cardiac 4D CT segmentation.

The script deliberately keeps DICOM identifiers and absolute source paths out of
committed artifacts.  It inventories candidate 4D series, converts one selected
series into one NIfTI volume per cardiac phase, combines heart-chamber masks into
a blood-pool mask, exports meshes, and creates quantitative QC manifests.

This is research software, not a medical device.  Automatic masks remain
``pending_manual_review`` until an expert has inspected every phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
import pydicom
from nibabel.affines import apply_affine
from pydicom.errors import InvalidDicomError
from scipy import ndimage
from skimage import measure
import trimesh


ALGORITHM_VERSION = "cardiac4d_v0.1.0"
PHASE_KEYWORDS = (
    "NominalPercentageOfCardiacPhase",
    "TemporalPositionIdentifier",
    "TriggerTime",
    "AcquisitionNumber",
)
INVENTORY_TAGS = [
    "Modality",
    "SOPClassUID",
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "FrameOfReferenceUID",
    "SeriesNumber",
    "SeriesDescription",
    "InstanceNumber",
    "AcquisitionNumber",
    "TemporalPositionIdentifier",
    "TriggerTime",
    "NominalPercentageOfCardiacPhase",
    "CardiacNumberOfImages",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
    "ImageType",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "PixelRepresentation",
    "BitsAllocated",
    "BitsStored",
    "RescaleSlope",
    "RescaleIntercept",
    "ConvolutionKernel",
    "ContrastBolusAgent",
    "ContrastBolusVolume",
    "Manufacturer",
    "ManufacturerModelName",
    "PatientPosition",
]
HEART_CHAMBER_MASKS = (
    "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
)
MYOCARDIUM_MASK = "heart_myocardium.nii.gz"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, Sequence):
        return [json_value(item) for item in value]
    return str(value)


def dicom_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file())


def read_header(path: Path, tags: Sequence[str] = INVENTORY_TAGS):
    return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(tags))


def vector(value: Any, length: int, name: str) -> np.ndarray:
    if value is None or len(value) != length:
        raise ValueError(f"{name}: expected {length} values")
    result = np.asarray([float(item) for item in value], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: contains non-finite values")
    return result


def unit(value: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 0:
        raise ValueError(f"{name}: zero-length vector")
    return value / length


def slice_coordinate(dataset) -> float | None:
    if not hasattr(dataset, "ImagePositionPatient") or not hasattr(
        dataset, "ImageOrientationPatient"
    ):
        return None
    orientation = vector(dataset.ImageOrientationPatient, 6, "ImageOrientationPatient")
    normal = unit(np.cross(orientation[:3], orientation[3:]), "slice normal")
    position = vector(dataset.ImagePositionPatient, 3, "ImagePositionPatient")
    return float(np.dot(normal, position))


def rounded_number(value: Any, digits: int = 6) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def geometry_signature(dataset) -> tuple[Any, ...]:
    return (
        int(getattr(dataset, "Rows", 0)),
        int(getattr(dataset, "Columns", 0)),
        tuple(round(float(x), 6) for x in getattr(dataset, "PixelSpacing", [])),
        tuple(
            round(float(x), 6)
            for x in getattr(dataset, "ImageOrientationPatient", [])
        ),
    )


def phase_candidate(records: Sequence[tuple[Path, Any]], keyword: str) -> dict[str, Any]:
    groups: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    missing = 0
    for record in records:
        value = getattr(record[1], keyword, None)
        key = rounded_number(value)
        if key is None:
            missing += 1
        else:
            groups[key].append(record)

    counts = sorted(len(items) for items in groups.values())
    group_qc = []
    for key, items in sorted(groups.items()):
        coordinates = [slice_coordinate(dataset) for _, dataset in items]
        valid_coordinates = [value for value in coordinates if value is not None]
        rounded_coordinates = [round(value, 3) for value in valid_coordinates]
        group_qc.append(
            {
                "value": key,
                "instances": len(items),
                "unique_slice_positions": len(set(rounded_coordinates)),
                "duplicate_slice_positions": len(rounded_coordinates)
                - len(set(rounded_coordinates)),
            }
        )

    usable = (
        missing == 0
        and len(groups) > 1
        and len(set(counts)) == 1
        and all(
            item["instances"] == item["unique_slice_positions"]
            for item in group_qc
        )
    )
    return {
        "keyword": keyword,
        "missing_instances": missing,
        "group_count": len(groups),
        "group_sizes": counts,
        "usable_complete_partition": usable,
        "groups": group_qc,
    }


def inspect_series(folder: Path, subject_id: str | None = None) -> dict[str, Any]:
    groups: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    rejected = 0
    for path in dicom_files(folder):
        try:
            dataset = read_header(path)
        except (InvalidDicomError, OSError, ValueError):
            rejected += 1
            continue
        if str(getattr(dataset, "Modality", "")) != "CT":
            continue
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not series_uid:
            rejected += 1
            continue
        groups[series_uid].append((path, dataset))

    series = []
    for series_uid, records in sorted(
        groups.items(),
        key=lambda item: str(getattr(item[1][0][1], "SeriesNumber", "")),
    ):
        first = records[0][1]
        coordinates = [slice_coordinate(dataset) for _, dataset in records]
        valid_coordinates = [value for value in coordinates if value is not None]
        signatures = Counter(geometry_signature(dataset) for _, dataset in records)
        phase_candidates = [phase_candidate(records, key) for key in PHASE_KEYWORDS]
        image_types = sorted(
            {
                "\\".join(str(item) for item in getattr(dataset, "ImageType", []))
                for _, dataset in records
            }
        )
        series.append(
            {
                "series_number": str(getattr(first, "SeriesNumber", "")),
                "series_description": str(
                    getattr(first, "SeriesDescription", "")
                ).strip(),
                "series_uid_sha256": sha256_text(series_uid),
                "study_uid_sha256": sha256_text(
                    str(getattr(first, "StudyInstanceUID", ""))
                ),
                "frame_of_reference_uid_sha256": sha256_text(
                    str(getattr(first, "FrameOfReferenceUID", ""))
                ),
                "instances": len(records),
                "unique_slice_positions_0_001mm": len(
                    {round(value, 3) for value in valid_coordinates}
                ),
                "geometry_signatures": [
                    {"signature": json_value(signature), "instances": count}
                    for signature, count in signatures.most_common()
                ],
                "image_types": image_types,
                "cardiac_number_of_images": sorted(
                    {
                        str(getattr(dataset, "CardiacNumberOfImages", ""))
                        for _, dataset in records
                        if hasattr(dataset, "CardiacNumberOfImages")
                    }
                ),
                "contrast_metadata_present": any(
                    hasattr(dataset, "ContrastBolusAgent")
                    or hasattr(dataset, "ContrastBolusVolume")
                    for _, dataset in records
                ),
                "manufacturer": str(getattr(first, "Manufacturer", "")).strip(),
                "model": str(getattr(first, "ManufacturerModelName", "")).strip(),
                "patient_position": str(
                    getattr(first, "PatientPosition", "")
                ).strip(),
                "phase_candidates": phase_candidates,
            }
        )

    return {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "subject_id": subject_id,
        "source_folder_name": folder.name,
        "absolute_source_path_omitted": True,
        "rejected_non_ct_or_invalid_files": rejected,
        "series": series,
        "runtime": {
            "python": platform.python_version(),
            "pydicom": pydicom.__version__,
        },
        "status": "inventory_only_not_segmentation",
    }


def choose_phase_keyword(records: Sequence[tuple[Path, Any]], requested: str) -> str:
    if requested != "auto":
        candidate = phase_candidate(records, requested)
        if not candidate["usable_complete_partition"]:
            raise ValueError(
                f"{requested} does not form a complete phase partition: {candidate}"
            )
        return requested
    for keyword in PHASE_KEYWORDS:
        candidate = phase_candidate(records, keyword)
        if candidate["usable_complete_partition"]:
            return keyword
    raise ValueError(
        "No DICOM phase tag forms a complete partition. Inspect the inventory and "
        "define a verified grouping rule before conversion."
    )


def load_single_series(folder: Path) -> list[tuple[Path, Any]]:
    groups: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    for path in dicom_files(folder):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except (InvalidDicomError, OSError, ValueError):
            continue
        if str(getattr(dataset, "Modality", "")) != "CT":
            continue
        uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if uid:
            groups[uid].append((path, dataset))
    if len(groups) != 1:
        raise ValueError(
            f"Selected folder must contain exactly one CT SeriesInstanceUID; found {len(groups)}"
        )
    return next(iter(groups.values()))


def dicom_phase_to_nifti(
    records: Sequence[tuple[Path, Any]], output_path: Path
) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("A 3D phase requires at least two slices")
    first = records[0][1]
    orientation = vector(first.ImageOrientationPatient, 6, "ImageOrientationPatient")
    row_direction = unit(orientation[:3], "DICOM row direction")
    column_direction = unit(orientation[3:], "DICOM column direction")
    normal = unit(np.cross(row_direction, column_direction), "DICOM slice normal")
    spacing = vector(first.PixelSpacing, 2, "PixelSpacing")

    sortable = []
    for path, header in records:
        if geometry_signature(header) != geometry_signature(first):
            raise ValueError("Geometry changes within one cardiac phase")
        position = vector(header.ImagePositionPatient, 3, "ImagePositionPatient")
        sortable.append((float(np.dot(normal, position)), path, header, position))
    sortable.sort(key=lambda item: item[0])
    coordinates = np.asarray([item[0] for item in sortable], dtype=np.float64)
    differences = np.diff(coordinates)
    if np.any(differences <= 1e-3):
        raise ValueError("Duplicate or reversed slice positions within a phase")
    slice_spacing = float(np.median(differences))
    if np.max(np.abs(differences - slice_spacing)) > 0.05:
        raise ValueError("Non-uniform slice spacing exceeds 0.05 mm")

    arrays = []
    slopes = []
    intercepts = []
    sop_hashes = []
    for _, path, header, _ in sortable:
        dataset = pydicom.dcmread(path)
        array = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        arrays.append(array * slope + intercept)
        slopes.append(slope)
        intercepts.append(intercept)
        sop_hashes.append(sha256_text(str(dataset.SOPInstanceUID)))
    volume = np.stack(arrays, axis=2)

    affine_lps = np.eye(4, dtype=np.float64)
    # numpy axis 0 is DICOM row; axis 1 is DICOM column.
    affine_lps[:3, 0] = column_direction * spacing[0]
    affine_lps[:3, 1] = row_direction * spacing[1]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = sortable[0][3]
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    image = nib.Nifti1Image(volume.astype(np.int16), affine_ras)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine_ras, code=1)
    image.set_sform(affine_ras, code=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output_path)
    return {
        "nifti": output_path.name,
        "shape": list(volume.shape),
        "voxel_spacing_mm": [float(spacing[0]), float(spacing[1]), slice_spacing],
        "affine_coordinate_system": "NIfTI RAS+",
        "dicom_source_coordinate_system": "DICOM patient LPS",
        "slice_count": len(records),
        "rescale_slope_values": sorted(set(slopes)),
        "rescale_intercept_values": sorted(set(intercepts)),
        "source_sop_uid_sha256": sorted(sop_hashes),
        "intensity_min_hu": int(np.min(volume)),
        "intensity_max_hu": int(np.max(volume)),
    }


def convert_series(
    folder: Path,
    output_dir: Path,
    phase_keyword: str,
    overwrite: bool,
) -> dict[str, Any]:
    records = load_single_series(folder)
    selected_keyword = choose_phase_keyword(records, phase_keyword)
    grouped: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    for record in records:
        value = rounded_number(getattr(record[1], selected_keyword, None))
        if value is None:
            raise ValueError(f"Missing {selected_keyword}")
        grouped[value].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    phases = []
    for index, (value, phase_records) in enumerate(
        sorted(grouped.items(), key=lambda item: float(item[0]))
    ):
        phase_id = f"phase_{index:02d}"
        phase_path = output_dir / f"{phase_id}.nii.gz"
        if phase_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {phase_path}")
        details = dicom_phase_to_nifti(phase_records, phase_path)
        details.update(
            {
                "phase_id": phase_id,
                "phase_index": index,
                "dicom_phase_keyword": selected_keyword,
                "dicom_phase_value": value,
            }
        )
        phases.append(details)

    manifest = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "source_series_folder_name": folder.name,
        "absolute_source_path_omitted": True,
        "phase_keyword": selected_keyword,
        "phase_count": len(phases),
        "phases": phases,
        "status": "converted_pending_image_qc",
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_totalsegmentator(
    input_dir: Path,
    output_dir: Path,
    executable: str,
    task: str,
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    phase_files = sorted(input_dir.glob("phase_*.nii.gz"))
    if not phase_files:
        raise FileNotFoundError(f"No phase_*.nii.gz files in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for phase_path in phase_files:
        phase_output = output_dir / phase_path.name.removesuffix(".nii.gz")
        expected = phase_output / MYOCARDIUM_MASK
        if expected.exists() and not overwrite:
            runs.append(
                {"phase_id": phase_output.name, "status": "reused_existing_output"}
            )
            continue
        phase_output.mkdir(parents=True, exist_ok=True)
        command = [executable, "-i", str(phase_path), "-o", str(phase_output)]
        if task == "total_v1":
            command += [
                "--roi_subset",
                "heart_myocardium",
                "heart_atrium_left",
                "heart_ventricle_left",
                "heart_atrium_right",
                "heart_ventricle_right",
            ]
        else:
            command += ["--task", task]
        if device:
            command += ["--device", device]
        started = utc_now()
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"TotalSegmentator failed for {phase_path.name} with code {result.returncode}"
            )
        runs.append(
            {
                "phase_id": phase_output.name,
                "status": "automatic_segmentation_pending_manual_review",
                "started_at": started,
                "finished_at": utc_now(),
                "command": [Path(command[0]).name, *command[1:]],
            }
        )
    manifest = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "backend": "TotalSegmentator",
        "task": task,
        "runs": runs,
        "status": "automatic_segmentation_pending_manual_review",
    }
    (output_dir / "segmentation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def same_grid(reference: nib.Nifti1Image, candidate: nib.Nifti1Image) -> bool:
    return reference.shape == candidate.shape and np.allclose(
        reference.affine, candidate.affine, atol=1e-4
    )


def clean_binary(mask: np.ndarray, min_component_ml: float, voxel_ml: float) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=bool)
    labels, count = ndimage.label(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    minimum_voxels = max(1, int(math.ceil(min_component_ml / voxel_ml)))
    keep = np.flatnonzero(sizes >= minimum_voxels)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def mask_metrics(mask: np.ndarray, affine: np.ndarray) -> dict[str, Any]:
    voxel_volume_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    voxel_count = int(np.count_nonzero(mask))
    labels, components = ndimage.label(mask)
    sizes = np.bincount(labels.ravel()) if components else np.asarray([0])
    component_sizes = sorted((int(value) for value in sizes[1:]), reverse=True)
    coordinates = np.argwhere(mask)
    if len(coordinates):
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        corners = np.asarray(
            [
                [i, j, k]
                for i in (minimum[0], maximum[0])
                for j in (minimum[1], maximum[1])
                for k in (minimum[2], maximum[2])
            ]
        )
        world = apply_affine(affine, corners)
        bbox_ras = [world.min(axis=0).tolist(), world.max(axis=0).tolist()]
    else:
        bbox_ras = None
    return {
        "voxel_count": voxel_count,
        "voxel_volume_mm3": voxel_volume_mm3,
        "volume_ml": voxel_count * voxel_volume_mm3 / 1000.0,
        "connected_components": int(components),
        "largest_component_fraction": (
            component_sizes[0] / voxel_count if voxel_count and component_sizes else 0.0
        ),
        "bounding_box_ras_mm": bbox_ras,
    }


def export_mesh(
    mask: np.ndarray, affine: np.ndarray, output_path: Path
) -> dict[str, Any]:
    if not np.any(mask):
        raise ValueError("Cannot export an empty mask")
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    vertices -= 1.0
    vertices_ras = apply_affine(affine, vertices)
    mesh = trimesh.Trimesh(vertices=vertices_ras, faces=faces, process=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return {
        "file": output_path.name,
        "coordinate_system": "NIfTI RAS+",
        "units": "mm (STL has no embedded unit metadata)",
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "mesh_volume_ml_if_watertight": (
            abs(float(mesh.volume)) / 1000.0 if mesh.is_watertight else None
        ),
    }


def build_products(
    segmentation_dir: Path,
    output_dir: Path,
    min_component_ml: float,
    overwrite: bool,
) -> dict[str, Any]:
    phase_dirs = sorted(path for path in segmentation_dir.glob("phase_*") if path.is_dir())
    if not phase_dirs:
        raise FileNotFoundError(f"No phase_* segmentation folders in {segmentation_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    phases = []
    for phase_dir in phase_dirs:
        output_phase = output_dir / phase_dir.name
        output_phase.mkdir(parents=True, exist_ok=True)
        images = [nib.load(str(phase_dir / name)) for name in HEART_CHAMBER_MASKS]
        myocardium_image = nib.load(str(phase_dir / MYOCARDIUM_MASK))
        reference = myocardium_image
        if not all(same_grid(reference, image) for image in images):
            raise ValueError(f"Mask grids differ in {phase_dir}")
        voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
        chamber_masks = [np.asanyarray(image.dataobj) > 0 for image in images]
        myocardium = np.asanyarray(myocardium_image.dataobj) > 0
        blood_pool = np.logical_or.reduce(chamber_masks)
        blood_pool = clean_binary(blood_pool, min_component_ml, voxel_ml)
        myocardium = clean_binary(myocardium, min_component_ml, voxel_ml)

        overlap = np.logical_and(blood_pool, myocardium)
        if np.any(overlap):
            myocardium = np.logical_and(myocardium, ~blood_pool)

        products = {}
        for label, mask in (("blood_pool", blood_pool), ("myocardium", myocardium)):
            nifti_path = output_phase / f"{label}.nii.gz"
            mesh_path = output_phase / f"{label}.stl"
            if (nifti_path.exists() or mesh_path.exists()) and not overwrite:
                raise FileExistsError(f"Refusing to overwrite products in {output_phase}")
            image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
            image.set_data_dtype(np.uint8)
            nib.save(image, nifti_path)
            metrics = mask_metrics(mask, reference.affine)
            mesh = export_mesh(mask, reference.affine, mesh_path)
            products[label] = {"metrics": metrics, "mesh": mesh}
            rows.append(
                {
                    "phase_id": phase_dir.name,
                    "label": label,
                    "volume_ml": f"{metrics['volume_ml']:.6f}",
                    "connected_components": metrics["connected_components"],
                    "largest_component_fraction": (
                        f"{metrics['largest_component_fraction']:.6f}"
                    ),
                    "mesh_watertight": mesh["watertight"],
                }
            )
        phases.append(
            {
                "phase_id": phase_dir.name,
                "overlap_voxels_removed_from_myocardium": int(np.count_nonzero(overlap)),
                "products": products,
                "qc_status": "pending_manual_review",
            }
        )

    csv_path = output_dir / "volumes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    temporal = {}
    for label in ("blood_pool", "myocardium"):
        values = [phase["products"][label]["metrics"]["volume_ml"] for phase in phases]
        jumps = [
            abs(values[index] - values[index - 1])
            / max(values[index - 1], 1e-12)
            for index in range(1, len(values))
        ]
        temporal[label] = {
            "minimum_volume_ml": min(values),
            "maximum_volume_ml": max(values),
            "range_ml": max(values) - min(values),
            "max_adjacent_relative_jump": max(jumps) if jumps else 0.0,
            "volumes_ml": values,
        }

    manifest = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utc_now(),
        "definitions": {
            "blood_pool": "union of four TotalSegmentator chamber masks; great vessels excluded",
            "myocardium": "TotalSegmentator heart_myocardium mask with blood-pool overlap removed",
            "volumes": "computed from binary voxels; STL smoothing is not used",
        },
        "postprocessing": {
            "minimum_component_volume_ml": min_component_ml,
            "overlap_rule": "blood pool takes precedence over myocardium",
        },
        "phases": phases,
        "temporal_qc": temporal,
        "status": "automatic_products_pending_manual_review",
    }
    (output_dir / "qc_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def command_inventory(args) -> None:
    manifest = inspect_series(Path(args.folder).resolve(), args.subject_id)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_convert(args) -> None:
    manifest = convert_series(
        Path(args.series_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.phase_key,
        args.overwrite,
    )
    print(f"Converted {manifest['phase_count']} phases using {manifest['phase_keyword']}")


def command_segment(args) -> None:
    manifest = run_totalsegmentator(
        Path(args.input_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.executable,
        args.task,
        args.device,
        args.overwrite,
    )
    print(f"Prepared {len(manifest['runs'])} automatic segmentations")


def command_products(args) -> None:
    manifest = build_products(
        Path(args.segmentation_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.min_component_ml,
        args.overwrite,
    )
    print(f"Built NIfTI/STL products for {len(manifest['phases'])} phases")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inspect CT and cardiac phase tags")
    inventory.add_argument("--folder", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--subject-id")
    inventory.set_defaults(function=command_inventory)

    convert = subparsers.add_parser("convert", help="Convert one 4D DICOM series to phase NIfTI files")
    convert.add_argument("--series-dir", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--phase-key", default="auto", choices=("auto", *PHASE_KEYWORDS))
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(function=command_convert)

    segment = subparsers.add_parser("segment", help="Run TotalSegmentator for all phase NIfTI files")
    segment.add_argument("--input-dir", required=True)
    segment.add_argument("--output-dir", required=True)
    segment.add_argument("--executable", default="TotalSegmentator")
    segment.add_argument(
        "--task", default="total_v1", choices=("total_v1", "heartchambers_highres")
    )
    segment.add_argument("--device", default="gpu")
    segment.add_argument("--overwrite", action="store_true")
    segment.set_defaults(function=command_segment)

    products = subparsers.add_parser("products", help="Build blood-pool/myocardium NIfTI, STL, and QC")
    products.add_argument("--segmentation-dir", required=True)
    products.add_argument("--output-dir", required=True)
    products.add_argument("--min-component-ml", type=float, default=0.10)
    products.add_argument("--overwrite", action="store_true")
    products.set_defaults(function=command_products)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
