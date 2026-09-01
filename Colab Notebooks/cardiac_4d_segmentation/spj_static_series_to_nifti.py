#!/usr/bin/env python
"""Convert one embedded static CT series from Inobitec SPJ directly to NIfTI.

No raw DICOM copies are created.  This is a fallback for a project that embeds a
static source series but does not contain the requested 4D phase series.
"""

from __future__ import annotations

import argparse
import io
import json
import mmap
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from cardiac4d import geometry_signature, sha256_text, unit, vector


def starts(mapped):
    result = []
    position = 0
    while True:
        found = mapped.find(b"DICM", position)
        if found < 0:
            return result
        if found >= 128:
            result.append(found - 128)
        position = found + 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spj", required=True)
    parser.add_argument("--series-number", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.spj).resolve()
    datasets = []
    with source.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            object_starts = starts(mapped)
            for index, start in enumerate(object_starts):
                end = (
                    object_starts[index + 1]
                    if index + 1 < len(object_starts)
                    else len(mapped)
                )
                chunk = io.BytesIO(mapped[start:end])
                try:
                    header = pydicom.dcmread(
                        chunk,
                        stop_before_pixels=True,
                        specific_tags=[
                            "Modality",
                            "SeriesNumber",
                            "SeriesInstanceUID",
                        ],
                    )
                except Exception:
                    continue
                if str(getattr(header, "Modality", "")) != "CT" or str(
                    getattr(header, "SeriesNumber", "")
                ) != str(args.series_number):
                    continue
                chunk.seek(0)
                dataset = pydicom.dcmread(chunk)
                datasets.append(dataset)

    if len(datasets) < 2:
        raise RuntimeError("Selected embedded CT series is missing or incomplete")
    first = datasets[0]
    orientation = vector(first.ImageOrientationPatient, 6, "ImageOrientationPatient")
    row_direction = unit(orientation[:3], "DICOM row direction")
    column_direction = unit(orientation[3:], "DICOM column direction")
    normal = unit(np.cross(row_direction, column_direction), "DICOM slice normal")
    spacing = vector(first.PixelSpacing, 2, "PixelSpacing")
    sortable = []
    for dataset in datasets:
        if geometry_signature(dataset) != geometry_signature(first):
            raise RuntimeError("Geometry changes inside selected embedded CT series")
        position = vector(dataset.ImagePositionPatient, 3, "ImagePositionPatient")
        sortable.append((float(np.dot(normal, position)), dataset, position))
    sortable.sort(key=lambda item: item[0])
    coordinates = np.asarray([item[0] for item in sortable], dtype=np.float64)
    differences = np.diff(coordinates)
    if np.any(differences <= 1e-3):
        raise RuntimeError("Duplicate or reversed embedded CT slice positions")
    slice_spacing = float(np.median(differences))
    if np.max(np.abs(differences - slice_spacing)) > 0.05:
        raise RuntimeError("Embedded CT has non-uniform slice spacing")

    arrays = []
    for _, dataset, _ in sortable:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        arrays.append(dataset.pixel_array.astype(np.float32) * slope + intercept)
    volume = np.stack(arrays, axis=2)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = column_direction * spacing[0]
    affine_lps[:3, 1] = row_direction * spacing[1]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = sortable[0][2]
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    image = nib.Nifti1Image(volume.astype(np.int16), affine_ras)
    image.set_qform(affine_ras, code=1)
    image.set_sform(affine_ras, code=1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_00.nii.gz"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    nib.save(image, output_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_container_name": source.name,
        "source_series_number": str(args.series_number),
        "series_uid_sha256": sha256_text(str(first.SeriesInstanceUID)),
        "phase_count": 1,
        "phases": [
            {
                "phase_id": "phase_00",
                "nifti": output_path.name,
                "shape": list(volume.shape),
                "voxel_spacing_mm": [
                    float(spacing[0]),
                    float(spacing[1]),
                    slice_spacing,
                ],
                "intensity_min_hu": int(np.min(volume)),
                "intensity_max_hu": int(np.max(volume)),
            }
        ],
        "status": "static_embedded_ct_not_4d",
        "warning": (
            "This SPJ exposed one static CT series only. It must not be used as a "
            "cardiac phase sequence or for dynamic volume estimation."
        ),
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Converted static embedded CT: {volume.shape}")


if __name__ == "__main__":
    main()

