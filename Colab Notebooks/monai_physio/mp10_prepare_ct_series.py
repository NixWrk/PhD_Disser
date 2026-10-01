"""Convert one classic single-frame DICOM CT series to an RAS-mm NIfTI.

Source DICOM files are read only. The caller supplies the input and output
paths; no medical path or identifier is embedded in this script or sidecar.
Mixed flat directories are filtered by DICOM SeriesNumber, then checked for
one frame of reference and regular physical slice positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def convert(source: Path, series_number: int, output: Path, alias: str) -> dict:
    selected = []
    for file in source.glob("*.dcm"):
        header = pydicom.dcmread(file, stop_before_pixels=True)
        if int(getattr(header, "SeriesNumber", -1)) == series_number:
            selected.append((file, header))
    if len(selected) < 2:
        raise ValueError("A volumetric series needs at least two DICOM slices")

    first = selected[0][1]
    rows, columns = int(first.Rows), int(first.Columns)
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    spacing_xy = np.asarray(first.PixelSpacing, dtype=float)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-4):
        raise ValueError("Invalid DICOM image orientation")

    frame_uid = str(first.FrameOfReferenceUID)
    entries = []
    for file, header in selected:
        if int(header.Rows) != rows or int(header.Columns) != columns:
            raise ValueError("Variable image size in selected series")
        if str(header.FrameOfReferenceUID) != frame_uid:
            raise ValueError("Multiple physical frames of reference")
        if not np.allclose(header.ImageOrientationPatient, orientation, atol=1e-5):
            raise ValueError("Variable slice orientation")
        if not np.allclose(header.PixelSpacing, spacing_xy, atol=1e-5):
            raise ValueError("Variable pixel spacing")
        if int(getattr(header, "NumberOfFrames", 1)) != 1:
            raise ValueError("Enhanced multi-frame DICOM needs another converter")
        origin = np.asarray(header.ImagePositionPatient, dtype=float)
        entries.append((float(np.dot(origin, normal)), file, origin))
    entries.sort(key=lambda item: item[0])
    positions = np.array([item[0] for item in entries])
    steps = np.diff(positions)
    step = float(np.median(steps))
    if step <= 0 or np.max(np.abs(steps - step)) > 0.01:
        raise ValueError("Irregular or repeated slice positions")

    volume = np.empty((len(entries), rows, columns), dtype=np.float32)
    source_hash = hashlib.sha256()
    for index, (_, file, _) in enumerate(entries):
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_hash.update(chunk)
        dicom = pydicom.dcmread(file)
        pixels = np.asarray(dicom.pixel_array, dtype=np.float32)
        volume[index] = (
            pixels * float(getattr(dicom, "RescaleSlope", 1))
            + float(getattr(dicom, "RescaleIntercept", 0))
        )

    lps = np.eye(4)
    lps[:3, 0] = row_direction * spacing_xy[1]
    lps[:3, 1] = column_direction * spacing_xy[0]
    lps[:3, 2] = normal * step
    lps[:3, 3] = entries[0][2]
    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps
    image = nib.Nifti1Image(volume.transpose(2, 1, 0), ras)
    image.header.set_xyzt_units("mm")
    image.set_qform(ras, code=1)
    image.set_sform(ras, code=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output)

    with output.open("rb") as stream:
        output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "schema": "mp10_single_series_ct_conversion_v1",
        "subject_alias": alias,
        "series_number": series_number,
        "source_dicom_count": len(entries),
        "ordered_source_bytes_sha256": source_hash.hexdigest(),
        "output_nifti_sha256": output_hash,
        "shape_xyz": list(image.shape),
        "voxel_spacing_mm": [float(spacing_xy[1]), float(spacing_xy[0]), step],
        "coverage_z_mm": float(positions[-1] - positions[0]),
        "coordinate_system": "NIfTI RAS+ mm; DICOM LPS converted explicitly",
        "hu_conversion": "per-slice RescaleSlope and RescaleIntercept",
        "respiratory_state": "author_reported_deep_inspiration_breath_hold",
        "note": "DICOM files were read only; no respiratory phase inferred from CT pixels",
    }
    sidecar = output.with_name(output.name + ".provenance.json")
    sidecar.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", required=True, type=Path)
    parser.add_argument("--series-number", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    result = convert(args.dicom_dir, args.series_number, args.output, args.alias)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
