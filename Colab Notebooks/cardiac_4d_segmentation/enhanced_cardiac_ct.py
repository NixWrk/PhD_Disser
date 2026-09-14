#!/usr/bin/env python
"""Convert one Enhanced CT multiframe instance per phase, without resampling.

Local input JSON: {"phases": [{"source_path": "phase.dcm", "phase_percent": 0}]}.
Relative source paths resolve against that JSON, never the working directory.
Use pseudonymous --case-id and --sequence-id. Only those supplied identifiers,
file hashes and whitelisted geometry enter the portable output manifest.

Pass <sequence_dir>/phases as --output-dir. This directory must not exist.
Outputs are published only after every
phase has passed validation. Phase percentages come from the input manifest;
they do not establish a measured cycle, RR interval or a time axis in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


ALGORITHM_VERSION = "enhanced_cardiac_ct_v0.1.0"
ENHANCED_CT_UID = "1.2.840.10008.5.1.4.1.1.2.1"
POSITION_ATOL_MM = 1e-3
ORIENTATION_ATOL = 1e-5
PHASE_KEY = "manifest.phase_percent"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_vector(value, size: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {label}") from None
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Nonfinite or malformed {label}")
    return result


def group_item(frame, shared, keyword: str):
    """Use a per-frame macro when present; otherwise require the shared macro."""
    for group, origin in ((frame, "per_frame"), (shared, "shared")):
        if hasattr(group, keyword):
            sequence = getattr(group, keyword)
            if len(sequence) != 1:
                raise ValueError(f"{keyword} must contain exactly one item")
            return sequence[0], origin
    raise ValueError(f"Missing {keyword} in Functional Groups")


def read_phase(source_path: Path) -> tuple[nib.Nifti1Image, dict]:
    """Read one 3D phase; output axes are rows, columns, sorted slices."""
    source_hash = sha256_file(source_path)
    try:
        dataset = pydicom.dcmread(source_path)
    except Exception:
        raise ValueError("Cannot read Enhanced CT source") from None
    if str(getattr(dataset, "SOPClassUID", "")) != ENHANCED_CT_UID:
        raise ValueError("Expected Enhanced CT Image Storage")
    if str(getattr(dataset, "Modality", "")) != "CT":
        raise ValueError("Expected CT modality")
    if int(getattr(dataset, "SamplesPerPixel", 0)) != 1:
        raise ValueError("Expected scalar CT pixels")
    count = int(getattr(dataset, "NumberOfFrames", 0))
    if count < 2:
        raise ValueError("A 3D phase requires at least two frames")
    frames = getattr(dataset, "PerFrameFunctionalGroupsSequence", [])
    shared_sequence = getattr(dataset, "SharedFunctionalGroupsSequence", [])
    if len(frames) != count or len(shared_sequence) > 1:
        raise ValueError("Functional Groups count does not match NumberOfFrames")
    shared = shared_sequence[0] if shared_sequence else pydicom.Dataset()

    positions, orientations, spacings, thicknesses = [], [], [], []
    slopes, intercepts, rescale_types = [], [], []
    origins = {key: [] for key in (
        "PlanePositionSequence", "PlaneOrientationSequence",
        "PixelMeasuresSequence", "PixelValueTransformationSequence",
    )}
    for frame in frames:
        macros = {}
        for key in origins:
            macros[key], origin = group_item(frame, shared, key)
            origins[key].append(origin)
        positions.append(finite_vector(
            getattr(macros["PlanePositionSequence"], "ImagePositionPatient", None),
            3, "ImagePositionPatient"))
        orientations.append(finite_vector(
            getattr(macros["PlaneOrientationSequence"], "ImageOrientationPatient", None),
            6, "ImageOrientationPatient"))
        measures = macros["PixelMeasuresSequence"]
        spacings.append(finite_vector(getattr(measures, "PixelSpacing", None), 2, "PixelSpacing"))
        thicknesses.append(finite_vector(
            [getattr(measures, "SliceThickness", None)], 1, "SliceThickness")[0])
        transform = macros["PixelValueTransformationSequence"]
        scale = finite_vector([getattr(transform, "RescaleSlope", None),
                               getattr(transform, "RescaleIntercept", None)], 2, "rescale")
        if scale[0] == 0:
            raise ValueError("RescaleSlope must be nonzero")
        rescale_type = str(getattr(transform, "RescaleType", "")).strip().upper()
        if rescale_type != "HU":
            raise ValueError("PixelValueTransformationSequence must declare RescaleType HU")
        slopes.append(scale[0]); intercepts.append(scale[1]); rescale_types.append(rescale_type)

    positions = np.asarray(positions)
    orientations = np.asarray(orientations)
    spacing = np.asarray(spacings)
    thicknesses = np.asarray(thicknesses)
    u, v = orientations[0, :3], orientations[0, 3:]
    if not (np.isclose(np.linalg.norm(u), 1, rtol=0, atol=ORIENTATION_ATOL)
            and np.isclose(np.linalg.norm(v), 1, rtol=0, atol=ORIENTATION_ATOL)
            and abs(np.dot(u, v)) <= ORIENTATION_ATOL):
        raise ValueError("ImageOrientationPatient must be orthonormal")
    if not np.allclose(orientations, orientations[0], rtol=0, atol=ORIENTATION_ATOL):
        raise ValueError("Mixed orientation within phase")
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=0, atol=POSITION_ATOL_MM):
        raise ValueError("Mixed or nonpositive PixelSpacing")
    if np.any(thicknesses <= 0) or not np.allclose(thicknesses, thicknesses[0], rtol=0, atol=POSITION_ATOL_MM):
        raise ValueError("Mixed or nonpositive SliceThickness")

    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    coordinates = positions @ normal
    order = np.argsort(coordinates)
    sorted_positions = positions[order]
    distances = np.diff(coordinates[order])
    if np.any(distances <= POSITION_ATOL_MM):
        raise ValueError("Duplicate slice positions within phase")
    step = float(np.median(distances))
    if not np.allclose(distances, step, rtol=0, atol=POSITION_ATOL_MM):
        raise ValueError("Nonuniform slice spacing")
    expected_positions = sorted_positions[0] + np.arange(count)[:, None] * step * normal
    if not np.allclose(sorted_positions, expected_positions, rtol=0, atol=POSITION_ATOL_MM):
        raise ValueError("Slice positions do not form an orthogonal regular stack")

    try:
        pixels = dataset.pixel_array
    except Exception:
        raise ValueError("Cannot decode pixel data with the installed DICOM dependencies") from None
    if pixels.shape != (count, int(dataset.Rows), int(dataset.Columns)):
        raise ValueError("Expected pixel axes frames, rows, columns")
    # Apply each frame's own transform BEFORE sorting and permuting axes.
    with np.errstate(over="ignore", invalid="ignore"):
        hu = pixels.astype(np.float64) * np.asarray(slopes)[:, None, None]
        hu += np.asarray(intercepts)[:, None, None]
        volume = np.transpose(hu[order], (1, 2, 0)).astype(np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Nonfinite HU values")

    affine_lps = np.eye(4)
    affine_lps[:3, 0] = v * spacing[0, 0]  # row index increments along second IOP vector
    affine_lps[:3, 1] = u * spacing[0, 1]  # column index increments along first IOP vector
    affine_lps[:3, 2] = normal * step
    affine_lps[:3, 3] = sorted_positions[0]
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    image = nib.Nifti1Image(volume, affine_ras)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine_ras, code=1)
    image.set_sform(affine_ras, code=1)

    def uid_hash(keyword):
        value = str(getattr(dataset, keyword, ""))
        return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None

    sop_hash = uid_hash("SOPInstanceUID")
    details = {
        "source_sha256": source_hash,
        "source_sop_uid_sha256": [sop_hash] if sop_hash else [],
        "frame_of_reference_uid_sha256": uid_hash("FrameOfReferenceUID"),
        "shape": list(volume.shape), "slice_count": count,
        "voxel_spacing_mm": [float(spacing[0, 0]), float(spacing[0, 1]), step],
        "slice_thickness_mm": float(thicknesses[0]),
        "slice_step_basis": "sorted ImagePositionPatient projected on slice normal",
        "affine_coordinate_system": "NIfTI RAS+",
        "dicom_source_coordinate_system": "DICOM patient LPS",
        "affine_ras_mm": affine_ras.tolist(), "spatial_units": "mm",
        "intensity_units": "HU", "output_dtype": "float32",
        "rescale_type_values": sorted(set(rescale_types)),
        "rescale_slope_values": sorted(set(float(x) for x in slopes)),
        "rescale_intercept_values": sorted(set(float(x) for x in intercepts)),
        "rescale_slope_by_slice": [float(slopes[i]) for i in order],
        "rescale_intercept_by_slice": [float(intercepts[i]) for i in order],
        "source_frame_numbers_1based": (order + 1).tolist(),
        "source_order": order.tolist(),
        "source_order_definition": "zero-based DICOM frame index for each output slice",
        "functional_group_sources_by_slice": {key: [values[i] for i in order] for key, values in origins.items()},
        "intensity_min_hu": float(volume.min()), "intensity_max_hu": float(volume.max()),
        "resampling_performed": False, "registration_performed": False,
    }
    if sha256_file(source_path) != source_hash:
        raise ValueError("Source changed during conversion")
    return image, details


def convert_manifest(manifest_path: Path, output_dir: Path, case_id: str, sequence_id: str) -> dict:
    for label, value in (("case_id", case_id), ("sequence_id", sequence_id)):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
            raise ValueError(f"{label} must be a pseudonymous ASCII identifier (1-64 characters)")
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new directory")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("phases") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest requires a nonempty phases list")
    for label, value in (("case_id", case_id), ("sequence_id", sequence_id)):
        if label in manifest and manifest[label] != value:
            raise ValueError(f"CLI {label} differs from the input manifest")
    selected = []
    seen_paths = set()
    for manifest_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("source_path"), str) or not entry["source_path"]:
            raise ValueError("Each phase requires source_path and phase_percent")
        value = entry.get("phase_percent")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value < 100:
            raise ValueError("phase_percent must be finite and in [0, 100)")
        source = Path(entry["source_path"])
        source = (manifest_path.parent / source).resolve()
        if not source.is_file():
            raise ValueError("A manifest source is not an existing local file")
        if source in seen_paths:
            raise ValueError("Duplicate source_path")
        seen_paths.add(source)
        selected.append((float(value), source, manifest_index))
    selected.sort(key=lambda item: item[0])
    if any(b[0] - a[0] <= 1e-8 for a, b in zip(selected, selected[1:])):
        raise ValueError("Duplicate phase_percent")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    phases = []
    seen_hashes = set()
    first_grid = None
    with tempfile.TemporaryDirectory(prefix=".enhanced-ct-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "phases"
        staged.mkdir()
        for index, (percent, source, manifest_index) in enumerate(selected):
            image, details = read_phase(source)
            if details["source_sha256"] in seen_hashes:
                raise ValueError("Duplicate source content across phases")
            seen_hashes.add(details["source_sha256"])
            if first_grid is None:
                first_grid = (image.shape, image.affine.copy(), details)
            else:
                shape, affine, first = first_grid
                if image.shape != shape or not np.allclose(image.affine, affine, rtol=0, atol=POSITION_ATOL_MM):
                    raise ValueError("Mixed geometry across phases; no registration or resampling is performed")
                if abs(details["slice_thickness_mm"] - first["slice_thickness_mm"]) > POSITION_ATOL_MM:
                    raise ValueError("Mixed SliceThickness across phases")
                if details["frame_of_reference_uid_sha256"] != first["frame_of_reference_uid_sha256"]:
                    raise ValueError("Mixed FrameOfReference across phases")
            phase_id = f"phase_{index:02d}"
            filename = f"{phase_id}.nii.gz"
            nib.save(image, staged / filename)
            phases.append({**details, "nifti": filename, "phase_id": phase_id,
                           "phase_index": index, "phase_percent": percent,
                           "source_manifest_phase_index": manifest_index,
                           "dicom_phase_keyword": PHASE_KEY, "dicom_phase_value": percent,
                           "phase_percent_source": "explicit_local_manifest",
                           "sha256": sha256_file(staged / filename)})
        result = {
            "schema_version": 1, "algorithm_version": ALGORITHM_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id, "sequence_id": sequence_id,
            "source_format": "Enhanced CT: one multiframe instance per 3D phase",
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "converter_sha256": sha256_file(Path(__file__)),
            "absolute_source_path_omitted": True, "raw_identifiers_omitted": True,
            "phase_keyword": PHASE_KEY, "phase_count": len(phases), "phases": phases,
            "time_basis": "manifest phase percentages only; no seconds, RR or cycle identity inferred",
            "geometry_tolerance_mm": POSITION_ATOL_MM,
            "orientation_tolerance": ORIENTATION_ATOL,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "nibabel": nib.__version__, "pydicom": pydicom.__version__},
            "status": "converted_pending_image_qc",
        }
        (staged / "conversion_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Output directory appeared during conversion")
        staged.rename(output_dir)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--sequence-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = convert_manifest(args.manifest, args.output_dir, args.case_id, args.sequence_id)
    except (ValueError, OSError) as exc:
        # OS errors may contain local paths. Do not echo those paths to logs.
        parser.exit(2, ("Conversion failed: local file access error\n" if isinstance(exc, OSError)
                        else f"Conversion failed: {exc}\n"))
    print(f"Converted {result['phase_count']} phases; status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
