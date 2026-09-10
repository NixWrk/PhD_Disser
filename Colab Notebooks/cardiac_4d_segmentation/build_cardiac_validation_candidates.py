#!/usr/bin/env python
"""Build explicitly named cardiac masks for independent manual validation.

The script does not improve anatomy by inference and does not overwrite the
existing 4D products.  It preserves the raw TotalSegmentator masks, constructs
only declared Boolean derivatives, and keeps the geometric residual
``total/heart minus four-chamber blood`` separate from myocardium.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


CHAMBER_FILES = {
    "left_atrium_blood": "heart_atrium_left.nii.gz",
    "left_ventricle_blood": "heart_ventricle_left.nii.gz",
    "right_atrium_blood": "heart_atrium_right.nii.gz",
    "right_ventricle_blood": "heart_ventricle_right.nii.gz",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{path}: expected 3D NIfTI, got {image.shape}")
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: non-finite values")
    if np.issubdtype(data.dtype, np.integer):
        if int(data.min()) < 0 or int(data.max()) > 1:
            raise ValueError(f"{path}: expected binary 0/1 mask")
    elif not np.all((data == 0) | (data == 1)):
        raise ValueError(f"{path}: expected binary 0/1 mask")
    mask = np.asarray(data == 1, dtype=bool)
    if not np.any(mask):
        raise ValueError(f"{path}: empty mask")
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    if int(qform_code) == 0 and int(sform_code) == 0:
        raise ValueError(f"{path}: both qform and sform codes are unset")
    if int(qform_code) > 0 and not np.allclose(qform, image.affine, atol=1e-4):
        raise ValueError(f"{path}: qform differs from selected affine")
    if int(sform_code) > 0 and not np.allclose(sform, image.affine, atol=1e-4):
        raise ValueError(f"{path}: sform differs from selected affine")
    return image, mask


def assert_same_grid(reference: nib.Nifti1Image, candidate: nib.Nifti1Image, context: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference.affine, candidate.affine, atol=1e-4
    ):
        raise ValueError(f"{context}: TotalSegmentator output grids differ")
    reference_unit = reference.header.get_xyzt_units()[0] or "unknown"
    candidate_unit = candidate.header.get_xyzt_units()[0] or "unknown"
    if reference_unit != candidate_unit:
        raise ValueError(
            f"{context}: spatial units differ ({reference_unit!r} != {candidate_unit!r})"
        )


def save_binary(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> dict[str, Any]:
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    image.header.set_xyzt_units("mm")
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    nib.save(image, str(path))
    voxel_volume_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
    voxel_count = int(np.count_nonzero(mask))
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "voxel_count": voxel_count,
        "volume_ml": voxel_count * voxel_volume_ml,
    }


def segmentation_provenance(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation manifest not found: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source_filename": path.name,
        "sha256": sha256_file(path),
        "backend": source.get("backend"),
        "backend_version": source.get("backend_version"),
        "python": source.get("python"),
        "torch": source.get("torch"),
        "cuda_runtime": source.get("cuda_runtime"),
        "gpu": source.get("gpu"),
        "models": source.get("models"),
        "source_status": source.get("status"),
        "absolute_path_omitted": True,
    }


def build_phase(
    phase_dir: Path,
    output_phase: Path,
    overwrite: bool = False,
    source_ct_path: Path | None = None,
    geometry_mm_confirmed: bool = False,
) -> dict[str, Any]:
    total_path = phase_dir / "total_heart" / "heart.nii.gz"
    highres_dir = phase_dir / "heartchambers_highres"
    myocardium_path = highres_dir / "heart_myocardium.nii.gz"
    total_image, total_heart = load_binary(total_path)
    input_spatial_unit = total_image.header.get_xyzt_units()[0] or "unknown"
    if input_spatial_unit != "mm" and not geometry_mm_confirmed:
        raise ValueError(
            f"{phase_dir.name}: spatial unit is {input_spatial_unit!r}; "
            "use --geometry-mm-confirmed only after checking DICOM/conversion provenance"
        )
    myocardium_image, myocardium_raw = load_binary(myocardium_path)
    assert_same_grid(total_image, myocardium_image, f"{phase_dir.name} myocardium")

    chamber_masks: dict[str, np.ndarray] = {}
    chamber_sources: dict[str, dict[str, Any]] = {}
    for label, filename in CHAMBER_FILES.items():
        chamber_path = highres_dir / filename
        image, mask = load_binary(chamber_path)
        assert_same_grid(total_image, image, f"{phase_dir.name} {label}")
        chamber_masks[label] = mask
        chamber_sources[label] = {
            "source_filename": filename,
            "sha256": sha256_file(chamber_path),
        }

    ct_source = None
    if source_ct_path is not None:
        source_ct_path = source_ct_path.resolve()
        if not source_ct_path.is_file():
            raise FileNotFoundError(f"Source CT phase not found: {source_ct_path}")
        ct_image = nib.load(str(source_ct_path))
        assert_same_grid(total_image, ct_image, f"{phase_dir.name} source CT")
        ct_source = {
            "source_filename": source_ct_path.name,
            "sha256": sha256_file(source_ct_path),
            "absolute_path_omitted": True,
        }

    blood_pool = np.logical_or.reduce(list(chamber_masks.values()))
    myocardium_no_blood_overlap = myocardium_raw & ~blood_pool
    geometric_residual = total_heart & ~blood_pool
    products = {
        "totalsegmentator_total_heart_raw": total_heart,
        **{
            f"totalsegmentator_heartchambers_highres_{label}_raw": mask
            for label, mask in chamber_masks.items()
        },
        "totalsegmentator_heartchambers_highres_blood_union": blood_pool,
        "totalsegmentator_heartchambers_highres_heart_myocardium_raw": myocardium_raw,
        "totalsegmentator_heartchambers_highres_heart_myocardium_postprocessed": myocardium_no_blood_overlap,
        "geometric_residual_total_heart_minus_highres_blood": geometric_residual,
    }

    output_phase.mkdir(parents=True, exist_ok=True)
    output_paths = [output_phase / f"{label}.nii.gz" for label in products]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_phase}: refusing to overwrite {len(existing)} existing products"
        )
    written = {
        label: save_binary(output_phase / f"{label}.nii.gz", mask, total_image)
        for label, mask in products.items()
    }
    return {
        "phase_id": phase_dir.name,
        "source": {
            "total_heart": {
                "task": "total",
                "class": "heart",
                "source_filename": total_path.name,
                "sha256": sha256_file(total_path),
            },
            "heartchambers_highres": {
                "task": "heartchambers_highres",
                "classes": [
                    "heart_atrium_left",
                    "heart_ventricle_left",
                    "heart_atrium_right",
                    "heart_ventricle_right",
                    "heart_myocardium",
                ],
                "heart_myocardium_sha256": sha256_file(myocardium_path),
                "chamber_sources": chamber_sources,
            },
            "source_ct_phase": ct_source,
            "absolute_paths_omitted": True,
        },
        "geometry": {
            "shape": list(total_image.shape),
            "orientation": list(nib.aff2axcodes(total_image.affine)),
            "voxel_spacing_mm": list(nib.affines.voxel_sizes(total_image.affine)),
            "input_spatial_unit_in_header": input_spatial_unit,
            "output_spatial_unit_in_header": "mm",
            "geometry_mm_confirmed_override": bool(geometry_mm_confirmed),
            "input_qform_code": int(total_image.header["qform_code"]),
            "input_sform_code": int(total_image.header["sform_code"]),
        },
        "boolean_derivations": {
            "blood_pool_four_chambers_union": "LA OR LV OR RA OR RV; aorta and pulmonary artery excluded",
            "heart_myocardium_no_blood_overlap": "raw heart_myocardium AND NOT four-chamber blood",
            "geometric_residual": "total/heart AND NOT four-chamber blood; this is not a myocardium label",
        },
        "qc": {
            "blood_voxels_outside_total_heart": int(
                np.count_nonzero(blood_pool & ~total_heart)
            ),
            "raw_myocardium_voxels_overlapping_blood": int(
                np.count_nonzero(myocardium_raw & blood_pool)
            ),
            "raw_myocardium_inside_geometric_residual_fraction": float(
                np.count_nonzero(myocardium_raw & geometric_residual)
                / np.count_nonzero(myocardium_raw)
            ),
            "interpretation": "internal geometry QC only; not accuracy against anatomy",
        },
        "products": written,
        "status": "automatic_candidates_pending_independent_manual_validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument(
        "--segmentation-manifest",
        type=Path,
        required=True,
        help="Exact TotalSegmentator run manifest; never selected automatically",
    )
    parser.add_argument("--ct-phase-dir", type=Path, required=True)
    parser.add_argument(
        "--geometry-mm-confirmed",
        action="store_true",
        help="Confirm legacy unknown-unit NIfTI geometry only after checking DICOM provenance",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    segmentation_dir = args.segmentation_dir.resolve()
    ct_phase_dir = args.ct_phase_dir.resolve()
    output_dir = args.output_dir.resolve()
    segmentation_run = segmentation_provenance(args.segmentation_manifest)
    if not ct_phase_dir.is_dir():
        raise FileNotFoundError(f"CT phase directory not found: {ct_phase_dir}")
    phase_dirs = sorted(path for path in segmentation_dir.glob("phase_*") if path.is_dir())
    stop = len(phase_dirs) if args.phase_stop is None else args.phase_stop
    selected = phase_dirs[args.phase_start : stop]
    if not selected:
        raise FileNotFoundError("No selected phase_* directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = [
        build_phase(
            phase_dir,
            output_dir / phase_dir.name,
            overwrite=args.overwrite,
            source_ct_path=ct_phase_dir / f"{phase_dir.name}.nii.gz",
            geometry_mm_confirmed=args.geometry_mm_confirmed,
        )
        for phase_dir in selected
    ]
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "segmentation_provenance": segmentation_run,
        "definitions": {
            "total_heart": "raw TotalSegmentator total/heart mask",
            "blood_pool": "Boolean union of four raw heartchambers_highres chamber masks",
            "heart_myocardium_raw": "raw TotalSegmentator heartchambers_highres/heart_myocardium class",
            "heart_myocardium_no_blood_overlap": "raw heart_myocardium after only the declared blood-overlap removal",
            "geometric_residual": "total/heart minus four-chamber blood; never interpreted as myocardium",
        },
        "phases": phases,
        "status": "automatic_candidates_pending_independent_manual_validation",
        "clinical_validation": False,
    }
    (output_dir / "validation_candidates_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built explicit validation candidates for {len(phases)} phases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
