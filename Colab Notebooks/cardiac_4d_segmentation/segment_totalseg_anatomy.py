#!/usr/bin/env python
"""Segment reference-phase thoracic anatomy with TotalSegmentator.

Outputs include individual raw masks plus grouped NIfTI and STL products for
bones, lungs, body within the acquired CT field of view, skeletal muscle, and
fat.  The TotalSegmentator license is read only from the user's local config.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from cardiac4d import clean_binary, export_mesh, mask_metrics, same_grid


LUNGS = (
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
)
BONE_EXACT = {
    "sacrum", "humerus_left", "humerus_right", "scapula_left", "scapula_right",
    "clavicula_left", "clavicula_right", "femur_left", "femur_right",
    "hip_left", "hip_right", "skull", "sternum",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_group(reference: nib.Nifti1Image, paths: list[Path], output_base: Path, min_component_ml: float = 0.0) -> dict:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No source masks for {output_base.name}")
    group = np.zeros(reference.shape, dtype=bool)
    used = []
    for path in existing:
        image = nib.load(str(path))
        if not same_grid(reference, image):
            raise RuntimeError(f"Grid mismatch: {path}")
        group |= np.asanyarray(image.dataobj) > 0
        used.append(path.stem.removesuffix(".nii"))
    if min_component_ml > 0:
        voxel_ml = abs(float(np.linalg.det(reference.affine[:3, :3]))) / 1000.0
        group = clean_binary(group, min_component_ml, voxel_ml)
    if not np.any(group):
        raise ValueError(f"Empty grouped mask: {output_base.name}")
    nifti_path = output_base.with_suffix(".nii.gz")
    stl_path = output_base.with_suffix(".stl")
    image = nib.Nifti1Image(group.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, nifti_path)
    result = {"source_classes": used, "minimum_component_ml": min_component_ml, "nifti": nifti_path.name, "stl": stl_path.name}
    result.update(mask_metrics(group, reference.affine))
    result["mesh"] = export_mesh(group, reference.affine, stl_path)
    return result


def all_bone_classes() -> tuple[str, ...]:
    from totalsegmentator.map_to_binary import class_map
    names = class_map["total"].values()
    return tuple(name for name in names if name.startswith(("vertebrae_", "rib_")) or name in BONE_EXACT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from totalsegmentator.python_api import totalsegmentator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_total = output_dir / "raw_total"
    raw_body = output_dir / "raw_body"
    raw_tissues = output_dir / "raw_tissue_types"
    products = output_dir / "products"
    for path in (raw_total, raw_body, raw_tissues, products):
        path.mkdir(parents=True, exist_ok=True)

    bones = all_bone_classes()
    total_expected = [raw_total / f"{name}.nii.gz" for name in (*LUNGS, *bones)]
    body_expected = [raw_body / f"{name}.nii.gz" for name in ("body_trunc", "body_extremities")]
    tissue_expected = [raw_tissues / f"{name}.nii.gz" for name in ("subcutaneous_fat", "torso_fat", "skeletal_muscle")]

    if args.overwrite or not all(path.is_file() for path in total_expected):
        totalsegmentator(input_path, raw_total, task="total", roi_subset=list((*LUNGS, *bones)), device="gpu", nr_thr_resamp=1, nr_thr_saving=1, quiet=args.quiet, no_derived_masks=True)
    if args.overwrite or not all(path.is_file() for path in body_expected):
        totalsegmentator(input_path, raw_body, task="body", device="gpu", nr_thr_resamp=1, nr_thr_saving=1, quiet=args.quiet, no_derived_masks=True)
    if args.overwrite or not all(path.is_file() for path in tissue_expected):
        totalsegmentator(input_path, raw_tissues, task="tissue_types", device="gpu", nr_thr_resamp=1, nr_thr_saving=1, quiet=args.quiet, no_derived_masks=True)

    missing = [str(path.relative_to(output_dir)) for path in (*total_expected, *body_expected, *tissue_expected) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing TotalSegmentator outputs: {missing}")

    reference = nib.load(str(input_path))
    grouped = {
        "bones_in_scan_fov": save_group(reference, total_expected[len(LUNGS):], products / "bones_in_scan_fov"),
        "lungs": save_group(reference, total_expected[:len(LUNGS)], products / "lungs", min_component_ml=1.0),
        "body_in_scan_fov": save_group(reference, body_expected, products / "body_in_scan_fov"),
        "skeletal_muscle_in_scan_fov": save_group(reference, [raw_tissues / "skeletal_muscle.nii.gz"], products / "skeletal_muscle_in_scan_fov"),
        "fat_in_scan_fov": save_group(reference, [raw_tissues / "subcutaneous_fat.nii.gz", raw_tissues / "torso_fat.nii.gz"], products / "fat_in_scan_fov"),
    }
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "subject_alias": args.subject_id,
        "reference_phase": input_path.name,
        "backend": "TotalSegmentator",
        "backend_version": importlib.metadata.version("TotalSegmentator"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "tasks": {"total": [*LUNGS, *bones], "body": ["body_trunc", "body_extremities"], "tissue_types": ["subcutaneous_fat", "torso_fat", "skeletal_muscle"]},
        "license": {"source": "local TotalSegmentator user configuration", "value_recorded": False},
        "products": grouped,
        "scope_note": "Body, bone, muscle, and fat masks cover only anatomy present inside the acquired CT field of view.",
        "temporal_note": "Static surrounding anatomy was segmented on one reference cardiac phase; cardiac structures use all phases in the separate 4D products.",
        "status": "automatic_segmentation_pending_manual_review",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "TotalSegmentator automatic research segmentation.\n"
        "Reference phase: phase_00. Products: NIfTI masks and STL meshes.\n"
        "All outputs are pending manual expert review and are not clinical diagnoses.\n"
        "The body mask means only the body portion present in the CT scan field of view.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: round(value["volume_ml"], 2) for key, value in grouped.items()}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
