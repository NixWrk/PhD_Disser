"""Locate nonpositive Jacobian voxels in a saved local registration field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-json", type=Path, required=True)
    parser.add_argument("--field-npz", type=Path, required=True)
    parser.add_argument("--products-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    runs = (Path(__file__).resolve().parent / "runs").resolve()
    if not args.output_json.resolve().is_relative_to(runs):
        parser.error("output-json must be within ignored runs/")
    report = json.loads(args.qc_json.read_text(encoding="utf-8"))
    with args.field_npz.open("rb") as source:
        field_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if field_hash != report["field_qc"]["local_field_sha256"]:
        raise ValueError("Field hash differs from registration QC")
    grid = report["grid_qc"]
    if grid["stride_ijk"] != [1, 1, 1]:
        parser.error("requires native-grid field")
    with np.load(args.field_npz) as field:
        vector = field["displacement_zyx_lps_mm"]
        image = sitk.GetImageFromArray(vector, isVector=True)
        image.SetOrigin(tuple(float(v) for v in field["origin_lps_mm"]))
        image.SetSpacing(tuple(float(v) for v in field["spacing_mm"]))
        image.SetDirection(tuple(float(v) for v in field["direction_lps"].ravel()))
    jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(image))
    bad = np.argwhere(jac <= 0)
    start = np.asarray(grid["crop_start_ijk"])
    stop = np.asarray(grid["crop_stop_ijk_exclusive"])
    mask_path = args.products_dir / report["fixed_target"] / "whole_heart.nii.gz"
    mask = np.asarray(nib.load(str(mask_path)).dataobj[
        tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    ]) > 0
    if tuple(jac.shape) != tuple(mask.shape[::-1]):
        raise ValueError("field and mask dimensions differ")
    heart_distance = ndi.distance_transform_edt(~mask, sampling=grid["evaluation_spacing_mm"])
    local_ijk = bad[:, ::-1]
    distances = heart_distance[tuple(local_ijk.T)] if len(bad) else np.array([])
    result = {
        "target": report["fixed_target"],
        "nonpositive_count": int(len(bad)),
        "nonpositive_fraction": float(len(bad) / jac.size),
        "minimum_jacobian": float(jac.min()),
        "inside_target_auto_heart_count": int(np.count_nonzero(mask[tuple(local_ijk.T)])) if len(bad) else 0,
        "nearest_auto_heart_distance_mm": float(distances.min()) if len(bad) else None,
        "farthest_auto_heart_distance_mm": float(distances.max()) if len(bad) else None,
        "nonpositive_global_ijk": (local_ijk + start).tolist(),
        "nonpositive_jacobians": jac[tuple(bad.T)].tolist() if len(bad) else [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
