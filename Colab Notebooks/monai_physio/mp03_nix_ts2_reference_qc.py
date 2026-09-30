"""Audit independently segmented TSv2 phase masks against the saved Greedy/FEM run.

"Independent" means independent of the phase-00 deformation. These automatic
masks have no accepted manual anatomical ground truth. Patient images stay in
ignored runs/; the tracked JSON contains only pseudonymized metrics and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
PHASES = (0, 1, 3, 7, 10)
PAIRED = HERE / "runs" / "mp02_nix_paired_fem"
OUTPUT = HERE / "mp03_nix_ts2_reference_qc.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + np.count_nonzero(b))


def surface(a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...]) -> dict:
    union = a | b
    bounds = []
    for axis in range(3):
        used = np.flatnonzero(np.any(union, axis=tuple(x for x in range(3) if x != axis)))
        bounds.append(slice(max(0, int(used[0]) - 1), min(a.shape[axis], int(used[-1]) + 2)))
    crop = tuple(bounds)
    a = a[crop]
    b = b[crop]
    del union
    sa = a & ~ndi.binary_erosion(a)
    sb = b & ~ndi.binary_erosion(b)
    distance_to_b = ndi.distance_transform_edt(~sb, sampling=spacing)
    da = distance_to_b[sa]
    del distance_to_b
    distance_to_a = ndi.distance_transform_edt(~sa, sampling=spacing)
    db = distance_to_a[sb]
    del distance_to_a
    count = len(da) + len(db)
    return {"assd_mm": float((da.sum() + db.sum()) / count),
            "hd95_mm": float(np.percentile(np.concatenate((da, db)), 95)),
            "surface_dice_2mm": float((np.count_nonzero(da <= 2) +
                                       np.count_nonzero(db <= 2)) / count),
            "surface_tolerance_mm": 2.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--prior-electrical", type=Path, required=True)
    args = parser.parse_args()
    prior = read(args.prior_electrical / "plan.json")
    paired_plan = read(PAIRED / "plan.json")
    paired_aggregate = read(HERE / "mp02_nix_paired_fem.json")
    prep = read(PAIRED / "preparation.json")
    products = read(args.subject_root / "products_v2" / "qc_manifest.json")
    segmentation_path = args.subject_root / "automatic_masks_v2" / "segmentation_manifest_gpu0_p0_12.json"
    segmentation = read(segmentation_path)
    conversion = args.subject_root / "phases" / "conversion_manifest.json"
    if (segmentation["backend_version"] != "2.18.0" or
            segmentation["models"]["whole_heart"]["task"] != "total" or
            segmentation["models"]["whole_heart"]["class"] != "heart" or
            products["definitions"]["whole_heart"] !=
            "independent TotalSegmentator v2 total/heart mask; not a union of the other products" or
            products["status"] != "automatic_products_pending_manual_review" or
            prior["source_prepared_sha256"] != paired_plan["source_prepared_sha256"] or
            sha(args.prior_electrical / "plan.json") != prep["source_plan_sha256"] or
            sha(PAIRED / "plan.json") != paired_aggregate["new_plan_sha256"]):
        raise ValueError("TSv2 or paired FEM provenance differs from frozen calculation")
    prior_inputs = {str(Path(x["path"]).resolve()): x["sha256"] for x in prior["inputs"]}
    rows = []
    source = None
    source_affine = None
    spacing = None
    for index in PHASES:
        phase = f"phase_{index:02d}"
        mask_path = args.subject_root / "products_v2" / phase / "whole_heart.nii.gz"
        ct_path = args.subject_root / "phases" / f"{phase}.nii.gz"
        saved = prior_inputs.get(str(mask_path.resolve()))
        if saved is None or saved != sha(mask_path):
            raise ValueError(f"TSv2 {phase} is not the mask used by baseline FEM")
        image = nib.load(str(mask_path))
        ct = nib.load(str(ct_path))
        if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, rtol=0, atol=1e-5):
            raise ValueError(f"TSv2 mask and CT geometry differ: {phase}")
        data = np.asarray(image.dataobj)
        if not np.all((data == 0) | (data == 1)):
            raise ValueError(f"Nonbinary TSv2 mask: {phase}")
        mask = data.astype(bool, copy=False)
        raw_path = args.subject_root / "automatic_masks_v2" / phase / "total_heart" / "heart.nii.gz"
        raw_image = nib.load(str(raw_path))
        if raw_image.shape != image.shape or not np.allclose(raw_image.affine, image.affine, rtol=0, atol=1e-5):
            raise ValueError(f"Raw TSv2 grid differs: {phase}")
        raw = np.asarray(raw_image.dataobj, dtype=bool)
        added = int(np.count_nonzero(mask & ~raw))
        removed = int(np.count_nonzero(raw & ~mask))
        if added:
            raise ValueError(f"Derived TSv2 product adds voxels absent from raw model: {phase}")
        product = next(x for x in products["phases"] if x["phase_id"] == phase)
        expected = product["products"]["whole_heart"]["metrics"]
        volume = int(np.count_nonzero(mask)) * abs(np.linalg.det(image.affine[:3, :3])) / 1000
        labels, components = ndi.label(mask)
        sizes = np.bincount(labels.ravel())[1:]
        largest = float(sizes.max() / mask.sum()) if sizes.size else 0.0
        del labels
        if (int(np.count_nonzero(mask)) != expected["voxel_count"] or
                not np.isclose(volume, expected["volume_ml"], rtol=0, atol=1e-5) or
                components != expected["connected_components"] or
                not np.isclose(largest, expected["largest_component_fraction"], rtol=0, atol=1e-8)):
            raise ValueError(f"TSv2 product QC differs: {phase}")
        row = {"phase": phase, "ct_sha256": sha(ct_path), "ts2_mask_sha256": saved,
               "raw_ts2_mask_sha256": sha(raw_path),
               "derived_product_added_voxels": added,
               "derived_product_removed_voxels": removed,
               "ts2_volume_ml": float(volume),
               "ts2_connected_components": components,
               "ts2_largest_component_fraction": largest,
               "ts2_touches_superior_fov": bool(np.any(mask[:, :, -1])),
               "ts2_source_qc_status": product["qc_status"]}
        if index == 0:
            source = mask
            source_affine = image.affine
            spacing = tuple(float(v) for v in nib.affines.voxel_sizes(image.affine))
        else:
            if not np.allclose(image.affine, source_affine, rtol=0, atol=1e-5):
                raise ValueError(f"Phase grid differs: {phase}")
            field = prep["phase_details"][phase]
            transported_file = PAIRED / f"{phase}_transported.nii.gz"
            if sha(transported_file) != field["transported_mask_sha256"]:
                raise ValueError(f"Transported mask changed since FEM: {phase}")
            transported_image = nib.load(str(transported_file))
            if not np.allclose(transported_image.affine, image.affine, rtol=0, atol=1e-5):
                raise ValueError(f"Transported grid differs: {phase}")
            transported = np.asarray(transported_image.dataobj, dtype=bool)
            row.update({"identity_dice_to_ts2": float(dice(source, mask)),
                        "greedy_dice_to_ts2": float(dice(transported, mask)),
                        "greedy_minus_identity_dice": float(dice(transported, mask) - dice(source, mask)),
                        "transported_volume_ml": field["fraction"]["mask_volume_ml"],
                        "transported_minus_ts2_volume_ml": float(field["fraction"]["mask_volume_ml"] - volume),
                        "greedy_surface_vs_ts2": surface(transported, mask, spacing)})
            if abs(row["greedy_dice_to_ts2"] - field["transported_vs_observed_dice"]) > 1e-8:
                raise ValueError(f"Full-resolution Dice differs from saved FEM: {phase}")
        rows.append(row)
        if index == 0:
            print(phase, "TSv2 baseline ml", round(volume, 2), flush=True)
        else:
            print(phase, "TSv2 ml", round(volume, 2),
                  "Greedy Dice", round(row["greedy_dice_to_ts2"], 5), flush=True)
    result = {"schema": "mp03_nix_ts2_reference_qc_v1", "subject_alias": "nix",
              "reference_role": "registration-independent automatic phase mask; not manual anatomical truth",
              "status": "ts2_accepted_by_author_for_relative_computational_comparison_only",
              "phase_ids": [r["phase"] for r in rows],
              "totalsegmentator_version": segmentation["backend_version"],
              "ts2_segmentation_manifest_sha256": sha(segmentation_path),
              "ts2_qc_manifest_sha256": sha(args.subject_root / "products_v2" / "qc_manifest.json"),
              "conversion_manifest_sha256": sha(conversion),
              "prior_fem_plan_sha256": sha(args.prior_electrical / "plan.json"),
              "paired_fem_plan_sha256": sha(PAIRED / "plan.json"),
              "fem_reuse_reason": "same byte-identical TSv2 masks and frozen FEM plans; no changed model input",
              "rows": rows}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
