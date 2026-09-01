#!/usr/bin/env python
"""Build whole-heart, blood-pool and myocardium NIfTI/STL products with QC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from cardiac4d import clean_binary, export_mesh, mask_metrics, same_grid, utc_now


CHAMBER_MASKS = (
    "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
)


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def dice(left: np.ndarray, right: np.ndarray) -> float | None:
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    denominator = left_count + right_count
    if not denominator:
        return None
    return 2.0 * int(np.count_nonzero(left & right)) / denominator


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return image, np.asanyarray(image.dataobj) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-start", type=int, default=0)
    parser.add_argument("--phase-stop", type=int)
    parser.add_argument("--min-component-ml", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    segmentation_dir = Path(args.segmentation_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    phase_dirs = sorted(path for path in segmentation_dir.glob("phase_*") if path.is_dir())
    stop = len(phase_dirs) if args.phase_stop is None else args.phase_stop
    phase_dirs = phase_dirs[args.phase_start : stop]
    if not phase_dirs:
        raise FileNotFoundError("No selected phase_* segmentation directories")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    phases = []
    for phase_dir in phase_dirs:
        whole_image, whole_heart = load_binary(phase_dir / "total_heart" / "heart.nii.gz")
        highres_dir = phase_dir / "heartchambers_highres"
        myocardium_image, myocardium = load_binary(highres_dir / "heart_myocardium.nii.gz")
        chamber_images_and_masks = [load_binary(highres_dir / name) for name in CHAMBER_MASKS]
        all_images = [myocardium_image, *[item[0] for item in chamber_images_and_masks]]
        if not all(same_grid(whole_image, image) for image in all_images):
            raise RuntimeError(f"{phase_dir.name}: TotalSegmentator output grids differ")

        voxel_ml = abs(float(np.linalg.det(whole_image.affine[:3, :3]))) / 1000.0
        whole_heart = clean_binary(whole_heart, args.min_component_ml, voxel_ml)
        blood_pool = np.logical_or.reduce([item[1] for item in chamber_images_and_masks])
        blood_pool = clean_binary(blood_pool, args.min_component_ml, voxel_ml)
        myocardium = clean_binary(myocardium, args.min_component_ml, voxel_ml)

        overlap = blood_pool & myocardium
        myocardium = myocardium & ~blood_pool
        highres_union = blood_pool | myocardium
        intersection = highres_union & whole_heart
        whole_count = int(np.count_nonzero(whole_heart))
        blood_count = int(np.count_nonzero(blood_pool))
        myocardium_count = int(np.count_nonzero(myocardium))
        union_count = int(np.count_nonzero(highres_union))
        intersection_count = int(np.count_nonzero(intersection))

        output_phase = output_dir / phase_dir.name
        output_phase.mkdir(parents=True, exist_ok=True)
        products = {}
        for label, mask in (
            ("whole_heart", whole_heart),
            ("blood_pool", blood_pool),
            ("myocardium", myocardium),
        ):
            nifti_path = output_phase / f"{label}.nii.gz"
            mesh_path = output_phase / f"{label}.stl"
            if (nifti_path.exists() or mesh_path.exists()) and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite products in {output_phase}")
            product_image = nib.Nifti1Image(mask.astype(np.uint8), whole_image.affine, whole_image.header)
            product_image.set_data_dtype(np.uint8)
            nib.save(product_image, nifti_path)
            metrics = mask_metrics(mask, whole_image.affine)
            mesh = export_mesh(mask, whole_image.affine, mesh_path)
            products[label] = {"metrics": metrics, "mesh": mesh}
            rows.append(
                {
                    "phase_id": phase_dir.name,
                    "label": label,
                    "volume_ml": f"{metrics['volume_ml']:.6f}",
                    "connected_components": metrics["connected_components"],
                    "largest_component_fraction": f"{metrics['largest_component_fraction']:.6f}",
                    "mesh_watertight": mesh["watertight"],
                }
            )

        phases.append(
            {
                "phase_id": phase_dir.name,
                "overlap_voxels_removed_from_myocardium": int(np.count_nonzero(overlap)),
                "cross_model_qc": {
                    "dice_whole_heart_vs_blood_plus_myocardium": dice(whole_heart, highres_union),
                    "blood_plus_myocardium_inside_whole_heart_fraction": fraction(intersection_count, union_count),
                    "whole_heart_covered_by_blood_plus_myocardium_fraction": fraction(intersection_count, whole_count),
                    "blood_pool_inside_whole_heart_fraction": fraction(int(np.count_nonzero(blood_pool & whole_heart)), blood_count),
                    "myocardium_inside_whole_heart_fraction": fraction(int(np.count_nonzero(myocardium & whole_heart)), myocardium_count),
                    "interpretation": "model-agreement QC only; not validation against ground truth",
                },
                "products": products,
                "qc_status": "pending_manual_review",
            }
        )

    with (output_dir / "volumes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    temporal = {}
    for label in ("whole_heart", "blood_pool", "myocardium"):
        volumes = [phase["products"][label]["metrics"]["volume_ml"] for phase in phases]
        jumps = [
            abs(volumes[index] - volumes[index - 1]) / max(volumes[index - 1], 1e-12)
            for index in range(1, len(volumes))
        ]
        temporal[label] = {
            "minimum_volume_ml": min(volumes),
            "maximum_volume_ml": max(volumes),
            "range_ml": max(volumes) - min(volumes),
            "max_adjacent_relative_jump": max(jumps) if jumps else 0.0,
            "volumes_ml": volumes,
        }

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "definitions": {
            "whole_heart": "independent TotalSegmentator v2 total/heart mask; not a union of the other products",
            "blood_pool": "union of four heartchambers_highres chamber masks; aorta and pulmonary artery excluded",
            "myocardium": "heartchambers_highres myocardium with any chamber overlap removed",
            "volumes": "computed from binary NIfTI voxels, never from smoothed STL",
        },
        "postprocessing": {
            "minimum_component_volume_ml": args.min_component_ml,
            "overlap_rule": "blood_pool takes precedence over myocardium",
            "whole_heart_cross_model_clipping": False,
        },
        "phases": phases,
        "temporal_qc": temporal,
        "status": "automatic_products_pending_manual_review",
    }
    (output_dir / "qc_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built three NIfTI/STL products for {len(phases)} phases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
