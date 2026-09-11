#!/usr/bin/env python
"""Build a compact QC passport for one static TotalSegmentator CT run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image) -> bool:
    return reference.shape == other.shape and np.allclose(reference.affine, other.affine, atol=1e-4)


def binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return image, np.asanyarray(image.dataobj) > 0


def manual_comparison(manual: dict, automatic_volumes: dict[str, float]) -> dict:
    by_label = {item["label"]: item for item in manual.get("items", [])}
    pairs = {
        "bones": ("manual_bones", "bones_in_scan_fov"),
        "lungs": ("manual_lungs", "lungs"),
        "body_vs_manual_soft_tissue": ("manual_soft_tissue", "body_in_scan_fov"),
        "blood_pool": ("manual_blood_pool", "blood_pool"),
    }
    result = {}
    for label, (manual_label, auto_label) in pairs.items():
        item = by_label.get(manual_label)
        if item is None:
            continue
        record = {
            "manual_label_from_filename": manual_label,
            "automatic_product": auto_label,
            "manual_stl_watertight": bool(item.get("watertight")),
            "manual_stl_components": item.get("components"),
            "automatic_volume_ml_from_nifti": automatic_volumes[auto_label],
        }
        if item.get("watertight"):
            manual_ml = abs(float(item["signed_volume_mesh_units_cubed"])) / 1000.0
            auto_ml = automatic_volumes[auto_label]
            record.update(
                {
                    "manual_mesh_volume_ml_if_export_units_are_mm": manual_ml,
                    "automatic_minus_manual_ml": auto_ml - manual_ml,
                    "automatic_minus_manual_percent": 100.0 * (auto_ml - manual_ml) / manual_ml,
                }
            )
        else:
            record["volume_comparison"] = "not_computed_because_manual_stl_is_not_watertight"
        result[label] = record
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing input, anatomy and cardiac_products")
    parser.add_argument("--manual-stl-inventory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    conversion = load_json(run_dir / "input" / "conversion_manifest.json")
    anatomy = load_json(run_dir / "anatomy" / "manifest.json")
    cardiac = load_json(run_dir / "cardiac_products" / "qc_manifest.json")
    if conversion.get("phase_count") != 1 or len(cardiac.get("phases", [])) != 1:
        raise ValueError("Static QC expects exactly one input volume and one cardiac product set")

    ct_path = run_dir / "input" / conversion["phases"][0]["nifti"]
    reference = nib.load(str(ct_path))
    product_paths = {
        name: run_dir / "anatomy" / "products" / details["nifti"]
        for name, details in anatomy["products"].items()
    }
    cardiac_phase = cardiac["phases"][0]
    product_paths.update(
        {
            name: run_dir / "cardiac_products" / cardiac_phase["phase_id"] / f"{name}.nii.gz"
            for name in cardiac_phase["products"]
        }
    )

    automatic_volumes = {
        name: float(details["volume_ml"])
        for name, details in anatomy["products"].items()
    }
    automatic_volumes.update(
        {
            name: float(details["metrics"]["volume_ml"])
            for name, details in cardiac_phase["products"].items()
        }
    )

    body_image, body = binary(product_paths["body_in_scan_fov"])
    if not same_grid(reference, body_image):
        raise RuntimeError("Body mask grid differs from the input CT")
    grid_checks = {}
    containment = {}
    for name, path in product_paths.items():
        image, mask = binary(path)
        grid_checks[name] = {
            "same_shape_and_affine_as_input": same_grid(reference, image),
            "nonempty": bool(np.any(mask)),
        }
        voxels = int(mask.sum())
        outside = int(np.logical_and(mask, ~body).sum())
        containment[name] = {
            "voxels_outside_body_mask": outside,
            "fraction_outside_body_mask": outside / voxels if voxels else None,
        }
    if not all(v["same_shape_and_affine_as_input"] and v["nonempty"] for v in grid_checks.values()):
        raise RuntimeError("At least one product failed grid or non-empty QC")

    manual = None
    comparison = None
    if args.manual_stl_inventory:
        manual = load_json(Path(args.manual_stl_inventory).resolve())
        comparison = manual_comparison(manual, automatic_volumes)

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "automatic_segmentation_pending_manual_review",
        "study_type": "single_static_contrast_ct_not_4d",
        "source": {
            "container_name": conversion.get("source_container_name"),
            "series_number": conversion.get("source_series_number"),
            "shape": conversion["phases"][0]["shape"],
            "voxel_spacing_mm": conversion["phases"][0]["voxel_spacing_mm"],
            "conversion_status": conversion.get("status"),
        },
        "software": {
            "backend": anatomy.get("backend"),
            "backend_version": anatomy.get("backend_version"),
            "torch": anatomy.get("torch"),
            "gpu": anatomy.get("gpu"),
            "license_value_recorded": False,
        },
        "automatic_volumes_ml_from_nifti": automatic_volumes,
        "grid_and_nonempty_checks": grid_checks,
        "containment_in_automatic_body_mask": containment,
        "heart_cross_model_qc": cardiac_phase["cross_model_qc"],
        "manual_stl_geometric_comparison": comparison,
        "limitations": [
            "All masks are automatic candidates and require slice-by-slice expert review before scientific inference.",
            "This study has one static volume; R-R dynamics, ED/ES, stroke volume and ejection fraction cannot be estimated from it.",
            "The myocardium class mainly represents left-ventricular myocardium and septum, not complete walls of every chamber.",
            "Body, bone, muscle and fat volumes describe only anatomy inside this CT field of view, not whole-body composition.",
            "Manual STL labels are taken from filenames; STL stores neither units nor anatomical coordinate-system metadata.",
            "Manual and automatic surfaces were not rasterized into a common verified grid, so Dice and surface distances are not reported.",
            "A non-watertight manual STL must not be used as a quantitative volume reference.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.name, "status": report["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
