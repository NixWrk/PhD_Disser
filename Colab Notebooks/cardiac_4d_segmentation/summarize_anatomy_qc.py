#!/usr/bin/env python
"""Validate grouped TotalSegmentator anatomy products and write cohort QC."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, path = value.split("=", 1)
    return alias.strip().lower(), Path(path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=parse_root)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    product_rows = []
    subject_rows = []
    failures = []

    for alias, root in args.root:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        reference_shape = None
        reference_affine = None
        volumes = {}
        for product, details in manifest["products"].items():
            nifti_path = root / "products" / details["nifti"]
            stl_path = root / "products" / details["stl"]
            image = nib.load(str(nifti_path))
            data = np.asanyarray(image.dataobj)
            unique = np.unique(data).tolist()
            binary = set(unique).issubset({0, 1})
            nonempty = bool(np.any(data))
            if reference_shape is None:
                reference_shape = image.shape
                reference_affine = image.affine
            same_grid = image.shape == reference_shape and np.allclose(image.affine, reference_affine, atol=1e-4)
            stl_nonempty = stl_path.is_file() and stl_path.stat().st_size > 84
            watertight = bool(details["mesh"]["watertight"])
            ok = binary and nonempty and same_grid and stl_nonempty and watertight
            if not ok:
                failures.append(f"{alias}:{product}")
            volumes[product] = float(details["volume_ml"])
            product_rows.append({
                "subject": alias,
                "product": product,
                "volume_ml": round(volumes[product], 4),
                "connected_components": int(details["connected_components"]),
                "nifti_binary": binary,
                "nifti_nonempty": nonempty,
                "same_product_grid": same_grid,
                "stl_nonempty": stl_nonempty,
                "stl_watertight_at_export": watertight,
                "mesh_vertices": int(details["mesh"]["vertices"]),
                "qc_ok": ok,
            })
        body = volumes["body_in_scan_fov"]
        raw_count = sum(1 for folder in ("raw_total", "raw_body", "raw_tissue_types") for _ in (root / folder).glob("*.nii.gz"))
        subject_rows.append({
            "subject": alias,
            "raw_mask_count": raw_count,
            "product_count": len(manifest["products"]),
            "muscle_fraction_of_body_fov": volumes["skeletal_muscle_in_scan_fov"] / body,
            "fat_fraction_of_body_fov": volumes["fat_in_scan_fov"] / body,
            "status": "file_and_geometry_qc_passed_pending_manual_review" if not any(item.startswith(alias + ":") for item in failures) else "qc_failed",
        })

    for filename, rows in (("anatomy_products_qc.csv", product_rows), ("anatomy_subjects_qc.csv", subject_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "subjects": subject_rows,
        "product_rows": len(product_rows),
        "failures": failures,
        "status": "file_and_geometry_qc_passed_pending_manual_review" if not failures else "qc_failed",
        "interpretation_limit": "File and geometry QC does not establish anatomical accuracy.",
    }
    (output_dir / "anatomy_qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
