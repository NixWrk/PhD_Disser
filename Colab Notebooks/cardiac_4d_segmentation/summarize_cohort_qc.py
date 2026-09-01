#!/usr/bin/env python
"""Create a compact cohort-level QC report from subject manifests."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


CHAMBERS = (
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
)


def volume_ml(path: Path) -> float:
    image = nib.load(str(path))
    voxels = int(np.count_nonzero(np.asanyarray(image.dataobj)))
    voxel_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))
    return voxels * voxel_mm3 / 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-root", required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--myocardium-range-fraction-limit", type=float, default=0.10)
    parser.add_argument("--adjacent-jump-limit", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(args.derived_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = []
    chamber_rows = []
    for subject_id in args.subjects:
        subject_root = root / subject_id
        conversion = json.loads(
            (subject_root / "phases" / "conversion_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        qc = json.loads(
            (subject_root / "products" / "qc_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        myocardium = qc["temporal_qc"]["myocardium"]
        blood = qc["temporal_qc"]["blood_pool"]
        myocardium_values = myocardium["volumes_ml"]
        myocardium_range_fraction = (
            myocardium["range_ml"] / statistics.median(myocardium_values)
        )
        all_watertight = all(
            phase["products"][label]["mesh"]["watertight"]
            for phase in qc["phases"]
            for label in ("blood_pool", "myocardium")
        )
        all_nonempty = all(
            phase["products"][label]["metrics"]["voxel_count"] > 0
            for phase in qc["phases"]
            for label in ("blood_pool", "myocardium")
        )

        for phase in qc["phases"]:
            phase_id = phase["phase_id"]
            row = {"subject_id": subject_id, "phase_id": phase_id}
            for chamber in CHAMBERS:
                row[f"{chamber}_ml"] = volume_ml(
                    subject_root / "automatic_masks" / phase_id / f"{chamber}.nii.gz"
                )
            chamber_rows.append(row)

        flags = []
        if myocardium_range_fraction > args.myocardium_range_fraction_limit:
            flags.append("myocardium_volume_range_exceeds_candidate_limit")
        if myocardium["max_adjacent_relative_jump"] > args.adjacent_jump_limit:
            flags.append("myocardium_adjacent_jump_exceeds_candidate_limit")
        if blood["max_adjacent_relative_jump"] > args.adjacent_jump_limit:
            flags.append("blood_pool_adjacent_jump_exceeds_candidate_limit")
        if not all_watertight:
            flags.append("non_watertight_mesh")
        if not all_nonempty:
            flags.append("empty_product")
        subjects.append(
            {
                "subject_id": subject_id,
                "source_series_folder_name": conversion["source_series_folder_name"],
                "phase_count": conversion["phase_count"],
                "phase_keyword": conversion["phase_keyword"],
                "phase_values_raw": [
                    phase["dicom_phase_value"] for phase in conversion["phases"]
                ],
                "shape": conversion["phases"][0]["shape"],
                "voxel_spacing_mm": conversion["phases"][0]["voxel_spacing_mm"],
                "blood_pool": blood,
                "myocardium": {
                    **myocardium,
                    "range_fraction_of_median": myocardium_range_fraction,
                },
                "all_meshes_watertight": all_watertight,
                "all_products_nonempty": all_nonempty,
                "automatic_qc_flags": flags,
                "status": "automatic_products_pending_manual_review",
            }
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_thresholds_not_validated": {
            "myocardium_range_fraction_limit": args.myocardium_range_fraction_limit,
            "adjacent_relative_jump_limit": args.adjacent_jump_limit,
        },
        "subjects": subjects,
        "interpretation": (
            "Flags are engineering screening rules, not validated biological or clinical "
            "acceptance limits. All automatic masks require phase-by-phase expert review."
        ),
    }
    (output_dir / "cohort_qc_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = ["subject_id", "phase_id", *[f"{name}_ml" for name in CHAMBERS]]
    with (output_dir / "chamber_volumes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chamber_rows)
    print(output_dir / "cohort_qc_summary.json")
    print(output_dir / "chamber_volumes.csv")


if __name__ == "__main__":
    main()

