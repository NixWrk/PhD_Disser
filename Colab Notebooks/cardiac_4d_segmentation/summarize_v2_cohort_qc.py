#!/usr/bin/env python
"""Create cohort QC for the three-product TotalSegmentator v2 pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("whole_heart", "blood_pool", "myocardium")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adjacent-jump-limit", type=float, default=0.10)
    parser.add_argument("--range-fraction-limit", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(args.derived_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = []
    rows = []
    for subject_id in args.subjects:
        subject_root = root / subject_id
        conversion = json.loads(
            (subject_root / "phases" / "conversion_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        qc = json.loads(
            (subject_root / "products_v2" / "qc_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        phase_count = conversion["phase_count"]
        if len(qc["phases"]) != phase_count:
            raise RuntimeError(
                f"{subject_id}: conversion has {phase_count} phases but products have {len(qc['phases'])}"
            )

        label_qc = {}
        flags = []
        for label in LABELS:
            temporal = qc["temporal_qc"][label]
            range_fraction = temporal["range_ml"] / statistics.median(
                temporal["volumes_ml"]
            )
            label_qc[label] = {
                **temporal,
                "range_fraction_of_median": range_fraction,
            }
            if range_fraction > args.range_fraction_limit:
                flags.append(f"{label}_range_fraction_exceeds_candidate_limit")
            if temporal["max_adjacent_relative_jump"] > args.adjacent_jump_limit:
                flags.append(f"{label}_adjacent_jump_exceeds_candidate_limit")

        all_watertight = all(
            phase["products"][label]["mesh"]["watertight"]
            for phase in qc["phases"]
            for label in LABELS
        )
        all_nonempty = all(
            phase["products"][label]["metrics"]["voxel_count"] > 0
            for phase in qc["phases"]
            for label in LABELS
        )
        if not all_watertight:
            flags.append("non_watertight_mesh")
        if not all_nonempty:
            flags.append("empty_product")

        cross_dice = [
            phase["cross_model_qc"]["dice_whole_heart_vs_blood_plus_myocardium"]
            for phase in qc["phases"]
        ]
        for phase in qc["phases"]:
            row = {"subject_id": subject_id, "phase_id": phase["phase_id"]}
            for label in LABELS:
                product = phase["products"][label]
                row[f"{label}_volume_ml"] = f"{product['metrics']['volume_ml']:.6f}"
                row[f"{label}_components"] = product["metrics"]["connected_components"]
            row["cross_model_dice"] = (
                f"{phase['cross_model_qc']['dice_whole_heart_vs_blood_plus_myocardium']:.6f}"
            )
            rows.append(row)

        subjects.append(
            {
                "subject_id": subject_id,
                "source_series_folder_name": conversion["source_series_folder_name"],
                "phase_count": phase_count,
                "phase_keyword": conversion["phase_keyword"],
                "phase_values_raw": [
                    phase["dicom_phase_value"] for phase in conversion["phases"]
                ],
                "shape": conversion["phases"][0]["shape"],
                "voxel_spacing_mm": conversion["phases"][0]["voxel_spacing_mm"],
                "labels": label_qc,
                "cross_model_dice_minimum": min(cross_dice),
                "cross_model_dice_maximum": max(cross_dice),
                "all_meshes_watertight": all_watertight,
                "all_products_nonempty": all_nonempty,
                "stl_count": phase_count * len(LABELS),
                "automatic_qc_flags": flags,
                "status": "automatic_products_pending_manual_review",
            }
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "whole_heart": "independent TotalSegmentator v2 total/heart mask",
            "blood_pool": "union of four heartchambers_highres chambers; great vessels excluded",
            "myocardium": "heartchambers_highres myocardium with chamber overlap removed",
        },
        "candidate_thresholds_not_validated": {
            "range_fraction_of_median_limit": args.range_fraction_limit,
            "adjacent_relative_jump_limit": args.adjacent_jump_limit,
        },
        "subjects": subjects,
        "total_phase_count": sum(item["phase_count"] for item in subjects),
        "total_stl_count": sum(item["stl_count"] for item in subjects),
        "interpretation": (
            "Engineering screening only. Cross-model agreement is not ground-truth validation; "
            "all automatic masks require phase-by-phase expert review."
        ),
    }
    (output_dir / "cohort_qc_summary_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "phase_volumes_v2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_dir / "cohort_qc_summary_v2.json")
    print(output_dir / "phase_volumes_v2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
