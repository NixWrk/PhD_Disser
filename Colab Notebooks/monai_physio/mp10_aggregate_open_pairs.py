"""Summarize deidentified MP-10 open-pair reports without source paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", nargs=3, action="append", metavar=("ALIAS", "EVALUATION", "SCENARIOS"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for alias, evaluation_name, scenarios_name in args.case:
        evaluation_path = Path(evaluation_name)
        scenarios_path = Path(scenarios_name)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
        if evaluation["pair_alias"] != alias or scenarios["subject_alias"] != alias:
            raise ValueError(f"Mismatched report alias: {alias}")
        if evaluation["scenario_report_sha256"] != sha256(scenarios_path):
            raise ValueError(f"Evaluation refers to a different scenario report: {alias}")
        if scenarios["stage_anchor"] != 0.7:
            raise ValueError(f"Unexpected reference stage: {alias}")
        stages = {float(item["stage"]): item for item in scenarios["stages"]}
        anchor = stages[0.7]["landmarks"]
        candidate = stages[0.0]["landmarks"]
        oracle = stages[0.9]["landmarks"]
        count = int(candidate["count"])
        if count != anchor["count"] or count != oracle["count"]:
            raise ValueError(f"Landmark counts differ across stages: {alias}")
        row = {
            "case_alias": alias,
            "landmark_count": count,
            "lung_mask_inhale_ml": evaluation["lung_mask_source"]["volume_ml"],
            "lung_mask_exhale_ml": evaluation["lung_mask_target"]["volume_ml"],
            "lung_mask_change_ml": evaluation["observed_lung_volume_change_ml"],
            "fit_nearest_vertex_mean_mm_stride8": evaluation["fit_nearest_vertex_mean_mm_stride8"],
            "lung_mesh_open_edges": evaluation["lung_mesh_open_edges_source"],
            "lung_model_vertex_displacement_mean_mm_stage_0": stages[0.0]["lung"]["vertex_displacement_mean_mm"],
            "landmark_model_displacement_mean_mm_stage_0": candidate["mean_predicted_displacement_mm"],
            "static_landmark_tre_mean_mm": anchor["static_tre_mean_mm"],
            "model_landmark_tre_mean_mm_stage_0": candidate["predicted_tre_mean_mm"],
            "tre_change_model_minus_static_mm": candidate["predicted_tre_mean_mm"] - anchor["static_tre_mean_mm"],
            "model_landmark_tre_mean_mm_stage_0_9_oracle": oracle["predicted_tre_mean_mm"],
            "heart_mask_centroid_shift_mm": evaluation["observed_heart_centroid_shift_norm_mm"],
            "heart_model_centroid_shift_mm_stage_0": evaluation["modeled_heart_centroid_shift_norm_mm"],
            "evaluation_report_sha256": sha256(evaluation_path),
            "scenario_report_sha256": sha256(scenarios_path),
        }
        rows.append(row)

    if len({item["case_alias"] for item in rows}) != len(rows):
        raise ValueError("Duplicate case alias")
    rows.sort(key=lambda item: item["case_alias"])
    total = sum(row["landmark_count"] for row in rows)
    weighted = lambda key: sum(row["landmark_count"] * row[key] for row in rows) / total
    report = {
        "schema": "mp10_open_pair_landmark_summary_v1",
        "status": "three_case_exploratory_transfer_check_not_general_validation",
        "assumed_stage_mapping": "inhale=0.7, exhale=0.0; no independent stage calibration",
        "case_count": len(rows),
        "landmark_count": total,
        "improved_case_count_stage_0": sum(row["tre_change_model_minus_static_mm"] < 0 for row in rows),
        "weighted_static_tre_mean_mm": weighted("static_landmark_tre_mean_mm"),
        "weighted_model_tre_mean_mm_stage_0": weighted("model_landmark_tre_mean_mm_stage_0"),
        "weighted_tre_change_model_minus_static_mm": weighted("tre_change_model_minus_static_mm"),
        "lung_volume_comparison": "source and target automatic masks only; model mesh volume unavailable because it has open edges",
        "heart_reference": "TotalSegmentator v2 automatic masks, not accepted manual contours",
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "case_count", "landmark_count", "improved_case_count_stage_0",
        "weighted_static_tre_mean_mm", "weighted_model_tre_mean_mm_stage_0",
        "weighted_tre_change_model_minus_static_mm",
    )}, indent=2))


if __name__ == "__main__":
    main()
