"""Publish only non-identifying aggregate metrics from the local TCIA phase run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "mp10_tcia_phase_validation_summary_v1",
        "date": "2026-10-01",
        "status": "one_public_case_automatic_mask_exploratory_validation",
        "dataset": "TCIA 4D-Lung, MONAI Physio 2026.07.1 tutorial subset",
        "case": "100_HM10395",
        "phase_rule": "gPPP -> network stage PPP/100; no stage chosen from the target outcome",
        "model_weights": "published 2026.07.1 checkpoint, no local retraining",
        "motion_network_holdout": "Tutorial 9 excludes case 100; checkpoint manifest not independently verified",
        "shape_model_holdout": "Tutorial 6 does not exclude case 100 g070; this is not full-pipeline leave-one-case-out",
        "segmenter": "TotalSegmentator v2 on each CT phase separately",
        "units": {"distance": "mm", "volume": "ml"},
        "experiments": {},
    }
    expected_phases = {f"g{i:03d}" for i in range(0, 100, 10)}
    for source in ("g070", "g000"):
        eval_path = args.run_dir / f"eval_{source}" / "evaluation.json"
        fit_path = args.run_dir / f"fit_{source}" / "fit_report.json"
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        rows = evaluation["metrics"]
        if {row["phase"] for row in rows} != expected_phases:
            raise ValueError(f"Incomplete respiratory series for {source}")
        if not evaluation["actual_source_ct_surface_warped_by_network_field"]:
            raise ValueError("The fitted mesh alone is not a CT-preserving prediction")
        if evaluation["source_phase"] != source:
            raise ValueError("Source phase mismatch")
        targets = [row for row in rows if row["phase"] != source]
        by_phase = {}
        for row in rows:
            static = row["lung_source_mask_copy_vs_observed"]["symmetric_mean_mm"]
            warped = row["lung_warped_ct_surface_vs_observed"]["symmetric_mean_mm"]
            by_phase[row["phase"]] = {
                "observed_lung_volume_ml": row["lung_volume_ml"],
                "predicted_lung_volume_ml": row["lung_predicted_volume_ml"],
                "predicted_lung_volume_error_ml": row["lung_predicted_volume_error_ml"],
                "lung_surface_static_mm": static,
                "lung_surface_ct_warp_mm": warped,
                "lung_surface_improvement_mm": static - warped,
                "lung_surface_fitted_mesh_mm": row["lung_neural_prediction_vs_observed"]["symmetric_mean_mm"],
                "lung_centroid_static_error_mm": row["lung_static_centroid_error_mm"],
                "lung_centroid_predicted_error_mm": row["lung_predicted_centroid_error_mm"],
                "heart_centroid_static_error_mm": row["heart_static_centroid_error_mm"],
                "heart_centroid_predicted_error_mm": row["heart_predicted_centroid_error_mm"],
                "observed_heart_volume_ml": row["heart_volume_ml"],
                "predicted_heart_volume_ml": row["heart_predicted_volume_ml"],
                "heart_surface_static_mm": row["heart_surface_static_vs_observed"]["symmetric_mean_mm"],
                "heart_surface_predicted_mm": row["heart_surface_predicted_vs_observed"]["symmetric_mean_mm"],
                "lung_field_coverage_fraction": row["lung_field_coverage_fraction"],
            }
        result["experiments"][source] = {
            "role": "published_reference_input" if source == "g070" else "inspiration_input_outside_published_fit_protocol",
            "fit_surface_mean_mm": fit["nearest_vertex_mean_mm_stride8"],
            "source_lung_volume_ml": fit["source_lung_volume_ml"],
            "evaluation_sha256": sha256(eval_path),
            "model_checkpoint_sha256": evaluation["model_checkpoint_sha256"],
            "phase_results": by_phase,
            "target_phase_count": len(targets),
            "lung_surface_improved_phase_count": sum(
                by_phase[row["phase"]]["lung_surface_improvement_mm"] > 0 for row in targets
            ),
            "mean_lung_surface_static_mm": float(np.mean([
                row["lung_source_mask_copy_vs_observed"]["symmetric_mean_mm"] for row in targets
            ])),
            "mean_lung_surface_ct_warp_mm": float(np.mean([
                row["lung_warped_ct_surface_vs_observed"]["symmetric_mean_mm"] for row in targets
            ])),
            "mean_heart_centroid_static_error_mm": float(np.mean([
                row["heart_static_centroid_error_mm"] for row in targets
            ])),
            "mean_heart_centroid_predicted_error_mm": float(np.mean([
                row["heart_predicted_centroid_error_mm"] for row in targets
            ])),
            "heart_surface_improved_phase_count": sum(
                row["heart_surface_predicted_vs_observed"]["symmetric_mean_mm"]
                < row["heart_surface_static_vs_observed"]["symmetric_mean_mm"]
                for row in targets
            ),
            "mean_heart_surface_static_mm": float(np.mean([
                row["heart_surface_static_vs_observed"]["symmetric_mean_mm"] for row in targets
            ])),
            "mean_heart_surface_predicted_mm": float(np.mean([
                row["heart_surface_predicted_vs_observed"]["symmetric_mean_mm"] for row in targets
            ])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
