"""Summarize lung volume and breathing contrast errors from saved aggregates."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    paths = [ROOT / "mp10_tcia_phase_validation_2026-10-01.json",
             ROOT / "mp10_dirlab_4dct_summary_2026-10-01.json"]
    tcia, dirlab = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    experiment = tcia["experiments"]["g000"]
    exhale = experiment["phase_results"]["g050"]
    source_volume = experiment["source_lung_volume_ml"]
    observed_reduction = source_volume - exhale["observed_lung_volume_ml"]
    predicted_reduction = source_volume - exhale["predicted_lung_volume_ml"]
    rows = []
    for case in dirlab["rows"]:
        if "mask_evaluation" not in case:
            continue
        phases = {row["phase"]: row["organs"]["lung"] for row in case["mask_evaluation"]["rows"]}
        observed = phases["T00"]["observed_volume_ml"] - phases["T50"]["observed_volume_ml"]
        predicted = phases["T00"]["predicted_volume_ml"] - phases["T50"]["predicted_volume_ml"]
        if observed <= 0:
            raise ValueError("End-inhale volume must exceed end-exhale in this summary")
        rows.append({"case": case["case"], "source_phase": "T70",
                     "observed_T00_to_T50_volume_reduction_ml": observed,
                     "predicted_T00_to_T50_volume_reduction_ml": predicted,
                     "volume_reduction_error_ml": abs(predicted - observed),
                     "volume_reduction_error_percent_observed_reduction": 100 * abs(predicted - observed) / observed,
                     "source_mask_evaluation_sha256": case["mask_evaluation"]["evaluation_sha256"]})
    result = {
        "schema": "mp10_lung_volume_error_summary_v1", "date": "2026-10-07",
        "units": "ml, percent", "reference": "automatic per-phase TotalSegmentator v2 masks; no accepted independent manual contours",
        "definitions": {
            "target_volume_error_percent": "100 * abs(predicted target volume - observed target volume) / observed target volume",
            "volume_reduction_error_percent": "100 * abs(predicted inhale-to-exhale reduction - observed reduction) / abs(observed reduction)",
            "tcia_prediction_baseline": "observed source CT volume; final prediction is a warped watertight CT-derived surface",
            "dirlab_prediction_baseline": "difference between two predicted phase surfaces from the same T70 source; common source offset cancels",
        },
        "tcia_one_case_inhale_input": {
            "source_phase": "g000", "target_phase": "g050",
            "source_lung_volume_ml": source_volume,
            "observed_target_lung_volume_ml": exhale["observed_lung_volume_ml"],
            "predicted_target_lung_volume_ml": exhale["predicted_lung_volume_ml"],
            "target_volume_error_ml": abs(exhale["predicted_lung_volume_error_ml"]),
            "target_volume_error_percent_observed_volume": 100 * abs(exhale["predicted_lung_volume_error_ml"]) / exhale["observed_lung_volume_ml"],
            "observed_volume_reduction_ml": observed_reduction,
            "predicted_volume_reduction_ml": predicted_reduction,
            "volume_reduction_error_percent_observed_reduction": 100 * abs(predicted_reduction - observed_reduction) / observed_reduction,
        },
        "dirlab_volume_case_count": len(rows), "dirlab_landmark_case_count": dirlab["case_count"],
        "dirlab_volume_rows": rows,
        "source_aggregates_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "scope": "lung volumes only; not heart chamber volume, stroke volume or ejection fraction",
    }
    output = ROOT / "mp10_volume_error_summary_2026-10-07.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"tcia_target_volume_error_percent": result["tcia_one_case_inhale_input"]["target_volume_error_percent_observed_volume"],
                      "tcia_reduction_error_percent": result["tcia_one_case_inhale_input"]["volume_reduction_error_percent_observed_reduction"],
                      "dirlab_reduction_error_percent": [r["volume_reduction_error_percent_observed_reduction"] for r in rows]}))


if __name__ == "__main__":
    main()
