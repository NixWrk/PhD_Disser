"""Compare T70-only respiratory predictions with withheld DIR-Lab phase masks.

Target masks are independently computed by frozen TotalSegmentator v2. They
serve as an automatic, not manually accepted, contour reference. Predicted
surfaces start from the segmented T70 CT, then use the saved neural field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import itk
import numpy as np
import pyvista as pv

from mp10_evaluate_tcia_phases import LOBES, mask_surface, sample_field, surface_error
from mp10_respiratory_phases import phase_stage, validate_scenario


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=["T50", "T00"])
    args = parser.parse_args()
    case_dir = args.case_dir
    output = case_dir / "mask_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    source = {}
    source_meta = {}
    for organ, names in (("lung", LOBES), ("heart", ("heart",))):
        _, source_meta[organ] = mask_surface(
            case_dir / "seg" / "T70", names, output / f"source_{organ}_T70.vtp"
        )
        source[organ] = pv.read(str(output / f"source_{organ}_T70.vtp"))
    scenario = case_dir / "scenario_T70"
    scenario_report = json.loads((scenario / "scenario_report.json").read_text(encoding="utf-8"))
    contract = validate_scenario(scenario_report, "dirlab")
    if scenario_report["source_phase"] != "T70":
        raise ValueError("Mask evaluation requires T70 source anatomy")
    reference_field = itk.imread(str(scenario / "field_stage_0.70.mha"))
    reference_shift = {}
    reference_coverage = {}
    for organ in source:
        reference_shift[organ], reference_coverage[organ] = sample_field(
            source[organ].points, reference_field
        )
    rows = []
    for phase in args.phases:
        stage = phase_stage(phase, "dirlab")
        target_field = itk.imread(str(scenario / f"field_stage_{stage:.2f}.mha"))
        row = {"phase": phase, "model_stage": stage, "organs": {}}
        for organ, names in (("lung", LOBES), ("heart", ("heart",))):
            target_points, target_meta = mask_surface(
                case_dir / "seg" / phase, names,
                output / f"observed_{organ}_{phase}.vtp",
            )
            shift, coverage = sample_field(source[organ].points, target_field)
            if min(coverage, reference_coverage[organ]) < 0.99:
                raise ValueError(f"{organ} field coverage below 99% at {phase}")
            predicted = source[organ].copy(deep=True)
            predicted.points = (
                source[organ].points + shift - reference_shift[organ]
            ).astype(np.float32)
            predicted_path = output / f"predicted_{organ}_{phase}.vtp"
            predicted.save(str(predicted_path))
            if target_meta["touches_ct_boundary"]:
                raise ValueError(f"{organ} touches acquired CT boundary at {phase}")
            row["organs"][organ] = {
                "observed_volume_ml": target_meta["volume_ml"],
                "observed_volume_change_from_T70_ml": (
                    target_meta["volume_ml"] - source_meta[organ]["volume_ml"]
                ),
                "predicted_volume_ml": (
                    float(predicted.volume / 1000)
                    if predicted.n_open_edges == 0 else None
                ),
                "predicted_surface_open_edges": int(predicted.n_open_edges),
                "field_coverage_fraction": coverage,
                "static_surface_error_mm": surface_error(source[organ].points, target_points),
                "predicted_surface_error_mm": surface_error(predicted.points, target_points),
                "target_mask_sha256": target_meta["mask_sha256"],
                "predicted_surface_sha256": sha256(predicted_path),
            }
            item = row["organs"][organ]
            if item["predicted_volume_ml"] is not None:
                item["predicted_volume_change_from_T70_ml"] = (
                    item["predicted_volume_ml"] - source_meta[organ]["volume_ml"]
                )
                item["predicted_volume_error_ml"] = (
                    item["predicted_volume_ml"] - item["observed_volume_ml"]
                )
        rows.append(row)
    report = {
        "schema": "mp10_dirlab_4dct_mask_evaluation_v1",
        "case_alias": json.loads((case_dir / "conversion_manifest.json").read_text(encoding="utf-8"))["case_alias"],
        "input_phase": "T70 only",
        "target_phases": args.phases,
        "phase_contract": contract,
        "reference": "TotalSegmentator v2 automatic phase masks, not accepted manual contours",
        "source_mask": source_meta,
        "source_field_coverage_fraction": reference_coverage,
        "conversion_manifest_sha256": sha256(case_dir / "conversion_manifest.json"),
        "scenario_report_sha256": sha256(scenario / "scenario_report.json"),
        "rows": rows,
        "limitations": [
            "The target contour reference is automatic and has correlated segmenter errors across phases.",
            "Heart motion is extrapolated from the lung field; the network was not trained on heart movement.",
            "Txx maps to stage xx/100 by protocol, without temporal alignment from the CT images.",
        ],
    }
    path = case_dir / "mask_evaluation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps({
            "phase": row["phase"],
            "lung_static_mm": row["organs"]["lung"]["static_surface_error_mm"]["symmetric_mean_mm"],
            "lung_predicted_mm": row["organs"]["lung"]["predicted_surface_error_mm"]["symmetric_mean_mm"],
            "lung_observed_delta_ml": row["organs"]["lung"]["observed_volume_change_from_T70_ml"],
            "lung_predicted_delta_ml": row["organs"]["lung"].get("predicted_volume_change_from_T70_ml"),
            "heart_static_mm": row["organs"]["heart"]["static_surface_error_mm"]["symmetric_mean_mm"],
            "heart_predicted_mm": row["organs"]["heart"]["predicted_surface_error_mm"]["symmetric_mean_mm"],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
