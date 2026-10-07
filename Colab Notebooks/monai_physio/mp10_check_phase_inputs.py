"""Audit saved phase inputs without refitting or changing neural fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mp10_respiratory_phases import phase_stage, validate_scenario


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    scenarios = [
        ("dirlab", args.run_root / "dirlab_4dct" /
         ("case_09_zflip" if n == 9 else f"case_{n:02d}") / "scenario_T70")
        for n in range(1, 11)
    ] + [("tcia", args.run_root / "tcia_validation" / f"scenario_{p}")
         for p in ("g070", "g000")]
    for dataset, directory in scenarios:
        path = directory / "scenario_report.json"
        report = load(path)
        contract = validate_scenario(report, dataset)
        if report["source_landmarks_sha256"] or report["target_landmarks_sha256"]:
            raise ValueError("Control landmarks entered a prediction")
        stages = contract["target_stages"]
        for stage in stages:
            if not (directory / f"field_stage_{stage:.2f}.mha").is_file():
                raise FileNotFoundError(f"Missing neural field at {stage}")
        if dataset == "dirlab":
            evaluation = load(directory.parent / "landmark_evaluation.json")
            labels = list(evaluation["sampled_75_trajectory_from_T50"])
            fit = load(directory.parent / "fit_T70" / "fit_report.json")
            conversion = load(directory.parent / "conversion_manifest.json")
            if fit["source_ct_sha256"] != conversion["phases"][report["source_phase"]]["nifti_sha256"]:
                raise ValueError("Source fit CT hash does not match the claimed DIR-Lab phase")
        else:
            evaluation = load(directory.parent / f"eval_{report['source_phase']}" / "evaluation.json")
            labels = [item["phase"] for item in evaluation["metrics"]]
            fit = load(directory.parent / f"fit_{report['source_phase']}" / "fit_report.json")
            if fit["source_ct_sha256"] != evaluation["ct_sha256"][report["source_phase"]]:
                raise ValueError("Source fit CT hash does not match the claimed TCIA phase")
            if any(item["model_stage"] != phase_stage(item["phase"], dataset)
                   for item in evaluation["metrics"]):
                raise ValueError("TCIA evaluation stage mismatch")
        mapping = {p: phase_stage(p, dataset) for p in labels}
        if any(s not in stages for s in mapping.values()):
            raise ValueError("Target phase missing from saved inference stages")
        rows.append({
            "scenario": directory.relative_to(args.run_root).as_posix(),
            "scenario_report_sha256": digest(path),
            "checkpoint_sha256": report["model_checkpoint_sha256"],
            "phase_contract": contract,
            "evaluation_phase_to_stage": mapping,
            "source_ct_phase_binding_hash_verified": True,
            "existing_numeric_inputs_need_change": False,
        })
    tcia = load(args.run_root / "tcia_validation" / "eval_g070" / "evaluation.json")
    volumes = {r["phase"]: r["lung_volume_ml"] for r in tcia["metrics"]}
    summary = {
        "schema": "mp10_saved_phase_input_audit_v1", "date": "2026-10-07",
        "status": "published_phase_convention_checked_on_saved_inputs",
        "scenario_count": len(rows), "rows": rows,
        "numeric_network_inputs_changed": False,
        "reason": "Existing Txx/100 and gPPP/100 inputs already match the published acquisition convention; g070 end-exhale wording is erroneous.",
        "tcia_volume_qc_only_not_phase_selection": {
            "observed_lung_volumes_ml": volumes,
            "maximum_volume_phase": max(volumes, key=volumes.get),
            "minimum_volume_phase": min(volumes, key=volumes.get),
            "reference_g070_is_not_volume_extreme": volumes["g050"] < volumes["g070"] < volumes["g000"],
        },
        "limitations": [
            "Acquisition conventions are verified; exact checkpoint patient and physiological manifests are absent.",
            "Equal breathing percentages do not imply equal lung volume, motion amplitude or deep-inspiration breath hold.",
            "Fitting a non-g070 source remains outside the published shape fitting protocol.",
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"checked_scenarios": len(rows), "numeric_inputs_changed": False,
                      "volume_max": max(volumes, key=volumes.get),
                      "volume_min": min(volumes, key=volumes.get)}))


if __name__ == "__main__":
    main()
