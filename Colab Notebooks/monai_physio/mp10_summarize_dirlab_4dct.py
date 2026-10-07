"""Build a public, non-image summary of the 10-case DIR-Lab 4DCT audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mp10_respiratory_phases import validate_scenario


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qc = {}
    for line in (args.run_root / "source_anatomy_qc.jsonl").read_text(encoding="utf-8-sig").splitlines():
        entry = json.loads(line)
        qc[Path(entry["case_dir"]).name] = entry
    rows = []
    checkpoint = None
    for number in range(1, 11):
        case_name = "case_09_zflip" if number == 9 else f"case_{number:02d}"
        case_dir = args.run_root / case_name
        landmark = load(case_dir / "landmark_evaluation.json")
        conversion = load(case_dir / "conversion_manifest.json")
        segment = load(case_dir / "seg" / "T70" / "segmentation_manifest.json")
        fit = load(case_dir / "fit_T70" / "fit_report.json")
        scenario = load(case_dir / "scenario_T70" / "scenario_report.json")
        contract = validate_scenario(scenario, "dirlab")
        if landmark.get("phase_contract") != contract:
            raise ValueError(f"Case {number}: rerun landmark evaluation with the checked phase contract")
        if landmark["conversion_manifest_sha256"] != sha256(case_dir / "conversion_manifest.json"):
            raise ValueError(f"Case {number}: landmark score uses stale conversion manifest")
        if segment["ct_sha256"] != conversion["phases"]["T70"]["nifti_sha256"]:
            raise ValueError(f"Case {number}: T70 segmentation does not match input CT")
        if fit["source_ct_sha256"] != segment["ct_sha256"] or fit["target_phases_used"]:
            raise ValueError(f"Case {number}: fit provenance includes a target phase or wrong CT")
        if scenario["source_fit_report_sha256"] != sha256(case_dir / "fit_T70" / "fit_report.json"):
            raise ValueError(f"Case {number}: scenario does not match fit report")
        if scenario["source_landmarks_sha256"] or scenario["target_landmarks_sha256"]:
            raise ValueError(f"Case {number}: landmarks entered the scenario calculation")
        if not qc[case_name]["source_only_anatomy_qc_pass"]:
            raise ValueError(f"Case {number}: source anatomy QC failed")
        current_checkpoint = landmark["model_checkpoint_sha256"]
        if checkpoint is None:
            checkpoint = current_checkpoint
        elif checkpoint != current_checkpoint:
            raise ValueError("Mixed network checkpoints")
        extreme = landmark["extreme_300_T50_to_T00"]
        row = {
            "case": number,
            "run_case_dir": case_name,
            "source_orientation": conversion["orientation"],
            "source_anatomy_qc": {
                key: qc[case_name][key] for key in (
                    "upper_minus_lower_z_voxels", "right_minus_left_x_voxels",
                    "source_only_anatomy_qc_pass",
                )
            },
            "T50_to_T00_expert_300": extreme,
            "sampled_75_by_phase": landmark["sampled_75_trajectory_from_T50"],
            "conversion_manifest_sha256": sha256(case_dir / "conversion_manifest.json"),
            "source_segmentation_manifest_sha256": sha256(case_dir / "seg" / "T70" / "segmentation_manifest.json"),
            "fit_report_sha256": landmark["fit_report_sha256"],
            "scenario_report_sha256": landmark["scenario_report_sha256"],
            "landmark_evaluation_sha256": sha256(case_dir / "landmark_evaluation.json"),
        }
        masks = case_dir / "mask_evaluation.json"
        if masks.is_file():
            report = load(masks)
            if report["conversion_manifest_sha256"] != sha256(case_dir / "conversion_manifest.json"):
                raise ValueError(f"Case {number}: mask score uses stale conversion manifest")
            row["mask_evaluation"] = {
                "source_lung_volume_ml": report["source_mask"]["lung"]["volume_ml"],
                "source_heart_volume_ml": report["source_mask"]["heart"]["volume_ml"],
                "rows": report["rows"],
                "evaluation_sha256": sha256(masks),
            }
        rows.append(row)
    observed = np.asarray([r["T50_to_T00_expert_300"]["observed_displacement_mean_mm"] for r in rows])
    predicted = np.asarray([r["T50_to_T00_expert_300"]["predicted_displacement_mean_mm"] for r in rows])
    residual = np.asarray([r["T50_to_T00_expert_300"]["predicted_tre_mean_mm"] for r in rows])
    summary = {
        "schema": "mp10_dirlab_4dct_ten_case_summary_v1",
        "status": "published_phase_convention_verified_anatomical_use_not_validated",
        "physiological_phase_correspondence_verified": True,
        "phase_verification_scope": "published acquisition bins and stage parser; individual amplitude and exact checkpoint metadata are not verified",
        "checkpoint_patient_overlap_verified": False,
        "phase_training_audit": "mp10_phase_training_audit_2026-10-07.json",
        "dataset": "Emory DIR-Lab 4DCT; 10 public cases; training overlap not verified",
        "source_phase": "T70 CT and TotalSegmentator v2 source masks only",
        "checkpoint_sha256": checkpoint,
        "temporal_rule": "Txx mapped directly to network stage xx/100, with T70 anchor; no outcome-dependent phase matching",
        "reference": "300 paired expert lung landmarks T50 to T00 per case; 75 trajectories T00-T50; automatic target masks only for contour subset",
        "case_count": len(rows),
        "extreme_landmark_count": 300 * len(rows),
        "extreme_case_mean_static_TRE_mm": float(observed.mean()),
        "extreme_case_mean_predicted_TRE_mm": float(residual.mean()),
        "extreme_case_mean_predicted_displacement_mm": float(predicted.mean()),
        "extreme_cases_improved": int(np.sum(residual < observed)),
        "extreme_cases_underpredicted_mean_displacement": int(np.sum(predicted < observed)),
        "extreme_case_mean_predicted_to_observed_amplitude_ratio": float(np.mean(predicted / observed)),
        "case09_automated_orientation_failure": {
            "initial_source_anatomy_qc_pass": qc["case_09"]["source_only_anatomy_qc_pass"],
            "initial_source_middle_lobe_ml": load(args.run_root / "case_09" / "seg" / "T70" / "segmentation_manifest.json")["masks"]["lung_middle_lobe_right"]["volume_ml"],
            "first_fit_error": "middle-lobe surface vanished on 2.5 mm fit grid",
            "resolution": "source-only T70 lobe anatomy QC selects superior axis reversal; no target-phase images or landmarks used",
        },
        "rows": rows,
        "limitations": [
            "Landmark score covers the complete frozen network plus surface-to-volume displacement postprocessing, not isolated neural mesh vertices.",
            "Fields at T50 landmark positions approximate displacement transfer from T70; no image-based temporal registration was performed.",
            "Source shape fitting and target contour comparison use TotalSegmentator v2; manual contour validation is absent.",
            "Case 5 and 9 superior-axis choices were corrected with source T70 lobe anatomy QC and must be counted as QC interventions.",
            "The packaged checkpoint training manifest was not independently audited for case overlap.",
            "The heart field is an extrapolation of lung motion, not a separately trained heart predictor.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ids = np.arange(1, 11)
    axes[0].plot(ids, observed, "o-", label="Наблюдаемое")
    axes[0].plot(ids, predicted, "s-", label="Прогноз")
    axes[0].set(xlabel="Случай DIR-Lab", ylabel="Среднее смещение, мм", xticks=ids)
    axes[0].legend(fontsize=8)
    axes[1].plot(ids, observed, "o-", label="Без движения")
    axes[1].plot(ids, residual, "s-", label="С моделью")
    axes[1].set(xlabel="Случай DIR-Lab", ylabel="Ошибка ориентиров, мм", xticks=ids)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    chart = args.output.with_suffix(".png")
    figure.savefig(chart, dpi=180)
    plt.close(figure)
    print(json.dumps({
        "cases": len(rows), "static_TRE_mm": float(observed.mean()),
        "predicted_TRE_mm": float(residual.mean()),
        "observed_motion_mm": float(observed.mean()),
        "predicted_motion_mm": float(predicted.mean()),
        "summary": str(args.output), "chart": str(chart),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
