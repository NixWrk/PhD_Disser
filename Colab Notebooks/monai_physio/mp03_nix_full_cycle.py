"""Screen full-resolution 00->other Nix registrations against automatic masks.

This is an internal technical screen only. No field is passed to FEM without an
independent accepted phase annotation and the previously specified geometry QC.
Dense fields and logs remain under ignored runs/.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def run_logged(command, logfile):
    with logfile.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed; inspect ignored local log {logfile.name}")


def summary(report):
    jac = report["field_qc"]["jacobian_of_pullback"]
    stationary = report["mask_qc"]["stationary"]
    warped = report["mask_qc"]["greedy"]
    surface = report["same_surface_qc"]
    return {
        "target_phase": report["fixed_target"],
        "input_sha256": report["input_sha256"],
        "smoothing": report["method"]["deformable_smoothing"],
        "stride_ijk": report["grid_qc"]["stride_ijk"],
        "stationary_dice_auto": stationary["dice_vs_auto_mask"],
        "warped_dice_auto": warped["dice_vs_auto_mask"],
        "stationary_assd_mm": stationary["surface_assd_mm"],
        "warped_assd_mm": warped["surface_assd_mm"],
        "min_sampled_jacobian": jac["minimum"],
        "nonpositive_sampled_jacobian_fraction": jac["nonpositive_fraction"],
        "source_mask_touches_fov": report["grid_qc"]["source_mask_touches_full_fov"],
        "target_mask_touches_fov": report["grid_qc"]["target_mask_touches_full_fov"],
        "local_field_sha256": report["field_qc"]["local_field_sha256"],
        "surface_vertex_count": surface["vertex_count"],
        "surface_faces_preserved": surface["same_vertices_and_faces_for_all_three_candidates"],
        "surface_mean_distance_stationary_mm": surface["mean_distance_mm"]["stationary"],
        "surface_mean_distance_warped_mm": surface["mean_distance_mm"]["greedy"],
        "surface_warped_p95_distance_mm": surface["greedy_p95_distance_mm"],
        "surface_vertices_outside_crop_fraction": surface["greedy_vertices_outside_crop_fraction"],
        "anatomical_validation": False,
        "screen": ("technical_candidate_only" if jac["minimum"] > 0 and
                   warped["dice_vs_auto_mask"] > stationary["dice_vs_auto_mask"]
                   else "technical_reject"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    args = parser.parse_args()
    subject_root = args.derived_root / "nix"
    phases = subject_root / "phases"
    products = subject_root / "products_v2"
    conversion = phases / "conversion_manifest.json"
    manifest = json.loads(conversion.read_text(encoding="utf-8"))
    if len(manifest["phases"]) != 12:
        raise ValueError("Expected 12 Nix CT phases")
    reference_ct_sha = digest(phases / "phase_00.nii.gz")
    reference_mask_sha = digest(products / "phase_00" / "whole_heart.nii.gz")
    summaries = []
    for phase_index in range(1, 12):
        phase = f"phase_{phase_index:02d}"
        if phase_index == 1:
            report_path = RUNS / "mp03_nix_fullres_00_01_smooth_qc.json"
        elif phase_index == 7:
            report_path = HERE / "mp03_nix_fullres_00_07_smooth_qc.json"
        else:
            rigid = RUNS / f"mp03_nix_00_{phase_index:02d}_rigid_qc.json"
            report_path = RUNS / f"mp03_nix_fullres_00_{phase_index:02d}_qc.json"
            field = RUNS / f"mp03_nix_fullres_00_{phase_index:02d}_field.npz"
            if not rigid.is_file():
                run_logged([
                    sys.executable, "-B", str(HERE / "mp03_04_pilot.py"),
                    "--phase-dir", str(phases), "--products-dir", str(products),
                    "--reference", "phase_00", "--target", phase,
                    "--subject-alias", "nix", "--output", str(rigid),
                ], RUNS / f"mp03_nix_00_{phase_index:02d}_rigid.log")
            if not report_path.is_file():
                run_logged([
                    sys.executable, "-B", str(HERE / "mp03_04_monai_pair.py"),
                    "--phase-dir", str(phases), "--products-dir", str(products),
                    "--reference-phase", "phase_00", "--target-phase", phase,
                    "--conversion-manifest", str(conversion), "--rigid-qc", str(rigid),
                    "--output-json", str(report_path), "--field-output", str(field),
                    "--stride", "1", "1", "1", "--crop-source-only", "--margin-mm", "25",
                    "--deformable-smoothing", "3.0vox 1.0vox",
                ], RUNS / f"mp03_nix_fullres_00_{phase_index:02d}.log")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["subject_alias"] != "nix" or report["moving_reference"] != "phase_00"
                or report["fixed_target"] != phase or
                report["grid_qc"]["stride_ijk"] != [1, 1, 1] or
                report["independent_anatomical_reference"]):
            raise ValueError(f"Report provenance mismatch for {phase}")
        expected = {
            "reference_ct": reference_ct_sha,
            "target_ct": digest(phases / f"{phase}.nii.gz"),
            "reference_mask": reference_mask_sha,
            "target_mask": digest(products / phase / "whole_heart.nii.gz"),
        }
        if report["input_sha256"] != expected:
            raise ValueError(f"Stored registration inputs no longer match {phase}")
        field = (HERE / report["field_qc"]["local_field_relative_path"]).resolve()
        if not field.is_relative_to(RUNS.resolve()) or digest(field) != report["field_qc"]["local_field_sha256"]:
            raise ValueError(f"Stored field hash mismatch for {phase}")
        summaries.append(summary(report))
        print(f"{phase}: screened", flush=True)
    result = {
        "schema": "mp03_nix_full_cycle_v1",
        "subject_alias": "nix",
        "reference_phase": "phase_00",
        "target_count": len(summaries),
        "results": summaries,
        "interpretation": "Full-resolution internal auto-mask screen, not independent anatomical validation or FEM admission",
    }
    (HERE / "mp03_nix_full_cycle.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
