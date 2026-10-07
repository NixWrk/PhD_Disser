"""Score the frozen lung model on independent DIR-Lab 4DCT landmarks.

One T70 CT is the only anatomy used for shape fitting. Target-phase CTs and
landmarks are withheld until evaluation. The network's surface motion is
extended into the lungs by MONAI Physio's deformation-field postprocessor;
therefore landmark scores assess that complete motion-transfer workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import itk
import numpy as np
from scipy.ndimage import map_coordinates

from mp10_respiratory_phases import phase_stage, validate_scenario


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def landmark_member(archive: ZipFile, phase: str, sample: bool) -> str:
    token = "4d-75_" if sample else "300_"
    matches = [
        name for name in archive.namelist()
        if name.lower().endswith(".txt") and token in name.lower()
        and f"_{phase.lower()}" in name.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {'sampled' if sample else 'extreme'} {phase} landmark file: {matches}")
    return matches[0]


def points_lps(archive: ZipFile, phase: str, sample: bool, manifest: dict) -> np.ndarray:
    name = landmark_member(archive, phase, sample)
    voxels = np.atleast_2d(np.loadtxt(io.BytesIO(archive.read(name)), dtype=float)) - 1.0
    if voxels.shape[1] != 3:
        raise ValueError(f"{phase}: expected Nx3 landmark coordinates")
    source = manifest["phases"]["T70"]
    shape = np.asarray(source["shape_xyz"], dtype=float)
    orientation = manifest["orientation"]
    if not orientation["right_is_high_index"]:
        voxels[:, 0] = shape[0] - 1 - voxels[:, 0]
    if orientation["posterior_is_high_row"]:
        voxels[:, 1] = shape[1] - 1 - voxels[:, 1]
    if not orientation["superior_is_high_index"]:
        voxels[:, 2] = shape[2] - 1 - voxels[:, 2]
    if not np.all((voxels >= 0) & (voxels < shape)):
        raise ValueError(f"{phase}: landmark outside CT image")
    ras = voxels * np.asarray(source["spacing_xyz_mm"])
    return ras * np.asarray([-1.0, -1.0, 1.0])


def field_at(points_lps: np.ndarray, field: itk.Image) -> tuple[np.ndarray, float]:
    array = itk.array_from_image(field)
    axes = np.asarray(field.GetDirection()) @ np.diag(np.asarray(field.GetSpacing()))
    ijk = np.linalg.solve(axes, (points_lps - np.asarray(field.GetOrigin())).T)
    size = np.asarray(field.GetLargestPossibleRegion().GetSize())
    coverage = float(np.mean(np.all((ijk >= 0) & (ijk <= size[:, None] - 1), axis=0)))
    coords = [ijk[2], ijk[1], ijk[0]]
    vectors = np.column_stack([
        map_coordinates(array[..., axis], coords, order=1, mode="constant", cval=0.0)
        for axis in range(3)
    ])
    return vectors, coverage


def score(source: np.ndarray, target: np.ndarray, shift: np.ndarray) -> dict[str, float | int]:
    if source.shape != target.shape or source.shape != shift.shape:
        raise ValueError("Landmark points and predicted shifts must have the same shape")
    actual = target - source
    static = np.linalg.norm(actual, axis=1)
    residual = np.linalg.norm(source + shift - target, axis=1)
    return {
        "landmark_count": len(source),
        "observed_displacement_mean_mm": float(static.mean()),
        "predicted_displacement_mean_mm": float(np.linalg.norm(shift, axis=1).mean()),
        "static_tre_mean_mm": float(static.mean()),
        "predicted_tre_mean_mm": float(residual.mean()),
        "predicted_tre_median_mm": float(np.median(residual)),
        "predicted_tre_p95_mm": float(np.quantile(residual, 0.95)),
        "mean_tre_change_predicted_minus_static_mm": float(residual.mean() - static.mean()),
        "fraction_landmarks_improved": float(np.mean(residual < static)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    conversion = json.loads((args.case_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    scenario_dir = args.case_dir / "scenario_T70"
    scenario = json.loads((scenario_dir / "scenario_report.json").read_text(encoding="utf-8"))
    contract = validate_scenario(scenario, "dirlab")
    if scenario["stage_anchor"] != 0.7 or scenario["source_phase"] != "T70":
        raise ValueError("Only the predeclared T70=0.7 reference protocol is scored")
    with ZipFile(args.archive) as archive:
        extreme_source = points_lps(archive, "T50", False, conversion)
        extreme_target = points_lps(archive, "T00", False, conversion)
        sampled_source = points_lps(archive, "T50", True, conversion)
        sampled_target = {
            phase: points_lps(archive, phase, True, conversion)
            for phase in ("T00", "T10", "T20", "T30", "T40", "T50")
        }
    needed = ["T00", "T10", "T20", "T30", "T40", "T50"]
    fields = {
        phase: itk.imread(str(scenario_dir / f"field_stage_{phase_stage(phase, 'dirlab'):.2f}.mha"))
        for phase in needed
    }
    base_extreme, base_coverage = field_at(extreme_source, fields["T50"])
    to_extreme, target_coverage = field_at(extreme_source, fields["T00"])
    if min(base_coverage, target_coverage) < 1.0:
        raise ValueError("Extreme landmarks are not covered by both fields")
    extreme = score(extreme_source, extreme_target, to_extreme - base_extreme)
    base_sampled, sampled_coverage = field_at(sampled_source, fields["T50"])
    if sampled_coverage < 1.0:
        raise ValueError("Sampled landmarks are not covered by reference field")
    trajectory = {}
    trajectory_coverage = {}
    for phase in needed:
        target_shift, coverage = field_at(sampled_source, fields[phase])
        if coverage < 1.0:
            raise ValueError(f"Sampled landmarks at {phase} not covered by field")
        trajectory_coverage[phase] = coverage
        trajectory[phase] = score(
            sampled_source, sampled_target[phase], target_shift - base_sampled
        )
    result = {
        "schema": "mp10_dirlab_4dct_landmark_evaluation_v1",
        "case_alias": conversion["case_alias"],
        "status": "independent_expert_landmarks_frozen_network_T70_input",
        "model_checkpoint_sha256": sha256(args.model_dir / "mgn_stage_model.pt"),
        "conversion_manifest_sha256": sha256(args.case_dir / "conversion_manifest.json"),
        "fit_report_sha256": sha256(args.case_dir / "fit_T70" / "fit_report.json"),
        "scenario_report_sha256": sha256(scenario_dir / "scenario_report.json"),
        "source_CT": "T70 only; shape fitted by TS2 masks and affine ICP/PCA",
        "target_phases_used_only_for_evaluation": needed,
        "stage_mapping": "DIR-Lab Txx -> network xx/100; no stage selected from outcome",
        "phase_contract": contract,
        "coordinate_system": "raw DIR-Lab one-based voxel xyz -> anatomically oriented RAS mm -> LPS mm",
        "metric_scope": "interior lung landmarks after MONAI Physio surface-to-volume field postprocessing",
        "field_pairing": "d(Txx)-d(T50) evaluated at T50 landmark positions, first-order displacement transfer",
        "field_coverage_fraction": {
            "extreme_T50": base_coverage,
            "extreme_T00": target_coverage,
            "sampled_T50": sampled_coverage,
            "sampled_phases": trajectory_coverage,
        },
        "extreme_300_T50_to_T00": extreme,
        "sampled_75_trajectory_from_T50": trajectory,
    }
    path = args.case_dir / "landmark_evaluation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_alias": result["case_alias"],
                      "extreme_300": extreme,
                      "sampled_phase_count": len(trajectory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
