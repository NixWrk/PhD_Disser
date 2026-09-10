from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmark_blood_segmentation_candidates.py"
SPEC = importlib.util.spec_from_file_location("benchmark_blood_segmentation_candidates", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, *, unit: str | None = "mm") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    if unit is not None:
        image.header.set_xyzt_units(unit)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))


def build_manifest(tmp_path: Path, *, with_reference: bool, overlap: bool = False) -> Path:
    shape = (12, 12, 12)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    records = []
    subjects = ["Adam", "Nix", "Georg"] if with_reference else ["Adam"]
    for subject_index, subject in enumerate(subjects):
        phase_id = "phase_00"
        ct = np.zeros(shape, dtype=np.int16)
        ct[2:10, 2:10, 2:10] = 100
        ct_path = tmp_path / "source" / subject / "phases" / "phase_00.nii.gz"
        save_nifti(ct_path, ct, affine)
        total = np.zeros(shape, dtype=np.uint8)
        total[2:10, 2:10, 2:10] = 1
        total_path = tmp_path / "source" / subject / "total_heart.nii.gz"
        save_nifti(total_path, total, affine)
        for chamber_index, chamber in enumerate(module.CHAMBERS):
            mask = np.zeros(shape, dtype=np.uint8)
            starts = ((3, 3, 3), (7, 3, 3), (3, 7, 3), (7, 7, 3))[chamber_index]
            x, y, z = starts
            mask[x : x + 2, y : y + 2, z : z + 2] = 1
            if overlap and chamber == "left_ventricle_blood":
                mask[3:5, 3:5, 3:5] = 1
            raw_path = tmp_path / "source" / subject / f"raw_{chamber}.nii.gz"
            save_nifti(raw_path, mask, affine)
            alternative = mask.copy()
            alternative[min(x + 2, shape[0] - 1), y, z] = 1
            alternative_path = tmp_path / "source" / subject / f"alternative_{chamber}.nii.gz"
            save_nifti(alternative_path, alternative, affine)
            reference_path = tmp_path / "source" / subject / f"manual_{chamber}.nii.gz"
            save_nifti(reference_path, mask, affine)
            candidates = [
                {
                    "candidate_id": f"raw_{chamber}",
                    "candidate_set_id": "raw_totalsegmentator",
                    "path": raw_path.relative_to(tmp_path).as_posix(),
                    "method_category": "raw_automatic",
                    "software_name": "TotalSegmentator",
                    "software_version": "2.18.0",
                    "model_or_method": "heartchambers_highres",
                    "independent_from_reference": True,
                },
                {
                    "candidate_id": f"alternative_{chamber}",
                    "candidate_set_id": "alternative_method",
                    "path": alternative_path.relative_to(tmp_path).as_posix(),
                    "method_category": "alternative_algorithm",
                    "software_name": "SyntheticAlternative",
                    "software_version": "1.0",
                    "model_or_method": "synthetic",
                    "independent_from_reference": True,
                    "manual_interaction_time_s": 2.5,
                    "manual_clicks": 3,
                },
            ]
            record = {
                "sample_id": f"{subject}_{phase_id}_{chamber}",
                "subject_id": subject,
                "phase_id": phase_id,
                "structure": chamber,
                "ct_path": ct_path.relative_to(tmp_path).as_posix(),
                "total_heart_path": total_path.relative_to(tmp_path).as_posix(),
                "candidates": candidates,
            }
            if with_reference:
                record["manual_reference"] = {
                    "reference_id": f"manual_{subject}_{chamber}",
                    "path": reference_path.relative_to(tmp_path).as_posix(),
                    "reference_kind": "manual_inobitec",
                    "independent_from_candidates": True,
                    "blinded_to_candidates": True,
                }
            records.append(record)
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_plan_locked_before_evaluation": True,
                "surface_tolerance_mm": 1.0,
                "surface_tolerance_locked_before_evaluation": True,
                "inference_scope": "pilot_descriptive_only",
                "geometry_mm_confirmed": False,
                "bootstrap": {"iterations": 200, "seed": 42, "confidence_level": 0.95},
                "samples": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_accuracy_compares_multiple_candidates_and_bootstraps_subjects(tmp_path):
    manifest_path = build_manifest(tmp_path, with_reference=True)
    report = module.benchmark(manifest_path, tmp_path / "external_benchmark")

    assert report["analysis_mode"] == "accuracy"
    assert report["status"] == "accuracy_metrics_computed_pending_interpretation"
    assert report["accuracy_rows"] == 24
    assert report["technical_groups"]
    raw_row = next(row for row in report["samples"] if row["structure"] == "left_atrium_blood")["candidates"][0]
    assert raw_row["method_category"] == "raw_automatic"
    assert len(report["bootstrap"]) == 8 * len(module.METRIC_NAMES)
    dice_summary = next(
        row for row in report["bootstrap"]
        if row["candidate_set_id"] == "raw_totalsegmentator" and row["metric"] == "dice"
    )
    assert dice_summary["estimate"] == pytest.approx(1.0)
    assert dice_summary["unique_subjects"] == 3
    assert dice_summary["status"] == "pilot_cluster_bootstrap_interval_not_population_ci"
    accuracy_csv = (tmp_path / "external_benchmark" / "benchmark_accuracy_metrics.csv").read_text(encoding="utf-8")
    assert "manual_interaction_time_s" in accuracy_csv
    encoded = json.dumps(report, ensure_ascii=False)
    assert str(tmp_path) not in encoded


def test_technical_only_reports_overlap_containment_and_boundary_qc(tmp_path):
    manifest_path = build_manifest(tmp_path, with_reference=False, overlap=True)
    report = module.benchmark(
        manifest_path,
        tmp_path / "external_technical",
        boundary_hu_qc=True,
    )

    assert report["analysis_mode"] == "technical_only"
    assert report["status"] == "no_accuracy_inference"
    assert report["accuracy_rows"] == 0
    group = report["technical_groups"][0]
    assert group["complete_four_chamber_set"] is True
    assert group["pairwise_overlaps"]["left_atrium_blood__left_ventricle_blood"]["overlap_voxels"] > 0
    assert group["union"]["outside_total_heart_voxels"] == 0
    technical_rows = [row for row in report["technical_groups"]]
    assert technical_rows
    csv_text = (tmp_path / "external_technical" / "benchmark_technical_metrics.csv").read_text(encoding="utf-8")
    assert "technical_pairwise_overlap_qc" in csv_text
    assert report["review"]["boundary_hu_qc_enabled"] is True


def test_benchmark_requires_locked_plan_and_rejects_repository_output(tmp_path):
    manifest_path = build_manifest(tmp_path, with_reference=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_plan_locked_before_evaluation"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="analysis_plan_locked_before_evaluation"):
        module.benchmark(manifest_path, tmp_path / "external")

    manifest["analysis_plan_locked_before_evaluation"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the repository"):
        module.benchmark(manifest_path, MODULE_PATH.parent / "derived")
