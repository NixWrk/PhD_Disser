from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "validate_manual_cardiac_masks.py"
SPEC = importlib.util.spec_from_file_location("validate_manual_cardiac_masks", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_compare_masks_reports_volume_dice_and_zero_surface_errors_for_identical_masks():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    metrics = module.compare_masks(
        mask,
        mask.copy(),
        np.diag([2.0, 3.0, 4.0, 1.0]),
        1.0,
    )
    assert metrics["candidate_volume_ml"] == pytest.approx(4 * 4 * 4 * 24 / 1000)
    assert metrics["reference_volume_ml"] == pytest.approx(metrics["candidate_volume_ml"])
    assert metrics["signed_volume_error_ml"] == pytest.approx(0.0)
    assert metrics["absolute_volume_error_ml"] == pytest.approx(0.0)
    assert metrics["signed_volume_error_percent"] == pytest.approx(0.0)
    assert metrics["absolute_volume_error_percent"] == pytest.approx(0.0)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["surface_dice"] == pytest.approx(1.0)
    assert metrics["assd_mm"] == pytest.approx(0.0)
    assert metrics["hd95_mm"] == pytest.approx(0.0)


def test_compare_masks_uses_affine_determinant_and_surface_tolerance():
    reference = np.zeros((12, 12, 12), dtype=bool)
    reference[3:8, 3:8, 3:8] = True
    candidate = np.zeros_like(reference)
    candidate[3:8, 3:8, 4:9] = True
    affine = np.diag([1.0, 1.0, 2.0, 1.0])

    tight = module.compare_masks(candidate, reference, affine, 0.5)
    loose = module.compare_masks(candidate, reference, affine, 2.0)
    assert tight["candidate_volume_ml"] == pytest.approx(tight["reference_volume_ml"])
    assert tight["dice"] == pytest.approx(4 / 5)
    assert tight["surface_dice"] < loose["surface_dice"]
    assert loose["surface_dice"] == pytest.approx(1.0)
    assert tight["assd_mm"] > 0.0
    assert tight["hd95_mm"] > 0.0


def test_surface_metrics_rejects_empty_masks():
    empty = np.zeros((4, 4, 4), dtype=bool)
    nonempty = empty.copy()
    nonempty[1:3, 1:3, 1:3] = True
    with pytest.raises(ValueError, match="empty"):
        module.surface_metrics(empty, nonempty, (1.0, 1.0, 1.0), 1.0)


def test_staple_consensus_accepts_three_repeated_annotations_and_rejects_two():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4, 4, 4] = True
    consensus, info = module.staple_consensus([mask, mask.copy(), mask.copy()])
    assert np.array_equal(consensus, mask)
    assert info["method"] == "staple"
    assert info["converged"] is True
    assert info["estimated_foreground_prior"] == pytest.approx(1 / mask.size)
    with pytest.raises(ValueError, match="at least three"):
        module.staple_consensus([mask, mask.copy()])


def test_cluster_bootstrap_averages_phases_within_subject_before_resampling():
    rows = []
    for subject_id, phase_values in {
        "Adam": [0.0, 2.0],
        "Nix": [4.0, 4.0],
        "Georg": [8.0],
    }.items():
        for phase_id, value in enumerate(phase_values):
            row = {
                "structure": "whole_heart",
                "candidate_id": "total",
                "subject_id": subject_id,
                "phase_id": f"phase_{phase_id}",
                "is_primary_reference": True,
            }
            row.update({metric: value for metric in module.METRIC_NAMES})
            rows.append(row)

    summaries = module.cluster_bootstrap_summary(
        rows,
        {"iterations": 2000, "seed": 1234, "confidence_level": 0.95},
        "pilot_descriptive_only",
    )
    summary = next(row for row in summaries if row["metric"] == "dice")
    assert summary["estimate"] == pytest.approx((1.0 + 4.0 + 8.0) / 3.0)
    assert summary["unique_subjects"] == 3
    assert summary["observations"] == 5
    assert summary["cluster_unit"] == "subject_id"
    assert summary["status"] == "pilot_cluster_bootstrap_interval_not_population_ci"
    assert summary["interval_lower"] <= summary["estimate"] <= summary["interval_upper"]


def test_end_to_end_report_omits_local_mask_paths(tmp_path):
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    candidate_path = tmp_path / "private" / "candidate.nii.gz"
    reference_path = tmp_path / "private" / "reference.nii.gz"
    candidate_path.parent.mkdir(parents=True)
    for path in (candidate_path, reference_path):
        image = nib.Nifti1Image(mask, affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, str(path))

    manifest = {
        "schema_version": 1,
        "inference_scope": "pilot_descriptive_only",
        "analysis_plan_locked_before_evaluation": True,
        "surface_tolerance_mm": 1.0,
        "surface_tolerance_locked_before_evaluation": True,
        "surface_tolerance_rationale": "Synthetic locked test tolerance.",
        "bootstrap": {
            "cluster_unit": "subject_id",
            "estimand": "mean_of_subject_means",
            "iterations": 100,
            "seed": 7,
            "confidence_level": 0.95,
        },
        "samples": [
            {
                "sample_id": "sample_1",
                "subject_id": "subject_alias",
                "phase_id": "phase_00",
                "structure": "whole_heart",
                "structure_definition_version": "test_v1",
                "candidates": [
                    {
                        "candidate_id": "candidate_1",
                        "path": str(candidate_path),
                        "source_kind": "totalsegmentator_total_heart_raw",
                        "software_name": "Synthetic",
                        "software_version": "1",
                        "model_task": "total",
                        "model_class": "heart",
                        "postprocessing": "none",
                        "label_value": 1,
                    }
                ],
                "manual_annotations": [
                    {
                        "annotation_id": "expert_a_session_1",
                        "path": str(reference_path),
                        "expert_id": "expert_a",
                        "session_id": "session_1",
                        "mask_version": "v1",
                        "source_project_sha256": hashlib.sha256(
                            b"synthetic_inobitec_project"
                        ).hexdigest(),
                        "annotation_mode": "manual_inobitec",
                        "independent_from_candidate": True,
                        "blinded_to_candidate_during_initial_annotation": True,
                        "label_value": 1,
                    },
                    {
                        "annotation_id": "expert_b_session_1",
                        "path": str(reference_path),
                        "expert_id": "expert_b",
                        "session_id": "session_1",
                        "mask_version": "v1",
                        "source_project_sha256": hashlib.sha256(
                            b"synthetic_inobitec_project"
                        ).hexdigest(),
                        "annotation_mode": "manual_inobitec",
                        "independent_from_candidate": True,
                        "blinded_to_candidate_during_initial_annotation": True,
                        "label_value": 1,
                    }
                ],
                "primary_reference": {
                    "method": "single",
                    "annotation_id": "expert_a_session_1",
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "report"
    report = module.run_validation(manifest_path, output_dir)

    provenance = report["samples"][0]["candidate_masks"]["candidate_1"]["provenance"]
    assert "path" not in provenance
    encoded = json.dumps(report)
    assert str(candidate_path) not in encoded
    assert str(reference_path) not in encoded
    assert report["status"] == "method_development_not_clinically_validated"
    assert report["reproducibility_comparisons"] == 1
    assert (output_dir / "validation_metrics.csv").is_file()
    assert (output_dir / "validation_report.json").is_file()
    bootstrap_csv = (output_dir / "cluster_bootstrap_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "manual_reproducibility" in bootstrap_csv
    assert "interexpert" in bootstrap_csv
