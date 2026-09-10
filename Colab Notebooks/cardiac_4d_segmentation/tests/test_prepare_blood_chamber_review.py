from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "prepare_blood_chamber_review.py"
SPEC = importlib.util.spec_from_file_location("prepare_blood_chamber_review", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def save_nifti(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    spatial_unit: str | None = "mm",
    forms: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    if spatial_unit is not None:
        image.header.set_xyzt_units(spatial_unit)
    if forms:
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=1)
    else:
        image.set_qform(affine, code=0)
        image.set_sform(affine, code=0)
    nib.save(image, str(path))


def make_manifest(tmp_path: Path, *, overlap: bool = False) -> tuple[Path, Path]:
    shape = (12, 12, 12)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    source_root = tmp_path / "source"
    phases = []
    for phase_index in range(2):
        phase_id = f"phase_{phase_index:02d}"
        ct_path = source_root / "ct" / f"{phase_id}.nii.gz"
        ct = np.zeros(shape, dtype=np.int16)
        ct[2:10, 2:10, 2:10] = 100
        save_nifti(ct_path, ct, affine)

        masks: dict[str, np.ndarray] = {}
        left_atrium = np.zeros(shape, dtype=np.uint8)
        left_atrium[3:5, 3:5, 3:5] = 1
        left_ventricle = np.zeros(shape, dtype=np.uint8)
        left_ventricle[6 : 8 + phase_index, 3:5, 3:5] = 1
        right_atrium = np.zeros(shape, dtype=np.uint8)
        right_atrium[3:5, 7:9, 3:5] = 1
        right_ventricle = np.zeros(shape, dtype=np.uint8)
        right_ventricle[6:8, 7:9, 3:5] = 1
        if overlap:
            left_ventricle[3:5, 3:5, 3:5] = 1
        masks.update(
            {
                "left_atrium_blood": left_atrium,
                "left_ventricle_blood": left_ventricle,
                "right_atrium_blood": right_atrium,
                "right_ventricle_blood": right_ventricle,
            }
        )
        chamber_paths = {}
        for chamber_id, mask in masks.items():
            path = source_root / "segmentation" / phase_id / f"{chamber_id}.nii.gz"
            save_nifti(path, mask, affine)
            chamber_paths[chamber_id] = str(path.relative_to(tmp_path).as_posix())

        total = np.zeros(shape, dtype=np.uint8)
        total[2:10, 2:10, 2:10] = 1
        # Deliberately leave one chamber voxel outside total to test QC-only
        # containment; the raw union must not be clipped.
        if phase_index == 0:
            total[3, 3, 3] = 0
        total_path = source_root / "segmentation" / phase_id / "total_heart.nii.gz"
        save_nifti(total_path, total, affine)

        phases.append(
            {
                "phase_id": phase_id,
                "ct_path": str(ct_path.relative_to(tmp_path).as_posix()),
                "chamber_masks": chamber_paths,
                "total_heart_path": str(total_path.relative_to(tmp_path).as_posix()),
            }
        )

    segmentation_manifest = tmp_path / "source" / "segmentation_manifest.json"
    segmentation_manifest.write_text(
        json.dumps(
            {
                "backend": "TotalSegmentator",
                "backend_version": "2.18.0",
                "models": {"total": "total", "chambers": "heartchambers_highres"},
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    review_manifest = tmp_path / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subject_id": "synthetic_subject",
                "analysis_plan_locked_before_review": True,
                "review_priority_thresholds": {
                    "outside_total_fraction_alert": 0.05,
                    "adjacent_relative_change_alert": 0.25,
                    "components_alert": 1,
                },
                "phases": phases,
            }
        ),
        encoding="utf-8",
    )
    return review_manifest, segmentation_manifest


def test_prepare_writes_raw_union_qc_and_disjoint_labelmap(tmp_path):
    manifest_path, segmentation_manifest = make_manifest(tmp_path)
    output_dir = tmp_path / "external_review"

    report = module.prepare_review(
        manifest_path,
        output_dir,
        segmentation_manifest_path=segmentation_manifest,
    )

    assert report["status"] == "automatic_blood_candidates_pending_manual_review"
    assert report["segmentation_provenance"]["backend_version"] == "2.18.0"
    assert len(report["phases"]) == 2
    phase0 = report["phases"][0]
    assert phase0["labelmap"]["status"] == "written_pairwise_disjoint_raw_masks"
    assert phase0["union"]["outside_total_heart_voxels"] == 1
    assert phase0["union"]["containment_fraction_in_total_heart"] < 1.0
    assert phase0["chambers"]["left_atrium_blood"]["outside_total_heart_voxels"] == 1
    assert phase0["chambers"]["left_atrium_blood"]["stats"]["volume_ml"] == pytest.approx(0.016)
    assert phase0["chambers"]["left_atrium_blood"]["adjacent_relative_change_fraction"] is None
    assert report["phases"][1]["chambers"]["left_ventricle_blood"]["previous_phase_id"] == "phase_00"

    labelmap = np.asanyarray(
        nib.load(str(output_dir / "phase_00" / "raw_blood_chambers_labelmap.nii.gz")).dataobj
    )
    assert set(np.unique(labelmap)) == {0, 1, 2, 3, 4}
    assert (output_dir / "phase_00" / "raw_left_atrium_blood.nii.gz").is_file()
    assert (output_dir / "phase_00" / "raw_blood_pool_four_chambers_union.nii.gz").is_file()
    assert (output_dir / "phase_00" / "qc_blood_inside_total_heart.nii.gz").is_file()
    assert (output_dir / "phase_00" / "qc_blood_outside_total_heart.nii.gz").is_file()
    assert (output_dir / "blood_phase_measurements.csv").is_file()
    assert (output_dir / "blood_review_queue.csv").is_file()
    assert (output_dir / "blood_review_report.json").is_file()

    encoded = json.dumps(report, ensure_ascii=False)
    assert str(tmp_path) not in encoded
    queue = report["review_queue"]
    assert len(queue) == 10  # four chambers + union for each of two phases
    assert all(item["requires_manual_review"] for item in queue)


def test_overlap_blocks_labelmap_without_priority_resolution(tmp_path):
    manifest_path, segmentation_manifest = make_manifest(tmp_path, overlap=True)
    output_dir = tmp_path / "external_review"

    report = module.prepare_review(
        manifest_path,
        output_dir,
        segmentation_manifest_path=segmentation_manifest,
    )
    phase0 = report["phases"][0]
    assert phase0["labelmap"]["status"] == "not_written_overlap_requires_manual_disambiguation"
    assert phase0["pairwise_overlaps"]["left_atrium_blood__left_ventricle_blood"]["overlap_voxels"] == 8
    assert not (output_dir / "phase_00" / "raw_blood_chambers_labelmap.nii.gz").exists()
    assert "overlap_blocking_labelmap" in phase0["chambers"]["left_atrium_blood"]["status"]
    assert "overlap_blocking_labelmap" in phase0["union"]["status"]
    assert "pairwise_overlap" in report["review_queue"][0]["flags"]


def test_strict_geometry_and_external_output_and_dry_run(tmp_path):
    manifest_path, segmentation_manifest = make_manifest(tmp_path)
    phase_mask = tmp_path / "source" / "segmentation" / "phase_00" / "left_atrium_blood.nii.gz"
    data = np.asanyarray(nib.load(str(phase_mask)).dataobj)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    save_nifti(phase_mask, data, affine, forms=False)
    with pytest.raises(ValueError, match="qform and sform codes are unset"):
        module.prepare_review(
            manifest_path,
            tmp_path / "external_review",
            segmentation_manifest_path=segmentation_manifest,
        )

    # Restore a valid source and verify dry-run performs no writes.
    save_nifti(phase_mask, data, affine)
    dry_output = tmp_path / "dry_run_output"
    report = module.prepare_review(
        manifest_path,
        dry_output,
        segmentation_manifest_path=segmentation_manifest,
        dry_run=True,
    )
    assert report["runtime"]["dry_run"] is True
    assert not dry_output.exists()

    with pytest.raises(ValueError, match="outside the repository"):
        module._assert_external_output_dir(MODULE_PATH.parent / "derived")
