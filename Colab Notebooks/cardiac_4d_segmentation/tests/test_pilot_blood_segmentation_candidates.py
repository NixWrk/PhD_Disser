from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "pilot_blood_segmentation_candidates.py"
SPEC = importlib.util.spec_from_file_location("pilot_blood_segmentation_candidates", MODULE_PATH)
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


def build_manifest(
    tmp_path: Path,
    *,
    subjects: tuple[str, ...] = ("Adam", "Adam"),
    with_reference: bool = False,
    nn_enabled: bool = True,
) -> Path:
    shape = (14, 14, 14)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    phases = []
    for phase_index, subject in enumerate(subjects):
        phase_id = f"phase_{phase_index:02d}"
        ct = np.zeros(shape, dtype=np.int16)
        ct[2:12, 2:12, 2:12] = 100
        ct_path = tmp_path / "source" / subject / f"ct_{phase_id}.nii.gz"
        save_nifti(ct_path, ct, affine)
        raw_masks = {}
        references = {}
        starts = ((2, 2, 2), (8, 2, 2), (2, 8, 2), (8, 8, 2))
        for chamber, (x, y, z) in zip(module.CHAMBERS, starts):
            mask = np.zeros(shape, dtype=np.uint8)
            mask[x : x + 3, y : y + 3, z : z + 3] = 1
            # A distant one-voxel component exercises the locked cleanup rule.
            if chamber == module.CHAMBERS[0]:
                mask[12, 12, 12] = 1
            raw_path = tmp_path / "source" / subject / f"raw_{phase_id}_{chamber}.nii.gz"
            save_nifti(raw_path, mask, affine)
            raw_masks[chamber] = {"path": raw_path.relative_to(tmp_path).as_posix()}
            if with_reference:
                reference = mask.copy()
                if chamber == module.CHAMBERS[0]:
                    reference[12, 12, 12] = 0
                ref_path = tmp_path / "source" / subject / f"ref_{phase_id}_{chamber}.nii.gz"
                save_nifti(ref_path, reference, affine)
                references[chamber] = {
                    "path": ref_path.relative_to(tmp_path).as_posix(),
                    "reference_id": f"manual_{subject}_{phase_id}_{chamber}",
                    "reference_kind": "manual_inobitec",
                    "independent_from_candidates": True,
                    "blinded_to_candidates": True,
                }
        phase = {
            "subject_id": subject,
            "phase_id": phase_id,
            "ct_path": ct_path.relative_to(tmp_path).as_posix(),
            "raw_masks": raw_masks,
        }
        if with_reference:
            phase["manual_reference"] = references
        phases.append(phase)
    manifest = {
        "schema_version": 1,
        "analysis_plan_locked_before_evaluation": True,
        "surface_tolerance_mm": 1.0,
        "surface_tolerance_locked_before_evaluation": True,
        "geometry_mm_confirmed": False,
        "required_phase_count": len(phases),
        "phase_selection": {
            "locked_before_evaluation": True,
            "phase_ids": [phase["phase_id"] for phase in phases],
        },
        "inference_scope": "pilot_descriptive_only",
        "raw_source": {
            "software_name": "TotalSegmentator",
            "software_version": "2.18.0",
            "model_or_method": "heartchambers_highres",
        },
        "conservative_cleanup": {
            "small_component_threshold_mm3": 50.0,
            "minimum_distance_mm": 1.0,
        },
        "nninteractive": {
            "enabled": nn_enabled,
            "model_dir": "not_used_by_fake_runner",
            "model_id": "nnInteractive_v1.0",
            "software_version": "2.5.1",
        },
        "bootstrap": {"iterations": 100, "seed": 42, "confidence_level": 0.95},
        "qc_png": False,
        "phases": phases,
    }
    manifest_path = tmp_path / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, image, spacing_mm, *, mode, initial_seg, positive_points, negative_points):
        self.calls.append({
            "mode": mode,
            "initial_seg": initial_seg,
            "positive_points": list(positive_points),
            "negative_points": list(negative_points),
        })
        result = np.zeros_like(image, dtype=bool)
        if mode == "prompt_from_ts_geometry":
            point = positive_points[0]
            result[tuple(point)] = True
        else:
            assert initial_seg is not None
            result = np.asarray(initial_seg, dtype=bool).copy()
        return result, {"fake_runner": True}


def test_conservative_cleanup_single_component_fast_path_skips_edt(monkeypatch):
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[3:7, 3:7, 3:7] = True
    image = nib.Nifti1Image(np.zeros(mask.shape, dtype=np.float32), np.eye(4))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("distance_transform_edt must not run for one component")

    monkeypatch.setattr(module.ndimage, "distance_transform_edt", fail_if_called)
    cleaned, details = module._conservative_cleanup(mask, image)

    assert np.array_equal(cleaned, mask)
    assert details["components_before"] == 1
    assert details["components_after"] == 1
    assert details["changed_voxels"] == 0
    assert details["fast_path"] == "single_component_no_distance_query"


def test_conservative_cleanup_ckdtree_matches_edt_on_anisotropic_grid():
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[5:11, 5:11, 5:11] = True
    # A detached one-voxel component 0.8 mm from the largest component: it is
    # small but must be retained because the threshold is strict (> 1 mm).
    mask[12, 7, 7] = True
    # A detached one-voxel component 1.2 mm from the largest component: remove.
    mask[14, 7, 7] = True
    # A detached component above 50 mm3: never query or remove it.
    mask[20:25, 20:25, 20:25] = True
    affine = np.diag([0.4, 1.3, 2.2, 1.0])
    image = nib.Nifti1Image(np.zeros(mask.shape, dtype=np.float32), affine)

    cleaned, details = module._conservative_cleanup(mask, image)

    structure = module.ndimage.generate_binary_structure(3, 3)
    labels, count = module.ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    largest_label = int(np.argmax(sizes[1:]) + 1)
    largest = labels == largest_label
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine))
    edt = module.ndimage.distance_transform_edt(~largest, sampling=spacing)
    expected = mask.copy()
    expected_removed = []
    voxel_volume_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    for label in range(1, count + 1):
        if label == largest_label:
            continue
        component = labels == label
        volume_mm3 = float(sizes[label]) * voxel_volume_mm3
        minimum_distance = float(np.min(edt[component]))
        if volume_mm3 < 50.0 and minimum_distance > 1.0:
            expected[component] = False
            expected_removed.append(label)

    assert np.array_equal(cleaned, expected)
    assert [entry["label"] for entry in details["removed_components"]] == expected_removed
    assert details["distance_method"] == "cKDTree_physical_voxel_centers"
    assert details["distance_query_components"] == 2


def test_pilot_generates_all_named_candidates_and_marks_agreement_only(tmp_path):
    manifest_path = build_manifest(tmp_path)
    runner = FakeRunner()
    output_dir = tmp_path / "external_pilot"
    report = module.run_pilot(
        manifest_path,
        output_dir,
        dry_run=False,
        run_nninteractive=True,
        nninteractive_runner=runner,
    )

    assert report["status"] == "no_accuracy_inference"
    assert report["analysis_mode"] == "technical_only"
    assert report["comparison_status"] == "agreement_only"
    assert report["phase_count_observed"] == 2
    assert len(report["technical_rows"]) == 2 * 4 * 4
    assert len(report["agreement_rows"]) == 2 * 4 * 6
    assert len(runner.calls) == 2 * 4 * 2
    prompt_calls = [call for call in runner.calls if call["mode"] == "prompt_from_ts_geometry"]
    initial_calls = [call for call in runner.calls if call["mode"] == "ts_initial_refinement"]
    assert prompt_calls and all(call["initial_seg"] is None for call in prompt_calls)
    assert initial_calls and all(call["initial_seg"] is not None for call in initial_calls)
    cleanup = next(
        candidate for candidate in report["phases"][0]["candidates"][module.CHAMBERS[0]]
        if candidate["candidate_id"] == "conservative_cleanup"
    )
    assert cleanup["cleanup_changed_voxels"] == 1
    assert cleanup["cleanup_details"]["connectivity"] == 26
    assert cleanup["cleanup_details"]["removed_components"]
    assert (output_dir / "pilot_report.json").is_file()
    assert (output_dir / "pilot_candidate_agreement.csv").is_file()
    assert not any(str(tmp_path) in json.dumps(value, ensure_ascii=False) for value in report.values())


def test_pilot_dry_run_writes_nothing_and_requires_explicit_nn_flag(tmp_path, monkeypatch):
    manifest_path = build_manifest(tmp_path, nn_enabled=True)
    output_dir = tmp_path / "external_dry_run"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run must not execute cleanup")

    monkeypatch.setattr(module, "_conservative_cleanup", fail_if_called)
    report = module.run_pilot(manifest_path, output_dir, dry_run=True, run_nninteractive=False)

    assert report["status"] == "dry_run_no_accuracy_inference"
    assert report["comparison_status"] == "not_evaluated"
    assert not output_dir.exists()
    statuses = {
        candidate["status"]
        for candidate in report["phases"][0]["candidates"][module.CHAMBERS[0]]
        if candidate["candidate_id"].startswith("nninteractive")
    }
    assert statuses == {"not_run_requires_explicit_flag"}
    non_nn_statuses = {
        candidate["status"]
        for candidate in report["phases"][0]["candidates"][module.CHAMBERS[0]]
        if not candidate["candidate_id"].startswith("nninteractive")
    }
    assert non_nn_statuses == {"not_run_dry_run"}
    assert report["technical_rows"] == []


def test_pilot_accuracy_uses_subject_cluster_bootstrap_and_locks_plan(tmp_path):
    manifest_path = build_manifest(tmp_path, subjects=("Adam", "Nix", "Georg"), with_reference=True, nn_enabled=False)
    report = module.run_pilot(manifest_path, tmp_path / "external_accuracy", dry_run=False)

    assert report["analysis_mode"] == "accuracy"
    assert report["accuracy_inference_allowed"] is True
    assert len(report["accuracy_rows"]) == 3 * 4 * 2
    assert report["bootstrap"]
    assert all(row["cluster_unit"] == "subject" for row in report["bootstrap"])
    assert all(row["unique_subjects"] == 3 for row in report["bootstrap"])
    assert all(row["status"] == "pilot_cluster_bootstrap_interval_not_population_ci" for row in report["bootstrap"])
    raw_row = next(row for row in report["accuracy_rows"] if row["candidate_id"] == "raw" and row["structure"] == module.CHAMBERS[1])
    assert raw_row["dice"] == pytest.approx(1.0)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_plan_locked_before_evaluation"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="analysis_plan_locked_before_evaluation"):
        module.run_pilot(manifest_path, tmp_path / "external_locked_error", dry_run=True)
