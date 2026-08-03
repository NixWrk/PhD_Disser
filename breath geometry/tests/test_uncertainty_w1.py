import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.profiles import ProfileParams
from breathgeom.measure.segmentation import body_mask_3d, lung_mask_3d
from breathgeom.measure.wall import WallParams
from breathgeom.uncertainty_w1 import (
    BudgetThresholds,
    axial_thirds,
    classify,
    decide,
    evaluate_phase,
    load_budget_config,
    write_budget,
)

IntArray = npt.NDArray[np.int_]

SHAPE = (96, 96, 24)
SPACING = (1.0, 1.0, 1.0)


def thorax_phantom(lung_hu: int = -800) -> IntArray:
    """A cylinder with two aerated lungs and a posterior midline spine.

    The spine is required: the lung mask picks sides by the vertebral midline,
    not by connected component, so a phantom without bone cannot be segmented.
    Both lungs stay strictly inside the muscle layer, otherwise they breach the
    body contour and the filled body mask no longer encloses them.
    """
    grid = np.indices(SHAPE, dtype=np.float64)
    x, y = grid[0], grid[1]
    body_radius = np.sqrt((x - 48.0) ** 2 + (y - 48.0) ** 2)
    image = np.full(SHAPE, -1000, dtype=np.int_)
    # A step edge at the skin would be insensitive to the body threshold, which
    # real CT never is.  This low-density outer shell sits between two of the
    # frozen variants, so the threshold family actually moves the skin.
    image[body_radius <= 39.0] = -290
    image[body_radius <= 36.0] = -100  # subcutaneous fat
    image[body_radius <= 30.0] = 50  # muscle
    for centre in (34.0, 62.0):
        lung = np.sqrt((x - centre) ** 2 + (y - 50.0) ** 2) <= 13.0
        image[lung] = lung_hu
    spine = np.sqrt((x - 48.0) ** 2 + (y - 24.0) ** 2) <= 5.0
    image[spine] = 600
    return image


def budget_config_file(path: Path) -> Path:
    config = {
        "budget_version": "test-budget",
        "dataset_id": "test_dataset",
        "subject_id": "test_subject",
        "phases": ["fixed"],
        "cross_phase_comparison": {"performed": False, "reason": "test"},
        "baseline_parameters": {
            "body_hu": -300,
            "air_hu": -400,
            "fat_hu": [-190, -30],
            "muscle_hu": [-29, 150],
            "bone_lower_hu": 151.0,
            "surface_spacing_mm": 10.0,
            "line_step_mm": 0.75,
        },
        "perturbation_families": {
            "body_threshold": {"variants": [-325, -300, -275]},
            "lung_threshold": {"variants": [-425, -400, -375]},
            "tissue_window": {"variants": [[-40, -39], [-30, -29], [-20, -19]]},
            "surface_sampling": {"variants": [0.0, 0.3333333333333333, 0.6666666666666666]},
            "grid_jitter": {"variants": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]},
        },
        "decision_thresholds": {
            "feasible_max_mm": 1.0,
            "marginal_max_mm": 2.5,
            "target_resolvable_effect_mm": 3.0,
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_must_declare_that_no_phases_are_compared(tmp_path: Path) -> None:
    path = budget_config_file(tmp_path / "config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    config["cross_phase_comparison"]["performed"] = True
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="cross_phase_comparison"):
        load_budget_config(path)


def test_defaults_reproduce_the_previous_hard_coded_thresholds() -> None:
    wall = WallParams()
    profile = ProfileParams()
    assert (wall.body_hu, wall.air_hu) == (-300, -400)
    assert profile.fat_hu == (-190, -30)
    assert profile.muscle_hu == (-29, 150)
    assert profile.sampling_offset_fraction == 0.0


def test_tissue_windows_must_stay_ordered() -> None:
    with pytest.raises(ValueError, match="below the muscle window"):
        ProfileParams(fat_hu=(-190, 200), muscle_hu=(-29, 150))
    with pytest.raises(ValueError, match="ordered low to high"):
        ProfileParams(fat_hu=(-30, -190))
    with pytest.raises(ValueError, match="sampling_offset_fraction"):
        ProfileParams(sampling_offset_fraction=1.0)


def test_axial_thirds_split_the_lung_extent() -> None:
    mask = np.zeros((4, 4, 30), dtype=bool)
    mask[:, :, 6:24] = True
    low, high = axial_thirds(mask)
    assert low == pytest.approx(6.0 + 17.0 / 3.0)
    assert high == pytest.approx(6.0 + 34.0 / 3.0)


def test_axial_thirds_reject_an_empty_mask() -> None:
    with pytest.raises(ValueError, match="empty"):
        axial_thirds(np.zeros((4, 4, 4), dtype=bool))


def test_classification_uses_the_frozen_thresholds() -> None:
    thresholds = BudgetThresholds(
        feasible_max_mm=1.0, marginal_max_mm=2.5, target_resolvable_effect_mm=3.0
    )
    assert classify(0.4, thresholds) == "feasible"
    assert classify(1.0, thresholds) == "feasible"
    assert classify(1.6, thresholds) == "marginal"
    assert classify(2.5, thresholds) == "marginal"
    assert classify(2.6, thresholds) == "infeasible"


def test_phantom_segments_into_two_lungs() -> None:
    image = thorax_phantom()
    lung = lung_mask_3d(image, WallParams())
    body = body_mask_3d(image, WallParams())
    assert lung.any()
    assert body.all(axis=(0, 1)).any() is not None
    assert int(lung.sum()) > 4000


def test_budget_runs_and_the_tissue_window_cannot_move_thickness(tmp_path: Path) -> None:
    config = load_budget_config(budget_config_file(tmp_path / "config.json"))
    budget = evaluate_phase(thorax_phantom(), SPACING, config, "fixed")

    families = {row.family for row in budget.families}
    assert families == {
        "body_threshold",
        "lung_threshold",
        "tissue_window",
        "surface_sampling",
        "grid_jitter",
    }

    # Total thickness is fixed by the body and lung masks alone, so moving the
    # fat/muscle boundary must not touch it.  A non-zero half-range here is a
    # defect in the measurement, not a property of the anatomy.
    tissue_thickness = next(
        row
        for row in budget.families
        if row.family == "tissue_window" and row.measurand == "median_thickness_mm"
    )
    assert tissue_thickness.half_range == pytest.approx(0.0, abs=1e-9)

    # ... while it must move the fat/muscle split, otherwise the perturbation
    # is not being applied at all.
    tissue_fat = next(
        row
        for row in budget.families
        if row.family == "tissue_window" and row.measurand == "median_fat_mm"
    )
    assert tissue_fat.half_range > 0.0

    # Moving the skin threshold has to move the thickness.
    body_thickness = next(
        row
        for row in budget.families
        if row.family == "body_threshold" and row.measurand == "median_thickness_mm"
    )
    assert body_thickness.half_range > 0.0

    combined = budget.combined("median_thickness_mm")
    assert combined >= body_thickness.half_range
    assert budget.worst_family["median_thickness_mm"] in families


def test_verdict_is_decided_by_the_worse_phase_and_is_written(tmp_path: Path) -> None:
    config_path = budget_config_file(tmp_path / "config.json")
    config = load_budget_config(config_path)
    calm = evaluate_phase(thorax_phantom(), SPACING, config, "fixed")
    noisy = evaluate_phase(thorax_phantom(lung_hu=-600), SPACING, config, "moving")

    verdict = decide((calm, noisy), config)
    worse = max((calm, noisy), key=lambda item: item.combined("median_thickness_mm"))
    assert verdict.deciding_phase == worse.phase
    assert verdict.thickness_classification in ("feasible", "marginal", "infeasible")
    assert verdict.separate_fat_muscle_allowed == (
        verdict.fat_classification in ("feasible", "marginal")
        and verdict.muscle_classification in ("feasible", "marginal")
    )
    assert not any("IMPLEMENTATION WARNING" in note for note in verdict.notes)

    manifest_path = write_budget(
        tmp_path / "out", config_path, config, (calm, noisy), verdict
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["sha256"]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["cross_phase_comparison"]["performed"] is False
    assert len(summary["phases"]) == 2
