import json
from pathlib import Path

import pytest
from test_rib_frame_r1 import SHAPE, SPACING, rib_phantom

from breathgeom.rib_frame_r3 import (
    RawRib,
    common_axial_window,
    detect_raw_ribs,
    evaluate_subject_r3,
    load_rib_config_r3,
    relaxed_for_diagnosis,
    summarise_phase,
)


def config_file(path: Path) -> Path:
    config = {
        "frame_version": "test-r3",
        "dataset_id": "test_dataset",
        "subjects": ["s1"],
        "phases": ["fixed", "moving"],
        "rib_identification": {
            "bone_hu_min": 200,
            "midline_slab_half_width_mm": 12.0,
            "min_component_voxels": 200,
            "max_fill_fraction": 0.2,
            "fragment_merge_gap_fraction": 0.4,
            "drop_ribs_touching_volume_edge": False,
            "common_axial_window": True,
        },
        "anchor": {"agreement_max_gap_fraction": 0.5},
        "rib_preflight_gates": {
            "ribs_per_side_min": 3,
            "ribs_per_side_max": 8,
            "count_must_match_between_phases": True,
            "max_gap_over_median_gap": 1.6,
            "left_right_z_tolerance_mm": 15.0,
            "anchor_must_agree": True,
        },
        "preflight_policy": {"delta_blocked_until_preflight_passes": True},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_rejects_the_r2_edge_rule(tmp_path: Path) -> None:
    path = config_file(tmp_path / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rib_identification"]["drop_ribs_touching_volume_edge"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="per phase|edge"):
        load_rib_config_r3(path)


def test_window_is_the_intersection_and_is_shared() -> None:
    phases = {
        "fixed": (
            RawRib("left", 10.0, True),
            RawRib("left", 30.0, False),
            RawRib("left", 90.0, False),
        ),
        "moving": (
            RawRib("left", 50.0, False),
            RawRib("left", 80.0, False),
            RawRib("left", 120.0, True),
        ),
    }
    assert common_axial_window(phases) == (50.0, 80.0)


def test_window_needs_a_fully_imaged_rib_and_an_overlap() -> None:
    with pytest.raises(ValueError, match="fully imaged"):
        common_axial_window({"fixed": (RawRib("left", 10.0, True),)})
    with pytest.raises(ValueError, match="share no axial window"):
        common_axial_window(
            {
                "fixed": (RawRib("left", 10.0, False), RawRib("left", 20.0, False)),
                "moving": (RawRib("left", 80.0, False), RawRib("left", 90.0, False)),
            }
        )


def test_the_same_window_is_applied_to_both_phases(tmp_path: Path) -> None:
    config = load_rib_config_r3(config_file(tmp_path / "config.json"))
    ribs = tuple(
        RawRib(side, z, False)
        for side in ("left", "right")
        for z in (20.0, 40.0, 60.0, 80.0, 100.0)
    )
    # A window that excludes the outermost rib must exclude it on both sides.
    inside = summarise_phase(
        ribs, (30.0, 90.0), config.params, subject_id="s1", phase="fixed"
    )
    assert inside.left_count == 3
    assert inside.right_count == 3


def test_r3_runs_end_to_end_on_the_phantom(tmp_path: Path) -> None:
    config = load_rib_config_r3(config_file(tmp_path / "config.json"))
    phantom = rib_phantom((30.0, 45.0, 60.0, 75.0, 90.0))
    raw = detect_raw_ribs(phantom, SPACING, config.params)
    assert len(raw) == 10  # five ribs, two sides

    phases, verdict = evaluate_subject_r3(
        {"fixed": phantom, "moving": phantom}, SPACING, config, "s1"
    )
    assert len(phases) == 2
    # Identical phases cannot disagree on anything.
    assert verdict.passes
    assert verdict.anchor_agrees


def test_relaxed_gates_are_diagnosis_only() -> None:
    from breathgeom.rib_frame_r2 import RibGatesR2

    strict = RibGatesR2()
    assert strict.count_must_match_between_phases
    assert not relaxed_for_diagnosis(strict).count_must_match_between_phases
    # Every other threshold must survive untouched.
    assert relaxed_for_diagnosis(strict).max_gap_over_median_gap == (
        strict.max_gap_over_median_gap
    )
    assert relaxed_for_diagnosis(strict).left_right_z_tolerance_mm == (
        strict.left_right_z_tolerance_mm
    )
    assert relaxed_for_diagnosis(strict).anchor_agreement_max_gap_fraction == (
        strict.anchor_agreement_max_gap_fraction
    )
    assert SHAPE[2] > 0
