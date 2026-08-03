import json
from pathlib import Path

import numpy as np
import pytest
from test_rib_frame_r1 import SHAPE, SPACING, rib_phantom

from breathgeom.rib_frame_r2 import (
    PhaseRibsR2,
    RibGatesR2,
    detect_ribs_r2,
    evaluate_preflight_r2,
    load_rib_config_r2,
    merge_fragments,
    write_preflight_r2,
)


def config_file(path: Path) -> Path:
    config = {
        "frame_version": "test-r2",
        "dataset_id": "test_dataset",
        "subjects": ["s1"],
        "phases": ["fixed", "moving"],
        "rib_identification": {
            "bone_hu_min": 200,
            "midline_slab_half_width_mm": 12.0,
            "min_component_voxels": 200,
            "max_fill_fraction": 0.2,
            "fragment_merge_gap_fraction": 0.4,
            "drop_ribs_touching_volume_edge": True,
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


def test_config_must_keep_the_anchor_gate(tmp_path: Path) -> None:
    path = config_file(tmp_path / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rib_preflight_gates"]["anchor_must_agree"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="anchor gate"):
        load_rib_config_r2(path)


def test_fragments_of_one_rib_collapse_but_real_ribs_do_not() -> None:
    # Four ribs 20 mm apart, one of which arrived as two fragments 3 mm apart.
    heights = np.asarray([10.0, 13.0, 30.0, 50.0, 70.0])
    merged, collapsed = merge_fragments(heights, 0.4)
    assert collapsed == 1
    assert len(merged) == 4
    assert merged[0] == pytest.approx(11.5)

    clean = np.asarray([10.0, 30.0, 50.0, 70.0])
    untouched, none = merge_fragments(clean, 0.4)
    assert none == 0
    assert np.allclose(untouched, clean)


def test_a_rib_cut_by_the_end_of_the_volume_is_dropped(tmp_path: Path) -> None:
    config = load_rib_config_r2(config_file(tmp_path / "config.json"))
    inside = detect_ribs_r2(
        rib_phantom((30.0, 45.0, 60.0, 75.0, 90.0)),
        SPACING,
        config.params,
        subject_id="s1",
        phase="fixed",
    )
    assert inside.left_count == 5
    assert inside.dropped_at_edge == 0

    # A rib placed at the very top is cut by the volume boundary.
    at_edge = detect_ribs_r2(
        rib_phantom((30.0, 45.0, 60.0, 75.0, 90.0, float(SHAPE[2] - 1))),
        SPACING,
        config.params,
        subject_id="s1",
        phase="fixed",
    )
    assert at_edge.dropped_at_edge > 0
    assert at_edge.left_count == 5


def _phase(
    phase: str,
    left: int,
    right: int,
    anchor: float = 90.0,
    gap: float = 20.0,
    symmetry: float = 2.0,
    ratio: float = 1.1,
) -> PhaseRibsR2:
    return PhaseRibsR2(
        subject_id="s1",
        phase=phase,
        left_count=left,
        right_count=right,
        dropped_at_edge=0,
        merged_fragments=0,
        left_gap_ratio=ratio,
        right_gap_ratio=ratio,
        median_gap_mm=gap,
        left_anchor_z_mm=anchor,
        right_anchor_z_mm=anchor,
        left_right_max_z_difference_mm=symmetry,
    )


def test_anchor_gate_refuses_an_ambiguous_match() -> None:
    gates = RibGatesR2(ribs_per_side_min=3, ribs_per_side_max=8)

    # Anchor moved 4 mm against a 20 mm spacing: unambiguous.
    close = evaluate_preflight_r2(
        (_phase("fixed", 5, 5), _phase("moving", 5, 5, anchor=94.0)), gates
    )
    assert close.passes

    # Anchor moved 14 mm, more than half a rib spacing: the match could be off
    # by a whole rib, so the gate must refuse rather than guess.
    far = evaluate_preflight_r2(
        (_phase("fixed", 5, 5), _phase("moving", 5, 5, anchor=104.0)), gates
    )
    assert not far.passes
    assert "anchor_ambiguous" in far.failure_reasons


def test_every_r1_gate_still_fires() -> None:
    gates = RibGatesR2(ribs_per_side_min=3, ribs_per_side_max=8)
    drifted = evaluate_preflight_r2((_phase("fixed", 5, 5), _phase("moving", 6, 5)), gates)
    assert "rib_count_differs_between_phases" in drifted.failure_reasons
    gapped = evaluate_preflight_r2(
        (_phase("fixed", 5, 5, ratio=2.0), _phase("moving", 5, 5)), gates
    )
    assert "irregular_rib_spacing" in gapped.failure_reasons
    skewed = evaluate_preflight_r2(
        (_phase("fixed", 5, 5, symmetry=40.0), _phase("moving", 5, 5)), gates
    )
    assert "left_right_asymmetry" in skewed.failure_reasons
    sparse = evaluate_preflight_r2((_phase("fixed", 2, 2), _phase("moving", 2, 2)), gates)
    assert "rib_count_out_of_range" in sparse.failure_reasons


def test_summary_counts_failures_and_blocks_the_delta(tmp_path: Path) -> None:
    config_path = config_file(tmp_path / "config.json")
    config = load_rib_config_r2(config_path)
    phases = (_phase("fixed", 5, 5), _phase("moving", 6, 5, anchor=104.0))
    verdicts = (evaluate_preflight_r2(phases, config.gates),)
    manifest_path = write_preflight_r2(
        tmp_path / "out", config_path, config, phases, verdicts
    )
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["delta_map_produced"] is False
    assert summary["failure_counts"]["anchor_ambiguous"] == 1
    assert summary["failure_counts"]["rib_count_differs_between_phases"] == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["config"]["sha256"]
