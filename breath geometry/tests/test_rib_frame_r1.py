import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.rib_frame_r1 import (
    PhaseRibs,
    Rib,
    RibPreflightGates,
    detect_ribs,
    evaluate_preflight,
    load_rib_frame_config,
    write_preflight,
)

IntArray = npt.NDArray[np.int_]

SHAPE = (96, 96, 120)
SPACING = (1.0, 1.0, 1.0)


def rib_phantom(rib_z: tuple[float, ...] = (30.0, 45.0, 60.0, 75.0, 90.0)) -> IntArray:
    """A body with a spine and thin arc-shaped ribs that slope anteriorly.

    Two properties are copied from the real anatomy on purpose.  The arcs make
    the bounding box nearly cubic, which is what defeated bounding-box
    elongation on real data.  The anterior slope makes the box span z as well,
    which is why a real rib fills so little of it.
    """
    grid = np.indices(SHAPE, dtype=np.float64)
    x, y, z = grid[0], grid[1], grid[2]
    radius = np.sqrt((x - 48.0) ** 2 + (y - 48.0) ** 2)
    image = np.full(SHAPE, -1000, dtype=np.int_)
    image[radius <= 40.0] = 50
    image[np.sqrt((x - 48.0) ** 2 + (y - 20.0) ** 2) <= 6.0] = 600  # spine

    for centre in rib_z:
        # The rib descends as it runs forward, exactly as ribs do supine.
        surface = centre - 0.35 * (y - 24.0)
        shell = (radius > 30.0) & (radius <= 33.0) & (np.abs(z - surface) <= 1.5)
        image[shell & (y > 24.0)] = 600
    return image


def config_file(path: Path, subjects: tuple[str, ...] = ("copd1",)) -> Path:
    config = {
        "frame_version": "test-frame",
        "dataset_id": "test_dataset",
        "subjects": list(subjects),
        "phases": ["fixed", "moving"],
        "rib_identification": {
            "bone_hu_min": 200,
            "midline_slab_half_width_mm": 12.0,
            "min_component_voxels": 200,
            "max_fill_fraction": 0.2,
        },
        "rib_preflight_gates": {
            "ribs_per_side_min": 4,
            "ribs_per_side_max": 8,
            "count_must_match_between_phases": True,
            "max_gap_over_median_gap": 1.6,
            "left_right_z_tolerance_mm": 15.0,
        },
        "preflight_policy": {"delta_blocked_until_preflight_passes": True},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_must_block_the_delta_map(tmp_path: Path) -> None:
    path = config_file(tmp_path / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["preflight_policy"]["delta_blocked_until_preflight_passes"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="block the delta map"):
        load_rib_frame_config(path)


def test_config_must_use_fill_fraction_not_elongation(tmp_path: Path) -> None:
    path = config_file(tmp_path / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["rib_identification"]["max_fill_fraction"]
    raw["rib_identification"]["min_elongation"] = 3.0
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="max_fill_fraction"):
        load_rib_frame_config(path)


def test_detection_finds_every_rib_on_both_sides(tmp_path: Path) -> None:
    config = load_rib_frame_config(config_file(tmp_path / "config.json"))
    found = detect_ribs(
        rib_phantom(), SPACING, config.detection, subject_id="phantom", phase="fixed"
    )
    assert found.left_count == 5
    assert found.right_count == 5
    assert found.left_right_max_z_difference_mm == pytest.approx(0.0, abs=1e-6)
    assert found.left_gap_ratio == pytest.approx(1.0, abs=0.05)
    orders = sorted(rib.order for rib in found.ribs if rib.side == "left")
    assert orders == [0, 1, 2, 3, 4]


def _phase(
    subject: str,
    phase: str,
    left: int,
    right: int,
    gap_ratio: float = 1.1,
    symmetry: float = 2.0,
) -> PhaseRibs:
    return PhaseRibs(
        subject_id=subject,
        phase=phase,
        left_count=left,
        right_count=right,
        left_gap_ratio=gap_ratio,
        right_gap_ratio=gap_ratio,
        left_right_max_z_difference_mm=symmetry,
        median_arc_extent_mm=200.0,
        ribs=(Rib("left", 0, 1000, 0.01, 10.0, 200.0),),
    )


def test_preflight_passes_a_consistent_subject() -> None:
    gates = RibPreflightGates(ribs_per_side_min=4, ribs_per_side_max=8)
    verdict = evaluate_preflight(
        (_phase("s", "fixed", 5, 5), _phase("s", "moving", 5, 5)), gates
    )
    assert verdict.passes
    assert verdict.failure_reasons == ""


def test_preflight_catches_each_miscount_mode() -> None:
    gates = RibPreflightGates(ribs_per_side_min=4, ribs_per_side_max=8)

    # One rib appears in the other phase: the dangerous mode, a shifted map.
    drifted = evaluate_preflight(
        (_phase("s", "fixed", 5, 5), _phase("s", "moving", 6, 5)), gates
    )
    assert not drifted.passes
    assert "rib_count_differs_between_phases" in drifted.failure_reasons

    # A missed rib leaves a double-width gap.
    gapped = evaluate_preflight(
        (_phase("s", "fixed", 5, 5, gap_ratio=2.1), _phase("s", "moving", 5, 5)), gates
    )
    assert not gapped.passes
    assert "irregular_rib_spacing" in gapped.failure_reasons

    # Sides indexed against each other.
    skewed = evaluate_preflight(
        (_phase("s", "fixed", 5, 5, symmetry=40.0), _phase("s", "moving", 5, 5)), gates
    )
    assert not skewed.passes
    assert "left_right_asymmetry" in skewed.failure_reasons

    # Far too few ribs to be a thorax.
    sparse = evaluate_preflight(
        (_phase("s", "fixed", 2, 2), _phase("s", "moving", 2, 2)), gates
    )
    assert not sparse.passes
    assert "rib_count_out_of_range" in sparse.failure_reasons


def test_written_summary_records_that_no_delta_was_produced(tmp_path: Path) -> None:
    config_path = config_file(tmp_path / "config.json")
    config = load_rib_frame_config(config_path)
    phases = (_phase("s", "fixed", 5, 5), _phase("s", "moving", 6, 5))
    verdicts = (evaluate_preflight(phases, config.gates),)

    manifest_path = write_preflight(
        tmp_path / "out", config_path, config, phases, verdicts
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["sha256"]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["delta_map_produced"] is False
    assert summary["subjects_passed"] == 0
    assert "blocks any delta map" in summary["delta_block_reason"]
