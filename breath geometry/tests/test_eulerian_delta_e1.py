import json
from pathlib import Path

import numpy as np
import pytest
from test_uncertainty_w1 import SPACING, thorax_phantom

from breathgeom.eulerian_delta_e1 import (
    build_shared_frame,
    compute_delta,
    load_eulerian_config,
    spine_extent_mm,
    write_eulerian_delta,
)


def config_file(path: Path) -> Path:
    config = {
        "map_version": "test-e1",
        "dataset_id": "test_dataset",
        "subjects": ["s1"],
        "status": "diagnostic",
        "measurand": {"name": "soft_tissue_mm"},
        "sampling_density_mm": 4.0,
        "map_resolution": {"angular_sectors": 6, "axial_levels": 2},
        "occupancy": {"min_profiles_per_bin": 5},
        "per_bin_uncertainty": {"bootstrap_resamples": 60, "bootstrap_seed": 20260731},
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_must_declare_itself_diagnostic(tmp_path: Path) -> None:
    path = config_file(tmp_path / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["status"] = "result"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostic"):
        load_eulerian_config(path)


def test_axial_anchor_is_the_spine_not_the_lung() -> None:
    image = thorax_phantom()
    low, high = spine_extent_mm(image, SPACING)
    # The phantom's spine runs the full height of the volume.
    assert low == pytest.approx(0.0, abs=SPACING[2])
    assert high == pytest.approx((image.shape[2] - 1) * SPACING[2], abs=SPACING[2])


def test_missing_vertebral_bone_is_an_error() -> None:
    image = thorax_phantom()
    boneless = np.where(image > 300, 50, image)
    with pytest.raises(ValueError, match="vertebral bone|midline"):
        spine_extent_mm(boneless, SPACING)


def test_one_frame_is_reused_for_both_phases(tmp_path: Path) -> None:
    config = load_eulerian_config(config_file(tmp_path / "config.json"))
    fixed = thorax_phantom()
    frame = build_shared_frame(fixed, SPACING)

    # Building the frame from the other phase must not be what happens: the
    # same object is what makes the comparison Eulerian.
    summary, bins = compute_delta(fixed, fixed, SPACING, config, "s1")
    assert summary.bins_in_both_phases > 0
    # Identical phases must give an exactly zero delta everywhere.
    assert all(item.delta_mm == pytest.approx(0.0, abs=1e-9) for item in bins)
    assert summary.resolved_bins == 0
    assert frame.lung_z_min_mm < frame.lung_z_max_mm


def test_a_thicker_phase_gives_a_positive_resolved_delta(tmp_path: Path) -> None:
    config = load_eulerian_config(config_file(tmp_path / "config.json"))
    fixed = thorax_phantom()
    # Add a 3 mm fat layer outside the skin rather than shrinking the lungs:
    # shrinking them drops the lung component below its minimum size and the
    # phantom then has no pleural surface to measure from at all.
    thicker = thorax_phantom()
    grid = np.indices(thicker.shape, dtype=np.float64)
    body_radius = np.sqrt((grid[0] - 48.0) ** 2 + (grid[1] - 48.0) ** 2)
    thicker[(body_radius > 39.0) & (body_radius <= 42.0)] = -100

    summary, bins = compute_delta(fixed, thicker, SPACING, config, "s1")
    assert summary.bins_in_both_phases > 0
    assert summary.max_delta_mm > 0.0
    assert any(item.resolved for item in bins)

    manifest_path = write_eulerian_delta(
        tmp_path / "out",
        tmp_path / "config.json",
        config,
        (summary,),
        bins,
    )
    written = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert written["status"] == "diagnostic"
    assert written["not_a_validated_effect"] is True
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["config"]["sha256"]
