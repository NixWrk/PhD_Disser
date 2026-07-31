import json
from pathlib import Path

import numpy as np
import pytest
from test_uncertainty_w1 import SPACING, thorax_phantom

from breathgeom.measure.segmentation import body_mask_3d, lung_mask_3d
from breathgeom.measure.wall import WallParams
from breathgeom.spatial_map_w2 import (
    MapEvaluation,
    MapResolution,
    bin_indices,
    bootstrap_median_se,
    build_frame,
    decide_map,
    evaluate_phase_map,
    load_spatial_map_config,
    select_configuration,
    write_spatial_map,
)


def map_config_file(path: Path) -> Path:
    config = {
        "map_version": "test-map",
        "dataset_id": "test_dataset",
        "subject_id": "test_subject",
        "phases": ["fixed"],
        "cross_phase_comparison": {"performed": False, "reason": "test"},
        "measurand": {"name": "soft_tissue_mm"},
        "sampling_density_variants_mm": [8.0, 5.0],
        "map_resolution_variants": [
            {"angular_sectors": 4, "axial_levels": 2},
            {"angular_sectors": 8, "axial_levels": 3},
        ],
        "per_bin_noise": {
            "primary": "bootstrap standard error of the per-bin median",
            "bootstrap_resamples": 80,
            "bootstrap_seed": 20260731,
        },
        "occupancy": {"min_profiles_per_bin": 5, "min_occupied_fraction": 0.5},
        "decision_thresholds": {
            "feasible_max_mm": 1.0,
            "marginal_max_mm": 2.5,
            "target_resolvable_effect_mm": 3.0,
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_config_must_measure_the_merged_soft_tissue(tmp_path: Path) -> None:
    path = map_config_file(tmp_path / "config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    config["measurand"]["name"] = "muscle_mm"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="D-044"):
        load_spatial_map_config(path)


def test_config_must_declare_that_no_phases_are_compared(tmp_path: Path) -> None:
    path = map_config_file(tmp_path / "config.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    config["cross_phase_comparison"]["performed"] = True
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="cross_phase_comparison"):
        load_spatial_map_config(path)


def test_frame_places_anterior_at_angle_zero() -> None:
    image = thorax_phantom()
    body = body_mask_3d(image, WallParams())
    lung = lung_mask_3d(image, WallParams())
    frame = build_frame(body, lung, SPACING)

    # The phantom is a centred cylinder, so the origin sits at its centre.
    middle = image.shape[2] // 2
    assert frame.centroid_x[middle] == pytest.approx(48.0, abs=1.0)
    assert frame.centroid_y[middle] == pytest.approx(48.0, abs=1.0)

    class Point:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.skin_x_mm, self.skin_y_mm, self.skin_z_mm = x, y, z

    z = float(middle) * SPACING[2]
    anterior = Point(48.0, 90.0, z)
    right = Point(90.0, 48.0, z)
    posterior = Point(48.0, 6.0, z)
    left = Point(6.0, 48.0, z)
    angle, _ = frame.coordinates((anterior, right, posterior, left))  # type: ignore[arg-type]
    assert angle[0] == pytest.approx(0.0, abs=1e-6)
    assert angle[1] == pytest.approx(0.25, abs=1e-6)
    assert angle[2] == pytest.approx(0.5, abs=1e-6)
    assert angle[3] == pytest.approx(0.75, abs=1e-6)


def test_bin_indices_wrap_angle_and_reject_outside_axial_span() -> None:
    resolution = MapResolution(angular_sectors=4, axial_levels=2)
    angle = np.asarray([0.0, 0.26, 0.99, 0.5])
    axial = np.asarray([0.1, 0.6, 0.9, -0.2])
    flat = bin_indices(angle, axial, resolution)
    assert flat[0] == 0
    assert flat[1] == 4 + 1
    assert flat[2] == 4 + 3
    assert flat[3] == -1


def test_bootstrap_median_se_is_unbiased_and_shrinks_with_sample_size() -> None:
    # The estimate for one bin is noisy -- it depends on the local density of
    # that particular sample near its median -- so the property that holds is
    # unbiasedness across bins, not accuracy on any single one.  Taking p95
    # across bins therefore errs conservative, never optimistic.
    generator = np.random.default_rng(11)
    sigma = 6.0
    for size, tolerance in ((200, 0.15), (2000, 0.10)):
        estimates = [
            bootstrap_median_se(
                generator.normal(30.0, sigma, size=size), 600, np.random.default_rng(index)
            )
            for index in range(40)
        ]
        asymptotic = 1.2533 * sigma / np.sqrt(size)
        assert float(np.mean(estimates)) == pytest.approx(asymptotic, rel=tolerance)

    small = float(
        np.mean(
            [
                bootstrap_median_se(
                    generator.normal(30.0, sigma, size=50), 600, np.random.default_rng(i)
                )
                for i in range(20)
            ]
        )
    )
    large = float(
        np.mean(
            [
                bootstrap_median_se(
                    generator.normal(30.0, sigma, size=2000), 600, np.random.default_rng(i)
                )
                for i in range(20)
            ]
        )
    )
    assert large < small
    assert np.isnan(bootstrap_median_se(np.asarray([1.0]), 10, np.random.default_rng(1)))


def _evaluation(density: float, resolution: str, qualifies: bool) -> MapEvaluation:
    return MapEvaluation(
        phase="fixed",
        sampling_density_mm=density,
        resolution=resolution,
        valid_profiles=100,
        mapped_profiles=100,
        occupied_bins=8,
        total_bins=8,
        occupied_fraction=1.0,
        p95_bootstrap_se_mm=0.5 if qualifies else 5.0,
        max_bootstrap_se_mm=1.0,
        median_bin_count=20.0,
        qualifies=qualifies,
    )


def test_selection_takes_the_finest_resolution_then_the_coarsest_density(
    tmp_path: Path,
) -> None:
    config = load_spatial_map_config(map_config_file(tmp_path / "config.json"))
    evaluations = (
        _evaluation(8.0, "4x2", True),
        _evaluation(5.0, "4x2", True),
        _evaluation(8.0, "8x3", True),
        _evaluation(5.0, "8x3", True),
    )
    # 8x3 has 24 bins against 8, so it wins on resolution; among the densities
    # that support it the coarsest is preferred because it is cheaper.
    assert select_configuration(evaluations, config) == (8.0, "8x3")

    coarse_only = (
        _evaluation(8.0, "4x2", True),
        _evaluation(5.0, "8x3", True),
        _evaluation(8.0, "8x3", False),
    )
    assert select_configuration(coarse_only, config) == (5.0, "8x3")


def test_no_selection_when_nothing_qualifies(tmp_path: Path) -> None:
    config = load_spatial_map_config(map_config_file(tmp_path / "config.json"))
    evaluations = (_evaluation(8.0, "4x2", False), _evaluation(5.0, "8x3", False))
    verdict = decide_map(evaluations, config)
    assert not verdict.selection_made
    assert verdict.selected_resolution is None
    assert any("no resolution is selected" in note for note in verdict.notes)


def test_selection_requires_every_phase_to_qualify(tmp_path: Path) -> None:
    path = map_config_file(tmp_path / "config.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["phases"] = ["fixed", "moving"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_spatial_map_config(path)

    one_phase_only = (
        _evaluation(8.0, "8x3", True),
        MapEvaluation(**{**vars(_evaluation(8.0, "8x3", False)), "phase": "moving"}),
    )
    assert select_configuration(one_phase_only, config) is None


def test_scan_runs_on_the_phantom_and_noise_falls_with_density(tmp_path: Path) -> None:
    config = load_spatial_map_config(map_config_file(tmp_path / "config.json"))
    evaluations, statistics = evaluate_phase_map(
        thorax_phantom(), SPACING, config, "fixed"
    )

    assert len(evaluations) == 4
    by_key = {(item.sampling_density_mm, item.resolution): item for item in evaluations}
    # A denser surface sampling can only add profiles, never remove them.
    assert by_key[(5.0, "4x2")].valid_profiles > by_key[(8.0, "4x2")].valid_profiles
    # A finer map spreads the same profiles over more bins.
    assert by_key[(8.0, "8x3")].total_bins > by_key[(8.0, "4x2")].total_bins
    assert all(item.bootstrap_se_mm >= 0.0 for item in statistics[(8.0, "4x2")])

    verdict = decide_map(evaluations, config)
    manifest_path = write_spatial_map(
        tmp_path / "out",
        tmp_path / "config.json",
        config,
        evaluations,
        {("fixed", density, label): rows for (density, label), rows in statistics.items()},
        verdict,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["sha256"]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["measurand"] == "soft_tissue_mm"
    assert summary["cross_phase_comparison"]["performed"] is False
