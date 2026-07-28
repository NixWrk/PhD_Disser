import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.shell import ShellParams, measure_shell
from breathgeom.measure.wall import AIR_HU, Side

IntArray = npt.NDArray[np.int16]

ROWS, COLUMNS, SLICES = 160, 160, 24
SPACING = (1.0, 1.0, 2.0)
BODY_CENTRE, BODY_RX, BODY_RY = (80.0, 80.0), 62.0, 48.0
# Two lungs, as in a thorax: the heart pushes the left one away from its wall,
# so the left wall comes out thicker. A single-lung phantom cannot test sides.
RIGHT_LUNG_CENTRE, RIGHT_LUNG_RADIUS = (100.0, 80.0), 25.0
LEFT_LUNG_CENTRE, LEFT_LUNG_RADIUS = (52.0, 80.0), 14.0
# Lateral wall at the widest point: body edge minus lung edge.
WALL_MM = (BODY_CENTRE[0] + BODY_RX) - (RIGHT_LUNG_CENTRE[0] + RIGHT_LUNG_RADIUS)
LEFT_WALL_MM = (LEFT_LUNG_CENTRE[0] - LEFT_LUNG_RADIUS) - (BODY_CENTRE[0] - BODY_RX)
LUNG_RADIUS = RIGHT_LUNG_RADIUS
MUSCLE_HU_FILL = 50
FAT_HU_FILL = -100
PARAMS = ShellParams(min_lung_area_mm2=200.0, min_lung_component_px=50, band_height_mm=1e6)


def torso(fat_band_mm: int = 0) -> IntArray:
    """Elliptical body with two round lungs at unequal wall depths."""
    volume = np.full((ROWS, COLUMNS, SLICES), -1000, dtype=np.int16)
    rows, columns = np.mgrid[0:ROWS, 0:COLUMNS]
    body = (
        ((rows - BODY_CENTRE[0]) / BODY_RX) ** 2 + ((columns - BODY_CENTRE[1]) / BODY_RY) ** 2
    ) <= 1.0
    right = (
        (rows - RIGHT_LUNG_CENTRE[0]) ** 2 + (columns - RIGHT_LUNG_CENTRE[1]) ** 2
        <= RIGHT_LUNG_RADIUS**2
    )
    left = (
        (rows - LEFT_LUNG_CENTRE[0]) ** 2 + (columns - LEFT_LUNG_CENTRE[1]) ** 2
        <= LEFT_LUNG_RADIUS**2
    )
    volume[np.repeat(body[:, :, None], SLICES, axis=2)] = MUSCLE_HU_FILL
    volume[np.repeat((right | left)[:, :, None], SLICES, axis=2)] = AIR_HU - 500
    if fat_band_mm:
        skin_band = body & ~(
            ((rows - BODY_CENTRE[0]) / (BODY_RX - fat_band_mm)) ** 2
            + ((columns - BODY_CENTRE[1]) / (BODY_RY - fat_band_mm)) ** 2
            <= 1.0
        )
        volume[np.repeat(skin_band[:, :, None], SLICES, axis=2)] = FAT_HU_FILL
    return volume


def test_lateral_wall_thickness_is_recovered() -> None:
    result = measure_shell(torso(), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.samples
    depths = result.depth_mm
    assert float(np.percentile(depths, 5)) == pytest.approx(WALL_MM, abs=2.0)


def test_medial_surface_is_not_sampled() -> None:
    """The decisive property: no path may run through the lung itself.

    Sampling the whole lung boundary instead of its lateral silhouette lets the
    medial surface in, and its shortest way out crosses the entire lung, so the
    depth jumps past the lung diameter.
    """
    result = measure_shell(torso(), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.samples
    assert float(result.depth_mm.max()) < WALL_MM + LUNG_RADIUS


def test_each_side_measures_its_own_wall() -> None:
    """The left lung is set further from its wall, so that wall reads thicker."""
    right = measure_shell(torso(), SPACING, side=Side.RIGHT, params=PARAMS)
    left = measure_shell(torso(), SPACING, side=Side.LEFT, params=PARAMS)

    assert right.samples and left.samples
    assert float(np.percentile(right.depth_mm, 5)) == pytest.approx(WALL_MM, abs=2.0)
    assert float(np.percentile(left.depth_mm, 5)) == pytest.approx(LEFT_WALL_MM, abs=2.0)
    assert LEFT_WALL_MM > WALL_MM


def test_composition_splits_fat_and_muscle() -> None:
    result = measure_shell(torso(fat_band_mm=6), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.samples
    assert float(np.median(result.fat_mm)) == pytest.approx(6.0, abs=2.0)
    assert float(np.median(result.muscle_mm)) > 5.0


def test_band_height_limits_the_slices_used() -> None:
    narrow = ShellParams(min_lung_area_mm2=200.0, min_lung_component_px=50,
                         band_height_mm=10.0)

    result = measure_shell(torso(), SPACING, side=Side.RIGHT, params=narrow)

    assert result.slices_used <= 1 + int(10.0 / SPACING[2])
    assert result.slices_used < SLICES


def test_volume_without_lung_reports_nothing() -> None:
    solid = np.full((60, 60, 20), MUSCLE_HU_FILL, dtype=np.int16)

    result = measure_shell(solid, SPACING, params=PARAMS)

    assert result.samples == ()
    assert result.band_first_slice is None
