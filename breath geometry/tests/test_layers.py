import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.layers import BONE_HU, LayerParams, measure_layers
from breathgeom.measure.wall import Side

IntArray = npt.NDArray[np.int16]

ROWS, COLUMNS, SLICES = 160, 160, 20
SPACING = (1.0, 1.0, 2.0)
BODY_ROWS, BODY_COLUMNS = (30, 130), (40, 120)
# Designed layer thicknesses on the right flank, counted inward from the skin.
FAT_MM, MUSCLE_MM = 8, 12
FAT_HU_FILL, MUSCLE_HU_FILL, RIB_HU_FILL = -100, 50, 600
SPINE_ROWS, SPINE_COLUMNS = (72, 88), (41, 49)
PARAMS = LayerParams(max_walk_mm=50.0)


def torso(fat_mm: int = FAT_MM, muscle_mm: int = MUSCLE_MM) -> IntArray:
    """Slab phantom: skin, then fat, then muscle, then a rib, then lung.

    A flat lateral face keeps every inward normal perpendicular, so the walk
    length equals the designed thickness and the test measures the classifier
    rather than the geometry.
    """
    volume = np.full((ROWS, COLUMNS, SLICES), -1000, dtype=np.int16)
    r0, r1 = BODY_ROWS
    c0, c1 = BODY_COLUMNS
    volume[r0:r1, c0:c1, :] = MUSCLE_HU_FILL
    volume[SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1], :] = RIB_HU_FILL

    face = r1 - 1
    volume[face - fat_mm + 1:face + 1, c0:c1, :] = FAT_HU_FILL
    muscle_end = face - fat_mm - muscle_mm
    volume[muscle_end - 3:muscle_end + 1, c0:c1, :] = RIB_HU_FILL
    volume[r0 + 20:muscle_end - 3, 50:110, :] = -900  # lung behind the rib
    return volume


def test_designed_layers_are_recovered() -> None:
    result = measure_layers(torso(), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.samples
    assert float(np.median(result.fat_mm)) == pytest.approx(FAT_MM, abs=1.5)
    assert float(np.median(result.muscle_mm)) == pytest.approx(MUSCLE_MM, abs=1.5)


def test_walks_end_on_the_rib() -> None:
    """A walk that never met bone did not find the inner boundary of the wall."""
    result = measure_layers(torso(), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.bone_fraction > 0.9


def test_thicker_fat_is_seen_as_thicker() -> None:
    thin = measure_layers(torso(fat_mm=4), SPACING, side=Side.RIGHT, params=PARAMS)
    thick = measure_layers(torso(fat_mm=16), SPACING, side=Side.RIGHT, params=PARAMS)

    assert float(np.median(thick.fat_mm)) - float(np.median(thin.fat_mm)) == pytest.approx(
        12.0, abs=2.0
    )
    # The muscle behind it must not move when only the fat changes.
    assert float(np.median(thick.muscle_mm)) == pytest.approx(
        float(np.median(thin.muscle_mm)), abs=1.5
    )


def test_band_limits_the_slices() -> None:
    result = measure_layers(
        torso(), SPACING, side=Side.RIGHT, params=PARAMS, band=(4, 8)
    )

    assert result.slices_used == 5
    assert {sample.slice_index for sample in result.samples} <= set(range(4, 9))


def test_layer_below_the_minimum_is_dropped() -> None:
    """A run of one or two voxels is speckle and must not be reported."""
    strict = LayerParams(max_walk_mm=50.0, min_layer_mm=12.0)

    result = measure_layers(torso(fat_mm=4), SPACING, side=Side.RIGHT, params=strict)

    assert result.samples
    assert float(np.median(result.fat_mm)) == 0.0


def test_bone_threshold_is_above_the_muscle_window() -> None:
    assert BONE_HU > 150
