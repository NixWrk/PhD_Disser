"""Tests for the parts of alignment this project owns.

The end-to-end fit is deliberately not exercised here. Driving ITK's masked
multi-resolution optimiser on a toy volume fails inside ITK itself, at every
voxel spacing tried, while the same call succeeds on all ten real DIR-Lab pairs.
A phantom test would therefore assert something about ITK's tolerance for small
volumes rather than about this code. End-to-end behaviour is checked on real
data and recorded in docs/RESEARCH_REPORT.md.

What is tested here is what this module decides: which voxels count as the
vertebral column, how overlap is scored, and how the resolution pyramid is cut
to the size of that column.
"""

import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.align import (
    MIN_MASKED_VOXELS_PER_LEVEL,
    _column_mask,
    _dice,
    pyramid_factors,
)
from breathgeom.measure.wall import AIR_HU, WallParams

IntArray = npt.NDArray[np.int16]

ROWS, COLUMNS, SLICES = 96, 96, 48
SPACING = (1.0, 1.0, 2.0)
MUSCLE_HU_FILL = 50
SPINE_HU_FILL = 600
RIB_HU_FILL = 700
SPINE_ROWS, SPINE_COLUMNS = (42, 54), (22, 32)
# Ribs are dense too and lie away from the midline. They must be excluded:
# they rotate with every breath, and that is the motion under study.
RIB_ROWS, RIB_COLUMNS = (12, 20), (24, 34)
PARAMS = WallParams()


def torso(with_ribs: bool = True) -> IntArray:
    """RAS+ phantom with a posterior midline column and a pair of lateral ribs."""
    volume = np.full((ROWS, COLUMNS, SLICES), -1000, dtype=np.int16)
    rows, columns = np.mgrid[0:ROWS, 0:COLUMNS]
    body = ((rows - 48) / 38.0) ** 2 + ((columns - 48) / 30.0) ** 2 <= 1.0
    lung = (rows - 62) ** 2 + (columns - 50) ** 2 <= 15**2
    volume[np.repeat(body[:, :, None], SLICES, axis=2)] = MUSCLE_HU_FILL
    volume[np.repeat(lung[:, :, None], SLICES, axis=2)] = AIR_HU - 500
    volume[SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1], :] = SPINE_HU_FILL
    if with_ribs:
        volume[RIB_ROWS[0]:RIB_ROWS[1], RIB_COLUMNS[0]:RIB_COLUMNS[1], :] = RIB_HU_FILL
        volume[ROWS - RIB_ROWS[1]:ROWS - RIB_ROWS[0], RIB_COLUMNS[0]:RIB_COLUMNS[1], :] = (
            RIB_HU_FILL
        )
    return volume


def test_column_mask_excludes_ribs() -> None:
    """Density alone would keep the ribs; distance from the midline removes them."""
    with_ribs = _column_mask(torso(with_ribs=True), SPACING, PARAMS)
    without = _column_mask(torso(with_ribs=False), SPACING, PARAMS)

    assert int(with_ribs.sum()) == int(without.sum())
    assert not with_ribs[RIB_ROWS[0]:RIB_ROWS[1], RIB_COLUMNS[0]:RIB_COLUMNS[1], :].any()


def test_column_mask_covers_the_vertebral_block() -> None:
    mask = _column_mask(torso(), SPACING, PARAMS)
    expected = (
        (SPINE_ROWS[1] - SPINE_ROWS[0]) * (SPINE_COLUMNS[1] - SPINE_COLUMNS[0]) * SLICES
    )

    assert int(mask.sum()) == pytest.approx(expected, rel=0.1)
    assert mask[SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1], :].all()


def test_column_mask_needs_bone() -> None:
    boneless = np.full((ROWS, COLUMNS, SLICES), MUSCLE_HU_FILL, dtype=np.int16)

    with pytest.raises(ValueError, match="midline cannot be located"):
        _column_mask(boneless, SPACING, PARAMS)


def test_dice_spans_disjoint_to_identical() -> None:
    first = np.zeros((4, 4, 4), dtype=bool)
    first[:2] = True
    second = np.zeros((4, 4, 4), dtype=bool)
    second[2:] = True

    assert _dice(first, first) == pytest.approx(1.0)
    assert _dice(first, second) == pytest.approx(0.0)
    assert _dice(np.zeros((2, 2, 2), dtype=bool), np.zeros((2, 2, 2), dtype=bool)) == 0.0


def test_pyramid_is_cut_to_the_column_size() -> None:
    """Every level must keep enough masked voxels for the joint histogram.

    A level that downsamples the mask into near-emptiness does not merely lose
    precision: Mattes mutual information becomes undefined and ITK raises.
    """
    assert pyramid_factors(10**6) == [4, 2, 1]
    assert pyramid_factors(10 * MIN_MASKED_VOXELS_PER_LEVEL) == [2, 1]
    assert pyramid_factors(MIN_MASKED_VOXELS_PER_LEVEL) == [1]
    assert pyramid_factors(0) == [1]
    for size in (0, 1, 500, 5000, 500_000):
        assert pyramid_factors(size)[-1] == 1
