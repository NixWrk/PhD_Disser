from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.wall import (
    AIR_HU,
    Reduction,
    Side,
    WallParams,
    load_ras,
    measure_wall,
)

IntArray = npt.NDArray[np.int16]

# Thin wall on the patient's right, thick on the left: only an asymmetric
# phantom catches a silent left-right flip.
RIGHT_WALL_MM = 10
LEFT_WALL_MM = 30
SPACING = (1.0, 1.0, 2.0)
PARAMS = WallParams(min_lung_area_mm2=200.0, min_lung_component_px=50)

# The array is wide enough that the body fits inside the inscribed
# reconstruction circle, so an untruncated phantom really reports no contact.
ROWS, COLUMNS, SLICES = 160, 160, 40
BODY_ROWS = (30, 130)  # left face at row 30, right face at row 129
BODY_COLUMNS = (40, 120)
LUNG_COLUMNS = (50, 110)
MUSCLE_HU_FILL = 50  # inside the muscle window
FAT_HU_FILL = -100  # inside the fat window
SPINE_HU_FILL = 600
# The midline is taken from the spine, so a phantom without one is not a
# thorax as far as the measurement is concerned.
SPINE_ROWS, SPINE_COLUMNS = (72, 88), (41, 49)


def torso(fat_band_mm: int = 0) -> IntArray:
    """RAS+ slab phantom with two lungs at unequal wall depths.

    Axis 0 increases towards the patient's right, so the right face is the high
    row index. A flat lateral face makes the shortest skin-to-lung path exactly
    the designed wall thickness, and the anterior/posterior faces fall outside
    the lateral sector.
    """
    volume = np.full((ROWS, COLUMNS, SLICES), -1000, dtype=np.int16)
    r0, r1 = BODY_ROWS
    c0, c1 = BODY_COLUMNS
    lc0, lc1 = LUNG_COLUMNS
    volume[r0:r1, c0:c1, :] = MUSCLE_HU_FILL
    volume[SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1], :] = SPINE_HU_FILL

    right_face = r1 - 1
    left_face = r0
    # nearest lung row is (stop - 1) on the right and (start) on the left
    volume[right_face - RIGHT_WALL_MM - 19 : right_face - RIGHT_WALL_MM + 1, lc0:lc1, :] = (
        AIR_HU - 500
    )
    volume[left_face + LEFT_WALL_MM : left_face + LEFT_WALL_MM + 20, lc0:lc1, :] = AIR_HU - 500

    if fat_band_mm:
        volume[right_face - fat_band_mm + 1 : right_face + 1, c0:c1, :] = FAT_HU_FILL
    return volume


def test_right_wall_is_thinner_than_left() -> None:
    volume = torso()

    right = measure_wall(volume, SPACING, side=Side.RIGHT, params=PARAMS)
    left = measure_wall(volume, SPACING, side=Side.LEFT, params=PARAMS)

    assert right.rays and left.rays
    assert float(np.median(right.thickness_mm)) == pytest.approx(RIGHT_WALL_MM, abs=1.5)
    assert float(np.median(left.thickness_mm)) == pytest.approx(LEFT_WALL_MM, abs=1.5)


def test_flipped_storage_yields_the_same_anatomy(tmp_path: Path) -> None:
    """A left-handed file must measure the same wall after canonicalisation.

    Without as_closest_canonical this test fails by returning the left wall for
    a right-side request, which is exactly how a flip stays invisible in a real
    pipeline.
    """
    volume = torso()
    cases = (
        ("ras.nii.gz", volume, np.diag([1.0, 1.0, 2.0, 1.0])),
        ("las.nii.gz", volume[::-1, :, :].copy(), np.diag([-1.0, 1.0, 2.0, 1.0])),
    )

    measured = []
    for name, array, affine in cases:
        path = tmp_path / name
        nib.save(nib.Nifti1Image(array, affine), path)
        loaded, spacing = load_ras(path)
        measured.append(measure_wall(loaded, spacing, side=Side.RIGHT, params=PARAMS))

    direct, flipped = measured
    assert direct.rays and flipped.rays
    assert float(np.median(direct.thickness_mm)) == pytest.approx(RIGHT_WALL_MM, abs=1.5)
    assert float(np.median(flipped.thickness_mm)) == pytest.approx(RIGHT_WALL_MM, abs=1.5)


def test_small_lung_slices_are_skipped() -> None:
    strict = WallParams(min_lung_area_mm2=1e9, min_lung_component_px=50)

    result = measure_wall(torso(), SPACING, side=Side.RIGHT, params=strict)

    assert result.rays == ()
    assert result.slices_skipped_small_lung == result.slices_in_lower_part


def test_volume_without_lung_reports_nothing() -> None:
    solid = np.full((ROWS, COLUMNS, SLICES), MUSCLE_HU_FILL, dtype=np.int16)
    solid[SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1], :] = SPINE_HU_FILL

    result = measure_wall(solid, SPACING, params=PARAMS)

    assert result.rays == ()
    assert result.lung_extent_mm is None
    assert result.lung_slice_first is None


def test_volume_without_a_spine_is_rejected() -> None:
    """The midline has no anchor without bone, so the side cannot be decided."""
    boneless = np.full((ROWS, COLUMNS, SLICES), MUSCLE_HU_FILL, dtype=np.int16)

    with pytest.raises(ValueError, match="midline cannot be located"):
        measure_wall(boneless, SPACING, params=PARAMS)


def test_composition_splits_fat_and_muscle() -> None:
    """Outer 4 mm of the right wall is fat, the remaining 6 mm muscle."""
    result = measure_wall(torso(fat_band_mm=4), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.rays
    assert float(np.median(result.thickness_mm)) == pytest.approx(RIGHT_WALL_MM, abs=1.5)
    assert float(np.median(result.fat_mm)) == pytest.approx(4.0, abs=1.5)
    assert float(np.median(result.muscle_mm)) == pytest.approx(6.0, abs=1.5)


def test_untruncated_phantom_reports_no_fov_contact() -> None:
    result = measure_wall(torso(), SPACING, side=Side.RIGHT, params=PARAMS)

    assert result.truncated_ray_count == 0
    assert result.fov_contact_fraction == 0.0


def curved_torso() -> IntArray:
    """Elliptical body with a round lung: real chests are not slabs."""
    volume = np.full((ROWS, COLUMNS, SLICES), -1000, dtype=np.int16)
    rows, columns = np.mgrid[0:ROWS, 0:COLUMNS]
    body = ((rows - 80) / 62.0) ** 2 + ((columns - 80) / 48.0) ** 2 <= 1.0
    lung = (rows - 95) ** 2 + (columns - 80) ** 2 <= 28**2
    volume[np.repeat(body[:, :, None], SLICES, axis=2)] = MUSCLE_HU_FILL
    volume[np.repeat(lung[:, :, None], SLICES, axis=2)] = AIR_HU - 500
    volume[SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1], :] = SPINE_HU_FILL
    return volume


def test_median_reduction_runs_above_min_on_a_curved_wall() -> None:
    """The two reductions do not measure the same quantity.

    Over a wide angular sector the rays near its edges reach a curved lung
    obliquely, so the median runs well above the perpendicular thickness that
    the minimum reports. Neither is wrong in itself; comparing across them is.
    """
    volume = curved_torso()
    common = {"min_lung_area_mm2": 200.0, "min_lung_component_px": 50}
    smallest = measure_wall(volume, SPACING, side=Side.RIGHT,
                            params=WallParams(reduction=Reduction.MIN, **common))
    typical = measure_wall(volume, SPACING, side=Side.RIGHT,
                           params=WallParams(reduction=Reduction.MEDIAN, **common))

    assert smallest.rays and typical.rays
    low = float(np.median(smallest.thickness_mm))
    high = float(np.median(typical.thickness_mm))
    assert high > low

    # On a flat wall with a parallel lung every ray is perpendicular, so the
    # two collapse onto the same number. The gap is created by curvature, and
    # on real anatomy it reaches a factor of two.
    flat = torso()
    flat_min = measure_wall(flat, SPACING, side=Side.RIGHT,
                            params=WallParams(reduction=Reduction.MIN, **common))
    flat_median = measure_wall(flat, SPACING, side=Side.RIGHT,
                               params=WallParams(reduction=Reduction.MEDIAN, **common))
    assert float(np.median(flat_median.thickness_mm)) == pytest.approx(
        float(np.median(flat_min.thickness_mm)), abs=0.5
    )
