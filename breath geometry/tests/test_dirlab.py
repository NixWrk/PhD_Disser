from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.io.dirlab import (
    HU_OFFSET,
    RAW_OUTSIDE,
    CaseGeometry,
    load_copdgene,
)

IntArray = npt.NDArray[np.int16]

GEOMETRY = CaseGeometry(64, 64, 40, 1.0, 1.0, 2.5)
AIR_HU_FILL = -1000
LUNG_HU_FILL = -850
SOFT_HU_FILL = 50


def phantom(right_at_high_row: bool, superior_at_high_slice: bool) -> IntArray:
    """(slice, row, column) stack in DIR-Lab layout, stored as HU + 1024.

    The right lung is larger, and the lung base — the wide end — is caudal.
    """
    stack = np.full((GEOMETRY.slices, GEOMETRY.rows, GEOMETRY.columns),
                    AIR_HU_FILL, dtype=np.int32)
    # The body must occupy well under half the reconstruction circle, as it does
    # in a real large-field thorax scan: external air has to stay the modal value.
    stack[:, 16:48, 18:46] = SOFT_HU_FILL

    for index in range(GEOMETRY.slices):
        # caudal half holds the wide lung base, cranial half tapers to nothing
        position = index / (GEOMETRY.slices - 1)
        anatomical = position if superior_at_high_slice else 1.0 - position
        if anatomical > 0.85:
            continue  # above the apex
        taper = 1.0 - max(anatomical - 0.15, 0.0) / 0.7
        wide = int(13 * taper)  # right lung
        narrow = int(8 * taper)  # left lung, displaced by the heart
        if narrow < 3:
            continue
        high_rows = slice(33, 33 + (wide if right_at_high_row else narrow))
        low_rows = slice(31 - (narrow if right_at_high_row else wide), 31)
        stack[index, high_rows, 22:42] = LUNG_HU_FILL
        stack[index, low_rows, 22:42] = LUNG_HU_FILL

    raw = (stack + HU_OFFSET).astype(np.int16)
    corner = np.mgrid[0:GEOMETRY.rows, 0:GEOMETRY.columns]
    radius = np.sqrt((corner[0] - 31.5) ** 2 + (corner[1] - 31.5) ** 2)
    raw[:, radius > 30] = RAW_OUTSIDE - 500  # out-of-field padding
    return raw


def write(tmp_path: Path, raw: IntArray, name: str = "case.img") -> Path:
    path = tmp_path / name
    raw.tofile(path)
    return path


@pytest.mark.parametrize("right_high", [True, False])
@pytest.mark.parametrize("superior_high", [True, False])
def test_orientation_is_recovered_for_every_storage_order(
    tmp_path: Path, right_high: bool, superior_high: bool
) -> None:
    """All four storage orders must yield the same anatomy in RAS+."""
    path = write(tmp_path, phantom(right_high, superior_high))

    volume, spacing, report = load_copdgene(path, "phantom", GEOMETRY)

    assert report.right_is_high_index is right_high
    assert report.superior_is_high_index is superior_high
    assert spacing == (GEOMETRY.spacing_y_mm, GEOMETRY.spacing_x_mm, GEOMETRY.spacing_z_mm)

    # In RAS+ the right lung (the larger one) must sit at high row indices
    lung = volume < -400
    half = lung.shape[0] // 2
    assert lung[half:].sum() > lung[:half].sum()
    # and the wide base must sit at low slice indices
    per_slice = lung.reshape(-1, lung.shape[2]).sum(axis=0)
    lower = per_slice[: len(per_slice) // 2].sum()
    upper = per_slice[len(per_slice) // 2 :].sum()
    assert lower > upper


def test_wrong_slice_count_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, phantom(True, True))
    wrong = CaseGeometry(64, 64, 39, 1.0, 1.0, 2.5)

    with pytest.raises(ValueError, match="published table says"):
        load_copdgene(path, "phantom", wrong)


def test_ragged_file_is_rejected(tmp_path: Path) -> None:
    raw = phantom(True, True).ravel()[:-7]
    path = write(tmp_path, raw.astype(np.int16), "ragged.img")

    with pytest.raises(ValueError, match="whole number of slices"):
        load_copdgene(path, "phantom", GEOMETRY)


def test_unshifted_intensities_are_rejected(tmp_path: Path) -> None:
    """A file already in HU must not be silently read as HU+1024.

    The tell is the floor, not a peak: shifting an already-correct file drives
    its air down to about -2024 HU, which nothing in a CT can be.
    """
    raw = phantom(True, True).astype(np.int32) - HU_OFFSET
    path = write(tmp_path, raw.astype(np.int16), "already_hu.img")

    with pytest.raises(ValueError, match="below -1100 HU"):
        load_copdgene(path, "phantom", GEOMETRY)


def test_shared_orientation_is_applied_verbatim(tmp_path: Path) -> None:
    """Passing a report must override inference, so both phases match."""
    inhale = write(tmp_path, phantom(True, True), "in.img")
    mirrored = write(tmp_path, phantom(False, True), "ex.img")

    _, _, anchor = load_copdgene(inhale, "phantom", GEOMETRY)
    _, _, reused = load_copdgene(mirrored, "phantom", GEOMETRY, orientation=anchor)
    _, _, inferred = load_copdgene(mirrored, "phantom", GEOMETRY)

    assert reused is anchor
    assert inferred.right_is_high_index is not anchor.right_is_high_index
