from pathlib import Path
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.io.dirlab import (
    COPDGENE_REQUIRED_SUFFIXES,
    HU_OFFSET,
    RAW_OUTSIDE,
    CaseGeometry,
    inventory_copdgene,
    load_copdgene,
)

IntArray = npt.NDArray[np.int16]

GEOMETRY = CaseGeometry(64, 64, 40, 1.0, 1.0, 2.5)
AIR_HU_FILL = -1000
LUNG_HU_FILL = -850
SOFT_HU_FILL = 50
SPINE_HU_FILL = 600

# In an axial DICOM the first in-plane index runs along the column direction
# cosine, so it is anteroposterior, and the second is left-right. The phantom is
# built in that order on purpose: a phantom that repeats the reader's own
# assumption cannot catch the reader getting it wrong.
BODY_ROWS, BODY_COLUMNS = (16, 48), (14, 50)
SPINE_ROWS, SPINE_COLUMNS = (40, 48), (24, 40)
RIGHT_LUNG_COLUMNS = (34, 47)
LEFT_LUNG_COLUMNS = (18, 28)
LUNG_ROWS = (20, 44)


def phantom(
    right_at_high_column: bool,
    posterior_at_high_row: bool,
    superior_at_high_slice: bool,
) -> IntArray:
    """(slice, row=AP, column=LR) stack stored as HU + 1024.

    The spine sits posterior and on the midline, the right lung is the wider of
    the two, and the lung base is the wide end.
    """
    stack = np.full((GEOMETRY.slices, GEOMETRY.rows, GEOMETRY.columns),
                    AIR_HU_FILL, dtype=np.int32)
    stack[:, BODY_ROWS[0]:BODY_ROWS[1], BODY_COLUMNS[0]:BODY_COLUMNS[1]] = SOFT_HU_FILL
    stack[:, SPINE_ROWS[0]:SPINE_ROWS[1], SPINE_COLUMNS[0]:SPINE_COLUMNS[1]] = SPINE_HU_FILL

    for index in range(GEOMETRY.slices):
        anatomical = index / (GEOMETRY.slices - 1)
        if anatomical > 0.85:
            continue  # above the apex
        taper = 1.0 - max(anatomical - 0.15, 0.0) / 0.7
        for start, stop in (RIGHT_LUNG_COLUMNS, LEFT_LUNG_COLUMNS):
            width = int((stop - start) * taper)
            if width < 3:
                continue
            rows = slice(LUNG_ROWS[0], LUNG_ROWS[1])
            stack[index, rows, start:start + width] = LUNG_HU_FILL

    if not right_at_high_column:
        stack = stack[:, :, ::-1]
    if not posterior_at_high_row:
        stack = stack[:, ::-1, :]
    if not superior_at_high_slice:
        stack = stack[::-1, :, :]

    raw = np.ascontiguousarray(stack + HU_OFFSET).astype(np.int16)
    grid = np.mgrid[0:GEOMETRY.rows, 0:GEOMETRY.columns]
    radius = np.sqrt((grid[0] - 31.5) ** 2 + (grid[1] - 31.5) ** 2)
    raw[:, radius > 30] = RAW_OUTSIDE - 500  # out-of-field padding
    return raw


def write(tmp_path: Path, raw: IntArray, name: str = "case.img") -> Path:
    path = tmp_path / name
    raw.tofile(path)
    return path


def test_inventory_distinguishes_archived_and_extracted_cases(tmp_path: Path) -> None:
    with ZipFile(tmp_path / "copd1.zip", "w") as archive:
        for suffix in COPDGENE_REQUIRED_SUFFIXES:
            archive.writestr(f"nested/copd1{suffix}", b"test")

    extracted = tmp_path / "extracted" / "copd2"
    extracted.mkdir(parents=True)
    for suffix in COPDGENE_REQUIRED_SUFFIXES:
        (extracted / f"copd2{suffix}").touch()

    rows = {row.case_id: row for row in inventory_copdgene(tmp_path)}

    assert len(rows) == 10
    assert rows["copd1"].archive_complete
    assert not rows["copd1"].extracted_complete
    assert rows["copd2"].extracted_complete
    assert not rows["copd2"].archive_complete
    assert not rows["copd10"].archive_complete
    assert not rows["copd10"].extracted_complete


@pytest.mark.parametrize("right_high", [True, False])
@pytest.mark.parametrize("posterior_high", [True, False])
@pytest.mark.parametrize("superior_high", [True, False])
def test_every_storage_order_yields_the_same_anatomy(
    tmp_path: Path, right_high: bool, posterior_high: bool, superior_high: bool
) -> None:
    """All eight storage orders must come out as one and the same RAS+ volume."""
    path = write(tmp_path, phantom(right_high, posterior_high, superior_high))

    volume, spacing, report = load_copdgene(path, "phantom", GEOMETRY)

    assert report.right_is_high_index is right_high
    assert report.posterior_is_high_row is posterior_high
    assert report.superior_is_high_index is superior_high
    # axis 0 = column spacing, axis 1 = row spacing, after the transpose
    assert spacing == (GEOMETRY.spacing_x_mm, GEOMETRY.spacing_y_mm, GEOMETRY.spacing_z_mm)

    lung = volume < -400
    half = lung.shape[0] // 2
    assert lung[half:].sum() > lung[:half].sum(), "right lung must sit at high axis 0"

    spine = volume > 400
    spine_axis1 = np.where(spine)[1].mean()
    body = volume > -300
    body_axis1 = np.where(body)[1].mean()
    assert spine_axis1 < body_axis1, "the spine is posterior, so it sits at low axis 1"

    per_slice = lung.sum(axis=(0, 1))
    lower = per_slice[: len(per_slice) // 2].sum()
    upper = per_slice[len(per_slice) // 2 :].sum()
    assert lower > upper, "the wide lung base must sit at low axis 2"


def test_wrong_slice_count_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, phantom(True, True, True))
    wrong = CaseGeometry(64, 64, 39, 1.0, 1.0, 2.5)

    with pytest.raises(ValueError, match="published table says"):
        load_copdgene(path, "phantom", wrong)


def test_ragged_file_is_rejected(tmp_path: Path) -> None:
    raw = phantom(True, True, True).ravel()[:-7]
    path = write(tmp_path, raw.astype(np.int16), "ragged.img")

    with pytest.raises(ValueError, match="whole number of slices"):
        load_copdgene(path, "phantom", GEOMETRY)


def test_unshifted_intensities_are_rejected(tmp_path: Path) -> None:
    """A file already in HU must not be silently read as HU+1024.

    The tell is the floor, not a peak: shifting an already-correct file drives
    its air down to about -2024 HU, which nothing in a CT can be.
    """
    raw = phantom(True, True, True).astype(np.int32) - HU_OFFSET
    path = write(tmp_path, raw.astype(np.int16), "already_hu.img")

    with pytest.raises(ValueError, match="below -1100 HU"):
        load_copdgene(path, "phantom", GEOMETRY)


def test_shared_orientation_is_applied_verbatim(tmp_path: Path) -> None:
    """Passing a report must override inference, so both phases match."""
    upright = write(tmp_path, phantom(True, True, True), "in.img")
    mirrored = write(tmp_path, phantom(False, True, True), "ex.img")

    _, _, anchor = load_copdgene(upright, "phantom", GEOMETRY)
    _, _, reused = load_copdgene(mirrored, "phantom", GEOMETRY, orientation=anchor)
    _, _, inferred = load_copdgene(mirrored, "phantom", GEOMETRY)

    assert reused is anchor
    assert inferred.right_is_high_index is not anchor.right_is_high_index
