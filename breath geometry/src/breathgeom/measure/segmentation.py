"""Conservative threshold masks used by registration and profile QC."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from breathgeom.measure.wall import (
    IntArray,
    Side,
    WallParams,
    anatomical_midline,
    body_mask,
    lung_mask,
)

BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


def body_mask_3d(volume_ras: IntArray, params: WallParams | None = None) -> BoolArray:
    """Largest filled body cross-section on every axial slice."""
    params = params or WallParams()
    output = np.zeros(volume_ras.shape, dtype=bool)
    for index in range(volume_ras.shape[2]):
        mask = body_mask(volume_ras[:, :, index], params)
        if mask is not None:
            output[:, :, index] = mask
    return output


def lung_mask_3d(volume_ras: IntArray, params: WallParams | None = None) -> BoolArray:
    """Both aerated lungs, separated from bowel gas by side and component size."""
    params = params or WallParams()
    midline = anatomical_midline(volume_ras, params)
    output = np.zeros(volume_ras.shape, dtype=bool)
    for index in range(volume_ras.shape[2]):
        slice_hu = volume_ras[:, :, index]
        body = body_mask(slice_hu, params)
        if body is None:
            continue
        for side in (Side.RIGHT, Side.LEFT):
            lung = lung_mask(slice_hu, body, side, params, midline)
            if lung is not None:
                output[:, :, index] |= lung
    # Close only sub-voxel gaps; large airways/fissures remain anatomically visible.
    closed: BoolArray = ndimage.binary_closing(output, np.ones((3, 3, 1), dtype=bool))
    return closed


def binary_surface(mask: BoolArray) -> BoolArray:
    """One-voxel internal surface of a binary volume."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    eroded: BoolArray = ndimage.binary_erosion(mask, np.ones((3, 3, 3), dtype=bool))
    return mask & ~eroded


__all__ = ["binary_surface", "body_mask_3d", "lung_mask_3d"]
