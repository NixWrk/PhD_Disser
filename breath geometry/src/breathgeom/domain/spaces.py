from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


LPS_TO_RAS: FloatArray = np.diag([-1.0, -1.0, 1.0, 1.0])


def lps_to_ras_points(points: npt.ArrayLike) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("Expected points with final dimension 3")
    result = values.copy()
    result[..., 0:2] *= -1.0
    return result


def ras_to_lps_points(points: npt.ArrayLike) -> FloatArray:
    return lps_to_ras_points(points)


def lps_to_ras_affine(affine_lps: npt.ArrayLike) -> FloatArray:
    affine = np.asarray(affine_lps, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("Expected a 4x4 affine")
    return LPS_TO_RAS @ affine


def ras_to_lps_affine(affine_ras: npt.ArrayLike) -> FloatArray:
    affine = np.asarray(affine_ras, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("Expected a 4x4 affine")
    return LPS_TO_RAS @ affine
