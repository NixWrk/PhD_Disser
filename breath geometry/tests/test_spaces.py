import numpy as np

from breathgeom.domain.spaces import (
    lps_to_ras_affine,
    lps_to_ras_points,
    ras_to_lps_affine,
    ras_to_lps_points,
)


def test_lps_ras_point_round_trip() -> None:
    points_lps = np.array([[10.0, 20.0, 30.0], [-4.0, 5.0, 6.0]])
    points_ras = lps_to_ras_points(points_lps)

    np.testing.assert_allclose(points_ras[0], [-10.0, -20.0, 30.0])
    np.testing.assert_allclose(ras_to_lps_points(points_ras), points_lps)


def test_lps_ras_affine_round_trip() -> None:
    affine_lps = np.array(
        [
            [0.8, 0.0, 0.0, 100.0],
            [0.0, 0.8, 0.0, -50.0],
            [0.0, 0.0, 1.5, 20.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    affine_ras = lps_to_ras_affine(affine_lps)
    np.testing.assert_allclose(ras_to_lps_affine(affine_ras), affine_lps)
