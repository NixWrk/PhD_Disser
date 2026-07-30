import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.registration import (
    BSplineParams,
    RegistrationParams,
    landmark_tre,
    mask_metrics,
    register_bspline,
    register_diffeomorphic,
    warp_mask,
)

IntArray = npt.NDArray[np.int16]


def shifted_blob(shift: int = 0) -> tuple[IntArray, npt.NDArray[np.bool_]]:
    grid = np.indices((40, 40, 32))
    radius = (
        (grid[0] - (20 + shift)) ** 2
        + (grid[1] - 20) ** 2
        + (grid[2] - 16) ** 2
    )
    mask = radius <= 8**2
    image = np.full(mask.shape, -1000, dtype=np.int16)
    image[mask] = 100
    return image, mask


def test_mask_metrics_are_physical_and_symmetric() -> None:
    _, first = shifted_blob(0)
    _, second = shifted_blob(2)
    forward = mask_metrics(first, second, (2.0, 1.0, 1.0))
    reverse = mask_metrics(second, first, (2.0, 1.0, 1.0))

    assert forward == reverse
    assert forward.dice < 1.0
    assert forward.surface_p95_mm == pytest.approx(np.sqrt(8.0))


def test_diffeomorphic_registration_improves_shifted_blob() -> None:
    fixed, fixed_mask = shifted_blob(0)
    moving, moving_mask = shifted_blob(2)
    params = RegistrationParams(level_spacing_mm=(1.0,), iterations=(30,))

    result = register_diffeomorphic(
        fixed,
        moving,
        (1.0, 1.0, 1.0),
        fixed_mask,
        moving_mask,
        params=params,
        run_rigid=False,
    )

    before = mask_metrics(fixed_mask, moving_mask, (1.0, 1.0, 1.0))
    after = mask_metrics(fixed_mask, result.warped_moving_mask, (1.0, 1.0, 1.0))
    assert after.dice > before.dice
    assert result.nonpositive_jacobian_fraction == 0.0
    assert result.displacement_mm.shape == fixed.shape + (3,)


def test_landmark_tre_samples_fixed_to_moving_field() -> None:
    field = np.zeros((10, 10, 10, 3), dtype=np.float32)
    field[..., 0] = 2.0
    fixed = np.array([[2.0, 3.0, 4.0], [5.0, 5.0, 5.0]])
    moving = fixed + np.array([2.0, 0.0, 0.0])

    assert np.mean(landmark_tre(fixed, moving)) == pytest.approx(2.0)
    assert np.mean(landmark_tre(fixed, moving, field, (1.0, 1.0, 1.0))) == pytest.approx(0.0)


def test_warp_mask_uses_fixed_to_moving_field_direction() -> None:
    moving = np.zeros((10, 10, 10), dtype=bool)
    moving[4, 5, 5] = True
    field = np.zeros(moving.shape + (3,), dtype=np.float32)
    field[..., 0] = 2.0

    warped = warp_mask(moving, field, (1.0, 1.0, 1.0))

    assert warped[2, 5, 5]
    assert int(warped.sum()) == 1


def test_bspline_registration_improves_shifted_blob() -> None:
    fixed, fixed_mask = shifted_blob(0)
    moving, moving_mask = shifted_blob(2)
    params = BSplineParams(
        control_point_spacing_mm=20.0,
        optimization_spacing_mm=1.0,
        iterations=20,
        sampling_fraction=0.5,
        shrink_factors=(1,),
        smoothing_sigmas_mm=(0.0,),
        mesh_scale_factors=(0,),
    )

    result = register_bspline(
        fixed, moving, (1.0, 1.0, 1.0), fixed_mask, moving_mask, params=params
    )

    before = mask_metrics(fixed_mask, moving_mask, (1.0, 1.0, 1.0))
    after = mask_metrics(fixed_mask, result.warped_moving_mask, (1.0, 1.0, 1.0))
    assert after.dice > before.dice
