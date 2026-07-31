import numpy as np
import pytest

from breathgeom.measure.convexadam_registration import (
    ConvexAdamParams,
    preprocessing_grid,
    resample_displacement_mm,
    voxel_displacement_to_mm,
)


def test_convexadam_parameters_reject_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="dtype"):
        ConvexAdamParams(dtype="float64")


def test_preprocessing_grid_never_upsamples_and_preserves_extent() -> None:
    shape = (512, 512, 120)
    spacing = (0.625, 0.625, 2.5)

    target_shape, target_spacing = preprocessing_grid(shape, spacing, 208)

    assert target_shape == (208, 208, 120)
    assert all(size % 2 == 0 for size in target_shape)
    source_extent = (np.asarray(shape) - 1) * np.asarray(spacing)
    target_extent = (np.asarray(target_shape) - 1) * np.asarray(target_spacing)
    assert target_extent == pytest.approx(source_extent)


def test_voxel_displacement_conversion_uses_component_spacing() -> None:
    field = np.ones((4, 5, 6, 3), dtype=np.float32)

    converted = voxel_displacement_to_mm(field, (0.5, 1.5, 2.5))

    assert converted[2, 3, 4].tolist() == pytest.approx([0.5, 1.5, 2.5])


def test_resampling_constant_physical_field_does_not_rescale_vectors() -> None:
    coarse = np.zeros((6, 8, 10, 3), dtype=np.float32)
    coarse[..., 0] = 3.0
    coarse[..., 1] = -2.0
    coarse[..., 2] = 5.0

    full = resample_displacement_mm(
        coarse,
        (2.0, 2.0, 2.0),
        (11, 15, 19),
        (1.0, 1.0, 1.0),
    )

    assert full.shape == (11, 15, 19, 3)
    assert full[5, 7, 9].tolist() == pytest.approx([3.0, -2.0, 5.0])
