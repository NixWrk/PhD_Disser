import numpy as np
import numpy.typing as npt
import pytest

from breathgeom.measure.profiles import (
    ProfileParams,
    extract_whole_body_profiles,
    pair_whole_body_profiles,
)

IntArray = npt.NDArray[np.int_]


def concentric_phantom() -> tuple[IntArray, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    grid = np.indices((64, 64, 64), dtype=np.float64)
    radius = np.sqrt(np.sum((grid - 31.5) ** 2, axis=0))
    body = radius <= 25.0
    lung = radius <= 10.0
    image = np.full(body.shape, -1000, dtype=np.int_)
    image[(radius > 10.0) & (radius <= 20.0)] = 50
    image[(radius > 20.0) & body] = -100
    image[lung] = -800
    return image, body, lung


def test_profiles_cover_whole_surface_and_partition_thickness() -> None:
    image, body, lung = concentric_phantom()
    result = extract_whole_body_profiles(
        image,
        (1.0, 1.0, 1.0),
        body,
        lung,
        params=ProfileParams(surface_spacing_mm=6.0, line_step_mm=0.5),
    )
    valid = [profile for profile in result.profiles if profile.valid]

    assert result.coverage == "whole-segmented-lung-surface_no-electrode-filter"
    assert len(valid) > 30
    assert {profile.region for profile in valid} == {
        "right",
        "left",
        "anterior",
        "posterior",
        "superior",
        "inferior",
    }
    for profile in valid:
        assert profile.fat_mm + profile.muscle_mm + profile.bone_mm + profile.other_mm == (
            pytest.approx(profile.thickness_mm)
        )
    assert np.median([profile.thickness_mm for profile in valid]) == pytest.approx(
        15.0, abs=1.5
    )
    assert np.median([profile.fat_mm for profile in valid]) == pytest.approx(5.0, abs=1.5)
    assert np.median([profile.muscle_mm for profile in valid]) == pytest.approx(
        10.0, abs=1.5
    )


def test_paired_profiles_refuse_failed_registration() -> None:
    image, body, lung = concentric_phantom()
    field = np.zeros(image.shape + (3,), dtype=np.float32)

    with pytest.raises(ValueError, match="blocked"):
        pair_whole_body_profiles(
            image,
            image,
            (1.0, 1.0, 1.0),
            body,
            body,
            lung,
            lung,
            field,
            registration_gate_pass=False,
        )
