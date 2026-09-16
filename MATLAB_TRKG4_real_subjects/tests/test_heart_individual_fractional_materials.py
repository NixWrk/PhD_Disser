"""Stage-1 tests for individual tetrahedral heart union geometry."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from heart_fractional_materials import uniform_tet_barycentric as shared_uniform_tet_barycentric
from heart_individual_fractional_materials import TetrahedralHeart, integrate_individual_fractions


def tet(origin=(0.0, 0.0, 0.0), size=1.0, *, reverse=False):
    origin = np.asarray(origin, dtype=float)
    result = np.array(
        [
            origin,
            origin + [size, 0.0, 0.0],
            origin + [0.0, size, 0.0],
            origin + [0.0, 0.0, size],
        ]
    )
    if reverse:
        result[[1, 2]] = result[[2, 1]]
    return result


def transformed(vertices, centre, volume_fraction=0.0, translation=(0.0, 0.0, 0.0)):
    scale = np.cbrt(1.0 + volume_fraction)
    return centre + scale * (vertices - centre) + np.asarray(translation)


def test_original_union_preserves_concavity_and_disconnected_components():
    first = tet(size=1.0)
    second = tet(origin=(3.0, 3.0, 0.0), size=1.0)
    heart = TetrahedralHeart(np.array([first, second]))

    inside = np.array([[0.1, 0.1, 0.1], [3.1, 3.1, 0.1]])
    convex_hull_false_positive = np.array([1.5, 1.5, 0.1])
    assert np.array_equal(heart.contains(inside), [True, True])
    assert not heart.contains(convex_hull_false_positive)
    assert heart.contains(np.array([[[0.1, 0.1, 0.1]], [[3.1, 3.1, 0.1]]])).shape == (2, 1)


def test_known_volume_centroid_and_signed_orientation_are_explicit():
    first = tet(origin=(0.0, 0.0, 0.0), size=2.0)
    second = tet(origin=(4.0, 0.0, 0.0), size=1.0, reverse=True)
    heart = TetrahedralHeart(np.array([first, second]))
    first_volume = 2.0**3 / 6.0
    second_volume = 1.0 / 6.0
    expected_centre = (first_volume * first.mean(axis=0) + second_volume * second.mean(axis=0)) / (
        first_volume + second_volume
    )
    assert heart.volume_ml == pytest.approx((first_volume + second_volume) * 1e6)
    np.testing.assert_allclose(heart.centre_m, expected_centre)
    np.testing.assert_allclose(heart.bounds_m[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(heart.bounds_m[1], [5.0, 2.0, 2.0])


def test_boundary_points_are_inside_and_invalid_tetrahedra_are_rejected():
    heart = TetrahedralHeart(np.array([tet()]))
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.25, 0.75], [0.0, 0.0, 1.0]])
    assert np.all(heart.contains(points))
    assert not heart.contains([0.4, 0.4, 0.4])
    with pytest.raises(ValueError, match="degenerate"):
        TetrahedralHeart(np.array([[[0.0, 0.0, 0.0]] * 4]))
    with pytest.raises(ValueError, match="finite"):
        TetrahedralHeart(np.array([tet()], dtype=float) * np.nan)
    with pytest.raises(ValueError, match="duplicate"):
        TetrahedralHeart(np.array([tet(), tet()]))


def test_nonorthogonal_rotated_translated_tet_uses_row_edge_barycentrics():
    base = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.1, 0.3], [0.2, 0.1, 1.4]],
        dtype=float,
    )
    theta = 0.47
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    offset = np.array([3.0, -2.0, 0.7])
    vertices = base @ rotation.T + offset
    heart = TetrahedralHeart(np.array([vertices]))
    weights = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    interior = weights @ vertices
    exterior = np.array([vertices[0] + 1.5 * (vertices[1] - vertices[0])])
    assert np.array_equal(heart.contains(interior), [True, True])
    assert not heart.contains(exterior)
    negative = TetrahedralHeart(np.array([vertices[[0, 2, 1, 3]]]))
    assert np.array_equal(negative.contains(interior), [True, True])


def test_unit_scaling_uses_metres_and_millilitres():
    size_m = 0.01
    heart = TetrahedralHeart(np.array([tet(size=size_m)]))
    assert heart.volume_ml == pytest.approx(size_m**3 / 6.0 * 1e6)
    np.testing.assert_allclose(heart.centre_m, [size_m / 4.0] * 3)


def test_inverse_translation_and_uniform_scaling_recovers_original_elements():
    original = np.array([tet(), tet(origin=(2.0, 0.0, 0.0), size=0.75)])
    heart = TetrahedralHeart(original)
    volume_fraction = 0.728
    translation = np.array([0.4, -0.25, 0.7])
    moved = transformed(original, heart.centre_m, volume_fraction, translation)
    fractions, diagnostics = integrate_individual_fractions(
        moved, heart, volume_fraction=volume_fraction, translation_m=translation, levels=(64, 512), batch_size=1
    )
    np.testing.assert_array_equal(fractions, np.ones_like(fractions))
    assert diagnostics["centroid_filter_used"] is False
    assert diagnostics["identity_transform"] is False
    assert diagnostics["transformed_heart_volume_ml"] == pytest.approx(heart.volume_ml * (1.0 + volume_fraction))


def test_identity_fractions_are_one_for_original_conforming_elements():
    original = np.array([tet(), tet(origin=(1.0, 0.0, 0.0), size=0.75)])
    heart = TetrahedralHeart(original)
    fractions, diagnostics = integrate_individual_fractions(original, heart, levels=(64, 512), batch_size=2)
    np.testing.assert_array_equal(fractions, np.ones_like(fractions))
    assert diagnostics["identity_transform"] is True
    assert diagnostics["inside_sample_count_by_level"] == [len(original) * 64, len(original) * 512]


def test_query_tet_crossing_heart_is_kept_when_centroid_is_outside():
    query = np.array([[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.0, 0.0], [1.2, 0.0, 1.0]]])
    heart = TetrahedralHeart(np.array([tet(size=1.0)]))
    assert np.linalg.norm(query.mean(axis=1)[0] - heart.centre_m) > 0.5
    fractions, diagnostics = integrate_individual_fractions(query, heart, levels=(64, 512))
    assert diagnostics["centroid_filter_used"] is False
    assert fractions[-1, 0] > 0.0


def test_nested_quadrature_is_the_shared_rule():
    barycentric = shared_uniform_tet_barycentric(512)
    fractions, diagnostics = integrate_individual_fractions(
        np.array([tet(size=0.5)]), TetrahedralHeart(np.array([tet()])), levels=(64, 512), barycentric=barycentric
    )
    assert np.array_equal(barycentric[:64], shared_uniform_tet_barycentric(64))
    assert diagnostics["levels"] == [64, 512]
    # The smaller tet is entirely inside the heart, so both nested levels agree.
    np.testing.assert_array_equal(fractions, np.ones_like(fractions))


def test_invalid_integration_parameters_do_not_silently_change_geometry():
    heart = TetrahedralHeart(np.array([tet()]))
    with pytest.raises(ValueError, match="greater than -1"):
        integrate_individual_fractions(np.array([tet()]), heart, volume_fraction=-1.0)
    with pytest.raises(ValueError, match="translation_m"):
        integrate_individual_fractions(np.array([tet()]), heart, translation_m=(0.0, 0.0))
    with pytest.raises(ValueError, match="levels"):
        integrate_individual_fractions(np.array([tet()]), heart, levels=(3, 8))

