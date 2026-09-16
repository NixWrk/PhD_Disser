"""Tests for conservative acceleration around the frozen heart integrator."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from heart_fractional_materials import uniform_tet_barycentric
from heart_individual_fractional_materials import (
    TetrahedralHeart,
    integrate_individual_fractions,
)
from heart_individual_fractional_fast import integrate_individual_fractions_fast


def tet(origin=(0.0, 0.0, 0.0), size=1.0):
    origin = np.asarray(origin, dtype=float)
    return np.array(
        [
            origin,
            origin + [size, 0.0, 0.0],
            origin + [0.0, size, 0.0],
            origin + [0.0, 0.0, size],
        ]
    )


def test_fast_matches_frozen_output_with_same_nested_points_and_transform():
    heart_vertices = np.array(
        [[[1.0, 2.0, 3.0], [2.0, 2.2, 3.0], [1.2, 3.1, 3.3], [1.4, 2.1, 4.4]]]
    )
    heart = TetrahedralHeart(heart_vertices)
    query = np.array(
        [
            heart_vertices[0],
            tet(origin=(8.0, 8.0, 8.0)),
            np.array([[1.1, 2.0, 3.0], [2.3, 2.2, 3.0], [1.3, 3.3, 3.3], [1.5, 2.1, 4.5]]),
        ]
    )
    barycentric = uniform_tet_barycentric(64)
    kwargs = dict(
        volume_fraction=0.125,
        translation_m=(0.4, -0.2, 0.7),
        levels=(16, 64),
        barycentric=barycentric,
        batch_size=2,
    )
    frozen, _ = integrate_individual_fractions(query, heart, **kwargs)
    fast, diagnostics = integrate_individual_fractions_fast(query, heart, **kwargs, chunk_size=2)
    np.testing.assert_array_equal(fast, frozen)
    json.dumps(diagnostics)
    assert diagnostics["centroid_filter_used"] is False


def test_vertices_in_different_tets_are_not_a_false_full_union_classification():
    heart = TetrahedralHeart(np.array([tet(), tet(origin=(3.0, 0.0, 0.0))]))
    query = np.array(
        [[[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [3.1, 0.2, 0.1], [3.2, 0.1, 0.2]]]
    )
    fractions, diagnostics = integrate_individual_fractions_fast(query, heart, levels=(16, 64))
    assert diagnostics["exact_full_tetrahedra_by_strict_same_tet"] == 0
    assert diagnostics["partial_quadrature_tetrahedra"] == 1
    assert fractions.shape == (2, 1)


def test_disjoint_aabbs_are_exact_zero_without_sampling():
    heart = TetrahedralHeart(np.array([tet()]))
    query = np.array([tet(origin=(10.0, 10.0, 10.0)), tet(origin=(-10.0, -10.0, -10.0))])
    fractions, diagnostics = integrate_individual_fractions_fast(heart.vertices_m * 0 + query, heart, levels=(512, 4096))
    np.testing.assert_array_equal(fractions, np.zeros_like(fractions))
    assert diagnostics["exact_empty_tetrahedra_by_conservative_aabb"] == 2
    assert diagnostics["partial_quadrature_tetrahedra"] == 0
    assert diagnostics["sample_points_evaluated_by_level"] == [0, 0]
    assert diagnostics["frozen_integrator_calls"] == 0


def test_skew_query_strictly_inside_one_tet_is_exact_full_without_sampling():
    heart_vertices = np.array(
        [[[1.0, 2.0, 3.0], [2.0, 2.2, 3.0], [1.2, 3.1, 3.3], [1.4, 2.1, 4.4]]]
    )
    heart = TetrahedralHeart(heart_vertices)
    weights = np.array(
        [[0.4, 0.2, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2], [0.2, 0.2, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4]]
    )
    query = np.array([weights @ heart_vertices[0]])
    fractions, diagnostics = integrate_individual_fractions_fast(vertices_m=query, heart=heart, levels=(512, 4096))
    np.testing.assert_array_equal(fractions, np.ones_like(fractions))
    assert diagnostics["exact_full_tetrahedra_by_strict_same_tet"] == 1
    assert diagnostics["sample_points_evaluated_by_level"] == [0, 0]


def test_tiny_intersection_survives_when_query_centroid_is_outside():
    heart = TetrahedralHeart(np.array([tet()]))
    query = np.array([[[0.2, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.0, 0.0], [1.2, 0.0, 1.0]]])
    assert np.linalg.norm(query.mean(axis=1)[0] - heart.centre_m) > 0.5
    fractions, diagnostics = integrate_individual_fractions_fast(query, heart, levels=(64, 512))
    assert diagnostics["exact_empty_tetrahedra_by_conservative_aabb"] == 0
    assert diagnostics["partial_quadrature_tetrahedra"] == 1
    assert fractions[-1, 0] > 0.0


def test_inverse_translation_and_scale_match_frozen_for_partial_rows():
    original = np.array([tet(), tet(origin=(3.0, 0.0, 0.0))])
    heart = TetrahedralHeart(original)
    vf = 0.728
    translation = np.array([0.4, -0.25, 0.7])
    scale = np.cbrt(1.0 + vf)
    moved = heart.centre_m + scale * (original - heart.centre_m) + translation
    barycentric = uniform_tet_barycentric(128)
    old, _ = integrate_individual_fractions(
        moved, heart, volume_fraction=vf, translation_m=translation, levels=(32, 128), barycentric=barycentric
    )
    new, _ = integrate_individual_fractions_fast(
        moved, heart, volume_fraction=vf, translation_m=translation, levels=(32, 128), barycentric=barycentric
    )
    np.testing.assert_array_equal(new, old)


def test_invalid_chunk_size_is_rejected():
    heart = TetrahedralHeart(np.array([tet()]))
    with pytest.raises(ValueError, match="chunk_size"):
        integrate_individual_fractions_fast(np.array([tet()]), heart, chunk_size=0)

