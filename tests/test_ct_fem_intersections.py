"""Disprove detector candidates only with geometric separation or adjacency."""
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ct_fem_surface_intersections import projected_pair_is_separate_or_adjacent, adjudicate


@pytest.mark.parametrize("second,shared,allowed", [
    ([[2, 2, 0], [2, .5, 1e-14], [.5, 2, -1e-14]], [], True),
    ([[.5, .5, 0], [2, .5, 0], [.5, 2, 0]], [], False),
    ([[.5, .5, -1], [.5, .5, 1], [1.5, .5, 0]], [], False),
    ([[2, 0, 0], [3, 0, 0], [2, 1, 0]], [], False),
    ([[2, 0, 0], [3, 0, 0], [2, 1, 0]], [[2, 0, 0]], True),
    ([[0, 0, 0], [2, 0, 0], [0, -2, 0]], [[0, 0, 0], [2, 0, 0]], True),
    ([[0, 0, 0], [2, 0, 0], [1, .5, 0]], [[0, 0, 0], [2, 0, 0]], False),
])
def test_projection_keeps_real_crossings_overlaps_and_unshared_touches_rejected(second, shared, allowed):
    first = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    assert bool(projected_pair_is_separate_or_adjacent(first, np.array(second, float),
        np.array(shared, float).reshape(-1, 3))) == allowed


def test_adjudication_does_not_move_or_repair_vertices():
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                         [2., 2., 0.], [2., .5, 1e-14], [.5, 2., -1e-14]])
    surface = trimesh.Trimesh(vertices, [[0, 1, 2], [3, 4, 5]], process=False)
    before = surface.vertices.copy()
    qc = adjudicate(surface, [0, 1])
    assert qc["passed"] and qc["dismissed_by_projection"] == 1
    assert np.array_equal(before, surface.vertices)
    assert not qc["coordinates_modified"] and qc["overlap_area_tolerance_mm2"] == 0


def test_adjudication_resource_limit_is_a_refusal():
    mesh = trimesh.creation.box()
    assert not adjudicate(mesh, np.zeros(5001, int))["passed"]
