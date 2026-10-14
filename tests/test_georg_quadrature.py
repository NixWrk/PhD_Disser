"""Homogeneous-box shortcuts must preserve arbitrary tissue interfaces."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
from georg_lateral_quadrature import uniform_boxes
from georg_lateral_search_mesh import search_edges


def test_certified_and_mixed_boxes():
    labels = np.zeros((7, 8, 9), np.uint8); labels[2:5, 2:6, 2:7] = 4
    low = np.array([[2, 2, 2], [0, 0, 0], [1, 1, 1], [-10, -10, -10]])
    high = np.array([[4, 5, 6], [1, 1, 1], [4, 5, 6], [-3, -3, -3]])
    np.testing.assert_array_equal(uniform_boxes(labels, low, high), [4, 0, -1, 0])


def test_outside_ct_is_background_and_cannot_certify_fat():
    labels = np.full((3, 4, 5), 4, np.uint8)
    low = np.array([[-1, 0, 0], [0, 0, 0], [5, 7, 9]])
    high = np.array([[1, 2, 3], [2, 3, 4], [8, 9, 11]])
    np.testing.assert_array_equal(uniform_boxes(labels, low, high), [-1, 4, 0])


def test_all_certificates_match_exhaustive_voxels():
    rng = np.random.default_rng(42); labels = rng.integers(0, 5, (5, 6, 7), dtype=np.uint8)
    low = rng.integers(-2, 6, (80, 3)); high = low+rng.integers(0, 4, (80, 3))
    result = uniform_boxes(labels, low, high)
    for lo, hi, material in zip(low, high, result):
        points = np.stack(np.meshgrid(*(np.arange(a, b+1) for a, b in zip(lo, hi)), indexing='ij'), -1).reshape(-1, 3)
        inside = ((points >= 0) & (points < labels.shape)).all(1)
        values = np.zeros(len(points), np.uint8); q = points[inside]
        values[inside] = labels[q[:, 0], q[:, 1], q[:, 2]]
        expected = int(values[0]) if np.all(values == values[0]) else -1
        assert material == expected


def test_search_refines_lateral_wall_and_preserves_artificial_caps():
    nodes=np.array([[-1,0,0],[-1,3,0],[-2,0,0],
                    [-1,0,1],[-1,3,1],[-1,0,3],
                    [-8,0,1],[-8,3,1],[-8,0,3]],float)
    faces=np.array([[0,1,2],[3,4,5],[6,7,8]])
    bounds=np.array([[-2,-5,0],[2,5,4]],float)
    edges=search_edges(nodes,faces,bounds,1.)
    assert set(map(tuple,edges))=={(3,4),(3,5),(4,5)}
