"""The voxel-cell convention and physical transform are essential to this pilot."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_cardiac_cycle_materials import sample_mask


def test_binary_voxel_cells_and_outside_domain():
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, 2, 3] = True
    points = np.array([[1, 2, 3], [.50001, 2, 3], [1.49999, 2, 3],
                       [.49999, 2, 3], [1.50001, 2, 3], [-1, 2, 3], [1, 2, 8]])
    assert sample_mask(points, mask, np.eye(4)).tolist() == [True, True, True, False, False, False, False]


def test_rigid_axes_and_mm_to_m_preserve_membership():
    mask = np.zeros((3, 4, 5), dtype=bool); mask[1, 2, 3] = True
    affine = np.array([[0, -.0007, 0, .205], [-.0007, 0, 0, .204],
                       [0, 0, .000625, -.237], [0, 0, 0, 1.]])
    points = np.array([[1, 2, 3], [0, 2, 3]])
    physical = points @ affine[:3, :3].T + affine[:3, 3]
    assert sample_mask(physical.reshape(1, 2, 3), mask, np.linalg.inv(affine)).tolist() == [[True, False]]
