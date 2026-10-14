from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MATLAB_TRKG4_real_subjects/tools'))
from georg_lateral_candidate_mesh import material_fractions, boundary_has_split
from georg_author_search_pipeline import mesh_comparison


class CandidateChecks(unittest.TestCase):
    def test_nested_prefix_is_independent_of_final_point_count(self):
        with tempfile.TemporaryDirectory() as name:
            seg = Path(name)
            shape = (8, 8, 8)
            mask = np.zeros(shape, np.uint8)
            mask[4:] = 1
            for tissue in ['static_ct', 'fat', 'heart', 'bone', 'lung']:
                a = mask if tissue == 'lung' else np.zeros(shape, np.uint8)
                nib.save(nib.Nifti1Image(a, np.eye(4)), seg / (tissue + '.nii.gz'))
            nodes = -np.array([[2., 2., -2.], [6., 2., -2.], [6., 6., -2.], [6., 2., -6.]])
            cells = np.array([[0, 1, 2, 3]])
            a = material_fractions(nodes, cells, seg, (64, 256))
            b = material_fractions(nodes, cells, seg, (64, 1024))
            np.testing.assert_array_equal(a[64], b[64])
            self.assertEqual(float(b[1024].sum()), 1.)
            self.assertGreater(float(b[1024][0, 1]), .1)
            self.assertLess(float(b[1024][0, 1]), .99)

    def test_interior_edge_does_not_change_exterior_triangle(self):
        boundary = np.array([[0, 1, 2], [0, 2, 3]])
        self.assertFalse(boundary_has_split(boundary, np.array([[4, 5]]), 6))
        self.assertTrue(boundary_has_split(boundary, np.array([[0, 2]]), 6))

    def test_spatial_failure_is_preserved(self):
        a = dict(identity=dict(model_plan_sha256='coarse'), fine_Z_ohm=[100., 60.])
        b = dict(identity=dict(model_plan_sha256='fine'), fine_Z_ohm=[101., 60.01])
        r = mesh_comparison(a, b)
        self.assertFalse(r['all_passed'])
        self.assertEqual(r['failed_values'], 1)
        self.assertFalse(r['global_convergence_proven'])


if __name__ == '__main__':
    unittest.main()
