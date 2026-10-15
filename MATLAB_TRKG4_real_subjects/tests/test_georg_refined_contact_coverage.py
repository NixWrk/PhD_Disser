"""Small saved-geometry checks; no FEM, MATLAB or real input arrays."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import georg_refined_contact_coverage as coverage


class FineContactCoverage(unittest.TestCase):
    def setUp(self):
        # Five independent triangles with areas 10, 1, 2, 3, 4 mm2.
        # Parent 1 has one child; parent 2 has four, independent of selections.
        self.nodes = np.array([point for i, area in enumerate([10, 1, 2, 3, 4])
                               for point in ([0, 0, i], [2*area, 0, i], [0, 1, i])], float)
        self.boundary = np.arange(1, 16).reshape(5, 3)
        self.mapping = np.array([[1, 2, 2, 2, 2]])
        self.forward = dict(patches=[dict(record_id='synthetic', faces=[[1, 4]], gids=[1],
                                          physical_patches=1, electrical_groups=1, xyz=[[0, 0, 0]])])

    def compute(self, **changes):
        values = dict(forward=self.forward, nodes_mm=self.nodes, boundary=self.boundary,
                      boundary_parent=self.mapping)
        values.update(changes)
        return coverage.fine_patch_coverage(**values)

    def test_actual_fine_indices_and_unequal_area_weights(self):
        row = self.compute()['contacts'][0]
        self.assertEqual(row['fine_face_count'], 2)
        self.assertEqual(row['refined_fine_face_count'], 1)
        self.assertEqual(row['area_mm2'], 13)
        self.assertEqual(row['refined_area_mm2'], 3)
        self.assertAlmostEqual(row['refined_boundary_area_fraction'], 3/13)
        self.assertFalse(row['all_faces_refined'])

    def test_children_outside_selected_patch_still_qualify(self):
        self.forward['patches'][0]['faces'] = [5]
        result = self.compute()
        self.assertTrue(result['all_contact_faces_refined'])
        self.assertEqual(result['contacts'][0]['maximum_parent_child_count'], 4)
        self.assertEqual(result['refined_boundary_area_fraction'], 1)
        self.assertFalse(result['volume_refinement_proven'])
        self.assertFalse(result['global_mesh_convergence_proven'])

    def test_description_and_total_ratios_are_area_weighted(self):
        self.forward['patches'][0].update(faces=[[1], [4]], gids=[1, 1],
                                          physical_patches=2, xyz=[[0, 0, 0]]*2)
        result = self.compute()
        self.assertEqual([row['refined_boundary_area_fraction'] for row in result['contacts']], [0, 1])
        self.assertAlmostEqual(result['descriptions'][0]['refined_boundary_area_fraction'], 3/13)
        self.assertAlmostEqual(result['refined_boundary_area_fraction'], 3/13)
        self.assertFalse(result['all_contact_faces_refined'])

    def test_repeated_conditions_and_group_ties_are_not_deduplicated(self):
        # A description with 28 + 12 physical contacts, all tied to group 1,
        # repeated in a second condition. Counts are descriptions, not uniques.
        record = self.forward['patches'][0]
        record.update(faces=[[2]]*40, gids=[1]*40, physical_patches=40, xyz=[[0, 0, 0]]*40)
        self.forward['patches'].append(copy.deepcopy(record))
        result = self.compute()
        self.assertEqual(result['physical_contact_description_count'], 80)
        self.assertEqual(result['physical_description_count'], 2)
        self.assertEqual(result['fine_face_count'], 80)
        self.assertEqual(result['area_mm2'], 80)
        self.assertEqual([row['description_index'] for row in result['descriptions']], [1, 2])

    def test_row_and_column_mapping_give_same_result(self):
        self.assertEqual(self.compute(), self.compute(boundary_parent=self.mapping.T))

    def test_invalid_mapping_length_shape_base_and_values(self):
        for mapping in ([1, 2], [[1, 2], [2, 2]], [0, 1, 1, 1, 1],
                        [1, 3, 3, 3, 3], [1, 2.5, 2, 2, 2],
                        [1, np.nan, 2, 2, 2], [1, 6, 6, 6, 6]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.compute(boundary_parent=mapping)

    def test_invalid_selected_fine_face_indices(self):
        for value in ([], [0], [-1], [6], [1.1], [np.inf], [True], ['1'], [1, 1], [[1, 2], [3, 4]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bad = copy.deepcopy(self.forward)
                bad['patches'][0]['faces'] = [value]
                self.compute(forward=bad)

    def test_boundary_indices_are_one_based_and_integer(self):
        for value in (0, 16, 1.5, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                boundary = self.boundary.astype(float)
                boundary[0, 0] = value
                self.compute(boundary=boundary)

    def test_degenerate_or_nonfinite_area_and_coordinates_rejected(self):
        boundary = self.boundary.copy()
        boundary[0, 1] = boundary[0, 0]
        with self.assertRaisesRegex(ValueError, 'areas'):
            self.compute(boundary=boundary)
        nodes = self.nodes.copy()
        nodes[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            self.compute(nodes_mm=nodes)
        nodes = self.nodes.copy()
        nodes[0] = [-1e308, -1e308, 0]
        nodes[1] = [1e308, -1e308, 0]
        nodes[2] = [-1e308, 1e308, 0]
        with self.assertRaisesRegex(ValueError, 'areas'):
            self.compute(nodes_mm=nodes)

    def test_contact_metadata_alignment_rejected(self):
        for fields in (dict(gids=[1, 1]), dict(physical_patches=2),
                       dict(gids=[2]), dict(xyz=[[0, 0, np.nan]])):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                forward = copy.deepcopy(self.forward)
                forward['patches'][0].update(fields)
                self.compute(forward=forward)

    def test_missing_physical_descriptions_rejected(self):
        for forward in ({}, dict(patches=[])):
            with self.subTest(forward=forward), self.assertRaises(ValueError):
                self.compute(forward=forward)

    def write_inputs(self, root):
        files = [root/'forward.json', root/'mesh.mat', root/'parents.mat']
        savemat(files[1], dict(nodes_mm=self.nodes, boundary=self.boundary, elems=np.zeros((2, 4))))
        savemat(files[2], dict(boundary_parent=self.mapping, element_parent=np.zeros(7)))
        forward = copy.deepcopy(self.forward)
        forward['identity'] = dict(contact_inheritance=dict(current_mesh_sha256=coverage.sha(files[1])))
        files[0].write_text(json.dumps(forward), encoding='utf-8')
        return files

    def test_file_binding_and_only_required_mat_variables_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            files = self.write_inputs(Path(directory))
            with patch.object(coverage, 'loadmat', wraps=coverage.loadmat) as loader:
                result = coverage.compute_coverage(*files)
            self.assertEqual([call.kwargs['variable_names'] for call in loader.call_args_list],
                             [['nodes_mm', 'boundary'], ['boundary_parent']])
            self.assertTrue(result['forward_current_mesh_identity_verified'])
            self.assertEqual(result['forward_sha256'], coverage.sha(files[0]))
            self.assertEqual(result['current_mesh_sha256'], coverage.sha(files[1]))
            self.assertEqual(result['parent_maps_sha256'], coverage.sha(files[2]))
            self.assertAlmostEqual(result['refined_boundary_area_fraction'], 3/13)

    def test_stale_forward_current_mesh_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            files = self.write_inputs(Path(directory))
            forward = json.loads(files[0].read_text())
            forward['identity']['contact_inheritance']['current_mesh_sha256'] = 'stale'
            files[0].write_text(json.dumps(forward), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'mesh SHA'):
                coverage.compute_coverage(*files)

    def test_cli_fresh_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = self.write_inputs(root)
            out = root/'coverage.json'
            args = ['--forward', str(files[0]), '--mesh', str(files[1]),
                    '--parent-maps', str(files[2]), '--out', str(out)]
            with patch('builtins.print'):
                coverage.main(args)
            original = out.read_bytes()
            self.assertFalse(json.loads(original)['all_contact_faces_refined'])
            with patch.object(coverage, 'compute_coverage') as compute:
                with self.assertRaises(FileExistsError):
                    coverage.main(args)
                compute.assert_not_called()
            self.assertEqual(out.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
