"""Real hashes of tiny temporary provenance files; no mesh/NIfTI parsing or FEM."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from georg_relocated_contact_refinement import verify_reference_transfer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


class ReferenceTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source_dir = self.root/'fitted_model'
        self.parent = self.root/'new_numerical_parent'
        self.source_dir.mkdir()
        self.parent.mkdir()
        self.source_masks, self.parent_masks = [], []
        for directory, paths in ((self.source_dir, self.source_masks), (self.parent, self.parent_masks)):
            for name in ('static_ct', 'fat', 'heart', 'bone', 'lung'):
                path = directory/(name + '.nii.gz')
                path.write_bytes(('synthetic identity bytes for ' + name).encode('ascii'))
                paths.append(path)
        # The .mat/.nii.gz suffixes represent provenance roles only; these
        # tiny fixtures are deliberately never parsed as numerical/medical data.
        self.pose = self.source_dir/'pose_master.mat'
        self.pose.write_bytes(b'unchanged synthetic pose master')
        self.source_mesh = self.source_dir/'mesh.mat'
        self.source_mesh.write_bytes(b'old numerical grid')
        self.parent_mesh = self.parent/'mesh.mat'
        self.parent_mesh.write_bytes(b'new numerical grid, different bytes')
        self.source_contacts = self.source_dir/'contacts.json'
        self.parent_contacts = self.parent/'contacts.json'
        for path in (self.source_contacts, self.parent_contacts):
            write_json(path, dict(coordinate_system='LPS', diameter_mm=5,
                                  arm_physical_centres_lps_mm=[[1, 2, 3]]))
        self.source_ref = self.source_dir/'reference_materials.local.json'
        self.parent_ref = self.parent/'reference_materials.local.json'
        for path in (self.source_ref, self.parent_ref):
            write_json(path, dict(frequency_hz=100000, bone_ohm_m=48.0977, heart_ohm_m=4.6488))
        self.geometry = self.root/'same_outer_geometry'
        self.geometry.mkdir()
        physical = dict(frequency_hz=100000,
                        material_order=['soft_background', 'lung', 'bone', 'heart', 'fat'],
                        contact_impedance_ohm_m2=0.01, source_rule='own_drive')
        self.source = dict(physical, pose_reference_mesh=str(self.pose),
                           contacts_file=str(self.source_contacts), geometry_directory=str(self.geometry),
                           numerical_quadrature_points=[4096, 16384], integration_seed=20261008,
                           mesh_file=str(self.source_mesh),
                           sources=self.rows([*self.source_masks, self.pose, self.source_contacts,
                                              self.source_ref, self.source_mesh]))
        self.template = dict(copy.deepcopy(physical), pose_reference_mesh=str(self.pose),
                             contacts_file=str(self.parent_contacts), geometry_directory=str(self.geometry),
                             numerical_quadrature_points=[1024, 4096], integration_seed=20261015,
                             mesh_file=str(self.parent_mesh),
                             sources=self.rows([*self.parent_masks, self.parent_contacts,
                                                self.parent_ref, self.parent_mesh]))
        self.lineages = dict(pose_sha256=digest(self.pose))
        self.plan = self.source_dir/'plan.local.json'
        self.candidate = dict(parameters=[0, 1, 2, 0.1, 1.1, 2.1, -10, -5, -2.5],
                              pose={'centre_lps_mm': [1, 2, 3]},
                              identity=dict(pose_reference_mesh_sha256=digest(self.pose),
                                            contact_file_sha256=digest(self.source_contacts)))
        self.repin_source_plan()

    @staticmethod
    def rows(paths):
        return [dict(path=str(path), sha256=digest(path)) for path in paths]

    def repin_source_plan(self):
        write_json(self.plan, self.source)
        self.candidate['identity']['model_plan_sha256'] = digest(self.plan)

    def verify(self):
        return verify_reference_transfer(self.candidate, self.parent, self.template, self.lineages, self.plan)

    def test_same_physical_inputs_allow_different_mesh_quadrature_and_seed(self):
        self.assertNotEqual(digest(self.source_mesh), digest(self.parent_mesh))
        self.assertNotEqual(self.source['numerical_quadrature_points'], self.template['numerical_quadrature_points'])
        self.assertNotEqual(self.source['integration_seed'], self.template['integration_seed'])
        before = copy.deepcopy((self.candidate, self.template, self.lineages))
        files = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        result = self.verify()
        self.assertFalse(result['fitted_in_current_parent_model'])
        self.assertFalse(result['predictions_or_Jacobian_transferred'])
        self.assertTrue(result['numerical_mesh_and_quadrature_may_differ'])
        self.assertEqual(result['source_model_plan_sha256'], digest(self.plan))
        self.assertEqual(result['source_model_plan'], str(self.plan.resolve()))
        self.assertEqual(set(result['additional_sources']),
                         {self.plan, *(Path(row['path']) for row in self.source['sources'])})
        self.assertEqual(before, (self.candidate, self.template, self.lineages))
        self.assertEqual(files, {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_wrong_candidate_source_plan_identity_rejected(self):
        self.candidate['identity']['model_plan_sha256'] = hashlib.sha256(b'not the fitted model').hexdigest()
        with self.assertRaisesRegex(ValueError, 'actual fitted numerical model'):
            self.verify()

    def test_actual_source_plan_change_with_stale_candidate_hash_rejected(self):
        changed = copy.deepcopy(self.source)
        changed['numerical_quadrature_points'] = [2048, 8192]
        write_json(self.plan, changed)
        with self.assertRaisesRegex(ValueError, 'actual fitted numerical model'):
            self.verify()

    def test_changed_source_dependency_bytes_rejected_despite_unchanged_plan(self):
        self.source_mesh.write_bytes(b'corrupted fitted-model dependency')
        with self.assertRaisesRegex(ValueError, 'Reference dependency changed'):
            self.verify()

    def test_each_of_five_actual_source_mask_changes_rejected(self):
        for path in self.source_masks:
            with self.subTest(mask=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b' changed')
                    with self.assertRaisesRegex(ValueError, 'Reference dependency changed'):
                        self.verify()
                finally:
                    path.write_bytes(original)

    def test_different_parent_mask_with_its_actual_updated_hash_rejected(self):
        for index, path in enumerate(self.parent_masks):
            with self.subTest(mask=path.name):
                original = path.read_bytes()
                row = self.template['sources'][index]
                old_hash = row['sha256']
                try:
                    path.write_bytes(original + b' different segmentation')
                    row['sha256'] = digest(path)
                    with self.assertRaisesRegex(ValueError, 'CT/tissue masks differ'):
                        self.verify()
                finally:
                    path.write_bytes(original)
                    row['sha256'] = old_hash

    def test_incomplete_or_conflicting_mask_inventory_rejected(self):
        original = copy.deepcopy(self.template['sources'])
        self.template['sources'] = original[1:]
        with self.assertRaisesRegex(ValueError, 'Incomplete CT'):
            self.verify()
        conflicting = dict(original[0], sha256=hashlib.sha256(b'other CT').hexdigest())
        self.template['sources'] = original + [conflicting]
        with self.assertRaisesRegex(ValueError, 'Ambiguous tissue mask'):
            self.verify()

    def test_different_pose_master_rejected(self):
        other_pose = self.parent/'other_pose.mat'
        other_pose.write_bytes(b'another pose master')
        self.lineages['pose_sha256'] = digest(other_pose)
        with self.assertRaisesRegex(ValueError, 'geometry, contacts or fixed materials changed'):
            self.verify()

    def test_changed_actual_parent_contact_inventory_rejected(self):
        write_json(self.parent_contacts, dict(coordinate_system='LPS', diameter_mm=6,
                                             arm_physical_centres_lps_mm=[[1, 2, 3]]))
        with self.assertRaisesRegex(ValueError, 'geometry, contacts or fixed materials changed'):
            self.verify()

    def test_changed_fixed_bone_or_heart_rho_reference_rejected(self):
        original = json.loads(self.parent_ref.read_text(encoding='utf-8'))
        for key in ('bone_ohm_m', 'heart_ohm_m'):
            with self.subTest(key=key):
                changed = dict(original)
                changed[key] += 0.01
                write_json(self.parent_ref, changed)
                with self.assertRaisesRegex(ValueError, 'geometry, contacts or fixed materials changed'):
                    self.verify()

    def test_physical_assumptions_cannot_be_relabelled_as_numerical_changes(self):
        for field, changed in (('frequency_hz', 50000), ('material_order', ['lung', 'fat']),
                               ('contact_impedance_ohm_m2', 0.02), ('source_rule', 'counter_drive')):
            with self.subTest(field=field):
                original = self.template[field]
                try:
                    self.template[field] = changed
                    with self.assertRaisesRegex(ValueError, 'physical assumption changed: ' + field):
                        self.verify()
                finally:
                    self.template[field] = original

    def test_outer_geometry_origin_change_rejected(self):
        self.template['geometry_directory'] = str(self.root/'other_outer_geometry')
        with self.assertRaisesRegex(ValueError, 'outer geometry origin changed'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
