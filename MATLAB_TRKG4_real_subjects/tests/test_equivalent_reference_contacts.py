"""Reference provenance may differ; physical inputs and their hashes may not."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from georg_quality_equivalent_reference_contacts import verify_reference_equivalence
from georg_quality_contact_variant import sha


class ReferenceEquivalence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=dict(parameters=[1.,2.],pose={'centre_lps_mm':[1.,2.,3.]},
            observations_sha256='observations',pose_reference_mesh_sha256='pose_mesh',
            common_parameters_not_fitted_here=True)
        self.refs=[copy.deepcopy(self.base),copy.deepcopy(self.base)]
        self.refs[1].update(selected_candidate_sha256='selection',common_parameters_not_fitted_here=False,
                            optimized_on_fine_remapped_mesh=True)
        self.files=[self.root/'a.json',self.root/'b.json']

    def tearDown(self):self.temp.cleanup()

    def bind(self):
        rows=[]
        for ref,file in zip(self.refs,self.files):
            file.write_text(json.dumps(ref),encoding='utf-8')
            rows.append(dict(parameters=ref['parameters'],pose=ref['pose'],
                identity=dict(reference_sha256=sha(file),observations_sha256=ref['observations_sha256'])))
        return rows

    def test_only_provenance_changes_are_allowed(self):
        result=verify_reference_equivalence(self.bind(),self.files)
        self.assertTrue(result['physical_reference_contents_equal_exactly'])
        self.assertNotEqual(*result['reference_sha256'])

    def test_changed_parameters_are_not_metadata(self):
        self.refs[1]['parameters'][0]+=1e-12
        with self.assertRaisesRegex(ValueError,'outside allowed'):
            verify_reference_equivalence(self.bind(),self.files)

    def test_pose_master_change_is_rejected(self):
        self.refs[1]['pose_reference_mesh_sha256']='another_pose_mesh'
        with self.assertRaisesRegex(ValueError,'outside allowed'):
            verify_reference_equivalence(self.bind(),self.files)

    def test_stale_reference_hash_is_rejected(self):
        rows=self.bind();self.files[1].write_text(json.dumps(self.base))
        with self.assertRaisesRegex(ValueError,'Actual reference hash'):
            verify_reference_equivalence(rows,self.files)


if __name__=='__main__':unittest.main()
