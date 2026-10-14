import importlib.util
from pathlib import Path
import copy
import unittest
import tempfile
import json

FILE=Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools/georg_quality_comparison.py'
spec=importlib.util.spec_from_file_location('separated_controls',FILE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def control(z):
    return dict(record_ids=['cross'],axes=['longitudinal'],states=['inhale'],
        parameters=[0]*9,pose={'centre':[1,2,3]},target_Z_ohm=[100],fine_Z_ohm=[z],
        identity={k:'same' for k in ('physical_contacts_sha256','reference_sha256',
            'observations_sha256','physical_montage_helper_sha256')},
        patches=[dict(record_id='cross',xyz=[[1,2,3]],gids=[1],
            electrical_groups=1,physical_patches=1)],
        patch_qc=dict(contacts=[dict(row=1,contact=1,group=1,area_mm2=19.6)]))


class SeparatedControls(unittest.TestCase):
    def test_total_cancellation_does_not_hide_failures(self):
        result=module.compare(control(100),control(102),control(100))
        self.assertTrue(result['total']['passed'])
        self.assertFalse(result['spatial']['passed'])
        self.assertFalse(result['fractions']['passed'])
        self.assertFalse(result['separated_comparison_passed'])

    def test_identical_physics_passes(self):
        result=module.compare(control(100),control(100),control(100))
        self.assertTrue(result['separated_comparison_passed'])
        self.assertFalse(result['global_mesh_convergence_proven'])

    def test_changed_pose_is_rejected(self):
        altered=control(100);altered['pose']['centre'][0]=2
        with self.assertRaisesRegex(ValueError,'pose'):
            module.compare(control(100),altered,control(100))

    def test_same_centres_cannot_hide_different_contact_area(self):
        altered=copy.deepcopy(control(100));altered['patch_qc']['contacts'][0]['area_mm2']+=.01
        with self.assertRaisesRegex(ValueError,'area'):
            module.compare(control(100),altered,control(100))

    def test_audit_rejects_a_different_executed_worker(self):
        with tempfile.TemporaryDirectory(dir=FILE.parents[2]/'.pytest_cache') as folder:
            base=Path(folder);worker=base/'worker.m';worker.write_text('new worker',encoding='utf-8')
            plan=base/'plan.local.json';plan.write_text(json.dumps(dict(sources=[])),encoding='utf-8')
            row=dict(identity=dict(worker_sha256='old worker',plan_sha256=module.sha(plan)))
            with self.assertRaisesRegex(ValueError,'worker'):
                module.verify_saved_identity(row,base,worker)

    def test_audit_rejects_a_different_executed_plan(self):
        with tempfile.TemporaryDirectory(dir=FILE.parents[2]/'.pytest_cache') as folder:
            base=Path(folder);worker=base/'worker.m';worker.write_text('same worker',encoding='utf-8')
            plan=base/'plan.local.json';plan.write_text(json.dumps(dict(sources=[])),encoding='utf-8')
            row=dict(identity=dict(worker_sha256=module.sha(worker),plan_sha256='wrong plan'))
            with self.assertRaisesRegex(ValueError,'plan'):
                module.verify_saved_identity(row,base,worker)


if __name__=='__main__':unittest.main()
