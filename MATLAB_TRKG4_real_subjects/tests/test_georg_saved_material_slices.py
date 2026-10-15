"""Known phase-separated errors, without loading real results or running FEM."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Colab Notebooks/scorpion_v2'))
from build_georg_convergence_report import saved_material_slices


class SavedMaterialSlices(unittest.TestCase):
    def fixture(self):
        best=dict(status='numerical_candidate_not_validated',identity={'model':'same'},
            pose={'centre':[1,2,3]},parameters=[0.]*9,predicted_Z_ohm=[1.]*24,evaluation=1)
        return dict(best=best,target=np.zeros(24),
            observations=pd.DataFrame({'state':['inhale']*12+['exhale']*12}),checkpoints=[best])

    def test_opposite_phase_is_replaced_by_the_selected_fixed_phase(self):
        d=self.fixture();row=copy.deepcopy(d['best']);row['evaluation']=2
        row['parameters'][0]=np.log(2.);row['parameters'][3]=np.log(7.)
        row['predicted_Z_ohm']=[2.]*12+[100.]*12;d['checkpoints'].append(row)
        sample=next(r for r in saved_material_slices(d) if r['state']=='inhale' and r['material']==0 and r['rho']==2.)
        self.assertEqual(sample['phase_RSS'],48.)
        self.assertEqual(sample['conditional_MSE'],2.5)  # (12*2^2 + 12*1^2)/24
        self.assertEqual(sample['delta_MSE'],1.5)
        self.assertEqual(sample['source_evaluations'],[2])

    def test_other_material_pose_identity_or_unfinished_changes_are_excluded(self):
        for defect in ('other_material','pose','identity','unfinished'):
            with self.subTest(defect=defect):
                d=self.fixture();row=copy.deepcopy(d['best']);row['evaluation']=2
                row['parameters'][0]=np.log(2.);row['predicted_Z_ohm']=[2.]*24
                if defect=='other_material':row['parameters'][1]=1.
                elif defect=='pose':row['pose']={'centre':[2,2,3]}
                elif defect=='identity':row['identity']={'model':'different'}
                else:row['status']='forward_calculation_running'
                d['checkpoints'].append(row)
                values=[r['rho'] for r in saved_material_slices(d) if r['state']=='inhale' and r['material']==0]
                self.assertEqual(values,[1.])

    def test_repeated_same_phase_points_are_kept_as_provenance(self):
        d=self.fixture();row=copy.deepcopy(d['best']);row['evaluation']=3
        row['parameters'][3]=1.;row['predicted_Z_ohm']=[1.]*12+[80.]*12
        d['checkpoints'].append(row)
        sample=next(r for r in saved_material_slices(d) if r['state']=='inhale' and r['material']==0)
        self.assertEqual(sample['source_evaluations'],[1,3])
        self.assertEqual(sample['duplicate_phase_RSS_difference'],0.)
        self.assertEqual(sample['conditional_MSE'],1.)


if __name__=='__main__':unittest.main()
