import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from georg_inverse_residual_diagnostics import analyse


class PairDiagnostics(unittest.TestCase):
    def fixture(self):
        observations = pd.DataFrame(dict(record_id=[f'record_{i}' for i in range(12)]*2,
            axis=['longitudinal']*24, state=['inhale']*12+['exhale']*12,Z_ohm=[50.]*24))
        jacobian = np.zeros((24,6));jacobian[:12,0]=1;jacobian[12:,3]=1
        values = np.r_[np.full(12,51.),np.full(12,53.)]
        row = dict(status='numerical_candidate_not_validated',parameters=[0.]*6,
            predicted_Z_ohm=values,target_Z_ohm=[50.]*24,jacobian_logrho=jacobian,
            own_transfer_ohm=values,counter_transfer_ohm=[7.]*24)
        return row,observations,[[.1,100.]]*3

    def test_independent_known_pair_decomposition_and_bounded_contrast(self):
        result = analyse(*self.fixture())
        self.assertAlmostEqual(result['SSE_ohm2'],120.)
        self.assertAlmostEqual(result['mean_SSE_ohm2'],96.)
        self.assertAlmostEqual(result['contrast_SSE_ohm2'],24.)
        self.assertAlmostEqual(result['contrast_SSE_fraction'],.2)
        linear=result['bounded_linear_contrast_diagnostic']
        self.assertLess(linear['contrast_RMSE_ohm'],1e-8)
        self.assertFalse(linear['nonlinear_feasibility_or_impossibility_proven'])
        self.assertFalse(result['quantitative_result_accepted'])
        self.assertTrue(all(not s['resistivities_or_pose_refitted'] for s in result['source_scenarios']))

    def test_duplicate_phase_or_changed_observation_is_rejected(self):
        for defect in ('duplicate_phase','changed_target'):
            row,observations,bounds=self.fixture()
            if defect=='duplicate_phase':observations.loc[12,'state']='inhale'
            else:row['target_Z_ohm'][0]=49.
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                analyse(row,observations,bounds)

    def test_out_of_bounds_or_unfinished_candidate_is_rejected(self):
        for defect in ('outside_bounds','unfinished'):
            row,observations,bounds=self.fixture()
            if defect=='outside_bounds':row['parameters'][0]=np.log(101.)
            else:row['status']='forward_calculation_running'
            with self.subTest(defect=defect),self.assertRaises(ValueError):
                analyse(row,observations,bounds)


if __name__=='__main__':unittest.main()
