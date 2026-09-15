import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_temporal_models import predict_state, evaluate


class TemporalModelTests(unittest.TestCase):
    def test_recovers_unseen_harmonic_state(self):
        t = np.array([.0, .1, .3, .55, .8, .95])
        def truth(x):
            return np.array([np.log(120000)+.15*np.sin(2*np.pi*x), 15+2*np.cos(2*np.pi*x), -4+np.sin(2*np.pi*x), 2+.5*np.cos(2*np.pi*x)])
        y = np.array([truth(x) for x in t])
        pred, _ = predict_state(t, y, .4, 'one_harmonic')
        np.testing.assert_allclose(pred, truth(.4), atol=1e-12)

    def test_rejects_extrapolation_and_under_supported_harmonic(self):
        with self.assertRaises(ValueError):
            predict_state([0,.5,1], np.zeros((3,4)), 1.1, 'linear_neighbors')
        with self.assertRaises(ValueError):
            predict_state([0,.5,1], np.zeros((3,4)), .3, 'one_harmonic')

    def test_holdout_excludes_target_and_keeps_cycles_separate(self):
        records=[]
        for cycle in range(2):
            for i in range(6):
                records.append({'subject':'synthetic','cycle_index':cycle,'region':'heart',
                    'phase_id':f'cycle{cycle}_phase{i}', 'phase_percent_within_cycle':i*15,
                    'moments':{'volume_ml':100+cycle*100, 'volume_mm3':100000+cycle*100000,
                               'centroid_mm':[cycle*20+i,0,0]}})
        result=evaluate(records[::-1])
        self.assertEqual(len(result['predictions']), 24)
        for row in result['predictions']:
            self.assertNotIn(row['phase_id'], row['training_phase_ids'])
            self.assertTrue(all(p.startswith(f"cycle{row['cycle_index']}_") for p in row['training_phase_ids']))
            self.assertAlmostEqual(row['volume_error_ml'], 0., places=8)
            if row['method']=='linear_neighbors':
                self.assertAlmostEqual(row['centroid_error_mm'], 0., places=10)
        self.assertEqual(result['skipped'], [])


if __name__=='__main__':
    unittest.main()
