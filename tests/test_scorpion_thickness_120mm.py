"""Fixed deep-layer recovery, immutable overlap and excluded-size forecasts."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Colab Notebooks/scorpion_v2'))
from lateral_thickness_extension import OUT,SelectedResponse,SCENARIOS
from lateral_pilot import DiskModel,fit_state,invert_pulse,read
from lateral_pilot_revision import OUT as PREVIOUS


class ThicknessExtension(unittest.TestCase):
    def test_fixed_110mm_synthetic_state_predicts_unseen_size(self):
        model=DiskModel([60,80,100,120,140],110)
        rho=np.array([4.,12.]);z,j=model.evaluate(rho)
        phase=np.arange(200)/200
        fraction=np.array([.001*np.sin(2*np.pi*phase),.02*np.cos(2*np.pi*phase)])
        wave=j@fraction;use=np.array([0,1,3,4]);train=SelectedResponse(model,use)
        fit=fit_state(train,z[use],np.array([[1,25],[1,40]]))
        pulse=invert_pulse(train,fit,wave[use])
        fitted_rho=np.array([fit['rho1_ohm_m'],fit['rho2_ohm_m']])
        _,jf=model.evaluate(fitted_rho)
        predicted=jf[2]@(np.array(pulse['delta_rho_ohm_m'])/fitted_rho[:,None])
        np.testing.assert_allclose(predicted,wave[2],atol=1e-8)

    @unittest.skipUnless((OUT/'results.local.json').is_file() and (PREVIOUS/'results.local.json').is_file(),
        'Requires local processed Georg signals')
    def test_overlap_retains_previous_baseline_data_and_fits(self):
        current=read(OUT/'results.local.json');old=read(PREVIOUS/'results.local.json')
        for case in current:
            previous=next(r for r in old if r['reference_140_record']==case['reference_140_record'])
            self.assertEqual(case['record_ids'],previous['record_ids'])
            self.assertEqual(case['channels'],previous['channels'])
            self.assertEqual([p['h_mm'] for p in case['profiles']],list(np.arange(5,121,5)))
            for before in previous['profiles']:
                after=next(p for p in case['profiles'] if p['h_mm']==before['h_mm'])
                for state in ['inhale','exhale']:
                    np.testing.assert_array_equal(before['fits'][state]['observed_ohm'],after['fits'][state]['observed_ohm'])
                    for key in ['rho1_ohm_m','rho2_ohm_m','rmse_ohm']:
                        self.assertAlmostEqual(before['fits'][state][key],after['fits'][state][key],places=8)

    @unittest.skipUnless((OUT/'results.local.json').is_file(), 'Requires local thickness-extension outputs')
    def test_deep_scenarios_and_forecasts_use_four_training_sizes(self):
        for case in read(OUT/'results.local.json'):
            for key,scenario in case['scenarios'].items():
                low,high=SCENARIOS[key][1:]
                self.assertGreaterEqual(scenario['best']['h_mm'],low)
                self.assertLessEqual(scenario['best']['h_mm'],high)
                for fold in scenario['withheld']:
                    self.assertNotIn(fold['size_mm'],fold['training_sizes_mm'])
                    self.assertEqual(len(fold['training_sizes_mm']),4)
                    self.assertGreaterEqual(fold['h_mm'],low)
                    self.assertLessEqual(fold['h_mm'],high)
                    model=DiskModel(case['sizes_mm'],fold['h_mm'])
                    z,j=model.evaluate(fold['rho_ohm_m']);index=case['sizes_mm'].index(fold['size_mm'])
                    self.assertAlmostEqual(z[index],fold['baseline_predicted_ohm'],places=9)
                    prediction=j[index]@np.array(fold['pulse_fractional_changes'])
                    np.testing.assert_allclose(prediction,fold['pulse_predicted_ohm'],atol=1e-12)


if __name__=='__main__':unittest.main()
