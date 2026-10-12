"""Measured field selection, excluded-size predictions and full paired CEM."""
from pathlib import Path
import json
import re
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VIEW=ROOT/'Colab Notebooks/scorpion_v2'
sys.path.insert(0,str(VIEW))
from lateral_pilot import DiskModel,read,sha
from lateral_pilot_revision import OUT as PILOT,fit_grid
from scorpion_comparisons import OUT as COMPARISONS
from scorpion_protocol import channel_assignment,ttrkg_configuration


class CorrectedScorpion(unittest.TestCase):
    def test_confirmation_does_not_resolve_the_late_variant(self):
        for number in range(12,19):
            roles=channel_assignment(f'SCV2-G-{number:03d}')
            self.assertEqual(roles['physical_channel_roles'],{'1':'ttrkg','2':'tepc' if number<18 else 'aepc'})
            self.assertEqual(roles['ttrkg']['status'],'author_confirmed')
        for number in range(7,11):
            for variant in ['1x1','1x4']:
                config=ttrkg_configuration(f'SCV2-G-{number:03d}',variant)
                self.assertEqual(config['variant'],variant)
                self.assertEqual(config['status'],'unresolved_hypothesis')

    @unittest.skipUnless((COMPARISONS/'ecg_evidence.local.json').is_file(),
        'Requires local signal-processing outputs; raw medical data stay outside Git')
    def test_ensemble_units_and_annotations_keep_their_meaning(self):
        evidence=read(COMPARISONS/'ecg_evidence.local.json')
        self.assertTrue(all('median_mV' in r['ensemble'] for r in evidence))
        self.assertTrue(all('median_ohm' not in r['ensemble'] for r in evidence))
        contract=read(COMPARISONS/'contract.local.json')
        annotation=ROOT/'Experiments/scorpion_v2_georg/annotations/scorpion_review/annotations_candidates.local.json'
        self.assertEqual(sha(annotation),contract['input_contract']['annotations_sha256'])
        self.assertFalse(contract['deep_hold_annotations_accepted'])
        self.assertFalse(contract['late_block_variant_identified'])

    @unittest.skipUnless((PILOT/'results.local.json').is_file(), 'Requires local corrected-pilot outputs')
    def test_corrected_pilot_uses_side_fields_not_arm_fields(self):
        results=read(PILOT/'results.local.json')
        for item in results:
            self.assertEqual(item['channels'],[1,1,1,1,2])
            self.assertEqual(item['record_ids'][-1],'SCV2-G-'+item['reference_140_record'])
            self.assertEqual(item['sizes_mm'],[60,80,100,120,140])
            for state,fit in item['best_on_grid']['fits'].items():
                levels=__import__('pandas').read_csv(COMPARISONS/'measured_levels.local.csv')
                actual=[levels[(levels.record_id==rid)&(levels.channel==channel)&(levels.state==state)].baseline_raw_median_ohm.iloc[0]
                    for rid,channel in zip(item['record_ids'],item['channels'])]
                np.testing.assert_allclose(fit['observed_ohm'],actual,atol=1e-10)

    @unittest.skipUnless((PILOT/'results.local.json').is_file(), 'Requires local corrected-pilot outputs')
    def test_withheld_predictions_reconstruct_from_training_parameters(self):
        for item in read(PILOT/'results.local.json'):
            for fold in item['withheld']:
                self.assertNotIn(fold['size_mm'],fold['training_sizes_mm'])
                model=DiskModel(item['sizes_mm'],fold['h_mm'])
                z,J=model.evaluate(fold['rho_ohm_m'])
                i=item['sizes_mm'].index(fold['size_mm'])
                self.assertAlmostEqual(z[i],fold['baseline_predicted_ohm'],places=10)
                if 'pulse_fractional_changes' in fold:
                    predicted=J[i]@np.array(fold['pulse_fractional_changes'])
                    np.testing.assert_allclose(predicted,fold['pulse_predicted_ohm'],atol=1e-12)

    def test_known_synthetic_state_predicts_unseen_size(self):
        sizes=np.array([60.,80.,100.,120.,140.]);truth=np.array([4.,12.])
        data=DiskModel(sizes,20).evaluate(truth)[0]
        use=np.array([True,False,True,True,True])
        _,best=fit_grid({'inhale':data[use],'exhale':data[use]},sizes[use],np.array([[1.,25.],[1.,40.]]),[5.,20.,60.])
        self.assertEqual(best['h_mm'],20.)
        fit=best['fits']['inhale']
        prediction=DiskModel([80],best['h_mm']).evaluate([fit['rho1_ohm_m'],fit['rho2_ohm_m']])[0][0]
        self.assertAlmostEqual(prediction,data[1],places=7)

    @unittest.skipUnless((ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_paired_montages_20261012/h6/forward_results.local.json').is_file(),
        'Requires local individual FEM outputs')
    def test_paired_operator_is_reciprocal_with_correct_material_indices(self):
        folder=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_paired_montages_20261012'
        plan=read(folder/'plan.local.json');forward=read(folder/'h6/forward_results.local.json')
        self.assertEqual(plan['free_material_indices_matlab'],[1,2,4,5])
        self.assertEqual(plan['fixed_material_index_matlab'],3)
        self.assertEqual(len(forward['cases']),17)
        for item in forward['cases']:
            self.assertTrue(item['contact_qc_passed'])
            T=np.array(item['port_transfer_ohm']);J=np.array(item['J_logrho_port_ohm'])
            self.assertEqual(J.shape,(4,4,5))
            np.testing.assert_allclose(T,T.T,atol=1e-7)
            np.testing.assert_allclose(J,J.swapaxes(0,1),atol=1e-7)
            self.assertFalse(item['quantitative_inversion_allowed'])

    def test_supervisor_keeps_details_and_removes_full_record_arrays(self):
        path=VIEW/'10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2_для_руководителя.html'
        page=path.read_text(encoding='utf-8');marker='const experimentPayload='
        payload,_=json.JSONDecoder().raw_decode(page.split(marker,1)[1])
        self.assertEqual(len(payload['records']),18)
        self.assertFalse(bool(re.search(r'<div[^>]*class="[^\"]*jp-CodeMirrorEditor',page)),
            'Reader export must omit source editors; rendered Markdown uses input-area wrappers too')
        self.assertFalse('from scorpion_comparisons import comparison_figure' in page,
            'The notebook Python source must not occur in the reader')
        self.assertFalse('<option value="full">' in page)
        for record in payload['records']:
            self.assertNotIn('full_samples',record)
            self.assertGreater(record['original_sample_rows'],len(record['detail_t']))
            self.assertEqual(len(record['detail_t']),5*record['fs'])


if __name__=='__main__':unittest.main()
