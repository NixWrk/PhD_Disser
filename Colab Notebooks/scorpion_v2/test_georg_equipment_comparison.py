"""Selection safety and source-locked reported-unit comparisons."""
import unittest
import numpy as np
from scipy.signal import butter, sosfiltfilt
import pandas as pd

import georg_equipment_comparison as E


class Selection(unittest.TestCase):
    def cycles(self):
        return [dict(start_r_s=float(i), stop_r_s=float(i + 1), rr_s=1.,
                     eligible_conservative=True) for i in range(8)]

    def test_gap_and_artifact_cannot_be_joined(self):
        cycles = self.cycles()
        cycles[3]['eligible_conservative'] = False
        self.assertEqual(E.choose_legacy_four(cycles, 0, 8), [4, 5, 6, 7, 8])
        del cycles[5]
        with self.assertRaises(ValueError):
            E.choose_legacy_four(cycles, 0, 8)

    def test_heart_rate_and_hold_boundaries_reject_invalid_blocks(self):
        cycles = self.cycles()[:4]
        cycles[1]['rr_s'] = 2.2
        with self.assertRaises(ValueError):
            E.choose_legacy_four(cycles, 0, 4)
        cycles[1]['rr_s'] = 1
        with self.assertRaises(ValueError):
            E.choose_legacy_four(cycles, .1, 4)


@unittest.skipUnless((E.OUT / 'contract.local.json').exists(), 'Local research data required')
class LocalSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = E.verify()
        cls.levels = E.level_frame()
        cls.segments = E.read(E.OUT / 'segments.local.json')

    def test_only_georg_and_target_sizes_with_component_exclusion(self):
        old = self.levels[self.levels.device == 'mgtu']
        self.assertEqual(set(old.size_mm), set(E.SIZES))
        self.assertTrue(all(r.startswith('exp02_georg_') for r in old.record_id))
        excluded = old[~old.usable]
        self.assertEqual(set(excluded.role), {'ttrkg'})
        self.assertEqual(set(excluded.size_mm), {100})
        self.assertTrue(excluded.median_ohm.isna().all())
        self.assertEqual(len(self.levels), 48)

    def test_legacy_pulse_uses_its_physical_source_units_and_full_record_filter(self):
        cfg = E.read(E.config_path())
        manifest = E.read(E.Path(cfg['derived_root']) / 'exp02/qc/10.01_record_manifest.candidate.json')
        records = {r['record_id']: r for r in manifest['records']}
        for s in self.segments:
            self.assertEqual(len(s['event_time_relative_s']), 5)
            if s['device'] != 'mgtu':
                continue
            r = records[s['record_id']]
            frame = pd.read_csv(E.Path(cfg['data_root']) / r['relative_path'], encoding='utf-8-sig')
            ch = 2 if s['role'] == 'lateral' else 1
            t = frame.TIME_s.to_numpy()
            a, b = s['window_original_time_s']
            ix = (t >= a - 1e-9) & (t <= b + 1e-9)
            raw = frame[f'RHEO_{ch}_mΩ'].to_numpy()
            filt = sosfiltfilt(butter(4, [.7, 12], fs=s['fs_hz'], btype='bandpass', output='sos'), raw)
            np.testing.assert_array_equal(s['reported_rheo_mohm'], raw[ix])
            np.testing.assert_array_equal(s['pulse_mohm'], filt[ix])
            np.testing.assert_array_equal(s['ecg_mV'], 1000 * frame.ECG_V.to_numpy()[ix])

    def test_scorpion_arrays_match_existing_four_cycle_physical_leads(self):
        upstream = {s['record_id']: s for s in E.read(E.P4.OUT / 'segments.local.json')}
        for s in self.segments:
            if s['device'] != 'scorpion':
                continue
            u = upstream[s['record_id']]
            role = 'lateral_horizontal' if s['role'] == 'lateral' else 'ttrkg'
            lead = next(x for x in u['leads'] if x['role'] == role)
            np.testing.assert_array_equal(s['pulse_mohm'], lead['pulse_mohm'])
            self.assertEqual(s['window_original_time_s'], u['window_original_time_s'])

    def test_fixed_materials_and_residual_sign_are_not_fitted(self):
        plan, forward, _ = E.FEM.verify()
        rho = E.read(E.FEM.PIPE / plan['reference_materials_file'])['rho_ohm_m']
        self.assertEqual(self.contract['literature_rho_ohm_m'], rho)
        self.assertFalse(self.contract['equipment_only_effect_identifiable'])
        self.assertFalse(self.contract['quantitative_fem_inversion_allowed'])
        f = E.fem_table()
        np.testing.assert_allclose(f['FEM − Scorpion, Ом'],
            f['КТ/FEM при литературных ρ, Ом'] - f['Scorpion, Ом'], atol=0, rtol=0)
        self.assertEqual(len(f), 44)
        for c in forward['cases']:
            self.assertEqual(c['rho_ohm_m'], rho)

    def test_comparison_scale_control_only_hides_scorpion_lateral_60mm(self):
        fig = E.pulse_figure('lateral')
        mask = fig.layout.updatemenus[0].buttons[1].args[0]['visible']
        hidden = [t for t, visible in zip(fig.data, mask) if not visible]
        self.assertEqual(len(hidden), 3)
        self.assertTrue(all(t.legendgroup == 'scorpion_60' for t in hidden))
        self.assertEqual(fig.layout.yaxis2.matches, 'y')


if __name__ == '__main__':
    unittest.main()
