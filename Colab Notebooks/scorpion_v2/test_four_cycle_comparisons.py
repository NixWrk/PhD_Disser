"""Timebase, physical lead and no-invented-cycle checks for displayed windows."""
import unittest

import numpy as np

from four_cycle_comparisons import choose_four, OUT, selected_leads, verify, figure
from lateral_pilot import EXPERIMENT, read


class SelectionChecks(unittest.TestCase):
    def test_four_intervals_require_five_events(self):
        self.assertIsNone(choose_four([1000, 2000, 3000, 4000], 0, 5000, 1000))
        np.testing.assert_array_equal(choose_four([1000, 2000, 3000, 4000, 5000], 0, 6000, 1000),
                                      [1000, 2000, 3000, 4000, 5000])

    def test_invalid_interval_is_not_removed_to_join_nonconsecutive_events(self):
        self.assertIsNone(choose_four([1000, 2000, 2100, 3000, 4000, 5000], 0, 6000, 1000))

    def test_terminal_event_must_stay_inside_half_open_hold(self):
        self.assertIsNone(choose_four([1000, 2000, 3000, 4000, 5000], 1000, 5000, 1000))


class SavedWindowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUT / 'contract.local.json').is_file():
            raise unittest.SkipTest('Subject-local signal caches and presentation contract are required')
        cls.contract = verify()
        cls.segments = read(OUT / 'segments.local.json')
        cls.sources = {s['record_id']: s for s in cls.contract['source_artifacts']}

    def test_groups_use_confirmed_physical_leads(self):
        main = [(s['number'], p['channel']) for s in self.segments for p in selected_leads('lateral', s)]
        self.assertEqual(main, [(4, 2), (7, 1), (8, 1), (9, 1), (10, 1)])
        self.assertEqual(sum(len(selected_leads('cross', s)) for s in self.segments), 4)
        ttrkg = [(s['number'], p['channel']) for s in self.segments for p in selected_leads('ttrkg', s)]
        self.assertEqual(len(ttrkg), 15)
        self.assertEqual([n for n, ch in ttrkg if ch == 2], [7, 8, 9, 10])
        self.assertEqual([(s['number'], p['channel']) for s in self.segments
                          for p in selected_leads('ttrkg_variants', s)], [(2, 1), (3, 1), (4, 1)])
        self.assertEqual([(s['number'], p['channel']) for s in self.segments for p in selected_leads('tepc', s)],
                         [(n, 2) for n in range(12, 18)])

    def test_samples_match_full_rate_cache_without_gains_or_resampling(self):
        for s in self.segments:
            start, end = s['window_samples_inclusive']
            np.testing.assert_array_equal(s['time_s'], (np.arange(start, end + 1) - start) / s['fs_hz'])
            with np.load(EXPERIMENT / self.sources[s['record_id']]['cache_path']) as arrays:
                for p in s['leads']:
                    np.testing.assert_array_equal(p['pulse_mohm'], 1000 * arrays[f'bei_pulse_band_{p["channel"]}'][start:end + 1])
                for e in s['ecg']:
                    np.testing.assert_array_equal(e['wave_mV'], arrays[f'ecg_filtered_{e["channel"]}'][start:end + 1])
                if s['n_cycles'] is None:
                    self.assertEqual(s['number'], 11)
                    self.assertIsNone(s['ecg_reference_channel'])
                    self.assertEqual(s['qrs_candidate_samples'], [])
                    self.assertEqual(end - start, 4 * s['fs_hz'])
                else:
                    self.assertEqual(s['n_cycles'], 4)
                    self.assertEqual(len(s['qrs_candidate_samples']), 5)
                    events = arrays[f'qrs_samples_{s["ecg_reference_channel"]}']
                    locations = np.searchsorted(events, s['qrs_candidate_samples'])
                    np.testing.assert_array_equal(np.diff(locations), [1, 1, 1, 1])
                    self.assertEqual(s['qrs_candidate_samples'][0], start)
                    self.assertEqual(s['qrs_candidate_samples'][-1], end)

    def test_windows_preserve_uncertainty_and_author_hint(self):
        for s in self.segments:
            start, end = s['window_samples_inclusive']
            if s['number'] == 16:
                self.assertGreaterEqual(start, 22000)
                self.assertLess(end, 30000)
                self.assertEqual(s['mode'], 'four_candidate_cycles_author_approximate_window')
                self.assertFalse(self.contract['author_display_hint']['existing_annotations_modified'])
            else:
                lo, hi = s['strict_inhale_window_samples_half_open']
                self.assertGreaterEqual(start, lo)
                self.assertLess(end, hi)
            self.assertFalse(s['breathing_author_accepted'])
            self.assertEqual(s['annotation_status'], 'pending_manual_review')

    def test_scale_controls_preserve_samples_and_group_ecg_with_pulse(self):
        lateral = figure('lateral')
        button = lateral.layout.updatemenus[0].buttons[1]
        hidden = [trace for trace, visible in zip(lateral.data, button.args[0]['visible']) if not visible]
        self.assertEqual(len(hidden), 3)  # pulse, reference ECG, boundary markers
        self.assertTrue(all(t.legendgroup == '010 · 60 мм · горизонтальная' for t in hidden))
        cross = figure('cross')
        common = cross.layout.updatemenus[0].buttons[1].args[0]
        self.assertEqual(common['yaxis.range'], common['yaxis2.range'])
        self.assertNotEqual(cross.layout.yaxis.range, cross.layout.yaxis2.range)

    def test_variant_plot_matches_three_confirmed_ttrkg_sources(self):
        comparison = figure('ttrkg_variants')
        pulses = [trace for trace in comparison.data if trace.yaxis == 'y']
        self.assertEqual([t.name[:3] for t in pulses], ['002', '004', '003'])
        for trace in pulses:
            item = next(s for s in self.segments if s['number'] == int(trace.name[:3]))
            lead = next(p for p in item['leads'] if p['channel'] == 1)
            self.assertEqual(lead['role'], 'ttrkg')
            np.testing.assert_array_equal(trace.y, lead['pulse_mohm'])

    def test_tepc_plot_pulses_are_channel_two_even_with_ttrkg_ecg(self):
        pulses = [trace for trace in figure('tepc').data if trace.yaxis == 'y']
        self.assertEqual(len(pulses), 6)
        for trace in pulses:
            item = next(s for s in self.segments if s['number'] == int(trace.name[:3]))
            lead = next(p for p in item['leads'] if p['channel'] == 2)
            self.assertEqual(lead['role'], 'tepc')
            np.testing.assert_array_equal(trace.y, lead['pulse_mohm'])


if __name__ == '__main__':
    unittest.main()
