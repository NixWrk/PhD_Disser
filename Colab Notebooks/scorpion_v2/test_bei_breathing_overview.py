import unittest
import numpy as np
from bei_breathing_overview import extrema_indices, segment


class BeiOverviewChecks(unittest.TestCase):
    def test_single_sample_downward_spike_survives_display_reduction(self):
        y = np.full(10000, 80.)
        y[3847] = -20.
        y[4003] = 140.
        indices = extrema_indices(y, 20)
        self.assertIn(3847, indices)
        self.assertIn(4003, indices)
        self.assertLess(len(indices), 100)

    def test_nonfinite_gap_is_retained_between_finite_neighbours(self):
        y = np.arange(100., dtype=float)
        y[43:47] = np.nan
        indices = extrema_indices(y, 3)
        self.assertTrue({42,43,46,47}.issubset(set(indices)))
        result = segment(np.arange(100)/1000, y, 0, 100)
        self.assertIn(None, result['y'])
        self.assertEqual(result['stats']['nonfinite'], 4)

    def test_half_open_stage_statistics_use_every_original_sample(self):
        t = np.arange(10000)/1000
        y = np.arange(10000., dtype=float)
        y[3001] = -1.
        y[8000] = -100.
        original = y.copy()
        result = segment(t, y, 2000, 8000)
        self.assertEqual(result['stats']['n'], 6000)
        self.assertEqual(result['stats']['below_zero'], 1)
        self.assertEqual(result['stats']['minimum'], -1.)
        self.assertEqual(result['stats']['maximum'], 7999.)
        self.assertEqual(result['stats']['median'], np.median(y[2000:8000]))
        np.testing.assert_array_equal(y, original)


if __name__ == '__main__':
    unittest.main()
