import unittest

import numpy as np

from ttrkg_analysis import (
    apply_fractional_operator,
    ensemble_waveform,
    matrix_diagnostics,
    residual_candidate,
    validate_fractional_operator,
)


class TtrkgAnalysisTests(unittest.TestCase):
    def test_ensemble_preserves_signed_peak_time(self):
        time = np.arange(0.0, 12.0, 0.002)
        rpeaks = np.arange(1.0, 11.0, 1.0)
        signal = -0.02 * np.exp(-((time[:, None] - (rpeaks[None, :] + 0.24)) / 0.05) ** 2).sum(axis=1)
        grid = np.arange(-0.15, 0.701, 0.005)
        result = ensemble_waveform(time, signal, rpeaks, (0.5, 11.5), grid, (-0.12, -0.02))
        self.assertEqual(result["n_beats"], len(rpeaks))
        self.assertLess(result["mean"].min(), 0.0)
        self.assertLessEqual(abs(grid[np.argmin(result["mean"])] - 0.24), 0.005)

    def test_ensemble_rejects_cycle_crossing_interval_boundary(self):
        time = np.arange(0.0, 8.0, 0.002)
        rpeaks = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.85])
        signal = np.sin(2.0 * np.pi * time)
        grid = np.arange(-0.15, 0.701, 0.005)
        result = ensemble_waveform(time, signal, rpeaks, (0.8, 6.0), grid, (-0.12, -0.02))
        self.assertEqual(result["n_beats"], 5)
        self.assertEqual(result["n_rejected"], 1)

    def test_fractional_operator_accepts_negative_regions(self):
        result = validate_fractional_operator(["soft", "lung", "other"], [0.7, 0.4, -0.1], complete_partition=True)
        self.assertTrue(result["has_negative"])
        self.assertAlmostEqual(result["sum"], 1.0)

    def test_fractional_operator_rejects_bad_complete_sum(self):
        with self.assertRaises(ValueError):
            validate_fractional_operator(["soft", "lung"], [0.3, 0.2], complete_partition=True)

    def test_operator_preserves_sign_and_phase(self):
        changes = np.array([[0.01, 0.0, -0.01], [-0.02, 0.01, 0.0]])
        predicted = apply_fractional_operator(np.array([0.5, -0.25]), changes)
        np.testing.assert_allclose(predicted, [0.010, -0.0025, -0.005])

    def test_residual_and_variance(self):
        result = residual_candidate([0.3, 0.2], [0.1, -0.1], [0.04, 0.04], [0.01, 0.01])
        np.testing.assert_allclose(result["residual"], [0.2, 0.3])
        np.testing.assert_allclose(result["variance"], [0.05, 0.05])

    def test_residual_rejects_negative_or_nonfinite_variance(self):
        with self.assertRaises(ValueError):
            residual_candidate([1.0], [0.0], [-0.1], [0.1])
        with self.assertRaises(ValueError):
            residual_candidate([1.0], [0.0], [np.nan], [0.1])

    def test_residual_rejects_impossible_covariance(self):
        with self.assertRaises(ValueError):
            residual_candidate([1.0], [0.0], [0.04], [0.01], [0.03])

    def test_rank_deficiency_is_exposed(self):
        result = matrix_diagnostics([[1.0, 2.0], [2.0, 4.0]])
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["nullspace"].shape, (2, 1))
        self.assertTrue(np.isinf(result["condition"]))


if __name__ == "__main__":
    unittest.main()
