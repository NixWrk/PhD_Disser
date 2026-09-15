import sys
import unittest
from pathlib import Path

import numpy as np


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOK_ROOT))

from two_layer_model import (  # noqa: E402
    apparent_resistivity,
    evaluate,
    geometry_from_size,
    isoimpedance_curves,
    transfer_impedance,
    transfer_impedance_coordinates,
)


class TwoLayerModelTests(unittest.TestCase):
    def setUp(self):
        self.rho1 = 5.0
        self.rho2 = 20.0
        self.h = 0.020
        self.a, self.b = geometry_from_size(0.140)

    def test_reference_apparent_resistivity_values(self):
        expected = {
            (0.050, 15.0): 5.58577,
            (0.050, 25.0): 5.81610,
            (0.090, 15.0): 6.78717,
            (0.090, 25.0): 7.54596,
            (0.140, 15.0): 8.26561,
            (0.140, 25.0): 9.79495,
        }
        for (size, rho2), target in expected.items():
            a, b = geometry_from_size(size)
            value = apparent_resistivity(
                transfer_impedance(self.rho1, rho2, self.h, a, b), a, b
            )
            self.assertAlmostEqual(value, target, places=5)

    def test_one_layer_limit(self):
        result = transfer_impedance(self.rho1, self.rho1, self.h, self.a, self.b)
        expected = self.rho1 / np.pi * (
            1.0 / (self.a - self.b) - 1.0 / (self.a + self.b)
        )
        self.assertAlmostEqual(result, expected, places=12)

    def test_small_and_large_spacing_limits(self):
        a_small, b_small = geometry_from_size(2e-5)
        rho_small = apparent_resistivity(
            transfer_impedance(self.rho1, self.rho2, self.h, a_small, b_small),
            a_small,
            b_small,
        )
        a_large, b_large = geometry_from_size(20.0)
        rho_large = apparent_resistivity(
            transfer_impedance(self.rho1, self.rho2, self.h, a_large, b_large),
            a_large,
            b_large,
        )
        self.assertTrue(np.isclose(rho_small, self.rho1, rtol=2e-6))
        self.assertTrue(np.isclose(rho_large, self.rho2, rtol=2e-3))

    def test_homogeneity_in_resistivity_and_geometry(self):
        base = transfer_impedance(self.rho1, self.rho2, self.h, self.a, self.b)
        scale = 3.7
        scaled_rho = transfer_impedance(
            scale * self.rho1, scale * self.rho2, self.h, self.a, self.b
        )
        scaled_geometry = transfer_impedance(
            self.rho1,
            self.rho2,
            scale * self.h,
            scale * self.a,
            scale * self.b,
        )
        self.assertAlmostEqual(scaled_rho / base, scale, places=10)
        self.assertAlmostEqual(scaled_geometry / base, 1.0 / scale, places=10)

    def test_analytic_derivatives_against_central_differences(self):
        result = evaluate(self.rho1, self.rho2, self.h, self.a, self.b)
        parameters = [self.rho1, self.rho2, self.h, self.a, self.b]
        analytic = [result.d_rho1, result.d_rho2, result.d_h, result.d_a, result.d_b]
        for index, derivative in enumerate(analytic):
            step = 1e-6 * parameters[index]
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            numeric = (
                transfer_impedance(*plus) - transfer_impedance(*minus)
            ) / (2.0 * step)
            self.assertTrue(np.isclose(derivative, numeric, rtol=2e-7, atol=1e-8))

    def test_reciprocity_for_explicit_coordinates(self):
        direct = transfer_impedance_coordinates(
            self.a, -self.a, self.b, -self.b, self.rho1, self.rho2, self.h
        )
        reciprocal = transfer_impedance_coordinates(
            self.b, -self.b, self.a, -self.a, self.rho1, self.rho2, self.h
        )
        closed = transfer_impedance(self.rho1, self.rho2, self.h, self.a, self.b)
        self.assertAlmostEqual(direct, reciprocal, places=11)
        self.assertAlmostEqual(direct, closed, places=11)

    def test_isoimpedance_curve_reconstructs_fixed_observation(self):
        sizes = np.asarray([0.050, 0.140])
        observed = np.asarray([
            transfer_impedance(self.rho1, self.rho2, self.h, *geometry_from_size(size))
            for size in sizes
        ])
        curves = isoimpedance_curves(
            sizes,
            observed,
            self.h,
            ratio_min=0.25,
            ratio_max=4.0,
            n_points=3,
            n_terms=1024,
        )
        np.testing.assert_allclose(curves["rho1_ohm_m"][:, -1], self.rho1, rtol=1e-10)
        np.testing.assert_allclose(curves["rho2_ohm_m"][:, -1], self.rho2, rtol=1e-10)
        self.assertTrue(np.all(curves["relative_kernel_change"] < 1e-10))

    def test_isoimpedance_curve_scales_with_observed_impedance(self):
        curves = isoimpedance_curves(
            [0.090],
            [12.0],
            self.h,
            ratio_min=0.1,
            ratio_max=10.0,
            n_points=5,
            n_terms=512,
        )
        doubled = isoimpedance_curves(
            [0.090],
            [24.0],
            self.h,
            ratio_min=0.1,
            ratio_max=10.0,
            n_points=5,
            n_terms=512,
        )
        np.testing.assert_allclose(
            doubled["rho1_ohm_m"], 2.0 * curves["rho1_ohm_m"]
        )
        np.testing.assert_allclose(
            doubled["rho2_ohm_m"], 2.0 * curves["rho2_ohm_m"]
        )

    def test_isoimpedance_curve_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            isoimpedance_curves([0.050, 0.060], [10.0], self.h)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            transfer_impedance(self.rho1, self.rho2, self.h, self.a, self.a)


if __name__ == "__main__":
    unittest.main()
