"""Independent mathematical checks; no CT or experimental files are needed."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from lateral_resistivity_range_analysis import ContractError, RangeSurrogate, _find_roots_at


class RangeNumericsTest(unittest.TestCase):
    def setUp(self):
        self.plan = {'grid_rho1': np.geomspace(.1, 20, 7),
                     'grid_rho2': np.geomspace(.1, 100, 8)}
        self.x, self.y = np.meshgrid(np.log(self.plan['grid_rho1']),
                                     np.log(self.plan['grid_rho2']), indexing='ij')
        self.scale = np.arange(1, 10)

    def test_scaled_response_and_log_derivatives(self):
        x, y = self.x, self.y
        z = np.exp(x)[..., None] * (2 + .01*x*x + .02*y*y)[..., None] * self.scale
        model = RangeSurrogate(self.plan, z)
        for x0, y0 in [(0., 1.), (-2., -1.), (2., 4.), (np.log(.1), np.log(100.))]:
            s = 2 + .01*x0*x0 + .02*y0*y0
            expected_z = np.exp(x0) * s * self.scale
            expected_j = np.c_[np.exp(x0) * (s + .02*x0) * self.scale,
                               np.exp(x0) * .04*y0 * self.scale]
            np.testing.assert_allclose(model.predict([x0, y0]), expected_z, atol=1e-10, rtol=1e-12)
            np.testing.assert_allclose(model.jacobian([x0, y0]), expected_j, atol=1e-10, rtol=1e-12)
        with self.assertRaises(ContractError):
            model.predict([np.log(.099), 0.])

    def test_multiple_roots_and_absent_solution(self):
        x = self.x
        z = np.exp(x)[..., None] * ((x + .5)*(x - 1))[..., None] * self.scale
        model = RangeSurrogate(self.plan, z)
        scan = np.linspace(np.log(.1), np.log(20), 401)
        roots = _find_roots_at(model, 0., 0, 1., scan)
        np.testing.assert_allclose(roots, [-.5, 1.], atol=1e-11, rtol=0)
        self.assertEqual(_find_roots_at(model, 1e6, 0, 1., scan), [])


if __name__ == '__main__':
    unittest.main()
