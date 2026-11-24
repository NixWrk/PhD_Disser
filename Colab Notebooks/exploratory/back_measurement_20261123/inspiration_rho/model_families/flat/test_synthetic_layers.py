"""Standalone synthetic mathematical checks: python -B test_synthetic_layers.py.

Requires only the adjacent layers.py, NumPy and SciPy. No data files are read
or written. All dimensions and resistivities below are arbitrary test inputs,
not anatomical measurements or material estimates. Units: mm, ohm m, ohm.
"""
import sys
sys.dont_write_bytecode = True
from functools import lru_cache
from itertools import product
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk
from layers import PlanarModel, surface_resistivity


SIZES_MM = tuple(48. * 2**i for i in range(3))
DEPTHS_MM = tuple(6. * 3**i for i in range(3))
DIAMETER_MM = 4.


@lru_cache(maxsize=None)
def disk_green(distance_m, radius_m, height_m=0.):
    """Average Coulomb kernel for two uniform disks, independent of layers.py.

    Disk-overlap area gives the distribution of their position difference;
    its angular integral is the complete elliptic integral K. The remaining
    radial integral uses adaptive quadrature, not the production moment series.
    """
    def integrand(u):
        shift = 2 * radius_m * u
        denominator = math.sqrt((distance_m + shift)**2 + height_m**2)
        overlap = math.acos(u) - u * math.sqrt(1 - u*u)
        return (32 / math.pi**2 * u * overlap
                * ellipk(4 * distance_m * shift / denominator**2) / denominator)
    return quad(integrand, 0, 1, epsabs=2e-11, epsrel=2e-13, limit=150)[0]


def two_layer_images(size_mm, rho_upper, rho_lower, h_mm, terms=48):
    """Analytical image-source solution, independently averaged over disks.

    beta=(rho_lower-rho_upper)/(rho_lower+rho_upper), image depth=2*n*h.
    Z=rho_upper/pi * [D(0)+2*sum(beta**n * D(2*n*h))],
    D(z)=G(L/4,z)-G(3*L/4,z). No spectral recurrence or Hankel nodes are reused.
    """
    radius = DIAMETER_MM / 2000
    d1, d3, h = size_mm / 4000, 3 * size_mm / 4000, h_mm / 1000
    beta = (rho_lower - rho_upper) / (rho_lower + rho_upper)
    def difference(z):
        return disk_green(d1, radius, z) - disk_green(d3, radius, z)
    z = rho_upper / math.pi * (difference(0.) + 2 * math.fsum(
        beta**n * difference(2*n*h) for n in range(1, terms+1)))
    # Each G(d,2*n*h) <= 1/(2*n*h); this bounds the omitted image terms.
    tail = (2 * rho_upper / math.pi * abs(beta)**(terms+1)
            / ((terms+1) * h * (1-abs(beta))))
    return z, tail


class SyntheticLayersTests(unittest.TestCase):
    def test_homogeneous_and_layer_reductions(self):
        for size in SIZES_MM:
            model = PlanarModel(size, diameter_mm=DIAMETER_MM)
            g = (disk_green(size/4000, DIAMETER_MM/2000)
                 - disk_green(3*size/4000, DIAMETER_MM/2000)) / math.pi
            checks = [
                (model.z([3, 3, 3], [6, 18]), 3*g),
                (model.z([2, 2, 6], [6, 18]), model.z([2, 6], 24)),
                (model.z([2, 6, 6], [6, 18]), model.z([2, 6], 6)),
                (model.z([9, 2, 6], [0, 18]), model.z([2, 6], 18)),
                (model.z([2, 9, 6], [6, 0]), model.z([2, 6], 6)),
                (model.z([2, 9, 6], [0, 0]), 6*g),
            ]
            for actual, expected in checks:
                self.assertAlmostEqual(actual, expected, delta=1e-8)

    def test_spectral_passivity_and_limits(self):
        k = np.r_[0., np.geomspace(1e-4, 1e6, 128), np.inf]
        model = PlanarModel(96, diameter_mm=DIAMETER_MM)
        for rhos, h in product(product((1., 4., 16.), repeat=3), ((3., 12.), (12., 48.))):
            response = surface_resistivity(k, rhos, h)
            self.assertTrue(np.isfinite(response).all())
            self.assertTrue(np.all(response >= min(rhos)*(1-1e-13)))
            self.assertTrue(np.all(response <= max(rhos)*(1+1e-13)))
            self.assertAlmostEqual(response[0], rhos[-1], delta=1e-12)
            self.assertAlmostEqual(response[-1], rhos[0], delta=1e-12)
            self.assertGreater(model.z(rhos, h), 0.)

    def test_two_layer_independent_images(self):
        worst = 0.
        for size, h, rhos in product(SIZES_MM, DEPTHS_MM, ((2., 6.), (6., 2.))):
            reference, tail = two_layer_images(size, *rhos, h)
            actual = PlanarModel(size, diameter_mm=DIAMETER_MM).z(rhos, h)
            self.assertLess(tail, 1e-10)  # Image truncation only, not total error.
            self.assertAlmostEqual(actual, reference, delta=1e-7)
            worst = max(worst, abs(actual-reference))
        print(f'Synthetic two-layer image checks: 18; max difference {worst:.3g} ohm')


if __name__ == '__main__':
    unittest.main(verbosity=2)
