"""Analytical isotropic resistive layers and four finite circular surface contacts.

Only NumPy/SciPy are required. No data files, inverse solver, FEM or MATLAB.
Units: rho [ohm m], h [mm], size/diameter [mm], k [1/m], returned Z [ohm].
Order is always top to bottom; the LAST material is a semi-infinite substrate.
Examples: (upper, lung), (fat, nonfat_soft, lung), or (wall, lung, wall).
The named tissue order is a caller's hypothesis, not inferred by this module.
There is no physiological bound on any material index; rho must be positive.

For a horizontal Fourier mode k, let R_i=k*V_i/j_i (ohm m). With continuity
of potential and normal current across a layer of thickness h_i (in metres),
  R_bottom = rho_bottom,
  R_i = rho_i*(R_{i+1}+rho_i*tanh(k*h_i)) /
                    (rho_i+R_{i+1}*tanh(k*h_i)).
This is the exact solution of V''-k**2 V=0, evaluated from bottom to top.
The actual spectral surface impedance is R_0/k (ohm m**2). R_0(0)=rho_bottom;
R_0(infinity)=rho_first_nonzero_layer. Zero-thickness layers disappear exactly.
R is real, positive and lies between the smallest and largest material rho.

Contacts have uniform imposed source current and area-averaged sensing voltage,
NOT equipotential CEM contact conditions. A/M/N/B centres are -L/2,-L/4,L/4,L/2;
I enters A and leaves B, and Z=(V_M-V_N)/I. No sign flips or fitted factors.
For disk radius a, F(k)=2*J1(a*k)/(a*k), d1=L/4, d3=3L/4:
  Z = rho_surface*H + (1/pi)*integral_0^infty (R(k)-rho_surface)*F(k)**2
                                             *(J0(k*d1)-J0(k*d3)) dk,
  H = (G(d1)-G(d3))/pi.
G(d) is the exact homogeneous disk-pair average of 1/distance. Its convergent
moment series for nonoverlapping disks is G(d)=sum(t_n)/d, t_0=1,
  t_n/t_{n-1} = (a/d)**2*(2*n-1)**2*(2*n+1)/(2*n*(n+1)*(n+2)).
Thus the slowly convergent oscillatory homogeneous Hankel tail is analytical.
The remaining integral decays exponentially with the first nonzero thickness.
The absolute tail is <=rho_surface*2/(pi*h)*[-log(1-exp(-2*h*kmax))].
This bounds truncation only; quadrature accuracy is checked in test_layers.py.

PlanarModel(size_mm).z(rhos,h_mm) caches thickness preparations. A scalar h
means two materials; [h1,h2] means three. prepare(h_mm) returns a SEPARATE
PreparedPlanarModel with z(rhos); it does not alter the parent's geometry.
Coefficient arrays are readonly. Exact zero thickness is supported; no small
positive thickness is silently rounded to zero. Very thin positive layers can
require many quadrature nodes and raise a resource-limit error explicitly.
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j0, j1

__all__ = ['PlanarModel', 'PreparedPlanarModel', 'surface_resistivity']


def _thickness(h_mm):
    if np.iscomplexobj(h_mm):
        raise ValueError('Thicknesses must be real')
    h = np.atleast_1d(np.asarray(h_mm, dtype=float))
    if h.ndim != 1 or len(h) not in (1, 2) or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError('h_mm must be one or two finite nonnegative thicknesses, top to bottom')
    return tuple(float(x) for x in h)


def _rho(rhos, count):
    if np.iscomplexobj(rhos):
        raise ValueError('Only real resistive materials are implemented; complex admittivity is not discarded')
    r = np.asarray(rhos, dtype=float)
    if r.shape != (count,) or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError(f'Expected {count} finite positive resistivities in ohm m, top to bottom')
    return r


def _active(h):
    ids = np.flatnonzero(np.asarray(h) > 0)
    return np.r_[ids, len(h)], np.asarray(h)[ids] / 1000


def surface_resistivity(k_m_inv, rhos, h_mm):
    """Return exact k*surface_impedance [ohm m], including k=0 and +infinity.

    This is NOT surface_impedance itself; divide by k for nonzero finite k.
    The k=0 impedance diverges for a monopole; the four-contact difference
    cancels that mode. At infinity surface_impedance tends to zero.
    """
    h = _thickness(h_mm)
    rho = _rho(rhos, len(h)+1)
    if np.iscomplexobj(k_m_inv):
        raise ValueError('Real nonnegative spatial frequency required')
    k = np.asarray(k_m_inv, dtype=float)
    if np.isnan(k).any() or np.any(k < 0):
        raise ValueError('k must be nonnegative, finite or +infinity, in inverse metres')
    indices, hm = _active(h)
    r = rho[indices]
    value = np.full_like(k, r[-1], dtype=float)
    for rr, hh in zip(r[-2::-1], hm[::-1]):
        t = np.tanh(k*hh)
        value = rr*(value+rr*t)/(rr+value*t)
    return float(value) if value.ndim == 0 else value


def _disk_green(d, a):
    """Analytical homogeneous disk-pair moment series, in inverse metres."""
    if d <= 2*a:
        raise ValueError('Disk-pair moment series requires disjoint contacts')
    total = term = 1.
    q = (a/d)**2
    for n in range(1, 10001):
        term *= q*(2*n-1)**2*(2*n+1)/(2*n*(n+1)*(n+2))
        total += term
        if term < 2e-16*total:
            return total/d
    raise ArithmeticError('Disk moment series did not converge; contacts too close')


class PlanarModel:
    """Fixed finite-disk geometry; cache reusable thickness quadratures.

    quad_order=16 and tail_decay=24 are numerical defaults, not fit parameters.
    rho .05..250 ohm m and the CT-scale thicknesses are tested. Positivity is
    the only material constraint imposed here. Finite lung thickness is simply
    z((wall,lung,wall),(h_wall,lung_h)); no tissue labels are hard coded.
    """
    def __init__(self, size_mm, diameter_mm=5., *, quad_order=16, tail_decay=24.,
                 cache_size=8, max_nodes=2_000_000):
        size_mm, diameter_mm = float(size_mm), float(diameter_mm)
        if not np.isfinite([size_mm,diameter_mm]).all() or diameter_mm <= 0 or size_mm/4 <= diameter_mm:
            raise ValueError('Positive diameter and size/4 > diameter are required for disjoint disks')
        if int(quad_order) != quad_order or quad_order < 8 or not np.isfinite(tail_decay) or tail_decay < 12:
            raise ValueError('quad_order >=8 (integer), tail_decay >=12 required')
        if int(cache_size) != cache_size or cache_size < 1 or int(max_nodes) != max_nodes or max_nodes < 32:
            raise ValueError('Positive integer cache_size and max_nodes >=32 required')
        self.size_mm, self.diameter_mm = size_mm, diameter_mm
        self.radius_m = diameter_mm/2000
        self.d1_m, self.d3_m = size_mm/4000, 3*size_mm/4000
        self.homogeneous_factor_m_inv = (_disk_green(self.d1_m,self.radius_m)-
                                         _disk_green(self.d3_m,self.radius_m))/np.pi
        self.quad_order, self.tail_decay = int(quad_order), float(tail_decay)
        self.cache_size, self.max_nodes = int(cache_size), int(max_nodes)
        self._prepared = OrderedDict()

    def prepare(self, h_mm):
        """Return cached separate PreparedPlanarModel; h_mm is not mutated."""
        h = _thickness(h_mm)
        if h not in self._prepared:
            self._prepared[h] = PreparedPlanarModel(self, h)
            if len(self._prepared) > self.cache_size:
                self._prepared.popitem(last=False)
        self._prepared.move_to_end(h)
        return self._prepared[h]

    def z(self, rhos, h_mm):
        """Transfer impedance [ohm] for material order top to bottom."""
        return self.prepare(h_mm).z(rhos)


class PreparedPlanarModel:
    """Readonly geometry/thickness coefficients, reusable for many rho vectors."""
    def __init__(self, parent, h_mm):
        h_mm = _thickness(h_mm)
        self.h_mm = h_mm
        self.n_materials = len(h_mm)+1
        self.active_indices, hm = _active(h_mm)
        self.homogeneous_factor_m_inv = parent.homogeneous_factor_m_inv
        if not len(hm):
            self.k_m_inv = np.empty(0)
            self.kernel_weights_m_inv = np.empty(0)
            self.tanh_kh = np.empty((0,0))
            self.kmax_m_inv = 0.
            self.tail_bound_factor_m_inv = 0.
        else:
            # Only the top NONZERO layer controls the exponential tail.
            first_h = hm[0]
            self.kmax_m_inv = parent.tail_decay/first_h
            panel_width = np.pi/(parent.d3_m+2*parent.radius_m)
            n = int(np.ceil(self.kmax_m_inv/panel_width))
            # Resolve near-zero spectral transitions even at large contrasts.
            low = np.geomspace(1e-7/hm.sum(), 1/hm.sum(), 32)
            if (n+len(low))*parent.quad_order > parent.max_nodes:
                raise ValueError('Positive layer too thin for max_nodes; increase max_nodes explicitly. '
                                 'Use zero only for a physically absent layer, never as an automatic approximation.')
            edges = np.unique(np.r_[np.linspace(0,self.kmax_m_inv,n+1),low[low<self.kmax_m_inv]])
            nodes, weights = leggauss(parent.quad_order)
            mids, half = (edges[1:]+edges[:-1])/2, np.diff(edges)/2
            k = (mids[:,None]+half[:,None]*nodes).ravel()
            dk = (half[:,None]*weights).ravel()
            ka = k*parent.radius_m
            disk = 2*j1(ka)/ka
            self.k_m_inv = k
            self.kernel_weights_m_inv = dk*disk**2*(j0(k*parent.d1_m)-j0(k*parent.d3_m))/np.pi
            self.tanh_kh = np.tanh(hm[:,None]*k)
            self.tail_bound_factor_m_inv = -2/(np.pi*first_h)*np.log1p(-np.exp(-2*first_h*self.kmax_m_inv))
        for a in (self.active_indices,self.k_m_inv,self.kernel_weights_m_inv,self.tanh_kh):
            a.flags.writeable = False

    @property
    def n_nodes(self):
        return len(self.k_m_inv)

    def z(self, rhos):
        rho = _rho(rhos,self.n_materials)[self.active_indices]
        if len(rho)==1 or np.all(rho==rho[0]):
            return float(rho[0]*self.homogeneous_factor_m_inv)
        value = np.full(self.n_nodes,rho[-1],dtype=float)
        for rr,t in zip(rho[-2::-1],self.tanh_kh[::-1]):
            value = rr*(value+rr*t)/(rr+value*t)
        result = rho[0]*self.homogeneous_factor_m_inv + np.dot(self.kernel_weights_m_inv,value-rho[0])
        if not np.isfinite(result) or result <= 0:
            raise ArithmeticError('Nonpositive/nonfinite canonical transfer impedance; check numerical resolution')
        return float(result)

    def tail_bound_ohm(self, rhos):
        """Conservative omitted-tail bound only, NOT total quadrature error."""
        rho = _rho(rhos,self.n_materials)[self.active_indices]
        return float(rho[0]*self.tail_bound_factor_m_inv)
