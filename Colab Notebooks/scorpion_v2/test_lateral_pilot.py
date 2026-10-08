import sys
from pathlib import Path

import numpy as np
import pytest

from lateral_pilot import DiskModel, candidate_windows, cycle_ensemble, fit_state, invert_pulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from two_layer_model import forward_curve


def test_finite_disk_point_limit_and_homogeneous_case():
    sizes = np.array([60., 100., 140.])
    for rho in ([2., 15.], [15., 2.], [3., 3.]):
        z, jlog = DiskModel(sizes, 20., diameter_mm=1e-4).evaluate(rho)
        zp, jp = forward_curve(sizes / 1000, *rho, .02)
        np.testing.assert_allclose(z, zp, rtol=1e-9)
        np.testing.assert_allclose(jlog, jp[:, :2] * rho, rtol=1e-8)


def test_finite_disk_jacobian_and_synthetic_inverse():
    model = DiskModel([60, 80, 100, 120, 140], 20)
    rho = np.array([3., 15.])
    z, jlog = model.evaluate(rho)
    for k in range(2):
        step = np.zeros(2); step[k] = 1e-5
        fd = (model.evaluate(rho * np.exp(step))[0] -
              model.evaluate(rho * np.exp(-step))[0]) / 2e-5
        np.testing.assert_allclose(jlog[:, k], fd, rtol=1e-8)
    fit = fit_state(model, z, np.array([[1., 30.], [2., 25.]]))
    np.testing.assert_allclose([fit['rho1_ohm_m'], fit['rho2_ohm_m']], rho, rtol=1e-6)
    fraction = np.array([.002 * np.sin(np.linspace(0, 2 * np.pi, 200)),
                         .003 * np.cos(np.linspace(0, 2 * np.pi, 200))])
    recovered = invert_pulse(model, fit, jlog @ fraction)
    np.testing.assert_allclose(recovered['delta_rho_ohm_m'], rho[:, None] * fraction, atol=1e-10)
    assert recovered['residual_percent'] < 1e-6


def test_candidate_windows_stay_inside_ambiguity_ranges_and_edges():
    annotation = {'status': 'pending_manual_review', 'author_accepted': False,
                  'boundary_uncertainty_ranges_samples': [[20_000, 25_000], [39_000, 42_000], [56_000, 61_000]]}
    assert candidate_windows(annotation, 1000, 75_725) == {
        'inhale': [27_000, 37_000], 'exhale': [63_000, 70_725]}
    assert annotation['status'] == 'pending_manual_review'
    assert annotation['author_accepted'] is False
    assert candidate_windows({}, 1000, 10_000) == {}


def test_incomplete_and_long_rr_cycles_are_not_interpolated_as_valid_beats():
    fs = 1000
    values = np.sin(2 * np.pi * np.arange(10_000) / fs)
    result = cycle_ensemble(values, np.array([0, 1000, 2000, 5000, 6000, 7000]),
                            500, 6500, fs)
    assert result['n_cycles'] == 2
    assert result['rr_rejected_sample_pairs'] == [[2000, 5000]]
    assert result['cycle_sample_pairs'] == [[1000, 2000], [5000, 6000]]
    assert abs(np.mean(result['median_ohm'])) < 1e-12


def test_disks_must_be_disjoint():
    with pytest.raises(ValueError, match='disjoint'):
        DiskModel([20], 10, diameter_mm=5)
