"""Единицы, полный размах, сохранение знака и отдельная политика QC циклов."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
from bmstu_pulse_resistivity import PHASE, normalized_cycle, pulse_statistics


def test_physical_units_full_range_and_signed_waveform():
    waves=np.tile(20*np.cos(2*np.pi*PHASE),(4,1))
    a=pulse_statistics(waves,[False]*4,10,200)
    assert a['delta_Z_median_ohm']==pytest.approx(.04)
    assert a['delta_rho_median_ohm_m']==pytest.approx(.002)
    assert a['median_wave_ohm_m'][0]==pytest.approx(.001)
    b=pulse_statistics(-waves,[False]*4,10,200)
    assert b['delta_rho_median_ohm_m']==pytest.approx(.002)
    assert b['median_wave_ohm_m'][0]==pytest.approx(-.001)


def test_rr_normalization_removes_constant_only():
    t=np.linspace(0,4,4001);y=17*np.cos(2*np.pi*t/1.2)
    a=normalized_cycle(t,y,1.2,2.4)
    b=normalized_cycle(t,y+1234,1.2,2.4)
    np.testing.assert_allclose(a,b,atol=1e-10)
    assert np.ptp(a)==pytest.approx(34,rel=1e-5)
    with pytest.raises(ValueError,match='пределы'):
        normalized_cycle(t,y,3.5,4.5)


def test_flagged_cycles_remain_primary_and_sensitivity_is_separate():
    shape=np.cos(2*np.pi*PHASE)
    waves=np.array([shape,shape,shape,3*shape,3*shape,3*shape,3*shape])
    a=pulse_statistics(waves,[False]*3+[True]*4,10,200)
    assert a['n_cycles']==7 and a['n_candidate_ecg_flags']==4
    assert a['delta_rho_median_ohm_m']==pytest.approx(.0003)
    assert a['without_candidate_ecg_flags']['n_cycles']==3
    assert a['without_candidate_ecg_flags']['delta_rho_median_ohm_m']==pytest.approx(.0001)


def test_range_of_ensemble_is_not_median_cycle_range():
    waves=np.array([np.cos(2*np.pi*(PHASE+p)) for p in (0,1/3,2/3)])
    a=pulse_statistics(waves,[False]*3,10,200)
    assert a['ensemble_peak_to_peak_ohm_m']<.6*a['delta_rho_median_ohm_m']
    with pytest.raises(ValueError,match='положительные'):
        pulse_statistics(waves,[False]*3,10,-200)
