"""Scientific guards for scope, units, polarity and out-of-range inversion."""
import copy
import numpy as np
import pandas as pd
import pytest

import scorpion_measured_pulse as M


def operator(z=(10.,9.,8.)):
    rows=[dict(geometry=g,rule=4096,transfers={'ttrkg':{'own_ohm':v}})
          for g,v in zip(['sphere_v095','sphere_v100','sphere_v105'],z)]
    return M.volume_operator(rows,100.,4096)


def test_milliohm_and_four_complete_cycles_without_sign_flip():
    t=np.arange(401)/100
    wave=.001*np.sin(2*np.pi*t)
    m=M.cycle_metrics(wave,[1000,1100,1200,1300,1400],100,[950,1450],origin=1000)
    assert len(m['cycles'])==4
    assert m['cycles'][0]['ptp_ohm']==pytest.approx(.002)
    assert m['delta_Z_from_R_ohm'][0][50]>0
    assert m['delta_Z_from_R_ohm'][0][150]<0


@pytest.mark.parametrize('events,inhale',[
    ([100,200,300,400],[0,500]),([100,200,200,300,400],[0,500]),
    ([100,200,300,400,500],[150,600]),([100,200,300,400,500],[0,500])])
def test_partial_or_invalid_selection_is_rejected(events,inhale):
    with pytest.raises(ValueError):M.cycle_metrics(np.zeros(601),events,100,inhale)


def test_nonmonotone_secant_does_not_silently_invert():
    op=operator((10.,10.1,10.05))
    assert op['secant_ohm_per_ml']>0
    assert op['half_slopes_ohm_per_ml'][1]<0
    d=M.amplitude_diagnostic(.2,op)
    assert d['equivalent_volume_span_ml'] is None
    assert d['inverse_status']=='blocked_nonmonotone'


def test_material_polarity_and_extrapolation_are_preserved():
    p,n=operator((8.,9.,10.)),operator()
    assert p['secant_ohm_per_ml']==-n['secant_ohm_per_ml']
    d=M.amplitude_diagnostic(3.,n)
    assert d['equivalent_volume_span_ml']==pytest.approx(15.)
    assert d['equivalent_span_exceeds_tested_volume_span']
    assert d['pulse_span_exceeds_sampled_model_span']
    assert not d['stroke_volume_estimated']


def test_ct_pairs_never_close_and_two_pairs_do_not_get_a_correlation():
    f=pd.DataFrame(dict(time_after_cycle_r_ms=[100,200,300],cycle_index=[0]*3,
        phase_id=['a','b','c'],rr_interval_ms=[900]*3,phase_percent_within_cycle=[11,22,33],
        temporal_resolution_ms=[140]*3,whole_heart_ml=[100,90,95],
        heart_atrium_left_ml=[10,11,10],heart_ventricle_left_ml=[30,25,27],
        heart_atrium_right_ml=[10,12,11],heart_ventricle_right_ml=[35,30,32]))
    r=M.ct_cycle(f,{'blood':operator()})
    assert all(c['n_pairs']==2 and c['delta_correlation'] is None for c in r['comparisons'])
    assert r['volume_only_projection'][0]['predicted_delta_Z_from_first_frame_ohm']==pytest.approx([0,2,1])
    assert not r['closed_cycle'] and not r['ED_ES_selected']


def test_phase_comparison_has_no_search_for_delay_or_sign():
    s=dict(phase_rr=[0,.5,1],delta_Z_from_R_ohm=[[0,-1,0]]*4)
    ct=[dict(cycle_index=0,phase_percent_within_cycle=[0,50,100],volumes_ml={'whole_heart_ml':[100,90,100]})]
    r=M.phase_sign_diagnostic(s,ct,[{'id':'blood','fine':operator()}])[0]
    assert all(x['same_sign_pairs']==0 for x in r['sign_counts_by_measured_cycle'])
    assert not r['delay_fitted'] and not r['gain_fitted']
    assert r['phase_alignment'].endswith('unvalidated')


def test_changed_fingerprint_is_not_accepted(tmp_path):
    p=tmp_path/'source.json';p.write_text('{}')
    h=M.sha(p);p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='fingerprint'):M.checked(p,h)
