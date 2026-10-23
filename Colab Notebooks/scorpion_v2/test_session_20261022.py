"""Input and leakage checks for the new session; no volunteer raw data needed."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_scorpion_session import COLUMNS, load_txt, name_context, identity_by_source
from session_20261022 import (ROLES, active_roles, inner_windows, choose_four,
                              suppress_close_candidates, confirmed_sampling)
from session_two_layer import fit_fixed_h


def test_trailing_tab_does_not_shift_fields(tmp_path):
    p=tmp_path/'test.txt'
    p.write_text('\t'.join(COLUMNS)+'\n'+
                 '\n'.join('\t'.join([str(i)]+['1,25']*10)+'\t' for i in [254,255,0]),encoding='utf8')
    frame=load_txt(p)
    assert frame.COUNT.tolist()==[254,255,0]
    assert np.array_equal(frame.iloc[:,1:].to_numpy(),np.full((3,10),1.25))


def test_roles_reverse_between_participants_and_off_ports_are_excluded():
    assert ROLES['nik']=={'ttrkg':1,'lateral':2}
    assert ROLES['yaroslav']=={'ttrkg':2,'lateral':1}
    assert active_roles('ttrkg_only')==['ttrkg']
    assert active_roles('lateral_only')==['lateral']
    assert active_roles('switching')==[]
    with pytest.raises(ValueError):active_roles('unknown')


def test_filename_context_does_not_assign_channels():
    context=name_context('оба канала 60 1х1 проыерка взаимности.txt')
    assert context['record_kind']=='reciprocity'
    assert context['lateral_size_mm']==60
    assert context['ttrkg_variant']=='1x1'
    assert context['physical_channel_roles'] is None


def test_boundaries_are_local_ordered_and_trimmed():
    windows=inner_windows([[10,12],[24,26],[40,42]],60000)
    assert windows=={'inhale':[14000,22000],'exhale':[44000,55000]}
    with pytest.raises(ValueError):inner_windows([[10,12],[11,26],[40,42]],60000)
    assert 'exhale' not in inner_windows([[10,12],[24,26],[54,56]],60000)


def test_four_cycles_use_five_existing_events_without_filling_missing_ones():
    peaks=np.arange(1000,9000,1000)
    got=choose_four(peaks,1500,7800)
    assert len(got)==5
    assert set(got).issubset(set(peaks))
    assert choose_four([1000,2000,3000,4000],0,6000) is None
    assert choose_four([1000,2000,2500,2600,4000],0,6000) is None
    with pytest.raises(ValueError):choose_four([1000,2000,1500],0,6000)


def test_close_event_suppression_keeps_original_strong_extremum():
    ecg=np.zeros(4000);ecg[[1000,1300,2000,2300,3000]]=[-2,.3,-2,.4,-2]
    original=np.array([1000,1300,2000,2300,3000])
    assert suppress_close_candidates(original,ecg).tolist()==[1000,2000,3000]
    assert original.tolist()==[1000,1300,2000,2300,3000]


def linked_inputs():
    inventory={'files':[{'record_id':'x','sha256':'h','text_qc':{'rows':12},'signal_input_candidate':True}]}
    matches={'matches':[{'record_id':'x','txt_sha256':'h','counter_equal_all_rows':True,'matched_rows':12,
        'settings_states_in_record':[{'data_packets':12,'control':{'u16_at_2':1000,
            'groups_from_14_u32_u16':[[100000000,7000]]}}]}]}
    return inventory,matches


def test_timebase_checks_every_row_and_source_setting():
    i,m=linked_inputs();assert confirmed_sampling(i,m)['working_fs_hz']==1000
    m['matches'][0]['settings_states_in_record'][0]['data_packets']=11
    with pytest.raises(ValueError):confirmed_sampling(i,m)
    i,m=linked_inputs();m['matches'][0]['settings_states_in_record'][0]['control']['groups_from_14_u32_u16']=[[100000000,1000]]
    with pytest.raises(ValueError):confirmed_sampling(i,m)


class LinearModel:
    a=np.array([[1.,2.],[3.,1.],[2.,4.]])
    def evaluate(self,rho):return self.a@rho,self.a*rho


def test_withheld_baseline_does_not_enter_rho_fit():
    model=LinearModel();data=model.a@np.array([2.,5.]);changed=data.copy();changed[2]+=100
    a=fit_fixed_h(model,data,[0,1]);b=fit_fixed_h(model,changed,[0,1])
    assert np.allclose(a['rho_ohm_m'],[2,5])
    assert np.allclose(a['rho_ohm_m'],b['rho_ohm_m'])
    assert b['residual_ohm'][2]==pytest.approx(-100)


def test_fixed_identity_uses_content_and_not_mutable_times():
    identity={'source_archive_sha256':'archive','records':[dict(record_id='N001',raw_relative_path='raw/a.txt',sha256='a'),dict(record_id='N002',raw_relative_path='raw/b.txt',sha256='b')]}
    imported={'archive_sha256':'archive','files':[dict(relative_path='b.txt',sha256='b',mtime=0),dict(relative_path='a.txt',sha256='a',mtime=999)]}
    assert identity_by_source(identity,imported)['raw/a.txt']['record_id']=='N001'
    imported['files'][1]['mtime']=-999
    assert identity_by_source(identity,imported)['raw/a.txt']['record_id']=='N001'
    imported['files'][1]['sha256']='changed'
    with pytest.raises(ValueError):identity_by_source(identity,imported)
