"""Electrical regrouping preserves bulk arrays and original contact geometry."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import ct_fem_core as core
import ct_fem_contact_regrouping as regroup
from test_ct_fem_preparation import fixture_config,mesh_artifact,write


def fixture(tmp_path):
    path,cfg=fixture_config(tmp_path)
    for row in cfg['contacts']['physical_centres_lps_mm'].values():row['nominal_area_mm2']=float(np.pi*2.5**2)
    write(path,cfg);mesh=mesh_artifact(tmp_path,path)
    parent=tmp_path/'original_materials';core.prepare('materials',path,parent,mesh)
    new=copy.deepcopy(cfg);s=new['contacts']['scenarios'][0];s['id']='reciprocal_wiring'
    s['drive_groups'],s['measure_groups']=s['measure_groups'],s['drive_groups']
    target=tmp_path/'wiring.json';write(target,new)
    return parent,target,new


def test_new_wiring_is_explicit_and_all_material_bytes_stay_identical(tmp_path):
    parent,config,_=fixture(tmp_path);out=tmp_path/'regrouped'
    original_contacts=core.sha(parent/'contacts.local.json')
    regroup.prepare(config,parent,out,reason='Synthetic current/sense regrouping')
    value=core.verify_artifact(out)
    assert value['stage']=='materials'
    for name in ['mesh.mat','fractions_q1024.mat','fractions_q4096.mat','mesh_quality.local.json']:
        assert core.sha(parent/name)==core.sha(out/name)
    assert core.sha(parent/'contacts.local.json')==original_contacts
    actual=core.read(out/'contacts.local.json');old=core.read(parent/'contacts.local.json')
    assert actual['physical_contacts']==old['physical_contacts']
    assert actual['scenarios'][0]['id']=='reciprocal_wiring'
    sidecar=tmp_path/'cem_input';core.prepare('cem_contacts_input',config,sidecar,out)
    assert core.verify_artifact(sidecar)['stage']=='cem_contacts_input'


@pytest.mark.parametrize('field',['centre','geometry','material','area'])
def test_regrouping_cannot_disguise_a_different_model(tmp_path,field):
    parent,path,cfg=fixture(tmp_path)
    if field=='centre':cfg['contacts']['physical_centres_lps_mm']['I_plus']['raw_lps_mm'][1]+=.5
    elif field=='area':cfg['contacts']['physical_centres_lps_mm']['I_plus']['nominal_area_mm2']*=2
    elif field=='geometry':cfg['body']['hu_threshold']=-440
    else:cfg['inputs']['mask_provenance']['method']='another segmentation'
    write(path,cfg)
    with pytest.raises(ValueError,match='only electrical scenarios'):
        regroup.prepare(path,parent,tmp_path/'rejected',reason='Invalid synthetic regrouping')
    assert not (tmp_path/'rejected').exists()
