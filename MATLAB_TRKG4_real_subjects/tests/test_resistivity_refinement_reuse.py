"""Protect reuse against silent changes in fixed physics and channel definition."""
import copy
import hashlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_resistivity_refinement import verify_reuse_model_binding


def fixture_contract(tmp_path):
    contact=tmp_path/'example_contact.mat';contact.write_bytes(b'fixture-contact-identity')
    digest=hashlib.sha256(contact.read_bytes()).hexdigest()
    plan={'mesh_sha256':'mesh','prepared_sha256':'prepared','frequency_hz':50000}
    old={'mesh_sha256':'mesh','prepared_fem_sha256':'prepared','study':{'frequency_hz':50000}}
    fixed={'bone_conductivity_S_m':.02}
    study={'source':{'frequency_hz':50000,'prepared_sha256':'prepared'},
      'parameters':[{'id':name,'block_indices':blocks,'fixed':is_fixed,'nominal_sigma':.02} for name,blocks,is_fixed in [('soft',[0,1],False),('heart',[2],False),('lung',[3],False),('bone',[4],True)]],
      'states':[{'conductivity':{'soft':.2,'heart':.3,'lung':.1,'bone':.02}}],
      'montages':[{'id':'example','contact_matrix':'example_contact.mat','contact_sha256':digest,'electrodes':['E1','E2','E3','E4'],'channels':[{'id':'standard','drive':[1,0,0,-1],'measure':[0,1,-1,0]}]}]}
    return plan,study,old,fixed,{'example_contact.mat':digest},tmp_path


def test_complete_reuse_contract_is_accepted(tmp_path):
    assert verify_reuse_model_binding(*fixture_contract(tmp_path))


@pytest.mark.parametrize('change',['bone','frequency','contact','roles','blocks','mesh','nominal_bone'])
def test_three_equal_variable_parameters_do_not_override_other_physics(tmp_path,change):
    plan,study,old,fixed,integrity,folder=fixture_contract(tmp_path)
    if change=='bone':study['states'][0]['conductivity']['bone']=.03
    elif change=='frequency':study['source']['frequency_hz']=100000
    elif change=='contact':study['montages'][0]['contact_sha256']='0'*64
    elif change=='roles':study['montages'][0]['channels'][0]['measure']=[1,0,0,-1]
    elif change=='blocks':study['parameters'][0]['block_indices']=[0]
    elif change=='mesh':old['mesh_sha256']='other-mesh'
    elif change=='nominal_bone':study['parameters'][3]['nominal_sigma']=.03
    with pytest.raises(AssertionError):verify_reuse_model_binding(plan,study,old,fixed,integrity,folder)
