from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
from georg_quality_general_audit import verify_contact_identity,PIPE,sha


@pytest.mark.parametrize('bad_field',['current_mesh_sha256','boundary_parent_map_sha256','mapping_helper_sha256'])
def test_foreign_executed_contact_identity_is_rejected(tmp_path,bad_field):
    mesh=tmp_path/'mesh.mat';mesh.write_bytes(b'example mesh')
    mapping=tmp_path/'map.mat';mapping.write_bytes(b'example map')
    plan=dict(exact_contact_parent_mesh=str(mesh),boundary_parent_map=str(mapping))
    identity=dict(contact_parent_mesh_sha256=sha(mesh),current_mesh_sha256=sha(mesh),
        boundary_parent_map_sha256=sha(mapping),mapping_helper_sha256=sha(PIPE/'src/trkg4_georg_contact_inheritance.m'))
    identity[bad_field]='foreign'
    with pytest.raises(ValueError,match=bad_field):
        verify_contact_identity(dict(identity=dict(contact_inheritance=identity)),plan,tmp_path)
