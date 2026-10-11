"""Synthetic checks do not establish anatomical fidelity of the real CT."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'
sys.path.insert(0,str(TOOLS))
from georg_ct_model import body_mask, surface_from_mask
from georg_model_contract import resolve
from scorpion_georg_fem_prepare import sha
from scorpion_georg_fem_prepare import surface_intersection_check
from georg_ct_fem_prepare import validate_tetrahedra


def test_exterior_excludes_table_and_fills_lung_cavity():
    hu=np.full((80,80,12),-1000,dtype=np.int16)
    hu[10:70,10:70,:]=20;hu[20:60,20:60,:]=-800
    hu[72:76,5:75,:]=300
    mask,_=body_mask(hu)
    assert mask[40,40,5] and not mask[74,40,5]
    assert np.all(mask[13:67,13:67,:])


def test_ras_to_lps_preserves_mm_and_closure():
    mask=np.ones((4,5,6),dtype=bool)
    affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[100.,200.,-50.]
    mesh=surface_from_mask(mask,affine)
    np.testing.assert_allclose(mesh.bounds,[[-107.,-213.5,-52.],[-99.,-198.5,-28.]])
    assert mesh.is_volume and mesh.is_watertight


def test_current_model_rejects_missing_qc(tmp_path):
    pointer=tmp_path/'pointer.json';pointer.write_text(json.dumps({'status':'pending'}))
    with pytest.raises(ValueError,match='not passed'):resolve(pointer)


def test_current_model_rejects_tampered_file(tmp_path,monkeypatch):
    import georg_model_contract as module
    monkeypatch.setattr(module,'PIPE',tmp_path)
    a=tmp_path/'a.dat';a.write_bytes(b'first')
    pointer=tmp_path/'pointer.json';pointer.write_text(json.dumps(dict(status='numerical_reference_checked',
        model_directory='model',files=[dict(path='a.dat',sha256=sha(a))])))
    assert resolve(pointer)[0]==tmp_path/'model'
    a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='file changed'):resolve(pointer)


def test_checked_geometry_does_not_authorize_unconverged_inversion(tmp_path,monkeypatch):
    import georg_model_contract as module
    monkeypatch.setattr(module,'PIPE',tmp_path)
    pointer=tmp_path/'pointer.json'
    pointer.write_text(json.dumps(dict(status='geometry_reference_checked',model_directory='model',files=[])))
    assert resolve(pointer)[0]==tmp_path/'model'
    with pytest.raises(ValueError,match='precision is unresolved'):resolve(pointer,require_converged=True)


def test_surface_qc_rejects_positive_area_overlap():
    nodes=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],
        [.2,.2,0.],[1.,.2,0.],[.2,1.,0.]])
    result=surface_intersection_check(nodes,np.array([[0,1,2],[3,4,5]]))
    assert result['proper_pairs']>0


def test_surface_qc_allows_shared_boundary_without_overlap():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    result=surface_intersection_check(nodes,np.array([[0,1,2],[1,3,2]]))
    assert result['proper_pairs']==0


def test_tiny_positive_tetrahedron_requires_exact_sign_and_volume():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1e-13]])
    result=validate_tetrahedra(nodes,np.array([[0,1,2,3]]),np.array([1e-13/6]))
    assert result['tiny_cells_verified']==1 and result['maximum_relative_volume_error']<1e-12


def test_zero_volume_is_not_accepted_as_a_tiny_cell():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    with pytest.raises(ValueError,match='Nonpositive'):
        validate_tetrahedra(nodes,np.array([[0,1,2,3]]),np.array([0.]))
