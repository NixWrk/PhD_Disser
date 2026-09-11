"""Algebraic multi-electrode controls, not anatomical validation."""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import run_electrode_sensitivity as runner
runner.np = np
runner.sp = sp
runner.sio = sio


def test_six_contact_network_multiple_roles_and_reciprocity():
    # Two tissue nodes joined by conductance 2 S; three distinct contacts
    # at each end. This deliberately small network tests indexing and units.
    B = sp.csr_matrix([[1.,-1.],[-1.,1.]])
    model = {'nodes': np.zeros((2,3)), 'blocks': [B]}
    blocks = runner.parameter_blocks(model, [{'id':'tissue','block_indices':[0]}], 6)
    C = np.zeros((8,8))
    for j in range(6):
        tissue = 0 if j < 3 else 1
        electrode = j+2
        C[tissue,tissue] += 100
        C[electrode,electrode] += 100
        C[tissue,electrode] -= 100
        C[electrode,tissue] -= 100
    montage = {'electrodes':list('ABCDEF'), 'channels':[
        {'drive':[1,0,0,-1,0,0], 'measure':[0,1,0,0,-1,0]},
        {'drive':[0,0,1,0,0,-1], 'measure':[1,0,0,-1,0,0]}]}
    q = runner.channel_rhs(2, montage)
    K = 2*blocks['tissue']+sp.csr_matrix(C)
    fields, residual = runner.solve_system(K, q, 0, backend='scipy')
    assert max(residual) < 1e-12
    for j in range(2):
        u,v = fields[:,2*j],fields[:,2*j+1]
        assert q[:,2*j+1]@u == pytest.approx(.5)
        assert q[:,2*j]@v == pytest.approx(.5)
        assert -v@(blocks['tissue']@u) == pytest.approx(-.25)


def test_missing_volume_block_rejected():
    model={'nodes':np.zeros((2,3)), 'blocks':[sp.eye(2),sp.eye(2)]}
    with pytest.raises(ValueError, match='partition'):
        runner.parameter_blocks(model,[{'id':'one','block_indices':[0]}],4)


def test_contact_matrix_is_bound_to_exact_model(tmp_path):
    C=sp.csr_matrix([[2.,0.,-2.,0.],[0.,3.,0.,-3.],[-2.,0.,2.,0.],[0.,-3.,0.,3.]])
    path=tmp_path/'contact.mat'
    sio.savemat(path,{'C':C,'ground':1,'matrix_error':1e-16,'prepared_sha256':'a'*64,'electrode_ids':np.array(['E1','E2'],dtype=object)})
    assert runner.load_contact(path,2,2,0,'a'*64).shape==(4,4)
    with pytest.raises(ValueError,match='different prepared model'):
        runner.load_contact(path,2,2,0,'b'*64)
    sio.savemat(path,{'C':C,'ground':1,'matrix_error':1e-16})
    with pytest.raises(ValueError,match='no verified binding'):
        runner.load_contact(path,2,2,0,'a'*64)


def test_indefinite_small_system_is_rejected():
    K=sp.csr_matrix([[1.,0.,0.],[0.,1.,2.],[0.,2.,1.]])
    q=np.array([[0.],[1.],[-1.]])
    with pytest.raises(np.linalg.LinAlgError):runner.solve_system(K,q,0,backend='scipy')


def test_nonfinite_system_is_rejected():
    K=sp.csr_matrix([[1.,0.],[0.,np.inf]])
    with pytest.raises(ArithmeticError,match='finite'):
        runner.solve_system(K,np.array([[0.],[1.]]),0,backend='scipy')


def test_completion_empty_inventory_and_corruption_rejected(tmp_path):
    (tmp_path/'data.csv').write_text('finite data')
    report={'status':'completed_numerical_checks_passed','scientific_status':'exploratory',
            'channel_states':1,'maps':0,'output_sha256':{'data.csv':runner.digest(tmp_path/'data.csv')}}
    def save_record():
        runner.write_json(tmp_path/'completion.json',report)
        (tmp_path/'completion.sha256').write_text(runner.digest(tmp_path/'completion.json'))
    save_record()
    runner.verify_completed_output(tmp_path,{'data.csv'},1,0,'exploratory')
    report['output_sha256']={}
    save_record()
    with pytest.raises(ValueError,match='inventory'):
        runner.verify_completed_output(tmp_path,{'data.csv'},1,0,'exploratory')
    (tmp_path/'completion.json').write_text('{}')
    with pytest.raises(ValueError,match='integrity'):
        runner.verify_completed_output(tmp_path,{'data.csv'},1,0,'exploratory')


def test_nonfinite_result_csv_rejected(tmp_path):
    with pytest.raises(ArithmeticError,match='Nonfinite'):
        runner.write_csv(tmp_path/'bad.csv',[{'Z':np.inf}])


@pytest.mark.parametrize('problem',['fractional_label','infinite_index','out_of_range','wrong_shape'])
def test_hdf_geometry_rejected_before_cast(tmp_path,problem):
    h5py=pytest.importorskip('h5py');runner.h5py=h5py
    path=tmp_path/'model.mat'
    nodes=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    elems=np.array([[1,2,3,4]],float);labels=np.array([[1.]],float)
    if problem=='fractional_label':labels[0,0]=1.9
    if problem=='infinite_index':elems[0,0]=np.inf
    if problem=='out_of_range':elems[0,0]=5
    if problem=='wrong_shape':elems=elems[:,:3]
    with h5py.File(path,'w') as f:
        f.create_dataset('fmdl_m/nodes',data=nodes.T)
        f.create_dataset('fmdl_m/elems',data=elems.T)
        f.create_dataset('tissue_id',data=labels)
    with pytest.raises(ValueError):runner.load_model(path)
