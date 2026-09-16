"""Export contact potentials and geometry checks for interpreting TEPC-7."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
import heart_replacement_electrical_pilot as core

def export():
    run=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/nix_cardiac_cycle_20260916/electrical'
    p=core.read(run/'plan.json')
    assert core.sha(p['source_prepared'])==p['source_prepared_sha256']
    rows=[]
    for m in range(2,8):
        target=run/'phase_00'/f'tepc_{m}'
        complete=core.read(target/'completion.json')
        assert complete['plan_sha256']==core.sha(run/'plan.json')
        for name in ('fields.npz','result.json'):
            assert core.sha(target/name)==complete['hashes'][name]
        with np.load(target/'fields.npz') as f:
            field=f['potentials_ohm'][:,0]
        u=field[-4:]
        result=core.read(target/'result.json')
        assert np.isclose(u[1]-u[2],result['Z_ohm'],rtol=0,atol=1e-12)
        rows.append({'montage':f'tepc_{m}','contact_potentials_mV_at_1mA':u.tolist(),
                     'voltage_difference_mV_at_1mA':float(u[1]-u[2]),
                     'fields_sha256':complete['hashes']['fields.npz']})
    geom=core.read(ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/tepc_preparation_20260911/contacts/tepc_7_geometry.json')
    assert geom['prepared_sha256']==p['source_prepared_sha256']
    rawpath=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/tepc_preparation_20260911/raw_json/tepc_7.json'
    raw=core.read(rawpath)
    assert core.sha(rawpath)==geom['source_json_sha256']
    assert raw['electrode_order']==['I_plus','V_plus','V_minus','I_minus']
    requested=np.array([raw['preliminary_electrode_centers_xyz_mm'][role] for role in raw['electrode_order']])
    assert np.allclose(requested,geom['requested_centres_xyz_mm'],rtol=0,atol=1e-9)
    offsets=np.linalg.norm(np.array(geom['requested_centres_xyz_mm'])-geom['actual_centres_xyz_mm'],axis=1)
    output={'state_id':'phase_00','rr_percent':14,'test_current_mA':1,
            'note':'1 mA is an illustrative linear rescaling, not a measured instrument current. Potentials use the solver ground; only differences are gauge invariant.',
            'contact_order':['I+','V+','V-','I-'],'contacts':rows,
            'tepc7_requested_to_actual_offsets_mm':offsets.tolist(),
            'tepc7_raw_json_sha256':core.sha(rawpath),
            'tepc7_raw_roles_verified':True,
            'tepc7_geometry_sha256':core.sha(ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/tepc_preparation_20260911/contacts/tepc_7_geometry.json'),
            'plan_sha256':core.sha(run/'plan.json')}
    dest=ROOT/'Colab Notebooks/cardiac_4d_segmentation/results/nix_cardiac_cycle_20260916/transfer_impedance_audit.json'
    core.write(dest,output)
    print('TEPC-7 requested/actual max offset mm:',max(offsets))
    print(dest)
if __name__=='__main__':export()
