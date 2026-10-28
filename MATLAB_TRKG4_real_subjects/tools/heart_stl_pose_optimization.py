"""Bounded deterministic electrical calibration of an equal-volume ellipsoid.

Three absolute impedances (minus/base/plus volume) are matched to the original
STL. Translation derivatives come from six executed +/-5 mm pose probes.
The fitted translation is a proposal until its independent FEM run is checked.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(base_report,pose_report,candidates_file,output):
    base=read(base_report);pose=read(pose_report)
    assert all(r['status']=='conditional_STL_geometry_electrical_calculations_completed' for r in [base,pose])
    rows={r['id']:r for p in [base,pose] for r in p['rows']}
    def curve(method):
        rs=[rows[f'heart4_{method}_v{n:03d}_q64'] for n in [95,100,105]]
        assert all(np.array_equal(rs[0]['rho_ohm_m'],r['rho_ohm_m']) for r in rs)
        return np.array([r['Ztt_ohm'] for r in rs])
    target=curve('STL');initial=curve('L1')
    A=np.column_stack([(curve(f'L1_{axis}p5')-curve(f'L1_{axis}m5'))/10 for axis in 'xyz'])
    solution=lsq_linear(A,target-initial,bounds=(-5,5),tol=1e-14,lsmr_tol=1e-14)
    delta=solution.x;prediction=initial+A@delta
    nominal=read(candidates_file);c=copy.deepcopy(nominal['L1'])
    c['center_mm']=(np.array(c['center_mm'])+delta).tolist()
    c['method_id']='L1_electrical_pose';c['electrical_translation_from_nominal_mm']=delta.tolist()
    c['status']='linearized_electrical_fit_requires_direct_FEM_validation'
    nominal['L1_electrical_pose']=c
    output.mkdir(parents=True,exist_ok=True)
    write(output/'candidate.local.json',nominal)
    sv=np.linalg.svd(A,compute_uv=False)
    result={'status':'proposed_pose_requires_direct_FEM_verification','method':'bounded_linear_least_squares_on_three_FEM_impedances',
        'translation_bounds_mm':[-5,5],'bounds_basis':'explicit computational search region, not measured installation uncertainty',
        'target_volume_steps':'same absolute +/-5 percent of source STL volume',
        'delta_center_mm':delta.tolist(),'volume_preserved':True,'target_Z_ohm':target.tolist(),
        'initial_Z_ohm':initial.tolist(),'predicted_Z_ohm':prediction.tolist(),
        'translation_jacobian_ohm_per_mm':A.tolist(),'singular_values_ohm_per_mm':sv.tolist(),
        'condition':float(sv[0]/sv[-1]),'initial_RMSE_mohm':float(np.sqrt(np.mean((initial-target)**2))*1000),
        'linearized_RMSE_mohm':float(np.sqrt(np.mean((prediction-target)**2))*1000),
        'active_bounds':solution.active_mask.tolist(),'solver_success':bool(solution.success),
        'sources_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [base_report,pose_report,candidates_file]},
        'AI_training':False,'clinical_validation':False}
    write(output/'pose_optimization.local.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--poses',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.base,a.poses,a.candidates,a.output)
