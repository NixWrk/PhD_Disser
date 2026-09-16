"""Numerical and provenance checks for notebook 33.09; no rerun of FEM."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(out):
    out = Path(out).resolve()
    plan = json.loads((out/'plan.json').read_text(encoding='utf-8'))
    identity = json.loads((out/'responses.csv.execution.json').read_text(encoding='utf-8'))
    assert identity['complete'] and identity['rows']==729
    assert identity['responses_sha256']==sha(out/'responses.csv')
    prepared_identity = identity['identity']['prepared_identity']
    assert prepared_identity['plan_sha256']==sha(out/'plan.json')
    assert prepared_identity['worker_sha256']==sha(ROOT/'src/run_lateral_array_design.m')
    assert plan['refined_mesh_sha256']==sha(out/'refined_mesh.mat')
    assert plan['points_sha256']==sha(out/'points.csv')
    assert plan['preparation_source_sha256']==sha(ROOT/'tools/lateral_array_design_prepare.py')
    assert plan['refinement_source_sha256']==sha(ROOT/'tools/refine_surface_contact_mesh.py')
    for name,digest in plan['source_hashes'].items():
        assert sha(ROOT/Path(name.replace('\\','/')))==digest, name
    data = pd.read_csv(out/'responses.csv')
    assert len(data)==729 and not data.duplicated(['point_id','model','L_mm']).any()
    grid=data[data.kind=='grid']; val=data[data.kind=='validation']
    assert len(grid)==49*9 and len(val)==16*2*9
    assert set(grid.model)=={'variable_transverse'}
    assert set(val.model)=={'variable_transverse','reference'}
    sizes=np.asarray(plan['sizes_mm'])
    for _,rows in data.groupby(['point_id','model']):
        assert np.array_equal(rows.L_mm.to_numpy(),sizes)
    for field in ['Z','dZ_dlogrho1','dZ_dlogrho2','relative_residual','reciprocity_abs']:
        assert np.isfinite(data[field]).all(),field
    assert data.relative_residual.max()<plan['acceptance']['system_relative_residual_max']
    assert data.reciprocity_abs.max()<plan['acceptance']['reciprocity_abs_ohm_max']
    contacts=pd.read_csv(out/'contact_qc.csv')
    assert len(contacts)==36 and contacts.actual_area_mm2.min()>0
    for c in ['area_relative_error','centroid_error_mm','radial_extent_mm']:
        assert contacts[c].max()<=plan['acceptance'][c+'_max']
    worst_resistance=np.inf
    worst_condition=0
    for _,rows in grid.groupby('point_id'):
        J=rows[['dZ_dlogrho1','dZ_dlogrho2']].to_numpy()
        sv=np.linalg.svd(J,compute_uv=False)
        assert sv[-1]>0 and np.linalg.matrix_rank(J)==2
        worst_resistance=min(worst_resistance,float(sv[-1]))
        worst_condition=max(worst_condition,float(sv[0]/sv[-1]))
    dq=json.loads((out/'derivative_qc.json').read_text())
    assert dq['finite_electrodes'] and dq['relative_error']<1e-5
    result={'status':'passed_declared_numerical_checks','scientific_status':'exploratory_hypothesis_not_validated',
            'response_rows':len(data),'grid_states':49,'offgrid_states_per_model':16,'contacts':len(contacts),
            'minimum_singular_value_all_nine_ohm':worst_resistance,
            'maximum_condition_all_nine':worst_condition,
            'global_spatial_convergence':False,'population_generalization_validated':False,
            'script_sha256':sha(Path(__file__)),'responses_sha256':sha(out/'responses.csv')}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'output/exploratory/lateral_array_design_20260916')
    verify(p.parse_args().out)
