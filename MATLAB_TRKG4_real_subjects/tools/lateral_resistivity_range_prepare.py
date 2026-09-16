"""Freeze an expanded diagnostic resistivity experiment on existing CEM matrices."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/exploratory/lateral_array_design_20260916'
OUT=ROOT/'output/exploratory/lateral_resistivity_range_20260916'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(out, lower=.1, nx=13, ny=17):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():
        raise FileExistsError('Expanded plan already frozen; do not replace it.')
    old=json.loads((BASE/'plan.json').read_text(encoding='utf-8'))
    execution=json.loads((BASE/'responses.csv.execution.json').read_text(encoding='utf-8'))
    assert execution['complete'] and execution['finite_diameter_mm']==5
    assert execution['responses_sha256']==sha(BASE/'responses.csv')
    assert sha(ROOT/'src/run_lateral_array_design.m')==execution['identity']['prepared_identity']['worker_sha256']
    bounds=np.array([[lower,20.],[lower,100.]])
    assert 0<lower<1 and nx>=4 and ny>=4
    axes=[np.geomspace(*bounds[j],n) for j,n in enumerate([nx,ny])]
    previous=pd.read_csv(BASE/'responses.csv')
    anchor=previous[(previous.kind=='grid')&(previous.model=='variable_transverse')].drop_duplicates('point_id')
    rows=[]
    for j,(_,r) in enumerate(anchor.iloc[[10,38]].iterrows()):
        rows.append(dict(point_id=f'a{j}',kind='anchor',rho1=r.rho1,rho2=r.rho2))
    for i,r1 in enumerate(axes[0]):
        for j,r2 in enumerate(axes[1]):
            rows.append(dict(point_id=f'g{i:02}_{j:02}',kind='grid',rho1=r1,rho2=r2))
    # Deterministic hold-out points covering the newly added rectangle.
    fractions=[np.array([.08,.33,.62,.91]),np.array([.06,.29,.61,.93])]
    va=[np.exp(np.log(bounds[j,0])+fractions[j]*np.log(bounds[j,1]/bounds[j,0])) for j in range(2)]
    for i,r1 in enumerate(va[0]):
        for j,r2 in enumerate(va[1]):
            rows.append(dict(point_id=f'v{i}_{j}',kind='validation',rho1=r1,rho2=r2))
    pd.DataFrame(rows).to_csv(out/'points.csv',index=False,float_format='%.17g')
    files={'base_plan':BASE/'plan.json','prepared_fem':BASE/'prepared_fem.mat',
           'base_responses':BASE/'responses.csv','base_execution':BASE/'responses.csv.execution.json',
           'best_sets':BASE/'best_by_count.csv','base_fits':BASE/'real_fits.csv',
           'observations':ROOT/Path(old['observations_source'].replace('\\','/'))}
    plan=dict(status='exploratory_hypothesis_not_validated',task='diagnostic expanded solution curves',
        model='variable_transverse',bounds_ohm_m=bounds.tolist(),
        literature_bounds_ohm_m=old['bounds_ohm_m'],grid_rho1=axes[0].tolist(),grid_rho2=axes[1].tolist(),
        sizes_mm=old['sizes_mm'],finite_electrodes=True,diameter_mm=5.,
        contact_ohm_m2=old['contact_ohm_m2'],electrode_count_per_measurement=4,
        geometry_fixed=True,simultaneous_all_sizes=False,
        sources={k:{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for k,p in files.items()},
        points_sha256=sha(out/'points.csv'),preparation_source_sha256=sha(Path(__file__)),
        acceptance={'system_relative_residual_max':1e-7,'reciprocity_abs_ohm_max':1e-7,
                    'anchor_Z_abs_error_ohm_max':1e-6,
                    'interpolation_absolute_allowance_ohm':.02,'interpolation_relative_allowance':.001},
        notes=['Expanded bounds are diagnostic, not physiological reference intervals.',
               'rho=0 is excluded; no extrapolation of the former spline is used.',
               'Best size sets remain the ones selected within the literature scenario.',
               'Existing fixed-geometry CEM matrices are read in place, without copying medical inputs.'])
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(out,len(rows),'parameter points; grid',nx,ny,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--lower',type=float,default=.1)
    p.add_argument('--nx',type=int,default=13);p.add_argument('--ny',type=int,default=17)
    a=p.parse_args();prepare(a.out.resolve(),a.lower,a.nx,a.ny)
