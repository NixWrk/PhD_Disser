"""Check direct finite-contact responses at selected inverse solutions."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from lateral_array_design_analysis import (
    load_plan, load_response_tables, SplineSurrogate, sha256_file, save_json,
)

ROOT=Path(__file__).resolve().parents[1]


def run(out):
    out=Path(out).resolve()
    plan=load_plan(out/'plan.json')
    tables=load_response_tables(out/'responses.csv',plan)
    spline=SplineSurrogate(plan,tables['grid_z'])
    best=pd.read_csv(out/'best_by_count.csv').set_index('k')
    obs_path=ROOT/Path(plan['observations_source'].replace('\\','/'))
    assert sha256_file(obs_path)==plan['observations_sha256']
    obs=pd.read_csv(obs_path).sort_values('L_mm')
    cases=pd.read_csv(out/'direct_check_cases.csv')
    direct=pd.read_csv(out/'direct_checks_responses.csv')
    identity=json.loads((out/'direct_checks_responses.csv.execution.json').read_text())
    base_identity=json.loads((out/'responses.csv.execution.json').read_text())
    assert identity['complete'] and identity['responses_sha256']==sha256_file(out/'direct_checks_responses.csv')
    assert identity['identity']['requests_sha256']==sha256_file(out/'direct_checks.csv')
    assert identity['identity']['prepared_identity']==base_identity['identity']['prepared_identity']
    assert len(cases)==8 and len(direct)==72
    main=pd.read_csv(out/'responses.csv');rows=[]
    for _,case in cases.iterrows():
        d=direct[direct.point_id==case.point_id].sort_values('L_mm')
        assert np.array_equal(d.L_mm,plan['sizes'])
        assert d.relative_residual.max()<1e-7 and d.reciprocity_abs.max()<1e-7
        rho=np.array([case.rho1_hat,case.rho2_hat])
        assert np.allclose(d[['rho1','rho2']],rho,rtol=1e-12,atol=1e-12)
        predicted=spline.predict(np.log(rho)); z=d.Z.to_numpy()
        indices=json.loads(best.loc[case.k,'subset_indices_json'])
        if case.kind=='real':
            target=obs[f'Z_{case.state}_hold_ohm'].to_numpy()
        else:
            target=main[(main.model=='reference')&(main.point_id==case.source_point_id)].sort_values('L_mm').Z.to_numpy()
        assert len(target)==9
        actual_rmse=float(np.sqrt(np.mean((z[indices]-target[indices])**2)))
        spline_rmse=float(np.sqrt(np.mean((predicted[indices]-target[indices])**2)))
        rows.append({'point_id':case.point_id,'k':int(case.k),'kind':case.kind,
                     'max_spline_Z_error_ohm':float(np.max(np.abs(z-predicted))),
                     'selected_direct_rmse_ohm':actual_rmse,'selected_spline_rmse_ohm':spline_rmse,
                     'selected_rmse_change_ohm':abs(actual_rmse-spline_rmse)})
    pd.DataFrame(rows).to_csv(out/'direct_check_summary.csv',index=False)
    result={'status':'computed_direct_comparison','points':len(cases),'rows':len(direct),
            'max_spline_Z_error_ohm':max(r['max_spline_Z_error_ohm'] for r in rows),
            'max_selected_rmse_change_ohm':max(r['selected_rmse_change_ohm'] for r in rows),
            'direct_optimization_repeated':False,'independent_tissue_truth':False,
            'input_sha256':{n:sha256_file(out/n) for n in ['plan.json','responses.csv','best_by_count.csv','direct_checks.csv','direct_check_cases.csv','direct_checks_responses.csv']},
            'script_sha256':sha256_file(Path(__file__))}
    save_json(out/'direct_check_qc.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def prepare(out):
    out=Path(out).resolve()
    inv=pd.read_csv(out/'inverse_validation.csv');real=pd.read_csv(out/'real_fits.csv')
    rows=[];cases=[]
    for k in [2,3,5,9]:
        subset=inv[(inv.k==k)&(inv.target=='ct_reference')]
        r=subset.loc[subset.rho2_abs_error_pct.idxmax()];pid=f'ct_k{k}'
        rows.append(dict(point_id=pid,kind='directcheck',model='variable_transverse',rho1=r.rho1_hat,rho2=r.rho2_hat))
        cases.append(dict(point_id=pid,kind='ct_reference',k=k,source_point_id=r.point_id,state='',rho1_hat=r.rho1_hat,rho2_hat=r.rho2_hat))
    for _,r in real[real.k.isin([2,9])].iterrows():
        pid=f'{r.state}_k{int(r.k)}'
        rows.append(dict(point_id=pid,kind='directcheck',model='variable_transverse',rho1=r.rho1_hat,rho2=r.rho2_hat))
        cases.append(dict(point_id=pid,kind='real',k=int(r.k),source_point_id='',state=r.state,rho1_hat=r.rho1_hat,rho2_hat=r.rho2_hat))
    for name,data in [('direct_checks.csv',rows),('direct_check_cases.csv',cases)]:
        payload=pd.DataFrame(data).to_csv(index=False).encode('utf-8')
        path=out/name
        # Compare line endings semantically; preserve the existing frozen file bytes.
        if path.exists(): assert path.read_text(encoding='utf-8')==payload.decode().replace('\r\n','\n'),name+' changed'
        else: path.write_bytes(payload)
    print('Direct check requests prepared or verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'output/exploratory/lateral_array_design_20260916')
    p.add_argument('--prepare',action='store_true');args=p.parse_args()
    prepare(args.out) if args.prepare else run(args.out)
