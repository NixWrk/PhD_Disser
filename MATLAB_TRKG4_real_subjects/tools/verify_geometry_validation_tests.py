"""Verify calculation identities and reported errors before notebook export."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from geometry_validation_tests import OUT, ROOT, sha, write_json


def verify(out=OUT):
    p=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    a=json.loads((out/'analysis_summary.json').read_text(encoding='utf-8'))
    c=json.loads((out/'cem_fit_summary.json').read_text(encoding='utf-8'))
    for name,h in p['input_sha256'].items():assert sha(ROOT/name)==h,name
    assert sha(out/'plan.json')==a['plan_sha256']==c['plan_sha256']
    assert sha(Path(__file__).with_name('geometry_validation_tests.py'))==a['analysis_source_sha256']
    assert sha(ROOT/'src/run_trkg4_geometry_validation_cem.m')==c['source_sha256']
    assert sha(out/'experimental_observations.csv')==c['observations_sha256']
    read=lambda s:pd.read_csv(out/(s+'.csv'))
    truth=read('reference_holdouts');s=read('synthetic_recovery');f=read('experimental_fits');h=read('conditional_holdouts')
    assert len(truth)==441 and not truth.duplicated(['case_index','L_mm']).any()
    assert len(s)==392 and not s.duplicated(['case_index','model']).any()
    assert len(f)==8 and len(h)==72 and not h.duplicated(['model','held_L_mm']).any()
    assert truth.relative_residual.max()<1e-7 and truth.reciprocity_abs_ohm.max()<1e-7
    np.testing.assert_allclose(s.rho1_error_pct,100*(s.rho1/s.true_rho1-1),atol=1e-10)
    np.testing.assert_allclose(s.rho2_error_pct,100*(s.rho2/s.true_rho2-1),atol=1e-10)
    bounds=np.asarray(p['bounds_ohm_m'])
    for name in ['synthetic_recovery','experimental_fits','conditional_holdouts','experimental_perturbations','narrow_lung_prior_fits','cem_multistart']:
        t=read(name)
        for i,k in enumerate(['rho1','rho2']):
            assert t[k].min()>=bounds[i,0]-1e-7 and t[k].max()<=bounds[i,1]+1e-7,name
    prediction=read('experimental_predictions')
    for name,g in prediction.groupby('model'):
        rmse=np.sqrt(np.mean((g.predicted_ohm-g.observed_ohm)**2))
        np.testing.assert_allclose(rmse,f.loc[f.model==name,'train_rmse'].iloc[0],atol=1e-10)
    np.testing.assert_allclose(h.held_residual_ohm,h.held_prediction_ohm-h.held_observed_ohm,atol=1e-10)
    dense=read('search_resolution_check')
    assert dense.rho1_difference.abs().max()<1e-4 and dense.rho2_difference.abs().max()<1e-4
    cem=read('cem_experimental_predictions')
    np.testing.assert_allclose(np.sqrt(np.mean((cem.predicted_ohm-cem.observed_ohm)**2)),c['best']['rmse_ohm'],atol=1e-8)
    assert (read('cem_multistart').exitflag>0).all()
    trace=read('cem_evaluation_trace')
    assert trace.relative_residual.max()<1e-7 and trace.reciprocity_abs_ohm.max()<1e-7
    d=json.loads((out/'cem_derivative_qc.json').read_text(encoding='utf-8'))
    assert d['passed'] and d['max_relative_error']<1e-4
    assert d['plan_sha256']==sha(out/'plan.json')
    assert d['prepared_sha256']==c['prepared_sha256']
    assert d['source_sha256']==sha(ROOT/'src/run_trkg4_geometry_validation_derivative_qc.m')
    assert len(read('cem_derivative_qc'))==8
    np.testing.assert_allclose(read('cem_derivative_qc').relative_error.max(),d['max_relative_error'])
    execution=json.loads((out/'fem_execution.json').read_text(encoding='utf-8'))
    assert execution['complete'] and execution['plan_sha256']==sha(out/'plan.json')
    assert execution['source_sha256']==sha(ROOT/'src/run_trkg4_geometry_validation.m')
    paths=[*out.glob('*.csv'),out/'plan.json',out/'analysis_summary.json',out/'fem_identity.json',out/'fem_execution.json',out/'cem_fit_summary.json',out/'cem_derivative_qc.json',out/'literature_selection.json']
    record={'technical_status':'passed','computational_status':'passed',
            'scientific_status':'exploratory_conditional_tests_not_in_vivo_accuracy_validation',
            'independent_CT_forward_cases':49,'synthetic_inversions':392,'experimental_models':8,
            'conditional_holdouts':72,'cem_starts':3,
            'source_sha256':sha(__file__),
            'artifacts':{x.name:sha(x) for x in paths}}
    write_json(out/'result_identity.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='artifacts'}))
    return record


if __name__=='__main__':verify()
