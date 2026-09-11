"""Read a frozen refinement only after every required FEM batch is sealed.

The loader rejects missing states, altered provenance and duplicate attempts.
Its arrays are ready for the independent convergence metrics module.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/exploratory/resistivity_refinement_20260910'
OLD = ROOT / 'output/exploratory/arm_sigma_20260908'
sys.path.insert(0, str(OLD/'python_solver_deps'))
import numpy as np
import pandas as pd
from analyze_resistivity_coverage import add_metrics, endpoint_excursions
from run_electrode_sensitivity import verify_completed_output
PARAMS = ['soft','heart','lung']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read_plan(base=BASE):
    path=base/'refinement_plan.json'
    assert sha(path)==(base/'refinement_plan.sha256').read_text().strip(), 'Plan hash mismatch'
    plan=json.loads(path.read_text(encoding='utf-8'))
    profile=json.loads((base/'rho_profile.json').read_text(encoding='utf-8'))
    assert sha(base/'rho_profile.json')==plan['profile_sha256']
    return plan, profile


def locate_completed_batch(base, batch):
    original=base/batch['output']
    candidates=[original]+sorted(original.parent.glob(original.name+'_attempt_*'))
    completed=[p for p in candidates if (p/'completion.json').exists()]
    if not completed:
        raise FileNotFoundError('Unfinished batch: '+batch['id'])
    if len(completed)!=1:
        raise ValueError('More than one completed attempt: '+batch['id'])
    return completed[0]


def verify_reuse_model_binding(plan, study, old_identity, fixed, integrity, old_folder=OLD):
    """Require the complete legacy parameter, contact and channel contract."""
    assert old_identity['mesh_sha256']==plan['mesh_sha256']
    assert old_identity['prepared_fem_sha256']==plan['prepared_sha256']==study['source']['prepared_sha256']
    assert old_identity['study']['frequency_hz']==plan['frequency_hz']==study['source']['frequency_hz']==50000
    expected=[('soft',[0,1],False),('heart',[2],False),('lung',[3],False),('bone',[4],True)]
    assert [(p['id'],p['block_indices'],p['fixed']) for p in study['parameters']]==expected
    assert study['parameters'][3]['nominal_sigma']==fixed['bone_conductivity_S_m']
    assert all(s['conductivity']['bone']==fixed['bone_conductivity_S_m'] for s in study['states'])
    assert len(study['montages'])==1
    montage=study['montages'][0]
    name=montage['id']+'_contact.mat'
    assert Path(montage['contact_matrix']).name==name
    assert montage['contact_sha256']==integrity[name]
    assert sha(old_folder/name)==integrity[name]
    assert montage['electrodes']==['E1','E2','E3','E4']
    assert montage['channels']==[{'id':'standard','drive':[1,0,0,-1],'measure':[0,1,-1,0]}]
    return True


def load_results(base=BASE):
    plan,profile=read_plan(base)
    frames=[]; sources={}
    old_identity=json.loads((OLD/'identity.json').read_text(encoding='utf-8'))
    fixed=json.loads((OLD/'fixed_parameters.json').read_text(encoding='utf-8'))
    integrity=json.loads((OLD/'result_integrity.json').read_text(encoding='utf-8'))['files']
    assert sha(OLD/'fixed_parameters.json')==integrity['fixed_parameters.json']
    assert sha(OLD/'prepared_fem.mat')==plan['prepared_sha256']
    expected_sources={name:value for name,value in plan['sources_sha256'].items()
                      if name in ['run_electrode_sensitivity.py','compartment_sensitivity.py','electrode_sensitivity_contract.py']}
    for batch in plan['batches']:
        folder=locate_completed_batch(base,batch)
        verify_completed_output(folder,{'identity.json','source_used.zip','channel_results.csv'},
                                batch['states'],0,'exploratory_hypothesis_not_validated')
        identity=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
        path=base/batch['study']
        assert sha(path)==batch['study_sha256']==identity['study_sha256']
        assert identity['sources']==expected_sources, 'Solver source drift between batches'
        study=identity['study']
        assert study['source']['prepared_sha256']==plan['prepared_sha256']
        assert len(study['montages'])==1 and study['montages'][0]['id']==batch['montage']
        verify_reuse_model_binding(plan,study,old_identity,fixed,integrity)
        frame=pd.read_csv(folder/'channel_results.csv')
        lookup={s['id']:s['conductivity'] for s in study['states']}
        assert len(frame)==len(lookup)==batch['states'] and set(frame.state)==set(lookup)
        assert frame.montage.eq(batch['montage']).all() and frame.channel.nunique()==1
        for key in PARAMS: frame['sigma_'+key]=[lookup[s][key] for s in frame.state]
        frame['role']=batch['role']; frame['origin']='new_direct_fem'
        sources[str(folder.relative_to(base)/'completion.json')]=sha(folder/'completion.json')
        frames.append(frame)
    for montage in plan['montages']:
        items=[x for x in plan['reuse'] if x['montage']==montage]
        if not items: continue
        path=(base/items[0]['source_csv']).resolve()
        assert all(x['source_sha256']==sha(path) for x in items)
        frame=pd.read_csv(path)
        lookup={str(int(r.case_index)):r for _,r in frame.iterrows()}
        rows=[]
        for item in items:
            row=lookup[str(int(float(item['case_index'])))].to_dict()
            for key in PARAMS: assert abs(row['sigma_'+key]-item['conductivity'][key])<1e-12
            row.update(montage=montage,state=item['state'],role=item['role'],origin='verified_prior_direct_fem',
                       channel='standard',Z_ohm=row['Z_direct_ohm'],max_relative_residual=row['relative_residual'],
                       reciprocity_absolute_ohm=abs(row['Z_direct_ohm']-row['Z_reciprocal_ohm']))
            rows.append(row)
        frames.append(pd.DataFrame(rows))
        sources[items[0]['source_csv']]=sha(path)
    frame=add_metrics(pd.concat(frames,ignore_index=True),profile)
    expected={(s['montage'],s['state']):s for s in plan['states']}
    assert len(frame)==len(expected)==plan['total_required_states']
    assert not frame.duplicated(['montage','state']).any()
    assert set(zip(frame.montage,frame.state))==set(expected)
    for _,row in frame.iterrows():
        wanted=expected[(row.montage,row.state)]
        assert row.role==wanted['role']
        assert np.allclose([row['rho_'+k] for k in PARAMS], [wanted['rho'][k] for k in PARAMS],rtol=1e-12,atol=1e-12)
    numeric=['Z_ohm']+[prefix+k for prefix in ['rho_','S_','B_','Q_'] for k in PARAMS]
    assert np.isfinite(frame[numeric].to_numpy()).all()
    return plan,profile,frame,sources


def to_arrays(plan,frame):
    axes=[np.asarray(plan['axes_rho'][k]) for k in PARAMS]
    grid=np.full((5,5,5,len(plan['montages']),3),np.nan)
    checks=sorted([s for s in plan['states'] if s['montage']==plan['montages'][0] and s['role']=='check5'],key=lambda s:s['state'])
    check_index={s['state']:i for i,s in enumerate(checks)}
    points=np.asarray([[s['rho'][k] for k in PARAMS] for s in checks])
    direct=np.full((64,len(plan['montages']),3),np.nan)
    index={m:i for i,m in enumerate(plan['montages'])}
    for _,row in frame.iterrows():
        values=row[['S_'+k for k in PARAMS]].to_numpy(float)
        if row.role=='check5':
            direct[check_index[row.state],index[row.montage]]=values
        else:
            cell=tuple(int(np.argmin(abs(axis-row['rho_'+k]))) for k,axis in zip(PARAMS,axes))
            assert np.allclose([axes[j][cell[j]] for j in range(3)],[row['rho_'+k] for k in PARAMS],rtol=1e-12,atol=1e-12), 'State is not a declared grid node'
            assert np.isnan(grid[cell+(index[row.montage],)]).all()
            grid[cell+(index[row.montage],)]=values
    assert np.isfinite(grid).all() and np.isfinite(direct).all()
    return axes,grid,points,direct


def audit_tables(plan, audit, points):
    individual=[]; pairs=[]
    for c,point in enumerate(points):
        coordinates={"rho_"+k:float(point[j]) for j,k in enumerate(PARAMS)}
        for m,montage in enumerate(plan['montages']):
            for j,k in enumerate(PARAMS):
                direct=float(audit['direct_S'][c,m,j]);error=float(audit['derivative_error'][c,m,j]);tol=float(audit['derivative_tolerance'][c,m,j])
                individual.append(dict(check_index=c,montage=montage,compartment=k,**coordinates,
                    direct_S=direct,interpolated_S=float(audit['interpolated_S'][c,m,j]),
                    S_absolute_error=error,S_tolerance=tol,S_error_over_tolerance=error/tol if tol else (0. if error==0 else np.inf),
                    S_relative_error=error/abs(direct) if direct else np.nan,S_pass=bool(audit['derivative_pass'][c,m,j]),
                    S_sign_reversal=bool(audit['sign_reversal'][c,m,j]),direct_Q=float(audit['direct_Q'][c,m,j]),
                    interpolated_Q=float(audit['interpolated_Q'][c,m,j]),Q_absolute_error=float(audit['Q_absolute_error'][c,m,j]),
                    Q_pass=bool(audit['Q_pass'][c,m,j])))
        for p,(a,b) in enumerate(audit['pairs']):
            for j,k in enumerate(PARAMS):
                row=dict(check_index=c,assembly_a=plan['montages'][a],assembly_b=plan['montages'][b],compartment=k,**coordinates)
                for name,prefix in [('S','pair_S'),('Q','pair_Q')]:
                    error=float(audit[prefix+'_error'][c,p,j]);tol=float(audit[prefix+'_tolerance'][c,p,j])
                    direct_field='pair_direct_delta_abs_S' if name=='S' else 'pair_direct_delta_Q'
                    interp_field='pair_interpolated_delta_abs_S' if name=='S' else 'pair_interpolated_delta_Q'
                    row.update({name+'_direct_difference':float(audit[direct_field][c,p,j]),
                        name+'_interpolated_difference':float(audit[interp_field][c,p,j]),name+'_absolute_error':error,
                        name+'_tolerance':tol,name+'_error_over_tolerance':error/tol if tol else (0. if error==0 else np.inf),
                        name+'_accuracy_pass':bool(audit[prefix+'_accuracy_pass'][c,p,j]),
                        name+'_unresolved':bool(audit[prefix+'_unresolved'][c,p,j]),
                        name+'_preference_reversal':bool(audit[prefix+'_preference_reversal'][c,p,j])})
                pairs.append(row)
    return pd.DataFrame(individual),pd.DataFrame(pairs)


def score_tables(plan, comparison):
    stages={'grid3':comparison['scorecards'][3], 'grid5':comparison['scorecards'][5],
            'grid5_plus_checks':comparison['extra_scorecard']}
    rows=[];changes=[];preferred=[]
    for stage,card in stages.items():
        for m,montage in enumerate(plan['montages']):
            for j,k in enumerate(PARAMS):
                rows.append(dict(stage=stage,montage=montage,compartment=k,
                    min_B=float(card['min_B'][m,j]),min_Q=float(card['min_Q'][m,j])))
    transitions=[('3_vs_5','grid3','grid5'),('5_vs_5_plus_checks','grid5','grid5_plus_checks')]
    for key,left,right in transitions:
        comparison_item=comparison['refinement_comparisons'][key]
        for m,montage in enumerate(plan['montages']):
            for j,k in enumerate(PARAMS):
                for metric in ['B','Q']:
                    name='min_'+metric
                    changes.append(dict(transition=key,montage=montage,compartment=k,metric=name,
                        old=float(stages[left][name][m,j]),new=float(stages[right][name][m,j]),
                        absolute_change=abs(float(comparison_item['delta_'+name][m,j])),
                        tolerance=float(comparison_item[name+'_tolerance'][m,j]),
                        **{'pass':bool(comparison_item[name+'_pass'][m,j])}))
    for j,k in enumerate(PARAMS):
        for metric in ['B','Q']:
            a=stages['grid3']['preferred_'+metric][j]
            b=stages['grid5']['preferred_'+metric][j]
            c=stages['grid5_plus_checks']['preferred_'+metric][j]
            preferred.append(dict(compartment=k,metric='Минимальное '+metric,
                grid3_numbers=[i+1 for i in a],grid5_numbers=[i+1 for i in b],augmented_numbers=[i+1 for i in c],
                stable=bool(a==b==c),grid3_montages=[plan['montages'][i] for i in a],
                grid5_montages=[plan['montages'][i] for i in b],augmented_montages=[plan['montages'][i] for i in c]))
    return pd.DataFrame(rows),pd.DataFrame(changes),preferred


def write_analysis(plan,profile,frame,sources,axes,grid,points,direct,audit,cards,stability,preferred_sets):
    individual,pairs=audit_tables(plan,audit,points)
    endpoints=endpoint_excursions(frame[frame.role!='check5'],profile,plan['axes_rho'])
    assert len(endpoints)==16*3*25
    decisions={
        'Полнота матрицы и численные проверки FEM':True,
        'Интерполяция производных по сопротивлению':bool(individual.S_pass.all()),
        'Интерполяция избирательности':bool(individual.Q_pass.all()),
        'Интерполяция парных разностей модулей производных':bool(pairs.S_accuracy_pass.all()),
        'Интерполяция парных разностей избирательности':bool(pairs.Q_accuracy_pass.all()),
        'Устойчивость знаков численно различимых пар':bool(audit['all_pair_S_robust_sign_pass'] and audit['all_pair_Q_robust_sign_pass']),
        'Устойчивость минимальных показателей':bool(stability['pass'].all()),
        'Устойчивость предпочтительных множеств':all(x['stable'] for x in preferred_sets),
    }
    out=BASE/'analysis';out.mkdir(exist_ok=True)
    failures=individual.groupby(['check_index','rho_soft','rho_heart','rho_lung','compartment']).agg(
        maximum_S_error_over_tolerance=('S_error_over_tolerance','max'),
        failed_assemblies=('S_pass',lambda x:int((~x).sum()))).reset_index()
    for name,table in [('states_rho.csv',frame),('interpolation_checks.csv',individual),('pair_checks.csv',pairs),
                       ('scorecard_levels.csv',cards),('scorecard_stability.csv',stability),
                       ('endpoint_excursions.csv',endpoints),('failure_regions.csv',failures)]:
        table.to_csv(out/name,index=False)
    by_comp=[]
    for k in PARAMS:
        a=individual[individual.compartment==k];p=pairs[pairs.compartment==k]
        by_comp.append(dict(compartment=k,derivative_failures=int((~a.S_pass).sum()),
            max_derivative_relative_error=float(a.S_relative_error.max()),max_Q_absolute_error=float(a.Q_absolute_error.max()),
            pair_S_failures=int((~p.S_accuracy_pass).sum()),pair_Q_failures=int((~p.Q_accuracy_pass).sum()),
            unresolved_S_pairs=int(p.S_unresolved.sum()),unresolved_Q_pairs=int(p.Q_unresolved.sum())))
    summary=dict(status='completed_exploratory_refinement_analysis',states=len(frame),
        new_fem_states=plan['new_fem_states'],reused_states=plan['reused_states'],
        derivative_checks=len(individual),pair_checks=len(pairs),criteria_decisions_ru=decisions,
        adequate_on_tested_states=all(decisions.values()),criteria=plan['criteria'],by_compartment=by_comp,
        preferred_sets=preferred_sets,max_system_relative_residual=float(frame.max_relative_residual.max()),
        max_reciprocity_absolute_ohm=float(frame.reciprocity_absolute_ohm.max()),
        continuous_domain_certified=False,spatial_convergence_evaluated=False,physical_validation=False,
        axis_attribution='Cell centres alone do not isolate which parameter axis causes the interpolation error.',
        next_uniform_level_if_needed=9,next_level_computed=False,
        plan_sha256=sha(BASE/'refinement_plan.json'),profile_sha256=sha(BASE/'rho_profile.json'),
        input_completion_sha256=sources,
        analysis_sources_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('resistivity_grid_convergence.py'),Path(__file__).with_name('resistivity_sensitivity_metrics.py'),Path(__file__).with_name('analyze_resistivity_coverage.py')]},
        output_sha256={p.name:sha(p) for p in out.glob('*.csv')})
    temporary=out/'refinement_summary.tmp'
    temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(out/'refinement_summary.json')
    print(json.dumps({k:summary[k] for k in ['status','states','adequate_on_tested_states','criteria_decisions_ru','by_compartment']},ensure_ascii=False,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    plan,profile,frame,sources=load_results()
    axes,grid,points,direct=to_arrays(plan,frame)
    if args.validate_only:
        print('Verified',len(frame),'states and complete arrays',grid.shape,direct.shape)
        return
    from resistivity_grid_convergence import audit_grid, compare_nested_scorecards
    widths=np.asarray([np.ptp(profile['bounds'][k]) for k in PARAMS])
    audit=audit_grid(axes,grid,points,direct,widths,plan['criteria'])
    comparison=compare_nested_scorecards(axes,grid,widths,extra_values=direct,criteria=plan['criteria'])
    cards,stability,preferred=score_tables(plan,comparison)
    write_analysis(plan,profile,frame,sources,axes,grid,points,direct,audit,cards,stability,preferred)


if __name__=='__main__': main()
