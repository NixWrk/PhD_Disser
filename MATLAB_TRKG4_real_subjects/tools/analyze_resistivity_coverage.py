"""Postprocess frozen FEM results in rho; no forward solves or local probes.

Full output requires all 16 independently sealed checking runs. A failed
interpolation criterion is an informative result, never promoted to convergence.
"""
from pathlib import Path
import hashlib, itertools, json, sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/exploratory/resistivity_sensitivity_20260910'
OLD=BASE.parent/'arm_sigma_20260908'
sys.path.insert(0,str(OLD/'python_solver_deps'))
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from resistivity_sensitivity_metrics import conductivity_to_resistivity, range_scaled_metrics
PARAMS=['soft','heart','lung']


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def seal_read(folder):
    completion=json.loads((folder/'completion.json').read_text(encoding='utf-8'))
    assert sha(folder/'completion.json')==(folder/'completion.sha256').read_text().strip()
    assert completion['status']=='completed_numerical_checks_passed'
    for filename,expected in completion['output_sha256'].items():
        assert sha(folder/filename)==expected,filename
    return completion


def add_metrics(frame,profile):
    frame=frame.copy()
    sigma=frame[['sigma_'+k for k in PARAMS]].to_numpy()
    derivatives=frame[['dZ_dsigma_'+k for k in PARAMS]].to_numpy()
    rho,S=conductivity_to_resistivity(sigma,derivatives)
    metrics=range_scaled_metrics(S,np.array([np.ptp(profile['bounds'][k]) for k in PARAMS]))
    for j,k in enumerate(PARAMS):
        frame['rho_'+k]=rho[:,j]
        frame['S_'+k]=S[:,j]
        frame['B_'+k]=metrics['absolute_response'][:,j]
        frame['Q_'+k]=metrics['selectivity'][:,j]
        frame['N_'+k]=metrics['nuisance_to_target'][:,j]
    return frame


def load_coarse(profile):
    manifest=json.loads((OLD/'result_integrity.json').read_text(encoding='utf-8'))['files']
    frames=[]
    for path in sorted(OLD.glob('in*_results.csv')):
        assert sha(path)==manifest[path.name],path
        frame=pd.read_csv(path)
        frame['montage']=path.name.removesuffix('_results.csv')
        frame['state']=['original_'+str(x) for x in frame.case_index]
        frame['sample']='coarse'
        frame['Z_ohm']=frame.Z_direct_ohm
        frame['max_relative_residual']=frame.relative_residual
        frame['reciprocity_absolute_ohm']=abs(frame.Z_direct_ohm-frame.Z_reciprocal_ohm)
        frame=add_metrics(frame,profile)
        keep=np.ones(len(frame),bool)
        for k in PARAMS:
            lo,hi=profile['bounds'][k]
            keep&=frame['rho_'+k].between(lo-1e-9,hi+1e-9)
        assert keep.sum()==18
        frames.append(frame.loc[keep])
    return pd.concat(frames,ignore_index=True)


def load_checks(profile):
    frames=[]
    plan=json.loads((BASE/'coverage_plan.json').read_text(encoding='utf-8'))
    for montage in plan['montages']:
        folder=BASE/'fem'/montage
        c=seal_read(folder)
        assert c['channel_states']==4 and c['maps']==0
        study_path=BASE/'studies'/f'{montage}.json'
        identity=json.loads((folder/'identity.json').read_text(encoding='utf-8'))
        assert identity['study_sha256']==sha(study_path)
        study=identity['study']
        frame=pd.read_csv(folder/'channel_results.csv')
        lookup={s['id']:s['conductivity'] for s in study['states']}
        assert set(frame.state)==set(lookup) and len(frame)==4
        for k in PARAMS: frame['sigma_'+k]=[lookup[s][k] for s in frame.state]
        frame['sample']='check'
        frames.append(add_metrics(frame,profile))
    return pd.concat(frames,ignore_index=True)


def audit_interpolation(coarse,checks,profile):
    axes=[np.array(profile['original_rho_axes'][k]) for k in PARAMS]
    fields=['Z_ohm']+['S_'+k for k in PARAMS]
    results=[]
    for montage,g in coarse.groupby('montage'):
        cube=np.full(tuple(map(len,axes))+(len(fields),),np.nan)
        for _,r in g.iterrows():
            idx=tuple(int(np.argmin(abs(axis-r['rho_'+k]))) for k,axis in zip(PARAMS,axes))
            assert np.isnan(cube[idx]).all(), 'Duplicate grid state'
            cube[idx]=r[fields].to_numpy(float)
        assert np.isfinite(cube).all(), 'Incomplete coarse Cartesian grid'
        c=checks[checks.montage==montage]
        estimated=RegularGridInterpolator(axes,cube)(c[['rho_'+k for k in PARAMS]].to_numpy())
        scales=np.array([np.ptp(profile['bounds'][k]) for k in PARAMS])
        q_est=range_scaled_metrics(estimated[:,1:],scales)['selectivity']
        for n,(_,r) in enumerate(c.iterrows()):
            for j,k in enumerate(PARAMS):
                direct=r['S_'+k]; error=abs(estimated[n,j+1]-direct)
                floor=.005*g['S_'+k].abs().max()
                tolerance=.05*abs(direct)+floor
                qerror=abs(q_est[n,j]-r['Q_'+k])
                results.append(dict(montage=montage,state=r.state,compartment=k,
                    rho_soft=r.rho_soft,rho_heart=r.rho_heart,rho_lung=r.rho_lung,
                    direct_S=direct,interpolated_S=estimated[n,j+1],absolute_error=error,
                    relative_error=error/abs(direct) if direct else np.nan,
                    tolerance=tolerance,error_over_tolerance=error/tolerance,
                    derivative_pass=bool(error<=tolerance),direct_Q=r['Q_'+k],
                    interpolated_Q=q_est[n,j],Q_absolute_error=qerror,Q_pass=bool(qerror<=.005),
                    Z_direct_ohm=r.Z_ohm,Z_interpolated_ohm=estimated[n,0],
                    Z_relative_error=abs(estimated[n,0]-r.Z_ohm)/abs(r.Z_ohm)))
    return pd.DataFrame(results)


def endpoint_excursions(coarse,profile,axes=None):
    """Endpoint changes on a declared Cartesian rho grid.

    Values reconstructed as 1 / sigma can differ from the declared grid by a
    few floating-point units. Grouping those values directly may split one
    physical background state into several groups. Map every value to a
    verified declared grid index before grouping, while retaining canonical
    rho values in the output.
    """
    declared=profile['original_rho_axes'] if axes is None else axes
    indexed=coarse.copy();axis_values={}
    for key in PARAMS:
        values=np.asarray(declared[key],dtype=float)
        assert values.ndim==1 and len(values)>=2 and np.all(np.diff(values)>0)
        assert np.allclose(values[[0,-1]],profile['bounds'][key],rtol=1e-12,atol=1e-12)
        observed=indexed['rho_'+key].to_numpy(float)
        nearest=np.argmin(abs(observed[:,None]-values[None,:]),axis=1)
        assert np.allclose(observed,values[nearest],rtol=1e-10,atol=1e-12), 'State is not on the declared rho grid'
        indexed['_rho_index_'+key]=nearest
        axis_values[key]=values
    assert not indexed.duplicated(['montage']+['_rho_index_'+k for k in PARAMS]).any()
    rows=[]
    for montage,g in indexed.groupby('montage'):
        for target in PARAMS:
            others=[k for k in PARAMS if k!=target]
            group_keys=['_rho_index_'+k for k in others]
            for state,h in g.groupby(group_keys):
                state=(state,) if not isinstance(state,tuple) else state
                lo,hi=profile['bounds'][target]
                target_index='_rho_index_'+target
                low=h[h[target_index]==0]
                high=h[h[target_index]==len(axis_values[target])-1]
                assert len(low)==len(high)==1, 'Missing or duplicate endpoint on declared rho grid'
                low,high=low.iloc[0],high.iloc[0]
                signed=high.Z_ohm-low.Z_ohm
                background={f'background_rho_{key}':float(axis_values[key][index]) for key,index in zip(others,state)}
                ordered=h.sort_values(target_index).Z_ohm
                rows.append(dict(montage=montage,compartment=target,**background,
                     signed_excursion_ohm=signed,excursion_ohm=abs(signed),
                     endpoint_secant_per_m=signed/(hi-lo),
                     sampled_monotone=bool(np.all(np.diff(ordered)>=0) or np.all(np.diff(ordered)<=0))))
    return pd.DataFrame(rows)


def scorecards(all_states,endpoints,audit):
    rows=[]
    for montage,g in all_states.groupby('montage'):
        for k in PARAMS:
            S=g['S_'+k]; B=g['B_'+k]; Q=g['Q_'+k]; N=g['N_'+k]
            e=endpoints[(endpoints.montage==montage)&(endpoints.compartment==k)]
            a=audit[(audit.montage==montage)&(audit.compartment==k)]
            rows.append(dict(montage=montage,compartment=k,states=len(g),
                S_min=S.min(),S_max=S.max(),S_abs_min=S.abs().min(),S_abs_max=S.abs().max(),
                observed_sign_reversal=bool(S.min()<0<S.max()),B_min=B.min(),B_max=B.max(),
                Q_min=Q.min(),Q_max=Q.max(),nuisance_to_target_max=N.max(),
                E_min=e.excursion_ohm.min(),E_max=e.excursion_ohm.max(),endpoint_backgrounds=len(e),
                derivative_checks_pass=bool(a.derivative_pass.all()),Q_checks_pass=bool(a.Q_pass.all()),
                continuous_domain_certified=False))
    return pd.DataFrame(rows)


def main():
    profile=json.loads((BASE/'literature/frozen/rho_profile.json').read_text(encoding='utf-8'))
    coarse=load_coarse(profile); checks=load_checks(profile)
    all_states=pd.concat([coarse,checks],ignore_index=True)
    assert len(coarse)==288 and len(checks)==64
    audit=audit_interpolation(coarse,checks,profile)
    endpoints=endpoint_excursions(coarse,profile)
    cards=scorecards(all_states,endpoints,audit)
    spread_rows=[]
    for state,g in all_states.groupby(['rho_'+k for k in PARAMS]):
        assert len(g)==16 and g.montage.nunique()==16
        for k in PARAMS:
            values=g['S_'+k].abs()
            spread_rows.append(dict(zip(['rho_'+j for j in PARAMS],state)) | {
                'compartment':k,'S_abs_min_across_assemblies':values.min(),
                'S_abs_max_across_assemblies':values.max(),
                'relative_spread':(values.max()-values.min())/values.max()})
    spread=pd.DataFrame(spread_rows)
    out=BASE/'analysis'; out.mkdir(exist_ok=True)
    for name,frame in [('states_rho.csv',all_states),('coverage_audit.csv',audit),
                       ('endpoint_excursions.csv',endpoints),('assembly_scorecards.csv',cards),('same_state_assembly_spread.csv',spread)]:
        frame.to_csv(out/name,index=False)
    summary={'status':'completed_exploratory_analysis','coarse_states':len(coarse),'new_fem_states':len(checks),
      'derivative_checks':len(audit),'failed_derivative_checks':int((~audit.derivative_pass).sum()),
      'failed_selectivity_checks':int((~audit.Q_pass).sum()),
      'max_derivative_relative_error':float(audit.relative_error.max()),
      'max_Q_absolute_error':float(audit.Q_absolute_error.max()),
      'max_Z_relative_error':float(audit.Z_relative_error.max()),
      'max_new_system_relative_residual':float(checks.max_relative_residual.max()),
      'max_new_reciprocity_absolute_ohm':float(checks.reciprocity_absolute_ohm.max()),
      'interior_check_pass':bool(audit.derivative_pass.all() and audit.Q_pass.all()),
      'continuous_domain_certified':False,'mesh_convergence':'not_evaluated_in_this_study',
      'physical_validation':'not_evaluated','profile_sha256':sha(BASE/'literature/frozen/rho_profile.json'),
      'output_sha256':{p.name:sha(p) for p in out.glob('*.csv')}}
    (out/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
