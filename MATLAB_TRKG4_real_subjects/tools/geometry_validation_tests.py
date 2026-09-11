"""Approved synthetic and measured-data tests with literature-based box constraints.

The existing 20.13 libraries and data are immutable inputs. CT anatomy is
accepted; in-vivo resistivities are unknown. PEM and analytical fits to measured
data remain conditional on the old, full-data-derived electrode pose.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from geometry_reduction_analysis import load_models, recover
from geometry_reduction_study import ROOT, DEFAULT_OUT, MODEL_LABELS

OUT = ROOT/'output/exploratory/geometry_validation_20260910'
LITERATURE = ROOT/'output/exploratory/arm_sigma_20260908/literature/Thermal_dielectric_acoustic_MR properties_database_V5.0(ASCII).txt'
EXPERIMENT = ROOT/'data/nik/experimental/nik_breath_hold_levels.csv'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def recover_bounded(spline, y, bounds, indices=None, grid_size=401):
    """Profile scale *inside* both absolute bounds at every contrast.

    Bounds are a 2x2 array, [rho1_lo,rho1_hi], [rho2_lo,rho2_hi].
    A grid resolves candidate minima; no global uniqueness is asserted.
    """
    y=np.asarray(y,dtype=float);bounds=np.asarray(bounds,dtype=float)
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds<=0) or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError('Two finite positive nonzero-width intervals are required')
    if y.ndim!=1 or not np.isfinite(y).all():raise ValueError('Finite vector required')
    ids=np.arange(len(y)) if indices is None else np.asarray(indices,dtype=int)
    if len(ids)<2 or len(set(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(y)):
        raise ValueError('At least two distinct valid measurement indices required')
    r1lo,r1hi=bounds[0];r2lo,r2hi=bounds[1]
    lower=max(float(spline.x[0]),float(np.log(r2lo/r1hi)))
    upper=min(float(spline.x[-1]),float(np.log(r2hi/r1lo)))
    if lower>=upper:raise ValueError('The admissible box has no contrast interval in the library')
    target=y[ids]
    def score(x,detail=False):
        q=np.exp(x);f=np.asarray(spline(x))[ids]
        lo=max(r1lo,r2lo/q);hi=min(r1hi,r2hi/q)
        if lo>hi and not np.isclose(lo,hi,rtol=1e-12):return np.inf
        denom=float(f@f)
        if denom<=0 or not np.isfinite(denom):raise ValueError('Zero or invalid forward response')
        scale=float(np.clip(f@target/denom,lo,max(lo,hi)))
        residual=scale*f-target;loss=float(residual@residual)
        return (loss,scale) if detail else loss
    grid=np.linspace(lower,upper,grid_size)
    # Changes of the active bound are explicit candidate points.
    grid=np.unique(np.r_[grid,np.clip(np.log([r2lo/r1lo,r2hi/r1hi]),lower,upper)])
    f=spline(grid)[:,ids];q=np.exp(grid)
    scales=np.clip(f@target/np.sum(f*f,axis=1),np.maximum(r1lo,r2lo/q),np.minimum(r1hi,r2hi/q))
    loss=np.sum((scales[:,None]*f-target)**2,axis=1)
    candidates=[(score(x),float(x)) for x in [lower,upper,*np.clip(np.log([r2lo/r1lo,r2hi/r1hi]),lower,upper)]]
    for k in np.flatnonzero((loss[1:-1]<=loss[:-2])&(loss[1:-1]<=loss[2:]))+1:
        opt=minimize_scalar(score,bounds=(grid[k-1],grid[k+1]),method='bounded',options={'xatol':1e-11})
        candidates.append((float(opt.fun),float(opt.x)))
    value,x=min(candidates);_,r1=score(x,True);r2=r1*np.exp(x)
    prediction=np.asarray(r1*spline(x));f=spline(x);d=spline(x,1)
    jac=r1*np.column_stack([f-d,d]);s=np.linalg.svd(jac[ids],compute_uv=False)
    at=np.isclose(np.array([r1,r2])[:,None],bounds,rtol=2e-6,atol=1e-9)
    result={'rho1':r1,'rho2':float(r2),'ratio':float(np.exp(x)),
            'train_rmse':float(np.sqrt(value/len(ids))),
            'rho1_at_bound':bool(at[0].any()),'rho2_at_bound':bool(at[1].any()),
            'rho1_bound_side':'lower' if at[0,0] else 'upper' if at[0,1] else '',
            'rho2_bound_side':'lower' if at[1,0] else 'upper' if at[1,1] else '',
            'condition_log_parameters':float(s[0]/s[-1]) if s[-1]>1e-14 else 1e99,
            'smallest_singular_value_ohm':float(s[-1]),'contrast_search_grid':grid_size}
    return result,prediction,jac


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Use existing plan or a fresh output directory')
    # Frozen numerical source: original ITIS V5.0 table, not rounded web values.
    rows=list(csv.reader(LITERATURE.read_text(encoding='cp1252').splitlines(),delimiter='\t'))
    anchors=[]
    for line,row in enumerate(rows,1):
        if len(row)<48 or row[1] not in ['Muscle','Lung (Inflated)','Lung (Deflated)','Fat']:continue
        delta=np.array([float(row[i]) for i in [28,31,35,38]])
        tau=np.array([float(row[i]) for i in [29,32,36,39]])*np.array([1e-12,1e-9,1e-6,1e-3])
        alpha=np.array([float(row[i]) for i in [30,33,37,40]])
        omega=2*np.pi*50e3
        permittivity=sum(delta/(1+(1j*omega*tau)**(1-alpha)))
        conductivity=float(row[34])-omega*8.8542e-12*permittivity.imag
        anchors.append({'tissue':row[1],'source_line':line,'frequency_hz':50000,
                        'conductivity_50khz_S_m':float(conductivity),'resistivity_50khz_ohm_m':float(1/conductivity),
                        'LF_min_S_m':float(row[46]),'LF_max_S_m':float(row[47]),
                        'LF_min_rho_ohm_m':1/float(row[47]),'LF_max_rho_ohm_m':1/float(row[46]),
                        'LF_frequency_scope':'up_to_1MHz_pooled_not_frequency_specific_norm'})
    a={v['tissue']:v for v in anchors};muscle=a['Muscle'];lung=a['Lung (Inflated)']
    bounds=[[muscle['LF_min_rho_ohm_m'],muscle['LF_max_rho_ohm_m']],
            [min(lung['LF_min_rho_ohm_m'],lung['resistivity_50khz_ohm_m']),lung['LF_max_rho_ohm_m']]]
    # Rectangular grid in physical units; every target is independently solved.
    cases=[]
    for i,(r1,r2) in enumerate(itertools.product(np.geomspace(*bounds[0],7),np.geomspace(*bounds[1],7))):
        cases.append({'case_index':i,'rho1':float(r1),'rho2':float(r2),'ratio':float(r2/r1)})
    m=pd.read_csv(EXPERIMENT);accepted=m.loc[m.is_independent==1].sort_values('L_mm')
    if len(accepted)!=9 or not accepted.L_mm.is_unique:raise ValueError('Invalid independent size manifest')
    sizes=accepted.L_mm.tolist()
    if sizes!=json.loads((DEFAULT_OUT/'geometry.json').read_text(encoding='utf-8'))['reference_geometry']['L_mm']:
        raise ValueError('Experimental/library montage sizes differ')
    inputs=[LITERATURE,EXPERIMENT,DEFAULT_OUT/'prepared_fem.mat',DEFAULT_OUT/'geometry.json',
            DEFAULT_OUT/'fem_identity.json',DEFAULT_OUT/'analytic_library.npz',
            *[DEFAULT_OUT/f'pem_{n}.csv' for n in ['reference','plane','ellipsoid','ellipsoid_depth']]]
    plan={'schema':'geometry_validation_v1','status':'exploratory_hypothesis_not_validated',
          'approved_tests':['synthetic_known_parameters','experimental_unknown_parameters'],
          'author_approval_date':'2026-09-10','ct_anatomy_accepted':True,'in_vivo_rhos_known':False,
          'rho1_material':'all_nonlung_background; muscle/soft-tissue literature prior per author',
          'rho2_material':'lung; inflated anchors for inhale',
          'bounds_ohm_m':bounds,'bounds_status':'literature_based_working_envelope_not_population_normal_limits',
          'bounds_assumptions':['Muscle LF mixed-direction spread transferred to scalar effective nonlung background.',
             'Inflated-lung LF spread united with Gabriel/ITIS 50kHz point; LF spread is not a 50kHz population interval.',
             'No independently measured subject-specific resistivity and no empirical tissue mixture calibration.',
             'Inflated and deflated endpoints are not interpreted as ordinary inhale/exhale of this volunteer.'],
          'frequency_hz':50000,'anchors':anchors,'synthetic_cases':cases,'sizes_mm':sizes,
          'experimental_state':'inhale_hold','experimental_column':'Z_inhale_hold_ohm',
          'frozen_pose_was_fitted_to_all_sizes':True,'holdout_validation_status':'conditional_on_full_data_pose_not_independent_end_to_end',
          'electrode_family':'PEM_centroid_for_spatial_models; nominal_planar_points_for_analytic_models',
          'point_contact_approximation_remains_in_experimental_fit':True,
          'perturbation_fraction':.01,'perturbation_distribution':'deterministic_one_channel_at_a_time_not_measured_noise',
          'source_urls':['https://itis.swiss/virtual-population/tissue-properties/downloads/database-v5-0',
             'https://itis.swiss/virtual-population/tissue-properties/database/low-frequency-conductivity',
             'https://itis.swiss/virtual-population/tissue-properties/database/dielectric-properties'],
          'source_policy':'ITIS V5.0 fixed numerical corpus; other search results not used for numeric bounds',
          'zotero_search':'API closed; metadata snapshot checked in relevant collections then whole library; no conductivity-specific title found',
          'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
    write_json(out/'plan.json',plan)
    pd.DataFrame(anchors).to_csv(out/'literature_anchors.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'synthetic_cases.csv',index=False)
    # Do not copy source filenames or raw physiological recordings into reports.
    accepted[['L_mm','Z_inhale_hold_ohm']].to_csv(out/'experimental_observations.csv',index=False)
    print(json.dumps({'bounds':bounds,'synthetic_cases':len(cases),'sizes':sizes}))


def analyse(out):
    t0=time.perf_counter();plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    for p,h in plan['input_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError(f'Changed input: {p}')
    _,models,_=load_models(DEFAULT_OUT,build_analytic=False)
    bounds=plan['bounds_ohm_m'];sizes=np.array(plan['sizes_mm'])
    direct=pd.read_csv(out/'reference_holdouts.csv');cases=pd.read_csv(out/'synthetic_cases.csv')
    if len(direct)!=9*len(cases):raise ValueError('Incomplete independently solved synthetic targets')
    synth=[];direct_errors=[];validation=[]
    for case in cases.itertuples(index=False):
        target=direct.loc[direct.case_index==case.case_index].sort_values('L_mm')
        if not np.array_equal(target.L_mm,sizes):raise ValueError('Unmatched synthetic sizes')
        y=target.Z_ohm.to_numpy()
        validation.append({'case_index':case.case_index,'max_interpolation_ohm':float(np.max(abs(case.rho1*models['reference'](np.log(case.ratio))-y)))})
        for name,spline in models.items():
            fit,pred,_=recover_bounded(spline,y,bounds)
            synth.append({'case_index':case.case_index,'model':name,'true_rho1':case.rho1,'true_rho2':case.rho2,
                          'rho1_error_pct':100*(fit['rho1']/case.rho1-1),'rho2_error_pct':100*(fit['rho2']/case.rho2-1),**fit})
            delta=case.rho1*spline(np.log(case.ratio))-y
            direct_errors.append({'case_index':case.case_index,'model':name,'rms_ohm':float(np.sqrt(np.mean(delta**2))),
                                  'relative_vector_error_pct':float(100*np.linalg.norm(delta)/np.linalg.norm(y))})
    validation=pd.DataFrame(validation)
    if validation.max_interpolation_ohm.max()>.001:raise ValueError('Refine library: independent reference error exceeds 1 mOhm')
    obs=pd.read_csv(out/'experimental_observations.csv');y=obs.Z_inhale_hold_ohm.to_numpy()
    fitted=[];predictions=[];held=[];perturbed=[];grid_checks=[]
    for name,spline in models.items():
        fit,pred,_=recover_bounded(spline,y,bounds);fitted.append({'model':name,**fit})
        check,_,_=recover_bounded(spline,y,bounds,grid_size=1601)
        grid_checks.append({'model':name,'rho1_difference':check['rho1']-fit['rho1'],
                            'rho2_difference':check['rho2']-fit['rho2'],'rmse_difference':check['train_rmse']-fit['train_rmse']})
        for L,value,target in zip(sizes,pred,y):predictions.append({'model':name,'L_mm':L,'observed_ohm':target,'predicted_ohm':value,'residual_ohm':value-target})
        for k,L in enumerate(sizes):
            ids=np.delete(np.arange(len(y)),k);fold,full,_=recover_bounded(spline,y,bounds,ids)
            held.append({'model':name,'held_L_mm':L,'held_observed_ohm':y[k],'held_prediction_ohm':full[k],
                         'held_residual_ohm':full[k]-y[k],**fold})
            for sign in [-1,1]:
                noisy=y.copy();noisy[k]*=1+sign*plan['perturbation_fraction']
                test,_,_=recover_bounded(spline,noisy,bounds)
                perturbed.append({'model':name,'perturbed_L_mm':L,'sign':sign,
                     'rho1_shift_from_fit_pct':100*(test['rho1']/fit['rho1']-1),
                     'rho2_shift_from_fit_pct':100*(test['rho2']/fit['rho2']-1),**test})
    # A narrower lung prior tests sensitivity to the frequency-model/LF choice.
    lung=next(a for a in plan['anchors'] if a['tissue']=='Lung (Inflated)')
    narrow=[bounds[0],[lung['LF_min_rho_ohm_m'],lung['LF_max_rho_ohm_m']]]
    boundary_comparison=[]
    for name,spline in models.items():
        fit,_,_=recover_bounded(spline,y,narrow)
        boundary_comparison.append({'model':name,**fit})
    for filename,rows in [('synthetic_recovery',synth),('synthetic_forward_errors',direct_errors),
            ('interpolation_validation',validation),('experimental_fits',fitted),('experimental_predictions',predictions),
            ('conditional_holdouts',held),('experimental_perturbations',perturbed),('search_resolution_check',grid_checks),
            ('narrow_lung_prior_fits',boundary_comparison)]:
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    summary={'status':'exploratory_hypothesis_not_validated','synthetic_test_executed':True,'experimental_test_executed':True,
          'scientific_validation':'conditional_comparison_not_known_in_vivo_parameter_accuracy',
          'synthetic_case_count':len(cases),'model_count':len(models),'experimental_observations':len(y),
          'max_reference_interpolation_error_ohm':float(validation.max_interpolation_ohm.max()),
          'analysis_elapsed_seconds':time.perf_counter()-t0,
          'analysis_source_sha256':sha(__file__),'plan_sha256':sha(out/'plan.json'),
          'reference_sha256':sha(out/'reference_holdouts.csv')}
    write_json(out/'analysis_summary.json',summary)
    print(pd.DataFrame(fitted).to_string(index=False));print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','analyse']);p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args();globals()[args.stage](args.out.resolve())
