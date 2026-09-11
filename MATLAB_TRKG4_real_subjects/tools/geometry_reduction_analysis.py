"""Inverse-recovery benchmark using independently solved CT/FEM holdouts.

No experimental curve is fitted here. FEM-based functions retain the fixed
outer body and nodal point contacts; planar formulas have a different outer
boundary/contact geometry and are reported as a separate family.
"""
from pathlib import Path
import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import minimize_scalar

from geometry_reduction_study import DEFAULT_OUT, ROOT, SOURCE, MODEL_LABELS, write_json, ellipsoid_interval
from compare_curved_two_tissue import analytic_curves, sha

ANALYTIC = ['planar_surface_h','adapted_mean_h','adapted_mean_kernel','adapted_finite_interval']


def curve_at(data, ratio, column='f_ohm_per_ohm_m'):
    representatives=data.groupby('ratio_index',sort=True).ratio.first()
    chosen_index=representatives.index[np.argmin(abs(representatives.to_numpy()-ratio))]
    chosen=float(representatives.loc[chosen_index])
    if not np.isclose(chosen,ratio,rtol=1e-10,atol=1e-12):
        raise ValueError(f'Missing independently solved ratio {ratio}')
    rows=data[data.ratio_index==chosen_index].sort_values('L_mm')
    if len(rows)!=9: raise ValueError('Incomplete FEM size series')
    return rows[column].to_numpy()


def recover(spline, y, indices=None):
    """Profile positive scale rho1 out; search all resolved contrast minima."""
    y=np.asarray(y,dtype=float)
    indices=np.arange(len(y)) if indices is None else np.asarray(indices)
    target=y[indices]
    lower,upper=float(spline.x[0]),float(spline.x[-1])
    grid=np.linspace(lower,upper,301)

    def score(log_ratio, detail=False):
        f=np.asarray(spline(log_ratio))[indices]
        scale=max(float(f@target/(f@f)),1e-12)
        residual=scale*f-target
        value=float(residual@residual)
        return (value,scale) if detail else value

    f=spline(grid)[:,indices]
    scale=np.maximum((f@target)/np.sum(f*f,axis=1),1e-12)
    loss=np.sum((scale[:,None]*f-target)**2,axis=1)
    candidates=[(score(lower),lower),(score(upper),upper)]
    for k in np.flatnonzero((loss[1:-1]<=loss[:-2])&(loss[1:-1]<=loss[2:]))+1:
        opt=minimize_scalar(score,bounds=(grid[k-1],grid[k+1]),method='bounded',options={'xatol':1e-11})
        candidates.append((float(opt.fun),float(opt.x)))
    candidates.sort()
    value,log_ratio=candidates[0];_,rho1=score(log_ratio,True)
    ratio=float(np.exp(log_ratio));rho2=rho1*ratio
    prediction=rho1*spline(log_ratio)
    f=spline(log_ratio);d=spline(log_ratio,1)
    jac=rho1*np.column_stack([f-d,d])
    singular=np.linalg.svd(jac[indices],compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>1e-14 else 1e99
    tied=sum(v<=value+max(1e-10,value*1e-6) for v,_ in candidates)
    return {'rho1':float(rho1),'rho2':float(rho2),'ratio':ratio,
            'train_rmse':float(np.sqrt(value/len(indices))),
            'all_rmse':float(np.sqrt(np.mean((prediction-y)**2))),
            'condition_log_parameters':condition,
            'contrast_at_bound':bool(min(log_ratio-lower,upper-log_ratio)<1e-6),
            'resolved_profile_minima':len(candidates)-2,'tied_candidates':tied},prediction,jac


def load_models(out, build_analytic=True):
    g=json.loads((out/'geometry.json').read_text(encoding='utf-8'))
    models={};tables={};knots=np.array(g['knots'])
    for name in g['model_names']:
        data=pd.read_csv(out/f'pem_{name}.csv')
        tables[name]=data
        values=np.stack([curve_at(data,k) for k in knots])
        derivatives=np.stack([curve_at(data,k,'df_dlog_ratio') for k in knots])
        models[name]=CubicHermiteSpline(np.log(knots),values,derivatives,axis=0,extrapolate=False)
    if build_analytic:
        source_g=g['reference_geometry'].copy()
        profile=pd.read_csv(SOURCE/'profile.csv')
        # Dense purely analytic grid; never calibrated to FEM or observations.
        aknots=np.geomspace(.5,32,97)
        arrays={name:[] for name in ANALYTIC}
        for ratio in aknots:
            gg=dict(source_g,rho_soft_ohm_m=1.,rho_lung_ohm_m=float(ratio))
            curves=analytic_curves(gg,profile,samples=401)
            for name in ANALYTIC: arrays[name].append(curves[name].to_numpy())
        packed={'ratios':aknots}
        for name in ANALYTIC:
            packed[name]=np.array(arrays[name]);models[name]=CubicSpline(np.log(aknots),packed[name],axis=0,extrapolate=False)
        np.savez_compressed(out/'analytic_library.npz',**packed)
    else:
        with np.load(out/'analytic_library.npz') as data:
            for name in ANALYTIC:models[name]=CubicSpline(np.log(data['ratios']),data[name],axis=0,extrapolate=False)
    return g,models,tables



def ray_diagnostics(out,g):
    # First interval of each actual union, including the retained distal CT.
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    stl=ROOT/'data/nik/stl/lungs_solid_v3_volume_fill.stl'
    assert sha(stl)==g['reference_geometry']['lung_stl_sha256']
    reader=vtk.vtkSTLReader();reader.SetFileName(str(stl));reader.Update()
    tree=vtk.vtkOBBTree();tree.SetDataSet(reader.GetOutput());tree.BuildLocator()
    centre=np.array(g['reference_geometry']['centre_mm']);frame=np.array(g['frame_columns_s_t_d'])
    direction=frame[:,2];rotation=np.array(g['ellipsoid_rotation']);radii=np.array(g['ellipsoid_radii_mm'])
    quadratic=(rotation/radii**2)@rotation.T;split=g['geometric_split_depth_mm']
    def first_interval(intervals):
        intervals=sorted((a,b) for a,b in intervals if b>a)
        if not intervals:return np.nan,np.nan
        start,end=intervals[0]
        for a,b in intervals[1:]:
            if a<=end:end=max(end,b)
            else:break
        return start,end
    rows=[]
    for ss in np.arange(-80,81,10):
        for tt in np.arange(-60,61,10):
            origin=centre+ss*frame[:,0]+tt*frame[:,1];hits=vtk.vtkPoints()
            tree.IntersectWithLine(origin-100*direction,origin+600*direction,hits,None)
            depth=sorted(float(x) for x in ((vtk_to_numpy(hits.GetData())-origin)@direction) if x>0) if hits.GetNumberOfPoints() else []
            assert len(depth)%2==0,'Unpaired ray intersections'
            intervals=list(zip(depth[::2],depth[1::2]));reference=first_interval(intervals)
            distal=[(max(a,split),b) for a,b in intervals if b>split]
            row={'s_mm':ss,'t_mm':tt,'entry_reference_mm':reference[0],'exit_reference_mm':reference[1]}
            for key,centre_key in [('ellipsoid','ellipsoid_centre_mm'),('anchored','anchored_centre_mm')]:
                a,b=ellipsoid_interval(origin,direction,np.array(g[centre_key]),quadratic)
                union=distal+([(float(a),float(b))] if np.isfinite(a) and b>a else [])
                a,b=first_interval(union);row[f'entry_{key}_mm']=a;row[f'exit_{key}_mm']=b
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'ray_diagnostics.csv',index=False)


def analyse(out):
    started=time.perf_counter()
    identity=json.loads((out/'fem_identity.json').read_text(encoding='utf-8'))
    assert sha(out/'geometry.json')==identity['geometry_sha256'], 'Stale FEM geometry'
    assert sha(ROOT/'src/run_trkg4_geometry_reduction.m')==identity['source_sha256'], 'FEM producer changed'
    assert sha(ROOT/'src/trkg4_cem_contact_matrix.m')==identity['contact_source_sha256'], 'CEM assembly changed'
    g,models,tables=load_models(out)
    assert sha(out/'geometry_masks.mat')==g['masks_sha256'], 'Stale material masks'
    ray_diagnostics(out,g)
    rho1=float(g['reference_geometry']['rho_soft_ohm_m']);rho2=float(g['reference_geometry']['rho_lung_ohm_m'])
    sizes=np.array(g['reference_geometry']['L_mm'])
    nominal=rho2/rho1
    old=pd.read_csv(SOURCE/'fem.csv').sort_values('L_mm')
    cem=pd.read_csv(out/'cem_reference.csv').sort_values('L_mm')
    reproduction=float(np.max(abs(cem.Z_cem_ohm.to_numpy()-old.Z_closed_lung_CEM_ohm.to_numpy())))
    assert reproduction<1e-6, f'CEM reference changed: {reproduction}'
    nominal_truth=rho1*curve_at(tables['reference'],nominal)

    validation=[]
    for name in g['model_names']:
        for ratio in g['truth_ratios']+g['check_ratios']:
            exact=curve_at(tables[name],ratio)
            residual=rho1*(models[name](np.log(ratio))-exact)
            validation.append({'model':name,'ratio':ratio,'max_abs_ohm':float(max(abs(residual))),
                               'rms_ohm':float(np.sqrt(np.mean(residual**2))),
                               'max_relative':float(max(abs(residual)/(rho1*np.maximum(abs(exact),1e-12))))})
    profile=pd.read_csv(SOURCE/'profile.csv')
    for ratio in g['truth_ratios']:
        gg=dict(g['reference_geometry'],rho_soft_ohm_m=1.,rho_lung_ohm_m=ratio)
        exact=analytic_curves(gg,profile,samples=801)
        for name in ANALYTIC:
            residual=rho1*(models[name](np.log(ratio))-exact[name].to_numpy())
            validation.append({'model':name,'ratio':ratio,'max_abs_ohm':float(max(abs(residual))),
                               'rms_ohm':float(np.sqrt(np.mean(residual**2))),
                               'max_relative':float(max(abs(residual)/(rho1*np.maximum(abs(exact[name].to_numpy()),1e-12))))})
    validation=pd.DataFrame(validation);validation.to_csv(out/'interpolation_validation.csv',index=False)
    assert validation.max_relative.max()<3e-4,'Interpolation is not sufficiently resolved; refine library before inference'
    homogeneous=max(float(np.max(abs(curve_at(t,1)-curve_at(tables['reference'],1)))) for t in tables.values())
    assert homogeneous<1e-6, 'Geometry-dependent homogeneous limit'

    forward=[];inverse=[];pair_rows=[];predictions=[];sensitivity=[]
    pairs=list(itertools.combinations(range(9),2))
    for ci,ratio in enumerate(g['truth_ratios']):
        true_rho2=rho1*ratio
        truth=rho1*curve_at(tables['reference'],ratio)
        for name,spline in models.items():
            direct=rho1*spline(np.log(ratio))
            fit,pred,jac=recover(spline,truth)
            inverse.append(dict(case_index=ci,model=name,true_rho1=rho1,true_rho2=true_rho2,
                rho1_error_pct=100*(fit['rho1']/rho1-1),rho2_error_pct=100*(fit['rho2']/true_rho2-1),**fit))
            forward.append({'case_index':ci,'model':name,'rmse_ohm':float(np.sqrt(np.mean((direct-truth)**2))),
                            'max_abs_ohm':float(max(abs(direct-truth)))})
            for i,L in enumerate(sizes):
                predictions.append({'case_index':ci,'model':name,'L_mm':L,'truth_ohm':truth[i],
                                    'before_fit_ohm':direct[i],'after_fit_ohm':pred[i]})
            if ci==1:
                f=spline(np.log(ratio));d=spline(np.log(ratio),1)
                for i,L in enumerate(sizes):
                    sensitivity.append({'model':name,'L_mm':L,'dZ_drho1':float(f[i]-d[i]),'dZ_drho2':float(d[i]/ratio)})
            for i,j in pairs:
                fitted,full,_=recover(spline,truth,[i,j])
                held=np.array([k for k in range(9) if k not in (i,j)])
                worst=0.
                for signs in itertools.product([-1,1],repeat=2):
                    disturbed=truth.copy();disturbed[[i,j]]*=1+.01*np.array(signs)
                    test,_,_=recover(spline,disturbed,[i,j])
                    worst=max(worst,abs(test['rho1']/rho1-1),abs(test['rho2']/true_rho2-1))
                pair_rows.append({'case_index':ci,'model':name,'L1_mm':sizes[i],'L2_mm':sizes[j],
                    'rho1_error_pct':100*(fitted['rho1']/rho1-1),'rho2_error_pct':100*(fitted['rho2']/true_rho2-1),
                    'holdout_rmse_ohm':float(np.sqrt(np.mean((full[held]-truth[held])**2))),
                    'worst_parameter_error_at_1pct_pct':100*worst,**fitted})
    pd.DataFrame(forward).to_csv(out/'forward_errors.csv',index=False)
    pd.DataFrame(inverse).to_csv(out/'inverse_recovery.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'pair_recovery.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'sensitivities.csv',index=False)

    # Actual CEM baseline is an electrode-model mismatch scenario, not a
    # synthetic matched-contact reference for the PEM geometry experiment.
    contact_inverse=[]
    for name,spline in models.items():
        fit,_,_=recover(spline,cem.Z_cem_ohm.to_numpy())
        contact_inverse.append(dict(model=name,rho1_error_pct=100*(fit['rho1']/rho1-1),rho2_error_pct=100*(fit['rho2']/rho2-1),**fit))
    pd.DataFrame(contact_inverse).to_csv(out/'cem_target_recovery.csv',index=False)
    contact=cem[['L_mm','Z_cem_ohm','Z_requested_pem_ohm']].copy()
    contact['Z_centroid_pem_ohm']=nominal_truth
    contact['CEM_minus_centroid_PEM_ohm']=contact.Z_cem_ohm-nominal_truth
    contact['requested_minus_centroid_PEM_ohm']=contact.Z_requested_pem_ohm-nominal_truth
    contact.to_csv(out/'contact_comparison.csv',index=False)

    times=[]
    for _ in range(21):
        tic=time.perf_counter();recover(models['reference'],nominal_truth);times.append(time.perf_counter()-tic)
    summary={'status':'exploratory_hypothesis_not_validated','rho1_true':rho1,'rho2_nominal':rho2,
        'nominal_case_index':1,'true_ratios':g['truth_ratios'],'contrast_search_bounds':[.5,32.],
        'two_point_perturbation_fraction':.01,'perturbation_is_scenario_not_measurement_noise':True,
        'CEM_reproduction_max_abs_ohm':reproduction,'homogeneous_model_spread_ohm_per_ohm_m':homogeneous,
        'interpolation_max_abs_ohm':float(validation.max_abs_ohm.max()),'interpolation_max_relative':float(validation.max_relative.max()),
        'FEM_library_factorizations':sum(t.ratio_index.nunique() for t in tables.values()),
        'FEM_library_elapsed_seconds':float(sum(t.groupby('ratio_index').elapsed_all_sizes_seconds.first().sum() for t in tables.values())),
        'online_inverse_median_seconds':float(np.median(times)),
        'CEM_minus_centroid_PEM_range_ohm':[float(contact.CEM_minus_centroid_PEM_ohm.min()),float(contact.CEM_minus_centroid_PEM_ohm.max())],
        'analysis_source_sha256':sha(__file__),'geometry_builder_current_sha256':sha(Path(__file__).with_name('geometry_reduction_study.py')),
        'fem_identity_sha256':sha(out/'fem_identity.json'),
        'input_sha256':{p.name:sha(p) for p in [out/'geometry.json',out/'geometry_masks.mat',out/'contacts.csv',out/'cem_reference.csv',*[out/f'pem_{n}.csv' for n in g['model_names']]]},
        'limitations':['One CT geometry and one volume mesh; no mesh/domain convergence established.',
        'Nodal point contacts are not finite CEM contacts; approximation error is measured separately at nominal resistivities.',
        'The synthetic reference is the original two-material PEM model, not measured tissue truth.',
        'All spatial reductions retain CT outer-body geometry and a distal lung-mask portion; they are not CT-free models.',
        'The proximal/distal split is an explicit geometric partition of a connected mask, not anatomical segmentation.',
        'Pair screening and deterministic perturbations on three contrasts do not establish an optimal clinical pair.',
        'Boundary-element, 2.5D and true mesh-refinement studies are not implemented in this notebook.'],
        'analysis_elapsed_seconds':time.perf_counter()-started}
    write_json(out/'analysis_summary.json',summary)
    print(pd.DataFrame(inverse).query('case_index==1')[['model','rho1','rho2','rho1_error_pct','rho2_error_pct','all_rmse','contrast_at_bound']].round(5).to_string(index=False))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','limitations']},ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();analyse(args.out.resolve())
