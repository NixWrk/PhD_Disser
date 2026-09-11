"""Postprocess a stopped exhaustive CEM scan; never run or resume FEM.

Tasks: stable experimental localization, common-h planar fits, held-out sizes,
and joint regions. Geometry export and reader rendering are separate consumers.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from threadpoolctl import threadpool_limits
from scan_layered_library import build_library,fit_batch,planar

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=ROOT/'output/full_pose_scan_accel_pilot_20260909_01'
DEFAULT_OUT=ROOT/'output/exploratory/scan_localization_20260910'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,obj):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def prepare(run,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'input_contract.json').exists():
        contract=read(out/'input_contract.json')
        assert contract['source_run']==str(run.resolve())
        for x in contract['inputs']:assert sha(x['path'])==x['sha256'],x['path']
        assert sha(out/'scan_arrays.npz')==contract['array_sha256']
        print('Verified prepared scan reused',flush=True);return
    spec=importlib.util.spec_from_file_location('full_scan_report',ROOT/'notebooks/build_trkg4_full_scan_report.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    scan=report.load_scan(run/'live',allow_partial=True)
    if scan['summary'].get('streaming_active'):raise ValueError('A stopped, stable snapshot is required')
    frames=[pd.read_csv(run/'live'/p) for p in scan['summary']['pose_files']]
    df=pd.concat(frames,ignore_index=True);valid=df.loc[df.status=='evaluated'].copy()
    flat=(valid.centre_index.to_numpy(int)-1)*len(scan['phi'])+pd.Index(scan['phi']).get_indexer(valid.phi_deg)
    order=np.argsort(flat);valid=valid.iloc[order];flat=flat[order]
    assert np.array_equal(flat,np.flatnonzero(scan['status'].ravel()==1))
    sizes=scan['summary']['experiment']['L_mm'];cols=[f'Z_L{int(x):03d}_ohm' for x in sizes]
    values=dict(flat=flat,Z_fem=valid[cols].to_numpy(float),rmse_experiment=valid.rmse_ohm.to_numpy(float),
        status=scan['status'],u=scan['u'],v=scan['v'],phi=scan['phi'],sizes=np.array(sizes),
        experiment=np.array(scan['summary']['experiment']['Z_ohm']),
        rho=np.array([scan['summary']['rho1'],scan['summary']['rho2']]),
        centres=scan['centres'][['centre_index','u_mm','v_mm','x_mm','y_mm','z_mm','h_vertex_mm']].to_numpy(float))
    np.savez_compressed(out/'scan_arrays.npz',**values)
    inputs=[dict(path=str(p),sha256=d) for p,d in scan['verified'].items()]
    for x in inputs:assert sha(x['path'])==x['sha256']
    contract=dict(source_run=str(run.resolve()),source_fingerprint=scan['summary']['fingerprint'],
        inputs=inputs,array_sha256=sha(out/'scan_arrays.npz'),counts=scan['counts'],
        status='exploratory_hypothesis_not_validated',complete_source=False,
        fixed_rho_source='previous joint fit, not independently measured tissue properties',
        FEM_materials=['soft_tissue','lungs','heart','bones'],
        contacts='saved finite CEM; planar comparison has nominal point contacts',
        user_scope='processed poses only; no new full FEM search',
        coordinate_interpretation='model-compatible candidates, not measured installation coordinates')
    write(out/'input_contract.json',contract)
    print('Prepared',len(flat),'valid curves,',scan['counts'],flush=True)

def fit(out):
    d=np.load(out/'scan_arrays.npz');r1,r2=d['rho'];sizes=d['sizes'];y=d['Z_fem'];n=len(y)
    lib=build_library(sizes,r1,r2);folder=out/'fits';folder.mkdir(exist_ok=True)
    identity={'array_sha256':sha(out/'scan_arrays.npz'),
              'fitter_sha256':sha(ROOT/'tools/scan_layered_library.py'),
              'producer_sha256':sha(__file__),'epsilon_ohm':lib['epsilon_ohm']}
    fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    summary={k:v for k,v in lib.items() if k not in ['h','z','asymptote','sizes']}
    summary.update(fingerprint=fp,identity=identity,knots=len(lib['h']),
                   global_RMSE_suboptimality_bound_ohm=2*lib['epsilon_ohm'],
                   h_domain='0 <= h < infinity; asymptote explicitly considered',
                   h_interpretation='effective approximation parameter, not anatomical thickness')
    write(out/'fitter_contract.json',summary);np.savez_compressed(out/'analytic_library.npz',h_mm=lib['h'],z=lib['z'],asymptote=lib['asymptote'])
    timer=time.perf_counter();parts=[]
    with threadpool_limits(limits=4):
        for first in range(0,n,8192):
            stop=min(first+8192,n);path=folder/f'part_{first:07d}.npz';side=Path(str(path)+'.json')
            if path.exists() and side.exists():
                c=read(side);assert c['fingerprint']==fp and c['sha256']==sha(path)
            else:
                results=[]
                for lo in range(first,stop,1024):
                    target=y[lo:min(lo+1024,stop)];f=fit_batch(target,lib)
                    exact=planar(f['h_mm'],sizes,r1,r2)
                    assert np.max(abs(exact-f['prediction']))<=lib['epsilon_ohm']+1e-7
                    f['rmse_polyline_ohm']=f['rmse_ohm'];f['rmse_ohm']=np.sqrt(np.mean((exact-target)**2,axis=1))
                    f['max_error_ohm']=np.max(abs(exact-target),axis=1);f['prediction']=exact
                    for j,L in enumerate(sizes):
                        actual=planar(f['loo_h_mm'][:,j],[L],r1,r2)[:,0]
                        assert np.max(abs(actual-f['loo_predictions'][:,j]))<=lib['epsilon_ohm']+1e-7
                        f['loo_predictions'][:,j]=actual
                    f['loo_rmse_ohm']=np.sqrt(np.mean((f['loo_predictions']-target)**2,axis=1))
                    results.append(f)
                joined={k:np.concatenate([x[k] for x in results]) for k in results[0]}
                joined['first']=np.array(first);joined['stop']=np.array(stop)
                temp=Path(str(path)+'.tmp.npz');np.savez_compressed(temp,**joined);temp.replace(path)
                write(side,dict(fingerprint=fp,sha256=sha(path),first=first,stop=stop))
            parts.append(path)
            write(out/'analysis_progress.json',dict(stage='fitting_common_h_and_nine_holdouts',complete=False,
                curves_done=stop,curves_total=n,elapsed_seconds=time.perf_counter()-timer))
            print(f'COMMON h + LOO: {stop}/{n}; {time.perf_counter()-timer:.1f}s',flush=True)
    results=[np.load(p) for p in parts]
    fit={k:np.concatenate([p[k] for p in results]) for k in results[0].files if k not in ['first','stop']}
    assert len(fit['h_mm'])==n
    for k in ['fitter_sha256','producer_sha256']:
        p=ROOT/'tools/scan_layered_library.py' if k=='fitter_sha256' else Path(__file__)
        assert sha(p)==identity[k],'Source changed while fitting'
    np.savez_compressed(out/'layered_fits.npz',**fit)
    ef=fit_batch(d['experiment'][None,:],lib,leave_one_out=False)
    ez=planar(ef['h_mm'],sizes,r1,r2)[0]
    write(out/'fit_summary.json',dict(curves=n,common_h_per_curve=1,sizes=len(sizes),
        held_out_fits=n*len(sizes),seconds=time.perf_counter()-timer,
        fitter_fingerprint=fp,fit_sha256=sha(out/'layered_fits.npz'),
        finite_h_count=int(np.isfinite(fit['h_mm']).sum()),infinite_h_count=int(fit['infinite'].sum()),
        median_h_mm=float(np.median(fit['h_mm'][np.isfinite(fit['h_mm'])])),
        median_2L_FEM_rmse_ohm=float(np.median(fit['rmse_ohm'])),
        median_LOO_rmse_ohm=float(np.median(fit['loo_rmse_ohm'])),
        planar_experiment_best_h_mm=float(ef['h_mm'][0]),
        planar_experiment_min_rmse_ohm=float(np.sqrt(np.mean((ez-d['experiment'])**2))),
        scientific_status='approximation of fixed-parameter CEM, not validation of measured location'))

def dense_metric(d,values):
    a=np.full(d['status'].size,np.nan);a[d['flat']]=values;return a.reshape(d['status'].shape)

def robustness(d,values,radius=2):
    shape=(len(d['v']),len(d['u']),len(d['phi']));width=2*radius+1;count=width**3
    state=d['status'].reshape(shape);valid=state==1;known=state>0
    full=ndimage.minimum_filter(valid.astype(np.uint8),size=width,mode='constant',cval=0).astype(bool)
    seen=ndimage.minimum_filter(known.astype(np.uint8),size=width,mode='constant',cval=0).astype(bool)
    x=dense_metric(d,values).reshape(shape)
    worst=ndimage.maximum_filter(np.where(valid,x,np.inf),size=width,mode='constant',cval=np.inf)
    worst[~full]=np.nan
    fractions=ndimage.uniform_filter(valid.astype(float),size=width,mode='constant',cval=0)
    return worst.reshape(d['status'].shape),seen.reshape(d['status'].shape),fractions.reshape(d['status'].shape)

def regions(out):
    d=np.load(out/'scan_arrays.npz');f=np.load(out/'layered_fits.npz');n=len(d['flat']);a=len(d['phi'])
    exp=d['rmse_experiment'];two=f['rmse_ohm'];joint=np.maximum(exp,two)
    maps={'experiment':dense_metric(d,exp),'two_layer':dense_metric(d,two),
          'h_eff':dense_metric(d,f['h_mm']),'loo':dense_metric(d,f['loo_rmse_ohm']),
          'joint':dense_metric(d,joint)}
    for name,values in [('experiment',exp),('joint',joint),('two_layer',two)]:
        maps['robust_'+name],seen,fraction=robustness(d,values)
    maps['neighbourhood_known']=seen;maps['neighbourhood_valid_fraction']=fraction
    ci=d['flat']//a;ai=d['flat']%a;c=d['centres'];lookup=np.full(d['status'].size,-1,int);lookup[d['flat']]=np.arange(n)
    table=pd.DataFrame(dict(flat_index=d['flat'],centre_index=ci+1,u_mm=c[ci,1],v_mm=c[ci,2],phi_deg=d['phi'][ai],
        rmse_experiment_ohm=exp,h_eff_mm=f['h_mm'],rmse_2L_FEM_ohm=two,
        loo_rmse_ohm=f['loo_rmse_ohm'],loo_h_min_mm=f['loo_h_min_mm'],loo_h_max_mm=f['loo_h_max_mm'],
        max_2L_error_ohm=f['max_error_ohm'],h_at_infinity=f['infinite'],
        robust_experiment_ohm=maps['robust_experiment'].ravel()[d['flat']],
        robust_joint_ohm=maps['robust_joint'].ravel()[d['flat']],
        neighbourhood_known=seen.ravel()[d['flat']],neighbourhood_valid_fraction=fraction.ravel()[d['flat']]))
    table.to_csv(out/'pose_analysis.csv',index=False)
    thresholds=[3.,4.,5.,7.5,10.];components=[];sweeps=[];requests={}
    def candidate(row,reason):requests.setdefault(int(row),[]).append(reason)
    for metric,col in [('experiment','rmse_experiment_ohm'),('robust_experiment','robust_experiment_ohm'),
                       ('two_layer','rmse_2L_FEM_ohm'),('robust_joint','robust_joint_ohm')]:
        if table[col].notna().any():candidate(table[col].idxmin(),'minimum_'+metric)
    candidate(np.argmin(joint),'minimum_joint')
    for T in thresholds:
        good=exp<=T;rob=table.robust_experiment_ohm.to_numpy()<=T
        sweeps.append(dict(experiment_threshold_ohm=T,good_poses=int(good.sum()),good_centres=int(np.unique(ci[good]).size),
            robust_poses=int(rob.sum()),robust_centres=int(np.unique(ci[rob]).size)))
        for metric in ['experiment','robust_experiment','joint','robust_joint']:
            value=maps[metric];mask=np.isfinite(value)&(value<=T)
            labels,num=ndimage.label(mask.reshape(len(d['v']),len(d['u']),a))
            counts=np.bincount(labels.ravel());order=np.argsort(-counts[1:])+1
            for label in order:
                if counts[label]==0:continue
                flat=np.flatnonzero(labels.ravel()==label);rows=lookup[flat];assert np.all(rows>=0)
                rep=rows[np.argmin(value.ravel()[flat])];tc=table.iloc[rep]
                coords=c[np.unique(ci[rows]),3:6]
                components.append(dict(metric=metric,threshold_ohm=T,label=int(label),poses=int(len(rows)),
                    centres=int(len(coords)),u_min_mm=float(table.u_mm.iloc[rows].min()),u_max_mm=float(table.u_mm.iloc[rows].max()),
                    v_min_mm=float(table.v_mm.iloc[rows].min()),v_max_mm=float(table.v_mm.iloc[rows].max()),
                    phi_min_deg=float(table.phi_deg.iloc[rows].min()),phi_max_deg=float(table.phi_deg.iloc[rows].max()),
                    span_x_mm=float(np.ptp(coords[:,0])),span_y_mm=float(np.ptp(coords[:,1])),span_z_mm=float(np.ptp(coords[:,2])),
                    representative_flat_index=int(d['flat'][rep]),minimum_metric_ohm=float(value.ravel()[d['flat'][rep]])))
                if T==5. and label in order[:5] and metric in ['experiment','robust_experiment','robust_joint']:
                    candidate(rep,f'{metric}_component_{label}_threshold_5')
    pd.DataFrame(sweeps).to_csv(out/'experimental_thresholds.csv',index=False)
    pd.DataFrame(components).to_csv(out/'regions.csv',index=False)
    # Nondominated tradeoff: keep strictly decreasing 2L error in experiment-error order.
    order=np.lexsort((d['flat'],two,exp));best=np.minimum.accumulate(two[order]);keep=np.r_[True,best[1:]<best[:-1]]
    pareto=order[keep];table.iloc[pareto].to_csv(out/'pareto_poses.csv',index=False)
    for i in np.linspace(0,len(pareto)-1,min(4,len(pareto))).astype(int):candidate(pareto[i],'pareto_tradeoff')
    req=table.iloc[list(requests)].copy();req.insert(0,'candidate_id',[f'C{i+1:02d}' for i in range(len(req))]);req['reasons']=[';'.join(requests[i]) for i in requests]
    req[['x_mm','y_mm','z_mm']]=c[req.centre_index.to_numpy(int)-1,3:6]
    req.to_csv(out/'candidate_requests.csv',index=False)
    np.savez_compressed(out/'analysis_maps.npz',**maps)
    joint_sweep=[]
    for te in thresholds:
        for tm in thresholds:
            good=(exp<=te)&(two<=tm);joint_sweep.append(dict(experiment_threshold_ohm=te,model_threshold_ohm=tm,
                poses=int(good.sum()),centres=int(np.unique(ci[good]).size)))
    pd.DataFrame(joint_sweep).to_csv(out/'joint_thresholds.csv',index=False)
    summary=dict(valid_poses=n,processed_centres=int(np.sum(np.any(d['status']>0,axis=1))),
        total_centres=len(c),total_poses=d['status'].size,candidates=len(req),pareto_poses=len(pareto),
        robust_neighbourhood=dict(u_grid_steps=2,v_grid_steps=2,phi_steps=2,poses=125,
            meaning='grid-parameter scenario, not a measured physical placement uncertainty',
            complete_observation_required=True,all_125_poses_must_be_valid=True),
        thresholds_ohm=thresholds,threshold_status='exploratory scenario thresholds, not calibrated confidence levels',
        minimum_experiment_rmse_ohm=float(exp.min()),minimum_2L_FEM_rmse_ohm=float(two.min()),
        minimum_joint_max_rmse_ohm=float(joint.min()),
        minimum_robust_experiment_rmse_ohm=float(np.nanmin(maps['robust_experiment'])),
        minimum_robust_joint_max_rmse_ohm=float(np.nanmin(maps['robust_joint'])),
        no_new_FEM=True,source_scan_still_stopped=True,physical_localization_validated=False)
    write(out/'analysis_summary.json',summary);write(out/'analysis_progress.json',dict(stage='analysis_complete_geometry_export_pending',complete=False,curves_done=n,curves_total=n))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fit','regions','all']);p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--out',type=Path,default=DEFAULT_OUT);o=p.parse_args()
    if o.stage in ['all','prepare']:prepare(o.run,o.out)
    if o.stage in ['all','fit']:fit(o.out)
    if o.stage in ['all','regions']:regions(o.out)

if __name__=='__main__':main()
