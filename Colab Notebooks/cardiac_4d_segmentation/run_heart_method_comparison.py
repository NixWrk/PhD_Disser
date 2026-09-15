"""Compare registered methods on identical masks, with resumable per-fit evidence.

No anatomy is moved between subjects or registered to the static FEM here.
The automatic target phase is the computational comparator (T0); I1--I3
transform the individual reference mask. Region definitions are orthogonal.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import json
import time
from pathlib import Path
import nibabel as nib
import numpy as np
from heart_reduced_geometry import _ellipsoid, sampled_dice, individual_isotropic_candidate
from heart_affine_geometry import individual_rigid_isotropic_candidate, individual_affine_moment_candidate
from heart_construction_variants import surface_quadrature, fit_surface_spheres
from heart_ct_overlays import select_examples
from run_reduced_heart_geometry import digest, load_crop, mask_path, serial, write_json
from heart_additional_sphere_methods import (fit_surface_sphere_fixed_volume,
    fit_algebraic_sphere, fit_iou_sphere_fixed_volume, fit_iou_sphere_free_volume,
    area_systematic_sample)
from heart_additional_ellipsoid_methods import (fit_surface_ellipsoid,
    fit_overlap_ellipsoid, ellipsoid_surface_distance)

METHODS = ['T0','I1','I2','I3','G1','G2','G3a','G3b','G3c','G4a','G4b','G5',
           'L1','L2','L3a','L3b','L3c','L3d','L4a','L4b']

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def get_mask(root, row):
    manifest=read_json(root/row['subject']/'phases/conversion_manifest.json')
    info=next(x for x in manifest['phases'] if x['phase_id']==row['phase_id'])
    ctpath=root/row['subject']/'phases'/info['nifti']
    if digest(ctpath)!=row['ct_sha256']:raise ValueError('CT changed since baseline')
    mask,affine,provenance=load_crop(mask_path(root,row['subject'],row['phase_id'],row['region']),info,nib.load(ctpath))
    if provenance['sha256']!=row['provenance']['sha256']:raise ValueError('Mask changed since baseline')
    return mask,affine

def moments(row):
    return {k:np.asarray(v) if isinstance(v,list) else v for k,v in row['moments'].items()}

def numeric_metrics(mask, affine, m, candidate, eval_points, eval_weights):
    candidate=dict(candidate)
    if 'semi_axes_mm' in candidate:candidate['sampling_family']='ellipsoid'
    overlap=sampled_dice(mask,affine,candidate,target_moments=m)
    metrics={'dice':overlap['dice_voxel_center_approx'],
        'volume_error_percent':100*(candidate['volume_mm3']/m['volume_mm3']-1),
        'center_shift_mm':float(np.linalg.norm(np.asarray(candidate['center_mm'])-m['centroid_mm'])),
        'overlap':overlap}
    if 'semi_axes_mm' in candidate:
        ep=np.asarray(eval_points,dtype=float);ew=np.asarray(eval_weights,dtype=float)
        if ep.ndim!=2 or ep.shape[1]!=3 or ew.shape!=(len(ep),) or not np.isfinite(ew).all() or np.any(ew<0) or ew.sum()<=0:
            raise ValueError('Invalid surface quadrature')
        ew=ew/ew.sum();squared=mean=0.
        for offset in range(0,len(ep),65536):
            d=ellipsoid_surface_distance(ep[offset:offset+65536],candidate['center_mm'],candidate['semi_axes_mm'],candidate['axes_matrix'])
            wc=ew[offset:offset+65536];squared+=float(wc@(d*d));mean+=float(wc@d)
        metrics['target_surface_to_candidate_RMS_mm']=float(np.sqrt(squared))
        metrics['target_surface_to_candidate_mean_mm']=mean
        metrics['surface_quadrature_point_count']=len(ep)
    else:
        metrics['target_surface_to_candidate_RMS_mm']=None
        metrics['surface_metric_limitation']='individual_surface_not_evaluated_by_analytic_ellipsoid_distance'
    return metrics

def candidates_for(method, row, refrow, mask, affine, refmask, refaffine, m, rm,
                   points, weights, surface, a):
    center=m['centroid_mm'];volume=m['volume_mm3'];r=float(np.cbrt(3*volume/(4*np.pi)))
    initial=dict(row['comparisons']['ellipsoid_volume_scaled']['parameters'])
    if method=='G1':return row['comparisons']['sphere']['parameters']
    if method=='L1':return initial
    if method=='L2':return row['raw_moment_ellipsoid']
    if method=='G2':
        radius=float(np.sqrt(5*np.trace(m['covariance_mm2'])/3))
        return _ellipsoid('sphere_second_moment',center,np.full(3,radius),np.eye(3),True,radius_mm=radius,preserves_volume=False)
    if method in ('I1','I2','I3'):
        function={'I1':individual_isotropic_candidate,'I2':individual_rigid_isotropic_candidate,'I3':individual_affine_moment_candidate}[method]
        return function(refmask,refaffine,mask,affine,reference_moments=rm,target_moments=m)
    p,w,sampling=area_systematic_sample(points,weights,a.fit_points)
    if method in ('G3a','G3b'):
        pair=fit_surface_spheres(p,w,center,surface['bounds_mm'])
        return pair[method=='G3b']
    bounds=np.asarray(surface['bounds_mm'])
    starts=np.stack([center,w@p,bounds.mean(axis=0)])
    if method=='G3c':return fit_surface_sphere_fixed_volume(p,w,volume,bounds,starts,max_nfev=a.surface_budget)
    if method=='G5':return fit_algebraic_sphere(p,w)
    if method in ('G4a','G4b'):
        kw=dict(max_evaluations=a.iou_budget,evaluation_spacing_mm=a.iou_step,
                verification_spacing_mm=None,check_subdivisions=1)
        if method=='G4a':return fit_iou_sphere_fixed_volume(mask,affine,volume,bounds,starts,[4.,2.,1.,.5],**kw)
        return fit_iou_sphere_free_volume(mask,affine,bounds,[.5*r,1.7*r],
            np.column_stack((starts,np.full(3,r))),[4.,2.,1.,.5],**kw)
    if method.startswith('L3'):
        return fit_surface_ellipsoid(points,weights,initial,method=method,target_volume_mm3=volume,
            max_fit_points=a.fit_points,max_nfev=a.surface_budget,n_starts=3)
    if method.startswith('L4'):
        return fit_overlap_ellipsoid(mask,affine,initial,method=method,target_volume_mm3=volume,
            max_nfev=a.iou_budget,n_starts=3,grid_step_mm=a.iou_step,max_grid_points=4_000_000,
            center_limit_mm=.5*r,axis_bounds_mm=(.35*r,1.8*r))
    raise ValueError(method)

def run(a):
    config=read_json(a.config);root=Path(config['derived_root']);out=Path(a.output)
    baseline=Path(a.baseline)/'geometry_comparison.jsonl'
    rows=[json.loads(x) for x in baseline.read_text(encoding='utf-8').splitlines()]
    selected=select_examples(rows) if a.selection=='examples' else [(r,next(x for x in rows if
        (x['subject'],x['cycle_index'],x['region'],x['is_reference_phase'])==
        (r['subject'],r['cycle_index'],r['region'],True))) for r in rows if r['region']==a.region]
    requested=a.methods.split(',') if a.methods else METHODS
    if not set(requested)<=set(METHODS):raise ValueError('Unregistered method')
    settings={k:v for k,v in vars(a).items() if k not in ('config','baseline','output')}
    code_names=['run_heart_method_comparison.py','heart_reduced_geometry.py','heart_construction_variants.py',
                'heart_additional_sphere_methods.py','heart_additional_ellipsoid_methods.py','heart_affine_geometry.py']
    fingerprint={'baseline_sha256':digest(baseline),'settings':settings,
                 'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in code_names},
                 'budget_semantics':{'G3c':'surface_budget max_nfev per start','L3':'surface_budget total objective evaluations'},
                 'surface_metric':'full_target_surface_area_quadrature'}
    if out.exists() and any(out.iterdir()) and not (out/'protocol.json').exists():
        raise ValueError('Existing output has no matching protocol')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and read_json(out/'protocol.json')!=fingerprint:
        raise ValueError('Resume refused: protocol or code changed; use a new output directory')
    write_json(out/'protocol.json',fingerprint)
    clouds=[]; completed=[]
    write_json(out/'status.json',{'status':'running','requested':len(selected)*len(requested),'completed':0})
    for row,refrow in selected:
        cid=f"{row['subject']}_{row['phase_id']}_{row['region']}"
        mask,affine=get_mask(root,row);refmask,refaffine=get_mask(root,refrow)
        m,rm=moments(row),moments(refrow)
        points,weights,surface=surface_quadrature(mask,affine)
        ep,ew,es=area_systematic_sample(points,weights,a.eval_points)
        clouds.append({'id':cid,'points_mm':ep.tolist(),'weights':ew.tolist(),
          'source_metadata':{'mask_sha256':row['provenance']['sha256'],'surface':surface,'sampling':es,
                             'region':row['region'],'reference_phase_id':refrow['phase_id']}})
        common={'case_id':cid,'subject':row['subject'],'phase_id':row['phase_id'],'region':row['region'],
            'cycle_index':row['cycle_index'],'reference_phase_id':refrow['phase_id'],
            'is_reference_phase':row['is_reference_phase'],'target_mask_sha256':row['provenance']['sha256'],
            'target_touches_CT_boundary':row['provenance']['touches_source_image_boundary'],
            'target_volume_ml':m['volume_ml'],'surface':surface,'evaluation_sampling':es,
            'anatomical_reference_independent':False}
        for method in requested:
            dest=out/f'{cid}__{method}.json'
            if dest.exists():
                record=read_json(dest)
                for key,expected in dict(common,method_id=method).items():
                    if record.get(key)!=serial(expected):raise ValueError('Resume record does not match case/method/provenance: '+key)
            else:
                start=time.monotonic();record=dict(common,method_id=method)
                try:
                    if method=='T0':
                        record.update(status='reference',parameters=None,metrics={'dice':1.,'volume_error_percent':0.,'center_shift_mm':0.,'target_surface_to_candidate_RMS_mm':0.},
                            interpretation='automatic_target_mask_self_comparison_by_definition')
                    else:
                        candidate=candidates_for(method,row,refrow,mask,affine,refmask,refaffine,m,rm,points,weights,surface,a)
                        detail=candidate if 'parameters' in candidate else {}
                        params=detail.get('parameters') if detail else candidate
                        record.update(status=detail.get('status',candidate.get('status','computed')),fit=detail)
                        if params is not None:
                            record['parameters']=serial(params)
                            record['metrics']=numeric_metrics(mask,affine,m,params,points,weights)
                            record['surface_metric_rule']='full_target_surface_area_quadrature_not_fit_subset'
                        else:record['metrics']=None
                except (ValueError,RuntimeError,FloatingPointError,MemoryError) as exc:
                    record.update(status='failed',reason=str(exc),metrics=None)
                record['elapsed_seconds']=time.monotonic()-start
                write_json(dest,serial(record))
            completed.append(record)
            print(cid,method,record['status'],round(record.get('elapsed_seconds',0),2),flush=True)
            write_json(out/'status.json',{'status':'running','requested':len(selected)*len(requested),'completed':len(completed),
                'last_case':cid,'last_method':method})
        write_json(out/'library_clouds.json',{'units':'mm','cases':clouds})
    write_json(out/'results.json',{'schema':'heart_method_comparison_v1','status':'exploratory_hypothesis_not_validated',
        'scope':a.selection,'protocol':fingerprint,'rows':completed,'FEM_executed_by_this_runner':False,
        'T0_is_automatic_target_phase_mask':True,'I_is_transformed_individual_reference_shape':True})
    write_json(out/'status.json',{'status':'completed','requested':len(selected)*len(requested),'completed':len(completed),
        'failures':sum(r.get('metrics') is None for r in completed)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p.add_argument('--selection',choices=['examples','all-phases'],default='examples');p.add_argument('--region',default='whole_heart')
    p.add_argument('--methods');p.add_argument('--fit-points',type=int,default=1024);p.add_argument('--eval-points',type=int,default=8192)
    p.add_argument('--surface-budget',type=int,default=1800);p.add_argument('--iou-budget',type=int,default=900)
    p.add_argument('--iou-step',type=float,default=3.)
    run(p.parse_args())
