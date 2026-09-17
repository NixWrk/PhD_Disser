"""Evaluate M1--M9 via the common numeric_metrics and compare 8192/4096 fits.

Read-only inputs; no MATLAB/FEM launch and no edits to the common runner/report.
The automatic target-phase mask is a computational comparator, not manual truth.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'Colab Notebooks/cardiac_4d_segmentation'
sys.path.insert(0,str(COMMON))
from run_heart_method_comparison import get_mask, moments, numeric_metrics
from heart_reduced_geometry import _ellipsoid
from run_reduced_heart_geometry import digest, serial, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parameters(row):
    if not row['valid']:
        return None
    centre=np.asarray(row['centre_mm'],float)
    axes=np.asarray(row['axes_mm'],float)
    rotation=np.asarray(row['orientation_columns'],float)
    assert centre.shape==(3,) and axes.shape==(3,) and rotation.shape==(3,3)
    assert np.isfinite(centre).all() and np.isfinite(axes).all() and (axes>0).all()
    assert np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-8)
    S=rotation@np.diag(axes**2)@rotation.T
    assert np.allclose(S,row['shape_tensor_mm2'],rtol=1e-10,atol=1e-8)
    ambiguous=any(abs(axes[i]-axes[j])<1e-6*axes.max() for i,j in itertools.combinations(range(3),2))
    candidate=_ellipsoid('library_'+row['method_id'],centre,axes,rotation,ambiguous,
        library_method_id=row['method_id'],library_method_name=row['method_name'],
        preserves_volume=False,sampling_family='ellipsoid')
    assert np.isclose(candidate['volume_mm3'],row['volume_mm3'],rtol=1e-10)
    if row['method_id'] in ('M7','M9'):
        candidate['radius_mm']=float(row['radius_mm'])
    return candidate


def load_fits(folder, clouds_hash, count):
    folder=Path(folder); inp=read(folder/'input.json'); out=read(folder/'results.json')
    assert inp['source']['sha256']==clouds_hash
    assert out['input_sha256']==digest(folder/'input.json')
    expected={(c['id'],m) for c in inp['cases'] for m in inp['methods']}
    actual=[(r['case_id'],r['method_id']) for r in out['fits']]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    for r in out['fits']:
        assert r['sampling']['fit_count']==count
        assert r['sampling']['policy']=='deterministic_weight_resample'
    return {(r['case_id'],r['method_id']):r for r in out['fits']}


def prepare_full_surface(args):
    from scipy.io import savemat, loadmat
    from heart_construction_variants import surface_quadrature
    import heart_library_fits as library
    target=Path(args.output)
    if target.exists(): raise FileExistsError('Use a new full-surface output directory')
    clouds=read(args.clouds)
    assert len(clouds['cases'])==3
    common_dir=Path(args.clouds).parent
    assert read(common_dir/'status.json')['status']=='completed'
    protocol=read(common_dir/'protocol.json')
    baseline=Path(args.baseline)/'geometry_comparison.jsonl'
    assert digest(baseline)==protocol['baseline_sha256']
    current_code={name:digest(COMMON/name) for name in protocol['code_sha256']}
    baseline_rows=[json.loads(x) for x in baseline.read_text(encoding='utf-8').splitlines()]
    derived=Path(read(args.config)['derived_root'])
    cases=[]; source_checks=[]
    for cloud in clouds['cases']:
        cid=cloud['id']
        selected=[r for r in baseline_rows if f"{r['subject']}_{r['phase_id']}_{r['region']}"==cid
                  and r['provenance']['sha256']==cloud['source_metadata']['mask_sha256']]
        assert len(selected)==1
        row=selected[0]; mask,affine=get_mask(derived,row)
        points,weights,surface=surface_quadrature(mask,affine)
        meta={'mask_sha256':row['provenance']['sha256'],'surface':surface,
              'region':row['region'],'subject':row['subject'],'phase_id':row['phase_id'],
              'reference_phase_id':cloud['source_metadata']['reference_phase_id'],
              'origin':'fresh_full_surface_quadrature_from_original_mask',
              'full_point_count':len(points),
              'full_points_sha256_le_f64':library.array_sha(points),
              'full_weights_sha256_le_f64':library.array_sha(weights),
              'target_touches_CT_boundary':row['provenance']['touches_source_image_boundary']}
        cases.append({'id':cid,'points_mm':points,'weights':weights,'source_metadata':meta})
        source_checks.append(dict(case_id=cid,**meta))
        print(cid,'fresh full surface',len(points),'points',flush=True)
    target.mkdir(parents=True)
    source=target/'full_surface_source.mat'
    savemat(source,{'units':'mm','cases':cases},do_compression=True)
    write_json(target/'full_surface_provenance.json',serial({
        'schema':'heart_library_full_surface_source_v1','cases':source_checks,
        'baseline_sha256':digest(baseline),'selection_clouds_sha256':digest(args.clouds),
        'selection_cloud_coordinates_used_for_fit':False,
        'full_surface_mat_sha256':digest(source),'common_protocol':protocol,
        'current_code_sha256':current_code,
        'adapter_sha256':digest(__file__)}))
    library.prepare(argparse.Namespace(input=str(source),synthetic=False,
        output=str(target/'fits_32768'),weight_policy='deterministic_weight_resample',
        sample_count=32768,regularization=0.,eta=1.,shared_mat=True))
    prepared=read(target/'fits_32768/input.json')
    shared=loadmat(target/'fits_32768/shared_points.mat')
    evaluation=[]
    for case in prepared['cases']:
        assert case['sampling']['fit_count']==32768
        assert case['sampling']['input_count']>32768
        evaluation.append({'id':case['id'],
            'points_mm':shared[case['points_mat_variable']].tolist(),
            'weights':np.full(32768,1/32768).tolist(),
            'source_metadata':case['source_metadata']})
    write_json(target/'evaluation_clouds_32768.json',{
        'units':'mm','cases':evaluation,'evaluation_origin':'same_fresh_32768_fit_sample',
        'full_quadrature_evaluation':False})


def run(args):
    out=Path(args.output)
    if out.exists(): raise FileExistsError('Use a fresh comparison output')
    clouds_path=Path(args.clouds); clouds=read(clouds_path)
    assert len(clouds['cases'])==3 and clouds['units']=='mm'
    common_dir=Path(args.common_dir) if args.common_dir else clouds_path.parent
    assert read(common_dir/'status.json')['status']=='completed'
    baseline=Path(args.baseline)/'geometry_comparison.jsonl'
    rows=[json.loads(line) for line in baseline.read_text(encoding='utf-8').splitlines()]
    protocol=read(common_dir/'protocol.json')
    assert digest(baseline)==protocol['baseline_sha256']
    # A fresh full-surface run freezes its own code version; evaluate BOTH
    # previous and refined fits with that same current version.
    active_code=protocol['code_sha256']
    if args.high_source:
        active_code=read(Path(args.high_source).with_name('full_surface_provenance.json'))['current_code_sha256']
    for name,sha in active_code.items():
        assert digest(COMMON/name)==sha,name
    chash=digest(clouds_path)
    high_n,low_n=args.high_count,args.low_count
    variants={str(high_n):load_fits(args.high_dir,digest(args.high_source) if args.high_source else chash,high_n),
              str(low_n):load_fits(args.low_dir,digest(args.low_source) if args.low_source else chash,low_n)}
    derived=Path(read(args.config)['derived_root'])
    out.mkdir(parents=True)
    records=[]
    for cloud in clouds['cases']:
        cid=cloud['id']
        match=[r for r in rows if f"{r['subject']}_{r['phase_id']}_{r['region']}"==cid
               and r['provenance']['sha256']==cloud['source_metadata']['mask_sha256']]
        assert len(match)==1, ('Ambiguous target',cid)
        row=match[0]; mask,affine=get_mask(derived,row); m=moments(row)
        points=np.asarray(cloud['points_mm'],float); weights=np.asarray(cloud['weights'],float)
        assert len(points)>=10 and len(weights)==len(points) and np.isfinite(weights).all() and (weights>=0).all()
        assert np.isclose(weights.sum(),1,rtol=0,atol=1e-10)
        weights=weights/weights.sum()
        for count,fitrows in variants.items():
            for mid in [f'M{i}' for i in range(1,10)]:
                fit=fitrows[cid,mid]; start=time.monotonic()
                record={'case_id':cid,'subject':row['subject'],'phase_id':row['phase_id'],
                        'region':row['region'],'method_id':mid,'method_name':fit['method_name'],
                        'fit_sample_count':int(count),'evaluation_sample_count':len(points),
                        'target_mask_sha256':row['provenance']['sha256'],
                        'target_touches_CT_boundary':row['provenance']['touches_source_image_boundary'],
                        'library_valid':fit['valid'],'library_status':fit['status'],
                        'library_error':fit['error_message'],'library_warning':fit['warning_message'],
                        'source_method_family':'MATLAB_library_registry_M',
                        'automatic_target_phase_mask_is_comparator':True,
                        'independent_anatomical_reference':False,'metrics':None,'parameters':None}
                if not fit['valid']:
                    record['status']='library_fit_failed'
                else:
                    try:
                        p=parameters(fit)
                        record['parameters']=serial(p)
                        record['metrics']=numeric_metrics(mask,affine,m,p,points,weights)
                        record['status']='computed'
                    except (AssertionError,ValueError,RuntimeError,FloatingPointError,MemoryError) as exc:
                        record['status']='geometry_evaluation_failed'
                        record['geometry_error']=str(exc)
                record['elapsed_seconds']=time.monotonic()-start
                records.append(record)
                write_json(out/f'{cid}__{mid}__{count}.json',serial(record))
                print(cid,mid,count,record['status'],round(record['elapsed_seconds'],2),flush=True)
    lookup={(r['case_id'],r['method_id'],r['fit_sample_count']):r for r in records}
    differences=[]; flips=[]
    for cid in [c['id'] for c in clouds['cases']]:
        for mid in [f'M{i}' for i in range(1,10)]:
            hi,lo=lookup[cid,mid,high_n],lookup[cid,mid,low_n]
            d={'case_id':cid,'method_id':mid,f'status_{high_n}':hi['status'],f'status_{low_n}':lo['status'],
               'direction':f'{high_n}_minus_{low_n}','comparable':hi['metrics'] is not None and lo['metrics'] is not None}
            if d['comparable']:
                hp,lp=hi['parameters'],lo['parameters']; hm,lm=hi['metrics'],lo['metrics']
                hS=variants[str(high_n)][cid,mid]['shape_tensor_mm2']
                lS=variants[str(low_n)][cid,mid]['shape_tensor_mm2']
                d.update(centre_displacement_mm=float(np.linalg.norm(np.array(hp['center_mm'])-lp['center_mm'])),
                    volume_relative_change_percent=100*(hp['volume_mm3']/lp['volume_mm3']-1),
                    max_semiaxis_relative_change_percent=float(100*np.max(np.abs(np.array(hp['semi_axes_mm'])/lp['semi_axes_mm']-1))),
                    shape_tensor_relative_difference=float(np.linalg.norm(np.array(hS)-lS)/np.linalg.norm(hS)),
                    dice_change=hm['dice']-lm['dice'],
                    volume_error_change_percentage_points=hm['volume_error_percent']-lm['volume_error_percent'],
                    surface_RMS_change_mm=hm['target_surface_to_candidate_RMS_mm']-lm['target_surface_to_candidate_RMS_mm'])
            differences.append(d)
        for m1,m2 in itertools.combinations([f'M{i}' for i in range(1,10)],2):
            four=[lookup[cid,mid,n] for n in (high_n,low_n) for mid in (m1,m2)]
            if any(r['metrics'] is None for r in four): continue
            for key in ('dice','target_surface_to_candidate_RMS_mm','volume_error_percent'):
                values=[r['metrics'][key] for r in four]
                if key=='volume_error_percent': values=list(map(abs,values))
                high=values[0]-values[1]; low=values[2]-values[3]
                if high*low<0 and min(abs(high),abs(low))>1e-10:
                    flips.append({'case_id':cid,'method_1':m1,'method_2':m2,'metric':key,
                                  f'difference_{high_n}':high,f'difference_{low_n}':low})
    result={'schema':'heart_library_geometry_comparison_v1','status':'completed',
            'clouds_sha256':chash,'baseline_sha256':digest(baseline),
            'fit_results_sha256':{str(n):digest(Path(folder)/'results.json') for n,folder in
                [(high_n,args.high_dir),(low_n,args.low_dir)]},
            'code_sha256':dict(active_code,**{Path(__file__).name:digest(__file__)}),
            'pilot_code_sha256':protocol['code_sha256'],
            'rows':records,'sampling_differences':differences,'pairwise_order_reversals':flips,
            'fit_counts':[high_n,low_n],
            'evaluation_point_counts':[len(c['points_mm']) for c in clouds['cases']],
            'limitation':'Fit sample origins are recorded separately. Both fits evaluated on identical evaluation points and common voxel-centre mask grid. Evaluation is sampled, not integration over all surface-quadrature points. Not CT resolution, FEM or physiological convergence.',
            'acceptance_thresholds_predeclared':False,'numerical_sufficiency_certified':False,
            'FEM_executed':False}
    write_json(out/'results.json',serial(result))
    print(json.dumps({'rows':len(records),'failed_rows':sum(r['metrics'] is None for r in records),
                      'pairwise_order_reversals':len(flips)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('config','baseline','clouds','high-dir','low-dir','output'):
        p.add_argument('--'+arg,required=arg not in ('high-dir','low-dir'))
    p.add_argument('--mode',choices=['compare','preparefullsurf'],default='compare')
    p.add_argument('--common-dir')
    p.add_argument('--high-source');p.add_argument('--low-source')
    p.add_argument('--high-count',type=int,default=8192);p.add_argument('--low-count',type=int,default=4096)
    args=p.parse_args()
    if args.mode=='preparefullsurf':prepare_full_surface(args)
    else:
        if not args.high_dir or not args.low_dir:p.error('Comparison requires both fit directories')
        run(args)
