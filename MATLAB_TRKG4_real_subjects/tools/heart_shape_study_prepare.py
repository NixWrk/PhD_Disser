"""Prepare matched STL/G1/L1 electric probes, reusing verified existing solves.

No solver is introduced: material quadrature and run_pipeline retain ownership
of their stages. All subject paths and new results are supplied in local JSON.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import trimesh

from heart_stl_materials import bounds, scaled


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


SHAPES=('STL','G1','L1')
POSES=('xm5','xp5','ym5','yp5','zm5','zp5')


def validate_reused(record, variant_row, common_cfg):
    """A current hash alone cannot prove which physical state a MAT contains."""
    d=loadmat(record['path'],simplify_cells=True)
    saved_cfg=read(record['config'])
    identity=d['identity']
    checks={'config_sha256':sha(record['config']),
            'plan_sha256':sha(saved_cfg['material_plan'])}
    for key,field in [('eit_model_sha256','eit_model_file'),('ttrkg_model_sha256','ttrkg_model_file')]:
        if saved_cfg[field]!=common_cfg[field]:raise ValueError('Reused model path differs: '+field)
        checks[key]=sha(common_cfg[field])
    for key in ('frequency_Hz','current_A','montage_basis','breath_interval_status'):
        if saved_cfg[key]!=common_cfg[key]:raise ValueError('Reused protocol differs: '+key)
    implementation='heart_stl_electrical_check.m' if record['format']=='electrical' else 'heart_stl_effective_fit.m'
    checks['implementation_sha256']=sha(Path(__file__).with_name(implementation))
    for key,value in checks.items():
        if identity.get(key)!=value:raise ValueError('Reused identity differs: '+key)
    rho=np.asarray(record['rho_ohm_m'],dtype=float)
    sigma=np.asarray(d['sigma'],dtype=float).reshape(-1)
    if sigma.shape!=(5,) or not np.allclose(sigma,1/rho,rtol=1e-12,atol=0):
        raise ValueError('Reused sigma/rho scenario differs')
    if not np.allclose(np.asarray(d['row']['rho_ohm_m']).reshape(-1),rho,rtol=1e-12,atol=0):
        raise ValueError('Reused row rho differs')
    row=d['row'];measure=d
    variant_key='variant' if record['format']=='electrical' else 'base_variant'
    if record['format']=='effective_fit_probe':
        row=np.asarray(d['probes'],dtype=object).reshape(-1)[record['probe_index']]
        measure=row;variant_key='variant'
    if row[variant_key]!=record['variant'] or row['material_sha256']!=variant_row['material_sha256']:
        raise ValueError('Reused variant/material differs')
    if not np.allclose(row['volume_ml'],variant_row['volume_ml'],rtol=1e-12,atol=1e-10):
        raise ValueError('Reused integrated volumes differ')
    if np.asarray(measure['V_eit']).size!=208 or np.asarray(measure['S_eit']).shape!=(208,5):
        raise ValueError('Reused EIT data unavailable')


def geometry(b):
    cfg=read(b/'stl_materials_v1.local.json')
    nominal=read(cfg['candidates_file'])
    base_plan=read(b/'stl_materials_v1/materials_q64_plan.local.json')
    for key in ('mesh_path','stl_path','baseline_materials_file','baseline_preparation','candidates_file'):
        expected=base_plan['sources'].get(cfg[key])
        if expected is None or sha(cfg[key])!=expected:
            raise ValueError('Changed source '+key)
    for name in ('fat','bone','lung'):
        path=cfg['mask_paths'][name]
        if sha(path)!=base_plan['sources'][path]:
            raise ValueError('Changed mask '+name)
    volume=float(nominal['STL']['volume_mm3'])
    for shape in SHAPES:
        if not np.isclose(nominal[shape]['volume_mm3'],volume,rtol=1e-10):
            raise ValueError('Selected shapes do not preserve the same nominal volume')
    source_bounds=trimesh.load_mesh(cfg['stl_path'],process=False).bounds
    boxes=[]
    for row in cfg['variants']:
        f=1+(row['factor']-1)*volume/float(nominal[row['method']]['volume_mm3'])
        boxes.append(bounds(scaled(nominal[row['method']],f),source_bounds))
    boxes=np.asarray(boxes); low=boxes[:,0].min(0); high=boxes[:,1].max(0)
    new_variants=[]
    # L1 translated maps already exist with this same integration envelope.
    for shape in ('STL','G1'):
        for pose in POSES:
            item=copy.deepcopy(nominal[shape])
            shift=np.zeros(3);shift['xyz'.index(pose[0])]=5*(-1 if pose[1]=='m' else 1)
            item['center_mm']=(np.asarray(item['center_mm'])+shift).tolist()
            item['method_id']=shape+'_'+pose
            if item['sampling_family']=='stl':
                matrix=np.eye(4);matrix[:3,:3]*=item['scale']
                matrix[:3,3]=np.asarray(item['center_mm'])-item['scale']*np.asarray(item['reference_centroid_mm'])
                item['transform_world']=matrix.tolist()
            box=bounds(item,source_bounds)
            nominal[item['method_id']]=item
            new_variants.append(dict(id=item['method_id']+'_v100_q64',method=item['method_id'],factor=1,q=64))
    # These three redundant material preparations test exact common-background reuse.
    new_variants += [dict(id=s+'_v100_q64',method=s,factor=1,q=64) for s in SHAPES]
    folder=b/'shape_materials_v1'
    if folder.exists():raise FileExistsError(folder)
    candidates=b/'shape_candidates_v1.local.json'
    material_cfg=b/'shape_materials_v1.local.json'
    cfg['selection_variants']=copy.deepcopy(cfg['variants'])+copy.deepcopy(new_variants)
    # Expand only the heart-test envelope. The background quadrature remains
    # exactly where it was, so translated shapes cannot silently change lung.
    cfg['background_quadrature_box_mm']=[low.tolist(),high.tolist()]
    cfg['variants']=new_variants
    cfg['output_dir']=str(folder)
    cfg['candidates_file']=str(candidates)
    write(candidates,nominal)
    write(material_cfg,cfg)
    print(json.dumps(dict(material_config=str(material_cfg),new_motion_maps=12,baseline_controls=3)))


def finalize(b):
    cfg=read(b/'stl_electrical_q64_v1.local.json')
    plans=[b/'stl_materials_v1/materials_q64_plan.local.json',
           b/'stl_pose_materials_q64_v1/materials_plan.local.json',
           b/'shape_materials_v1/materials_plan.local.json']
    loaded=[read(p) for p in plans]
    base=loaded[0]
    variants={}
    for path,plan in zip(plans,loaded):
        for key in ('mesh_sha256','stl_sha256','seed','quadrature_rule'):
            if plan[key]!=base[key]:raise ValueError('Different material setup: '+key)
        for row in plan['variants']:
            item=copy.deepcopy(row)
            material=(path.parent/item['material_file']).resolve()
            if sha(material)!=item['material_sha256']:raise ValueError('Changed material '+item['id'])
            item['material_file']=os.path.relpath(material,b)
            if item['id'] in variants:
                old=b/variants[item['id']]['material_file']
                if not np.array_equal(loadmat(old)['fractions'],loadmat(material)['fractions']):
                    raise ValueError('Common-background control failed: '+item['id'])
                continue
            variants[item['id']]=item
    fit_file=b/'stl_effective_fit_q64_v1/effective_fit.local.json'
    fit=read(fit_file)
    fit_row=next(r for r in fit['rows'] if r['id']=='STL')
    if not fit_row['converged']:raise ValueError('STL conditional fit incomplete')
    for key,file_key in [('eit_model_sha256','eit_model_file'),('ttrkg_model_sha256','ttrkg_model_file'),('reference_sha256','reference_prediction_file')]:
        if fit['identity'][key]!=sha(cfg[file_key]):raise ValueError('Fit/model identity differs')
    scenarios={'diagnostic4':cfg['jobs'][0]['rho_ohm_m'],'commonSTLfit':fit_row['rho_ohm_m']}
    common_plan=b/'shape_electrical_materials_v1.local.json'
    output=b/'shape_electrical_v1'
    new_config=b/'shape_electrical_v1.local.json'
    manifest_path=b/'shape_electrical_study_v1.local.json'
    for p in (common_plan,output,new_config,manifest_path):
        if p.exists():raise FileExistsError(p)
    records=[];jobs=[];wanted=set()
    for scenario,rho in scenarios.items():
        for shape in SHAPES:
            states=[('base',95),('base',100),('base',105)]+[(p,100) for p in POSES]
            for pose,level in states:
                var=f'{shape}_{"" if pose=="base" else pose+"_"}v{level:03d}_q64'
                wanted.add(var)
                record=dict(shape=shape,scenario=scenario,pose=pose,level=level,variant=var,
                            rho_ohm_m=rho,format='electrical',probe_index=None,
                            computational_probe_not_measured_motion=True)
                if scenario=='diagnostic4' and shape=='STL' and pose=='base':
                    record.update(path=str(b/'stl_electrical_q64_v1'/f'heart4_{var}.mat'),
                                  config=str(b/'stl_electrical_q64_v1.local.json'),reused=True)
                elif scenario=='diagnostic4' and shape=='L1':
                    record.update(path=str(b/'motion_electrical_v2'/f'diagnostic4_{var}.mat'),
                                  config=str(b/'motion_electrical_v2.local.json'),reused=True)
                else:
                    job_id=f'{scenario}_{var}'
                    jobs.append(dict(id=job_id,variant=var,scenario=scenario,rho_ohm_m=rho,eit=True))
                    record.update(path=str(output/(job_id+'.mat')),config=str(new_config),reused=False)
                if record['reused']:
                    validate_reused(record,variants[var],cfg)
                    record['mat_sha256']=sha(record['path'])
                records.append(record)
    if len(records)!=54 or len(jobs)!=42:raise ValueError('Unexpected study size')
    merged=copy.deepcopy(base);merged['variants']=[variants[v] for v in sorted(wanted)]
    merged['source_plans_sha256']={str(p):sha(p) for p in plans}
    merged['source_selected_tetrahedra']={str(p):d['selected_tetrahedra'] for p,d in zip(plans,loaded)}
    merged['selected_tetrahedra']=max(d['selected_tetrahedra'] for d in loaded)
    merged['integration_envelope_note']='Heart probe envelope extended; original background quadrature envelope frozen; per-source counts retained'
    merged['background_quadrature_box_mm']=loaded[-1]['background_quadrature_box_mm']
    merged['shared_background_control']='STL/G1/L1 baseline fractional arrays identical to earlier maps'
    cfg['jobs']=jobs;cfg['output_dir']=str(output);cfg['material_plan']=str(common_plan)
    cfg['shape_comparison']=dict(forms=list(SHAPES),shared_material_scenarios=scenarios,
                                new_FEM_jobs=42,reused_states=12,no_phase_CT_input=True,
                                numerical_motion_step_mm=5,volume_factors=[.95,1,1.05],
                                conditional_parameters_not_tissue_truth=True)
    write(common_plan,merged);write(new_config,cfg)
    manifest=dict(schema='heart_shape_electrical_study_v1',shapes=list(SHAPES),scenarios=scenarios,
                  probes=records,new_config=str(new_config),material_plan=str(common_plan),
                  frequency_Hz=cfg['frequency_Hz'],current_A=cfg['current_A'],
                  montage_basis=cfg['montage_basis'],breath_interval_status=cfg['breath_interval_status'],
                  model_files={k:cfg[k] for k in ('eit_model_file','ttrkg_model_file','reference_prediction_file')},
                  preparation_QC=dict(common_background_arrays_identical=True,baseline_controls=3,
                                      selected_tetrahedra=base['selected_tetrahedra']),
                  sources_sha256={str(p):sha(p) for p in [*plans,new_config,common_plan,fit_file,
                       b/'shape_candidates_v1.local.json',b/'shape_materials_v1.local.json',Path(__file__)]},
                  limits=['Original STL is an anatomical geometric reference, not clinical ground truth',
                          'Original surface scaled/transformed as controlled probes, not phase CT anatomy',
                          'Common rho values are conditional scenarios, not measured synchronous tissue changes',
                          'One fixed TT montage; no clinical noise distribution or motion law assumed',
                          'EIT geometry distinguishability is diagnostic, not the operational target',
                          'Material replacement is not mechanical deformation of background organs',
                          'q64 local quadrature, no global mesh convergence claim'])
    write(manifest_path,manifest)
    print(json.dumps(dict(config=str(new_config),manifest=str(manifest_path),new_jobs=len(jobs),reused=12)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['geometry','finalize']);p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args();globals()[args.stage](args.data_root.resolve())
