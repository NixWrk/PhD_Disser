"""Exploratory Georg ventricular-mask proposals with explicit provenance and QC.

Uses the installed nnInteractive adapter; no primary masks or functional results
are overwritten. Candidate volumes are not accepted EDV/ESV or a manual reference.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from cardiac_function_analysis import verify_grid
from pilot_blood_segmentation_candidates import NNInteractiveRunner,sha256_file,_save_binary_mask

PHASES=('phase_04','phase_08','phase_10','phase_14')
CHAMBERS=('heart_ventricle_left','heart_ventricle_right')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derived-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prompt-manifest',type=Path)
    parser.add_argument('--model-dir',type=Path,default=Path(__file__).parent/'.nninteractive-models/models/nnInteractive_v1.0')
    args=parser.parse_args();source=args.derived_root/'georg';out=args.output_dir.resolve()
    if out==source.resolve() or source.resolve().is_relative_to(out):raise ValueError('Unsafe candidate output root')
    out.mkdir(parents=True,exist_ok=True)
    prompts=json.loads(args.prompt_manifest.read_text(encoding='utf-8')) if args.prompt_manifest else None
    source_manifest=source/'phases/conversion_manifest.json'
    conversion=json.loads(source_manifest.read_text(encoding='utf-8'));infos={r['phase_id']:r for r in conversion['phases']}
    plan={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'subject':'georg','phases':PHASES,'chambers':CHAMBERS,
          'status':'exploratory_hypothesis_not_validated','method':'nninteractive_ts_initial_refinement',
          'definition_version':'ventricular_cavity_envelope_pilot_20260910',
          'definition':{'papillary_muscles':'included_in_cavity_envelope','trabeculae':'included_in_cavity_envelope','LV_bounds':'mitral_and_aortic_valve_planes','RV_bounds':'tricuspid_and_pulmonary_valve_planes','valve_plane_status':'not_manually_accepted','atria':'unchanged_automatic_raw'},
          'contrast_provenance':'Author statement 2026-09-10: residual contrast at Georg examination; sufficient blood/myocardium contrast reported only for Georg',
          'selection':'Previously selected mask-volume extremes across separate short and long RR intervals; not accepted ED/ES',
          'primary_masks_overwritten':False,'independent_reference':False,'morphology_or_HU_threshold_correction':False,
          'quantitative_acceptance':'No accuracy claim without independent reference; no cycle merging or target SV/EF fitting',
          'code_sha256':sha256_file(Path(__file__)),'adapter_sha256':sha256_file(Path(__file__).with_name('pilot_blood_segmentation_candidates.py')),
          'conversion_manifest_sha256':sha256_file(source_manifest),'environment':{},'phases_processed':[]}
    if prompts:
        plan['method']='nninteractive_ts_initial_with_structure_points'
        plan['prompt_manifest_sha256']=sha256_file(args.prompt_manifest)
        plan['prompts']=prompts
    existing=out/'pilot_manifest.json'
    if existing.exists():raise FileExistsError('Use a new output directory; candidate provenance must not be overwritten')
    existing.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    import torch,importlib.metadata
    torch.cuda.set_per_process_memory_fraction(.40,torch.device(args.device))
    config={'model_dir':str(args.model_dir.resolve()),'device':args.device,'use_torch_compile':False,'verbose':False,'torch_n_threads':8,'enable_undo':False}
    print('Loading local nnInteractive model',flush=True)
    runner=NNInteractiveRunner(config,Path(__file__).parent)
    plan['model_provenance']=runner.model_provenance
    plan['environment']={p:importlib.metadata.version(p) for p in ['nninteractive','torch','nibabel','numpy','scipy']}
    plan['device']=args.device
    measurements=[]
    try:
        for pid in PHASES:
            ct_path=source/'phases'/f'{pid}.nii.gz';ct_img=nib.load(str(ct_path));verify_grid(ct_img,infos[pid])
            ct=np.asarray(ct_img.dataobj,dtype=np.float32);spacing=np.linalg.norm(ct_img.affine[:3,:3],axis=0)
            voxel=abs(float(np.linalg.det(ct_img.affine[:3,:3])))/1000
            info={'phase_id':pid,'ct_sha256':sha256_file(ct_path),'source_axis_codes':nib.aff2axcodes(ct_img.affine),'source_spacing_mm':spacing.tolist(),'candidates':[]}
            other_masks={}
            for ch in ['heart_atrium_left','heart_atrium_right','heart_ventricle_left','heart_ventricle_right']:
                path=source/'automatic_masks_v2'/pid/'heartchambers_highres'/f'{ch}.nii.gz';img=nib.load(str(path));verify_grid(img,infos[pid],ct_img)
                other_masks[ch]=np.asarray(img.dataobj)>0
            predictions={}
            for ch in CHAMBERS:
                raw_path=source/'automatic_masks_v2'/pid/'heartchambers_highres'/f'{ch}.nii.gz';raw=other_masks[ch]
                destination=out/pid/'nninteractive_proposal'/f'{ch}.nii.gz'
                print(f'Predicting {pid}/{ch}',flush=True)
                if prompts is None:
                    prediction,details=runner.predict(ct,spacing,mode='ts_initial_refinement',initial_seg=raw,positive_points=[],negative_points=[])
                else:
                    spec=next(r for r in prompts['items'] if r['phase_id']==pid and r['chamber']==ch)
                    if spec['ct_sha256']!=info['ct_sha256'] or spec['source_mask_sha256'][ch]!=sha256_file(raw_path):raise ValueError('Prompt inputs changed')
                    target=np.zeros(ct.shape,dtype=np.uint8);started=time.perf_counter()
                    runner.session.set_image(ct[None,...],{'spacing':tuple(float(x) for x in spacing)})
                    runner.session.set_target_buffer(target)
                    runner.session.add_initial_seg_interaction(raw.astype(np.uint8),run_prediction=False)
                    interactions=[(p,True) for p in spec['positive_points']]+[(p,False) for p in spec['negative_points']]
                    for index,(point,include) in enumerate(interactions):
                        runner.session.add_point_interaction(tuple(point),include_interaction=include,run_prediction=True)
                    prediction=target>0;details={'elapsed_seconds':time.perf_counter()-started,'positive_points':spec['positive_points'],'negative_points':spec['negative_points'],'prediction_after_each_point':True}

                if prediction.shape!=raw.shape:raise ValueError('Prediction grid changed')
                _save_binary_mask(destination,prediction,ct_img);predictions[ch]=prediction
                added=prediction & ~raw;removed=raw & ~prediction
                _save_binary_mask(destination.with_name(ch+'_added.nii.gz'),added,ct_img)
                _save_binary_mask(destination.with_name(ch+'_removed.nii.gz'),removed,ct_img)
                counts=np.bincount(ndimage.label(prediction,structure=np.ones((3,3,3)))[0].ravel())[1:]
                row={'phase_id':pid,'chamber':ch,'raw_ml':int(raw.sum())*voxel,'candidate_ml':int(prediction.sum())*voxel,
                     'delta_ml':int(prediction.sum())*voxel-int(raw.sum())*voxel,'added_ml':int(added.sum())*voxel,'removed_ml':int(removed.sum())*voxel,
                     'agreement_dice_not_accuracy':2*int((raw & prediction).sum())/(int(raw.sum())+int(prediction.sum())),
                     'components_26':len(counts),'largest_component_fraction':float(counts.max()/counts.sum()) if len(counts) else 0,
                     'status':'proposal_pending_visual_review','independent_reference':False}
                for name,mask in [('added',added),('removed',removed),('raw_interior',ndimage.binary_erosion(raw,iterations=3))]:
                    v=ct[mask];row[name+'_HU_median']=float(np.median(v)) if len(v) else None
                for other,mask in other_masks.items():
                    if other!=ch:row['overlap_with_raw_'+other+'_ml']=int((prediction & mask).sum())*voxel
                measurements.append(row)
                info['candidates'].append({'chamber':ch,'raw_sha256':sha256_file(raw_path),'candidate_file':destination.relative_to(out).as_posix(),'candidate_sha256':sha256_file(destination),'details':details,'metrics':row})
                print(json.dumps(row,ensure_ascii=False),flush=True)
            info['candidate_ventricle_overlap_ml']=int((predictions[CHAMBERS[0]] & predictions[CHAMBERS[1]]).sum())*voxel
            plan['phases_processed'].append(info);existing.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
            with (out/'candidate_measurements.csv').open('w',encoding='utf-8-sig',newline='') as f:
                keys=list(dict.fromkeys(k for r in measurements for k in r));writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(measurements)
            del ct,predictions,other_masks
    finally:
        runner.close()
    plan['finished_at']=datetime.now(timezone.utc).isoformat();plan['run_status']='completed_proposals_not_accepted'
    existing.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    import shutil
    source_copy=out/'source_code';source_copy.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),source_copy/Path(__file__).name)
    print('Completed',len(measurements),'ventricular proposals',flush=True)

if __name__=='__main__':main()
