from pathlib import Path
import json,hashlib,sys
root=Path('Colab Notebooks/cardiac_4d_segmentation').resolve();sys.path.insert(0,str(root))
from run_historical_cardiac_sequence import digest
from segment_totalseg_v2 import assert_same_grid,HIGHRES_MASKS
work=root/'results/old_verification_20260914';cfg=json.loads((work/'run_config.local.json').read_text(encoding='utf8'));out=Path(cfg['output_dir']);seq=Path(cfg['sequence_dir'])
statepath=out/'run_status.json';old=json.loads(statepath.read_text(encoding='utf8'))
assert old['input_signatures']['segmenter']==digest(root/'segment_totalseg_v2.py')
assert old['input_signatures']['analyzer']==digest(root/'analyze_historical_cardiac_sequence.py')
conversion=seq/'phases/conversion_manifest.json';assert old['input_signatures']['conversion']==digest(conversion)
m=json.loads(conversion.read_text(encoding='utf8'));expected=[p['nifti'] for p in m['phases']]
assert expected==[p.name for p in sorted((seq/'phases').glob('phase_*.nii.gz'))]
new=json.loads(json.dumps(old));new['completed_phase_ids']=[];new['phase_hashes']={};models=set();adopted=[]
for p in m['phases']:
 i=p['phase_index'];pid=p['phase_id'];phase_dir=out/'automatic_masks_v2'/pid
 if not phase_dir.exists():continue
 mp=out/'automatic_masks_v2'/f'segmentation_manifest_gpu0_p{i}_{i+1}.json'
 assert mp.is_file(),pid
 sm=json.loads(mp.read_text(encoding='utf8'));assert sm['backend_version']=='2.18.0' and len(sm['runs'])==1 and sm['runs'][0]['phase_id']==pid and sm['runs'][0]['status']=='automatic_segmentation_pending_manual_review'
 files=[phase_dir/'total_heart/heart.nii.gz']+[phase_dir/'heartchambers_highres'/x for x in HIGHRES_MASKS]
 assert all(f.is_file() for f in files)
 ct=seq/'phases'/p['nifti'];assert digest(ct)==p['sha256'];assert_same_grid(ct,files)
 models.add(hashlib.sha256(json.dumps({k:sm[k] for k in ['backend_version','torch','cuda_runtime','models','model_files']},sort_keys=True).encode()).hexdigest())
 assert len(sm['model_files'])==43
 hashes={str(f.relative_to(out)):digest(f) for f in [*files,mp]}
 if pid in old['completed_phase_ids']:assert hashes==old['phase_hashes'][pid]
 else:adopted.append(pid)
 new['completed_phase_ids'].append(pid);new['phase_hashes'][pid]=hashes
assert len(models)==1
new['model_signature']=next(iter(models))
new['input_signatures']={'runner':digest(root/'run_historical_cardiac_sequence.py'),'config':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'conversion':digest(conversion),'segmenter':digest(root/'segment_totalseg_v2.py'),'analyzer':digest(root/'analyze_historical_cardiac_sequence.py')}
new.update(status='prepared_after_review_fix',stage='pilot_segmentation',pilot_report_written=False)
receipt={'reason':'Owned parent paused to fix resume guards; active segmentation child allowed to finish','old_runner_sha256':digest(work/'runner_before_review.py'),'new_runner_sha256':digest(root/'run_historical_cardiac_sequence.py'),'old_input_signatures':old['input_signatures'],'new_input_signatures':new['input_signatures'],'completed_phase_ids':new['completed_phase_ids'],'adopted_finished_child_phases':adopted,'all_8_masks_per_phase_grid_checked':True,'model_files_per_phase':43,'model_environment_signatures_equal':True,'pilot_QC_still_required':True}
(work/'run_status_before_review_fix.local.json').write_text(json.dumps(old,indent=2),encoding='utf8')
(work/'runner_reconciliation.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
statepath.write_text(json.dumps(new,indent=2),encoding='utf8')
print(json.dumps({'retained_completed_phases':len(new['completed_phase_ids']),'adopted_finished_child':adopted,'models_consistent':True}))
