"""Extend a saved geometry comparison with two framewise affine candidates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import nibabel as nib
import numpy as np

from run_reduced_heart_geometry import load_crop, mask_path, digest, serial, write_json
from heart_reduced_geometry import mask_moments, sampled_dice
from heart_affine_geometry import individual_rigid_isotropic_candidate, individual_affine_moment_candidate


def run(config_path, baseline, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    root, baseline, output = Path(config['derived_root']), Path(baseline), Path(output)
    rows = [json.loads(line) for line in (baseline/'geometry_comparison.jsonl').read_text(encoding='utf-8').splitlines()]
    base_status = json.loads((baseline/'run_status.json').read_text(encoding='utf-8'))
    if base_status['status'] != 'completed' or len(rows) != base_status['completed_phase_region_pairs']:
        raise ValueError('Incomplete baseline calculation')
    output.mkdir(parents=True, exist_ok=False)
    status = {'status':'running','completed_pairs':0,'expected_pairs':len(rows),
              'evidence_status':'exploratory_hypothesis_not_validated',
              'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'run_status.json',status)
    provenance = {'baseline_records_sha256':digest(baseline/'geometry_comparison.jsonl'),
        'config_sha256':digest(config_path), 'code_sha256':{name:digest(Path(__file__).with_name(name)) for name in
        ('run_affine_heart_geometry.py','heart_affine_geometry.py','heart_reduced_geometry.py','run_reduced_heart_geometry.py')}}
    write_json(output/'input_provenance.local.json',provenance)
    try:
        manifests={subject:json.loads((root/subject/'phases/conversion_manifest.json').read_text(encoding='utf-8'))
                   for subject in config['subjects']}
        references={}
        phase_cache={}
        started=time.monotonic()
        with (output/'affine_comparison.jsonl').open('w',encoding='utf-8') as stream:
            for index,row in enumerate(rows):
                subject,phase,region,cycle=(row[k] for k in ('subject','phase_id','region','cycle_index'))
                phase_key=subject,phase
                if phase_key not in phase_cache:
                    info=next(p for p in manifests[subject]['phases'] if p['phase_id']==phase)
                    ct_path=root/subject/'phases'/info['nifti']
                    if digest(ct_path)!=row['ct_sha256']:
                        raise ValueError('CT differs from baseline')
                    phase_cache[phase_key]=(info,nib.load(str(ct_path)))
                info,ct=phase_cache[phase_key]
                target,affine,mask_provenance=load_crop(mask_path(root,subject,phase,region),info,ct)
                if mask_provenance['sha256']!=row['provenance']['sha256']:
                    raise ValueError('Target mask differs from baseline')
                moments=mask_moments(target,affine)
                for field in ('volume_mm3','centroid_mm','covariance_mm2'):
                    if not np.allclose(moments[field],row['moments'][field],atol=1e-8,rtol=1e-10):
                        raise ValueError('Moments differ from baseline')
                ref_key=subject,cycle,region
                if ref_key not in references:
                    if not row['is_reference_phase']:
                        raise ValueError('Baseline is not reference-first within each cycle')
                    references[ref_key]=(target,affine,moments,phase,mask_provenance['sha256'])
                ref_mask,ref_affine,ref_moments,ref_phase,ref_sha=references[ref_key]
                if ref_phase!=row['reference_phase_id'] or ref_sha!=row['reference_mask_sha256']:
                    raise ValueError('Reference differs from baseline')
                comparisons={}
                for name,fn in [('individual_rotation',individual_rigid_isotropic_candidate),
                                ('individual_affine',individual_affine_moment_candidate)]:
                    candidate=fn(ref_mask,ref_affine,target,affine,reference_moments=ref_moments,target_moments=moments)
                    overlap=sampled_dice(target,affine,candidate,target_moments=moments)
                    if row['is_reference_phase'] and abs(overlap['dice_voxel_center_approx']-1)>1e-12:
                        raise ValueError('Reference identity failed')
                    comparisons[name]={'parameters':serial(candidate),'overlap':serial(overlap)}
                record={k:row[k] for k in ('subject','phase_id','cycle_index','region','is_reference_phase','reference_phase_id')}
                record.update(mask_sha256=mask_provenance['sha256'],reference_mask_sha256=ref_sha,
                              comparisons=comparisons)
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                status.update(completed_pairs=index+1,last_completed=f'{subject}/{phase}/{region}')
                write_json(output/'run_status.json',status)
                if (index+1)%6==0:
                    print(f'{index+1}/{len(rows)} pairs; {time.monotonic()-started:.0f}s',flush=True)
        status.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        write_json(output/'run_status.json',status)
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(output/'run_status.json',status)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.config,args.baseline,args.output)
