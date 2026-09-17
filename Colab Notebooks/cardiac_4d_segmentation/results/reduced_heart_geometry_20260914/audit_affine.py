from pathlib import Path
import hashlib,json
import numpy as np
p=Path('Colab Notebooks/cardiac_4d_segmentation/results/reduced_heart_geometry_20260914/exploratory')
b=[json.loads(s) for s in (p/'comparison_02/geometry_comparison.jsonl').read_text(encoding='utf-8').splitlines()]
a=[json.loads(s) for s in (p/'affine_01/affine_comparison.jsonl').read_text(encoding='utf-8').splitlines()]
key=lambda r:(r['subject'],r['cycle_index'],r['phase_id'],r['region'])
lookup={key(r):r for r in b}
assert len(a)==234 and len({key(r) for r in a})==234
checks={'unique_rows_and_baseline_keyset':{key(r) for r in a}==set(lookup),'nominal_identity':True,'volume':True,'positive_transform_determinants':True,'sample_count_Dice':True,'mask_hashes':True,'reference_hashes':True,'covariance_transform':True}
summary=[]
for r in a:
 base=lookup[key(r)];ref=lookup[(r['subject'],r['cycle_index'],r['reference_phase_id'],r['region'])]
 checks['mask_hashes'] &= r['mask_sha256']==base['provenance']['sha256']
 checks['reference_hashes'] &= r['reference_mask_sha256']==base['reference_mask_sha256']
 for model,c in r['comparisons'].items():
  par,o=c['parameters'],c['overlap'];F=np.array(par['transform_world'])[:3,:3]
  checks['volume'] &= bool(np.isclose(par['volume_ml'],base['moments']['volume_ml'],rtol=1e-10,atol=1e-9))
  checks['positive_transform_determinants'] &= bool(np.linalg.det(F)>0)
  checks['covariance_transform'] &= bool(np.allclose(F @ np.array(ref['moments']['covariance_mm2']) @ F.T,par['covariance_mm2'],rtol=1e-10,atol=1e-8))
  expected=2*o['intersection_sample_count']/(o['target_voxel_count']+o['candidate_sample_count'])
  checks['sample_count_Dice'] &= abs(o['dice_voxel_center_approx']-expected)<1e-12
  if r['is_reference_phase']: checks['nominal_identity'] &= abs(o['dice_voxel_center_approx']-1)<1e-12
for region in sorted({r['region'] for r in a}):
 chosen=[r for r in a if r['region']==region and not r['is_reference_phase']]
 for model in ['individual_rotation','individual_affine']:
  diffs=np.array([r['comparisons'][model]['overlap']['dice_voxel_center_approx']-lookup[key(r)]['comparisons']['individual_isotropic']['overlap']['dice_voxel_center_approx'] for r in chosen])
  summary.append({'region':region,'model':model,'n':len(chosen),'median_paired_delta':float(np.median(diffs)),'improved_count':int(np.count_nonzero(diffs>0))})
assert all(checks.values()),checks
raw=(p/'affine_01/affine_comparison.jsonl').read_bytes()
receipt={'status':'passed','scope':'saved_affine_geometry_arithmetic_and_input_identity_only','checks':checks,
 'rows':len(a),'reference_rows':sum(r['is_reference_phase'] for r in a),'records_sha256':hashlib.sha256(raw).hexdigest(),
 'code_sha256':{name:hashlib.sha256((p.parents[2]/name).read_bytes()).hexdigest() for name in ['heart_affine_geometry.py','heart_reduced_geometry.py','run_affine_heart_geometry.py']},
 'paired_summary':summary,'independent_anatomical_validation':False,'electrical_validation':False}
(p/'affine_01/numerical_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'checks':checks,'rows':len(a),'reference_rows':receipt['reference_rows']},ensure_ascii=False))
