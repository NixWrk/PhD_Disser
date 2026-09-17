from pathlib import Path
import json,csv
root=Path('Colab Notebooks/cardiac_4d_segmentation');work=root/'results/reduced_heart_geometry_20260914';work.mkdir(parents=True,exist_ok=True)
data=Path(r'D:\Аспа\Bitrix\ЭИТЛ\02 Big_data\3D\_DERIVED_CARDIAC4D')
rows=list(csv.DictReader((root/'heart_rr_analysis/chamber_volumes_by_rr.csv').open(encoding='utf-8-sig')))
counts={}
for subject in ['adam','nix','georg']:
 m=json.loads((data/subject/'phases/conversion_manifest.json').read_text(encoding='utf8'))
 counts[subject]={'phases':len(m['phases']),'cycles':sorted({int(float(r['cycle_index'])) for r in rows if r['subject']==subject})}
 for p in m['phases']:
  folder=data/subject/'products_v2'/p['phase_id'];assert (folder/'whole_heart.nii.gz').is_file() and (folder/'blood_pool.nii.gz').is_file()
  for c in ['heart_atrium_left','heart_ventricle_left','heart_atrium_right','heart_ventricle_right']:
   assert (data/subject/'automatic_masks_v2'/p['phase_id']/'heartchambers_highres'/f'{c}.nii.gz').is_file()
config={'derived_root':str(data),'subjects':['adam','nix','georg'],'time_table':str((root/'heart_rr_analysis/chamber_volumes_by_rr.csv').resolve()),'output_dir':str(work.resolve()/'comparison_01'),'mask_definitions':{'whole_heart':'products_v2/whole_heart, same target as existing 21.03','blood_pool':'products_v2/blood_pool, union and existing component cleanup retained','chambers':'raw heartchambers_highres outputs'},'reference_rule':'earliest available phase within each subject/cycle; no ED label','independent_reference_available':False}
(work/'run_config.local.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(counts));print('All 39 phases x 6 target masks exist. No segmentation invoked.')
