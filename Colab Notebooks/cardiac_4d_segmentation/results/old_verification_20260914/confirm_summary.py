from pathlib import Path
import json
p=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
r=json.loads((p/'header_summary.json').read_text(encoding='utf8'))
g=[g for x in r for g in x['groups'] if g['phases']]
print(json.dumps({'studies':len(r),'objects':sum(x['dicom_objects'] for x in r),'full_20_phase_groups':sum(len(x['phases'])==20 for x in g),'phase_groups':[(x['case'],[(g['series'],g['phases']) for g in x['groups'] if 0<len(g['phases'])<20]) for x in r]},ensure_ascii=False))
# report actual frame tag names for one full sequence, without patient identifiers
import pydicom
inv=json.loads((p/'dicom_inventory.local.json').read_text(encoding='utf8'))
s=next(s for s in inv[1]['series'] if 'CARDIAC_PHASE' in s['sample_tags'].get('ImageType',''))
d=pydicom.dcmread(Path(inv[1]['folder'])/s['files'][0],stop_before_pixels=True)
for area,ds in [('shared',d.SharedFunctionalGroupsSequence[0]),('frame',d.PerFrameFunctionalGroupsSequence[0])]:
 print(area)
 for e in ds.iterall():
  if e.VR!='SQ' and any(w in e.keyword for w in ['Cardiac','Trigger','Temporal','Contrast','Acquisition','Dimension','Stack']):print(e.keyword,str(e.value)[:180])
