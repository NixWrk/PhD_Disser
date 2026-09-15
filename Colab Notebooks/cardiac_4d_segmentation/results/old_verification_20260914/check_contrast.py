from pathlib import Path
import json,pydicom
p=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
a=json.loads((p/'dicom_inventory.local.json').read_text(encoding='utf8'))
for r in a:
 flags=[]
 for s in r['series']:
  d=pydicom.dcmread(Path(r['folder'])/s['files'][0],stop_before_pixels=True)
  for e in d.iterall():
   if e.keyword=='ContrastBolusAgentAdministered':flags.append(str(e.value))
 print(r['case'],'contrast flags',sorted(set(flags)),'CT_objects_with_flags',len(flags))
