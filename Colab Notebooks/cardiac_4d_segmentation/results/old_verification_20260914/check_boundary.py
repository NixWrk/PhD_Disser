from pathlib import Path
import json,numpy as np,nibabel as nib
root=Path('Colab Notebooks/cardiac_4d_segmentation');w=root/'results/old_verification_20260914';out=w/'run_01'
s=json.loads((out/'analysis/automatic_sequence_summary.json').read_text(encoding='utf8'))
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('seq_analysis',root/'analyze_historical_cardiac_sequence.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
for r in s['rows']:
 if not r['touches_image_boundary']:continue
 a=np.asanyarray(nib.load(out/'automatic_masks_v2'/r['phase_id']/mod.MASKS[r['structure']][0]).dataobj)>0
 faces={f'axis{axis}_{side}':int(a.take(index,axis=axis).sum()) for axis in range(3) for side,index in [('first',0),('last',-1)]}
 rows.append({'phase_percent':r['phase_percent'],'structure':r['structure'],'boundary_voxels_by_face':{k:v for k,v in faces.items() if v}})
(w/'pilot_boundary_qc.json').write_text(json.dumps({'rows':rows,'interpretation':'Boundary contact requires checking field-of-view truncation; not itself a measured segmentation error.'},indent=2),encoding='utf8')
print(json.dumps(rows))
