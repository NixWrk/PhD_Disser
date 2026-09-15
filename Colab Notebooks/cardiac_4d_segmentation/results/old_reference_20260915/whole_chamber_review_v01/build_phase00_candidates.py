from pathlib import Path
import json,hashlib,sys
import numpy as np,nibabel as nib
from scipy import ndimage
root=Path(__file__).parent;base=root.parents[2];sys.path.insert(0,str(base))
from render_authored_heart_contours import render_envelope
from prepare_old_reference_workspace import binary_nifti
s=json.loads((root/'phase00_authored_spec.local.json').read_text(encoding='utf-8'));ct=nib.load(s['ct_path']);a=np.asarray(ct.dataobj)
assert hashlib.sha256(Path(s['ct_path']).read_bytes()).hexdigest()==s['ct_sha256']
spacing=np.linalg.norm(ct.affine[:3,:3],axis=0);p=s['long_axis_frame'];ii,jj,kk=np.ogrid[:a.shape[0],:a.shape[1],:a.shape[2]]
u=sum((q*sp-origin)*direction for q,sp,origin,direction in zip([ii,jj,kk],spacing,p['origin_scaled_ijk_mm'],p['direction_u']))
v=sum((q*sp-origin)*direction for q,sp,origin,direction in zip([ii,jj,kk],spacing,p['origin_scaled_ijk_mm'],p['direction_v']))
env={ch:render_envelope(a.shape,c,spacing[:2]) for ch,c in s['contours'].items()}
out=root/'phase00_candidate_v01';out.mkdir(exist_ok=False)
vol=abs(np.linalg.det(ct.affine[:3,:3]))/1000;records=[]
for ch,e in env.items():binary_nifti(e,ct,out/(ch+'_envelope.nii.gz'))
for shift in [-3,0,3]:
 for hu in ([170,190,210] if shift==0 else [190]):
  bounds={'LV':v<=84+shift,'LA':v>84+shift,'RV':v+.2*u<=76+shift,'RA':v+.2*u>76+shift}
  masks={}
  for ch in env:
   m=env[ch]&bounds[ch]&(a>=hu)
   labels,n=ndimage.label(m,structure=ndimage.generate_binary_structure(3,1));sizes=np.bincount(labels.ravel());sizes[0]=0
   m=labels==int(sizes.argmax()) if n else m;masks[ch]=m
   suffix=f'HU{hu}_boundary{shift:+d}mm';name=f'{ch}_{suffix}.nii.gz';binary_nifti(m,ct,out/name)
   records.append({'chamber':ch,'HU':hu,'boundary_shift_mm':shift,'file':name,'volume_ml':float(m.sum()*vol),'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest(),'k_range':np.flatnonzero(m.any(axis=(0,1)))[[0,-1]].tolist() if m.any() else [],'accepted':False,'full_chamber':False})
  overlap=sum(m.astype('uint8') for m in masks.values())>1
  if overlap.any():raise ValueError(f'Overlapping chambers: {overlap.sum()}')
(out/'provenance.json').write_text(json.dumps({'source_ct_sha256':s['ct_sha256'],'source_spec_sha256':hashlib.sha256((root/'phase00_authored_spec.local.json').read_bytes()).hexdigest(),'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'accepted':False,'status':'four_chamber_extent_candidates_pending_visual_review','automatic_comparator_used':False,'policy':s['policy'],'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([r for r in records if r['HU']==190 and r['boundary_shift_mm']==0],indent=2))
