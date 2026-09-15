from pathlib import Path
import json,sys,hashlib
import nibabel as nib,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;sys.path.insert(0,str(root.parents[2]))
from render_authored_heart_contours import render_envelope
from prepare_old_reference_workspace import binary_nifti
s=json.loads((root/'phase00_outer_spec.local.json').read_text(encoding='utf-8'));ct=nib.load(root.parent/'sequence_02/phases/phase_00.nii.gz');a=np.asarray(ct.dataobj);spacing=np.linalg.norm(ct.affine[:3,:3],axis=0)
outer=render_envelope(a.shape,s['contours'],spacing[:2]);blood=np.zeros(a.shape,bool)
for ch in ['LV','RV','LA','RA']:blood|=np.asarray(nib.load(root/f'phase00_candidate_v02/{ch}_HU190_boundary+0mm.nii.gz').dataobj)>0
out=root/'phase00_outer_v01';out.mkdir(exist_ok=False)
# Residual deliberately not labelled as an accepted myocardial segmentation.
residual=outer&~blood
soft=residual&(a>=0)&(a<190)
for name,m in [('outer_heart_candidate',outer),('wall_residual_candidate',residual),('myocardium_intensity_candidate',soft)]:binary_nifti(m,ct,out/(name+'.nii.gz'))
checks={'accepted':False,'myocardium_is_reference':False,'blood_outside_outer_voxels':int((blood&~outer).sum()),'source_ct_sha256':hashlib.sha256((root.parent/'sequence_02/phases/phase_00.nii.gz').read_bytes()).hexdigest(),'definitions':{'outer_heart_candidate':'Authored anatomical envelope, vessel and endpoint review pending','wall_residual_candidate':'Authored outer envelope minus four chamber blood masks, may include vessels, unsegmented blood and other tissue','myocardium_intensity_candidate':'Wall residual restricted to 0<=HU<190, not a specific validated myocardial segmentation'},'policy':s,'volumes_ml':{name:float(m.sum()*np.prod(spacing)/1000) for name,m in [('outer',outer),('residual',residual),('intensity_candidate',soft)]}}
(out/'provenance.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,4,figsize=(16,9),layout='constrained')
for ax,k in zip(axs.flat,[4,8,14,20,26,32,38,44]):
 ax.imshow(a[:,:,k],origin='upper',cmap='gray',vmin=0,vmax=500);ax.contour(outer[:,:,k],levels=[.5],colors=['#fff04b'],linewidths=1);ax.contour(soft[:,:,k],levels=[.5],colors=['#ff5353'],linewidths=.55);ax.set_xlim(90,370);ax.set_ylim(350,95);ax.set_title(f'k={k}');ax.grid(alpha=.15)
fig.savefig(root/'phase00_outer_candidates.jpg',dpi=120,pil_kwargs={'quality':89});plt.close(fig)
print(json.dumps({k:v for k,v in checks.items() if k!='policy'},ensure_ascii=False,indent=2))
