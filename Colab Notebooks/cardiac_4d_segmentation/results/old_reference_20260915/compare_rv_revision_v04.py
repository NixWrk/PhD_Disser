from pathlib import Path
import json,hashlib
import numpy as np,nibabel as nib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;out=root/'body_review_v06'
ct=nib.load(root/'sequence_02/phases/phase_00.nii.gz');a=np.asarray(ct.dataobj)
oldimg=nib.load(root/'agent_drafts/sequence02_phase00_RV_body_v02/RV_visible_body_agent_HU190.nii.gz');newimg=nib.load(root/'agent_drafts/sequence02_phase00_RV_body_v04/RV_visible_body_agent_HU190.nii.gz')
old=np.asarray(oldimg.dataobj)>0;new=np.asarray(newimg.dataobj)>0;lv=np.asarray(nib.load(root/'agent_drafts/sequence02_phase00_LV_body_v04/LV_visible_body_agent_HU190.nii.gz').dataobj)>0
vox=abs(np.linalg.det(ct.affine[:3,:3]))/1000
assert np.array_equal(ct.affine,newimg.affine) and np.array_equal(old[:,:,16:],new[:,:,16:])
assert not (new&lv).any()
r={'reference_accepted':False,'partial_volume_only':True,'old_ml':float(old.sum()*vox),'new_ml':float(new.sum()*vox),'added_ml':float((new&~old).sum()*vox),'removed_ml':float((old&~new).sum()*vox),'changed_slices':np.flatnonzero(np.any(old!=new,axis=(0,1))).tolist(),'overlap_voxels':int((new&lv).sum()),'grid_unchanged':True,'k16_and_above_unchanged':True,'per_slice':[{'k':k,'old_ml':float(old[:,:,k].sum()*vox),'new_ml':float(new[:,:,k].sum()*vox)} for k in range(4,25)]}
r['delta_ml']=r['new_ml']-r['old_ml'];r['delta_percent']=r['delta_ml']/r['old_ml']*100
(out/'RV_revision_comparison.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
for start in [4,8,12]:
 fig,axs=plt.subplots(2,4,figsize=(16,8),layout='constrained')
 for col,k in enumerate(range(start,start+4)):
  for row in range(2):
   ax=axs[row,col];ax.imshow(a[:,:,k],cmap='gray',vmin=0,vmax=400,origin='upper');ax.set_xlim(120,310);ax.set_ylim(275,100);ax.grid(alpha=.15);ax.set_title(f'Срез {k}: '+('исходная КТ' if row==0 else 'было / исправлено'))
   if row:
    ax.contour(old[:,:,k],levels=[.5],colors=['#ffad33'],linestyles='--',linewidths=1)
    ax.contour(new[:,:,k],levels=[.5],colors=['#00c9ff'],linewidths=1)
 fig.savefig(out/f'RV_before_after_{start}_{start+3}.jpg',dpi=120,pil_kwargs={'quality':89});plt.close(fig)
print(json.dumps({k:v for k,v in r.items() if k!='per_slice'},indent=2))
