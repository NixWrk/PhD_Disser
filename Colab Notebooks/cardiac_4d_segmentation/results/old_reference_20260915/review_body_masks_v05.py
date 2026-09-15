from pathlib import Path
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
root=Path(__file__).parent
ct=nib.load(root/'sequence_02/phases/phase_00.nii.gz');a=np.asarray(ct.dataobj)
lv=np.asarray(nib.load(root/'agent_drafts/sequence02_phase00_LV_body_v04/LV_visible_body_agent_HU190.nii.gz').dataobj).astype(bool)
rv=np.asarray(nib.load(root/'agent_drafts/sequence02_phase00_RV_body_v03/RV_visible_body_agent_HU190.nii.gz').dataobj).astype(bool)
out=root/'body_review_v05';out.mkdir(exist_ok=True)
for page,start in enumerate([4,12,20],1):
    fig,axs=plt.subplots(2,4,figsize=(15,8),layout='constrained')
    for ax,k in zip(axs.ravel(),range(start,start+8)):
        ax.imshow(a[:,:,k],origin='upper',cmap='gray',vmin=0,vmax=400)
        for mask,color in [(lv,'#36ff70'),(rv,'#00c9ff')]:
            if mask[:,:,k].any():ax.contour(mask[:,:,k],levels=[.5],colors=[color],linewidths=.9)
        ax.set_xlim(100,380);ax.set_ylim(330,90);ax.grid(alpha=.15)
        ax.set_title(f'Срез {k}')
    fig.savefig(out/f'axial_page{page}.jpg',dpi=120,pil_kwargs={'quality':89});plt.close(fig)
planes=json.loads((root/'oblique_planes.local.json').read_text())
u=np.linspace(-40,80,481);v=np.linspace(-15,125,561);U,V=np.meshgrid(u,v)
fig,axs=plt.subplots(1,2,figsize=(12,7),layout='constrained')
for ax,p in zip(axs,planes[:2]):
    ijk=(np.array(p['origin_scaled_ijk_mm'])[:,None,None]+np.array(p['direction_u'])[:,None,None]*U+np.array(p['direction_v'])[:,None,None]*V)/np.array(p['spacing_mm'])[:,None,None]
    values=map_coordinates(a,ijk,order=1,cval=-1000,mode='constant')
    ax.imshow(values,origin='lower',extent=(u[0],u[-1],v[0],v[-1]),cmap='gray',vmin=0,vmax=400)
    for mask,color in [(lv,'#36ff70'),(rv,'#00c9ff')]:
        values=map_coordinates(mask.astype(float),ijk,order=0,cval=0,mode='constant')
        ax.contour(U,V,values,levels=[.5],colors=[color],linewidths=1)
    ax.set_title(('Продольная плоскость 1' if p == planes[0] else 'Продольная плоскость 2')+' | частичные маски');ax.set_xlabel('u, mm');ax.set_ylabel('v, mm');ax.grid(alpha=.15)
fig.savefig(out/'oblique_partial_boundaries.jpg',dpi=130,pil_kwargs={'quality':89});plt.close(fig)
print('Overlap voxels',int((lv&rv).sum()))
