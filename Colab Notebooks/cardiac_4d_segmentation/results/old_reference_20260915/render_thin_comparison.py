from pathlib import Path
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
coarse=nib.load(root/'sequence_02/phases/phase_00.nii.gz')
thin=nib.load(root/'thin_phase00_series10/phase_00.nii.gz')
planes=json.loads((root/'oblique_planes.local.json').read_text())
u=np.linspace(-40,80,601);v=np.linspace(-15,125,701);U,V=np.meshgrid(u,v)
fig,axes=plt.subplots(2,2,figsize=(12,13),layout='constrained')
for row,plane in enumerate(planes[:2]):
    spacing=np.array(plane['spacing_mm'])
    ijk=(np.array(plane['origin_scaled_ijk_mm'])[:,None,None]+np.array(plane['direction_u'])[:,None,None]*U+np.array(plane['direction_v'])[:,None,None]*V)/spacing[:,None,None]
    ras=np.einsum('ab,bij->aij',coarse.affine[:3,:3],ijk)+coarse.affine[:3,3,None,None]
    for col,(im,name) in enumerate([(coarse,'5 mm / step 2.5'),(thin,'0.5 mm / step 0.25')]):
        inv=np.linalg.inv(im.affine);indices=np.einsum('ab,bij->aij',inv[:3,:3],ras)+inv[:3,3,None,None]
        values=map_coordinates(np.asarray(im.dataobj),indices,order=1,cval=-1000,mode='constant')
        ax=axes[row,col];ax.imshow(values,origin='lower',extent=(u[0],u[-1],v[0],v[-1]),cmap='gray',vmin=0,vmax=400)
        ax.set_title(name+' | '+plane['name']);ax.set_xticks(np.arange(-40,81,10));ax.set_yticks(np.arange(-10,126,10));ax.grid(alpha=.2)
        ax.set_xlabel('u, mm');ax.set_ylabel('v, mm')
fig.savefig(root/'thin_coarse_same_physical_planes.jpg',dpi=120,pil_kwargs={'quality':90})
