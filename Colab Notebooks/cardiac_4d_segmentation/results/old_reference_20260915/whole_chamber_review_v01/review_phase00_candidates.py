from pathlib import Path
import numpy as np,nibabel as nib
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;a=np.asarray(nib.load(root.parent/'sequence_02/phases/phase_00.nii.gz').dataobj)
masks={ch:np.asarray(nib.load(root/f'phase00_candidate_v01/{ch}_HU190_boundary+0mm.nii.gz').dataobj)>0 for ch in ['LV','RV','LA','RA']}
colors=['#32ff65','#00caff','#ffa529','#ff59dd']
for start in [14,22,30,38]:
 fig,axs=plt.subplots(2,4,figsize=(17,9),layout='constrained')
 for ax,k in zip(axs.flat,range(start,start+8)):
  ax.imshow(a[:,:,k],origin='upper',cmap='gray',vmin=0,vmax=500)
  for (ch,m),color in zip(masks.items(),colors):
   if m[:,:,k].any():ax.contour(m[:,:,k],levels=[.5],colors=[color],linewidths=.9)
  ax.set_xlim(90,370);ax.set_ylim(350,100);ax.set_title(f'k={k}');ax.grid(alpha=.15)
 fig.savefig(root/f'phase00_candidates_{start}.jpg',dpi=125,pil_kwargs={'quality':90});plt.close(fig)
