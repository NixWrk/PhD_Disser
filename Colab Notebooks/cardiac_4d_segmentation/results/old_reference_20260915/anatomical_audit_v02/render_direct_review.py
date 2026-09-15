from pathlib import Path
import numpy as np,nibabel as nib
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).parent
colors={"LV":"#56e66b","RV":"#27d9ff","LA":"#ffad32","RA":"#ff6de3"}
for idx in [0,8]:
 pid=f"phase_{idx:02d}";cache=np.load(root/(pid+"_review_cache.npz"));ct=cache['ct']
 masks={ch:np.asarray(nib.load(root/"direct_v01"/pid/(ch+"_cavity_envelope.nii.gz")).dataobj)[85:385,80:390,:60] for ch in colors}
 for page,start in enumerate(range(2,50,12),1):
  fig,axs=plt.subplots(3,4,figsize=(16,12),layout="constrained")
  for ax,k in zip(axs.flat,range(start,start+12)):
   ax.imshow(ct[:,:,k],origin="upper",extent=[80,390,385,85],cmap="gray",vmin=0,vmax=550)
   for ch,m in masks.items():
    if m[:,:,k].any():ax.contour(np.arange(80,390),np.arange(85,385),m[:,:,k],levels=[.5],colors=[colors[ch]],linewidths=.9)
   ax.set_xlim(90,375);ax.set_ylim(365,85);ax.set_title(f"Direct contours {idx*5}%  k={k}");ax.grid(alpha=.15);ax.tick_params(labelsize=7)
  fig.suptitle("LV green; RV cyan; LA orange; RA magenta — unblinded candidate, no CT-intensity threshold",fontsize=12)
  fig.savefig(root/f"{pid}_direct_{page}.jpg",dpi=110,pil_kwargs={"quality":90});plt.close(fig)
