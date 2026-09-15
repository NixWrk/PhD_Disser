"""CT-based review panels and technical diagnostics; no automatic anatomical acceptance."""
from pathlib import Path
import argparse,json,hashlib
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from compare_authored_cardiac_volumes import sha,read_mask
STRUCTURES={"LV":("heart_ventricle_left","#56e66b"),"RV":("heart_ventricle_right","#27d9ff"),"LA":("heart_atrium_left","#ffad32"),"RA":("heart_atrium_right","#ff6de3"),"myo":("heart_myocardium","#f65353"),"aorta":("aorta","#ffff33"),"PA":("pulmonary_artery","#eeeeee")}
def review(root,out,phases):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 allrows=[]
 for idx in phases:
  pid=f"phase_{idx:02d}";ctpath=root/"sequence_02/phases"/(pid+".nii.gz");ct=nib.load(ctpath);a=np.asarray(ct.dataobj);vox=abs(np.linalg.det(ct.affine[:3,:3]))/1000
  folder=root/"sequence02_ts_comparator_v1/automatic_masks_v2"/pid/"heartchambers_highres";masks={}
  for key,(name,color) in STRUCTURES.items():
   mask=read_mask(folder/(name+".nii.gz"),ct);masks[key]=mask;coords=np.argwhere(mask);values=a[mask];cc,n=ndimage.label(mask)
   sizes=np.bincount(cc.ravel());sizes[0]=0
   allrows.append(dict(phase=idx,region=key,volume_ml=float(mask.sum()*vox),centroid_ijk=coords.mean(axis=0).tolist(),bounds_ijk=[coords.min(axis=0).tolist(),coords.max(axis=0).tolist()],hu_quantiles=np.percentile(values,[5,25,50,75,95]).tolist(),fraction_below_170=float((values<170).mean()),component_count=n,largest_component_fraction=float(sizes.max()/mask.sum()),source_ct_sha256=sha(ctpath),mask_sha256=sha(folder/(name+".nii.gz"))))
  np.savez_compressed(out/(pid+"_review_cache.npz"),ct=a[85:385,80:390,:60],**{key:v[85:385,80:390,:60] for key,v in masks.items()})
  for page,start in enumerate(range(2,50,12),1):
   fig,axs=plt.subplots(3,4,figsize=(16,12),layout="constrained")
   for ax,k in zip(axs.flat,range(start,start+12)):
    ax.imshow(a[:,:,k],origin="upper",cmap="gray",vmin=0,vmax=550)
    for key,mask in masks.items():
     if mask[:,:,k].any():ax.contour(mask[:,:,k],levels=[.5],colors=[STRUCTURES[key][1]],linewidths=.75)
    ax.set_xlim(90,375);ax.set_ylim(365,85);ax.set_title(f"phase {idx*5}%  k={k}");ax.tick_params(labelsize=7);ax.grid(alpha=.15)
   fig.suptitle("TS: LV green; RV cyan; LA orange; RA magenta; myo red; aorta yellow; PA white",fontsize=12)
   fig.savefig(out/f"{pid}_axial_{page}.jpg",dpi=110,pil_kwargs={"quality":91});plt.close(fig)
 (out/"ts_technical_diagnostics.json").write_text(json.dumps(allrows,indent=2),encoding="utf-8")
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--phases",type=int,nargs="+",default=[0,8]);a=p.parse_args();review(a.root,a.output,a.phases)
