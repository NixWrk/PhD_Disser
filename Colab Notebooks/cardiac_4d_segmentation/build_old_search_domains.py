"""Reconstruct four disjoint search domains from the frozen OLD baseline envelopes."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import nibabel as nib

def build(root,output):
 root=Path(root);output=Path(output)
 if output.exists():raise FileExistsError(output)
 spec=json.loads((root/"phase00_authored_spec_v02.local.json").read_text(encoding="utf-8"));ct=nib.load(spec["ct_path"])
 if hashlib.sha256(Path(spec["ct_path"]).read_bytes()).hexdigest()!=spec["ct_sha256"]:raise ValueError("CT changed")
 spacing=np.linalg.norm(ct.affine[:3,:3],axis=0);frame=spec["long_axis_frame"];grids=np.ogrid[tuple(slice(0,n) for n in ct.shape)]
 def coord(direction):return sum((q*sp-origin)*d for q,sp,origin,d in zip(grids,spacing,frame["origin_scaled_ijk_mm"],frame[direction]))
 u=coord("direction_u");v=coord("direction_v");envelopes={}
 for ch in ["LV","RV","LA","RA"]:
  img=nib.load(root/"phase00_candidate_v02"/(ch+"_envelope.nii.gz"))
  if img.shape!=ct.shape or not np.allclose(img.affine,ct.affine,atol=1e-5,rtol=0):raise ValueError("Envelope grid differs")
  envelopes[ch]=np.asarray(img.dataobj)>0
 bounds={"LV":v<=84,"LA":v>84,"RV":(~envelopes["RA"])|(v+.2*u<=76),"RA":(~envelopes["RV"])|(v+.2*u>76)}
 labels=np.zeros(ct.shape,dtype=np.uint8)
 for i,ch in enumerate(["LV","RV","LA","RA"],1):
  mask=envelopes[ch]&bounds[ch]
  if np.any(mask&(labels>0)):raise ValueError("Overlapping search domains")
  labels[mask]=i
 output.mkdir(parents=True);header=ct.header.copy();header.set_data_dtype(np.uint8);img=nib.Nifti1Image(labels,ct.affine,header)
 nib.save(img,output/"chamber_domains.nii.gz")
 return labels
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();build(a.root,a.output)
