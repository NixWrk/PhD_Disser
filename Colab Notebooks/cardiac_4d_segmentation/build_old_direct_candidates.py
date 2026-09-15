"""Direct reviewed per-phase cavity envelopes, separated from HU-derived subsets."""
from pathlib import Path
import argparse,json
import nibabel as nib
import numpy as np
from scipy import ndimage
from render_authored_heart_contours import render_envelope
from prepare_old_reference_workspace import binary_nifti
from compare_authored_cardiac_volumes import sha

CHAMBERS=["LV","RV","LA","RA"]
def build(root,review,out):
 root=Path(root);review=Path(review);out=Path(out)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);rows=[]
 for sp in sorted(review.glob("phase_*_direct_contours_v01.local.json")):
  spec=json.loads(sp.read_text(encoding="utf-8"));pid=spec["phase_id"];ct=nib.load(spec["ct_path"]);a=np.asarray(ct.dataobj);spacing=np.linalg.norm(ct.affine[:3,:3],axis=0);voxel_ml=abs(np.linalg.det(ct.affine[:3,:3]))/1000
  if sha(spec["ct_path"])!=spec["source_ct_sha256"]:raise ValueError("CT mismatch")
  masks={ch:render_envelope(a.shape,c,spacing[:2]) for ch,c in spec["contours"].items()}
  if pid=="phase_00":
   old=nib.load(root/"whole_chamber_review_v01/phase00_search_domains_v01/chamber_domains.nii.gz");d=np.asarray(old.dataobj)
   for i,ch in enumerate(CHAMBERS,1):
    if ch not in masks:masks[ch]=d==i
  conflict=sum(m.astype(np.uint8) for m in masks.values())>1
  folder=out/pid;folder.mkdir();binary_nifti(conflict,ct,folder/"uncertain_overlap.nii.gz");records={}
  for ch in CHAMBERS:
   # Ambiguous overlap excluded from all chambers, retained in a separate uncertainty mask.
   envelope=masks[ch]&~conflict
   for name,m in [("cavity_envelope",envelope)]+[(f"HU{hu}",envelope&(a>=hu)) for hu in [170,190,210]]:
    path=folder/f"{ch}_{name}.nii.gz";binary_nifti(m,ct,path)
    records[ch+"_"+name]=dict(file=path.name,sha256=sha(path),volume_ml=float(m.sum()*voxel_ml))
    rows.append(dict(phase_id=pid,phase_percent=spec["phase_percent"],region=ch,definition=name,volume_ml=float(m.sum()*voxel_ml)))
  receipt=dict(phase_id=pid,phase_percent=spec["phase_percent"],source_ct_sha256=spec["source_ct_sha256"],source_spec_sha256=sha(sp),builder_sha256=sha(__file__),method="Direct per-phase CT cavity envelope; no cross-phase transport; unblinded visual annotation",overlap_excluded_ml=float(conflict.sum()*voxel_ml),accepted=False,independent_reference=False,source_spec=str(sp.resolve()),records=records)
  (folder/"receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
  print(pid,"overlap excluded",round(receipt["overlap_excluded_ml"],3),[(ch,round(records[ch+"_cavity_envelope"]["volume_ml"],2)) for ch in CHAMBERS],flush=True)
 (out/"volumes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--review",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();build(a.root,a.review,a.output)
