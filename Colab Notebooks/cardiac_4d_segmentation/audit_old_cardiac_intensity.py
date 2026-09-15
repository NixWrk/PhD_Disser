"""Audit the distinction between chamber labels and CT-intensity-selected subsets."""
from pathlib import Path
import argparse,json,csv
import numpy as np,nibabel as nib
from compare_authored_cardiac_volumes import sha
NAMES={"LV":"heart_ventricle_left","RV":"heart_ventricle_right","LA":"heart_atrium_left","RA":"heart_atrium_right","myo":"heart_myocardium"}
def audit(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
 for idx in range(20):
  pid=f"phase_{idx:02d}";path=root/"sequence_02/phases"/(pid+".nii.gz");im=nib.load(path);a=np.asarray(im.dataobj);v=abs(np.linalg.det(im.affine[:3,:3]))/1000
  for ch,name in NAMES.items():
   p=root/"sequence02_ts_comparator_v1/automatic_masks_v2"/pid/"heartchambers_highres"/(name+".nii.gz");m=nib.load(p)
   if im.shape!=m.shape or not np.allclose(im.affine,m.affine,rtol=0,atol=1e-5):raise ValueError("Grid mismatch")
   mask=np.asarray(m.dataobj)>0;vals=a[mask]
   rows.append(dict(phase_id=pid,phase_percent=idx*5,region=ch,mask_sha256=sha(p),source_ct_sha256=sha(path),raw_volume_ml=float(len(vals)*v),**{f"HU{hu}_subset_ml":float((vals>=hu).sum()*v) for hu in [170,190,210]},fraction_below_170=float((vals<170).mean()),median_hu=float(np.median(vals))))
  print(pid,flush=True)
 (out/"intensity_audit.json").write_text(json.dumps(dict(status="definition_audit_not_tissue_truth",rows=rows),indent=2),encoding="utf-8")
 with (out/"intensity_audit.csv").open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();audit(a.root,a.output)
