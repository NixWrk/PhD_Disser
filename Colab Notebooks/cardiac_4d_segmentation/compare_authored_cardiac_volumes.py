"""Compare chamber volumes and phase differences on exactly matched CT grids.

An unaccepted authored or propagated candidate permits exploratory disagreement
analysis only. This command never validates a reference or computes SV/EF.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import nibabel as nib
import numpy as np

CHAMBERS=("LV","RV","LA","RA")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_mask(path,ct):
    img=nib.load(path)
    if img.shape!=ct.shape or not np.allclose(img.affine,ct.affine,rtol=0,atol=1e-5):
        raise ValueError("Mask and source CT grids differ")
    if img.header.get_xyzt_units()[0]!="mm":
        raise ValueError("Mask spatial units must be mm")
    a=np.asarray(img.dataobj)
    if not np.isfinite(a).all() or not np.isin(a,[0,1]).all():
        raise ValueError("Expected a finite binary mask")
    if not a.any():
        raise ValueError("Empty chamber mask cannot support volume-change comparison")
    return a.astype(bool)

def compare(manifest,output):
    manifest=Path(manifest);cfg=json.loads(manifest.read_text(encoding="utf-8-sig"))
    phases=cfg["phases"];ids=[p["phase_id"] for p in phases]
    if len(phases)<2 or len(set(ids))!=len(ids):
        raise ValueError("At least two distinct phases required")
    if cfg["baseline_phase"] not in ids:
        raise ValueError("Explicit baseline phase is absent")
    out=Path(output)
    if out.exists():raise FileExistsError("Use a new comparison output directory")
    rows=[];provenance=[]
    for phase in phases:
        if sha(phase["ct_path"])!=phase["ct_sha256"]:
            raise ValueError("Source CT hash mismatch")
        ct=nib.load(phase["ct_path"])
        if len(ct.shape)!=3 or ct.header.get_xyzt_units()[0]!="mm":
            raise ValueError("Expected a 3D source CT in mm")
        voxel_ml=abs(float(np.linalg.det(ct.affine[:3,:3])))/1000
        if not np.isfinite(voxel_ml) or voxel_ml<=0:raise ValueError("Invalid physical voxel volume")
        values={};hashes={}
        for method in ["authored","automatic"]:
            if phase[method+"_source_ct_sha256"]!=phase["ct_sha256"]:
                raise ValueError("Mask provenance refers to a different source CT")
            masks=[]
            for ch in CHAMBERS:
                path=phase[method+"_masks"][ch];mask=read_mask(path,ct);masks.append(mask)
                actual=sha(path)
                if actual!=phase[method+"_mask_hashes"][ch]:raise ValueError("Mask hash mismatch")
                hashes[method+"_"+ch]=actual
                values[method+"_"+ch]=float(mask.sum()*voxel_ml)
            if np.any(sum(m.astype(np.uint8) for m in masks)>1):
                raise ValueError("Chamber masks overlap")
            values[method+"_blood_pool_sum"]=sum(values[method+"_"+ch] for ch in CHAMBERS)
        for ch in (*CHAMBERS,"blood_pool_sum"):
            av=values["authored_"+ch];tv=values["automatic_"+ch]
            rows.append({"phase_id":phase["phase_id"],"phase_percent":phase["phase_percent"],"region":ch,
                         "authored_ml":av,"automatic_ml":tv,"disagreement_ml":tv-av,
                         "authored_method":phase["authored_method"],"reference_accepted":phase.get("reference_accepted",False),
                         "whole_chamber_confirmed":phase.get("whole_chamber_confirmed",False)})
        provenance.append({"phase_id":phase["phase_id"],"ct_sha256":phase["ct_sha256"],"mask_hashes":hashes})
    baseline={r["region"]:r for r in rows if r["phase_id"]==cfg["baseline_phase"]}
    differences=[]
    for r in rows:
        b=baseline[r["region"]];da=r["authored_ml"]-b["authored_ml"];dt=r["automatic_ml"]-b["automatic_ml"]
        differences.append({"phase_id":r["phase_id"],"phase_percent":r["phase_percent"],"baseline_phase":cfg["baseline_phase"],
                            "region":r["region"],"delta_authored_ml":da,"delta_automatic_ml":dt,
                            "disagreement_delta_ml":dt-da,
                            "change_authored_percent_of_baseline":100*da/b["authored_ml"],
                            "change_automatic_percent_of_baseline":100*dt/b["automatic_ml"]})
    out.mkdir(parents=True)
    for name,data in [("volumes",rows),("phase_differences",differences)]:
        with (out/(name+".csv")).open("w",encoding="utf-8",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary={"status":"exploratory_method_disagreement_not_accuracy_validation","phase_count":len(phases),
             "baseline_phase":cfg["baseline_phase"],"manifest_sha256":sha(manifest),"code_sha256":sha(__file__),
             "reference_accepted_for_all_phases":all(p.get("reference_accepted",False) for p in phases),
             "whole_chamber_confirmed_for_all_phases":all(p.get("whole_chamber_confirmed",False) for p in phases),
             "SV_EF_computed":False,"blood_pool_sum_definition":"Sum of four blood cavities, excludes myocardium; not whole-heart volume",
             "rows":rows,"phase_differences":differences,"provenance":provenance,"limitations":cfg.get("limitations",[])}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=compare(a.manifest,a.output);print(r["status"],r["phase_count"])
