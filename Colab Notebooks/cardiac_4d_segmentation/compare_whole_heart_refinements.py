"""Compare complete phase sets before/after declared numerical refinements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_reduced_heart_geometry import digest, write_json


def read_records(root):
    rows={}
    for p in root.glob("nix_*.json"):
        v=json.loads(p.read_text(encoding="utf-8"))
        key=v["subject"],v["phase_id"],v["method_id"]
        if key in rows:raise ValueError("Duplicate refinement record")
        rows[key]=(v,p)
    return rows


def run(args):
    pairs=[("sphere_grid", "project_sphere_overlap", "refine_sphere_overlap"),
        ("ellipsoid_grid", "project_ellipsoid_overlap_retry", "refine_ellipsoid_overlap"),
        ("surface_points", "project_surface", "refine_surface"),
        ("library_points", "library/fits_8192/scoring", "library/fits_32768/scoring")]
    rows=[];sources={}
    for label,old,new in pairs:
        first,second=read_records(args.root/old),read_records(args.root/new)
        if not first or set(first)!=set(second):raise ValueError("Different phase/method sets: "+label)
        for key,(a,ap) in first.items():
            b,bp=second[key]
            assert a["target_mask_sha256"]==b["target_mask_sha256"]
            sources[str(ap)]=digest(ap);sources[str(bp)]=digest(bp)
            r=dict(refinement=label,subject=key[0],phase_id=key[1],method_id=key[2],
                cycle_index=a["cycle_index"],is_reference_phase=a["is_reference_phase"],
                first_status=a["status"],second_status=b["status"])
            if a.get("metrics") is None or b.get("metrics") is None:
                r["status"]="unavailable_pair"
            else:
                x,y=a["parameters"],b["parameters"]
                r.update(status="paired",volume_difference_ml=(y["volume_mm3"]-x["volume_mm3"])/1000,
                    center_difference_mm=float(np.linalg.norm(np.asarray(y["center_mm"])-x["center_mm"])),
                    dice_difference=b["metrics"]["dice"]-a["metrics"]["dice"],
                    full_surface_rms_difference_mm=b["metrics"]["target_surface_to_candidate_RMS_mm"]-
                        a["metrics"]["target_surface_to_candidate_RMS_mm"])
            rows.append(r)
    summaries=[]
    for method in sorted({r["method_id"] for r in rows}):
        selected=[r for r in rows if r["method_id"]==method]
        valid=[r for r in selected if r["status"]=="paired"]
        s=dict(method_id=method,refinement=selected[0]["refinement"],requested=len(selected),paired=len(valid))
        for name in ("volume_difference_ml","center_difference_mm","dice_difference","full_surface_rms_difference_mm"):
            values=np.array([r[name] for r in valid])
            s["max_abs_"+name]=float(np.max(np.abs(values))) if len(values) else None
            s["rms_"+name]=float(np.sqrt(np.mean(values**2))) if len(values) else None
        summaries.append(s)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in (("paired_differences.csv",rows),("summary.csv",summaries)):
        keys=list(dict.fromkeys(k for row in data for k in row))
        with (args.output/name).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(data)
    write_json(args.output/"refinements.local.json",dict(status="numerical_sensitivity_completed",
        implementation_sha256=digest(__file__),source_hashes=sources,rows=rows,summaries=summaries,
        limits=["same anatomy and algorithms; changed numerical density/budget only",
            "pairwise differences are not an anatomical uncertainty bound",
            "close method order may remain unresolved",
            "input status retained independently of metric availability"]))
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())
