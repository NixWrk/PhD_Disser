"""Evaluate I1--I3 with every available single CT anchor, without refitting masks.

Each anchor is held fixed across both recorded R--R intervals. Target CT
moments remain known: this is a descriptive geometry sensitivity check,
not an electrical or one-image prediction of unobserved moments.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from heart_reduced_geometry import individual_isotropic_candidate, sampled_dice
from heart_affine_geometry import individual_rigid_isotropic_candidate, individual_affine_moment_candidate
from run_heart_method_comparison import get_mask, moments
from run_reduced_heart_geometry import digest, serial, write_json


def run(args):
    cfg=json.loads((args.inputs/"config.local.json").read_text(encoding="utf-8"))
    src=args.inputs/"geometry_comparison.jsonl"
    rows=[json.loads(s) for s in src.read_text(encoding="utf-8").splitlines()]
    if len({r["subject"] for r in rows}) != 1:
        raise ValueError("One subject required")
    source_root=Path(cfg["derived_root"])
    cache={r["phase_id"]:(*get_mask(source_root,r),moments(r)) for r in rows}
    functions={"I1":individual_isotropic_candidate,"I2":individual_rigid_isotropic_candidate,
        "I3":individual_affine_moment_candidate}
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for ref in rows:
        rm,ra,rmo=cache[ref["phase_id"]]
        for target in rows:
            tm,ta,tmo=cache[target["phase_id"]]
            for name,fn in functions.items():
                params=fn(rm,ra,tm,ta,reference_moments=rmo,target_moments=tmo)
                overlap=sampled_dice(tm,ta,params,target_moments=tmo)
                volume_error=float(params["volume_mm3"]/1000-tmo["volume_ml"])
                center_error=float(np.linalg.norm(params["center_mm"]-tmo["centroid_mm"]))
                assert abs(volume_error)<1e-8 and center_error<1e-8
                self_comparison=ref["phase_id"]==target["phase_id"]
                if self_comparison:
                    assert abs(overlap["dice_voxel_center_approx"]-1)<1e-12
                record=dict(method_id=name,reference_phase_id=ref["phase_id"],
                    reference_cycle_index=ref["cycle_index"],target_phase_id=target["phase_id"],
                    target_cycle_index=target["cycle_index"],rr_percent=target["rr_percent"],
                    reference_mask_sha256=ref["provenance"]["sha256"],
                    target_mask_sha256=target["provenance"]["sha256"],
                    self_comparison=self_comparison,dice=overlap["dice_voxel_center_approx"],
                    volume_error_ml=volume_error,center_error_mm=center_error,
                    sampled_volume_error_percent=100*(overlap["candidate_sampled_volume_ml"]/
                        params["volume_ml"]-1),
                    parameters=serial(params))
                records.append(record)
                write_json(args.output/f"{ref['phase_id']}__{target['phase_id']}__{name}.json",record)
        write_json(args.output/"status.json",dict(status="running",completed=len(records),requested=len(rows)**2*3))
        print(ref["phase_id"],len(records),flush=True)
    summaries=[]
    for name in functions:
        for ref in rows:
            selected=[r for r in records if r["method_id"]==name and
                r["reference_phase_id"]==ref["phase_id"] and not r["self_comparison"]]
            dice=np.array([r["dice"] for r in selected])
            summaries.append(dict(method_id=name,reference_phase_id=ref["phase_id"],
                reference_cycle_index=ref["cycle_index"],count=len(selected),
                mean_dice=float(dice.mean()),median_dice=float(np.median(dice)),
                min_dice=float(dice.min()),max_dice=float(dice.max())))
    write_json(args.output/"anchor_comparison.local.json",dict(
        status="all_single_anchor_pairs_completed_not_prediction_validation",
        implementation_sha256=digest(__file__),baseline_sha256=digest(src),
        subject=rows[0]["subject"],phase_count=len(rows),record_count=len(records),
        independent_subject_count=1, summaries=summaries,
        target_moments_known=True,raw_masks_changed=False,
        volume_and_center_errors_are_construction_checks=True,
        source_code_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ("heart_reduced_geometry.py","heart_affine_geometry.py")},
        limits=["one fixed reference per sweep, spanning two R-R intervals",
            "target mask geometry supplied to every transformation",
            "all masks touch the superior CT field boundary",
            "sampling and segmentation uncertainties are not independent anatomical validation"]))
    with (args.output/"anchor_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    write_json(args.output/"status.json",dict(status="completed",completed=len(records),requested=len(rows)**2*3))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())
