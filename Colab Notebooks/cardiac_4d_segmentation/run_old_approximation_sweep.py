"""Run every implemented geometric representation on sealed OLD 4D CT masks.

This is a geometry-only comparison. It neither reproduces RFBR's inverse
electrical hodograph nor registers subjects between the two studies.
The local case manifest contains paths and remains outside Git.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from heart_construction_variants import surface_quadrature
from heart_reduced_geometry import fit_reduced_models
from run_heart_method_comparison import METHODS, candidates_for, numeric_metrics
from run_reduced_heart_geometry import digest, serial, write_json


CHAMBERS = ("heart_ventricle_left", "heart_ventricle_right",
            "heart_atrium_left", "heart_atrium_right")
SCRIPT_DIR = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_phase(sequence, masks, phase, status):
    phase_id = phase["phase_id"]
    ct_path = sequence / phase["nifti"]
    if digest(ct_path) != phase["sha256"]:
        raise ValueError(f"CT digest changed: {phase_id}")
    ct = nib.load(str(ct_path))
    if not np.allclose(ct.affine, phase["affine_ras_mm"], atol=1e-5):
        raise ValueError(f"CT affine changed: {phase_id}")
    relative = [f"automatic_masks_v2/{phase_id}/heartchambers_highres/{name}.nii.gz"
                for name in CHAMBERS]
    recorded = {key.replace("\\", "/"): value
                for key, value in status["phase_hashes"][phase_id].items()}
    union = np.zeros(ct.shape, dtype=bool)
    hashes = {}
    for name in relative:
        path = masks / name
        value = digest(path)
        if value != recorded.get(name):
            raise ValueError(f"Segmentation digest changed: {name}")
        image = nib.load(str(path))
        if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, atol=1e-4):
            raise ValueError(f"Mask grid differs from CT: {name}")
        union |= np.asanyarray(image.dataobj) > 0
        hashes[name] = value
    if not union.any():
        raise ValueError(f"Empty chamber union: {phase_id}")
    where = np.where(union)
    lower = np.array([int(np.min(v)) for v in where])
    upper = np.array([int(np.max(v)) + 1 for v in where])
    cropped = union[tuple(slice(lo, hi) for lo, hi in zip(lower, upper))].copy()
    affine = ct.affine.copy()
    affine[:3, 3] = ct.affine[:3, :3] @ lower + ct.affine[:3, 3]
    return cropped, affine, hashes


def base_row(mask, affine):
    fitted = fit_reduced_models(mask, affine)
    return {
        "moments": fitted["moments"],
        "raw_moment_ellipsoid": fitted["ellipsoid_raw_moment"],
        "comparisons": {
            "sphere": {"parameters": fitted["sphere"]},
            "ellipsoid_volume_scaled": {"parameters": fitted["ellipsoid_volume_scaled"]},
        },
    }


def run(args):
    config = load_json(args.config)
    output = Path(args.output)
    selected = METHODS if args.methods == "all" else args.methods.split(",")
    if len(selected) != len(set(selected)) or not set(selected) <= set(METHODS):
        raise ValueError("Methods must be distinct registered method identifiers")
    code_names = ("run_old_approximation_sweep.py", "run_heart_method_comparison.py",
                  "heart_reduced_geometry.py", "heart_construction_variants.py",
                  "heart_additional_sphere_methods.py",
                  "heart_additional_ellipsoid_methods.py", "heart_affine_geometry.py")
    protocol = {
        "schema": "old_approximation_sweep_v1",
        "config_sha256": digest(args.config),
        "methods": selected,
        "max_phases_per_case": args.max_phases_per_case,
        "settings": {name: getattr(args, name) for name in
                     ("fit_points", "eval_points", "surface_budget", "iou_budget",
                      "iou_step")},
        "code_sha256": {name: digest(SCRIPT_DIR / name) for name in code_names},
        "target": "union_of_four_chamber_masks_without_great_vessels",
        "coordinate_frame": "NIfTI RAS+ mm",
        "rfbr_pairing_verified": False,
        "electrical_calculation": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if load_json(protocol_path) != protocol:
            raise ValueError("Output belongs to a different protocol; choose a new directory")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output has no protocol")
    else:
        write_json(protocol_path, protocol)
    cases = config["cases"]
    completed = 0
    failures = 0
    phase_total = 0
    for case in cases:
        case_id = case["case_id"]
        sequence = Path(case["sequence_dir"])
        masks = Path(case["masks_dir"])
        manifest_path = sequence / "conversion_manifest.json"
        status_path = masks / "status/run_status.json"
        if not status_path.exists():
            status_path = masks / "run_status.json"
        manifest, status = load_json(manifest_path), load_json(status_path)
        if status["status"] != "completed":
            raise ValueError(f"Segmentation incomplete: {case_id}")
        phases = sorted(manifest["phases"], key=lambda p: p["phase_percent"])
        if [phase["phase_percent"] for phase in phases] != list(range(0, 100, 5)):
            raise ValueError(f"Expected 20 consecutive 5% phases: {case_id}")
        if args.max_phases_per_case:
            phases = phases[:args.max_phases_per_case]
        phase_total += len(phases)
        refmask = refaffine = refrow = rm = None
        for phase in phases:
            phase_id = phase["phase_id"]
            mask, affine, input_hashes = load_phase(sequence, masks, phase, status)
            row = base_row(mask, affine)
            m = row["moments"]
            if refrow is None:
                refmask, refaffine, refrow, rm = mask, affine, row, m
            points, weights, surface = surface_quadrature(mask, affine)
            for method in selected:
                dest = output / case_id / f"{phase_id}__{method}.json"
                if dest.exists():
                    record = load_json(dest)
                    if (record.get("input_hashes") != input_hashes
                            or record.get("phase_percent") != phase["phase_percent"]):
                        raise ValueError(f"Stale result: {dest}")
                else:
                    start = time.monotonic()
                    record = {
                        "case_id": case_id, "phase_id": phase_id,
                        "phase_percent": phase["phase_percent"],
                        "reference_phase_id": phases[0]["phase_id"],
                        "method_id": method, "input_hashes": input_hashes,
                        "target_volume_ml": m["volume_ml"],
                        "target_center_ras_mm": serial(m["centroid_mm"]),
                        "rfbr_direct_numerical_comparison": False,
                    }
                    try:
                        if method == "T0":
                            record.update(status="reference", parameters=None,
                                          metrics={"dice": 1.0, "volume_error_percent": 0.0,
                                                   "center_shift_mm": 0.0})
                        else:
                            candidate = candidates_for(
                                method, row, refrow, mask, affine, refmask, refaffine,
                                m, rm, points, weights, surface, args)
                            detail = candidate if "parameters" in candidate else {}
                            params = detail.get("parameters") if detail else candidate
                            record["status"] = detail.get(
                                "status", candidate.get("status", "computed"))
                            record["fit"] = serial(detail)
                            record["parameters"] = serial(params)
                            record["metrics"] = (numeric_metrics(
                                mask, affine, m, params, points, weights)
                                if params is not None else None)
                    except (ValueError, RuntimeError, FloatingPointError, MemoryError) as exc:
                        record.update(status="failed", reason=str(exc),
                                      parameters=None, metrics=None)
                    record["elapsed_seconds"] = time.monotonic() - start
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    write_json(dest, record)
                completed += 1
                failures += record.get("metrics") is None
                write_json(output / "status.json", {
                    "status": "running", "completed": completed,
                    "failures": failures, "last_case": case_id,
                    "last_phase": phase_id, "last_method": method,
                    "requested": len(cases) * len(phases) * len(selected),
                })
                print(case_id, phase_id, method, record["status"],
                      round(record.get("elapsed_seconds", 0), 2), flush=True)
    write_json(output / "status.json", {
        "status": "completed", "completed": completed, "failures": failures,
        "phase_count": phase_total, "method_count": len(selected),
        "rfbr_direct_numerical_comparison": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--max-phases-per-case", type=int)
    parser.add_argument("--fit-points", type=int, default=1024)
    parser.add_argument("--eval-points", type=int, default=8192)
    parser.add_argument("--surface-budget", type=int, default=1800)
    parser.add_argument("--iou-budget", type=int, default=900)
    parser.add_argument("--iou-step", type=float, default=3.0)
    run(parser.parse_args())
