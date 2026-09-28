"""Score M1--M9 library geometry against the same OLD CT masks as G/L/I.

One immutable result per phase/method makes interruption and independent review
possible. A valid MATLAB quadric is not treated as anatomical validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from heart_reduced_geometry import mask_moments, sampled_dice
from run_old_approximation_sweep import load_json, load_phase
from run_reduced_heart_geometry import digest, write_json


def run(args):
    config = load_json(args.config)
    matlab = load_json(args.matlab_results)
    prepared = load_json(args.matlab_input)
    if matlab["input_sha256"] != digest(args.matlab_input):
        raise ValueError("MATLAB results do not match exact prepared input")
    if len(matlab["fits"]) != 9 * len(prepared["cases"]):
        raise ValueError("MATLAB fit count does not match prepared cases")
    source = {case["id"]: case for case in prepared["cases"]}
    groups = {}
    for row in matlab["fits"]:
        if row["case_id"] not in source:
            raise ValueError("Unknown MATLAB case")
        groups.setdefault(row["case_id"], []).append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "old_library_fit_scoring_v1",
        "config_sha256": digest(args.config),
        "matlab_input_sha256": digest(args.matlab_input),
        "matlab_results_sha256": digest(args.matlab_results),
        "code_sha256": digest(__file__),
        "target": "union_of_four_chamber_masks_without_great_vessels",
        "overlap_metric": "sampled_dice from heart_reduced_geometry",
        "rfbr_historical_preprocessing_reproduced": False,
    }
    if (output / "protocol.json").exists():
        if load_json(output / "protocol.json") != protocol:
            raise ValueError("Existing output uses another protocol")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output without protocol")
    else:
        write_json(output / "protocol.json", protocol)
    count = failures = 0
    for case in config["cases"]:
        case_id = case["case_id"]
        sequence, masks = Path(case["sequence_dir"]), Path(case["masks_dir"])
        manifest = load_json(sequence / "conversion_manifest.json")
        status_path = masks / "status/run_status.json"
        if not status_path.exists():
            status_path = masks / "run_status.json"
        status = load_json(status_path)
        for phase in sorted(manifest["phases"], key=lambda p: p["phase_percent"]):
            cid = f"{case_id}__{phase['phase_id']}"
            if cid not in groups or len(groups[cid]) != 9:
                raise ValueError(f"Expected nine library methods for {cid}")
            mask, affine, hashes = load_phase(sequence, masks, phase, status)
            m = mask_moments(mask, affine)
            for fit in groups[cid]:
                method = fit["method_id"]
                dest = output / case_id / f"{phase['phase_id']}__{method}.json"
                if dest.exists():
                    record = load_json(dest)
                    if record["input_hashes"] != hashes:
                        raise ValueError(f"Stale mask result: {dest}")
                else:
                    record = {
                        "case_id": case_id, "phase_id": phase["phase_id"],
                        "phase_percent": phase["phase_percent"],
                        "method_id": method, "method_name": fit["method_name"],
                        "input_hashes": hashes,
                        "target_volume_ml": m["volume_ml"],
                        "target_center_ras_mm": m["centroid_mm"].tolist(),
                        "matlab_valid_geometry": fit["valid"],
                        "parameters": {
                            "center_mm": fit["centre_mm"],
                            "semi_axes_mm": fit["axes_mm"],
                            "axes_matrix": fit["orientation_columns"],
                            "volume_mm3": fit["volume_mm3"],
                        } if fit["valid"] else None,
                    }
                    if fit["valid"]:
                        candidate = dict(
                            kind="library_ellipsoid", sampling_family="ellipsoid",
                            **record["parameters"])
                        candidate["volume_ml"] = candidate["volume_mm3"] / 1000
                        try:
                            overlap = sampled_dice(mask, affine, candidate,
                                                   target_moments=m)
                            record["metrics"] = {
                                "dice": overlap["dice_voxel_center_approx"],
                                "volume_error_percent": 100 * (
                                    candidate["volume_mm3"] / m["volume_mm3"] - 1),
                                "center_shift_mm": float(np.linalg.norm(
                                    np.asarray(candidate["center_mm"]) - m["centroid_mm"])),
                                "overlap": overlap,
                            }
                            record["status"] = "scored"
                        except (ValueError, RuntimeError, FloatingPointError, MemoryError) as exc:
                            record.update(status="scoring_failed",
                                          reason=str(exc), metrics=None)
                    else:
                        record.update(status="matlab_invalid_geometry",
                                      metrics=None)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    write_json(dest, record)
                count += 1
                failures += record["metrics"] is None
                write_json(output / "status.json", {
                    "status": "running", "completed": count, "failures": failures,
                    "requested": 9 * len(prepared["cases"]), "last_case": cid,
                    "last_method": method})
                print(cid, method, record["status"], flush=True)
    write_json(output / "status.json", {
        "status": "completed", "completed": count, "failures": failures,
        "requested": 9 * len(prepared["cases"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--matlab-input", required=True)
    parser.add_argument("--matlab-results", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
