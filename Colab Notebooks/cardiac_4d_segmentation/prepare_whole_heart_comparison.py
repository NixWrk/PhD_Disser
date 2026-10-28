"""Prepare a verified subject subset and area-sampled clouds for existing fitters.

No segmentation, registration, FEM or phase interpolation is performed.
Local paths are supplied by the caller; raw medical inputs remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heart_reduced_geometry import mask_moments
from heart_construction_variants import surface_quadrature
from heart_additional_sphere_methods import area_systematic_sample
from run_heart_method_comparison import get_mask
from run_reduced_heart_geometry import digest, serial, write_json


def run(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    baseline = args.baseline / "geometry_comparison.jsonl"
    rows = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["subject"] == args.subject and r["region"] == "whole_heart"]
    if not rows or len({r["phase_id"] for r in rows}) != len(rows):
        raise ValueError("Expected distinct phases of one subject's whole heart")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    root = Path(cfg["derived_root"])
    clouds = {8192: [], 32768: []}
    audit = []
    for row in rows:
        mask, affine = get_mask(root, row)  # Checks CT, affine, grid and mask hashes.
        current = mask_moments(mask, affine)
        for field in ("volume_mm3", "centroid_mm", "covariance_mm2"):
            if not np.allclose(current[field], row["moments"][field], rtol=1e-10, atol=1e-8):
                raise ValueError("Changed geometric definition: " + row["phase_id"] + "/" + field)
        points, weights, surface = surface_quadrature(mask, affine)
        cid = f"{args.subject}_{row['phase_id']}_whole_heart"
        metadata = dict(subject=args.subject, phase_id=row["phase_id"],
            region="whole_heart", cycle_index=row["cycle_index"],
            reference_phase_id=row["reference_phase_id"],
            mask_sha256=row["provenance"]["sha256"], ct_sha256=row["ct_sha256"],
            target_touches_CT_boundary=row["provenance"]["touches_source_image_boundary"],
            coordinate_system="NIfTI RAS+ mm", surface=surface)
        for count in clouds:
            p, w, sampling = area_systematic_sample(points, weights, count)
            clouds[count].append(dict(id=cid, points_mm=p, weights=w,
                source_metadata=dict(metadata, sampling=sampling)))
        audit.append(dict(metadata, moments_recomputed=True,
            target_volume_ml=current["volume_ml"], target_center_mm=current["centroid_mm"]))
        print(row["phase_id"], "hash/grid/moments passed", flush=True)
    for cycle in sorted({r["cycle_index"] for r in rows}):
        same = [r for r in rows if r["cycle_index"] == cycle]
        refs = [r for r in same if r["is_reference_phase"]]
        if len(refs) != 1 or any(r["reference_phase_id"] != refs[0]["phase_id"] for r in same):
            raise ValueError("Inconsistent within-cycle reference")
    (out / "geometry_comparison.jsonl").write_text(
        "".join(json.dumps(serial(r), ensure_ascii=False, allow_nan=False)+"\n" for r in rows),
        encoding="utf-8")
    cfg.update(subjects=[args.subject], output_dir=str(out.resolve()))
    write_json(out / "config.local.json", cfg)
    for count, cases in clouds.items():
        name = "library_clouds.json" if count == 8192 else "library_clouds_32768.json"
        write_json(out / name, dict(units="mm", cases=cases))
    write_json(out / "input_audit.local.json", dict(
        status="verified_existing_whole_heart_inputs_not_anatomical_validation",
        subject=args.subject, phase_count=len(rows),
        source_baseline=str(baseline.resolve()), baseline_sha256=digest(baseline),
        source_config_sha256=digest(args.config), implementation_sha256=digest(__file__),
        source_masks_modified=False, all_phases_touch_CT_boundary=all(r["target_touches_CT_boundary"] for r in audit),
        phases=audit, artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--subject", required=True)
    run(p.parse_args())
