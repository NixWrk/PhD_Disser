"""Export area-sampled OLD CT chamber surfaces for independent library fits.

The output remains local. The RFBR contour filling is not reproduced here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from heart_additional_sphere_methods import area_systematic_sample
from heart_construction_variants import surface_quadrature
from run_old_approximation_sweep import load_json, load_phase
from run_reduced_heart_geometry import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=8192)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for case in load_json(args.config)["cases"]:
        sequence = Path(case["sequence_dir"])
        masks = Path(case["masks_dir"])
        manifest_path = sequence / "conversion_manifest.json"
        status_path = masks / "status/run_status.json"
        if not status_path.exists():
            status_path = masks / "run_status.json"
        manifest = load_json(manifest_path)
        status = load_json(status_path)
        if status["status"] != "completed":
            raise ValueError(f"Segmentation incomplete: {case['case_id']}")
        phases = sorted(manifest["phases"], key=lambda p: p["phase_percent"])
        if [p["phase_percent"] for p in phases] != list(range(0, 100, 5)):
            raise ValueError(f"Expected 20 consecutive phases: {case['case_id']}")
        for phase in phases:
            mask, affine, hashes = load_phase(sequence, masks, phase, status)
            points, weights, surface = surface_quadrature(mask, affine)
            sampled, sampled_weights, sampling = area_systematic_sample(
                points, weights, args.sample_count)
            cases.append({
                "id": f"{case['case_id']}__{phase['phase_id']}",
                "points_mm": sampled.tolist(),
                "weights": sampled_weights.tolist(),
                "source_metadata": {
                    "case_id": case["case_id"],
                    "phase_id": phase["phase_id"],
                    "phase_percent": phase["phase_percent"],
                    "region": "union_of_four_chamber_masks_without_great_vessels",
                    "coordinate_system": "NIfTI RAS+ mm",
                    "mask_hashes": hashes,
                    "ct_sha256": phase["sha256"],
                    "conversion_manifest_sha256": digest(manifest_path),
                    "segmentation_status_sha256": digest(status_path),
                    "surface": surface,
                    "sampling": sampling,
                    "rfbr_septal_filling_reproduced": False,
                },
            })
            print(case["case_id"], phase["phase_id"], len(sampled), flush=True)
    write_json(output, {
        "units": "mm",
        "schema": "old_chamber_surface_clouds_v1",
        "config_sha256": digest(args.config),
        "cases": cases,
    })
    print("completed", len(cases), output, flush=True)


if __name__ == "__main__":
    main()
