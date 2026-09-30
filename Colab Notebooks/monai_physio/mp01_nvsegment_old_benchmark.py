"""Compare the separate pretrained NV-Segment-CTMR model on OLD CT phases.

Direct contours are unblinded candidate annotations, not independent truth.
Only aggregate, path-free numbers are written to the tracked JSON output.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mp01_old_ct_benchmark import CLASSES, comparison, read_mask, sha256, volume


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "mp01_nvsegment_old_benchmark.json")
    args = parser.parse_args()
    phases = {}
    for phase in ("phase_00", "phase_08"):
        ct_path = args.old_root / "sequence_02" / "phases" / f"{phase}.nii.gz"
        ct = nib.load(str(ct_path))
        spacing = np.linalg.norm(ct.affine[:3, :3], axis=0)
        voxel_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000
        nv_dir = HERE / "runs" / f"mp01_old_{phase.replace('phase_', 'phase')}_nvsegment_prompted"
        nv_manifest = json.loads((nv_dir / "manifest.json").read_text(encoding="utf-8"))
        if (nv_manifest["ct_sha256"] != sha256(ct_path) or
                nv_manifest["label_prompt"] != [149, 151, 152, 153, 154]):
            raise ValueError("NV model output has different CT provenance")
        ts_dir = (args.old_root / "sequence02_ts_comparator_v1" /
                  "automatic_masks_v2" / phase / "heartchambers_highres")
        direct_dir = args.old_root / "anatomical_audit_v02" / "direct_v01" / phase
        classes = {}
        for name, (_, ts_name, direct_name) in CLASSES.items():
            nv_name = "left_ventricle_myocardium" if name == "myocardium" else name
            nv = read_mask(nv_dir / f"nvsegment_{nv_name}.nii.gz", ct)
            ts = read_mask(ts_dir / ts_name, ct)
            record = {
                "nv_volume_ml": volume(nv, voxel_ml),
                "project_ts_volume_ml": volume(ts, voxel_ml),
                "nv_vs_project_ts": comparison(nv, ts, spacing),
                "nv_top_fov_voxels": int(nv[:, :, -1].sum()),
            }
            if direct_name:
                direct = read_mask(direct_dir / direct_name, ct)
                record["candidate_direct_volume_ml"] = volume(direct, voxel_ml)
                record["nv_vs_candidate_direct"] = comparison(nv, direct, spacing)
                record["ts_vs_candidate_direct"] = comparison(ts, direct, spacing)
            classes[name] = record
        phases[phase] = {
            "ct_sha256": nv_manifest["ct_sha256"],
            "nv_labelmap_sha256": nv_manifest["labelmap_sha256"],
            "classes": classes,
        }
        print(f"{phase}: compared", flush=True)
    changes = {}
    for name in CLASSES:
        early = phases["phase_00"]["classes"][name]
        late = phases["phase_08"]["classes"][name]
        changes[name] = {
            "nv_delta_ml": late["nv_volume_ml"] - early["nv_volume_ml"],
            "project_ts_delta_ml": late["project_ts_volume_ml"] - early["project_ts_volume_ml"],
        }
        if "candidate_direct_volume_ml" in early:
            changes[name]["candidate_direct_delta_ml"] = (
                late["candidate_direct_volume_ml"] - early["candidate_direct_volume_ml"]
            )
    result = {
        "schema": "mp01_nvsegment_old_benchmark_v1",
        "method": "Pretrained NVIDIA NV-Segment-CTMR with explicit chamber/myocardium prompts via MONAI Physio on OLD CT phases 0% and 40%",
        "reference_status": "Direct contours are unblinded, unaccepted candidate annotations",
        "phases": phases,
        "volume_changes_00_to_08": changes,
        "decision": "not_admitted_as_replacement_for_quantitative_chamber_dynamics_on_this_case",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
