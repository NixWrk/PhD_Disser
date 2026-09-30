"""Compare all 20 OLD CT phases with project TotalSegmentator output.

This is agreement between automatic pipelines, not independent anatomy truth.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mp01_old_ct_benchmark import CLASSES, read_mask


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads((HERE / "mp01_old_full_cycle.json").read_text(encoding="utf-8"))
    if runs["phase_count"] != 20:
        raise ValueError("Full 20-phase run required")
    records = []
    for entry in runs["outputs"]:
        phase = entry["phase"]
        ct = nib.load(str(args.old_root / "sequence_02" / "phases" / f"{phase}.nii.gz"))
        voxel_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000
        run_dir = HERE / "runs" / entry["local_run_name"]
        ts_dir = (args.old_root / "sequence02_ts_comparator_v1" /
                  "automatic_masks_v2" / phase / "heartchambers_highres")
        classes = {}
        for name, (monai_name, ts_name, _) in CLASSES.items():
            monai = read_mask(run_dir / f"mp01_02_{monai_name}.nii.gz", ct)
            ts = read_mask(ts_dir / ts_name, ct)
            n_m, n_t = int(monai.sum()), int(ts.sum())
            overlap = int(np.count_nonzero(monai & ts))
            classes[name] = {
                "monai_volume_ml": n_m * voxel_ml,
                "project_ts_volume_ml": n_t * voxel_ml,
                "monai_minus_ts_ml": (n_m - n_t) * voxel_ml,
                "dice_monai_vs_ts": 2 * overlap / (n_m + n_t),
                "monai_top_fov_voxels": int(monai[:, :, -1].sum()),
                "project_ts_top_fov_voxels": int(ts[:, :, -1].sum()),
            }
        records.append({"phase": phase, "classes": classes})
        print(f"{phase}: QC complete", flush=True)
    summary = {}
    dynamics = {}
    for name in CLASSES:
        series = [record["classes"][name] for record in records]
        dice = [item["dice_monai_vs_ts"] for item in series]
        volume_diff = [item["monai_minus_ts_ml"] for item in series]
        summary[name] = {
            "dice_min": min(dice), "dice_mean": float(np.mean(dice)), "dice_max": max(dice),
            "max_abs_volume_difference_ml": max(abs(x) for x in volume_diff),
            "monai_phases_touching_top_fov": sum(item["monai_top_fov_voxels"] > 0 for item in series),
            "project_ts_phases_touching_top_fov": sum(item["project_ts_top_fov_voxels"] > 0 for item in series),
        }
        dynamics[name] = {
            "max_abs_difference_of_phase00_referenced_changes_ml": max(
                abs((item["monai_volume_ml"] - series[0]["monai_volume_ml"])
                    - (item["project_ts_volume_ml"] - series[0]["project_ts_volume_ml"]))
                for item in series),
        }
    result = {
        "schema": "mp01_old_cycle_qc_v1",
        "case_alias": "historical_01_sequence_02",
        "phase_count": 20,
        "reference_semantics": "Project TotalSegmentator v2 is another automatic run, not independent ground truth",
        "per_phase": records,
        "summary": summary,
        "volume_dynamics": dynamics,
        "independent_manual_validation": False,
        "decision": "technical_agreement_only",
    }
    (HERE / "mp01_old_cycle_qc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
