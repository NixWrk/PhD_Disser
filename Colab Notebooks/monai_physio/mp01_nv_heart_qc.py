"""Compare default NV whole-heart output with OLD CT TotalSegmentator heart.

This tests automatic agreement only; no independent outer-heart contour exists.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mp01_old_ct_benchmark import comparison, read_mask, sha256, volume


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("phase_00", "phase_08"), required=True)
    args = parser.parse_args()
    phase = args.phase
    ct_path = args.old_root / "sequence_02" / "phases" / f"{phase}.nii.gz"
    ct = nib.load(str(ct_path))
    spacing = np.linalg.norm(ct.affine[:3, :3], axis=0)
    voxel_ml = abs(float(np.linalg.det(ct.affine[:3, :3]))) / 1000
    nv_dir = HERE / "runs" / f"mp01_old_{phase.replace('phase_', 'phase')}_nvsegment"
    manifest = json.loads((nv_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["ct_sha256"] != sha256(ct_path) or manifest.get("label_prompt") is not None:
        raise ValueError("Expected default CT_BODY output from this exact CT")
    nv = read_mask(nv_dir / "nvsegment_heart.nii.gz", ct)
    ts_path = (args.old_root / "sequence02_ts_comparator_v1" /
               "automatic_masks_v2" / phase / "total_heart" / "heart.nii.gz")
    ts = read_mask(ts_path, ct)
    result = {
        "schema": "mp01_nv_heart_qc_v1",
        "phase": phase,
        "nv_volume_ml": volume(nv, voxel_ml),
        "ts_volume_ml": volume(ts, voxel_ml),
        "nv_vs_ts": comparison(nv, ts, spacing),
        "nv_top_fov_voxels": int(nv[:, :, -1].sum()),
        "ts_top_fov_voxels": int(ts[:, :, -1].sum()),
        "reference_status": "Two automatic masks; no independent accepted outer-heart contour",
    }
    (HERE / f"mp01_nv_heart_{phase}_qc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": phase, "dice": result["nv_vs_ts"]["dice"]}))


if __name__ == "__main__":
    main()
