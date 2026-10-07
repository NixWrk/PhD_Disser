"""Segment selected DIR-Lab 4DCT phases with frozen TotalSegmentator v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator


NAMES = (
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right", "heart",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--phases", nargs="+", required=True)
    args = parser.parse_args()
    for phase in args.phases:
        ct_path = args.case_dir / f"ct_{phase}.nii.gz"
        seg_dir = args.case_dir / "seg" / phase
        seg_dir.mkdir(parents=True, exist_ok=True)
        if not all((seg_dir / f"{name}.nii.gz").is_file() for name in NAMES):
            print(f"{phase}: running TotalSegmentator v2", flush=True)
            totalsegmentator(
                input=ct_path, output=seg_dir, task="total", roi_subset=list(NAMES),
                fast=False, device="gpu", quiet=True,
            )
        ct = nib.load(ct_path)
        masks = {}
        for name in NAMES:
            path = seg_dir / f"{name}.nii.gz"
            img = nib.load(path)
            if img.shape != ct.shape or not np.allclose(img.affine, ct.affine, atol=1e-4):
                raise ValueError(f"{phase} {name}: grid mismatch")
            voxels = np.asanyarray(img.dataobj) > 0
            faces = [
                [int(np.count_nonzero(np.take(voxels, k, axis=axis)))
                 for k in (0, voxels.shape[axis] - 1)]
                for axis in range(3)
            ]
            masks[name] = {
                "voxels": int(np.count_nonzero(voxels)),
                "volume_ml": float(np.count_nonzero(voxels) * abs(np.linalg.det(img.affine[:3, :3])) / 1000),
                "face_voxels_low_high": faces,
                "touches_ct_boundary": any(v > 0 for pair in faces for v in pair),
                "mask_sha256": sha256(path),
            }
        manifest = {
            "schema": "mp10_dirlab_4dct_segmentation_v1",
            "phase": phase,
            "ct_sha256": sha256(ct_path),
            "segmenter": "TotalSegmentator v2 total task full resolution",
            "totalsegmentator_version": version("TotalSegmentator"),
            "masks": masks,
        }
        (seg_dir / "segmentation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{phase}: lung volume {sum(masks[n]['volume_ml'] for n in NAMES[:5]):.1f} ml; "
              f"boundary contact {any(masks[n]['touches_ct_boundary'] for n in NAMES[:5])}", flush=True)


if __name__ == "__main__":
    main()
