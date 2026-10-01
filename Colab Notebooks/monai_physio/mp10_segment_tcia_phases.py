"""Segment fixed TCIA respiratory frames with the same frozen TotalSegmentator v2."""

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
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    "heart",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=["g070", "g000", "g050", "g090"])
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for phase in args.phases:
        ct_path = args.case_dir / f"100_HM10395_{phase}.nii.gz"
        seg_dir = args.output_root / phase
        seg_dir.mkdir(exist_ok=True)
        ready = all((seg_dir / f"{name}.nii.gz").is_file() for name in NAMES)
        print(f"{phase}: {'checking saved masks' if ready else 'running TotalSegmentator v2'}", flush=True)
        if not ready:
            totalsegmentator(
                input=ct_path, output=seg_dir, task="total", roi_subset=list(NAMES),
                fast=False, device="gpu", quiet=True,
            )
        ct = nib.load(ct_path)
        manifest = {
            "phase": phase,
            "ct_sha256": sha256(ct_path),
            "segmentation_model": "TotalSegmentator v2",
            "totalsegmentator_version": version("TotalSegmentator"),
            "roi_subset": list(NAMES),
            "masks": {},
        }
        for name in NAMES:
            path = seg_dir / f"{name}.nii.gz"
            mask = nib.load(path)
            if mask.shape != ct.shape or not np.allclose(mask.affine, ct.affine, atol=1e-4):
                raise ValueError(f"{name} grid does not match CT for {phase}")
            count = int(np.count_nonzero(np.asarray(mask.dataobj)))
            if count == 0:
                raise ValueError(f"{name} mask empty for {phase}")
            manifest["masks"][name] = {"voxels": count, "sha256": sha256(path)}
        (seg_dir / "segmentation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{phase}: masks verified", flush=True)


if __name__ == "__main__":
    main()
