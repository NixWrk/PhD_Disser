"""Audit phase T70 orientation using only its segmented lobe anatomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


LOBES = (
    "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for case_dir in args.case_dirs:
        centroids = {}
        volumes = {}
        for lobe in LOBES:
            image = nib.load(case_dir / "seg" / "T70" / f"{lobe}.nii.gz")
            points = np.argwhere(np.asarray(image.dataobj) > 0)
            centroids[lobe] = np.mean(points, axis=0).tolist()
            volumes[lobe] = float(len(points) * abs(np.linalg.det(image.affine[:3, :3])) / 1000)
        upper = (centroids[LOBES[0]][2] + centroids[LOBES[2]][2]) / 2
        lower = (centroids[LOBES[1]][2] + centroids[LOBES[4]][2]) / 2
        left_x = (centroids[LOBES[0]][0] + centroids[LOBES[1]][0]) / 2
        right_x = (centroids[LOBES[2]][0] + centroids[LOBES[4]][0]) / 2
        print(json.dumps({
            "case_dir": str(case_dir),
            "upper_minus_lower_z_voxels": upper - lower,
            "right_minus_left_x_voxels": right_x - left_x,
            "lobes_ml": volumes,
            "source_only_anatomy_qc_pass": upper > lower and right_x > left_x
            and all(volume > 30 for volume in volumes.values()),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
