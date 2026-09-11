#!/usr/bin/env python
"""Render orthogonal CT overlays for grouped static anatomy products."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PRODUCTS = (
    ("bones_in_scan_fov", "Bones", "autumn", (-200, 1400)),
    ("lungs", "Lungs", "winter", (-1000, 300)),
    ("body_in_scan_fov", "Body in scan FOV", "Greens", (-150, 500)),
    ("skeletal_muscle_in_scan_fov", "Skeletal muscle", "Reds", (-150, 250)),
    ("fat_in_scan_fov", "Fat", "Oranges", (-200, 200)),
    ("whole_heart", "Whole heart", "Purples", (-150, 650)),
)


def oriented(array: np.ndarray) -> np.ndarray:
    return np.rot90(array)


def panel(axis, ct: np.ndarray, mask: np.ndarray, title: str, cmap: str, window: tuple[int, int]) -> None:
    axis.imshow(oriented(ct), cmap="gray", vmin=window[0], vmax=window[1])
    axis.imshow(
        np.ma.masked_where(~oriented(mask), oriented(mask)),
        cmap=cmap,
        alpha=0.38,
        vmin=0,
        vmax=1,
    )
    axis.set_title(title)
    axis.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ct_image = nib.load(str(run_dir / "input" / "phase_00.nii.gz"))
    ct = np.asanyarray(ct_image.dataobj)
    paths = {
        "bones_in_scan_fov": run_dir / "anatomy" / "products" / "bones_in_scan_fov.nii.gz",
        "lungs": run_dir / "anatomy" / "products" / "lungs.nii.gz",
        "body_in_scan_fov": run_dir / "anatomy" / "products" / "body_in_scan_fov.nii.gz",
        "skeletal_muscle_in_scan_fov": run_dir / "anatomy" / "products" / "skeletal_muscle_in_scan_fov.nii.gz",
        "fat_in_scan_fov": run_dir / "anatomy" / "products" / "fat_in_scan_fov.nii.gz",
        "whole_heart": run_dir / "cardiac_products" / "phase_00" / "whole_heart.nii.gz",
    }

    figure, axes = plt.subplots(len(PRODUCTS), 3, figsize=(15, 27), constrained_layout=True)
    for row, (key, label, cmap, window) in enumerate(PRODUCTS):
        image = nib.load(str(paths[key]))
        if image.shape != ct_image.shape or not np.allclose(image.affine, ct_image.affine, atol=1e-4):
            raise ValueError(f"Grid mismatch: {key}")
        mask = np.asanyarray(image.dataobj) > 0
        coords = np.argwhere(mask)
        if not len(coords):
            raise ValueError(f"Empty mask: {key}")
        center = np.rint(coords.mean(axis=0)).astype(int)
        panel(axes[row, 0], ct[:, :, center[2]], mask[:, :, center[2]], f"{label}: axial", cmap, window)
        panel(axes[row, 1], ct[:, center[1], :], mask[:, center[1], :], f"{label}: coronal", cmap, window)
        panel(axes[row, 2], ct[center[0], :, :], mask[center[0], :, :], f"{label}: sagittal", cmap, window)
    figure.suptitle("Static automatic segmentation: orthogonal QC, pending expert review", fontsize=16)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140, facecolor="white")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
