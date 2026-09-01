#!/usr/bin/env python
"""Render de-identified orthogonal CT/mask overlays for manual QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def oriented(array: np.ndarray) -> np.ndarray:
    return np.rot90(array)


def overlay(axis, ct, blood, myocardium, title):
    axis.imshow(oriented(ct), cmap="gray", vmin=-150, vmax=650)
    axis.imshow(
        np.ma.masked_where(~oriented(blood), oriented(blood)),
        cmap="Blues",
        alpha=0.42,
        vmin=0,
        vmax=1,
    )
    axis.imshow(
        np.ma.masked_where(~oriented(myocardium), oriented(myocardium)),
        cmap="Reds",
        alpha=0.48,
        vmin=0,
        vmax=1,
    )
    axis.set_title(title)
    axis.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--blood-pool", required=True)
    parser.add_argument("--myocardium", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ct_image = nib.load(args.ct)
    blood_image = nib.load(args.blood_pool)
    myocardium_image = nib.load(args.myocardium)
    if ct_image.shape != blood_image.shape or ct_image.shape != myocardium_image.shape:
        raise ValueError("CT and masks must have the same grid")
    if not np.allclose(ct_image.affine, blood_image.affine, atol=1e-4) or not np.allclose(
        ct_image.affine, myocardium_image.affine, atol=1e-4
    ):
        raise ValueError("CT and mask affines differ")

    ct = np.asanyarray(ct_image.dataobj)
    blood = np.asanyarray(blood_image.dataobj) > 0
    myocardium = np.asanyarray(myocardium_image.dataobj) > 0
    combined = blood | myocardium
    coords = np.argwhere(combined)
    if not len(coords):
        raise ValueError("Masks are empty")
    center = np.rint(coords.mean(axis=0)).astype(int)
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    axial_indices = np.rint(np.linspace(minimum[2], maximum[2], 6)).astype(int)

    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for axis, index in zip(axes.flat, axial_indices):
        overlay(
            axis,
            ct[:, :, index],
            blood[:, :, index],
            myocardium[:, :, index],
            f"axial k={index}",
        )
    figure.suptitle("Blood pool (blue) / myocardium (red); automatic, pending review")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)

    orthogonal_output = output.with_name(output.stem + "_orthogonal" + output.suffix)
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    overlay(
        axes[0],
        ct[:, :, center[2]],
        blood[:, :, center[2]],
        myocardium[:, :, center[2]],
        f"axial k={center[2]}",
    )
    overlay(
        axes[1],
        ct[:, center[1], :],
        blood[:, center[1], :],
        myocardium[:, center[1], :],
        f"coronal j={center[1]}",
    )
    overlay(
        axes[2],
        ct[center[0], :, :],
        blood[center[0], :, :],
        myocardium[center[0], :, :],
        f"sagittal i={center[0]}",
    )
    figure.suptitle("Automatic segmentation geometry check")
    figure.savefig(orthogonal_output, dpi=150, facecolor="white")
    plt.close(figure)
    print(output)
    print(orthogonal_output)


if __name__ == "__main__":
    main()

