"""Scan physical closing radii on one rasterization without writing STL files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from solid_fill_stl import keep_largest_components, rasterize_stl


def euclidean_binary_closing(mask: np.ndarray, radius_mm: float, spacing_mm: float) -> np.ndarray:
    """Exact Euclidean-ball closing evaluated with two distance transforms."""
    outside_distance = ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
    dilated = outside_distance <= radius_mm
    del outside_distance
    inside_distance = ndimage.distance_transform_edt(dilated, sampling=spacing_mm)
    closed = inside_distance > radius_mm
    return closed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--spacing-mm", type=float, default=1.0)
    parser.add_argument("--radii-mm", type=float, nargs="+", required=True)
    args = parser.parse_args()

    padding = max(args.radii_mm) + 2 * args.spacing_mm
    mask, _origin = rasterize_stl(args.source, args.spacing_mm, padding)
    voxel_volume = args.spacing_mm**3
    print(f"grid_shape_zyx={mask.shape} raw_volume_l={mask.sum() * voxel_volume / 1e6:.6f}")
    for radius in args.radii_mm:
        closed = euclidean_binary_closing(mask, radius, args.spacing_mm)
        before_fill = int(closed.sum())
        filled = ndimage.binary_fill_holes(closed)
        after_fill = int(filled.sum())
        largest = keep_largest_components(filled, 1)
        final = int(largest.sum())
        components = int(ndimage.label(largest)[1])
        print(
            f"radius_mm={radius:g} before_fill_l={before_fill * voxel_volume / 1e6:.6f} "
            f"after_fill_l={after_fill * voxel_volume / 1e6:.6f} "
            f"largest_l={final * voxel_volume / 1e6:.6f} components={components}"
        )


if __name__ == "__main__":
    main()
