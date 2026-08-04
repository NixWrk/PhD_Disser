"""Print topology diagnostics for one or more STL triangle meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def analyze(path: Path) -> None:
    mesh = pv.read(path).clean().triangulate()
    print(f"\n{path}")
    print(f"  vertices: {mesh.n_points:,}")
    print(f"  faces: {mesh.n_cells:,}")
    print(f"  bounds mm: {np.array2string(np.asarray(mesh.bounds).reshape(3, 2), precision=3)}")
    print(f"  extents mm: {np.array2string(np.asarray(mesh.length), precision=3)} diagonal")
    print(f"  open/non-manifold edges: {mesh.n_open_edges:,}")
    print(f"  manifold: {mesh.is_manifold}")
    try:
        print(f"  volume: {mesh.volume:,.3f} mm^3")
    except Exception as exc:
        print(f"  volume: unavailable ({exc})")

    components = list(mesh.split_bodies(label=False))
    components.sort(key=lambda item: item.n_cells, reverse=True)
    print(f"  connected components: {len(components)}")
    for i, component in enumerate(components[:20], 1):
        surface = component.extract_surface().clean().triangulate()
        try:
            volume = surface.volume
        except Exception:
            volume = float("nan")
        bounds = np.asarray(surface.bounds).reshape(3, 2)
        print(
            f"    {i:2d}: faces={surface.n_cells:,}, vertices={surface.n_points:,}, "
            f"open_edges={surface.n_open_edges:,}, manifold={surface.is_manifold}, "
            f"volume={volume:,.3f}, bounds={np.array2string(bounds, precision=2)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        analyze(path)


if __name__ == "__main__":
    main()
