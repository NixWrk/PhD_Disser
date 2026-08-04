"""Measure candidate-vertex distances to a reference STL surface."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree


def load(path: Path):
    bpy.ops.wm.stl_import(filepath=str(path.resolve()))
    return bpy.context.active_object


def percentile(sorted_values, fraction):
    if not sorted_values:
        return math.nan
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass reference and candidate STL paths after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 2:
        raise SystemExit("Expected: reference.stl candidate.stl")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    reference = load(Path(args[0]))
    candidate = load(Path(args[1]))

    vertices = [reference.matrix_world @ vertex.co for vertex in reference.data.vertices]
    polygons = [tuple(polygon.vertices) for polygon in reference.data.polygons]
    tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=True)

    distances = []
    for vertex in candidate.data.vertices:
        point = candidate.matrix_world @ vertex.co
        nearest = tree.find_nearest(point)
        if nearest is not None:
            distances.append(nearest[3])
    distances.sort()
    mean = sum(distances) / len(distances)
    print(f"SURFACE_DISTANCE reference={args[0]} candidate={args[1]}")
    print(f"  samples={len(distances):,}")
    print(f"  mean_mm={mean:.6f}")
    print(f"  p50_mm={percentile(distances, 0.50):.6f}")
    print(f"  p95_mm={percentile(distances, 0.95):.6f}")
    print(f"  p99_mm={percentile(distances, 0.99):.6f}")
    print(f"  max_mm={distances[-1]:.6f}")


if __name__ == "__main__":
    main()
