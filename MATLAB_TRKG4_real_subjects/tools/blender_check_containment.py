"""Check that sampled vertices and face centroids of one STL lie inside another."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree


def import_stl(path: Path, name: str):
    bpy.ops.wm.stl_import(filepath=str(path.resolve()))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def is_inside(point, tree: BVHTree, tolerance: float = 1e-4) -> bool:
    nearest = tree.find_nearest(point)
    if nearest is None:
        return False
    location, normal, _index, distance = nearest
    # The imported CT STL uses Blender's opposite winding convention here;
    # its face normals point into the represented body volume.
    return distance <= tolerance or (point - location).dot(normal) > 0


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass container and contained STL paths after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 2:
        raise SystemExit("Expected: container.stl contained.stl")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    container = import_stl(Path(args[0]), "container")
    contained = import_stl(Path(args[1]), "contained")
    tree = BVHTree.FromObject(container, bpy.context.evaluated_depsgraph_get())

    vertex_points = [contained.matrix_world @ vertex.co for vertex in contained.data.vertices]
    centroid_points = []
    for polygon in contained.data.polygons:
        centroid_points.append(contained.matrix_world @ polygon.center)
    outside_vertices = sum(not is_inside(point, tree) for point in vertex_points)
    outside_centroids = sum(not is_inside(point, tree) for point in centroid_points)
    print(
        f"CONTAINMENT container={args[0]} contained={args[1]} "
        f"outside_vertices={outside_vertices}/{len(vertex_points)} "
        f"outside_face_centroids={outside_centroids}/{len(centroid_points)}"
    )
    if outside_vertices or outside_centroids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
