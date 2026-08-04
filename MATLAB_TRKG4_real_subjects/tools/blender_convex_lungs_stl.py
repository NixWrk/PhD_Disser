"""Build separate filled convex hulls for the left and right lungs.

The source segmentation may contain bronchi, vessels, tunnels or a bridge
between the lungs.  Splitting its point cloud at the sagittal mid-plane and
building one convex hull per side produces two closed solid masks suitable for
an effective lung-tissue-plus-air conductivity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector


def make_hull(name: str, coordinates: list[Vector]):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    vertices = [bm.verts.new(coordinate) for coordinate in coordinates]
    bm.verts.ensure_lookup_table()
    result = bmesh.ops.convex_hull(bm, input=vertices, use_existing_faces=False)
    unused = [item for item in result.get("geom_unused", []) if isinstance(item, bmesh.types.BMVert)]
    if unused:
        bmesh.ops.delete(bm, geom=unused, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass input, output and split X after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 3:
        raise SystemExit("Expected: input.stl output.stl split_x_mm")

    source_path = Path(args[0]).resolve()
    target_path = Path(args[1]).resolve()
    split_x = float(args[2])
    target_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.stl_import(filepath=str(source_path))
    source = bpy.context.active_object
    coordinates = [source.matrix_world @ vertex.co for vertex in source.data.vertices]
    left_coordinates = [coordinate for coordinate in coordinates if coordinate.x < split_x]
    right_coordinates = [coordinate for coordinate in coordinates if coordinate.x >= split_x]
    if len(left_coordinates) < 4 or len(right_coordinates) < 4:
        raise RuntimeError("The split plane did not produce two valid point clouds")

    left = make_hull("left_lung_solid_convex", left_coordinates)
    right = make_hull("right_lung_solid_convex", right_coordinates)
    bpy.data.objects.remove(source, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    left.select_set(True)
    right.select_set(True)
    bpy.context.view_layer.objects.active = left
    bpy.ops.wm.stl_export(
        filepath=str(target_path), export_selected_objects=True, ascii_format=False
    )
    print(
        f"CONVEX_LUNGS source={source_path} target={target_path} split_x_mm={split_x} "
        f"left_input_vertices={len(left_coordinates):,} right_input_vertices={len(right_coordinates):,} "
        f"left_faces={len(left.data.polygons):,} right_faces={len(right.data.polygons):,}"
    )


if __name__ == "__main__":
    main()
