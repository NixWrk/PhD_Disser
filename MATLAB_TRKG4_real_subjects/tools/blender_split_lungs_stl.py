"""Split a joined lung surface into closed left/right solids.

The small medial gap prevents coincident cap faces when both solids are exported
to one STL file.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy


def keep_largest_face_component(bm):
    remaining = set(bm.faces)
    components = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component = {seed}
        while queue:
            face = queue.popleft()
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
        components.append(component)
    if not components:
        return
    keep_faces = max(components, key=len)
    keep_verts = {vert for face in keep_faces for vert in face.verts}
    remove_verts = [vert for vert in bm.verts if vert not in keep_verts]
    if remove_verts:
        bmesh.ops.delete(bm, geom=remove_verts, context="VERTS")


def bisect_half(source, cut_x, keep_left, name):
    obj = source.copy()
    obj.data = source.data.copy()
    bpy.context.collection.objects.link(obj)
    obj.name = name

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    geometry = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(
        bm,
        geom=geometry,
        dist=1e-5,
        plane_co=(cut_x, 0.0, 0.0),
        plane_no=(1.0, 0.0, 0.0),
        clear_inner=not keep_left,
        clear_outer=keep_left,
    )
    boundary = [edge for edge in bm.edges if len(edge.link_faces) == 1]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    keep_largest_face_component(bm)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    print(
        f"HALF_RESULT name={name} vertices={len(obj.data.vertices):,} "
        f"faces={len(obj.data.polygons):,}"
    )
    return obj


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass input, output, split X, gap and voxel size after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 5:
        raise SystemExit("Expected: input.stl output.stl split_x_mm gap_mm voxel_size_mm")

    source_path = Path(args[0]).resolve()
    target_path = Path(args[1]).resolve()
    split_x = float(args[2])
    gap = float(args[3])
    voxel_size = float(args[4])
    target_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.stl_import(filepath=str(source_path))
    source = bpy.context.active_object
    left = bisect_half(source, split_x - gap / 2, True, "left_lung")
    right = bisect_half(source, split_x + gap / 2, False, "right_lung")
    bpy.data.objects.remove(source, do_unlink=True)

    if voxel_size > 0:
        for obj in (left, right):
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            obj.data.remesh_voxel_size = voxel_size
            obj.data.remesh_voxel_adaptivity = 0.0
            bpy.ops.object.voxel_remesh()

    bpy.ops.object.select_all(action="DESELECT")
    left.select_set(True)
    right.select_set(True)
    bpy.context.view_layer.objects.active = left
    bpy.ops.wm.stl_export(
        filepath=str(target_path), export_selected_objects=True, ascii_format=False
    )
    print(
        f"SPLIT_LUNGS source={source_path} target={target_path} "
        f"split_x_mm={split_x} gap_mm={gap} voxel_size_mm={voxel_size}"
    )


if __name__ == "__main__":
    main()
