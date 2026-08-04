"""Create a manifold STL candidate using Blender's voxel remesher.

Run with:
  blender --background --factory-startup --python tools/blender_voxel_remesh_stl.py \
    -- input.stl output.stl voxel_size_mm
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass input, output and voxel size after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 3:
        raise SystemExit("Expected: input.stl output.stl voxel_size_mm")

    source = Path(args[0]).resolve()
    target = Path(args[1]).resolve()
    voxel_size = float(args[2])
    target.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.stl_import(filepath=str(source))
    obj = bpy.context.active_object
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(target), export_selected_objects=True, ascii_format=False)
    print(f"VOXEL_REMESH source={source} target={target} voxel_size_mm={voxel_size}")
    print(f"VOXEL_REMESH_RESULT vertices={len(obj.data.vertices):,} faces={len(obj.data.polygons):,}")


if __name__ == "__main__":
    main()
