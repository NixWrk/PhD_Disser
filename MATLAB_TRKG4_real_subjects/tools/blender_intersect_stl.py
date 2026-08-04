"""Export the exact Boolean intersection of two closed STL surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy


def load(path: Path):
    bpy.ops.wm.stl_import(filepath=str(path.resolve()))
    return bpy.context.active_object


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass A, B and output STL paths after --")
    args = sys.argv[separator + 1 :]
    if len(args) != 3:
        raise SystemExit("Expected: a.stl b.stl intersection.stl")

    target = Path(args[2]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    a = load(Path(args[0]))
    b = load(Path(args[1]))

    modifier = a.modifiers.new(name="exact_intersection", type="BOOLEAN")
    modifier.operation = "INTERSECT"
    modifier.solver = "EXACT"
    modifier.object = b
    bpy.ops.object.select_all(action="DESELECT")
    a.select_set(True)
    bpy.context.view_layer.objects.active = a
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    print(f"INTERSECTION_RESULT vertices={len(a.data.vertices):,} faces={len(a.data.polygons):,}")

    bpy.ops.object.select_all(action="DESELECT")
    a.select_set(True)
    bpy.context.view_layer.objects.active = a
    bpy.ops.wm.stl_export(filepath=str(target), export_selected_objects=True, ascii_format=False)
    print(f"INTERSECTION a={args[0]} b={args[1]} target={target}")


if __name__ == "__main__":
    main()
