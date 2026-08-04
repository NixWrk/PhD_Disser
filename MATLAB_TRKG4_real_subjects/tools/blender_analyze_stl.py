"""Analyze STL topology with Blender's mesh kernel.

Run with:
  blender --background --factory-startup --python tools/blender_analyze_stl.py -- file.stl ...
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy


def face_components(bm: bmesh.types.BMesh) -> list[list[bmesh.types.BMFace]]:
    remaining = set(bm.faces)
    components = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component = [seed]
        while queue:
            face = queue.popleft()
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def component_volume(faces: list[bmesh.types.BMFace]) -> float:
    # Signed tetrahedral volume relative to the origin.
    volume = 0.0
    for face in faces:
        verts = [loop.vert.co for loop in face.loops]
        if len(verts) < 3:
            continue
        origin = verts[0]
        for i in range(1, len(verts) - 1):
            volume += origin.dot(verts[i].cross(verts[i + 1])) / 6.0
    return volume


def analyze(path: Path) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.stl_import(filepath=str(path.resolve()))
    obj = bpy.context.active_object
    mesh = obj.data

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.normal_update()

    boundary = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
    nonmanifold = sum(1 for edge in bm.edges if len(edge.link_faces) != 2)
    components = face_components(bm)
    bounds = [[min(v.co[i] for v in bm.verts), max(v.co[i] for v in bm.verts)] for i in range(3)]

    print(f"\nSTL_DIAGNOSTIC {path}")
    print(f"  vertices={len(bm.verts):,}")
    print(f"  edges={len(bm.edges):,}")
    print(f"  faces={len(bm.faces):,}")
    print(f"  boundary_edges={boundary:,}")
    print(f"  nonmanifold_edges={nonmanifold:,}")
    print(f"  components={len(components)}")
    print(f"  signed_volume_mm3={bm.calc_volume(signed=True):,.3f}")
    euler = len(bm.verts) - len(bm.edges) + len(bm.faces)
    print(f"  euler_characteristic={euler}")
    print(f"  bounds_mm={bounds}")
    for index, faces in enumerate(components[:20], 1):
        vertices = {vert for face in faces for vert in face.verts}
        edges = {edge for face in faces for edge in face.edges}
        component_euler = len(vertices) - len(edges) + len(faces)
        genus = (2 - component_euler) / 2
        print(
            f"    component={index} faces={len(faces):,} "
            f"vertices={len(vertices):,} euler={component_euler} genus={genus:g} "
            f"signed_volume_mm3={component_volume(faces):,.3f}"
        )
    bm.free()


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass STL paths after --")
    for argument in sys.argv[separator + 1 :]:
        analyze(Path(argument))


if __name__ == "__main__":
    main()
