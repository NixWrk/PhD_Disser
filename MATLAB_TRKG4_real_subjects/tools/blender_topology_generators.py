"""Locate tree-cotree topology generators of a closed STL surface."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import bmesh
import bpy


def tree_path(u, v, parent, depth):
    left = []
    right = []
    while depth[u] > depth[v]:
        left.append(u)
        u = parent[u]
    while depth[v] > depth[u]:
        right.append(v)
        v = parent[v]
    while u != v:
        left.append(u)
        right.append(v)
        u = parent[u]
        v = parent[v]
    left.append(u)
    return left + list(reversed(right))


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        raise SystemExit("Pass one STL path after --")
    path = Path(sys.argv[separator + 1]).resolve()

    bpy.ops.wm.stl_import(filepath=str(path))
    bm = bmesh.new()
    bm.from_mesh(bpy.context.active_object.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for sequence in (bm.verts, bm.edges, bm.faces):
        sequence.index_update()

    parent = [-1] * len(bm.verts)
    depth = [0] * len(bm.verts)
    primal_edges = set()
    queue = deque([0])
    parent[0] = 0
    while queue:
        vertex_index = queue.popleft()
        vertex = bm.verts[vertex_index]
        for edge in vertex.link_edges:
            other = edge.other_vert(vertex).index
            if parent[other] == -1:
                parent[other] = vertex_index
                depth[other] = depth[vertex_index] + 1
                primal_edges.add(edge.index)
                queue.append(other)

    dual_seen = {0}
    dual_edges = set()
    queue = deque([0])
    while queue:
        face_index = queue.popleft()
        face = bm.faces[face_index]
        for edge in face.edges:
            if edge.index in primal_edges or len(edge.link_faces) != 2:
                continue
            other_face = edge.link_faces[0] if edge.link_faces[1].index == face_index else edge.link_faces[1]
            if other_face.index not in dual_seen:
                dual_seen.add(other_face.index)
                dual_edges.add(edge.index)
                queue.append(other_face.index)

    generators = [
        edge for edge in bm.edges
        if edge.index not in primal_edges and edge.index not in dual_edges
    ]
    print(f"TOPOLOGY_GENERATORS path={path}")
    print(f"  generators={len(generators)} expected_2g={len(generators)}")
    for number, edge in enumerate(generators, 1):
        u, v = edge.verts[0].index, edge.verts[1].index
        indices = tree_path(u, v, parent, depth)
        coords = [bm.verts[index].co for index in indices]
        bounds = [[min(co[axis] for co in coords), max(co[axis] for co in coords)] for axis in range(3)]
        centroid = [sum(co[axis] for co in coords) / len(coords) for axis in range(3)]
        length = sum((coords[i + 1] - coords[i]).length for i in range(len(coords) - 1))
        length += (edge.verts[1].co - edge.verts[0].co).length
        print(
            f"  generator={number} cycle_vertices={len(indices)} length_mm={length:.3f} "
            f"centroid={centroid} bounds={bounds} edge_midpoint={(edge.verts[0].co + edge.verts[1].co) / 2}"
        )
    bm.free()


if __name__ == "__main__":
    main()
