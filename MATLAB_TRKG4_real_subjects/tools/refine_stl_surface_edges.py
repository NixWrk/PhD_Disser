"""Conformingly split long STL edges without changing the surface geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stl import Mode, mesh


def vertex_key(point: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in point)


def refine_once(
    triangles: np.ndarray,
    max_edge_mm: float,
    centres_mm: np.ndarray | None = None,
    radius_mm: float | None = None,
) -> tuple[np.ndarray, int]:
    """Split every globally marked long edge using conforming triangle templates."""
    marked: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    if centres_mm is None:
        selected = np.ones(len(triangles), dtype=bool)
    else:
        selected = np.zeros(len(triangles), dtype=bool)
        for centre in centres_mm:
            selected |= np.min(np.linalg.norm(triangles - centre, axis=2), axis=1) <= radius_mm
    for tri, use_face in zip(triangles, selected):
        if not use_face:
            continue
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if np.linalg.norm(tri[j] - tri[i]) > max_edge_mm:
                a, b = vertex_key(tri[i]), vertex_key(tri[j])
                marked.add((a, b) if a < b else (b, a))
    if not marked:
        return triangles, 0

    midpoint = {edge: (np.asarray(edge[0]) + np.asarray(edge[1])) / 2 for edge in marked}
    output: list[np.ndarray] = []
    for tri in triangles:
        keys = [vertex_key(point) for point in tri]
        edge_keys = []
        is_marked = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = (keys[i], keys[j]) if keys[i] < keys[j] else (keys[j], keys[i])
            edge_keys.append(edge)
            is_marked.append(edge in marked)
        count = sum(is_marked)
        a, b, c = tri
        mab = midpoint.get(edge_keys[0])
        mbc = midpoint.get(edge_keys[1])
        mca = midpoint.get(edge_keys[2])
        if count == 0:
            output.append(tri)
        elif count == 1:
            if is_marked[0]:
                output.extend(([a, mab, c], [mab, b, c]))
            elif is_marked[1]:
                output.extend(([b, mbc, a], [mbc, c, a]))
            else:
                output.extend(([c, mca, b], [mca, a, b]))
        elif count == 2:
            if not is_marked[0]:
                output.extend(([c, mca, mbc], [a, b, mca], [b, mbc, mca]))
            elif not is_marked[1]:
                output.extend(([a, mab, mca], [b, c, mab], [c, mca, mab]))
            else:
                output.extend(([b, mbc, mab], [c, a, mbc], [a, mab, mbc]))
        else:
            output.extend(
                ([a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca])
            )
    return np.asarray(output, dtype=np.float64), len(marked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--max-edge-mm", type=float, default=4.0)
    parser.add_argument(
        "--centre-mm", type=float, nargs=3, action="append",
        help="Repeat for local refinement around one or more electrode centres.",
    )
    parser.add_argument("--radius-mm", type=float, default=35.0)
    args = parser.parse_args()

    source = mesh.Mesh.from_file(str(args.source.resolve()))
    triangles = np.asarray(source.vectors, dtype=np.float64)
    initial_faces = len(triangles)
    centres = None if args.centre_mm is None else np.asarray(args.centre_mm, dtype=float)
    iterations = 0
    while True:
        triangles, split_edges = refine_once(
            triangles, args.max_edge_mm, centres, args.radius_mm
        )
        if split_edges == 0:
            break
        iterations += 1
        print(f"iteration={iterations} split_edges={split_edges:,} faces={len(triangles):,}")

    result = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    result.vectors[:] = triangles.astype(np.float32)
    result.update_normals()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(args.target.resolve()), mode=Mode.BINARY)
    lengths = np.linalg.norm(
        triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2
    )
    print(f"REFINE_STL source={args.source.resolve()} target={args.target.resolve()}")
    print(f"  max_edge_mm={args.max_edge_mm} iterations={iterations}")
    if centres is not None:
        print(f"  local_centres={len(centres)} radius_mm={args.radius_mm}")
    print(f"  faces={initial_faces:,}->{len(triangles):,} actual_max_edge_mm={lengths.max():.6f}")


if __name__ == "__main__":
    main()
