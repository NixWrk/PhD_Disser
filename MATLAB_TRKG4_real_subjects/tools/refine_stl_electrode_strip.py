"""Conformingly subdivide STL triangles near the fitted electrode axis.

New vertices are face centres and edge midpoints, so the piecewise-planar
outer geometry is preserved exactly.  Faces adjacent to the selected strip
are split as needed to avoid T-junctions and keep the STL watertight.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


STL_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        stream.read(80)
        count = struct.unpack("<I", stream.read(4))[0]
        records = np.fromfile(stream, dtype=STL_DTYPE, count=count)
    if len(records) != count:
        raise ValueError(f"Truncated binary STL: expected {count}, got {len(records)}")
    flat = records["vertices"].reshape(-1, 3).astype(np.float64)
    vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    records = np.zeros(len(faces), dtype=STL_DTYPE)
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = triangles.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"TRKG4 conforming electrode-strip refinement".ljust(80, b" "))
        stream.write(struct.pack("<I", len(records)))
        records.tofile(stream)


def subdivide_strip(
    vertices: np.ndarray,
    faces: np.ndarray,
    centre: np.ndarray,
    axis: np.ndarray,
    half_length: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    triangles = vertices[faces]
    face_centres = triangles.mean(axis=1)
    delta = face_centres - centre
    axial = delta @ axis
    radial = np.linalg.norm(delta - axial[:, None] * axis, axis=1)
    selected = (np.abs(axial) <= half_length) & (radial <= radius)

    marked_edges: set[tuple[int, int]] = set()
    for face in faces[selected]:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            marked_edges.add((min(int(a), int(b)), max(int(a), int(b))))

    vertex_list = vertices.tolist()
    edge_midpoint: dict[tuple[int, int], int] = {}
    for edge in marked_edges:
        midpoint = 0.5 * (vertices[edge[0]] + vertices[edge[1]])
        edge_midpoint[edge] = len(vertex_list)
        vertex_list.append(midpoint.tolist())

    new_faces: list[list[int]] = []
    affected = 0
    for face in faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        boundary: list[int] = [a]
        for start, end in ((a, b), (b, c), (c, a)):
            edge = (min(start, end), max(start, end))
            if edge in edge_midpoint:
                boundary.append(edge_midpoint[edge])
            if end != a:
                boundary.append(end)
        if len(boundary) == 3:
            new_faces.append(boundary)
            continue

        affected += 1
        face_centre = np.mean(np.asarray([vertex_list[index] for index in (a, b, c)]), axis=0)
        centre_index = len(vertex_list)
        vertex_list.append(face_centre.tolist())
        for index, current in enumerate(boundary):
            following = boundary[(index + 1) % len(boundary)]
            new_faces.append([current, following, centre_index])

    return (
        np.asarray(vertex_list, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64),
        affected,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--centre-mm", type=float, nargs=3, required=True)
    parser.add_argument("--axis", type=float, nargs=3, required=True)
    parser.add_argument("--half-length-mm", type=float, default=90.0)
    parser.add_argument("--radius-mm", type=float, default=24.0)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    centre = np.asarray(args.centre_mm, dtype=float)
    axis = np.asarray(args.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    vertices, faces = read_binary_stl(args.source)
    initial_vertices, initial_faces = len(vertices), len(faces)
    for pass_index in range(args.passes):
        vertices, faces, affected = subdivide_strip(
            vertices, faces, centre, axis, args.half_length_mm, args.radius_mm
        )
        print(
            f"pass={pass_index + 1} affected_faces={affected:,} "
            f"vertices={len(vertices):,} faces={len(faces):,}"
        )
    write_binary_stl(args.target, vertices, faces)
    print(
        f"REFINED_STL source={args.source.resolve()} target={args.target.resolve()} "
        f"vertices={initial_vertices:,}->{len(vertices):,} "
        f"faces={initial_faces:,}->{len(faces):,}"
    )


if __name__ == "__main__":
    main()
