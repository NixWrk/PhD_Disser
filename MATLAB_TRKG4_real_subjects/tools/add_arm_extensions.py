"""Add cylindrical computational extensions to two open arm sections of an STL.

The input is expected to be a CT-derived external surface with either two
simple, planar open sections or two connected coplanar cap groups at the
extreme coordinates of the selected axis.  A cylinder is added outwards from
each section; ``overlap`` is an additional inward part used by the boolean
union.

STL does not carry unit metadata.  The unit therefore has to be declared with
``--units`` (millimetres by default for CT meshes), and the declaration is
recorded in the JSON report.  No subject name or absolute path is part of the
algorithm or the report contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh


SCHEMA = "trkg4_arm_extensions_v1"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class ArmExtensionError(ValueError):
    """An input geometry or parameter cannot be accepted safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmExtensionError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ArmExtensionError(f"{name} must be a finite number")
    return number


def _validate_parameters(
    *,
    length: float,
    radius: float | str,
    overlap: float,
    segments: int,
    section_tolerance: float,
    units: str,
    axis: str,
) -> None:
    if axis not in AXIS_INDEX:
        raise ArmExtensionError("axis must be one of x, y, z")
    if units not in {"mm", "m"}:
        raise ArmExtensionError("units must be mm or m")
    if not math.isfinite(length) or length <= 0:
        raise ArmExtensionError("length must be positive")
    if isinstance(radius, str):
        if radius != "auto":
            raise ArmExtensionError("radius must be a positive number or auto")
    elif not math.isfinite(radius) or radius <= 0:
        raise ArmExtensionError("radius must be a positive number or auto")
    if not math.isfinite(overlap) or overlap < 0:
        raise ArmExtensionError("overlap must be non-negative")
    if int(segments) != segments or segments < 3:
        raise ArmExtensionError("segments must be an integer of at least 3")
    if not math.isfinite(section_tolerance) or section_tolerance <= 0:
        raise ArmExtensionError("section_tolerance must be positive")


def load_surface(path: Path) -> trimesh.Trimesh:
    """Load one triangle surface and reject scenes or unusable geometry."""
    try:
        loaded = trimesh.load_mesh(path, process=False)
    except Exception as exc:  # pragma: no cover - backend-specific errors
        raise ArmExtensionError(f"cannot read STL {path.name}: {exc}") from exc
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) != 1:
            raise ArmExtensionError(
                "source STL contains multiple mesh components; section detection is ambiguous"
            )
        loaded = next(iter(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ArmExtensionError("source STL did not produce a triangle mesh")

    mesh = loaded.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ArmExtensionError("source STL contains non-finite or invalid vertex coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ArmExtensionError("source STL contains no triangular faces")
    areas = np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
        axis=1,
    ) * 0.5
    if not np.isfinite(areas).all() or np.any(areas <= 1e-12):
        raise ArmExtensionError("source STL contains degenerate or invalid triangles")
    components = mesh.split(only_watertight=False)
    if len(components) != 1:
        raise ArmExtensionError(
            f"source STL has {len(components)} connected components; a single external body is required"
        )
    return mesh


def _edge_topology(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts, edges


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Return ordered vertex loops; refuse non-manifold or non-cyclic edges."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    unique, counts, _ = _edge_topology(faces)
    if np.any(counts > 2):
        raise ArmExtensionError("source STL contains non-manifold edges")
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return []

    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in boundary:
        first, second = int(first), int(second)
        adjacency[first].append(second)
        adjacency[second].append(first)
    bad_degree = {vertex: len(neighbours) for vertex, neighbours in adjacency.items() if len(neighbours) != 2}
    if bad_degree:
        raise ArmExtensionError(
            "open section is ambiguous: boundary edges do not form simple closed loops"
        )

    unseen = {tuple(edge) for edge in boundary.tolist()}
    loops: list[np.ndarray] = []
    while unseen:
        edge = min(unseen)
        start, current = edge
        previous: int | None = start
        ordered = [start, current]
        unseen.discard(edge)
        while True:
            candidates = [item for item in adjacency[current] if item != previous]
            if not candidates:
                raise ArmExtensionError("failed to order an open section boundary")
            next_vertex = candidates[0]
            unseen.discard(tuple(sorted((current, next_vertex))))
            if next_vertex == start:
                break
            ordered.append(next_vertex)
            previous, current = current, next_vertex
            if len(ordered) > len(adjacency):
                raise ArmExtensionError("open section boundary does not close")
        if len(ordered) < 3:
            raise ArmExtensionError("open section boundary has fewer than three vertices")
        loops.append(np.asarray(ordered, dtype=np.int64))
    return loops



def _selected_face_groups(face_indices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Group selected coplanar faces by shared edges."""
    selected = {int(value) for value in np.asarray(face_indices, dtype=np.int64)}
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index in selected:
        face = faces[face_index]
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces[tuple(sorted((int(first), int(second))))].append(face_index)
    neighbours: dict[int, set[int]] = {face_index: set() for face_index in selected}
    for owners in edge_to_faces.values():
        for first in owners:
            neighbours[first].update(second for second in owners if second != first)
    groups: list[np.ndarray] = []
    unseen = set(selected)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        group = []
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in neighbours[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        groups.append(np.asarray(sorted(group), dtype=np.int64))
    return groups


def _describe_closed_cap(
    mesh: trimesh.Trimesh,
    axis: str,
    side: str,
    section_tolerance: float,
) -> dict[str, object] | None:
    """Detect one connected coplanar face group at an axis extreme."""
    axis_index = AXIS_INDEX[axis]
    perpendicular = [index for index in range(3) if index != axis_index]
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    coordinate = vertices[:, axis_index]
    extreme = float(np.min(coordinate) if side == "left" else np.max(coordinate))
    triangles = vertices[faces]
    triangle_axis = triangles[:, :, axis_index]
    flat = np.ptp(triangle_axis, axis=1)
    centre_axis = np.mean(triangle_axis, axis=1)
    selected = np.flatnonzero((flat <= section_tolerance) & (np.abs(centre_axis - extreme) <= section_tolerance))
    if len(selected) == 0:
        return None
    groups = _selected_face_groups(selected, faces)
    if len(groups) != 1:
        raise ArmExtensionError(
            f"ambiguous closed cap at {side} {axis.upper()} extreme: found {len(groups)} coplanar face groups"
        )
    group = groups[0]
    group_triangles = triangles[group]
    area_vectors = np.cross(
        group_triangles[:, 1] - group_triangles[:, 0],
        group_triangles[:, 2] - group_triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(area_vectors, axis=1)
    if not np.isfinite(areas).all() or float(np.sum(areas)) <= 1e-12:
        raise ArmExtensionError(f"ambiguous closed cap at {side}: cap area is zero or invalid")
    centroids = np.mean(group_triangles, axis=1)
    centre = np.average(centroids, axis=0, weights=areas)
    projected = group_triangles.reshape(-1, 3)[:, perpendicular]
    radial = np.linalg.norm(projected - centre[perpendicular], axis=1)
    return {
        "kind": "closed_cap",
        "loop": None,
        "axis_index": axis_index,
        "perpendicular": perpendicular,
        "cut": float(centre[axis_index]),
        "center_perpendicular": np.asarray(centre[perpendicular], dtype=float),
        "center_xyz": np.asarray(centre, dtype=float),
        "area": float(np.sum(areas)),
        "radius_estimate": float(math.sqrt(float(np.sum(areas)) / math.pi)),
        "radial_median": float(np.median(radial)),
        "radial_min": float(np.min(radial)),
        "radial_max": float(np.max(radial)),
        "axis_spread": float(np.max(flat[group])),
        "cap_faces": int(len(group)),
        "boundary_vertices": None,
    }


def _detect_sections(
    mesh: trimesh.Trimesh,
    axis: str,
    section_tolerance: float,
) -> tuple[list[dict[str, object]], str]:
    """Choose either two open sections or two closed extreme caps."""
    loops = boundary_loops(mesh)
    caps = [
        _describe_closed_cap(mesh, axis, side, section_tolerance)
        for side in ("left", "right")
    ]
    cap_count = sum(cap is not None for cap in caps)
    if cap_count == 2:
        if loops:
            raise ArmExtensionError(
                "mixed surface topology: closed extreme caps and additional open boundaries are present"
            )
        sections = [cap for cap in caps if cap is not None]
        mode = "closed_capped"
    elif cap_count != 0:
        raise ArmExtensionError(
            "ambiguous surface topology: exactly two matching extreme caps or two open sections are required"
        )
    else:
        if len(loops) != 2:
            raise ArmExtensionError(
                f"ambiguous open sections: found {len(loops)}, exactly two extreme sections are required"
            )
        sections = [describe_section(mesh, loop, axis, section_tolerance) for loop in loops]
        mode = "open_sections"
    sections.sort(key=lambda item: float(item["cut"]))
    separation = float(sections[1]["cut"]) - float(sections[0]["cut"])
    if separation <= section_tolerance:
        raise ArmExtensionError(
            "ambiguous sections: the two extreme coordinates are not sufficiently separated"
        )
    sections[0]["side"] = "left"
    sections[1]["side"] = "right"
    return sections, mode

def _polygon_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    )


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    cross = points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)
    area_twice = float(np.sum(cross))
    if abs(area_twice) <= 1e-12:
        return points.mean(axis=0)
    return np.asarray(
        [
            np.sum((points[:, 0] + np.roll(points[:, 0], -1)) * cross) / (3 * area_twice),
            np.sum((points[:, 1] + np.roll(points[:, 1], -1)) * cross) / (3 * area_twice),
        ],
        dtype=float,
    )


def describe_section(
    mesh: trimesh.Trimesh,
    loop: np.ndarray,
    axis: str,
    section_tolerance: float,
) -> dict[str, object]:
    axis_index = AXIS_INDEX[axis]
    perpendicular = [index for index in range(3) if index != axis_index]
    points = np.asarray(mesh.vertices, dtype=float)[loop]
    axis_values = points[:, axis_index]
    spread = float(np.ptp(axis_values))
    if spread > section_tolerance:
        raise ArmExtensionError(
            f"open section is ambiguous: its {axis.upper()} coordinate spread "
            f"{spread:.6g} exceeds section_tolerance={section_tolerance:.6g}"
        )
    projected = points[:, perpendicular]
    centre = _polygon_centroid(projected)
    area = abs(_polygon_area(projected))
    if area <= 1e-12:
        raise ArmExtensionError("open section is ambiguous: projected area is zero")
    radial = np.linalg.norm(projected - centre, axis=1)
    if not np.isfinite(radial).all() or float(np.median(radial)) <= 0:
        raise ArmExtensionError("open section is ambiguous: radius cannot be estimated")
    cut = float(np.median(axis_values))
    angle = np.mod(np.arctan2(projected[:, 1] - centre[1], projected[:, 0] - centre[0]), 2 * np.pi)
    if np.min(np.diff(np.sort(angle))) <= 1e-12:
        raise ArmExtensionError("open section is ambiguous: repeated angular vertices")
    return {
        "kind": "open_section",
        "loop": loop,
        "axis_index": axis_index,
        "perpendicular": perpendicular,
        "cut": cut,
        "center_perpendicular": centre,
        "area": area,
        "radius_estimate": float(math.sqrt(area / math.pi)),
        "radial_median": float(np.median(radial)),
        "radial_min": float(np.min(radial)),
        "radial_max": float(np.max(radial)),
        "axis_spread": spread,
    }


def _axis_point(section: dict[str, object]) -> np.ndarray:
    point = np.zeros(3, dtype=float)
    axis_index = int(section["axis_index"])
    perpendicular = list(section["perpendicular"])
    centre = np.asarray(section["center_perpendicular"], dtype=float)
    point[axis_index] = float(section["cut"])
    point[perpendicular] = centre
    return point


def _ring(
    section: dict[str, object],
    coordinate: float,
    radius: float,
    segments: int,
    vertices: list[list[float]],
) -> np.ndarray:
    axis_index = int(section["axis_index"])
    perpendicular = list(section["perpendicular"])
    centre = np.asarray(section["center_perpendicular"], dtype=float)
    angles = 2 * np.pi * np.arange(segments, dtype=float) / segments
    points = np.zeros((segments, 3), dtype=float)
    points[:, axis_index] = coordinate
    points[:, perpendicular[0]] = centre[0] + radius * np.cos(angles)
    points[:, perpendicular[1]] = centre[1] + radius * np.sin(angles)
    first = len(vertices)
    vertices.extend(points.tolist())
    return np.arange(first, first + segments, dtype=np.int64)


def _angles(points: np.ndarray, section: dict[str, object]) -> np.ndarray:
    perpendicular = list(section["perpendicular"])
    centre = np.asarray(section["center_perpendicular"], dtype=float)
    projected = points[:, perpendicular]
    return np.mod(np.arctan2(projected[:, 1] - centre[1], projected[:, 0] - centre[0]), 2 * np.pi)


def _bridge_loops(
    first: np.ndarray,
    second: np.ndarray,
    vertices: list[list[float]],
    section: dict[str, object],
) -> list[list[int]]:
    """Triangulate a strip between two cyclic loops with different counts."""
    coordinates = np.asarray(vertices, dtype=float)
    first_angles = _angles(coordinates[first], section)
    second_angles = _angles(coordinates[second], section)
    first_order = np.argsort(first_angles)
    second_order = np.argsort(second_angles)
    first = first[first_order]
    second = second[second_order]
    first_angles = first_angles[first_order]
    second_angles = second_angles[second_order]
    first_ext = np.concatenate((first_angles, [first_angles[0] + 2 * np.pi]))
    second_ext = np.concatenate((second_angles, [second_angles[0] + 2 * np.pi]))
    faces: list[list[int]] = []
    i = j = 0
    while i < len(first) or j < len(second):
        current_first = int(first[i % len(first)])
        current_second = int(second[j % len(second)])
        next_first = first_ext[i + 1] if i < len(first) else math.inf
        next_second = second_ext[j + 1] if j < len(second) else math.inf
        if next_first <= next_second:
            faces.append([current_first, current_second, int(first[(i + 1) % len(first)])])
            i += 1
        else:
            faces.append([current_first, current_second, int(second[(j + 1) % len(second)])])
            j += 1
    return faces


def _cap_ring(
    ring: np.ndarray,
    section: dict[str, object],
    coordinate: float,
    vertices: list[list[float]],
) -> list[list[int]]:
    centre = _axis_point(section)
    centre[int(section["axis_index"])] = coordinate
    centre_index = len(vertices)
    vertices.append(centre.tolist())
    return [
        [centre_index, int(ring[index]), int(ring[(index + 1) % len(ring)])]
        for index in range(len(ring))
    ]


def _cylinder_mesh(
    section: dict[str, object],
    radius: float,
    length: float,
    overlap: float,
    segments: int,
) -> tuple[trimesh.Trimesh, float, float]:
    """Build a closed cylinder whose inner cap overlaps the source volume."""
    direction = -1.0 if section["side"] == "left" else 1.0
    cut = float(section["cut"])
    inner = cut - direction * overlap
    far = cut + direction * length
    axis_index = int(section["axis_index"])
    perpendicular = list(section["perpendicular"])
    centre = np.asarray(section["center_perpendicular"], dtype=float)
    angles = 2 * np.pi * np.arange(segments, dtype=float) / segments
    points = np.zeros((2 * segments + 2, 3), dtype=float)
    points[:segments, axis_index] = inner
    points[segments : 2 * segments, axis_index] = far
    points[:segments, perpendicular[0]] = centre[0] + radius * np.cos(angles)
    points[:segments, perpendicular[1]] = centre[1] + radius * np.sin(angles)
    points[segments : 2 * segments, perpendicular[0]] = centre[0] + radius * np.cos(angles)
    points[segments : 2 * segments, perpendicular[1]] = centre[1] + radius * np.sin(angles)
    points[-2, axis_index] = inner
    points[-2, perpendicular] = centre
    points[-1, axis_index] = far
    points[-1, perpendicular] = centre
    inner_centre = 2 * segments
    far_centre = inner_centre + 1
    faces: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend(
            (
                [index, segments + index, segments + nxt],
                [index, segments + nxt, nxt],
                [inner_centre, nxt, index],
                [far_centre, segments + index, segments + nxt],
            )
        )
    output = trimesh.Trimesh(vertices=points, faces=np.asarray(faces, dtype=np.int64), process=False)
    trimesh.repair.fix_normals(output, multibody=False)
    return output, inner, far


def _close_open_surface(mesh: trimesh.Trimesh, sections: Sequence[dict[str, object]]) -> trimesh.Trimesh:
    """Close only the two accepted planar open sections before boolean union."""
    vertices = np.asarray(mesh.vertices, dtype=float).tolist()
    faces = np.asarray(mesh.faces, dtype=np.int64).tolist()
    for section in sections:
        loop = np.asarray(section["loop"], dtype=np.int64)
        centre_index = len(vertices)
        vertices.append(_axis_point(section).tolist())
        for index in range(len(loop)):
            faces.append([centre_index, int(loop[index]), int(loop[(index + 1) % len(loop)])])
    closed = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int64), process=False)
    trimesh.repair.fix_normals(closed, multibody=False)
    return closed


def _mesh_stats(mesh: trimesh.Trimesh) -> dict[str, object]:
    unique, counts, _ = _edge_topology(np.asarray(mesh.faces, dtype=np.int64))
    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))
    components = len(mesh.split(only_watertight=False))
    volume = float(mesh.volume) if mesh.is_volume and math.isfinite(float(mesh.volume)) else None
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "connected_components": components,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "volume": volume,
        "euler_number": int(mesh.euler_number),
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
    }


def _require_closed_positive_volume(mesh: trimesh.Trimesh, context: str) -> trimesh.Trimesh:
    trimesh.repair.fix_normals(mesh, multibody=False)
    if mesh.is_volume and mesh.volume < 0:
        mesh.faces = mesh.faces[:, ::-1]
    stats = _mesh_stats(mesh)
    if not (
        stats["watertight"]
        and stats["winding_consistent"]
        and stats["is_volume"]
        and stats["connected_components"] == 1
        and stats["boundary_edges"] == 0
        and stats["nonmanifold_edges"] == 0
        and stats["volume"] is not None
        and float(stats["volume"]) > 0
    ):
        raise ArmExtensionError(
            f"{context} failed strict closed-volume validation: "
            + json.dumps(stats, ensure_ascii=False)
        )
    return mesh


def _boolean_union_manifold(
    base: trimesh.Trimesh,
    cylinders: Sequence[trimesh.Trimesh],
) -> trimesh.Trimesh:
    try:
        import manifold3d  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ArmExtensionError(
            "reliable FEM boolean union requires the manifold3d package"
        ) from exc
    engines = set(getattr(trimesh.boolean, "engines_available", set()))
    if "manifold" not in engines:
        raise ArmExtensionError(
            f"trimesh does not expose the manifold boolean engine; available={sorted(engines)}"
        )
    try:
        result = trimesh.boolean.union(
            [base, *cylinders], engine="manifold", check_volume=True
        )
    except Exception as exc:  # pragma: no cover - backend-specific failure
        raise ArmExtensionError(f"manifold boolean union failed: {exc}") from exc
    if result is None or not isinstance(result, trimesh.Trimesh):
        raise ArmExtensionError("manifold boolean union returned no single triangle mesh")
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    return _require_closed_positive_volume(result, "manifold boolean result")


def _arm_report(
    section: dict[str, object],
    radius: float,
    axis: str,
    length: float,
    overlap: float,
) -> dict[str, object]:
    cut = float(section["cut"])
    direction = -1.0 if section["side"] == "left" else 1.0
    inner = cut - direction * overlap
    far = cut + direction * length
    point = _axis_point(section)
    return {
        "section_type": section["kind"],
        "cut_x": float(point[0]),
        "center_y": float(point[1]),
        "center_z": float(point[2]),
        "radius": float(radius),
        "cylinder_x_min": float(min(inner, far)) if axis == "x" else None,
        "cylinder_x_max": float(max(inner, far)) if axis == "x" else None,
        "cut_coordinate": cut,
        "cylinder_axis_min": float(min(inner, far)),
        "cylinder_axis_max": float(max(inner, far)),
        "direction": "negative_axis" if direction < 0 else "positive_axis",
        "length_outward": float(length),
        "overlap_inward": float(overlap),
        "section_area": float(section["area"]),
        "cap_area": float(section["area"]) if section["kind"] == "closed_cap" else None,
        "section_radius_estimate": float(section["radius_estimate"]),
        "section_axis_spread": float(section["axis_spread"]),
        "boundary_vertices": section.get("boundary_vertices"),
        "cap_faces": section.get("cap_faces"),
    }


def _select_sections(mesh: trimesh.Trimesh, axis: str, section_tolerance: float) -> tuple[list[dict[str, object]], str]:
    return _detect_sections(mesh, axis, section_tolerance)


def add_arm_extensions(
    source: Path,
    target: Path,
    report_path: Path,
    *,
    axis: str = "x",
    length: float = 180.0,
    radius: float | str = "auto",
    overlap: float = 12.0,
    segments: int = 96,
    units: str = "mm",
    section_tolerance: float = 0.5,
    method: str = "manifold",
) -> dict[str, object]:
    """Create a closed FEM surface by a mandatory manifold boolean union."""
    axis = str(axis).lower()
    _validate_parameters(
        length=length,
        radius=radius,
        overlap=overlap,
        segments=segments,
        section_tolerance=section_tolerance,
        units=units,
        axis=axis,
    )
    if method not in {"auto", "boolean", "manifold"}:
        raise ArmExtensionError(
            "simple surface stitching is disabled; method must use the manifold boolean engine"
        )
    source = Path(source)
    target = Path(target)
    report_path = Path(report_path)
    if source.resolve() == target.resolve():
        raise ArmExtensionError("source and target STL paths must be different")
    if not source.is_file():
        raise ArmExtensionError(f"source STL does not exist: {source}")

    mesh = load_surface(source)
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = np.ptp(bounds, axis=1)
    if not np.isfinite(extents).all() or np.any(extents <= 0):
        raise ArmExtensionError("source STL has invalid or zero coordinate extent")
    sections, source_mode = _select_sections(mesh, axis, section_tolerance)
    radius_by_section = [
        float(section["radius_estimate"]) if radius == "auto" else float(radius)
        for section in sections
    ]
    if any(value <= 0 or not math.isfinite(value) for value in radius_by_section):
        raise ArmExtensionError("computed cylinder radius is invalid")

    source_stats = _mesh_stats(mesh)
    if source_mode == "closed_capped":
        base = _require_closed_positive_volume(mesh.copy(), "closed capped source")
    else:
        base = _require_closed_positive_volume(
            _close_open_surface(mesh, sections), "open source after accepted caps"
        )
    cylinders = [
        _cylinder_mesh(section, radius_value, float(length), float(overlap), int(segments))[0]
        for section, radius_value in zip(sections, radius_by_section)
    ]
    output = _boolean_union_manifold(base, cylinders)

    target.parent.mkdir(parents=True, exist_ok=True)
    output.export(target, file_type="stl")
    checked = _require_closed_positive_volume(load_surface(target), "serialized target")
    target_stats = _mesh_stats(checked)
    base_stats = _mesh_stats(base)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ok",
        "source_mode": source_mode,
        "units": units,
        "unit_check": {
            "declared_units": units,
            "stl_unit_metadata": "not_present_in_binary_or_ascii_stl",
            "coordinate_values_finite": True,
            "positive_source_extents": True,
            "interpretation": "all coordinates and geometric parameters use the declared units",
        },
        "axis": axis,
        "parameters": {
            "length": float(length),
            "radius": radius,
            "overlap": float(overlap),
            "segments": int(segments),
            "section_tolerance": float(section_tolerance),
            "method_requested": method,
            "method_used": "trimesh_boolean_manifold3d",
            "boolean_engine": "manifold",
        },
        "source": {
            "name": source.name,
            "sha256": file_sha256(source),
            "topology": source_stats,
        },
        "boolean_input": {"topology": base_stats},
        "target": {
            "name": target.name,
            "sha256": file_sha256(target),
            "topology": target_stats,
        },
        "topology": {
            "source": source_stats,
            "boolean_input": base_stats,
            "target": target_stats,
            "volume_change": (
                float(target_stats["volume"]) - float(source_stats["volume"])
                if source_stats["volume"] is not None
                else None
            ),
        },
        "arm_extension": {
            "left": _arm_report(sections[0], radius_by_section[0], axis, float(length), float(overlap)),
            "right": _arm_report(sections[1], radius_by_section[1], axis, float(length), float(overlap)),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _radius_argument(value: str) -> float | str:
    if value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("radius must be a positive number or auto") from exc


def _axis_argument(value: str) -> str:
    normalized = value.lower()
    if normalized not in AXIS_INDEX:
        raise argparse.ArgumentTypeError("axis must be one of x, y, z")
    return normalized

def _positive(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _nonnegative(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add two artificial cylindrical arm extensions to a CT-derived "
            "surface with either two open end contours or two closed planar "
            "end caps at the selected axis extrema."
        )
    )
    parser.add_argument(
        "source_stl", type=Path,
        help="source CT-derived STL with two unambiguous arm end sections",
    )
    parser.add_argument("target_stl", type=Path, help="target closed STL")
    parser.add_argument("report_json", type=Path, help="JSON report path")
    parser.add_argument("--axis", type=_axis_argument, default="x", help="extension axis (default: x)")
    parser.add_argument(
        "--units", choices=("mm", "m"), default="mm", help="declared STL units (default: mm)"
    )
    parser.add_argument("--length", type=_positive, default=180.0, help="outward length in declared units")
    parser.add_argument(
        "--radius", type=_radius_argument, default="auto", help="cylinder radius or auto (default: auto)"
    )
    parser.add_argument("--overlap", type=_nonnegative, default=12.0, help="inward overlap in declared units")
    parser.add_argument("--segments", type=int, default=96, help="cylinder circumference segments")
    parser.add_argument(
        "--section-tolerance",
        type=_positive,
        default=0.5,
        help="maximum open-section spread along the axis in declared units",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "boolean", "manifold"),
        default="manifold",
        help="mandatory trimesh boolean method backed by manifold3d",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = add_arm_extensions(
            args.source_stl,
            args.target_stl,
            args.report_json,
            axis=args.axis,
            length=args.length,
            radius=args.radius,
            overlap=args.overlap,
            segments=args.segments,
            units=args.units,
            section_tolerance=args.section_tolerance,
            method=args.method,
        )
    except ArmExtensionError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
