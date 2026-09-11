"""Replace coarse artificial-arm side walls with a <=1 mm structured surface.

The CT-derived torso and the existing arm/body joins and end caps are preserved.
Only the very long triangles forming the artificial cylindrical side walls are
removed.  Their two boundary loops are connected by a regular cylindrical grid
fine enough to place the parameterised arm electrodes reproducibly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh


def self_intersecting_face_count(path: Path) -> int:
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(path))
    mesh_set.compute_selection_by_self_intersections_per_face()
    selected = np.asarray(
        mesh_set.current_mesh().face_selection_array(), dtype=bool
    )
    return int(np.count_nonzero(selected))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            counts[key] += 1
            directed[key] = (int(u), int(v))
    return [directed[key] for key, count in counts.items() if count == 1]


def edge_components(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    by_vertex: dict[int, list[int]] = defaultdict(list)
    for index, (a, b) in enumerate(edges):
        by_vertex[a].append(index)
        by_vertex[b].append(index)
    unseen = set(range(len(edges)))
    components: list[list[tuple[int, int]]] = []
    while unseen:
        stack = [unseen.pop()]
        indices: list[int] = []
        while stack:
            index = stack.pop()
            indices.append(index)
            a, b = edges[index]
            for vertex in (a, b):
                for neighbour in by_vertex[vertex]:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)
        components.append([edges[index] for index in indices])
    return components


def ordered_loop(component: list[tuple[int, int]]) -> np.ndarray:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in component:
        adjacency[a].append(b)
        adjacency[b].append(a)
    bad = {vertex: len(neighbours) for vertex, neighbours in adjacency.items() if len(neighbours) != 2}
    if bad:
        raise RuntimeError(f"Arm-side boundary is not a simple loop: {bad}")
    start = min(adjacency)
    result = [start]
    previous = None
    current = start
    while True:
        candidates = [item for item in adjacency[current] if item != previous]
        nxt = candidates[0]
        if nxt == start:
            break
        result.append(nxt)
        previous, current = current, nxt
        if len(result) > len(adjacency):
            raise RuntimeError("Failed to close arm-side boundary loop")
    if len(result) != len(adjacency):
        raise RuntimeError("Arm-side boundary loop did not include every vertex")
    return np.asarray(result, dtype=np.int64)


def sort_loop_by_angle(loop: np.ndarray, vertices: np.ndarray, centre_yz: np.ndarray) -> np.ndarray:
    yz = vertices[loop, 1:3] - centre_yz
    angle = np.mod(np.arctan2(yz[:, 1], yz[:, 0]), 2 * np.pi)
    return loop[np.argsort(angle)]


def append_ring(vertices: list[list[float]], x: float, centre_yz: np.ndarray, radius: float, count: int) -> np.ndarray:
    angle = 2 * np.pi * np.arange(count) / count
    start = len(vertices)
    vertices.extend(
        np.column_stack(
            (
                np.full(count, x),
                centre_yz[0] + radius * np.cos(angle),
                centre_yz[1] + radius * np.sin(angle),
            )
        ).tolist()
    )
    return np.arange(start, start + count, dtype=np.int64)


def outward_face(face: list[int], vertices: list[list[float]], centre_yz: np.ndarray) -> list[int]:
    points = np.asarray([vertices[index] for index in face], dtype=float)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    radial = points.mean(axis=0)[1:3] - centre_yz
    if np.dot(normal[1:3], radial) < 0:
        return [face[0], face[2], face[1]]
    return face


def connect_equal_rings(
    first: np.ndarray,
    second: np.ndarray,
    vertices: list[list[float]],
    centre_yz: np.ndarray,
) -> list[list[int]]:
    faces: list[list[int]] = []
    count = len(first)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append(outward_face([int(first[index]), int(second[index]), int(second[nxt])], vertices, centre_yz))
        faces.append(outward_face([int(first[index]), int(second[nxt]), int(first[nxt])], vertices, centre_yz))
    return faces


def connect_unequal_rings(
    first: np.ndarray,
    second: np.ndarray,
    vertices: list[list[float]],
    centre_yz: np.ndarray,
) -> list[list[int]]:
    """Triangulate between cyclic rings using their common angular parameter."""
    faces: list[list[int]] = []
    n_first, n_second = len(first), len(second)
    coordinates = np.asarray(vertices)
    first_angles = np.mod(
        np.arctan2(
            coordinates[first, 2] - centre_yz[1],
            coordinates[first, 1] - centre_yz[0],
        ), 2 * np.pi)
    second_angles = np.mod(
        np.arctan2(
            coordinates[second, 2] - centre_yz[1],
            coordinates[second, 1] - centre_yz[0],
        ), 2 * np.pi)
    i = j = 0
    while i < n_first or j < n_second:
        next_first = first_angles[(i + 1) % n_first] + (2 * np.pi if i + 1 >= n_first else 0) if i < n_first else math.inf
        next_second = second_angles[(j + 1) % n_second] + (2 * np.pi if j + 1 >= n_second else 0) if j < n_second else math.inf
        a = int(first[i % n_first])
        b = int(second[j % n_second])
        if next_first <= next_second:
            a_next = int(first[(i + 1) % n_first])
            face = [a, b, a_next]
            i += 1
        else:
            b_next = int(second[(j + 1) % n_second])
            face = [a, b, b_next]
            j += 1
        faces.append(outward_face(face, vertices, centre_yz))
    return faces


def surface_edge_stats(triangles: np.ndarray) -> dict[str, float]:
    edges = np.concatenate(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        )
    )
    percentiles = np.percentile(edges, [0, 50, 90, 95, 99, 100])
    return dict(zip(("min", "p50", "p90", "p95", "p99", "max"), map(float, percentiles)))


def clip_side_at_plane(
    selected_faces: np.ndarray,
    original_vertices: np.ndarray,
    vertices: list[list[float]],
    plane_x: float,
    outward_sign: float,
) -> tuple[list[list[int]], np.ndarray]:
    """Keep the original join-side part and expose one planar cut loop."""
    intersection_cache: dict[tuple[int, int], int] = {}
    cut_edges: list[tuple[int, int]] = []
    kept_faces: list[list[int]] = []

    def is_inside(index: int) -> bool:
        return outward_sign * (vertices[index][0] - plane_x) <= 1e-9

    def intersection(first: int, second: int) -> int:
        key = (min(first, second), max(first, second))
        if key in intersection_cache:
            return intersection_cache[key]
        p0 = original_vertices[first]
        p1 = original_vertices[second]
        fraction = (plane_x - p0[0]) / (p1[0] - p0[0])
        point = p0 + fraction * (p1 - p0)
        point[0] = plane_x
        index = len(vertices)
        vertices.append(point.tolist())
        intersection_cache[key] = index
        return index

    for face in selected_faces:
        polygon: list[int] = []
        crossings: list[int] = []
        for offset in range(3):
            first = int(face[offset])
            second = int(face[(offset + 1) % 3])
            first_inside = is_inside(first)
            second_inside = is_inside(second)
            if second_inside:
                if not first_inside:
                    item = intersection(first, second)
                    polygon.append(item)
                    crossings.append(item)
                polygon.append(second)
            elif first_inside:
                item = intersection(first, second)
                polygon.append(item)
                crossings.append(item)
        if len(polygon) < 3:
            continue
        for offset in range(1, len(polygon) - 1):
            kept_faces.append([polygon[0], polygon[offset], polygon[offset + 1]])
        unique_crossings = list(dict.fromkeys(crossings))
        if len(unique_crossings) == 2:
            cut_edges.append((unique_crossings[0], unique_crossings[1]))

    components = edge_components(cut_edges)
    if len(components) != 1:
        raise RuntimeError(
            f"Expected one clipping loop, found {len(components)}"
        )
    cut_loop = ordered_loop(components[0])
    return kept_faces, cut_loop


def replace_arm_side(
    side: str,
    arm: dict[str, float],
    vertices: list[list[float]],
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    remove_mask: np.ndarray,
    target_mm: float,
    largest_half_width_mm: float,
) -> tuple[list[list[int]], dict[str, object]]:
    centre_yz = np.asarray([arm["center_y"], arm["center_z"]], dtype=float)
    radius = float(arm["radius"])
    triangles = original_vertices[original_faces]
    radial = np.linalg.norm(triangles[:, :, 1:3] - centre_yz, axis=2)
    spans = np.ptp(triangles[:, :, 0], axis=1)
    on_cylinder = np.all(np.abs(radial - radius) <= 0.75, axis=1)
    within_arm_span = np.all(
        (triangles[:, :, 0] >= arm["cylinder_x_min"] - 1.0)
        & (triangles[:, :, 0] <= arm["cylinder_x_max"] + 1.0),
        axis=1,
    )
    selected = (
        on_cylinder
        & within_arm_span
        & (spans >= 0.70 * abs(arm["cylinder_x_max"] - arm["cylinder_x_min"]))
    )
    selected_faces = original_faces[selected]
    if len(selected_faces) < 150:
        raise RuntimeError(f"{side}: expected coarse artificial-arm side faces, found {len(selected_faces)}")
    remove_mask |= selected

    components = edge_components(boundary_edges(selected_faces))
    if len(components) != 2:
        raise RuntimeError(f"{side}: expected two arm-side boundary loops, found {len(components)}")
    loops = [ordered_loop(component) for component in components]
    mean_x = [float(original_vertices[loop, 0].mean()) for loop in loops]
    far_target = arm["cylinder_x_min"] if side == "left" else arm["cylinder_x_max"]
    far_index = int(np.argmin(np.abs(np.asarray(mean_x) - far_target)))
    near_index = 1 - far_index
    near_loop = sort_loop_by_angle(loops[near_index], original_vertices, centre_yz)
    far_loop = sort_loop_by_angle(loops[far_index], original_vertices, centre_yz)

    sign = -1.0 if side == "left" else 1.0
    near_outward_x = (
        float(original_vertices[near_loop, 0].min())
        if side == "left"
        else float(original_vertices[near_loop, 0].max())
    )
    cut_x = near_outward_x + sign * 0.5
    preserved_faces, cut_loop = clip_side_at_plane(
        selected_faces, original_vertices, vertices, cut_x, sign
    )
    cut_loop = sort_loop_by_angle(cut_loop, np.asarray(vertices), centre_yz)
    regular_start_x = cut_x + sign * target_mm
    first_electrode_edge_x = arm["cut_x"] + sign * (20.0 - largest_half_width_mm)
    if sign * (first_electrode_edge_x - regular_start_x) < 0:
        raise RuntimeError(f"{side}: regular surface starts after the first electrode")
    far_x = float(np.mean(original_vertices[far_loop, 0]))
    regular_end_x = far_x - sign * target_mm
    if sign * (regular_end_x - regular_start_x) <= 0:
        raise RuntimeError(f"{side}: invalid regular arm interval")

    circumference_count = int(math.ceil(2 * math.pi * radius / target_mm))
    axial_intervals = int(math.ceil(abs(regular_end_x - regular_start_x) / target_mm))
    x_values = np.linspace(regular_start_x, regular_end_x, axial_intervals + 1)
    rings = [append_ring(vertices, float(x), centre_yz, radius, circumference_count) for x in x_values]

    new_faces: list[list[int]] = list(preserved_faces)
    new_faces.extend(connect_unequal_rings(cut_loop, rings[0], vertices, centre_yz))
    for first, second in zip(rings[:-1], rings[1:]):
        new_faces.extend(connect_equal_rings(first, second, vertices, centre_yz))
    new_faces.extend(connect_unequal_rings(rings[-1], far_loop, vertices, centre_yz))

    near_radial = np.linalg.norm(original_vertices[near_loop, 1:3] - centre_yz, axis=1)
    far_radial = np.linalg.norm(original_vertices[far_loop, 1:3] - centre_yz, axis=1)
    report = {
        "removed_coarse_faces": int(selected.sum()),
        "near_loop_vertices": int(len(near_loop)),
        "far_loop_vertices": int(len(far_loop)),
        "near_loop_x_range_mm": [float(original_vertices[near_loop, 0].min()), float(original_vertices[near_loop, 0].max())],
        "far_loop_x_range_mm": [float(original_vertices[far_loop, 0].min()), float(original_vertices[far_loop, 0].max())],
        "near_loop_radial_error_max_mm": float(np.max(np.abs(near_radial - radius))),
        "far_loop_radial_error_max_mm": float(np.max(np.abs(far_radial - radius))),
        "preserved_join_side_faces": int(len(preserved_faces)),
        "cut_loop_vertices": int(len(cut_loop)),
        "cut_plane_x_mm": float(cut_x),
        "regular_x_range_mm": [float(x_values.min()), float(x_values.max())],
        "regular_axial_intervals": axial_intervals,
        "regular_axial_step_max_mm": float(np.max(np.abs(np.diff(x_values)))),
        "regular_circumference_vertices": circumference_count,
        "regular_circumferential_chord_mm": float(2 * radius * math.sin(math.pi / circumference_count)),
        "new_faces": int(len(new_faces)),
    }
    return new_faces, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("preparation_report", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("qc_report", type=Path)
    parser.add_argument("--target-mm", type=float, default=1.0)
    parser.add_argument("--largest-electrode-half-width-mm", type=float, default=7.5)
    args = parser.parse_args()
    if args.target_mm <= 0:
        raise ValueError("--target-mm must be positive")

    records = json.loads(args.preparation_report.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        if "arm_extension" not in records:
            raise ValueError(
                "preparation report object has no arm_extension geometry"
            )
        body = records
    elif isinstance(records, list):
        body = next(
            (record for record in records
             if isinstance(record, dict) and record.get("name") == "body"),
            None,
        )
        if body is None:
            raise ValueError("legacy preparation report has no body record")
    else:
        raise ValueError("preparation report must be a JSON object or array")
    arms = body["arm_extension"]
    source_mesh = trimesh.load_mesh(args.source, process=False)
    # Binary STL stores triangle corners independently. Merge identical
    # coordinates before detecting the two topological boundary loops.
    source_mesh.merge_vertices()
    original_vertices = np.asarray(source_mesh.vertices, dtype=float)
    original_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    vertices = original_vertices.tolist()
    remove_mask = np.zeros(len(original_faces), dtype=bool)
    added_faces: list[list[int]] = []
    arm_reports: dict[str, object] = {}
    for side in ("left", "right"):
        faces, report = replace_arm_side(
            side,
            arms[side],
            vertices,
            original_vertices,
            original_faces,
            remove_mask,
            args.target_mm,
            args.largest_electrode_half_width_mm,
        )
        added_faces.extend(faces)
        arm_reports[side] = report

    faces = np.vstack((original_faces[~remove_mask], np.asarray(added_faces, dtype=np.int64)))
    output_mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=faces, process=False)
    trimesh.repair.fix_normals(output_mesh, multibody=False)
    output_mesh.remove_unreferenced_vertices()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.qc_report.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_bytes(trimesh.exchange.stl.export_stl(output_mesh))

    output_triangles = np.asarray(output_mesh.triangles)
    parameter_mask = np.zeros(len(output_triangles), dtype=bool)
    for side, sign in (("left", -1.0), ("right", 1.0)):
        arm = arms[side]
        centre_yz = np.asarray([arm["center_y"], arm["center_z"]])
        centres = []
        for inner in (20.0, 40.0):
            centres.extend((arm["cut_x"] + sign * inner, arm["cut_x"] + sign * (inner + 80.0), arm["cut_x"] + sign * (inner + 100.0)))
        axial = output_triangles[:, :, 0]
        radial = np.linalg.norm(output_triangles[:, :, 1:3] - centre_yz, axis=2)
        for centre in centres:
            parameter_mask |= np.any(np.abs(axial - centre) <= args.largest_electrode_half_width_mm, axis=1) & np.all(np.abs(radial - arm["radius"]) <= 0.8, axis=1)

    volume_change = (float(output_mesh.volume) - float(source_mesh.volume)) / float(source_mesh.volume)
    self_intersections = self_intersecting_face_count(args.target)
    report = {
        "status": "candidate_surface_not_yet_accepted_fem_mesh",
        "interpretation": (
            f"<={args.target_mm:g} mm structured surface on all allowed "
            "electrode bands; torso, joins and end caps preserved"
        ),
        "source": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source),
        "target": str(args.target.resolve()),
        "target_sha256": file_sha256(args.target),
        "target_mm": args.target_mm,
        "source_vertices": int(len(source_mesh.vertices)),
        "source_faces": int(len(source_mesh.faces)),
        "source_volume_l": float(source_mesh.volume / 1e6),
        "target_vertices": int(len(output_mesh.vertices)),
        "target_faces": int(len(output_mesh.faces)),
        "target_volume_l": float(output_mesh.volume / 1e6),
        "relative_volume_change": volume_change,
        "watertight": bool(output_mesh.is_watertight),
        "winding_consistent": bool(output_mesh.is_winding_consistent),
        "self_intersecting_faces": self_intersections,
        "euler_number": int(output_mesh.euler_number),
        "parameter_band_surface_faces": int(parameter_mask.sum()),
        "parameter_band_edge_stats_mm": surface_edge_stats(output_triangles[parameter_mask]),
        "arms": arm_reports,
    }
    report["surface_qc_pass"] = bool(
        report["watertight"]
        and report["winding_consistent"]
        and report["self_intersecting_faces"] == 0
        and abs(volume_change) <= 0.005
        and report["parameter_band_edge_stats_mm"]["max"] <= math.sqrt(2) * args.target_mm * 1.01
    )
    args.qc_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["surface_qc_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
