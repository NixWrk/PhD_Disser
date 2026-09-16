"""Bounded conforming refinement around the heart candidate region.

The parent is an immutable EIDORS ``prepared_fem.mat`` in metres. Candidate
regions are selected by vertex-AABB overlap. Every edge of every selected
tetrahedron is marked and the existing global edge-bisection primitive is used
unchanged. No boundary edge is allowed, so the original outer/contact surface
node IDs remain valid.

This module writes only a mesh artifact and a QC sidecar. It does not build
EIDORS matrices or run a solver.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import refine_surface_contact_mesh as _surface_refinement
except ModuleNotFoundError:
    from tools import refine_surface_contact_mesh as _surface_refinement

# The existing module imports numpy only inside its CLI entry point. Supplying
# its module-global name lets us reuse split_simplices without copying it.
_surface_refinement.np = np
split_simplices = _surface_refinement.split_simplices

TET_EDGE_PAIRS = tuple(itertools.combinations(range(4), 2))
CHECK_NAMES = (
    "original_vertices_unchanged",
    "outer_boundary_unchanged",
    "tissue_labels_inherited",
    "all_positive_tetrahedra",
    "all_marked_edge_incident_tetrahedra_split",
    "per_parent_volumes_preserved",
    "original_heart_volume_preserved",
    "local_new_node_faces_have_two_incident_tetrahedra",
    "heart_volume_centroid_scaling_explicit",
    "vertex_aabb_selection_used",
    "marked_edges_are_interior",
    "quality_and_local_edge_qc_recorded",
)


class RefinementError(ValueError):
    """Input or geometric QC failure."""


class BoundaryRefinementError(RefinementError):
    """The requested refinement would change the outer/contact surface."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefinementError(message)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _signed_tetra_volumes(
    nodes_m: np.ndarray, tets_zero_based: np.ndarray, chunk_size: int = 100_000
) -> np.ndarray:
    """Return signed tetrahedron volumes without materialising all vertices."""
    result = np.empty(len(tets_zero_based), dtype=np.float64)
    for start in range(0, len(tets_zero_based), chunk_size):
        stop = min(start + chunk_size, len(tets_zero_based))
        p = nodes_m[tets_zero_based[start:stop]]
        result[start:stop] = (
            np.einsum(
                "ij,ij->i",
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                p[:, 3] - p[:, 0],
            )
            / 6.0
        )
    return result


def _validate_mesh_arrays(
    nodes_m: np.ndarray,
    tets_zero_based: np.ndarray,
    boundary_zero_based: np.ndarray,
    tissue_id: np.ndarray,
) -> None:
    _require(
        nodes_m.ndim == 2 and nodes_m.shape[1] == 3 and np.isfinite(nodes_m).all(),
        "nodes_m must be a finite N-by-3 array",
    )
    _require(
        tets_zero_based.ndim == 2
        and tets_zero_based.shape[1] == 4
        and np.issubdtype(tets_zero_based.dtype, np.integer),
        "elements must be an integer E-by-4 array",
    )
    _require(
        boundary_zero_based.ndim == 2
        and boundary_zero_based.shape[1] == 3
        and np.issubdtype(boundary_zero_based.dtype, np.integer),
        "boundary must be an integer B-by-3 array",
    )
    _require(
        tissue_id.ndim == 1
        and len(tissue_id) == len(tets_zero_based)
        and np.issubdtype(tissue_id.dtype, np.integer),
        "tissue_id must contain one integer label per tetrahedron",
    )
    _require(len(tets_zero_based) > 0 and len(boundary_zero_based) > 0, "Mesh is empty")
    _require(
        int(tets_zero_based.min()) >= 0
        and int(tets_zero_based.max()) < len(nodes_m)
        and int(boundary_zero_based.min()) >= 0
        and int(boundary_zero_based.max()) < len(nodes_m),
        "Mesh connectivity is outside the node range",
    )
    _require(
        np.all(_signed_tetra_volumes(nodes_m, tets_zero_based) > 0),
        "Input tetrahedra must have positive determinants",
    )
    _require(np.any(tissue_id == 3), "Input has no tissue_id == 3 heart tetrahedra")


def _edge_codes(edges: np.ndarray, node_count: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    return edges[:, 0] * np.int64(node_count) + edges[:, 1]


def _unique_edges_from_tets(tets_zero_based: np.ndarray) -> np.ndarray:
    parts = [tets_zero_based[:, [a, b]] for a, b in TET_EDGE_PAIRS]
    edges = np.sort(np.concatenate(parts, axis=0).astype(np.int64), axis=1)
    return np.unique(edges, axis=0)


def _boundary_edges(boundary_zero_based: np.ndarray) -> np.ndarray:
    parts = [
        boundary_zero_based[:, [0, 1]],
        boundary_zero_based[:, [0, 2]],
        boundary_zero_based[:, [1, 2]],
    ]
    return np.unique(
        np.sort(np.concatenate(parts, axis=0).astype(np.int64), axis=1), axis=0
    )


def _incident_tet_mask(
    tets_zero_based: np.ndarray, marked_edges: np.ndarray, node_count: int
) -> np.ndarray:
    """Find every parent tetrahedron containing at least one marked edge."""
    marked = _edge_codes(marked_edges, node_count)
    hit = np.zeros(len(tets_zero_based), dtype=bool)
    for a, b in TET_EDGE_PAIRS:
        pair = np.sort(tets_zero_based[:, [a, b]].astype(np.int64), axis=1)
        hit |= np.isin(_edge_codes(pair, node_count), marked)
    return hit


def _local_new_node_face_counts(
    tets_zero_based: np.ndarray, new_node_start: int
) -> np.ndarray:
    """Count only faces containing appended midpoint nodes."""
    with_new_node = np.any(tets_zero_based >= new_node_start, axis=1)
    if not np.any(with_new_node):
        return np.empty(0, dtype=np.int64)
    local = tets_zero_based[with_new_node]
    faces = np.concatenate(
        [
            local[:, [1, 2, 3]],
            local[:, [0, 2, 3]],
            local[:, [0, 1, 3]],
            local[:, [0, 1, 2]],
        ]
    )
    faces = faces[np.any(faces >= new_node_start, axis=1)]
    faces.sort(axis=1)
    _unique, counts = np.unique(faces, axis=0, return_counts=True)
    return counts


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(
        len(values) > 0 and np.isfinite(values).all(),
        "QC distribution is empty or nonfinite",
    )
    q01, q05, q50, q95, q99 = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p01": float(q01),
        "p05": float(q05),
        "median": float(q50),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(values.max()),
    }


def _tetra_quality(nodes_m: np.ndarray, tets_zero_based: np.ndarray) -> np.ndarray:
    """Return unit-normalised mean-ratio tetrahedral quality in (0, 1]."""
    result = np.empty(len(tets_zero_based), dtype=np.float64)
    for start in range(0, len(tets_zero_based), 100_000):
        stop = min(start + 100_000, len(tets_zero_based))
        p = nodes_m[tets_zero_based[start:stop]]
        volume = np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        ) / 6.0
        edge_sum = np.zeros(len(p), dtype=np.float64)
        for a, b in TET_EDGE_PAIRS:
            delta = p[:, a] - p[:, b]
            edge_sum += np.einsum("ij,ij->i", delta, delta)
        result[start:stop] = 12.0 * np.power(3.0 * volume, 2.0 / 3.0) / edge_sum
    return result


def _edge_length_distribution(nodes_m: np.ndarray, tets_zero_based: np.ndarray) -> dict:
    edges = _unique_edges_from_tets(tets_zero_based)
    lengths_mm = np.linalg.norm(
        nodes_m[edges[:, 0]] - nodes_m[edges[:, 1]], axis=1
    ) * 1000.0
    return _distribution(lengths_mm)


def _volume_by_label(volumes_m3: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    upper = max(4, int(labels.max()))
    sums = np.bincount(labels, weights=volumes_m3, minlength=upper + 1)
    return {str(label): float(sums[label]) for label in range(1, upper + 1)}


def _validate_local_refinement(
    old_nodes: np.ndarray,
    old_tets: np.ndarray,
    old_boundary: np.ndarray,
    old_labels: np.ndarray,
    new_nodes: np.ndarray,
    new_tets: np.ndarray,
    new_boundary: np.ndarray,
    new_labels: np.ndarray,
    new_parent_ids: np.ndarray,
    child_parent_zero: np.ndarray,
    marked_edges: np.ndarray,
    expected_incident: np.ndarray,
    old_volumes: np.ndarray,
) -> dict:
    old_node_count = len(old_nodes)
    old_tet_count = len(old_tets)
    _require(np.array_equal(new_nodes[:old_node_count], old_nodes), "Original node prefix changed")
    _require(np.array_equal(new_boundary, old_boundary), "Outer boundary changed")
    _require(
        np.array_equal(new_labels, old_labels[child_parent_zero]),
        "Tissue labels were not inherited from parent tetrahedra",
    )
    _require(
        np.array_equal(new_parent_ids, new_parent_ids.astype(np.int64))
        and np.all(new_parent_ids >= 1),
        "Invalid one-based parent_tet_id",
    )
    _require(
        np.array_equal(np.unique(child_parent_zero), np.arange(old_tet_count, dtype=np.int64)),
        "Some current parent tetrahedron has no child",
    )

    signed = _signed_tetra_volumes(new_nodes, new_tets)
    _require(np.all(signed > 0), "Refinement created a nonpositive tetrahedron")
    children_per_parent = np.bincount(child_parent_zero, minlength=old_tet_count)
    actual_incident = children_per_parent > 1
    _require(
        np.array_equal(actual_incident, expected_incident),
        "Not every incident tetrahedron was split consistently",
    )
    per_parent = np.bincount(child_parent_zero, weights=signed, minlength=old_tet_count)
    _require(
        np.allclose(per_parent, old_volumes, rtol=1e-10, atol=1e-18),
        "Refinement changed a parent tetrahedron volume",
    )

    expected_midpoints = (
        old_nodes[marked_edges[:, 0]] + old_nodes[marked_edges[:, 1]]
    ) / 2.0
    _require(
        np.allclose(new_nodes[old_node_count:], expected_midpoints, rtol=0.0, atol=1e-15),
        "Appended nodes are not marked-edge midpoints",
    )
    local_face_counts = _local_new_node_face_counts(new_tets, old_node_count)
    _require(
        len(local_face_counts) > 0 and np.all(local_face_counts == 2),
        "Refinement has a hanging or nonmanifold local face",
    )

    local_old_tets = old_tets[expected_incident]
    local_new_tets = new_tets[
        np.isin(child_parent_zero, np.flatnonzero(expected_incident))
    ]
    volume_before = _volume_by_label(old_volumes, old_labels)
    volume_after = _volume_by_label(signed, new_labels)
    original_heart_error = abs(volume_after["3"] - volume_before["3"])
    checks = {
        "original_vertices_unchanged": True,
        "outer_boundary_unchanged": True,
        "tissue_labels_inherited": True,
        "all_positive_tetrahedra": True,
        "all_marked_edge_incident_tetrahedra_split": True,
        "per_parent_volumes_preserved": True,
        "original_heart_volume_preserved": True,
        "local_new_node_faces_have_two_incident_tetrahedra": True,
        "quality_and_local_edge_qc_recorded": True,
    }
    metrics = {
        "old_nodes": int(len(old_nodes)),
        "new_nodes": int(len(new_nodes)),
        "old_tetrahedra": int(len(old_tets)),
        "new_tetrahedra": int(len(new_tets)),
        "marked_edges": int(len(marked_edges)),
        "split_parent_tetrahedra": int(np.count_nonzero(actual_incident)),
        "minimum_signed_tetra_volume_m3": float(signed.min()),
        "maximum_parent_volume_error_m3": float(np.max(np.abs(per_parent - old_volumes))),
        "volume_by_tissue_id_m3": {"before": volume_before, "after": volume_after},
        "original_heart_volume_m3": {
            "before": volume_before["3"],
            "after": volume_after["3"],
            "absolute_error": original_heart_error,
        },
        "quality_distribution": {
            "before": _distribution(_tetra_quality(old_nodes, old_tets)),
            "after": _distribution(_tetra_quality(new_nodes, new_tets)),
        },
        "local_edge_length_mm": {
            "before_affected_parent_tets": _edge_length_distribution(old_nodes, local_old_tets),
            "after_refined_children": _edge_length_distribution(new_nodes, local_new_tets),
            "marked_edges_before": _distribution(
                np.linalg.norm(
                    old_nodes[marked_edges[:, 0]] - old_nodes[marked_edges[:, 1]], axis=1
                )
                * 1000.0
            ),
        },
    }
    return {"checks": checks, "metrics": metrics}


def refine_marked_edges(
    nodes_m: np.ndarray,
    tets_zero_based: np.ndarray,
    boundary_zero_based: np.ndarray,
    tissue_id: np.ndarray,
    marked_edges: np.ndarray,
    parent_tet_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Refine one level using the unchanged global midpoint splitter."""
    nodes = np.asarray(nodes_m, dtype=np.float64)
    tets = np.asarray(tets_zero_based, dtype=np.int64)
    boundary = np.asarray(boundary_zero_based, dtype=np.int64)
    labels = np.asarray(tissue_id).reshape(-1).astype(np.int64, copy=False)
    _validate_mesh_arrays(nodes, tets, boundary, labels)

    edges = np.asarray(marked_edges, dtype=np.int64)
    _require(edges.ndim == 2 and edges.shape[1] == 2 and len(edges) > 0, "No marked edges")
    edges = np.sort(edges, axis=1)
    _require(
        int(edges.min()) >= 0 and int(edges.max()) < len(nodes),
        "Marked edge is outside the node range",
    )
    if np.any(edges[:, 0] == edges[:, 1]):
        raise RefinementError("Marked edge has identical endpoints")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise RefinementError("Marked edges are not unique")

    boundary_edge_codes = _edge_codes(_boundary_edges(boundary), len(nodes))
    marked_codes = _edge_codes(edges, len(nodes))
    boundary_hits = np.flatnonzero(np.isin(marked_codes, boundary_edge_codes))
    if len(boundary_hits):
        raise BoundaryRefinementError(
            "Marked edges touch the outer/contact boundary; rerun explicitly with a "
            f"smaller collar. offending_edges={len(boundary_hits)}"
        )

    expected_incident = _incident_tet_mask(tets, edges, len(nodes))
    _require(np.any(expected_incident), "Marked edges are not incident to an input tetrahedron")
    old_volumes = _signed_tetra_volumes(nodes, tets)
    if parent_tet_id is None:
        input_parent_ids = np.arange(1, len(tets) + 1, dtype=np.int64)
    else:
        input_parent_ids = np.asarray(parent_tet_id, dtype=np.int64).reshape(-1)
        _require(
            len(input_parent_ids) == len(tets) and np.all(input_parent_ids >= 1),
            "parent_tet_id must be one-based and match the input tetrahedra",
        )

    child_tets, child_parent_zero = split_simplices(tets, edges, len(nodes))
    child_tets = np.asarray(child_tets, dtype=np.int64)
    child_parent_zero = np.asarray(child_parent_zero, dtype=np.int64)
    child_labels = labels[child_parent_zero]
    child_parent_ids = input_parent_ids[child_parent_zero]
    # Boundary edges were rejected, so this is an exact copy. The legacy
    # splitter has no empty-child-width case for an unaffected boundary array.
    child_boundary = boundary.copy()
    child_nodes = np.vstack(
        [nodes, (nodes[edges[:, 0]] + nodes[edges[:, 1]]) / 2.0]
    )

    qc = _validate_local_refinement(
        nodes,
        tets,
        boundary,
        labels,
        child_nodes,
        child_tets,
        child_boundary,
        child_labels,
        child_parent_ids,
        child_parent_zero,
        edges,
        expected_incident,
        old_volumes,
    )
    return child_nodes, child_tets, child_boundary, child_labels, child_parent_ids, qc


def _candidate_shape(
    candidate: dict, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        centre = np.asarray(candidate["centre_m"], dtype=np.float64)
        axes = np.asarray(candidate["semiaxes_m"], dtype=np.float64)
        rotation = np.asarray(candidate["axes_columns"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RefinementError(f"Invalid {name} candidate parameters") from exc
    _require(
        centre.shape == (3,)
        and axes.shape == (3,)
        and rotation.shape == (3, 3)
        and np.isfinite(centre).all()
        and np.isfinite(axes).all()
        and np.isfinite(rotation).all()
        and np.all(axes > 0),
        f"Invalid {name} candidate parameter shapes or values",
    )
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-10),
        f"{name} axes_columns must be orthonormal",
    )
    return centre, axes, rotation


def _read_candidates(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    nested = value.get("candidates", {})
    _require(isinstance(nested, dict), "candidate_parameters.candidates must be an object")
    candidates = dict(nested)
    for key in ("originalheart", "original_heart", "sphere", "ellipsoid", "ellipse"):
        if key in value:
            candidates.setdefault(key, value[key])
    if "originalheart" not in candidates:
        candidates["originalheart"] = candidates.get("original_heart")
    if "ellipsoid" not in candidates:
        candidates["ellipsoid"] = candidates.get("ellipse")
    _require(
        isinstance(candidates.get("originalheart"), dict)
        and isinstance(candidates.get("sphere"), dict)
        and isinstance(candidates.get("ellipsoid"), dict),
        "candidate_parameters.json must contain originalheart, sphere, and ellipsoid",
    )
    return candidates


def build_candidate_boxes(
    nodes_m: np.ndarray,
    tets_zero_based: np.ndarray,
    tissue_id: np.ndarray,
    candidates: dict,
    max_volume_fraction: float = 0.01,
) -> list[dict]:
    """Build nominal and +/- volume-scaled boxes for all three geometries."""
    _require(
        np.isfinite(max_volume_fraction) and 0 <= max_volume_fraction < 1,
        "max_volume_fraction must be in [0, 1)",
    )
    labels = np.asarray(tissue_id).reshape(-1)
    heart_tets = np.flatnonzero(labels == 3)
    _require(len(heart_tets) > 0, "Input has no tissue_id == 3 heart tetrahedra")
    heart_vertices = nodes_m[np.unique(tets_zero_based[heart_tets])]
    heart_volumes = _signed_tetra_volumes(nodes_m, tets_zero_based[heart_tets])
    source_centroid = (
        heart_volumes[:, None]
        * nodes_m[tets_zero_based[heart_tets]].mean(axis=1)
    ).sum(axis=0) / heart_volumes.sum()

    original = candidates["originalheart"]
    declared_centre = np.asarray(original.get("centre_m"), dtype=np.float64)
    _require(
        declared_centre.shape == (3,) and np.isfinite(declared_centre).all(),
        "originalheart must declare centre_m",
    )
    _require(
        np.allclose(declared_centre, source_centroid, rtol=0.0, atol=1e-7),
        "originalheart scaling centre is not the input heart volume centroid",
    )

    boxes: list[dict] = []
    fractions = (-max_volume_fraction, 0.0, max_volume_fraction)
    for signed_fraction in fractions:
        scale = (1.0 + signed_fraction) ** (1.0 / 3.0)
        transformed = declared_centre + scale * (heart_vertices - declared_centre)
        boxes.append(
            {
                "geometry": "originalheart",
                "signed_volume_fraction": float(signed_fraction),
                "lower_m": transformed.min(axis=0),
                "upper_m": transformed.max(axis=0),
                "scale_about_volume_centroid": float(scale),
            }
        )

    for geometry in ("sphere", "ellipsoid"):
        centre, axes, rotation = _candidate_shape(candidates[geometry], geometry)
        for signed_fraction in fractions:
            scale = (1.0 + signed_fraction) ** (1.0 / 3.0)
            extent = np.linalg.norm(rotation * (axes * scale), axis=1)
            boxes.append(
                {
                    "geometry": geometry,
                    "signed_volume_fraction": float(signed_fraction),
                    "lower_m": centre - extent,
                    "upper_m": centre + extent,
                    "scale_about_volume_centroid": float(scale),
                }
            )
    return boxes


def select_tetrahedra_by_vertex_aabb(
    nodes_m: np.ndarray,
    tets_zero_based: np.ndarray,
    boxes: Sequence[dict],
    collar_mm: float = 15.0,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Select input tetrahedra whose vertex AABB overlaps any expanded box."""
    _require(np.isfinite(collar_mm) and collar_mm >= 0, "collar_mm must be nonnegative")
    _require(chunk_size > 0, "chunk_size must be positive")
    collar_m = float(collar_mm) / 1000.0
    expanded = [
        (
            np.asarray(box["lower_m"], dtype=np.float64) - collar_m,
            np.asarray(box["upper_m"], dtype=np.float64) + collar_m,
        )
        for box in boxes
    ]
    selected: list[np.ndarray] = []
    for start in range(0, len(tets_zero_based), chunk_size):
        stop = min(start + chunk_size, len(tets_zero_based))
        vertices = nodes_m[tets_zero_based[start:stop]]
        lower = vertices.min(axis=1)
        upper = vertices.max(axis=1)
        hit = np.zeros(len(vertices), dtype=bool)
        for box_lower, box_upper in expanded:
            hit |= np.all((upper >= box_lower) & (lower <= box_upper), axis=1)
        if np.any(hit):
            selected.append(start + np.flatnonzero(hit))
    _require(selected, "Candidate union does not overlap any input tetrahedron")
    return np.concatenate(selected).astype(np.int64, copy=False)


def refine_heart_mesh(
    nodes_m: np.ndarray,
    tets_zero_based: np.ndarray,
    boundary_zero_based: np.ndarray,
    tissue_id: np.ndarray,
    candidates: dict,
    collar_mm: float = 15.0,
    max_volume_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select and refine one bounded heart-neighbourhood level."""
    nodes = np.asarray(nodes_m, dtype=np.float64)
    tets = np.asarray(tets_zero_based, dtype=np.int64)
    boundary = np.asarray(boundary_zero_based, dtype=np.int64)
    labels = np.asarray(tissue_id).reshape(-1).astype(np.int64, copy=False)
    _validate_mesh_arrays(nodes, tets, boundary, labels)
    boxes = build_candidate_boxes(nodes, tets, labels, candidates, max_volume_fraction)
    selected = select_tetrahedra_by_vertex_aabb(nodes, tets, boxes, collar_mm)
    marked_edges = _unique_edges_from_tets(tets[selected])
    new_nodes, new_tets, new_boundary, new_labels, parents, qc = refine_marked_edges(
        nodes, tets, boundary, labels, marked_edges
    )
    checks = dict(qc["checks"])
    checks.update(
        {
            "heart_volume_centroid_scaling_explicit": True,
            "vertex_aabb_selection_used": True,
            "marked_edges_are_interior": True,
        }
    )
    metrics = dict(qc["metrics"])
    metrics.update(
        {
            "selected_tetrahedra_by_vertex_aabb": int(len(selected)),
            "candidate_box_count": int(len(boxes)),
            "collar_mm": float(collar_mm),
            "max_volume_fraction": float(max_volume_fraction),
        }
    )
    return new_nodes, new_tets, new_boundary, new_labels, parents, {
        "selected_tetrahedra_zero_based": selected,
        "marked_edges_zero_based": marked_edges,
        "candidate_boxes": [
            {
                **box,
                "lower_m": np.asarray(box["lower_m"]).tolist(),
                "upper_m": np.asarray(box["upper_m"]).tolist(),
            }
            for box in boxes
        ],
        "checks": checks,
        "metrics": metrics,
    }


def _matlab_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").replace("\x00", "").strip()
    array = np.asarray(value)
    if array.dtype.kind in "ui":
        return "".join(chr(int(item)) for item in array.reshape(-1) if int(item)).strip()
    if array.dtype.kind == "S":
        return b"".join(bytes(item) for item in array.reshape(-1)).decode(
            "utf-8", errors="strict"
        ).replace("\x00", "").strip()
    if array.dtype.kind == "U":
        return "".join(str(item) for item in array.reshape(-1)).replace("\x00", "").strip()
    return str(value).replace("\x00", "").strip()


def load_prepared_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MATLAB v7.3 prepared FEM arrays; connectivity becomes zero-based."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - deployment diagnostic
        raise RefinementError("h5py is required to read prepared_fem.mat") from exc
    try:
        with h5py.File(path, "r") as handle:
            _require("coordinate_units" in handle, "prepared_fem.mat lacks coordinate_units")
            units = _matlab_string(handle["coordinate_units"][()])
            _require(
                units == "m",
                f"prepared_fem.mat coordinate_units must be 'm', got {units!r}",
            )
            nodes = np.asarray(handle["fmdl_m/nodes"][()]).T.astype(np.float64)
            tets = np.asarray(handle["fmdl_m/elems"][()]).T.astype(np.int64) - 1
            boundary = (
                np.asarray(handle["fmdl_m/boundary"][()]).T.astype(np.int64) - 1
            )
            labels = np.asarray(handle["tissue_id"][()]).reshape(-1).astype(np.int64)
    except RefinementError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise RefinementError(
            "source-prepared must contain fmdl_m/nodes, fmdl_m/elems, "
            "fmdl_m/boundary, and tissue_id in MATLAB v7.3 layout"
        ) from exc
    _validate_mesh_arrays(nodes, tets, boundary, labels)
    return nodes, tets, boundary, labels


def _source_file_for_module(module) -> Path:
    path = Path(module.__file__).resolve()
    source = path.with_suffix(".py") if path.suffix == ".pyc" else path
    return source if source.is_file() else path


def provenance() -> dict:
    implementation = Path(__file__).resolve()
    splitter = _source_file_for_module(_surface_refinement)
    return {
        "implementation": {"path": str(implementation), "sha256": sha256(implementation)},
        "split_simplices_source": {
            "function": "refine_surface_contact_mesh.split_simplices",
            "path": str(splitter),
            "sha256": sha256(splitter),
        },
    }


def _empty_checks() -> dict:
    return {name: False for name in CHECK_NAMES}


def _write_failure_report(
    output_dir: Path,
    source: Path,
    candidate_path: Path,
    error: Exception,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source) if source.is_file() else None
    candidate_hash = sha256(candidate_path) if candidate_path.is_file() else None
    report = {
        "schema": "trkg4_heart_volume_refinement_v1",
        "status": "failed_heart_volume_refinement",
        "source_prepared": str(source),
        "source_prepared_sha256": source_hash,
        "candidate_parameters": str(candidate_path),
        "candidate_parameters_sha256": candidate_hash,
        "refined_mesh_sha256": None,
        "checks": _empty_checks(),
        "provenance": provenance(),
        "error": f"{type(error).__name__}: {error}",
        "physical_validation": False,
        "spatial_solution_convergence": False,
    }
    report_path = output_dir / "refinement_report.json"
    if not report_path.exists():
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def _write_output(
    output_dir: Path,
    source: Path,
    candidates_path: Path,
    new_nodes: np.ndarray,
    new_tets: np.ndarray,
    new_boundary: np.ndarray,
    new_labels: np.ndarray,
    parent_ids: np.ndarray,
    audit: dict,
) -> dict:
    from scipy.io import savemat

    checks = dict(audit["checks"])
    _require(
        set(checks) >= set(CHECK_NAMES)
        and all(isinstance(value, (bool, np.bool_)) for value in checks.values())
        and all(bool(value) for value in checks.values()),
        "Successful output requires a boolean all-true checks dictionary",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    mesh_path = output_dir / "refined_mesh.mat"
    source_hash = sha256(source)
    candidate_hash = sha256(candidates_path)
    savemat(
        mesh_path,
        {
            "nodes_m": new_nodes,
            "elems": new_tets + 1,
            "boundary": new_boundary + 1,
            "tissue_id": new_labels[:, None],
            "parent_tet_id": parent_ids[:, None],
            "source_prepared_sha256": source_hash,
        },
        do_compression=False,
    )
    metadata = {
        "schema": "trkg4_heart_volume_refinement_v1",
        "status": "passed_heart_volume_refinement",
        "source_prepared": str(source.resolve()),
        "source_prepared_sha256": source_hash,
        "candidate_parameters": str(candidates_path.resolve()),
        "candidate_parameters_sha256": candidate_hash,
        "refined_mesh": "refined_mesh.mat",
        "coordinates": "m",
        "connectivity_in_output": "one_based",
        "tissue_id_heart": 3,
        "mesh": {
            "nodes": int(len(new_nodes)),
            "tetrahedra": int(len(new_tets)),
            "boundary_triangles": int(len(new_boundary)),
        },
        "parent_tet_id": {
            "base": "one_based_input_tetrahedron_id",
            "unique_parents": int(len(np.unique(parent_ids))),
        },
        "checks": {name: bool(value) for name, value in checks.items()},
        "metrics": audit["metrics"],
        "candidate_boxes": audit["candidate_boxes"],
        "provenance": provenance(),
        "refined_mesh_sha256": sha256(mesh_path),
        "physical_validation": False,
        "spatial_solution_convergence": False,
    }
    (output_dir / "refinement_report.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def run(
    source_prepared: Path,
    candidate_parameters: Path,
    output_dir: Path,
    collar_mm: float = 15.0,
    max_volume_fraction: float = 0.01,
) -> dict:
    source = Path(source_prepared).resolve()
    candidate_path = Path(candidate_parameters).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"Refusing to overwrite existing output directory: {output}")
    try:
        _require(source.is_file(), f"Missing source-prepared file: {source}")
        _require(candidate_path.is_file(), f"Missing candidate-parameters file: {candidate_path}")
        nodes, tets, boundary, labels = load_prepared_mesh(source)
        candidates = _read_candidates(candidate_path)
        new_nodes, new_tets, new_boundary, new_labels, parents, audit = refine_heart_mesh(
            nodes,
            tets,
            boundary,
            labels,
            candidates,
            collar_mm=collar_mm,
            max_volume_fraction=max_volume_fraction,
        )
        return _write_output(
            output,
            source,
            candidate_path,
            new_nodes,
            new_tets,
            new_boundary,
            new_labels,
            parents,
            audit,
        )
    except Exception as error:
        return _write_failure_report(output, source, candidate_path, error)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prepared", required=True, type=Path)
    parser.add_argument("--candidate-parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--collar-mm", type=float, default=15.0)
    parser.add_argument("--max-volume-fraction", type=float, default=0.01)
    args = parser.parse_args(argv)
    metadata = run(
        args.source_prepared,
        args.candidate_parameters,
        args.output_dir,
        collar_mm=args.collar_mm,
        max_volume_fraction=args.max_volume_fraction,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    if metadata["status"] != "passed_heart_volume_refinement":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
