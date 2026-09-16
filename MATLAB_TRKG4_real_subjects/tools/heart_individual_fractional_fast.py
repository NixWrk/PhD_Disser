"""Conservative acceleration wrapper for individual-heart quadrature.

This module deliberately leaves the frozen
``heart_individual_fractional_materials`` implementation unchanged.  It
inverse-maps query vertices into the original-heart frame, then works in
bounded query chunks.  A query tetrahedron is assigned an exact zero only
when its conservative AABB is disjoint from every original heart-tetrahedron
AABB.  It is assigned an exact one only when all four vertices are strictly
inside one and the same original tetrahedron; convexity then proves that the
whole query tetrahedron is inside that tetrahedron.  Every other query is
passed to the frozen integrator with the original transform and the same
shared barycentric points.

The AABB margins make the broad phase conservative: false candidates are
allowed, missed intersections are not.  Positive-volume overlap assumptions
and floating-point boundary conventions remain those of
``TetrahedralHeart``.  The wrapper performs no FEM work and no all-pairs
tetrahedron-overlap validation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from heart_fractional_materials import uniform_tet_barycentric
    from heart_individual_fractional_materials import (
        TetrahedralHeart,
        integrate_individual_fractions,
    )
except ImportError:  # pragma: no cover - package import fallback
    from .heart_fractional_materials import uniform_tet_barycentric
    from .heart_individual_fractional_materials import (
        TetrahedralHeart,
        integrate_individual_fractions,
    )


__all__ = ["integrate_individual_fractions_fast"]

_EPS = np.finfo(np.float64).eps
_AABB_MARGIN_ULPS = 256.0
_BARYCENTRIC_MARGIN_ULPS = 256.0
_DEGENERATE_MARGIN_ULPS = 4096.0


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_query_tetrahedra(vertices_m: Any) -> np.ndarray:
    vertices = _finite_array(vertices_m, name="vertices_m")
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError("vertices_m must have shape (N, 4, 3)")
    if len(vertices) == 0:
        raise ValueError("vertices_m must contain at least one tetrahedron")
    edges = vertices[:, 1:] - vertices[:, :1]
    determinants = np.einsum(
        "ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])
    )
    edge_scale = np.max(np.abs(edges), axis=(1, 2))
    threshold = _DEGENERATE_MARGIN_ULPS * _EPS * np.maximum(edge_scale, np.finfo(float).tiny) ** 3
    if np.any(~np.isfinite(determinants)) or np.any(np.abs(determinants) <= threshold):
        first = int(np.flatnonzero(~np.isfinite(determinants) | (np.abs(determinants) <= threshold))[0])
        raise ValueError(f"vertices_m contains a degenerate tetrahedron at index {first}")
    return np.array(vertices, dtype=np.float64, copy=True)


def _validate_levels(levels: Any) -> tuple[int, ...]:
    try:
        parsed = tuple(levels)
    except TypeError as exc:
        raise ValueError("levels must be a non-empty sequence of powers of two") from exc
    if not parsed or any(isinstance(level, bool) or not isinstance(level, (int, np.integer)) for level in parsed):
        raise ValueError("levels must be a non-empty sequence of integer powers of two")
    parsed = tuple(int(level) for level in parsed)
    if tuple(sorted(set(parsed))) != parsed or any(level <= 0 or level & (level - 1) for level in parsed):
        raise ValueError("levels must be strictly increasing powers of two")
    return parsed


def _validate_barycentric(barycentric: Any, maximum_level: int) -> np.ndarray:
    bary = _finite_array(barycentric, name="barycentric")
    if bary.shape != (maximum_level, 4):
        raise ValueError(f"barycentric must have shape ({maximum_level}, 4)")
    if np.any(bary < 0.0) or not np.allclose(bary.sum(axis=1), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("barycentric rows must be non-negative and sum to one")
    return np.array(bary, dtype=np.float64, copy=False)


def _conservative_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    margin = _AABB_MARGIN_ULPS * _EPS * scale
    try:
        with np.errstate(over="raise", invalid="raise"):
            safe_lower = np.nextafter(lower - margin, -np.inf)
            safe_upper = np.nextafter(upper + margin, np.inf)
    except FloatingPointError as exc:
        raise ValueError("AABB expansion produced non-finite bounds") from exc
    if not np.isfinite(safe_lower).all() or not np.isfinite(safe_upper).all():
        raise ValueError("AABB expansion produced non-finite bounds")
    return safe_lower, safe_upper


def _heart_barycentric_data(heart: TetrahedralHeart) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(heart.vertices_m, dtype=np.float64)
    v0 = vertices[:, 0]
    edges = vertices[:, 1:] - v0[:, None, :]
    try:
        inverse_edges = np.linalg.inv(edges)
        condition = np.linalg.cond(edges)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - constructor guards this
        raise ValueError("heart contains a singular tetrahedron") from exc
    tolerance = _BARYCENTRIC_MARGIN_ULPS * _EPS * np.maximum(1.0, condition)
    tolerance = np.minimum(tolerance, 1e-8)
    lower, upper = _conservative_bounds(vertices.min(axis=1), vertices.max(axis=1))
    return v0, inverse_edges, tolerance, lower, upper


def _inverse_map_vertices(
    vertices: np.ndarray,
    centre_m: np.ndarray,
    scale: float,
    translation_m: np.ndarray,
) -> np.ndarray:
    if scale == 1.0 and np.array_equal(translation_m, np.zeros(3)):
        return np.array(vertices, dtype=np.float64, copy=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mapped = centre_m + (vertices - translation_m - centre_m) / scale
    except FloatingPointError as exc:
        raise ValueError("inverse heart transform produced non-finite vertices") from exc
    if not np.isfinite(mapped).all():
        raise ValueError("inverse heart transform produced non-finite vertices")
    return mapped


def integrate_individual_fractions_fast(
    vertices_m: Any,
    heart: TetrahedralHeart,
    volume_fraction: float = 0.0,
    translation_m: Any = (0.0, 0.0, 0.0),
    levels: Any = (512, 4096),
    barycentric: Any = None,
    batch_size: int = 32,
    chunk_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accelerate frozen individual-heart fractions with safe classifications.

    The API matches ``integrate_individual_fractions`` and adds
    ``chunk_size`` for the temporary query-AABB block.  Returned fractions have
    shape ``(len(levels), len(vertices_m))``.  AABB-empty and strict-same-tet
    rows are filled without quadrature; all partial rows use the frozen
    integrator, so the quadrature rule and transform semantics are unchanged.
    """

    if not isinstance(heart, TetrahedralHeart):
        raise TypeError("heart must be a TetrahedralHeart")
    query_vertices = _validate_query_tetrahedra(vertices_m)
    parsed_levels = _validate_levels(levels)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")
    batch_size = int(batch_size)
    chunk_size = int(chunk_size)

    fraction_value = _finite_array(volume_fraction, name="volume_fraction")
    if fraction_value.ndim != 0:
        raise ValueError("volume_fraction must be a finite scalar")
    volume_fraction = float(fraction_value)
    if 1.0 + volume_fraction <= 0.0:
        raise ValueError("volume_fraction must be greater than -1")
    translation = _finite_array(translation_m, name="translation_m")
    if translation.shape != (3,):
        raise ValueError("translation_m must have shape (3,)")

    maximum_level = parsed_levels[-1]
    if barycentric is None:
        bary = uniform_tet_barycentric(maximum_level)
    else:
        bary = _validate_barycentric(barycentric, maximum_level)

    scale = float(np.cbrt(1.0 + volume_fraction))
    centre = np.asarray(heart.centre_m, dtype=np.float64)
    mapped_vertices = _inverse_map_vertices(query_vertices, centre, scale, translation)
    v0, inverse_edges, bary_tolerance, heart_lower, heart_upper = _heart_barycentric_data(heart)

    query_count = len(query_vertices)
    heart_count = len(heart.vertices_m)
    exact_empty = np.zeros(query_count, dtype=bool)
    exact_full = np.zeros(query_count, dtype=bool)
    aabb_pair_tests = 0
    aabb_overlap_pairs = 0
    full_vertex_tests = 0

    # The only dense temporary is (chunk_size, heart_count), bounded by the
    # default 256 x ~1963 query requested for the real mesh.
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        block = mapped_vertices[start:stop]
        query_lower, query_upper = _conservative_bounds(block.min(axis=1), block.max(axis=1))
        overlap = np.all(
            (query_upper[:, None, :] >= heart_lower[None, :, :])
            & (query_lower[:, None, :] <= heart_upper[None, :, :]),
            axis=2,
        )
        aabb_pair_tests += int((stop - start) * heart_count)
        aabb_overlap_pairs += int(np.count_nonzero(overlap))
        overlaps_any = np.any(overlap, axis=1)
        exact_empty[start:stop] = ~overlaps_any
        local_candidates = np.flatnonzero(overlaps_any)
        if len(local_candidates) == 0:
            continue

        candidate_hearts = np.flatnonzero(np.any(overlap, axis=0))
        for heart_index in candidate_hearts:
            eligible = local_candidates[overlap[local_candidates, heart_index]]
            if len(eligible) == 0:
                continue
            global_eligible = start + eligible
            eligible = eligible[~exact_full[global_eligible]]
            if len(eligible) == 0:
                continue
            full_vertex_tests += int(len(eligible) * 4)
            points = block[eligible]
            barycentric_vertices = np.einsum(
                "qvi,ij->qvj",
                points - v0[heart_index],
                inverse_edges[heart_index],
                optimize=True,
            )
            lambda0 = 1.0 - barycentric_vertices.sum(axis=2)
            tolerance = bary_tolerance[heart_index]
            strictly_inside = np.all(barycentric_vertices > tolerance, axis=(1, 2)) & np.all(
                lambda0 > tolerance, axis=1
            )
            exact_full[start + eligible[strictly_inside]] = True

    partial = ~(exact_empty | exact_full)
    fractions = np.zeros((len(parsed_levels), query_count), dtype=np.float64)
    base_diagnostics: dict[str, Any] | None = None
    partial_fraction = np.zeros((len(parsed_levels), int(np.count_nonzero(partial))), dtype=np.float64)
    if np.any(partial):
        # Keep the original vertices and transform arguments for this call.
        # This preserves the frozen integrator's floating-point operation order
        # exactly while the inverse-mapped vertices above drive only safe
        # classification and pruning.
        partial_fraction, base_diagnostics = integrate_individual_fractions(
            query_vertices[partial],
            heart,
            volume_fraction=volume_fraction,
            translation_m=translation,
            levels=parsed_levels,
            barycentric=bary,
            batch_size=batch_size,
        )
        fractions[:, partial] = partial_fraction
    fractions[:, exact_full] = 1.0

    if not np.isfinite(fractions).all() or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("computed fractions must be finite and lie in [0, 1]")

    partial_count = int(np.count_nonzero(partial))
    exact_empty_count = int(np.count_nonzero(exact_empty))
    exact_full_count = int(np.count_nonzero(exact_full))
    if base_diagnostics is None:
        inside_samples = [0] * len(parsed_levels)
        sampled_points = [0] * len(parsed_levels)
        base_zero = np.zeros(len(parsed_levels), dtype=np.int64)
        base_one = np.zeros(len(parsed_levels), dtype=np.int64)
        base_bvh_nodes = base_aabb_tests = base_bary_tests = 0
    else:
        inside_samples = list(base_diagnostics["inside_sample_count_by_level"])
        sampled_points = list(base_diagnostics["sample_points_evaluated_by_level"])
        base_zero = np.count_nonzero(partial_fraction == 0.0, axis=1)
        base_one = np.count_nonzero(partial_fraction == 1.0, axis=1)
        base_bvh_nodes = int(base_diagnostics["bvh_node_tests"])
        base_aabb_tests = int(base_diagnostics["tetrahedron_aabb_tests"])
        base_bary_tests = int(base_diagnostics["barycentric_point_tests"])

    empty_by_level = exact_empty_count + base_zero
    full_by_level = exact_full_count + base_one
    diagnostics: dict[str, Any] = {
        "input_tetrahedra": query_count,
        "heart_tetrahedra": heart_count,
        "levels": list(parsed_levels),
        "barycentric_points_per_tet": maximum_level,
        "exact_empty_tetrahedra_by_conservative_aabb": exact_empty_count,
        "exact_full_tetrahedra_by_strict_same_tet": exact_full_count,
        "partial_quadrature_tetrahedra": partial_count,
        "query_tetrahedra_pruned_by_aabb": exact_empty_count,
        "query_tetrahedra_pruned_by_strict_containment": exact_full_count,
        "sample_points_evaluated_by_level": sampled_points,
        "inside_sample_count_by_level": inside_samples,
        "empty_fraction_tetrahedra_by_level": empty_by_level.tolist(),
        "unit_fraction_tetrahedra_by_level": full_by_level.tolist(),
        "aabb_chunk_size": chunk_size,
        "aabb_pair_tests": aabb_pair_tests,
        "aabb_overlap_pairs": aabb_overlap_pairs,
        "strict_full_vertex_tests": full_vertex_tests,
        "frozen_integrator_calls": 0 if base_diagnostics is None else 1,
        "frozen_bvh_node_tests": base_bvh_nodes,
        "frozen_tetrahedron_aabb_tests": base_aabb_tests,
        "frozen_barycentric_point_tests": base_bary_tests,
        "broad_phase": "conservative query-AABB versus original-heart-tetrahedron-AABB blocks",
        "centroid_filter_used": False,
        "full_classification_requires_one_same_original_tetrahedron": True,
        "union_membership": "OR of original tetrahedra",
        "volume_fraction": volume_fraction,
        "scale": scale,
        "translation_m": translation.tolist(),
        "inverse_mapped_query_vertices": True,
        "partial_integrator_frame": "original query vertices with frozen inverse transform",
        "identity_transform": volume_fraction == 0.0 and np.array_equal(translation, np.zeros(3)),
        "heart_volume_ml": heart.volume_ml,
        "transformed_heart_volume_ml": heart.volume_ml * (1.0 + volume_fraction),
        "heart_centre_m": centre.tolist(),
        "overlap_assumption": heart.overlap_assumption,
        "batch_size": batch_size,
    }
    if base_diagnostics is not None:
        diagnostics["frozen_integrator_diagnostics"] = base_diagnostics
    return fractions, diagnostics

