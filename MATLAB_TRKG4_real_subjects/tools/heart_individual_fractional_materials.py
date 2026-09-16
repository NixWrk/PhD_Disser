"""Stage-1 geometry for fractions in an individual tetrahedral heart union.

The heart is the *original union of the supplied tetrahedra*.  Membership is
the logical OR of barycentric membership in those tetrahedra.  No convex hull,
Delaunay tetrahedralisation, surface reconstruction, or centroid filter is
used.  A conservative AABB hierarchy only removes tetrahedra whose boxes are
separated from a query point; the individual barycentric test decides
membership.

``TetrahedralHeart`` expects coordinates in metres.  Tetrahedron orientation
may be positive or negative: the signed determinant is retained internally,
while ``volume_ml`` and the volume centroid use its absolute value.  This is
appropriate for an oriented or inconsistently oriented non-overlapping mesh
with shared faces.  A positive-volume overlap between distinct input
tetrahedra would make the volume and centroid sums double-count that region;
the constructor therefore rejects exact duplicate tetrahedra and explicitly
records the remaining non-overlap assumption, but deliberately does not run
an O(N^2) overlap test on a real mesh.

``integrate_individual_fractions`` samples each query tetrahedron at nested,
uniform barycentric points shared with ``heart_fractional_materials``.  The
heart is transformed forward as

    x' = centre + scale * (x - centre) + translation,
    scale = (1 + volume_fraction) ** (1/3),

so the integration maps query points through the exact inverse transform.  The
returned fractions are quadrature estimates; point membership itself is
geometric (within a small floating-point tolerance on boundaries).
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    # The quadrature rule is intentionally shared with the existing ellipsoid
    # implementation.  The existing file is not modified by this module.
    from heart_fractional_materials import uniform_tet_barycentric
except ImportError:  # pragma: no cover - useful when imported as a package
    from .heart_fractional_materials import uniform_tet_barycentric


__all__ = ["TetrahedralHeart", "integrate_individual_fractions", "uniform_tet_barycentric"]

_FLOAT_EPS = np.finfo(np.float64).eps
_AABB_MARGIN_ULPS = 64.0
_BARYCENTRIC_MARGIN_ULPS = 256.0
_DEFAULT_LEAF_SIZE = 16


def _as_finite_float_array(value: Any, *, name: str) -> np.ndarray:
    """Convert an input to float64 and reject malformed or non-finite data."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _signed_determinants(vertices: np.ndarray) -> np.ndarray:
    """Return six times the signed volume for ``N x 4 x 3`` tetrahedra."""

    edges = vertices[:, 1:] - vertices[:, :1]
    return np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))


def _degenerate_mask(vertices: np.ndarray, signed_determinants: np.ndarray) -> np.ndarray:
    """Use a scale-relative determinant guard rather than an absolute metre cutoff."""

    edges = vertices[:, 1:] - vertices[:, :1]
    edge_scale = np.max(np.abs(edges), axis=(1, 2))
    # This rejects exact degeneracy and numerically unresolved slivers while
    # remaining scale invariant for coordinates expressed in metres.
    threshold = 4096.0 * _FLOAT_EPS * np.maximum(edge_scale, np.finfo(float).tiny) ** 3
    return np.abs(signed_determinants) <= threshold


def _validate_tetrahedra(vertices_m: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate tetrahedral coordinates and return a private float64 copy plus dets."""

    vertices = _as_finite_float_array(vertices_m, name=name)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError(f"{name} must have shape (N, 4, 3)")
    if vertices.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one tetrahedron")
    signed = _signed_determinants(vertices)
    if not np.isfinite(signed).all():
        raise ValueError(f"{name} has non-finite tetrahedron determinants")
    degenerate = _degenerate_mask(vertices, signed)
    if np.any(degenerate):
        first = int(np.flatnonzero(degenerate)[0])
        raise ValueError(f"{name} contains a degenerate or numerically unresolved tetrahedron at index {first}")
    return np.array(vertices, dtype=np.float64, copy=True), signed


def _reject_exact_duplicate_tetrahedra(vertices: np.ndarray) -> None:
    """Reject the cheap overlap failure mode without pairwise geometry tests."""

    canonical = np.empty_like(vertices)
    for index, tetrahedron in enumerate(vertices):
        order = np.lexsort((tetrahedron[:, 2], tetrahedron[:, 1], tetrahedron[:, 0]))
        canonical[index] = tetrahedron[order]
    _, counts = np.unique(canonical.reshape(len(vertices), -1), axis=0, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(
            "heart contains exact duplicate tetrahedra; positive-volume overlap "
            "is incompatible with the volume-centroid assumption"
        )


def _conservative_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand finite AABBs so roundoff cannot turn an intersection into a miss."""

    scale = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    margin = _AABB_MARGIN_ULPS * _FLOAT_EPS * scale
    with np.errstate(over="raise", invalid="raise"):
        expanded_lower = np.nextafter(lower - margin, -np.inf)
        expanded_upper = np.nextafter(upper + margin, np.inf)
    if not np.isfinite(expanded_lower).all() or not np.isfinite(expanded_upper).all():
        raise ValueError("AABB expansion produced non-finite bounds")
    return expanded_lower, expanded_upper


def _validate_levels(levels: Any) -> tuple[int, ...]:
    try:
        result = tuple(levels)
    except TypeError as exc:
        raise ValueError("levels must be a non-empty sequence of powers of two") from exc
    if not result or any(isinstance(level, bool) or not isinstance(level, (int, np.integer)) for level in result):
        raise ValueError("levels must be a non-empty sequence of integer powers of two")
    result = tuple(int(level) for level in result)
    if tuple(sorted(set(result))) != result or any(level <= 0 or level & (level - 1) for level in result):
        raise ValueError("levels must be strictly increasing powers of two")
    return result


def _validate_barycentric_rule(barycentric: Any, maximum_level: int) -> np.ndarray:
    bary = _as_finite_float_array(barycentric, name="barycentric")
    if bary.shape != (maximum_level, 4):
        raise ValueError(f"barycentric must have shape ({maximum_level}, 4)")
    if np.any(bary < 0.0) or not np.allclose(bary.sum(axis=1), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("barycentric rows must be non-negative and sum to one")
    return np.array(bary, dtype=np.float64, copy=False)


class TetrahedralHeart:
    """Exact point-membership index for the supplied tetrahedral heart union.

    Parameters
    ----------
    vertices_m:
        Array with shape ``(N, 4, 3)`` in metres.  The union is defined by
        these exact input tetrahedra, including concavities and disconnected
        components.  Shared faces are allowed.  Distinct tetrahedra are
        assumed not to overlap at positive volume; that assumption is needed
        for the volume centroid and is not established by an all-pairs test.
    """

    def __init__(self, vertices_m: Any) -> None:
        vertices, signed_determinants = _validate_tetrahedra(vertices_m, name="vertices_m")
        _reject_exact_duplicate_tetrahedra(vertices)

        self.vertices_m = vertices
        self._signed_determinants = signed_determinants
        self._tet_volumes_m3 = np.abs(signed_determinants) / 6.0
        self._tet_lower = vertices.min(axis=1)
        self._tet_upper = vertices.max(axis=1)
        self._tet_lower_safe, self._tet_upper_safe = _conservative_bounds(self._tet_lower, self._tet_upper)
        self._v0 = vertices[:, 0]
        edges = vertices[:, 1:] - self._v0[:, None, :]
        try:
            self._inverse_edges = np.linalg.inv(edges)
            condition = np.linalg.cond(edges)
        except np.linalg.LinAlgError as exc:
            raise ValueError("vertices_m contains a singular tetrahedron") from exc
        if not np.isfinite(self._inverse_edges).all() or not np.isfinite(condition).all():
            raise ValueError("vertices_m produced non-finite barycentric transforms")
        self._barycentric_tolerance = _BARYCENTRIC_MARGIN_ULPS * _FLOAT_EPS * np.maximum(1.0, condition)
        self._barycentric_tolerance = np.minimum(self._barycentric_tolerance, 1e-8)

        total_volume_m3 = float(np.sum(self._tet_volumes_m3, dtype=np.float64))
        if not np.isfinite(total_volume_m3) or total_volume_m3 <= 0.0:
            raise ValueError("heart tetrahedron volumes must have a finite positive sum")
        weighted_centres = vertices.mean(axis=1) * self._tet_volumes_m3[:, None]
        centre = np.sum(weighted_centres, axis=0, dtype=np.float64) / total_volume_m3
        if not np.isfinite(centre).all():
            raise ValueError("heart volume centroid is non-finite")

        self.volume_ml = total_volume_m3 * 1e6
        self.centre_m = np.array(centre, dtype=np.float64, copy=True)
        self.bounds_m = (
            np.array(vertices.min(axis=(0, 1)), dtype=np.float64, copy=True),
            np.array(vertices.max(axis=(0, 1)), dtype=np.float64, copy=True),
        )
        self.overlap_assumption = (
            "positive-volume overlap absent; shared faces allowed; exact duplicates rejected; "
            "no all-pairs overlap validation performed"
        )
        self._build_aabb_hierarchy()

    def _build_aabb_hierarchy(self) -> None:
        """Build a balanced median-split hierarchy over tetrahedron AABBs."""

        tetrahedron_count = len(self.vertices_m)
        leaf_size = _DEFAULT_LEAF_SIZE
        node_lower: list[np.ndarray] = []
        node_upper: list[np.ndarray] = []
        node_left: list[int] = []
        node_right: list[int] = []
        node_start: list[int] = []
        node_stop: list[int] = []
        leaf_order: list[int] = []
        centres = 0.5 * (self._tet_lower + self._tet_upper)

        def build(indices: np.ndarray) -> int:
            node = len(node_lower)
            lower = np.min(self._tet_lower[indices], axis=0)
            upper = np.max(self._tet_upper[indices], axis=0)
            node_lower.append(lower)
            node_upper.append(upper)
            node_left.append(-1)
            node_right.append(-1)
            node_start.append(-1)
            node_stop.append(-1)
            if len(indices) <= leaf_size:
                start = len(leaf_order)
                leaf_order.extend(int(index) for index in indices)
                node_start[node] = start
                node_stop[node] = len(leaf_order)
                return node

            axis = int(np.argmax(upper - lower))
            order = np.argsort(centres[indices, axis], kind="mergesort")
            midpoint = len(indices) // 2
            left = build(indices[order[:midpoint]])
            right = build(indices[order[midpoint:]])
            node_left[node] = left
            node_right[node] = right
            return node

        root = build(np.arange(tetrahedron_count, dtype=np.int64))
        self._node_lower, self._node_upper = _conservative_bounds(
            np.asarray(node_lower, dtype=np.float64), np.asarray(node_upper, dtype=np.float64)
        )
        self._node_left = np.asarray(node_left, dtype=np.int64)
        self._node_right = np.asarray(node_right, dtype=np.int64)
        self._node_start = np.asarray(node_start, dtype=np.int64)
        self._node_stop = np.asarray(node_stop, dtype=np.int64)
        self._leaf_order = np.asarray(leaf_order, dtype=np.int64)
        self._root = root

    @staticmethod
    def _point_box_mask(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.all((points >= lower) & (points <= upper), axis=1)

    def _contains_flat(self, points: np.ndarray, *, with_diagnostics: bool = False):
        """Vectorized hierarchy traversal for an ``M x 3`` finite point array."""

        point_count = len(points)
        result = np.zeros(point_count, dtype=bool)
        diagnostics = {
            "bvh_node_tests": 0,
            "tetrahedron_aabb_tests": 0,
            "barycentric_point_tests": 0,
        }
        if point_count == 0:
            return (result, diagnostics) if with_diagnostics else result

        stack: list[tuple[int, np.ndarray]] = [(self._root, np.arange(point_count, dtype=np.int64))]
        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            unresolved = indices[~result[indices]]
            if len(unresolved) == 0:
                continue
            diagnostics["bvh_node_tests"] += 1
            in_node = self._point_box_mask(points[unresolved], self._node_lower[node], self._node_upper[node])
            indices = unresolved[in_node]
            if len(indices) == 0:
                continue

            left = int(self._node_left[node])
            if left < 0:
                for position in range(int(self._node_start[node]), int(self._node_stop[node])):
                    tetrahedron = int(self._leaf_order[position])
                    unresolved = indices[~result[indices]]
                    if len(unresolved) == 0:
                        break
                    diagnostics["tetrahedron_aabb_tests"] += len(unresolved)
                    in_tetrahedron_box = self._point_box_mask(
                        points[unresolved], self._tet_lower_safe[tetrahedron], self._tet_upper_safe[tetrahedron]
                    )
                    candidates = unresolved[in_tetrahedron_box]
                    if len(candidates) == 0:
                        continue
                    diagnostics["barycentric_point_tests"] += len(candidates)
                    # ``edges`` are ROW vectors, so lambda = (p-v0) @ inv(edges).
                    barycentric = (points[candidates] - self._v0[tetrahedron]) @ self._inverse_edges[tetrahedron]
                    lambda0 = 1.0 - barycentric.sum(axis=1)
                    tolerance = self._barycentric_tolerance[tetrahedron]
                    inside = np.all(barycentric >= -tolerance, axis=1) & (lambda0 >= -tolerance)
                    result[candidates[inside]] = True
                continue

            right = int(self._node_right[node])
            # Children are tested when popped, keeping the same conservative
            # AABB rule at every level and avoiding a dense point x node table.
            stack.append((right, indices))
            stack.append((left, indices))

        return (result, diagnostics) if with_diagnostics else result

    def contains(self, points: Any) -> np.ndarray | bool:
        """Return membership in the original tetrahedral union.

        ``points`` may have any leading shape ending in three coordinates; the
        returned boolean array has the same leading shape.  A single ``(3,)``
        point returns a Python ``bool``.  Boundary points are included, with a
        scale-aware floating-point tolerance in barycentric coordinates.
        """

        array = _as_finite_float_array(points, name="points")
        if array.ndim == 0 or array.shape[-1:] != (3,):
            raise ValueError("points must have shape (..., 3)")
        leading_shape = array.shape[:-1]
        flat = array.reshape((-1, 3))
        result = self._contains_flat(flat)
        if not leading_shape:
            return bool(result[0])
        return result.reshape(leading_shape)


def _inverse_transform(
    points: np.ndarray,
    centre_m: np.ndarray,
    scale: float,
    translation_m: np.ndarray,
    identity: bool,
) -> np.ndarray:
    """Map points from the transformed heart frame back to the original frame."""

    if identity:
        return points
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mapped = centre_m + (points - translation_m - centre_m) / scale
    if not np.isfinite(mapped).all():
        raise ValueError("inverse heart transform produced non-finite points")
    return mapped


def integrate_individual_fractions(
    vertices_m: Any,
    heart: TetrahedralHeart,
    volume_fraction: float = 0.0,
    translation_m: Any = (0.0, 0.0, 0.0),
    levels: Any = (512, 4096),
    barycentric: Any = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate the fraction of each query tetrahedron inside a transformed heart.

    The result has shape ``(len(levels), len(vertices_m))``.  ``volume_fraction``
    changes heart volume by the factor ``1 + volume_fraction``.  The transform
    is a uniform scale about the original union volume centroid followed by
    ``translation_m``; query points are mapped through its inverse before the
    exact union-membership query.  ``diagnostics`` contains integer sample and
    broad-phase counts, transform metadata, and the explicit non-overlap
    assumption.
    """

    if not isinstance(heart, TetrahedralHeart):
        raise TypeError("heart must be a TetrahedralHeart")
    query_vertices, _ = _validate_tetrahedra(vertices_m, name="vertices_m")
    parsed_levels = _validate_levels(levels)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    batch_size = int(batch_size)

    fraction_value = _as_finite_float_array(volume_fraction, name="volume_fraction")
    if fraction_value.ndim != 0:
        raise ValueError("volume_fraction must be a finite scalar")
    volume_fraction = float(fraction_value)
    if 1.0 + volume_fraction <= 0.0:
        raise ValueError("volume_fraction must be greater than -1")
    translation = _as_finite_float_array(translation_m, name="translation_m")
    if translation.shape != (3,):
        raise ValueError("translation_m must have shape (3,)")

    maximum_level = parsed_levels[-1]
    if barycentric is None:
        # Calling the existing helper preserves its deterministic nested
        # scrambled-Sobol construction and seed contract.
        bary = uniform_tet_barycentric(maximum_level)
    else:
        bary = _validate_barycentric_rule(barycentric, maximum_level)

    scale = float(np.cbrt(1.0 + volume_fraction))
    identity = volume_fraction == 0.0 and np.array_equal(translation, np.zeros(3))
    query_count = len(query_vertices)
    fractions = np.zeros((len(parsed_levels), query_count), dtype=np.float64)
    inside_samples = np.zeros(len(parsed_levels), dtype=np.int64)
    full_counts = np.zeros(len(parsed_levels), dtype=np.int64)
    empty_counts = np.zeros(len(parsed_levels), dtype=np.int64)
    candidate_tet_tests = 0
    barycentric_point_tests = 0
    bvh_node_tests = 0

    for start in range(0, query_count, batch_size):
        stop = min(start + batch_size, query_count)
        batch_vertices = query_vertices[start:stop]
        samples = np.einsum("pv,bvc->bpc", bary, batch_vertices, optimize=True)
        transformed_samples = _inverse_transform(
            samples.reshape((-1, 3)), heart.centre_m, scale, translation, identity
        )
        membership, query_diagnostics = heart._contains_flat(transformed_samples, with_diagnostics=True)
        membership = membership.reshape((stop - start, maximum_level))
        cumulative = np.cumsum(membership, axis=1, dtype=np.int64)
        bvh_node_tests += int(query_diagnostics["bvh_node_tests"])
        candidate_tet_tests += int(query_diagnostics["tetrahedron_aabb_tests"])
        barycentric_point_tests += int(query_diagnostics["barycentric_point_tests"])
        for level_index, level in enumerate(parsed_levels):
            count = cumulative[:, level - 1]
            fractions[level_index, start:stop] = count / float(level)
            inside_samples[level_index] += int(np.sum(count, dtype=np.int64))
            full_counts[level_index] += int(np.count_nonzero(count == level))
            empty_counts[level_index] += int(np.count_nonzero(count == 0))

    if not np.isfinite(fractions).all() or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("computed fractions must be finite and lie in [0, 1]")

    partial_counts = query_count - full_counts - empty_counts
    diagnostics: dict[str, Any] = {
        "input_tetrahedra": query_count,
        "heart_tetrahedra": len(heart.vertices_m),
        "levels": list(parsed_levels),
        "barycentric_points_per_tet": maximum_level,
        "sample_points_evaluated_by_level": [query_count * level for level in parsed_levels],
        "inside_sample_count_by_level": inside_samples.tolist(),
        "full_sample_fraction_tetrahedra_by_level": full_counts.tolist(),
        "empty_sample_fraction_tetrahedra_by_level": empty_counts.tolist(),
        "partial_sample_fraction_tetrahedra_by_level": partial_counts.tolist(),
        "zero_fraction_tetrahedra_by_level": empty_counts.tolist(),
        "unit_fraction_tetrahedra_by_level": full_counts.tolist(),
        "bvh_node_tests": bvh_node_tests,
        "tetrahedron_aabb_tests": candidate_tet_tests,
        "barycentric_point_tests": barycentric_point_tests,
        "broad_phase": "conservative AABB hierarchy over original heart tetrahedra",
        "centroid_filter_used": False,
        "union_membership": "OR of original tetrahedra",
        "volume_fraction": volume_fraction,
        "scale": scale,
        "translation_m": translation.tolist(),
        "identity_transform": identity,
        "heart_volume_ml": heart.volume_ml,
        "transformed_heart_volume_ml": heart.volume_ml * (1.0 + volume_fraction),
        "heart_centre_m": heart.centre_m.tolist(),
        "overlap_assumption": heart.overlap_assumption,
        "batch_size": batch_size,
    }
    return fractions, diagnostics

