"""Pure geometry of full binary voxel cells; no I/O, fitting to signals or FEM.

Affines map voxel CENTERS to Cartesian physical coordinates in millimetres.
The caller must establish units and coordinate provenance before using this API.
`mask_moments` and `fit_reduced_models` return dictionaries with NumPy arrays.
`individual_isotropic_candidate` additionally retains the reference mask by
reference (do not mutate it while sampling); omit that array from JSON reports.

Volumes and moments describe continuous unions of affine voxel cells.
`sampled_dice` instead samples candidate occupancy at TARGET native voxel
centers on an expanded lattice. It is NOT exact continuous Dice. Individual
shapes use nearest-neighbor pullback; their rendered volume is approximate even
when their continuous transformed volume equals the target volume analytically.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np


DEFAULT_CHUNK_SIZE = 65536


def _affine(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 affine in mm")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("Expected an affine homogeneous last row")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not np.isfinite(determinant) or determinant == 0:
        raise ValueError("Singular or nonfinite spatial affine")
    return matrix


def _chunk_size(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return int(value)


def mask_moments(mask, affine, *, chunk_size=DEFAULT_CHUNK_SIZE):
    """Volume, centroid and normalized central covariance of full voxel cells.

Includes index-space I/12 within every voxel, then transforms covariance to mm².
Batch central moments avoid subtracting large raw second moments. Even a
noncontiguous mask is traversed in bounded chunks through its flat iterator.
"""
    mask = np.asarray(mask)
    affine = _affine(affine)
    chunk_size = _chunk_size(chunk_size)
    if mask.ndim != 3 or not mask.size:
        raise ValueError("Expected a nonempty binary 3D mask")
    count = 0
    mean = np.zeros(3)
    scatter = np.zeros((3, 3))
    lower = np.array(mask.shape, dtype=np.int64)
    upper = np.full(3, -1, dtype=np.int64)
    for start in range(0, mask.size, chunk_size):
        values = mask.flat[start:start + chunk_size]
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Expected finite binary mask values")
        occupied = np.flatnonzero(values) + start
        if not len(occupied):
            continue
        points = np.column_stack(np.unravel_index(occupied, mask.shape)).astype(np.float64)
        n = len(points)
        batch_mean = points.mean(axis=0)
        centered = points - batch_mean
        delta = batch_mean - mean
        total = count + n
        scatter += centered.T @ centered + np.outer(delta, delta) * (count * n / total)
        mean += delta * (n / total)
        count = total
        lower = np.minimum(lower, points.min(axis=0).astype(np.int64))
        upper = np.maximum(upper, points.max(axis=0).astype(np.int64))
    if not count:
        raise ValueError("Empty foreground mask")
    linear = affine[:3, :3]
    covariance = linear @ (scatter / count + np.eye(3) / 12.0) @ linear.T
    covariance = (covariance + covariance.T) / 2
    voxel_mm3 = abs(float(np.linalg.det(linear)))
    volume = count * voxel_mm3
    centroid = linear @ mean + affine[:3, 3]
    if not np.isfinite(covariance).all() or not np.isfinite(centroid).all() or not np.isfinite(volume):
        raise ValueError("Nonfinite physical moments")
    return {
        "voxel_count": count, "voxel_volume_mm3": voxel_mm3,
        "volume_mm3": volume, "volume_ml": volume / 1000.0,
        "centroid_mm": centroid, "covariance_mm2": covariance,
        "bbox_index_centers": np.stack((lower, upper)),
        "coordinate_system": "input_affine_physical_mm",
        "moment_definition": "uniform_full_voxel_cells_including_index_I_over_12",
    }


def _eigenframe(covariance, eigen_rtol):
    if not np.isfinite(eigen_rtol) or not 0 <= eigen_rtol < 1:
        raise ValueError("eigen_rtol must be finite and in [0, 1)")
    values, axes = np.linalg.eigh(covariance)
    values, axes = values[::-1], axes[:, ::-1].copy()
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Physical covariance must be numerically positive definite")
    # Resolve signs deterministically where eigenvectors are identifiable.
    for column in range(3):
        pivot = np.argmax(np.abs(axes[:, column]))
        if axes[pivot, column] < 0:
            axes[:, column] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    ambiguous = any(abs(values[i] - values[j]) <= eigen_rtol * max(values[i], values[j])
                    for i, j in ((0, 1), (0, 2), (1, 2)))
    return values, axes, bool(ambiguous)


def _ellipsoid(kind, center, semi_axes, axes, degenerate, **metadata):
    semi_axes = np.asarray(semi_axes, dtype=np.float64)
    volume = 4.0 * math.pi / 3.0 * float(np.prod(semi_axes))
    return {
        "kind": kind, "center_mm": np.array(center, copy=True),
        "semi_axes_mm": semi_axes.copy(), "axes_matrix": axes.copy(),
        "volume_mm3": volume, "volume_ml": volume / 1000.0,
        "covariance_mm2": (axes * (semi_axes ** 2 / 5.0)) @ axes.T,
        "orientation_degenerate": degenerate,
        "axes_convention": "columns_are_physical_axes_descending_size_det_plus_one",
        "volume_definition": "analytic_continuous_ellipsoid",
        **metadata,
    }


def fit_reduced_models(mask, affine, *, eigen_rtol=1e-6, chunk_size=DEFAULT_CHUNK_SIZE):
    """Return moments and three separately named geometric candidates.

Raw-moment ellipsoid: a_i=sqrt(5*lambda_i), preserving covariance, not volume.
Volume-scaled ellipsoid: apply ONE common factor to those a_i, preserving axis
ratios and volume, not covariance. Degenerate orientations are not trajectories.
"""
    moments = mask_moments(mask, affine, chunk_size=chunk_size)
    values, axes, degenerate = _eigenframe(moments["covariance_mm2"], eigen_rtol)
    volume, center = moments["volume_mm3"], moments["centroid_mm"]
    radius = float(np.cbrt(3 * volume / (4 * math.pi)))
    raw_axes = np.sqrt(5 * values)
    raw_volume = 4 * math.pi / 3 * float(np.prod(raw_axes))
    scale = float(np.cbrt(volume / raw_volume))
    return {
        "moments": moments,
        "sphere": _ellipsoid("sphere_equal_volume_centroid", center, np.full(3, radius), np.eye(3), True,
                             radius_mm=radius, preserves_volume=True, preserves_covariance=False),
        "ellipsoid_raw_moment": _ellipsoid("ellipsoid_raw_moment", center, raw_axes, axes, degenerate,
                             eigenvalues_mm2=values.copy(), eigen_rtol=eigen_rtol,
                             common_axis_scale=1.0, preserves_volume=False, preserves_covariance=True),
        "ellipsoid_volume_scaled": _ellipsoid("ellipsoid_volume_scaled", center, raw_axes * scale, axes, degenerate,
                             eigenvalues_mm2=values.copy(), eigen_rtol=eigen_rtol,
                             common_axis_scale=scale, preserves_volume=True, preserves_covariance=False),
    }


def individual_isotropic_candidate(reference_mask, reference_affine, target_mask, target_affine,
                                   *, reference_moments=None, target_moments=None,
                                   chunk_size=DEFAULT_CHUNK_SIZE):
    """Transform x -> c_target + s*(x-c_reference), s=(V_target/V_reference)^(1/3).

This preserves the reference shape up to isotropic scale and translation. It is
a geometric control, not an anisotropic deformation, learned mode or physiology.
The continuous transformed voxel union has the target volume analytically.
Optional moments must come from mask_moments for the same unchanged mask and
affine. Cached inputs are trusted, avoiding repeated full-mask scans.
"""
    chunk_size = _chunk_size(chunk_size)
    reference = (mask_moments(reference_mask, reference_affine, chunk_size=chunk_size)
                 if reference_moments is None else reference_moments)
    target = (mask_moments(target_mask, target_affine, chunk_size=chunk_size)
              if target_moments is None else target_moments)
    scale = float(np.cbrt(target["volume_mm3"] / reference["volume_mm3"]))
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = target["centroid_mm"] - scale * reference["centroid_mm"]
    return {
        "kind": "individual_isotropic_volume_centroid", "scale": scale,
        "center_mm": target["centroid_mm"].copy(),
        "reference_centroid_mm": reference["centroid_mm"].copy(),
        "centroid_translation_mm": target["centroid_mm"] - reference["centroid_mm"],
        "transform_world": transform,
        "volume_mm3": reference["volume_mm3"] * scale ** 3,
        "volume_ml": reference["volume_ml"] * scale ** 3,
        "covariance_mm2": reference["covariance_mm2"] * scale ** 2,
        "reference_mask": np.asarray(reference_mask),
        "reference_affine": _affine(reference_affine).copy(),
        "reference_bbox_index_centers": reference["bbox_index_centers"].copy(),
        "volume_definition": "analytic_scaled_reference_full_voxel_union",
        "shape_preservation": "reference_shape_up_to_isotropic_scale_and_translation",
        "rendering": "nearest_neighbor_pullback_at_target_voxel_centers",
        "sampled_volume_is_approximate": True,
    }


def _lookup(mask, indices):
    inside = np.all((indices >= 0) & (indices < np.asarray(mask.shape)), axis=1)
    values = np.zeros(len(indices), dtype=bool)
    selected = indices[inside]
    values[inside] = mask[tuple(selected.T)] > 0
    return values


def sampled_dice(target_mask, target_affine, candidate, *, target_moments=None,
                 chunk_size=DEFAULT_CHUNK_SIZE):
    """Native-center Dice on target-mask bbox UNION the full candidate bbox.

Lattice indices may be negative or beyond the target image. All candidate
samples there count in the denominator; target occupancy there is zero.
Only chunk_size lattice points are materialized at a time, not the full volume.
Individual pullback uses floor(index+0.5), i.e. half-open reference voxel cells.
The full affine of a reference candidate transforms both its bounding corners
and the pullback; isotropic and anisotropic reference candidates use this path.
Optional target_moments must describe this same unchanged target mask and affine;
the cache is trusted rather than verified by another full-mask scan.
"""
    chunk_size = _chunk_size(chunk_size)
    target_mask = np.asarray(target_mask)
    target_affine = _affine(target_affine)
    target = (mask_moments(target_mask, target_affine, chunk_size=chunk_size)
              if target_moments is None else target_moments)
    inverse_target = np.linalg.inv(target_affine)
    kind = candidate["kind"]
    individual = kind in ("individual_isotropic_volume_centroid",
                          "individual_rigid_isotropic_volume_centroid",
                          "individual_affine_moment_volume_centroid")
    if individual:
        transform = _affine(candidate["transform_world"])
        reference_affine = _affine(candidate["reference_affine"])
        index_transform = inverse_target @ transform @ reference_affine
        bounds = np.asarray(candidate["reference_bbox_index_centers"], dtype=float)
        corners = np.array(list(product(*zip(bounds[0] - 0.5, bounds[1] + 0.5))))
        corners = corners @ index_transform[:3, :3].T + index_transform[:3, 3]
        candidate_lower, candidate_upper = corners.min(axis=0), corners.max(axis=0)
        pullback = np.linalg.inv(reference_affine) @ np.linalg.inv(transform) @ target_affine
        reference_mask = candidate["reference_mask"]
    elif kind in ("sphere_equal_volume_centroid", "sphere_second_moment",
                  "sphere_surface_fixed_center", "sphere_surface_free_center",
                  "ellipsoid_raw_moment", "ellipsoid_volume_scaled") or candidate.get("sampling_family") == "ellipsoid":
        center = np.asarray(candidate["center_mm"], dtype=float)
        axes = np.asarray(candidate["axes_matrix"], dtype=float)
        radii = np.asarray(candidate["semi_axes_mm"], dtype=float)
        if center.shape != (3,) or axes.shape != (3, 3) or radii.shape != (3,):
            raise ValueError("Malformed ellipsoid candidate")
        if not np.isfinite(center).all() or not np.isfinite(axes).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("Invalid ellipsoid candidate")
        if not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-8):
            raise ValueError("Ellipsoid axes must be orthonormal")
        index_center = inverse_target[:3, :3] @ center + inverse_target[:3, 3]
        extent = np.linalg.norm((inverse_target[:3, :3] @ axes) * radii, axis=1)
        candidate_lower, candidate_upper = index_center - extent, index_center + extent
    else:
        raise ValueError("Unknown candidate kind")
    if not np.isfinite(candidate_lower).all() or not np.isfinite(candidate_upper).all():
        raise ValueError("Nonfinite sampling bounds")
    lower_float = np.floor(np.minimum(target["bbox_index_centers"][0], candidate_lower) - 1e-9)
    upper_float = np.ceil(np.maximum(target["bbox_index_centers"][1], candidate_upper) + 1e-9)
    if np.any(np.abs(np.concatenate((lower_float, upper_float))) >= np.iinfo(np.int64).max / 4):
        raise ValueError("Sampling bounds exceed integer capacity")
    lower, upper = lower_float.astype(np.int64), upper_float.astype(np.int64)
    shape = tuple(int(value) for value in upper - lower + 1)
    total = math.prod(shape)
    if total > np.iinfo(np.int64).max:
        raise ValueError("Sampling grid exceeds integer capacity")
    candidate_count = intersection = 0
    for start in range(0, total, chunk_size):
        flat = np.arange(start, min(total, start + chunk_size), dtype=np.int64)
        indices = np.column_stack(np.unravel_index(flat, shape)) + lower
        target_values = _lookup(target_mask, indices)
        if individual:
            reference_indices = indices @ pullback[:3, :3].T + pullback[:3, 3]
            nearest = np.floor(reference_indices + 0.5).astype(np.int64)
            candidate_values = _lookup(reference_mask, nearest)
        else:
            world = indices @ target_affine[:3, :3].T + target_affine[:3, 3]
            local = ((world - center) @ axes) / radii
            candidate_values = np.einsum("ij,ij->i", local, local) <= 1.0 + 1e-12
        candidate_count += int(np.count_nonzero(candidate_values))
        intersection += int(np.count_nonzero(candidate_values & target_values))
    denominator = target["voxel_count"] + candidate_count
    return {
        "dice_voxel_center_approx": 2.0 * intersection / denominator,
        "target_voxel_count": target["voxel_count"],
        "candidate_sample_count": candidate_count, "intersection_sample_count": intersection,
        "candidate_sampled_volume_ml": candidate_count * target["voxel_volume_mm3"] / 1000,
        "candidate_continuous_volume_ml": candidate["volume_ml"],
        "grid_bounds_index_inclusive": np.stack((lower, upper)), "grid_shape": shape,
        "sampling": "target_native_voxel_centers_on_extended_index_grid",
        "exact_continuous_dice": False, "candidate_clipped_to_image_extent": False,
        "individual_tie_rule": "floor(reference_index_plus_0.5)" if individual else None,
    }
