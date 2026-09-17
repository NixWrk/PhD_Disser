"""Framewise reduced reference shapes from volume and principal moments.

These are geometric candidates, not measured anatomical motion or a temporal
prediction. The target frame supplies its own moments; no Dice is optimized.
Affines map voxel centers to a common Cartesian physical space in millimetres.
Moment caches must describe the same unchanged mask and affine; they are trusted
to avoid full-mask rescans. Returned reference_mask is retained by reference,
must stay unchanged during sampling, and should be omitted from JSON exports.

Principal axes are paired by descending eigenvalue, without permutations.
Four proper sign flips resolve only discrete sign ambiguity. Near-equal
eigenvalues leave a continuous orientation ambiguity: the returned orientation
is a numerical convention, not an anatomical correspondence.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from . import heart_reduced_geometry as geometry
else:
    import heart_reduced_geometry as geometry


DEFAULT_CHUNK_SIZE = geometry.DEFAULT_CHUNK_SIZE
_PROPER_FLIPS = np.array(((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))


def _degenerate_axes(values, eigen_rtol):
    gaps = np.abs(values[:, None] - values[None, :])
    close = gaps <= eigen_rtol * np.maximum(values[:, None], values[None, :])
    np.fill_diagonal(close, False)
    return close.any(axis=1)


def _principal_alignment(reference, target, eigen_rtol):
    ref_values, ref_axes, ref_degenerate = geometry._eigenframe(
        reference["covariance_mm2"], eigen_rtol)
    target_values, target_axes, target_degenerate = geometry._eigenframe(
        target["covariance_mm2"], eigen_rtol)
    rotations = np.array([(target_axes * flip) @ ref_axes.T for flip in _PROPER_FLIPS])
    # For proper rotations, maximizing trace minimizes angle in [0, pi].
    traces = np.trace(rotations, axis1=1, axis2=2)
    best = int(np.argmax(traces))
    aligned_target = target_axes * _PROPER_FLIPS[best]
    rotation = rotations[best]
    return {
        "reference_eigenvalues_mm2": ref_values,
        "target_eigenvalues_mm2": target_values,
        "reference_axes_matrix": ref_axes,
        "target_axes_matrix": aligned_target,
        "rotation_matrix": rotation,
        "rotation_angle_rad": float(np.arccos(np.clip((traces[best] - 1) / 2, -1, 1))),
        "selected_target_sign_flip": _PROPER_FLIPS[best].copy(),
        "orientation_selection": "minimum_rotation_among_four_proper_sign_flips",
        "orientation_sign_tie": bool(np.count_nonzero(np.abs(traces - traces[best]) <= 1e-12) > 1),
        "orientation_degenerate": ref_degenerate or target_degenerate,
        "reference_orientation_degenerate": ref_degenerate,
        "target_orientation_degenerate": target_degenerate,
        "reference_degenerate_axes": _degenerate_axes(ref_values, eigen_rtol),
        "target_degenerate_axes": _degenerate_axes(target_values, eigen_rtol),
        "eigen_rtol": eigen_rtol,
        "axes_convention": "columns_are_physical_axes_paired_by_descending_eigenvalue_det_plus_one",
        "rotation_is_anatomical_motion": False,
    }


def _candidate(reference_mask, reference_affine, target_mask, target_affine, *,
               anisotropic, reference_moments, target_moments, eigen_rtol, chunk_size):
    chunk_size = geometry._chunk_size(chunk_size)
    reference_affine = geometry._affine(reference_affine)
    target_affine = geometry._affine(target_affine)
    reference = (geometry.mask_moments(reference_mask, reference_affine, chunk_size=chunk_size)
                 if reference_moments is None else reference_moments)
    target = (geometry.mask_moments(target_mask, target_affine, chunk_size=chunk_size)
              if target_moments is None else target_moments)
    alignment = _principal_alignment(reference, target, eigen_rtol)
    volumes = np.array([reference["volume_mm3"], target["volume_mm3"]], dtype=float)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Reference and target volumes must be finite and positive")
    volume_ratio = float(volumes[1] / volumes[0])
    raw_scales = (np.sqrt(alignment["target_eigenvalues_mm2"] /
                          alignment["reference_eigenvalues_mm2"])
                  if anisotropic else np.ones(3))
    common_scale = float(np.cbrt(volume_ratio / np.prod(raw_scales)))
    principal_scales = common_scale * raw_scales
    linear = (alignment["target_axes_matrix"] * principal_scales) @ alignment["reference_axes_matrix"].T
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(linear).all() or not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("Nonfinite or nonpositive candidate determinant")
    if not np.isclose(determinant, volume_ratio, rtol=1e-10, atol=0):
        raise ValueError("Candidate is too ill-conditioned to preserve the volume ratio")
    ref_center = np.asarray(reference["centroid_mm"], dtype=float)
    target_center = np.asarray(target["centroid_mm"], dtype=float)
    if ref_center.shape != (3,) or target_center.shape != (3,) or not (
            np.isfinite(ref_center).all() and np.isfinite(target_center).all()):
        raise ValueError("Expected finite 3D centroids")
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = target_center - linear @ ref_center
    covariance = linear @ reference["covariance_mm2"] @ linear.T
    covariance = (covariance + covariance.T) / 2
    volume = float(reference["volume_mm3"] * determinant)
    if not np.isfinite(transform).all() or not np.isfinite(covariance).all() or not np.isfinite(volume):
        raise ValueError("Nonfinite transformed geometry")
    return {
        "kind": ("individual_affine_moment_volume_centroid" if anisotropic
                 else "individual_rigid_isotropic_volume_centroid"),
        "center_mm": target_center.copy(), "reference_centroid_mm": ref_center.copy(),
        "centroid_translation_mm": target_center - ref_center,
        "transform_world": transform,
        "principal_scales": principal_scales, "raw_principal_scales": raw_scales,
        "common_scale": common_scale,
        "principal_scales_order": "paired_descending_eigenvalues_not_sorted_scale_magnitudes",
        "determinant": determinant, "target_to_reference_volume_ratio": volume_ratio,
        "volume_mm3": volume, "volume_ml": volume / 1000,
        "covariance_mm2": covariance,
        "target_covariance_mm2": np.array(target["covariance_mm2"], copy=True),
        "covariance_definition": ("common_scale_squared_times_target_covariance" if anisotropic
                                  else "scaled_rotated_reference_covariance"),
        "guarantees_target_covariance": False, "preserves_volume": True,
        "reference_mask": np.asarray(reference_mask),
        "reference_affine": reference_affine.copy(),
        "reference_bbox_index_centers": np.array(reference["bbox_index_centers"], copy=True),
        "volume_definition": "analytic_affine_transformed_reference_full_voxel_union",
        "shape_preservation": ("reference_voxel_union_under_full_affine" if anisotropic
                               else "reference_shape_up_to_isotropic_scale_rotation_translation"),
        "rendering": "nearest_neighbor_pullback_at_target_voxel_centers",
        "sampled_volume_is_approximate": True,
        "parameter_estimation": "separate_target_frame_moments_relative_to_reference",
        "temporal_prediction": False, "target_dice_optimized": False,
        **alignment,
    }


def individual_rigid_isotropic_candidate(
        reference_mask, reference_affine, target_mask, target_affine, *,
        reference_moments=None, target_moments=None,
        eigen_rtol=1e-6, chunk_size=DEFAULT_CHUNK_SIZE):
    """Return x -> c_target + s*R*(x-c_reference), s=(V_target/V_ref)^(1/3).

    R aligns principal axes with the smallest rotation over four proper sign
    flips. The continuous transformed reference volume equals target volume.
    Principal shape ratios are retained; target covariance need not match.
    See the module docstring for cache and orientation-ambiguity contracts.
    """
    return _candidate(
        reference_mask, reference_affine, target_mask, target_affine,
        anisotropic=False, reference_moments=reference_moments,
        target_moments=target_moments, eigen_rtol=eigen_rtol, chunk_size=chunk_size)


def individual_affine_moment_candidate(
        reference_mask, reference_affine, target_mask, target_affine, *,
        reference_moments=None, target_moments=None,
        eigen_rtol=1e-6, chunk_size=DEFAULT_CHUNK_SIZE):
    """Return x -> c_target + F*(x-c_reference), a volume-corrected moment map.

    F = c * Q_target * diag(sqrt(lambda_target/lambda_ref)) * Q_ref.T,
    c = ((V_target/V_ref) / product(raw_principal_scales))**(1/3).
    This preserves the relative principal scales and continuous target volume.
    Its covariance is c**2 * Cov_target, generally NOT exactly Cov_target.
    The principal-axis convention does not identify a general physical affine
    deformation from second moments alone, and does not predict another frame.
    """
    return _candidate(
        reference_mask, reference_affine, target_mask, target_affine,
        anisotropic=True, reference_moments=reference_moments,
        target_moments=target_moments, eigen_rtol=eigen_rtol, chunk_size=chunk_size)
