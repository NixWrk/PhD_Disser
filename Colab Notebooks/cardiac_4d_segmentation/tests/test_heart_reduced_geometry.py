"""Analytic and lattice controls, without medical data or disk-backed volumes."""
from __future__ import annotations

import importlib.util
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "heart_reduced_geometry.py"
SPEC = importlib.util.spec_from_file_location("heart_reduced_geometry", MODULE_PATH)
geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(geometry)


def oblique_affine(reflected=False):
    u = np.array([np.sqrt(0.5), np.sqrt(0.5), 0])
    v = np.array([-0.5, 0.5, np.sqrt(0.5)])
    rotation = np.column_stack((u, v, np.cross(u, v)))
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-2 if reflected else 2, 3, 4])
    affine[:3, 3] = [12, -5, 8]
    return affine


class ReducedGeometryTests(unittest.TestCase):
    def test_analytic_cuboid_moments_in_oblique_and_reflected_affine(self):
        mask = np.zeros((5, 7, 8), dtype=np.uint8)
        mask[1:3, 2:5, 1:5] = 1
        lengths = np.array([2, 3, 4])
        mean_index = np.array([1, 2, 1]) + (lengths - 1) / 2
        for reflected in (False, True):
            with self.subTest(reflected=reflected):
                affine = oblique_affine(reflected)
                result = geometry.mask_moments(mask, affine, chunk_size=7)
                self.assertEqual(result["voxel_count"], 24)
                self.assertAlmostEqual(result["volume_mm3"], 24 * 24)
                np.testing.assert_allclose(result["centroid_mm"],
                    affine[:3, :3] @ mean_index + affine[:3, 3], atol=1e-12)
                # Continuous uniform cuboid covariance, NOT just point centers.
                expected = affine[:3, :3] @ np.diag(lengths ** 2 / 12) @ affine[:3, :3].T
                np.testing.assert_allclose(result["covariance_mm2"], expected, atol=1e-12)

    def test_single_voxel_has_within_cell_covariance(self):
        affine = oblique_affine()
        result = geometry.mask_moments(np.ones((1, 1, 1), dtype=bool), affine)
        np.testing.assert_allclose(result["covariance_mm2"], affine[:3, :3] @ affine[:3, :3].T / 12)
        self.assertTrue(np.all(np.linalg.eigvalsh(result["covariance_mm2"]) > 0))

    def test_ellipsoid_variants_equal_volume_ratios_and_proper_axes(self):
        for reflected in (False, True):
            with self.subTest(reflected=reflected):
                models = geometry.fit_reduced_models(np.ones((2, 3, 4), dtype=bool), oblique_affine(reflected))
                moments = models["moments"]
                raw = models["ellipsoid_raw_moment"]
                scaled = models["ellipsoid_volume_scaled"]
                np.testing.assert_allclose(raw["covariance_mm2"], moments["covariance_mm2"], atol=1e-12)
                self.assertFalse(np.isclose(raw["volume_mm3"], moments["volume_mm3"]))
                self.assertAlmostEqual(scaled["volume_mm3"], moments["volume_mm3"], places=10)
                self.assertAlmostEqual(models["sphere"]["volume_mm3"], moments["volume_mm3"], places=10)
                np.testing.assert_allclose(scaled["semi_axes_mm"] / raw["semi_axes_mm"], scaled["common_axis_scale"])
                np.testing.assert_allclose(scaled["semi_axes_mm"] / scaled["semi_axes_mm"][0],
                                           raw["semi_axes_mm"] / raw["semi_axes_mm"][0])
                self.assertFalse(scaled["orientation_degenerate"])
                for candidate in (raw, scaled):
                    np.testing.assert_allclose(candidate["axes_matrix"].T @ candidate["axes_matrix"], np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(candidate["axes_matrix"]), 1)
                    np.testing.assert_allclose(candidate["center_mm"], moments["centroid_mm"])

    def test_equal_eigenvalues_flag_orientation_ambiguity(self):
        models = geometry.fit_reduced_models(np.ones((3, 3, 3)), np.eye(4))
        self.assertTrue(models["ellipsoid_raw_moment"]["orientation_degenerate"])
        self.assertTrue(models["ellipsoid_volume_scaled"]["orientation_degenerate"])
        self.assertAlmostEqual(np.linalg.det(models["ellipsoid_raw_moment"]["axes_matrix"]), 1)

    def test_dice_includes_candidate_outside_image(self):
        mask = np.ones((1, 1, 9), dtype=bool)
        sphere = geometry.fit_reduced_models(mask, np.eye(4))["sphere"]
        # r≈1.29: center plus six axial neighbors are inside; four neighbors
        # lie outside the 1x1 image cross-section and MUST enter the denominator.
        for chunk in (2, 1000):
            with self.subTest(chunk_size=chunk):
                result = geometry.sampled_dice(mask, np.eye(4), sphere, chunk_size=chunk)
                self.assertEqual(result["candidate_sample_count"], 7)
                self.assertEqual(result["intersection_sample_count"], 3)
                self.assertEqual(result["dice_voxel_center_approx"], 6 / 16)
                self.assertLess(result["grid_bounds_index_inclusive"][0, 0], 0)
                self.assertGreater(result["grid_bounds_index_inclusive"][1, 0], 0)
                self.assertFalse(result["exact_continuous_dice"])
                self.assertAlmostEqual(result["candidate_continuous_volume_ml"], 0.009)
                self.assertAlmostEqual(result["candidate_sampled_volume_ml"], 0.007)

    def test_individual_identity_on_sparse_oblique_mask(self):
        storage = np.zeros((8, 5, 6), dtype=np.uint8)
        storage[0:4:2, 1:4, 2:5] = 1
        storage[6, 4, 5] = 1
        mask = storage[::2]  # noncontiguous input is intentionally supported
        before = mask.copy()
        affine = oblique_affine(reflected=True)
        candidate = geometry.individual_isotropic_candidate(mask, affine, mask, affine, chunk_size=5)
        np.testing.assert_allclose(candidate["transform_world"], np.eye(4), atol=1e-12)
        result = geometry.sampled_dice(mask, affine, candidate, chunk_size=11)
        self.assertEqual(result["dice_voxel_center_approx"], 1)
        self.assertEqual(result["candidate_sample_count"], np.count_nonzero(mask))
        np.testing.assert_array_equal(mask, before)

    def test_individual_scale_and_translation_preserve_continuous_volume(self):
        reference = np.ones((1, 1, 1), dtype=bool)
        target = np.zeros((6, 6, 6), dtype=bool)
        target[3:5, 3:5, 3:5] = 1
        candidate = geometry.individual_isotropic_candidate(reference, np.eye(4), target, np.eye(4))
        self.assertAlmostEqual(candidate["scale"], 2)
        np.testing.assert_allclose(candidate["center_mm"], [3.5, 3.5, 3.5])
        self.assertAlmostEqual(candidate["volume_mm3"], 8)
        result = geometry.sampled_dice(target, np.eye(4), candidate, chunk_size=17)
        self.assertEqual(result["candidate_sample_count"], 8)
        self.assertEqual(result["dice_voxel_center_approx"], 1)
        self.assertTrue(candidate["sampled_volume_is_approximate"])

    def test_cached_moments_avoid_repeated_mask_scans(self):
        reference = np.ones((1, 1, 1), dtype=bool)
        target = np.ones((2, 2, 2), dtype=bool)
        affine = oblique_affine()
        reference_moments = geometry.mask_moments(reference, affine)
        models = geometry.fit_reduced_models(target, affine)
        with mock.patch.object(geometry, "mask_moments", side_effect=AssertionError("cache ignored")):
            individual = geometry.individual_isotropic_candidate(
                reference, affine, target, affine,
                reference_moments=reference_moments, target_moments=models["moments"])
            for candidate in (individual, models["sphere"], models["ellipsoid_volume_scaled"]):
                result = geometry.sampled_dice(target, affine, candidate, target_moments=models["moments"])
                self.assertGreater(result["dice_voxel_center_approx"], 0)
            result = geometry.sampled_dice(target, affine, individual, target_moments=models["moments"])
            self.assertEqual(result["dice_voxel_center_approx"], 1)

    def test_crop_with_adjusted_affine_preserves_models_and_sampled_dice(self):
        target = np.zeros((8, 9, 12), dtype=bool)
        target[3, 4, 2:11] = True
        affine = oblique_affine(reflected=True)
        cropped = target[3:4, 4:5, 2:11]
        cropped_affine = affine.copy()
        cropped_affine[:3, 3] += affine[:3, :3] @ np.array([3, 4, 2])
        full_models = geometry.fit_reduced_models(target, affine)
        cropped_models = geometry.fit_reduced_models(cropped, cropped_affine)
        for field in ("volume_mm3", "centroid_mm", "covariance_mm2"):
            np.testing.assert_allclose(full_models["moments"][field], cropped_models["moments"][field], atol=1e-12)
        full_individual = geometry.individual_isotropic_candidate(target, affine, target, affine)
        cropped_individual = geometry.individual_isotropic_candidate(cropped, cropped_affine, cropped, cropped_affine)
        pairs = [(full_models[key], cropped_models[key])
                 for key in ("sphere", "ellipsoid_raw_moment", "ellipsoid_volume_scaled")]
        pairs.append((full_individual, cropped_individual))
        for full_candidate, cropped_candidate in pairs:
            full_result = geometry.sampled_dice(target, affine, full_candidate, chunk_size=13)
            cropped_result = geometry.sampled_dice(cropped, cropped_affine, cropped_candidate, chunk_size=7)
            for field in ("candidate_sample_count", "intersection_sample_count", "dice_voxel_center_approx"):
                self.assertEqual(full_result[field], cropped_result[field])

    def test_invalid_masks_and_affines_are_rejected(self):
        for mask in (np.zeros((2, 2, 2)), np.full((2, 2, 2), 2), np.full((2, 2, 2), np.nan), np.ones((2, 2))):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                geometry.mask_moments(mask, np.eye(4))
        for affine in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)):
            with self.subTest(affine_shape=affine.shape), self.assertRaises(ValueError):
                geometry.mask_moments(np.ones((1, 1, 1)), affine)
        with self.assertRaises(ValueError):
            geometry.mask_moments(np.ones((1, 1, 1)), np.eye(4), chunk_size=0)


if __name__ == "__main__":
    unittest.main()
