"""Light synthetic geometry checks only; no CT I/O or FEM."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_additional_sphere_methods import (
    fit_surface_sphere_fixed_volume, fit_algebraic_sphere,
    fit_iou_sphere_fixed_volume, fit_iou_sphere_free_volume,
    sampled_sphere_iou, area_systematic_sample, sphere_surface_metrics,
)


def surface(count=100):
    k = np.arange(count)
    z = 1-2*(k+.5)/count
    theta = k*np.pi*(3-np.sqrt(5))
    return np.column_stack((np.sqrt(1-z*z)*np.cos(theta),
                            np.sqrt(1-z*z)*np.sin(theta), z))


def volume(radius):
    return 4*np.pi/3*radius**3


def sphere_mask():
    index = np.indices((9, 9, 9)).transpose(1, 2, 3, 0)
    return np.sum((index-4)**2, axis=-1) <= 2.6**2


class SurfaceTests(unittest.TestCase):
    def test_known_sphere_nonuniform_weights_both_methods(self):
        center = np.array([12., -7., 21.])
        r = 8.
        points = center+r*surface()
        weights = np.linspace(.2, 2., len(points))
        fits = [fit_algebraic_sphere(points, weights),
            fit_surface_sphere_fixed_volume(points, weights, volume(r),
                [center-r, center+r], [center+[1, 2, -1], center-1])]
        for fit in fits:
            with self.subTest(kind=fit['kind']):
                np.testing.assert_allclose(fit['center_mm'], center, atol=1e-7)
                self.assertAlmostEqual(fit['radius_mm'], r, places=7)
                self.assertLess(sphere_surface_metrics(points, weights,
                    fit['center_mm'], fit['radius_mm'])['surface_RMS_mm'], 1e-7)

    def test_fixed_volume_on_nonspherical_surface(self):
        points = surface()*[7., 4., 2.]+[3, -2, 8]
        fit = fit_surface_sphere_fixed_volume(points, np.ones(len(points)), 350.,
            [[-5, -8, 0], [11, 4, 16]], [[3, -2, 8], [4, -3, 7]])
        self.assertAlmostEqual(fit['volume_mm3'], 350., places=10)
        self.assertTrue(fit['preserves_volume'])
        self.assertFalse(fit['diagnostics']['global_optimum_certified'])
        self.assertEqual(fit['criterion']['units'], 'mm2')

    def test_similarity_equivariance_and_objective_units(self):
        points = surface()*[7., 4., 2.]+[3, -2, 8]
        weights = np.linspace(.3, 2, len(points))
        shift = np.array([1e5, -2e5, 7e5]); scale = 3.
        bounds = np.array([[-5, -8, 0], [11, 4, 16.]])
        starts = np.array([[3, -2, 8], [4, -3, 7.]])
        pairs = [
            (fit_algebraic_sphere(points, weights),
             fit_algebraic_sphere(points*scale+shift, weights*scale**2), 4),
            (fit_surface_sphere_fixed_volume(points, weights, 350, bounds, starts),
             fit_surface_sphere_fixed_volume(points*scale+shift, weights*scale**2,
                 350*scale**3, bounds*scale+shift, starts*scale+shift), 2)]
        for original, moved, power in pairs:
            with self.subTest(kind=original['kind']):
                np.testing.assert_allclose(moved['center_mm'], original['center_mm']*scale+shift,
                                           rtol=0, atol=2e-6)
                self.assertAlmostEqual(moved['radius_mm']/scale, original['radius_mm'], places=7)
                self.assertAlmostEqual(moved['criterion']['value']/scale**power,
                                       original['criterion']['value'], places=6)

    def test_algebraic_area_weighting_and_not_radial_mean(self):
        directions = np.concatenate((np.eye(3), -np.eye(3)))
        points = np.concatenate((2*directions, 4*directions))
        weights = np.r_[np.ones(6), 3*np.ones(6)]
        fit = fit_algebraic_sphere(points, weights)
        np.testing.assert_allclose(fit['center_mm'], 0, atol=1e-12)
        self.assertAlmostEqual(fit['radius_mm'], np.sqrt(13))
        self.assertNotAlmostEqual(fit['radius_mm'], 3.5)
        self.assertEqual(fit['criterion']['units'], 'mm4')
        self.assertAlmostEqual(fit['criterion']['value'], 27.)

    def test_degenerate_algebraic_and_invalid_inputs_rejected(self):
        points = surface()
        points[:, 2] = 0
        with self.assertRaisesRegex(ValueError, 'Rank-deficient'):
            fit_algebraic_sphere(points, np.ones(len(points)))
        for weights in [np.zeros(100), -np.ones(100), np.full(100, np.nan)]:
            with self.subTest(weights=weights[0]), self.assertRaises(ValueError):
                fit_algebraic_sphere(surface(), weights)
        with self.assertRaises(ValueError):
            fit_surface_sphere_fixed_volume(surface(), np.ones(100), -1,
                [[-2]*3, [2]*3], [[0]*3])
        with self.assertRaisesRegex(ValueError, 'no silent clipping'):
            fit_surface_sphere_fixed_volume(surface(), np.ones(100), 4,
                [[-2]*3, [2]*3], [[3]*3])

    def test_sample_area_cdf_and_full_metrics(self):
        points = surface(4)
        sampled, weights, diag = area_systematic_sample(points, [1, 2, 3, 4], 100)
        np.testing.assert_allclose(sampled, points)
        np.testing.assert_allclose(weights, [.1, .2, .3, .4])
        self.assertEqual(diag['unique_count'], 4)
        full = sphere_surface_metrics(points*2, [1, 2, 3, 4], [0, 0, 0], 1, chunk_size=1)
        self.assertAlmostEqual(full['surface_RMS_mm'], 1)
        self.assertAlmostEqual(full['algebraic_RMS_mm2'], 3)


class IoUTests(unittest.TestCase):
    def test_known_native_sphere(self):
        result = sampled_sphere_iou(sphere_mask(), np.eye(4), [4]*3, 2.6, chunk_size=29)
        self.assertEqual(result['iou'], 1.)
        self.assertFalse(result['candidate_clipped_to_image_extent'])

    def test_outside_candidate_penalized_and_padding_invariant(self):
        mask = np.ones((1, 1, 1), dtype=bool)
        original = sampled_sphere_iou(mask, np.eye(4), [0]*3, 3., subdivisions=2)
        self.assertGreater(original['candidate_sample_count'], original['target_sample_count'])
        self.assertLess(original['iou'], .02)
        padded = np.pad(mask, 4)
        affine = np.eye(4); affine[:3, 3] = -4
        expanded = sampled_sphere_iou(padded, affine, [0]*3, 3., subdivisions=2)
        self.assertEqual(original['iou'], expanded['iou'])

    def test_full_affine_similarity_and_shear(self):
        mask = sphere_mask()
        affine = np.array([[1, .2, 0, 8], [0, 2, 0, -3], [0, 0, 1.5, 7], [0, 0, 0, 1.]])
        c = affine[:3, :3]@np.array([4]*3)+affine[:3, 3]
        original = sampled_sphere_iou(mask, affine, c, 4.1, subdivisions=2)
        transform = np.array([[0, -3, 0, 12], [3, 0, 0, -7], [0, 0, 3, 100], [0, 0, 0, 1.]])
        moved = sampled_sphere_iou(mask, transform@affine,
            transform[:3, :3]@c+transform[:3, 3], 12.3, subdivisions=2)
        self.assertEqual(original['iou'], moved['iou'])

    def test_native_fit_fixed_and_free_known_sphere(self):
        common = dict(evaluation_spacing_mm=None, verification_spacing_mm=None,
                      check_subdivisions=2, max_evaluations=160)
        fixed = fit_iou_sphere_fixed_volume(sphere_mask(), np.eye(4), volume(2.6),
            [[2]*3, [6]*3], [[4]*3, [5, 4, 4]], [1., .5, .25], **common)
        free = fit_iou_sphere_free_volume(sphere_mask(), np.eye(4), [[2]*3, [6]*3],
            [1, 4], [[4, 4, 4, 2.6], [5, 4, 4, 2.1]], [1., .5, .25], **common)
        for fit in [fixed, free]:
            self.assertEqual(fit['criterion']['value'], 1.)
            for trial in fit['diagnostics']['trials']:
                self.assertGreaterEqual(trial['iou'], trial['initial_iou'])
            self.assertFalse(fit['diagnostics']['global_optimum_certified'])
        self.assertAlmostEqual(fixed['volume_mm3'], volume(2.6), places=10)
        self.assertFalse(free['preserves_volume'])

    def test_physical_grid_budget_and_refinement_recorded(self):
        fit = fit_iou_sphere_fixed_volume(sphere_mask(), np.eye(4), volume(2.6),
            [[2]*3, [6]*3], [[5]*3, [4]*3], [1., .5], max_evaluations=4)
        diag = fit['diagnostics']
        self.assertLessEqual(diag['optimization_evaluations'], 4)
        self.assertEqual(diag['sampling']['spacing_mm'], 2.)
        self.assertEqual(diag['refined_sampling_check']['spacing_mm'], 1.)
        self.assertFalse(diag['refined_sampling_check']['reoptimized'])
        self.assertEqual(fit['status'], 'search_budget_exhausted')

    def test_physical_grid_translation_scale_and_no_cropping(self):
        kwargs = dict(max_evaluations=20, verification_spacing_mm=None, check_subdivisions=None)
        mask = sphere_mask()
        original = fit_iou_sphere_fixed_volume(mask, np.eye(4), volume(2.6),
            [[2]*3, [6]*3], [[4]*3], [1., .5], **kwargs)
        affine = np.eye(4); affine[:3, :3] *= 3; affine[:3, 3] = [11, -8, 21]
        shift = affine[:3, 3]
        moved = fit_iou_sphere_fixed_volume(mask, affine, volume(7.8),
            np.array([[2]*3, [6]*3])*3+shift, np.array([[4]*3])*3+shift,
            [3., 1.5], evaluation_spacing_mm=6, **kwargs)
        self.assertEqual(original['criterion']['value'], moved['criterion']['value'])
        np.testing.assert_allclose(moved['center_mm'], original['center_mm']*3+shift)
        with self.assertRaisesRegex(ValueError, 'clipping forbidden'):
            fit_iou_sphere_fixed_volume(np.ones((1,1,1), bool), np.eye(4), volume(3),
                [[-1]*3, [1]*3], [[0]*3], [1.], evaluation_bounds_mm=[[-.5]*3, [.5]*3],
                **kwargs)

    def test_physical_grid_huge_sphere_cannot_win_by_crop(self):
        kwargs = dict(max_evaluations=2, evaluation_spacing_mm=1.,
                      verification_spacing_mm=None, check_subdivisions=None)
        fit = fit_iou_sphere_free_volume(np.ones((1,1,1), bool), np.eye(4),
            [[-1]*3, [1]*3], [.25, 5.], [[0,0,0,.5], [0,0,0,3.]], [1., .5], **kwargs)
        self.assertEqual(fit['radius_mm'], .5)
        self.assertLess(fit['diagnostics']['trials'][1]['iou'], .02)

    def test_budget_limit_raises_instead_of_clipping(self):
        with self.assertRaisesRegex(ValueError, 'never crop'):
            sampled_sphere_iou(np.ones((1,1,1)), np.eye(4), [0]*3, 20,
                               max_samples_per_evaluation=20)
        with self.assertRaisesRegex(ValueError, 'cropping forbidden'):
            fit_iou_sphere_fixed_volume(sphere_mask(), np.eye(4), volume(2.6),
                [[2]*3, [6]*3], [[4]*3], [1.], max_samples_per_evaluation=20)


if __name__ == '__main__':
    unittest.main()
