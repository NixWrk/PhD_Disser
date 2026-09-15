"""Synthetic numerical checks, not anatomical or electrical validation.

Run: python -B -m unittest discover -s <this directory> -p
     test_heart_additional_ellipsoid_methods.py
"""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_additional_ellipsoid_methods import (
    _ParameterSpace, _scanline_overlap, _overlap, area_subsample, ellipsoid_surface_distance,
    evaluate_overlap_ellipsoid, fit_overlap_ellipsoid, fit_surface_ellipsoid,
)
from heart_reduced_geometry import fit_reduced_models


def sphere_points(n):
    k = np.arange(n)
    z = 1-2*(k+.5)/n
    theta = k*np.pi*(3-np.sqrt(5))
    return np.column_stack((np.sqrt(1-z*z)*np.cos(theta),
                            np.sqrt(1-z*z)*np.sin(theta), z))


def example():
    center = np.array([2., -1., 3.])
    axes = np.array([6., 4., 2.])
    rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    return center, axes, rotation


class ClosestDistanceTests(unittest.TestCase):
    def test_sphere_signed_distance_including_centre(self):
        p = np.array([[0., 0., 0.], [1., 2., 0.], [0., 0., 5.], [3., 0., 0.]])
        d = ellipsoid_surface_distance(p, np.zeros(3), np.full(3, 3.), signed=True)
        np.testing.assert_allclose(d, np.linalg.norm(p, axis=1)-3, atol=2e-14)

    def test_interior_hard_case_is_not_radial(self):
        a = np.array([3., 2., 1.])
        p = np.array([[1., 0., 0.], [0., 0., 0.], [4., 0., 0.]])
        d, closest = ellipsoid_surface_distance(p, np.zeros(3), a, return_closest=True)
        expected = [9/8, 0., np.sqrt(1-(3/8)**2)]
        np.testing.assert_allclose(closest[0], expected, atol=1e-12)
        np.testing.assert_allclose(d, [np.sqrt(7/8), 1., 1.], atol=1e-12)
        self.assertLess(d[0], 2.)  # radial intersection would give 2 mm

    def test_repeated_shortest_axes_and_tiny_components(self):
        p = np.array([[.5, 0., 0.], [0., 0., 0.], [.5, 1e-20, 0.], [4., 0., 0.]])
        d, q = ellipsoid_surface_distance(p, np.zeros(3), [3., 1., 1.], return_closest=True)
        np.testing.assert_allclose(np.sum((q/[3., 1., 1.])**2, axis=1), 1, atol=1e-12)
        self.assertAlmostEqual(d[0], d[2], places=12)
        self.assertAlmostEqual(d[1], 1.)

    def test_known_normal_offsets_and_rigid_similarity(self):
        c, a, r = example()
        u = sphere_points(100)
        surface = c+(u*a) @ r.T
        normals = u/a
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals = normals @ r.T
        query = np.r_[surface+.01*normals, surface-.01*normals]
        d, q = ellipsoid_surface_distance(query, c, a, r, signed=True, return_closest=True)
        np.testing.assert_allclose(d, np.r_[np.full(100, .01), np.full(100, -.01)], atol=1e-12)
        np.testing.assert_allclose(q, np.r_[surface, surface], atol=1e-11)
        t = Rotation.from_rotvec([.8, .2, -.4]).as_matrix()
        shift, scale = np.array([100., -20., 5.]), 7.5
        transformed = scale*query @ t.T + shift
        td, tq = ellipsoid_surface_distance(transformed, scale*c @ t.T+shift,
            scale*a, t @ r, signed=True, return_closest=True)
        np.testing.assert_allclose(td, scale*d, atol=2e-12)
        np.testing.assert_allclose(tq, scale*q @ t.T+shift, atol=2e-12)

    def test_against_independent_angular_optimization(self):
        # A separate parameterization and solver check the global branch on
        # interior and exterior points, without reusing the secular equation.
        axes = np.array([4., 2.5, 1.])
        queries = np.array([[.2, .3, .1], [2., .1, .0], [5., 1., 2.], [-1., 3., .4]])
        distances = ellipsoid_surface_distance(queries, np.zeros(3), axes)
        def point(angles):
            theta, phi = angles
            return axes*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        for query, distance in zip(queries, distances):
            fits = [minimize(lambda x: np.sum((point(x)-query)**2), [theta, phi],
                    method="BFGS", options={"gtol": 1e-10})
                    for theta in [.2, 1.5, 2.9] for phi in [0., 1.5, 3., 4.5]]
            independent = np.sqrt(min(f.fun for f in fits))
            self.assertAlmostEqual(distance, independent, places=7)

    def test_bad_rotation_rejected(self):
        with self.assertRaises(ValueError):
            ellipsoid_surface_distance(np.zeros((1, 3)), np.zeros(3), [1, 2, 3], np.diag([1, 1, -1]))


class SurfaceFittingTests(unittest.TestCase):
    def test_area_sampling_is_deterministic_and_weighted(self):
        p = np.column_stack((np.arange(4000), np.zeros((4000, 2))))
        w = np.r_[np.ones(2000), np.full(2000, 9.)]
        selected, weights, indices = area_subsample(p, w, 1000)
        _, other_weights, other_indices = area_subsample(p, w*100, 1000)
        np.testing.assert_array_equal(indices, other_indices)
        np.testing.assert_allclose(weights, other_weights)
        self.assertAlmostEqual(weights[selected[:, 0] >= 2000].sum(), .9)
        self.assertLessEqual(len(selected), 1000)

    def test_all_four_surface_methods_recover_known_ellipsoid(self):
        c, a, r = example()
        points = c+(sphere_points(80)*a) @ r.T
        vp = c+(sphere_points(360)*a) @ r.T
        volume = 4*np.pi/3*np.prod(a)
        for method in ["L3a", "L3b", "L3c", "L3d"]:
            with self.subTest(method=method):
                init = {"center_mm": c if method in ("L3a", "L3c") else c+[.2, -.15, .1],
                        "semi_axes_mm": a*[1.05, .96, 1.02], "axes_matrix": r}
                result = fit_surface_ellipsoid(points, np.linspace(.5, 2., len(points)), init,
                    method=method, target_volume_mm3=volume, max_nfev=900, n_starts=2,
                    validation_points=vp, validation_weights=np.ones(len(vp)))
                self.assertEqual(result["status"], "completed")
                self.assertLess(result["validation_metrics"]["surface_RMS_mm"], 5e-5)
                self.assertLessEqual(result["optimization"]["nfev_total"], 900)
                param = result["parameters"]
                self.assertEqual(param["kind"], "ellipsoid_project_"+method)
                np.testing.assert_allclose(param["center_mm"], c, atol=1e-4)
                np.testing.assert_allclose(param["semi_axes_mm"], a, atol=1e-4)
                if method in ("L3c", "L3d"):
                    self.assertAlmostEqual(param["volume_mm3"]/volume, 1., places=13)

    def test_fixed_volume_parameterization_at_many_feasible_points(self):
        c, a, r = example()
        volume = 200.
        space = _ParameterSpace({"center_mm": c, "semi_axes_mm": a, "axes_matrix": r},
                                 True, True, volume, None, None)
        for x in space.starts(50):
            self.assertTrue(space.feasible(x))
            _, semi, rotation = space.decode(x)
            self.assertAlmostEqual(4*np.pi/3*np.prod(semi)/volume, 1., places=13)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)

    def test_dense_validation_does_not_change_fitted_parameters(self):
        c, a, r = example()
        p = c+(sphere_points(120)*a) @ r.T
        initial = {"center_mm": c, "semi_axes_mm": a*1.03, "axes_matrix": r}
        first = fit_surface_ellipsoid(p, np.ones(len(p)), initial, n_starts=1, max_nfev=300)
        second = fit_surface_ellipsoid(p, np.ones(len(p)), initial, n_starts=1, max_nfev=300,
            validation_points=p+10, validation_weights=np.ones(len(p)))
        self.assertEqual(first["status"], "completed")
        np.testing.assert_array_equal(first["parameters"]["semi_axes_mm"], second["parameters"]["semi_axes_mm"])
        self.assertGreater(second["validation_metrics"]["surface_RMS_mm"], 1.)

    def test_fit_similarity_invariance(self):
        c, a, r = example()
        p = c+(sphere_points(100)*a) @ r.T
        init = {"center_mm": c+[.2, -.1, .15], "semi_axes_mm": a*1.04, "axes_matrix": r}
        kwargs = dict(method="L3d", target_volume_mm3=4*np.pi/3*np.prod(a),
                      max_nfev=600, n_starts=2)
        first = fit_surface_ellipsoid(p, np.ones(len(p)), init, **kwargs)
        t = Rotation.from_rotvec([.4, .6, -.2]).as_matrix()
        scale, shift = 3., np.array([11., -2., 8.])
        moved_init = {"center_mm": scale*init["center_mm"] @ t.T+shift,
                      "semi_axes_mm": scale*init["semi_axes_mm"], "axes_matrix": t @ r}
        kwargs["target_volume_mm3"] *= scale**3
        second = fit_surface_ellipsoid(scale*p @ t.T+shift, np.ones(len(p)), moved_init, **kwargs)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        np.testing.assert_allclose(second["parameters"]["center_mm"],
            scale*first["parameters"]["center_mm"] @ t.T+shift, atol=2e-4)
        np.testing.assert_allclose(second["parameters"]["covariance_mm2"],
            scale**2*t @ first["parameters"]["covariance_mm2"] @ t.T, atol=2e-3)

    def test_exhausted_budget_has_no_moment_fallback(self):
        c, a, r = example()
        result = fit_surface_ellipsoid(c+sphere_points(50)*a, np.ones(50),
            {"center_mm": c, "semi_axes_mm": a, "axes_matrix": r}, max_nfev=3, n_starts=3)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["parameters"])
        self.assertEqual(result["optimization"]["nfev_total"], 3)


class OverlapTests(unittest.TestCase):
    def test_full_candidate_counted_outside_image(self):
        mask = np.ones((1, 1, 1), dtype=bool)
        params = {"center_mm": np.zeros(3), "semi_axes_mm": np.full(3, 3.), "axes_matrix": np.eye(3)}
        result = evaluate_overlap_ellipsoid(mask, np.eye(4), params, grid_step_mm=.25)
        metrics = result["metrics"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(metrics["target_count"], 64)
        self.assertGreater(metrics["candidate_count"], 7000)
        self.assertLess(metrics["IoU_sampled"], .01)
        self.assertEqual(metrics["intersection_count"], metrics["target_count"])
        self.assertTrue(result["sampling"]["whole_candidate_counted"])

    def test_disjoint_candidate_outside_image(self):
        params = {"center_mm": np.array([10., 0., 0.]), "semi_axes_mm": np.ones(3), "axes_matrix": np.eye(3)}
        result = evaluate_overlap_ellipsoid(np.ones((1, 1, 1), bool), np.eye(4), params, grid_step_mm=.5)
        self.assertGreater(result["metrics"]["candidate_count"], 0)
        self.assertEqual(result["metrics"]["IoU_sampled"], 0.)

    def test_sheared_affine_matches_direct_occupancy(self):
        affine = np.array([[1., .3, 0., 2.], [0., 1.2, .2, -3.], [0., 0., .8, 1.], [0, 0, 0, 1.]])
        mask = np.ones((2, 3, 2), bool)
        params = {"center_mm": affine[:3, :3] @ [.5, 1., .5]+affine[:3, 3],
                  "semi_axes_mm": np.array([3., 2., 1.]), "axes_matrix": np.eye(3)}
        result = evaluate_overlap_ellipsoid(mask, affine, params, grid_step_mm=.4)
        meta = result["sampling"]
        lo, hi = np.array(meta["index_bounds"])
        q = np.column_stack(np.unravel_index(np.arange(np.prod(hi-lo+1)), tuple(hi-lo+1)))+lo
        points = q*.4
        local = np.linalg.solve(affine[:3, :3], (points-affine[:3, 3]).T).T
        target = np.all((local >= -.5) & (local < np.array(mask.shape)-.5), axis=1)
        candidate = np.sum(((points-params["center_mm"])/params["semi_axes_mm"])**2, axis=1) <= 1
        self.assertEqual(result["metrics"]["candidate_count"], int(candidate.sum()))
        self.assertEqual(result["metrics"]["intersection_count"], int((candidate & target).sum()))

    def test_bounded_overlap_pilot_both_methods(self):
        q = np.indices((13, 11, 9)).reshape(3, -1).T
        mask = (np.sum(((q-[6, 5, 4])/[4, 3, 2])**2, axis=1) <= 1).reshape(13, 11, 9)
        affine = np.diag([2., 2., 2., 1.])
        initial = fit_reduced_models(mask, affine)["ellipsoid_volume_scaled"]
        for method in ("L4a", "L4b"):
            with self.subTest(method=method):
                result = fit_overlap_ellipsoid(mask, affine, initial, method=method,
                    grid_step_mm=2., max_nfev=1600, n_starts=2, max_grid_points=100000,
                    center_limit_mm=2., axis_bounds_mm=[2., 12.], validation_grid_step_mm=1.5)
                self.assertEqual(result["status"], "completed")
                self.assertGreater(result["fit_metrics"]["IoU_sampled"], .9)
                self.assertLessEqual(result["optimization"]["nfev_total"], 1600)
                self.assertEqual(result["validation_metrics"]["status"], "completed")
                self.assertEqual(result["parameters"]["kind"], "ellipsoid_project_"+method)
                if method == "L4a":
                    self.assertAlmostEqual(result["parameters"]["volume_mm3"]/(8*mask.sum()), 1., places=13)

    def test_scanline_counts_match_dense_lattice_with_rotations(self):
        rng = np.random.default_rng(39)
        frame = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
        origin = np.array([10., -20., 30.])
        lattice = np.indices((31, 31, 31)).reshape(3, -1).T-15
        points = origin+lattice @ frame.T
        target = np.sum(lattice*lattice, axis=1) < 36
        meta = {"frame": frame.tolist(), "origin_mm": origin.tolist(), "grid_step_mm": 1.}
        for i in range(15):
            c = origin+frame @ rng.uniform(-2., 2., 3)
            a = rng.uniform(2., 6., 3)
            r = frame @ Rotation.from_rotvec(rng.uniform(-1., 1., 3)).as_matrix()
            with self.subTest(i=i):
                dense = _overlap(points, target, c, a, r)
                scan = _scanline_overlap(points[target], meta, c, a, r)
                self.assertEqual(scan, dense)
        # Integer tangencies are included consistently in both implementations.
        self.assertEqual(_overlap(points, target, origin, np.full(3, 4.), frame),
                         _scanline_overlap(points[target], meta, origin, np.full(3, 4.), frame))

    def test_grid_limit_returns_unsupported_without_fallback(self):
        mask = np.ones((3, 3, 3), bool)
        initial = fit_reduced_models(mask, np.eye(4))["ellipsoid_volume_scaled"]
        result = fit_overlap_ellipsoid(mask, np.eye(4), initial, max_grid_points=10)
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["parameters"])
        self.assertIn("whole_candidate_lattice_requires", result["reason"])


if __name__ == "__main__":
    unittest.main()
