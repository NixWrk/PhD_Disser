"""Synthetic integration review; no medical data or FEM.

Expected failures document observed runner defects without editing shared code.
When the coordinator fixes one, remove its expectedFailure decorator and retain
the assertion as a regression check. All disk writes use TemporaryDirectory.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_heart_method_comparison as runner
from heart_reduced_geometry import (
    _ellipsoid, mask_moments, sampled_dice, fit_reduced_models,
    individual_isotropic_candidate,
)


def unit_surface(count=40):
    z = 1-2*(np.arange(count)+.5)/count
    theta = np.arange(count)*np.pi*(3-np.sqrt(5))
    return np.column_stack((np.sqrt(1-z*z)*np.cos(theta),
                            np.sqrt(1-z*z)*np.sin(theta), z))


def sphere(radius=2.4, center=(3., 3., 3.), kind='sphere_algebraic_unit_quadratic'):
    return _ellipsoid(kind, center, np.full(3, radius), np.eye(3), True,
                      radius_mm=radius, sampling_family='ellipsoid')


def synthetic_case():
    indices = np.indices((7, 7, 7)).transpose(1, 2, 3, 0)
    mask = np.sum((indices-3)**2, axis=-1) <= 2.4**2
    affine = np.eye(4)
    models = fit_reduced_models(mask, affine)
    row = {
        'subject': 'synthetic', 'phase_id': 'phase00', 'region': 'whole_heart',
        'cycle_index': 0, 'is_reference_phase': True,
        'ct_sha256': 'synthetic-no-ct',
        'provenance': {'sha256': 'synthetic-mask', 'touches_source_image_boundary': False},
        'moments': runner.serial(models['moments']),
        'comparisons': {
            'sphere': {'parameters': runner.serial(models['sphere'])},
            'ellipsoid_volume_scaled': {'parameters': runner.serial(models['ellipsoid_volume_scaled'])}},
        'raw_moment_ellipsoid': runner.serial(models['ellipsoid_raw_moment']),
    }
    return mask, affine, models['moments'], row


class CommonMetricTests(unittest.TestCase):
    def test_known_sphere_new_family_matches_legacy_and_has_zero_surface_error(self):
        mask, affine, m, _ = synthetic_case()
        points = 3+2.4*unit_surface()
        weights = np.full(len(points), 1/len(points))
        new = sphere()
        original = copy.deepcopy(new)
        legacy = dict(new, kind='sphere_equal_volume_centroid')
        legacy.pop('sampling_family')
        new_metrics = runner.numeric_metrics(mask, affine, m, new, points, weights)
        old_metrics = runner.numeric_metrics(mask, affine, m, legacy, points, weights)
        self.assertEqual(new_metrics['dice'], 1.)
        self.assertEqual(new_metrics['dice'], old_metrics['dice'])
        self.assertLess(new_metrics['target_surface_to_candidate_RMS_mm'], 1e-10)
        self.assertAlmostEqual(new_metrics['volume_error_percent'],
            100*(4*np.pi*2.4**3/3/m['volume_mm3']-1))
        self.assertEqual(new['kind'], original['kind'])
        np.testing.assert_array_equal(new['center_mm'], original['center_mm'])

    def test_common_surface_metric_uses_geometric_mm_for_algebraic_candidate(self):
        mask, affine, m, _ = synthetic_case()
        directions = unit_surface(20)
        points = np.concatenate((3+3.4*directions, 3+5.4*directions))
        weights = np.r_[np.full(20, .25/20), np.full(20, .75/20)]
        candidate = sphere()
        candidate['criterion'] = {'value': 12345., 'units': 'mm4'}
        metrics = runner.numeric_metrics(mask, affine, m, candidate, points, weights)
        self.assertAlmostEqual(metrics['target_surface_to_candidate_RMS_mm'], np.sqrt(7), places=8)
        self.assertAlmostEqual(metrics['target_surface_to_candidate_mean_mm'], 2.5, places=8)

    def test_family_dispatch_does_not_clip_and_agrees_with_independent_enumeration(self):
        mask = np.ones((1, 1, 1), dtype=bool)
        candidate = sphere(3., (0., 0., 0.), 'arbitrary_registered_geometric_method')
        result = sampled_dice(mask, np.eye(4), candidate, chunk_size=17)
        grid = np.indices((9, 9, 9)).reshape(3, -1).T-4
        count = int(np.count_nonzero(np.sum(grid**2, axis=1) <= 9))
        self.assertEqual(result['candidate_sample_count'], count)
        self.assertAlmostEqual(result['dice_voxel_center_approx'], 2/(1+count))
        self.assertFalse(result['candidate_clipped_to_image_extent'])
        padded = np.pad(mask, 4)
        affine = np.eye(4); affine[:3, 3] = -4
        expanded = sampled_dice(padded, affine, candidate)
        self.assertEqual(expanded['dice_voxel_center_approx'], result['dice_voxel_center_approx'])

    def test_new_family_rotated_ellipsoid_on_sheared_affine(self):
        mask = np.ones((2, 2, 2), dtype=bool)
        affine = np.array([[1, .2, 0, 0], [0, 1.4, 0, 0], [0, 0, .8, 0], [0, 0, 0, 1.]])
        angle = .47
        axes = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        candidate = _ellipsoid('new_ellipsoid_method', [0, 0, 0], np.array([3.2, 2., 1.]),
                               axes, False, sampling_family='ellipsoid')
        actual = sampled_dice(mask, affine, candidate, chunk_size=23)
        index = np.indices((17, 17, 17)).reshape(3, -1).T-8
        world = index@affine[:3, :3].T
        occupied = np.sum(((world@axes)/candidate['semi_axes_mm'])**2, axis=1) <= 1+1e-12
        target = np.all((index >= 0) & (index < 2), axis=1)
        expected = 2*np.count_nonzero(target & occupied)/(8+np.count_nonzero(occupied))
        self.assertAlmostEqual(actual['dice_voxel_center_approx'], expected)

    def test_family_dispatch_preserves_validation_and_requires_opt_in(self):
        candidate = sphere()
        candidate.pop('sampling_family')
        with self.assertRaisesRegex(ValueError, 'Unknown candidate kind'):
            sampled_dice(np.ones((2,2,2)), np.eye(4), candidate)
        for axes, radii in [(np.ones((3,3)), np.ones(3)), (np.eye(3), [-1, 2, 3])]:
            bad = dict(sphere(), axes_matrix=axes, semi_axes_mm=np.asarray(radii))
            with self.assertRaises(ValueError):
                sampled_dice(np.ones((2,2,2)), np.eye(4), bad)

    def test_individual_representation_is_not_recast_as_ellipsoid(self):
        mask, affine, m, _ = synthetic_case()
        candidate = individual_isotropic_candidate(mask, affine, mask, affine)
        metrics = runner.numeric_metrics(mask, affine, m, candidate,
                                         3+2.4*unit_surface(), np.full(40, 1/40))
        self.assertEqual(metrics['dice'], 1.)
        self.assertEqual(metrics['volume_error_percent'], 0.)
        self.assertIsNone(metrics['target_surface_to_candidate_RMS_mm'])
        self.assertNotIn('sampling_family', candidate)

    def test_project_ids_map_to_distinct_new_kind_names(self):
        mask, affine, m, row = synthetic_case()
        points = 3+2.4*unit_surface()
        weights = np.full(len(points), 1/len(points))
        metadata = {'bounds_mm': [[.5]*3, [5.5]*3]}
        args = SimpleNamespace(fit_points=40, surface_budget=30, iou_budget=3, iou_step=2.)
        expected = {'G3c': 'sphere_surface_fixed_volume',
                    'G4a': 'sphere_iou_fixed_volume', 'G4b': 'sphere_iou_free_volume',
                    'G5': 'sphere_algebraic_unit_quadratic'}
        self.assertEqual(len(runner.METHODS), len(set(runner.METHODS)))
        for method, kind in expected.items():
            with self.subTest(method=method):
                candidate = runner.candidates_for(method, row, row, mask, affine, mask, affine,
                    m, m, points, weights, metadata, args)
                self.assertEqual(candidate['kind'], kind)
                metrics = runner.numeric_metrics(mask, affine, m, candidate, points, weights)
                self.assertTrue(np.isfinite(metrics['dice']))


class RunnerEvidenceTests(unittest.TestCase):
    def run_synthetic(self, *, candidate=None, region='whole_heart', methods='G5',
                      eval_points=8, existing_record=None, expected_surface=None):
        """Patch medical acquisition only; execute actual record writing/metrics."""
        mask, affine, m, row = synthetic_case()
        if candidate is None:
            candidate = sphere()
        with tempfile.TemporaryDirectory(prefix='heart_method_review_') as directory:
            directory = Path(directory)
            config = directory/'config.json'
            config.write_text(json.dumps({'derived_root': 'synthetic-no-medical-root'}), encoding='utf-8')
            baseline = directory/'baseline'; baseline.mkdir()
            (baseline/'geometry_comparison.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
            output = directory/'output'
            if existing_record is not None:
                output.mkdir()
                (output/'synthetic_phase00_whole_heart__G5.json').write_text(
                    json.dumps(existing_record), encoding='utf-8')
            args = SimpleNamespace(config=str(config), baseline=str(baseline), output=str(output),
                selection='examples', region=region, methods=methods, fit_points=40,
                eval_points=eval_points, surface_budget=30, iou_budget=3, iou_step=2.)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner, 'get_mask', return_value=(mask, affine)))
                stack.enter_context(patch.object(runner, 'select_examples', return_value=[(row, row)]))
                stack.enter_context(patch.object(runner, 'candidates_for', return_value=candidate))
                if expected_surface is not None:
                    stack.enter_context(patch.object(runner, 'surface_quadrature', return_value=expected_surface))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runner.run(args)
            return json.loads((output/'results.json').read_text(encoding='utf-8'))

    def test_record_id_and_case_provenance_and_T0_baseline(self):
        result = self.run_synthetic(methods='T0,G5')
        self.assertFalse(result['FEM_executed_by_this_runner'])
        self.assertTrue(result['T0_is_automatic_target_phase_mask'])
        self.assertTrue(result['I_is_transformed_individual_reference_shape'])
        self.assertEqual([row['method_id'] for row in result['rows']], ['T0', 'G5'])
        for row in result['rows']:
            self.assertEqual(row['case_id'], 'synthetic_phase00_whole_heart')
            self.assertEqual(row['target_mask_sha256'], 'synthetic-mask')
            self.assertFalse(row['anatomical_reference_independent'])
        self.assertEqual(result['rows'][0]['metrics']['dice'], 1.)
        self.assertEqual(result['rows'][1]['parameters']['kind'], 'sphere_algebraic_unit_quadratic')

    def test_explicit_failed_wrapper_has_no_fallback_sphere(self):
        result = self.run_synthetic(candidate={
            'parameters': None, 'status': 'failed', 'reason': 'synthetic solver failure'})
        record = result['rows'][0]
        self.assertEqual(record['status'], 'failed')
        self.assertIsNone(record['metrics'])
        self.assertIsNone(record.get('parameters'))

    def test_unconverged_flat_candidate_keeps_its_geometry_not_G1(self):
        candidate = dict(sphere(1.7, kind='sphere_surface_fixed_volume'),
                         status='solver_not_converged', diagnostics={'trials': [{'success': False}]})
        result = self.run_synthetic(candidate=candidate)
        record = result['rows'][0]
        self.assertEqual(record['parameters']['kind'], 'sphere_surface_fixed_volume')
        self.assertEqual(record['parameters']['radius_mm'], 1.7)
        self.assertEqual(record['parameters']['status'], 'solver_not_converged')
        # The separate expected-failure test documents loss of the outer status.

    def test_solver_budget_exhaustion_must_survive_record_status(self):
        candidate = dict(sphere(), status='search_budget_exhausted',
                         diagnostics={'optimization_evaluations': 3, 'max_evaluations': 3})
        result = self.run_synthetic(candidate=candidate)
        self.assertEqual(result['rows'][0]['status'], 'search_budget_exhausted')

    def test_final_surface_RMS_must_use_full_quadrature(self):
        directions = unit_surface(32)
        radii = np.full(32, 2.4); radii[0] = 7.4
        points = 3+radii[:,None]*directions
        weights = np.ones(32)
        result = self.run_synthetic(eval_points=8,
            expected_surface=(points, weights, {'bounds_mm': [points.min(0).tolist(), points.max(0).tolist()]}))
        actual = result['rows'][0]['metrics']['target_surface_to_candidate_RMS_mm']
        expected = np.sqrt(np.mean((radii-2.4)**2))
        self.assertAlmostEqual(actual, expected, places=8)

    def test_resume_must_reject_foreign_case_or_method_record(self):
        foreign = {'case_id': 'other_subject_phase', 'method_id': 'L1', 'status': 'computed',
                   'metrics': {'dice': .5}}
        with self.assertRaises(ValueError):
            self.run_synthetic(existing_record=foreign)

    def test_G3c_must_honor_explicit_surface_budget(self):
        mask, affine, m, row = synthetic_case()
        args = SimpleNamespace(fit_points=40, surface_budget=7, iou_budget=3, iou_step=2.)
        with patch.object(runner, 'fit_surface_sphere_fixed_volume', return_value=sphere()) as fit:
            runner.candidates_for('G3c', row, row, mask, affine, mask, affine,
                m, m, 3+2.4*unit_surface(), np.full(40, 1/40),
                {'bounds_mm': [[.5]*3, [5.5]*3]}, args)
        self.assertEqual(fit.call_args.kwargs['max_nfev'], args.surface_budget)


if __name__ == '__main__':
    unittest.main()
