"""Small saved-forward gate regressions: no FEM, MAT files or real source reads.

The mocked controller isolates numerical admission decisions from integration
identity. The real identity/declaration guard is tested separately with mocked I/O.
"""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import georg_convergence_continue as checks
import georg_quality_comparison as saved_checks


LEADS = [('002', 'longitudinal'), ('003', 'longitudinal'),
         ('004', 'longitudinal'), ('005', 'perpendicular'),
         ('006', 'longitudinal'), ('006', 'perpendicular'),
         ('007', 'longitudinal'), ('008', 'longitudinal'),
         ('009', 'longitudinal'), ('010', 'longitudinal'),
         ('011', 'longitudinal'), ('011', 'perpendicular')]


def forward_row(contrast=.01, coarse_contrast=None):
    identity = {name: 'a' * 64 for name in (
        'physical_contacts_sha256', 'reference_sha256',
        'observations_sha256', 'physical_montage_helper_sha256')}
    return dict(parameters=[0.] * 9, pose={'centre_lps_mm': [1., 2., 3.]},
        record_ids=[f'SCV2-G-{rid}' for rid, _ in LEADS for _ in range(2)],
        axes=[axis for _, axis in LEADS for _ in range(2)],
        states=['inhale', 'exhale'] * 12, target_Z_ohm=[80.] * 24,
        source_rule='own_drive', identity=identity,
        fine_Z_ohm=[80., 80. + contrast] * 12,
        coarse_Z_ohm=[80., 80. + (contrast if coarse_contrast is None else coarse_contrast)] * 12,
        quadrature_passed=True, system_checks_passed=True, patch_qc={'passed': True})


class FixedPointChecks(unittest.TestCase):
    def test_identical_point_and_all_pairs_pass(self):
        result = checks.fixed_point_compare(forward_row(), forward_row())
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['respiratory']['left']), 12)

    def test_parameters_pose_and_reference_changes_are_rejected(self):
        for kind in ('parameters', 'pose', 'reference_sha256'):
            with self.subTest(kind=kind):
                left, right = forward_row(), forward_row()
                if kind == 'parameters': right[kind][0] += 1e-12
                elif kind == 'pose': right[kind]['centre_lps_mm'][0] += 1e-12
                else: right['identity'][kind] = 'b' * 64
                with self.assertRaises(ValueError): checks.fixed_point_compare(left, right)

    def test_phase_order_mismatch_is_rejected(self):
        right = forward_row(); right['states'][0:2] = ['exhale', 'inhale']
        with self.assertRaises(ValueError): checks.fixed_point_compare(forward_row(), right)

    def test_missing_or_ambiguous_exhale_is_rejected(self):
        for ambiguous in (False, True):
            with self.subTest(ambiguous=ambiguous):
                row = forward_row()
                row['record_ids'][1 if not ambiguous else 3] = row['record_ids'][2 if not ambiguous else 1]
                with self.assertRaises(ValueError): checks.paired_contrasts(row, row['fine_Z_ohm'])

    def test_duplicate_inhale_and_orphan_exhale_are_rejected(self):
        row = forward_row()
        row['record_ids'][2] = row['record_ids'][0]
        row['axes'][2] = row['axes'][0]
        with self.assertRaises(ValueError): checks.paired_contrasts(row, row['fine_Z_ohm'])

    def test_contrast_failure_blocks_an_otherwise_passing_level_comparison(self):
        result = checks.fixed_point_compare(forward_row(.01), forward_row(.02))
        self.assertEqual(result['failed_values'], 0)
        self.assertFalse(result['respiratory']['passed'])
        self.assertFalse(result['passed'])

    def test_nested_rule_contrast_failure_is_isolated_from_seed_identity(self):
        result = checks.nested_rule_compare(forward_row(.01, coarse_contrast=.05))
        self.assertEqual(result['failed_values'], 0)
        self.assertTrue(result['reported_quadrature_passed'])
        self.assertFalse(result['respiratory']['passed'])
        self.assertFalse(result['passed'])


class NumericalAdmission(unittest.TestCase):
    def run_mocked_gate(self, *, audit_pass=True, contact_pass=True,
                        seed_contrasts=(.01, .01), coarse_contrast=None,
                        mesh_contrast=None):
        work = checks.PIPE / 'output/exploratory/synthetic_gate_no_files'
        job = checks.ConvergenceRun.__new__(checks.ConvergenceRun)
        job.work = work
        job.config = {'hdf5_library': None}
        job.wait_file = Mock()
        job.publish = Mock()
        job.start_inverse = Mock()
        job.forward = Mock(side_effect=lambda model, label, **kw: work / (label + '.local.json'))
        parent = forward_row(.01)
        low = forward_row(.01 if mesh_contrast is None else mesh_contrast)
        seed1 = forward_row(seed_contrasts[0], coarse_contrast)
        seed2 = forward_row(seed_contrasts[1], coarse_contrast)
        rows = {'parent_direct_forward.local.json': parent,
                'level3_inherited_forward.local.json': low,
                'level3_remapped_forward.local.json': low,
                'level3_direct_forward.local.json': low,
                'integration_high_seed_1_forward.local.json': seed1,
                'integration_high_seed_2_forward.local.json': seed2}
        # Default fixtures declare the two expected seeds. Controller tests use
        # a verified-identity stub; declaration/cache failures are isolated below.
        plan = dict(numerical_quadrature_points=[4096, 16384], integration_seed=20261008,
                    integration_stability={'schema': 'georg_integration_stability_v1',
                        'seed': 20261008, 'points': [4096, 16384]})
        def read_fake(path):
            path = Path(path)
            if path.name == 'plan.local.json':
                result = copy.deepcopy(plan)
                seed = 20261015 if path.parent.name.endswith('20261015') else 20261008
                result['integration_seed'] = result['integration_stability']['seed'] = seed
                return result
            return copy.deepcopy(rows[path.name])
        audit = dict(all_checks_passed=audit_pass,
            spatial={'passed': audit_pass}, fractions={'passed': audit_pass}, total={'passed': audit_pass})
        writes = {}
        with patch.object(checks, 'read', side_effect=read_fake), \
             patch.object(checks, 'sha', return_value='a' * 64), \
             patch.object(checks, 'write_new', side_effect=lambda path, value: writes.update({Path(path).name: copy.deepcopy(value)})), \
             patch.object(checks, 'compile_audit', return_value=audit), \
             patch.object(checks, 'compare_saved_controls', return_value={'passed': contact_pass}), \
             patch.object(checks, 'verify_independent_integrations', return_value={
                 'passed': True, 'seeds': [20261008, 20261015],
                 'actual_forward_plan_and_cache_verified': True}) as identity_check:
            job.run()
            identity_check.assert_called_once()
        return job, writes['initial_convergence_gate.local.json']

    def test_failed_spatial_or_contact_gate_never_starts_inverse(self):
        for audit_pass, contact_pass in ((False, True), (True, False)):
            with self.subTest(audit=audit_pass, contact=contact_pass):
                job, gate = self.run_mocked_gate(audit_pass=audit_pass, contact_pass=contact_pass)
                self.assertTrue(gate['integration_identity']['passed'])
                self.assertFalse(gate['passed'])
                job.start_inverse.assert_not_called()

    def test_failed_independent_seed_contrast_never_starts_inverse(self):
        job, gate = self.run_mocked_gate(seed_contrasts=(.01, .02))
        self.assertTrue(gate['integration_identity']['passed'])
        self.assertFalse(gate['comparisons']['independent_seed']['respiratory']['passed'])
        self.assertFalse(gate['passed'])
        job.start_inverse.assert_not_called()

    def test_nested_quadrature_contrast_failure_never_starts_inverse(self):
        self.assertTrue(checks.difference(forward_row(.05)['fine_Z_ohm'],
                                         forward_row(.01)['fine_Z_ohm'], relative=.001)['passed'])
        job, gate = self.run_mocked_gate(coarse_contrast=.05)
        self.assertTrue(gate['integration_identity']['passed'])
        self.assertFalse(gate['comparisons']['nested_rules_seed_1']['respiratory']['passed'])
        self.assertFalse(gate['passed'])
        job.start_inverse.assert_not_called()

    def test_spatial_contrast_failure_never_starts_inverse(self):
        self.assertTrue(checks.difference(forward_row(.01)['fine_Z_ohm'],
                                         forward_row(.05)['fine_Z_ohm'])['passed'])
        job, gate = self.run_mocked_gate(mesh_contrast=.05, seed_contrasts=(.05, .05))
        self.assertTrue(gate['integration_identity']['passed'])
        self.assertFalse(gate['comparisons']['spatial_respiratory']['respiratory']['passed'])
        self.assertFalse(gate['passed'])
        job.start_inverse.assert_not_called()

    def test_accepted_gate_calls_inverse_exactly_once(self):
        job, gate = self.run_mocked_gate()
        self.assertTrue(gate['passed'])
        job.start_inverse.assert_called_once()


class ResumeCompletedControls(unittest.TestCase):
    """Exercise the real forward resume branch with no files or MATLAB calls."""
    def invoke(self, *, enabled=True, explicit_reference=False, derivatives=False,
               change=None, reference_change=None, fd_rows=None, verification_error=None):
        work = checks.PIPE / 'output/exploratory/synthetic_resume_no_files'
        self.target = work/'saved_control.local.json'
        self.model = (work/'model').resolve()
        self.reference = work/('custom_reference.local.json' if explicit_reference
                              else 'reference_selected_v1.local.json')
        observations = work/'inputs/observations.local.csv'
        def sha_fake(path):
            return hashlib.sha256(Path(path).as_posix().encode('utf-8')).hexdigest()
        prior = forward_row()
        prior['pose']['electrodes'] = [dict(size_mm=60, axis='longitudinal',
            centres_lps_mm=[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])]
        prior['derivatives'] = (copy.deepcopy(fd_rows) if fd_rows is not None else
            [{'record_id': 'SCV2-G-006', 'passed': True},
             {'record_id': 'SCV2-G-011', 'passed': True}])
        prior['identity'].update(reference_sha256=sha_fake(self.reference),
            observations_sha256=sha_fake(observations), prepared_sha256='d'*64)
        expected = {key: copy.deepcopy(prior[key]) for key in ('parameters', 'pose')}
        if change == 'parameters': prior['parameters'][8] += 1e-12
        elif change == 'pose': prior['pose']['electrodes'][0]['centres_lps_mm'][1][2] += 1e-12
        elif change in ('observations_sha256', 'reference_sha256'):
            prior['identity'][change] = 'b'*64
        if reference_change == 'parameters': expected['parameters'][0] += 1e-12
        elif reference_change == 'pose':
            expected['pose']['electrodes'][0]['centres_lps_mm'][0][0] += 1e-12
        self.prior = copy.deepcopy(prior)
        job = checks.ConvergenceRun.__new__(checks.ConvergenceRun)
        job.work = work
        job.config = {'sources': []}
        if enabled is not None: job.config['resume_completed_controls'] = enabled
        job.matlab = Mock(side_effect=AssertionError('Resume must not launch MATLAB'))
        job.publish = Mock()
        self.job = job
        self.verifier = Mock(return_value=({'verified_plan': True}, self.model/'verified_cache.mat'),
                             side_effect=verification_error)
        def read_fake(path):
            path = Path(path)
            if path == self.target: return copy.deepcopy(prior)
            if path == self.reference: return copy.deepcopy(expected)
            raise AssertionError('Unexpected resume input read: ' + str(path))
        self.reader = Mock(side_effect=read_fake)
        with patch.object(Path, 'exists', autospec=True, side_effect=lambda path: path == self.target), \
             patch.object(checks, 'read', self.reader), \
             patch.object(checks, 'sha', side_effect=sha_fake), \
             patch.object(checks, 'verify_saved_identity', self.verifier), \
             patch.object(checks, 'write', side_effect=AssertionError('Resume must not rewrite config')), \
             patch.object(checks, 'write_new', side_effect=AssertionError('Resume must not rewrite results')):
            return job.forward(self.model, 'saved_control', derivatives=derivatives,
                               reference=self.reference if explicit_reference else None)

    def assert_verified_without_matlab(self):
        self.verifier.assert_called_once_with(self.prior, self.model,
                                              checks.PIPE/'src/run_georg_quality_forward.m')
        self.job.matlab.assert_not_called()

    def test_valid_resume_verifies_actual_model_and_skips_matlab(self):
        for explicit_reference in (False, True):
            with self.subTest(explicit_reference=explicit_reference):
                result = self.invoke(explicit_reference=explicit_reference, derivatives=True)
                self.assertEqual(result, self.target)
                self.assert_verified_without_matlab()
                self.reader.assert_any_call(self.reference)
                self.job.publish.assert_called_once_with(resumed_completed_control='saved_control')
                self.assertEqual(self.job.config['sources'], [])

    def test_disabled_or_missing_resume_setting_rejects_existing_output(self):
        for enabled in (False, None):
            with self.subTest(enabled=enabled), self.assertRaisesRegex(ValueError, 'No unverified reuse'):
                self.invoke(enabled=enabled)
            self.verifier.assert_not_called()
            self.reader.assert_not_called()
            self.job.matlab.assert_not_called()
            self.job.publish.assert_not_called()

    def test_changed_params_full_pose_reference_or_observations_rejects_resume(self):
        for change in ('parameters', 'pose', 'reference_sha256', 'observations_sha256'):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'frozen reference or observations'):
                self.invoke(change=change)
            self.assert_verified_without_matlab()
            self.job.publish.assert_not_called()

    def test_requested_reference_full_content_is_checked(self):
        for field in ('parameters', 'pose'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'frozen reference or observations'):
                self.invoke(explicit_reference=True, reference_change=field)
            self.assert_verified_without_matlab()
            self.reader.assert_any_call(self.reference)
            self.job.publish.assert_not_called()

    def test_requested_fd_requires_two_passed_controls(self):
        for rows in ([], [{'passed': True}], [{'passed': True}, {'passed': False}]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'derivative controls are incomplete'):
                self.invoke(derivatives=True, fd_rows=rows)
            self.assert_verified_without_matlab()
            self.job.publish.assert_not_called()

    def test_fd_is_not_required_when_not_requested(self):
        self.assertEqual(self.invoke(derivatives=False, fd_rows=[]), self.target)
        self.assert_verified_without_matlab()

    def test_cache_or_source_identity_failure_propagates_without_matlab(self):
        for message in ('Executed material cache changed', 'Prepared model source changed'):
            error = ValueError(message)
            with self.subTest(message=message), self.assertRaises(ValueError) as raised:
                self.invoke(verification_error=error)
            self.assertIs(raised.exception, error)
            self.assert_verified_without_matlab()
            self.job.publish.assert_not_called()


class ImmutableReceiptResume(unittest.TestCase):
    def job(self, enabled=True):
        job=object.__new__(checks.ConvergenceRun)
        job.config={'resume_completed_controls':enabled}
        return job

    def test_identical_existing_receipt_is_not_rewritten(self):
        result={'passed':True,'sources':[{'sha256':'a'*64}]}
        with patch.object(Path,'exists',return_value=True), \
             patch.object(checks,'read',return_value=copy.deepcopy(result)), \
             patch.object(checks,'write_new') as writer:
            self.job().save_receipt(Path('synthetic_receipt.json'),result)
            writer.assert_not_called()

    def test_changed_source_or_disabled_resume_rejects_receipt(self):
        result={'passed':True,'sources':[{'sha256':'a'*64}]}
        for enabled,prior in [(False,result),(True,{'passed':True,'sources':[{'sha256':'b'*64}]})]:
            with self.subTest(enabled=enabled), patch.object(Path,'exists',return_value=True), \
                 patch.object(checks,'read',return_value=prior), \
                 patch.object(checks,'write_new') as writer, self.assertRaises(ValueError):
                self.job(enabled).save_receipt(Path('synthetic_receipt.json'),result)
            writer.assert_not_called()

    def test_new_receipt_uses_exclusive_writer(self):
        path=Path('synthetic_receipt.json');result={'passed':False}
        with patch.object(Path,'exists',return_value=False), patch.object(checks,'write_new') as writer:
            self.job().save_receipt(path,result)
            writer.assert_called_once_with(path,result)


class IndependentIntegrationBindings(unittest.TestCase):
    """Real declaration/cache guard and real saved-identity checker, mocked I/O."""
    def verify(self, *, same_seed=False, changed_source=False, changed_cache=False,
               bad_metadata=False, bad_fractions=False, si_flag=True, length_unit='mm', node_scale=.001,
               bad_scale_helper=False):
        root = checks.PIPE / 'output/exploratory/synthetic_bindings_no_files'
        models = [root / f'integration_q16384_seed{seed}' for seed in (20261008, 20261015)]
        def stable_sha(path):
            path = Path(path)
            # One common bulk mesh; all other synthetic files have distinct hashes.
            value = 'common-mesh' if path.name == 'mesh.mat' else path.as_posix()
            return hashlib.sha256(value.encode('utf-8')).hexdigest()
        plans, metadata, rows = {}, {}, []
        for index, model in enumerate(models):
            seed = 20261008 if same_seed or index == 0 else 20261015
            plan = dict(numerical_quadrature_points=[4096, 16384], integration_seed=seed,
                integration_stability={'schema': 'georg_integration_stability_v1',
                    'seed': seed, 'points': [4096, 16384]},
                sources=[{'path': str(model/'sampler.py'), 'sha256': stable_sha(model/'sampler.py')}])
            plans[model/'plan.local.json'] = plan
            cache = model/'prepared_material_fixture.mat'
            row = forward_row()
            row['identity'].update(worker_sha256=stable_sha(checks.PIPE/'src/run_georg_quality_forward.m'),
                plan_sha256=stable_sha(model/'plan.local.json'), prepared_sha256=stable_sha(cache))
            rows.append(row)
            identity = dict(plan_sha256=stable_sha(model/'plan.local.json'),
                mesh_sha256=stable_sha(model/'mesh.mat'),
                worker_sha256=stable_sha(checks.PIPE/'src/trkg4_georg_lateral_materials.m'),
                quadrature_points=[4096, 16384], material_rules=dict(
                    quadrature_points=[4096, 16384], scale_nodes_to_si=si_flag,
                    length_unit=length_unit, node_scale_to_si=node_scale,
                    scale_helper_sha256='b'*64 if bad_scale_helper else stable_sha(checks.PIPE/'src/trkg4_scale_fmdl_to_si.m'),
                    fractions_sha256=[stable_sha(model/f'fractions_q{q}.mat') for q in (4096, 16384)]))
            metadata[cache] = identity
        if bad_metadata: metadata[models[1]/'prepared_material_fixture.mat']['plan_sha256'] = 'b'*64
        if bad_fractions:
            metadata[models[1]/'prepared_material_fixture.mat']['material_rules']['fractions_sha256'][1] = 'b'*64
        def sha_fake(path):
            path = Path(path)
            if path.parent == models[1] and ((changed_source and path.name == 'sampler.py') or
                                             (changed_cache and path.name == 'prepared_material_fixture.mat')):
                return 'c'*64
            return stable_sha(path)
        def read_fake(path): return copy.deepcopy(plans[Path(path)])
        def glob_fake(path, pattern):
            self.assertEqual(pattern, 'prepared_material_*.mat')
            return iter([path/'prepared_material_fixture.mat'])
        with patch.object(checks, 'read', side_effect=read_fake), \
             patch.object(checks, 'sha', side_effect=sha_fake), \
             patch.object(saved_checks, 'read', side_effect=read_fake), \
             patch.object(saved_checks, 'sha', side_effect=sha_fake), \
             patch.object(Path, 'glob', autospec=True, side_effect=glob_fake), \
             patch.object(checks, 'read_prepared_metadata', side_effect=lambda path, library:
                 (copy.deepcopy(metadata[Path(path)]), 0.)) as metadata_reader:
            result = checks.verify_independent_integrations(models, rows, None)
            if same_seed: metadata_reader.assert_not_called()
            else: self.assertEqual(metadata_reader.call_count, 2)
            return result

    def test_valid_distinct_seeds_and_actual_metadata_pass(self):
        self.assertTrue(self.verify()['passed'])

    def test_actual_hdf5_matlab_logical_one_is_accepted(self):
        self.assertTrue(self.verify(si_flag=1)['passed'])

    def test_invalid_flag_units_scale_or_helper_are_rejected(self):
        for kwargs in ({'si_flag':0},{'si_flag':1.0},{'si_flag':'1'},
                       {'length_unit':'m'},{'node_scale':1.},{'bad_scale_helper':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.verify(**kwargs)

    def test_same_seed_real_declaration_guard_rejects_before_cache_reads(self):
        result = self.verify(same_seed=True)
        self.assertFalse(result['passed'])
        self.assertEqual(result['seeds'], [20261008, 20261008])

    def test_changed_source_or_cache_is_rejected(self):
        for field in ('changed_source', 'changed_cache', 'bad_metadata', 'bad_fractions'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(**{field: True})


if __name__ == '__main__':
    unittest.main()
