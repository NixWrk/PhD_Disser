"""Small synthetic contracts/results only; never launch a FEM solve."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import run_surface_sensitivity as runner
DEPS = TOOLS.parent/'output/exploratory/arm_sigma_20260908/python_solver_deps'
runner.configure(DEPS)
import prepare_surface_sensitivity as prepare


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def seal_json(path, value):
    put(path, value)
    path.with_suffix('.sha256').write_text(runner.sha(path), encoding='ascii')


def make_fixture(base, montages=6):
    profile = dict(frequency_hz=50000, bounds=dict(soft=[2., 6.], heart=[1., 3.], lung=[4., 8.]), bone_fixed_rho_ohm_m=40.)
    put(base/'rho_profile.json', profile)
    source = {}
    for name, key in [('prepared_fem', 'prepared_sha256'), ('mesh', 'mesh_sha256'), ('surface', 'surface_sha256'), ('model_identity', 'model_identity_sha256'), ('mesh_acceptance', 'mesh_acceptance_sha256')]:
        path = base/('inputs/'+name+'.fixture')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('synthetic '+name).encode())
        source[name], source[key] = str(path.relative_to(base)), runner.sha(path)
    (base/'source_archive.zip').write_bytes(b'synthetic archive')
    parameters = [dict(id=k, block_indices=b, nominal_sigma=.2 if k != 'bone' else .025, fixed=k == 'bone') for k, b in [('soft', [0, 1]), ('heart', [2]), ('lung', [3]), ('bone', [4])]]
    inputs, contacts = [], []
    for i in range(2, 2+montages):
        mid = f'tepc_{i}'
        path = base/f'raw_json/{mid}.json'
        put(path, {'synthetic': mid})
        inputs.append(dict(id=mid, json=str(path.relative_to(base)), sha256=runner.sha(path)))
        contact = base/f'contacts/{mid}.mat'
        contact.parent.mkdir(parents=True, exist_ok=True)
        contact.write_bytes(('synthetic contact '+mid).encode())
        diagnostics = base/f'contacts/{mid}_diagnostics.csv'
        geometry = base/f'contacts/{mid}_geometry.json'
        diagnostics.write_text('synthetic diagnostics', encoding='utf-8')
        put(geometry, {'synthetic': mid})
        contacts.append(dict(
            status='passed',
            id=mid,
            contact_matrix=str(contact.relative_to(base)),
            contact_sha256=runner.sha(contact),
            diagnostics_file=str(diagnostics.relative_to(base)),
            diagnostics_sha256=runner.sha(diagnostics),
            geometry_file=str(geometry.relative_to(base)),
            geometry_sha256=runner.sha(geometry),
            source_json_sha256=runner.sha(path),
            matrix_error=1e-15,
        ))
    manifest = dict(source=source, inputs=inputs, frequency_hz=50000, parameters=parameters, archive_sha256=runner.sha(base/'source_archive.zip'), profile_sha256=runner.sha(base/'rho_profile.json'))
    seal_json(base/'manifest.json', manifest)
    put(base/'contacts/preparation.json', dict(status='passed', manifest_sha256=runner.sha(base/'manifest.json'), prepared_sha256=source['prepared_sha256'], montages=contacts))
    prepare.seal(base)


def fake_completion(base, plan, job, study, folder=None):
    folder = folder or base/job['output']
    folder.mkdir(parents=True, exist_ok=True)
    path, m = base/job['study'], study['montages'][0]
    identity = dict(study=study, study_sha256=job['study_sha256'], inputs={str((path.parent/study['source']['prepared_fem']).resolve()): plan['prepared_sha256'], str((path.parent/m['contact_matrix']).resolve()): m['contact_sha256']}, sources={k: plan['sources_sha256'][k] for k in runner.SOLVER}, threads=4, solver='PARDISO float64', status=runner.STATUS)
    put(folder/'identity.json', identity)
    (folder/'source_used.zip').write_bytes(b'SYNTHETIC_TEST_NOT_FEM')
    rows = []
    scale = int(job['montage'].split('_')[-1])
    for s in study['states']:
        rho = {k: 1/s['conductivity'][k] for k in runner.PARAMS}
        z = scale * sum((i+1)*rho[k]**2 for i, k in enumerate(runner.PARAMS))
        rows.append(dict(montage=job['montage'], state=s['id'], channel='standard', Z_ohm=z, reciprocal_Z_ohm=z, reciprocity_absolute_ohm=0., max_relative_residual=1e-12, **{'dZ_dsigma_'+k: -2*scale*(i+1)*rho[k]**3 for i, k in enumerate(runner.PARAMS)}))
    runner.pd.DataFrame(rows).to_csv(folder/'channel_results.csv', index=False)
    reseal_completion(folder, job['states'])
    return folder


def reseal_completion(folder, count):
    report = dict(status='completed_numerical_checks_passed', scientific_status=runner.STATUS, channel_states=count, maps=0, output_sha256={n: runner.sha(folder/n) for n in runner.ARTIFACTS})
    seal_json(folder/'completion.json', report)


class SurfaceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_dir = tempfile.TemporaryDirectory()
        cls.template = Path(cls.template_dir.name)/'fixture'
        make_fixture(cls.template)

    @classmethod
    def tearDownClass(cls):
        cls.template_dir.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)/'study'
        shutil.copytree(self.template, self.base)
        self.plan, self.profile, self.studies = runner.check_inputs(self.base)

    def save_plan(self):
        seal_json(self.base/'surface_plan.json', self.plan)

    def complete(self, jobs):
        for b in jobs:
            fake_completion(self.base, self.plan, b, self.studies[b['id']])

    def test_check_is_readonly_and_explicit_mode_required(self):
        before = {p.relative_to(self.base): runner.sha(p) for p in self.base.rglob('*') if p.is_file()}
        with patch.object(runner.subprocess, 'run') as process, redirect_stdout(io.StringIO()):
            runner.main(['--base', str(self.base), '--deps', str(DEPS), '--check'])
            process.assert_not_called()
        self.assertEqual(before, {p.relative_to(self.base): runner.sha(p) for p in self.base.rglob('*') if p.is_file()})
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            runner.main(['--base', str(self.base), '--deps', str(DEPS)])

    def test_manifest_without_plan_reports_contacts_pending(self):
        (self.base/'surface_plan.json').unlink()
        (self.base/'surface_plan.sha256').unlink()
        (self.base/'contacts/preparation.json').unlink()
        stdout = io.StringIO()
        with patch.object(runner.subprocess, 'run') as process, redirect_stdout(stdout):
            runner.main(['--base', str(self.base), '--deps', str(DEPS), '--check'])
        report = json.loads(stdout.getvalue())
        self.assertEqual(report['status'], 'contacts_pending')
        self.assertFalse(report['ready_for_fem'])
        process.assert_not_called()

    def test_hash_drift_refused_for_all_inputs(self):
        paths = ['surface_plan.json', 'manifest.json', 'rho_profile.json', 'contacts/preparation.json', 'raw_json/tepc_2.json', 'inputs/prepared_fem.fixture', 'inputs/mesh_acceptance.fixture', 'contacts/tepc_2.mat', self.plan['batches'][0]['study']]
        for name in paths:
            with self.subTest(name=name):
                path = self.base/name
                original = path.read_bytes()
                try:
                    path.write_bytes(original+b' ')
                    with self.assertRaises(ValueError):
                        runner.check_inputs(self.base)
                finally:
                    path.write_bytes(original)

    def test_sealed_source_drift_refused(self):
        self.plan['sources_sha256']['run_surface_sensitivity.py'] = '0'*64
        self.save_plan()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            runner.check_inputs(self.base)

    def test_missing_unexpected_duplicate_plan_states(self):
        original = copy.deepcopy(self.plan)
        for change in ('missing', 'duplicate', 'wrong_role', 'wrong_rho', 'missing_batch', 'duplicate_batch', 'output_escape'):
            with self.subTest(change=change):
                self.plan = copy.deepcopy(original)
                if change == 'missing': self.plan['states'].pop()
                if change == 'duplicate': self.plan['states'][-1] = self.plan['states'][0]
                if change == 'wrong_role': self.plan['states'][0]['role'] = 'check5'
                if change == 'wrong_rho': self.plan['states'][0]['rho']['soft'] += .2
                if change == 'missing_batch': self.plan['batches'].pop()
                if change == 'duplicate_batch': self.plan['batches'].append(self.plan['batches'][0])
                if change == 'output_escape': self.plan['batches'][0]['output'] = '../outside'
                self.save_plan()
                with self.assertRaises(ValueError): runner.check_inputs(self.base)

    def test_pilot_whole_compartment_state_contract(self):
        b = self.plan['pilots'][0]
        path = self.base/b['study']
        study = runner.read(path)
        # A hidden simultaneous nuisance perturbation must fail even with new seals.
        study['states'][1]['conductivity']['lung'] *= 1.001
        put(path, study)
        b['study_sha256'] = runner.sha(path)
        self.save_plan()
        with self.assertRaisesRegex(ValueError, 'Rho perturbation'):
            runner.check_inputs(self.base)

    def test_successful_pilot_math_and_receipt_revalidation(self):
        self.complete(self.plan['pilots'])
        report = runner.verified_pilot(self.base, self.plan, self.studies, save=True)
        self.assertEqual((report['states'], report['montages'], report['derivative_checks']), (78, 6, 18))
        self.assertTrue(report['passed'])
        self.assertEqual(report, runner.verified_pilot(self.base, self.plan, self.studies))
        for r in report['checks']:
            self.assertAlmostEqual(r['analytic_S'], r['fd_full'], places=7)
            self.assertAlmostEqual(r['analytic_S'], r['fd_half'], places=7)
        (self.base/'pilot_checks.json').write_text('{}')
        with self.assertRaises(ValueError): runner.verified_pilot(self.base, self.plan, self.studies)

    def test_wrong_derivative_sign_and_half_step_failure_block_pilot(self):
        self.complete(self.plan['pilots'])
        frame, sources = runner.load_frames(self.base, self.plan, self.studies, self.plan['pilots'])
        frame.loc[frame.state == 'pilot_base', 'dZ_dsigma_soft'] *= -1
        self.assertFalse(runner.pilot_checks(self.base, self.plan, frame, sources)['passed'])
        frame.loc[frame.state == 'pilot_base', 'dZ_dsigma_soft'] *= -1
        frame.loc[frame.state == 'pilot_soft_plus_half', 'Z_ohm'] += .1
        self.assertFalse(runner.pilot_checks(self.base, self.plan, frame, sources)['passed'])
        frame, sources = runner.load_frames(self.base, self.plan, self.studies, self.plan['pilots'])
        # Both differences may agree with S separately, yet disagree with each other.
        for mid in self.plan['montages']:
            S = 2*int(mid.split('_')[-1])*self.plan['axes_rho']['soft'][2]
            h = runner.np.ptp(self.plan['axes_rho']['soft'])*.001
            for step, divisor, factor in [('full', 1, 1.004), ('half', 2, .996)]:
                minus = frame.loc[(frame.montage == mid) & (frame.state == f'pilot_soft_minus_{step}'), 'Z_ohm'].iloc[0]
                frame.loc[(frame.montage == mid) & (frame.state == f'pilot_soft_plus_{step}'), 'Z_ohm'] = minus+2*h/divisor*S*factor
        report = runner.pilot_checks(self.base, self.plan, frame, sources)
        soft = next(r for r in report['checks'] if r['compartment'] == 'soft')
        self.assertTrue(all(abs(soft['analytic_S']-soft['fd_'+step]) <= tol for step, tol in zip(['full', 'half'], soft['tolerances'])))
        self.assertFalse(report['passed'])

    def test_missing_duplicate_and_unexpected_result_states(self):
        b = self.plan['pilots'][0]
        folder = fake_completion(self.base, self.plan, b, self.studies[b['id']])
        path = folder/'channel_results.csv'
        original = runner.pd.read_csv(path)
        for kind in ('missing', 'duplicate', 'unexpected'):
            with self.subTest(kind=kind):
                frame = original.copy()
                if kind == 'missing': frame = frame.iloc[:-1]
                if kind == 'duplicate': frame.loc[1, 'state'] = frame.loc[0, 'state']
                if kind == 'unexpected': frame.loc[1, 'state'] = 'not_planned'
                frame.to_csv(path, index=False)
                reseal_completion(folder, 13)
                with self.assertRaises(ValueError): runner.load_batch(self.base, self.plan, b, self.studies[b['id']])

    def test_completion_and_contact_provenance_refusal(self):
        b = self.plan['pilots'][0]
        folder = fake_completion(self.base, self.plan, b, self.studies[b['id']])
        identity = runner.read(folder/'identity.json')
        identity['inputs'][next(iter(identity['inputs']))] = '0'*64
        put(folder/'identity.json', identity)
        reseal_completion(folder, 13)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            runner.load_batch(self.base, self.plan, b, self.studies[b['id']])

    def test_duplicate_completions_refused(self):
        b = self.plan['pilots'][0]
        folder = fake_completion(self.base, self.plan, b, self.studies[b['id']])
        shutil.copytree(folder, folder.with_name(folder.name+'_attempt_01'))
        with self.assertRaisesRegex(ValueError, 'Duplicate completed'):
            runner.completed_folder(self.base, b)

    def test_numerical_qc_rechecked_even_when_output_resealed(self):
        b = self.plan['pilots'][0]
        folder = fake_completion(self.base, self.plan, b, self.studies[b['id']])
        path = folder/'channel_results.csv'
        frame = runner.pd.read_csv(path)
        frame.loc[0, 'max_relative_residual'] = 1e-3
        frame.to_csv(path, index=False)
        reseal_completion(folder, 13)
        with self.assertRaisesRegex(ValueError, 'residual/reciprocity'):
            runner.load_batch(self.base, self.plan, b, self.studies[b['id']])

    def test_csv_roundtrip_preserves_small_reciprocity_difference(self):
        b = self.plan['pilots'][0]
        folder = fake_completion(self.base, self.plan, b, self.studies[b['id']])
        path = folder/'channel_results.csv'
        frame = runner.pd.read_csv(path, float_precision='round_trip')
        z, zr = 102.33451869319245, 102.33451869319399
        frame.loc[0, ['Z_ohm', 'reciprocal_Z_ohm', 'reciprocity_absolute_ohm']] = [z, zr, abs(z-zr)]
        frame.to_csv(path, index=False)
        reseal_completion(folder, 13)
        loaded, _ = runner.load_batch(self.base, self.plan, b, self.studies[b['id']])
        self.assertEqual(loaded.loc[0, 'Z_ohm'], z)
        self.assertEqual(loaded.loc[0, 'reciprocal_Z_ohm'], zr)

    def test_full_without_verified_pilot_never_launches(self):
        with patch.object(runner, 'guard_older_run'), patch.object(runner.subprocess, 'run') as process:
            with self.assertRaises(ValueError):
                runner.main(['--base', str(self.base), '--deps', str(DEPS), '--matlab', 'matlab', '--full'])
            process.assert_not_called()

    def test_old_running_job_blocks_new_fem(self):
        with patch.object(runner, 'ROOT', self.base):
            put(self.base/'output/exploratory/resistivity_refinement_20260910/progress.json', {'status': 'running'})
            with self.assertRaisesRegex(ValueError, 'running'):
                runner.guard_older_run()

    def test_restart_uses_new_attempt_and_canonical_four_threads(self):
        self.complete(self.plan['pilots'][1:])
        b = self.plan['pilots'][0]
        partial = self.base/b['output']
        partial.mkdir(parents=True)
        (partial/'unfinished.txt').write_text('preserve me')
        (self.base/'logs').mkdir()
        # A crashed launch may leave only a log and no output directory.
        (self.base/'logs'/(b['id']+'_attempt_01.log')).write_text('failed before output')
        args = SimpleNamespace(matlab=Path('matlab'), python='python-test', deps=DEPS)
        def launch(command, **kw):
            self.assertEqual(command[:2], ['matlab', '-batch'])
            self.assertIn("run_pipeline('Profile','electrode_sensitivity'", command[2])
            self.assertIn("'Threads',4", command[2])
            self.assertEqual(kw['env']['OMP_NUM_THREADS'], '4')
            fake_completion(self.base, self.plan, b, self.studies[b['id']], partial.with_name(partial.name+'_attempt_02'))
        with patch.object(runner, 'guard_older_run'), patch.object(runner.subprocess, 'run', side_effect=launch) as process:
            runner.launch_stage(self.base, self.plan, self.studies, args, 'pilot')
        self.assertEqual(process.call_count, 1)
        self.assertEqual((partial/'unfinished.txt').read_text(), 'preserve me')
        self.assertTrue(runner.read(self.base/'pilot_checks.json')['passed'])
        self.assertEqual((self.base/'logs'/(b['id']+'_attempt_01.log')).read_text(), 'failed before output')

    def test_reusable_plan_supports_one_two_and_seven_montages(self):
        for n in (1, 2, 7):
            with self.subTest(n=n):
                base = Path(self.temp.name)/f'new_geometries_{n}'
                make_fixture(base, montages=n)
                plan, profile, studies = runner.check_inputs(base)
                self.assertEqual(len(plan['states']), 189*n)
                self.assertEqual(len(plan['pilots']), n)
                for b in plan['pilots']:
                    fake_completion(base, plan, b, studies[b['id']])
                report = runner.verified_pilot(base, plan, studies, save=True)
                self.assertEqual((report['states'], report['montages']), (13*n, n))
                if n <= 2:
                    for b in plan['batches']:
                        fake_completion(base, plan, b, studies[b['id']])
                    summary = runner.analyze(base, plan, profile, studies)
                    self.assertTrue(summary['adequate_on_tested_states'])
                    self.assertEqual(summary['states'], 189*n)
                    self.assertEqual(summary['pair_checks'], 64*3*n*(n-1)//2)
                    self.assertEqual(summary['pair_comparison_applicable'], n > 1)
                    self.assertEqual(len(runner.pd.read_csv(base/'analysis/pair_checks.csv')), summary['pair_checks'])
                    self.assertEqual(len(runner.pd.read_csv(base/'analysis/endpoints.csv')), 75*n)

    def test_empty_montage_list_is_rejected(self):
        self.plan['montages'] = []
        self.save_plan()
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            runner.check_inputs(self.base)

    def test_preparation_matrix_error_accepts_exact_tolerance(self):
        path = self.base/'contacts/preparation.json'
        prep = runner.read(path)
        for error, accepted in [(1e-12, True), (runner.np.nextafter(1e-12, runner.np.inf), False)]:
            with self.subTest(error=error):
                prep['montages'][0]['matrix_error'] = error
                put(path, prep)
                self.plan['preparation_sha256'] = runner.sha(path)
                self.save_plan()
                if accepted:
                    runner.check_inputs(self.base)
                else:
                    with self.assertRaisesRegex(ValueError, 'Unverified contact'):
                        runner.check_inputs(self.base)

    def test_full_synthetic_analysis_has_six_montages_and_no_legacy_reuse(self):
        self.complete(self.plan['batches'])
        result = runner.analyze(self.base, self.plan, self.profile, self.studies)
        self.assertEqual(result['states'], 1134)
        self.assertEqual(result['reused_states'], 0)
        self.assertTrue(result['adequate_on_tested_states'])
        self.assertFalse(result['physical_validation'])
        self.assertFalse(result['spatial_convergence_evaluated'])
        self.assertEqual(len(runner.pd.read_csv(self.base/'analysis/endpoints.csv')), 450)
        for filename, digest in result['output_sha256'].items():
            self.assertEqual(runner.sha(self.base/'analysis'/filename), digest)


if __name__ == '__main__':
    unittest.main()
