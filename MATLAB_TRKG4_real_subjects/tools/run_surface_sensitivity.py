"""Bounded TEPC whole-compartment rho study; every mode is explicit, no local probes.

Plan paths are relative to --base; study inputs relative to the study file.
Manifest/preparation/profile use the prepare_surface_sensitivity.seal contract:
manifest.json, contacts/preparation.json, rho_profile.json. sources_sha256 keys
name tools. An imported manifest without a sealed plan is contacts_pending.
Only --pilot and --full launch MATLAB, sequentially, with four solver threads.
"""
from pathlib import Path, PureWindowsPath
from contextlib import contextmanager
from itertools import product
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ('soft', 'heart', 'lung')
SOLVER = ('run_electrode_sensitivity.py', 'compartment_sensitivity.py', 'electrode_sensitivity_contract.py')
ANALYSIS = ('analyze_resistivity_refinement.py', 'analyze_resistivity_coverage.py',
            'resistivity_grid_convergence.py', 'resistivity_sensitivity_metrics.py')
ARTIFACTS = {'identity.json', 'source_used.zip', 'channel_results.csv'}
STATUS = 'exploratory_hypothesis_not_validated'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def relative(parent, value, contained=False):
    require(isinstance(value, str) and value and not PureWindowsPath(value).drive and
            not value.startswith(('/', '\\')) and '\x00' not in value, 'Expected relative path: ' + str(value))
    path = (parent / value).resolve()
    require(not contained or path.is_relative_to(parent.resolve()), 'Path escapes base: ' + value)
    return path


def configure(deps):
    if deps:
        sys.path.insert(0, str(Path(deps).resolve()))
    global np, pd, DEFAULT_CRITERIA, validate_study, verify_completed_output
    import numpy as np
    import pandas as pd
    from resistivity_grid_convergence import DEFAULT_CRITERIA
    from electrode_sensitivity_contract import validate_study
    from run_electrode_sensitivity import verify_completed_output


def pilot_states(plan):
    centre = {k: plan['axes_rho'][k][2] for k in PARAMS}
    result = {'pilot_base': centre}
    for k in PARAMS:
        h = (plan['axes_rho'][k][-1] - plan['axes_rho'][k][0]) * .001
        for sign, multiplier in [('plus', 1), ('minus', -1)]:
            for step, divisor in [('full', 1), ('half', 2)]:
                result[f'pilot_{k}_{sign}_{step}'] = dict(centre, **{k: centre[k] + multiplier*h/divisor})
    return result


def check_inputs(base):
    """Read/hash every sealed input once; no FEM, result writes or model loading."""
    plan_path = base / 'surface_plan.json'
    require(sha(plan_path) == (base/'surface_plan.sha256').read_text().strip(), 'Plan hash mismatch')
    plan = read(plan_path)
    montages = plan['montages']
    n = len(montages)
    require(n > 0 and len(set(montages)) == n, 'Expected nonempty unique montage IDs')
    hashes = {}
    def verify(path, expected):
        if path not in hashes:
            hashes[path] = sha(path)
        require(hashes[path] == expected, 'Input hash mismatch: ' + str(path))
    from prepare_surface_sensitivity import verify_manifest
    manifest = verify_manifest(base)
    verify(base/'manifest.json', plan['manifest_sha256'])
    verify(base/'contacts/preparation.json', plan['preparation_sha256'])
    verify(base/'rho_profile.json', plan['profile_sha256'])
    profile, prep = read(base/'rho_profile.json'), read(base/'contacts/preparation.json')
    require(prep['status'] == 'passed' and prep['manifest_sha256'] == plan['manifest_sha256'] and
            prep['prepared_sha256'] == manifest['source']['prepared_sha256'] == plan['prepared_sha256'], 'Contact preparation has not passed/is stale')
    contacts = {c['id']: c for c in prep['montages']}
    inputs = {m['id']: m for m in manifest['inputs']}
    require(len(inputs) == len(manifest['inputs']) == len(contacts) == len(prep['montages']) == n and
            set(contacts) == set(inputs) == set(plan['montages']), 'Prepared/input montage mismatch')
    for mid, c in contacts.items():
        require(c['source_json_sha256'] == inputs[mid]['sha256'] and
                np.isfinite(c['matrix_error']) and 0 <= c['matrix_error'] <= 1e-12, 'Unverified contact model: '+mid)
        verify(relative(base, c['contact_matrix']), c['contact_sha256'])
    for name, digest in [('prepared_fem', 'prepared_sha256'), ('mesh', 'mesh_sha256'), ('surface', 'surface_sha256'),
                         ('model_identity', 'model_identity_sha256'), ('mesh_acceptance', 'mesh_acceptance_sha256')]:
        hashes[relative(base, manifest['source'][name])] = manifest['source'][digest]
    for m in manifest['inputs']:
        hashes[relative(base, m['json'])] = m['sha256']
    sources = plan['sources_sha256']
    require(set(SOLVER + ANALYSIS + ('run_surface_sensitivity.py', 'prepare_surface_sensitivity.py')) <= set(sources), 'Incomplete solver/analysis source inventory')
    for name, digest in sources.items():
        verify(relative(ROOT/'tools', name, True), digest)
    require(plan['criteria'] == DEFAULT_CRITERIA and plan['pilot_h_fraction'] == .001, 'Unapproved numerical criteria')
    require(profile['frequency_hz'] == plan['frequency_hz'], 'Profile frequency mismatch')
    axes = plan['axes_rho']
    require(set(axes) == set(PARAMS), 'Expected three rho axes')
    for k in PARAMS:
        a = np.asarray(axes[k], dtype=float)
        require(a.shape == (5,) and np.isfinite(a).all() and (a > 0).all() and (np.diff(a) > 0).all(), 'Invalid axis: '+k)
        require(np.allclose(a[[0, -1]], profile['bounds'][k], rtol=1e-12, atol=1e-12), 'Profile bounds mismatch')
    expected_geometry = {}
    for indices in product(range(5), repeat=3):
        expected_geometry[tuple(axes[k][i] for k, i in zip(PARAMS, indices))] = 'grid3' if all(i % 2 == 0 for i in indices) else 'grid5_added'
    for indices in product(range(4), repeat=3):
        expected_geometry[tuple((axes[k][i]+axes[k][i+1])/2 for k, i in zip(PARAMS, indices))] = 'check5'
    states = {(s['montage'], s['state']): s for s in plan['states']}
    require(plan.get('reused_states', 0) == 0 and plan.get('total_required_states', 189*n) == 189*n and plan.get('new_fem_states', 189*n) == 189*n, 'Unexpected declared reuse/count')
    require(len(states) == len(plan['states']) == 189*n, 'Duplicate/missing/unexpected plan states')
    reference = None
    for m in montages:
        rows = [s for s in plan['states'] if s['montage'] == m]
        geometry = {tuple(s['rho'][k] for k in PARAMS): s['role'] for s in rows}
        require(len(rows) == len(geometry) == 189 and geometry == expected_geometry, 'Not a full 125+64 Cartesian design: '+m)
        named = {s['state']: (s['rho'], s['role']) for s in rows}
        require(reference is None or named == reference, 'State names must agree across montages')
        reference = named
    require(len(plan['pilots']) == n and {b['montage'] for b in plan['pilots']} == set(montages), 'Missing/duplicate pilot montage')
    jobs = plan['batches'] + plan['pilots']
    seen, ids, outputs, studies = set(), set(), [], {}
    prepared_paths = set()
    for b in jobs:
        require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', b['id']) and b['id'] not in ids, 'Invalid/duplicate batch ID')
        ids.add(b['id'])
        path = relative(base, b['study'], True)
        output = relative(base, b['output'], True)
        require(output != base.resolve() and all(output != p and not output.is_relative_to(p) and not p.is_relative_to(output) for p in outputs), 'Overlapping batch output paths')
        outputs.append(output)
        verify(path, b['study_sha256'])
        study = validate_study(read(path))
        require(b['montage'] in montages and len(study['states']) == b['states'], 'Batch state count/montage mismatch')
        source = study['source']
        require(source['prepared_sha256'] == plan['prepared_sha256'] and source['frequency_hz'] == plan['frequency_hz'], 'Prepared model/frequency mismatch')
        prepared = relative(path.parent, source['prepared_fem'])
        verify(prepared, plan['prepared_sha256'])
        prepared_paths.add(prepared)
        require([(p['id'], p['block_indices'], p['fixed']) for p in study['parameters']] ==
                [('soft', [0, 1], False), ('heart', [2], False), ('lung', [3], False), ('bone', [4], True)], 'Whole-compartment block contract mismatch')
        require(len(study['montages']) == 1, 'Expected one montage per batch')
        m = study['montages'][0]
        require(m['id'] == b['montage'] and m['electrodes'] == ['E1', 'E2', 'E3', 'E4'] and len(m['channels']) == 1, 'Electrode/channel contract mismatch')
        require(m['channels'][0]['drive'] == [1, 0, 0, -1] and m['channels'][0]['measure'] == [0, 1, -1, 0], 'Unexpected drive/measurement roles')
        c = contacts[m['id']]
        require(relative(path.parent, m['contact_matrix']) == relative(base, c['contact_matrix']) and m['contact_sha256'] == c['contact_sha256'], 'Study contact differs from prepared QC')
        verify(relative(path.parent, m['contact_matrix']), m['contact_sha256'])
        require(study['analysis'] == {'map_states': [], 'save_fields': False}, 'Local maps/field storage not part of this bounded plan')
        pilot = b in plan['pilots']
        wanted = pilot_states(plan) if pilot else {s['state']: s['rho'] for s in plan['states'] if s['montage'] == b['montage'] and s['role'] == b['role']}
        require(not pilot or b['states'] == 13, 'Expected 13 pilot states')
        actual = {s['id'] for s in study['states']}
        require(actual == set(wanted) if pilot else actual <= set(wanted), 'Unexpected study states')
        for s in study['states']:
            require(np.allclose([1/s['conductivity'][k] for k in PARAMS], [wanted[s['id']][k] for k in PARAMS], rtol=1e-12, atol=1e-12), 'Rho perturbation/state mismatch')
            require(np.isclose(s['conductivity']['bone'], 1/profile['bone_fixed_rho_ohm_m'], rtol=1e-12, atol=0), 'Bone conductivity changed')
            if not pilot:
                key = (b['montage'], s['id'])
                require(key not in seen, 'Duplicate state across batches')
                seen.add(key)
        studies[b['id']] = study
    require(seen == set(states) and len(prepared_paths) == 1, 'Incomplete full plan/shared model mismatch')
    require(all(not p.is_relative_to(out) and p != out for p in hashes for out in outputs), 'Output would overwrite sealed inputs')
    return plan, profile, studies


def completed_folder(base, batch):
    original = relative(base, batch['output'], True)
    candidates = [original] + sorted(original.parent.glob(original.name+'_attempt_*'))
    require(all(p.resolve().is_relative_to(base.resolve()) for p in candidates), 'Attempt path escapes base')
    completed = [p for p in candidates if (p/'completion.json').exists() or (p/'completion.sha256').exists()]
    require(len(completed) <= 1, 'Duplicate completed attempts: '+batch['id'])
    return completed[0] if completed else None


def load_batch(base, plan, batch, study):
    folder = completed_folder(base, batch)
    require(folder is not None, 'Unfinished batch: '+batch['id'])
    verify_completed_output(folder, ARTIFACTS, batch['states'], 0, STATUS)
    identity = read(folder/'identity.json')
    path = relative(base, batch['study'], True)
    m = study['montages'][0]
    inputs = {str(relative(path.parent, study['source']['prepared_fem'])): plan['prepared_sha256'],
              str(relative(path.parent, m['contact_matrix'])): m['contact_sha256']}
    require(identity['study'] == study and identity['study_sha256'] == batch['study_sha256'] and
            identity['inputs'] == inputs and identity['sources'] == {k: plan['sources_sha256'][k] for k in SOLVER} and
            identity['threads'] == 4 and identity['solver'] == 'PARDISO float64' and identity['status'] == STATUS, 'Completed batch provenance mismatch')
    frame = pd.read_csv(folder/'channel_results.csv', float_precision='round_trip')
    lookup = {s['id']: s['conductivity'] for s in study['states']}
    require(len(frame) == len(lookup) == batch['states'] and frame.state.is_unique and set(frame.state) == set(lookup), 'Missing/unexpected/duplicate result states')
    require(frame.montage.eq(batch['montage']).all() and frame.channel.eq(m['channels'][0]['id']).all(), 'Result montage/channel mismatch')
    numeric = ['Z_ohm', 'reciprocal_Z_ohm', 'max_relative_residual', 'reciprocity_absolute_ohm'] + ['dZ_dsigma_'+k for k in PARAMS]
    require(np.isfinite(frame[numeric].to_numpy(float)).all(), 'Nonfinite FEM result')
    residual = frame.max_relative_residual.to_numpy()
    rec = abs(frame.Z_ohm - frame.reciprocal_Z_ohm)
    require(((residual >= 0) & (residual < 1e-7)).all() and (rec <= 1e-8+1e-8*np.maximum(abs(frame.Z_ohm), abs(frame.reciprocal_Z_ohm))).all() and np.allclose(rec, frame.reciprocity_absolute_ohm, rtol=1e-10, atol=1e-14), 'FEM residual/reciprocity failed')
    for k in PARAMS:
        frame['sigma_'+k] = [lookup[s][k] for s in frame.state]
    frame['role'] = batch.get('role', 'pilot')
    return frame, {str((folder/'completion.json').relative_to(base.resolve())): sha(folder/'completion.json')}


def load_frames(base, plan, studies, jobs):
    frames, sources = [], {}
    for b in jobs:
        frame, source = load_batch(base, plan, b, studies[b['id']])
        frames.append(frame)
        sources.update(source)
    frame = pd.concat(frames, ignore_index=True)
    require(not frame.duplicated(['montage', 'state']).any(), 'Duplicate completed states')
    return frame, sources


def pilot_checks(base, plan, frame, sources):
    rows = []
    for montage in plan['montages']:
        g = frame[frame.montage == montage].set_index('state')
        require(len(g) == 13 and g.index.is_unique and set(g.index) == set(pilot_states(plan)), 'Pilot needs exactly 13 unique whole-compartment states')
        for k in PARAMS:
            h = np.ptp(plan['axes_rho'][k]) * .001
            S = -g.loc['pilot_base', 'sigma_'+k]**2 * g.loc['pilot_base', 'dZ_dsigma_'+k]
            fd = [(g.loc[f'pilot_{k}_plus_{step}', 'Z_ohm'] - g.loc[f'pilot_{k}_minus_{step}', 'Z_ohm'])/(2*h/divisor) for step, divisor in [('full', 1), ('half', 2)]]
            tol = [1e-8 + .005*max(abs(value), abs(S)) for value in fd]
            step_tol = 1e-8 + .005*max(map(abs, fd))
            passed = bool(np.isfinite([S, *fd]).all() and all(abs(value-S) <= t for value, t in zip(fd, tol)) and abs(fd[0]-fd[1]) <= step_tol)
            rows.append(dict(montage=montage, compartment=k, h=float(h), analytic_S=float(S), fd_full=float(fd[0]), fd_half=float(fd[1]), tolerances=tol, step_tolerance=step_tol, passed=passed))
    return dict(plan_sha256=sha(base/'surface_plan.json'), runner_sha256=sha(Path(__file__)), input_completion_sha256=sources,
                states=len(frame), montages=len(plan['montages']), derivative_checks=len(rows), pilot_h_fraction=.001,
                passed=all(r['passed'] for r in rows), checks=rows, physical_validation=False, spatial_convergence_evaluated=False)


def verified_pilot(base, plan, studies, save=False):
    frame, sources = load_frames(base, plan, studies, plan['pilots'])
    report = pilot_checks(base, plan, frame, sources)
    path = base/'pilot_checks.json'
    if save:
        write(path, report)
        path.with_suffix('.sha256').write_text(sha(path)+'\n', encoding='ascii')
    else:
        require(sha(path) == path.with_suffix('.sha256').read_text().strip() and read(path) == report, 'Pilot check receipt is stale/changed')
    require(report['passed'], 'Pilot derivative checks failed; full run is blocked')
    return report


def guard_older_run():
    path = ROOT/'output/exploratory/resistivity_refinement_20260910/progress.json'
    require(not path.exists() or read(path).get('status') != 'running', 'Older resistivity refinement is running; launch TEPC later (no automatic waiting).')


@contextmanager
def exclusive_run(base):
    # OS lock is released on process exit; a surviving file is not a stale lock.
    with (base/'launcher.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def launch_stage(base, plan, studies, args, mode):
    jobs = plan['pilots'] if mode == 'pilot' else plan['batches']
    progress = dict(status='running', mode=mode, completed_batches=0, completed_states=0, total_batches=len(jobs))
    for b in jobs:
        guard_older_run()
        require(check_inputs(base)[0] == plan, 'Plan changed during execution')
        if completed_folder(base, b) is None:
            original = relative(base, b['output'], True)
            logs = relative(base, 'logs', True)
            logs.mkdir(exist_ok=True)
            output, attempt = original, 0
            while output.exists() or (logs/(b['id']+f'_attempt_{attempt:02d}.log')).exists():
                attempt += 1
                output = original.with_name(original.name+f'_attempt_{attempt:02d}')
            q = lambda value: "'"+str(value).replace("'", "''")+"'"
            command = f"addpath({q(ROOT)});run_pipeline('Profile','electrode_sensitivity','StudyFile',{q(relative(base,b['study'],True))},'OutputDirectory',{q(output)},'PythonExecutable',{q(args.python)},'DependenciesDirectory',{q(args.deps.resolve())},'Threads',4);"
            progress['current_batch'] = b['id']
            write(base/'progress.json', progress)
            with (logs/(b['id']+f'_attempt_{attempt:02d}.log')).open('x', encoding='utf-8') as log:
                env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUTF8='1')
                subprocess.run([str(args.matlab), '-batch', command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        load_batch(base, plan, b, studies[b['id']])
        progress.update(completed_batches=progress['completed_batches']+1, completed_states=progress['completed_states']+b['states'])
        write(base/'progress.json', progress)
    require(check_inputs(base)[0] == plan, 'Plan changed during execution')
    if mode == 'pilot':
        verified_pilot(base, plan, studies, save=True)
    progress.update(status='pilot_checks_passed' if mode == 'pilot' else 'fem_complete_analysis_pending', current_batch='')
    write(base/'progress.json', progress)


def analyze(base, plan, profile, studies):
    from analyze_resistivity_refinement import to_arrays, audit_tables, score_tables
    from analyze_resistivity_coverage import add_metrics, endpoint_excursions
    from resistivity_grid_convergence import audit_grid, compare_nested_scorecards
    frame, sources = load_frames(base, plan, studies, plan['batches'])
    require(len(frame) == 189*len(plan['montages']), 'Incomplete full surface montage matrix')
    frame = add_metrics(frame, profile)
    axes, grid, points, direct = to_arrays(plan, frame)
    widths = np.array([np.ptp(profile['bounds'][k]) for k in PARAMS])
    audit = audit_grid(axes, grid, points, direct, widths, plan['criteria'])
    comparison = compare_nested_scorecards(axes, grid, widths, extra_values=direct, criteria=plan['criteria'])
    individual, pairs = audit_tables(plan, audit, points)
    if pairs.empty:
        pairs = pd.DataFrame(columns=['check_index', 'assembly_a', 'assembly_b', 'compartment', *['rho_'+k for k in PARAMS]] + [k+'_'+field for k in ('S', 'Q') for field in ('direct_difference', 'interpolated_difference', 'absolute_error', 'tolerance', 'error_over_tolerance', 'accuracy_pass', 'unresolved', 'preference_reversal')])
    cards, stability, preferred = score_tables(plan, comparison)
    endpoints = endpoint_excursions(frame[frame.role != 'check5'], profile, plan['axes_rho'])
    require(len(endpoints) == 75*len(plan['montages']), 'Incomplete endpoint comparisons')
    decisions = dict(complete_matrix=True, derivative_interpolation=bool(individual.S_pass.all()), selectivity_interpolation=bool(individual.Q_pass.all()), pair_S_accuracy=bool(audit['all_pair_S_accuracy_pass']), pair_Q_accuracy=bool(audit['all_pair_Q_accuracy_pass']), pair_preferences=bool(audit['all_pair_S_robust_sign_pass'] and audit['all_pair_Q_robust_sign_pass']), score_stability=bool(stability['pass'].all()), preferred_sets_stable=all(p['stable'] for p in preferred))
    require(check_inputs(base)[0] == plan, 'Inputs changed during analysis')
    out = relative(base, 'analysis', True)
    out.mkdir(exist_ok=True)
    tables = dict(states_rho=frame, interpolation_checks=individual, pair_checks=pairs, scorecards=cards, stability=stability, endpoints=endpoints)
    for name, table in tables.items():
        table.to_csv(out/(name+'.csv'), index=False)
    report = dict(status='completed_exploratory_surface_analysis', states=len(frame), montages=len(plan['montages']), pair_comparison_applicable=len(plan['montages']) > 1, derivative_checks=len(individual), pair_checks=len(pairs), reused_states=0, criteria=plan['criteria'], criteria_decisions=decisions, adequate_on_tested_states=all(decisions.values()), preferred_sets=preferred, plan_sha256=sha(base/'surface_plan.json'), runner_sha256=sha(Path(__file__)), input_completion_sha256=sources, sources_sha256=plan['sources_sha256'], output_sha256={name+'.csv': sha(out/(name+'.csv')) for name in tables}, continuous_domain_certified=False, spatial_convergence_evaluated=False, physical_validation=False)
    write(out/'summary.json', report)
    (out/'summary.sha256').write_text(sha(out/'summary.json')+'\n', encoding='ascii')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--deps', required=True, type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--matlab', type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('pilot', 'full', 'analyze', 'check'):
        modes.add_argument('--'+mode, action='store_true')
    args = parser.parse_args(argv)
    if (args.pilot or args.full) and args.matlab is None:
        parser.error('--matlab is required for FEM modes')
    configure(args.deps)
    base = args.base.resolve()
    if args.check and not (base/'surface_plan.json').exists() and not (base/'surface_plan.sha256').exists():
        from prepare_surface_sensitivity import verify_manifest
        manifest = verify_manifest(base)
        print(json.dumps(dict(status='contacts_pending', input_manifest_verified=True, ready_for_fem=False, montages=[m['id'] for m in manifest['inputs']])))
        return
    plan, profile, studies = check_inputs(base)
    if args.check:
        n = len(plan['montages'])
        print(json.dumps(dict(status='sealed_inputs_verified', ready_for_pilot=True, montages=n, full_states=189*n, pilot_states=13*n, fem_launched=False)))
    elif args.analyze:
        verified_pilot(base, plan, studies)
        print(json.dumps(analyze(base, plan, profile, studies), ensure_ascii=False))
    else:
        guard_older_run()
        with exclusive_run(base):
            if args.full:
                verified_pilot(base, plan, studies)
            try:
                launch_stage(base, plan, studies, args, 'pilot' if args.pilot else 'full')
            except BaseException as error:
                write(base/'progress.json', dict(status='failed', mode='pilot' if args.pilot else 'full', message=str(error)))
                raise


if __name__ == '__main__':
    main()
