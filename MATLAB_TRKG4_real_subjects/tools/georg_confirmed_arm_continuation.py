"""Fresh 1x1-input search after the author's confirmed arm-wire correction.

Old outputs are immutable. Parameters are initial guesses only, with a new
observation identity. All controls and optimization use the canonical kernel.
The original whole-task deadline is retained; a new search allocation is explicit.
"""
from pathlib import Path
import argparse, copy, csv, os, shutil, time
import numpy as np
from georg_quality_rebuild_pipeline import QualityRun, PIPE, quote
from georg_quality_comparison import read, sha
from georg_quality_general_audit import compile_audit
from georg_quality_contact_coverage import patch_coverage
from scorpion_georg_fem_prepare import write


def prepare(main, out, search_deadline, evaluations=24):
    main, out = Path(main).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('A fresh corrected-input directory is required')
    cfg = copy.deepcopy(read(main / 'run_config.local.json'))
    if not time.time() < search_deadline < cfg['deadline_unix']:
        raise ValueError('Reserve controls within the original whole-task deadline')
    deep = main / 'postcheck_v2/deep_final'
    models = [(main / cfg['control_models'][0]).resolve(), deep / 'fine2_inherited', deep / 'fine2']
    if not all(read(p / 'mesh_quality.local.json')['passed'] for p in models):
        raise ValueError('All corrected-search meshes must pass shape QC')
    out.mkdir(parents=True)
    inputs = out / 'inputs'; inputs.mkdir()
    for name in ('observations.local.csv', 'inputs.local.json', 'pose_grid.local.json',
                 'author_centre_20261012.local.json'):
        shutil.copyfile(main / 'inputs' / name, inputs / name)
    with (inputs / 'observations.local.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    changed = []
    for row in rows:
        if 7 <= int(row['record_id'].rsplit('-', 1)[1]) <= 10:
            changed.append(dict(record_id=row['record_id'], channel=row['channel'], state=row['state'],
                                previous_arm_montage=row['arm_montage'], arm_montage='1x1'))
            row['arm_montage'] = '1x1'
    with (inputs / 'observations.local.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    confirmation = out / 'author_arm_confirmation_20261014.local.json'
    write(confirmation, dict(origin='direct human reply in current chat, 2026-10-14',
        cross_006_011_all_arm_wires_removed=True, late_ttrkg_007_010='1x1',
        physical_arm_pads_remain_an_assumption=True,
        changed_rows=changed, numerical_observation_values_changed=False,
        channels_axes_windows_changed=False, prior_observations_sha256=sha(main / 'inputs/observations.local.csv')))
    passport = read(inputs / 'inputs.local.json')
    passport.update(late_arm_montage_scenario='1x1', late_arm_montage_confirmed=True,
        late_arm_montage_alternatives=[],
        cross_forward_rule='all arm wires removed by author confirmation; physically remaining pads assumed independently floating',
        updated_protocol_interpretation='Author confirms 007-010 1x1 and complete wire removal for 006/011.',
        previous_input_contract_sha256=sha(main / 'inputs/inputs.local.json'),
        new_input_producer_sha256=sha(Path(__file__)),
        corrected_boundary_conditions=True)
    passport['outputs_sha256']['observations.local.csv'] = sha(inputs / 'observations.local.csv')
    passport['sources'].append(dict(path=str(confirmation), sha256=sha(confirmation)))
    write(inputs / 'inputs.local.json', passport)
    old = read(main / 'selected_candidate.local.json')['best']
    warm = dict(parameters=old['parameters'], pose=old['pose'],
        identity=dict(author_centre_sha256=sha(inputs / 'author_centre_20261012.local.json'),
                      observations_sha256=sha(inputs / 'observations.local.csv')),
        status='initial_guess_from_wrong_1x4_run_no_old_predictions_or_fit_identity_transferred',
        source_candidate_sha256=sha(main / 'selected_candidate.local.json'),
        boundary_correction_sha256=sha(confirmation))
    write(out / 'warm_seed.local.json', warm)
    grid = read(inputs / 'pose_grid.local.json')
    pose = copy.deepcopy(old['pose']); pose['pose_id'] = 'selected_previous_model'
    grid['poses'] = [pose]; write(inputs / 'pose_grid.local.json', grid)
    reference = read(main / 'reference_selected_v1.local.json')
    reference.update(observations_sha256=sha(inputs / 'observations.local.csv'),
        common_parameters_not_fitted_here=True, optimized_on_fine_remapped_mesh=False,
        boundary_correction_sha256=sha(confirmation),
        parameters_are_initial_guess_from_wrong_boundary_run=True)
    reference.pop('selected_candidate_sha256', None)
    write(out / 'reference_initial.local.json', reference)
    contract = read(main / 'candidate_model_contract.local.json')
    contract['fem_directory'] = models[-1].relative_to(PIPE).as_posix()
    contract['files'] = [r for r in contract['files'] if 'contact_mesh_v1/fine2' not in r['path']]
    for name in ('plan.local.json', 'mesh.mat', 'mesh_quality.local.json'):
        p = models[-1] / name
        contract['files'].append(dict(path=p.relative_to(PIPE).as_posix(), sha256=sha(p)))
    write(out / 'candidate_model_contract.local.json', contract)
    cfg.update(work_directory=out.relative_to(PIPE).as_posix(), input_directory='inputs',
        control_models=[os.path.relpath(p, out).replace('\\', '/') for p in models],
        initial_control_files=['coarse_initial.local.json', 'inherited_initial.local.json', 'remapped_initial.local.json'],
        initial_reference_file='reference_initial.local.json', initial_audit_file='initial_numerical_audit.local.json',
        stage_suffix='confirmed_1x1', run_label='confirmed_1x1_final_contacts',
        max_evaluations=evaluations, start_indices=[1], search_deadline_unix=search_deadline,
        additional_corrected_search_allocation=evaluations, original_whole_task_deadline_retained=True,
        old_search_budget_is_not_relabelled=True, independent_contact_master=True,
        confirmed_arm_correction=True, profile_max_evaluations=8)
    files = [Path(__file__), confirmation, *inputs.glob('*'), out / 'warm_seed.local.json',
             out / 'reference_initial.local.json', out / 'candidate_model_contract.local.json']
    cfg['sources'] += [dict(path=p.relative_to(PIPE).as_posix(), sha256=sha(p)) for p in files]
    write(out / 'run_config.local.json', cfg); write(out / 'run_config.initial.local.json', cfg)
    return cfg


class ConfirmedRun(QualityRun):
    def rel(self, p): return Path(p).resolve().relative_to(PIPE).as_posix()

    def matlab(self, label, args, search=False):
        if search:
            old = quote(self.rel(self.work / 'warm_seed.local.json'))
            new = quote(self.rel(self.work / 'warm_profile.local.json'))
            args = [new if x == old else x for x in args]
        return super().matlab(label, args, search)

    def run(self):
        try:
            self.publish()
            models = [(self.work / p).resolve() for p in self.config['control_models']]
            files = [self.work / p for p in self.config['initial_control_files']]
            for i, (model, file) in enumerate(zip(models, files, strict=True)):
                args = ["'Profile'", "'georg_quality_forward'", "'ModelDir'", quote(self.rel(model)),
                    "'InputDir'", quote(self.rel(self.work / 'inputs')), "'ReferenceFile'", quote(self.rel(self.work / 'reference_initial.local.json')),
                    "'OutputFile'", quote(self.rel(file))]
                if i == 0: args += ["'CheckDerivatives'", 'true']
                if not self.matlab(('coarse_initial', 'inherited_initial', 'remapped_initial')[i], args):
                    raise TimeoutError('Corrected initial control deadline')
            initial = compile_audit(*models, files)
            write(self.work / self.config['initial_audit_file'], initial)
            required = ('shape_QC_passed', 'energy_inheritance_passed', 'quadrature_passed', 'derivatives_passed', 'system_checks_passed')
            if not all(initial[k] for k in required):
                raise ValueError('Corrected preliminary numerical prerequisites failed')
            args = ["'Profile'", "'georg_lateral_inverse'", "'OutputDir'", quote(self.rel(self.work / 'inputs')),
                "'ModelDir'", quote(self.rel(models[-1])), "'ModelContractFile'", quote(self.rel(self.work / 'candidate_model_contract.local.json')),
                "'NumericalPolicy'", "'nix_preliminary'", "'NumericalAuditFile'", quote(self.rel(self.work / self.config['initial_audit_file'])),
                "'SearchMode'", "'profile_grid'", "'WarmStartFile'", quote(self.rel(self.work / 'warm_seed.local.json')),
                "'StartIndices'", '[1]', "'ProfileMaxEvaluations'", str(self.config['profile_max_evaluations']), "'ProfileMaxIterations'", '7']
            if not self.matlab('fixed_pose_corrected_materials', args):
                raise TimeoutError('Corrected conditional profile deadline')
            profile = read(self.work / 'inputs/fits_own_drive/selected_previous_model.local.json')
            warm = read(self.work / 'warm_seed.local.json')
            warm['parameters'][:6] = np.log([*profile['rho_inhale_ohm_m'], *profile['rho_exhale_ohm_m']]).tolist()
            bounds = np.log(np.asarray(read(self.work / 'inputs/inputs.local.json')['search_bounds_ohm_m']))
            for k in range(6):
                lo, hi = bounds[k % 3]
                for edge in (lo, hi):
                    if abs(warm['parameters'][k] - edge) <= 1e-8: warm['parameters'][k] = float(edge)
            warm.update(status='fresh_corrected_boundary_profile_initial_guess', profile_source_sha256=sha(self.work / 'inputs/fits_own_drive/selected_previous_model.local.json'))
            write(self.work / 'warm_profile.local.json', warm)
            self.config['sources'].append(dict(path=self.rel(self.work / 'warm_profile.local.json'), sha256=sha(self.work / 'warm_profile.local.json')))
            write(self.work / 'run_config.local.json', self.config)
            self.publish(fixed_pose_corrected_RMSE_ohm=profile['RMSE_ohm'])
            super().run()
            coverage = patch_coverage(self.work / 'coarse_forward_selected_v1.local.json', models[0] / 'mesh.mat', models[-1] / 'parent_maps.mat')
            write(self.work / 'contact_coverage_final.local.json', coverage)
            self.publish(status='complete_with_explicit_numerical_results',
                         all_final_contact_faces_refined=coverage['all_actual_contact_faces_refined'], global_mesh_convergence_proven=False)
        except BaseException as error:
            self.publish(status='failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--search-deadline-unix', type=float); p.add_argument('--evaluations', type=int, default=24)
    p.add_argument('--config', type=Path); a = p.parse_args()
    if a.config: ConfirmedRun(a.config).run()
    else:
        if not a.main or not a.out or not a.search_deadline_unix: p.error('Prepare needs main, out and deadline')
        prepare(a.main, a.out, a.search_deadline_unix, a.evaluations)
