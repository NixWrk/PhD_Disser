"""Fixed-point numerical checks first; joint inversion is gated by their success.

All FEM and inverse calls use run_pipeline. No relaxation of tolerances,
automatic shared-model promotion, source gain or channel reassignment.
"""
from pathlib import Path
import argparse
import copy
import datetime as dt
import json
import time
import numpy as np
from georg_quality_rebuild_pipeline import QualityRun, PIPE, quote
from georg_quality_selected_refinement import read, sha, write_new
from georg_quality_general_audit import compile_audit
from georg_quality_contact_variant import compare_saved_controls, read_prepared_metadata
from georg_quality_comparison import difference, verify_saved_identity
from scorpion_georg_fem_prepare import write


def paired_contrasts(row, values):
    values = np.asarray(values)
    if values.shape != (24,) or not np.all(np.isfinite(values)):
        raise ValueError('Expected 24 finite levels')
    grouped = {}
    for i, (record, axis, state) in enumerate(zip(row['record_ids'], row['axes'], row['states'], strict=True)):
        if state not in ('inhale', 'exhale'):
            raise ValueError('Unexpected respiratory state')
        key = (record, axis)
        states = grouped.setdefault(key, {})
        if state in states:
            raise ValueError('Duplicate respiratory state within a measured lead')
        states[state] = i
    if len(grouped) != 12 or any(set(states) != {'inhale', 'exhale'} for states in grouped.values()):
        raise ValueError('Expected exactly one inhale and exhale for every lead')
    pairs = []
    for i, state in enumerate(row['states']):
        if state == 'inhale':
            ids = [j for j, s in enumerate(row['states']) if s == 'exhale' and
                   row['record_ids'][j] == row['record_ids'][i] and row['axes'][j] == row['axes'][i]]
            if len(ids) != 1:
                raise ValueError('Respiratory pair is ambiguous')
            pairs.append(values[ids[0]] - values[i])
    if len(pairs) != 12:
        raise ValueError('All 12 respiratory pairs are required')
    return np.asarray(pairs)


def fixed_point_compare(left, right, relative=.001):
    for field in ('parameters', 'pose', 'record_ids', 'axes', 'states', 'target_Z_ohm', 'source_rule'):
        if left[field] != right[field]:
            raise ValueError('Fixed-point comparison changed ' + field)
    for field in ('physical_contacts_sha256', 'reference_sha256', 'observations_sha256',
                  'physical_montage_helper_sha256'):
        if left['identity'][field] != right['identity'][field]:
            raise ValueError('Fixed physical source changed ' + field)
    answer = difference(left['fine_Z_ohm'], right['fine_Z_ohm'], relative=relative)
    lc = paired_contrasts(left, left['fine_Z_ohm'])
    rc = paired_contrasts(right, right['fine_Z_ohm'])
    delta = np.abs(rc-lc)
    tolerance = .002 + .01*np.abs(rc)
    answer['respiratory'] = dict(left=lc.tolist(), right=rc.tolist(), difference_ohm=delta.tolist(),
        tolerance_ohm=tolerance.tolist(), tolerance_formula='0.002+0.01*abs(right contrast)',
        scope='Additional predeclared numerical screen, not measurement uncertainty',
        passed=bool(np.all(delta <= tolerance)))
    answer['passed'] = bool(answer['passed'] and answer['respiratory']['passed'])
    return answer


def nested_rule_compare(row):
    low = copy.deepcopy(row)
    low['fine_Z_ohm'] = row['coarse_Z_ohm']
    result = fixed_point_compare(low, row, relative=.001)
    result['reported_quadrature_passed'] = row['quadrature_passed']
    result['passed'] = result['passed'] and row['quadrature_passed']
    return result


def verify_independent_integrations(models, rows, library):
    plans = [read(model/'plan.local.json') for model in models]
    seeds = [plan.get('integration_seed') for plan in plans]
    declaration_ok = seeds == [20261008, 20261015] and all(
        plan.get('numerical_quadrature_points') == [4096, 16384] and
        plan.get('integration_stability', {}).get('schema') == 'georg_integration_stability_v1' and
        plan['integration_stability'].get('seed') == seed and
        plan['integration_stability'].get('points') == [4096, 16384]
        for plan, seed in zip(plans, seeds, strict=True))
    if not declaration_ok:
        return dict(passed=False, seeds=seeds, reason='Missing or non-independent integration declarations')
    if sha(models[0]/'mesh.mat') != sha(models[1]/'mesh.mat'):
        raise ValueError('Independent integration changed the mesh')
    facts = []
    for model, row in zip(models, rows, strict=True):
        plan, cache = verify_saved_identity(row, model, PIPE/'src/run_georg_quality_forward.m')
        identity, homogeneous = read_prepared_metadata(cache, library)
        if (identity['plan_sha256'] != sha(model/'plan.local.json') or
                identity['mesh_sha256'] != sha(model/'mesh.mat') or
                identity['worker_sha256'] != sha(PIPE/'src/trkg4_georg_lateral_materials.m') or
                identity['quadrature_points'] != [4096, 16384] or not 0 <= homogeneous < 1e-12):
            raise ValueError('Actual integration cache does not match its declared model')
        signature = identity['material_rules']
        if (signature['quadrature_points'] != [4096, 16384] or
                signature['fractions_sha256'] != [sha(model/f'fractions_q{q}.mat') for q in (4096,16384)] or
                signature['scale_nodes_to_si'] is not True or signature['node_scale_to_si'] != .001):
            raise ValueError('Actual integration fractions or SI units changed')
        facts.append(dict(model_plan_sha256=sha(model/'plan.local.json'), prepared_sha256=sha(cache),
                          mesh_sha256=sha(model/'mesh.mat'), seed=plan['integration_seed']))
    return dict(passed=True, seeds=seeds, actual_forward_plan_and_cache_verified=True, models=facts)


class ConvergenceRun(QualityRun):
    def rel(self, path):
        return Path(path).resolve().relative_to(PIPE).as_posix()

    def wait_file(self, path):
        while not path.is_file():
            if time.time() >= self.search_deadline:
                raise TimeoutError('Preparation/control wait exhausted the numerical search window')
            time.sleep(15)

    def forward(self, model, label, derivatives=False, reference=None):
        target = self.work/(label+'.local.json')
        if target.exists():
            raise ValueError('No unverified reuse of a previous forward result')
        model = Path(model).resolve()
        self.config['sources'].append(dict(path=self.rel(model/'plan.local.json'), sha256=sha(model/'plan.local.json')))
        write(self.work/'run_config_continuation.local.json', self.config)
        args = ["'Profile'", "'georg_quality_forward'", "'ModelDir'", quote(self.rel(model)),
            "'InputDir'", quote(self.rel(self.work/'inputs')),
            "'ReferenceFile'", quote(self.rel(reference or self.work/'reference_selected_v1.local.json')),
            "'OutputFile'", quote(self.rel(target))]
        if derivatives:
            args += ["'CheckDerivatives'", 'true']
        if not self.matlab(label, args):
            raise TimeoutError('Numerical FEM deadline')
        return target

    def run(self):
        try:
            # The independently launched parent calculation owns its process.
            # Do not run two enormous MATLAB factorizations concurrently.
            self.wait_file(self.work/'parent_direct_forward.local.json')
            parent = read(self.work/'parent_direct_forward.local.json')
            if not parent['patch_qc']['passed'] or not parent['system_checks_passed']:
                raise ValueError('Parent contact/system check failed')
            self.publish(status='checking_nested_level_3')
            fi = self.forward(self.work/'level3_inherited', 'level3_inherited_forward')
            fr = self.forward(self.work/'level3_remapped', 'level3_remapped_forward')
            models = [self.work/p for p in ('parent_direct', 'level3_inherited', 'level3_remapped')]
            audit = compile_audit(*models, [self.work/'parent_direct_forward.local.json', fi, fr])
            write_new(self.work/'level3_numerical_audit.local.json', audit)
            self.publish(status='checking_contact_discretization', mesh_space_passed=audit['spatial']['passed'],
                tissue_representation_passed=audit['fractions']['passed'], total_passed=audit['total']['passed'])
            fd = self.forward(self.work/'level3_direct', 'level3_direct_forward')
            contact = compare_saved_controls(self.work/'level3_remapped', self.work/'level3_direct', fr, fd,
                                             hdf5_library=self.config['hdf5_library'])
            write_new(self.work/'level3_contact_comparison.local.json', contact)
            high_models = [self.work/'integration_q16384_seed20261008', self.work/'integration_q16384_seed20261015']
            high_files = []
            for k, model in enumerate(high_models):
                self.wait_file(model/'plan.local.json')
                high_files.append(self.forward(model, f'integration_high_seed_{k+1}_forward', derivatives=k==0))
            rows = [read(p) for p in high_files]
            integration_identity = verify_independent_integrations(high_models, rows, self.config['hdf5_library'])
            comparisons = dict(
                spatial_respiratory=fixed_point_compare(parent, read(fi), relative=.005),
                tissue_respiratory=fixed_point_compare(read(fi), read(fr), relative=.005),
                contact_respiratory=fixed_point_compare(read(fr), read(fd), relative=.005),
                rule_growth=fixed_point_compare(read(fd), rows[0]),
                independent_seed=fixed_point_compare(rows[0], rows[1]))
            for k, row in enumerate(rows):
                comparisons[f'nested_rules_seed_{k+1}'] = nested_rule_compare(row)
            stability_passed = integration_identity['passed'] and all(r['passed'] for r in comparisons.values()) and all(
                r['quadrature_passed'] and r['system_checks_passed'] and r['patch_qc']['passed'] for r in rows)
            gate = dict(schema='georg_fixed_point_convergence_gate_v1',
                spatial_and_tissue_checks_passed=audit['all_checks_passed'],
                contact_discretization_passed=contact['passed'], integration_stability_passed=stability_passed,
                comparisons=comparisons, integration_identity=integration_identity,
                parameters=parent['parameters'], pose=parent['pose'],
                point_only=True, global_mesh_convergence_proven=False,
                quantitative_result_accepted=False, measurement_uncertainty_estimated=False,
                scope='Frozen pose and resistivities; 24 levels / 12 contrasts / finite connected and floating contacts',
                sources=[dict(path=self.rel(p),sha256=sha(p)) for p in
                    [self.work/'level3_numerical_audit.local.json', self.work/'level3_contact_comparison.local.json',
                     self.work/'reference_selected_v1.local.json', *high_files]])
            gate['passed'] = audit['all_checks_passed'] and contact['passed'] and stability_passed
            write_new(self.work/'initial_convergence_gate.local.json', gate)
            if not gate['passed']:
                self.publish(status='additional_refinement_required', numerical_gate_passed=False,
                    joint_inversion_started=False, failures_preserved=True)
                return
            self.start_inverse(high_models[0], rows[0], gate)
        except BaseException as error:
            self.publish(status='failed', error_type=type(error).__name__, error=str(error))
            raise

    def start_inverse(self, model, forward, gate):
        # Numerical admission is scoped to the tested point and protocol.
        # The shared geometry/model pointer is never changed here.
        prior = read(PIPE/'output/exploratory/georg_confirmed_arm_continuation_20261014/candidate_model_contract.local.json')
        prior.update(status='geometry_reference_checked', mesh_convergence_passed=False,
            integration_comparison_passed=False, quantitative_inversion_allowed=False,
            fem_directory=self.rel(model), numerical_precision_scope=gate['scope'],
            local_fixed_point_numerical_checks_passed=True,
            global_mesh_convergence_proven=False, independent_physiological_validation=False,
            acceptance_from_saved_gate_sha256=sha(self.work/'initial_convergence_gate.local.json'))
        prior['files'] = [r for r in prior['files'] if not any(
            name in r['path'] for name in ('/fine2/', '/fine2_inherited/'))]
        for p in [model/'plan.local.json', model/'mesh.mat', model/'mesh_quality.local.json',
                  self.work/'initial_convergence_gate.local.json']:
            prior['files'].append(dict(path=self.rel(p),sha256=sha(p)))
        contract = self.work/'point_checked_model_contract.local.json'
        write_new(contract, prior)
        self.config['sources'].append(dict(path=self.rel(contract),sha256=sha(contract)))
        local_audit = read(self.work/'level3_numerical_audit.local.json')
        local_audit.update(model_plan_sha256=sha(model/'plan.local.json'),
            prepared_sha256=forward['identity']['prepared_sha256'],
            numerical_precision_scope=gate['scope'], global_mesh_convergence_proven=False,
            initial_stronger_rule_gate_sha256=sha(self.work/'initial_convergence_gate.local.json'),
            interpretation='New qualified local numerical audit including stronger rules; not a global model admission')
        local_audit['derivatives_passed'] = len(forward['derivatives']) == 2 and all(
            r['passed'] for r in forward['derivatives'])
        if not local_audit['all_checks_passed'] or not local_audit['derivatives_passed']:
            raise ValueError('Stronger-rule local numerical prerequisites failed')
        local_audit['sources'] += [dict(path=self.rel(p),sha256=sha(p)) for p in
            [self.work/'initial_convergence_gate.local.json', model/'plan.local.json',
             self.work/'integration_high_seed_1_forward.local.json']]
        local_audit_file = self.work/'qualified_local_numerical_audit.local.json'
        write_new(local_audit_file, local_audit)
        self.config['sources'].append(dict(path=self.rel(local_audit_file),sha256=sha(local_audit_file)))
        self.publish(status='numerical_point_accepted_starting_material_profile', numerical_gate_passed=True,
            common_model_pointer_changed=False, physiological_validation=False)
        args = ["'Profile'", "'georg_lateral_inverse'", "'OutputDir'", quote(self.rel(self.work/'inputs')),
            "'ModelDir'", quote(self.rel(model)), "'ModelContractFile'", quote(self.rel(contract)),
            "'NumericalPolicy'", "'nix_preliminary'", "'NumericalAuditFile'", quote(self.rel(local_audit_file)),
            "'SearchMode'", "'profile_grid'",
            "'WarmStartFile'", quote(self.rel(self.work/'warm_seed.local.json')),
            "'ProfileMaxEvaluations'", '12', "'ProfileMaxIterations'", '8']
        if not self.matlab('fixed_pose_material_profile', args, search=True):
            self.publish(status='material_profile_time_limit', new_rhos_accepted=False)
            return
        profilefile = self.work/'inputs/fits_own_drive/selected_previous_model.local.json'
        profile = read(profilefile)
        warm = read(self.work/'warm_seed.local.json')
        warm['parameters'][:6] = np.log([*profile['rho_inhale_ohm_m'], *profile['rho_exhale_ohm_m']]).tolist()
        bounds = np.log(np.asarray(read(self.work/'inputs/inputs.local.json')['search_bounds_ohm_m']))
        for k in range(6):
            lo, hi = bounds[k % 3]
            for edge in (lo, hi):
                if abs(warm['parameters'][k]-edge) <= 1e-8:
                    warm['parameters'][k] = float(edge)
            if not lo <= warm['parameters'][k] <= hi:
                raise ValueError('Profile is outside the unchanged material bounds')
        warmfile = self.work/'warm_profile.local.json'
        write_new(warmfile, warm)
        self.config['sources'].append(dict(path=self.rel(warmfile),sha256=sha(warmfile)))
        self.publish(status='starting_joint_materials_and_placement', fixed_pose_RMSE_ohm=profile['RMSE_ohm'])
        args = ["'Profile'", "'georg_lateral_inverse'", "'OutputDir'", quote(self.rel(self.work/'inputs')),
            "'ModelDir'", quote(self.rel(model)), "'ModelContractFile'", quote(self.rel(contract)),
            "'NumericalPolicy'", "'nix_preliminary'", "'NumericalAuditFile'", quote(self.rel(local_audit_file)),
            "'WarmStartFile'", quote(self.rel(warmfile)),
            "'RunLabel'", quote(self.config['run_label']), "'MaxEvaluations'", str(self.config['max_evaluations']),
            "'CentreFile'", quote(self.rel(self.work/'inputs/author_centre_20261012.local.json')),
            "'CentreHalfWidthMM'", '[30,30]', "'AngleHalfWidthDeg'", '20',
            "'StartIndices'", '[1]', "'JointParameterOrder'", '[1,4,3,6,7,8,9,2,5]']
        complete = self.matlab('joint_materials_and_placement', args, search=True)
        directory = self.work/'inputs'/('joint_own_drive_'+self.config['run_label'])
        if complete:
            selected = read(directory/'execution.local.json')
        else:
            points = [read(p) for p in directory.glob('evaluation_*.local.json')]
            points = [p for p in points if p['status'] == 'numerical_candidate_not_validated']
            if not points:
                self.publish(status='joint_time_limit_no_completed_candidate')
                return
            selected = dict(best=min(points,key=lambda p:p['mean_square_ohm2']),
                search_stopped_for_time=True, resolution_reached=False, quantitative_result_accepted=False)
        target = self.work/'selected_new_candidate.local.json'
        write_new(target, selected)
        reference = read(self.work/'reference_selected_v1.local.json')
        reference.update(parameters=selected['best']['parameters'], pose=selected['best']['pose'],
                         selected_candidate_sha256=sha(target))
        ref = self.work/'reference_new_candidate.local.json'
        write_new(ref, reference)
        self.publish(status='checking_new_selected_candidate', quantitative_result_accepted=False)
        checked = []
        for name in ('parent_direct', 'level3_inherited', 'level3_remapped', 'level3_direct'):
            checked.append(self.forward(self.work/name, name+'_new_candidate',
                                        derivatives=name=='parent_direct', reference=ref))
        final = compile_audit(*[self.work/p for p in ('parent_direct','level3_inherited','level3_remapped')],checked[:3])
        final_contact = compare_saved_controls(self.work/'level3_remapped',self.work/'level3_direct',*checked[2:4],
                                              hdf5_library=self.config['hdf5_library'])
        write_new(self.work/'selected_new_numerical_audit.local.json', final)
        write_new(self.work/'selected_new_contact_comparison.local.json', final_contact)
        selected_high = [self.forward(self.work/name, name+'_new_candidate', reference=ref, derivatives=k==0)
            for k, name in enumerate(('integration_q16384_seed20261008', 'integration_q16384_seed20261015'))]
        selected_rows = [read(p) for p in selected_high]
        selected_identity = verify_independent_integrations(
            [self.work/name for name in ('integration_q16384_seed20261008','integration_q16384_seed20261015')],
            selected_rows,self.config['hdf5_library'])
        selected_integration = dict(
            spatial_respiratory=fixed_point_compare(read(checked[0]), read(checked[1]), relative=.005),
            tissue_respiratory=fixed_point_compare(read(checked[1]), read(checked[2]), relative=.005),
            contact_respiratory=fixed_point_compare(read(checked[2]), read(checked[3]), relative=.005),
            rule_growth=fixed_point_compare(read(checked[3]), selected_rows[0]),
            independent_seed=fixed_point_compare(*selected_rows))
        for k,row in enumerate(selected_rows):
            selected_integration[f'nested_rules_seed_{k+1}'] = nested_rule_compare(row)
        selected_integration['passed'] = selected_identity['passed'] and all(r['passed'] for r in selected_integration.values()) and all(
            r['quadrature_passed'] and r['system_checks_passed'] and r['patch_qc']['passed'] for r in selected_rows)
        selected_integration['selected_candidate_sha256'] = sha(target)
        selected_integration['quantitative_result_accepted'] = False
        selected_integration['integration_identity'] = selected_identity
        write_new(self.work/'selected_new_integration_stability.local.json', selected_integration)
        self.publish(status='joint_finished_with_selected_checks', selected_numerical_checks_passed=
            final['all_checks_passed'] and final_contact['passed'] and selected_integration['passed'],
            quantitative_result_accepted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a = p.parse_args()
    ConvergenceRun(a.config).run()
