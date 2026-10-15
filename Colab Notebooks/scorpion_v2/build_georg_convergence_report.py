"""Read-only progress/final reader for the independent Georg convergence run.

Reads completed saved results only; never launches preparation, MATLAB or FEM.
Large arrays are bound by recorded identities, not rehashed by this reader.
"""
from pathlib import Path
import argparse
import copy
import datetime as dt
import hashlib
import html
import io
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML

from build_georg_quality_repeat_report import PIPE, number, table, figure_html
from build_georg_contact_continuation_report import numbered_table

FOLDER = Path(__file__).resolve().parent
WORK = PIPE/'output/exploratory/georg_convergence_20261014'
STEM = '33.31_Численная_сходимость_и_совместный_подбор_Георгия_Scorpion_v2'
MODELS = ('parent_direct', 'level3_inherited', 'level3_remapped', 'level3_direct',
          'integration_q16384_seed20261008', 'integration_q16384_seed20261015')
LABELS = ('Родительская сетка: непосредственные контакты',
          'Третий уровень: унаследованные доли и контакты',
          'Третий уровень: пересчитанные доли, унаследованные контакты',
          'Третий уровень: непосредственные контакты',
          '16384 пробы: основное случайное перемешивание',
          '16384 пробы: независимое случайное перемешивание')
INITIAL_FILES = ('parent_direct_forward', 'level3_inherited_forward', 'level3_remapped_forward',
                 'level3_direct_forward', 'integration_high_seed_1_forward', 'integration_high_seed_2_forward')
FINAL_FILES = tuple(name+'_new_candidate' for name in MODELS)
RELOCATED_MODELS=('parent_direct','relocated_inherited','relocated_remapped','relocated_direct',
    'relocated_q16384_seed20261008','relocated_q16384_seed20261015')
RELOCATED_FILES=('parent_direct_new_candidate',*[name+'_forward' for name in RELOCATED_MODELS[1:]])
PALETTE = ('#777777', '#3989ad', '#8c61aa', '#dc872e', '#36895d', '#b35d8c', '#4252b1', '#ad492b', '#111111')
STATE_LABELS = {
    'parent_direct_forward_convergence_20261014': 'Первый прямой контроль родительской сетки',
    'level3_inherited_forward_convergence_20261014': 'Третий уровень: расчёт с унаследованными долями и контактами',
    'level3_remapped_forward_convergence_20261014': 'Третий уровень: расчёт с пересчитанными долями',
    'level3_direct_forward_convergence_20261014': 'Третий уровень: расчёт с непосредственными контактами',
    'integration_high_seed_1_forward_convergence_20261014': 'Прямой контроль усиленного правила интегрирования',
    'integration_high_seed_2_forward_convergence_20261014': 'Прямой контроль независимого перемешивания',
    'fixed_pose_material_profile_convergence_20261014': 'Условный подбор сопротивлений при фиксированном положении',
    'joint_materials_and_placement_convergence_20261014': 'Совместный подбор сопротивлений и положения',
    'checking_nested_level_3': 'Контроли третьего уровня сетки',
    'checking_contact_discretization': 'Проверка дискретизации контактов',
    'additional_refinement_required': 'Численный допуск не выдержан; требуется дальнейшее уточнение',
    'numerical_point_accepted_starting_material_profile': 'Численный допуск в точке выдержан; начинается подбор сопротивлений',
    'starting_joint_materials_and_placement': 'Совместный подбор сопротивлений и положения',
    'checking_new_selected_candidate': 'Проверки нового выбранного кандидата',
    'joint_finished_with_selected_checks': 'Поиск и проверки выбранного кандидата завершены',
    'material_profile_time_limit': 'Условный подбор остановлен по сроку',
    'joint_time_limit_no_completed_candidate': 'Поиск остановлен по сроку без готового кандидата',
    'failed': 'Процедура завершилась с ошибкой',
    'waiting_preexisting_selected_control': 'Ожидание ранее запущенного прямого контроля',
    'checking_new_selected_candidate_after_interruption': 'Проверка лучшей сохранённой точки после остановки процесса',
    'superseded_by_relocated_contact_checks': 'Контроли направлены на сетку под новые контактные площадки',
}
for model,label in zip(MODELS,LABELS,strict=True):
    STATE_LABELS[model+'_new_candidate_convergence_20261014']='Проверка нового кандидата: '+label
STATE_LABELS.update({key+suffix:value for key,value in tuple(STATE_LABELS.items())
    if key.endswith('_convergence_20261014') for suffix in ('_v3','_v4')})


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PIPE/path).resolve()


class Snapshot:
    def __init__(self):
        self.files = {}; self.data = {}; self.guard_records = {}; self.deferred = {}

    def bytes(self, path):
        path = Path(path).resolve()
        if path not in self.data:
            payload = path.read_bytes(); self.data[path] = payload
            self.files[str(path)] = hashlib.sha256(payload).hexdigest()
        return self.data[path]

    def json(self, path, optional=False):
        path = Path(path)
        if optional and not path.is_file():return None
        try:return json.loads(self.bytes(path).decode('utf-8-sig'))
        except (json.JSONDecodeError, UnicodeError):
            if optional:return None  # incomplete live publication stays pending
            raise

    def guard(self, rows):
        for row in rows:
            path = source_path(row['path']); expected = row['sha256']
            key = (str(path), expected)
            if key in self.guard_records:continue
            if path.suffix.lower() not in ('.m', '.py', '.json', '.csv', '.md'):
                self.deferred[key] = dict(path=str(path), sha256=expected,
                    verification='recorded_large_artifact_identity_not_rehashed');continue
            candidates = [path]
            if path.suffix.lower() in ('.m', '.py'):
                if row.get('snapshot'):candidates.append(source_path(row['snapshot']))
                try:relative = path.relative_to(PIPE)
                except ValueError:relative = None
                if relative is not None:
                    archive = PIPE/'archive/executed_georg_continuation_20261014'
                    candidates += [archive/relative, archive/'before_priority'/relative,
                                   archive/'priority_executed'/relative]
                    candidates += [root/relative for root in sorted(
                        (PIPE/'archive').glob('executed_georg_convergence_20261014*'))]
                    if relative.as_posix() == 'tools/georg_convergence_continue.py':
                        candidates.append(PIPE/'archive/executed_georg_convergence_20261014_v2'/relative)
            verified = None
            for candidate in candidates:
                if candidate.is_file():
                    digest = hashlib.sha256(self.bytes(candidate)).hexdigest()
                    if digest == expected:verified = candidate;break
            if verified is None:raise ValueError('Reader source SHA changed: '+str(path))
            self.guard_records[key] = dict(path=str(path), sha256=expected,
                verified_path=str(verified), verification='exact_bytes')


def paired_rows(observations):
    result = []
    for i, row in observations.iterrows():
        if row.state != 'inhale':continue
        matches = observations.index[(observations.state == 'exhale') &
            (observations.record_id == row.record_id) & (observations.axis == row.axis)].tolist()
        if len(matches) != 1:raise ValueError('Respiratory pair is ambiguous')
        axis = 'продольная' if row.axis == 'longitudinal' else 'поперечная'
        result.append((i, matches[0], row.record_id.rsplit('-', 1)[-1]+' · '+str(row.size_mm)+' мм · '+axis))
    if len(result) != 12 or len({i for a,b,_ in result for i in (a,b)}) != 24:
        raise ValueError('Expected exactly 24 levels in 12 respiratory pairs')
    return result


def vector(row):return np.asarray(row['fine_Z_ohm'] if 'fine_Z_ohm' in row else row['predicted_Z_ohm'], float)


def contrasts(d, values):return np.array([values[j]-values[i] for i,j,_ in d['pairs']])


def check_vector(d, row):
    z = vector(row)
    if z.shape != (24,) or not np.isfinite(z).all():raise ValueError('Incomplete/nonfinite prediction vector')
    np.testing.assert_allclose(row['target_Z_ohm'], d['target'], rtol=0, atol=1e-12)
    return z


def guard_worker(snap, path, digest):snap.guard([dict(path=str(path), sha256=digest)])


def validate_forward(d, row, model, reference):
    snap = d['snapshot'];plan = d['plans'][model];identity = row['identity']
    if identity['plan_sha256'] != snap.files[str((d['work']/model/'plan.local.json').resolve())]:
        raise ValueError('Forward belongs to another model plan')
    guard_worker(snap, PIPE/'src/run_georg_quality_forward.m', identity['worker_sha256'])
    if 'material_cache_helper_sha256' in identity:
        guard_worker(snap, PIPE/'src/trkg4_georg_lateral_materials.m', identity['material_cache_helper_sha256'])
    guard_worker(snap, PIPE/'src/trkg4_georg_physical_montages.m', identity['physical_montage_helper_sha256'])
    if identity['observations_sha256'] != d['observation_sha']:
        raise ValueError('Forward observation identity changed')
    reference_file, ref = reference
    if identity['reference_sha256'] != snap.files[str(reference_file.resolve())]:
        raise ValueError('Forward reference identity changed')
    if (row['parameters'],row['pose']) != (ref['parameters'],ref['pose']):
        raise ValueError('Forward is not at the full declared reference point')
    for field,column in [('record_ids','record_id'),('axes','axis'),('states','state')]:
        if row[field] != d['observations'][column].tolist():raise ValueError('Forward row order changed')
    check_vector(d, row)
    contact = identity['contact_inheritance']
    pins = {str(source_path(r['path'])):r['sha256'] for r in plan['sources']}
    for field,path in [('current_mesh_sha256',d['work']/model/'mesh.mat'),
            ('contact_parent_mesh_sha256',source_path(plan['exact_contact_parent_mesh']))]:
        if contact[field] != pins.get(str(path.resolve())):raise ValueError('Forward geometry/master identity changed')
    if plan.get('boundary_parent_map'):
        if contact['boundary_parent_map_sha256'] != pins.get(str(source_path(plan['boundary_parent_map']))):
            raise ValueError('Forward contact parent-map identity changed')
    elif contact['boundary_parent_map_sha256'] != '':raise ValueError('Unexpected contact parent map')
    contact_file = source_path(plan['contacts_file'])
    if identity['physical_contacts_sha256'] != snap.files.get(str(contact_file)):
        snap.guard([dict(path=str(contact_file),sha256=identity['physical_contacts_sha256'])])


def validate_fit(d, fit, joint=False):
    identity = fit['identity'];snap = d['snapshot'];check_vector(d, fit)
    for field,value in [('observations_sha256',d['observation_sha']),
            ('input_sha256',snap.files[str((d['work']/'inputs/inputs.local.json').resolve())])]:
        if identity[field] != value:raise ValueError('New fit uses other observations/passport')
    model = MODELS[4]
    if d['plans'][model] is None or identity['model_plan_sha256'] != snap.files[str((d['work']/model/'plan.local.json').resolve())]:
        raise ValueError('New fit uses another integration model')
    if fit.get('source_rule',identity['excitation_scenario']) != 'own_drive':raise ValueError('New fit changed the declared source scenario')
    if identity['search_mode'] != ('nix_joint' if joint else 'profile_grid'):
        raise ValueError('New fit uses another search mode')
    guard_worker(snap, PIPE/'src/run_georg_lateral_inverse.m', identity['worker_sha256'])
    if not d['gate'] or d['gate']['passed'] is not True:raise ValueError('New fit exists without a passed numerical gate')
    contract_file = d['work']/'point_checked_model_contract.local.json'
    contract = snap.json(contract_file)
    if (identity['model_contract_sha256'] != snap.files[str(contract_file.resolve())] or
            contract['acceptance_from_saved_gate_sha256'] != snap.files[str((d['work']/'initial_convergence_gate.local.json').resolve())]):
        raise ValueError('New fit is not bound to its point-only numerical admission')
    if (contract['status'] != 'geometry_reference_checked' or
            source_path(contract['fem_directory']) != (d['work']/model).resolve() or
            contract.get('local_fixed_point_numerical_checks_passed') is not True or
            any(contract.get(key) is not False for key in ('mesh_convergence_passed',
                'integration_comparison_passed','quantitative_inversion_allowed',
                'global_mesh_convergence_proven','independent_physiological_validation'))):
        raise ValueError('Local admission was changed into a global model acceptance')
    snap.guard(contract.get('files',[]))
    audit_file=d['work']/'qualified_local_numerical_audit.local.json';audit=snap.json(audit_file)
    snap.guard(audit.get('sources',[]))
    if (identity['numerical_policy'] != 'nix_preliminary' or
            identity['numerical_audit_sha256'] != snap.files[str(audit_file.resolve())] or
            audit['initial_stronger_rule_gate_sha256'] != snap.files[str((d['work']/'initial_convergence_gate.local.json').resolve())] or
            audit['model_plan_sha256'] != identity['model_plan_sha256'] or
            audit['prepared_sha256'] != identity['prepared_sha256'] or
            audit.get('global_mesh_convergence_proven') is not False or
            not audit['all_checks_passed'] or not audit['derivatives_passed']):
        raise ValueError('New fit lacks its qualified local numerical audit')
    rhos = np.asarray([*fit['rho_inhale_ohm_m'],*fit['rho_exhale_ohm_m']])
    if rhos.shape != (6,) or not np.isfinite(rhos).all() or np.any(rhos<=0):raise ValueError('Invalid fitted material vector')
    rmse=fit['RMSE_ohm'] if 'RMSE_ohm' in fit else np.sqrt(fit['mean_square_ohm2'])
    np.testing.assert_allclose(rmse,
        np.sqrt(np.mean((vector(fit)-d['target'])**2)),rtol=1e-10,atol=1e-10)
    if not joint:
        pose = d['reference'][1]['pose']
        if fit['centre_lps_mm'] != pose['centre_lps_mm'] or fit['electrodes'] != pose['electrodes']:
            raise ValueError('Conditional material fit changed the fixed placement')


def level_difference(left,right,relative=.005):
    delta = np.asarray(right)-np.asarray(left);tolerance=.02+relative*np.abs(right)
    return dict(signed_difference_ohm=delta, difference_ohm=abs(delta), tolerance_ohm=tolerance,
        failed_values=int(np.count_nonzero(abs(delta)>tolerance)), passed=bool(np.all(abs(delta)<=tolerance)))


def check_difference(saved, actual):
    for key in ('difference_ohm','tolerance_ohm','signed_difference_ohm'):
        if key in saved:np.testing.assert_allclose(saved[key],actual[key],rtol=0,atol=1e-12)
    if 'failed_values' in saved and saved['failed_values'] != actual['failed_values']:
        raise ValueError('Saved numerical failure count changed')
    # Gate differences can additionally include a stricter contrast screen.
    declared=saved.get('contact_difference_passed',saved['passed'])
    if 'respiratory' not in saved and declared != actual['passed']:
        raise ValueError('Saved numerical pass flag contradicts its vectors')


def check_screen(saved, actual, quadrature=True):
    check_difference(saved,actual);r=saved['respiratory']
    for key in ('left','right'):
        np.testing.assert_allclose(r[key],actual['contrast_'+key],rtol=0,atol=1e-12)
    for key,value in [('difference_ohm',abs(actual['contrast_delta'])),
            ('tolerance_ohm',actual['contrast_tolerance'])]:
        np.testing.assert_allclose(r[key],value,rtol=0,atol=1e-12)
    if (r['passed'] != actual['contrast_passed'] or
            saved['passed'] != bool(actual['passed'] and actual['contrast_passed'] and quadrature)):
        raise ValueError('Combined level/contrast/nested quadrature flag changed')


def contrast_difference(d,left,right):
    lc,rc=contrasts(d,np.asarray(left)),contrasts(d,np.asarray(right));delta=rc-lc
    tolerance=.002+.01*abs(rc)
    return dict(contrast_left=lc,contrast_right=rc,contrast_delta=delta,
        contrast_tolerance=tolerance,contrast_failed=int(np.count_nonzero(abs(delta)>tolerance)),
        contrast_passed=bool(np.all(abs(delta)<=tolerance)))


def check_integration_identity(d,receipt,rows,models=MODELS[4:]):
    if receipt['passed'] is not True:return
    if receipt['seeds'] != [20261008,20261015] or receipt.get('actual_forward_plan_and_cache_verified') is not True:
        raise ValueError('Independent integration receipt is incomplete')
    if len(receipt['models']) != 2:raise ValueError('Independent integration receipt lacks a model')
    for model,row,fact,seed in zip(models,rows,receipt['models'],receipt['seeds'],strict=True):
        plan=d['plans'][model];declared=plan.get('integration_stability',{})
        if (plan.get('integration_seed') != seed or plan['numerical_quadrature_points'] != [4096,16384] or
                declared.get('schema') != 'georg_integration_stability_v1' or
                declared.get('points') != [4096,16384] or declared.get('seed') != seed or
                fact['seed'] != seed or fact['model_plan_sha256'] != row['identity']['plan_sha256'] or
                fact['prepared_sha256'] != row['identity']['prepared_sha256'] or
                fact['mesh_sha256'] != row['identity']['contact_inheritance']['current_mesh_sha256']):
            raise ValueError('Independent integration receipt differs from actual saved plan/cache identities')


def validate_residual_receipt(d,receipt,candidate_file,candidate):
    if candidate is None:raise ValueError('Residual diagnostics lack their completed candidate')
    snap=d['snapshot'];pins={str(source_path(r['path'])):r['sha256'] for r in receipt['sources']}
    required=(candidate_file,d['work']/'inputs/observations.local.csv',d['work']/'inputs/inputs.local.json')
    for path in required:
        if pins.get(str(path.resolve())) != snap.files.get(str(path.resolve())):
            raise ValueError('Residual diagnostics candidate/input SHA changed')
    producer=PIPE/'tools/georg_inverse_residual_diagnostics.py'
    if str(producer.resolve()) not in pins:raise ValueError('Residual diagnostics producer is not pinned')
    snap.guard(receipt['sources'])
    if (receipt['schema']!='georg_completed_point_residual_diagnostics_v1' or
            receipt['observation_count']!=24 or receipt['pair_count']!=12 or
            receipt['quantitative_result_accepted'] is not False or receipt['independently_validated'] is not False or
            receipt['original_inverse_objective_unchanged'] is not True):
        raise ValueError('Residual diagnostics scope changed')
    z=check_vector(d,candidate);J=np.asarray(candidate['jacobian_logrho'],float)
    if J.shape!=(24,6) or not np.isfinite(J).all():raise ValueError('Candidate material Jacobian is incomplete')
    error=z-d['target'];sse=float(error@error)
    np.testing.assert_allclose([receipt['SSE_ohm2'],receipt['RMSE_ohm']],
        [sse,np.sqrt(sse/24)],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(receipt['mean_SSE_ohm2']+receipt['contrast_SSE_ohm2'],sse,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(receipt['contrast_SSE_fraction'],receipt['contrast_SSE_ohm2']/sse if sse else 0,rtol=1e-12,atol=1e-12)
    if len(receipt['pairs'])!=12:raise ValueError('Residual diagnostics pairs are incomplete')
    for pair,(i,j,_) in zip(receipt['pairs'],d['pairs'],strict=True):
        obs=d['observations'].iloc[i]
        if (pair['record_id'],pair['axis'])!=(obs.record_id,obs.axis):raise ValueError('Residual diagnostics pair order changed')
        np.testing.assert_allclose([pair['measured_contrast_ohm'],pair['model_contrast_ohm'],pair['mean_error_ohm']],
            [d['target'][j]-d['target'][i],z[j]-z[i],(error[i]+error[j])/2],rtol=0,atol=1e-12)
        np.testing.assert_allclose(pair['pair_SSE_fraction'],(error[i]**2+error[j]**2)/sse if sse else 0,
            rtol=1e-12,atol=1e-12)
    gradient=np.asarray(receipt['gradient_MSE_per_logrho'],float)
    if gradient.shape!=(6,) or not np.isfinite(gradient).all():raise ValueError('Residual gradient is incomplete')
    np.testing.assert_allclose(gradient,2*J.T@error/24,rtol=1e-12,atol=1e-12)
    params=np.asarray(candidate.get('parameters',np.log([*candidate['rho_inhale_ohm_m'],
        *candidate['rho_exhale_ohm_m']])),float)[:6]
    bounds=np.log(np.asarray(d['inputs']['search_bounds_ohm_m']))
    for field,edge in [('lower_bound_parameters',bounds[:,0]),('upper_bound_parameters',bounds[:,1])]:
        if receipt[field]!=np.isclose(params,np.tile(edge,2),rtol=0,atol=1e-4).tolist():
            raise ValueError('Material gradient bound classification changed')
    if [r['state'] for r in receipt['local_sensitivity']]!=['inhale','exhale']:
        raise ValueError('Sensitivity phase order changed')
    for r in receipt['local_sensitivity']:
        for key,shape in [('norms',(3,)),('cosine',(3,3)),('singular_values',(3,))]:
            value=np.asarray(r[key],float)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Incomplete local sensitivity')
    projection=receipt['bounded_linear_contrast_diagnostic']
    for key,shape in [('logrho_step',(6,)),('linear_contrast_ohm',(12,))]:
        value=np.asarray(projection[key],float)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Incomplete linear contrast diagnostic')
    if any(projection[k] is not False for k in ('local_linearization_validity_verified',
            'mean_levels_constrained','nonlinear_feasibility_or_impossibility_proven')):
        raise ValueError('Linear contrast diagnostic was relabelled as a nonlinear/physical result')
    for scenario in receipt['source_scenarios']:
        if (scenario['name'] not in ('own_drive','in_phase','opposite_phase','quadrature_magnitude') or
                scenario['experimentally_confirmed'] is not False or scenario['resistivities_or_pose_refitted'] is not False or
                scenario['equal_source_amplitudes_assumed'] != (scenario['name']!='own_drive')):
            raise ValueError('Unfitted source scenario scope changed')
        if np.asarray(scenario['contrast_ohm']).shape!=(12,):raise ValueError('Scenario contrasts are incomplete')


def validate_coverage_receipt(d,receipt):
    parent=d['forwards'][0];quality=d['mesh_quality'];snap=d['snapshot']
    if (parent is None or quality is None or
            receipt['coarse_forward_sha256']!=snap.files[str((d['work']/'parent_direct_forward.local.json').resolve())] or
            receipt['coarse_mesh_sha256']!=parent['identity']['contact_inheritance']['current_mesh_sha256'] or
            receipt['parent_maps_sha256']!=quality['parent_maps_sha256'] or
            receipt['global_mesh_convergence_proven'] is not False):
        raise ValueError('Contact coverage is not bound to the actual parent direct footprints/maps')
    guard_worker(snap,PIPE/'tools/georg_quality_contact_coverage.py',receipt['producer_sha256'])
    expected=[(r['record_id'],i+1,g) for r in parent['patches'] for i,g in enumerate(r['gids'])]
    actual=[(r['record_id'],r['physical_contact'],r['electrical_group']) for r in receipt['contacts']]
    if actual!=expected:raise ValueError('Contact coverage description order/physical groups changed')
    areas=np.asarray([r['area_mm2'] for r in receipt['contacts']])
    np.testing.assert_allclose(areas,[r['area_mm2'] for r in parent['patch_qc']['contacts']],rtol=0,atol=1e-9)
    fractions=np.asarray([r['refined_boundary_area_fraction'] for r in receipt['contacts']])
    if not np.isfinite(fractions).all() or np.any((fractions<0)|(fractions>1)):
        raise ValueError('Invalid contact coverage fractions')
    if (receipt['all_actual_contact_faces_refined']!=all(r['all_faces_refined'] for r in receipt['contacts']) or
            receipt['minimum_refined_boundary_area_fraction']!=float(min(fractions))):
        raise ValueError('Contact coverage aggregate changed')


def validate_mesh_sections(d,receipt):
    snap=d['snapshot'];candidate_file=d['work']/'selected_candidate.local.json'
    candidate=snap.json(candidate_file);candidate=candidate.get('best',candidate)
    if (receipt['schema']!='georg_frozen_pose_mesh_sections_v1' or receipt['coordinate_units']!='mm' or
            receipt['source_candidate_sha256']!=snap.files[str(candidate_file.resolve())] or
            (candidate['parameters'],candidate['pose'])!=(d['reference'][1]['parameters'],d['reference'][1]['pose']) or
            receipt['numerical_or_physiological_acceptance'] is not False or receipt['FEM_was_run'] is not False):
        raise ValueError('Mesh sections do not describe the frozen initial point')
    snap.guard(receipt['sources'])
    sections=receipt['sections']
    if len(sections)!=4 or {(s['model'],s['axis']) for s in sections}!={
            (m,a) for m in ('parent','child') for a in ('longitudinal','perpendicular')}:
        raise ValueError('Mesh section pair/axis set is incomplete')
    pose=d['reference'][1]['pose']
    for s in sections:
        model=MODELS[0] if s['model']=='parent' else MODELS[3]
        if (d['quality'][model] is None or s['mesh_sha256']!=d['quality'][model]['mesh_sha256'] or
                s['window_mm']!=[-80,80,-5,85] or s['is_plane_intersection'] is not True or
                s['is_2D_shape_quality_test'] is not False or len(s['x'])!=len(s['y'])):
            raise ValueError('Displayed section mesh/window/scope changed')
        horizontal=pose['primary_unit_xyz' if s['axis']=='longitudinal' else 'perpendicular_unit_xyz']
        np.testing.assert_allclose(s['centre_lps_mm'],pose['centre_lps_mm'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(s['horizontal_unit_xyz'],horizontal,rtol=0,atol=1e-12)
        np.testing.assert_allclose(s['inward_unit_xyz'],-np.asarray(pose['normal_unit_xyz']),rtol=0,atol=1e-12)


def validate_fine_coverage(d,receipt,forward_file,forward,quality=None):
    snap=d['snapshot'];quality=quality if quality is not None else d['mesh_quality']
    if (forward is None or quality is None or
            receipt['schema']!='georg_actual_fine_contact_coverage_v1' or
            receipt['forward_sha256']!=snap.files[str(forward_file.resolve())] or
            receipt['current_mesh_sha256']!=forward['identity']['contact_inheritance']['current_mesh_sha256'] or
            receipt['parent_maps_sha256']!=quality['parent_maps_sha256'] or
            receipt['forward_current_mesh_identity_verified'] is not True):
        raise ValueError('Fine coverage is not bound to its actual fine forward/mesh/maps')
    if any(receipt[k] is not False for k in ('volume_refinement_proven','global_mesh_convergence_proven',
            'whole_search_window_convergence_proven','independent_physical_validation','quantitative_result_accepted')):
        raise ValueError('Fine contact coverage was relabelled as volume/physical validation')
    snap.guard(receipt['sources'])
    guard_worker(snap,PIPE/'tools/georg_refined_contact_coverage.py',receipt['producer_sha256'])
    expected=[(r['record_id'],i+1,g,tuple(r['xyz'][i])) for r in forward['patches'] for i,g in enumerate(r['gids'])]
    actual=[(r['record_id'],r['physical_contact'],r['electrical_group'],tuple(r['centre_mm'])) for r in receipt['contacts']]
    if actual!=expected:raise ValueError('Fine coverage actual contact centres/groups changed')
    fractions=np.asarray([r['refined_boundary_area_fraction'] for r in receipt['contacts']])
    if not np.isfinite(fractions).all() or np.any((fractions<0)|(fractions>1)):
        raise ValueError('Invalid fine coverage fractions')
    if (receipt['physical_description_count']!=len(forward['patches']) or
            receipt['physical_contact_description_count']!=len(expected) or
            receipt['minimum_refined_boundary_area_fraction']!=float(min(fractions)) or
            receipt['all_contact_faces_refined']!=all(r['all_faces_refined'] for r in receipt['contacts'])):
        raise ValueError('Fine coverage aggregate changed')


def comparisons(d, final=False):
    rows = d['final_forwards'] if final else d['forwards'];result=[]
    definitions = [(0,1,'Пространство FEM',.005),(1,2,'Пересчёт долей тканей',.005),
        (0,2,'Суммарное изменение',.005),(2,3,'Непосредственный выбор контактных граней',.005),
        (3,4,'Увеличение числа проб',.001),(4,5,'Независимое случайное перемешивание',.001)]
    for a,b,name,relative in definitions:
        left,right = rows[a],rows[b]
        if left is None or right is None:result.append(dict(name=name,pending=True));continue
        for field in ('parameters','pose','target_Z_ohm','record_ids','axes','states','source_rule'):
            if left[field] != right[field]:raise ValueError('A numerical comparison changes '+field)
        if a in (1,2,3,4) and b in (2,3,4,5):
            if left['identity']['contact_inheritance']['current_mesh_sha256'] != right['identity']['contact_inheritance']['current_mesh_sha256']:
                raise ValueError('Isolated fraction/contact/integration comparison changed the volume mesh')
        actual=level_difference(vector(left),vector(right),relative)
        result.append(dict(name=name,pending=False,**actual,
            **contrast_difference(d,vector(left),vector(right)),relative=relative))
    for i,row in enumerate(rows[4:],4):
        name='Вложенные правила: '+('основное' if i==4 else 'независимое')+' перемешивание'
        if row is None:result.append(dict(name=name,pending=True));continue
        result.append(dict(name=name,pending=False,**level_difference(row['coarse_Z_ohm'],row['fine_Z_ohm'],.001),
            **contrast_difference(d,row['coarse_Z_ohm'],row['fine_Z_ohm']),relative=.001))
    return result


def load_configuration(snap,work):
    initial_file=work/'run_config.local.json';initial=snap.json(initial_file);snap.guard(initial['sources'])
    runtime_file=work/'run_config_continuation.local.json';runtime=snap.json(runtime_file,optional=True)
    versions=[]
    for path in work.glob('run_config_continuation.v*.local.json'):
        match=re.fullmatch(r'run_config_continuation\.v(\d+)\.local\.json',path.name)
        if match:versions.append((int(match[1]),path,snap.json(path)))
    versions.sort(key=lambda row:row[0])
    fixed=('work_directory','run_label','deadline_unix','control_models','initial_control_files',
        'max_evaluations','input_directory')
    for config in [runtime,*[row[2] for row in versions]]:
        if config is None:continue
        for key in fixed:
            if config[key] != initial[key]:raise ValueError('Continuation changed the frozen run contract: '+key)
        snap.guard(config['sources'])
    declared_by_suffix={initial['stage_suffix']:initial}
    for version,path,declared in versions:
        suffix=initial['stage_suffix'] if version==2 else initial['stage_suffix']+'_v'+str(version)
        reserve=2 if version==2 else 3 if version==3 else 4
        starts=[1] if version<5 else [1,2]
        if (declared['stage_suffix'] != suffix or
                (version>=3 and declared.get('resume_completed_controls') is not True) or
                declared['start_indices'] != starts or
                (version>=5 and declared.get('joint_starts_share_total_evaluation_budget') is not True) or
                abs(declared['deadline_unix']-declared['search_deadline_unix']-reserve*3600)>1e-6 or
                declared['search_deadline_unix']>initial['search_deadline_unix']):
            raise ValueError('Revised controller start/deadline contract changed')
        declared_by_suffix[suffix]=declared
    if runtime is not None:
        expected=declared_by_suffix.get(runtime['stage_suffix'])
        if expected is None or any(runtime[key]!=expected[key] for key in
                                  ('search_deadline_unix','start_indices')):
            raise ValueError('Runtime search deadline lacks its explicit matching configuration')
    config=versions[-1][2] if versions else runtime if runtime is not None else initial
    path=versions[-1][1] if versions else runtime_file if runtime is not None else initial_file
    return initial,config,path


def load(work=WORK):
    work=Path(work).resolve();snap=Snapshot();initial_cfg,cfg,cfg_file=load_configuration(snap,work)
    payload=snap.bytes(work/'inputs/observations.local.csv');obs=pd.read_csv(io.BytesIO(payload))
    quiet=obs[obs.state=='quiet_breathing'].copy()
    if len(quiet) != 1 or quiet.iloc[0].record_id != 'SCV2-G-001' or not np.isfinite(quiet.Z_ohm).all():
        raise ValueError('The independent quiet-breathing observation is missing or ambiguous')
    obs=obs[obs.state.isin(['inhale','exhale'])].reset_index(drop=True)
    if len(obs)!=24 or not np.isfinite(obs.Z_ohm).all():raise ValueError('The reader requires 24 finite measured levels')
    inputs=snap.json(work/'inputs/inputs.local.json');snap.guard(inputs['sources'])
    if (inputs['late_arm_montage_scenario']!='1x1' or not inputs['late_arm_montage_confirmed'] or
            inputs['annotation_acceptance_changed'] or inputs['annotation_status']!='pending_manual_review'):
        raise ValueError('Input connection/annotation contract changed')
    for _,row in obs.iterrows():
        record=int(row.record_id.rsplit('-',1)[-1])
        if 7<=record<=10 and row.arm_montage!='1x1':raise ValueError('Late confirmed arm montage changed')
    reference_file=work/'reference_selected_v1.local.json';reference=snap.json(reference_file)
    if reference['observations_sha256']!=snap.files[str((work/'inputs/observations.local.csv').resolve())]:
        raise ValueError('Initial reference does not bind these measurements')
    d=dict(work=work,snapshot=snap,config=cfg,initial_config=initial_cfg,configuration_file=cfg_file,
        inputs=inputs,observations=obs,quiet=quiet,target=obs.Z_ohm.to_numpy(),
        observation_sha=snap.files[str((work/'inputs/observations.local.csv').resolve())],
        pairs=paired_rows(obs),reference=(reference_file,reference),
        state=snap.json(work/'run_state.local.json',optional=True) or {},plans={},quality={},progress={})
    note_file=work/'contact_size_author_note.local.json';d['contact_size_note']=snap.json(note_file,optional=True)
    if d['contact_size_note'] is not None:
        note=d['contact_size_note']
        if (note['schema']!='georg_contact_size_author_note_v1' or
                note['current_frozen_FEM_diameter_mm']!=inputs['assumed_contact_diameter_mm'] or
                any(note[key] is not False for key in ('independently_measured',
                    'effective_electrical_contact_area_measured','existing_inputs_or_plans_changed'))):
            raise ValueError('Author contact-size note changed its uncertainty/frozen-scenario scope')
        snap.guard([dict(path=str(note_file),sha256=snap.files[str(note_file.resolve())])])
    for model in MODELS:
        base=work/model;plan=snap.json(base/'plan.local.json',optional=True);d['plans'][model]=plan
        if plan is not None:snap.guard(plan['sources'])
        d['quality'][model]=snap.json(base/'mesh_quality.local.json',optional=True)
        if plan is not None and d['quality'][model] is not None:
            pins={str(source_path(r['path'])):r['sha256'] for r in plan['sources']}
            if d['quality'][model]['mesh_sha256']!=pins.get(str((base/'mesh.mat').resolve())):
                raise ValueError('Displayed mesh quality is not bound to this model')
        d['progress'][model]=dict(contract=snap.json(base/'integration_contract.local.json',optional=True),
            checkpoint=snap.json(base/'fraction_checkpoint.local.json',optional=True))
    d['mesh_quality']=snap.json(work/'mesh_level_3_r3/fine2/mesh_quality.local.json',optional=True)
    d['preparation']=snap.json(work/'mesh_level_3_r3/preparation.local.json',optional=True)
    if d['preparation']:snap.guard(d['preparation']['sources'])
    d['reference_materials']=snap.json(work/'parent_direct/reference_materials.local.json')
    d['forwards']=[]
    for filename,model in zip(INITIAL_FILES,MODELS,strict=True):
        row=snap.json(work/(filename+'.local.json'),optional=True)
        if row is not None:validate_forward(d,row,model,d['reference'])
        d['forwards'].append(row)
    d['repeat']=snap.json(work/'parent_repeat_reproducibility.local.json',optional=True)
    if d['repeat'] is not None:
        receipt=d['repeat'];current=d['forwards'][0]
        oldfile=work.parent/'georg_confirmed_arm_continuation_20261014/contact_check_priority_v1/direct_forward.local.json'
        snap.guard([dict(path=str(oldfile),sha256=receipt['old_forward_sha256'])]);old=snap.json(oldfile)
        if current is None or receipt['new_forward_sha256'] != snap.files[str((work/'parent_direct_forward.local.json').resolve())]:
            raise ValueError('Reproducibility receipt is not bound to the current parent direct result')
        for key,field in [('parameters_equal_exactly','parameters'),('entire_pose_equal_exactly','pose'),
                ('targets_equal_exactly','target_Z_ohm')]:
            if receipt[key] is not True or old[field] != current[field]:raise ValueError('Reproducibility changed '+field)
        for field in ('record_ids','axes','states','source_rule'):
            if old[field] != current[field]:raise ValueError('Reproducibility changed row order/source')
        for field in ('physical_contacts_sha256','observations_sha256','physical_montage_helper_sha256'):
            if old['identity'][field] != current['identity'][field]:raise ValueError('Reproducibility changed physical source')
        for field in ('current_mesh_sha256','contact_parent_mesh_sha256','boundary_parent_map_sha256'):
            if old['identity']['contact_inheritance'][field] != current['identity']['contact_inheritance'][field]:
                raise ValueError('Reproducibility changed the parent mesh or contact lineage')
        maximum=float(np.max(abs(check_vector(d,old)-vector(current))))
        if receipt['maximum_Z_difference_ohm'] != maximum or receipt['physiological_validation'] is not False:
            raise ValueError('Reproducibility scope or difference changed')
        guard_worker(snap,PIPE/'src/trkg4_georg_lateral_materials.m',receipt['new_material_worker_sha256'])
    d['gate']=snap.json(work/'initial_convergence_gate.local.json',optional=True)
    d['initial_audit']=snap.json(work/'level3_numerical_audit.local.json',optional=True)
    d['initial_contact']=snap.json(work/'level3_contact_comparison.local.json',optional=True)
    for audit in (d['initial_audit'],d['initial_contact'],d['gate']):
        if audit is not None:snap.guard(audit.get('sources',[]))
    d['initial_differences']=comparisons(d)
    if d['initial_audit']:
        for key,index in [('spatial',0),('fractions',1),('total',2)]:
            if d['initial_differences'][index]['pending']:raise ValueError('Audit exists without its controls')
            check_difference(d['initial_audit'][key],d['initial_differences'][index])
    if d['initial_contact']:
        if d['initial_differences'][3]['pending']:raise ValueError('Contact comparison lacks its controls')
        check_difference(d['initial_contact'],d['initial_differences'][3])
    if d['gate']:
        gate=d['gate']
        if (gate['parameters'],gate['pose'])!=(reference['parameters'],reference['pose']):raise ValueError('Gate changed its point')
        expected=bool(gate['spatial_and_tissue_checks_passed'] and gate['contact_discretization_passed'] and gate['integration_stability_passed'])
        if (gate['passed']!=expected or gate['point_only'] is not True or gate['quantitative_result_accepted'] is not False or
                gate['global_mesh_convergence_proven'] is not False):
            raise ValueError('Numerical admission scope/logic changed')
        if d['initial_audit'] is None or d['initial_contact'] is None or (
                gate['spatial_and_tissue_checks_passed']!=d['initial_audit']['all_checks_passed'] or
                gate['contact_discretization_passed']!=d['initial_contact']['passed']):
            raise ValueError('Gate is not bound to its separate audit/contact outcomes')
        for key,index in [('spatial_respiratory',0),('tissue_respiratory',1),('contact_respiratory',3),
                ('rule_growth',4),('independent_seed',5),('nested_rules_seed_1',6),('nested_rules_seed_2',7)]:
            actual=d['initial_differences'][index]
            if actual['pending']:raise ValueError('Gate exists without completed integration controls')
            quadrature=d['forwards'][index-2]['quadrature_passed'] if index>=6 else True
            check_screen(gate['comparisons'][key],actual,quadrature)
        high=d['forwards'][4:]
        check_integration_identity(d,gate['integration_identity'],high)
        stability=gate['integration_identity']['passed'] and all(r['passed'] for r in gate['comparisons'].values()) and all(
            r['quadrature_passed'] and r['system_checks_passed'] and r['patch_qc']['passed'] for r in high)
        if gate['integration_stability_passed']!=stability:raise ValueError('Gate integration prerequisites changed')
    d['profile']=snap.json(work/'inputs/fits_own_drive/selected_previous_model.local.json',optional=True)
    if d['profile'] is not None:validate_fit(d,d['profile'])
    d['selected']=snap.json(work/'selected_new_candidate.local.json',optional=True)
    d['interrupted_state']=snap.json(work/'interrupted_run_state_20261015.local.json',optional=True)
    d['best']=d['selected'].get('best') if d['selected'] else None
    if d['best'] is not None:validate_fit(d,d['best'],joint=True)
    if d['selected'] and d['selected'].get('search_interrupted'):
        recovery=d['selected'];snap.guard(recovery['checkpoint_inventory'])
        source=source_path(recovery['source_checkpoint']);row=snap.json(source)
        if (row!=d['best'] or recovery['source_checkpoint_sha256']!=snap.files[str(source)] or
                recovery['quantitative_result_accepted'] is not False or
                recovery['resolution_reached'] is not False or
                recovery['search_not_resumed_after_search_deadline'] is not True):
            raise ValueError('Interrupted search selection lost its exact checkpoint/scope')
    new_reference_file=work/'reference_new_candidate.local.json';new_reference=snap.json(new_reference_file,optional=True)
    if new_reference is not None:
        if d['best'] is None or new_reference['selected_candidate_sha256']!=snap.files[str((work/'selected_new_candidate.local.json').resolve())]:
            raise ValueError('Final reference is not bound to the new candidate')
        if (new_reference['parameters'],new_reference['pose'])!=(d['best']['parameters'],d['best']['pose']):
            raise ValueError('Final reference changed new parameters/whole pose')
    d['final_forwards']=[]
    for filename,model in zip(FINAL_FILES,MODELS,strict=True):
        row=snap.json(work/(filename+'.local.json'),optional=True)
        if row is not None:
            if new_reference is None:raise ValueError('Selected control lacks its own new reference')
            validate_forward(d,row,model,(new_reference_file,new_reference))
        d['final_forwards'].append(row)
    d['final_audit']=snap.json(work/'selected_new_numerical_audit.local.json',optional=True)
    d['final_contact']=snap.json(work/'selected_new_contact_comparison.local.json',optional=True)
    d['final_integration']=snap.json(work/'selected_new_integration_stability.local.json',optional=True)
    for audit in (d['final_audit'],d['final_contact'],d['final_integration']):
        if audit is not None:snap.guard(audit.get('sources',[]))
    if d['final_integration'] and d['final_integration']['selected_candidate_sha256']!=snap.files.get(str((work/'selected_new_candidate.local.json').resolve())):
        raise ValueError('Final integration belongs to another selected candidate')
    d['final_differences']=comparisons(d,True)
    if d['final_audit']:
        for key,index in [('spatial',0),('fractions',1),('total',2)]:
            if d['final_differences'][index]['pending']:raise ValueError('Final audit lacks its own control')
            check_difference(d['final_audit'][key],d['final_differences'][index])
    if d['final_contact']:
        if d['final_differences'][3]['pending']:raise ValueError('Final contact comparison lacks its own controls')
        check_difference(d['final_contact'],d['final_differences'][3])
    if d['final_integration']:
        for key,index in [('spatial_respiratory',0),('tissue_respiratory',1),('contact_respiratory',3),
                ('rule_growth',4),('independent_seed',5),('nested_rules_seed_1',6),('nested_rules_seed_2',7)]:
            actual=d['final_differences'][index]
            if actual['pending']:raise ValueError('Final integration lacks its own controls')
            quadrature=d['final_forwards'][index-2]['quadrature_passed'] if index>=6 else True
            check_screen(d['final_integration'][key],actual,quadrature)
        high=d['final_forwards'][4:];receipt=d['final_integration']['integration_identity']
        check_integration_identity(d,receipt,high)
        expected=receipt['passed'] and all(d['final_integration'][key]['passed'] for key in (
            'spatial_respiratory','tissue_respiratory','contact_respiratory','rule_growth',
            'independent_seed','nested_rules_seed_1','nested_rules_seed_2')) and all(
            r['quadrature_passed'] and r['system_checks_passed'] and r['patch_qc']['passed'] for r in high)
        if d['final_integration']['passed'] != expected:raise ValueError('Final integration prerequisites changed')
    directory=work/'inputs'/('joint_own_drive_'+cfg['run_label']);d['checkpoints']=[]
    for path in sorted(directory.glob('evaluation_*.local.json')):
        row=snap.json(path,optional=True)
        if row is None:continue
        if row['status']=='numerical_candidate_not_validated':validate_fit(d,row,joint=True)
        d['checkpoints'].append(row)
    d['hypotheses']=snap.json(work/'review_hypotheses.local.json',optional=True)
    if d['hypotheses']:
        snap.guard(d['hypotheses']['source_pins'])
        if d['hypotheses']['quantitative_result_accepted'] is not False:raise ValueError('Historical review scope changed')
    d['coverage']=snap.json(work/'level3_actual_contact_coverage.local.json',optional=True)
    if d['coverage'] is not None:validate_coverage_receipt(d,d['coverage'])
    d['fine_coverage']={}
    for key,filename,forward_file,forward in [
            ('initial','initial_fine_actual_contact_coverage',work/'integration_high_seed_1_forward.local.json',d['forwards'][4]),
            ('selected','selected_fine_actual_contact_coverage',work/(FINAL_FILES[4]+'.local.json'),d['final_forwards'][4])]:
        receipt=snap.json(work/(filename+'.local.json'),optional=True)
        if receipt is not None:validate_fine_coverage(d,receipt,forward_file,forward)
        d['fine_coverage'][key]=receipt
    d['coverage_probe']=snap.json(work/'selected_contact_geometry_probe_coverage.local.json',optional=True)
    if d['coverage_probe'] is not None:
        receipt=d['coverage_probe'];probe_file=work/'selected_contact_geometry_probe.local.json';probe=snap.json(probe_file)
        snap.guard(receipt['sources']);guard_worker(snap,PIPE/'tools/georg_refined_contact_coverage.py',receipt['producer_sha256'])
        if (d['best'] is None or receipt['forward_sha256']!=snap.files[str(probe_file.resolve())] or
                receipt['current_mesh_sha256']!=d['forwards'][4]['identity']['contact_inheritance']['current_mesh_sha256'] or
                receipt['parent_maps_sha256']!=d['mesh_quality']['parent_maps_sha256'] or
                (probe['pose'],probe['parameters'])!=(d['best']['pose'],d['best']['parameters']) or
                receipt['quantitative_result_accepted'] is not False):
            raise ValueError('Selected contact-only probe does not describe the actual selected geometry')
    d['residual_diagnostics']={}
    for key,filename,candidate_file,candidate in [
            ('previous','previous_candidate_residual_diagnostics',work/'selected_candidate.local.json',None),
            ('profile','profile_residual_diagnostics',work/'inputs/fits_own_drive/selected_previous_model.local.json',d['profile']),
            ('selected','selected_new_residual_diagnostics',work/'selected_new_candidate.local.json',d['best'])]:
        receipt=snap.json(work/(filename+'.local.json'),optional=True)
        if receipt is not None:
            if key=='previous':
                old=snap.json(candidate_file);candidate=old.get('best',old)
                if (candidate['parameters'],candidate['pose'])!=(reference['parameters'],reference['pose']):
                    raise ValueError('Previous residual diagnostics do not describe the inherited initial point')
            validate_residual_receipt(d,receipt,candidate_file,candidate)
        d['residual_diagnostics'][key]=receipt
    d['mesh_sections']=snap.json(work/'mesh_sections.local.json',optional=True)
    if d['mesh_sections'] is not None:validate_mesh_sections(d,d['mesh_sections'])
    d['display_surface']=snap.json(work/'calculation_surface_display.local.json',optional=True)
    if d['display_surface'] is not None:
        surface=d['display_surface'];snap.guard(surface['sources'])
        if (surface['schema']!='georg_calculation_boundary_display_v1' or surface['display_only'] is not True or
                surface['calculation_geometry_changed'] is not False or surface['FEM_was_run'] is not False or
                surface['calculation_mesh_sha256']!=d['quality']['level3_direct']['mesh_sha256']):
            raise ValueError('Display boundary changed its calculation identity or scope')
        vertices=np.asarray(surface['vertices_lps_mm']);faces=np.asarray(surface['faces_zero_based'])
        if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or
                faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(vertices)):
            raise ValueError('Invalid display surface indexing')
    d['author_centre']=snap.json(work/'inputs/author_centre_20261012.local.json')
    d['inverse_consistency']=snap.json(work/'initial_inverse_forward_consistency.local.json',optional=True)
    if d['inverse_consistency'] is not None:
        agreement=d['inverse_consistency']
        source=work/'inputs/fits_own_drive/pose_0001_evaluation_0001.local.json';row=snap.json(source)
        if (agreement['first_profile_evaluation_sha256']!=snap.files[str(source.resolve())] or
                agreement['audited_forward_sha256']!=snap.files[str((work/'integration_high_seed_1_forward.local.json').resolve())] or
                agreement['quantitative_result_accepted'] is not False or
                agreement['complete_electrode_centres_equal'] is not True):
            raise ValueError('Inverse/forward agreement changed its bound point or source')
        actual=float(np.max(abs(vector(row)-vector(d['forwards'][4]))))
        parameter_delta=float(np.max(abs(np.asarray(row['parameters'])-np.asarray(d['forwards'][4]['parameters'][:6]))))
        same_centres=row['pose']['electrodes']==d['forwards'][4]['pose']['electrodes']
        if (actual!=agreement['maximum_Z_difference_ohm'] or parameter_delta!=agreement['material_parameter_maximum_difference'] or
                same_centres!=agreement['complete_electrode_centres_equal'] or
                agreement['passed']!=(same_centres and parameter_delta<1e-8 and actual<agreement['tolerance_ohm'])):
            raise ValueError('Inverse/forward agreement does not match the saved vectors')
    d['dispatch_failure']=snap.json(work/'independent_rule_dispatch_failure.local.json',optional=True)
    d['unit_metadata_failure']=snap.json(work/'independent_rule_unit_metadata_failure.local.json',optional=True)
    if d['unit_metadata_failure'] is not None:
        note=d['unit_metadata_failure']
        if (note['actual_length_unit']!='mm' or note['actual_node_scale_to_si']!=.001 or
                note['actual_scale_nodes_to_si']!=1 or note['actual_fraction_hashes_match'] is not True or
                note['calculation_arrays_or_units_changed'] is not False or note['numerical_tolerances_changed'] is not False):
            raise ValueError('Recorded unit metadata failure does not preserve actual SI units')
    if d['dispatch_failure'] is not None:
        failure=d['dispatch_failure']
        if any(failure[k] is not False for k in ('fraction_plans_or_arrays_changed',
                'failed_stage_produced_a_numerical_result','existing_four_controls_changed')):
            raise ValueError('Historical dispatch failure changed its numerical scope')
        snap.guard([dict(path=str(PIPE/'src/trkg4_georg_check_materials.m'),sha256=failure['legacy_dispatch_source_sha256'])])
    d['relocated_state']=snap.json(work/'relocated_run_state.local.json',optional=True)
    d['relocated_quality']=snap.json(work/'relocated_contacts_r3_v2/fine2/mesh_quality.local.json',optional=True)
    d['relocated_preparation']=snap.json(work/'relocated_contacts_r3_v2/preparation.local.json',optional=True)
    if d['relocated_preparation']:
        prep=d['relocated_preparation'];snap.guard(prep['sources'])
        if (prep['candidate_source_sha256']!=snap.files.get(str((work/'selected_new_candidate.local.json').resolve())) or
                prep['candidate_provenance']['fitted_in_current_parent_model'] is not False or
                prep['candidate_provenance']['predictions_or_Jacobian_transferred'] is not False):
            raise ValueError('Relocated refinement transferred predictions or lost its selected reference')
    d['relocated_attempts']=[snap.json(p) for p in sorted((work/'relocated_contacts_r3_v2').glob('quality_attempt_*.local.json'))]
    d['relocated_forwards']=[]
    for model,filename in zip(RELOCATED_MODELS,RELOCATED_FILES,strict=True):
        if model!='parent_direct':
            d['plans'][model]=snap.json(work/model/'plan.local.json',optional=True)
            d['quality'][model]=snap.json(work/model/'mesh_quality.local.json',optional=True)
            if d['plans'][model]:snap.guard(d['plans'][model]['sources'])
        row=snap.json(work/(filename+'.local.json'),optional=True)
        if row is not None:
            if new_reference is None or d['plans'][model] is None:raise ValueError('Relocated forward lacks model/reference')
            validate_forward(d,row,model,(new_reference_file,new_reference))
        d['relocated_forwards'].append(row)
    d['relocated_differences']=comparisons(dict(d,final_forwards=d['relocated_forwards']),True)
    d['relocated_audit']=snap.json(work/'relocated_selected_audit.local.json',optional=True)
    d['relocated_coverage']=snap.json(work/'relocated_fine_contact_coverage.local.json',optional=True)
    if d['relocated_coverage']:
        validate_fine_coverage(d,d['relocated_coverage'],work/(RELOCATED_FILES[4]+'.local.json'),
            d['relocated_forwards'][4],quality=d['relocated_quality'])
    if d['relocated_audit']:
        audit=d['relocated_audit'];snap.guard(audit['sources'])
        if (audit['schema']!='georg_relocated_selected_numerical_audit_v1' or
                audit['selected_candidate_sha256']!=snap.files[str((work/'selected_new_candidate.local.json').resolve())] or
                audit['reference_sha256']!=snap.files[str(new_reference_file.resolve())] or
                any(audit[k] is not False for k in ('global_mesh_convergence_proven','quantitative_result_accepted'))):
            raise ValueError('Relocated audit lost its fixed-point scope')
        for key,index in [('spatial',0),('fractions',1),('total',2)]:
            check_difference(audit['spatial_tissue_audit'][key],d['relocated_differences'][index])
        check_difference(audit['contact_comparison'],d['relocated_differences'][3])
        contact=audit['contact_comparison']
        if (contact.get('matrix_entries_compared') is not True or
                contact.get('actual_bulk_entries_bitwise_identical') is not True or
                len(contact.get('matrix_fingerprints',[]))!=10):
            raise ValueError('Independent contact comparison lacks equality of all actual coefficient matrices')
        for key,index in [('spatial_respiratory',0),('tissue_respiratory',1),('contact_respiratory',3),
                ('rule_growth',4),('independent_seed',5),('nested_rules_seed_1',6),('nested_rules_seed_2',7)]:
            quadrature=d['relocated_forwards'][index-2]['quadrature_passed'] if index>=6 else True
            check_screen(audit['comparisons'][key],d['relocated_differences'][index],quadrature)
        check_integration_identity(d,audit['integration_identity'],d['relocated_forwards'][4:],RELOCATED_MODELS[4:])
        if d['relocated_coverage'] is None:raise ValueError('Relocated audit lacks actual coverage')
        expected=(audit['spatial_tissue_audit']['all_checks_passed'] and audit['contact_comparison']['passed'] and
            audit['integration_identity']['passed'] and all(r['passed'] for r in audit['comparisons'].values()) and
            d['relocated_coverage']['all_contact_faces_refined'] and all(r['system_checks_passed'] and
            r['quadrature_passed'] and r['patch_qc']['passed'] for r in d['relocated_forwards'][4:]) and
            len(d['relocated_forwards'][4]['derivatives'])==2 and all(r['passed'] for r in d['relocated_forwards'][4]['derivatives']))
        if audit['all_checks_passed']!=bool(expected):raise ValueError('Relocated numerical admission changed')
    plan=d['plans'][MODELS[0]]
    masks={Path(r['path']).name:r for r in plan['sources'] if Path(r['path']).name in
        ('static_ct.nii.gz','lung.nii.gz','bone.nii.gz','heart.nii.gz','fat.nii.gz')}
    directory=source_path(masks['lung.nii.gz']['path']).parent
    d['segmentation_passport']=snap.json(directory/'segmentation.json',optional=True)
    if d['segmentation_passport']:
        passport=d['segmentation_passport']
        if passport['complete'] is not True or passport['ct_sha256']!=masks['static_ct.nii.gz']['sha256']:
            raise ValueError('Segmentation passport is not bound to this completed CT processing')
        for name in ('lung','bone','heart','fat'):
            if passport['products'][name]['sha256']!=masks[name+'.nii.gz']['sha256']:
                raise ValueError('Reported automatic mask differs from the actual FEM source')
    test_receipt=work/'synthetic_test_receipt.final.local.json'
    if not test_receipt.is_file():test_receipt=work/'synthetic_test_receipt.local.json'
    d['synthetic_tests']=snap.json(test_receipt,optional=True)
    if d['synthetic_tests']:
        snap.guard(d['synthetic_tests']['sources'])
        if d['synthetic_tests']['FEM_or_real_array_validation'] is not False:
            raise ValueError('Synthetic tests were relabelled as real-data validation')
    d['as_of']=dt.datetime.now(dt.timezone.utc).isoformat()
    return d


def completed_relocated_control(d):
    for index,label in [(4,'16384 пробы, основное перемешивание'),
            (5,'16384 пробы, независимое перемешивание'),(3,'4096 проб')]:
        row=d['relocated_forwards'][index]
        if row is not None:return row,label
    return None,None


def available_series(d):
    series=[(label,row) for label,row in zip(LABELS,d['forwards'],strict=True) if row is not None]
    if d['profile'] is not None:series.append(('Новый условный подбор сопротивлений',d['profile']))
    if d['best'] is not None:series.append(('Новый совместно выбранный кандидат',d['best']))
    control,label=completed_relocated_control(d)
    if control is not None:series.append(('Контроль целевой сетки: '+label,control))
    return series


def status_table(d):
    cfg=d['config'];moscow=dt.timezone(dt.timedelta(hours=3));gate=d['gate']
    current=d['relocated_state'] or d['state']
    status=re.sub(r'_v\d+$','',current.get('status',''))
    label=STATE_LABELS.get(status,'Выполняются численные проверки; итог не опубликован')
    if status.startswith('fixed_pose_material_profile_'):label='Подбор сопротивлений при фиксированном положении'
    if status.startswith('joint_materials_and_placement_'):label='Совместный подбор сопротивлений и положения'
    if '_new_candidate_' in status:label='Повторная численная проверка нового выбранного кандидата'
    if d['relocated_audit'] is not None:
        label=('Численные проверки выбранной точки на целевой сетке завершены; допуски выдержаны'
            if d['relocated_audit']['all_checks_passed'] else
            'Численные проверки выбранной точки завершены; полный допуск не выдержан')
    rows=[['Последнее сохранённое состояние',label],
        ['Общий срок, по Москве',dt.datetime.fromtimestamp(cfg['deadline_unix'],moscow).strftime('%d.%m.%Y %H:%M:%S')],
        ['Действующий срок поиска, по Москве',dt.datetime.fromtimestamp(cfg['search_deadline_unix'],moscow).strftime('%d.%m.%Y %H:%M:%S')],
        ['Резерв до общего срока',number((cfg['deadline_unix']-cfg['search_deadline_unix'])/3600,1)+' ч'],
        ['Заявленный предел совместных оценок',cfg['max_evaluations']],
        ['Начальные приближения совместного поиска','Прежнее уточнённое и стандартное Ника' if cfg['start_indices']==[1,2] else 'Прежнее уточнённое'],
        ['Начальный численный допуск','Ожидается полный результат' if gate is None else ('Выдержан в проверенной точке' if gate['passed'] else 'Не выдержан; переход к подбору не разрешён')],
        ['Новый условный подбор сопротивлений','Результат сохранён и связан с новым расчётом' if d['profile'] is not None else 'Готовый результат отсутствует'],
        ['Новый совместно выбранный кандидат','Сохранён; полнота поиска оценивается отдельно' if d['best'] is not None else 'Готовый результат отсутствует'],
        ['Количественное или физиологическое принятие','Не заявляется']]
    for label,row in zip(LABELS,d['forwards'],strict=True):rows.append([label,'Новый прямой результат сохранён' if row is not None else 'Ожидается завершённый результат'])
    return table('Состояние численных проверок и ограниченного поиска',['Стадия или условие','Сохранённые сведения'],rows)


def montage_label(row):
    if pd.isna(row.arm_montage):
        return 'Провода сняты' if int(row.record_id.rsplit('-',1)[-1]) in (6,11) else 'Не указан'
    return str(row.arm_montage).replace('x','×')


def observations_table(d):
    rows=[]
    for _,r in d['observations'].iterrows():
        rows.append([r.record_id.rsplit('-',1)[-1],int(r.channel),int(r.size_mm),
            'Продольная' if r.axis=='longitudinal' else 'Поперечная',
            'Вдох' if r.state=='inhale' else 'Выдох',montage_label(r),number(r.Z_ohm,6),
            number(r.start_s,3)+'–'+number(r.end_s,3),int(r.fs_hz),'Кандидатная разметка'])
    return table('Измеренные уровни и происхождение дыхательных интервалов',
        ['Запись','Канал','Размер, мм','Ось','Фаза','Монтаж рук','Уровень, Ом','Интервал, с','Частота записи, Гц','Статус'],rows)


def quiet_table(d):
    rows=[]
    for _,r in d['quiet'].iterrows():
        rows.append([r.record_id.rsplit('-',1)[-1],'BEI'+str(int(r.channel)),int(r.size_mm),
            'Продольная' if r.axis=='longitudinal' else 'Поперечная',montage_label(r),
            number(r.Z_ohm,6),number(r.start_s,3)+'–'+number(r.end_s,3),
            'Спокойное дыхание; фаза задержки не назначена','Вне 24 уровней подбора'])
    return table('Отдельное описательное наблюдение спокойного дыхания',
        ['Запись','Канал','Размер, мм','Ось','Монтаж рук','Уровень, Ом','Интервал, с','Состояние','Использование'],rows)


def reproducibility_table(d):
    row=d['forwards'][0];receipt=d['repeat']
    if row is None:return table('Воспроизводимость прямого расчёта родительской сетки',
        ['Проверка','Результат'],[['Прямой расчёт','Ожидается завершённый результат']])
    rmse=np.sqrt(np.mean((vector(row)-d['target'])**2))
    rows=[['Новый прямой расчёт: RMSE',number(rmse,12)+' Ом'],
        ['Вложенные правила интегрирования','Выдержаны' if row['quadrature_passed'] else 'Не выдержаны'],
        ['Системные проверки','Пройдены' if row['system_checks_passed'] else 'Не пройдены'],
        ['Проверки производных в записях 006 и 011',
            ('Пройдены' if len(row['derivatives'])==2 and all(r['passed'] for r in row['derivatives']) else 'Не пройдены')]]
    if receipt is None:rows.append(['Сравнение с прежним прямым расчётом','Отдельное свидетельство ещё не опубликовано'])
    else:
        rows += [['Максимум разности с прежним прямым расчётом',number(receipt['maximum_Z_difference_ohm'],12)+' Ом'],
            ['Параметры, полное положение и измеренные цели','Совпадают точно'],
            ['Область вывода','Воспроизводимость сохранённого прямого расчёта; физиологическая валидация не выполнена']]
    return table('Воспроизводимость прямого расчёта родительской сетки',['Проверка','Результат'],rows)


def properties_table(d):
    ref=d['reference_materials'];rho=ref['rho_ohm_m'];p=d['plans'][MODELS[0]];inputs=d['inputs']
    diameter=inputs['assumed_contact_diameter_mm']
    rows=[['Частота электрической модели',str(ref['frequency_hz'])+' Гц','Сохранённый файл свойств'],
        ['Кость: фиксированное сопротивление',number(rho[2],10)+' Ом·м','Вся маска кости представлена кортикальной костью'],
        ['Сердце: фиксированное сопротивление',number(rho[3],11)+' Ом·м','Вся маска сердца представлена миокардом'],
        ['Опорный фон',number(rho[0],12)+' Ом·м','Принят как для мышцы; фон охватывает весь объём вне четырёх масок'],
        ['Свободные фазовые параметры','1 — фон; 2 — лёгкое; 3 — жир','В полном порядке материалов это позиции 1, 2 и 5'],
        ['Техническая область поиска','0,1–100 Ом·м','Физиологическая норма и доверительный интервал не установлены'],
        ['Диаметр диска',number(diameter,1)+' мм','Принят; не измерен'],
        ['Площадки рук в модели',inputs['constant_arm_physical_contact_count'],'Их присутствие после снятия проводов — допущение'],
        ['Номинальная площадь',number(np.pi*(diameter/2)**2,9)+' мм²','Площадь круга по принятому диаметру'],
        ['Удельный контактный импеданс',format(p['contact_impedance_ohm_m2'],'.15g').replace('.',',')+' Ом·м²','Фиксирован; принят без индивидуального измерения']]
    if d['contact_size_note'] is not None:
        note=d['contact_size_note']
        rows += [['Площадка по приблизительному воспоминанию автора',
            number(note['metal_contact_diameter_mm_approximate'],1)+' мм','Не является независимым измерением'],
            ['Область геля по приблизительному воспоминанию автора',
            number(note['gel_footprint_diameter_mm_approximate'],1)+' мм','Эффективная электрическая площадь контакта не измерена']]
    for label,row in zip(LABELS,d['forwards'],strict=True):
        if row is None:continue
        contacts=row['patch_qc']['contacts'];areas=[r['area_mm2'] for r in contacts]
        rows.append(['Фактическая площадь: '+label,number(min(areas),9)+'–'+number(max(areas),9)+' мм²',
            str(len(contacts))+' описаний граней в '+str(len({r['row'] for r in contacts}))+' условиях записи'])
    return table('Материалы и конечные контактные площадки',['Вход','Значение','Смысл и допущение'],rows)


def mesh_table(d):
    rows=[]
    for label,model in [('Родительская сетка',MODELS[0]),('Третий уровень',MODELS[2])]:
        q=d['quality'][model]
        if q is None:rows.append([label,*['Ожидается']*7]);continue
        rows.append([label,q['nodes'],q['tetrahedra'],number(q['mean_ratio_minimum'],9),number(q['edge_condition_maximum'],3),
            number(q.get('relative_volume_difference',0),12),'Да' if q['all_positive'] else 'Нет','Да' if q['passed'] else 'Нет'])
    return table('Форма тетраэдров и сохранение геометрии',
        ['Сетка','Узлы','Тетраэдры','Минимальный показатель формы','Обусловленность рёбер','Изменение объёма','Положительные объёмы','Контроль формы пройден'],rows)


def integration_table(d):
    rows=[]
    for label,model in zip(LABELS,MODELS,strict=True):
        plan=d['plans'][model];progress=d['progress'][model];contract=progress['contract'];cp=progress['checkpoint']
        if plan is not None:
            points=plan['numerical_quadrature_points'];seed=plan.get('integration_seed')
            if seed is None and model in MODELS[:4]:seed=20261008
            rows.append([label,' / '.join(map(str,points)),seed,'План опубликован',
                'Точное наследование' if plan['fraction_origin']=='exact parent inheritance' else 'Интегрирование масок'])
        elif contract is not None:
            identity=contract['identity'];done=cp.get('mixed_cells_completed') if cp else None
            total=cp.get('mapping',{}).get('mixed_cells') if cp else None
            rows.append([label,' / '.join(map(str,identity['points'])),identity['seed'],
                'Интегрирование не завершено: '+str(done if done is not None else 'неизвестно')+' из '+str(total if total is not None else 'неизвестно'),
                'Новые доли ещё не доступны для завершённого прямого контроля'])
        else:rows.append([label,'Ожидается','Ожидается','План ещё не опубликован','Готовый результат отсутствует'])
    return table('Правила интегрирования и независимое случайное перемешивание',
        ['Модель','Проб на элемент','Начальное число генератора','Состояние подготовки','Происхождение долей'],rows)


def numerical_sets(d):
    r=d['relocated_audit']
    return [
        ('Начальная фиксированная точка',d['initial_differences'],d['initial_audit'],d['initial_contact'],d['gate']),
        ('Новый угол: первоначальное сгущение',d['final_differences'],d['final_audit'],d['final_contact'],d['final_integration']),
        ('Новый угол: целевое сгущение',d['relocated_differences'],r['spatial_tissue_audit'] if r else None,
            r['contact_comparison'] if r else None,r)]


def numerical_table(d):
    rows=[]
    for point,values,audit,contact,stability in numerical_sets(d):
        for index,r in enumerate(values):
            if r['pending']:
                state=('Заменено целевым сгущением' if point=='Новый угол: первоначальное сгущение' and d['coverage_probe'] else 'Ожидается')
                rows.append([point,r['name'],state,'—','Не установлено','—']);continue
            published=(audit is not None if index<3 else contact is not None if index==3 else stability is not None)
            rows.append([point,r['name'],number(max(r['difference_ohm']),9),r['failed_values'],
                ('Выдержан' if r['passed'] else 'Не выдержан')+('' if published else ' по сохранённым векторам; полный аудит ожидается'),
                number(max(abs(r['contrast_delta'])),9)])
    return table('Раздельные численные изменения при неизменных параметрах и положении',
        ['Точка','Влияние','Максимум |ΔZ|, Ом','Превышения из 24','Допуск уровней','Максимум изменения контраста, Ом'],rows)


def contrast_screen_table(d):
    rows=[]
    for point,values,_,_,published in numerical_sets(d):
        for i in (0,1,3,4,5,6,7):
            r=values[i]
            if r['pending']:
                state=('Заменено целевым сгущением' if point=='Новый угол: первоначальное сгущение' and d['coverage_probe'] else 'Ожидается')
                rows.append([point,r['name'],state,'—','Не установлено']);continue
            state=('Выдержан' if r['contrast_passed'] else 'Не выдержан')+('' if published is not None else ' по сохранённым векторам; полный допуск ожидается')
            rows.append([point,r['name'],number(max(abs(r['contrast_delta'])),9),r['contrast_failed'],state])
    return table('Численная устойчивость всех 12 дыхательных контрастов',
        ['Точка','Сравнение','Максимум изменения контраста, Ом','Превышения из 12','Контрастный допуск'],rows)


def prerequisites_table(d):
    rows=[]
    relocated=d['relocated_audit']
    for label,audit in [('Начальная точка',d['initial_audit']),
            ('Новая точка: целевое сгущение',relocated['spatial_tissue_audit'] if relocated else None)]:
        for key,name in [('shape_QC_passed','Форма сетки'),('energy_inheritance_passed','Наследование энергии поля'),
                ('quadrature_passed','Вложенные правила интегрирования'),('derivatives_passed','Производные'),
                ('system_checks_passed','Системы и контактные площадки'),('all_checks_passed','Все допуски раздельного сеточного аудита')]:
            rows.append([label,name,'Ожидается' if audit is None else ('Выдержан' if audit[key] else 'Не выдержан')])
    gate=d['gate']
    for key,name in [('contact_discretization_passed','Дискретизация контактов'),('integration_stability_passed','Увеличение числа проб и смена перемешивания'),('passed','Допуск к поиску в проверенной точке')]:
        rows.append(['Начальная точка',name,'Ожидается' if gate is None else ('Выдержан' if gate[key] else 'Не выдержан')])
    return table('Предварительные проверки и численный допуск',['Точка','Проверка','Состояние'],rows)


def fit_table(d):
    rows=[]
    for label,row in [('Условный подбор при фиксированном положении',d['profile']),('Совместно выбранный кандидат',d['best']),
            ('Прямой контроль на целевом сгущении; без нового подбора',d['relocated_forwards'][4])]:
        if row is None:rows.append([label,*['Готовый результат отсутствует']*5]);continue
        z=vector(row);error=z-d['target'];a=[];b=[]
        for i,j,_ in d['pairs']:a.append((error[i]+error[j])/2);b.append((error[j]-error[i])/2)
        sse=float(error@error);contrast_sse=2*np.sum(np.square(b))
        rows.append([label,number(np.sqrt(np.mean(error**2)),9),number(np.sqrt(np.mean(np.square(a))),9),
            number(np.sqrt(np.mean(np.square(b))),9),number(100*contrast_sse/sse,6),
            int(np.count_nonzero(np.sign(contrasts(d,z))!=np.sign(contrasts(d,d['target']))))])
    return table('Новые результаты подбора и разложение ошибки',
        ['Результат','RMSE, Ом','RMS ошибки среднего, Ом','RMS ошибки половины контраста, Ом','Контрастная часть SSE, %','Несовпадений знака из 12'],rows)


def rho_pose_table(d):
    rows=[]
    for label,r in [('Условный подбор',d['profile']),('Совместно выбранный кандидат',d['best'])]:
        if r is None:continue
        for k,material in enumerate(('Фон','Лёгкое','Жир')):
            near=any(np.isclose(v,.1,rtol=0,atol=1e-6) or np.isclose(v,100,rtol=0,atol=1e-6)
                for v in (r['rho_inhale_ohm_m'][k],r['rho_exhale_ohm_m'][k]))
            rows.append([label,material,number(r['rho_inhale_ohm_m'][k],9),number(r['rho_exhale_ohm_m'][k],9),'Да' if near else 'Нет'])
    if not rows:rows=[['Новый подбор','Готовый результат отсутствует','—','—','Не установлено']]
    return table('Новые фазовые сопротивления; близость к пределу — только читательская отметка',
        ['Результат','Материал','Вдох, Ом·м','Выдох, Ом·м','Близость к границе ≤10⁻⁶ Ом·м'],rows)


def preparation_changes_table(d):
    return table('Изменения вычислительной подготовки в этом блоке',
        ['Изменение','Назначение и проверка'],[
        ['Сгущение всех контактных областей','28 боковых и 12 ручных направляющих; проверены геометрия, форма элементов и покрытие всех фактических контактных граней сгущением'],
        ['Независимое интегрирование','Все неоднородные элементы пересчитаны для 4096 и 16384 проб и двух начальных значений генератора'],
        ['Выбор обработчика усиленного правила','Первый запуск завершился до численного результата: старый обработчик требовал родительскую модель. Обработчик направлен к каноническому сборщику с полной проверкой новой декларации' if d['dispatch_failure'] else 'Отдельный исправленный запуск не отмечен'],
        ['Чтение метаданных единиц','Логическое значение MATLAB прочитано из HDF5 как целое 1. Исправлена проверка его типа; подтверждены миллиметры, масштаб 0,001, исходники и реальные доли. Единицы и матрицы не изменялись' if d['unit_metadata_failure'] else 'Отдельное исправление чтения не отмечено'],
        ['Отделение влияния контактных граней','Для целевой сетки подтверждено побитовое совпадение численных значений и структуры всех десяти коэффициентных матриц. Проверка прямого наследования кэша заменена проверкой фактических матриц двух самостоятельных сборок' if d['relocated_audit'] and d['relocated_audit']['contact_comparison'].get('actual_bulk_entries_bitwise_identical') else 'Полное сравнение коэффициентных матриц ещё не опубликовано'],
        ['Возобновление завершённых контролей','Проверяются действительные планы, матрицы, полная поза, параметры и наблюдения; аудит должен воспроизводиться точно, без перезаписи'],
        ['Согласованность прямой и обратной постановок','Первая полная оценка оптимизатора воспроизвела все 24 проверенных импеданса точно при прежних параметрах и всех центрах контактов' if d['inverse_consistency'] and d['inverse_consistency']['passed'] else 'Собственная проверка ещё не опубликована'],
        ['Старты и время поиска','Один старт условного подбора; два исходных приближения совместного поиска в общем бюджете; четыре часа оставлены для итоговых контролей'],
        ['Описание контакта','Размер площадки около 5 мм и области геля около 10 мм — приближённое воспоминание автора; эффективная электрическая площадь не измерена']])


def placement_coordinates_table(d):
    rows=[];best=d['best']
    if best is not None:
        p=best['pose']
        rows.append(['Общий центр','—','—',*[number(v,3) for v in p['centre_lps_mm']]])
        for block in p['electrodes']:
            axis='Продольная' if block['axis']=='longitudinal' else 'Поперечная'
            for role,xyz in zip(('I+','U+','U−','I−'),block['centres_lps_mm'],strict=True):
                rows.append([block['size_mm'],axis,role,*[number(v,3) for v in xyz]])
    else:rows=[['Новый кандидат ещё не выбран','—','—','—','—','—']]
    return table('Расчётные поверхностные центры всех боковых контактов нового кандидата',
        ['Размер, мм','Ось','Контакт','x LPS, мм','y LPS, мм','z LPS, мм'],rows)


def levels_table(d):
    series=available_series(d);labels=observation_labels(d)
    return table('Все 24 уровня: измерение и только завершённые новые расчёты',
        ['Наблюдение','Измерение, Ом',*[name+', Ом' for name,_ in series]],
        [[label,number(d['target'][i],6),*[number(vector(row)[i],6) for _,row in series]] for i,label in enumerate(labels)])


def contrasts_table(d):
    series=available_series(d)
    return table('Все 12 дыхательных контрастов: выдох минус вдох',
        ['Запись, размер и ось','Измерение, Ом',*[name+', Ом' for name,_ in series]],
        [[label,number(d['target'][j]-d['target'][i],6),*[number(vector(r)[j]-vector(r)[i],6) for _,r in series]] for i,j,label in d['pairs']])


def search_table(d):
    rows=d['checkpoints'];valid=[r for r in rows if r['status']=='numerical_candidate_not_validated']
    answer=[['Запланированные старты совместного поиска',', '.join(map(str,d['config']['start_indices']))],
        ['Траекторий поиска из лучшего старта','Одна; два старта не означают два независимо оптимизированных минимума'],
        ['Общий предел поисковых оценок',d['config']['max_evaluations']],
        ['Готовые численные оценки',len(valid)],['Отказы геометрии',sum(r['status']=='invalid_geometry' for r in rows)],
        ['Начатые оценки без численного результата',sum(r['status']=='forward_calculation_running' for r in rows)],
        ['Глобальность минимума','Не установлена'],['Достижение разрешения',
            'Не подтверждено' if d['selected'] is None else ('Да' if d['selected'].get('search',{}).get('resolution_reached',False) else 'Не подтверждено')]]
    if d['profile'] is not None:
        r=d['profile'];answer.extend([['Оценок условного подбора',r['evaluations']],
            ['Коды завершения условного подбора',str(r['start_exitflags'])],
            ['Положительный код условного подбора','Да' if r['all_starts_converged'] else 'Нет']])
    if d['selected'] is not None:
        search=d['selected'].get('search',{})
        for key,label in [('evaluations','Поисковых оценок по итоговой записи'),('completed_sweeps','Завершённых обходов'),
                ('sweeps','Начатых обходов'),('budget_reached','Достигнут предел оценок')]:
            answer.append([label,str(search[key]) if key in search else 'Нет итоговой записи алгоритма'])
        answer.append(['Остановка по времени','Да' if d['selected'].get('search_stopped_for_time',False) else 'Не отмечена'])
        if 'final_step' in search:
            answer.append(['Последний поисковый шаг: шесть lnρ; два смещения, мм; угол, °',
                '; '.join(number(v,6) for v in search['final_step'])])
    if d['best'] is not None:
        pose=d['best']['pose'];answer.extend([['Смещения центра, мм','; '.join(number(v,6) for v in pose['offsets_uv_mm'])],
            ['Угол, °',number(pose['angle_deg'],6)],['Общий центр LPS, мм','; '.join(number(v,6) for v in pose['centre_lps_mm'])]])
    for r in rows:
        value=number(np.sqrt(r['mean_square_ohm2']),9)+' Ом' if r['status']=='numerical_candidate_not_validated' else 'Отказ геометрии' if r['status']=='invalid_geometry' else ('Незавершено; процесс остановлен' if d['selected'] and d['selected'].get('search_interrupted') else 'Ожидается численный результат')
        answer.append(['Оценка '+str(r['evaluation']),value])
    return table('Фактический объём совместного поиска',['Характеристика или оценка','Сохранённый результат'],answer)


def historical_diagnostics_table(d):
    review=d['hypotheses']
    if review is None:return table('Диагностика прежней точки',['Состояние'],[['Сохранённый разбор отсутствует']])
    a=review['numerical_analysis'];pairs=a['pair_diagnostics']
    concentrated=sum(r['pair_SSE_fraction'] for r in pairs if (r['record_id'].endswith('-009') and r['channel']==1) or
        (r['record_id'].endswith('-011') and r['channel']==2 and r['axis']=='perpendicular'))
    decomp=a['objective_decomposition'];g=a['local_sensitivity']['MSE_gradient_per_logrho']
    return table('Математическая диагностика только прежней сохранённой точки',['Характеристика','Значение','Область вывода'],[
        ['RMSE',number(a['RMSE_ohm'],9)+' Ом','Прежняя точка, не новый минимум'],
        ['Доля SSE двух выделенных пар',number(100*concentrated,6)+' %','Запись 009, канал 1; запись 011, канал 2, поперечная ось'],
        ['Контрастная часть SSE',number(100*decomp['contrast_fraction_of_SSE'],6)+' %','Разложение ошибок прежней точки'],
        ['Градиент MSE по логарифму фона: вдох / выдох',number(g[0],6)+' / '+number(g[3],6),'Локальный аналитический якобиан прежней точки'],
        ['Несовпадения знака',str(a['contrast_wrong_sign_count'])+' из 12','Глобальная невозможность контраста не установлена'],
        ['Независимая оценка ковариации измерений','Отсутствует','Разброс блоков не заменяет стандартной ошибки измерения']])


DIAGNOSTIC_LABELS=(('previous','Прежняя точка: унаследованные контакты'),
    ('profile','Новый условный подбор при фиксированном положении'),
    ('selected','Новый совместно выбранный кандидат'))


def actual_coverage_table(d):
    r=d['coverage']
    if r is None:return table('Покрытие фактических контактных граней',['Проверка','Состояние'],
        [['Фактические площадки начальной точки','Свидетельство ещё не опубликовано']])
    rows=[['Все выбранные контактные грани сгущены','Да' if r['all_actual_contact_faces_refined'] else 'Нет'],
        ['Минимальная доля сгущённой площади',number(r['minimum_refined_boundary_area_fraction'],9)],
        ['Описания площадок по условиям записи',len(r['contacts'])],
        ['Условия записи',len({c['record_id'] for c in r['contacts']})],
        ['Область вывода','Фактические площадки прямого родительского расчёта в начальной точке'],
        ['Покрытие нового выбранного положения','Начальное свидетельство на него не переносится']]
    selection=d['mesh_quality'].get('local_selection',{})
    if 'lateral_contact_count' in selection and 'physical_arm_contact_count' in selection:
        rows.append(['Направляющие уточнения',str(selection['lateral_contact_count'])+' боковых и '+
            str(selection['physical_arm_contact_count'])+' для рук; альтернативные размеры не установлены одновременно'])
    return table('Покрытие фактических контактных граней',['Проверка','Сохранённое свидетельство'],rows)


def fine_contact_coverage_table(d):
    rows=[]
    for r,label in [(d['fine_coverage']['initial'],'Начальная точка'),
            (d['fine_coverage']['selected'],'Новый угол: первоначальное сгущение'),
            (d['relocated_coverage'],'Новый угол: целевое сгущение')]:
        if r is None:
            state=('Контроль заменён целевым сгущением' if label=='Новый угол: первоначальное сгущение' and d['coverage_probe']
                else 'Свидетельство ещё отсутствует')
            rows.append([label,state,*['—']*5]);continue
        for record in r['descriptions']:
            rows.append([label,record['record_id'],record['physical_contact_count'],record['fine_face_count'],
                record['unrefined_fine_face_count'],number(100*record['refined_boundary_area_fraction'],6),
                'Да' if record['all_contact_faces_refined'] else 'Нет'])
    return table('Покрытие фактических контактных граней сгущённой сетки',
        ['Точка','Условие записи','Описаний контактов','Включений граней','Несгущённых включений',
         'Сгущённая доля площади, %','Все грани сгущены'],rows)


def relocated_mesh_table(d):
    q=d['relocated_quality'];prep=d['relocated_preparation'];rows=[]
    if q is None:return table('Сетка под новые площадки',['Показатель','Результат'],[['Подготовка','Полный результат отсутствует']])
    rows=[['Узлов',q['nodes']],['Тетраэдров',q['tetrahedra']],['Положительность объёмов','Да' if q['all_positive'] else 'Нет'],
        ['Минимальный показатель формы',number(q['mean_ratio_minimum'],12)],
        ['Перцентили показателя формы: 1 / 5 / 50 / 95',
            ' / '.join(number(v,6) for v in q['mean_ratio_quantiles'])],
        ['Максимальная обусловленность рёбер',number(q['edge_condition_maximum'],6)],
        ['Перцентили обусловленности рёбер: 50 / 95 / 99',
            ' / '.join(number(v,6) for v in q['edge_condition_quantiles'])],
        ['Элементов с нарушением порога формы / обусловленности',
            str(q['failing_mean_ratio'])+' / '+str(q['failing_edge_condition'])],
        ['Заранее заданные пороги: форма не ниже / обусловленность не выше',
            number(q['quality_limits']['minimum_mean_ratio'],3)+' / '+number(q['quality_limits']['maximum_edge_condition'],0)],
        ['Относительное изменение объёма',number(q['relative_volume_difference'],12)],
        ['Согласованность тетраэдральной границы проверена','Да' if q['conforming_boundary_verified'] else 'Нет'],
        ['Контроль формы','Пройден' if q['passed'] else 'Не пройден'],
        ['Технический предел узлов',q['maximum_fine_nodes']],['Радиус выбора родителей, мм',number(q['local_selection']['radius_mm'],1)],
        ['Боковых / ручных направляющих','28 / 12'],['Попыток с ограниченным добавлением родителей',q['promotion_attempts']],
        ['Параметры подобраны на этой сетке','Нет; перенесены как фиксированный опорный сценарий']]
    if prep:rows.append(['Происхождение','Те же КТ, маски, наружная геометрия, конечные контакты и фиксированные ткани; другая численная сетка'])
    for attempt in d['relocated_attempts']:
        quality=attempt['quality'];rows.append(['Попытка '+str(attempt['attempt']),
            ('Пройдена' if quality['passed'] else 'Не пройдена')+'; min q='+number(quality.get('mean_ratio_minimum',0),9)+
            '; добавлено родителей: '+str(attempt['promoted_parents'])])
    return table('Целевое сгущение под фактические площадки нового угла',['Показатель','Результат'],rows)


def relocated_control_status_table(d):
    state=d['relocated_state'];audit=d['relocated_audit'];rows=[]
    rows.append(['Подготовка целевой сетки','Готова' if d['relocated_quality'] else 'Ожидается'])
    for model,row in zip(RELOCATED_MODELS,d['relocated_forwards'],strict=True):
        name=dict(zip(RELOCATED_MODELS,('Родительская сетка','Унаследованные доли','Пересчитанные доли',
            'Непосредственные контакты','16384 пробы, основное перемешивание','16384 пробы, независимое перемешивание')))[model]
        rows.append([name,'Готовый прямой контроль отсутствует' if row is None else
            'Сохранён; системы и контакты: '+('пройдены' if row['system_checks_passed'] and row['patch_qc']['passed'] else 'не пройдены')])
    if state and state['status']=='failed':rows.append(['Ошибка процедуры',state.get('error','Не уточнена')])
    coverage=d['relocated_coverage']
    rows.append(['Все фактические контактные грани сгущены','Ожидается' if coverage is None else ('Да' if coverage['all_contact_faces_refined'] else 'Нет')])
    rows.append(['Полный численный допуск в этой точке','Ожидается полный аудит' if audit is None else
        ('Пройден' if audit['all_checks_passed'] else 'Не пройден')])
    rows.append(['Количественное физиологическое принятие','Не заявляется'])
    return table('Проверки выбранной точки на целевом сгущении',['Контроль','Состояние'],rows)


def rejected_contacts_table(d):
    rows=[]
    for r in d['checkpoints']:
        if r['status']!='invalid_geometry':continue
        contacts=r.get('patch_qc',{}).get('contacts',[])
        if not contacts:rows.append([r['evaluation'],*['Контактные показатели не сохранены']*4]);continue
        area=max(c['area_error'] for c in contacts);centroid=max(c['centroid_offset_mm'] for c in contacts)
        extent=max(c['extent_mm'] for c in contacts)
        causes=[]
        if area>=.12:causes.append('Площадь')
        if centroid>=1:causes.append('Центроид')
        if extent>=3.7:causes.append('Протяжённость')
        rows.append([r['evaluation'],number(100*area,6),number(centroid,6),number(extent,6),'; '.join(causes) or 'Отдельная причина не установлена'])
    if not rows:rows=[['Отказы не сохранены',*['—']*4]]
    return table('Отказы конечных контактных площадок во время поиска',
        ['Оценка','Максимальная ошибка площади, %; порог <12 %','Смещение центроида, мм; порог <1',
         'Максимальное расстояние узла от центра, мм; порог <3,7','Нарушенный контактный критерий'],rows)


def interrupted_search_table(d):
    selected=d['selected'];state=d['interrupted_state']
    rows=[['Остановка процесса','Не отмечена' if not selected or not selected.get('search_interrupted') else 'Подтверждено отсутствие процесса; причина не установлена']]
    if state:
        time=dt.datetime.fromisoformat(state['updated_utc']).astimezone(dt.timezone(dt.timedelta(hours=3)))
        rows.append(['Последнее состояние до остановки, Москва',time.strftime('%d.%m.%Y %H:%M:%S')])
    if selected and selected.get('search_interrupted'):
        inventory=selected['checkpoint_inventory'];rows.extend([
            ['Завершённых численных оценок',selected['completed_numerical_evaluations']],
            ['Отказов геометрии',sum(r['status']=='invalid_geometry' for r in inventory)],
            ['Незавершённых оценок',sum(r['status']=='forward_calculation_running' for r in inventory)],
            ['Разрешение совместного поиска','Не достигнуто и не подтверждено'],
            ['Поиск после установленного срока','Не возобновлялся; выполнены только последующие проверки']])
    probe=d['coverage_probe']
    if probe:
        rows.extend([['Покрытие контактов нового угла первоначальным сгущением','Полное' if probe['all_contact_faces_refined'] else 'Неполное'],
            ['Минимальная доля сгущённой площади отдельного контакта, %',number(100*probe['minimum_refined_boundary_area_fraction'],6)],
            ['Доля по всем повторным включениям площадок, %',number(100*probe['refined_boundary_area_fraction'],6)]])
    return table('Остановка поиска и отдельная проверка перенесённых площадок',['Сведения','Сохранённый результат'],rows)


def diagnostic_split_table(d):
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None:rows.append([label,*['Свидетельство ещё не опубликовано']*4]);continue
        rows.append([label,number(r['RMSE_ohm'],9),number(np.sqrt(r['mean_SSE_ohm2']/24),9),
            number(np.sqrt(r['contrast_SSE_ohm2']/24),9),number(100*r['contrast_SSE_fraction'],6)])
    return table('Сохранённое разложение остаточной ошибки в каждой отдельной точке',
        ['Точка','RMSE, Ом','RMS ошибки среднего, Ом','RMS ошибки половины контраста, Ом','Контрастная доля SSE, %'],rows)


def pair_residual_contribution_table(d):
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None:rows.append([label,*['Ожидается']*5]);continue
        for pair in r['pairs']:
            rows.append([label,pair['record_id'],
                'Продольная' if pair['axis']=='longitudinal' else 'Поперечная',
                number(pair['mean_error_ohm'],9),
                number(pair['model_contrast_ohm']-pair['measured_contrast_ohm'],9),
                number(100*pair['pair_SSE_fraction'],6)])
    return table('Ошибка средних уровней и вклад каждого отведения в общий критерий',
        ['Точка','Запись','Ось','Ошибка среднего: модель минус измерение, Ом',
         'Ошибка контраста выдох−вдох, Ом','Доля общего SSE, %'],rows)


def connection_transition_table(d):
    rows=[]
    transitions=[('004','006','longitudinal','ТТРКГ 1×4 → боковой крест'),
        ('005','006','perpendicular','ТТРКГ 1×4 → боковой крест'),
        ('010','011','longitudinal','ТТРКГ 1×1 → боковой крест')]
    obs=d['observations']
    for before,after,axis,meaning in transitions:
        for state in ('inhale','exhale'):
            first=obs.index[(obs.record_id=='SCV2-G-'+before)&(obs.axis==axis)&(obs.state==state)].tolist()
            second=obs.index[(obs.record_id=='SCV2-G-'+after)&(obs.axis==axis)&(obs.state==state)].tolist()
            if len(first)!=1 or len(second)!=1:raise ValueError('Connection transition is ambiguous')
            i,j=first[0],second[0];measured=d['target'][j]-d['target'][i]
            for label,candidate in [('Условный подбор',d['profile']),('Новый совместный кандидат',d['best'])]:
                if candidate is None:rows.append([label,before+' → '+after,meaning,'Вдох' if state=='inhale' else 'Выдох',*['Ожидается']*3]);continue
                model=vector(candidate)[j]-vector(candidate)[i]
                rows.append([label,before+' → '+after,meaning,'Вдох' if state=='inhale' else 'Выдох',
                    number(measured,9),number(model,9),number(model-measured,9)])
    return table('Межзаписные изменения при переходе к кресту на той же номинальной боковой сборке',
        ['Точка','Записи','Изменение соседнего монтажа','Фаза',
         'Измерение: после минус до, Ом','Модель: после минус до, Ом','Расхождение, Ом'],rows)


def local_sensitivity_table(d):
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None:rows.append([label,*['Ожидается']*5]);continue
        for k,s in enumerate(r['local_sensitivity']):
            cosine=np.asarray(s['cosine']);condition=s['condition_number']
            rows.append([label,'Вдох' if s['state']=='inhale' else 'Выдох',
                '; '.join(number(v,6) for v in s['norms']),
                '; '.join(number(cosine[a,b],6) for a,b in [(0,1),(0,2),(1,2)]),
                '; '.join(number(v,6) for v in r['gradient_MSE_per_logrho'][3*k:3*k+3]),
                number(condition,3) if condition is not None else 'Не конечна'])
    control,rule=completed_relocated_control(d)
    if control is not None:
        J=np.asarray(control['jacobian_logrho'],float);gradient=2*J.T@(vector(control)-d['target'])/24
        for k,state in enumerate(('inhale','exhale')):
            matrix=J[d['observations'].state.to_numpy()==state,3*k:3*k+3]
            norms=np.linalg.norm(matrix,axis=0);cosine=matrix.T@matrix/np.maximum(np.outer(norms,norms),np.finfo(float).tiny)
            singular=np.linalg.svd(matrix,compute_uv=False)
            rows.append(['Целевая сетка: '+rule,'Вдох' if state=='inhale' else 'Выдох',
                '; '.join(number(v,6) for v in norms),
                '; '.join(number(cosine[a,b],6) for a,b in [(0,1),(0,2),(1,2)]),
                '; '.join(number(v,6) for v in gradient[3*k:3*k+3]),
                number(singular[0]/singular[-1],3) if singular[-1] else 'Не конечна'])
    return table('Локальные материальные чувствительности сохранённых точек',
        ['Точка','Фаза','Нормы J: фон; лёгкое; жир, Ом','Косинусы: фон–лёгкое; фон–жир; лёгкое–жир',
         'Градиент MSE: фон; лёгкое; жир, Ом²','Обусловленность фазового J'],rows)


def linear_projection_table(d):
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None:rows.append([label,*['Ожидается']*4]);continue
        p=r['bounded_linear_contrast_diagnostic']
        rows.append([label,'; '.join(number(v,6) for v in p['logrho_step']),
            number(p['largest_absolute_log_step'],6),number(p['contrast_RMSE_ohm'],9),
            'Завершено' if p['solver_success'] else 'Успех линейного алгоритма не подтверждён'])
    return table('Ограниченная линейная диагностика дыхательного контраста',
        ['Точка','Шесть шагов Δlnρ: вдох, затем выдох; фон, лёгкое, жир','Максимум |Δlnρ|',
         'RMSE линейного контраста, Ом','Линейный алгоритм'],rows)


def material_bound_gradient_table(d):
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None:rows.append([label,*['Ожидается']*6]);continue
        gradient=np.asarray(r['gradient_MSE_per_logrho'],float)
        lower=np.asarray(r['lower_bound_parameters'],bool)
        upper=np.asarray(r['upper_bound_parameters'],bool)
        for i,g in enumerate(gradient):
            blocked=bool((lower[i] and g>0) or (upper[i] and g<0))
            rows.append([label,'Вдох' if i<3 else 'Выдох',('Фон','Лёгкое','Жир')[i%3],
                'Нижний' if lower[i] else ('Верхний' if upper[i] else 'Внутри'),
                number(g,9),number(0. if blocked else g,9),
                'Да' if blocked else 'Нет'])
    return table('Градиент по сопротивлениям с учётом активных технических пределов',
        ['Точка','Фаза','Материал','Предел при |Δlnρ|≤10⁻⁴','Градиент MSE, Ом²',
         'Допустимая компонента градиента, Ом²','Спуск заблокирован пределом'],rows)


def linear_contrasts_table(d):
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None:rows.append([label,*['Ожидается']*5]);continue
        for pair,value in zip(r['pairs'],r['bounded_linear_contrast_diagnostic']['linear_contrast_ohm'],strict=True):
            name=pair['record_id'].rsplit('-',1)[-1]+' · '+('продольная' if pair['axis']=='longitudinal' else 'поперечная')
            rows.append([label,name,number(pair['measured_contrast_ohm'],6),number(pair['model_contrast_ohm'],6),
                number(value,6),number(value-pair['measured_contrast_ohm'],6)])
    return table('Все контрасты сохранённой линейной диагностики: выдох минус вдох',
        ['Точка','Запись и ось','Измеренный контраст, Ом','Исходный модельный, Ом','Линейный, Ом','Линейный минус измеренный, Ом'],rows)


def source_scenarios_table(d):
    names={'own_drive':('Собственный источник','—'), 'in_phase':('Софазные источники','0'),
        'opposite_phase':('Противофазные источники','π'), 'quadrature_magnitude':('Модуль квадратурной суммы','π/2')}
    rows=[]
    for key,label in DIAGNOSTIC_LABELS:
        r=d['residual_diagnostics'][key]
        if r is None or not r['source_scenarios']:rows.append([label,*['Сценарии ещё не сохранены']*5]);continue
        for scenario in r['source_scenarios']:
            name,phase=names[scenario['name']]
            rows.append([label,name,phase,number(scenario['RMSE_ohm'],9),scenario['wrong_signs'],
                'Равные амплитуды приняты' if scenario['equal_source_amplitudes_assumed'] else 'Встречный источник исключён'])
    return table('Неподогнанные сценарии источников при тех же сопротивлениях и положении',
        ['Точка','Сценарий','Относительная фаза φ, рад','RMSE уровней, Ом','Несовпадений знака из 12','Допущение'],rows)


def saved_material_slices(d):
    best=d['best']
    if best is None:return []
    p=np.asarray(best['parameters']);error=vector(best)-d['target'];base=float(error@error);result=[]
    for phase,state in enumerate(('inhale','exhale')):
        ids=(d['observations'].state==state).to_numpy();fixed_other=float(error[~ids]@error[~ids])
        for material in range(3):
            coordinate=3*phase+material;fixed=[i for i in range(3*phase,3*phase+3) if i!=coordinate]
            samples={}
            for row in d['checkpoints']:
                if (row['status']!='numerical_candidate_not_validated' or row['identity']!=best['identity'] or
                        row['pose']!=best['pose']):continue
                q=np.asarray(row['parameters'])
                if not np.allclose(q[fixed],p[fixed],rtol=0,atol=1e-12):continue
                residual=vector(row)[ids]-d['target'][ids];rss=float(residual@residual)
                sample=samples.setdefault(float(q[coordinate]),dict(state=state,material=material,
                    logrho=float(q[coordinate]),rho=float(np.exp(q[coordinate])),phase_RSS=rss,
                    conditional_MSE=(rss+fixed_other)/24,delta_MSE=(rss+fixed_other-base)/24,
                    source_evaluations=[],duplicate_phase_RSS_difference=0.))
                sample['source_evaluations'].append(row['evaluation'])
                sample['duplicate_phase_RSS_difference']=max(sample['duplicate_phase_RSS_difference'],abs(rss-sample['phase_RSS']))
            result.extend(samples[key] for key in sorted(samples))
    return result


def saved_material_slices_table(d):
    rows=[[('Вдох' if r['state']=='inhale' else 'Выдох'),('Фон','Лёгкое','Жир')[r['material']],
        number(r['rho'],9),number(r['phase_RSS'],12),number(r['conditional_MSE'],12),number(r['delta_MSE'],12),
        ', '.join(map(str,r['source_evaluations'])),number(r['duplicate_phase_RSS_difference'],12)] for r in saved_material_slices(d)]
    if not rows:rows=[['Полный выбранный кандидат отсутствует',*['—']*7]]
    return table('Срезы невязки по материалам из уже завершённых FEM-оценок при одном положении',
        ['Фаза','Материал','ρ, Ом·м','Фазовая SSE, Ом²','Условная MSE всех уровней, Ом²','ΔMSE к выбранной точке, Ом²',
         'Исходные номера оценок','Максимальная разность повторных фазовых SSE, Ом²'],rows)


def saved_material_slices_figure(d,include_js=False):
    from plotly.subplots import make_subplots
    samples=saved_material_slices(d)
    if not samples:return HTML('<p>Сохранённые срезы при выбранном положении пока отсутствуют.</p>')
    fig=make_subplots(rows=2,cols=3,subplot_titles=[phase+': '+material for phase in ('Вдох','Выдох') for material in ('фон','лёгкое','жир')],
        horizontal_spacing=.09,vertical_spacing=.22)
    best=d['best'];params=np.asarray(best['parameters'])
    for phase,state in enumerate(('inhale','exhale')):
        for material in range(3):
            rows=[r for r in samples if r['state']==state and r['material']==material];name=('Вдох' if phase==0 else 'Выдох')+': '+('фон','лёгкое','жир')[material]
            fig.add_trace(go.Scatter(x=[r['rho'] for r in rows],y=[r['delta_MSE'] for r in rows],mode='lines+markers',
                name=name,legendgroup=name,marker=dict(color=PALETTE[3*phase+material],symbol='circle'),
                customdata=[', '.join(map(str,r['source_evaluations'])) for r in rows],
                hovertemplate='ρ=%{x:.9g} Ом·м<br>ΔMSE=%{y:.9g} Ом²<br>Оценки: %{customdata}<extra>%{fullData.name}</extra>'),row=phase+1,col=material+1)
            fig.add_trace(go.Scatter(x=[float(np.exp(params[3*phase+material]))],y=[0.],mode='markers',
                name='Выбранная точка',legendgroup=name,showlegend=False,marker=dict(color='#b52d34',symbol='x',size=10)),row=phase+1,col=material+1)
            fig.add_hline(y=0,line_dash='dash',line_color='#333333',row=phase+1,col=material+1)
            fig.update_xaxes(type='log',title_text='ρ, Ом·м',row=phase+1,col=material+1)
            fig.update_yaxes(title_text='ΔMSE, Ом²',row=phase+1,col=material+1)
    fig.update_layout(height=830,legend=dict(orientation='h',y=1.17),margin=dict(t=150))
    return figure_html(fig,include_js)


def hypotheses_table(d):
    rows=[['Численные ошибки','Раздельные новые контроли пространства, долей и контактов',
        'Допуски пройдены в выбранной точке; сходимость всей области поиска не установлена' if d['relocated_audit'] and d['relocated_audit']['all_checks_passed'] else 'До завершения контролей модельное расхождение не отделено от дискретизации'],
        ['Нестационарность прежней точки','Новый условный подбор и затем совместный поиск','Прежняя точка не является установленным минимумом по материалам'],
        ['Доминирование абсолютных уровней','Раздельные ошибки средних и контрастов','Произвольные новые веса и доверительные интервалы не вводятся'],
        ['Статическая геометрия двух фаз','Проверка размерного и направленного отклика','Дыхательное движение анатомии не измерено и не моделируется'],
        ['Фиксированные кость и сердце','Отдельные документированные сценарии после численного допуска','Производные по фиксированным свойствам здесь не рассчитаны'],
        ['Межзаписные изменения монтажа','Сравнение записей 005 → 006 и 010 → 011','Это не независимые повторы и не доказательство дрейфа прибора'],
        ['Неизмеренная схема возбуждения','Сценарии собственного и встречного источников при тех же параметрах','Меньший RMSE сценария не устанавливает реальную токовую фазу']]
    return table('Проверяемые гипотезы и нерешённые вопросы',['Вопрос','Разделяющая проверка','Ограничение вывода'],rows)


def provenance_table(d):
    snap=d['snapshot']
    tests=d['synthetic_tests'];test_status=('Отдельное свидетельство отсутствует' if tests is None else
        str(tests['passed_tests'])+' тест; '+str(tests['passed_subtests'])+' подтеста; малые синтетические данные, без FEM и медицинских массивов')
    return table('Происхождение и границы проверки отчёта',['Объект','Проверка'],[
        ['Измерения, код, планы и прочитанные малые паспорта','Точные байты и контрольные суммы; исторический код — только по совпавшим архивным байтам'],
        ['Параметры и полное положение каждого прямого контроля','Связаны с собственными опорными данными и порядком 24 наблюдений'],
        ['Новый подобранный кандидат','Собственный файл нового поиска; начальный допуск и проверки выбранной точки разделены'],
        ['Крупные сетки, доли, подготовленные матрицы и медицинские массивы','Сохранённые идентичности; повторное хеширование крупных массивов этим отчётом не выполнялось'],
        ['Проверенные малые или программные источники',len(snap.guard_records)],
        ['Модульные проверки программной алгебры и происхождения',test_status],
        ['Самостоятельная физиологическая валидация','Не выполнена']])


def observation_labels(d):
    return [r.record_id.rsplit('-',1)[-1]+' · '+str(int(r.size_mm))+' мм · '+
        ('продольная' if r.axis=='longitudinal' else 'поперечная')+' · '+
        ('вдох' if r.state=='inhale' else 'выдох') for _,r in d['observations'].iterrows()]


def levels_figure(d,include_js=False):
    fig=go.Figure();labels=observation_labels(d)
    for state,name,color in [('inhale','Измерение: вдох','#b7482b'),('exhale','Измерение: выдох','#23778a')]:
        ids=np.flatnonzero(d['observations'].state.to_numpy()==state)
        fig.add_trace(go.Scatter(x=[labels[i] for i in ids],y=d['target'][ids],mode='lines+markers',name=name,
            line=dict(color=color,dash='dot'),marker=dict(symbol='circle',size=8)))
    for k,(name,row) in enumerate(available_series(d)):
        fig.add_trace(go.Scatter(x=labels,y=vector(row),mode='lines+markers',name=name,
            line=dict(color=PALETTE[k%len(PALETTE)]),marker=dict(symbol='square',size=7)))
    fig.update_layout(xaxis=dict(title='Запись, размер, ось и фаза',tickangle=-60,
            categoryorder='array',categoryarray=labels),
        yaxis_title='Передаточный импеданс, Ом',height=800,
        legend=dict(groupclick='toggleitem',orientation='h',yanchor='bottom',y=1.03,xanchor='left',x=0),
        margin=dict(t=160,b=210))
    return figure_html(fig,include_js)


def mesh_section_figure(d,axis,include_js=False):
    if d['mesh_sections'] is None:return HTML('<p>Сохранённые сечения сетки в начальном положении ещё не опубликованы.</p>')
    fig=go.Figure()
    for model,label,color in [('parent','Родительская сетка','#888888'),('child','Третий уровень','#2c77bb')]:
        s=next(s for s in d['mesh_sections']['sections'] if (s['model'],s['axis'])==(model,axis))
        fig.add_trace(go.Scattergl(x=s['x'],y=s['y'],mode='lines',name=label,
            line=dict(color=color,width=.6),hoverinfo='skip',connectgaps=False))
    fig.add_trace(go.Scatter(x=[0],y=[0],mode='markers',name='Начальный общий центр (0, 0)',
        marker=dict(color='#222222',symbol='diamond',size=8)))
    name='продольной' if axis=='longitudinal' else 'поперечной'
    fig.update_layout(title='Сечение вдоль '+name+' оси: начальное фиксированное положение',
        xaxis=dict(title='Смещение вдоль '+name+' оси, мм',range=[-80,80],constrain='domain'),
        yaxis=dict(title='Глубина вдоль внутренней нормали, мм',range=[85,-5],scaleanchor='x',scaleratio=1),
        legend=dict(orientation='h',yanchor='bottom',y=1.03,x=0,groupclick='toggleitem'),
        height=700,margin=dict(t=120),uirevision='initial-section-'+axis)
    return figure_html(fig,include_js)


def longitudinal_section_figure(d,include_js=False):return mesh_section_figure(d,'longitudinal',include_js)


def perpendicular_section_figure(d,include_js=False):return mesh_section_figure(d,'perpendicular',include_js)


def contrasts_figure(d,include_js=False):
    fig=go.Figure();labels=[label for _,_,label in d['pairs']]
    fig.add_trace(go.Scatter(x=labels,y=contrasts(d,d['target']),mode='markers',name='Измерение',
        marker=dict(symbol='circle',color='#222222',size=9)))
    for k,(name,row) in enumerate(available_series(d)):
        fig.add_trace(go.Scatter(x=labels,y=contrasts(d,vector(row)),mode='markers',name=name,
            marker=dict(symbol='square',color=PALETTE[k%len(PALETTE)],size=8)))
    fig.add_trace(go.Scatter(x=labels,y=[0]*12,mode='lines',name='Нулевой контраст',line=dict(color='#222222',dash='dash')))
    fig.update_layout(xaxis=dict(title='Запись, размер и ось',tickangle=-55),yaxis_title='Выдох − вдох, Ом',height=700,margin=dict(b=190))
    return figure_html(fig,include_js)


def numerical_figure(d,include_js=False):
    fig=go.Figure();labels=observation_labels(d);found=False
    points=('Начальная точка','Новый угол, прежняя сетка','Новый угол, целевая сетка')
    names=('Пространство FEM','Доли тканей','Суммарное изменение','Контактные грани','Число проб',
        'Другое перемешивание','Вложенные: основное','Вложенные: независимое')
    for dataset,(point,rows,_,_,_) in enumerate(numerical_sets(d)):
        for k,row in enumerate(rows):
            if row['pending']:continue
            found=True;fig.add_trace(go.Scatter(x=labels,y=row['difference_ohm']/row['tolerance_ohm'],mode='markers',
                name=points[dataset]+': '+names[k],marker=dict(symbol=('circle','diamond','triangle-up')[dataset],
                color=PALETTE[k%len(PALETTE)],size=7)))
    if not found:return HTML('<p>Новые пары завершённых численных контролей пока отсутствуют. График численной сходимости не сформирован.</p>')
    fig.add_trace(go.Scatter(x=labels,y=[1]*24,mode='lines',name='Граница допуска',line=dict(color='#222222',dash='dash')))
    fig.update_layout(xaxis=dict(title='Наблюдение',tickangle=-60),yaxis_title='|ΔZ| / допуск уровней',height=760,margin=dict(b=210))
    return figure_html(fig,include_js)


def history_figure(d,include_js=False):
    rows=d['checkpoints'];valid=[r for r in rows if r['status']=='numerical_candidate_not_validated']
    if not valid:return HTML('<p>Готовых численных оценок нового совместного поиска пока нет. Прежняя история поиска не подставлена.</p>')
    ids=[r['evaluation'] for r in valid];values=np.sqrt([r['mean_square_ohm2'] for r in valid]);fig=go.Figure()
    fig.add_trace(go.Scatter(x=ids,y=values,mode='markers',name='Готовая оценка',marker=dict(color='#777777',symbol='circle')))
    fig.add_trace(go.Scatter(x=ids,y=np.minimum.accumulate(values),mode='lines',name='Лучшее среди завершённых оценок',line=dict(color='#326ba8')))
    failed=[r['evaluation'] for r in rows if r['status']=='invalid_geometry']
    if failed:fig.add_trace(go.Scatter(x=failed,y=[max(values)]*len(failed),mode='markers',name='Отказ геометрии: условная высота',marker=dict(symbol='x',color='#bd3535',size=10)))
    fig.update_layout(xaxis_title='Номер новой оценки',yaxis_title='RMSE, Ом',height=550)
    return figure_html(fig,include_js)


def placement_figure(d,include_js=False):
    if d['best'] is None:return HTML('<p>Нового выбранного положения нет. Прежняя картинка и исходное приближение не показаны как новый результат.</p>')
    pose=d['best']['pose'];fig=go.Figure()
    if d['display_surface'] is not None:
        v=np.asarray(d['display_surface']['vertices_lps_mm']);f=np.asarray(d['display_surface']['faces_zero_based'])
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            color='#c6aa93',opacity=.25,name='Наружная граница модели: обзор',showlegend=True,hoverinfo='skip'))
    for block in pose['electrodes']:
        xyz=np.asarray(block['centres_lps_mm']);longitudinal=block['axis']=='longitudinal'
        label=str(block['size_mm'])+' мм · '+('продольная' if longitudinal else 'поперечная')
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines+markers',name=label,
            line=dict(color='#df8b26' if longitudinal else '#3989ad'),
            marker=dict(size=4,color='#df8b26' if longitudinal else '#3989ad',symbol=['circle','square','square','circle']),
            text=['I+','U+','U−','I−'],hovertemplate='%{text}<br>LPS, мм: %{x:.3f}; %{y:.3f}; %{z:.3f}<extra>%{fullData.name}</extra>'))
    c=pose['centre_lps_mm'];fig.add_trace(go.Scatter3d(x=[c[0]],y=[c[1]],z=[c[2]],mode='markers',
        name='Новый общий центр',marker=dict(color='#bd3535',symbol='diamond',size=7)))
    author=d['author_centre']['centre_lps_mm']
    fig.add_trace(go.Scatter3d(x=[author[0]],y=[author[1]],z=[author[2]],mode='markers',
        name='Приближённый центр автора',marker=dict(color='#666666',symbol='x',size=6)))
    fig.update_layout(scene=dict(xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм',
        aspectmode='data',camera=dict(eye=dict(x=-1.4,y=-1.2,z=.7)),
        uirevision='preserve-new-candidate-view'),uirevision='preserve-new-candidate-view',height=650)
    return figure_html(fig,include_js)


def result_summary(d):
    if d['best'] is None:return 'Полный новый кандидат ещё не сохранён. Результаты подготовки и начальных проверок приведены отдельно.\n\n'
    z=vector(d['best']);rmse=float(np.sqrt(np.mean((z-d['target'])**2)))
    wrong=int(np.count_nonzero(np.sign(contrasts(d,z))!=np.sign(contrasts(d,d['target']))))
    chosen=d['selected'];completed=chosen.get('completed_numerical_evaluations',sum(r['status']=='numerical_candidate_not_validated' for r in d['checkpoints']))
    paragraph=(f'**Сохранённый результат поиска:** лучшая RMSE среди {completed} завершённых допустимых оценок — {number(rmse,6)} Ом; '
        f'формальное несовпадение дыхательного знака сохраняется у {wrong} из 12 отведений. Разрешение поиска не подтверждено. '
        'Часть сопротивлений достигла технической границы, а стационарность выбранной точки не подтверждена. '
        'Обоснованные индивидуальные сопротивления в этом прогоне пока не получены.\n\n')
    audit=d['relocated_audit'];control,rule=completed_relocated_control(d)
    if control is not None:
        value=float(np.sqrt(np.mean((vector(control)-d['target'])**2)))
        signs=int(np.count_nonzero(np.sign(contrasts(d,vector(control)))!=np.sign(contrasts(d,d['target']))))
        paragraph+=f'Прямой контроль тех же параметров на сетке под новые площадки ({rule}) даёт RMSE {number(value,6)} Ом и {signs} несовпадений дыхательного знака из 12. '
    paragraph+=('Полный численный аудит целевой сетки ещё не опубликован.' if audit is None else
        ('Численные проверки в выбранной точке на целевой сетке пройдены.' if audit['all_checks_passed'] else
         'Полный численный допуск выбранной точки на целевой сетке не пройден.'))
    return paragraph+' Это состояние численной модели; независимая физиологическая валидация не выполнена.\n\n'


def selected_residual_summary(d):
    diagnostic=d['residual_diagnostics']['selected']
    if diagnostic is None:return 'Разбор новой выбранной точки ещё не опубликован.\n\n'
    pairs=sorted(diagnostic['pairs'],key=lambda row:row['pair_SSE_fraction'],reverse=True)[:2]
    names=['запись '+row['record_id'].rsplit('-',1)[-1]+', '+
        ('продольная' if row['axis']=='longitudinal' else 'поперечная')+' ось' for row in pairs]
    share=100*sum(row['pair_SSE_fraction'] for row in pairs)
    ratios=[row['norms'][0]/row['norms'][1] for row in diagnostic['local_sensitivity']]
    return (f"В новой выбранной точке на сетке поиска на две пары — {'; '.join(names)} — приходится {number(share,2)} % SSE. "
        f"Дыхательная составляющая даёт {number(100*diagnostic['contrast_SSE_fraction'],3)} % SSE. "
        'Таким образом, текущий критерий в основном отражает расхождение средних абсолютных уровней. '
        'Эта структура ошибки не устанавливает дефект записи и не служит основанием для её исключения.\n\n'
        f'Отношение нормы чувствительности к логарифму сопротивления фона к соответствующей норме лёгкого '
        f'составляет примерно {number(ratios[0],0)} на вдохе и {number(ratios[1],0)} на выдохе. '
        'Это локальная характеристика выбранных сопротивлений и положения на сетке поиска. '
        'Она объясняет слабую численную различимость изменения сопротивления лёгкого в этой точке; '
        'возможность лучшего нелинейного решения в другой части области поиска остаётся открытой.\n\n')


def text(d):
    q=d['mesh_quality'];prep=d['preparation'];shape='Готовая проверка формы третьего уровня ещё не опубликована.'
    if q is not None:
        promotions=max(0,int(q['promotion_attempts'])-1)
        shape=(f"Для начальной точки третий уровень содержит **{q['nodes']} узла и {q['tetrahedra']} тетраэдра**. Минимальный безразмерный показатель формы — **{number(q['mean_ratio_minimum'],9)}**, обусловленность рёбер — **{number(q['edge_condition_maximum'],3)}**. После исходной попытки выполнено раундов дополнительного уточнения соседних элементов: **{promotions}**. Относительное изменение общего объёма — **{number(q['relative_volume_difference'],12)}**.")
    repeat='Отдельное свидетельство воспроизводимости прямого расчёта ещё не опубликовано.'
    if d['repeat'] is not None and d['forwards'][0] is not None:
        rmse=np.sqrt(np.mean((vector(d['forwards'][0])-d['target'])**2))
        repeat=(f"Для завершённого повторения максимальная разность равна {number(d['repeat']['maximum_Z_difference_ohm'],12)} Ом; RMSE прямой исходной модели — {number(rmse,12)} Ом.")
    segmentation_note='Происхождение тканевых масок не подтверждено отдельным паспортом в этой сборке.\n\n'
    if d['segmentation_passport']:
        s=d['segmentation_passport']
        segmentation_note=(f"Четыре тканевые маски лёгкого, кости, сердца и жира получены автоматической сегментацией {s['backend']} {s['version']} полного статического КТ. Паспорт и контрольные суммы масок сопоставлены с фактическими входами FEM. В паспорте указано отсутствие ручной валидации этих автоматических масок. Их одинаковое использование в численных вариантах позволяет разделять ошибки дискретизации, но не устанавливает анатомической точности сегментации.\n\n")
    contact_note=''
    if d['contact_size_note'] is not None:
        note=d['contact_size_note']
        contact_note=(f"\n\nАвтор приблизительно вспоминает размер контактной площадки {number(note['metal_contact_diameter_mm_approximate'],1)} мм и области геля {number(note['gel_footprint_diameter_mm_approximate'],1)} мм. Это воспоминание с неопределённостью, без независимого измерения эффективной электрической площади. Гель может расширять область ввода тока, однако его протяжённость сама по себе не задаёт диаметр контакта полной электродной модели. Текущий расчёт сохраняет принятый диск диаметром {number(note['current_frozen_FEM_diameter_mm'],1)} мм; расчёт с диаметром 10 мм здесь не выполнен. Проверка иной площади требует отдельного явно заданного сценария и собственных контактных и сеточных контролей.\n\n")
    return r'''# Численная сходимость и совместный подбор: Георг, Scorpion v2

Это самостоятельный исследовательский расчёт после исправления позднего монтажа рук. Его порядок — сначала проверка численных ошибок при фиксированных сопротивлениях и полном положении электродов, затем условный подбор материалов и совместный поиск сопротивлений и положения. Все выводы относятся к развитию методики. Конечноэлементная модель (FEM) использует полную электродную модель (CEM) с конечными контактными площадками. RMSE — среднеквадратическая невязка 24 уровней, MSE — средний квадрат невязки, равный RMSE², SSE — сумма квадратов ошибок; RMS обозначает среднеквадратический уровень отдельной составляющей ошибки.

'''+result_summary(d)+r'''

<!-- status_table -->
<!-- preparation_changes_table -->

При отсутствии полного результата стадия отмечается как незавершённая либо заменённая другим контролем. Такие стадии не заполняются прежними импедансами, прежними сопротивлениями или начальным приближением. Сохранённое состояние процесса не заменяет завершённого численного свидетельства. Общий срок ограничивает поиск и контрольные расчёты; остановка по времени не доказывает достижения разрешения или оптимума.

## Измерения, монтаж и рабочие допущения

В целевую функцию включены 24 уровня на кандидатных задержках вдоха и выдоха. Для диагностики сохранены все 12 парных контрастов. Запись спокойного дыхания в целевую функцию не включена. Автор подтвердил монтаж ТТРКГ 1×1 в записях 007–010 и полное снятие проводов рук в записях 006 и 011. Ранние монтажи сохраняют собственные обозначения; подтверждение позднего 1×1 не переносится на весь эксперимент. Наличие оставшихся физических площадок рук принято как допущение: при снятых проводах они моделируются отдельно плавающими, с нулевым внешним током.

Обхват обеих рук 37 см на уровне токовых электродов сообщён автором. Руки представлены цилиндрами постоянного сечения; вне поля КТ их объём отнесён к фону с тем же свободным ρ₁. Цилиндрическая форма и перенос одного обхвата по длине руки являются геометрическими допущениями, а не результатом индивидуальной сегментации руки.

Дыхательные команды прибором не размечены. Интервалы восстановлены экспертно; текущая кандидатная разметка ожидает ручного принятия. Ни новая сетка, ни снижение невязки не меняют этот статус. Одна индивидуальная КТ глубокого вдоха используется для обеих фаз. Фазовые сопротивления подбираются раздельно, а положение сборок общее. Сопоставимость состояния в последовательных записях и статическая геометрия двух фаз остаются допущениями.

<!-- observations_table -->
<!-- quiet_table -->

Запись 001 содержит самостоятельное описательное наблюдение спокойного дыхания: 81,33071 Ом в канале BEI2, продольная сборка 140 мм, монтаж ТТРКГ 1×1. Этот уровень сохранён в свежей таблице измерений, но не является 25-й строкой целевой функции. Спокойному дыханию не присваивается задержка на вдохе или выдохе. Сопоставление с расчётом собственного источника возможно только при отдельно определённом дыхательном состоянии и применимой схеме подключения; соответствующее фазовое сопротивление здесь не назначено и модельная разность не вычислена.

<!-- properties_table -->

Кость целиком представлена кортикальной костью, вся маска сердца — миокардом; отдельные кровь и миокард внутри неё не моделируются. Фон включает весь объём вне масок лёгкого, кости, сердца и жира, хотя его опорное свойство взято как для мышцы. Фиксированные свойства и технические границы 0,1–100 Ом·м не являются независимыми индивидуальными референсами или физиологическими нормами. Производные по фиксированным свойствам кости и сердца здесь не рассчитаны; их количественное влияние не объявляется малым.

Диаметр диска 5 мм и удельный контактный импеданс приняты без индивидуального измерения. Номинальная площадь равна $\pi(d/2)^2$; фактическая площадь — сумма выбранных граничных граней, без независимого измерения контакта с кожей. Повторные описания контактов в разных записях не являются числом уникальных физических электродов.

'''+contact_note+r'''Реальные токовые фазы и способ регистрации смеси двух источников аппаратно не установлены. Здесь сохранён сценарий собственного источника, без подгонки произвольного усиления, смещения, канального знака или перестановки ролей.

'''+segmentation_note+r'''## Прямая и обратная задачи

**Модельное допущение.** В фиксированном объёме КТ решается квазистатическая задача $\nabla\cdot(\sigma\nabla\psi)=0$, где $\psi$ — электрический потенциал, $\sigma=1/\rho$ — вещественная изотропная проводимость. Частота расчётного сценария равна 100 кГц. Комплексные и анизотропные свойства тканей, дыхательное перемещение границ и изменение формы грудной клетки в этой постановке не моделируются. Свободные ρ₁, ρ₂, ρ₃ относятся соответственно к оставшемуся фону, целой маске лёгкого и жиру; кость и сердце сохраняют отдельные фиксированные свойства. Эквивалентная толщина h здесь не подбирается: используется полный неоднородный объём КТ.

На конечной площадке электрической группы $l$ полная электродная модель задаёт $\psi+z_c\sigma\,\partial\psi/\partial n=V_l$ и интегральное условие $\int_{E_l}\sigma\,\partial\psi/\partial n\,dS=I_l$. Здесь $E_l$ — объединение площадок группы, $V_l$ — общий электродный потенциал, $I_l$ — заданный внешний ток группы, $z_c$ — удельный контактный импеданс в Ом·м², $n$ — наружная нормаль. Одна группа может объединять несколько физических площадок. Вне контактов нормальный ток равен нулю; сумма токов групп равна нулю. Для собственного источника задаётся единичный ток между I+ и I−, а внешний ток измерительных и плавающих групп равен нулю. Передаточный импеданс вычисляется как $Z=(V_{U+}-V_{U-})/I$. Такая нормировка использует линейность принятой модели и не является измерением амплитуды тока прибора.

**Численная модель.** КТ-маски задают доли $f_{e,m}$ пяти материалов внутри каждого тетраэдра $e$; $m$ — номер материала. Полный вектор удельных сопротивлений материалов имеет порядок $(r_1,r_2,r_3,r_4,r_5)=(\rho_1,\rho_2,\rho_{\text{кость}},\rho_{\text{сердце}},\rho_3)$. Здесь $r_m$ обозначает удельное сопротивление материала, а ρ₁, ρ₂, ρ₃ — три свободных сопротивления фона, лёгкого и жира. В расчёте используется $\sigma_e=\sum_{m=1}^{5}f_{e,m}/r_m$, а не среднее сопротивление. Доли в сумме равны единице; координаты переводятся из миллиметров в метры. При проверке пространства FEM доли точно наследуются от родительских элементов. Затем на той же дочерней сетке они пересчитываются по КТ, чтобы отдельно оценить влияние представления тканей. Контактные группы и конечная площадь сохраняются при этих двух сравнениях; изменение контактной дискретизации проверяется следующим самостоятельным контролем.

**Обратная задача.** Минимизируется $\mathrm{MSE}=\frac{1}{24}\sum_{k=1}^{24}[Z_{\mathrm{FEM},k}(\theta)-Z_{\text{изм},k}]^2$, с одинаковым весом всех 24 уровней. Вектор θ содержит шесть фазовых логарифмов сопротивлений, два поверхностных смещения общего центра и угол сборок. Логарифм берётся от численного значения ρ в Ом·м, то есть относительно 1 Ом·м. Для каждого положения сохраняются все семь альтернативных боковых сборок, включая обе оси крестов. Номинальные точки I расположены на ±d/2, точки U — на ±d/4 вдоль оси сборки, после чего проецируются на поверхность; номинальное расстояние не объявляется измеренной геодезической длиной по коже. Положение общее для вдоха и выдоха, а сопротивления двух фаз независимы. Дыхательные контрасты используются для диагностики и численных проверок, без замены ими исходной целевой функции или добавления новых весов.

## Геометрическое уточнение и правила интегрирования

Для формы тетраэдра используется безразмерный показатель $q=12(3V)^{2/3}/\sum_{\alpha=1}^{6}l_\alpha^2$, где $V$ — его объём, $l_\alpha$ — длины шести рёбер. Для правильного тетраэдра q=1; при вырождении q стремится к нулю. Обусловленность рёбер — отношение крайних сингулярных значений матрицы трёх рёбер из одной вершины. В таблице целевой сетки показаны также перцентили этих характеристик и число элементов за заранее установленными порогами. Это проверки дискретной геометрии; точность импеданса оценивается отдельными прямыми расчётами.

'''+shape+r'''

<!-- mesh_table -->
<!-- actual_coverage_table -->
<!-- fine_contact_coverage_table -->
<!-- relocated_mesh_table -->
<!-- integration_table -->

<!-- longitudinal_section_figure -->

**Рисунок 1. Сечение сетки вдоль продольной оси.** Серые линии — пересечения тетраэдров родительской сетки с плоскостью, синие — пересечения тетраэдров третьего уровня. Чёрный ромб обозначает начальный общий центр с координатами (0, 0) в плоскости. Ряды переключаются независимо в собственной легенде. Плоскость задана сохранённым начальным положением, до нового подбора. По горизонтали показано смещение от центра, по вертикали — глубина вдоль внутренней нормали; положительная глубина направлена внутрь и вниз на рисунке. Окно составляет −80…80 мм по горизонтали и −5…85 мм по глубине. Это реальные пересечения тетраэдров, без изображения тканей КТ и без оценки качества трёхмерных элементов по двумерному рисунку.

<!-- perpendicular_section_figure -->

**Рисунок 2. Сечение сетки вдоль поперечной оси.** Серые линии — родительская сетка, синие — третий уровень, чёрный ромб — начальный общий центр. Обе сетки пересечены одной плоскостью начального фиксированного положения; это не новое найденное положение и не изображение тканей КТ. Ряды отключаются независимо в легенде этого рисунка. По горизонтали показано смещение вдоль поперечной оси, по вертикали — глубина внутрь тела с обращённой шкалой. Окно и масштаб совпадают с продольным сечением: −80…80 мм и −5…85 мм. Плотность линий служит иллюстрацией сетки; численные показатели качества определяются отдельными трёхмерными проверками.

Пределы формы сохранены: показатель формы не ниже 0,01, обусловленность рёбер не выше 10000, ориентированные объёмы положительны. Расширение разрешённого числа узлов до 1,5 миллиона является явным ресурсным изменением; допуски формы и импеданса не ослабляются. Геометрический контроль и неизменность объёма сами по себе не устанавливают сходимости электрического решения.

Свидетельство покрытия связано с фактическими граничными гранями прямого родительского расчёта и картой их дочерних граней. Радиус направляющих сам по себе не служит доказательством покрытия. Описания площадок суммируются по условиям записи: 168 описаний не означают 168 уникальных физических электродов. Проверка относится к начальному положению; при перемещении сборок требуется отдельное свидетельство покрытия, а сходимость всего объёма этим контролем не устанавливается.

Дополнительная проверка берёт непосредственно контактные грани сгущённой сетки из сохранённого прямого расчёта. Грань отмечена как сгущённая, если её непосредственная родительская грань имеет более одного потомка во всей новой границе. Доля вычислена по площади треугольников; повторные включения одной грани в разных условиях сохранены. Эта проверка не использует старую контактную площадку для определения новой и связана с собственными координатами и хэшами файлов. Начальная и вновь найденная точки показаны отдельно; отсутствие нового свидетельства не заменяется начальным результатом. Такое покрытие проверяет локальную границу контактов, без доказательства сходимости объёмного решения.

При переходе к третьему уровню отдельно рассматриваются увеличение пространства FEM с точным наследованием проводимости и контактов, затем пересчёт тканевых долей на той же объёмной сетке. Непосредственный выбор контактных граней сравнивается отдельно при неизменных сопротивлениях, положении и объёмной сетке. Для проверки интегрирования предусмотрены вложенные правила 4096 и 16384 проб и два явно сохранённых начальных числа случайного перемешивания. Во второй подготовке все смешанные тетраэдры интегрируются заново; сертифицированные однородные элементы имеют точные доли. Приоритет масок — лёгкое, кость, сердце, жир, фон. Используется ближайший воксель с округлением вниз после прибавления 0,5; область вне поля КТ принята фоном. Изменение начального числа генератора проверяет устойчивость численной интеграции, без создания новых измерений или оценки физиологической неопределённости.

<!-- reproducibility_table -->

Повторяемость прямого родительского расчёта проверяется отдельно от сходимости: сравниваются прежний и новый расчёты с непосредственным выбором контактных граней, точно теми же сопротивлениями, полным положением и целевыми уровнями. '''+repeat+r''' Эти сведения не означают улучшения соответствия эксперименту и не подтверждают сходимости третьего уровня. Прежняя невязка около 13,682 Ом получена с унаследованными контактами; это другая контактная дискретизация при тех же сопротивлениях и положении.

## Численный допуск в фиксированной точке

<!-- numerical_table -->
<!-- contrast_screen_table -->
<!-- prerequisites_table -->
<!-- numerical_figure -->

**Рисунок 3. Раздельные численные изменения.** Каждый ряд относится к подписанному сравнению и точке. Круги — начальная точка, ромбы — новый угол на первоначальном сгущении, треугольники — новый угол на целевом сгущении. Цвет различает пространство, доли, суммарное изменение, контакты, увеличение числа проб и независимое перемешивание. Высота — |ΔZ|, делённое на допуск уровней. Чёрный пунктир на уровне 1 — граница допуска. При отсутствии пары завершённых контролей её ряд не строится. Каждая легенда действует только на собственный рисунок; ряды отключаются независимо.

Для пространства, долей, суммарного изменения и контактов сохранён допуск уровней 0,02 + 0,005|Z правого расчёта| Ом. Для увеличения числа проб, смены перемешивания и вложенных правил используется более строгий относительный коэффициент 0,001 при том же абсолютном слагаемом 0,02 Ом. Дополнительный допуск изменения модельного дыхательного контраста равен 0,002 + 0,01|контраст правого расчёта| Ом. Он проверяется для пространства, долей, контактов, увеличения числа проб, независимого перемешивания и обоих вложенных правил. Для суммарного пространственного и тканевого изменения сохранён отдельный контроль уровней. Эти заранее заданные численные критерии не являются погрешностью экспериментальных измерений. Контроллер также проверяет независимость двух перемешиваний по фактически использованным планам, подготовленным матрицам, долям и переводу координат в метры; до публикации свидетельства этот контроль ожидается.

Только завершённый положительный допуск по всем требуемым составляющим разрешает новый исследовательский подбор через канонический предварительный интерфейс. При отрицательном допуске контроллер останавливается перед подбором и сохраняет необходимость дальнейшего уточнения. Новый локальный численный паспорт связывает проверки пространства, долей, контактов, правил и перемешиваний с одной проверенной точкой и протоколом. Общий контракт сохраняет статус проверенной опорной геометрии; глобальный допуск к количественной инверсии не выставляется. Локальные проверки не устанавливают глобальной сеточной сходимости, достоверности тканей, независимой анатомической регистрации или пригодности модели для оценки функций сердца. При изменении параметров и положения новые численные проверки требуются отдельно.

## Все уровни и дыхательные контрасты

<!-- levels_table -->
<!-- levels_figure -->

**Рисунок 4. Все 24 уровня.** Круги и пунктир — измерения: красный ряд соответствует вдоху, сине-зелёный — выдоху. Квадраты и сплошные линии — только завершённые новые расчёты, перечисленные в легенде; цвет различает численную модель или этап подбора. Новые отсутствующие ряды не заменяются прежними. Связывающие линии сравнивают дискретные записи, без временной интерполяции между ними.

<!-- contrasts_table -->
<!-- contrasts_figure -->

**Рисунок 5. Все 12 контрастов, выдох минус вдох.** Чёрные круги — измерения, цветные квадраты — завершённые новые модельные результаты с собственными подписями. Чёрный пунктир — нулевой контраст. Значения разных размеров и осей сохранены раздельно. Отсутствие ряда означает отсутствие готового результата, без вывода о невозможности соответствующего знака в модели.

## Условный подбор и совместный поиск

<!-- fit_table -->
<!-- rho_pose_table -->
<!-- search_table -->
<!-- interrupted_search_table -->
<!-- rejected_contacts_table -->
<!-- relocated_control_status_table -->
<!-- history_figure -->

**Рисунок 6. История нового совместного поиска.** Серые круги — завершённые численные оценки, синяя линия — лучшее значение среди них к данному номеру. Красные кресты, если имеются, — геометрические отказы на условной высоте максимальной завершённой невязки; собственной RMSE им не присваивается. Начатые оценки без численного вектора не считаются готовыми кандидатами. Номер оценки не является временем или независимым экспериментом.

Технический статус отказа геометрии включает непрохождение контактного контроля на рабочей сетке. Его пороги приведены в отдельной таблице: ошибка площади менее 12 %, смещение центроида менее 1 мм и расстояние контактного узла от центра менее 3,7 мм. Такой отказ не означает доказанной анатомической невозможности установки. Отброшенное положение может потребовать собственного сгущения и повторной проверки конечной площадки. Поэтому контактные отказы, неполное покрытие областей поиска и недостигнутое разрешение ограничивают вывод о найденном центре.

При остановке процесса выбран лучший полностью сохранённый вектор; незавершённая оценка исключена из выбора. Причина остановки расчётного процесса не установлена, поэтому это событие не названо штатным достижением предела времени, числа оценок или разрешения. После срока поиска новые поисковые точки не рассчитывались. Отдельные прямые проверки продолжаются в пределах первоначального общего срока. Такой выбор не доказывает стационарности по сопротивлениям или положению. В частности, ненулевые допустимые компоненты материального градиента показывают ещё доступные локальные направления спуска.

Контактная проверка при новом угле выполнена отдельно от решения электрической задачи. Если покрытие первоначальным сгущением неполное, даже малая разность двух прежних сеток не считается достаточным контролем площадок нового положения. Дополнительное сгущение строится по всем 28 боковым и 12 ручным центрам этой точки. Параметры и координаты переносятся только как фиксированный опорный сценарий; прежние импедансы и якобиан не переносятся на новую сетку. Геометрические алгоритмы и допуски формы сохраняются, а отрицательные попытки сетки остаются в протоколе.

Первоначальное и целевое локальные сгущения построены независимо от одной родительской сетки с 860255 узлами. Целевая сетка не является вложенным продолжением сетки с 1279834 узлами: области уточнения выбраны заново по новому углу. Поэтому меньшее итоговое число узлов само по себе не означает ни потери, ни улучшения точности. Для контроля пространства сравниваются общий родитель и его целевой дочерний вариант; тканевые доли при этом наследуются точно.

После отрицательной проверки покрытия оставшиеся контроли направлены на целевую сетку. Уже завершённый родительский расчёт сохранён; начатый контроль дочерней сетки первоначального угла остановлен как заменённый, без присвоения ему численного результата. На целевой сетке вновь разделены пространство FEM, тканевые доли, контактная дискретизация, рост числа проб и независимое перемешивание. Перенесённая точка здесь проверяется прямым расчётом; дополнительный подбор на целевой сетке этим контролем не выполняется. Пройденный численный допуск не заменяет нового поиска и не делает граничные сопротивления индивидуальными измерениями.

В действующем контроллере условный подбор сопротивлений при фиксированном положении ограничен 12 вычислениями целевой функции и 8 итерациями; предусмотрен один явно выбранный старт из сохранённого начального приближения. Затем предусмотрен совместный поиск шести фазовых сопротивлений, двух смещений и угла. Для него заявлены два начальных приближения: уточнённая прежняя точка и стандартное приближение методики Ника. Они используют общий предел числа оценок и времени. Дополнительные профили невязки остаются предложением, пока не сохранены результаты их расчёта. Реальное число завершённых стартов и оценок определяется сохранёнными результатами. Положительный код завершения сам по себе не устанавливает глобального минимума. Близость к границе в таблице отмечается с абсолютным допуском 10⁻⁶ Ом·м только для подписи; параметры и технические пределы не изменяются.

Для ошибок пары eᵢ и eₑ, равных модели минус измерение соответственно на вдохе и выдохе, вводятся a=(eᵢ+eₑ)/2 и b=(eₑ−eᵢ)/2. Тогда RMSE² = среднее по 12 парам a² + среднее b². Здесь отдельно показаны ошибка среднего уровня и ошибка половины контраста. Меньшая общая RMSE не обязательно означает правильное воспроизведение дыхательного знака. После нового выбора сопротивлений или положения начальные проверки не переносятся на итоговую точку.

Число несовпадений знака служит дополнительным диагностическим показателем и не входит в целевую функцию. Амплитуды дыхательных разностей неодинаковы, поэтому уменьшение числа ошибочных знаков не обязательно минимизирует MSE. Сопоставление средних уровней, амплитуд и знаков сохраняется раздельно; новые произвольные веса для улучшения знакового показателя не вводятся.

<!-- placement_figure -->
<!-- placement_coordinates_table -->

**Рисунок 7. Расчётные центры нового выбранного положения.** Оранжевые ряды — продольные сборки, сине-зелёные — поперечные; круги — токовые контакты I, квадраты — измерительные контакты U; красный ромб — новый общий центр, серый крест — приближённый центр автора. Подсказки у точек задают I+, U+, U−, I−. Линии соединяют центры для отображения размещения; это не токовые траектории. Носители разных размеров показаны как альтернативные условия установки, а не одновременно установленные контакты. Представлены сохранённые координаты LPS в миллиметрах: x направлена влево, y назад, z вверх. Бежевая полупрозрачная поверхность, если доступна, — обзорная копия фактической наружной границы расчётной модели, включая цилиндрические руки. Только для рисунка поверхность упрощена до 12000 треугольников; расчётная граница и контактные матрицы не изменялись. Погрешность обзорного упрощения не установлена. Новая независимая регистрация КТ не выполнялась. Таблица содержит все 28 расчётных поверхностных центров; три десятичных разряда задают форму записи, а не измеренную точность установки. Камера сохраняется при переключении рядов. Пока нового выбранного положения нет, рисунок отсутствует; прежняя сцена не подставляется.

## Локальная алгебра сохранённых результатов

'''+selected_residual_summary(d)+r'''

<!-- diagnostic_split_table -->
<!-- pair_residual_contribution_table -->
<!-- connection_transition_table -->
<!-- local_sensitivity_table -->
<!-- material_bound_gradient_table -->
<!-- saved_material_slices_table -->
<!-- saved_material_slices_figure -->

**Рисунок 8. Срезы критерия по сопротивлениям из сохранённого поиска.** Шесть панелей разделяют вдох и выдох, фон, лёгкое и жир. Цветные круги — значения, полученные из завершённых FEM-оценок при том же полном положении; красный крест — сопротивление выбранной точки, чёрный пунктир — её нулевое ΔMSE. Сплошные линии только соединяют рассчитанные точки; промежуточные значения не вычислены. Ось ρ логарифмическая, единицы — Ом·м; вертикальная ось показывает изменение MSE всех 24 уровней относительно выбранной точки, в Ом². Вертикальные масштабы панелей различаются; величины сопоставляют по подписям шкал. Легенда позволяет независимо отключать фазовые материальные ряды.

Эти срезы относятся к численной модели, на которой выполнялся поиск. При фиксированном положении две фазы в данной статической постановке рассчитываются независимо. Для каждой точки среза берётся SSE рассматриваемой фазы из исходной оценки и прибавляется неизменная SSE противоположной фазы выбранного кандидата. Включаются только оценки с точно тем же полным положением и идентичностью модели, при неизменных двух остальных фазовых сопротивлениях. Параметры противоположной фазы в исходной оценке могли отличаться; её вектор при таком сложении не используется. Источники и максимальные различия повторных фазовых SSE приведены в таблице. Новые электрические задачи не решаются, а повторные численные оценки не превращаются в независимые экспериментальные измерения.

Это срезы при фиксированных остальных параметрах, без их повторной оптимизации. Профиль с переоптимизацией остальных сопротивлений и положения не рассчитан; сетка проб не задаёт доверительный интервал. Даже ухудшение критерия в единственной внутренней точке у верхнего предела не устанавливает определённость граничного сопротивления. Срезы и допустимый градиент помогают отличить крупный поисковый шаг от достигнутой стационарности; отсутствие более узких проб сохраняется как ограничение. На целевую сетку эти срезы автоматически не переносятся.

Каждое свидетельство связано контрольной суммой с собственным файлом кандидата, неизменными измерениями и паспортом входов. Прежняя точка с унаследованными контактами, новый условный подбор и новый совместный кандидат показаны раздельно. Отсутствующая новая диагностика не заполняется прежним якобианом или прежними сопротивлениями. Основные диагностические свидетельства рассчитаны отдельным алгебраическим модулем. Дополнительно отчёт суммирует сохранённые фазовые ошибки для показанных срезов; новых прямых расчётов и линейных оптимизаций он не выполняет.

При переходах 004 → 006, 005 → 006 и 010 → 011 сопоставлена одна номинальная боковая сборка с общим центром и одинаково названной дыхательной фазой. Разность «после минус до» является межзаписной и отличается от дыхательной разности внутри записи. Меняется соседний монтаж, но одновременно проходят время и повторные дыхательные пробы. Сравнение показывает, насколько сохранённая модельная точка воспроизводит эти переходы при принятом объединении записей; само по себе оно не устанавливает межканальное влияние, изменение контакта или физиологическую причину. Эти условия не являются независимыми контролируемыми повторами.

Материальный якобиан J содержит производные уровней импеданса по шести фазовым логарифмам сопротивлений при сохранённом положении. Фазовые нормы его трёх столбцов приведены в порядке фон, лёгкое, жир; единица — Ом на единичное изменение логарифма сопротивления. Косинусы характеризуют сходство этих столбцов, а обусловленность — отношение крайних сингулярных значений фазовой матрицы. Градиент MSE равен 2Jᵀe/24, где e — модель минус измерение. Все характеристики локальны; они не устанавливают глобальной идентифицируемости или стационарности ещё не завершённого поиска.

Строки целевой сетки вычислены отдельно из якобиана её завершённого прямого контроля. Параметры и положение при этом сохраняются; новый подбор не выполняется. Эти строки позволяют проверить, сохраняется ли слабая чувствительность к лёгкому после уточнения сетки, долей и контактных площадок. Прежние производные на новую сетку не переносятся.

Допуски сходимости заданы для импедансов и дыхательных контрастов, без автоматического переноса на сопротивления. При слабой чувствительности к лёгкому небольшое численное изменение прямого отклика может быть существенным для его обратной оценки. Поэтому положительный допуск по Z ещё не подтверждает устойчивости ρ₂. Повторная оптимизация на сравниваемых сетках и интеграционных правилах и профили сопротивлений с переоптимизацией остальных параметров здесь не выполнены. Численная устойчивость восстановленных сопротивлений остаётся отдельным нерешённым вопросом.

В таблице активных пределов используется вычислительная отметка |Δlnρ|≤10⁻⁴. При нижнем пределе положительная компонента градиента направляла бы спуск за допустимую область; при верхнем пределе то же относится к отрицательной компоненте. Только эти компоненты заменены нулём в столбце допустимого градиента. Это локальная алгебраическая проверка направления спуска, без новой оптимизации. Её порог не задаёт неопределённость сопротивления. Шесть материальных столбцов не проверяют идентифицируемость всех девяти параметров совместного подбора: производные по положению и ориентации здесь отсутствуют. Значение у предела нельзя считать определённым индивидуальным сопротивлением без отдельного исследования профиля невязки.

<!-- linear_projection_table -->
<!-- linear_contrasts_table -->

Линейная диагностика использует исходный модельный контраст и его поправку Aη, где строка A равна разности соответствующих строк J на выдохе и вдохе, а η содержит шесть независимых фазовых шагов Δlnρ. Сохранённая линейная задача приближает измеренные 12 контрастов при ограничении итоговых логарифмов исходными техническими границами 0,1–100 Ом·м. Это другая задача, чем подбор всех 24 абсолютных уровней: средние уровни здесь не ограничены. Применимость линеаризации для найденного шага не проверена. Даже ограниченный границами широкий шаг не трактуется как физическое изменение тканей, новый нелинейный минимум или доказательство невозможности дыхательного знака.

<!-- source_scenarios_table -->

Сценарии источников используют сохранённые собственный и встречный отклики при тех же сопротивлениях и положении, без повторного подбора. При относительной фазе φ=0 берётся их знаковая сумма, при φ=π — знаковая разность, при φ=π/2 — модуль квадратурной суммы. Сценарии двух источников предполагают равные амплитуды; эти амплитуды, относительная фаза и способ регистрации экспериментально не подтверждены. Сценарная RMSE и число несовпадений знака описывают вычисленную гипотезу и не выбирают реальную схему прибора.

## Диагностика прежней точки и нерешённые гипотезы

<!-- historical_diagnostics_table -->
<!-- hypotheses_table -->

Этот раздел относится только к ранее сохранённой исправленной точке с унаследованными контактами и RMSE около 13,682 Ом, перенесённой как начальное приближение. Исторические производные относятся к этой контактной дискретизации, без переноса на новый прямой расчёт с RMSE около 13,796 Ом. Исторический разбор сам по себе не определяет свойства нового кандидата; его собственная диагностика показана отдельно. В прежней точке пары записи 009, канал 1, и записи 011, канал 2, поперечная ось, давали около 86,5 % SSE; контрастная доля SSE была около 0,44 %. Точные значения берутся из сохранённого независимого разбора. Это математическая структура ошибки в одной точке, без доказательства причины экспериментального расхождения.

Положительные компоненты градиента MSE по логарифму сопротивления фона в обеих фазах, при внутренних фоновых параметрах, предсказывают локальное уменьшение критерия при их снижении. Поэтому прежнюю точку нельзя называть установленным минимумом по материалам. Локальный якобиан и формальная линейная проекция контраста не доказывают глобальной нелинейной несовместимости или неидентифицируемости постановки. Широкий логарифмический шаг за пределы 0,1–100 Ом·м не интерпретируется как физическое решение.

Разброс внутриоконных блоков не является готовой стандартной ошибкой медианы, ковариацией измерений или набором независимых повторов. Переключения соседних подключений смешаны с последовательной регистрацией; их причины не разделены. Ковариация, доверительные интервалы и статистическая устойчивость почти нулевых контрастов здесь не оценены. Новые p-значения и произвольный уровень шума не вводятся. Сценарии токовой фазы при прежних сопротивлениях остаются отдельной диагностикой; меньшая сценарная RMSE не подтверждает реальную схему прибора.

## Происхождение и пределы вывода

**Предлагаемый следующий расчётный этап, не результат этого прогона.** Продолжение совместного поиска должно учитывать контактный контроль каждой геометрической пробы. При фиксированном положении следует доводить подбор материалов с проверенными производными, затем сравнивать положения по полученным значениям критерия. Это позволит отдельно проверить влияние крупного координатного шага и слабой чувствительности к лёгкому. Потребуются профили с переоптимизацией остальных параметров и повторение подбора на сравниваемых численных моделях; один срез или малая ошибка прямого расчёта их не заменяет.

Если после этих проверок большая невязка и различия дыхательного знака сохранятся, следующими разделяющими проверками остаются изменение анатомической геометрии при дыхании, схема совместного возбуждения и сопоставимость последовательных записей. Пока эти причины не разделены, граничные сопротивления не передаются как подтверждённые поправки для ТТРКГ и не используются для количественного вывода УО или ФВ.

<!-- provenance_table -->

Отчёт читает сохранённые результаты и проверяет точные байты малых паспортов, таблиц, кода и планов. Крупные сетки, маски, доли и подготовленные матрицы связаны по записанным идентичностям; их повторное хеширование в читательской сборке не выполняется и не заявляется. Исполненный исторический код принимается только при точном совпадении архивной контрольной суммы. Источники каждого нового результата сохраняются отдельно от прежней диагностики. Положительный геометрический или численный контроль не заменяет независимой физической, экспериментальной или клинической валидации. Общий указатель принятой модели этой сборкой не изменяется.
'''


def build(work=WORK, out=FOLDER):
    import nbformat as nbf
    from nbconvert import HTMLExporter
    d=load(work);reader=text(d);names=list(dict.fromkeys(re.findall(r'<!-- (\w+) -->',reader)))
    if any(not callable(globals().get(name)) for name in names):raise ValueError('Unknown reader function')
    relative=d['work'].relative_to(PIPE).as_posix();cells=[];markdown=[];table_index=0;figure_index=0;js=True
    setup='from build_georg_convergence_report import load, PIPE, numbered_table, '+', '.join(names)+f'\nreport_data=load(PIPE / {relative!r})'
    cells.append(nbf.v4.new_code_cell(setup,execution_count=1,metadata=dict(hide_input=True)))
    for piece in re.split(r'(<!-- \w+ -->)',reader):
        if piece.startswith('<!-- '):
            key=piece[5:-4].strip();function=globals()[key]
            if key.endswith('table'):
                table_index+=1;value=numbered_table(function(d),table_index)
                code=f'display(numbered_table({key}(report_data),{table_index}))';markdown.append(value.data)
            else:
                include_js=js;value=function(d,include_js=include_js)
                if 'plotly' in value.data.lower():js=False
                figure_index+=1;anchor='figure-'+str(figure_index)
                value=HTML('<div id="'+anchor+'"></div>'+value.data)
                code=f'display({key}(report_data,include_js={include_js!r}))'
                markdown.append('[Открыть интерактивный рисунок]('+STEM+'.html#'+anchor+')' if 'plotly' in value.data.lower() else value.data)
            cell=nbf.v4.new_code_cell('from IPython.display import display\n'+code,
                execution_count=1+sum(c.cell_type=='code' for c in cells),metadata=dict(hide_input=True))
            cell.outputs=[nbf.v4.new_output('display_data',data={'text/html':value.data})];cells.append(cell)
        elif piece.strip():cells.append(nbf.v4.new_markdown_cell(piece.strip()));markdown.append(piece.strip())
    snap=d['snapshot'];metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        execution_mode='saved results reader; no FEM',report_as_of_utc=d['as_of'],quantitative_result_accepted=False,
        global_mesh_convergence_proven=False,reader_language_review='pending_final_full_model_pass',
        initial_gate_passed=d['gate'].get('passed') if d['gate'] else None,
        effective_run_configuration_path=str(d['configuration_file']),
        effective_run_configuration_sha256=snap.files[str(d['configuration_file'].resolve())],
        original_run_configuration_sha256=snap.files[str((d['work']/'run_config.local.json').resolve())],
        new_candidate_available=d['best'] is not None,raw_large_artifact_hashes_recomputed=False,
        residual_diagnostics_available={key:value is not None for key,value in d['residual_diagnostics'].items()},
        actual_initial_contact_coverage_available=d['coverage'] is not None,
        actual_fine_contact_coverage_available={key:value is not None for key,value in d['fine_coverage'].items()},
        relocated_mesh_quality_passed=d['relocated_quality']['passed'] if d['relocated_quality'] else None,
        relocated_selected_numerical_gate_passed=d['relocated_audit']['all_checks_passed'] if d['relocated_audit'] else None,
        relocated_contact_coverage_available=d['relocated_coverage'] is not None,
        selected_search_interrupted=bool(d['selected'] and d['selected'].get('search_interrupted')),
        inverse_parameter_identification_accepted=False,
        frozen_initial_mesh_sections_available=d['mesh_sections'] is not None,
        reader_inputs_sha256=snap.files,exact_source_guards=list(snap.guard_records.values()),
        recorded_large_artifact_identities=list(snap.deferred.values()),generator_sha256=file_sha(__file__),
        helper_sha256={name:file_sha(FOLDER/name) for name in ('build_georg_quality_repeat_report.py','build_georg_contact_continuation_report.py')})
    nb=nbf.v4.new_notebook(cells=cells,metadata=metadata);nbf.validate(nb)
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    rendered,_=exporter.from_notebook_node(nb)
    rendered=rendered.replace('</head>','<style>details{margin:1em 0}summary{cursor:pointer;font-weight:600}.scientific-table{overflow-x:auto}</style></head>')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=[out/(STEM+suffix) for suffix in ('.ipynb','.html','.md')]
    with files[0].open('w',encoding='utf-8',newline='\n') as stream:nbf.write(nb,stream)
    files[1].write_text(rendered.replace('\r\n','\n'),encoding='utf-8',newline='\n')
    files[2].write_text(('\n\n'.join(markdown)+'\n').replace('\r\n','\n'),encoding='utf-8',newline='\n')
    return files


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,default=WORK)
    parser.add_argument('--out',type=Path,default=FOLDER);args=parser.parse_args()
    for path in build(args.work,args.out):print(path)
