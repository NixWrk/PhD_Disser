"""Read-only report of all-contact refinement and continued Georg inversion."""
from pathlib import Path
import copy,json,re
import numpy as np
from build_georg_quality_repeat_report import (
    PIPE,PREPARATION,sha,read,number,table,figure_html,mesh_table as previous_mesh_table,
    fit_figure,history_figure,placement_figure,sensitivity_figure,
    coarse_section_figure,fine_section_figure,numerical_figure,
    residual_table as previous_residual_table,coordinate_table as previous_coordinate_table)
FOLDER=Path(__file__).resolve().parent
HISTORY_WORK=PIPE/'output/exploratory/georg_contact_continuation_20261014_v2'
WORK=PIPE/'output/exploratory/georg_confirmed_arm_continuation_20261014'
W3_CONFIG='run_config_priority.local.json'
W3_CONTACT_CHECK='contact_check_priority_v1'
STEM='33.30_Сгущение_у_всех_контактов_и_продолжение_подбора_Георгия_Scorpion_v2'


def verify_executed_source(path,digest,config):
    path=Path(path)
    if path.exists() and sha(path)==digest:return
    if path.suffix.lower() not in ('.m','.py'):
        raise ValueError('Displayed numerical/medical input changed: '+str(path))
    try:relative=path.resolve().relative_to(PIPE).as_posix()
    except ValueError:relative=None
    archived=[]
    for row in config['sources']:
        candidate=Path(row['path'])
        if not candidate.is_absolute():candidate=PIPE/candidate
        if candidate.resolve()==path.resolve() and row['sha256']==digest and 'snapshot' in row:
            archived.append(PIPE/row['snapshot'])
    if relative and (relative.startswith('src/') or relative=='run_pipeline.m'):
        archived.append(PIPE/'archive/executed_georg_continuation_20261014'/relative)
        if relative=='src/run_georg_lateral_inverse.m':
            archived.append(PIPE/'archive/executed_georg_continuation_20261014/before_priority'/relative)
    if relative in ('src/run_georg_lateral_inverse.m','tools/georg_lateral_inverse_prepare.py',
            'src/run_ct_fem_contacts_prepare.m'):
        archived.append(PIPE/'archive/executed_georg_continuation_20261014/priority_executed'/relative)
    if not any(p.exists() and sha(p)==digest for p in archived):
        raise ValueError('Executed source changed without its recorded byte snapshot: '+str(path))


def source_path(value):
    path=Path(value)
    return path if path.is_absolute() else PIPE/path


def guard_sources(rows,config):
    for row in rows:
        verify_executed_source(source_path(row['path']),row['sha256'],config)


def priority_contract(work,cfg):
    """Bind the explicit search-order change to preserved code and unchanged controls."""
    work=Path(work);prior=read(work/'run_config_joint_only.local.json')
    change_file=work/'explicit_parameter_order_change.local.json';change=read(change_file)
    order=cfg['parameter_evaluation_order']
    if order!=[7,8,9,1,4,2,5,3,6]:raise ValueError('Unexpected geometry-priority parameter order')
    if not (change['default_order_remains_1_to_9'] and change['original_deadlines_retained'] and
            all(change[k] is False for k in ('physical_kernels_changed','parameter_bounds_changed',
                'observation_values_changed','old_predictions_reused'))):
        raise ValueError('Priority change exceeded its declared search-order scope')
    for key in ('work_directory','control_models','initial_control_files','initial_reference_file',
            'initial_audit_file','input_directory','deadline_unix','search_deadline_unix',
            'additional_corrected_search_allocation','conditional_profile_omitted_for_original_time_budget'):
        if cfg[key]!=prior[key]:raise ValueError('Priority handoff changed completed input/control contract: '+key)
    inverse=(PIPE/'src/run_georg_lateral_inverse.m').resolve()
    changes=change['changes']
    if len(changes)!=1 or source_path(changes[0]['path']).resolve()!=inverse:
        raise ValueError('Expected one explicitly repinned inverse source')
    row=changes[0];archive=source_path(row['executed_previous_snapshot'])
    expected_archive=PIPE/'archive/executed_georg_continuation_20261014/before_priority/src/run_georg_lateral_inverse.m'
    if archive.resolve()!=expected_archive.resolve() or not archive.exists() or sha(archive)!=row['previous_sha256']:
        raise ValueError('The exact pre-priority inverse bytes are not preserved')
    old_sources={(source_path(r['path']).resolve(),r['sha256']) for r in prior['sources']}
    new_sources={(source_path(r['path']).resolve(),r['sha256']) for r in cfg['sources']}
    if ((inverse,row['previous_sha256']) not in old_sources or
            (inverse,row['new_sha256']) not in new_sources or
            not (old_sources-{(inverse,row['previous_sha256'])})<=new_sources):
        raise ValueError('Priority configuration changed another previously pinned dependency')
    old_paths={path for path,_ in old_sources}
    if any(path!=inverse and path in old_paths and (path,digest) not in old_sources for path,digest in new_sources):
        raise ValueError('Priority configuration repinned another existing source')
    if {digest for path,digest in new_sources if path==inverse}!={row['new_sha256']}:
        raise ValueError('Priority inverse source identity is ambiguous')
    required=[(archive.resolve(),row['previous_sha256']),(change_file.resolve(),sha(change_file))]
    if not set(required)<=new_sources:raise ValueError('Priority change and previous bytes are not pinned')
    verify_executed_source(inverse,row['new_sha256'],cfg)
    replaced=source_path(cfg['replaced_search_directory']).resolve()
    if replaced!=(work/'inputs'/('joint_own_drive_'+prior['run_label'])).resolve():
        raise ValueError('Replaced-search provenance belongs to another run')
    old_rows=[read(p) for p in sorted(replaced.glob('evaluation_*.local.json'))]
    started=max((r['evaluation'] for r in old_rows),default=0)
    if started!=cfg['replaced_search_started_evaluations']:
        raise ValueError('Replaced-search started count differs from its saved checkpoints')
    if cfg['max_evaluations']!=max(16,prior['max_evaluations']-started):
        raise ValueError('Priority allocation differs from its explicit controller rule')
    audit=read(work/cfg['initial_audit_file'])
    if not audit['complete']:raise ValueError('Priority handoff requires the completed initial audit')
    pinned={source_path(r['path']).resolve():r['sha256'] for r in audit['sources']}
    for name in cfg['initial_control_files']:
        file=(work/name).resolve()
        if pinned.get(file)!=sha(file):raise ValueError('Priority handoff changed an initial control')
    return dict(change=change,replaced_rows=old_rows,prior_config=prior)


def priority_budget_rows(d):
    cfg=d['config'];root=source_path(cfg['replaced_search_directory'])
    rows=[read(p) for p in root.glob('evaluation_*.local.json')]
    return [
        ['Исправленный расчёт: первоначальная отдельная квота',cfg['additional_corrected_search_allocation']],
        ['Исправленный расчёт: начатые оценки заменённой попытки совместного поиска без предварительного профиля',cfg['replaced_search_started_evaluations']],
        ['Исправленный расчёт: готовые численные промежуточные результаты заменённого запуска',sum(r['status']=='numerical_candidate_not_validated' for r in rows)],
        ['Исправленный расчёт: незавершённые оценки заменённого запуска',sum(r['status']=='forward_calculation_running' for r in rows)],
        ['Исправленный расчёт: предел нового поиска с приоритетом геометрии',cfg['max_evaluations']],
        ['Исправленный расчёт: порядок проверки параметров (исходные индексы 1–9)',', '.join(map(str,cfg['parameter_evaluation_order']))],
        ['Исправленный расчёт: фиксированный профиль','Пропущен до выполнения из-за исходного лимита времени']]


def confirmed_inputs(work=WORK,history=HISTORY_WORK):
    """Small input-contract review; does not require or run any numerical stage."""
    import pandas as pd
    work,history=map(Path,(work,history))
    cfg=read(work/W3_CONFIG);priority=priority_contract(work,cfg)
    if not (cfg.get('confirmed_arm_correction') and
            cfg.get('conditional_profile_omitted_for_original_time_budget') is True):
        raise ValueError('Expected corrected priority run configuration without a fixed profile')
    if (PIPE/cfg['work_directory']).resolve()!=work.resolve():
        raise ValueError('Corrected configuration belongs to another work directory')
    passport=read(work/'inputs/inputs.local.json');old_passport=read(history/'inputs/inputs.local.json')
    confirmation_file=work/'author_arm_confirmation_20261014.local.json';confirmation=read(confirmation_file)
    old_csv=history/'inputs/observations.local.csv';new_csv=work/'inputs/observations.local.csv'
    if confirmation['prior_observations_sha256']!=sha(old_csv):
        raise ValueError('Boundary correction belongs to another historical observation file')
    if passport['previous_input_contract_sha256']!=sha(history/'inputs/inputs.local.json'):
        raise ValueError('Corrected passport belongs to another historical passport')
    if passport['outputs_sha256']['observations.local.csv']!=sha(new_csv):
        raise ValueError('Corrected observation identity changed')
    if not (confirmation['cross_006_011_all_arm_wires_removed'] and
            confirmation['late_ttrkg_007_010']=='1x1' and
            confirmation['physical_arm_pads_remain_an_assumption'] and
            not confirmation['numerical_observation_values_changed'] and
            not confirmation['channels_axes_windows_changed']):
        raise ValueError('Unexpected human boundary-confirmation contract')
    if not (passport['corrected_boundary_conditions'] and passport['late_arm_montage_confirmed'] and
            passport['late_arm_montage_scenario']=='1x1' and passport['late_arm_montage_alternatives']==[]):
        raise ValueError('Corrected passport is inconsistent with the human confirmation')
    if passport['search_bounds_ohm_m']!=old_passport['search_bounds_ohm_m']:
        raise ValueError('Boundary correction changed the technical material bounds')
    # Compare literal CSV fields, including sample windows/order and all numeric strings.
    old=pd.read_csv(old_csv,dtype=str,keep_default_na=False)
    new=pd.read_csv(new_csv,dtype=str,keep_default_na=False)
    if list(old)!=list(new) or len(old)!=len(new) or not old.drop(columns='arm_montage').equals(new.drop(columns='arm_montage')):
        raise ValueError('Boundary correction changed levels, roles, windows or observation order')
    changes=[]
    for a,b in zip(old.to_dict('records'),new.to_dict('records'),strict=True):
        late=7<=int(a['record_id'].rsplit('-',1)[1])<=10
        if late:
            if (a['arm_montage'],b['arm_montage'])!=('1x4','1x1'):
                raise ValueError('Expected only the declared 007-010 1x4 to 1x1 correction')
            changes.append(dict(record_id=b['record_id'],channel=b['channel'],state=b['state'],
                previous_arm_montage=a['arm_montage'],arm_montage=b['arm_montage']))
        elif a['arm_montage']!=b['arm_montage']:
            raise ValueError('An undeclared arm montage changed')
    if changes!=confirmation['changed_rows']:
        raise ValueError('Declared boundary changes differ from the actual CSV changes')
    seed=read(work/'warm_seed.local.json');reference=read(work/cfg['initial_reference_file'])
    if seed['source_candidate_sha256']!=sha(history/'selected_candidate.local.json'):
        raise ValueError('Corrected initial guess came from another historical candidate')
    for row in (seed,reference):
        identity=row.get('identity',row)
        if identity['observations_sha256']!=sha(new_csv) or row['boundary_correction_sha256']!=sha(confirmation_file):
            raise ValueError('Initial guess/reference has another boundary or observation identity')
        if identity['author_centre_sha256']!=sha(work/'inputs/author_centre_20261012.local.json'):
            raise ValueError('Corrected author-centre identity changed')
    old_best=read(history/'selected_candidate.local.json')['best']
    if any((row['parameters'],row['pose'])!=(old_best['parameters'],old_best['pose']) for row in (seed,reference)):
        raise ValueError('The isolated BC comparison requires the unchanged historical initial point')
    if 'best' in seed or 'predicted_Z_ohm' in seed:
        raise ValueError('Historical predictions must not be transferred to the corrected run')
    return dict(config=cfg,passport=passport,confirmation=confirmation,warm_seed=seed,priority=priority)


def validate_observation_rows(row,observations):
    if len(observations)!=24:
        raise ValueError('Exactly 24 hold observations are required')
    for field,column in [('record_ids','record_id'),('axes','axis'),('states','state')]:
        if row[field]!=observations[column].tolist():
            raise ValueError('Saved numerical observation order changed: '+field)
    np.testing.assert_allclose(row['target_Z_ohm'],observations.Z_ohm.to_numpy(),rtol=0,atol=1e-12)


def validate_forward(row,reference,observations,observation_sha,reference_sha,model):
    validate_observation_rows(row,observations)
    if (row['parameters'],row['pose'])!=(reference['parameters'],reference['pose']):
        raise ValueError('A fixed numerical control uses another parameter vector or pose')
    identity=row['identity']
    if identity['observations_sha256']!=observation_sha or identity['reference_sha256']!=reference_sha:
        raise ValueError('A fixed numerical control uses another observation/reference identity')
    if identity['plan_sha256']!=sha(model/'plan.local.json'):
        raise ValueError('A fixed numerical control belongs to another model plan')
    if np.asarray(row['fine_Z_ohm']).shape!=(24,) or not np.isfinite(row['fine_Z_ohm']).all():
        raise ValueError('Expected 24 finite numerical levels')


def selection_control_reuse(work,cfg,best):
    """Authorize original reference identities only through the exact-point receipt."""
    file=work/'selection_controls_reuse.local.json'
    if not file.exists():return None
    receipt=read(file)
    if (receipt['schema']!='exact_selected_point_control_reuse_v1' or receipt['FEM_rerun'] is not False or
            not all(receipt[k] is True for k in ('parameters_and_complete_pose_equal_exactly',
                'targets_equal_exactly','original_controls_unchanged','negative_numerical_results_preserved'))):
        raise ValueError('Invalid exact-point control-reuse contract')
    if receipt['selected_candidate_sha256']!=sha(work/'selected_candidate.local.json'):
        raise ValueError('Reuse receipt belongs to another selected candidate')
    reference_file=work/cfg['initial_reference_file'];reference=read(reference_file)
    if receipt['reference_initial_sha256']!=sha(reference_file):raise ValueError('Original reuse reference changed')
    if best is None or any(reference[k]!=best[k] for k in ('parameters','pose')):
        raise ValueError('Reuse requires exact parameters and the WHOLE original pose')
    verify_executed_source(PIPE/'tools/georg_confirmed_same_point_finalize.py',receipt['producer_sha256'],cfg)
    sources=[work/p for p in cfg['initial_control_files']]
    aliases=[work/(p+'_selected_v1.local.json') for p in ('coarse_forward','fine_inherited','fine_remapped')]
    if len(receipt['copies'])!=3:raise ValueError('Expected all three fixed-point control aliases')
    for record,source,alias in zip(receipt['copies'],sources,aliases,strict=True):
        if source_path(record['source']).resolve()!=source.resolve() or source_path(record['selection_alias']).resolve()!=alias.resolve():
            raise ValueError('Unexpected initial source or selected alias in reuse receipt')
        if sha(source)!=record['source_sha256'] or sha(alias)!=record['alias_sha256'] or source.read_bytes()!=alias.read_bytes():
            raise ValueError('An initial control/alias is not the preserved byte-for-byte copy')
        row=read(source)
        if any(row[k]!=best[k] for k in ('parameters','pose','target_Z_ohm')):
            raise ValueError('Reused control has another exact point or target vector')
        if row['identity']['reference_sha256']!=receipt['reference_initial_sha256']:
            raise ValueError('Reused control lost its original reference identity')
    difference=float(np.max(np.abs(np.asarray(read(sources[-1])['fine_Z_ohm'])-np.asarray(best['predicted_Z_ohm']))))
    if (receipt['allowed_replay_difference_ohm']!=2e-8 or not np.isfinite(difference) or difference>2e-8 or
            difference!=receipt['inverse_fixed_forward_maximum_difference_ohm']):
        raise ValueError('Inverse/fixed-forward agreement does not match the exact reuse receipt')
    initial=read(work/cfg['initial_audit_file']);final=read(work/'numerical_audit_selected_v1.local.json')
    if (final.get('completed_controls_reused_at_exactly_identical_point') is not True or
            final['original_initial_audit_sha256']!=sha(work/cfg['initial_audit_file']) or
            final['selected_candidate_sha256']!=receipt['selected_candidate_sha256']):
        raise ValueError('Reused final audit lacks its selected/initial binding')
    if any(final[k]!=v for k,v in initial.items() if k!='sources'):
        raise ValueError('A numerical result or operator identity changed during exact-point reuse')
    original_sources={(source_path(r['path']).resolve(),r['sha256']) for r in initial['sources']}
    final_sources={(source_path(r['path']).resolve(),r['sha256']) for r in final['sources']}
    expected=(original_sources-{(p.resolve(),sha(p)) for p in sources})|{(p.resolve(),sha(p)) for p in aliases}
    if final_sources!=expected:raise ValueError('Reused audit changed another executed dependency')
    return receipt


def coverage_mesh_hashes(models):
    return dict(coarse_mesh_sha256=sha(models[0]/'mesh.mat'),
        parent_maps_sha256=sha(models[-1]/'parent_maps.mat'))


def validate_contact_coverage(coverage,forward_file,mesh_hashes,cfg):
    if coverage['coarse_forward_sha256']!=sha(forward_file) or any(coverage[k]!=v for k,v in mesh_hashes.items()):
        raise ValueError('Contact coverage uses another forward/mesh/parent map')
    verify_executed_source(PIPE/'tools/georg_quality_contact_coverage.py',coverage['producer_sha256'],cfg)
    forward=read(forward_file)
    expected=[(r['record_id'],k+1,group) for r in forward['patches'] for k,group in enumerate(r['gids'])]
    actual=[(r['record_id'],r['physical_contact'],r['electrical_group']) for r in coverage['contacts']]
    if actual!=expected or not actual:raise ValueError('Coverage does not address all actual physical CEM patches')
    fractions=np.asarray([r['refined_boundary_area_fraction'] for r in coverage['contacts']])
    if not np.isfinite(fractions).all() or np.any((fractions<0)|(fractions>1)):
        raise ValueError('Invalid actual-contact refined area fraction')
    if any(not np.isfinite(r['area_mm2']) or r['area_mm2']<=0 for r in coverage['contacts']):
        raise ValueError('Coverage contains an invalid finite contact area')
    if (coverage['all_actual_contact_faces_refined']!=all(r['all_faces_refined'] for r in coverage['contacts']) or
            not np.isclose(coverage['minimum_refined_boundary_area_fraction'],fractions.min(),rtol=0,atol=1e-12)):
        raise ValueError('Coverage summary differs from the actual patch rows')


def initial_contact_coverage(work,models,cfg,mesh_hashes=None):
    file=work/'contact_coverage_initial.local.json'
    if not file.exists():return None
    coverage=read(file)
    validate_contact_coverage(coverage,work/cfg['initial_control_files'][0],
        coverage_mesh_hashes(models) if mesh_hashes is None else mesh_hashes,cfg)
    return coverage


def validate_audit(audit,files,forwards,models,observations,cfg,strict_paths):
    validate_observation_rows(dict(audit,target_Z_ohm=forwards[0]['target_Z_ohm']),observations)
    if not audit['complete'] or list(map(lambda x:str(Path(x).resolve()),audit['model_directories']))!=list(map(str,models)):
        raise ValueError('Numerical audit is incomplete or uses another model chain')
    guard_sources(audit['sources'],cfg)
    pinned={source_path(r['path']).resolve():r['sha256'] for r in audit['sources']}
    for file in files:
        if (strict_paths and pinned.get(file.resolve())!=sha(file)) or (not strict_paths and sha(file) not in pinned.values()):
            raise ValueError('Numerical audit is not bound to the displayed saved control')
    for field,key in [('model_plan_sha256','plan_sha256'),('prepared_sha256','prepared_sha256'),
            ('physical_contacts_sha256','physical_contacts_sha256'),
            ('physical_montage_helper_sha256','physical_montage_helper_sha256')]:
        if audit[field]!=forwards[2]['identity'][key]:
            raise ValueError('Numerical audit and remapped control identity differ: '+field)
    for key,left,right in [('spatial',0,1),('fractions',1,2),('total',0,2)]:
        a=np.asarray(forwards[left]['fine_Z_ohm']);b=np.asarray(forwards[right]['fine_Z_ohm']);r=audit[key]
        np.testing.assert_allclose(r['signed_difference_ohm'],b-a,rtol=0,atol=1e-12)
        np.testing.assert_allclose(r['difference_ohm'],abs(b-a),rtol=0,atol=1e-12)
        np.testing.assert_allclose(r['tolerance_ohm'],.02+.005*abs(b),rtol=0,atol=1e-12)


def load_trial(work,confirmed=False):
    import pandas as pd
    work=Path(work);state=read(work/'run_state.local.json')
    if state['status']!='complete_with_explicit_numerical_results':
        raise ValueError('Completed W3 search and all selected-candidate controls are required' if confirmed else 'Historical search is incomplete')
    if confirmed and ('all_final_contact_faces_refined' not in state or not (work/'contact_coverage_final.local.json').exists()):
        raise ValueError('W3 final contact-coverage publication is not complete')
    cfg=read(work/(W3_CONFIG if confirmed else 'run_config.local.json'))
    bootstrap=read(work/'run_config.initial.local.json');guard_sources(cfg['sources'],cfg)
    models=[(work/p).resolve() for p in cfg['control_models']]
    observations=pd.read_csv(work/'inputs/observations.local.csv')
    observations=observations[observations.state!='quiet_breathing'].reset_index(drop=True)
    selected=read(work/'selected_candidate.local.json');best=selected['best']
    expected_inverse=next(r['sha256'] for r in cfg['sources'] if source_path(r['path']).resolve()==(PIPE/'src/run_georg_lateral_inverse.m').resolve())
    if best['identity']['worker_sha256']!=expected_inverse:
        raise ValueError('Selected candidate belongs to another executed inverse source')
    contract_file=work/'candidate_model_contract.local.json';contract=read(contract_file)
    if best['identity']['model_contract_sha256']!=sha(contract_file):
        raise ValueError('Selected candidate uses another model contract')
    guard_sources(contract['files'],cfg)
    observation_sha=sha(work/'inputs/observations.local.csv')
    if best['identity']['observations_sha256']!=observation_sha or best['identity']['input_sha256']!=sha(work/'inputs/inputs.local.json'):
        raise ValueError('Selected candidate uses another observation/passport identity')
    np.testing.assert_allclose(best['target_Z_ohm'],observations.Z_ohm.to_numpy(),rtol=0,atol=1e-12)
    run=work/'inputs'/('joint_own_drive_'+cfg['run_label'])
    if 'search' not in selected:
        if not selected.get('search_stopped_for_time'):raise ValueError('Missing search-stop provenance')
        # Reader-only summary; never rewrite the executed checkpoint.
        rows=[read(p) for p in run.glob('evaluation_*.local.json')]
        completed=[r for r in rows if r['status'] in ('numerical_candidate_not_validated','invalid_geometry')]
        if not completed:raise ValueError('No completed search evaluations')
        selected=copy.deepcopy(selected)
        selected['search']=dict(evaluations=max(r['evaluation'] for r in completed),resolution_reached=False,
            completed_sweeps=None,search_stopped_for_time=True)
    seed=read(work/'warm_seed.local.json')
    initial_files=[work/p for p in cfg['initial_control_files']]
    final_files=[work/(p+'_selected_v1.local.json') for p in ('coarse_forward','fine_inherited','fine_remapped')]
    initial_forwards=[read(p) for p in initial_files];final_forwards=[read(p) for p in final_files]
    reuse=selection_control_reuse(work,cfg,best) if confirmed else None
    if confirmed and state.get('completed_controls_reused_at_exactly_identical_point',False)!=bool(reuse):
        raise ValueError('Completion state and exact-point reuse receipt differ')
    initial_reference_file=work/cfg.get('initial_reference_file','reference_author_v2.local.json')
    # Historical initial files are copied controls from W1 with their original reference SHA.
    initial_reference=read(initial_reference_file) if confirmed else initial_forwards[0]
    for files,forwards,reference,reference_sha in [
            (initial_files,initial_forwards,initial_reference,sha(initial_reference_file) if confirmed else initial_forwards[0]['identity']['reference_sha256']),
            (final_files,final_forwards,best,reuse['reference_initial_sha256'] if reuse else sha(work/'reference_selected_v1.local.json'))]:
        for row,model in zip(forwards,models,strict=True):
            validate_forward(row,reference,observations,observation_sha,reference_sha,model)
    final_reference=read(work/'reference_selected_v1.local.json')
    selected_sha=sha(work/'selected_candidate.local.json')
    if final_reference['selected_candidate_sha256']!=selected_sha or (final_reference['parameters'],final_reference['pose'])!=(best['parameters'],best['pose']):
        raise ValueError('Final reference is not bound to the selected candidate')
    initial=read(work/cfg['initial_audit_file']);final=read(work/'numerical_audit_selected_v1.local.json')
    if final['selected_candidate_sha256']!=selected_sha:raise ValueError('Final audit belongs to another candidate')
    for audit,files,forwards in [(initial,initial_files,initial_forwards),(final,final_files,final_forwards)]:
        validate_audit(audit,files,forwards,models,observations,cfg,confirmed)
    for key in ('prepared_sha256','model_plan_sha256'):
        if best['identity'][key]!=final[key]:raise ValueError('Candidate and final control use another material model')
    coverage=read(work/'contact_coverage_final.local.json')
    mesh_hashes=coverage_mesh_hashes(models)
    validate_contact_coverage(coverage,final_files[0],mesh_hashes,cfg)
    initial_coverage=initial_contact_coverage(work,models,cfg,mesh_hashes) if confirmed else read(work/'coverage_author_selected.local.json')
    if confirmed and state['all_final_contact_faces_refined']!=coverage['all_actual_contact_faces_refined']:
        raise ValueError('Final coverage and published completion state differ')
    plan=read(models[-1]/'plan.local.json');masks={}
    for name in ('lung','bone','heart','fat'):
        matched=[r for r in plan['sources'] if Path(r['path']).name==name+'.nii.gz']
        if len(matched)!=1 or sha(source_path(matched[0]['path']))!=matched[0]['sha256']:
            raise ValueError('Displayed mask differs from the executed model')
        masks[name]=source_path(matched[0]['path'])
    profile=None if confirmed else read(work/'inputs/fits_own_drive/selected_previous_model.local.json')
    if profile is not None and profile['identity']['observations_sha256']!=observation_sha:
        raise ValueError('Historical fixed profile belongs to another observation identity')
    old=None if confirmed else seed['best']
    d=dict(work=work,state=state,config=cfg,bootstrap=bootstrap,models=models,selected=selected,best=best,
        initial=initial,final=final,old=old,profile=profile,warm_seed=seed,confirmed=confirmed,
        input_directory=work/'inputs',preparation=PREPARATION,masks=masks,observations=observations,
        coarse=read(models[0]/'mesh_quality.local.json'),fine=read(models[-1]/'mesh_quality.local.json'),
        initial_coverage=initial_coverage,coverage=coverage,
        forwards=initial_forwards,final_forwards=final_forwards,control_reuse=reuse,run_directory=run)
    pairs(d)
    return d


def load_failed_trial(work):
    """Terminal failure only: preserve available evidence without manufacturing audits."""
    import pandas as pd
    work=Path(work);state=read(work/'run_state.local.json')
    if state['status']!='failed':raise ValueError('Incomplete diagnostic requires an explicitly failed main run')
    cfg=read(work/W3_CONFIG);guard_sources(cfg['sources'],cfg)
    contract=read(work/'candidate_model_contract.local.json');guard_sources(contract['files'],cfg)
    models=[(work/p).resolve() for p in cfg['control_models']]
    observations=pd.read_csv(work/'inputs/observations.local.csv')
    observations=observations[observations.state!='quiet_breathing'].reset_index(drop=True)
    row_order=dict(record_ids=observations.record_id.tolist(),axes=observations.axis.tolist(),states=observations.state.tolist())
    seed=read(work/'warm_seed.local.json');run=work/'inputs'/('joint_own_drive_'+cfg['run_label'])
    checkpoints=[(p,read(p)) for p in sorted(run.glob('evaluation_*.local.json'))]
    valid=[(p,r) for p,r in checkpoints if r['status']=='numerical_candidate_not_validated']
    selected_file=work/'selected_candidate.local.json'
    selected=read(selected_file) if selected_file.exists() else None
    candidate_file=selected_file if selected is not None else (min(valid,key=lambda item:item[1]['mean_square_ohm2'])[0] if valid else None)
    best=selected['best'] if selected is not None else (read(candidate_file) if candidate_file is not None else None)
    if best is not None:
        expected_inverse=next(r['sha256'] for r in cfg['sources'] if source_path(r['path']).resolve()==(PIPE/'src/run_georg_lateral_inverse.m').resolve())
        if best['identity']['worker_sha256']!=expected_inverse:
            raise ValueError('Incomplete priority candidate belongs to another inverse source')
        if (best['identity']['observations_sha256']!=sha(work/'inputs/observations.local.csv') or
                best['identity']['input_sha256']!=sha(work/'inputs/inputs.local.json') or
                best['identity']['model_contract_sha256']!=sha(work/'candidate_model_contract.local.json') or
                best['identity']['model_plan_sha256']!=sha(models[-1]/'plan.local.json')):
            raise ValueError('Incomplete candidate is not bound to the corrected W3 inputs/model')
        np.testing.assert_allclose(best['target_Z_ohm'],observations.Z_ohm.to_numpy(),rtol=0,atol=1e-12)
        if np.asarray(best['predicted_Z_ohm']).shape!=(24,) or not np.isfinite(best['predicted_Z_ohm']).all():
            raise ValueError('Incomplete candidate does not contain 24 finite levels')
    initial_reference_file=work/cfg['initial_reference_file'];initial_reference=read(initial_reference_file)
    initial_files=[work/p for p in cfg['initial_control_files']]
    final_files=[work/(p+'_selected_v1.local.json') for p in ('coarse_forward','fine_inherited','fine_remapped')]
    forwards=[read(p) if p.exists() else None for p in initial_files]
    final_forwards=[read(p) if p.exists() else None for p in final_files]
    reuse=selection_control_reuse(work,cfg,best)
    for row,model in zip(forwards,models,strict=True):
        if row is not None:
            validate_forward(row,initial_reference,observations,sha(work/'inputs/observations.local.csv'),sha(initial_reference_file),model)
    final_reference_file=work/'reference_selected_v1.local.json'
    if any(r is not None for r in final_forwards):
        if selected is None or not final_reference_file.exists():raise ValueError('Final forward lacks a selected candidate/reference')
        reference=read(final_reference_file)
        if reference['selected_candidate_sha256']!=sha(selected_file) or (reference['parameters'],reference['pose'])!=(best['parameters'],best['pose']):
            raise ValueError('Incomplete final reference is not bound to the saved selection')
        for row,model in zip(final_forwards,models,strict=True):
            if row is not None:
                validate_forward(row,best,observations,sha(work/'inputs/observations.local.csv'),
                    reuse['reference_initial_sha256'] if reuse else sha(final_reference_file),model)
    audits=[]
    for name,files,rows in [(cfg['initial_audit_file'],initial_files,forwards),
            ('numerical_audit_selected_v1.local.json',final_files,final_forwards)]:
        file=work/name;audit=read(file) if file.exists() else None
        if audit is not None:
            if any(r is None for r in rows):raise ValueError('Published audit has a missing control')
            validate_audit(audit,files,rows,models,observations,cfg,True)
            if name=='numerical_audit_selected_v1.local.json' and audit['selected_candidate_sha256']!=sha(selected_file):
                raise ValueError('Incomplete final audit belongs to another selected candidate')
        audits.append(audit)
    coverage_file=work/'contact_coverage_final.local.json';coverage=read(coverage_file) if coverage_file.exists() else None
    mesh_hashes=coverage_mesh_hashes(models) if coverage is not None or (work/'contact_coverage_initial.local.json').exists() else None
    if coverage is not None:
        if final_forwards[0] is None:raise ValueError('Final coverage has no final coarse control')
        validate_contact_coverage(coverage,final_files[0],mesh_hashes,cfg)
    initial_coverage=initial_contact_coverage(work,models,cfg,mesh_hashes)
    d=dict(work=work,state=state,config=cfg,bootstrap=read(work/'run_config.initial.local.json'),models=models,
        observations=observations,row_order=row_order,input_directory=work/'inputs',warm_seed=seed,confirmed=True,
        incomplete=True,best=best,selected=selected,candidate_file=candidate_file,checkpoints=checkpoints,
        profile=None,old=None,forwards=forwards,final_forwards=final_forwards,initial=audits[0],final=audits[1],
        initial_coverage=initial_coverage,coverage=coverage,control_reuse=reuse,run_directory=run,quantitative_result_accepted=False)
    pairs(d)
    return d


def load_preparation(history):
    """Reuse checked geometry, not the cancelled WRONG-BC postcheck forwards."""
    work=history['work'];cfg=history['config'];post=work/'postcheck_v2'
    post_state=read(post/'run_state.local.json')
    if post_state['status']!='stopped_after_author_boundary_confirmation' or not post_state['quality_preparation_complete']:
        raise ValueError('Expected stopped WRONG-BC postcheck with completed good mesh preparation')
    deep_model=source_path(post_state['deep_model']);quality=read(deep_model/'fine2/mesh_quality.local.json')
    if not quality['passed']:raise ValueError('Reused depth mesh has not passed quality preparation')
    if (quality['quality_limits']!={'minimum_mean_ratio':.01,'maximum_edge_condition':10000.} or
            not quality['all_positive'] or quality['mean_ratio_minimum']<.01 or
            quality['edge_condition_maximum']>10000. or not quality['within_predeclared_size_limit']):
        raise ValueError('Reused depth mesh violates the unchanged quality screens')
    distribution=read(post/'fraction_distribution_final.local.json')
    if distribution['mesh_sha256']!=sha(deep_model/'fine2/mesh.mat') or quality['mesh_sha256']!=distribution['mesh_sha256']:
        raise ValueError('Depth fraction comparison belongs to another mesh')
    guard_sources(distribution['sources'],cfg)
    old_distribution=read(work/'fraction_distribution.local.json');guard_sources(old_distribution['sources'],cfg)
    ray_file=post/'contact_CT_profiles_final.local.json'
    if not ray_file.exists():ray_file=work/'contact_anatomical_depths.local.json'
    rays=read(ray_file)
    for block in history['best']['pose']['electrodes']:
        found=[r for r in rays['contacts'] if (r['size_mm'],r['axis'])==(block['size_mm'],block['axis'])]
        np.testing.assert_allclose([r['source_centre_lps_mm'] for r in found],block['centres_lps_mm'],rtol=0,atol=1e-10)
    body_root=work/'generic_body_distance_diagnostic_v1';body_file=body_root/'body_distance_diagnostic.local.json'
    body=read(body_file);binding=read(body_root/'executed_sources_binding.local.json')
    if binding['parent_report_sha256']!=sha(body_file):raise ValueError('Historical body binding belongs to another result')
    resolved={r['original_path']:r for r in binding['executed_source_resolution']}
    if {p:r['sha256'] for p,r in resolved.items()}!=body['sources']:
        raise ValueError('Historical body-source binding does not cover all inputs')
    archive=PIPE/'archive/executed_ct_fem_body_20261014'
    for path,digest in body['sources'].items():
        source=Path(resolved[path]['verification_path']);fallback=archive/'diagnostic'/Path(path).name
        if not ((source.exists() and sha(source)==digest) or (fallback.exists() and sha(fallback)==digest)):
            raise ValueError('Body diagnostic source changed: '+path)
    body_plot=body_root/'worst_sections_ru_v2.local.png';body_meta=read(body_plot.with_suffix('.json'))
    if sha(body_plot)!=body_meta['plot_sha256'] or sha(body_file)!=body_meta['report_sha256']:
        raise ValueError('Body plot and numerical diagnostic differ')
    if sha(archive/'plot/ct_fem_body_diagnostic.py')!=body_meta['plotter_sha256']:
        raise ValueError('Executed body plotter changed')
    plot_binding=read(body_root/'russian_plot_source_binding.local.json')
    if plot_binding['plot_sha256']!=body_meta['plot_sha256']:raise ValueError('Body plot binding belongs to another figure')
    for row in plot_binding['source_files']:
        if sha(archive/'plot'/Path(row['original_path']).name)!=row['sha256']:
            raise ValueError('Executed body plot dependency changed')
    for path,digest in body_meta['sources'].items():
        if body['sources'].get(path)!=digest:raise ValueError('Body plot changed a displayed CT/mask input')
    return dict(post_state=post_state,post_failure=read(work/'postcheck_v1/deep_final/preparation_failed.local.json'),
        deep_model=deep_model,deep_quality=quality,deep_distribution=distribution,distribution=old_distribution,
        rays=rays,body=body,body_plot=body_plot,body_plot_metadata=body_meta)


def load_contact_check(d):
    root=d['work']/W3_CONTACT_CHECK;result_file=root/'result.local.json'
    equivalent_file=root/'result_equivalent_reference.local.json'
    explicit_equivalence=equivalent_file.exists()
    original=read(result_file) if result_file.exists() else None
    if explicit_equivalence:result_file=equivalent_file
    if not result_file.exists():
        return dict(status='pending' if root.exists() else 'missing',passed=None,job_status=optional_job_status(root))
    result=read(result_file)
    result['job_status']=optional_job_status(root)
    completed_status='completed_with_proven_reference_equivalence' if explicit_equivalence else 'completed_fixed_bulk_contact_comparison'
    if result['status']!=completed_status:
        if result.get('passed') is not None:raise ValueError('An uncompleted contact check cannot have a pass/fail result')
        return result
    file=root/('comparison_equivalent_reference.local.json' if explicit_equivalence else 'comparison.local.json')
    if not file.exists():return dict(status='pending_comparison_file',passed=None)
    if result['comparison_sha256']!=sha(file):raise ValueError('Optional contact comparison changed')
    comparison=read(file);cfg=read(root/'run_config.local.json');guard_sources(cfg['sources'],cfg)
    for key in ('run_label','parameter_evaluation_order','control_models','deadline_unix','search_deadline_unix'):
        if cfg[key]!=d['config'][key]:raise ValueError('Optional contact check belongs to another search configuration')
    guard_sources(comparison['sources'],cfg)
    direct_file=root/'direct_forward.local.json';direct=read(direct_file)
    reference_sha=sha(d['work']/'reference_selected_v1.local.json')
    if d.get('control_reuse') and direct['identity']['reference_sha256']==d['control_reuse']['reference_initial_sha256']:
        reference_sha=d['control_reuse']['reference_initial_sha256']
    validate_forward(direct,d['best'],d['observations'],sha(d['input_directory']/'observations.local.csv'),
        reference_sha,root/'direct')
    inherited_file=d['work']/'fine_remapped_selected_v1.local.json'
    pinned={source_path(r['path']).resolve():r['sha256'] for r in comparison['sources']}
    for f in (inherited_file,direct_file):
        if pinned.get(f.resolve())!=sha(f):raise ValueError('Contact comparison is not bound to W3 final controls')
    if explicit_equivalence:
        if not result['original_reference_gate_failure_preserved'] or original is None or original['status']!='failed':
            raise ValueError('Original reference-gate failure was not preserved')
        proof=comparison['reference_equivalence'];references=[d['work']/'reference_initial.local.json',d['work']/'reference_selected_v1.local.json']
        forward_rows=[d['final_forwards'][2],direct];values=[]
        for reference,row in zip(references,forward_rows,strict=True):
            digest=sha(reference)
            if pinned.get(reference.resolve())!=digest or row['identity']['reference_sha256']!=digest:
                raise ValueError('Reference-equivalence proof does not bind both actual reference identities')
            value=read(reference)
            if any(value[k]!=row[k] for k in ('parameters','pose')) or value['observations_sha256']!=row['identity']['observations_sha256']:
                raise ValueError('Actual reference physical content differs from its forward')
            values.append(value)
        allowed={'selected_candidate_sha256','common_parameters_not_fitted_here','optimized_on_fine_remapped_mesh'}
        changed=sorted(k for k in allowed if values[0].get(k)!=values[1].get(k))
        if ({k:v for k,v in values[0].items() if k not in allowed}!={k:v for k,v in values[1].items() if k not in allowed} or
                proof['reference_sha256']!=[sha(p) for p in references] or proof['changed_provenance_fields']!=changed or
                proof['schema']!='exact_physical_reference_equivalence_v1' or
                proof['physical_reference_contents_equal_exactly'] is not True or proof['reference_hashes_not_rewritten'] is not True):
            raise ValueError('References differ outside the three allowed provenance fields')
    if (comparison['parameters'],comparison['pose'])!=(d['best']['parameters'],d['best']['pose']):
        raise ValueError('Optional contact comparison changed the final parameters or pose')
    for key in ('record_ids','axes','states'):
        if comparison[key]!=d['final'][key]:raise ValueError('Optional contact comparison changed observation order')
    if (not comparison['bulk_identical_verified'] or result['passed']!=comparison['passed'] or
            comparison['actual_parent_prepared_ancestry_verified'] is not True or
            comparison['units_and_material_cache_identity_verified_here'] is not True):
        raise ValueError('Optional contact-comparison completion contract is inconsistent')
    np.testing.assert_allclose(comparison['inherited_Z_ohm'],d['final_forwards'][2]['fine_Z_ohm'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(comparison['direct_Z_ohm'],direct['fine_Z_ohm'],rtol=0,atol=1e-12)
    delta=np.asarray(comparison['direct_Z_ohm'])-np.asarray(comparison['inherited_Z_ohm'])
    tolerance=.02+.005*np.abs(comparison['direct_Z_ohm'])
    np.testing.assert_allclose(comparison['signed_difference_ohm'],delta,rtol=0,atol=1e-12)
    np.testing.assert_allclose(comparison['difference_ohm'],abs(delta),rtol=0,atol=1e-12)
    np.testing.assert_allclose(comparison['tolerance_ohm'],tolerance,rtol=0,atol=1e-12)
    if comparison['failed_values']!=int(np.count_nonzero(abs(delta)>tolerance)):
        raise ValueError('Contact failure count differs from saved forward arithmetic')
    return dict(result,comparison=comparison,direct=direct,
        original_gate_failure=original if explicit_equivalence else None,
        result_filename=result_file.name,comparison_filename=file.name)


def optional_job_status(root):
    file=root/'run_state.local.json'
    if not file.exists():return None
    state=read(file)
    return dict(status=state.get('status'),last_stage=state.get('stages',[])[-1:] or [],
        error_type=state.get('error_type'),error=state.get('error'),updated_utc=state.get('updated_utc'))


def load_material_update(d):
    root=d['work']/'material_update_v1';result_file=root/'result.local.json'
    if not root.exists():return dict(status='missing',fit=None)
    cfg_file=root/'run_config.local.json'
    if not cfg_file.exists():return dict(status='pending_configuration',fit=None)
    cfg=read(cfg_file)
    for key in ('control_models','deadline_unix','initial_audit_file','input_directory'):
        if cfg[key]!=d['config'][key]:raise ValueError('Conditional material update changed its retained model/input contract')
    if (source_path(cfg['work_directory']).resolve()!=root.resolve() or cfg['profile_max_evaluations']!=4 or
            cfg['profile_max_iterations']!=3):raise ValueError('Unexpected conditional material-update scope/budget')
    initial_sources={(source_path(r['path']).resolve(),r['sha256']) for r in d['config']['sources']}
    stage_sources={(source_path(r['path']).resolve(),r['sha256']) for r in cfg['sources']}
    if not initial_sources<=stage_sources:raise ValueError('Conditional stage changed an existing frozen dependency')
    # Main dependencies are already guarded by load_trial; verify only new stage pins here.
    guard_sources([r for r in cfg['sources'] if (source_path(r['path']).resolve(),r['sha256']) not in initial_sources],cfg)
    if not result_file.exists():return dict(status='pending',fit=None,config=cfg,job_status=optional_job_status(root))
    result=read(result_file);result.update(config=cfg,fit=None,job_status=optional_job_status(root))
    if result.get('quantitative_result_accepted') is not False:raise ValueError('Conditional update cannot be accepted by its completion flag')
    if result['status']=='failed':return result
    for key,expected in [('maximum_evaluations',4),('maximum_iterations',3),('location_refitted_here',False)]:
        if result[key]!=expected:raise ValueError('Conditional result does not match its bounded fixed-pose scope')
    if result.get('fit_file') is None:
        if result['status']=='completed_conditional_material_fit':result['status']='completed_without_fit'
        return result
    fit_file=source_path(result['fit_file'])
    expected_file=d['input_directory']/'fits_own_drive/selected_previous_model.local.json'
    if fit_file.resolve()!=expected_file.resolve():raise ValueError('Conditional result points outside its own W3 fit artifact')
    if not fit_file.exists():return dict(result,status='fit_file_missing',fit=None)
    if result['fit_sha256']!=sha(fit_file):raise ValueError('Conditional material-fit artifact changed')
    try:fit=read(fit_file)
    except (ValueError,UnicodeError):return dict(result,status='fit_file_incomplete',fit=None)
    if d['best'] is None:raise ValueError('Conditional fit cannot be bound without a retained joint candidate')
    identity=fit['identity']
    expected_identity=dict(observations_sha256=sha(d['input_directory']/'observations.local.csv'),
        input_sha256=sha(d['input_directory']/'inputs.local.json'),grid_sha256=sha(d['input_directory']/'pose_grid.local.json'),
        model_contract_sha256=sha(d['work']/'candidate_model_contract.local.json'),
        model_plan_sha256=sha(d['models'][-1]/'plan.local.json'),
        warm_start_sha256=sha(d['work']/'warm_seed.local.json'),
        numerical_audit_sha256=sha(d['work']/d['config']['initial_audit_file']),
        prepared_sha256=d['best']['identity']['prepared_sha256'])
    if any(identity[k]!=v for k,v in expected_identity.items()) or identity['search_mode']!='profile_grid' or fit['source_rule']!='own_drive':
        raise ValueError('Conditional fit uses another observation/model/start/source contract')
    worker_digest=next(r['sha256'] for r in cfg['sources'] if source_path(r['path']).resolve()==(PIPE/'src/run_georg_lateral_inverse.m').resolve())
    if identity['worker_sha256']!=worker_digest:raise ValueError('Conditional fit belongs to another executed inverse')
    verify_executed_source(PIPE/'src/run_georg_lateral_inverse.m',worker_digest,cfg)
    poses=read(d['input_directory']/'pose_grid.local.json')['poses']
    matching=[p for p in poses if p['pose_id']==fit['pose_id']]
    if len(matching)!=1 or d['best'] is None or any(matching[0].get(k)!=v for k,v in d['best']['pose'].items()):
        raise ValueError('Conditional fit grid did not retain the complete selected joint pose')
    if fit['centre_lps_mm']!=d['best']['pose']['centre_lps_mm'] or fit['electrodes']!=d['best']['pose']['electrodes']:
        raise ValueError('Conditional material fit moved its centre/contact coordinates')
    target=d['observations'].Z_ohm.to_numpy();z=np.asarray(fit['predicted_Z_ohm'])
    np.testing.assert_allclose(fit['target_Z_ohm'],target,rtol=0,atol=1e-12)
    if z.shape!=(24,) or not np.isfinite(z).all():raise ValueError('Conditional fit requires 24 finite own-source levels')
    np.testing.assert_allclose(fit['RMSE_ohm'],np.sqrt(np.mean((z-target)**2)),rtol=1e-12,atol=1e-12)
    rhos=np.asarray([*fit['rho_inhale_ohm_m'],*fit['rho_exhale_ohm_m']])
    if rhos.shape!=(6,) or not np.isfinite(rhos).all() or np.any(rhos<=0):raise ValueError('Invalid conditional phase material vectors')
    same_rho=bool(np.array_equal(rhos,np.asarray([*d['best']['rho_inhale_ohm_m'],*d['best']['rho_exhale_ohm_m']])))
    plan=read(d['models'][-1]/'plan.local.json')
    # The pinned profile_grid implementation calls objective with rule=2 (4096 in this plan).
    points=plan['numerical_quadrature_points'][1]
    return dict(result,fit=fit,fit_path=fit_file,same_joint_rho_exactly=same_rho,
        optimization_rule_index=2,optimization_rule_points=points,
        new_rho_numerical_controls_available=False)


def load(work=WORK):
    # A live run is gated; a terminal failure gets an explicit incomplete chapter.
    state=read(Path(work)/'run_state.local.json')
    d=load_failed_trial(work) if state['status']=='failed' else load_trial(work,confirmed=True)
    contract=confirmed_inputs(work);d.update(contract)
    history=load_trial(HISTORY_WORK);preparation=load_preparation(history);history.update(preparation)
    d.update(preparation,history=history)
    if d['models'][-1]!=(d['deep_model']/'fine2').resolve():
        raise ValueError('W3 is not bound to the reported reused depth preparation')
    d['contact_check']=load_contact_check(d)
    d['material_update']=load_material_update(d)
    if d['forwards'][0] is not None:boundary_correction(d)
    return d


def rename(value,old,new):
    from IPython.display import HTML
    return HTML(value.data.replace(old,new,1))


def numbered_table(value,index):
    """Number reader captions in document order without changing table contents."""
    import html
    from IPython.display import HTML
    def caption(match):
        title=html.unescape(match.group(1));scope=''
        if title.startswith('Таблица прежнего расчёта · '):
            scope='Прежний расчёт. ';title=title.removeprefix('Таблица прежнего расчёта · ')
        else:title=title.removeprefix('Таблица ')
        title=re.sub(r'^\d+\.\s*','',title)
        heads={'отдельного условного ρ-подбора исправленного расчёта':'Условный подбор ρ в исправленном расчёте',
            'поиска ':'Поиск ','исправленного расчёта.':'Исправленный расчёт.',
            'незавершённого исправленного расчёта.':'Незавершённый исправленный расчёт.',
            'уровней ':'Уровни ','контрастов ':'Контрасты ','контактов.':'Контакты.',
            'происхождения.':'Происхождение.','подготовки.':'Подготовка.',
            'сетки.':'Сетка.','долей.':'Доли.','расстояний.':'Расстояния.',
            'граничных условий.':'Граничные условия.',
            'отдельного условного ρ-подбора ':'Отдельный условный подбор ρ: '}
        for old,new in heads.items():
            if title.startswith(old):title=new+title[len(old):];break
        if scope and title.startswith('Поиск прежнего расчёта.'):
            title=title.replace('Поиск прежнего расчёта.','Поиск.',1)
        return '<summary>'+html.escape(f'Таблица {index}. '+scope+title)+'</summary>'
    return HTML(re.sub(r'<summary>(.*?)</summary>',caption,value.data,count=1,flags=re.S))


def reader_status(value):
    """Human descriptions for presentation; recorded status fields stay unchanged."""
    return {
        'pending':'Ожидается результат','missing':'Результат отсутствует',
        'pending_configuration':'Конфигурация ещё не сохранена',
        'pending_comparison_file':'Сравнение ещё не опубликовано',
        'not_executed_original_time_limit':'Не выполнено из-за исходного срока',
        'not_executed_main_failed':'Не выполнено после ошибки основного расчёта',
        'interrupted_original_time_limit':'Прервано по исходному общему сроку',
        'failed':'Процедура завершилась с ошибкой',
        'complete_with_explicit_numerical_results':'Процедура завершена; численные результаты сохранены',
        'completed_fixed_bulk_contact_comparison':'Сравнение контактов при фиксированном объёме завершено',
        'completed_with_proven_reference_equivalence':'Сравнение завершено с доказанной эквивалентностью опорных данных',
        'completed_conditional_material_fit':'Условный подбор сопротивлений завершён',
        'completed_without_fit':'Процедура завершена без сохранённого результата подгонки',
        'fit_file_missing':'Файл результата подгонки отсутствует',
        'fit_file_incomplete':'Файл результата подгонки неполон',
        'fixed_pose_material_update_conditional_materials':'Подбор сопротивлений при фиксированном положении',
        'fixed_bulk_direct_contacts':'Непосредственный выбор контактных граней при фиксированном объёме',
        'deadline_stop':'Остановлено по сроку',
    }.get(value,'Служебное состояние сохранено в техническом паспорте')


def reader_error(value):
    if 'Timeout' in str(value) or 'deadline' in str(value).lower():
        return 'Превышен исходный срок выполнения'
    if 'reference' in str(value).lower():return 'Не совпали контрольные суммы опорных данных'
    return 'Ошибка выполнения; подробная причина сохранена в техническом паспорте'


def body_correction_table(d):
    body=d['body'];distances=body['distances'];rows=[
        ['Добавленные воксели',body['body']['added']],
        ['Удалённые воксели',body['body']['removed']],
        ['Доля изменённых вокселей тела',number(body['body']['changed_relative'],8)],
        ['Объём изменённых вокселей, мл',number(body['body']['changed_volume_ml'],4)]]
    for key,name in [('original_to_regularized','До → после'),('regularized_to_original','После → до')]:
        value=distances[key]
        rows.extend([[name+': выборочный максимум, мм',number(value['max_mm'],6)],
            [name+': превышения 1,25 мм среди 15 000 проб',value['above_declared_threshold']],
            [name+': максимум локальной проверки вершин, мм',number(value['local_vertex_probe']['max_mm'],6)]])
    return table('Таблица 20. Геометрическое изменение КТ-контура при одном замыкании',
        ['Характеристика','Результат'],rows)


def body_correction_figure(d,include_js=False):
    import base64
    from IPython.display import HTML
    encoded=base64.b64encode(d['body_plot'].read_bytes()).decode('ascii')
    return HTML('<img alt="Сечения КТ и контуры до и после замыкания" style="width:100%;height:auto" src="data:image/png;base64,'+encoded+'">')


def observation_labels(d):
    a=d['final'] if d['final'] is not None else d['row_order']
    return [r.rsplit('-',1)[-1]+' · '+('продольная' if axis=='longitudinal' else 'поперечная')+
        ' · '+('вдох' if state=='inhale' else 'выдох') for r,axis,state in zip(a['record_ids'],a['axes'],a['states'],strict=True)]


def boundary_correction(d):
    """Isolated BC effect on the SAME parent, pose and rho, not final-fit improvement."""
    old=d['history']['final_forwards'][0];new=d['forwards'][0]
    for key in ('parameters','pose','record_ids','axes','states','target_Z_ohm','source_rule'):
        if old[key]!=new[key]:raise ValueError('Isolated BC comparison changed '+key)
    for key in ('plan_sha256','prepared_sha256','physical_contacts_sha256','physical_montage_helper_sha256'):
        if old['identity'][key]!=new['identity'][key]:
            raise ValueError('Isolated BC comparison changed the parent model/kernel: '+key)
    if old['identity']['contact_inheritance']!=new['identity']['contact_inheritance']:
        raise ValueError('Isolated BC comparison changed the contact master or parent mesh')
    if d['history']['models'][0]!=d['models'][0]:raise ValueError('Isolated BC comparison requires the same parent mesh')
    x=np.asarray(old['fine_Z_ohm']);y=np.asarray(new['fine_Z_ohm']);target=np.asarray(new['target_Z_ohm'])
    changed=np.array([7<=int(r.rsplit('-',1)[1])<=10 for r in new['record_ids']]);delta=y-x
    return dict(old=x,new=y,signed=delta,absolute=abs(delta),changed=changed,
        old_rmse=float(np.sqrt(np.mean((x-target)**2))),new_rmse=float(np.sqrt(np.mean((y-target)**2))),
        maximum=float(np.max(abs(delta))),other_maximum=float(np.max(abs(delta[~changed]))),
        other_passed=bool(np.all(abs(delta[~changed])<=1e-8)))


def boundary_correction_table(d):
    r=boundary_correction(d)
    return table('Таблица граничных условий. Изолированная замена 1×4 на 1×1 на родительской сетке',
        ['Наблюдение','Изменён монтаж 007–010','Прежний расчёт, Ом','Исправленный расчёт, Ом','исправленный расчёт минус прежний, Ом','Модуль изменения, Ом'],
        [[label,'Да' if r['changed'][i] else 'Нет',*[number(v[i],10) for v in
            (r['old'],r['new'],r['signed'],r['absolute'])]] for i,label in enumerate(observation_labels(d))])


def boundary_correction_figure(d,include_js=False):
    import plotly.graph_objects as go
    r=boundary_correction(d);labels=observation_labels(d);fig=go.Figure()
    for name,values,color,symbol in [('Знаковое изменение: исправленный расчёт минус прежний',r['signed'],'#326ba8','circle'),
            ('Модуль изменения',r['absolute'],'#df8b26','square')]:
        fig.add_trace(go.Scatter(x=labels,y=values,mode='markers',name=name,
            marker=dict(color=color,symbol=symbol,size=9)))
    fig.add_trace(go.Scatter(x=labels,y=np.zeros(24),mode='lines',name='Нулевое изменение',
        line=dict(color='#222222',dash='dash')))
    fig.update_layout(xaxis=dict(title='Запись, ось и состояние',tickangle=-55),
        yaxis_title='Изменение импеданса при замене граничных условий, Ом',height=660,margin=dict(l=85,r=35,t=100,b=180))
    return figure_html(fig,include_js)


def mesh_table(d):return rename(previous_mesh_table(d),'Таблица 3.','Таблица 1.')
def residual_table(d):return rename(previous_residual_table(d),'Таблица 11.','Таблица 6.')
def coordinate_table(d):return rename(previous_coordinate_table(d),'Таблица 12.','Таблица 13.')


def pairs(d):
    a=d['final'] if d['final'] is not None else d['row_order'];result=[]
    if len(a['states'])!=24 or set(a['states'])!={'inhale','exhale'}:
        raise ValueError('Expected 24 inhale/exhale levels')
    for i,state in enumerate(a['states']):
        if state!='inhale':continue
        found=[j for j,s2 in enumerate(a['states']) if s2=='exhale' and
            a['record_ids'][j]==a['record_ids'][i] and a['axes'][j]==a['axes'][i]]
        if len(found)!=1:raise ValueError('Ambiguous respiratory pair')
        label=a['record_ids'][i].rsplit('-',1)[-1]+' · '+('продольная' if a['axes'][i]=='longitudinal' else 'поперечная')
        result.append((i,found[0],label))
    if len(result)!=12 or len({v for i,j,_ in result for v in (i,j)})!=24:
        raise ValueError('Respiratory pairs do not cover exactly 12 distinct measured leads')
    return result


def stages(d):
    if d['confirmed']:
        return [('Исправленный расчёт: свежий контроль с пересчитанными долями начального приближения',np.asarray(d['forwards'][2]['fine_Z_ohm'])),
            ('Исправленный расчёт: выбранный кандидат ограниченного совместного поиска',np.asarray(d['best']['predicted_Z_ohm']))]
    return [('Прежний расчёт: прежний кандидат',np.asarray(d['old']['predicted_Z_ohm'])),
        ('Прежний расчёт: новая сетка, прежние параметры',np.asarray(d['forwards'][2]['fine_Z_ohm'])),
        ('Прежний расчёт: фиксированный центр',np.asarray(d['profile']['predicted_Z_ohm'])),
        ('Прежний расчёт: совместный подбор',np.asarray(d['best']['predicted_Z_ohm']))]


def components(d,z):
    r=z-np.asarray(d['best']['target_Z_ohm']);p=pairs(d)
    a=np.array([(r[i]+r[j])/2 for i,j,_ in p]);b=np.array([(r[j]-r[i])/2 for i,j,_ in p])
    level=float(np.mean(a*a));contrast=float(np.mean(b*b));total=float(np.mean(r*r))
    if not np.isclose(total,level+contrast,rtol=1e-12):raise ValueError('Residual decomposition failed')
    return np.sqrt(total),np.sqrt(level),np.sqrt(contrast),100*contrast/total if total else 0.


def coverage_table(d):
    rows=[]
    for label,c in [('Перед подбором',d['initial_coverage']),('После подбора',d['coverage'])]:
        if c is None:continue
        for record in dict.fromkeys(r['record_id'] for r in c['contacts']):
            r=[v for v in c['contacts'] if v['record_id']==record]
            rows.append([label,record.rsplit('-',1)[-1],len(r),sum(v['all_faces_refined'] for v in r),
                number(100*min(v['refined_boundary_area_fraction'] for v in r),2)])
    return table('Таблица 2. Покрытие фактических площадок сгущением',
        ['Условия','Запись','Площадок','Полностью покрыто','Минимальная доля площади, %'],rows)


def initial_coverage_statement(d):
    c=d['initial_coverage']
    if c is None:return 'Начальная проверка покрытия исправленного расчёта не сохранена; её прохождение не заявляется.'
    return ('Начальная проверка исправленного расчёта связана с расчётом на родительской сетке и фактической картой происхождения элементов глубинной сетки:  '+
        str(sum(r['all_faces_refined'] for r in c['contacts']))+' из '+str(len(c['contacts']))+
        ' конечных контактных площадок CEM полностью покрыты сгущением; минимальная доля уточнённой площади — '+
        number(100*c['minimum_refined_boundary_area_fraction'],6)+' %. Проверены реальные граничные грани площадок, а не только направляющие центры.')


def numerical_table(d):
    rows=[]
    final_label='Выбранная точка: те же сохранённые контроли' if d.get('control_reuse') else 'Новый выбранный кандидат'
    for label,a in [('Начальное приближение: свежие контроли' if d['confirmed'] else 'Прежний расчёт: прежние параметры и положение',d['initial']),(final_label,d['final'])]:
        for key,name in [('spatial','Пространство FEM'),('fractions','Тканевые доли'),('total','Суммарное изменение')]:
            r=a[key];rows.append([label,name,number(max(r['difference_ohm']),5),r['failed_values'],number(r['maximum_ratio'],4)])
    return table('Таблица 3. Раздельные численные сравнения на новой ветви сеток',
        ['Условия','Изменение','Максимум |ΔZ|, Ом','Превышения из 24','Максимум |ΔZ|/допуск'],rows)


def initial_controls_table(d):
    rows=[];target=d['observations'].Z_ohm.to_numpy()
    for label,row in zip(('Родитель','Глубинная сетка: унаследованные доли','Глубинная сетка: пересчитанные доли'),d['forwards'],strict=True):
        rows.append([label,number(np.sqrt(np.mean((np.asarray(row['fine_Z_ohm'])-target)**2)),9)
            if row is not None else 'Не рассчитано'])
    audit=d['initial']
    for key,name in [('shape_QC_passed','Контроль формы'),('energy_inheritance_passed','Наследование энергии'),
            ('quadrature_passed','Интегрирование'),('derivatives_passed','Производные'),
            ('system_checks_passed','Линейные системы'),('all_checks_passed','Все численные допуски, включая пространство/доли')]:
        rows.append([name,'Нет сохранённого аудита' if audit is None else ('Да' if audit[key] else 'Нет')])
    return table('Таблица исправленного расчёта. Начальная точка: RMSE и раздельные статусы', ['Контроль','Результат'],rows)


def stages_table(d):
    return table('Таблица 4. Изменение невязки и её составляющих',
        ['Этап','RMSE, Ом','Ошибка средних уровней, Ом','Ошибка половины контраста, Ом','Доля контраста в SSE, %'],
        [[name,*[number(v,4) for v in components(d,z)]] for name,z in stages(d)])


def rho_table(d):
    rows=[]
    if d['confirmed']:
        results=[('Исправленный расчёт: только начальное приближение',np.exp(d['warm_seed']['parameters'][:6])),
            ('Исправленный расчёт: доступный кандидат без итогового принятия' if d.get('incomplete') else 'Исправленный расчёт: выбранный кандидат',
             np.exp(d['best']['parameters'][:6]))]
    else:
        results=[(name,[*r['rho_inhale_ohm_m'],*r['rho_exhale_ohm_m']]) for name,r in
            [('Прежний расчёт: прежний кандидат',d['old']),('Прежний расчёт: фиксированный центр',d['profile']),('Прежний расчёт: совместный подбор',d['best'])]]
    bounds=read(d['input_directory']/'inputs.local.json')['search_bounds_ohm_m']
    for name,rhos in results:
        for k,material in enumerate(['Мягкотканный фон','Лёгкое','Жир']):
            lo,hi=bounds[k];at_bound=any(np.isclose(v,lo,rtol=0,atol=1e-6) or
                np.isclose(v,hi,rtol=0,atol=1e-6) for v in (rhos[k],rhos[k+3]))
            rows.append([name,material,number(rhos[k],6),number(rhos[k+3],6),'Да' if at_bound else 'Нет'])
    return table('Таблица 5. Удельные сопротивления как параметры исследовательского расчёта',
        ['Этап','Материал','Вдох, Ом·м','Выдох, Ом·м','Близость к техническому пределу (≤10⁻⁶ Ом·м)'],rows)


def fixed_properties_table(d):
    reference=read(d['deep_model']/'fine2/reference_materials.local.json')
    inputs=read(d['input_directory']/'inputs.local.json')
    materials=reference['materials'];rho=reference['rho_ohm_m']
    rows=[['Частота',str(reference['frequency_hz'])+' Гц','Частота расчётных электрических свойств'],
        ['Кость',number(rho[materials.index('bone')],10)+' Ом·м','Фиксировано; вся маска кости представлена кортикальной костью'],
        ['Целое сердце',number(rho[materials.index('heart')],11)+' Ом·м','Фиксировано; вся маска сердца представлена миокардом'],
        ['Опорный мягкотканный фон',number(rho[materials.index('soft_background')],12)+' Ом·м',
         'Опорное значение принято как для мышцы; сопротивление фона затем подбирается']]
    names={'soft_background':'мягкотканный фон','lung':'лёгкое','fat':'жир'}
    for index,(material,bounds) in enumerate(zip(inputs['free_materials'],inputs['search_bounds_ohm_m'],strict=True),1):
        rows.append(['Свободный фазовый параметр ρ'+str(index),names[material],
            'Технический диапазон '+number(bounds[0],1)+'–'+number(bounds[1],0)+' Ом·м; индивидуальная физиологическая норма не установлена'])
    return table('Таблица Опорные свойства и фиксированные материалы',
        ['Свойство или параметр','Значение или материал','Роль и допущение'],rows)


def contact_properties_table(d):
    inputs=read(d['input_directory']/'inputs.local.json');diameter=inputs['assumed_contact_diameter_mm']
    rows=[['Диаметр диска',number(diameter,1)+' мм',
        'Паспорт входов: '+('измерен' if inputs['contact_diameter_measured'] else 'принят как допущение; не измерен')],
        ['Номинальная площадь диска',number(np.pi*(diameter/2)**2,9)+' мм²',
         'Геометрическая площадь π(d/2)² по принятому диаметру d']]
    for name,relative in zip(('Родительская сетка','Сгущённая сетка: унаследованные доли',
            'Сгущённая сетка: пересчитанные доли'),d['config']['control_models'],strict=True):
        plan=read((d['work']/relative).resolve()/'plan.local.json')
        rows.append(['Удельный контактный импеданс: '+name,
            format(plan['contact_impedance_ohm_m2'],'.15g').replace('.',',')+' Ом·м²',
            'План этой модели: фиксирован; '+('измерен' if plan['contact_impedance_is_measured'] else 'принят как допущение; не измерен')])
    coverage=d['coverage'];coarse=d['forwards'][0]
    for name,contacts,source in [
            ('Фактическая площадь в выбранной точке',coverage['contacts'] if coverage is not None else None,
             'Итоговая проверка покрытия фактических контактных граней'),
            ('Фактическая площадь в начальной точке',coarse['patch_qc']['contacts'] if coarse is not None else None,
             'Начальный контроль контактных граней родительской сетки')]:
        value=('Не сохранена' if not contacts else number(min(r['area_mm2'] for r in contacts),9)+'–'+
            number(max(r['area_mm2'] for r in contacts),9)+' мм²')
        rows.append([name,value,source+'; минимум–максимум по сохранённым описаниям'])
    if coverage is not None:
        rows.append(['Объём итоговой проверки',str(len(coverage['contacts']))+' описаний; '+
            str(len({r['record_id'] for r in coverage['contacts']}))+' условий записи',
            'Повторные описания площадок в разных условиях, без подсчёта уникальных физических электродов'])
    if coarse is not None:
        contacts=coarse['patch_qc']['contacts']
        rows.append(['Объём начальной проверки родительской сетки',str(len(contacts))+' описаний; '+
            str(len({r['row'] for r in contacts}))+' условий записи','Сохранённая проверка контактных граней начального расчёта'])
    return table('Таблица Контактные диски и принятый контактный импеданс',
        ['Характеристика','Значение или диапазон','Источник и смысл'],rows)


def respiratory_table(d):
    target=np.asarray(d['best']['target_Z_ohm']);s=stages(d)
    return table('Таблица 7. Все парные дыхательные изменения',
        ['Запись и ось','Эксперимент, Ом',*[n+', Ом' for n,_ in s]],
        [[label,number(target[j]-target[i],4),*[number(z[j]-z[i],4) for _,z in s]] for i,j,label in pairs(d)])


def sensitivity_table(d):
    singular=np.asarray(d['best']['singular_values'])
    return table('Таблица 9. Сингулярные числа якобиана шести сопротивлений',
        ['Номер','Сингулярное число, Ом'],[[i+1,number(v,6)] for i,v in enumerate(singular)])


def repeated_table(d):
    a=d['final'];rows=[]
    for pair_label,axis,records in [('140 мм поперечная: 005 → 006','perpendicular',('005','006')),
                                    ('60 мм продольная: 010 → 011','longitudinal',('010','011'))]:
        for state,label in [('inhale','Вдох'),('exhale','Выдох')]:
            ix=[next(i for i in range(24) if a['record_ids'][i].endswith(r) and
                a['axes'][i]==axis and a['states'][i]==state) for r in records]
            for name,z in [('Эксперимент',np.asarray(d['best']['target_Z_ohm'])),*stages(d)]:
                rows.append([pair_label,label,name,*[number(z[i],4) for i in ix],number(z[ix[1]]-z[ix[0]],4)])
    return table('Таблица 11. Сборки, сохранявшиеся при переходе к крестам',
        ['Повтор','Состояние','Ряд','Отдельная запись, Ом','Запись креста, Ом','Крест минус отдельная запись, Ом'],rows)


def source_projection(d):
    b=d['best'];own=np.asarray(b['own_transfer_ohm']);counter=np.asarray(b['counter_transfer_ohm'])
    target=np.asarray(b['target_Z_ohm']);denominator=counter@counter
    alpha=float(counter@(target-own)/denominator) if denominator else 0.
    bounded=float(np.clip(alpha,-1,1));required=[];a=d['final']
    for state in ('inhale','exhale'):
        ix=[next(i for i in range(24) if a['record_ids'][i].endswith(r) and
            a['axes'][i]=='longitudinal' and a['states'][i]==state) for r in ('010','011')]
        i,j=ix;delta=counter[j]-counter[i]
        required.append(float((target[j]-target[i]-own[j]+own[i])/delta) if delta else None)
    return dict(alpha=alpha,bounded_alpha=bounded,predicted=own+bounded*counter,required_for_repeat=required)


def source_table(d):
    b=d['best'];own=np.asarray(b['own_transfer_ohm']);counter=np.asarray(b['counter_transfer_ohm'])
    target=np.asarray(b['target_Z_ohm']);rows=[]
    projection=source_projection(d)
    for name,z,alpha in [('Собственный источник',own,'0'),('Равные токи, совпадающая фаза',own+counter,'1'),
        ('Равные токи, противоположные фазы',own-counter,'−1'),('Равные токи, 90° и модуль',np.hypot(own,counter),'Неприменимо'),
        ('Условный вещественный подбор доли второго источника',projection['predicted'],number(projection['bounded_alpha'],5))]:
        mismatch=sum(np.sign(z[j]-z[i])!=np.sign(target[j]-target[i]) for i,j,_ in pairs(d))
        rows.append([name,alpha,number(np.sqrt(np.mean((z-target)**2)),4),mismatch])
    return table('Таблица 12. Сценарии источников при одинаковых параметрах и положении',
        ['Условия возбуждения','Коэффициент α','RMSE, Ом','Несовпадения знака из 12'],rows)


def respiratory_figure(d,include_js=False):
    import plotly.graph_objects as go
    target=np.asarray(d['best']['target_Z_ohm']);p=pairs(d);fig=go.Figure()
    for name,z,symbol,color in [('Эксперимент',target,'circle','#222222'),
        *[(n,z,'square',c) for (n,z),c in zip(stages(d),['#777777','#3989ad','#8c61aa','#dc872e'])]]:
        fig.add_trace(go.Scatter(x=[v[2] for v in p],y=[z[j]-z[i] for i,j,_ in p],
            mode='markers',name=name,marker=dict(symbol=symbol,size=9,color=color)))
    fig.add_trace(go.Scatter(x=[v[2] for v in p],y=[0]*len(p),mode='lines',name='Нет изменения',line=dict(color='#333333',dash='dash')))
    fig.update_layout(xaxis=dict(title='Запись и ось',tickangle=-45),yaxis_title='Z выдоха − Z вдоха, Ом',
        height=710,margin=dict(l=85,r=30,t=115,b=170))
    return figure_html(fig,include_js)


def depth_table(d):
    return table('Таблица 14. Профили по локальным нормалям всех боковых контактов',
        ['Размер, мм','Ось','Контакт','Первое попадание в маску лёгкого, мм','Смещение начала при проекции, мм'],
        [[r['size_mm'],'Продольная' if r['axis']=='longitudinal' else 'Поперечная',r['contact'],
            number(r['first_lung_depth_mm'],1) if r['first_lung_depth_mm'] is not None else 'Нет на первых 80 мм',
            number(r['projection_distance_mm'],6)] for r in d['rays']['contacts']])


def depth_figure(d,include_js=False):
    import plotly.graph_objects as go
    rows=d['rays']['contacts'];labels=[str(r['size_mm'])+' мм · '+
        ('продольная' if r['axis']=='longitudinal' else 'поперечная')+' · '+r['contact'] for r in rows]
    fig=go.Figure()
    for tissue,name,color in [(0,'Мягкотканный фон','#c6aa93'),(1,'Лёгкое','#4ba3bb'),
        (2,'Кость','#91918c'),(3,'Целое сердце','#b55061'),(4,'Жир','#d9ad3c'),(-1,'Вне поля КТ','#333333')]:
        x=[];y=[]
        for label,r in zip(labels,rows):
            for depth,code in zip(r['depth_mm'],r['tissue_label']):
                if code==tissue:x.append(depth);y.append(label)
        if x:fig.add_trace(go.Scattergl(x=x,y=y,mode='markers',name=name,
            marker=dict(symbol='square',size=5,color=color),hovertemplate='%{y}<br>%{x:.1f} мм<extra>%{fullData.name}</extra>'))
    found=[(label,r['first_lung_depth_mm']) for label,r in zip(labels,rows) if r['first_lung_depth_mm'] is not None]
    fig.add_trace(go.Scatter(x=[v for _,v in found],y=[n for n,_ in found],mode='markers',
        name='Первое попадание в лёгкое',marker=dict(symbol='diamond-open',size=9,color='#222222')))
    fig.update_layout(xaxis_title='Расстояние вдоль принятой внутренней нормали, мм',
        yaxis=dict(title='Сборка и контакт',categoryorder='array',categoryarray=labels,autorange='reversed'),
        height=950,margin=dict(l=235,r=40,t=85,b=80))
    return figure_html(fig,include_js)


def preparation_attempts_table(d):
    rows=[]
    for a in d['post_failure']['attempts']:
        q=a['quality'];rows.append(['Первая подготовка: отказ',a['attempt'],q['nodes'],q['tetrahedra'],
            number(q['mean_ratio_minimum'],9),q['failing_mean_ratio'],'Нет'])
    q=d['deep_quality'];rows.append(['Вторая подготовка: завершена','Итог',q['nodes'],q['tetrahedra'],
        number(q['mean_ratio_minimum'],9),q['failing_mean_ratio'],'Да' if q['passed'] else 'Нет'])
    return table('Таблица подготовки. Неудачные попытки и пригодная по форме глубинная сетка',
        ['Ветвь','Попытка','Узлы','Тетраэдры','Минимальный безразмерный показатель формы','Нарушений формы','Контроль формы'],rows)


def deep_quality_table(d):
    q=d['deep_quality']
    return table('Таблица сетки. Геометрический контроль повторно используемой глубинной подготовки',
        ['Характеристика','Результат'],[
        ['Узлы',q['nodes']],['Тетраэдры',q['tetrahedra']],
        ['Минимальный показатель формы',number(q['mean_ratio_minimum'],9)],
        ['Максимальная обусловленность матрицы рёбер',number(q['edge_condition_maximum'],3)],
        ['Нарушения допуска формы',q['failing_mean_ratio']],
        ['Нарушения допуска обусловленности рёбер',q['failing_edge_condition']],
        ['Все ориентированные объёмы положительны','Да' if q['all_positive'] else 'Нет'],
        ['Относительное изменение общего объёма',number(q['relative_volume_difference'],12)],
        ['FEM-контроли прежних граничных условий после второй подготовки','Остановлены; завершённая проверка не заявляется']])


def fraction_distribution_table(d):
    rows=[]
    for branch,distribution in [('Прежний расчёт: поверхностное уточнение',d['distribution']),
            ('Исправленный расчёт: повторно используемая глубинная сетка',d['deep_distribution'])]:
        for rule in distribution['rules']:
            for k,name in enumerate(('Мягкотканный фон','Лёгкое','Кость','Целое сердце','Жир')):
                rows.append([branch,rule['points'],rule['changed_elements'],name,
                    rule['changed_elements_by_material'][k],number(rule['maximum_absolute_fraction_change'][k],6)])
    return table('Таблица долей. Наследование → повторное интегрирование на той же сетке',
        ['Сетка','Проб на элемент','Изменена хотя бы одна доля','Материал','Элементов с изменением','Максимум |Δдоля|'],rows)


def contact_check_table(d):
    c=d['contact_check'];r=c.get('comparison')
    status={'pending':'Ожидается результат','missing':'Результат отсутствует',
        'pending_comparison_file':'Результат сравнения ещё не опубликован',
        'not_executed_original_time_limit':'Не выполнено: исходный лимит времени',
        'not_executed_main_failed':'Не выполнено: основной расчёт завершился ошибкой',
        'interrupted_original_time_limit':'Прервано: исходный лимит времени',
        'failed':'Ошибка выполнения','completed_fixed_bulk_contact_comparison':'Завершённое сравнение при фиксированном объёме'}
    status['completed_with_proven_reference_equivalence']='Завершено с доказанной эквивалентностью физических опорных данных'
    rows=[['Состояние',reader_status(c['status'])],
        ['Выдержан допуск','Нет завершённого свидетельства' if r is None else ('Да' if r['passed'] else 'Нет')]]
    if r is not None:
        rows.extend([['Максимум |Z прямых контактов − Z наследуемых|, Ом',number(max(r['difference_ohm']),8)],
            ['Превышения из 24',r['failed_values']],['Максимум |ΔZ|/допуск',number(r['maximum_ratio'],5)],
            ['Контроль интегрирования','Да' if r['quadrature_checks_passed'] else 'Нет'],
            ['Объёмная сетка и коэффициенты неизменны','Да' if r['bulk_identical_verified'] else 'Нет'],
            ['Происхождение подготовленных материалов проверено','Да' if r['actual_parent_prepared_ancestry_verified'] else 'Нет'],
            ['Единицы и соответствие подготовленных материалов проверены','Да' if r['units_and_material_cache_identity_verified_here'] else 'Нет'],
            ['Элементы матриц сравнивались здесь','Да' if r['matrix_entries_compared'] else 'Нет; проверка по контрольным суммам и сохранённым сведениям о происхождении']])
    if 'remaining_seconds' in c:rows.append(['Оставалось времени, с',number(c['remaining_seconds'],1)])
    if 'error' in c:rows.append(['Сохранённая ошибка',reader_error(c['error'])])
    if c.get('job_status'):
        rows.append(['Последнее сохранённое состояние процесса',reader_status(c['job_status']['status'])])
    if c.get('original_gate_failure'):
        rows.append(['Первоначальный отказ проверки опорных данных','Сохранён отдельно; несовпадение контрольных сумм при одинаковом физическом содержании'])
    return table('Таблица контактов. Необязательная проверка дискретизации площадок исправленного расчёта',
        ['Характеристика','Результат'],rows)


def contact_check_statement(d):
    c=d['contact_check'];r=c.get('comparison')
    if r is None:return 'Завершённого контактного сравнения нет: '+reader_status(c['status'])+'.'
    text=('Контактное сравнение завершено; допуск '+('выдержан' if r['passed'] else 'не выдержан')+
        ', максимум разности импедансов при непосредственном и унаследованном выборе контактных граней '+
        number(max(r['difference_ohm']),12)+' Ом, превышений '+str(r['failed_values'])+' из 24. ')
    if c.get('original_gate_failure'):
        text+=('Первоначальный отказ возник только при проверке контрольных сумм опорных данных и сохранён отдельно. '+
            'Итоговое сравнение учитывает доказанную эквивалентность этих данных: обе контрольные суммы проверены, '+
            'полные опорные записи совпадают после исключения только трёх полей происхождения. '+
            'Опорные данные, результаты FEM и сведения об их происхождении не переписаны. ')
    return text+'Проверки источников, единиц и происхождения подготовленных материалов не означают прохождения допуска импеданса.'


def contact_check_figure(d,include_js=False):
    import plotly.graph_objects as go
    r=d['contact_check'].get('comparison')
    if r is None:
        from IPython.display import HTML
        return HTML('<p>График контактной проверки отсутствует: завершённый результат не сохранён. Это не прохождение допуска.</p>')
    labels=observation_labels(d);fig=go.Figure()
    fig.add_trace(go.Scatter(x=labels,y=np.asarray(r['difference_ohm'])/np.asarray(r['tolerance_ohm']),
        mode='markers',name='Новый выбор граней: |ΔZ| / допуск',marker=dict(symbol='diamond',color='#8b60aa',size=9)))
    fig.add_trace(go.Scatter(x=labels,y=np.ones(24),mode='lines',name='Граница допуска',line=dict(color='#222222',dash='dash')))
    fig.update_layout(xaxis=dict(title='Запись, ось и состояние',tickangle=-55),yaxis_title='Модуль изменения / допуск',
        height=660,margin=dict(l=85,r=35,t=90,b=180))
    return figure_html(fig,include_js)


def controls_levels_table(d):
    a=d['final'];target=np.asarray(d['best']['target_Z_ohm']);z=np.asarray(d['best']['predicted_Z_ohm'])
    other=[np.asarray(r['fine_Z_ohm']) for r in d['final_forwards']]
    title=('Таблица уровней исправленного расчёта. Выбранный кандидат и копии трёх начальных контролей' if d.get('control_reuse') else
        'Таблица уровней исправленного расчёта. Итоговый поиск и три свежих контроля при его параметрах')
    return table(title,
        ['Наблюдение','Измерение, Ом','Выбранный кандидат, Ом','Родитель, Ом','Наследование, Ом','Повторное интегрирование, Ом'],
        [[label,*[number(v[i],6) for v in (target,z,*other)]] for i,label in enumerate(observation_labels(d))])


def control_reuse_table(d):
    r=d.get('control_reuse')
    if r is None:return table('Таблица исправленного расчёта. Происхождение итоговых контролей',['Характеристика','Результат'],
        [['Подтверждение повторного использования','Нет; итоговые контроли должны иметь собственное подтверждённое происхождение']])
    rows=[['ρ и вся поза равны начальной точке точно','Да'],['Векторы измеренных уровней равны точно','Да'],
        ['Максимум расхождения обратного подбора и прямого расчёта с пересчитанными долями, Ом',number(r['inverse_fixed_forward_maximum_difference_ohm'],12)],
        ['Сохранённый допуск сопоставления, Ом',number(r['allowed_replay_difference_ohm'],12)],
        ['Новый FEM после поиска','Не выполнялся для этой тройки'],
        ['Отрицательные исходные численные результаты','Сохранены'],
        ['Связь с выбранным кандидатом','Проверена по контрольной сумме'],
        ['Исходные опорные данные контролей','Сохранены и проверены']]
    rows.extend([[name+': начальный контроль → итоговая копия','Точные байтовые копии; контрольные суммы совпадают']
        for name in ('Родительская сетка','Сгущённая сетка: унаследованные доли','Сгущённая сетка: пересчитанные доли')])
    return table('Таблица исправленного расчёта. Подтверждение точного повторного использования контролей',['Характеристика','Результат'],rows)


def control_reuse_statement(d):
    r=d.get('control_reuse')
    if r is None:return 'Начальные и итоговые контроли выполнены отдельными тройками. В каждой тройке параметры и поза фиксированы; между тройками они могут различаться.'
    return ('Выбранные ρ и вся поза точно совпали с исходной опорной точкой. Три уже выполненных начальных контроля '+
        'скопированы байт-в-байт как контроли выбранной точки; контрольные суммы исходных данных сохранены. '+
        'Максимум расхождения обратного подбора и прямого расчёта на сгущённой сетке с пересчитанными долями равен '+number(r['inverse_fixed_forward_maximum_difference_ohm'],12)+
        ' Ом при допуске 2·10⁻⁸ Ом. Сохранённое подтверждение связывает копии с выбранным кандидатом, а новый аудит сохраняет '+
        'исходные отрицательные результаты. Это одна выполненная тройка, привязанная к той же выбранной точке, '+
        'без новой тройки FEM-расчётов после поиска. Начатый повторный расчёт на родительской сетке остановлен до получения результата.')


def material_update_table(d):
    r=d['material_update'];fit=r.get('fit');cfg=r.get('config',{})
    rows=[['Состояние отдельной стадии',reader_status(r['status'])],
        ['Сохранённый лимит оценок',cfg.get('profile_max_evaluations','Не сохранён')],
        ['Сохранённый лимит итераций',cfg.get('profile_max_iterations','Не сохранён')],
        ['Готовый проверенный результат подгонки','Да' if fit is not None else 'Нет'],
        ['Начало условного подбора','Одно начальное приближение; новые независимые старты не заявляются'],
        ['Кандидат совместного поиска заменён результатом подгонки','Нет'],['Количественное принятие','Нет']]
    if r.get('job_status'):rows.append(['Последнее сохранённое состояние процесса',reader_status(r['job_status']['status'])])
    if 'fit_file' in r:rows.append(['Файл результата подгонки','Не сохранён' if r['fit_file'] is None else 'Сохранён и проверен'])
    if 'error' in r:rows.append(['Сохранённая ошибка',reader_error(r['error'])])
    if fit is not None:
        rows.extend([['Фактические вычисления целевой функции',fit['evaluations']],
            ['Все старты завершились положительным кодом завершения','Да' if fit['all_starts_converged'] else 'Нет'],
            ['RMSE условного подбора, Ом',number(fit['RMSE_ohm'],9)],
            ['Несовпадения знака из 12',sign_mismatches(d,np.asarray(fit['predicted_Z_ohm']))],
            ['Правило интегрирования целевой функции',str(r['optimization_rule_index'])+'; '+str(r['optimization_rule_points'])+' проб'],
            ['Новая проверка 1024 → 4096 при подобранных ρ','Не выполнена этой стадией'],
            ['ρ точно равны параметрам кандидата совместного поиска','Да' if r['same_joint_rho_exactly'] else 'Нет'],
            ['Новые раздельные численные контроли этого ρ-подбора','Не выполнены/не передаются из контролей совместно выбранной точки'],
            ['Происхождение отдельного результата подгонки','Проверено по контрольной сумме']])
    return table('Таблица отдельного условного ρ-подбора исправленного расчёта',['Характеристика','Результат'],rows)


def material_update_rho_table(d):
    fit=d['material_update'].get('fit');rows=[]
    for label,row in [('Сохранённый кандидат совместного поиска',d['best']),('Отдельный условный подбор ρ',fit)]:
        if row is None:continue
        flags=row.get('bound_parameters')
        for k,name in enumerate(('Мягкотканный фон','Лёгкое','Жир')):
            rows.append([label,name,number(row['rho_inhale_ohm_m'][k],9),number(row['rho_exhale_ohm_m'][k],9),
                'Не сохранён' if flags is None else ('Да' if flags[k] or flags[k+3] else 'Нет')])
    return table('Таблица ρ исправленного расчёта. Условный подбор не заменяет кандидата совместного поиска',
        ['Результат','Материал','Вдох, Ом·м','Выдох, Ом·м','Сохранённый флаг близости к границам'],rows)


def material_update_contrast_table(d):
    fit=d['material_update'].get('fit');target=d['observations'].Z_ohm.to_numpy()
    joint=np.asarray(d['best']['predicted_Z_ohm']) if d['best'] is not None else None
    z=np.asarray(fit['predicted_Z_ohm']) if fit is not None else None
    return table('Таблица контрастов исправленном расчёте. Измерение, совместный и условный подбор',
        ['Запись и ось','Измерение, Ом','Кандидат совместного поиска, Ом','Условный подбор ρ, Ом'],
        [[label,number(target[j]-target[i],6),number(joint[j]-joint[i],6) if joint is not None else 'Нет кандидата',
          number(z[j]-z[i],6) if z is not None else 'Нет готового результата подгонки'] for i,j,label in pairs(d)])


def material_update_figure(d,include_js=False):
    fit=d['material_update'].get('fit')
    if fit is None:
        from IPython.display import HTML
        return HTML('<p>Готового условного ρ-подбора нет; начальное приближение не показано как результат подгонки.</p>')
    view=dict(d);view['best']=fit
    return fit_figure(view,include_js=include_js)


def material_update_text(d):
    r=d['material_update'];fit=r.get('fit')
    text='''## Исправленная постановка с 1×1: отдельный условный ρ-подбор после ограниченного геометрического поиска

Отдельно запущен ограниченный подбор шести фазовых логарифмов сопротивлений методом наименьших квадратов с аналитическим якобианом при сохранённой позе. Лимиты — 4 вычисления целевой функции и 3 итерации, в пределах исходного общего срока. Пропуск профиля перед совместным поиском относится к более раннему плану и не отменяет последующего условного подбора. Все 24 уровня остаются в целевой функции сценария собственного источника. Кандидат совместного поиска, подтверждение повторного использования и численные контроли этой точки сохраняются независимо от условного подбора.

<!-- material_update_table -->
'''
    if fit is None:
        if r['status']=='interrupted_original_time_limit':
            return text+'\nПодбор остановлен по исходному общему сроку; файл результата подгонки и его контрольная сумма отсутствуют. Проверенного промежуточного результата и выбранного вектора ρ нет. Завершённые вычисления метода наименьших квадратов в журнале не задают сохранённого выбранного решения и не позволяют сообщить новый вектор ρ, его RMSE или дыхательные знаки. Сохранённая совместно выбранная точка и её отрицательные численные контроли остаются самостоятельным результатом.\n\n'
        return text+'\nГотовый проверенный результат подгонки отсутствует. Показано последнее сохранённое состояние процесса; без файла результата завершение не предполагается. При остановке по сроку без результата подгонки задача остаётся незавершённой. Начальное приближение не представлено как подобранный вектор ρ.\n\n'
    text+='''
<!-- material_update_rho_table -->
<!-- material_update_contrast_table -->
<!-- material_update_figure -->

**Рисунок отдельного условного ρ-подбора.** Красные ряды — вдох, сине-зелёные — выдох; круги и пунктир — измерение, квадраты и сплошная линия — только условный результат подгонки. Ряды отключаются независимо. Связывающие линии не являются временной интерполяцией.

'''
    text+=('Точность целевой функции — правило '+str(r['optimization_rule_index'])+' модели, '+str(r['optimization_rule_points'])+
        ' проб. Это использованная дискретизация, без отдельного доказательства сходимости интегрирования при новых ρ. '+
        'Код завершения и завершение MATLAB не устанавливают полного оптимума. При изменении ρ прежние '+
        'пространственные, материальные и контактные проверки совместно выбранной точки не являются проверками нового результата подгонки. '+
        'Условная невязка и дыхательные знаки описывают именно этот вектор на статической КТ вдоха, '+
        'без принятия физиологических параметров или глобальной идентифицируемости.\n\n')
    return text


def deep_section_figure(d,include_js=False):
    import sys
    import plotly.graph_objects as go
    from scipy.io import loadmat
    sys.path.insert(0,str(PIPE/'tools'))
    from georg_quality_mesh_section import section_lines
    m=loadmat(d['deep_model']/'fine2/mesh.mat',variable_names=['nodes_mm','elems']);p=d['best']['pose']
    lines=section_lines(m['nodes_mm'],m['elems'].astype(np.int64)-1,
        np.asarray(p['centre_lps_mm']),p['primary_unit_xyz'],-np.asarray(p['normal_unit_xyz']),window=(-80,80,-5,85))
    fig=go.Figure(go.Scattergl(x=lines['x'],y=lines['y'],mode='lines',
        line=dict(width=.6,color='#326ba8'),name='Сечения тетраэдров',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0],y=[0],mode='markers',name='Центр сборок',
        marker=dict(color='#bd3535',size=10,symbol='diamond')))
    fig.update_layout(xaxis=dict(title='Вдоль основной оси от центра, мм',range=[-80,80]),
        yaxis=dict(title='По внутренней нормали от центра, мм',range=[85,-5],scaleanchor='x',scaleratio=1),
        height=780,margin=dict(l=95,r=35,t=80,b=85))
    return figure_html(fig,include_js)


def conditional_figure(d,include_js=False):
    import plotly.graph_objects as go
    seed=np.asarray(read(d['work']/'warm_profile.local.json')['parameters'])
    rows=[read(p) for p in sorted(d['run_directory'].glob('evaluation_*.local.json'))]
    fig=go.Figure()
    for k,name in enumerate(['Фон: вдох','Лёгкое: вдох','Жир: вдох','Фон: выдох','Лёгкое: выдох','Жир: выдох']):
        values=[(np.exp(seed[k]),d['profile']['RMSE_ohm'])]
        other=[j for j in range(9) if j!=k]
        for r in rows:
            if r['status']!='numerical_candidate_not_validated':continue
            x=np.asarray(r['parameters'])
            if np.allclose(x[other],seed[other],rtol=0,atol=1e-9):
                values.append((np.exp(x[k]),np.sqrt(r['mean_square_ohm2'])))
        values=sorted(set(values))
        fig.add_trace(go.Scatter(x=[x for x,_ in values],y=[y for _,y in values],
            mode='markers',name=name,marker=dict(size=9,symbol='circle' if k<3 else 'diamond')))
    fig.update_layout(xaxis=dict(title='Изменяемое сопротивление, Ом·м',type='log'),
        yaxis_title='RMSE всех 24 уровней, Ом',height=620,margin=dict(l=90,r=35,t=110,b=90))
    return figure_html(fig,include_js)


def contrast_linearization(d,row):
    J=np.asarray(row['jacobian_logrho']);p=pairs(d)
    H=np.array([(J[i,:3]+J[j,3:])/2 for i,j,_ in p])
    delta=np.log(row['rho_exhale_ohm_m'])-np.log(row['rho_inhale_ohm_m'])
    contributions=H*delta
    z=np.asarray(row['predicted_Z_ohm']);exact=np.array([z[j]-z[i] for i,j,_ in p])
    return delta,contributions,float(np.max(np.abs(contributions.sum(1)-exact)))


def contrast_contributions_table(d):
    rows=[]
    for label,r in [('Выбранный кандидат исправленного расчёта',d['best'])] if d['confirmed'] else [('Фиксированный центр прежнего расчёта',d['profile']),('Совместный подбор прежнего расчёта',d['best'])]:
        delta,c,error=contrast_linearization(d,r)
        for k,name in enumerate(['Фон','Лёгкое','Жир']):
            rows.append([label,name,number(delta[k],10),number(c[:,k].min(),8),number(c[:,k].max(),8),number(error,8)])
    return table('Таблица 8. Локальное разложение модельного дыхательного изменения',
        ['Результат','Материал','ln(ρ выдоха / ρ вдоха)','Минимальный вклад, Ом',
         'Максимальный вклад, Ом','Максимальная ошибка суммы вкладов, Ом'],rows)


def contrast_projection(d):
    # Conditional diagnostic at the OLD centre, not another inverse solution.
    r=d['profile'];J=np.asarray(r['jacobian_logrho']);p=pairs(d)
    H=np.array([(J[i,:3]+J[j,3:])/2 for i,j,_ in p])
    z=np.asarray(r['predicted_Z_ohm']);t=np.asarray(r['target_Z_ohm'])
    base=np.array([z[j]-z[i] for i,j,_ in p]);measured=np.array([t[j]-t[i] for i,j,_ in p])
    eta=np.linalg.lstsq(H,measured-base,rcond=None)[0]
    projected=base+H@eta;residual=projected-measured;singular=np.linalg.svd(H,compute_uv=False)
    return dict(eta=eta,measured=measured,projected=projected,residual=residual,singular=singular,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_SSE_fraction=float(np.sum(residual**2)/np.sum(measured**2)),
        sign_mismatches=int(np.count_nonzero(np.sign(measured)!=np.sign(projected))))


def projection_table(d):
    r=contrast_projection(d)
    return table('Таблица 10. Неограниченная локальная проекция дыхательного контраста при прежнем центре',
        ['Характеристика','Результат'],[
        ['Формальная поправка к разности логарифмов: фон; лёгкое; жир','; '.join(number(v,6) for v in r['eta'])],
        ['Сингулярные числа матрицы H, Ом','; '.join(number(v,6) for v in r['singular'])],
        ['Отношение крайних сингулярных чисел',number(r['singular'][0]/r['singular'][-1],2)],
        ['RMS остаточного контраста, Ом',number(r['residual_rms'],6)],
        ['Доля необъяснённой суммы квадратов контраста, %',number(100*r['residual_SSE_fraction'],3)],
        ['Несовпадения знака из 12',r['sign_mismatches']],
        ['Допустимость найденной поправки','Вне области локального приближения и технических пределов']])


def search_table(d):
    from datetime import datetime,timezone
    cfg=d['config'];search=d['selected']['search'];rows=[read(p) for p in d['run_directory'].glob('evaluation_*.local.json')]
    budget=(priority_budget_rows(d)
        if d['confirmed'] else [
            ['Прежний расчёт: исходный суммарный предел',cfg.get('initial_joint_evaluation_budget',80)],
            ['Прежний расчёт: начатые оценки первой остановленной попытки',cfg.get('prior_search_started_evaluations',23)],
            ['Прежний расчёт: выделенный остаток исходного лимита',cfg['max_evaluations']]])
    details=[['Завершённые оценки поиска, включая отказы',search['evaluations']],
        ['Отказы контактов/геометрии внутри поиска',sum(r['status']=='invalid_geometry' and r['evaluation']<=search['evaluations'] for r in rows)],
        ['Свежие проверки кандидата после поиска',sum(r['evaluation']>search['evaluations'] and r['status']=='numerical_candidate_not_validated' for r in rows)],
        ['Начатые оценки без результата',sum(r['status']=='forward_calculation_running' for r in rows)],
        ['Остановка по времени','Да' if d['selected'].get('search_stopped_for_time') or search.get('search_stopped_for_time') else 'Не отмечена в сохранённом результате'],
        ['Остановка по квоте','Да' if search.get('budget_reached') else 'Не отмечена в сохранённом результате'],
        ['Полные обходы направлений',search.get('completed_sweeps') if search.get('completed_sweeps') is not None else 'Не сохранены'],
        ['Целевое разрешение поиска достигнуто','Да' if search['resolution_reached'] else 'Нет'],
        ['Срок остановки поиска, UTC',datetime.fromtimestamp(cfg['search_deadline_unix'],timezone.utc).isoformat()],
        ['Общий исходный срок, UTC',datetime.fromtimestamp(cfg['deadline_unix'],timezone.utc).isoformat()],
        ['Смещения центра в касательной плоскости, мм','; '.join(number(v,3) for v in d['best']['pose']['offsets_uv_mm'])],
        ['Угол, °',number(d['best']['pose']['angle_deg'],3)],['Глобальность минимума','Не установлена']]
    if d['confirmed']:
        coverage=search_direction_coverage(d)
        details.extend([['Готовые численные кандидаты поиска с приоритетом геометрии',coverage['valid_points']],
            ['Новые численные кандидаты с изменением ρ',coverage['rho_points']],
            ['Новые численные кандидаты с изменением угла',coverage['angle_points']],
            ['Исходные индексы параметров, изменённых в готовых численных кандидатах',
             ', '.join(map(str,coverage['calculated_indices'])) or 'Нет'],
            ['Начатые индексы, включая отказ и прерванную пробу',', '.join(map(str,coverage['attempted_indices'])) or 'Нет']])
    return table('Таблица поиска '+('исправленного расчёта' if d['confirmed'] else 'прежнего расчёта')+'. Объём и ограничения',
        ['Характеристика','Результат'],budget+details)


def search_direction_coverage(d):
    seed=np.asarray(d['warm_seed']['parameters']);rows=[read(p) for p in d['run_directory'].glob('evaluation_*.local.json')]
    valid=[r for r in rows if r['status']=='numerical_candidate_not_validated']
    def changed(row):return set((np.flatnonzero(np.asarray(row['parameters'])!=seed)+1).tolist())
    calculated=set().union(*(changed(r) for r in valid))
    attempted=set().union(*(changed(r) for r in rows))
    return dict(valid_points=len(valid),rho_points=sum(bool(changed(r)&set(range(1,7))) for r in valid),
        angle_points=sum(9 in changed(r) for r in valid),calculated_indices=sorted(calculated),attempted_indices=sorted(attempted),
        invalid_points=sum(r['status']=='invalid_geometry' for r in rows),
        unfinished_points=sum(r['status']=='forward_calculation_running' for r in rows))


def provenance_table(d):
    rows=[]
    labels={'run_config.local.json':'Исполненная конфигурация','run_config_priority.local.json':'Конфигурация с приоритетом геометрии',
        'run_config.initial.local.json':'Неизменяемая исходная конфигурация','observations.local.csv':'Измеренные уровни',
        'inputs.local.json':'Паспорт входных данных','warm_seed.local.json':'Начальное приближение',
        'selected_candidate.local.json':'Выбранный кандидат','initial_numerical_audit.local.json':'Начальный численный аудит',
        'numerical_audit_selected_v1.local.json':'Аудит выбранной точки','contact_coverage_initial.local.json':'Начальное покрытие контактов',
        'coverage_author_selected.local.json':'Покрытие прежнего положения','contact_coverage_final.local.json':'Итоговое покрытие контактов',
        'author_arm_confirmation_20261014.local.json':'Подтверждение монтажа автором','run_config_joint_only.local.json':'Конфигурация заменённого совместного поиска',
        'controller_before_priority.local.json':'Состояние до смены порядка','explicit_parameter_order_change.local.json':'Явное изменение порядка проверок',
        'selection_controls_reuse.local.json':'Подтверждение повторного использования контролей','result.local.json':'Сведения об остановке условного подбора'}
    if d.get('material_update',{}).get('fit') is not None:labels['result.local.json']='Результат условного подбора'
    for label,trial,config_name in [('Прежняя постановка с 1×4',d['history'],'run_config.local.json'),
            ('Исправленная постановка с 1×1',d,W3_CONFIG)]:
        for file in [trial['work']/config_name,trial['work']/'run_config.initial.local.json',
                trial['input_directory']/'observations.local.csv',trial['input_directory']/'inputs.local.json',
                trial['work']/'warm_seed.local.json',trial['work']/'selected_candidate.local.json',
                trial['work']/'initial_numerical_audit.local.json',trial['work']/'numerical_audit_selected_v1.local.json',
                trial['work']/('contact_coverage_initial.local.json' if trial['confirmed'] else 'coverage_author_selected.local.json'),
                trial['work']/'contact_coverage_final.local.json']:
            rows.append([label,labels[file.name],'Сохранено; реквизиты приведены в техническом паспорте' if file.exists() else 'Не выполнен/не сохранён'])
    file=d['work']/'author_arm_confirmation_20261014.local.json'
    rows.append(['Подтверждение автора',labels[file.name],'Сохранено; реквизиты приведены в техническом паспорте'])
    for name in ('run_config_joint_only.local.json','controller_before_priority.local.json','explicit_parameter_order_change.local.json',
            'selection_controls_reuse.local.json'):
        file=d['work']/name
        rows.append(['Приоритет геометрии / происхождение контролей',labels[file.name],'Сохранено; реквизиты приведены в техническом паспорте' if file.exists() else 'Не выполнен/не сохранён'])
    for name in ('material_update_v1/run_config.local.json','material_update_v1/result.local.json'):
        file=d['work']/name
        rows.append(['Отдельный условный ρ-подбор',labels[file.name],'Сохранено в техническом паспорте' if file.exists() else 'Не выполнено/не сохранено'])
    if d.get('material_update',{}).get('fit') is not None:
        file=d['material_update']['fit_path']
        rows.append(['Отдельный условный подбор ρ','Самостоятельный результат','Сохранено; реквизиты приведены в техническом паспорте'])
    return table('Таблица происхождения. Раздельные идентичности прежнего расчёта и исправленного расчёта',
        ['Постановка','Сохранённые данные','Проверка происхождения'],rows)


def rejected_table(d):
    rows=[]
    meanings={'trkg4:poseOutsideRightTorso':'Выход контакта с правой поверхности',
        'trkg4:poseOutsideCTTorso':'Выход с КТ-поверхности туловища',
        'trkg4:poseOnCTCap':'Недостаточный зазор от искусственного среза',
        'trkg4:patchOverlap':'Перекрытие площадок',
        'electrode_faces_by_area:noSeed':'Не найдена начальная грань площадки',
        'electrode_faces_by_area:patchTooSmall':'Недостаточная площадь выбранных граней'}
    for p in sorted(d['run_directory'].glob('evaluation_*.local.json')):
        r=read(p)
        if r['status']!='invalid_geometry':continue
        q=r.get('patch_qc') or {};c=q.get('contacts',[]) if isinstance(q,dict) else []
        values=[number(max(v[key] for v in c)*scale,3) if c else 'Не рассчитано'
            for key,scale in [('area_error',100),('centroid_offset_mm',1),('extent_mm',1)]]
        rows.append([r['evaluation'],*[number(v,3) for v in r['parameters'][-3:]],
            meanings.get(r.get('failure_code'),'Нарушен допуск дискретной площадки'),*values])
    return table('Таблица 19. Отклонённые положения и качество площадок',
        ['Оценка','Первое смещение, мм','Второе смещение, мм','Угол, °','Основание отказа',
         'Максимальная ошибка площади, %','Максимальное смещение центроида, мм','Максимальная протяжённость, мм'],rows)


def span_table(d):
    if d['best']['pose'].get('nominal_spacing_is_geodesic') is not False:
        raise ValueError('Projected chord table requires the explicit tangent/projection convention')
    rows=[]
    for block in d['best']['pose']['electrodes']:
        xyz=np.asarray(block['centres_lps_mm'],float)
        rows.append([block['size_mm'],'Продольная' if block['axis']=='longitudinal' else 'Поперечная',
            number(np.linalg.norm(xyz[3]-xyz[0]),6),number(np.linalg.norm(xyz[2]-xyz[1]),6)])
    return table('Таблица расстояний. Хорды между расчётными центрами после проекции',
        ['Номинальный размер, мм','Ось','I+ → I−, хорда, мм','U+ → U−, хорда, мм'],rows)


HISTORICAL_FUNCTIONS=('mesh_table','coverage_table','numerical_table','numerical_figure','stages_table',
    'rho_table','fit_figure','residual_table','respiratory_figure','respiratory_table','history_figure',
    'sensitivity_figure','sensitivity_table','repeated_table','source_table','coordinate_table',
    'conditional_figure','contrast_contributions_table','projection_table','search_table','rejected_table','span_table')


def historical_view(function):
    def view(d,**kwargs):
        result=function(d['history'],**kwargs)
        if function.__name__.endswith('table'):
            return rename(result,'Таблица ','Таблица прежнего расчёта · ')
        return result
    return view


for _name in HISTORICAL_FUNCTIONS:
    globals()['historical_'+_name]=historical_view(globals()[_name])


def sign_mismatches(d,z):
    target=np.asarray(d['best']['target_Z_ohm'])
    return sum(np.sign(z[j]-z[i])!=np.sign(target[j]-target[i]) for i,j,_ in pairs(d))


def incomplete_status_table(d):
    state=d['state'];rows=[['Состояние основной процедуры',reader_status(state['status'])],
        ['Сохранённая причина',reader_error(state.get('error_type','')+' '+state.get('error',''))],
        *priority_budget_rows(d),
        ['Завершённые промежуточные численные результаты',sum(r['status']=='numerical_candidate_not_validated' for _,r in d['checkpoints'])],
        ['Отказы геометрии',sum(r['status']=='invalid_geometry' for _,r in d['checkpoints'])],
        ['Начатые оценки без результата',sum(r['status']=='forward_calculation_running' for _,r in d['checkpoints'])],
        ['Целевое разрешение','Не подтверждается незавершённой процедурой'],
        ['Количественное принятие','Нет'],['Глобальная сходимость/валидация','Не установлены']]
    for label,files in [('Начальный',d['forwards']),('Итоговый',d['final_forwards'])]:
        for name,row in zip(('Родитель','Точная наследуемая доля','Повторное интегрирование'),files,strict=True):
            rows.append([label+': '+name,('Точная байтовая копия начального контроля' if label=='Итоговый' and d.get('control_reuse') else 'Сохранён свежий расчёт') if row is not None else 'Не выполнен/не сохранён'])
    for label,audit in [('Начальный',d['initial']),('Итоговый',d['final'])]:
        rows.append([label+': раздельный численный аудит','Не выполнен/не опубликован' if audit is None else
            ('Сохранён; все допуски выдержаны' if audit['all_checks_passed'] else 'Сохранён; не все допуски выдержаны')])
    rows.append(['Покрытие итоговых контактов','Нет итоговой проверки' if d['coverage'] is None else
        ('Полное' if d['coverage']['all_actual_contact_faces_refined'] else 'Неполное')])
    rows.append(['Покрытие начальных фактических площадок','Нет начальной проверки' if d['initial_coverage'] is None else
        ('Полное' if d['initial_coverage']['all_actual_contact_faces_refined'] else 'Неполное')])
    if d['candidate_file'] is not None:
        rows.extend([['Источник доступного кандидата','Сохранённый результат исправленной постановки'],
            ['Происхождение доступного кандидата','Проверено по контрольной сумме'],
            ['Кандидат является принятым решением','Нет']])
    return table('Таблица незавершённого исправленного расчёта. Фактически сохранённые стадии', ['Характеристика','Результат'],rows)


def incomplete_numerical_table(d):
    rows=[]
    for label,audit in [('Начальное приближение',d['initial']),('Итоговый кандидат',d['final'])]:
        for key,name in [('spatial','Пространство FEM'),('fractions','Тканевые доли'),('total','Суммарное изменение')]:
            if audit is None:rows.append([label,name,'Не выполнен/не опубликован','—','Не установлено'])
            else:
                r=audit[key];rows.append([label,name,number(max(r['difference_ohm']),7),r['failed_values'],'Да' if r['passed'] else 'Нет'])
    return table('Таблица незавершённого исправленного расчёта. Начальные и итоговые сравнения раздельно',
        ['Точка','Сравнение','Максимум |ΔZ|, Ом','Превышения из 24','Допуск выдержан'],rows)


def initial_numerical_figure(d,include_js=False):
    return numerical_figure(d,key='initial',include_js=include_js)


def incomplete_levels_table(d):
    vectors=[d['best']['predicted_Z_ohm'] if d['best'] is not None else None]
    vectors.extend(r['fine_Z_ohm'] if r is not None else None for r in (*d['forwards'],*d['final_forwards']))
    return table('Таблица незавершённого исправленного расчёта. Только реально сохранённые 24 уровня',
        ['Наблюдение','Измерение, Ом','Доступный кандидат, Ом','Начальный родитель, Ом','Начальная сгущённая сетка: унаследованные доли, Ом',
         'Начальная сгущённая сетка: пересчитанные доли, Ом','Итоговый родитель, Ом','Итоговая сгущённая сетка: унаследованные доли, Ом','Итоговая сгущённая сетка: пересчитанные доли, Ом'],
        [[label,number(d['observations'].Z_ohm.iloc[i],6),
          *[number(v[i],6) if v is not None else 'Не рассчитано' for v in vectors]] for i,label in enumerate(observation_labels(d))])


def incomplete_contrast_table(d):
    target=d['observations'].Z_ohm.to_numpy();vectors=[d['best']['predicted_Z_ohm'] if d['best'] is not None else None]
    vectors.extend(r['fine_Z_ohm'] if r is not None else None for r in (*d['forwards'],*d['final_forwards']))
    return table('Таблица незавершённого исправленного расчёта. Все 12 контрастов из сохранённых векторов',
        ['Запись и ось','Измерение, Ом','Доступный кандидат, Ом','Начальный родитель, Ом','Начальная сгущённая сетка: унаследованные доли, Ом',
         'Начальная сгущённая сетка: пересчитанные доли, Ом','Итоговый родитель, Ом','Итоговая сгущённая сетка: унаследованные доли, Ом','Итоговая сгущённая сетка: пересчитанные доли, Ом'],
        [[label,number(target[j]-target[i],6),
          *[number(v[j]-v[i],6) if v is not None else 'Не рассчитано' for v in vectors]] for i,j,label in pairs(d)])


def incomplete_text(d):
    text='''## Исправленная постановка с 1×1: незавершённый расчёт после явной остановки с ошибкой

Исправленный расчёт завершился с ошибкой или по исходному лимиту времени. Это самостоятельная неполная диагностика подтверждённого монтажа 1×1, без количественного принятия. Прежняя постановка с 1×4 выше сохраняется как расчёт с ошибочными граничными условиями. В исправленной постановке фиксированный профиль пропущен до начала его вычисления. Старая квота 23+57, первоначальная отдельная квота исправленного расчёта, начатые оценки остановленной попытки совместного поиска без предварительного профиля и новый лимит поиска с приоритетом геометрии показываются раздельно. Порядок проверок параметров — [7, 8, 9, 1, 4, 2, 5, 3, 6]; физические параметры возвращаются в исходные индексы, порядок по умолчанию 1:9 сохранён. Три начальных контроля завершены до этого перехода, без новой маркировки их исполнения. В доступном времени новый поиск мог не закончить даже один полный обход направлений.

<!-- incomplete_status_table -->
<!-- incomplete_numerical_table -->
<!-- initial_controls_table -->
<!-- coverage_table -->

Начальные проверки формы, наследования энергии, интегрирования, производных и систем рассматриваются отдельно от допусков изменения импеданса. Прохождение предварительных критериев позволяет исследовательское продолжение; оно не отменяет отрицательных пространственных/материальных сравнений и не устанавливает сеточной сходимости. Отсутствующий итоговый аудит не восстановлен из начальных контролей или прежнего расчёта. Количественное принятие не заявляется независимо от наличия отдельных численных результатов. Если тройка расчётов сохранена, но официальный аудит не опубликован, в таблице уровней показаны сами векторы; прохождение раздельных проверок не заявляется.

'''
    text+=initial_coverage_statement(d)+' Начальное покрытие не заменяет отсутствующую итоговую проверку: новая поза может выйти из области сгущения.\n\n'
    if d['forwards'][0] is not None:
        bc=boundary_correction(d)
        text+=f'''Изолированное исправление граничных условий проверено по свежему начальному родителю исправленного расчёта и выбранному родителю прежнего расчёта: одинаковые ρ, поза, сетка и числовые наблюдения. Максимум |ΔZ| = **{number(bc['maximum'],10)} Ом**, для остальных 16 наблюдений — **{number(bc['other_maximum'],12)} Ом**; проверка ≤10⁻⁸ Ом **{'пройдена' if bc['other_passed'] else 'не пройдена'}**. RMSE фиксированного родителя: **{number(bc['old_rmse'],10)} → {number(bc['new_rmse'],10)} Ом**. Само изменение граничных условий в этой точке не объясняет большую остаточную невязку и не доказывает неисправность прибора; реальные токи и фазы не установлены.

<!-- boundary_correction_table -->
<!-- boundary_correction_figure -->

**Рисунок незавершённого исправленного расчёта 1. Изолированное изменение граничных условий.** Синие круги — исправленный расчёт минус прежний, оранжевые квадраты — модуль; чёрный пунктир — ноль. Ряды отключаются независимо. Это сравнение фиксированного родителя, без смешивания с новым сгущением или подбором.

'''
    else:text+='Свежий начальный родитель исправленного расчёта отсутствует: изолированное сравнение граничных условий не выполнено.\n\n'
    if d['initial'] is not None:
        text+='''<!-- initial_numerical_figure -->

**Рисунок незавершённого исправленного расчёта 2. Только начальная тройка.** Синие столбцы — пространство FEM, оранжевые — доли, фиолетовые — суммарное изменение; чёрный пунктир на уровне 1 — граница допуска. Высота — |ΔZ|/допуск. Все три расчёта относятся к перенесённому начальному приближению, а не к новому итоговому кандидату.

'''
    if d['best'] is not None:
        r=components(d,np.asarray(d['best']['predicted_Z_ohm']))
        label='Сохранённый выбранный кандидат' if d['selected'] is not None else 'Лучший доступный свежий промежуточный результат'
        text+=f'''**{label}** имеет RMSE **{number(r[0],7)} Ом**, RMS ошибки среднего уровня **{number(r[1],6)} Ом**, RMS ошибки половины контраста **{number(r[2],6)} Ом**. Это фактически рассчитанный вектор исправленного расчёта, без объявления завершённого оптимума. Несовпадений дыхательного знака — **{sign_mismatches(d,np.asarray(d['best']['predicted_Z_ohm']))} из 12**; короткий или прерванный поиск не доказывает глобальной невозможности нужного контраста или неидентифицируемости модели.

<!-- fit_figure -->
<!-- rho_table -->
<!-- coordinate_table -->
<!-- span_table -->

**Рисунок незавершённого исправленного расчёта 3. Доступный кандидат.** Красные ряды — вдох, сине-зелёные — выдох; пунктир и круги — измерение, сплошная линия и квадраты — только доступный кандидат исправленного расчёта. Это не результат отсутствующих итоговых контролей. Таблица расстояний показывает расчётные хорды после касательной конструкции и проекции, без измерения дуг; номинальный размер не задан как геодезическая длина по поверхности. Более короткая хорда на кривой коже сама по себе не является доказанной ошибкой.

'''
    else:text+='Нового рассчитанного кандидата исправленного расчёта нет. Перенесённые параметры остаются только начальным приближением, без подстановки старых предсказаний.\n\n'
    text+='''<!-- incomplete_levels_table -->
<!-- incomplete_contrast_table -->
<!-- contact_check_table -->
<!-- contact_check_figure -->

**Необязательный рисунок незавершённого исправленного расчёта 4. Контактные грани.** Используется самостоятельная проверка контактов исправленной постановки; состояние заменённой до выполнения расчёта попытки не принимается за итоговый результат. При завершённом сравнении фиолетовые ромбы — |ΔZ|/допуск выбора граней на том же объёме; чёрный пунктир на уровне 1 — граница. При отсутствии результата показано сохранённое состояние; график и прохождение допуска не заявляются. Итоговые пространственные, материальные и контактные проверки, которых нет, остаются невыполненными. Начальные проверки не перенесены на итоговую точку. Статическая КТ глубокого вдоха для обеих фаз остаётся допущением; локальная J-проекция исторической точки не даёт глобального доказательства для исправленного расчёта.

'''
    text+=contact_check_statement(d)+'\n\n'
    return text


def historical_text(d):
    h=d['history'];hr=components(h,np.asarray(h['best']['predicted_Z_ohm']));projection=contrast_projection(h)
    rows=d['rays']['contacts'];depths=[v['first_lung_depth_mm'] for v in rows if v['first_lung_depth_mm'] is not None]
    fraction=next(v for v in d['deep_distribution']['rules'] if v['points']==4096)
    old_fraction=next(v for v in d['distribution']['rules'] if v['points']==4096)
    return fr'''# Сгущение у всех контактов и продолжение подбора: Георг, Scorpion v2

В отчёте сопоставлены **прежняя постановка с 1×4 для записей 007–010** и **исправленная постановка с подтверждённым автором монтажом 1×1**. Рисунки обозначены отдельными последовательностями для прежнего и исправленного расчётов. Конечноэлементная модель (FEM) использует полную электродную модель (CEM) с конечными контактными площадками. Среднеквадратическая невязка обозначена RMSE, сумма квадратов ошибок — SSE; RMS обозначает среднеквадратический уровень отдельной составляющей ошибки. Система координат LPS задаёт направления влево, назад и вверх; координаты приведены в миллиметрах. Численные результаты прежнего расчёта сохранены как диагностика ранее выполненного расчёта. Глубинная геометрическая подготовка повторно используется в исправленном расчёте, а начальные электрические расчёты и подбор исправленного расчёта имеют собственные исправленные входы и подтверждённое происхождение. Ниже указано, выполнены ли контроли отдельно или повторно использованы при точно совпавшей точке.

Подтверждённый автором факт: в записях 006 и 011 все провода рук были сняты; в 007–010 использована ТТРКГ 1×1. Сохранение самих физических площадок рук остаётся допущением. При снятых проводах каждая присутствующая площадка моделируется отдельно плавающей, с нулевым внешним током. Альтернативные соединения рук в этом отчёте не проверяются.

Измеренные 24 уровня, их порядок, роли каналов и интервалы обработки между прежним и исправленным расчётами сохранены. Исправлено поле монтажа рук только для 007–010. Подтверждение соединений не является принятием дыхательной разметки: паспорт сохраняет её статус кандидата для ручного просмотра. Прибор не записывал временные метки дыхательных команд; экспертная разметка не становится непосредственно измеренным событием.

Рабочая постановка использует три удельных сопротивления — мягкотканного фона, лёгкого и жира — отдельно для вдоха и выдоха, при общем положении сборок. Материалы костей и целого сердца фиксированы. **Одна статическая КТ на глубоком вдохе используется для обеих фаз**; деформация анатомии при выдохе не моделируется. Сопоставимость состояния и положения в последовательных записях — рабочие допущения. Собственный источник задаёт выбранный расчётный сценарий; физическая фаза и отношение токов двух источников непосредственно не установлены.

## Материалы и технические границы

<!-- fixed_properties_table -->
<!-- contact_properties_table -->

Контактные площадки CEM заданы круглыми дисками принятого диаметра 5 мм; диаметр не измерен. Номинальная площадь π(d/2)² рассчитана по этому диаметру. Фактическая площадь — сумма площадей выбранных граничных граней, а не независимо измеренная площадь контакта с кожей. Диапазоны в таблице раздельно относятся к итоговой проверке покрытия выбранной точки и начальному контролю родительской сетки. Удельный контактный импеданс фиксирован в планах трёх моделей и принят как допущение, без индивидуального измерения. Сохранённые 168 описаний относятся к десяти условиям записи с повторяющимися площадками; это не 168 уникальных физических электродов и не число одновременно установленных контактов.

Фиксированные свойства кости и сердца взяты из сохранённого файла материалов глубинной модели при частоте 100000 Гц (100 кГц). Вся маска кости представлена кортикальной костью, а вся маска сердца — миокардом; кровь и миокард внутри неё отдельно не моделируются. Это модельные допущения, а не независимые индивидуальные референсы. Количественное влияние этих фиксированных свойств не устанавливается без отдельных производных по ним.

Опорное сопротивление мягкотканного фона принято как для мышцы. При этом фон объединяет весь объём модели вне масок лёгкого, кости, сердца и жира; мышечное опорное значение не подтверждает однородности этого объёма. Свободные фазовые параметры имеют индексы 1 — мягкотканный фон, 2 — лёгкое, 3 — жир; в полном порядке пяти материалов им соответствуют позиции 1, 2 и 5. Они подбираются отдельно для вдоха и выдоха. Диапазон 0,1–100 Ом·м является технической областью поиска, а не физиологической нормой; индивидуальные сопротивления тканей не измерены независимо.

Все рисунки имеют самостоятельные средства просмотра и отключения рядов. Трёхмерный рисунок сохраняет камеру при переключении легенды. Таблицы раскрываются отдельно. Линии между номерами наблюдений служат сравнению дискретных записей и не являются временной интерполяцией.

## Прежняя постановка с 1×4: диагностика граничных условий

Исходный совместный лимит был 80 оценок: 23 начатые оценки первой попытки сохранены отдельно, затем для прежнего расчёта выделены 57. Это история исходного лимита; она не включает новую квоту исправленного расчёта. Старый фиксированный профиль при прежнем центре завершился с сохранённым числом {h['profile']['evaluations']} вычислений целевой функции и RMSE **{number(h['profile']['RMSE_ohm'],7)} Ом**. По сохранённым сведениям о длительности он занял около 94 минут, а поиск с лимитом 57 — около 215 минут; эти длительности описывают ресурсное ограничение, а не математическую сходимость.

Для старого продолжения одновременно менялись технические границы сопротивлений до 0,1–100 Ом·м и способ оптимизации. Снижение невязки относительно прежней постановки не устанавливает причинного эффекта оптимизатора отдельно от расширения границ. Эти пределы не являются индивидуально валидированными физиологическими нормами. Представление `exp(log(100)) = 100.00000000000004` отличается от 100 только округлением арифметики с плавающей точкой; оно не означает расширения технической границы. Числа в сохранённых результатах не подменяются. Для читательской отметки близости к техническим границам 0,1 и 100 Ом·м принят абсолютный допуск 10⁻⁶ Ом·м без относительного допуска. Он используется только в подписи таблицы: параметры, их границы и результаты расчётов не изменяются. Попадание параметра на границу не даёт его двусторонней оценки или доверительного интервала.

<!-- historical_mesh_table -->
<!-- historical_coverage_table -->
<!-- historical_numerical_table -->
<!-- historical_numerical_figure -->

**Рисунок прежнего расчёта 1. Численные контроли выбранной точки прежнего расчёта.** Синие столбцы — изменение пространства FEM при точном наследовании долей; оранжевые — повторное интегрирование масок на той же сетке; фиолетовые — суммарное изменение родитель → сгущённая сетка с пересчитанными долями. Высота — |ΔZ|, делённое на сохранённый допуск 0,02 + 0,005|Z правого расчёта| Ом. Чёрный пунктир на уровне 1 — граница допуска. Параметры и координаты в тройке фиксированы, без новой оптимизации. Итоговые допуски прежнего расчёта **{'выдержаны' if h['final']['all_checks_passed'] else 'выдержаны не все'}**; завершение процедуры само по себе не разрешает количественную интерпретацию.

<!-- historical_stages_table -->
<!-- historical_rho_table -->
<!-- historical_fit_figure -->

**Рисунок прежнего расчёта 2. Все уровни при ошибочных граничных условиях.** Красные ряды — вдох, сине-зелёные — выдох; пунктир и круги — измерение, сплошная линия и квадраты — модель. По горизонтали — запись, размер и ось сборки. RMSE выбранной точки прежнего расчёта равна **{number(hr[0],7)} Ом**.

<!-- historical_residual_table -->
<!-- historical_respiratory_figure -->

**Рисунок прежнего расчёта 3. Контраст ΔZ = Z выдоха − Z вдоха.** Чёрные круги — измерение; серые квадраты — прежний кандидат; синие — новая поверхностная сетка с прежними параметрами; фиолетовые — фиксированный профиль; оранжевые — совместный кандидат прежнего расчёта. Чёрный пунктир — нулевой контраст. Знак не совпадает в **{sign_mismatches(h,np.asarray(h['best']['predicted_Z_ohm']))} из 12 пар** для выбранной точки прежнего расчёта с ошибочными граничными условиями.

<!-- historical_respiratory_table -->
<!-- historical_contrast_contributions_table -->
<!-- historical_projection_table -->

Индексы $i$ и $e$ обозначают вдох и выдох соответственно; верхние индексы $model$ и $obs$ — модель и наблюдение. Для ошибок пары $e_i=Z_i^{{model}}-Z_i^{{obs}}$, $e_e=Z_e^{{model}}-Z_e^{{obs}}$ введены $a=(e_i+e_e)/2$ и $b=(e_e-e_i)/2$. Тогда $RMSE^2=mean_{{12}}(a^2)+mean_{{12}}(b^2)$. Для выбранной точки прежнего расчёта RMS ошибки среднего уровня равна **{number(hr[1],6)} Ом**, RMS ошибки половины контраста — **{number(hr[2],6)} Ом**; вклад последней в SSE — **{number(hr[3],4)} %**. Абсолютные уровни и парный дыхательный контраст анализируются отдельно.

Локальная проекция относится только к **старому фиксированному профилю прежнего расчёта**. Для каждой пары $H=(J_{{i,\rho_i}}+J_{{e,\rho_e}})/2$, столбцы — производные по логарифмам трёх сопротивлений. Использована фактическая модельная разность $\Delta Z_0$ и формальная поправка $\eta=\arg\min\|H\eta-(\Delta Z_{{obs}}-\Delta Z_0)\|^2$. Проекция — $\Delta Z_0+H\eta$; остаток — проекция минус измеренный контраст. RMS остатка равна **{number(projection['residual_rms'],6)} Ом**. Большой логарифмический шаг выходит из области локального приближения; формальный результат не является новым физическим решением. Локальный якобиан не доказывает глобальной несовместимости общего состояния и не переносится автоматически на новый центр исправленного расчёта.

<!-- historical_repeated_table -->
<!-- historical_source_table -->

Повторные 005 → 006 и 010 → 011 сохраняют боковую сборку, но меняют подключение соседних рук и условия регистрации. Разность в таблице — запись креста минус отдельная запись, отдельно для обеих фаз. Она проверяет совместимость предположения об общем состоянии с конкретной схемой возбуждения, без доказательства дрейфа, аппаратной неисправности или иной причины. Сценарии источников вычислены из сохранённых векторов передаточного импеданса при тех же параметрах: α = 0, +1, −1; 90° показаны как $\sqrt{{Z_1^2+Z_2^2}}$, где $Z_1$ и $Z_2$ — вклады собственного и встречного источников. Это модуль при принятой квадратуре. Ограниченный вещественный α — только диагностическая проекция. Ни один сценарий не устанавливает реальную фазу или способ регистрации сигнала прибора.

<!-- historical_sensitivity_table -->
<!-- historical_sensitivity_figure -->

**Рисунок прежнего расчёта 4. Якобиан шести логарифмов ρ.** Цвет ячейки — знаковая производная импеданса, Ом, по логарифму параметра; строки — 24 наблюдения, столбцы — три материала отдельно для вдоха и выдоха. Сингулярные числа характеризуют локальную информативность этой точки, без экспериментальной валидации или полного анализа положения.

<!-- historical_conditional_figure -->

**Рисунок прежнего расчёта 5. Сохранённые однопараметрические пробы.** Круги — вдох, ромбы — выдох; цвет и название ряда указывают материал. По горизонтали — изменяемое ρ в логарифмической шкале, по вертикали — RMSE 24 уровней. Включены только пробы, у которых остальные восемь параметров совпадают с началом старого совместного поиска. Это не завершённые профили правдоподобия.

<!-- historical_history_figure -->

**Рисунок прежнего расчёта 6. Ход поиска.** Серые точки — рассчитанные кандидаты, синяя линия — лучший достигнутый RMSE. Зелёные открытые ромбы, если присутствуют, — свежие проверки после поиска; красные кресты — отказ геометрии на условной высоте максимального рассчитанного RMSE, без собственного значения целевой функции. Номер оценки не является временем или номером независимого опыта.

<!-- historical_search_table -->
<!-- historical_rejected_table -->

## Подготовка глубинной сетки и остановленные проверки прежнего расчёта

Первая подготовка глубинной сетки завершилась **отказом по качеству**: минимум показателя формы в трёх сохранённых попытках оказался ниже 0,01. Лимиты не ослаблялись. Следующая подготовка дала пригодную по форме сетку с **{d['deep_quality']['nodes']} узлами и {d['deep_quality']['tetrahedra']} тетраэдрами**, минимум безразмерного показателя формы **{number(d['deep_quality']['mean_ratio_minimum'],9)}**. Её геометрия и доли использованы как входы исправленного расчёта.

<!-- preparation_attempts_table -->
<!-- deep_quality_table -->

Дальнейшие электрические проверки прежней постановки остановлены после уточнения автором монтажа рук. Отсутствующие проверки глубинного сгущения и непосредственного выбора контактных граней в прежней постановке не трактуются как пройденные и не требуются для чтения подготовки. Качество формы, положительные объёмы и наследование геометрии не устанавливают достаточности FEM-дискретизации: это отдельные численные сравнения исправленного расчёта ниже.

<!-- fraction_distribution_table -->

На поверхностной сетке прежнего расчёта при 4096 пробах изменились доли в **{old_fraction['changed_elements']}** элементах; изменения затрагивали фон и жир, без изменений лёгкого, костей и сердца. Этот узкий охват нельзя переносить на глубинную сетку: здесь изменены **{fraction['changed_elements']}** элементов, в том числе лёгкое — **{fraction['changed_elements_by_material'][1]}**, кость — **{fraction['changed_elements_by_material'][2]}**, фон — **{fraction['changed_elements_by_material'][0]}**, жир — **{fraction['changed_elements_by_material'][4]}**; целое сердце — **{fraction['changed_elements_by_material'][3]}**. Счётчики по материалам пересекаются и не суммируются как независимые элементы. Доли сравниваются на одной объёмной сетке, без изменения координат. Это дискретизация состава тканей, без электрического разложения вкладов компартментов и без независимой проверки сегментации.

<!-- depth_table -->
<!-- depth_figure -->

**Рисунок прежнего расчёта 7. КТ-профили старого положения прежнего расчёта.** Квадраты обозначают материал проб вдоль принятой внутренней нормали: бежевый — фон, синий — лёгкое, серый — кость, красный — целое сердце, жёлтый — жир, чёрный — вне поля КТ. Открытый ромб — первое попадание в маску лёгкого. Из {len(rows)} лучей в {len(depths)} лёгкое найдено на глубине {number(min(depths),1)}–{number(max(depths),1)} мм; для остальных нет попадания на первых 80 мм. Это не отсутствие лёгкого во всём чувствительном объёме. Лучи относятся к подготовке при прежних координатах прежнего расчёта; при изменении позы исправленного расчёта они не переименовываются в новые итоговые КТ-профили. Геометрическое сгущение по лучам не является картой электрической чувствительности.


'''


def body_text(d):
    body=d['body'];forward=body['distances']['original_to_regularized'];reverse=body['distances']['regularized_to_original']
    worst=forward['local_vertex_probe']['worst_locations'][0];other=reverse['local_vertex_probe']['worst_locations'][0]
    worst_point='; '.join(number(v,6) for v in worst['source_lps_mm'])
    other_point='; '.join(number(v,6) for v in other['source_lps_mm'])
    component=next(v for v in body['body']['components_26'] if v['component']==worst['changed_component_26'])
    box=component['bbox_lps_mm'];box_text=', '.join(axis+' '+number(box[0][k],6)+'…'+number(box[1][k],6)+' мм' for k,axis in enumerate(('x','y','z')))
    return fr'''## Изменение КТ-контура в общей подготовке

Общий [порядок подготовки](../../docs/CT_FEM_PREPARATION_STANDARD.md) и [скрипт](../../tools/prepare_ct_fem.py) относятся к развитию воспроизводимой методики. Наличие общего интерфейса и малых программных проверок не означает универсальной количественной применимости моделей. Ранее исполненные исходники и отдельные паспорта сохраняются.

Для КТ Георга 512 × 512 × 280 с шагом 0,878906 × 0,878906 × 1,25 мм исходная HU-маска и результат одного шестисоседнего замыкания воспроизвели ранее использованные воксельные маски. Изменено {body['body']['changed_voxels']} вокселей. Пилотная подготовка поверхности по реальной КТ с пределом 1,25 мм дала отрицательный результат: раньше этот предел контролировал экспорт **уже исправленного** контура, а в общем интерфейсе первоначально был перенесён на сравнение **до → после топологической коррекции**. Это разные операции и пары поверхностей. Предел нельзя автоматически расширять ради прохождения допуска; требуется обоснованный явный геометрический сценарий.

<!-- body_correction_table -->
<!-- body_correction_figure -->

**Рисунок подготовки геометрии 1. Локальные сечения КТ.** Синие контуры — до замыкания, оранжевые — после; красные кресты — точки локальной пробы вершин. Верхний ряд — до → после, нижний — после → до. Показаны три ближайших воксельных сечения; плоскости могут не проходить точно через трёхмерную точку. Контуры сечений построены на уровне 0,5, измеренные поверхности — на уровне 0,49. Индексы отсчитываются от нуля; координаты — LPS, мм. Анатомическое принятие коррекции не заявляется.

До → после: выборочный максимум **{number(forward['max_mm'],9)} мм**, локальные вершины **{number(forward['local_vertex_probe']['max_mm'],9)} мм** в точке **[{worst_point}] LPS мм**. После → до: выборочный максимум **{number(reverse['max_mm'],9)} мм**, локальный **{number(reverse['local_vertex_probe']['max_mm'],9)} мм** в точке **[{other_point}] LPS мм**. Самое сильное проверенное изменение связано с компонентой изменённых вокселей №{worst['changed_component_26']} при 26-соседней связности, рядом с вокселем {worst['nearest_changed_voxel']}. Эта компонента содержит {component['added']} добавленных вокселей в верхней части объёма КТ: её LPS-границы {box_text}; это пространственная область, без анатомического диагноза. Это локализация по КТ, без анатомического принятия. Максимумы разных наборов проб не являются глобальным расстоянием Хаусдорфа. Замороженная FEM-геометрия при диагностике не изменялась.

## Происхождение и ограничения

<!-- provenance_table -->

Входы прежнего расчёта остаются неизменными; изменяемые после расчёта программные файлы проверяются по сохранённым контрольным суммам и архивам точных исполненных байтов. Для исправленного расчёта проверяются собственные наблюдения и паспорт, опорные данные, параметры и координаты, планы моделей, подготовленные материалы и источники контрольных файлов. Исторические и новые выборки не переименовываются друг в друга. Исходная конфигурация, конфигурация остановленной попытки совместного поиска без предварительного профиля и фактическая конфигурация с приоритетом геометрии показываются раздельно.

Это отчёт о разработке методики. Ни завершение процедуры, ни качество сетки, ни малая невязка одной точки не устанавливают независимой валидации тканей, глобальной сеточной сходимости, идентифицируемости всех параметров или функций сердца. Технические границы 0,1–100 Ом·м сохранены; автоматического расширения нет. Дыхательная разметка, физическое возбуждение, влияние контактов и статическая геометрия для двух фаз остаются отдельными вопросами. Отрицательные контроли и неполные проверки сохраняются в выводах.
'''


def text(d):
    if d.get('incomplete'):return historical_text(d)+incomplete_text(d)+material_update_text(d)+body_text(d)
    b=d['best'];z=np.asarray(b['predicted_Z_ohm']);r=components(d,z);bc=boundary_correction(d)
    centre='; '.join(number(v,4) for v in b['pose']['centre_lps_mm'])
    singular=np.asarray(b['singular_values']);condition=float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf
    search=d['selected']['search'];contact=d['contact_check'];directions=search_direction_coverage(d)
    contact_text=contact_check_statement(d)
    return historical_text(d)+fr'''## Исправленная постановка с 1×1: изолированный эффект исправления граничных условий

Перед новым поиском повторён расчёт на **том же родителе**, при точно тех же ρ и позе, что в выбранной точке прежнего расчёта. Сопоставлены выбранный расчёт прежней постановки на родительской сетке и начальный расчёт исправленной постановки на той же сетке; используется сохранённый результат 4096 проб. Маски, объёмная сетка, исходный каталог физических контактов и числовые наблюдения совпадают. Единственное изменённое поле входов — 1×4 → 1×1 для 007–010; условия раздельно плавающих площадок при крестах уже использовались в старом расчёте.

<!-- boundary_correction_table -->
<!-- boundary_correction_figure -->

**Рисунок исправленного расчёта 1. Исправление соединений на неизменном родителе.** Синие круги показывают знаковую разность исправленного и прежнего расчётов, оранжевые квадраты — её модуль; чёрный пунктир — нулевое изменение. Каждый ряд отключается независимо. Максимум модуля изменения — **{number(bc['maximum'],10)} Ом**. Для остальных 16 наблюдений максимум — **{number(bc['other_maximum'],12)} Ом**; проверка ≤10⁻⁸ Ом **{'пройдена' if bc['other_passed'] else 'не пройдена'}**. RMSE на фиксированном родителе изменяется **{number(bc['old_rmse'],10)} → {number(bc['new_rmse'],10)} Ом**.

Это изолирует влияние подтверждённых граничных условий от глубинного сгущения и оптимизации. Само исправление соединений в этой точке не объясняет большую остаточную невязку собственного источника. Оно не доказывает ошибку прибора: реальные фазы и токи не установлены. Разность итоговых RMSE прежнего и исправленного расчётов включает другие изменения и не является изолированным эффектом граничных условий.

## Исправленная постановка с 1×1: самостоятельный ограниченный совместный поиск

Для соблюдения исходного срока профиль сопротивлений при фиксированном центре пропущен **до выполнения**. Исходная конфигурация сохранена отдельно; её план профиля не выдается за исполненную стадию. Унаследованные сведения о лимитах 80 и 23 оценок и прежнем начальном приближении относятся к истории расчётов. Три начальных контроля завершены до смены порядка и сохранены с их прежними сведениями о происхождении; они не переименовываются в новые расчёты. Новый поиск запущен из того же сохранённого начального приближения исправленного расчёта. Порядок проверок исходных параметров теперь [7, 8, 9, 1, 4, 2, 5, 3, 6]: сначала два смещения центра и угол, затем сопротивления фона, лёгкого и жира для обеих фаз. Дополнительная настройка порядка проверок возвращает физические параметры в исходную последовательность; порядок по умолчанию от 1 до 9 сохранён. Исполненная до изменения порядка версия обратного расчёта сохранена в архиве и проверяется по прежней контрольной сумме. Уравнения, границы сопротивлений тканей и физическое ядро не изменялись.

Параметры прежнего расчёта перенесены только как начальное приближение. Старые предсказания и сведения о происхождении подгонки не перенесены. Первый ряд исправленного расчёта в таблицах — свежий контроль с пересчитанными долями этого приближения на глубинной сетке. Первоначальное число совместных оценок, выделенных исправленному расчёту, — **{d['config']['additional_corrected_search_allocation']}**. Перед изменением приоритета в заменённой попытке совместного поиска без предварительного профиля число начатых оценок — **{d['config']['replaced_search_started_evaluations']}**; готовых численных промежуточных результатов — **{sum(v['status']=='numerical_candidate_not_validated' for v in d['priority']['replaced_rows'])}**. Предельное число оценок поиска с приоритетом геометрии — **{d['config']['max_evaluations']}**, при прежних сроках остановки поиска и всей задачи. Начатая без результата оценка не считается готовым кандидатом и не прибавляется к завершённым оценкам поиска с приоритетом геометрии. Число фактически завершённых оценок, включая отказы, — **{search['evaluations']}**; достижение целевого разрешения **{'отмечено' if search['resolution_reached'] else 'не подтверждено'}**. Готовых численных кандидатов — **{directions['valid_points']}**, геометрических отказов — **{directions['invalid_points']}**, начатых оценок без численного результата — **{directions['unfinished_points']}**. Новых готовых кандидатов с изменением ρ — **{directions['rho_points']}**, с изменением угла — **{directions['angle_points']}**. Поэтому объявленный приоритет геометрии не означает выполненного обхода всех геометрических направлений или новых материальных проб. Поиск ограничен временем и числом оценок; выбранный кандидат не объявляется полным оптимумом или доказательством глобальной неидентифицируемости.

<!-- search_table -->
<!-- rejected_table -->
<!-- history_figure -->

**Рисунок исправленного расчёта 2. Новый поиск.** Серые точки — RMSE рассчитанных кандидатов, синяя линия — лучшее достигнутое значение. Зелёные открытые ромбы, если присутствуют, — свежие проверки после поиска. Красные кресты — отказ геометрии на условном уровне, без вычисленной RMSE. Небольшой доступный объём поиска ограничивает выводы даже при улучшении невязки.

<!-- rho_table -->
<!-- coordinate_table -->
<!-- span_table -->
<!-- historical_span_table -->

Расстояния между I+ и I−, U+ и U− рассчитаны непосредственно из сохранённых итоговых координат каждой постановки. Это **хорды после проекции**, без измерения поверхностной дуги. В расчёте положения использован метод Ника: смещения в касательной плоскости с последующей проекцией на треугольную поверхность. Номинальное расстояние в этой конструкции не задаёт геодезическую длину по поверхности. На кривой коже хорда может быть короче поверхностной дуги; её уменьшение само по себе не является доказанной ошибкой. Физический смысл номинального расстояния и его связь с геодезической длиной носителя требуют отдельной проверки.

<!-- placement_figure -->

**Рисунок исправленного расчёта 3. Итоговое расчётное положение.** Наружный контур полупрозрачен; маски тканей включаются через легенду и подписаны по материалам. Оранжевые ряды — продольные сборки, сине-зелёные — поперечные; круги — I+, I−, квадраты — U+, U−. Линии соединяют центры для отображения расположения, без утверждения о физическом соединении площадок. Фиолетовые точки — постоянные площадки рук (их присутствие — допущение), коричневый крест — исходная ориентировочная точка автора, малиновый ромб — найденный общий центр. Поверхности упрощены для отображения; рисунок не подменяет фактические FEM-грани контактов. Выбранный центр LPS, мм: **[{centre}]**. Ракурс и масштаб сохраняются при отключении рядов.

<!-- stages_table -->
<!-- fit_figure -->

**Рисунок исправленного расчёта 4. Все 24 уровня.** Красные ряды — вдох, сине-зелёные — выдох. Пунктир с кругами — эксперимент; сплошные линии с квадратами — модель исправленного расчёта. По горизонтали — записи, размеры и оси. RMSE выбранного кандидата — **{number(r[0],7)} Ом**; тканевые параметры не приняты как индивидуально валидированные значения.

<!-- residual_table -->
<!-- controls_levels_table -->
<!-- respiratory_figure -->

**Рисунок исправленного расчёта 5. Все 12 контрастов.** Чёрные круги — измеренные $Z_e-Z_i$; серые квадраты — свежий контроль с пересчитанными долями начального приближения исправленного расчёта; синие квадраты — выбранный совместный кандидат. Чёрный пунктир — нулевой контраст. Несовпадений знака в выбранной точке — **{sign_mismatches(d,z)} из 12**. Неудача воспроизведения знака этой точкой и коротким поиском по направлениям не доказывает математической невозможности такого контраста во всей постановке.

<!-- respiratory_table -->
<!-- contrast_contributions_table -->

Разложение SSE даёт RMS ошибки среднего уровня **{number(r[1],6)} Ом**, RMS ошибки половины контраста **{number(r[2],6)} Ом**, долю контраста **{number(r[3],4)} %**. Таблица локальных вкладов использует якобиан выбранной точки исправленного расчёта и фактическую разность логарифмов её фазовых сопротивлений; ошибка суммы линейных вкладов относительно точного модельного контраста указана отдельно. Линейная диагностика не заменяет нелинейной модели.

<!-- repeated_table -->
<!-- sensitivity_table -->
<!-- sensitivity_figure -->

**Рисунок исправленного расчёта 6. Якобиан выбранной точки.** Цвет — знаковая производная уровня по одному из шести логарифмов ρ, Ом. Материалы и фазы подписаны по столбцам, наблюдения — по строкам. Отношение крайних сингулярных чисел — **{number(condition,3)}**. Слабая чувствительность, зависимость столбцов и достижение границ требуют дополнительных профилей и стартов; короткий поиск не заменяет этих проверок. Изменение эффективных ρ на статической КТ не считается анатомической вентиляцией или оценкой функций сердца.

## Исправленная постановка с 1×1: происхождение численных контролей и итоговое покрытие

Продолжение разрешено по предварительным техническим критериям формы, энергии, интегрирования, производных и систем, без отмены отрицательного контроля пространства и долей. Невязка более 13 Ом в начальной точке сохраняется; её численные изменения не устанавливают достигнутой сеточной сходимости. {control_reuse_statement(d)} Контрольная тройка — родитель → сгущённая сетка с унаследованными долями → та же сгущённая сетка с пересчитанными долями. Первый переход изменяет пространство FEM при точном наследовании долей и конечных площадок; второй — интегрирование масок при тех же узлах и тетраэдрах. Сравнение родитель → сгущённая сетка с пересчитанными долями показывает суммарный эффект. Внутри контрольной тройки параметры и положение всех сборок фиксированы. При повторном использовании её повторное отображение в начальном и выбранном аудитах не считается независимой численной проверкой.

<!-- mesh_table -->
<!-- control_reuse_table -->
<!-- initial_controls_table -->
<!-- numerical_table -->
<!-- numerical_figure -->

**Рисунок исправленного расчёта 7. Контроли итоговой точки.** Синие столбцы — пространство FEM, оранжевые — доли тканей, фиолетовые — суммарное изменение. Высота — модуль изменения к допуску 0,02 + 0,005|Z правого расчёта| Ом; чёрный пунктир на уровне 1 — граница. Противоположные знаки изменений не скрывают отказ отдельной составляющей. Ряды отключаются независимо.

<!-- coverage_table -->

Начальные проверки исправленного расчёта **{'выдержали все сохранённые допуски' if d['initial']['all_checks_passed'] else 'не выдержали все сохранённые допуски'}**; итоговые — **{'выдержали все сохранённые допуски' if d['final']['all_checks_passed'] else 'не выдержали все сохранённые допуски'}**. Покрытие именно итоговых площадок исправленного расчёта сгущением **{'полное' if d['coverage']['all_actual_contact_faces_refined'] else 'неполное'}**. {initial_coverage_statement(d)} Начальное покрытие не переносится на итоговое положение: при движении центра или угла площадки могут выйти за область направляющих. Для итоговой точки требуется отдельная сохранённая проверка покрытия, связанная с её собственным расчётом на родительской сетке. При неполном покрытии или отрицательном контроле завершение процедуры не разрешает количественную интерпретацию.

<!-- coarse_section_figure -->
<!-- deep_section_figure -->

**Рисунки исправленного расчёта 8 и 9. Сечения родительской и глубинной сеток в новом центре.** Синие отрезки — сечения тетраэдров, красный ромб — центр. По горизонтали — расстояние вдоль основной оси, по вертикали — вдоль принятой внутренней нормали, мм. Масштаб каждого рисунка независим. Сечения иллюстрируют сетку; достаточность электрической дискретизации устанавливается сравнениями импедансов.

<!-- contact_check_table -->
<!-- contact_check_figure -->

**Необязательный рисунок исправленного расчёта 10. Контактные грани при фиксированном объёме.** Используется самостоятельная проверка контактов исправленной постановки. Прежняя попытка этой проверки заменена до электрического расчёта и не является итоговым свидетельством. При завершённом сравнении фиолетовые ромбы показывают модуль изменения к допуску при непосредственном выборе граней на той же объёмной сетке; чёрный пунктир на уровне 1 — граница. Ряды отключаются отдельно. Параметры, поза, тканевые коэффициенты и физические электрические группы фиксированы. При отсутствии результата показано текстовое состояние, без графика и без признания проверки пройденной. **{contact_text}** Это не проверка реального контактного сопротивления кожи или глобальной/обратной приемлемости модели.


'''+material_update_text(d)+body_text(d)


def reader_functions(reader):
    names=list(dict.fromkeys(re.findall(r'<!-- (\w+) -->',reader)))
    missing=[name for name in names if not callable(globals().get(name))]
    if missing:raise ValueError('Unknown reader markers: '+', '.join(missing))
    return {name:globals()[name] for name in names}


def build(work=WORK):
    import nbformat as nbf
    from nbconvert import HTMLExporter
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    d=load(work);reader=text(d)
    functions=reader_functions(reader)
    relative=Path(work).resolve().relative_to(PIPE).as_posix()
    setup='from build_georg_contact_continuation_report import load, PIPE, numbered_table, '+', '.join(functions)+f'\nreport_data=load(PIPE / {relative!r})'
    shell=InteractiveShell.instance();cells=[];js=True;added=False;table_index=0
    result=shell.run_cell(setup,store_history=False)
    if result.error_before_exec or result.error_in_exec:raise RuntimeError('Reader initialization failed')
    for piece in re.split(r'(<!-- \w+ -->)',reader):
        if piece.startswith('<!-- '):
            key=piece[5:-4].strip()
            if key not in functions:raise ValueError('Unknown reader marker')
            if not added:
                c=nbf.v4.new_code_cell(setup);c.execution_count=1;cells.append(c);added=True
            code=f'display({key}(report_data'+(f',include_js={js!r}' if key.endswith('figure') else '')+'))'
            if key.endswith('table'):
                table_index+=1;code=f'display(numbered_table({key}(report_data),{table_index}))'
            if key.endswith('figure'):js=False
            shell.push({key:functions[key]});c=nbf.v4.new_code_cell('from IPython.display import display\n'+code)
            with capture_output(display=True) as capture:result=shell.run_cell(c.source,store_history=False)
            if result.error_before_exec or result.error_in_exec:raise RuntimeError('Reader cell failed: '+key)
            c.execution_count=sum(x.cell_type=='code' for x in cells)+1
            c.outputs=[nbf.v4.new_output('display_data',data=x.data,metadata=x.metadata) for x in capture.outputs]
            cells.append(c)
        elif piece.strip():cells.append(nbf.v4.new_markdown_cell(piece.strip()))
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        execution_mode='exact reader cells; no FEM',quantitative_result_accepted=False,
        selected_candidate_sha256=sha(Path(work)/'selected_candidate.local.json') if (Path(work)/'selected_candidate.local.json').exists() else None,
        main_status=d['state']['status'],computation_incomplete=d.get('incomplete',False),
        available_candidate_sha256=sha(d['candidate_file']) if d.get('candidate_file') is not None else None,
        corrected_observations_sha256=sha(d['input_directory']/'observations.local.csv'),
        corrected_config_sha256=sha(d['work']/W3_CONFIG),
        corrected_config_file=W3_CONFIG,parameter_evaluation_order=d['config']['parameter_evaluation_order'],
        replaced_search_started_evaluations=d['config']['replaced_search_started_evaluations'],
        priority_max_evaluations=d['config']['max_evaluations'],optional_contact_check_directory=W3_CONTACT_CHECK,
        initial_contact_coverage_sha256=sha(d['work']/'contact_coverage_initial.local.json') if (d['work']/'contact_coverage_initial.local.json').exists() else None,
        historical_selected_candidate_sha256=sha(d['history']['work']/'selected_candidate.local.json'),
        historical_observations_sha256=sha(d['history']['input_directory']/'observations.local.csv'),
        conditional_profile_omitted_for_original_time_budget=True,
        later_conditional_material_update_status=d['material_update']['status'],
        later_conditional_fit_sha256=d['material_update'].get('fit_sha256') if d['material_update'].get('fit') is not None else None,
        conditional_fit_replaces_joint_selected=False,
        optional_contact_check_status=d['contact_check']['status'],
        optional_contact_result_filename=d['contact_check'].get('result_filename'),
        original_contact_reference_gate_failure_preserved=bool(d['contact_check'].get('original_gate_failure')),
        completed_controls_reused_at_exactly_identical_point=bool(d.get('control_reuse')),
        selected_control_FEM_rerun=False if d.get('control_reuse') else None,
        selection_controls_reuse_sha256=sha(d['work']/'selection_controls_reuse.local.json') if d.get('control_reuse') else None))
    nbf.validate(nb);nbf.write(nb,FOLDER/(STEM+'.ipynb'))
    exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    html=html.replace('</head>','<style>details{margin:1em 0}summary{cursor:pointer;font-weight:600}.scientific-table{overflow-x:auto}</style></head>')
    (FOLDER/(STEM+'.html')).write_text(html,encoding='utf-8')
    (FOLDER/(STEM+'.md')).write_text(reader,encoding='utf-8')


if __name__=='__main__':build();print(STEM)
