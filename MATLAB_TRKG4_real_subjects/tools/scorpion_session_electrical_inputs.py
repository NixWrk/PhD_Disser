"""Подготовка наблюдений и литературных сценариев 100 кГц без подгонки шкалы."""
from pathlib import Path
import argparse
import csv
import math
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import ct_fem_core as core


SUBJECTS=('georg','nik','yaroslav')
STATES=('inhale','exhale')


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=core.sha(path))


def reference_materials(registry):
    r=core.read(registry)
    names=('Muscle','Lung (Inflated)','Bone (Cortical)','Heart Muscle','Fat')
    def values(tissues):
        result=[r['tissues'][n]['gabriel_at_frequency']['100000']['rho_ohm_m'] for n in tissues]
        if not all(math.isfinite(x) and x>0 for x in result):
            raise ValueError('Требуются положительные конечные rho для 100 кГц')
        return result
    return dict(frequency_hz=100000,material_order=core.MATERIALS,
        inhale_rho_ohm_m=values(names),exhale_rho_ohm_m=values((names[0],'Lung (Deflated)',*names[2:])),
        registry=pin(registry),registry_sha256=core.sha(registry),registry_id=r['registry_id'],
        registry_version=r['schema_version'],tissue_assignments=dict(zip(core.MATERIALS,names)),
        source='Частотные записи Gabriel из литературного реестра проекта',
        nominal_heart_is_myocardial_endpoint_of_effective_whole_heart=True,
        blood_fraction_is_measured=False,individual_tissue_values_are_measured=False,
        geometric_breathing_motion_modelled=False,
        exhale_status='conditional_material_change_on_fixed_deep_inspiration_CT',
        fitted_gain=False,individual_fitted=False,mesh_convergence=False)


def observation(record,role,state,case_id='',channel=None,value=None,reason='',evidence=None,window=None,qc=None):
    if value is not None and not math.isfinite(float(value)):
        raise ValueError('Неконечное экспериментальное значение импеданса')
    return dict(record_id=record['record_id'],case_id=case_id,role=role,state=state,
        Z_ohm=None if value is None else float(value),channel=channel,source_sha256=record['source_sha256'],
        status='excluded' if reason else 'eligible_conditional',exclusion_reason=reason,
        annotation_status='pending_manual_review',manual_annotation_accepted=False,
        role_evidence=evidence or {},window=window or {},source_qc=qc or {})


def source_pin(record,experiment):
    result=pin(experiment/record['source_relative_path'])
    if result['sha256']!=record['source_sha256']:
        raise ValueError('Изменился исходный файл: '+record['record_id'])
    return result


def session_observations(subject,summary,summary_file,annotations=None):
    experiment=summary_file.parents[3]
    annotation_path=Path(annotations) if annotations else experiment/'annotations/session_1140/candidate_annotations.local.json'
    ann=core.read(annotation_path)
    if core.sha(annotation_path)!=summary['annotations_sha256']:
        raise ValueError('Разметка сессии не соответствует summary')
    by_id={a['record_id']:a for a in ann['records']}
    records={r['record_id']:r for r in summary['records'] if r['subject']==subject}
    if not records:raise ValueError('В summary нет записей выбранного участника')
    sources=[pin(annotation_path)]+[source_pin(r,experiment) for r in records.values()]
    rows=[]
    for v in summary['levels']:
        if v['record_id'] not in records:continue
        rec=records[v['record_id']];context=rec['context'];role=v['role'];annotation=by_id[rec['record_id']]
        if annotation['source_sha256']!=rec['source_sha256']:raise ValueError('Разметка относится к другой записи')
        reason='' if context['record_kind']=='both' else 'single_source_or_reciprocity_requires_separate_physical_CEM_scenario'
        if role not in rec['active_roles'] or role not in ('lateral','ttrkg'):reason='role_not_active_or_not_supported'
        channel=rec['physical_channel_roles'].get(role)
        if channel not in (1,2):reason='physical_channel_role_unresolved'
        rows.append(observation(rec,role,v['state'],f"L{context['lateral_size_mm']}_TT{context['ttrkg_variant']}",
            channel,v['baseline_ohm'],reason,dict(source=rec['roles_evidence'],mapping=rec['physical_channel_roles']),
            dict(start_s=v['start_s'],end_s=v['end_s'],annotation=annotation),
            dict(level=v,ecg=rec['ecg_qc'],summary_status=summary['status'])))
    represented={(r['record_id'],r['role'],r['state']) for r in rows}
    for rec in records.values():
        for role,channel in rec['physical_channel_roles'].items():
            for state in STATES:
                if (rec['record_id'],role,state) not in represented:
                    rows.append(observation(rec,role,state,channel=channel,
                        reason='no_state_level_in_summary_or_switching_control_record',
                        evidence=dict(context=rec['context']),qc=dict(ecg=rec['ecg_qc'])))
    return rows,sources,dict(kind='session_20261022',annotation_source=pin(annotation_path),
        sampling=summary['sampling'],source_status=summary['status'])


def georg_observations(summary,summary_file,levels=None,annotations=None):
    folder=ROOT/'Colab Notebooks/scorpion_v2';sys.path.insert(0,str(folder))
    import scorpion_protocol as protocol
    experiment=summary_file.parents[2]
    levels_file=Path(levels) if levels else experiment/'derived/exploratory/protocol_revision_20261012/measured_levels.local.csv'
    contract_file=levels_file.with_name('contract.local.json');contract=core.read(contract_file)
    annotation_path=Path(annotations) if annotations else experiment/'annotations/scorpion_review/annotations_candidates.local.json'
    ann=core.read(annotation_path);protocol_path=folder/'scorpion_protocol.py'
    for expected,path in [(contract['protocol_sha256'],protocol_path),
            (contract['input_contract']['review_summary_sha256'],summary_file),
            (contract['input_contract']['annotations_sha256'],annotation_path)]:
        if core.sha(path)!=expected:raise ValueError('Источник Георгия не соответствует протоколу: '+str(path))
    by_id={r['record_id']:r for r in summary['records']
        if r['fs_hz_scenario']==(400 if r['record_id'][-3:] in ('005','006') else 1000)}
    by_ann={(a['record_id'],a['fs_hz_scenario']):a for a in ann['records']}
    primary={'001':140,'002':140,'003':140,'004':140,'007':120,'008':100,'009':80,'010':60}
    sources=[pin(p) for p in (levels_file,contract_file,annotation_path,protocol_path,folder/'scorpion_qc.py')]
    sources += [source_pin(r,experiment) for r in by_id.values()]
    rows=[]
    with levels_file.open(encoding='utf-8-sig',newline='') as stream:measured=list(csv.DictReader(stream))
    for v in measured:
        rec=by_id[v['record_id']];rid=rec['record_id'];short=rid[-3:];assignment=protocol.channel_assignment(rid)
        channel=int(v['channel']);original_role=assignment['physical_channel_roles'][str(channel)]
        if v['role']!=original_role:raise ValueError('Роль измерения не соответствует протоколу: '+rid)
        role='lateral' if original_role=='lateral_horizontal' else original_role
        configuration=assignment['ttrkg'];variant=configuration['variant'] if configuration else ''
        annotation=by_ann[rid,rec['fs_hz_scenario']]
        if annotation['source_sha256']!=rec['source_sha256']:raise ValueError('Разметка относится к другой записи: '+rid)
        reason='' if short in primary and role in ('lateral','ttrkg') else 'outside_primary_lateral_TTRKG_protocol'
        rows.append(observation(rec,role,v['state'],f'L{primary[short]}_TT{variant}' if short in primary else '',
            channel,v['baseline_raw_median_ohm'],reason,assignment,
            dict(start_s=float(v['start_s']),end_s=float(v['end_s']),breathing=annotation['breathing']),
            dict(flags=rec['flags'],channels=rec['channels'],timebase_status=rec['timebase_status'],
                fs_hz_scenario=rec['fs_hz_scenario'],measured_snapshot_row=v)))
    represented={(r['record_id'],r['channel'],r['state']) for r in rows}
    for rid,rec in by_id.items():
        assignment=protocol.channel_assignment(rid)
        for channel,role in assignment['physical_channel_roles'].items():
            for state in STATES:
                if (rid,int(channel),state) not in represented:
                    rows.append(observation(rec,role,state,channel=int(channel),reason='no_state_level_in_protocol_snapshot',
                        evidence=assignment,qc=dict(flags=rec['flags'])))
    return rows,sources,dict(kind='georg_protocol_snapshot',protocol_revision=protocol.REVISION,
        annotation_source=pin(annotation_path),levels_source=pin(levels_file),
        role_continuity_008_010_is_assumption=True,source_status=summary['status'])


def build(subject,model,summary_file,*,summary_kind='auto',levels=None,annotations=None):
    if subject not in SUBJECTS:raise ValueError('Неизвестный участник')
    model=Path(model).resolve();summary_file=Path(summary_file).resolve()
    if not model.is_dir():raise ValueError('Требуется существующий каталог электрической подготовки')
    targets=[model/'reference_materials.local.json',model/'electrical_inputs.local.json']
    if any(p.exists() for p in targets):raise FileExistsError('Существующие входы неизменяемы; используйте новую подготовку')
    summary=core.read(summary_file)
    kind=('georg' if subject=='georg' else 'session') if summary_kind=='auto' else summary_kind
    if (kind=='georg')!=(subject=='georg'):raise ValueError('Тип summary не соответствует участнику')
    if kind=='georg':rows,sources,context=georg_observations(summary,summary_file,levels,annotations)
    else:rows,sources,context=session_observations(subject,summary,summary_file,annotations)
    keys=[(r['record_id'],r['channel'],r['role'],r['state']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Повторная пара запись/роль/состояние')
    registry=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    reference=reference_materials(registry);core.write_new(targets[0],reference)
    result=dict(schema='scorpion_electrical_observations_v2',subject_alias=subject,summary_source=pin(summary_file),
        source_files=[pin(summary_file),*sources,pin(registry)],source_context=context,observations=rows,
        reference_materials_sha256=core.sha(targets[0]),current_per_source_A=.007,current_amplitude_convention='unknown',
        own_drive_is_working_source_rule=True,source_relative_phase='unknown',producer_sha256=core.sha(__file__),
        fitted_gain=False,individual_fitted=False,mesh_convergence=False,clinical_validation=False)
    core.write_new(targets[1],result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',required=True,choices=SUBJECTS)
    for k in ('model','summary'):p.add_argument('--'+k,required=True)
    p.add_argument('--summary-kind',default='auto',choices=('auto','session','georg'))
    p.add_argument('--levels',help='Сохранённый measured_levels.local.csv Георгия')
    p.add_argument('--annotations',help='Кандидатная разметка выбранной сессии')
    a=p.parse_args();result=build(a.subject,a.model,a.summary,summary_kind=a.summary_kind,levels=a.levels,annotations=a.annotations)
    print('Наблюдений:',len(result['observations']),'; условно включены:',
        sum(r['status']=='eligible_conditional' for r in result['observations']))
