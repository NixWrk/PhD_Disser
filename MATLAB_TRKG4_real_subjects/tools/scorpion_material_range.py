"""Новая ограниченная серия rho на готовых канонических материальных блоках."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ['soft_background', 'lung', 'bone', 'heart', 'fat']
PRIMARY = 'L140_TT2x2'
MUSCLE = (2., 2.5, 3.)
LUNG = (2., 4., 7., 10., 12.)
COARSE = {(2., 2.), (2., 12.), (3., 2.), (3., 12.), (2.5, 7.)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_pin(value):
    require(sha(value['path']) == value['sha256'], 'Изменился источник: ' + value['path'])


def rows_for_scan(reference):
    rows = []
    points = [(2.5, 7.)] + [p for p in itertools.product(MUSCLE, LUNG) if p != (2.5, 7.)]
    for q in (4096, 1024):
        for muscle, lung in points:
            if q == 1024 and (muscle, lung) not in COARSE:
                continue
            rho = list(reference['inhale_rho_ohm_m'])
            rho[0], rho[1], rho[3] = muscle, lung, 1.35
            point = f'm{muscle:g}_l{lung:g}'.replace('.', 'p')
            rows.append(dict(id=f'{point}_q{q}', point_id=point, geometry='wholeheart_ct',
                operator_id='baseline', rule=q, parts_row=1 if q == 1024 else 2,
                rho_ohm_m=rho, finite_difference=q == 4096 and (muscle, lung) == (2.5, 7.)))
    return rows


def prepare(base, action, output, cases=None):
    base, output = Path(base).resolve(), Path(output).resolve()
    require(action in ('scan', 'heart'), 'Неизвестное действие')
    require(not output.exists(), 'Нужен новый каталог серии; старые результаты неизменяемы')
    forward = base / 'electrical/forward.local.json'; b = read(forward)
    subject = b['subject_alias']
    require(subject in ('nik', 'georg', 'yaroslav'), 'Неизвестный участник')
    require(b['FEM_was_run'] and not b['limited_run'] and b['frequency_hz'] == 100000,
        'Требуется полный собственный baseline 100 кГц')
    require(b['derivative_checks']['performed'] and b['derivative_checks']['passed'] and
        b['derivative_checks']['active_material_indices'] == [1, 2, 4, 5], 'Baseline FD не пройден')
    require(all(c['system_relative_residual'] < 1e-7 and c['reciprocity_ohm'] < 1e-7
        for c in b['system_checks']), 'Baseline CEM QC не пройден')
    reference_path = base / 'reference_materials.local.json'; reference = read(reference_path)
    require(reference['frequency_hz'] == 100000 and reference['material_order'] == ORDER,
        'Неверный частотный профиль или порядок тканей')
    require(sha(reference_path) == b['identity']['reference_sha256'], 'Изменён baseline reference')
    mesh = base / 'materials/mesh.mat'; cem = base / 'cem/cem_contacts.mat'
    inputs = base / 'electrical_inputs.local.json'; observations = read(inputs)
    for name, path in [('mesh_sha256', mesh), ('cem_sha256', cem), ('inputs_sha256', inputs)]:
        require(sha(path) == b['identity'][name], 'Несовпадение собственного baseline: ' + name)
    require(observations['subject_alias'] == subject, 'Чужие наблюдения')
    available = sorted({a['case_id'] for a in b['answers']})
    selected = available if cases == 'all' or (cases is None and action == 'scan') else (
        [PRIMARY] if cases is None else cases.split(','))
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(available),
        'Неизвестные либо повторные монтажи')
    require(PRIMARY in selected, 'Для принятой проверки требуется основной монтаж L140_TT2x2')
    for case in selected:
        for q in (1024, 4096):
            require(sum(a['case_id'] == case and a['state'] == 'inhale' and a['rule'] == q
                for a in b['answers']) == 1, 'Нет собственного baseline для монтажа/q')
    worker = ROOT / 'MATLAB_TRKG4_real_subjects/src/run_scorpion_material_range.m'
    sources = [pin(p) for p in (forward, reference_path, mesh, cem, inputs, worker, __file__)]
    for p in observations['source_files']:
        check_pin(p); sources.append(p)
    registry = reference['registry']; check_pin(registry); sources.append(registry)
    operators, rows = [], []
    if action == 'scan':
        parts = base / 'electrical/prepared_materials.mat'
        operators = [dict(id='baseline', layout='2x5', path=str(parts), sha256=sha(parts))]
        rows = rows_for_scan(reference)
    else:
        pilot = base / 'heart_pilot/plan.local.json'; h = read(pilot)
        require(h['subject_alias'] == subject and h['mesh_sha256'] == b['identity']['mesh_sha256']
            and h['material_order'] == ORDER, 'Heart pilot относится к другой модели')
        sources.append(pin(pilot))
        require(len(h['variants']) == 8, 'Требуются восемь подготовленных variant/q')
        for variant in h['variants']:
            directory = pilot.parent / ('assemble_' + variant['id'])
            report_file = directory / 'report.local.json'; report = read(report_file)
            parts = directory / 'parts.mat'
            require(report['status'] == 'canonical_weighted_assembly_checked' and
                report['identity']['subject_alias'] == subject and
                report['identity']['mesh_sha256'] == b['identity']['mesh_sha256'] and
                report['identity']['plan_sha256'] == sha(pilot) and
                all(report['identity'][key] == b['identity'][key] for key in
                    ('field_kernel_sha256', 'scale_kernel_sha256', 'reference_sha256')) and
                report['homogeneous_error'] < 1e-12 and report['heterogeneous_error'] < 1e-12,
                'Не принята собственная каноническая сборка heart variant')
            require(report['parts_sha256'] == sha(parts), 'Изменились heart parts')
            check_pin(dict(path=variant['file'], sha256=variant['sha256']))
            sources += [pin(report_file), pin(variant['file'])]
            operators.append(dict(id=variant['id'], layout='1x5', path=str(parts), sha256=sha(parts)))
            rho = list(reference['inhale_rho_ohm_m']); rho[0], rho[1], rho[3] = 2.5, 7., 1.35
            rows.append(dict(id=variant['id'], point_id='m2p5_l7', geometry=variant['geometry'],
                operator_id=variant['id'], rule=variant['q'], parts_row=1,
                rho_ohm_m=rho, finite_difference=False))
        expected = {(g, q) for g in ('wholeheart_ct', 'sphere_v095', 'sphere_v100', 'sphere_v105')
            for q in (1024, 4096)}
        require({(r['geometry'], r['rule']) for r in rows} == expected,
            'Неполный набор wholeheart/sphere/q')
    sources += [dict(path=o['path'], sha256=o['sha256']) for o in operators]
    unique = {p['path']: p for p in sources}
    result = dict(schema='scorpion_material_range_plan_v1', action=action, subject_alias=subject,
        frequency_hz=100000, material_order=ORDER, active_material_indices=[1, 2, 4, 5], threads=6,
        case_ids=selected, primary_case=PRIMARY, rows=rows, operator_sources=operators,
        source_files=list(unique.values()), baseline_forward=str(forward), observations_file=str(inputs),
        mesh_mat=str(mesh), cem_mat=str(cem), reference_materials=str(reference_path),
        identity=dict(mesh_sha256=b['identity']['mesh_sha256'], cem_sha256=b['identity']['cem_sha256'],
            reference_sha256=b['identity']['reference_sha256'], registry_sha256=registry['sha256'],
            field_kernel_sha256=b['identity']['field_kernel_sha256'],
            scale_kernel_sha256=b['identity']['scale_kernel_sha256'],
            response_kernel_sha256=b['identity']['response_kernel_sha256']),
        parameter_source='Прямое указание автора 26.10.2026; диапазоны коллеги и первый однородный blood-сценарий',
        soft_background_is_effective_unlabelled_mixture=True,
        muscle_value_assignment='Диапазон мышц присвоен эффективному фону всех неотмеченных мягких тканей; отдельной анатомической muscle-маски нет',
        blood_replacement_is_tissue_validation=False,
        rho_muscle_grid=list(MUSCLE), rho_lung_grid=list(LUNG), rho_wholeheart_blood_ohm_m=1.35,
        wholeheart_is_author_blood_replacement=True, blood_fraction_is_measured=False,
        fat_and_bone_fixed_registry_are_assumptions=True, geometry='fixed_deep_inspiration_CT',
        current_per_source_A=.007, current_is_Z_multiplier=False, excitation_scenario='own_drive',
        source_relative_phase='unknown', FD_scope='center_primary_q4096_four_logrho' if action == 'scan' else 'not_run_for_heart',
        q_scope='corners_and_center_only' if action == 'scan' else 'all_four_geometry_pairs',
        mesh_convergence=False, individual_fitted=False, fitted_gain=False,
        clinical_validation=False, stroke_volume_estimated=False, reader_tables_default_open=False)
    for p in result['source_files']:check_pin(p)
    output.mkdir(parents=True)
    with (output / 'plan.local.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('base', 'out'):prep.add_argument('--' + name, required=True)
    prep.add_argument('--action', required=True, choices=('scan', 'heart'))
    prep.add_argument('--cases', help='all либо перечисление case_id через запятую')
    a = p.parse_args(); plan = prepare(a.base, a.action, a.out, a.cases)
    print('Подготовлен план:', Path(a.out).resolve() / 'plan.local.json', '; строк:', len(plan['rows']))
