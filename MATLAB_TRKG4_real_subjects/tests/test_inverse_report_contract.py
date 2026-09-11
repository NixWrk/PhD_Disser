from __future__ import annotations

import ast
import hashlib
import json
import os
import runpy
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'notebooks' / 'build_trkg4_inverse_report.py'
MAIN = ROOT / 'src' / 'run_trkg4_inverse_inhale.m'


@pytest.fixture(scope='module')
def generated_report(tmp_path_factory):
    target = tmp_path_factory.mktemp('generated_report') / 'report.ipynb'
    with patch.dict(os.environ, {'TRKG4_REPORT_TARGET': str(target)}):
        runpy.run_path(str(BUILDER), run_name='test_builder')
    return nbformat.read(target, as_version=4)


def _embedded_main_cell(notebook) -> str:
    return next(cell.source for cell in notebook.cells
                if cell.cell_type == 'code'
                and cell.source.startswith('from pathlib import Path'))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract_fixture(tmp_path: Path, compute_jacobian: bool, partial=False) -> tuple[str, str]:
    tag = f'fixture_20260908_{"jac" if compute_jacobian else "nojac"}'
    out = tmp_path / 'output'
    out.mkdir()

    summary_name = f'nik_trkg4_inverse_inhale_fit_summary_{tag}.csv'
    summary = (
        'rho_soft_ohm_m,rho_lungs_ohm_m,centre_u_mm,centre_v_mm,phi_deg,'
        'centre_x_mm,centre_y_mm,centre_z_mm,h_centre_mm,'
        'experimental_slope_ohm_per_mm,fem_slope_ohm_per_mm,'
        'rms_residual_ohm,mae_residual_ohm,function_evaluations,'
        'mesh_nodes,mesh_tetrahedra\n'
        '2.5,17.4,1.0,2.0,-3.0,10.0,20.0,30.0,15.0,0.1,0.11,0.2,0.15,1,100,200\n')

    files = {
        summary_name: summary,
        f'nik_trkg4_inverse_inhale_comparison_{tag}.csv':
            'L_mm,Z_experiment_ohm,Z_FEM_ohm,residual_ohm,source_file\n'
            '50,70,70,0,fixture.csv\n',
        f'nik_trkg4_inverse_inhale_depth_curve_{tag}.csv':
            'axis_s_mm,h_inward_ray_mm,h_nearest_lung_mm,skin_x_mm,skin_y_mm,skin_z_mm\n'
            '0,15,20,0,0,0\n',
        f'nik_trkg4_inverse_inhale_electrodes_{tag}.csv':
            'L_mm,electrode,patch_area_mm2\n50,I_plus,1\n',
        f'nik_trkg4_inverse_inhale_grid_geometry_{tag}.csv':
            'name,value\nfixture,1\n',
    }

    parameter_names = [
        'rho_soft_ohm_m', 'rho_lungs_ohm_m', 'centre_u_mm',
        'centre_v_mm', 'phi_deg']
    sensitivity_lines = [
        'parameter,finite_difference_step_requested,finite_difference_step_effective,'
        'difference_scheme,plausible_scale,jacobian_column_norm,'
        'local_crlb_std_at_sigma5ohm,status,finite_difference_half_step_requested,'
        'finite_difference_half_step_effective,half_difference_scheme,half_step_status,'
        'step_check_status,relative_difference_at_half_step,'
        'invalid_error_identifier,half_invalid_error_identifier'
    ]
    statuses = (['valid', 'valid_after_step_reduction', 'valid_with_bound_clipping',
                 'unavailable_bounds', 'unavailable_geometry'] if partial else ['valid'] * 5)
    for parameter, computed_status, scale in zip(parameter_names, statuses, [1, 5, 10, 10, 10]):
        status = computed_status if compute_jacobian else 'not_computed'
        numeric = (f'0.2,0.2,central,{scale},1,nan' if compute_jacobian
                   else f'nan,nan,not_computed,{scale},nan,nan')
        tail = '0.1,0.1,central,valid,distinct_effective_samples,0,, '
        if not compute_jacobian:
            tail = 'nan,nan,not_computed,not_computed,not_computed,nan,, '
        if partial and computed_status.startswith('unavailable'):
            numeric = f'0.2,nan,unavailable,{scale},nan,nan'
            tail = '0.1,nan,unavailable,unavailable_geometry,unavailable,nan,, '
        sensitivity_lines.append(f'{parameter},{numeric},{status},{tail}')
    files[f'nik_trkg4_inverse_inhale_parameter_sensitivity_{tag}.csv'] = \
        '\n'.join(sensitivity_lines) + '\n'

    if compute_jacobian:
        files[f'nik_trkg4_inverse_inhale_jacobian_{tag}.csv'] = (
            'L_mm,dZ_drho_soft,dZ_drho_lungs,dZ_dcentre_u_per_mm,'
            'dZ_dcentre_v_per_mm,dZ_dphi_per_deg\n'
            + ('50,1,2,3,nan,nan\n' if partial else '50,1,2,3,4,5\n'))
        details = [
            {'parameter': parameter, 'finite_difference_step_requested': 0.2,
             'status': status} for parameter, status in zip(parameter_names, statuses)]
        diagnostics = {
            'semantic_status': 'finite_step_secants_not_pose_fisher_information',
            'step_check_details': details,
            'step_check': {'details': details},
        }
    else:
        files[f'nik_trkg4_inverse_inhale_jacobian_{tag}.csv'] = (
            'L_mm,dZ_drho_soft,dZ_drho_lungs,dZ_dcentre_u_per_mm,'
            'dZ_dcentre_v_per_mm,dZ_dphi_per_deg\n50,nan,nan,nan,nan,nan\n')
        diagnostics = {'semantic_status': 'not_computed'}
    diagnostics_name = f'nik_trkg4_inverse_inhale_jacobian_diagnostics_{tag}.json'
    files[diagnostics_name] = json.dumps(diagnostics)

    history_name = f'nik_trkg4_inverse_inhale_history_{tag}.csv'
    files[history_name] = 'evaluation,objective\n1,0.2\n'
    direct_history_name = f'inverse_direct_search_history_{tag}.csv'
    files[direct_history_name] = 'evaluation,objective\n999,999\n'

    for name, content in files.items():
        (out / name).write_text(content, encoding='utf-8')

    artifacts = [
        {'name': name, 'path': f'output/{name}', 'sha256': _sha256(out / name)}
        for name in files if name != direct_history_name
    ]
    contract = {
        'schema_version': 1,
        'result_tag': tag,
        'parameters': {
            'rho_soft_ohm_m': 2.5,
            'rho_lungs_ohm_m': 17.4,
            'centre_u_mm': 1.0,
            'centre_v_mm': 2.0,
            'phi_deg': -3.0,
        },
        'data_selection': 'fixture',
        'patch_builder_mode': 'disjoint',
        'compute_jacobian': compute_jacobian,
        'geometry_qc_passed': True,
        'status': 'numerical_candidate_not_validated',
        'units': {'coordinates': 'mm', 'resistivity': 'ohm_m',
                  'transfer_impedance': 'ohm', 'contact_impedance': 'ohm_m2'},
        'inputs': [{'name': 'fixture_model', 'path': 'data/fixture.mat',
                    'sha256': '0' * 64}],
        'artifacts': artifacts,
    }
    (out / f'nik_trkg4_inverse_inhale_contract_{tag}.json').write_text(
        json.dumps(contract), encoding='utf-8')
    return tag, history_name


def test_builder_tracks_real_main_schema_and_guards_optional_cells(generated_report):
    builder_source = BUILDER.read_text(encoding='utf-8')
    producer_source = MAIN.read_text(encoding='utf-8')
    assert "'TRKG4_RESULT_TAG', 'auditfix_20260908'" in builder_source
    assert 'nik_trkg4_inverse_inhale_jacobian_diagnostics_{result_tag}.json' in builder_source
    assert "contract.inputs должен быть непустым массивом записей" in builder_source
    assert "jacobian_diagnostics.get('step_check_details')" in builder_source
    assert "step_check.get('details')" in builder_source
    assert "step_checks" not in builder_source
    assert "get('steps')" not in builder_source
    assert "contract.inputs = model_identity.inputs" in producer_source
    assert "contract.artifacts = local_file_records(artifact_files, root)" in producer_source
    assert 'nik_trkg4_inverse_inhale_jacobian_diagnostics_%s.json' in producer_source

    notebook = generated_report
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    for source in code_cells:
        ast.parse(source)

    historical_refs = [
        source for source in code_cells
        if ('fast_pem_summary' in source or 'fast_cem_summary' in source)
        and not source.lstrip().startswith('from pathlib import Path')
    ]
    assert historical_refs
    assert all(source.lstrip().startswith('if historical_pem_available:')
               for source in historical_refs)
    jacobian_guard = 'if compute_jacobian and jacobian_step_diagnostics_ok and jacobian_numeric_available:'
    assert any('np.linalg.svd' in source
               and source.lstrip().startswith(jacobian_guard)
               for source in code_cells)
    assert any('delta_rho12 = np.linalg.lstsq' in source
               and source.lstrip().startswith(jacobian_guard)
               for source in code_cells)
    joined = '\n'.join(code_cells)
    assert 'np.linalg.pinv' not in joined
    assert 'covariance =' not in joined
    assert 'CRLB' not in joined


@pytest.mark.parametrize(('compute_jacobian', 'partial'), [(True, False), (False, False), (True, True)])
def test_embedded_bootstrap_accepts_real_contract_fixture_and_history_provenance(
        tmp_path, monkeypatch, compute_jacobian, partial, generated_report):
    tag, history_name = _write_contract_fixture(tmp_path, compute_jacobian, partial)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRKG4_RESULT_TAG', tag)
    namespace = {}
    exec(compile(_embedded_main_cell(generated_report), 'embedded_main_bootstrap', 'exec'), namespace)

    assert namespace['result_contract']['result_tag'] == tag
    assert namespace['result_contract']['inputs']
    assert isinstance(namespace['result_contract']['inputs'], list)
    assert namespace['history_source'] == history_name
    assert namespace['history'].objective.iloc[0] == pytest.approx(0.2)
    assert namespace['historical_pem_available'] is False
    assert 'fast_pem_summary' not in namespace
    assert 'fast_cem' not in namespace
    if compute_jacobian:
        assert bool(namespace['jacobian_numeric_available']) is (not partial)
        assert namespace['jacobian_step_diagnostics_ok'] is True
    else:
        assert bool(namespace['jacobian_numeric_available']) is False
        assert namespace['jacobian_diagnostics']['semantic_status'] == 'not_computed'
        assert namespace['status_values'] == {'not_computed'}

    status = namespace['report_status_markdown']
    assert f'`{tag}`' in status
    assert 'геометрический QC пройден по контракту' in status
    assert 'единицы контракта согласованы' in status
    assert 'numerical_candidate_not_validated' in status
    assert 'число отклонений неизвестно' in status
    assert namespace['report_history_counts']['rows'] == 1
    assert namespace['report_history_counts']['invalid'] is None
    if not compute_jacobian or partial:
        assert 'SVD и дыхательная линейная подгонка пропущены' in status
        expected_secant_text = 'not_computed' if not compute_jacobian else 'конечных столбцов 3/5'
        assert expected_secant_text in status
        # Execute the actual generated cells; missing physics inputs must never be read here.
        for cell in generated_report.cells:
            if cell.cell_type == 'code' and cell.source.startswith('if compute_jacobian and'):
                exec(compile(cell.source, 'guarded_report_cell', 'exec'), namespace)
        assert 'weak_delta_z' not in namespace
        assert 'delta_rho12' not in namespace


def test_generated_visible_claims(generated_report):
    # Section 18 is removed by the builder. Audit the emitted notebook, not dead source.
    visible = '\n'.join(cell.source for cell in generated_report.cells)
    assert '## 18. Контроль положения' not in visible
    for stale_claim in (
        'Все локальные оценки обусловленности в этом разделе рассчитаны заново',
        'Для каждой пробы выполнялись девять FEM-расчётов',
        'заново рассчитан Якобиан',
        'Слабое направление меняет всю кривую только',
        'Если ошибочно оставить геометрию вдоха',
        'Диагностика ошибочного предположения',
        'недостаточности фиксированной геометрии',
        'около независимой оценки по КТ/HU',
        '100 FEM-оценок; лучший на оценке 93',
        'также имеет локальный максимум при 60 мм',
        'выполнен в разделе 18',
    ):
        assert stale_claim not in visible
    for qualification in (
        'может содержать кость',
        'подстановка $h=h_c^{surface}$ является допущением',
        'не непосредственно измеренное независимое свойство',
        'невязка не доказывает геометрический вклад',
        'Конечный нелинейный отклик при той же геометрии',
        'полный FEM в смещённой точке здесь не пересчитывался',
        'SVD и дыхательная линейная подгонка пропущены',
    ):
        assert qualification in visible


def _update_artifact_hash(tmp_path, tag, path):
    contract_path = tmp_path / 'output' / f'nik_trkg4_inverse_inhale_contract_{tag}.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    record = next(record for record in contract['artifacts'] if record['name'] == path.name)
    record['sha256'] = _sha256(path)
    contract_path.write_text(json.dumps(contract), encoding='utf-8')


def test_verified_history_counts_before_filtering(tmp_path, monkeypatch, generated_report):
    tag, history_name = _write_contract_fixture(tmp_path, False)
    history_path = tmp_path / 'output' / history_name
    history_path.write_text(
        'evaluation,objective,status\n1,0.2,evaluated\n2,1e12,invalid_geometry\n'
        '3,nan,unknown\n', encoding='utf-8')
    _update_artifact_hash(tmp_path, tag, history_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRKG4_RESULT_TAG', tag)
    namespace = {}
    exec(_embedded_main_cell(generated_report), namespace)
    assert namespace['report_history_counts'] == {'rows': 3, 'valid': 1, 'invalid': 1}
    assert len(namespace['history']) == 1
    status = namespace['report_status_markdown']
    assert 'valid/evaluated=1, invalid_geometry=1, не классифицировано=1' in status
    assert 'не счётчик FEM-решений' in status
    # Corrupting a verified file must block the report, including the status counts.
    history_path.write_text('evaluation,objective\n1,1\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Хэш артефакта не совпал'):
        exec(_embedded_main_cell(generated_report), {})


def test_unverified_history_and_legacy_qc_status(tmp_path, monkeypatch, generated_report):
    tag, _ = _write_contract_fixture(tmp_path, False)
    contract_path = tmp_path / 'output' / f'nik_trkg4_inverse_inhale_contract_{tag}.json'
    contract_path.unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRKG4_RESULT_TAG', tag)
    namespace = {}
    exec(_embedded_main_cell(generated_report), namespace)
    status = namespace['report_status_markdown']
    assert 'исторический результат, не прошедший приёмку по новому QC' in status
    assert 'новый QC не подтверждён контрактом' in status
    assert 'valid/invalid не установлены' in status
    assert 'единицы контракта согласованы' not in status
    assert namespace['report_history_counts'] is None


@pytest.mark.parametrize('unit_change', ['missing', 'mismatch'])
def test_status_does_not_invent_units_or_qc(tmp_path, monkeypatch, generated_report, unit_change):
    tag, _ = _write_contract_fixture(tmp_path, False)
    contract_path = tmp_path / 'output' / f'nik_trkg4_inverse_inhale_contract_{tag}.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['geometry_qc_passed'] = False
    if unit_change == 'missing':
        del contract['units']
    else:
        contract['units']['coordinates'] = 'm'
    contract_path.write_text(json.dumps(contract), encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRKG4_RESULT_TAG', tag)
    namespace = {}
    exec(_embedded_main_cell(generated_report), namespace)
    status = namespace['report_status_markdown']
    assert 'геометрический QC не пройден по контракту' in status
    assert 'единицы в контракте отсутствуют или не совпадают' in status
    assert 'единицы контракта согласованы' not in status


def test_counts_use_verified_driver_history_when_available(tmp_path, monkeypatch, generated_report):
    tag, _ = _write_contract_fixture(tmp_path, False)
    out = tmp_path / 'output'
    driver = out / f'nik_trkg4_inverse_direct_search_history_{tag}.csv'
    driver.write_text('evaluation,objective,status\n1,0.2,evaluated\n2,1e12,invalid_geometry\n',
                      encoding='utf-8')
    contract_path = out / f'nik_trkg4_inverse_inhale_contract_{tag}.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['artifacts'].append({'name': driver.name, 'path': f'output/{driver.name}',
                                  'sha256': _sha256(driver)})
    contract_path.write_text(json.dumps(contract), encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRKG4_RESULT_TAG', tag)
    namespace = {}
    exec(_embedded_main_cell(generated_report), namespace)
    assert namespace['history_source'] == driver.name
    assert namespace['report_history_counts'] == {'rows': 2, 'valid': 1, 'invalid': 1}
