"""Build the self-contained executed-report notebook for the TRKG4 inverse fit."""
import os
import hashlib

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get(
    "TRKG4_REPORT_TARGET",
    ROOT / "notebooks" / "TRKG4_inverse_inhale_results.ipynb"))
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def guarded_code(text: str, condition: str, skipped_message: str) -> None:
    body = text.strip()
    indented = "\n".join(
        ('    ' + line) if line else '' for line in body.splitlines())
    code(
        f"if {condition}:\n"
        f"{indented}\n"
        "else:\n"
        f"    display(Markdown({skipped_message!r}))")


def optional_historical_code(text: str) -> None:
    guarded_code(
        text,
        'historical_pem_available',
        'Исторические PEM/CEM-файлы отсутствуют; раздел пропущен. '
        'Это не блокирует основной отчёт.')


def jacobian_code(text: str) -> None:
    guarded_code(
        text,
        'compute_jacobian and jacobian_step_diagnostics_ok and jacobian_numeric_available',
        'Якобиан не рассчитан или его диагностика неполна; числовой '
        'раздел чувствительности пропущен.')


def breathing_linear_code(text: str) -> None:
    guarded_code(
        text,
        'compute_jacobian and jacobian_step_diagnostics_ok and jacobian_numeric_available',
        'Дыхательная линейная подгонка пропущена: для выбранного '
        'контракта нет пригодного Якобиана.')


md(r"""
# TRKG4: восстановление параметров по серии электродных сборок на вдохе

Этот ноутбук представляет собой воспроизводимый расчётный отчёт. В нём последовательно рассмотрены:

1. исходная CT/FEM-анатомия и положение всех электродных сборок;
2. косой DICOM-срез через найденную ось сборок;
3. границы кожи, лёгких, сердца и костей на этой плоскости;
4. определение и аппроксимация геометрического расстояния кожа–лёгкое $h(s)$;
5. способ учёта трёхмерной геометрии и $h(s)$ в прямой FEM-задаче;
6. математическая постановка обратной задачи, допущения и ограничения;
7. итоговые параметры, $Z(L)$, остатки и локальная идентифицируемость.

**Состояние модели:** оболочка тела с растром 1 мм и локально сгущённая сетка `v5_1mm_local2mm`; заполненная модель лёгких `v3`, где
$\rho_2$ относится к эффективной смеси лёгочной ткани и воздуха. Всей маске сердца назначена
литературная проводимость сердечной мышцы, костям — кортикальной кости. Отдельной маски крови
для Ника нет, поэтому кровь самостоятельной областью FEM не представлена.

**Основной расчёт во всём ноутбуке:** расчёт, выбранный через `TRKG4_RESULT_TAG`; по умолчанию
это `auditfix_20260908`. Архивная координата до перепрогона и
отклонённая точка `fmincon` сохранены только для сравнения происхождения изменений; они не
подменяют итоговую координату. Исторический PEM/CEM-скрининг также явно отделён от основного
результата.
""")

md(r"""
## Что было сделано — простыми словами

1. В эксперименте было 10 файлов электродных сборок. Проверка субъектного манифеста и самих
   файлов показала: запись 90 мм является независимой, а запись 100 мм — её поздней
   перезаписанной копией. Поэтому в расчёт входят **9 независимых размеров**: 50, 60, 70, 80,
   90, 110, 120, 130 и 140 мм; 100 мм исключён.
2. Считалось, что все сборки ставились в один общий центр и вдоль одной общей оси.
3. Для каждого размера четыре электрода первоначально располагались на этой оси в точках
   $[-L/2,-L/4,+L/4,+L/2]$.
4. Каждая из четырёх точек проецировалась на реальную STL-поверхность кожи. Поэтому после
   проекции электроды следуют кривизне тела и не обязаны оставаться в одной идеальной плоскости.
5. В ходе поиска изменялись пять величин: $\rho_1$, $\rho_2$, две координаты общего центра по
   коже и угол поворота общей оси $\varphi$.
6. Координаты центра $(x,y,z)$ не менялись независимо: центр всегда оставался на поверхности
   кожи. Трёхмерные координаты вычислялись после проекции.
7. Полная допустимая кривая содержит **9 FEM-значений** — по одному на размер сборки.
   В исправленном поиске недопустимая геометрия может быть отклонена до решения FEM;
   число проб и число решённых FEM-задач поэтому различаются.
8. Полученные девять значений $Z_{FEM}(L)$ сравнивались с девятью измеренными значениями.
   Сравнивались и отдельные точки, и общий наклон зависимости $Z(L)$.
9. Проводимости целой маски сердца и костей не подгонялись. Отдельная маска крови отсутствует;
   это явно принятое ограничение модели.
10. Толщина $h$ не подставлялась в FEM одним числом. Реальная граница лёгкого уже находится
    внутри трёхмерной CT/STL-модели, поэтому изменение глубины до лёгкого учитывалось самой
    геометрией.

**Что пока не подгонялось:** отдельная ошибка расстояния между электродами каждой сборки и
индивидуальный пространственный наклон каждой жёсткой сборки.
""")

md(r"""
## 1. Математическая модель

### Электродная геометрия

Для сборки номинального размера $L$ четыре центра задаются одним общим центром $\mathbf c$ и
касательной осью $\mathbf a_\varphi$:

$$
\widetilde{\mathbf r}_{e}(L)=\mathbf c+q_e L\,\mathbf a_\varphi,
\qquad
q_e\in\left\{-\frac12,-\frac14,+\frac14,+\frac12\right\},
$$

$$
\mathbf r_e(L)=\Pi_{\partial\Omega_{skin}}\!\left(\widetilde{\mathbf r}_{e}(L)\right),
$$

где $\Pi$ — непрерывная проекция на треугольную поверхность кожи. Поэтому трёхмерные
координаты центра не являются тремя независимыми переменными: свободны две координаты
$(u,v)$ на поверхности, а $(x,y,z)$ вычисляются из CT/STL.

### Проводимость

$$
\sigma(\mathbf r;\theta)=
\begin{cases}
1/\rho_1, & \mathbf r\in\Omega_{soft},\\
1/\rho_2, & \mathbf r\in\Omega_{lung},\\
\sigma_{heart}^{ITIS}, & \mathbf r\in\Omega_{heart},\\
\sigma_{bone}^{ITIS}, & \mathbf r\in\Omega_{bone}.
\end{cases}
$$

Внутри объёма решается уравнение квазистатического поля

$$
\nabla\!\cdot\!\left(\sigma\nabla u\right)=0.
$$

Используется полная электродная модель (complete electrode model, CEM):

$$
u+z_c\sigma\frac{\partial u}{\partial n}=U_\ell\quad\text{на }e_\ell,
\qquad
\int_{e_\ell}\sigma\frac{\partial u}{\partial n}\,dS=I_\ell,
$$

а измеряемый передаточный импеданс равен

$$
Z(L;\theta)=\frac{U_{V+}-U_{V-}}{I_{I+}}.
$$
""")

code(r"""
from pathlib import Path
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
import pydicom
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from stl import mesh as stl_mesh

warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 50)
pd.set_option('display.precision', 5)

# Store Plotly figures as ordinary HTML outputs.  The first figure embeds
# plotly.js, so the exported report remains interactive without internet.
_plotly_js_embedded = False
def show_plotly(figure):
    global _plotly_js_embedded
    display(HTML(figure.to_html(
        full_html=False,
        include_plotlyjs=True if not _plotly_js_embedded else False,
    )))
    _plotly_js_embedded = True

ROOT = Path.cwd()
if not (ROOT / 'output').is_dir():
    ROOT = ROOT.parent
OUT = ROOT / 'output'
STL = ROOT / 'data' / 'nik' / 'stl'
dicom_env = os.environ.get('KALMYKOV_DICOM_DIR', '').strip()
DICOM_DIR = Path(dicom_env) if dicom_env else None

# The report follows the result set selected by the executor.  Keep the
# current result as the default, but make a different validated result set
# selectable without editing this source.
result_tag = os.environ.get(
    'TRKG4_RESULT_TAG', 'auditfix_20260908').strip()
if not result_tag:
    raise ValueError('TRKG4_RESULT_TAG must be a non-empty result tag')
rejected_result_tag = 'v5_1mm_local2mm'
baseline_summary_file = OUT / 'nik_trkg4_inverse_inhale_fit_summary.csv'
baseline_comparison_file = OUT / 'nik_trkg4_inverse_inhale_comparison.csv'
baseline_available = baseline_summary_file.is_file() and baseline_comparison_file.is_file()
baseline_summary = pd.read_csv(baseline_summary_file).iloc[0] if baseline_available else None
baseline_comparison = pd.read_csv(baseline_comparison_file) if baseline_available else None
pre_rerun_dir = OUT / 'repro_check_20260901_before'
pre_rerun_summary_file = pre_rerun_dir / f'nik_trkg4_inverse_inhale_fit_summary_{rejected_result_tag}.csv'
pre_rerun_comparison_file = pre_rerun_dir / f'nik_trkg4_inverse_inhale_comparison_{rejected_result_tag}.csv'
pre_rerun_available = pre_rerun_summary_file.is_file() and pre_rerun_comparison_file.is_file()
pre_rerun_summary = (
    pd.read_csv(pre_rerun_summary_file).iloc[0] if pre_rerun_available else None)
pre_rerun_comparison = (
    pd.read_csv(pre_rerun_comparison_file) if pre_rerun_available else None)
legacy_patch_tags = {
    'earliest': 'repro_oldpoint_legacy_patch',
    'reconstructed_fast': 'repro_oldpoint_legacy_fast_patch',
}
legacy_patch_files = {
    key: {
        'summary': OUT / f'nik_trkg4_inverse_inhale_fit_summary_{tag}.csv',
        'comparison': OUT / f'nik_trkg4_inverse_inhale_comparison_{tag}.csv',
        'electrodes': OUT / f'nik_trkg4_inverse_inhale_electrodes_{tag}.csv',
        'history': OUT / f'nik_trkg4_inverse_inhale_history_{tag}.csv',
    }
    for key, tag in legacy_patch_tags.items()
}
pre_rerun_electrodes_file = pre_rerun_dir / (
    f'nik_trkg4_inverse_inhale_electrodes_{rejected_result_tag}.csv')
legacy_patch_available = (
    pre_rerun_electrodes_file.is_file()
    and all(path.is_file() for group in legacy_patch_files.values()
            for path in group.values()))
if legacy_patch_available:
    legacy_patch_summary = {
        key: pd.read_csv(paths['summary']).iloc[0]
        for key, paths in legacy_patch_files.items()}
    legacy_patch_comparison = {
        key: pd.read_csv(paths['comparison'])
        for key, paths in legacy_patch_files.items()}
    legacy_patch_electrodes = {
        key: pd.read_csv(paths['electrodes'])
        for key, paths in legacy_patch_files.items()}
    legacy_patch_history = {
        key: pd.read_csv(paths['history'])
        for key, paths in legacy_patch_files.items()}
primary_files = {
    'summary': OUT / f'nik_trkg4_inverse_inhale_fit_summary_{result_tag}.csv',
    'comparison': OUT / f'nik_trkg4_inverse_inhale_comparison_{result_tag}.csv',
    'depth': OUT / f'nik_trkg4_inverse_inhale_depth_curve_{result_tag}.csv',
    'electrodes': OUT / f'nik_trkg4_inverse_inhale_electrodes_{result_tag}.csv',
    'grid_geometry': OUT / f'nik_trkg4_inverse_inhale_grid_geometry_{result_tag}.csv',
    'parameter_sensitivity': OUT / f'nik_trkg4_inverse_inhale_parameter_sensitivity_{result_tag}.csv',
    'history': OUT / f'inverse_inhale_history_{result_tag}.csv',
    'matlab_history': OUT / f'nik_trkg4_inverse_inhale_history_{result_tag}.csv',
    'driver_history': OUT / f'nik_trkg4_inverse_direct_search_history_{result_tag}.csv',
    'direct_history': OUT / f'inverse_direct_search_history_{result_tag}.csv',
    'jacobian': OUT / f'nik_trkg4_inverse_inhale_jacobian_{result_tag}.csv',
    'jacobian_diagnostics': OUT / f'nik_trkg4_inverse_inhale_jacobian_diagnostics_{result_tag}.json',
    'contract': OUT / f'nik_trkg4_inverse_inhale_contract_{result_tag}.json',
}
missing_primary = [str(path) for key, path in primary_files.items()
                   if key not in ('history', 'matlab_history', 'driver_history', 'direct_history', 'jacobian',
                                  'jacobian_diagnostics', 'contract')
                   and not path.is_file()]
if missing_primary:
    raise FileNotFoundError(
        'У выбранного result_tag отсутствуют основные CSV: '
        + ', '.join(missing_primary))

summary = pd.read_csv(primary_files['summary']).iloc[0]
comparison = pd.read_csv(primary_files['comparison'])
depth = pd.read_csv(primary_files['depth'])
electrodes = pd.read_csv(primary_files['electrodes'])
grids = pd.read_csv(primary_files['grid_geometry'])
sensitivity = pd.read_csv(primary_files['parameter_sensitivity'])
history_file = None
history_available = False
history_source = None
history = pd.DataFrame(columns=['evaluation', 'objective'])


def _contract_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f'Поле contract.{name} должно иметь тип bool')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


contract_available = primary_files['contract'].is_file()
if contract_available:
    result_contract = json.loads(
        primary_files['contract'].read_text(encoding='utf-8'))
    if result_contract.get('result_tag') != result_tag:
        raise ValueError(
            f'contract result_tag={result_contract.get("result_tag")!r} '
            f'не совпадает с выбранным {result_tag!r}')
    for field in ('parameters', 'inputs', 'patch_builder_mode',
                  'compute_jacobian', 'geometry_qc_passed', 'artifacts'):
        if field not in result_contract:
            raise ValueError(
                f'В nik_trkg4_inverse_inhale_contract_{result_tag}.json '
                f'отсутствует поле {field!r}')
    if not isinstance(result_contract['parameters'], dict):
        raise ValueError('contract.parameters должен быть объектом')
    if (not isinstance(result_contract['inputs'], list)
            or not result_contract['inputs']
            or not all(isinstance(record, dict)
                       for record in result_contract['inputs'])):
        raise ValueError('contract.inputs должен быть непустым массивом записей')
    if not isinstance(result_contract['patch_builder_mode'], str) or not result_contract['patch_builder_mode'].strip():
        raise ValueError('contract.patch_builder_mode должен быть непустой строкой')
    if result_contract.get('status') != 'numerical_candidate_not_validated':
        raise ValueError(
            'Основной контракт должен иметь status='
            'numerical_candidate_not_validated')
    compute_jacobian = _contract_bool(
        result_contract['compute_jacobian'], 'compute_jacobian')
    geometry_qc_passed = _contract_bool(
        result_contract['geometry_qc_passed'], 'geometry_qc_passed')

    parameter_contract_names = {
        'rho_soft_ohm_m', 'rho_lungs_ohm_m', 'centre_u_mm', 'centre_v_mm', 'phi_deg'}
    missing_parameters = sorted(
        parameter_contract_names - set(result_contract['parameters']))
    if missing_parameters:
        raise ValueError(
            'В contract.parameters отсутствуют: ' + ', '.join(missing_parameters))
    for name in parameter_contract_names:
        contract_value = float(result_contract['parameters'][name])
        summary_value = float(summary[name])
        if not np.isclose(contract_value, summary_value, rtol=0, atol=1e-10):
            raise ValueError(
                f'contract.parameters[{name!r}]={contract_value} не совпадает '
                f'с summary={summary_value}')

    artifact_records = result_contract['artifacts']
    if not isinstance(artifact_records, list) or not artifact_records:
        raise ValueError('contract.artifacts должен быть непустым массивом')
    verified_artifacts = set()
    for record in artifact_records:
        if not isinstance(record, dict) or not all(
                field in record for field in ('name', 'path', 'sha256')):
            raise ValueError('Каждый contract.artifacts должен иметь name/path/sha256')
        name = str(record['name'])
        relative_path = Path(str(record['path']))
        if (relative_path.is_absolute()
                or relative_path.as_posix() != f'output/{name}'):
            raise ValueError(
                f'Артефакт {name} должен иметь path=output/{name!s}')
        path = (ROOT / relative_path).resolve()
        if path.parent != OUT.resolve() or path.name != name:
            raise ValueError(f'Недопустимый путь артефакта {name}: {relative_path}')
        if not path.is_file():
            raise FileNotFoundError(f'Артефакт из contract.artifacts не найден: {path}')
        if not isinstance(record['sha256'], str) or len(record['sha256']) != 64:
            raise ValueError(f'Некорректный SHA-256 для {name}')
        if _sha256(path).lower() != record['sha256'].lower():
            raise ValueError(f'Хэш артефакта не совпал: {name}')
        verified_artifacts.add(name)
    required_artifact_names = {
        primary_files['summary'].name,
        primary_files['comparison'].name,
        primary_files['parameter_sensitivity'].name,
        primary_files['jacobian'].name,
        primary_files['jacobian_diagnostics'].name,
    }
    missing_artifacts = sorted(required_artifact_names - verified_artifacts)
    if missing_artifacts:
        raise ValueError(
            'В contract.artifacts отсутствуют обязательные файлы: '
            + ', '.join(missing_artifacts))

    if not primary_files['jacobian_diagnostics'].is_file():
        raise FileNotFoundError(
            f'Отсутствует обязательный jacobian_diagnostics для нового tag {result_tag}')
    jacobian_diagnostics = json.loads(
        primary_files['jacobian_diagnostics'].read_text(encoding='utf-8'))
    expected_diagnostics_status = (
        'finite_step_secants_not_pose_fisher_information'
        if compute_jacobian else 'not_computed')
    if jacobian_diagnostics.get('semantic_status') != expected_diagnostics_status:
        raise ValueError(
            'Некорректный semantic_status в nik_trkg4_inverse_inhale_jacobian_diagnostics')
    step_check_details = jacobian_diagnostics.get('step_check_details')
    step_check = jacobian_diagnostics.get('step_check')
    if compute_jacobian:
        if not isinstance(step_check_details, list) or not step_check_details:
            raise ValueError(
                'jacobian_diagnostics.step_check_details должен быть непустым массивом')
        if not isinstance(step_check, dict) or not isinstance(step_check.get('details'), list):
            raise ValueError(
                'jacobian_diagnostics.step_check.details должен быть массивом')
        for details in (step_check_details, step_check['details']):
            for detail in details:
                if not isinstance(detail, dict):
                    raise ValueError('Каждая проверка шага должна быть записью')
                if not any(name in detail for name in ('parameter', 'parameter_index')):
                    raise ValueError('Проверка шага не содержит parameter/parameter_index')
                if not any(name in detail for name in (
                        'signed_step', 'step', 'requested_step',
                        'finite_difference_step_requested')):
                    raise ValueError('Проверка шага не содержит сведения о шаге')
    jacobian_step_diagnostics_ok = True
    sensitivity_status = sensitivity.get('status')
    if sensitivity_status is None:
        raise ValueError('Новый parameter_sensitivity CSV обязан иметь колонку status')
    status_values = {str(value).strip().lower()
                     for value in sensitivity_status.dropna()}
    allowed_jacobian_statuses = {
        'valid', 'valid_after_step_reduction', 'valid_with_bound_clipping',
        'unavailable_bounds', 'unavailable_geometry'}
    if compute_jacobian and (
            not status_values or not status_values.issubset(allowed_jacobian_statuses)):
        raise ValueError(
            'Для compute_jacobian=true status parameter_sensitivity содержит '
            f'неподдерживаемые статусы: {sorted(status_values - allowed_jacobian_statuses)}')
    if not compute_jacobian:
        if status_values != {'not_computed'}:
            raise ValueError(
                'Для compute_jacobian=false status должен быть not_computed')
        numeric_diagnostics = sensitivity.drop(
            columns=['status', 'parameter', 'difference_scheme', 'plausible_scale'],
            errors='ignore').apply(pd.to_numeric, errors='coerce')
        if numeric_diagnostics.size and np.isfinite(
                numeric_diagnostics.to_numpy(float)).any():
            raise ValueError(
                'При compute_jacobian=false диагностические значения должны быть NaN')
else:
    # The preserved old tag is allowed as a visible historical/current snapshot.
    # It is never labelled as a new validated run and cannot satisfy the new
    # artifact contract by implication.
    result_contract = {
        'result_tag': result_tag,
        'status': 'legacy_current_snapshot_without_contract',
    }
    compute_jacobian = primary_files['jacobian'].is_file()
    geometry_qc_passed = None
    jacobian_diagnostics = None
    jacobian_step_diagnostics_ok = False
    sensitivity_status = sensitivity.get('status')
    status_values = ({str(value).strip().lower()
                      for value in sensitivity_status.dropna()}
                     if sensitivity_status is not None else {'legacy_unlabelled'})

if contract_available:
    # A verified driver history can include rejected proposals omitted by finalise.
    history_candidates = [primary_files[key] for key in (
        'driver_history', 'direct_history', 'matlab_history', 'history')
        if primary_files[key].name in verified_artifacts]
else:
    history_candidates = [primary_files['history'], primary_files['matlab_history']]

history_file = next((path for path in history_candidates if path.is_file()), None)
history_available = history_file is not None
history_source = history_file.name if history_available else None
history = (pd.read_csv(history_file) if history_available
           else pd.DataFrame(columns=['evaluation', 'objective']))
# Keep the unfiltered, verified rows for status counts; the plot uses finite candidates.
report_history_counts = None
if contract_available and history_available:
    report_history_counts = {'rows': len(history), 'valid': None, 'invalid': None}
    history_objective = pd.to_numeric(history.get(
        'objective', pd.Series(np.nan, index=history.index)), errors='coerce')
    if 'status' in history:
        row_status = history.status.fillna('').astype(str).str.strip()
        report_history_counts['valid'] = int((
            row_status.eq('evaluated') & np.isfinite(history_objective)).sum())
        report_history_counts['invalid'] = int(row_status.eq('invalid_geometry').sum())
    else:
        report_history_counts['finite_objective'] = int(np.isfinite(history_objective).sum())
if history_available and 'objective' in history:
    history = history[history.objective < 1e11].copy()

# A contract can legitimately publish a NaN Jacobian when the producer only
# wrote the compatibility columns.  All SVD/linear-fit cells must skip that
# case explicitly instead of reusing an older sensitivity result.
jacobian_numeric_available = False
if compute_jacobian and primary_files['jacobian'].is_file():
    _jacobian_probe = pd.read_csv(primary_files['jacobian'])
    _jacobian_values = _jacobian_probe.drop(
        columns=['L_mm'], errors='ignore').apply(pd.to_numeric, errors='coerce')
    jacobian_numeric_available = (
        not _jacobian_values.empty
        and _jacobian_values.shape[1] == 5
        and np.isfinite(_jacobian_values.to_numpy(float)).all())


# Visible status describes only the selected candidate and its verified artifacts.
expected_units = {'coordinates': 'mm', 'resistivity': 'ohm_m',
                  'transfer_impedance': 'ohm', 'contact_impedance': 'ohm_m2'}
if contract_available:
    qc_text = ('геометрический QC пройден по контракту'
               if geometry_qc_passed else 'геометрический QC не пройден по контракту')
    unit_records = result_contract.get('units', {})
    units_match = isinstance(unit_records, dict) and all(
        unit_records.get(key) == value for key, value in expected_units.items())
    units_text = ('единицы контракта согласованы: мм, Ом·м, Ом, Ом·м²'
                  if units_match else
                  'единицы в контракте отсутствуют или не совпадают с ожидаемыми мм, Ом·м, Ом, Ом·м²')
    candidate_text = ('numerical_candidate_not_validated — численный кандидат; '
                      'физическая и экспериментальная валидация не выполнена этим отчётом')
else:
    qc_text = ('исторический результат, не прошедший приёмку по новому QC; '
               'новый QC не подтверждён контрактом')
    units_text = 'единицы отчёта: мм, Ом·м, Ом, Ом·м²; контрактной проверки единиц нет'
    candidate_text = 'исторический снимок без нового контракта; физическая валидация не установлена'

if report_history_counts is None:
    history_status_text = 'hash-verified история отсутствует; valid/invalid не установлены'
elif report_history_counts['valid'] is None:
    history_status_text = (
        f"{history_source}: {report_history_counts['rows']} строк, "
        f"с конечным objective — {report_history_counts['finite_objective']}; "
        'valid/invalid_geometry не размечены, число отклонений неизвестно')
else:
    history_unknown = (report_history_counts['rows'] - report_history_counts['valid']
                       - report_history_counts['invalid'])
    history_status_text = (
        f"{history_source}: valid/evaluated={report_history_counts['valid']}, "
        f"invalid_geometry={report_history_counts['invalid']}, "
        f"не классифицировано={history_unknown}")

if not contract_available:
    secant_status_text = 'исторические секущие без новой шаговой диагностики; числовые разделы пропущены'
elif not compute_jacobian:
    secant_status_text = 'not_computed (compute_jacobian=false); SVD и дыхательная линейная подгонка пропущены'
else:
    available_secants = int(np.isfinite(_jacobian_values.to_numpy(float)).all(axis=0).sum())
    secant_status_text = (
        f"{jacobian_diagnostics['semantic_status']}; конечных столбцов {available_secants}/5; "
        f"статусы: {', '.join(sorted(status_values))}. "
        + ('SVD — диагностика линейной суррогатной модели по секущим.'
           if jacobian_numeric_available else
           'SVD и дыхательная линейная подгонка пропущены: секущие частично недоступны.'))
report_status_markdown = (
    f'### Статус выбранного расчёта `{result_tag}`\n\n'
    f'- QC: {qc_text}; {units_text}.\n'
    f'- История: {history_status_text}. Это сохранённые строки, а не счётчик FEM-решений.\n'
    f'- Секущие: {secant_status_text}\n'
    f'- Статус результата: {candidate_text}.')
display(Markdown(report_status_markdown))

rejected_files = {
    'summary': OUT / f'nik_trkg4_inverse_inhale_fit_summary_{rejected_result_tag}.csv',
    'comparison': OUT / f'nik_trkg4_inverse_inhale_comparison_{rejected_result_tag}.csv',
    'history': OUT / f'nik_trkg4_inverse_inhale_history_{rejected_result_tag}.csv',
}
rejected_available = all(path.is_file() for path in rejected_files.values())
if rejected_available:
    rejected_summary = pd.read_csv(rejected_files['summary']).iloc[0]
    rejected_comparison = pd.read_csv(rejected_files['comparison'])
    rejected_history = pd.read_csv(rejected_files['history'])

historical_pem_files = {
    'pem_summary': OUT / 'nik_trkg4_fast_pem_scan_summary.csv',
    'pem_all': OUT / 'nik_trkg4_fast_pem_scan_all.csv',
    'pem_validation': OUT / 'nik_trkg4_fast_pem_validation.csv',
    'cem_summary': OUT / f'nik_trkg4_fast_cem_refine_summary_{rejected_result_tag}.csv',
    'cem': OUT / f'nik_trkg4_fast_cem_refine_{rejected_result_tag}.csv',
}
historical_pem_available = all(path.is_file() for path in historical_pem_files.values())
if historical_pem_available:
    fast_pem_summary = pd.read_csv(historical_pem_files['pem_summary']).iloc[0]
    fast_pem_all = pd.read_csv(historical_pem_files['pem_all'])
    fast_pem_validation = pd.read_csv(historical_pem_files['pem_validation'])
    fast_cem_summary = pd.read_csv(historical_pem_files['cem_summary']).iloc[0]
    fast_cem = pd.read_csv(historical_pem_files['cem'])
ablation_tags = {
    'oldpoint_legacy': 'ablation_fixed_oldpoint_legacy100',
    'oldpoint_manifest': 'ablation_fixed_oldpoint_manifest90',
    'oldstart_manifest': 'ablation_opt_oldstart_manifest90',
}
ablation_files = {
    key: {
        'summary': OUT / f'nik_trkg4_inverse_inhale_fit_summary_{tag}.csv',
        'comparison': OUT / f'nik_trkg4_inverse_inhale_comparison_{tag}.csv',
        'history': OUT / f'nik_trkg4_inverse_inhale_history_{tag}.csv',
    }
    for key, tag in ablation_tags.items()
}
ablation_available = all(
    path.is_file() for group in ablation_files.values() for path in group.values())
if ablation_available:
    ablation_summary = {
        key: pd.read_csv(paths['summary']).iloc[0]
        for key, paths in ablation_files.items()
    }
    ablation_comparison = {
        key: pd.read_csv(paths['comparison'])
        for key, paths in ablation_files.items()
    }
    ablation_history = {
        key: pd.read_csv(paths['history'])
        for key, paths in ablation_files.items()
    }
restartable_tags = {
    'legacy_fast': 'legacy_fast_restartable_manifest90',
    'disjoint': 'disjoint_restartable_manifest90',
}
restartable_files = {
    key: {
        'summary': OUT / f'nik_trkg4_inverse_inhale_fit_summary_{tag}.csv',
        'comparison': OUT / f'nik_trkg4_inverse_inhale_comparison_{tag}.csv',
        'electrodes': OUT / f'nik_trkg4_inverse_inhale_electrodes_{tag}.csv',
        'history': OUT / f'inverse_direct_search_history_{tag}.csv',
        'search_summary': OUT / f'nik_trkg4_inverse_direct_search_summary_{tag}.json',
    }
    for key, tag in restartable_tags.items()
}
restartable_available = all(
    path.is_file() for group in restartable_files.values() for path in group.values())
if restartable_available:
    restartable_summary = {
        key: pd.read_csv(paths['summary']).iloc[0]
        for key, paths in restartable_files.items()}
    restartable_comparison = {
        key: pd.read_csv(paths['comparison'])
        for key, paths in restartable_files.items()}
    restartable_electrodes = {
        key: pd.read_csv(paths['electrodes'])
        for key, paths in restartable_files.items()}
    restartable_history = {
        key: pd.read_csv(paths['history'])
        for key, paths in restartable_files.items()}
    restartable_search = {
        key: json.loads(paths['search_summary'].read_text(encoding='utf-8'))
        for key, paths in restartable_files.items()}

print(f'Проект: {ROOT}')
print(f'Основной result_tag: {result_tag}')
print(f'FEM: {int(summary.mesh_nodes):,} узлов, {int(summary.mesh_tetrahedra):,} тетраэдров')
print(f'Происхождение результата: '
      f'{"новый contract-tag, контракт проверен" if contract_available else "сохранённый старый tag/history без нового контракта"}')
print(f'Контракт: статус={result_contract["status"]}; '
      f'geometry_qc_passed={geometry_qc_passed}; '
      f'compute_jacobian={compute_jacobian}')
print(f'История основного поиска: {history_source or "отсутствует; график поиска пропущен"}')
print(f'Jacobian diagnostics: шаговые проверки={jacobian_step_diagnostics_ok}')
print(f'DICOM: {DICOM_DIR}' if DICOM_DIR is not None else
      'DICOM: не задан; визуализация HU будет пропущена')
print('Историческая сетка v4: доступна' if baseline_available else
      'Историческая сетка v4: файлы отсутствуют, сравнение будет пропущено')
print('Снимок v5 до перепрогона: доступен' if pre_rerun_available else
      'Снимок v5 до перепрогона: отсутствует')
print('Причинные контрольные прогоны: доступны' if ablation_available else
      'Причинные контрольные прогоны: отсутствуют')
print('Расширенный прямой поиск: доступен' if restartable_available else
      'Расширенный прямой поиск: отсутствует')
print('Реконструкция прежних контактных функций: доступна' if legacy_patch_available else
      'Реконструкция прежних контактных функций: отсутствует')
""")

md(r"""
## 2. Исходная DICOM-серия и согласование координат

Используется серия `2 / LUNG`, а не scout и не кардиальная фазовая реконструкция. Косой срез
интерполируется непосредственно из HU-объёма в системе координат пациента. STL-модели
экспортированы в той же системе миллиметров, поэтому дополнительная ручная регистрация не вводится.
""")

code(r"""
def load_dicom_series(dicom_dir):
    if dicom_dir is None:
        return None
    dicom_files = list(dicom_dir.glob('*CT_NA.2.*.dcm'))
    if not dicom_files:
        return None

    headers = []
    for path in dicom_files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        headers.append((path, ds))

    iop = np.asarray(headers[0][1].ImageOrientationPatient, float)
    direction_columns = iop[:3]
    direction_rows = iop[3:]
    direction_slices = np.cross(direction_columns, direction_rows)
    headers.sort(key=lambda item: np.dot(
        np.asarray(item[1].ImagePositionPatient, float), direction_slices))

    first = headers[0][1]
    rows, columns = int(first.Rows), int(first.Columns)
    pixel_spacing = np.asarray(first.PixelSpacing, float)
    volume_hu = np.empty((len(headers), rows, columns), dtype=np.float32)
    slice_coordinates = np.empty(len(headers))

    for index, (path, header) in enumerate(headers):
        ds = pydicom.dcmread(str(path))
        volume_hu[index] = (ds.pixel_array.astype(np.float32) *
                            float(ds.RescaleSlope) + float(ds.RescaleIntercept))
        slice_coordinates[index] = np.dot(
            np.asarray(ds.ImagePositionPatient, float), direction_slices)

    origin = np.asarray(headers[0][1].ImagePositionPatient, float)
    row_coordinates = np.dot(origin, direction_rows) + np.arange(rows) * pixel_spacing[0]
    column_coordinates = (np.dot(origin, direction_columns) +
                          np.arange(columns) * pixel_spacing[1])
    hu_interpolator = RegularGridInterpolator(
        (slice_coordinates, row_coordinates, column_coordinates),
        volume_hu, bounds_error=False, fill_value=np.nan
    )
    info = pd.DataFrame({
        'параметр': ['SeriesDescription', 'число срезов', 'матрица', 'pixel spacing, мм',
                     'шаг по z, мм', 'диапазон z, мм', 'HU slope/intercept'],
        'значение': [str(first.SeriesDescription), len(headers), f'{rows}×{columns}',
                     f'{pixel_spacing[0]:.6f} × {pixel_spacing[1]:.6f}',
                     f'{np.median(np.diff(slice_coordinates)):.6f}',
                     f'{slice_coordinates.min():.3f} … {slice_coordinates.max():.3f}',
                     f'{float(first.RescaleSlope):g} / {float(first.RescaleIntercept):g}']
    })
    return (hu_interpolator, direction_slices, direction_rows,
            direction_columns, column_coordinates, info)

dicom_series = load_dicom_series(DICOM_DIR)
if dicom_series is None:
    hu_interpolator = None
    direction_slices = direction_rows = direction_columns = None
    column_coordinates = None
    display(Markdown(
        '**DICOM не загружен.** Задайте `KALMYKOV_DICOM_DIR` для HU-фона. '
        'Расчёт по STL/FEM и геометрический срез продолжаются без него.'
    ))
else:
    (hu_interpolator, direction_slices, direction_rows,
     direction_columns, column_coordinates, dicom_info) = dicom_series
    display(dicom_info)
""")

code(r"""
def load_stl_triangles(path):
    return stl_mesh.Mesh.from_file(str(path)).vectors.astype(np.float64)

triangles = {
    'body': load_stl_triangles(STL / 'body_solid_v3_1mm_r10_volume_fill.stl'),
    'lungs': load_stl_triangles(STL / 'lungs_solid_v3_volume_fill.stl'),
    'heart': load_stl_triangles(STL / 'heart.stl'),
    'bones': load_stl_triangles(STL / 'bones.stl'),
}

# The plotting limits must cover the complete body.  Earlier versions used a
# torso crop (x=-230...170 mm) even though the arms extend to x=-373...355 mm;
# Plotly therefore showed artificial open cuts at both arm ends.
body_points = triangles['body'].reshape(-1, 3)
body_min = body_points.min(axis=0)
body_max = body_points.max(axis=0)
body_span = body_max - body_min
body_margin = np.maximum(5.0, 0.025 * body_span)
body_ranges = [
    [float(body_min[k] - body_margin[k]), float(body_max[k] + body_margin[k])]
    for k in range(3)
]

def closed_mesh_diagnostics(tri):
    points, inverse = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(edges, axis=1)
    unique_edges, edge_counts = np.unique(undirected, axis=0, return_counts=True)
    graph = coo_matrix(
        (np.ones(2 * len(unique_edges)),
         (np.r_[unique_edges[:, 0], unique_edges[:, 1]],
          np.r_[unique_edges[:, 1], unique_edges[:, 0]])),
        shape=(len(points), len(points))
    )
    components = int(connected_components(graph, directed=False, return_labels=False))
    chi = int(len(points) - len(unique_edges) + len(faces))
    genus = (2 * components - chi) / 2
    signed_volume_l = np.sum(
        np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    ) / 6e6
    return {
        'components': components,
        'vertices': len(points),
        'triangles': len(faces),
        'boundary edges': int(np.sum(edge_counts == 1)),
        'non-manifold edges': int(np.sum(edge_counts > 2)),
        'Euler characteristic': chi,
        'genus': genus,
        'signed volume, L': signed_volume_l,
    }

def nearest_triangle_surface_distance(point, tri):
    # Exact Euclidean distance from one point to a triangular surface.
    point = np.asarray(point, dtype=float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum('ij,ij->i', normal, normal)
    valid = normal_sq > 1e-18

    signed_scale = np.zeros(len(tri))
    signed_scale[valid] = np.einsum('ij,ij->i', ap[valid], normal[valid]) / normal_sq[valid]
    projection = point - signed_scale[:, None] * normal
    v2 = projection - a
    dot00 = np.einsum('ij,ij->i', ab, ab)
    dot01 = np.einsum('ij,ij->i', ab, ac)
    dot11 = np.einsum('ij,ij->i', ac, ac)
    dot20 = np.einsum('ij,ij->i', v2, ab)
    dot21 = np.einsum('ij,ij->i', v2, ac)
    denom = dot00 * dot11 - dot01 ** 2
    bary_valid = np.abs(denom) > 1e-18
    u = np.full(len(tri), np.nan)
    v = np.full(len(tri), np.nan)
    u[bary_valid] = (dot11[bary_valid] * dot20[bary_valid] - dot01[bary_valid] * dot21[bary_valid]) / denom[bary_valid]
    v[bary_valid] = (dot00[bary_valid] * dot21[bary_valid] - dot01[bary_valid] * dot20[bary_valid]) / denom[bary_valid]
    inside = valid & bary_valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    plane_distance = np.linalg.norm(point - projection, axis=1)

    def segment_distance(x, y):
        edge = y - x
        edge_sq = np.einsum('ij,ij->i', edge, edge)
        t = np.zeros(len(edge))
        nonzero = edge_sq > 1e-18
        t[nonzero] = np.einsum('ij,ij->i', point - x[nonzero], edge[nonzero]) / edge_sq[nonzero]
        t = np.clip(t, 0.0, 1.0)
        return np.linalg.norm(point - (x + t[:, None] * edge), axis=1)

    edge_distance = np.minimum.reduce([
        segment_distance(a, b), segment_distance(b, c), segment_distance(c, a)
    ])
    distance = np.where(inside, plane_distance, edge_distance)
    return float(np.min(distance))

centre = summary[['centre_x_mm', 'centre_y_mm', 'centre_z_mm']].to_numpy(dtype=float)
h_centre_surface_mm = nearest_triangle_surface_distance(centre, triangles['lungs'])
h_centre_vertex_mm = float(summary.h_centre_mm)
axis = summary[['axis_x', 'axis_y', 'axis_z']].to_numpy(dtype=float)
axis /= np.linalg.norm(axis)

# The second in-plane direction points from the skin centre toward the nearest lung surface.
lung_points = triangles['lungs'].reshape(-1, 3)
nearest_lung = lung_points[np.argmin(np.linalg.norm(lung_points - centre, axis=1))]
inward = nearest_lung - centre
inward -= np.dot(inward, axis) * axis
inward /= np.linalg.norm(inward)
plane_normal = np.cross(axis, inward)
plane_normal /= np.linalg.norm(plane_normal)

geometry_frame = pd.DataFrame(
    [centre, axis, inward, plane_normal],
    index=['центр, мм', 'ось a', 'направление внутрь b', 'нормаль плоскости n'],
    columns=['x', 'y', 'z']
)
display(geometry_frame)
""")

md(r"""
## 3. Трёхмерная анатомия, ось и электроды

Интерактивная модель ниже показывает наружную поверхность, заполненные лёгкие, сердце, кости,
все 36 фактических центроидов электродных площадок и отдельно сборку 140 мм. Прямоугольник —
плоскость косого CT-среза из следующего раздела. Все координаты даны в миллиметрах.
В общей сцене все анатомические объекты показаны разными цветами и полупрозрачно, как в HTML
ручной установки электродов. Каркасная сетка поверх STL не рисуется.

Сразу после общей анатомической сцены приведён второй, специально контрастный внешний вид:
в нём показаны только полная непрореженная непрозрачная поверхность тела и электроды, без лёгких,
сердца и костей. Он нужен именно для проверки прилегания сборок к коже. Любой слой можно
временно выключить нажатием на его название в легенде Plotly.

Вторая интерактивная 3D-схема специально упрощена и показывает смысл $\varphi$: это угол между
исходной осью $\mathbf a_0$ и найденной осью $\mathbf a_\varphi$ **внутри касательной плоскости
кожи**. Чёрная стрелка — нормаль кожи. Таким образом, $\varphi$ не является независимым
наклоном всей плоскости сборки относительно тела.
""")

md(r"""
### Контроль замкнутости и масштаба наружной поверхности

Ниже проверяется рабочий `body_solid_v3_1mm_r10_volume_fill.stl`, из которого построена объёмная FEM-сетка v5. Важны три независимых свойства:

1. `boundary edges = 0` — у поверхности нет открытых краёв;
2. `non-manifold edges = 0` — каждое ребро принадлежит ровно двум треугольникам;
3. одна компонента и `genus = 0` — наружная оболочка топологически эквивалентна сфере и не содержит сквозных тоннелей.

Прежняя визуализация ошибочно ограничивала координату $x$ диапазоном $-230\ldots170$ мм, хотя полная поверхность с руками занимает примерно $-373\ldots355$ мм. Plotly пересекал STL плоскостями границ графика, поэтому торцы рук выглядели как настоящие отверстия. Это был дефект **только визуализации**: диапазоны осей не участвуют в FEM-расчёте. Теперь показана вся поверхность, включена ортографическая камера, а единицы по всем осям одинаковы (`aspectmode='data'`).

Большой размер по $x$ — не масштабная деформация: это размах обеих рук. Поэтому полный объект физически значительно шире, чем толще в передне-заднем направлении. Для проверки ниже явно напечатаны границы и размеры по каждой оси.

Крайние участки удлинённых рук выходят за поле зрения выбранной DICOM-серии ($x\approx-205\ldots186$ мм). Их торцы являются закрывающими крышками конечной FEM-области, а не CT-анатомией и не отверстиями. Диапазон электродов $x\approx-154\ldots-146$ мм и удаление от торца более 200 мм относились к исторической позе. Для выбранного тега положение следует проверять по его координатам; эти расстояния автоматически не переносятся. Это допущение наружной FEM-границы нужно учитывать отдельно от точности CT-анатомии.
""")

code(r"""
body_topology = pd.DataFrame([closed_mesh_diagnostics(triangles['body'])])
body_bounds = pd.DataFrame({
    'ось': ['x', 'y', 'z'],
    'минимум, мм': body_min,
    'максимум, мм': body_max,
    'полный размер, мм': body_span,
})
extent_rows = [
    {'объект': 'наружная FEM-оболочка',
     'минимум x, мм': body_min[0], 'максимум x, мм': body_max[0]},
    {'объект': 'все электроды по x',
     'минимум x, мм': electrodes.patch_centroid_x_mm.min(),
     'максимум x, мм': electrodes.patch_centroid_x_mm.max()},
]
if column_coordinates is not None:
    extent_rows.insert(1, {
        'объект': 'поле DICOM по x',
        'минимум x, мм': column_coordinates.min(),
        'максимум x, мм': column_coordinates.max(),
    })
extent_check = pd.DataFrame(extent_rows)
display(body_topology)
display(body_bounds)
display(extent_check)
""")

md(r"""
### Какая пространственная дискретизация здесь нужна

Нужно различать три разных шага:

- исходный DICOM: $0.765625\times0.765625\times1.25$ мм;
- растр, использованный при заполнении наружной оболочки тела: 1 мм;
- локальный размер поверхностных треугольников и тетраэдров FEM около электродов.

Растр заполнения тела уменьшен с 2 до **1 мм**. При таком разрешении морфологическое замыкание радиусом 8 мм уже сохраняло 12 тоннелей (`genus=12`) и не позволяло заполнить полости; минимальным проверенным радиусом стал 10 мм. Он дал замкнутый объём 20.171 л, практически совпадающий со старым объёмом 20.156 л. Переход сразу к 0.5 мм увеличил бы число вокселей ещё примерно в 8 раз, но не добавил бы исходной КТ-информации между срезами 1.25 мм.

В новой FEM-сетке локальные размеры заданы как 2, 4 и 8 мм в цилиндрах радиусом 20, 40 и 70 мм около принятой оси; вне их — до 25 мм. Итоговая Delaunay-сетка содержит 153 415 узлов и 721 943 тетраэдра. Сумма объёмов тетраэдров 20.170763 л совпадает с STL до машинной точности. Вариант Frontal был отвергнут: он заполнил только 17.31 л. Медианный эквивалентный шаг наружных треугольников уменьшился с 2.56 до 1.52 мм.

При этом уменьшение вокселя не исправляет видимые «отверстия» и не исправляет пропорции графика: первое было вызвано обрезкой диапазона Plotly, второе проверяется физическими размерами и одинаковым масштабом осей.
""")

code(r"""
def sampled_mesh_trace(tri, max_faces, name, color, opacity):
    count = len(tri)
    take = np.linspace(0, count - 1, min(count, max_faces), dtype=int)
    selected = tri[take]
    points, inverse = np.unique(selected.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, color=color, opacity=opacity,
        flatshading=False, hoverinfo='name', lighting=dict(
            ambient=0.68, diffuse=0.72, specular=0.12, roughness=0.82,
            fresnel=0.05
        ), lightposition=dict(x=-300, y=-200, z=100)
    )

fig3d = go.Figure()
fig3d.add_trace(sampled_mesh_trace(triangles['body'], len(triangles['body']), 'кожа / тело', '#d8c7b6', 0.14))
fig3d.add_trace(sampled_mesh_trace(triangles['lungs'], len(triangles['lungs']), 'лёгкие v3', 'deepskyblue', 0.28))
fig3d.add_trace(sampled_mesh_trace(triangles['heart'], len(triangles['heart']), 'сердце', 'crimson', 0.55))
fig3d.add_trace(sampled_mesh_trace(triangles['bones'], len(triangles['bones']), 'кости', 'ivory', 0.24))

all_xyz = electrodes[['patch_centroid_x_mm', 'patch_centroid_y_mm', 'patch_centroid_z_mm']].to_numpy()
fig3d.add_trace(go.Scatter3d(
    x=all_xyz[:, 0], y=all_xyz[:, 1], z=all_xyz[:, 2], mode='markers',
    marker=dict(size=3, color=electrodes.L_mm, colorscale='Viridis', colorbar=dict(title='L, мм')),
    text=[f'L={L:g}, {e}' for L, e in zip(electrodes.L_mm, electrodes.electrode)],
    hovertemplate='%{text}<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>',
    name='все сборки'
))

grid140 = electrodes[electrodes.L_mm == 140]
p140 = grid140[['patch_centroid_x_mm', 'patch_centroid_y_mm', 'patch_centroid_z_mm']].to_numpy()
fig3d.add_trace(go.Scatter3d(
    x=p140[:, 0], y=p140[:, 1], z=p140[:, 2], mode='markers+text',
    marker=dict(size=7, color=['red', 'orange', 'lime', 'blue']),
    text=grid140.electrode, textposition='top center', name='сборка 140 мм'
))

axis_points = centre + np.array([-85, 85])[:, None] * axis
fig3d.add_trace(go.Scatter3d(
    x=axis_points[:, 0], y=axis_points[:, 1], z=axis_points[:, 2],
    mode='lines', line=dict(color='magenta', width=8), name='общая ось'
))
fig3d.add_trace(go.Scatter3d(
    x=[centre[0]], y=[centre[1]], z=[centre[2]], mode='markers',
    marker=dict(size=8, color='black', symbol='diamond'), name='общий центр'
))

corners = np.array([
    centre - 90 * axis - 15 * inward,
    centre + 90 * axis - 15 * inward,
    centre + 90 * axis + 125 * inward,
    centre - 90 * axis + 125 * inward,
])
fig3d.add_trace(go.Mesh3d(
    x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
    i=[0, 0], j=[1, 2], k=[2, 3], color='gold', opacity=0.18,
    name='плоскость CT-среза'
))

fig3d.update_layout(
    title='CT/STL-анатомия и положение электродных сборок',
    height=720, margin=dict(l=0, r=0, t=45, b=0),
    scene=dict(
        aspectmode='data',
        xaxis=dict(title='x, мм', range=body_ranges[0]),
        yaxis=dict(title='y, мм', range=body_ranges[1]),
        zaxis=dict(title='z, мм', range=body_ranges[2]),
        camera=dict(eye=dict(x=-1.7, y=-1.4, z=0.7),
                    projection=dict(type='orthographic'))
    )
)
show_plotly(fig3d)

# A separate exterior-only view is intentionally much less transparent. It
# avoids the visual competition between the body shell and internal organs.
surface_fig = go.Figure()
surface_fig.add_trace(sampled_mesh_trace(
    triangles['body'], len(triangles['body']), 'полная наружная поверхность', '#d8a56d', 1.0
))
surface_fig.add_trace(go.Scatter3d(
    x=all_xyz[:, 0], y=all_xyz[:, 1], z=all_xyz[:, 2], mode='markers',
    marker=dict(size=5.5, color=electrodes.L_mm, colorscale='Viridis',
                colorbar=dict(title='L, мм')),
    text=[f'L={L:g}, {e}' for L, e in zip(electrodes.L_mm, electrodes.electrode)],
    hovertemplate='%{text}<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>',
    name='центры электродных площадок'
))
surface_fig.add_trace(go.Scatter3d(
    x=p140[:, 0], y=p140[:, 1], z=p140[:, 2], mode='markers+text',
    marker=dict(size=9, color=['red', 'orange', 'lime', 'blue'],
                line=dict(color='black', width=1.5)),
    text=grid140.electrode, textposition='top center', name='сборка 140 мм'
))
surface_fig.add_trace(go.Scatter3d(
    x=axis_points[:, 0], y=axis_points[:, 1], z=axis_points[:, 2], mode='lines',
    line=dict(color='magenta', width=9), name='общая ось'
))
surface_fig.add_trace(go.Scatter3d(
    x=[centre[0]], y=[centre[1]], z=[centre[2]], mode='markers',
    marker=dict(size=9, color='black', symbol='diamond'), name='общий центр'
))
surface_fig.update_layout(
    title='Наружная поверхность тела — отдельный контрастный вид без внутренних органов',
    height=720, margin=dict(l=0, r=0, t=45, b=0),
    paper_bgcolor='white',
    scene=dict(
        bgcolor='rgb(235,241,247)',
        aspectmode='data',
        xaxis=dict(title='x, мм', range=body_ranges[0]),
        yaxis=dict(title='y, мм', range=body_ranges[1]),
        zaxis=dict(title='z, мм', range=body_ranges[2]),
        camera=dict(eye=dict(x=-1.7, y=-1.4, z=0.7),
                    projection=dict(type='orthographic'))
    )
)
show_plotly(surface_fig)

# A deliberately simple local 3-D diagram of phi. The geometry is expressed
# in the tangent frame of the skin centre so the small fitted angle is visible.
def rodrigues(vector, rotation_axis, angle_rad):
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    return (vector * np.cos(angle_rad)
            + np.cross(rotation_axis, vector) * np.sin(angle_rad)
            + rotation_axis * np.dot(rotation_axis, vector) * (1 - np.cos(angle_rad)))

phi_rad = np.deg2rad(float(summary.phi_deg))
skin_outward = -inward
reference_axis_3d = rodrigues(axis, skin_outward, -phi_rad)
reference_axis_3d -= np.dot(reference_axis_3d, skin_outward) * skin_outward
reference_axis_3d /= np.linalg.norm(reference_axis_3d)
reference_side_3d = np.cross(skin_outward, reference_axis_3d)
reference_side_3d /= np.linalg.norm(reference_side_3d)

axis_local = np.array([
    np.dot(axis, reference_axis_3d),
    np.dot(axis, reference_side_3d),
    np.dot(axis, skin_outward),
])

phi_fig = go.Figure()
plane_u = np.array([-90, 90, 90, -90])
plane_v = np.array([-28, -28, 28, 28])
phi_fig.add_trace(go.Mesh3d(
    x=plane_u, y=plane_v, z=np.zeros(4), i=[0, 0], j=[1, 2], k=[2, 3],
    color='lightblue', opacity=0.28, name='касательная плоскость кожи'
))
phi_fig.add_trace(go.Scatter3d(
    x=[-85, 85], y=[0, 0], z=[0, 0], mode='lines+text',
    line=dict(color='gray', width=8, dash='dash'), text=['', 'a₀'],
    textposition='top center', name='исходная ось a₀'
))
fitted_line = np.array([-85, 85])[:, None] * axis_local
phi_fig.add_trace(go.Scatter3d(
    x=fitted_line[:, 0], y=fitted_line[:, 1], z=fitted_line[:, 2], mode='lines+text',
    line=dict(color='magenta', width=9), text=['', 'aφ'],
    textposition='bottom center', name='найденная ось aφ'
))
nominal_offsets = np.array([-70, -35, 35, 70])
nominal_points = nominal_offsets[:, None] * axis_local
phi_fig.add_trace(go.Scatter3d(
    x=nominal_points[:, 0], y=nominal_points[:, 1], z=nominal_points[:, 2],
    mode='markers+text', marker=dict(size=7, color=['red', 'orange', 'lime', 'blue']),
    text=['I+', 'V+', 'V−', 'I−'], textposition='top center', name='электроды L=140 мм'
))
phi_fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, 38], mode='lines+text',
    line=dict(color='black', width=7), text=['', 'n кожи'],
    textposition='top center', name='нормаль кожи'
))
angles = np.linspace(0, phi_rad, 40)
arc = np.column_stack((70 * np.cos(angles), 70 * np.sin(angles), np.zeros_like(angles)))
phi_fig.add_trace(go.Scatter3d(
    x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode='lines+text',
    line=dict(color='darkorange', width=10),
    text=[''] * 39 + [f'φ={summary.phi_deg:.2f}°'],
    textposition='middle right', name='угол φ'
))
phi_fig.update_layout(
    title='Что означает φ: поворот оси внутри касательной плоскости кожи',
    height=560, margin=dict(l=0, r=0, t=50, b=0),
    scene=dict(
        xaxis_title='направление исходной оси, мм',
        yaxis_title='поперёк оси, мм', zaxis_title='нормаль, мм',
        xaxis=dict(range=[-95, 95]), yaxis=dict(range=[-32, 32]), zaxis=dict(range=[-10, 45]),
        aspectmode='manual', aspectratio=dict(x=2.6, y=1.1, z=0.8),
        camera=dict(eye=dict(x=1.15, y=-1.65, z=1.15))
    )
)
show_plotly(phi_fig)
""")

md(r"""
### 3.1. Где разрешалось искать центр и угол

Оптимизатор не перебирал $x,y,z$ независимо. Он перемещал исходную точку по двум направлениям
касательной плоскости на $u,v\in[-120,120]$ мм, после чего каждая пробная точка проецировалась
обратно на кожу. Поэтому допустимая область в 3D — криволинейный участок поверхности, а не
прямоугольный параллелепипед. Жёлтая область ниже показывает этот участок.

Угол меняется в уточнённом диапазоне $\varphi\in[-20^\circ,+20^\circ]$ относительно исходной оси.
Минимальная и максимальная серые линии показывают границы этого диапазона. Диапазоны $x,y,z$
в таблице — ограничивающий параллелепипед подсвеченного участка, а не три независимых интервала оптимизации.

Для этой отдельной иллюстрации жёлтый участок получен на сетке $61\times61$ и отображён через
ближайшие вершины STL тела, поэтому его зубчатая граница и приведённые диапазоны $x,y,z$ являются
визуальной дискретной оценкой. В самой обратной задаче каждая точка проецировалась на ближайший
треугольник FEM-поверхности с вычислением барицентрических координат; это более точная операция,
чем использованная только для рисунка выборка вершин.
""")

code(r"""
reference_grid = pd.read_csv(STL.parent / 'electrodes' / 'electrodes_4_right_rib_140mm_xyz_mm.csv')
reference_xyz = reference_grid[['x_mm', 'y_mm', 'z_mm']].to_numpy(dtype=float)
reference_centre = reference_xyz.mean(axis=0)
reference_axis = reference_xyz[-1] - reference_xyz[0]
reference_axis /= np.linalg.norm(reference_axis)

body_points = np.unique(triangles['body'].reshape(-1, 3), axis=0)
body_tree = cKDTree(body_points)
_, nearest_reference_index = body_tree.query(reference_centre)
reference_surface = body_points[nearest_reference_index]

# Use the local inward direction already obtained from CT; outward is its opposite.
reference_normal = skin_outward.copy()
reference_u = reference_axis - np.dot(reference_axis, reference_normal) * reference_normal
reference_u /= np.linalg.norm(reference_u)
reference_v = np.cross(reference_normal, reference_u)
reference_v /= np.linalg.norm(reference_v)

u_samples = np.linspace(-120, 120, 61)
v_samples = np.linspace(-120, 120, 61)
U_search, V_search = np.meshgrid(u_samples, v_samples)
raw_search = (reference_surface
              + U_search.reshape(-1, 1) * reference_u
              + V_search.reshape(-1, 1) * reference_v)
_, projected_indices = body_tree.query(raw_search)
search_patch_xyz = body_points[projected_indices]

archive_search_values = (
    [pre_rerun_summary.centre_u_mm, pre_rerun_summary.centre_v_mm,
     pre_rerun_summary.phi_deg, pre_rerun_summary.centre_x_mm,
     pre_rerun_summary.centre_y_mm, pre_rerun_summary.centre_z_mm]
    if pre_rerun_available else [np.nan] * 6)
search_limits = pd.DataFrame({
    'параметр': ['u, мм', 'v, мм', 'φ, град', 'x, мм (вычисляемый)',
                 'y, мм (вычисляемый)', 'z, мм (вычисляемый)'],
    'минимум': [-120, -120, -20,
                search_patch_xyz[:, 0].min(), search_patch_xyz[:, 1].min(), search_patch_xyz[:, 2].min()],
    'максимум': [120, 120, 20,
                 search_patch_xyz[:, 0].max(), search_patch_xyz[:, 1].max(), search_patch_xyz[:, 2].max()],
    'итоговый disjoint': [summary.centre_u_mm, summary.centre_v_mm, summary.phi_deg,
                summary.centre_x_mm, summary.centre_y_mm, summary.centre_z_mm],
    'архив до перепрогона': archive_search_values,
    'тип ограничения': ['независимая граница', 'независимая граница', 'независимая граница',
                        'следствие проекции', 'следствие проекции', 'следствие проекции']
})
display(search_limits)

primary_electrode_xyz = electrodes[
    ['patch_centroid_x_mm', 'patch_centroid_y_mm',
     'patch_centroid_z_mm']].to_numpy(float)
search_fig = go.Figure()
search_fig.add_trace(sampled_mesh_trace(triangles['body'], len(triangles['body']), 'поверхность тела', '#d8c7b6', 0.24))
search_fig.add_trace(go.Scatter3d(
    x=primary_electrode_xyz[:, 0], y=primary_electrode_xyz[:, 1],
    z=primary_electrode_xyz[:, 2], mode='markers',
    marker=dict(size=5.5, color=electrodes.L_mm, colorscale='Turbo',
                cmin=50, cmax=140, colorbar=dict(title='L, мм'),
                line=dict(color='black', width=0.6)),
    text=[f'L={L:g} мм; {name}' for L, name in zip(
        electrodes.L_mm, electrodes.electrode)],
    hovertemplate='%{text}<br>x=%{x:.1f}; y=%{y:.1f}; z=%{z:.1f} мм<extra></extra>',
    name='итоговые электродные площадки'
))
search_fig.add_trace(go.Scatter3d(
    x=search_patch_xyz[:, 0], y=search_patch_xyz[:, 1], z=search_patch_xyz[:, 2],
    mode='markers', marker=dict(size=3.2, color='#ffd21f', opacity=0.80),
    name='разрешённая область центра'
))
search_fig.add_trace(go.Scatter3d(
    x=[reference_surface[0]], y=[reference_surface[1]], z=[reference_surface[2]],
    mode='markers+text', marker=dict(size=8, color='black'), text=['исходный центр'],
    textposition='top center', name='исходный центр'
))
search_fig.add_trace(go.Scatter3d(
    x=[centre[0]], y=[centre[1]], z=[centre[2]], mode='markers+text',
    marker=dict(size=9, color='magenta', symbol='diamond'), text=['итоговый disjoint'],
    textposition='top center', name='итоговый disjoint'
))
if pre_rerun_available:
    archive_search_centre = pre_rerun_summary[
        ['centre_x_mm', 'centre_y_mm', 'centre_z_mm']].to_numpy(float)
    archive_search_axis = pre_rerun_summary[['axis_x', 'axis_y', 'axis_z']].to_numpy(float)
    archive_search_axis /= np.linalg.norm(archive_search_axis)
    archive_search_line = (
        archive_search_centre + np.array([-70, 70])[:, None] * archive_search_axis)
    search_fig.add_trace(go.Scatter3d(
        x=archive_search_line[:, 0], y=archive_search_line[:, 1],
        z=archive_search_line[:, 2], mode='lines',
        line=dict(color='#d62728', width=7, dash='dash'), name='архивная ось'))
    search_fig.add_trace(go.Scatter3d(
        x=[archive_search_centre[0]], y=[archive_search_centre[1]],
        z=[archive_search_centre[2]], mode='markers+text',
        marker=dict(size=8, color='#d62728', symbol='square'),
        text=['архивная координата'], textposition='bottom center',
        name='архивная координата'))

for angle_deg, color, dash, label in [(-20, 'gray', 'dash', 'φ min = −20°'),
                                      (20, 'gray', 'dash', 'φ max = +20°'),
                                      (summary.phi_deg, 'magenta', 'solid', 'итоговая φ disjoint')]:
    direction = rodrigues(reference_u, reference_normal, np.deg2rad(angle_deg))
    endpoints = centre + np.array([-70, 70])[:, None] * direction
    search_fig.add_trace(go.Scatter3d(
        x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2], mode='lines',
        line=dict(color=color, width=7, dash=dash), name=label
    ))

search_fig.update_layout(
    title='Область поиска и итоговое положение сборок на полной STL-поверхности', height=700,
    margin=dict(l=0, r=0, t=45, b=0),
    scene=dict(aspectmode='data',
               xaxis=dict(title='x, мм', range=body_ranges[0]),
               yaxis=dict(title='y, мм', range=body_ranges[1]),
               zaxis=dict(title='z, мм', range=body_ranges[2]),
               camera=dict(eye=dict(x=-1.7, y=-1.2, z=0.7),
                           projection=dict(type='orthographic')))
)
show_plotly(search_fig)
display(Markdown(
    'Цветные точки показывают центроиды 36 итоговых электродных площадок `disjoint`. '
    'Пурпурные центр и ось соответствуют основному расчёту; красные пунктирная ось '
    'и квадрат — архивной координате до перепрогона. Жёлтые точки задают разрешённую '
    'область положения общего центра. Показана вся наружная STL-поверхность без '
    'пространственной обрезки.'))
""")

md(r"""
## 4. Настоящий косой CT-срез через электродную ось

Координаты точки плоскости:

$$
\mathbf r(s,d)=\mathbf c+s\,\mathbf a_\varphi+d\,\mathbf b,
$$

где $s$ идёт вдоль электродной сборки, а $d>0$ — от кожи внутрь грудной клетки. Значения HU
получены трилинейной интерполяцией исходного DICOM-объёма. Цветные линии — независимое
пересечение этой же плоскости с STL-моделями, полученными по КТ. Их совпадение с анатомическими границами на HU
является визуальной проверкой регистрации.

Важно различать следующие координаты и расстояния:

1. В FEM электрод — это набор граничных треугольников. Его фактическая координата в таблице —
   центроид площади этой площадки в 3D.
2. После проекции четырёх электродов на криволинейную кожу они в общем случае **не лежат точно
   в одной плоскости среза**. На рисунке кругом показана ортогональная проекция 3D-центроида в
   плоскость. Потерянная координата $\ell=(\mathbf r_e-\mathbf c)\cdot\mathbf n_{plane}$ указана
   в таблице ниже.
3. Крест на жёлтой линии — пересечение самой плоскости с кожей при том же $s$. Пунктир между
   крестом и кругом показывает, почему проекция центроида может визуально оказаться выше или
   ниже контура кожи.
4. Зелёный отрезок $h_{plane}(s)$ — расстояние между контурами кожи и лёгкого **в этой
   плоскости**. Это не евклидово расстояние от конкретного 3D-электрода до лёгкого.
5. В исходном MATLAB-коде для мягкого априорного штрафа обратной задачи использовалось $h_c^{vertex}$ —
   расстояние от общего центра до ближайшей **вершины** STL лёгкого (`knnsearch`). Это
   дискретная аппроксимация расстояния до поверхности. Для сравнения с обычной двухслойной
   моделью ниже отдельно вычисляется точное расстояние до ближайшей точки треугольника
   $h_c^{surface}$. В самом FEM ни одна из этих глубин явно не подставляется: используется
   полная трёхмерная геометрия.
""")

code(r"""
def plane_segments(tri, point, normal, basis_s, basis_d, tolerance=1e-7):
    signed = np.einsum('fvi,i->fv', tri - point, normal)
    candidate = tri[(signed.min(axis=1) <= tolerance) & (signed.max(axis=1) >= -tolerance)]
    candidate_signed = signed[(signed.min(axis=1) <= tolerance) & (signed.max(axis=1) >= -tolerance)]
    segments = []
    for face, distances in zip(candidate, candidate_signed):
        hits = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            di, dj = distances[i], distances[j]
            if abs(di) <= tolerance:
                hits.append(face[i])
            if di * dj < 0:
                t = di / (di - dj)
                hits.append(face[i] + t * (face[j] - face[i]))
        if len(hits) < 2:
            continue
        unique = np.unique(np.round(np.asarray(hits), 6), axis=0)
        if len(unique) < 2:
            continue
        if len(unique) > 2:
            distances_pair = np.linalg.norm(unique[:, None] - unique[None, :], axis=2)
            i, j = np.unravel_index(np.argmax(distances_pair), distances_pair.shape)
            unique = unique[[i, j]]
        relative = unique - point
        segments.append(np.column_stack((relative @ basis_s, relative @ basis_d)))
    return segments

s_grid = np.linspace(-95, 95, 381)
d_grid = np.linspace(-20, 135, 311)
S, D = np.meshgrid(s_grid, d_grid)
points = centre + S[..., None] * axis + D[..., None] * inward
if hu_interpolator is None:
    oblique_hu = None
else:
    queries = np.column_stack((
        points.reshape(-1, 3) @ direction_slices,
        points.reshape(-1, 3) @ direction_rows,
        points.reshape(-1, 3) @ direction_columns,
    ))
    oblique_hu = hu_interpolator(queries).reshape(D.shape)

section_segments = {
    key: plane_segments(value, centre, plane_normal, axis, inward)
    for key, value in triangles.items()
}

def contour_intersections_at_s(segments, s_value):
    depths = []
    for segment in segments:
        x0, y0 = segment[0]
        x1, y1 = segment[1]
        if (x0 - s_value) * (x1 - s_value) > 0 or abs(x1 - x0) < 1e-9:
            continue
        fraction = (s_value - x0) / (x1 - x0)
        if -1e-8 <= fraction <= 1 + 1e-8:
            depths.append(y0 + fraction * (y1 - y0))
    return np.asarray(depths)

def unique_sorted_depths(values, minimum_separation=1e-3):
    '''Collapse duplicate STL edge hits while retaining distinct branches.'''
    ordered = np.sort(np.asarray(values, dtype=float))
    ordered = ordered[np.isfinite(ordered)]
    if len(ordered) == 0:
        return ordered
    unique = [ordered[0]]
    for value in ordered[1:]:
        if value - unique[-1] > minimum_separation:
            unique.append(value)
    return np.asarray(unique)

# h in the displayed plane: choose the skin crossing nearest d=0 and the
# first lung crossing deeper than that skin point.
s_profile = np.arange(-80.0, 80.1, 2.0)
skin_profile = np.full_like(s_profile, np.nan)
lung_profile = np.full_like(s_profile, np.nan)
lung_crossings_by_s = []
for index, s_value in enumerate(s_profile):
    skin_hits = contour_intersections_at_s(section_segments['body'], s_value)
    if len(skin_hits) == 0:
        lung_crossings_by_s.append(np.asarray([], dtype=float))
        continue
    skin_profile[index] = skin_hits[np.argmin(np.abs(skin_hits))]
    lung_hits = contour_intersections_at_s(section_segments['lungs'], s_value)
    deeper = unique_sorted_depths(
        lung_hits[lung_hits > skin_profile[index] + 1.0])
    lung_crossings_by_s.append(deeper)
    if len(deeper):
        lung_profile[index] = deeper.min()
max_lung_branches = max((len(values) for values in lung_crossings_by_s), default=0)
lung_branch_profiles = np.full(
    (len(s_profile), max_lung_branches), np.nan, dtype=float)
for index, values in enumerate(lung_crossings_by_s):
    lung_branch_profiles[index, :len(values)] = values
h_plane = lung_profile - skin_profile
# This is a geometric skin-to-first-lung-boundary distance in the selected
# plane.  It is not a soft-tissue thickness unless every intervening tissue
# class has been explicitly segmented and removed from the path.
bone_crossings_by_s = []
for index, s_value in enumerate(s_profile):
    if not (np.isfinite(skin_profile[index]) and np.isfinite(lung_profile[index])):
        bone_crossings_by_s.append(np.asarray([], dtype=float))
        continue
    bone_hits = unique_sorted_depths(
        contour_intersections_at_s(section_segments['bones'], s_value))
    bone_crossings_by_s.append(bone_hits[
        (bone_hits > skin_profile[index] + 1.0)
        & (bone_hits < lung_profile[index] - 1.0)])
bone_path_flags = np.array([len(values) > 0 for values in bone_crossings_by_s])

fig, ax = plt.subplots(figsize=(13, 8))
if oblique_hu is not None:
    image = ax.imshow(
        oblique_hu, extent=[s_grid.min(), s_grid.max(), d_grid.min(), d_grid.max()],
        origin='lower', cmap='gray', vmin=-1000, vmax=450, aspect='equal'
    )
else:
    image = None
    ax.set_facecolor('#1d2730')
colors = {'body': 'yellow', 'lungs': '#00e5ff', 'heart': '#ff334f', 'bones': 'white'}
labels = {'body': 'кожа', 'lungs': 'лёгкие v3', 'heart': 'сердце', 'bones': 'кости'}
for key in ('body', 'lungs', 'heart', 'bones'):
    ax.add_collection(LineCollection(section_segments[key], colors=colors[key], linewidths=1.4, label=labels[key]))

projected140 = p140 - centre
electrode_s = projected140 @ axis
electrode_d = projected140 @ inward
electrode_lateral = projected140 @ plane_normal
skin_d_at_electrode_s = np.interp(electrode_s, s_profile, skin_profile)
h_plane_at_electrode_s = np.interp(electrode_s, s_profile, h_plane)
nearest_lung_from_patch = np.array([
    np.min(np.linalg.norm(lung_points - point, axis=1)) for point in p140
])

ax.scatter(electrode_s, electrode_d, s=65, c=['red', 'orange', 'lime', 'blue'], edgecolor='black', zorder=5)
ax.scatter(electrode_s, skin_d_at_electrode_s, s=75, marker='x', c='yellow',
           linewidths=2.2, zorder=6, label='кожа в плоскости при том же s')
for x, y_projected, y_skin in zip(electrode_s, electrode_d, skin_d_at_electrode_s):
    ax.plot([x, x], [y_projected, y_skin], color='yellow', ls='--', lw=1.1, alpha=0.85)
for x, y, label in zip(electrode_s, electrode_d, grid140.electrode):
    ax.annotate(label, (x, y), xytext=(4, -12), textcoords='offset points', color='yellow', weight='bold')

for s_mark in (-60, -30, 0, 30, 60):
    index = np.argmin(np.abs(s_profile - s_mark))
    if np.isfinite(h_plane[index]):
        ax.plot([s_profile[index], s_profile[index]],
                [skin_profile[index], lung_profile[index]], color='lime', lw=2.2)
        ax.text(s_profile[index] + 1.5,
                0.5 * (skin_profile[index] + lung_profile[index]),
                f'h={h_plane[index]:.1f}', color='lime', fontsize=8, rotation=90,
                va='center')

ax.axhline(0, color='magenta', ls=':', lw=1, label='центр/касательная кожи')
ax.set(xlabel='$s$ вдоль сборки, мм', ylabel='$d$ внутрь, мм',
       title=('Косой DICOM-срез в плоскости общей оси электродов'
              if oblique_hu is not None else
              'STL-контуры в плоскости общей оси электродов (без DICOM)'))
ax.set_xlim(-95, 95)
ax.set_ylim(135, -20)  # skin at the top, depth increases downwards
ct_legend = ax.legend(loc='lower right', ncol=2, facecolor='#111820',
                      edgecolor='white', framealpha=0.88)
for text_item in ct_legend.get_texts():
    text_item.set_color('white')
if image is not None:
    fig.colorbar(image, ax=ax, label='HU', shrink=0.8)
fig.tight_layout()

electrode_plane_table = pd.DataFrame({
    'электрод': grid140.electrode.to_numpy(),
    's проекции, мм': electrode_s,
    'd проекции, мм': electrode_d,
    'ℓ вне плоскости, мм': electrode_lateral,
    'd кожи в плоскости, мм': skin_d_at_electrode_s,
    'расстояние кожа–лёгкое в плоскости, мм': h_plane_at_electrode_s,
    'кость между кожей и лёгким в плоскости': np.interp(
        electrode_s, s_profile, bone_path_flags.astype(float)) > 0.5,
    '3D min до лёгкого от центроида, мм': nearest_lung_from_patch,
})
display(electrode_plane_table)
""")

md(r"""
### 4.1. Отдельный STL-срез той же плоскостью

На предыдущем рисунке HU-фон и STL-контуры показаны вместе. Ниже оставлена только геометрия,
которая действительно поступает в FEM. Поэтому этот рисунок не зависит от оконного уровня HU,
интерполяции DICOM и визуального совпадения полутонов. Это точное пересечение плоскости
$\mathbf r(s,d)=\mathbf c+s\mathbf a_\varphi+d\mathbf b$ с треугольными поверхностями STL.

- толстая жёлтая линия — наружная поверхность тела;
- голубая — заполненная модель лёгких;
- красная — сердце;
- белая — кости;
- цветные круги — ортогональные проекции центров электродных площадок;
- жёлтые кресты — точки пересечения плоскости с кожей при том же $s$;
- зелёные отрезки — геометрическое расстояние $d_{skin\to lung}(s)$ от кожи до
  первого контура лёгкого. Если отрезок пересекает кость, это не является толщиной
  мягких тканей.

Если DICOM-срез выглядит неоднозначно, именно этот STL-срез следует использовать для проверки
положения электродов и закона изменения $h$. При этом он показывает не исходные HU, а уже
предобработанную геометрию расчётной модели.
""")

code(r"""
stl_fig, stl_ax = plt.subplots(figsize=(13, 8), facecolor='#101820')
stl_ax.set_facecolor('#101820')
stl_colors = {'body': '#ffd400', 'lungs': '#00e5ff', 'heart': '#ff4055', 'bones': '#ffffff'}
stl_widths = {'body': 3.8, 'lungs': 2.8, 'heart': 2.4, 'bones': 1.8}
stl_labels = {'body': 'наружная поверхность тела', 'lungs': 'лёгкие v3',
              'heart': 'сердце', 'bones': 'кости'}
for key in ('body', 'lungs', 'heart', 'bones'):
    stl_ax.add_collection(LineCollection(
        section_segments[key], colors=stl_colors[key], linewidths=stl_widths[key],
        label=stl_labels[key], zorder=2 if key == 'body' else 3
    ))

electrode_colors = ['#ff3b30', '#ff9500', '#34c759', '#0a84ff']
stl_ax.scatter(electrode_s, electrode_d, s=90, c=electrode_colors,
               edgecolor='black', linewidth=0.8, zorder=7, label='проекции электродов')
stl_ax.scatter(electrode_s, skin_d_at_electrode_s, s=90, marker='x', c='#ffd400',
               linewidths=2.5, zorder=8, label='кожа при том же s')
for x, y_projected, y_skin, label in zip(
        electrode_s, electrode_d, skin_d_at_electrode_s, grid140.electrode):
    stl_ax.plot([x, x], [y_projected, y_skin], color='#ffd400', ls='--', lw=1.2, alpha=0.9)
    stl_ax.annotate(label, (x, y_projected), xytext=(5, -14), textcoords='offset points',
                    color='white', weight='bold', zorder=9)

for s_mark in (-60, -30, 0, 30, 60):
    index = np.argmin(np.abs(s_profile - s_mark))
    if np.isfinite(h_plane[index]):
        stl_ax.plot([s_profile[index], s_profile[index]],
                    [skin_profile[index], lung_profile[index]], color='#39ff14', lw=3.0, zorder=6)
        stl_ax.text(s_profile[index] + 1.8,
                    0.5 * (skin_profile[index] + lung_profile[index]),
                    f'h={h_plane[index]:.1f} мм', color='#39ff14', fontsize=9,
                    rotation=90, va='center', zorder=9)

stl_ax.axhline(0, color='#ff2dff', ls=':', lw=1.2, label='центр / касательная кожи')
stl_ax.set_xlim(-95, 95)
stl_ax.set_ylim(135, -20)
stl_ax.set_aspect('equal', adjustable='box')
stl_ax.set_xlabel('$s$ вдоль общей оси электродов, мм', color='white')
stl_ax.set_ylabel('$d$ от кожи внутрь, мм', color='white')
stl_ax.set_title('STL-срез в плоскости электродов — геометрия, используемая FEM',
                 color='white', weight='bold')
stl_ax.tick_params(colors='white')
for spine in stl_ax.spines.values():
    spine.set_color('#b0bec5')
legend = stl_ax.legend(loc='lower right', ncol=2, facecolor='#18242e',
                       edgecolor='#b0bec5', framealpha=0.95)
for text_item in legend.get_texts():
    text_item.set_color('white')
stl_fig.tight_layout()
""")

md(r"""
## 5. Геометрическое расстояние кожа–лёгкое и эффективный $h$

В показанной плоскости непосредственно определяется расстояние между первым пересечением
наружной поверхности тела и первым более глубоким пересечением контура лёгкого:

$$
d_{skin\to lung}(s)=d_{lung}(s)-d_{skin}(s).
$$

Именно эти отрезки зелёным цветом нанесены на DICOM-срез. Это расстояние кожа–лёгкое
в выбранном срезе, а не автоматически толщина мягких тканей. Текущая разметка содержит
кости отдельным классом, и отрезок от кожи до лёгкого может пересекать кость. Поэтому
толщина мягких тканей без костей в текущем отчёте не считается.

Дополнительно из полной трёхмерной CT/STL-геометрии вычисляются две диагностические величины:

$$
h_{ray}(s)=\min\{t>0:\ \mathbf r_{skin}(s)-t\,\mathbf n_{skin}(s)\in\partial\Omega_{lung}\},
$$

$$
h_{near}(s)=\min_{\mathbf q\in\partial\Omega_{lung}}
\|\mathbf r_{skin}(s)-\mathbf q\|.
$$

$h_{ray}$ соответствует движению внутрь по локальной нормали кожи, а $h_{near}$ — ближайшей
точке лёгкого. Луч может пройти мимо ближней поверхности, поэтому отдельные выбросы $h_{ray}$
не используются для аппроксимации. Гладкий полином четвёртой степени робастно подгоняется к
$d_{skin\to lung}(s)$, включая возможный путь через кость. Это не оценка мягкотканной
толщины. Электрический параметр $h_{eff}$ классической аналитической модели — отдельный
эффективный параметр, который должен идентифицироваться по критерию эквивалентности с FEM;
он не объявляется равным ни расстоянию кожа–лёгкое, ни толщине мягких тканей без отдельной
проверки.
**Эта аппроксимация не заменяет STL в FEM-решателе:** при расчёте используется полная трёхмерная граница лёгких;
кривая нужна для контроля, интерпретации и возможной редуцированной модели.
""")

code(r"""
valid_plane = np.isfinite(h_plane) & (s_profile >= -70) & (s_profile <= 70)
s_valid = s_profile[valid_plane]
h_valid = h_plane[valid_plane]
t_valid = s_valid / 70.0

# Robust iteratively reweighted fourth-degree fit. This prevents a single
# contour ambiguity from controlling the reduced h(s) approximation.
design = np.column_stack([t_valid ** power for power in range(5)])
weights = np.ones(len(h_valid))
for _ in range(12):
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_h = h_valid * np.sqrt(weights)
    coefficients = np.linalg.lstsq(weighted_design, weighted_h, rcond=None)[0]
    residual_fit = h_valid - design @ coefficients
    robust_scale = max(1.4826 * np.median(np.abs(residual_fit - np.median(residual_fit))), 0.25)
    scaled = np.abs(residual_fit) / (1.5 * robust_scale)
    weights = np.where(scaled <= 1, 1.0, 1.0 / scaled)

s_dense = np.linspace(-70, 70, 401)
t_dense = s_dense / 70.0
h_approx = np.polynomial.polynomial.polyval(t_dense, coefficients)
h_at_samples = np.polynomial.polynomial.polyval(t_valid, coefficients)
h_fit_rmse = np.sqrt(np.mean((h_at_samples - h_valid) ** 2))

terms = [f'{coefficients[0]:.3f}']
for power, coefficient in enumerate(coefficients[1:], start=1):
    terms.append(f'{coefficient:+.3f}t^{power}')
display(Markdown(
    '$$h_{approx}(s)=' + ''.join(terms) +
    r',\qquad t=s/70.$$' +
    f'\n\nRMSE относительно расстояния кожа–лёгкое между контурами на показанном срезе: **{h_fit_rmse:.3f} мм**.'
))

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(s_profile, h_plane, 'o-', ms=3.5, color='tab:green',
        label='$d_{skin\to lung}$: между контурами на DICOM/STL-срезе')
ax.plot(depth.axis_s_mm, depth.h_inward_ray_mm, 'o', alpha=0.35,
        label='$h_{ray}$: локальная нормаль кожи (диагностика)')
ax.plot(depth.axis_s_mm, depth.h_nearest_lung_mm, 's--', ms=3, alpha=0.55,
        label='$h_{near}$: ближайшая точка (диагностика)')
ax.plot(s_dense, h_approx, 'k-', lw=2.4, label='робастный полином 4-й степени')
for L in sorted(comparison.L_mm.unique()):
    ax.axvline(-L / 2, color='gray', alpha=0.05)
    ax.axvline(+L / 2, color='gray', alpha=0.05)
ax.set(xlabel='$s$ вдоль общей оси, мм', ylabel='$h(s)$, мм',
       title='Изменение геометрического расстояния кожа–лёгкое')
ax.legend()
fig.tight_layout()
""")

md(r"""
## 6. Как геометрия и эффективный $h$ учитываются в расчёте

В плоской двухслойной модели пришлось бы явно подставлять одно эффективное $h_{eff}$
или его редуцированный профиль в аналитическую формулу. В текущей CT/FEM-модели
$h_{eff}$ **не является отдельным входным числом**. Последовательность такая:

$$
\text{CT/STL кожи и лёгких}
\longrightarrow \text{tissue\_id каждого тетраэдра}
\longrightarrow \sigma(\mathbf r)
\longrightarrow \nabla\cdot(\sigma\nabla u)=0
\longrightarrow Z(L).
$$

Когда меняются $(u,v,\varphi)$ или размер $L$, электроды перемещаются по коже. Под каждым из них
меняется распределение тока в трёхмерных тканевых областях, включая кости и лёгкое. Поэтому
пространственное изменение $h$, кривизна кожи, боковой уход тока, сердце и кости входят в
решение одновременно. Показанная выше $h(s)$ — сечение этой трёхмерной геометрии для контроля.

Таким образом, геометрическое расстояние кожа–лёгкое, мягкотканная толщина без костей
и эффективный $h_{eff}$ аналитической модели — разные величины. Равенство между ними
требует отдельной идентификационной проверки.
""")

md(r"""
## 7. Обратная задача

Оптимизируемый вектор:

$$
\theta=(\rho_1,\rho_2,u,v,\varphi).
$$

Для остатков $r_i=Z_{FEM}(L_i;\theta)-Z_{exp}(L_i)$ используется функция Хьюбера

$$
H_\delta(r)=
\begin{cases}
\frac12r^2,& |r|\le\delta,\\
\delta(|r|-\frac12\delta),& |r|>\delta,
\end{cases}\qquad \delta=5\ \Omega.
$$

Полный критерий:

$$
J(\theta)=\overline{H_5(r_i)}
+0.20\left[(b_{FEM}-b_{exp})\Delta L\right]^2
+\left(\frac{h_c-15}{7}\right)^2
+P_{7\ldots30}(h_c)
+0.25\left(\frac{\rho_2-\rho_{2,CT}}{6}\right)^2.
$$

Здесь $b$ — робастный наклон $Z(L)$, $h_c$ в реализованном MATLAB-коде — расстояние от центра
до ближайшей вершины STL лёгкого, а визуальная оценка 15 мм используется только как мягкое
априорное ограничение. При достаточно плотном STL это приближает расстояние до треугольной
поверхности; величина этой дискретизационной разницы явно показана ниже. Путь кожа–лёгкое
может содержать кость: $h_c$ не является отдельно измеренной мягкотканной толщиной или $h_{eff}$.

### Почему критерий именно такой

| Часть критерия | Зачем она нужна | Откуда взялось число |
|---|---|---|
| $\overline{H_5(r_i)}$ | Согласовать все девять абсолютных значений $Z$. Хьюбер не позволяет одной плохой сборке полностью определить решение. | Порог 5 Ω взят как рабочий масштаб измерительного и модельного разброса: ниже него ошибка квадратичная, выше — растёт линейно. Это инженерный выбор, а не физическая константа. |
| $0.20[(b_{FEM}-b_{exp})\Delta L]^2$ | Отдельно сохранить общий наклон кривой $Z(L)$, даже если отдельные точки имеют выбросы. $\Delta L=140-50=90$ мм переводит ошибку наклона обратно в омы на всём диапазоне. | Вес 0.20 выбран эмпирически, чтобы наклон влиял на поиск, но не подавлял девять точечных остатков. |
| $[(h_c-15)/7]^2$ | Удержать центр около первоначальной CT-оценки и не позволить геометрии компенсировать ошибку $\rho_2$. | 15 мм — прежняя визуальная оценка по CT; 7 мм — намеренно широкая мягкая ширина, а не измеренная стандартная ошибка. |
| $P_{7\ldots30}(h_c)$ | Отсечь решения, где центр оказывается над анатомически другой областью лёгкого. | Использовано $2\max(h_c-30,0)^2+2\max(7-h_c,0)^2$. Границы 7–30 мм и коэффициент 2 заданы как анатомический защитный диапазон. |
| $0.25[(\rho_2-\rho_{2,CT})/6]^2$ | Регуляризовать $\rho_2$ около модельной оценки по КТ/HU и Maxwell-Garnett. | $\rho_{2,CT}=17.4067$ Ом·м; масштаб 6 Ом·м выбран широким, а множитель 0.25 делает априорный штраф слабее основной ошибки $Z$. |

$\rho_{2,CT}$ — модельная оценка по HU, воздушной доле и принятому сопротивлению
тканевой матрицы, а не непосредственно измеренное независимое свойство лёгкого.

Следовательно, это **регуляризованный инженерный критерий**, а не полностью идентифицированная
статистическая функция правдоподобия. Для строгой вероятностной интерпретации порог Хьюбера,
дисперсию измерений и веса априорных штрафов нужно оценить по повторным установкам электродов
и повторным КТ- и электрическим исследованиям.

### Диапазоны поиска

$$
\rho_1\in[2,10]\ \Omega\cdot\text{м},\qquad
\rho_2\in[10,32]\ \Omega\cdot\text{м},
$$

$$
u,v\in[-120,120]\ \text{мм},\qquad
\varphi\in[-20^\circ,20^\circ].
$$

$x,y,z$ получались после проекции $(u,v)$ на кожу и поэтому не имели независимых нижних и
верхних границ. Их фактический диапазон показан на отдельной 3D-визуализации выше.
""")

code(r"""
parameter_table = pd.DataFrame({
    'параметр': ['$\\rho_1$', '$\\rho_2$', '$u$', '$v$', '$\\varphi$',
                 '$x_c$', '$y_c$', '$z_c$', '$h_c$'],
    'итоговое локальное значение': [summary.rho_soft_ohm_m, summary.rho_lungs_ohm_m,
               summary.centre_u_mm, summary.centre_v_mm, summary.phi_deg,
               summary.centre_x_mm, summary.centre_y_mm, summary.centre_z_mm,
               summary.h_centre_mm],
    'единица': ['Ом·м', 'Ом·м', 'мм', 'мм', 'град', 'мм', 'мм', 'мм', 'мм'],
    'смысл': ['мягкие ткани', 'лёгкое: ткань + воздух', 'смещение по поверхности',
              'смещение по поверхности', 'поворот общей оси', 'CT-координата центра',
              'CT-координата центра', 'CT-координата центра', 'глубина до лёгкого в центре']
})
display(parameter_table)

metric_table = pd.DataFrame({
    'метрика': ['RMS', 'MAE', 'наклон эксперимента', 'наклон FEM',
                'разность наклонов', 'число обусловленности масштабированной матрицы секущих'],
    'значение': [summary.rms_residual_ohm, summary.mae_residual_ohm,
                 summary.experimental_slope_ohm_per_mm, summary.fem_slope_ohm_per_mm,
                 summary.fem_slope_ohm_per_mm - summary.experimental_slope_ohm_per_mm,
                 summary.jacobian_condition_scaled],
    'единица': ['Ом', 'Ом', 'Ом/мм', 'Ом/мм', 'Ом/мм', '—']
})
if not (compute_jacobian and jacobian_step_diagnostics_ok and jacobian_numeric_available):
    metric_table = metric_table.iloc[:5]
display(metric_table)
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(comparison.L_mm, comparison.Z_experiment_ohm, 'o-', lw=2, label='эксперимент, вдох')
ax.plot(comparison.L_mm, comparison.Z_FEM_ohm, 's-', lw=2, label=f'CT/FEM: {result_tag}')
ax.set(xlabel='L, мм', ylabel='Z, Ом', title='Измеренный и рассчитанный импеданс')
ax.legend()

ax = axes[0, 1]
colors_residual = np.where(comparison.residual_ohm >= 0, 'tab:red', 'tab:blue')
ax.bar(comparison.L_mm.astype(str), comparison.residual_ohm, color=colors_residual)
ax.axhline(0, color='black', lw=1)
ax.set(xlabel='L, мм', ylabel='$Z_{FEM}-Z_{exp}$, Ом', title='Остатки по каждой сборке')

ax = axes[1, 0]
ax.plot(grids.L_nominal_mm, grids.I_to_I_actual_mm, 'o-', label='I−I на сетке')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm, 'k--', label='I−I номинально')
ax.plot(grids.L_nominal_mm, grids.V_to_V_actual_mm, 's-', label='V−V на сетке')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm / 2, 'k:', label='V−V номинально')
ax.set(xlabel='номинальный L, мм', ylabel='фактический размах, мм', title='Дискретизация электродных площадок')
ax.legend(fontsize=8)

ax = axes[1, 1]
best_so_far = np.minimum.accumulate(history.objective.to_numpy())
ax.plot(history.evaluation, history.objective, '.', alpha=0.45, label='проверенная точка')
ax.plot(history.evaluation, best_so_far, 'r-', lw=2, label='лучшее J')
ax.set(xlabel='номер обращения к критерию', ylabel='J', title='Ход оптимизации', yscale='log')
ax.legend()

fig.tight_layout()
display(comparison)
""")

md(r"""
## 8. Секущие по шагу и диагностика дискретного CEM-оператора

В CSV сохраняются прежние колонки производных для совместимости. В отчёте они
интерпретируются только как конечные разности на явно заданных шагах:

$$
J^{(\Delta)}_{ij}=\frac{Z(L_i;\theta+\Delta_j)-Z(L_i;\theta-\Delta_j)}
{2\Delta_j}.
$$

При конечном выборе CEM-площадок малое изменение позы может оставить тот же набор
граничных граней, а затем вызвать скачок при смене площадки. Поэтому такая секущая
зависит от выбранного шага и характеризует изменение отклика на этом масштабе. Она не является
производной непрерывной позы, информацией Фишера или оценкой точности положения.
В таблице запрошенный шаг означает возмущение параметра; расстояние между пробами
для центральной схемы равно удвоенному шагу и входит в знаменатель.
Псевдообратная матрица не используется для выдачи стандартной ошибки: при потере ранга
она не превращает неидентифицируемое направление в конечную или нулевую неопределённость.
""")

jacobian_code(r"""
display(sensitivity.drop(columns=['local_crlb_std_at_sigma5ohm'], errors='ignore').rename(columns={
    'parameter': 'параметр',
    'finite_difference_step_requested': 'запрошенный шаг конечной разности',
    'finite_difference_step_effective': 'расстояние между пробами (знаменатель)',
    'difference_scheme': 'схема разности',
    'plausible_scale': 'заданный масштаб параметра',
    'jacobian_column_norm': 'норма столбца конечной секущей',
}))

display(Markdown(
    'Для сопоставления норм столбцы умножены на заданные масштабы параметров из таблицы; '
    'все показанные нормы имеют размерность Ом. Масштабы не являются оценками неопределённости.'))
fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
ax.bar(sensitivity.parameter,
       sensitivity.jacobian_column_norm * sensitivity.plausible_scale, color='tab:cyan')
ax.set_yscale('log')
ax.tick_params(axis='x', rotation=30)
ax.set(ylabel='$||J^{finite\\ step}_j s_j||_2$, Ом',
       title='Нормы секущих после масштабирования параметров')
""")

md(r"""
## 9. Допущения и границы применимости

1. Все сборки имеют один физический центр; свободны общая позиция на коже и угол $\varphi$.
2. Номинальные смещения электродов равны $[-L/2,-L/4,+L/4,+L/2]$. Индивидуальные ошибки
   расстояний пока не оптимизируются: одно значение $Z$ на сборку не позволяет независимо
   восстановить ещё девять поправок без сильной регуляризации.
3. Площадки имеют одну заданную площадь и одинаковый контактный импеданс; гель, давление и
   локальная неоднородность кожи отдельно не моделируются.
4. Мягкие ткани и лёгкие внутри своих доменов изотропны и кусочно-однородны. Кожа, жир и мышцы
   пока объединены в $\rho_1$.
5. $\rho_2$ — эффективное сопротивление заполненного лёгкого, то есть ткани вместе с воздухом,
   а не сопротивление одной паренхимы.
6. Целая маска сердца и кости фиксированы по литературным данным ITIS; их геометрия не
   оптимизируется. Отдельной маски крови нет, и она не моделируется как отдельная область.
7. Используется вещественная квазистатическая модель на 50 кГц; диэлектрическая часть и
   частотная дисперсия в этом цикле не восстанавливаются.
8. DICOM и STL считаются уже зарегистрированными в одной системе координат пациента. Косой
   HU-срез выше позволяет это проверить визуально.
9. Состояние дыхания CT и экспериментального удержания вдоха может отличаться. Это один из
   источников систематической ошибки геометрии лёгких.
10. Для Ника запись 100 мм исключена как поздняя перезаписанная копия независимой записи
    90 мм. Это субъектно-специфическое решение, а не глобальное исключение размера 100 мм.
""")

code(r"""
largest = comparison.iloc[np.argmax(np.abs(comparison.residual_ohm.to_numpy()))]
area_min, area_mean, area_max = electrodes.patch_area_mm2.agg(['min', 'mean', 'max'])
function_evaluations = summary.get('function_evaluations', np.nan)
if pd.notna(function_evaluations):
    function_evaluations_text = f'{int(float(function_evaluations)):,}'
else:
    function_evaluations_text = 'не записано'

if history_available and not history.empty and 'objective' in history:
    best_history = history.loc[history.objective.idxmin()]
    if 'evaluation' in history:
        history_text = (
            f'{history_source}: {len(history)} строк; минимум на сохранённой '
            f'оценке {best_history.evaluation:g}')
    else:
        history_text = f'{history_source}: {len(history)} строк; минимум сохранён'
else:
    history_text = 'основная история отсутствует или не содержит objective; прогресс не реконструируется'

if compute_jacobian and jacobian_step_diagnostics_ok and jacobian_numeric_available:
    jacobian_text = (
        'Конечные секущие и их спектр показаны ниже; это шагозависимая '
        'диагностика дискретной геометрии, а не неопределённость позы.')
else:
    jacobian_text = (
        'Числовая диагностика конечных секущих пропущена: J отсутствует, '
        'содержит NaN, отсутствует новая шаговая диагностика или compute_jacobian=false.')

display(Markdown(f'''## 10. Итоговый локальный результат прямого поиска

- Выбран result tag **`{result_tag}`**; в fit summary записано обращений к критерию (не FEM-решений): **{function_evaluations_text}**.
- Источник истории: **{history_text}**. Одна строка finalise является допустимым результатом и
  не расширяется до вымышленной траектории поиска.
- Получены $\\rho_1={summary.rho_soft_ohm_m:.3f}$ Ом·м и
  $\\rho_2={summary.rho_lungs_ohm_m:.3f}$ Ом·м.
- Центр оси: $({summary.centre_x_mm:.2f}, {summary.centre_y_mm:.2f}, {summary.centre_z_mm:.2f})$ мм;
  $\\varphi={summary.phi_deg:.2f}^\\circ$.
- $h_c={summary.h_centre_mm:.2f}$ мм; полная кривая $h(s)$ показана выше.
- RMS = **{summary.rms_residual_ohm:.2f} Ом**, MAE = **{summary.mae_residual_ohm:.2f} Ом**.
- Наклон: эксперимент {summary.experimental_slope_ohm_per_mm:.4f}, FEM {summary.fem_slope_ohm_per_mm:.4f} Ом/мм.
- Наибольший остаток: L={largest.L_mm:g} мм, {largest.residual_ohm:+.2f} Ом.
- Площади площадок после сгущения: {area_min:.2f}…{area_max:.2f} мм², средняя {area_mean:.2f} мм².

{jacobian_text}

Под «лучшим» здесь понимается только наименьшее значение заранее заданного критерия $J$ среди
проверенных FEM-вариантов. Критерий объединяет остатки отдельных $Z(L)$, расхождение наклона,
CT-ограничение глубины и мягкий априорный штраф для $\\rho_2$. Это не означает доказанной
анатомической истинности или единственности решения.
'''))
""")

md(r"""
## 11. Что было бы при обычной двухслойной и при «адаптированной» модели
**Ограничение сравнения.** Во всех расчётах этого раздела используются параметры
выбранного кандидата из раздела 10. Поэтому сравнение ниже характеризует
поведение моделей при фиксированном наборе параметров, но не устанавливает их окончательное
ранжирование.


Ниже сравниваются три геометрических представления одной области.

### 11.1. Обычная двухслойная модель

Кожа и граница лёгкого считаются двумя параллельными плоскостями. В этом сравнении в
аналитическую формулу условно подставляется постоянное геометрическое расстояние:

$$
h=h_c^{surface}=\min_{\mathbf q\in\partial\Omega_{lung}}\|\mathbf c-\mathbf q\|.
$$

Это точное ближайшее евклидово расстояние от найденного общего центра на коже до треугольной
поверхности STL лёгкого. Значение 15 мм остаётся только первоначальным мягким априорным ограничением и в данном
сравнении обычной модели не используется. Отрезок кожа–лёгкое может пересекать кость;
подстановка $h=h_c^{surface}$ является допущением сравнения, а не измерением толщины мягких
тканей или идентификацией электрически эффективного $h_{eff}$.

Введём ядро метода изображений

$$
\mathcal F(d,h)=\frac1d+2\sum_{i=1}^{N}
\frac{k^i}{\sqrt{d^2+(2ih)^2}},
\qquad
k=\frac{\rho_2-\rho_1}{\rho_2+\rho_1}.
$$

Для симметричной сборки $A=-L/2$, $M=-L/4$, $N=L/4$, $B=L/2$:

$$
Z_{2layer}(L,h)=\frac{\rho_1}{2\pi}
\left[2\mathcal F(L/4,h)-2\mathcal F(3L/4,h)\right].
$$

### 11.2. Реальная CT/FEM-модель

Кожа и лёгкое остаются криволинейными трёхмерными поверхностями. Ток свободно растекается во
всех направлениях через мягкие ткани, лёгкие, сердце и кости. Закрытой формулы для $Z$ нет —
решается уравнение поля на тетраэдральной сетке.

### 11.3. «Адаптированная» плоская модель

Сначала криволинейная кожа математически выпрямляется в линию $d=0$. После этого положение
границы лёгкого относительно кожи становится функцией

$$
h_{flat}(s)=d_{lung}(s)-d_{skin}(s).
$$

Для пары электродов в координатах $s_p,s_q$ вводится средняя глубина вдоль пути

$$
\overline h_{pq}=\frac{1}{|s_q-s_p|}
\int_{\min(s_p,s_q)}^{\max(s_p,s_q)}h_{approx}(s)\,ds.
$$

**Что означает эта средняя глубина и зачем она нужна.** Формула зеркальных источников
может принять только одно число $h$ для конкретной пары «токовый электрод — измерительный
электрод». Но после выпрямления кожи граница лёгкого имеет разную глубину в разных точках
между электродами. Поэтому профиль $h_{approx}(s)$ сжимается в одно число — его арифметическое
среднее на отрезке между данной парой электродов. Для четырёхэлектродного измерения получаются
четыре, вообще говоря разные, величины $\overline h_{AM}$, $\overline h_{BM}$,
$\overline h_{AN}$ и $\overline h_{BN}$.

Это не утверждение, что ток идёт по прямому отрезку между электродами, и не точное
усреднение трёхмерной плотности тока. Это явно введённое приближение, позволяющее учесть
изменение глубины лёгкого в старой аналитической формуле, которая изначально построена для
плоской границы с одним постоянным $h$.

Тогда локально-плоская адаптированная формула имеет вид

$$
\boxed{
Z_{adapt}=\frac{\rho_1}{2\pi}\left[
\mathcal F(|M-A|,\overline h_{AM})
-\mathcal F(|M-B|,\overline h_{BM})
-\mathcal F(|N-A|,\overline h_{AN})
+\mathcal F(|N-B|,\overline h_{BN})
\right].}
$$

Если $h_{flat}(s)$ постоянно, все $\overline h_{pq}=h$, и формула превращается в обычную
двухслойную. При кривой границе это **явно определённое приближение**, а не точное решение:
метод зеркальных источников строго справедлив только для параллельных плоскостей.
""")

code(r"""
# Two-column visual comparison requested: geometry is on the left, the depth
# representation actually used by that model is on the right.
fig, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
x_visual = np.linspace(-80, 80, 401)
h_constant = float(h_centre_surface_mm)
display(pd.DataFrame({
    'определение h в найденном центре': [
        'до ближайшей вершины STL (как в априорном штрафе обратной задачи)',
        'до ближайшей точки треугольной поверхности (обычная модель)'
    ],
    'h, мм': [h_centre_vertex_mm, h_centre_surface_mm],
    'роль': ['только априорный штраф обратной задачи', 'константа h в сравнении моделей']
}))

# Row 1: ordinary two-layer model.
ax = axes[0, 0]
ax.plot(x_visual, np.zeros_like(x_visual), color='saddlebrown', lw=3, label='плоская кожа')
ax.plot(x_visual, np.full_like(x_visual, h_constant), color='deepskyblue', lw=3, label='плоское лёгкое')
ax.fill_between(x_visual, 0, h_constant, color='peachpuff', alpha=0.65, label='ρ₁')
ax.fill_between(x_visual, h_constant, 65, color='lightskyblue', alpha=0.35, label='ρ₂')
ax.scatter([-70, -35, 35, 70], [0, 0, 0, 0], c=['red', 'orange', 'lime', 'blue'], s=55, zorder=5)
ax.set(title='Обычная: две параллельные плоскости', xlabel='s, мм', ylabel='d, мм',
       xlim=(-82, 82), ylim=(60, -8))
ax.legend(ncol=2, fontsize=8)

ax = axes[0, 1]
ax.plot(x_visual, np.full_like(x_visual, h_constant), 'k-', lw=2.5)
ax.set(title=f'Обычная модель использует одно h=h_c={h_constant:.2f} мм', xlabel='s, мм', ylabel='h, мм',
       xlim=(-82, 82), ylim=(0, 60))

# Row 2: real CT geometry in the selected plane.
ax = axes[1, 0]
if oblique_hu is not None:
    ax.imshow(oblique_hu,
              extent=[s_grid.min(), s_grid.max(), d_grid.min(), d_grid.max()],
              origin='lower', cmap='gray', vmin=-1000, vmax=450, aspect='equal')
else:
    ax.set_facecolor('#1d2730')
ax.add_collection(LineCollection(section_segments['body'], colors='yellow', linewidths=1.4))
ax.add_collection(LineCollection(section_segments['lungs'], colors='#00e5ff', linewidths=1.5))
ax.set(title=('Реальная: DICOM + CT/STL-контуры' if oblique_hu is not None else
              'Реальная: CT/STL-контуры без DICOM'),
       xlabel='s, мм', ylabel='d, мм', xlim=(-82, 82), ylim=(75, -15))

ax = axes[1, 1]
ax.plot(s_profile, skin_profile, color='saddlebrown', lw=2.5, label='$d_{skin}(s)$')
ax.plot(s_profile, lung_profile, color='deepskyblue', lw=2.5, label='$d_{lung}(s)$')
ax.fill_between(s_profile, skin_profile, lung_profile, where=np.isfinite(h_plane),
                color='peachpuff', alpha=0.65)
for branch_index in range(1, lung_branch_profiles.shape[1]):
    branch = lung_branch_profiles[:, branch_index]
    if np.any(np.isfinite(branch)):
        ax.plot(s_profile, branch, color='#007c91', lw=1.4, ls='--', alpha=0.8,
                label=('\u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 '
                       '\u043f\u0435\u0440\u0435\u0441\u0435\u0447\u0435\u043d\u0438\u044f \u043b\u0451\u0433\u043a\u043e\u0433\u043e '
                       '(\u0434\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u043a\u0430)'
                       if branch_index == 1 else None))
ax.set(title='Реальная кривизна кожи и лёгкого', xlabel='s, мм', ylabel='d, мм',
       xlim=(-82, 82), ylim=(65, -15))
ax.legend()

# Row 3: skin flattened; the relative lung curvature is preserved.
h_flat_visual = np.polynomial.polynomial.polyval(x_visual / 70.0, coefficients)
lung_flat_branch_profiles = lung_branch_profiles - skin_profile[:, None]
ax = axes[2, 0]
ax.plot(x_visual, np.zeros_like(x_visual), color='saddlebrown', lw=3, label='выпрямленная кожа')
ax.plot(x_visual, h_flat_visual, color='deepskyblue', lw=3, label='$h_{approx}(s)$')
ax.fill_between(x_visual, 0, h_flat_visual, color='peachpuff', alpha=0.65, label='ρ₁')
ax.fill_between(x_visual, h_flat_visual, 65, color='lightskyblue', alpha=0.35, label='ρ₂')
ax.scatter([-70, -35, 35, 70], [0, 0, 0, 0], c=['red', 'orange', 'lime', 'blue'], s=55, zorder=5)
for branch_index in range(1, lung_flat_branch_profiles.shape[1]):
    branch = lung_flat_branch_profiles[:, branch_index]
    if np.any(np.isfinite(branch)):
        ax.plot(s_profile, branch, color='#007c91', lw=1.8, ls='--', alpha=0.9,
                label=('дополнительная граница лёгкого '
                       '(в формуле не используется)'
                       if branch_index == 1 else None))
ax.text(0.02, 0.97,
        'Пунктир — дополнительные CT/STL-пересечения; '
        '$Z_{adapt}$ использует только ближайшую границу',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.78, edgecolor='none'))
ax.set(title='Адаптированная: кожа выпрямлена, лёгкое остаётся кривым',
       xlabel='s, мм', ylabel='d, мм', xlim=(-82, 82), ylim=(60, -8))
ax.legend(ncol=2, fontsize=8)
display(Markdown(
    'Замкнутая поверхность лёгкого может пересекать выбранную плоскость несколько раз. '
    'После выпрямления кожи все найденные CT/STL-границы сохранены на рисунке: ближайшая '
    'показана сплошной линией, дополнительные, включая дальний контур нижней части лёгкого, '
    '— пунктиром. В расчёт $Z_{adapt}$ входит только аппроксимация ближайшей границы. '
    'Следовательно, схема не утверждает, что упрощённая формула учитывает всю форму лёгкого; '
    'полный замкнутый объём используется только в CT/FEM-модели.'))

ax = axes[2, 1]
ax.plot(s_profile, h_plane, 'o', ms=3, color='tab:green', label='$h_{flat}$ из среза')
ax.plot(x_visual, h_flat_visual, 'k-', lw=2.5, label='$h_{approx}(s)$')
ax.set(title='Глубина, используемая адаптированной моделью', xlabel='s, мм', ylabel='h, мм',
       xlim=(-82, 82), ylim=(0, 60))
ax.legend()
""")

code(r"""
def image_kernel(d_mm, h_mm, rho1, rho2, terms=100):
    d_m = abs(float(d_mm)) / 1000.0
    h_m = float(h_mm) / 1000.0
    reflection = (rho2 - rho1) / (rho2 + rho1)
    order = np.arange(1, terms + 1, dtype=float)
    return (1.0 / d_m
            + 2.0 * np.sum(reflection ** order /
                           np.sqrt(d_m ** 2 + (2.0 * order * h_m) ** 2)))

def z_two_layer(L_mm, rho1, rho2, h_mm):
    d_near = L_mm / 4.0
    d_far = 3.0 * L_mm / 4.0
    return rho1 / (2.0 * np.pi) * (
        2.0 * image_kernel(d_near, h_mm, rho1, rho2)
        - 2.0 * image_kernel(d_far, h_mm, rho1, rho2)
    )

def h_approx_function(s_mm):
    return np.polynomial.polynomial.polyval(np.asarray(s_mm) / 70.0, coefficients)

def path_mean_h(s1_mm, s2_mm):
    sample_s = np.linspace(min(s1_mm, s2_mm), max(s1_mm, s2_mm), 401)
    return np.trapezoid(h_approx_function(sample_s), sample_s) / abs(s2_mm - s1_mm)

def z_adapted(L_mm, rho1, rho2):
    A, M, N, B = -L_mm / 2.0, -L_mm / 4.0, L_mm / 4.0, L_mm / 2.0
    F_AM = image_kernel(M - A, path_mean_h(A, M), rho1, rho2)
    F_BM = image_kernel(M - B, path_mean_h(B, M), rho1, rho2)
    F_AN = image_kernel(N - A, path_mean_h(A, N), rho1, rho2)
    F_BN = image_kernel(N - B, path_mean_h(B, N), rho1, rho2)
    return rho1 / (2.0 * np.pi) * (F_AM - F_BM - F_AN + F_BN)

L_values = comparison.L_mm.to_numpy(dtype=float)
Z_experiment = comparison.Z_experiment_ohm.to_numpy(dtype=float)
Z_fem = comparison.Z_FEM_ohm.to_numpy(dtype=float)
rho1_fem = float(summary.rho_soft_ohm_m)
rho2_fem = float(summary.rho_lungs_ohm_m)

# Direct comparison: all models receive the same rho1 and rho2. This isolates
# the effect of geometry/formula instead of hiding it by re-fitting materials.
Z_two_direct = np.array([z_two_layer(L, rho1_fem, rho2_fem, h_constant) for L in L_values])
Z_adapt_direct = np.array([z_adapted(L, rho1_fem, rho2_fem) for L in L_values])

pair_depth_rows = []
for L in L_values:
    A, M, N, B = -L / 2.0, -L / 4.0, L / 4.0, L / 2.0
    pair_depth_rows.append({
        'L, мм': L,
        'h̄_AM, мм': path_mean_h(A, M),
        'h̄_BM, мм': path_mean_h(B, M),
        'h̄_AN, мм': path_mean_h(A, N),
        'h̄_BN, мм': path_mean_h(B, N),
    })
display(Markdown('### Средние глубины, фактически подставленные в адаптированную формулу'))
display(pd.DataFrame(pair_depth_rows))

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(L_values, Z_experiment, 'ko-', lw=2.5, label='измерение')
ax.plot(L_values, Z_two_direct, 'o--', lw=2,
        label=f'обычная двухслойная, h=h_c={h_constant:.2f} мм')
ax.plot(L_values, Z_adapt_direct, 'd-.', lw=2, label='адаптированная, $h_{approx}(s)$')
ax.plot(L_values, Z_fem, 's-', lw=2.5, label='реальная CT/FEM')
ax.set(xlabel='L, мм', ylabel='Z, Ом',
       title='Одинаковые $\\rho_1,\\rho_2$: влияние геометрии и формулы')
ax.legend()
fig.tight_layout()

def error_row(name, values):
    residual = np.asarray(values) - Z_experiment
    return {
        'модель': name,
        'RMS, Ом': np.sqrt(np.mean(residual ** 2)),
        'MAE, Ом': np.mean(np.abs(residual)),
        'bias, Ом': np.mean(residual),
        'max |ошибка|, Ом': np.max(np.abs(residual)),
        'наклон Z(L), Ом/мм': np.polyfit(L_values, values, 1)[0],
    }

direct_error_table = pd.DataFrame([
    error_row('обычная двухслойная, h=h_c', Z_two_direct),
    error_row('адаптированная h(s)', Z_adapt_direct),
    error_row('CT/FEM', Z_fem),
])
display(Markdown('### Ошибка без повторной подгонки $\\rho_1,\\rho_2$'))
display(direct_error_table)
""")

md(r"""
### 11.4. Что здесь называется калибровкой модели

Калибровка — это отдельная подгонка свободных параметров **каждой** упрощённой модели по тем
же девяти экспериментальным точкам $Z_{exp}(L_i)$. Для неё решается обычная задача наименьших
квадратов

$$
\min_{\theta}\sum_{i=1}^{9}\left[Z_{model}(L_i;\theta)-Z_{exp}(L_i)\right]^2.
$$

- для обычной модели с фиксированным $h=h_c$ подгоняются $\theta=(\rho_1,\rho_2)$;
- для обычной модели со свободной глубиной подгоняются $\theta=(\rho_1,\rho_2,h)$;
- для адаптированной модели профиль $h_{approx}(s)$ остаётся взятым из CT, а подгоняются
  $\theta=(\rho_1,\rho_2)$;
- CT/FEM в этой последней таблице заново не калибруется: показан итоговый локальный кандидат
  выбранного тега из раздела 10.

Использованы одинаковые ограничения $2\leq\rho_1\leq10$ Ом·м и
$8\leq\rho_2\leq40$ Ом·м; для варианта со свободной глубиной
$3\leq h\leq60$ мм. Эти границы шире основных диапазонов обратной задачи и нужны, чтобы
проверить, способна ли упрощённая модель компенсировать ошибку геометрии искажёнными
параметрами. Попадание результата на границу диапазона — предупреждение о такой компенсации,
а не подтверждение физической корректности найденных $\rho$ или $h$.
""")

code(r"""
# A second, fair model-class comparison: let each reduced model re-fit the
# material parameters. This shows how much empirical calibration can conceal
# an incorrect geometric model.
plain_fixed_fit = least_squares(
    lambda x: np.array([z_two_layer(L, x[0], x[1], h_constant) for L in L_values]) - Z_experiment,
    x0=[rho1_fem, rho2_fem], bounds=([2.0, 8.0], [10.0, 40.0])
)
plain_free_fit = least_squares(
    lambda x: np.array([z_two_layer(L, x[0], x[1], x[2]) for L in L_values]) - Z_experiment,
    x0=[rho1_fem, rho2_fem, h_constant], bounds=([2.0, 8.0, 3.0], [10.0, 40.0, 60.0])
)
adapted_fit = least_squares(
    lambda x: np.array([z_adapted(L, x[0], x[1]) for L in L_values]) - Z_experiment,
    x0=[rho1_fem, rho2_fem], bounds=([2.0, 8.0], [10.0, 40.0])
)

Z_plain_fixed_fit = np.array([z_two_layer(L, *plain_fixed_fit.x, h_constant) for L in L_values])
Z_plain_free_fit = np.array([z_two_layer(L, *plain_free_fit.x) for L in L_values])
Z_adapted_fit = np.array([z_adapted(L, *adapted_fit.x) for L in L_values])

calibrated_table = pd.DataFrame([
    {**error_row('обычная, h=h_c, подгонка ρ₁ρ₂', Z_plain_fixed_fit),
     'ρ₁, Ом·м': plain_fixed_fit.x[0], 'ρ₂, Ом·м': plain_fixed_fit.x[1], 'h, мм': h_constant},
    {**error_row('обычная, подгонка ρ₁ρ₂h', Z_plain_free_fit),
     'ρ₁, Ом·м': plain_free_fit.x[0], 'ρ₂, Ом·м': plain_free_fit.x[1], 'h, мм': plain_free_fit.x[2]},
    {**error_row('адаптированная, подгонка ρ₁ρ₂', Z_adapted_fit),
     'ρ₁, Ом·м': adapted_fit.x[0], 'ρ₂, Ом·м': adapted_fit.x[1], 'h, мм': np.nan},
    {**error_row('CT/FEM', Z_fem),
     'ρ₁, Ом·м': rho1_fem, 'ρ₂, Ом·м': rho2_fem, 'h, мм': h_centre_surface_mm},
])
display(Markdown('### Минимальная ошибка после отдельной калибровки каждой модели'))
display(calibrated_table)

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(L_values, Z_experiment, 'ko-', lw=2.5, label='измерение')
ax.plot(L_values, Z_plain_fixed_fit, 'o--', label='обычная h=h_c, после подгонки')
ax.plot(L_values, Z_plain_free_fit, '^--', label='обычная h свободно, после подгонки')
ax.plot(L_values, Z_adapted_fit, 'd-.', label='адаптированная, после подгонки')
ax.plot(L_values, Z_fem, 's-', lw=2.2, label='CT/FEM')
ax.set(xlabel='L, мм', ylabel='Z, Ом',
       title='Что можно скрыть повторной калибровкой параметров')
ax.legend()
fig.tight_layout()

display(Markdown(f'''### Технический вывод сравнения

Сравнение не доказывает превосходство одной модели: упрощённые модели подогнаны отдельно,
а CT/FEM-кривая относится к выбранному локальному кандидату. Числа ниже служат
диагностикой возможной компенсации параметров, а не глобальным ранжированием моделей.

- При одинаковых $\\rho_1,\\rho_2$ обычная модель с $h=h_c={h_constant:.2f}$ мм имеет
  RMS **{direct_error_table.iloc[0]['RMS, Ом']:.2f} Ом**, адаптированная локально-плоская —
  **{direct_error_table.iloc[1]['RMS, Ом']:.2f} Ом**, CT/FEM —
  **{direct_error_table.iloc[2]['RMS, Ом']:.2f} Ом**.
- После подгонки обычная модель с фиксированным $h=h_c$ достигает RMS
  **{calibrated_table.iloc[0]['RMS, Ом']:.2f} Ом**, но использует смещённые эффективные
  сопротивления.
- Если разрешить обычной модели свободно менять $h$, оптимум уходит к
  $h={plain_free_fit.x[2]:.2f}$ мм и $\\rho_1={plain_free_fit.x[0]:.2f}$ Ом·м. Это не похоже на
  CT-геометрию и показывает взаимную компенсацию $h$, $\\rho_1$ и $\\rho_2$.
- Простая адаптация через среднюю $\\overline h_{{pq}}$ не обязана улучшать результат: она
  учитывает профиль глубины, но всё ещё игнорирует трёхмерное боковое растекание тока,
  локальные нормали, сердце и кости. Если нужна точность полной кривой границы, это уже задача
  FEM, а не метода изображений.
'''))
""")

md(r"""
## 12. Почему от 50 до 60 мм импеданс сначала растёт — ошибка это или нет?

Для однородного полупространства и обычной плоской двухслойной модели с неизменными
$\rho_1,\rho_2,h$ ожидается в основном монотонное уменьшение $Z$ при увеличении $L$.
Поэтому максимум около 60 мм требует проверки, но сам по себе ещё не доказывает ошибку.

Здесь одновременно действуют причины, отсутствующие в плоской формуле:

1. при изменении $L$ все четыре электрода перемещаются по криволинейной коже над разными
   участками лёгкого, рёбер и сердца;
2. глубина лёгкого и нормаль кожи под каждой парой меняются не монотонно;
3. сборки ставились отдельно вручную, поэтому возможны ошибки расстояния, оси, прижима и
   контактного сопротивления;
4. FEM-электрод состоит из конечного набора граничных треугольников. При изменении $L$ состав
   площадки меняется дискретно, что может создавать сеточную немонотонность.

В JSON имеется по одному агрегированному уровню вдоха и выдоха для каждой установки. Повторных
независимых установок и исходного временного ряда импеданса в этой папке нет. Поэтому можно
проверить воспроизводимость пика между состояниями и моделью, но нельзя окончательно отделить
физический эффект от систематической ошибки конкретной установки.
""")

code(r"""
timestamp_dir = ROOT.parent / 'Colab Notebooks' / 'timestamps'
hold_rows = []
for L in L_values.astype(int):
    record = json.loads((timestamp_dir / f'{L}nik.json').read_text(encoding='utf-8'))
    hold_rows.append({
        'L, мм': L,
        'вдох, Ом': float(record['hold_levels']['вдох']),
        'выдох, Ом': float(record['hold_levels']['выдох']),
    })
hold_table = pd.DataFrame(hold_rows)
hold_table['вдох−выдох, Ом'] = hold_table['вдох, Ом'] - hold_table['выдох, Ом']
hold_table['FEM, Ом'] = Z_fem
hold_table['обычная 2-layer, Ом'] = Z_two_direct

geometry_diagnostic = grids.copy()
geometry_diagnostic['ошибка I-span, %'] = (
    100 * (geometry_diagnostic.I_to_I_actual_mm - geometry_diagnostic.L_nominal_mm)
    / geometry_diagnostic.L_nominal_mm
)
geometry_diagnostic['ошибка V-span, %'] = (
    100 * (geometry_diagnostic.V_to_V_actual_mm - geometry_diagnostic.L_nominal_mm / 2)
    / (geometry_diagnostic.L_nominal_mm / 2)
)
area_summary = electrodes.groupby('L_mm').patch_area_mm2.agg(['min', 'max', 'mean']).reset_index()
area_summary.columns = ['L, мм', 'area min, мм²', 'area max, мм²', 'area mean, мм²']

delta_table = pd.DataFrame({
    'переход L, мм': [f'{int(a)}→{int(b)}' for a, b in zip(L_values[:-1], L_values[1:])],
    'ΔZ вдох, Ом': np.diff(hold_table['вдох, Ом']),
    'ΔZ выдох, Ом': np.diff(hold_table['выдох, Ом']),
    'ΔZ FEM, Ом': np.diff(Z_fem),
    'ΔZ обычная, Ом': np.diff(Z_two_direct),
})

display(Markdown('### Уровни в обоих дыхательных состояниях'))
display(hold_table)
display(Markdown('### Приращения между соседними размерами'))
display(delta_table)
display(Markdown('### Фактическая дискретная геометрия FEM-электродов'))
display(geometry_diagnostic[[
    'L_nominal_mm', 'I_to_I_actual_mm', 'V_to_V_actual_mm',
    'ошибка I-span, %', 'ошибка V-span, %'
]])
display(area_summary)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
ax = axes[0, 0]
ax.plot(L_values, hold_table['вдох, Ом'], 'ko-', lw=2.4, label='эксперимент: вдох')
ax.plot(L_values, hold_table['выдох, Ом'], 'o--', color='tab:gray', label='эксперимент: выдох')
ax.plot(L_values, Z_fem, 's-', color='tab:blue', lw=2.2, label='CT/FEM')
ax.plot(L_values, Z_two_direct, 'd-.', color='tab:orange', label='обычная 2-layer')
ax.axvspan(50, 70, color='gold', alpha=0.16)
ax.set(title='Пик 50–70 мм в эксперименте и моделях', xlabel='L, мм', ylabel='Z, Ом')
ax.legend()

ax = axes[0, 1]
x_delta = np.arange(len(delta_table))
ax.axhline(0, color='black', lw=1)
ax.plot(x_delta, delta_table['ΔZ вдох, Ом'], 'ko-', label='вдох')
ax.plot(x_delta, delta_table['ΔZ выдох, Ом'], 'o--', color='tab:gray', label='выдох')
ax.plot(x_delta, delta_table['ΔZ FEM, Ом'], 's-', color='tab:blue', label='CT/FEM')
ax.plot(x_delta, delta_table['ΔZ обычная, Ом'], 'd-.', color='tab:orange', label='2-layer')
ax.set_xticks(x_delta, delta_table['переход L, мм'], rotation=35, ha='right')
ax.set(title='Знак изменения между соседними сборками', ylabel='ΔZ, Ом')
ax.legend()

ax = axes[1, 0]
ax.plot(grids.L_nominal_mm, grids.I_to_I_actual_mm, 'o-', label='I–I фактически')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm, 'k--', label='I–I номинально')
ax.plot(grids.L_nominal_mm, grids.V_to_V_actual_mm, 's-', label='V–V фактически')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm / 2, color='gray', ls='--', label='V–V номинально')
ax.set(title='Номинальные и FEM-расстояния электродов', xlabel='L, мм', ylabel='расстояние, мм')
ax.legend()

ax = axes[1, 1]
for label, group in electrodes.groupby('electrode'):
    ax.plot(group.L_mm, group.patch_area_mm2, 'o-', label=label)
ax.axhline(np.pi * 10**2 / 4 * 0.9, color='black', ls='--', label='целевая площадь')
ax.set(title='Дискретная площадь электродных площадок', xlabel='L, мм', ylabel='площадь, мм²')
ax.legend(ncol=2, fontsize=8)

peak_inhale = hold_table.loc[hold_table['L, мм'] == 60, 'вдох, Ом'].iloc[0]
neighbor_mean = hold_table.loc[hold_table['L, мм'].isin([50, 70]), 'вдох, Ом'].mean()
display(Markdown(f'''### Вывод по максимуму при 60 мм

- На вдохе 60-мм сборка выше среднего соседей 50 и 70 мм на
  **{peak_inhale - neighbor_mean:.2f} Ом**.
- На выдохе максимум сохраняется, следовательно, он не создаётся только выбором интервала вдоха.
- Локальный максимум FEM при 60 мм для `{result_tag}`: **{"да" if Z_fem[1] > max(Z_fem[0], Z_fem[2]) else "нет"}**.
  $Z_{{50}}={Z_fem[0]:.2f}$, $Z_{{60}}={Z_fem[1]:.2f}$,
  $Z_{{70}}={Z_fem[2]:.2f}$ Ом. Для обычной плоской модели такой максимум:
  **{"да" if Z_two_direct[1] > max(Z_two_direct[0], Z_two_direct[2]) else "нет"}**.
- Поэтому считать точку 60 мм автоматически ошибочной нельзя. Но FEM использует дискретно
  меняющиеся площадки, а эксперимент не содержит повторных переустановок. Наиболее строгая
  следующая проверка — 3–5 повторных установок сборок 50, 60 и 70 мм плюс расчёт на ещё более
  сгущённой поверхностной сетке с контролем одинаковой площади площадок.
'''))
""")

md(r"""
## 13. Исторический диагностический PEM/CEM-скрининг геометрии
**Статус раздела.** Этот скрининг выполнен до итогового прямого поиска. В нём зафиксированы
$\rho_1,\rho_2$ из отклонённой точки `fmincon`. Поэтому он показывает только поведение
геометрического критерия при тех старых сопротивлениях и не участвует в выборе итоговой
координаты `disjoint`.


Отклонённая совместная подгонка пяти параметров служит исходной точкой скрининга, но не доказывает, что
найден глобальный минимум. Для дополнительной проверки координаты центра задаются на регулярной
сетке

$$
u,v=-120,-119,\ldots,119,120\ \text{мм},
$$

а для каждого различимого центра FEM-поверхности явно проверяется

$$
\varphi=-20,-19,\ldots,19,20^\circ.
$$

На этом этапе $\rho_1$ и $\rho_2$ зафиксированы в значениях отклонённой подгонки `fmincon`. Следовательно,
это плотный поиск **только геометрии**, а не повторная глобальная оптимизация всех пяти
неизвестных. После проекции, объединения одинаковых FEM-узлов и фильтра по толщине число
фактически проверенных центров и поз приведено в таблице ниже. Шаг входных координат 1 мм не
означает миллиметрового физического разрешения: оно ограничено поверхностной FEM-сеткой и
дискретным выбором граней площадок.

### 13.1. Быстрый PEM-скрининг

Для фиксированных $\rho_1,\rho_2$ один раз собирается и факторизуется объёмная матрица
$K(\rho)$. Передаточный импеданс для точечных электродов получается из матрицы Грина:

$$
Z_{ABMN}=(\mathbf e_M-\mathbf e_N)^T
K^{-1}(\mathbf e_A-\mathbf e_B).
$$

Все правые части решаются пакетно, а одинаковые граничные узлы объединяются. После построения
кэша полный перебор занимает секунды. Но PEM игнорирует конечную площадь площадки, поэтому
используется только как предварительный фильтр.

### 13.2. Точная низкоранговая CEM-поправка

Для лучших PEM-областей возвращаются реальные площадки и контактный импеданс. Объёмная часть
матрицы остаётся той же, а каждая площадка добавляет локальные блоки $D,B,C$:

$$
E=\begin{bmatrix}K+D&B\\B^T&C\end{bmatrix}.
$$

Через заранее рассчитанную подматрицу $G=K^{-1}_{PP}$ для узлов площадок решаются только малые
системы

$$
Q=(I+GD)^{-1}GB,
\qquad
(C-B^TQ)U=I_e.
$$

Это не новая физическая модель: проверка в контрольной исторической позе совпала с полным EIDORS
`fwd_solve` с RMS порядка $10^{-6}$ Ом.
""")

optional_historical_code(r"""
performance = pd.DataFrame([
    {
        'этап': 'глобальный PEM, 1 мм / 1°',
        'исходных центров': int(fast_pem_summary.raw_centre_count),
        'различимых центров': int(fast_pem_summary.unique_anatomical_centre_count),
        'позиций центр–φ': int(fast_pem_summary.pose_count),
        'время основного скана, с': fast_pem_summary.scan_seconds,
        'валидация RMS, Ом': fast_pem_summary.validation_RMSE_ohm,
    },
    {
        'этап': 'низкоранговая CEM для лучших PEM-кандидатов',
        'исходных центров': int(fast_cem_summary.requested_pose_count),
        'различимых центров': int(fast_cem.loc[
            ~fast_cem.is_validation.astype(bool),
            ['centre_u_mm', 'centre_v_mm']].drop_duplicates().shape[0]),
        'позиций центр–φ': int(fast_cem_summary.evaluated_pose_count),
        'время основного скана, с': (fast_cem_summary.geometry_seconds
                                     + fast_cem_summary.green_solve_seconds
                                     + fast_cem_summary.local_CEM_seconds),
        'валидация RMS, Ом': fast_cem_summary.validation_curve_RMSE_ohm,
    },
])
display(performance)

stl_rendering = pd.DataFrame({
    'STL': ['тело', 'лёгкие', 'сердце', 'кости'],
    'отрисовано треугольников': [len(triangles[k]) for k in ['body', 'lungs', 'heart', 'bones']],
    'режим': ['все грани, indexed Mesh3d'] * 4,
})
display(Markdown('### Точность 3D-отрисовки STL'))
display(stl_rendering)

# Global PEM map: for every distinct centre retain the explicitly best phi.
global_best_index = fast_pem_all.groupby(['centre_u_mm', 'centre_v_mm']).fast_objective.idxmin()
global_best = fast_pem_all.loc[global_best_index].copy()

# The current CEM stage evaluates the best PEM poses, not a regular local grid.
cem_candidates = fast_cem[~fast_cem.is_validation.astype(bool)].copy()
cem_candidates = cem_candidates.sort_values('fast_CEM_objective').reset_index(drop=True)
best_exact = cem_candidates.iloc[0]
cem_z_columns = [f'Z_fast_CEM_L{int(L):03d}_ohm' for L in comparison.L_mm]
best_exact_z = best_exact[cem_z_columns].to_numpy(dtype=float)
best_exact_residual = best_exact_z - comparison.Z_experiment_ohm.to_numpy(dtype=float)
best_exact_rms = float(np.sqrt(np.mean(best_exact_residual ** 2)))
best_exact_mae = float(np.mean(np.abs(best_exact_residual)))

fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
ax = axes[0, 0]
sc = ax.scatter(global_best.centre_u_mm, global_best.centre_v_mm,
                c=global_best.fast_objective, s=22, cmap='viridis_r')
ax.scatter(rejected_summary.centre_u_mm, rejected_summary.centre_v_mm, marker='*', s=180,
           color='red', edgecolor='black', label='отклонённая точка fmincon')
ax.scatter(best_exact.centre_u_mm, best_exact.centre_v_mm, marker='X', s=150,
           color='lime', edgecolor='black', label='лучший CEM-кандидат')
ax.set(title='Глобальный PEM: минимум по всем φ для каждого центра',
       xlabel='u, мм', ylabel='v, мм', aspect='equal')
ax.legend(); fig.colorbar(sc, ax=ax, label='быстрый критерий')

ax = axes[0, 1]
sc = ax.scatter(global_best.centre_u_mm, global_best.centre_v_mm,
                c=global_best.phi_deg, s=22, cmap='twilight',
                vmin=fast_pem_summary.phi_min_deg, vmax=fast_pem_summary.phi_max_deg)
ax.set(title='Какой φ выбран PEM в каждом центре', xlabel='u, мм', ylabel='v, мм', aspect='equal')
fig.colorbar(sc, ax=ax, label='φ, град')

ax = axes[1, 0]
sc = ax.scatter(cem_candidates.centre_u_mm, cem_candidates.centre_v_mm,
                c=cem_candidates.fast_CEM_objective, s=45, cmap='magma_r')
ax.scatter(best_exact.centre_u_mm, best_exact.centre_v_mm, marker='X', s=160,
           color='lime', edgecolor='black')
ax.set(title='Точно пересчитанные CEM-кандидаты', xlabel='u, мм', ylabel='v, мм', aspect='equal')
fig.colorbar(sc, ax=ax, label='CEM-критерий')

ax = axes[1, 1]
ax.scatter(cem_candidates.phi_deg, cem_candidates.fast_CEM_objective, s=35, alpha=0.8)
ax.axvline(fast_pem_summary.phi_min_deg, color='red', ls='--', label='граница поиска')
ax.axvline(fast_pem_summary.phi_max_deg, color='red', ls='--')
ax.set(title='CEM-критерий и угол', xlabel='φ, град', ylabel='CEM-критерий')
ax.legend()

display(Markdown('### Лучшие точные CEM-положения'))
display(fast_cem.head(12)[[
    'centre_u_mm', 'centre_v_mm', 'phi_deg', 'h_centre_mm',
    'fast_CEM_objective', 'slope_ohm_per_mm', 'is_validation'
]])
""")

md(r"""
### 13.3. Почему результат при шаге 1 мм нельзя считать субмиллиметровым

Текущий конечный электрод задаётся целым набором треугольных граней. При малом изменении центра
или угла набор граней некоторое время не меняется, а затем меняется скачком. Поэтому критерий
является кусочно-постоянным/скачкообразным по геометрии. Формальный минимум на сетке 1 мм может
оказаться минимумом конкретного дискретного набора граней, а не непрерывной физической задачи.

Особенно важно, попал ли лучший кандидат внутрь исследованного диапазона. Минимум на границе
не подтверждает локальный минимум: критерий может продолжать уменьшаться за пределами диапазона.
Ниже показаны ранжирование CEM-кандидатов и кривая импеданса лучшего из них.
""")

optional_historical_code(r"""
phi_on_boundary = bool(
    np.isclose(best_exact.phi_deg, fast_pem_summary.phi_min_deg)
    or np.isclose(best_exact.phi_deg, fast_pem_summary.phi_max_deg)
)
phi_boundary_text = 'да' if phi_on_boundary else 'нет'

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
axes[0].plot(np.arange(1, len(cem_candidates) + 1),
             cem_candidates.fast_CEM_objective, 'o-', ms=4)
axes[0].set(title='Ранжирование пересчитанных CEM-кандидатов',
            xlabel='ранг кандидата', ylabel='CEM-критерий')

axes[1].plot(comparison.L_mm, comparison.Z_experiment_ohm, 'ko-', lw=2,
             label='эксперимент')
axes[1].plot(comparison.L_mm, best_exact_z, 'o-', lw=2,
             label='лучший CEM-кандидат')
axes[1].set(title='Как читать итоговую кривую', xlabel='L, мм', ylabel='Z, Ом')
axes[1].legend()

display(Markdown(fr'''### Вывод плотного поиска

- PEM проверил **{int(fast_pem_summary.pose_count):,}** комбинаций для
  **{int(fast_pem_summary.unique_anatomical_centre_count):,}** различимых центров и
  {int(fast_pem_summary.phi_count)} углов.
- PEM работает быстро, но его RMS относительно CEM равен
  **{fast_pem_summary.validation_RMSE_ohm:.2f} Ом** при корреляции
  **{fast_pem_summary.validation_correlation:.3f}**; одного PEM недостаточно для окончательного выбора.
- Из {int(fast_cem_summary.requested_pose_count)} запрошенных CEM-поз
  **{int(fast_cem_summary.rejected_geometry_count)}** отвергнуты, потому что на дискретной поверхности
  для них нельзя построить четыре допустимые непересекающиеся площадки. Оценено
  **{int(fast_cem_summary.evaluated_pose_count)}** поз.
- CEM-решатель с низкоранговым обновлением проверен против полного EIDORS в контрольной позе: RMS различия кривых
  **{fast_cem_summary.validation_curve_RMSE_ohm:.6g} Ом**. Это численная проверка решателя,
  а не физиологическая валидация найденного положения.
- Лучший из пересчитанных CEM-кандидатов:
  $u={best_exact.centre_u_mm:.3f}$ мм,
  $v={best_exact.centre_v_mm:.3f}$ мм,
  $\varphi={best_exact.phi_deg:.1f}^\circ$, $J={best_exact.fast_CEM_objective:.3f}$,
  RMS остатков **{best_exact_rms:.3f} Ом**, MAE **{best_exact_mae:.3f} Ом**.
- Лучший угол находится на границе поиска: **{phi_boundary_text}**. В текущем прогоне это
  $\varphi={best_exact.phi_deg:.1f}^\circ$ при диапазоне
  [{fast_pem_summary.phi_min_deg:.0f}; {fast_pem_summary.phi_max_deg:.0f}]°. Поэтому локальный
  минимум по углу не установлен.
- Отклонённая подгонка `fmincon` и исторический CEM-скрининг указывают на разные положения:
  $({rejected_summary.centre_u_mm:.1f}, {rejected_summary.centre_v_mm:.1f}, {rejected_summary.phi_deg:.1f}^\circ)$ против
  $({best_exact.centre_u_mm:.1f}, {best_exact.centre_v_mm:.1f}, {best_exact.phi_deg:.1f}^\circ)$.
  Поскольку в CEM-скрининге $\rho_1,\rho_2$ не переоптимизировались, это ещё не совместное
  решение пяти неизвестных и не альтернатива итоговому `disjoint`.
'''))
""")

md(r"""
## 14. Исторический пересчёт на сетке v5: 1-мм оболочка и раздельные CEM-площадки

Этот раздел описывает прежние прогоны; он не подтверждает статус нового `auditfix_20260908`.

Новая версия не является простым повтором старого расчёта с большим числом тетраэдров.
В ходе полного перепрогона были обнаружены и исправлены ошибки воспроизводимости:

1. Gmsh Frontal формально завершал построение, но тетраэдры заполняли только 17.31 л из 20.17 л.
   Принят вариант Delaunay, где объёмы STL и тетраэдров совпадают.
2. Для сборки 50 мм прежнее независимое выращивание площадок давало 4 общие грани и 8 общих
   узлов у `V_minus` и `I_minus`. Это частично замыкало два электрода и создавало ложный скачок
   $Z(50)$. Теперь грани сначала разделяются между ближайшими электродами, а повторное использование
   узлов запрещено.
3. Входной манифест исправлен: для Ника независима запись 90 мм, а 100 мм является её поздней
   копией и исключается.
4. Inverse, PEM и CEM теперь используют один и тот же артефакт сетки v5, один загрузчик данных
   и согласованно именованные результаты. Добавлены очистка кэша EIDORS, обработка опорного узла и
   контролируемое отбрасывание геометрически недопустимых CEM-поз.

Старый градиентный прогон и исторический CEM-кандидат сохранены как диагностика.
Последующий `disjoint_restartable_manifest90` является историческим снимком без подтверждения
нового QC. Основной кандидат определяется только выбранным тегом и статусом в начале отчёта.
""")

optional_historical_code(r"""
if baseline_available:
    mesh_comparison = pd.DataFrame({
        'версия': ['исторический v4: иной входной набор',
                   'текущий v5: 90 мм включён, 100 мм исключён'],
        'узлы': [int(baseline_summary.mesh_nodes), int(summary.mesh_nodes)],
        'тетраэдры': [int(baseline_summary.mesh_tetrahedra), int(summary.mesh_tetrahedra)],
        'объём FEM, л': [18.973231, 20.170763],
        'медианный шаг поверхности, мм': [2.556, 1.520],
        'RMSE Z, Ом': [baseline_summary.rms_residual_ohm, summary.rms_residual_ohm],
    })
    parameter_comparison = pd.DataFrame({
        'параметр': ['rho_1, Ом·м', 'rho_2, Ом·м', 'u, мм', 'v, мм',
                     'phi, град', 'h центра, мм', 'наклон, Ом/мм'],
        'исторический v4': [baseline_summary.rho_soft_ohm_m, baseline_summary.rho_lungs_ohm_m,
               baseline_summary.centre_u_mm, baseline_summary.centre_v_mm,
               baseline_summary.phi_deg, baseline_summary.h_centre_mm,
               baseline_summary.fem_slope_ohm_per_mm],
        'текущий v5': [summary.rho_soft_ohm_m, summary.rho_lungs_ohm_m,
               summary.centre_u_mm, summary.centre_v_mm,
               summary.phi_deg, summary.h_centre_mm,
               summary.fem_slope_ohm_per_mm],
    })
    display(Markdown(
        '**Важно:** исторический v4 использовал 100 мм вместо независимой записи 90 мм. '
        'Поэтому различие RMSE нельзя приписывать только изменению сетки.'
    ))
    display(mesh_comparison)
    display(parameter_comparison)
else:
    display(Markdown(
        '**Сравнение с исторической сеткой v4 пропущено:** два старых CSV отсутствуют. '
        'Это не мешает отчёту по текущему расчёту v5.'
    ))
    display(pd.DataFrame({
        'версия': ['v5: 1 мм / local 2 мм'],
        'узлы': [int(summary.mesh_nodes)],
        'тетраэдры': [int(summary.mesh_tetrahedra)],
        'RMSE Z, Ом': [summary.rms_residual_ohm],
    }))

if pre_rerun_available:
    display(Markdown('### Сопоставление с состоянием непосредственно до перепрогона'))
    display(pd.DataFrame({
        'расчёт': ['v5 до перепрогона', 'v5: итоговый прямой поиск disjoint',
                   'v5: исторический CEM-кандидат'],
        'входной ряд': ['100 мм включён, 90 мм исключён',
                       '90 мм включён, 100 мм исключён',
                       '90 мм включён, 100 мм исключён'],
        'RMSE, Ом': [pre_rerun_summary.rms_residual_ohm,
                     summary.rms_residual_ohm, best_exact_rms],
        'статус': ['архивный результат с неверной субъектной атрибуцией копии',
                   f'выбранный тег {result_tag}; обращения к критерию: {function_evaluations_text}',
                   ('рассчитан при rho из отклонённой точки fmincon; '
                    'лучший угол находится на границе поиска')],
    }))
    display(Markdown(
        'Архивный RMS нельзя сравнивать с новыми двумя строками как чистый эффект алгоритма: '
        'изменены входной размерный ряд и согласованность используемых артефактов.'
    ))

if pre_rerun_available and ablation_available:
    display(Markdown('### Разложение причин сдвига найденной точки'))

    if legacy_patch_available:
        earliest = legacy_patch_summary['earliest']
        reconstructed_fast = legacy_patch_summary['reconstructed_fast']
        earliest_objective = float(legacy_patch_history['earliest'].objective.min())
        reconstructed_fast_objective = float(
            legacy_patch_history['reconstructed_fast'].objective.min())

        archived_electrodes = pd.read_csv(pre_rerun_electrodes_file)
        reconstructed_electrodes = legacy_patch_electrodes['reconstructed_fast']
        geometry_check = archived_electrodes.merge(
            reconstructed_electrodes, on=['L_mm', 'electrode'],
            suffixes=('_archive', '_reconstructed'))
        geometry_check['centroid_error_mm'] = np.sqrt(sum(
            (geometry_check[f'patch_centroid_{axis}_mm_reconstructed']
             - geometry_check[f'patch_centroid_{axis}_mm_archive'])**2
            for axis in 'xyz'))
        geometry_check['area_error_mm2'] = (
            geometry_check.patch_area_mm2_reconstructed
            - geometry_check.patch_area_mm2_archive)
        exact_geometry_count = int((
            (geometry_check.centroid_error_mm.abs() < 1e-10)
            & (geometry_check.area_error_mm2.abs() < 1e-10)).sum())
        max_centroid_error = float(geometry_check.centroid_error_mm.abs().max())
        max_area_error = float(geometry_check.area_error_mm2.abs().max())

        reconstructed_curve = legacy_patch_comparison['reconstructed_fast']
        archived_curve = pre_rerun_comparison.set_index('L_mm').loc[
            reconstructed_curve.L_mm]
        exact_impedance_count = int(np.isclose(
            reconstructed_curve.Z_FEM_ohm.to_numpy(float),
            archived_curve.Z_FEM_ohm.to_numpy(float),
            rtol=0, atol=1e-10).sum())

        display(Markdown('#### Проверка восстановленных контактных функций'))
        display(pd.DataFrame({
            'вариант построения электродного пятна': [
                'архивный результат до перепрогона',
                'самый ранний алгоритм: независимые пятна с общими узлами',
                'реконструкция быстрого алгоритма с разделением узлов',
            ],
            'RMSE, Ом': [pre_rerun_summary.rms_residual_ohm,
                         earliest.rms_residual_ohm,
                         reconstructed_fast.rms_residual_ohm],
            'J': [np.nan, earliest_objective, reconstructed_fast_objective],
            'статус': [
                'сохранённые числа; исходная функция отсутствует',
                'точно повторена по доступному раннему исходнику',
                'функциональная реконструкция по установленным правилам',
            ],
        }))

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.plot(pre_rerun_comparison.L_mm, pre_rerun_comparison.Z_FEM_ohm,
                's--', label='архивная кривая')
        ax.plot(legacy_patch_comparison['earliest'].L_mm,
                legacy_patch_comparison['earliest'].Z_FEM_ohm,
                'o:', label='самый ранний алгоритм')
        ax.plot(reconstructed_curve.L_mm, reconstructed_curve.Z_FEM_ohm,
                'd-', label='реконструкция быстрого алгоритма')
        ax.set(title='Что воспроизводят восстановленные контактные функции',
               xlabel='L, мм', ylabel='Z, Ом')
        ax.legend()

        display(Markdown(fr'''
Архивная геометрия содержит 36 электродных пятен. Реконструкция быстрого алгоритма
совпала с ней для **{exact_geometry_count} из 36** пятен. Два оставшихся расхождения
относятся к электроду $V_-$ сборок 50 и 70 мм; максимальное смещение центроида равно
**{max_centroid_error:.3f} мм**, максимальное отличие площади —
**{max_area_error:.3f} мм²**. Рассчитанный импеданс точно совпал с архивом для
**{exact_impedance_count} из 9** сборок. В прежней точке RMSE снизился с
{earliest.rms_residual_ohm:.3f} Ом для самого раннего алгоритма до
{reconstructed_fast.rms_residual_ohm:.3f} Ом для реконструкции, но архивное значение
{pre_rerun_summary.rms_residual_ohm:.3f} Ом полностью не воспроизведено.

Таким образом, восстановлено влияние прежнего правила разделения электродных узлов и
подтверждена высокая чувствительность результата к одной-двум краевым граням при малых
расстояниях между электродами. Точное происхождение двух архивных пятен остаётся
неустановленным. Реконструированный режим используется только для проверки происхождения
чисел и не заменяет текущую физически ограниченную модель контакта.
'''))

    old_legacy = ablation_summary['oldpoint_legacy']
    old_manifest = ablation_summary['oldpoint_manifest']
    recovered = ablation_summary['oldstart_manifest']
    old_legacy_history = ablation_history['oldpoint_legacy']
    old_manifest_history = ablation_history['oldpoint_manifest']
    recovered_history = ablation_history['oldstart_manifest']
    old_legacy_curve = ablation_comparison['oldpoint_legacy']
    recovered_curve = ablation_comparison['oldstart_manifest']

    old_objective = float(old_legacy_history.objective.min())
    manifest_objective = float(old_manifest_history.objective.min())
    recovered_objective = float(recovered_history.objective.min())
    rejected_objective = float(rejected_history.objective.min())
    cem_objective = float(fast_cem.fast_CEM_objective.min())
    recovered_shift_from_archive_mm = float(np.hypot(
        recovered.centre_u_mm - pre_rerun_summary.centre_u_mm,
        recovered.centre_v_mm - pre_rerun_summary.centre_v_mm))
    rejected_shift_from_recovered_mm = float(np.hypot(
        rejected_summary.centre_u_mm - recovered.centre_u_mm,
        rejected_summary.centre_v_mm - recovered.centre_v_mm))

    archived_curve_aligned = pre_rerun_comparison.set_index('L_mm').loc[
        old_legacy_curve.L_mm, 'Z_FEM_ohm'].to_numpy(float)
    forward_change = old_legacy_curve.Z_FEM_ohm.to_numpy(float) - archived_curve_aligned
    forward_change_rms = float(np.sqrt(np.mean(forward_change**2)))
    forward_change_max = float(np.max(np.abs(forward_change)))
    forward_change_max_L = int(old_legacy_curve.iloc[np.argmax(np.abs(forward_change))].L_mm)

    cause_table = pd.DataFrame({
        'контрольный расчёт': [
            'архивный v5 непосредственно до перепрогона',
            'прежняя точка, прежняя привязка 100 мм, текущий код',
            'прежняя точка, исправленная привязка 90 мм, текущий код',
            'поиск из прежней точки, исправленная привязка',
            'поиск fmincon из нулевого смещения — отклонён',
            'исторический CEM-кандидат при rho из отклонённой точки',
        ],
        'u, мм': [pre_rerun_summary.centre_u_mm, old_legacy.centre_u_mm,
                 old_manifest.centre_u_mm, recovered.centre_u_mm,
                 rejected_summary.centre_u_mm, fast_cem.iloc[0].centre_u_mm],
        'v, мм': [pre_rerun_summary.centre_v_mm, old_legacy.centre_v_mm,
                 old_manifest.centre_v_mm, recovered.centre_v_mm,
                 rejected_summary.centre_v_mm, fast_cem.iloc[0].centre_v_mm],
        'phi, град': [pre_rerun_summary.phi_deg, old_legacy.phi_deg,
                     old_manifest.phi_deg, recovered.phi_deg,
                     rejected_summary.phi_deg, fast_cem.iloc[0].phi_deg],
        'RMSE, Ом': [pre_rerun_summary.rms_residual_ohm, old_legacy.rms_residual_ohm,
                    old_manifest.rms_residual_ohm, recovered.rms_residual_ohm,
                    rejected_summary.rms_residual_ohm, best_exact_rms],
        'J': [np.nan, old_objective, manifest_objective, recovered_objective,
              rejected_objective, cem_objective],
        'смысл сравнения': [
            'историческая точка; иной алгоритм построения контакта',
            'изолировано влияние текущей контактной модели',
            'изолирована замена 100 на 90 мм',
            'проверено влияние начального приближения',
            'неудачный локальный путь градиентного решателя',
            'геометрический скрининг без повторной оценки rho',
        ],
    })
    display(cause_table)

    delta_table = pd.DataFrame({
        'проверяемое изменение': [
            'построение контактных площадок при неизменных параметрах и данных',
            'замена привязки 100 на 90 мм при неизменных параметрах',
            'смена старта: отклонённая точка fmincon вместо поиска из прежней области',
        ],
        'изменение показателя': [
            f'RMSE +{old_legacy.rms_residual_ohm-pre_rerun_summary.rms_residual_ohm:.3f} Ом; '
            f'RMS сдвига самой FEM-кривой {forward_change_rms:.3f} Ом',
            f'RMSE {old_manifest.rms_residual_ohm-old_legacy.rms_residual_ohm:+.3f} Ом; '
            f'J {manifest_objective-old_objective:+.3f}',
            f'J {rejected_objective-recovered_objective:+.3f}; '
            f'RMSE {rejected_summary.rms_residual_ohm-recovered.rms_residual_ohm:+.3f} Ом',
        ],
        'оценка влияния на пространственный сдвиг': [
            'существенно изменяет прямую задачу, но сама по себе не требует ухода центра',
            'малое; в прежней точке RMSE даже немного уменьшается',
            'определяющее для полученного ухода v и фиксации угла',
        ],
    })
    display(delta_table)

    def objective_parts_for(row, curve, objective_value):
        residual = curve.residual_ohm.to_numpy(float)
        abs_residual = np.abs(residual)
        huber = 0.5*np.minimum(abs_residual, 5.0)**2 + 5.0*np.maximum(abs_residual-5.0, 0)
        slope_error = (float(row.fem_slope_ohm_per_mm)
                       - float(row.experimental_slope_ohm_per_mm))*90.0
        h = float(row.h_centre_mm)
        rho2 = float(row.rho_lungs_ohm_m)
        parts = [
            huber.mean(),
            0.20*slope_error**2,
            ((h-15.0)/7.0)**2,
            2.0*max(h-30.0, 0.0)**2 + 2.0*max(7.0-h, 0.0)**2,
            0.25*((rho2-17.4067)/6.0)**2,
        ]
        return parts + [objective_value]

    objective_comparison = pd.DataFrame(
        [
            objective_parts_for(recovered, recovered_curve, recovered_objective),
            objective_parts_for(rejected_summary, rejected_comparison, rejected_objective),
        ],
        index=['поиск из прежней точки', 'записанная точка из нулевого старта'],
        columns=['Huber по отдельным Z', 'штраф наклона', 'штраф CT-глубины',
                 'штраф выхода h за диапазон', 'штраф rho2', 'J'],
    )
    display(Markdown('#### Почему записанная точка хуже по собственному критерию'))
    display(objective_comparison)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].plot(rejected_comparison.L_mm, rejected_comparison.Z_experiment_ohm, 'ko-', lw=2,
                 label='эксперимент: независимая запись 90 мм')
    axes[0].plot(pre_rerun_comparison.L_mm, pre_rerun_comparison.Z_FEM_ohm,
                 's--', label='архивная FEM-кривая')
    axes[0].plot(old_legacy_curve.L_mm, old_legacy_curve.Z_FEM_ohm,
                 'o--', label='та же точка, текущая контактная модель')
    axes[0].plot(recovered_curve.L_mm, recovered_curve.Z_FEM_ohm,
                 'd-', label='поиск из прежней точки')
    axes[0].plot(rejected_comparison.L_mm, rejected_comparison.Z_FEM_ohm,
                 'x-', label='записанная точка из нулевого старта')
    axes[0].set(title='Какая часть изменения возникла в прямой задаче',
                xlabel='L, мм', ylabel='Z, Ом')
    axes[0].legend(fontsize=8)

    objective_labels = ['прежняя точка,\nпрежняя привязка',
                        'прежняя точка,\nисправленная привязка',
                        'поиск из\nпрежней точки',
                        'нулевой старт',
                        'CEM при\nфиксированных rho']
    objective_values = [old_objective, manifest_objective, recovered_objective,
                        rejected_objective, cem_objective]
    axes[1].bar(objective_labels, objective_values,
                color=['#7f7f7f', '#4c78a8', '#2ca02c', '#d62728', '#ffbf00'])
    axes[1].set(title='Меньшее J означает лучшее согласование по принятому критерию',
                ylabel='J')
    axes[1].tick_params(axis='x', labelsize=8)

    display(Markdown(fr'''#### Интерпретация контрольных прогонов

При неизменных параметрах, прежнем наборе измерений и той же объёмной FEM-сетке переход к
текущему построению контактных площадок изменил рассчитанную кривую на
**{forward_change_rms:.3f} Ом RMS**. Наибольшее изменение составило
**{forward_change_max:.3f} Ом** для сборки **{forward_change_max_L} мм**. Поэтому архивный
RMSE {pre_rerun_summary.rms_residual_ohm:.3f} Ом нельзя воспроизвести простым возвратом
параметров: он относится к прежней дискретизации контакта.

Замена дублированной записи 100 мм на независимую запись 90 мм не вызвала большого сдвига.
В обеих записях сохранены одинаковые уровни: 74,580 Ом на вдохе и 70,327 Ом на выдохе;
изменился аргумент $L$, к которому относится это измерение, а не само значение импеданса.
В прежней точке RMSE изменился с **{old_legacy.rms_residual_ohm:.3f}** до
**{old_manifest.rms_residual_ohm:.3f} Ом**, то есть уменьшился на
**{old_legacy.rms_residual_ohm-old_manifest.rms_residual_ohm:.3f} Ом**. Изменение $J$
составило только **{manifest_objective-old_objective:+.3f}**.

Определяющим оказался путь оптимизации. При старте из прежней области поиск остался рядом с
ней: $(u,v,\varphi)=({recovered.centre_u_mm:.3f}, {recovered.centre_v_mm:.3f},
{recovered.phi_deg:.1f}^\circ)$ и дал $J={recovered_objective:.3f}$. Расстояние от архивного
центра в координатах касательной плоскости составило только
**{recovered_shift_from_archive_mm:.3f} мм**, а угол не изменился. Напротив, точка из нулевого
старта отстоит от этого контрольного решения на **{rejected_shift_from_recovered_mm:.3f} мм** и
отличается по углу на **{abs(rejected_summary.phi_deg-recovered.phi_deg):.1f}°**.
Записанная после
перепрогона точка из нулевого смещения имеет $J={rejected_objective:.3f}$, хотя меньшее значение
критерия уже найдено. Её вклад от несовпадения отдельных $Z$ равен
**{objective_comparison.loc['записанная точка из нулевого старта', 'Huber по отдельным Z']:.2f}**,
а штраф наклона — **{objective_comparison.loc['записанная точка из нулевого старта', 'штраф наклона']:.2f}**.
Следовательно, пространственный уход этой точки не подтверждается принятым критерием.

Причина поведения решателя численная: при выборе конечного набора поверхностных граней
целевая функция по координатам электродов кусочно-постоянна и меняется скачками. Микроскопический
шаг конечных разностей `fmincon` не видит этих скачков. Поэтому при нулевом старте угол остался
ровно $0^\circ$, а при старте из прежней точки — ровно $-10^\circ$. Сдвиг по $v$ в первом
прогоне в значительной части следовал штрафу за CT-глубину, а не минимуму расхождения с
измеренным $Z(L)$.

CEM-кандидат имеет меньший RMSE, но $J={cem_objective:.3f}$ и найден при фиксированных
$\rho_1,\rho_2$ из неудачной точки; кроме того, его угол лежит на границе поиска. Его следует
сохранить как отдельный диагностический кандидат, но не считать новой итоговой локализацией.
'''))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
axes[0].plot(comparison.L_mm, comparison.Z_experiment_ohm, 'ko-', lw=2,
             label='эксперимент')
if baseline_available:
    axes[0].plot(baseline_comparison.L_mm, baseline_comparison.Z_FEM_ohm, 's--',
                 label='исторический v4, иной набор')
axes[0].plot(comparison.L_mm, comparison.Z_FEM_ohm, 'o-', lw=2,
             label='v5, итоговый disjoint')
axes[0].plot(comparison.L_mm, best_exact_z, 'd-', lw=2,
             label='исторический CEM-кандидат')
axes[0].set(title='Измеренная и рассчитанные кривые',
            xlabel='L, мм', ylabel='Z, Ом')
axes[0].legend()

x = np.arange(len(comparison))
axes[1].bar(x - 0.18, comparison.residual_ohm, width=0.36,
            label='итоговый disjoint')
axes[1].bar(x + 0.18, best_exact_residual, width=0.36,
            label='исторический CEM-кандидат')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xticks(x, comparison.L_mm.astype(int))
axes[1].set(title='Как читать остатки: нуль означает совпадение',
            xlabel='L, мм', ylabel='$Z_{модель}-Z_{эксперимент}$, Ом')
axes[1].legend()
""")

optional_historical_code(r"""
overlap = pd.read_csv(OUT / 'nik_trkg4_patch_overlap_v5_1mm_local2mm.csv')
timing = pd.DataFrame({
    'этап': ['геометрия площадок', 'факторизация', 'решения с матрицей Грина', 'локальные CEM'],
    'секунды': [fast_cem_summary.geometry_seconds, fast_cem_summary.factor_seconds,
                fast_cem_summary.green_solve_seconds, fast_cem_summary.local_CEM_seconds],
})
display(Markdown('### Исторический контроль раздельности площадок v5_1mm_local2mm'))
display(pd.DataFrame({
    'проверено пар электродов': [len(overlap)],
    'пар с общими узлами': [(overlap.shared_nodes > 0).sum()],
    'пар с общими гранями': [(overlap.shared_faces > 0).sum()],
}))
display(Markdown('### Измеренное время низкорангового CEM-пересчёта'))
display(timing)

display(Markdown(fr'''### Основной результат v5 и статус исторического CEM-скрининга

- Выбранный кандидат `{result_tag}`: $\rho_1={summary.rho_soft_ohm_m:.3f}$ Ом·м,
  $\rho_2={summary.rho_lungs_ohm_m:.3f}$ Ом·м,
  $(u,v,\varphi)=({summary.centre_u_mm:.3f}, {summary.centre_v_mm:.3f},
  {summary.phi_deg:.1f}^\circ)$, $h_{{centre}}={summary.h_centre_mm:.2f}$ мм.
- Его наклон FEM: **{summary.fem_slope_ohm_per_mm:.6f} Ом/мм** при экспериментальном
  **{summary.experimental_slope_ohm_per_mm:.6f} Ом/мм**;
  RMSE **{summary.rms_residual_ohm:.3f} Ом**, MAE **{summary.mae_residual_ohm:.3f} Ом**;
  секущие: {secant_status_text}
- Исторический CEM-кандидат рассчитан при $\rho_1,\rho_2$ отклонённой точки `fmincon`, а не
  при итоговых значениях `disjoint`:
  $(u,v,\varphi)=({best_exact.centre_u_mm:.1f}, {best_exact.centre_v_mm:.1f},
  {best_exact.phi_deg:.1f}^\circ)$, $h_{{centre}}={best_exact.h_centre_mm:.2f}$ мм;
  RMSE **{best_exact_rms:.3f} Ом**, MAE **{best_exact_mae:.3f} Ом**.
- Его угол равен границе старого диапазона ({fast_pem_summary.phi_min_deg:.0f}°), поэтому этот
  результат сохраняется только как диагностика и не конкурирует с итоговой координатой.

Построение площадок занимает {fast_cem_summary.geometry_seconds:.1f} с, факторизация —
{fast_cem_summary.factor_seconds:.2f} с, а {int(fast_cem_summary.evaluated_pose_count)} малых
CEM-решений — {fast_cem_summary.local_CEM_seconds:.1f} с в историческом прогоне. Эти времена
не относятся к выбранному тегу. Статус нового кандидата указан в начале отчёта.
'''))
""")

md(r"""
## 15. Аудит принятой STL/FEM-модели

Этот раздел фиксирует не только имена файлов, но и физический смысл выполненной
предобработки. Полная обязательная запись находится в
`docs/INVERSE_IDENTIFIABILITY_AND_BREATHING_STATES.md`.

### Наружный объём тела

1. Исходный `body.stl` растеризован в изотропную маску 1 мм.
2. Морфологическое замыкание радиусом 8 мм оставляло 12 тоннелей и давало после заполнения только 16.12 л.
3. Минимальным проверенным вариантом, закрывшим тоннели, стало замыкание радиусом 10 мм.
4. Выполнены заполнение замкнутых полостей, выбор основной компоненты, извлечение поверхности
   и прореживание сетки с сохранением топологии.
5. Принят `body_solid_v3_1mm_r10_volume_fill.stl`: одна замкнутая двумерно-многообразная компонента,
   0 открытых/неманифолдных рёбер, 20.171 л, 77 562 вершины и 155 120 граней.

Весь объём внутри кожи сначала заполнен фоновой мягкой тканью. Лёгкие, целая маска сердца и кости
переопределяют материал соответствующих тетраэдров. Поэтому внутренних пустот тела в FEM нет.
Файл маски крови для Ника отсутствует; проводимость крови рассчитана в конфигурации, но ни одному
тетраэдру не назначена. Вся доступная маска сердца получила проводимость сердечной мышцы.

Морфологическое замыкание радиусом 10 мм является вычислительным ремонтом негерметичной маски и может локально менять
поверхность около узких щелей. В плоскости рёберных электродов совпадение с DICOM проверено,
но глобальное отсутствие отклонения наружной поверхности строго не доказано. Искусственные
торцы удлинённых рук лежат вне DICOM и не используются в этой обратной задаче.

### Лёгкие

Используется `lungs_solid_v3_volume_fill.stl`, полученный из `lungs_ct_full_clean.stl`, а не
старый `lungs.stl` и не выпуклая оболочка. Контрольная маска 1 мм изменилась при заполнении всего на
один воксель: очищенная STL-модель уже задавала практически заполненный объём. Получено 4.242 л,
одна замкнутая двумерно-многообразная компонента и 0 открытых/неманифолдных рёбер. Наружная трахея сохранена
до $z_{max}=19.49$ мм.

$\rho_2$ назначается эффективному объёму «лёгочная ткань + воздух». Внутренние бронхи не
являются отдельными пустотами. Заполнение не восстанавливает ветви, которых не было во входной
сегментации; для отдельной проводимости дыхательных путей нужен независимый `airways.stl`.

### FEM

Принята Delaunay-сетка `nik_body_solid_v3_1mm_local2mm_alg1.msh`: 153 415 узлов,
721 943 тетраэдра и объём 20.170763 л. Шаг около рабочей оси примерно 2 мм, затем 4 и 8 мм,
вдали — до 25 мм. Это локально уточнённая, а не равномерная 1–2-мм сетка всего тела.
Экспериментальные уточнения поверхности электродов на руках в текущую конфигурацию не включены.
""")

md(r"""
## 16. Почему численная точка не означает единственное физическое решение
Числовые оценки в этом разделе выводятся только при наличии проверенной шаговой диагностики
и полностью конечной матрицы секущих выбранного тега. При `compute_jacobian=false` или частично
недоступных столбцах раздел пропускается. Спектр характеризует линейную суррогатную модель по
секущим и не даёт доверительного интервала или доказательства физической единственности.


Для одной сборки и фиксированного $h$ имеется одно уравнение

$$Z_L=F_L(\rho_1,\rho_2,h),$$

поэтому допустимые $(\rho_1,\rho_2)$ образуют кривую. Текущая задача возвращает точку, потому
что связывает девять независимых размеров общими параметрами
$\theta=(\rho_1,\rho_2,u,v,\varphi)$, вычисляет $h$ из STL и добавляет границы и априорные штрафы.
Для Ника включена независимая запись 90 мм; запись 100 мм исключена как её поздняя
перезаписанная копия.

Критерий:

$$
\begin{aligned}
J={}&\operatorname{mean}[\operatorname{Huber}_5(Z_{FEM}-Z_{exp})]
+0.20[(b_{FEM}-b_{exp})90]^2\\
&+[(h_c-15)/7]^2+P_{h\notin[7,30]}
+0.25[(\rho_2-17.4067)/6]^2.
\end{aligned}
$$

Наклон вычислен из тех же значений $Z$ и не является дополнительным независимым измерением.
Сведения о выполненном поиске берутся из выбранного summary и проверенной истории.
Полная допустимая кривая включает девять FEM-значений; отклонение `invalid_geometry` возможно
до FEM. Наличие и полнота расчёта секущих определяются `compute_jacobian` и diagnostics,
а не фактом существования CSV с совместимыми именами колонок.
Исторический PEM/CEM-скрининг с фиксированными сопротивлениями сохранён отдельно и в основной
результат не входит. Получен локальный вычислительный кандидат, а не доказанный глобальный или
единственный физический минимум.
Лишние десятичные знаки не являются физической точностью.
""")

jacobian_code(r"""
jacobian = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_jacobian_{result_tag}.csv')
parameter_names = ['rho1', 'rho2', 'u', 'v', 'phi']
parameter_scales = np.array([1.0, 5.0, 10.0, 10.0, 10.0])
J = jacobian.iloc[:, 1:].to_numpy(float)
J_scaled = J * parameter_scales
_, singular_values, right_vectors = np.linalg.svd(J_scaled, full_matrices=False)
weak_scaled = right_vectors[-1]
weak_physical = weak_scaled * parameter_scales
weak_delta_z = J @ weak_physical

residual = comparison.residual_ohm.to_numpy(float)
abs_residual = np.abs(residual)
huber = 0.5 * np.minimum(abs_residual, 5.0)**2 + 5.0 * np.maximum(abs_residual - 5.0, 0)
slope_difference = float(summary.fem_slope_ohm_per_mm-summary.experimental_slope_ohm_per_mm)
objective_parts = pd.DataFrame({
    'часть критерия': ['средняя функция Хьюбера по Z', 'штраф наклона', 'априорный штраф по h',
                       'выход h за 7–30 мм', 'априорный штраф по rho2'],
    'вклад': [huber.mean(), 0.20*(90*slope_difference)**2,
              ((float(summary.h_centre_mm)-15)/7)**2,
              2*max(float(summary.h_centre_mm)-30, 0)**2
              + 2*max(7-float(summary.h_centre_mm), 0)**2,
              0.25*((float(summary.rho_lungs_ohm_m)-17.4067)/6)**2]
})

weak_table = pd.DataFrame({
    'параметр': parameter_names,
    'компенсирующее изменение': weak_physical,
    'единица': ['Ом·м', 'Ом·м', 'мм', 'мм', 'град']
})
display(objective_parts)
display(pd.DataFrame({'масштабированное сингулярное число': singular_values}))
display(weak_table)
display(Markdown(
    f'Линейная суррогатная модель по секущим предсказывает для слабого направления **'
    f'{np.sqrt(np.mean(weak_delta_z**2)):.3f} Ом RMS** и не более '
    f'**{np.max(np.abs(weak_delta_z)):.3f} Ом**; '
    f'число обусловленности = **{singular_values[0]/singular_values[-1]:.1f}**. '
    'Это предсказание J @ delta_theta; полный FEM в смещённой точке здесь не пересчитывался. '
    'Результат не является оценкой неопределённости позы.'
))

fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
ax.semilogy(np.arange(1, len(singular_values) + 1), singular_values, 'o-', lw=2)
ax.set(xticks=np.arange(1, len(singular_values) + 1),
       xlabel='номер сингулярного числа', ylabel='s',
       title='Спектр масштабированного finite-step Jacobian')
""")

jacobian_code(r"""
display(Markdown(f'''### Интерпретация обусловленности

Для выбранного масштаба параметров спектр конечных секущих имеет число обусловленности
**{singular_values[0]/singular_values[-1]:.1f}**. Слабейшее направление показывает
компенсацию параметров в локальной численной диагностике.

Эти значения получены на явно заданных конечных шагах для дискретных CEM-площадок.
Они зависят от шага и не являются производной непрерывной позы, информацией Фишера,
оценкой стандартной ошибки или доказательством идентифицируемости положения.
'''))
""")

md(r"""
## 17. Глубокий вдох и выдох без отдельной CT выдоха
**Ограничение расчёта.** Линейная подгонка ниже доступна только при полностью конечных
секущих выбранного кандидата на вдохе и проверенной шаговой диагностике. При
`compute_jacobian=false` или частично недоступных секущих она пропускается.
Её невязка не доказывает геометрический вклад: изменение сопротивления может выходить за
область применимости линеаризации. Полученные $\Delta\rho$ не являются физиологическими оценками.


Для Ника сохранены уровни `BASE_2` на задержке глубокого вдоха и глубокого выдоха. Вдох и выдох
для каждого размера записаны при одной и той же установке соответствующей сборки, поэтому
разность частично подавляет постоянную составляющую и контактную ошибку. Она не устраняет изменение
формы грудной клетки, объёма лёгкого и профиля $h(s)$.

Спокойное и усиленное дыхание следует анализировать как временные сигналы внутри каждой записи:
они дают повторяемость, шум и промежуточные дыхательные уровни, но не являются одновременной
многомерной кривой по всем размерам.
""")

breathing_linear_code(r"""
breath = pd.read_csv(ROOT / 'data' / 'nik' / 'experimental' / 'nik_breath_hold_levels.csv')
breath_independent = breath[breath.is_independent == 1].copy().reset_index(drop=True)
jacobian_by_L = jacobian.set_index('L_mm').loc[breath_independent.L_mm]
delta_measured = breath_independent.delta_in_minus_ex_ohm.to_numpy(float)
j_rho1 = jacobian_by_L.dZ_drho_soft.to_numpy(float)
j_rho2 = jacobian_by_L.dZ_drho_lungs.to_numpy(float)

delta_rho2_only = float(j_rho2 @ delta_measured / (j_rho2 @ j_rho2))
delta_pred_rho2_only = j_rho2 * delta_rho2_only
delta_rho12 = np.linalg.lstsq(np.column_stack((j_rho1, j_rho2)),
                              delta_measured, rcond=None)[0]
delta_pred_rho12 = np.column_stack((j_rho1, j_rho2)) @ delta_rho12

diagnostic_delta = pd.DataFrame({
    'линейная суррогатная модель': ['секущие при фиксированной геометрии, только rho2',
                                    'секущие при фиксированной геометрии, rho1 и rho2'],
    'delta rho1, Ом·м': [0.0, delta_rho12[0]],
    'delta rho2, Ом·м': [delta_rho2_only, delta_rho12[1]],
    'RMSE delta Z, Ом': [np.sqrt(np.mean((delta_pred_rho2_only-delta_measured)**2)),
                         np.sqrt(np.mean((delta_pred_rho12-delta_measured)**2))]
})

legacy_and_ct = pd.DataFrame({
    'источник': ['CT/HU + Maxwell-Garnett, глубокий вдох',
                 'старая двухслойная совместная оценка, вдох',
                 'старая двухслойная совместная оценка, выдох',
                 f'выбранный FEM-кандидат {result_tag}, вдох'],
    'rho2, Ом·м': [17.4067, 17.6575, 14.7108, summary.rho_lungs_ohm_m],
    'оговорка': ['f_air=0.799, rho_matrix=2.5 Ом·м',
                 'фиксированное h=15 мм', 'фиксированное h=15 мм',
                 'идентифицируемость не установлена']
})

q_values = np.arange(0.50, 1.001, 0.10)
f_air_in = 0.799
rho_matrix = 2.5
f_air_ex = 1.0 - (1.0-f_air_in)/q_values
rho2_ex_scenarios = rho_matrix*(1.0+f_air_ex/2.0)/(1.0-f_air_ex)
volume_scenarios = pd.DataFrame({
    'q=V_ex/V_in': q_values,
    'f_air_ex': f_air_ex,
    'rho2_ex по Maxwell-Garnett, Ом·м': rho2_ex_scenarios,
})

display(breath_independent)
display(diagnostic_delta)
display(legacy_and_ct)
display(volume_scenarios)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
axes[0].plot(breath_independent.L_mm, breath_independent.Z_inhale_hold_ohm,
             'o-', lw=2, label='глубокий вдох, задержка')
axes[0].plot(breath_independent.L_mm, breath_independent.Z_exhale_hold_ohm,
             's-', lw=2, label='глубокий выдох, задержка')
axes[0].set(xlabel='L, мм', ylabel='Z, Ом', title='Сохранённые уровни задержек дыхания')
axes[0].legend()
axes[1].plot(breath_independent.L_mm, delta_measured, 'ko-', lw=2,
             label='измеренное delta Z')
axes[1].plot(breath_independent.L_mm, delta_pred_rho2_only, 's--',
             label=f'только delta rho2={delta_rho2_only:.1f} Ом·м')
axes[1].plot(breath_independent.L_mm, delta_pred_rho12, '^--',
             label=f'delta rho1={delta_rho12[0]:.2f}, delta rho2={delta_rho12[1]:.1f}')
axes[1].set(xlabel='L, мм', ylabel='Z_in-Z_ex, Ом',
            title='Линейная подгонка по секущим при фиксированной геометрии')
axes[1].legend()
""")

md(r"""
### Что можно считать предварительным диапазоном

Большое $\Delta\rho$ или невязка линейной подгонки не доказывают необходимость изменения
геометрии. Конечный нелинейный отклик при той же геометрии может существенно отличаться от
предсказания по секущим. Линейная диагностика не разделяет нелинейность, изменение геометрии,
свойства тканей и модельную ошибку; количественный вклад изменения $h(s)$ здесь не установлен.
Исторические линейные оценки не переносятся в выбранный тег. Значение около 17.4 Ом·м из
CT/HU и Maxwell-Garnett является модельной априорной оценкой, не прямым измерением $\rho_2$.

Можно построить сценарный диапазон через неизвестное отношение объёмов
$q=V_{lung,ex}/V_{lung,in}$. При постоянном объёме тканевой матрицы

$$f_{air,ex}=1-\frac{1-f_{air,in}}{q},$$

а в использованной модели Maxwell-Garnett

$$\rho_2(f)=\rho_m\frac{1+f/2}{1-f}.$$

При $f_{air,in}=0.799$ и $\rho_m=2.5$ Ом·м диапазон $q=0.5\ldots0.9$ соответствует
$\rho_{2,ex}\approx8.1\ldots15.5$ Ом·м. Это поисковая сценарная область, а не доверительный
интервал и не результат выбранного FEM-прогона. Для ограничения $q$ нужны согласованные
оценки объёмов обоих состояний; сама спирометрическая разность не определяет абсолютное
отношение объёмов, а перенос на $\rho_{2,ex}$ сохраняет допущения модели смеси.

Для первого полного многофазного поиска можно использовать широкие сценарные границы:

- $\rho_{2,in}=14\ldots30$ Ом·м с модельной CT/HU-оценкой около 17.4 Ом·м;
- $\rho_{2,ex}=8\ldots16$ Ом·м только как сценарий при $q=0.5\ldots0.9$, расширяя его при
  другой оценке объёма или сопротивления матрицы.

### Возможное дальнейшее исследование без CT выдоха

Следующая постановка здесь не исполняется; её необходимость и идентифицируемость не доказаны
линейной подгонкой.

CT глубокого вдоха может служить геометрическим якорем. Для выдоха можно рассмотреть семейство допустимых
деформаций $D(q,\eta)$, которое изменяет объём лёгкого, положение диафрагмы, поверхность кожи
и профиль $h_{ex}(s)$. В таком исследовании можно использовать абсолютный вдох и разность состояний:

$$
J=J_{in}[Z_{in}^{FEM}-Z_{in}^{exp}]
+w_\Delta J_\Delta[(Z_{in}^{FEM}-Z_{ex}^{FEM})-\Delta Z^{exp}]
+J_{CT}+J_{volume}+J_{deformation}.
$$

Общими оставляются $\rho_1$, электродная ось и параметры контакта. $\rho_2$ разных состояний
лучше связать через объём/воздушную долю, а не подбирать независимо. Чередование
«геометрия → $\rho$ → геометрия» допустимо как алгоритм, но каждый результат необходимо
проверять профильным критерием: при фиксированном $\rho_{2,in}$ или $q_{ex}$ все остальные
параметры переоптимизируются заново. Иначе итерация может остановиться в произвольной точке
длинной неидентифицируемой долины.
## 18. Контроль положения электродных сборок: расширенный прямой поиск

Этот раздел заменяет предварительную трактовку большого пространственного сдвига из раздела 14.
Для двух способов построения электродных площадок выполнен возобновляемый прямой поиск с
физически осмысленными шагами. Оба варианта использовали один и тот же исправленный ряд из
девяти независимых размеров, одну объёмную FEM-сетку, одну целевую функцию и одинаковый набор
начальных областей. Различался только способ выбора граничных треугольников электродов.

`legacy_fast` служит реконструкцией утраченного исторического алгоритма и нужен для проверки
происхождения архивных чисел. `disjoint` является текущим физически ограниченным вариантом:
электродные площадки связны и не имеют общих узлов.
""")

code(r"""
if restartable_available and pre_rerun_available:
    archive_fit = pre_rerun_summary
    shifted_fit = rejected_summary
    legacy_fit = restartable_summary['legacy_fast']
    disjoint_fit = restartable_summary['disjoint']
    legacy_search = restartable_search['legacy_fast']
    disjoint_search = restartable_search['disjoint']
    legacy_curve = restartable_comparison['legacy_fast']
    disjoint_curve = restartable_comparison['disjoint']
    legacy_hist = restartable_history['legacy_fast'].copy()
    disjoint_hist = restartable_history['disjoint'].copy()
    disjoint_invalid = int((disjoint_hist.objective >= 1e11).sum())

    def tangent_shift(left, right):
        return float(np.hypot(float(left.centre_u_mm-right.centre_u_mm),
                              float(left.centre_v_mm-right.centre_v_mm)))

    legacy_shift = tangent_shift(legacy_fit, archive_fit)
    disjoint_shift = tangent_shift(disjoint_fit, archive_fit)
    rejected_shift = tangent_shift(shifted_fit, archive_fit)
    legacy_best_eval = int(legacy_hist.loc[legacy_hist.objective.idxmin(), 'evaluation'])
    disjoint_best_eval = int(disjoint_hist.loc[disjoint_hist.objective.idxmin(), 'evaluation'])
    legacy_j25 = float(legacy_hist.loc[legacy_hist.evaluation <= 25, 'objective'].min())
    disjoint_j25 = float(disjoint_hist.loc[disjoint_hist.evaluation <= 25, 'objective'].min())

    final_table = pd.DataFrame({
        'расчёт': [
            'архив до перепрогона',
            'градиентный прогон из нулевого смещения — отклонён',
            'legacy_fast: прямой поиск',
            'disjoint: прямой поиск',
        ],
        'размерный ряд': [
            '100 мм включён; 90 мм исключён',
            '90 мм включён; 100 мм исключён',
            '90 мм включён; 100 мм исключён',
            '90 мм включён; 100 мм исключён',
        ],
        'оценки FEM': [np.nan, int(len(rejected_history)),
                       int(legacy_search['function_evaluations']),
                       int(disjoint_search['function_evaluations'])],
        'J': [np.nan, float(rejected_history.objective.min()),
              float(legacy_search['objective']), float(disjoint_search['objective'])],
        'RMSE, Ом': [archive_fit.rms_residual_ohm, shifted_fit.rms_residual_ohm,
                     legacy_fit.rms_residual_ohm, disjoint_fit.rms_residual_ohm],
        'rho1, Ом·м': [archive_fit.rho_soft_ohm_m, shifted_fit.rho_soft_ohm_m,
                       legacy_fit.rho_soft_ohm_m, disjoint_fit.rho_soft_ohm_m],
        'rho2, Ом·м': [archive_fit.rho_lungs_ohm_m, shifted_fit.rho_lungs_ohm_m,
                       legacy_fit.rho_lungs_ohm_m, disjoint_fit.rho_lungs_ohm_m],
        'u, мм': [archive_fit.centre_u_mm, shifted_fit.centre_u_mm,
                  legacy_fit.centre_u_mm, disjoint_fit.centre_u_mm],
        'v, мм': [archive_fit.centre_v_mm, shifted_fit.centre_v_mm,
                  legacy_fit.centre_v_mm, disjoint_fit.centre_v_mm],
        'phi, град': [archive_fit.phi_deg, shifted_fit.phi_deg,
                      legacy_fit.phi_deg, disjoint_fit.phi_deg],
        'h центра, мм': [archive_fit.h_centre_mm, shifted_fit.h_centre_mm,
                         legacy_fit.h_centre_mm, disjoint_fit.h_centre_mm],
        'сдвиг центра от архива, мм': [0.0, rejected_shift, legacy_shift, disjoint_shift],
    })
    display(final_table)

    influence_table = pd.DataFrame({
        'изменение': [
            'числа измеренного импеданса',
            'атрибуция дублированной записи',
            'объёмная FEM-сетка между двумя MAT-файлами v5',
            'дискретизация контактных площадок',
            'метод поиска положения',
        ],
        'установленный результат': [
            'не изменились',
            'то же значение перенесено с L=100 на независимый L=90',
            'nodes, elems, boundary, elem_sigma и tissue_id совпадают поэлементно',
            '34 из 36 архивных пятен восстановлены точно; различаются V− при 50 и 70 мм',
            'fmincon дал худшую точку; прямой поиск вернул центр в архивную область',
        ],
        'влияние на итог': [
            'нулевое',
            'малое при фиксированных параметрах; меняется аргумент L',
            'нулевое для объёмной прямой задачи; хэши MAT различаются из-за других полей',
            'заметно меняет Z(50) и частично Z(70)',
            'определяло ложный большой пространственный сдвиг',
        ],
    })
    display(Markdown('### Какие изменения действительно повлияли на результат'))
    display(influence_table)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    axes[0, 0].plot(legacy_curve.L_mm, legacy_curve.Z_experiment_ohm,
                    'ko-', lw=2, label='эксперимент: независимый L=90 мм')
    axes[0, 0].plot(pre_rerun_comparison.L_mm, pre_rerun_comparison.Z_FEM_ohm,
                    's--', label='архив: L=100 мм вместо 90 мм')
    axes[0, 0].plot(legacy_curve.L_mm, legacy_curve.Z_FEM_ohm,
                    'd-', label='legacy_fast')
    axes[0, 0].plot(disjoint_curve.L_mm, disjoint_curve.Z_FEM_ohm,
                    'o-', label='disjoint')
    axes[0, 0].set(title='Измеренный и рассчитанный импеданс',
                   xlabel='L, мм', ylabel='Z, Ом')
    axes[0, 0].legend(fontsize=8)

    for hist_frame, label, color in [
        (legacy_hist, 'legacy_fast', '#2ca02c'),
        (disjoint_hist, 'disjoint', '#1f77b4'),
    ]:
        valid = hist_frame[hist_frame.objective < 1e11].sort_values('evaluation')
        axes[0, 1].plot(valid.evaluation, valid.objective, '.', color=color, alpha=0.35)
        axes[0, 1].plot(valid.evaluation,
                        np.minimum.accumulate(valid.objective.to_numpy(float)),
                        '-', color=color, lw=2, label=label)
    axes[0, 1].axvline(25, color='gray', ls='--', lw=1, label='25 оценок')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set(title='Ход поиска: линия показывает лучшее найденное J',
                   xlabel='номер FEM-оценки', ylabel='J, логарифмическая шкала')
    axes[0, 1].legend(fontsize=8)

    pose_rows = [
        ('архив', archive_fit, '#111111'),
        ('отклонённый fmincon', shifted_fit, '#d62728'),
        ('legacy_fast', legacy_fit, '#2ca02c'),
        ('disjoint', disjoint_fit, '#1f77b4'),
    ]
    for label, row, color in pose_rows:
        axes[1, 0].scatter(row.centre_u_mm, row.centre_v_mm, s=65, color=color)
        axes[1, 0].annotate(label, (row.centre_u_mm, row.centre_v_mm),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_aspect('equal', adjustable='datalim')
    axes[1, 0].set(title='Положение общего центра в касательной плоскости',
                   xlabel='u, мм', ylabel='v, мм')

    residual_x = np.arange(len(legacy_curve))
    axes[1, 1].bar(residual_x-0.2, legacy_curve.residual_ohm, width=0.4,
                   label='legacy_fast')
    axes[1, 1].bar(residual_x+0.2, disjoint_curve.residual_ohm, width=0.4,
                   label='disjoint')
    axes[1, 1].axhline(0, color='black', lw=0.8)
    axes[1, 1].set_xticks(residual_x, legacy_curve.L_mm.astype(int))
    axes[1, 1].set(title='Остатки: модель минус эксперимент',
                   xlabel='L, мм', ylabel='остаток, Ом')
    axes[1, 1].legend()

    pose_fig = go.Figure()
    pose_fig.add_trace(sampled_mesh_trace(
        triangles['body'], 45000, 'локальный участок STL тела', '#d8a56d', 0.48))
    centres_xyz = []
    for label, row, color in pose_rows:
        point = row[['centre_x_mm', 'centre_y_mm', 'centre_z_mm']].to_numpy(float)
        direction = row[['axis_x', 'axis_y', 'axis_z']].to_numpy(float)
        direction /= np.linalg.norm(direction)
        line = point + np.array([-70, 70])[:, None] * direction
        centres_xyz.append(point)
        pose_fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
            line=dict(color=color, width=7), name=f'{label}: ось'))
        pose_fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode='markers+text',
            marker=dict(size=7, color=color), text=[label], textposition='top center',
            name=f'{label}: центр'))
    local_mid = np.mean(np.vstack(centres_xyz), axis=0)
    pose_fig.update_layout(
        title='Архивная и повторно найденные оси непосредственно на CT/STL-поверхности',
        height=680, margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='x, мм', range=[local_mid[0]-95, local_mid[0]+95]),
            yaxis=dict(title='y, мм', range=[local_mid[1]-95, local_mid[1]+95]),
            zaxis=dict(title='z, мм', range=[local_mid[2]-95, local_mid[2]+95]),
            camera=dict(eye=dict(x=-1.5, y=-1.7, z=0.8),
                        projection=dict(type='orthographic'))))
    show_plotly(pose_fig)
    final_full_fit = disjoint_fit
    final_full_electrodes = restartable_electrodes['disjoint']
    archive_full_centre = archive_fit[
        ['centre_x_mm', 'centre_y_mm', 'centre_z_mm']].to_numpy(float)
    archive_full_axis = archive_fit[['axis_x', 'axis_y', 'axis_z']].to_numpy(float)
    archive_full_axis /= np.linalg.norm(archive_full_axis)
    archive_full_axis_points = (
        archive_full_centre + np.array([-90, 90])[:, None] * archive_full_axis)
    final_centre = final_full_fit[
        ['centre_x_mm', 'centre_y_mm', 'centre_z_mm']].to_numpy(float)
    final_axis = final_full_fit[['axis_x', 'axis_y', 'axis_z']].to_numpy(float)
    final_axis /= np.linalg.norm(final_axis)
    final_axis_points = final_centre + np.array([-90, 90])[:, None] * final_axis
    final_xyz = final_full_electrodes[
        ['patch_centroid_x_mm', 'patch_centroid_y_mm',
         'patch_centroid_z_mm']].to_numpy(float)

    full_pose_fig = go.Figure()
    full_pose_fig.add_trace(sampled_mesh_trace(
        triangles['body'], len(triangles['body']),
        'полная наружная STL-поверхность тела', '#d8a56d', 0.72))
    full_pose_fig.add_trace(go.Scatter3d(
        x=archive_full_axis_points[:, 0], y=archive_full_axis_points[:, 1],
        z=archive_full_axis_points[:, 2], mode='lines',
        line=dict(color='#111111', width=8, dash='dash'),
        name='старая архивная ось'))
    full_pose_fig.add_trace(go.Scatter3d(
        x=[archive_full_centre[0]], y=[archive_full_centre[1]],
        z=[archive_full_centre[2]], mode='markers+text',
        marker=dict(size=8, color='#111111', symbol='square'),
        text=['старая архивная координата'], textposition='bottom center',
        name='старая архивная координата'))
    full_pose_fig.add_trace(go.Scatter3d(
        x=final_xyz[:, 0], y=final_xyz[:, 1], z=final_xyz[:, 2],
        mode='markers',
        marker=dict(size=6, color=final_full_electrodes.L_mm,
                    colorscale='Turbo', cmin=50, cmax=140,
                    colorbar=dict(title='L, мм'),
                    line=dict(color='black', width=0.7)),
        text=[f'L={L:g} мм; {name}' for L, name in zip(
            final_full_electrodes.L_mm, final_full_electrodes.electrode)],
        hovertemplate='%{text}<br>x=%{x:.1f}; y=%{y:.1f}; z=%{z:.1f} мм<extra></extra>',
        name='фактические центроиды электродных площадок'))
    full_pose_fig.add_trace(go.Scatter3d(
        x=final_axis_points[:, 0], y=final_axis_points[:, 1],
        z=final_axis_points[:, 2], mode='lines',
        line=dict(color='magenta', width=12),
        name='итоговая ось disjoint'))
    full_pose_fig.add_trace(go.Scatter3d(
        x=[final_centre[0]], y=[final_centre[1]], z=[final_centre[2]],
        mode='markers+text', marker=dict(size=9, color='magenta', symbol='diamond'),
        text=['итоговая координата disjoint'], textposition='top center',
        name='итоговая координата disjoint'))
    full_pose_fig.update_layout(
        title=('Итоговое и архивное расположение на полной STL-поверхности '
               '(без пространственной обрезки)'),
        height=760, margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='x, мм', range=body_ranges[0]),
            yaxis=dict(title='y, мм', range=body_ranges[1]),
            zaxis=dict(title='z, мм', range=body_ranges[2]),
            camera=dict(eye=dict(x=-1.7, y=-1.4, z=0.7),
                        projection=dict(type='orthographic'))),
        legend=dict(orientation='h', yanchor='bottom', y=0.01,
                    xanchor='center', x=0.5))
    show_plotly(full_pose_fig)
    display(Markdown(
        'На этом рисунке показана вся наружная STL-поверхность, включая обе руки; '
        'диапазоны осей не обрезаны. Чёрная пунктирная линия и квадрат — старая '
        'архивная ось и координата. Пурпурная линия и ромб — итоговый результат '
        '`disjoint`. Цветные точки — фактические центроиды 36 '
        'электродных площадок для девяти размеров. Поскольку все сборки имеют общий '
        'центр и общую ось, точки разных размеров частично перекрываются. Рисунок '
        'можно вращать и увеличивать в HTML.'))

    archive_electrodes = pd.read_csv(pre_rerun_electrodes_file)
    electrode_sets = [
        ('архив', archive_electrodes, '#111111', 's'),
        ('legacy_fast', restartable_electrodes['legacy_fast'], '#2ca02c', 'd'),
        ('disjoint', restartable_electrodes['disjoint'], '#1f77b4', 'o'),
    ]
    fig, patch_axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    electrode_order = ['I_plus', 'V_plus', 'V_minus', 'I_minus']
    for patch_ax, L_value in zip(patch_axes, [50, 70]):
        for label, frame, color, marker in electrode_sets:
            subset = frame[frame.L_mm == L_value].copy()
            subset['order'] = pd.Categorical(subset.electrode, electrode_order, ordered=True)
            subset = subset.sort_values('order')
            patch_ax.plot(subset.patch_centroid_y_mm, subset.patch_centroid_z_mm,
                          marker=marker, color=color, lw=1.5, label=label)
            if label == 'disjoint':
                for _, electrode_row in subset.iterrows():
                    patch_ax.annotate(electrode_row.electrode,
                                      (electrode_row.patch_centroid_y_mm,
                                       electrode_row.patch_centroid_z_mm),
                                      xytext=(4, 4), textcoords='offset points', fontsize=8)
        patch_ax.set_aspect('equal', adjustable='datalim')
        patch_ax.set(title=f'Центроиды электродных площадок, L={L_value} мм',
                     xlabel='y, мм', ylabel='z, мм')
        patch_ax.legend(fontsize=8)

    display(Markdown(fr'''### Как читать итоговые графики

На графике хода поиска точки — все вычисленные допустимые положения, а сплошная линия —
наименьшее значение $J$, найденное к данному моменту. После 25 оценок `legacy_fast` уже имел
$J={legacy_j25:.4f}$, но для `disjoint` было найдено лишь $J={disjoint_j25:.4f}$. Итоговый
`disjoint`-минимум $J={disjoint_search['objective']:.4f}$ появился на оценке
**{disjoint_best_eval}**. Поэтому 25 точек действительно было недостаточно. `legacy_fast`
достиг разрешения заданного шага после **{legacy_search['function_evaluations']}** оценок;
`disjoint` проверен до **{disjoint_search['function_evaluations']}** оценок. Три его пробные
геометрии были отклонены, потому что связную площадку требуемой площади нельзя было построить
без пересечения с соседним электродом.

На графике остатков нулевая линия означает точное совпадение. Положительный столбец означает,
что FEM завышает импеданс, отрицательный — занижает. RMSE сворачивает эти девять остатков в
одно число, но не показывает, на каком размере возникло расхождение; поэтому кривую и остатки
нужно рассматривать вместе.

На трёхмерном рисунке показаны не абстрактные $(u,v)$, а вычисленные центры и оси на STL тела,
полученной по КТ. Большая красная ось — отклонённый результат `fmincon`. Две новые оси остаются
рядом с архивной: сдвиг центра `legacy_fast` равен **{legacy_shift:.3f} мм**, `disjoint` —
**{disjoint_shift:.3f} мм**. У `disjoint` угол отличается от архивного на
**{abs(float(disjoint_fit.phi_deg-archive_fit.phi_deg)):.3f}°**.

Два последних рисунка увеличивают сборки 50 и 70 мм. Они показывают центроиды площадок, а не
сами архивные треугольники: списки граней старого результата не сохранились. Перебор размеров
KD-списка, четырёх правил расстояния и всех 24 порядков обработки электродов не устранил два
остаточных расхождения `V_minus`. Поэтому точное происхождение этих двух архивных площадок
остаётся нерешённым вопросом.

**Вывод расчётной модели.** Большой уход центра после перепрогона был численным следствием
неподходящего градиентного поиска по кусочно-постоянной дискретной геометрии. Он не подтверждён
ни меньшим $J$, ни расширенным прямым поиском. При текущих допущениях устойчивая область центра
лежит рядом с архивной, но различие угла и $\rho_2$ между двумя способами построения контакта
показывает сохраняющуюся модельную неопределённость. Полученные точки являются локальными
вычислительными кандидатами, а не доказательством единственности или физической истинности
решения.
'''))
else:
    display(Markdown(
        '**Расширенное сравнение пропущено:** отсутствует один или несколько расчётных CSV/JSON.'))
""")


md(r"""
Обязательный следующий результат должен быть не одной точкой, а графиками
$J_{min}(\rho_{2,in})$ и $J_{min}(q_{ex})$ с бутстрэп-оценкой по временным окнам и отдельным сборкам.
""")

# The former section 18 was a migration audit, not a second scientific result.
# Its evidence remains in output/ and in docs/PIPELINE_FOR_NOTEBOOKS.md.  Keep
# the reader-facing notebook focused on the canonical disjoint calculation;
# the necessary archive/current coordinate comparison is now shown in section 3.
for cell_index, cell in enumerate(cells):
    marker = '\n## 18. Контроль положения электродных сборок'
    if cell.cell_type == 'markdown' and marker in cell.source:
        cell.source = cell.source.split(marker, 1)[0].rstrip()
        if cell_index + 1 < len(cells) and cells[cell_index + 1].cell_type == 'code':
            del cells[cell_index + 1]
        break

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
TARGET.write_text(nbf.writes(nb), encoding="utf-8")
print(TARGET)
