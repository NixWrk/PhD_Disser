from pathlib import Path
import hashlib
import json
import nbformat
import os
import runpy
import shutil

PROJECT = Path(__file__).resolve().parents[3]
MODEL = PROJECT / 'MATLAB_TRKG4_real_subjects'
OUT = Path(__file__).resolve().parent
NAME = '20.10_КТ_FEM_локализация_боковых_матриц_МГТУ'
DEST = PROJECT / 'Colab Notebooks' / (NAME + '.ipynb')
OLD = MODEL / 'notebooks/TRKG4_inverse_inhale_results.ipynb'


def change(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


# Freeze the precise producer of the already computed layered comparison.
comparison_dir = MODEL / 'output/two_tissue_auditfix_20260908'
comparison = json.loads((comparison_dir / 'comparison_summary.json').read_text(encoding='utf-8'))
producer = MODEL / 'tools/compare_curved_two_tissue.py'
producer_hash = hashlib.sha256(producer.read_bytes()).hexdigest()
assert producer_hash == comparison['source_sha256']
source_used = comparison_dir / 'source_used' / f'compare_curved_two_tissue_{producer_hash}.py'
source_used.parent.mkdir(exist_ok=True)
assert not source_used.exists()
shutil.copy2(producer, source_used)

source = producer.read_text(encoding='utf-8')
source = change(source, "NB = ROOT / 'notebooks/TRKG4_inverse_inhale_results.ipynb'",
                f"NB = ROOT.parent / 'Colab Notebooks' / '{NAME}.ipynb'")
source = change(source, "for file in [NB, NB.with_name('build_trkg4_inverse_report.py')]:",
                "for file in [NB, ROOT / 'notebooks/build_trkg4_inverse_report.py']:")
producer.write_text(source, encoding='utf-8')

builder = MODEL / 'notebooks/build_trkg4_inverse_report.py'
source = builder.read_text(encoding='utf-8')
source = change(source, 'ROOT / "notebooks" / "TRKG4_inverse_inhale_results.ipynb"',
                f'ROOT.parent / "Colab Notebooks" / "{NAME}.ipynb"')
old_root = "ROOT = Path.cwd()\nif not (ROOT / 'output').is_dir():\n    ROOT = ROOT.parent"
new_root = """_working_dir = Path.cwd().resolve()
_model_candidates = (
    _working_dir / 'MATLAB_TRKG4_real_subjects',
    _working_dir.parent / 'MATLAB_TRKG4_real_subjects',
    _working_dir,
    _working_dir.parent,
)
ROOT = next((p for p in _model_candidates if (p / 'output').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Не найден расчётный каталог MATLAB_TRKG4_real_subjects/output')"""
source = change(source, old_root, new_root)
old_check = "    assert hashlib.sha256((ROOT / 'tools/compare_curved_two_tissue.py').read_bytes()).hexdigest() == two['source_sha256']"
new_check = """    two_current_source = ROOT / 'tools/compare_curved_two_tissue.py'
    two_current_hash = hashlib.sha256(two_current_source.read_bytes()).hexdigest()
    two_source_snapshot = two_dir / 'source_used' / f"compare_curved_two_tissue_{two['source_sha256']}.py"
    two_verified_source = (two_current_source if two_current_hash == two['source_sha256']
                           else two_source_snapshot)
    assert two_verified_source.is_file(), 'Не найдена версия кода сохранённого расчёта'
    assert hashlib.sha256(two_verified_source.read_bytes()).hexdigest() == two['source_sha256']
    if two_verified_source != two_current_source:
        display(Markdown('**Происхождение сравнения:** рабочий скрипт изменён после расчёта; '
                         'его сохранённая версия проверена по исходной контрольной сумме. '
                         'Эта проверка не является повторным расчётом.'))"""
source = change(source, old_check, new_check)
navigation = """**Место в последовательности:** [карта проекта](00.00_Карта_проекта.md) →
[20.01–20.04: КТ и исходные свойства](20.10_КТ_FEM_локализация_боковых_матриц_МГТУ.md#входы-и-происхождение) → **20.10: данный отчёт** →
[20.11: чувствительность](20.11_КТ_FEM_чувствительность_боковых_сборок.ipynb) →
[20.12: применимость упрощённых моделей](20.12_Критерий_Lmax_плоскослоистой_модели.ipynb) →
[32: выбор размеров](32.02_Выбор_пары_при_известной_h.ipynb).
Аналитическая основа: [30.01](30.01_Прямая_двуслойная_модель_боковой_сборки.md)
и [30.04](30.04_Вычислительное_ядро_двуслойной_модели.ipynb).
[Паспорт этапа](20.10_КТ_FEM_локализация_боковых_матриц_МГТУ.md) определяет входы,
выходы и статус передачи результата. Нумерация задаёт порядок исследования;
численные результаты не становятся принятыми входами следующих этапов автоматически.
"""
insertion = """# Reader links are relative to the canonical notebook in Colab Notebooks.
for cell in cells:
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace('../../Colab%20Notebooks/', '')
        cell.source = cell.source.replace('../docs/PIPELINE_FOR_NOTEBOOKS.md',
                                          '../MATLAB_TRKG4_real_subjects/docs/PIPELINE_FOR_NOTEBOOKS.md')
cells[0].source = cells[0].source.replace('# ', '# 20.10 ', 1)
_heading, _rest = cells[0].source.split('\\n', 1)
cells[0].source = _heading + '\\n\\n' + """ + repr(navigation) + " + '\\n' + _rest.lstrip('\n')\n\n"
source = change(source, 'nb["cells"] = cells\n', insertion + 'nb["cells"] = cells\n')
source = change(source, 'TARGET.write_text(nbf.writes(nb), encoding="utf-8")',
                "nb['metadata']['kalmykov_corpus'] = {'id': '20.10', 'status': 'numerical_candidate_not_validated', 'canonical_owner': 'Colab Notebooks', 'experiment': 2, 'outputs_in_git': False}\nTARGET.write_text(nbf.writes(nb), encoding=\"utf-8\")")
builder.write_text(source, encoding='utf-8')

updater = MODEL / 'notebooks/update_scientific_report.py'
source = updater.read_text(encoding='utf-8')
source = change(source, "TARGET = ROOT / 'notebooks/TRKG4_inverse_inhale_results.ipynb'",
                f"TARGET = ROOT.parent / 'Colab Notebooks' / '{NAME}.ipynb'")
source = change(source, 'str(p.relative_to(ROOT)): digest(p)', 'str(p.relative_to(ROOT.parent)): digest(p)')
updater.write_text(source, encoding='utf-8')

# Generate the destination structure, without executing any notebook cell.
os.environ['TRKG4_REPORT_TARGET'] = str(OUT / 'generated.ipynb')
namespace = runpy.run_path(str(builder))
del os.environ['TRKG4_REPORT_TARGET']
generated = nbformat.read(OUT / 'generated.ipynb', 4)
original = nbformat.read(OLD, 4)
assert len(generated.cells) == len(original.cells) == 56
code_mapping = []
for i, (before, after) in enumerate(zip(original.cells, generated.cells)):
    assert before.cell_type == after.cell_type
    if before.cell_type != 'code':
        continue
    expected = before.source
    if i == 3:
        expected = change(expected, old_root, new_root)
    elif i == 32:
        expected = change(expected, old_check, new_check)
    assert expected == after.source, f'Unexpected computational change in cell {i}'
    after.execution_count = before.execution_count
    after.outputs = before.outputs
    after.metadata = before.metadata
    after.id = before.id
    code_mapping.append({'index': i, 'id': before.id, 'change': 'root discovery' if i == 3 else 'source provenance validation' if i == 32 else 'none'})
    if i == 32:
        after.outputs.insert(0, nbformat.v4.new_output('display_data', data={'text/markdown':
            '**Происхождение сравнения:** рабочий скрипт изменён после расчёта; '
            'его сохранённая версия проверена по исходной контрольной сумме. '
            'Эта проверка не является повторным расчётом.'}))
generated.metadata['corpus_integration'] = {'date': '2026-09-09', 'source_sha256': hashlib.sha256(OLD.read_bytes()).hexdigest(),
    'scope': 'path resolution, navigation and preserved producer provenance; numerical outputs retained', 'execution_performed': False}
nbformat.validate(generated)
assert DEST.resolve().is_relative_to(PROJECT) and OLD.resolve().is_relative_to(PROJECT)
assert not DEST.exists()
OLD.rename(DEST)
nbformat.write(generated, DEST)

# Keep old incoming links usable without maintaining a second executable report.
pointer = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
    '# Отчёт включён в основную последовательность\n\n'
    f'Полный ноутбук перенесён в [20.10 — КТ/FEM-локализация и сравнение моделей боковых сборок](../../Colab%20Notebooks/{NAME}.ipynb).\n\n'
    f'[Читательская HTML-версия](../../Colab%20Notebooks/{NAME}.html).\n\n'
    'Этот файл служит указателем для прежних ссылок. Вычислений и результатов здесь нет. '
    'Генератор и MATLAB/Python-ядро остаются в прежней вычислительной ветке.')],
    metadata={'kalmykov_corpus': {'status': 'moved_pointer_only', 'canonical_notebook': f'../../Colab Notebooks/{NAME}.ipynb'}})
nbformat.write(pointer, OLD)
OLD.with_suffix('.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Отчёт 20.10 перенесён</title>'
    f'<p>Полный отчёт находится в <a href="../../Colab%20Notebooks/{NAME}.html">основной последовательности: 20.10</a>.</p></html>', encoding='utf-8')
(OUT / 'code_cell_mapping.json').write_text(json.dumps(code_mapping, ensure_ascii=False, indent=2), encoding='utf-8')
print(DEST)
