"""Render the saved model-preparation audit. This module never runs FEM."""
from pathlib import Path
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / 'MATLAB_TRKG4_real_subjects'
FOLDER = Path(__file__).parent
OUT = PIPE / 'output/exploratory/georg_model_audit_20261013'
STEM = '20.22_Аудит_подготовки_КТ_FEM_модели_Георгия_Scorpion_v2'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scientific_number(value):
    mantissa, separator, exponent = f'{value:.6g}'.partition('e')
    mantissa = mantissa.replace('.', ',')
    if not separator:
        return mantissa
    superscript = str(int(exponent)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return mantissa + '·10' + superscript


def load_evidence():
    reference = read(OUT / 'shared_reference_inventory.local.json')
    pointer = PIPE / 'output/georg_current_model.local.json'
    if sha(pointer) != reference['pointer_sha256']:
        raise ValueError('The shared reference changed after the audit inventory.')
    for item in reference['checked_files']:
        if not item['match'] or sha(PIPE / item['path']) != item['actual_sha256']:
            raise ValueError('An audited reference file changed: ' + item['path'])
    lineage = read(OUT / 'mesh_lineage.local.json')
    for stage in lineage['stages']:
        if sha(PIPE / stage['plan_relative_path']) != stage['plan_sha256']:
            raise ValueError('An audited preparation plan changed.')
    return reference, lineage


def closed_table(title, columns, rows):
    from IPython.display import HTML
    header = ''.join('<th>' + html.escape(str(x)) + '</th>' for x in columns)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>'
                                  for x in row) + '</tr>' for row in rows)
    return HTML('<details><summary>' + html.escape(title) + '</summary>'
                '<div class="audit-table"><table><thead><tr>' + header +
                '</tr></thead><tbody>' + body + '</tbody></table></div></details>')


def reference_table():
    reference, _ = load_evidence()
    return closed_table('Таблица 1. Происхождение проверенной общей модели',
        ['Проверка', 'Результат'], [
            ['Закреплённые файлы', len(reference['checked_files'])],
            ['Совпадение контрольных сумм', sum(x['match'] for x in reference['checked_files'])],
            ['Независимая точность анатомии', 'Не подтверждена'],
            ['Сеточная сходимость', 'Не подтверждена'],
            ['Количественная инверсия в строгом режиме', 'Не разрешена текущим численным статусом'],
            ['Изменение общей модели при аудите', 'Не выполнялось'],
        ])


def lineage_table():
    _, lineage = load_evidence()
    rows = []
    for stage in lineage['stages']:
        volume = stage['minimum_volume_mm3']
        rows.append([stage['label'], f"{stage['nodes']:,}".replace(',', ' '),
            f"{stage['tetrahedra']:,}".replace(',', ' '),
            'Не указан' if volume is None else scientific_number(volume),
            'Не применяется' if stage['quadrature_points'] is None else
            ' / '.join(str(x) for x in stage['quadrature_points'])])
    return closed_table('Таблица 2. Сохранённые этапы построения и уточнения сетки',
        ['Этап', 'Узлы', 'Тетраэдры', 'Минимальный объём по паспорту, мм³',
         'Числа точек интегрирования долей тканей'], rows)


def build():
    import nbformat as nbf
    from nbconvert import HTMLExporter
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    text = (OUT / 'reader_text.local.md').read_text(encoding='utf-8')
    intro, body = text.split('\n<!-- evidence tables -->\n', 1)
    cells = [nbf.v4.new_markdown_cell(intro),
             nbf.v4.new_code_cell('from IPython.display import display\n'
                 'from build_georg_model_preparation_audit import reference_table, lineage_table\n'
                 'display(reference_table())\ndisplay(lineage_table())'),
             nbf.v4.new_markdown_cell(body)]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'audit_scope': 'saved model preparation, read-only; no new FEM',
        'execution_mode': 'exact reader cell executed in-process with IPython'})
    shell = InteractiveShell.instance()
    with capture_output(display=True) as captured:
        result = shell.run_cell(cells[1].source, store_history=False)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError('Audit reader cell failed.')
    if len(captured.outputs) != 2:
        raise RuntimeError('Audit tables were not rendered.')
    cells[1].execution_count = 1
    cells[1].outputs = [nbf.v4.new_output('display_data', data=x.data,
                         metadata=x.metadata) for x in captured.outputs]
    notebook.cells = cells
    nbf.write(notebook, FOLDER / (STEM + '.ipynb'))
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    reader, _ = exporter.from_notebook_node(notebook)
    reader = reader.replace('</head>', '<style>.audit-table{overflow-x:auto}'
        'details{margin:1em 0}summary{cursor:pointer;font-weight:600}'
        'table{white-space:normal}th,td{vertical-align:top}</style></head>')
    (FOLDER / (STEM + '.html')).write_text(reader, encoding='utf-8')
    (FOLDER / (STEM + '.md')).write_text(text.replace('\n<!-- evidence tables -->\n', '\n'), encoding='utf-8')
    nbf.validate(notebook)
    print(json.dumps({'notebook': STEM, 'executed_cells': 1,
                      'new_FEM_executed': False, 'reference_files_checked': 91}))


if __name__ == '__main__':
    build()
