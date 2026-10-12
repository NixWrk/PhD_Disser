"""Concise reader export from the executed 10.00 notebook, without recomputation."""
from pathlib import Path
import re
import json
import nbformat
from nbconvert import HTMLExporter
from lateral_pilot import save, sha
from scorpion_comparisons import OUT, verify

FOLDER=Path(__file__).parent
STEM='10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2'


def build():
    verify()
    source=FOLDER/(STEM+'.ipynb')
    notebook=nbformat.read(source,4)
    cells=[nbformat.v4.new_markdown_cell('''# Scorpion v2: сигналы и электродные сборки Георгия

МГТУ, 05.10.2026. Частота зондирования — 100 кГц.

При 140 мм подтверждены ТТРКГ 1×1, 2×2 и 1×4. В ТЕПС/АЕПС
канал 1 — ТТРКГ 1×1; канал 2 — соответствующая сборка.
Для 007–010 вариант 1×1 остаётся гипотезой; можно сравнить его с 1×4.
Координаты ТЕПС реконструированы автором, остальные монтажи показаны
номинально. При смене записи предусмотрено сохранение ракурса.

Обзорные линии сохраняют экстремумы 900 временных интервалов;
пятисекундная ЭКГ показана без прореживания. Полные отсчёты доступны
в основном отчёте. Спектры и медианы рассчитаны по полным массивам.

Дыхательные границы и события ЭКГ — кандидатная разметка. Медианы
показывают общий BEI в выбранных задержках на глубоком вдохе и выдохе;
точность этих интервалов ещё проверяется. Это этап разработки методики.''')]
    widget=next(c for c in notebook.cells if c.cell_type=='code' and c.source=='display(HTML(widget()))')
    widget=nbformat.from_dict(dict(widget))
    for output in widget.outputs:
        if 'text/html' in output.get('data',{}):
            content=output.data['text/html']
            marker='const experimentPayload='
            first=content.index(marker)+len(marker)
            payload,end=json.JSONDecoder().raw_decode(content[first:])
            for record in payload['records']:
                record['original_sample_rows']=record['full_samples']['rows']
                del record['full_samples']
            content=content[:first]+json.dumps(payload,separators=(',',':'),ensure_ascii=False)+content[first+end:]
            content=re.sub(r'(<select id="experiment-sampling">).*?(</select>)',
                r'\1<option value="extrema">Краткий обзор — минимумы и максимумы 900 интервалов</option>\2',content,flags=re.S)
            content=re.sub(r'(<select id="experiment-channel">).*?(</select>)',
                r'\1<option value="0">Текущее назначение каналов</option>\2',content,flags=re.S)
            content=re.sub(r'<fieldset><legend>([^<]+)</legend>(.*?)</fieldset>',
                r'<details><summary>\1</summary><fieldset>\2</fieldset></details>',content,flags=re.S)
            output.data['text/html']=content
    cells.append(widget)
    for group,title in [('ttrkg','Сравнение ТТРКГ'),('lateral','Сравнение боковых сборок')]:
        cells.append(nbformat.v4.new_markdown_cell('## '+title+'\n\nСлева — базовые кривые, справа и в таблице — медианы всех исходных отсчётов. Время отсчитывается отдельно для каждой записи; знак «?» обозначает неподтверждённый вариант сборки.'))
        cells.append(next(c for c in notebook.cells if c.cell_type=='code' and f"comparison_figure('{group}')" in c.source))
    cells.append(nbformat.v4.new_markdown_cell('''## Сопоставление ЭКГ

Слева — шкала мВ, справа — нормированная форма медианного цикла.
ЭКГ поддерживает проверку подключения; однозначно установить 1×1
или 1×4 в 007–010 по имеющимся сигналам пока не удалось.

[Полный отчёт с описанием обработки](10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2.html).'''))
    cells.append(next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('display(HTML(ecg_figure()')))
    compact=nbformat.v4.new_notebook(cells=cells,metadata=notebook.metadata)
    exporter=HTMLExporter()
    exporter.exclude_input=exporter.exclude_input_prompt=exporter.exclude_output_prompt=True
    page,_=exporter.from_notebook_node(compact)
    target=FOLDER/(STEM+'_для_руководителя.html')
    target.write_text(page,encoding='utf-8')
    save(OUT/'supervisor_export.local.json',dict(source_notebook_sha256=sha(source),
        html_sha256=sha(target),code_excluded=True,numerical_recomputation=False,
        input=verify()['outputs'],producer_sha256=sha(__file__)))
    print(target)
    return target


if __name__=='__main__':build()
