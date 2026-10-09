"""Preserve the completed first pass without repeating its numerical calculation."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from lateral_pilot import ROOT, read, sha


def build():
    folder = Path(__file__).parent
    path = next(folder.glob('33.20*.ipynb'))
    out = ROOT / 'Experiments/scorpion_v2_georg/derived/exploratory/33.20_lateral_pilot'
    manifest = read(out / 'manifest.local.json')
    if manifest['code_sha256'] != sha(folder / 'lateral_pilot.py'):
        raise ValueError('The saved first pass belongs to a different numerical module')
    nb = nbformat.read(path, 4)
    nb.cells = [c for c in nb.cells if c.cell_type == 'markdown' and not c.source.startswith('## Сохранённые результаты')]
    for cell in nb.cells:
        if cell.source.startswith('# 33.20.'):
            history = 'Исторический первый проход до уточнения каналов от 09.10.2026. Прежняя постоянная нумерация не соответствует сообщённому переходу 004–007. Рабочий пересчёт находится в [33.21](33.21_Множество_решений_двуслойной_модели_Scorpion_v2.html); результаты этого первого прохода не передаются в коррекцию ТТРКГ.'
            old_history = 'Сохранённый первый проход. Его литературные ограничения не используются в расширенной проверке 33.21 и индивидуальной КТ/FEM-инверсии 33.22.'
            cell.source = cell.source.replace('\n\n'+old_history, '').replace('\n\n'+history, '')
            marker = '**Статус: разработка методики; исследовательские сценарии без независимой валидации.**'
            cell.source = cell.source.replace(marker, marker+'\n\n'+history)
        cell.source = cell.source.replace(
            'Роли импедансных каналов постоянны во всём горизонтальном ряду, но их числовое назначение не восстановлено: расчёт сохраняет два общих сценария.',
            'В этой исторической постановке принималась постоянная нумерация ролей во всём горизонтальном ряду и сохранялись два общих сценария. Уточнение автора от 09.10.2026 заменило это допущение; исходные результаты ниже сохранены для сопоставления.')
        cell.source = cell.source.replace(
            'Полный индивидуальный результат находится только в исключённом из Git `Experiments/scorpion_v2_georg/derived/exploratory/33.20_lateral_pilot/`.',
            'Сохранённые таблицы и графики приведены ниже. Полные производные массивы находятся в исключённом из Git `Experiments/scorpion_v2_georg/derived/exploratory/33.20_lateral_pilot/`.')
    nb.cells.append(nbformat.v4.new_markdown_cell('## Сохранённые результаты первого прохода\n\nНиже воспроизведён завершённый расчёт с его исходными допущениями. Численные результаты не пересчитываются при формировании этого представления.'))
    nb.cells.append(nbformat.v4.new_code_cell('''from pathlib import Path
from IPython.display import HTML, display
from bs4 import BeautifulSoup
from plotly.offline import get_plotlyjs
from lateral_pilot import ROOT, read, sha
import re

out = ROOT / "Experiments/scorpion_v2_georg/derived/exploratory/33.20_lateral_pilot"
manifest = read(out / "manifest.local.json")
if manifest["code_sha256"] != sha(ROOT / "Colab Notebooks/scorpion_v2/lateral_pilot.py"):
    raise ValueError("Изменилась версия вычислительного модуля первого прохода")
page = BeautifulSoup((out / "index.html").read_text(encoding="utf-8"), "html.parser")
headers = {
    "record_id": "Запись", "montage": "Монтаж", "state": "Дыхание", "channel": "Канал",
    "start_s": "Начало, с", "end_s": "Конец, с",
    "baseline_raw_median_ohm": "Медиана общего BEI, Ом",
    "slow_p05_p95_span_ohm": "Разность 95-го и 5-го процентилей медленной кривой, Ом",
    "second_minus_first_half_slow_ohm": "Разность второй и первой половин медленной кривой, Ом",
    "n_candidate_cycles": "Число кандидатных циклов",
    "median_curve_ptp_mohm": "Размах медианной кривой, мОм",
    "межцикловое SD, мОм": "Межцикловое стандартное отклонение, мОм",
    "h сетки, мм": "Заданная h, мм", "граница ρ": "Достигнута граница ρ",
    "край сетки h": "Достигнута граница h", "при δρ₁=0, %": "Пульсовая невязка при δρ₁=0, %"
}
for element in page.select("th"):
    if element.get_text(strip=True) in headers:
        element.string = headers[element.get_text(strip=True)]
for element in page.select("td"):
    value = element.get_text(strip=True)
    value = {"inhale": "Вдох", "exhale": "Выдох", "True": "Да", "False": "Нет", "NaN": "Не определено"}.get(value, value)
    value = re.sub(r"^SCV2-G-(\\d+)$", r"Запись \\1", value)
    if re.fullmatch(r"-?\\d+\\.\\d+", value):
        value = value.replace(".", ",").replace("-", "−")
    element.string = value
for element in page.select("table"):
    element.wrap(page.new_tag("div", attrs={"style": "overflow-x:auto"}))
start = page.find("h2")
content = str(start) + "".join(str(node) for node in start.next_siblings)
content = content.replace("inhale", "вдох").replace("exhale", "выдох")
content = content.replace("RMSE, Ом", "Среднеквадратичная невязка, Ом")
content = content.replace("Подтверждённое автором постоянство ролей сохраняется: перестановка каналов между записями не вводится ради подгонки.", "В этой исторической редакции принималась постоянная нумерация ролей. Уточнение автора от 09.10.2026 заменило это допущение; исправленный размерный ряд используется в 33.21.")
display(HTML("<script>" + get_plotlyjs() + "</script>" + content))'''))
    nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
    NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(folder)}}).execute()
    nbformat.write(nb, path)
    exporter = HTMLExporter()
    exporter.exclude_input = exporter.exclude_input_prompt = exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(nb)
    path.with_suffix('.html').write_text(html, encoding='utf-8')
    path.with_suffix('.md').write_text('\n\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')+'\n', encoding='utf-8')


if __name__ == '__main__':
    build()
