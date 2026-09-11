"""Clarify reference, electrode ranking and author-required physiological domain.

This is an editorial update, not a constrained inversion or a new FEM run.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import ast
import hashlib
import json
import runpy
import shutil

import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

REV = Path(__file__).resolve().parent
OUT = REV.parent
ROOT = OUT.parents[2]
PROJECT = ROOT.parent
GEN = ROOT/'notebooks/build_geometry_reduction_report.py'
NB = next((PROJECT/'Colab Notebooks').glob('20.13*.ipynb'))
BEFORE = REV/'before'
BEFORE.mkdir(exist_ok=True)
for p in [GEN,NB,NB.with_suffix('.html'),NB.with_suffix('.md'),
          OUT/'language_review.json',OUT/'completion_checks.json',
          OUT/'final_artifact_manifest.json',OUT/'reader_visible_text.txt']:
    dest=BEFORE/p.name
    if not dest.exists():shutil.copy2(p,dest)

source=(BEFORE/GEN.name).read_text(encoding='utf-8')
calls=[n.value for n in ast.parse(source).body if isinstance(n,ast.Expr)
       and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
       and n.value.func.id=='md']
texts=[ast.literal_eval(c.args[0]).strip() for c in calls]
assert len(texts)==14
additions={}

additions[4]=r'''

**Выбор по разделу 5.** CEM здесь служит расчётной опорой для конечных
контактов; её нельзя объявить экспериментальным победителем по сравнению
с собственным результатом. Среди двух точечных приближений ближе к CEM
вариант у центроидов площадок: среднеквадратичное расхождение по девяти
размерам составляет 5,596 Ом против 8,758 Ом у точек в запрошенных
координатах. Максимальные абсолютные расхождения равны соответственно
13,000 и 21,805 Ом. Следовательно, при выборе точечного приближения
для данного монтажа предпочтителен вариант у центроидов по этой метрике.
Его достаточная точность для восстановления тканей этим не установлена.
Раздел 5 сравнивает электродные условия при одной КТ-маске; выбор
геометрического упрощения лёгкого выполняется в других разделах.
'''

additions[5]=r'''

**Смысл нуля в однородном пределе.** При $q=1$ оба материала имеют
одинаковое сопротивление, поэтому перенос внутренней границы не меняет
пространственный оператор с общей наружной поверхностью и электродами.
Нулевое различие четырёх пространственных моделей в этом тесте не
означает нулевую ошибку плоской модели при неоднородных тканях.
'''

phys=r'''
### 8.4. Физиологическая допустимость как условие выбора

**Подтверждённое требование автора.** Основной интерес представляет
восстановление физиологически допустимых $\rho_1$ и $\rho_2$. Хорошая
подгонка импеданса при недопустимых сопротивлениях не делает модель
кандидатом для практической методики. Расчёты вне принятой области
сохраняются как диагностика несостоятельности упрощения или поведения
обратной задачи; они не участвуют в выборе пригодной модели наравне
с допустимыми решениями.

Проверка должна охватывать и сопротивления, заданные при генерации
основных синтетических данных, и сопротивления, полученные инверсией.
Сначала задаётся допустимая область в Ом·м для соответствующих частоты,
состояния дыхания и принятой классификации тканей. Внутри неё сравниваются
ошибки обоих параметров, остаток импеданса, прогноз и устойчивость.
Попадание в физиологическую область является необходимым, но само по себе
недостаточным условием: физиологически правдоподобная оценка также может
существенно отличаться от известного сопротивления данного объекта.

В текущем опыте численные физиологические границы для двух эффективных
материалов не установлены. Интервал контраста 0,5–32 — техническая
область поиска; он не определяет допустимость тканей. Номинальная пара
5,55217 и 30,70695 Ом·м также не является независимо подтверждённым
физиологическим эталоном. Поэтому сохранённые строки пока имеют
неоценённую физиологическую допустимость. Например, значению
65,426 Ом·м нельзя приписать статус допустимого или недопустимого
только по величине ошибки относительно номинального числа.

Для $\rho_1$ границы должны относиться ко всей эффективной нелёгочной
группе, включающей кости и поддиафрагмальные ткани, согласно постановке
автора. Диапазон одной мышцы нельзя автоматически переносить на эту
группу. Принятое объединение не отменяет требования физиологической
допустимости; оно требует обоснования диапазона именно эффективного
параметра. Согласование этих границ предшествует новому основному
ранжированию моделей.
'''
additions[7]=phys

zero=r'''
**Какая линия имеет почти нулевую ошибку.** На рисунке 5 это синяя
кривая «КТ: исходная граница лёгкого», а не плоская модель. Серая
горизонталь на нуле — ориентир идеального восстановления. Ошибка каждого
параметра вычисляется относительно сопротивления, заданного при создании
синтетических данных, по формуле раздела 1. Данные создаёт исходная
КТ-модель; затем её же геометрия используется в контрольной инверсии
через интерполяционную библиотеку. Поэтому близость синей кривой к нулю
проверяет численную согласованность и не является независимым доказательством
преимущества КТ-модели на реальных измерениях.

В номинальном случае ошибка $\rho_2$ равна примерно +0,00013% у КТ,
−64,75% у плоской границы внутри тела и −1,81% у плоской аналитической
модели с постоянной глубиной. Последняя в рисунок 5 не включена;
она представлена в таблице раздела 7. Сравнение не использует плоскую
модель в качестве источника синтетических наблюдений.

'''
texts[8]=texts[8].replace('**Результат модели.** Для плоской пространственной границы ошибка',zero+'**Результат модели.** Для плоской пространственной границы ошибка',1)
assert zero in texts[8]

additions[8]=r'''

### 9.4. Как применять чувствительность в физиологической области

Чувствительность и обусловленность следует использовать для основного
сравнения при допустимых сопротивлениях. Небольшое число обусловленности
вне этой области характеризует только математическое поведение модели.
Оно не оправдывает недопустимую оценку и не повышает её практическую
ценность. Таблица этого раздела остаётся диагностикой прежних
неограниченных физиологическими границами решений; новым ранжированием
допустимых моделей она не является.

После задания положительных границ $\rho_j^{\min},\rho_j^{\max}$
инверсия должна выполняться внутри них. При фиксированном $q>0$
допустимый интервал для масштаба имеет вид

$$\rho_1\in[\ell(q),u(q)],\qquad
\ell(q)=\max\left(\rho_1^{\min},\frac{\rho_2^{\min}}q\right),\qquad
u(q)=\min\left(\rho_1^{\max},\frac{\rho_2^{\max}}q\right),$$
где $\ell,u$ выражаются в Ом·м и учитывают одновременно границы
обоих материалов, поскольку $\rho_2=q\rho_1$. При $\ell>u$ данный
контраст недопустим. При непустом интервале решение для масштаба из
раздела 7 ограничивается этим интервалом **внутри каждого шага поиска**,
после чего заново вычисляется остаток и выбирается контраст. Простая
обрезка двух готовых оценок после прежнего поиска такой задачи не решает.

Это спецификация следующей ограниченной инверсии, не описание уже
выполненного пересчёта. Приведённая формула предполагает, что принятая
область задаётся независимыми интервалами двух сопротивлений. Если
потребуется совместное физиологическое ограничение, его нужно включить
отдельно. Выход решения на принятую границу фиксируется: он может
указывать на недостаточность модели или ограничений и не доказывает,
что ткань действительно имеет граничное сопротивление.
'''

additions[12]=r'''

**Приоритет дальнейшего выбора после уточнения автора.** Допустимая
область сопротивлений должна быть задана до практического ранжирования.
Для каждого кандидата нужны результаты инверсии в этой области и
проверка, сохраняется ли приемлемое совпадение импедансов при допустимых
параметрах. Если совпадение достигается лишь за её пределами, кандидат
не проходит этот критерий в проверенных условиях. Прежнее уменьшение
остатка у эллипсоида характеризует математическую компенсацию ошибки
геометрии; само по себе оно не является основанием рекомендовать
эллипсоид для оценки физиологических сопротивлений.
'''

for i,addition in additions.items():texts[i]+='\n'+addition.strip()+'\n'
lines=source.splitlines(keepends=True)
for i in reversed(range(len(calls))):
    c=calls[i]
    lines[c.lineno-1:c.end_lineno]=["md(r'''\n"+texts[i]+"\n''')\n"]
updated=''.join(lines)
ast.parse(updated)
GEN.write_text(updated,encoding='utf-8')
captured=[]
with patch('nbformat.write',side_effect=lambda obj,*a,**kw:captured.append(obj)):
    runpy.run_path(str(GEN),run_name='__main__')
old=nbformat.read(BEFORE/NB.name,as_version=4)
new=captured[0]
for a,b in zip(old.cells,new.cells):
    assert a.cell_type==b.cell_type
    if a.cell_type=='code':
        assert a.source==b.source
        b.clear();b.update(deepcopy(a))
    else:b.id=a.id;b.metadata=deepcopy(a.metadata)
new.metadata=deepcopy(old.metadata)
nbformat.validate(new)
nbformat.write(new,NB)

exp=HTMLExporter(template_name='lab')
exp.exclude_input=True;exp.exclude_input_prompt=True;exp.exclude_output_prompt=True
html,_=exp.from_notebook_node(new)
NB.with_suffix('.html').write_text(html,encoding='utf-8')
soup=BeautifulSoup(html,'html.parser');body=soup.select_one('.jp-Notebook')
assert not soup.select('.jp-CodeCell .jp-InputArea')
assert len(body.find_all('table'))==12
assert len(body.find_all('img'))==7
assert [dict(c) for c in old.cells if c.cell_type=='code']==[dict(c) for c in new.cells if c.cell_type=='code']
for e in body.find_all(['style','script']):e.decompose()
reader=body.get_text('\n',strip=True)
(REV/'reader_visible_text.txt').write_text(reader,encoding='utf-8')
(REV/'added_reader_text.md').write_text('\n\n'.join([*additions.values(),zero]),encoding='utf-8')

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'status':'editorial_checks_passed_review_pending','code_cells_executed':0,
        'code_cells_and_outputs_preserved':12,'figures_unchanged':7,'html_tables':12,
        'physiological_admissibility_required_by_author':True,
        'numerical_physiological_bounds':'not_established',
        'constrained_inversion_executed':False,
        'notebook_sha256':sha(NB),'html_sha256':sha(NB.with_suffix('.html')),
        'reader_sha256':sha(REV/'reader_visible_text.txt'),'generator_sha256':sha(GEN),
        'reader_characters':len(reader)}
(REV/'editorial_checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
