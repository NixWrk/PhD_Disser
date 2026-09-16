"""Editorial change only: preserve all scientific outputs and computation statements."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone

import nbformat
import pandas as pd
import numpy as np
from nbconvert import HTMLExporter

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'Colab Notebooks').is_dir())
OUT = Path(__file__).parent
NB = next((ROOT/'Colab Notebooks').glob('33.08*.ipynb'))
GEN = ROOT/'MATLAB_TRKG4_real_subjects/notebooks/build_lateral_real_application.py'
BACKUP = OUT/'before_conclusion_edit_20260916'
BACKUP.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for p in [NB, NB.with_suffix('.html'), NB.with_suffix('.language-review.json'), GEN]:
    target=BACKUP/p.name
    if target.exists():
        assert sha(target)==sha(p), f'Backup differs: {target}'
    else:
        shutil.copy2(p,target)

before=nbformat.read(NB,as_version=4)
nb=copy.deepcopy(before)
gen=GEN.read_text(encoding='utf-8')
old_gen=gen
frozen={name:sha(OUT/name) for name in before.metadata.study.input_sha256}

def markdown_replace(index, old, new):
    global gen
    assert nb.cells[index].source.count(old)==1
    assert gen.count(old)==1
    nb.cells[index].source=nb.cells[index].source.replace(old,new)
    gen=gen.replace(old,new)

for index,new in {
2:'''**Вывод по исходным данным.** Получаемые оценки относятся к одному человеку и двум выбранным задержкам дыхания. Базовые уровни и пульсовые кривые взяты из его записей. КТ задаёт геометрию, а сопоставимость состояния между записями разных сборок принимается как допущение. На выдохе используется геометрия вдоха, поэтому возможное влияние дыхательного движения тканей в этом расчёте может быть отнесено к изменению их сопротивлений.''',
3:'''**Вывод по постановке.** Сравнение позволяет увидеть влияние двух разных решений: как упрощена форма лёгкого и как представлены электроды. Основным результатом считается расчёт с конечными электродами. Сопоставление с полной КТ показывает, насколько изменяются оценки при упрощении формы лёгкого; сопоставление с точечными электродами — насколько они зависят от описания контактов. Ни одно из этих сравнений само по себе не устанавливает истинные свойства тканей.'''
}.items():
    old=nb.cells[index].source
    markdown_replace(index,old,old+'\n\n'+new)

old=nb.cells[7].source.split('\n\n',1)[1]
markdown_replace(7,old,'''**Вывод по базовому расчёту.** Модель 4 и полная КТ дают почти одинаковые кривые импеданса и близкие оценки сопротивлений. В этом примере выбранное упрощение формы лёгкого мало меняет результат восстановления. Переход от точечных к конечным электродам меняет результат заметно сильнее и уменьшает расхождение с измерениями. Поэтому при выборе быстрой модели важно сохранять проверенное описание электродов.

Оставшиеся расхождения зависят от размера сборки. Одного среднего показателя RMSE недостаточно, чтобы объяснить их причину. Хорошее совпадение кривых также не доказывает, что сопротивления тканей найдены точно: разные сочетания параметров могут давать близкий импеданс. Далее проверяется, можно ли упростить дыхательный расчёт и принять одно базовое ρ₁ для вдоха и выдоха.''')

old=nb.cells[12].source.split('\n\n',1)[1]
markdown_replace(12,old,'''**Вывод по пульсовым оценкам.** Записанные пульсовые изменения импеданса удалось разложить на две расчётные кривые: для мягких тканей и для лёгкого. При замене модели 4 полной КТ эти кривые меняются мало. В рамках выбранной модели пульсовой размах сопротивления лёгкого на вдохе больше, чем на выдохе; размах сопротивления мягких тканей в обоих состояниях значительно меньше.

Измеренный пульсовой импеданс является входом расчёта. Разделение этого сигнала между тканями зависит от модели, найденных базовых сопротивлений и допущения об отсутствии пульсового движения тканей. Кроме того, кривые построены по последовательным записям сборок и описывают медианный цикл, а не одно одновременно измеренное сокращение. Нулевой уровень задан центрированием циклов. Поэтому согласование двух геометрических моделей ещё не доказывает точность тканевых пульсовых оценок.

Малое изменение ρ₁ может заметно влиять на импеданс, если сборки к нему чувствительны. Одного сравнения размахов δρ₁ и δρ₂ недостаточно, чтобы исключить мягкие ткани. Следующий расчёт прямо проверяет последствия такого исключения.''')

old=nb.cells[17].source.split('\n\n## Источники и повторение расчёта')[0]
markdown_replace(17,old,'''**Общий итог.** Получен работающий расчёт, который переводит реальные измерения боковых сборок в оценки базовых и пульсовых свойств мягких тканей и лёгкого. Выбранная модель 4 даёт результаты, близкие к полной КТ-модели для этого человека и принятых условий. Представление конечных электродов оказалось существенно важнее проверенного упрощения формы лёгкого. При этом точность самих тканевых оценок пока не доказана.

Расчёт даёт небольшое дыхательное изменение базового сопротивления мягких тканей и значительно большее изменение сопротивления лёгкого. Однако ρ₂ на вдохе достигает верхней границы поиска. Кроме того, геометрия для вдоха и выдоха одинакова. Поэтому найденную дыхательную разность пока нельзя полностью приписать изменению свойств лёгкого: в ней может отражаться и неучтённое движение тканей.

**Возможности упрощения различаются.** Постоянное базовое ρ₁ — кандидат на дальнейшую проверку: при таком условии согласование с измерениями ухудшается умеренно. Исключение пульсового δρ₁ заметно меняет лёгочную пульсовую оценку и увеличивает остаток сигнала, который модель не объясняет. Поэтому малость δρ₁ сама по себе не обосновывает его исключение. Допустимая погрешность обоих упрощений ещё не задана.

**Что остаётся проверить.** Для базовой задачи прежде всего нужно разобраться с граничной оценкой лёгкого на вдохе и последствиями неизменной геометрии при дыхании. Сохраняются заданные автором сценарии изменения толщины: размах до 2 мм при глубоком дыхании и до 1 мм при обычном. Здесь эти сценарии ещё не рассчитаны. Дополнительно остаются вопросы к двум принятым материалам, фиксированному контактному сопротивлению, представлению электродных площадок на сетке и влиянию отмеченных проблем качества записей. Для пульсовой задачи нужно проверить, насколько точна линейная связь между малыми изменениями импеданса и сопротивлений в найденных базовых точках.

Этот ноутбук даёт исходный расчёт для последовательной проверки перечисленных допущений. В каждой следующей проверке следует менять одно условие при тех же измерениях и оценивать, насколько меняются восстановленные параметры. Выбор оптимальной пары сборок, перенос на другого человека и применение к обычному дыханию пока не обоснованы. Перед использованием параметров для коррекции ТТРКГ требуется установить допустимую погрешность и проверить сокращённый набор размеров. Функции сердца в этом ноутбуке не рассчитывались.''')

def md_calls(source):
    return sorted([n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='display' and len(n.args)==1
        and isinstance(n.args[0],ast.Call) and isinstance(n.args[0].func,ast.Name)
        and n.args[0].func.id=='Markdown'],key=lambda n:(n.lineno,n.col_offset))

fits=pd.read_csv(OUT/'baseline_fits.csv')
main=fits[(fits.model=='variable_transverse')&(fits.electrodes=='CEM')].set_index('state')
planar=fits[fits.model=='planar_nominal'].set_index('state')
bounds=np.asarray(json.loads((OUT/'input_contract.json').read_text(encoding='utf-8'))['bounds_ohm_m'])
number=lambda x,places=3:f'{float(x):.{places}f}'.replace('.',',')
def bound_label(rho,k):
    if np.isclose(rho,bounds[k,0],rtol=1e-6):return 'нижняя граница'
    if np.isclose(rho,bounds[k,1],rtol=1e-6):return 'верхняя граница'
    return 'внутри диапазона'
shared=json.loads((OUT/'cem_variable_transverse_shared.json').read_text(encoding='utf-8'))['best']
env=dict(ai=main.loc['inhale'],ae=main.loc['exhale'],planar=planar,number=number,bound_label=bound_label,
    ri=shared['rho'][1],re=shared['rho'][2],outside={'Вдох':101,'Выдох':0})

def display_replace(index, ordinal, expression):
    global gen
    c=nb.cells[index]
    old=ast.get_source_segment(c.source,md_calls(c.source)[ordinal])
    new='display(Markdown('+expression+'))'
    assert c.source.count(old)==1 and gen.count(old)==1,(index,ordinal)
    c.source=c.source.replace(old,new)
    gen=gen.replace(old,new)
    outputs=[o for o in c.outputs if 'text/markdown' in o.get('data',{})]
    outputs[ordinal].data['text/markdown']=eval(compile(ast.parse('('+expression+')',mode='eval'),'<editorial text>','eval'),env)

display_replace(5,0,'''f"Плоское аналитическое приближение с центральной глубиной из 20.16 даёт RMSE "
    f"**{number(planar.loc['inhale'].train_rmse)} Ом на вдохе** и **{number(planar.loc['exhale'].train_rmse)} Ом на выдохе**. "
    "В этом примере оно хуже воспроизводит измеренные импедансы, чем выбранная модель с конечными электродами. "
    "Разницу нельзя объяснить только формой лёгкого: одновременно меняются геометрия и описание электродов."''')
display_replace(5,1,'''f"Для модели 4: ρ₂ на вдохе — **{bound_label(ai.rho2,1)}**, на выдохе — **{bound_label(ae.rho2,1)}**. "
    "На вдохе расчёт дошёл до максимально разрешённого сопротивления лёгкого. Это означает, что найденное число зависит от границы поиска. "
    "Его пока нельзя считать точно установленным свойством ткани. На выдохе оценка не упирается в ограничение, но и её точность ещё не подтверждена независимым измерением."''')
display_replace(9,0,'''f"В основном сценарии разность «вдох минус выдох» составляет "
    f"Δρ₁ = **{number(ai.rho1-ae.rho1,5)} Ом·м**, Δρ₂ = **{number(ai.rho2-ae.rho2,5)} Ом·м**. "
    f"При общем ρ₁ дыхательная разность Δρ₂ составляет **{number(ri-re,5)} Ом·м**. "
    "\\n\\n**Вывод по дыхательному упрощению.** Одно базовое ρ₁ для вдоха и выдоха умеренно ухудшает согласование с измерениями. "
    "Поэтому постоянное ρ₁ можно рассматривать как кандидат на упрощение. Но это условие меняет найденное ρ₂: "
    "на выдохе оно достигает нижней границы поиска, а на вдохе остаётся на верхней. "
    "Модель может переносить часть изменений, которые раньше объясняла мягкими тканями, в лёгочный параметр. "
    "Приемлемость такого переноса ещё нужно проверить. "
    "\\n\\nЭта проверка относится к базовым уровням между двумя задержками дыхания. "
    "Она не означает, что сопротивление мягких тканей постоянно внутри сердечного цикла. Пульсовое δρ₁ рассматривается отдельно в следующих разделах."''')
display_replace(14,0,'''"**Вывод по исключению пульсового δρ₁.** Если запретить пульсовое изменение мягких тканей, "
    "расчёт увеличивает пульсовой размах сопротивления лёгкого и хуже воспроизводит записанный сигнал. "
    "Особенно заметно это на вдохе. Следовательно, маленькое δρ₁ не означает, что им можно пренебречь: "
    "его исключение меняет оценку δρ₂. "
    "\\n\\nЭто показывает последствия упрощения в принятой модели. "
    "Улучшение подгонки при двух параметрах само по себе ещё не доказывает правильность разделения физиологических источников. "
    "Чтобы решить, допустимо ли исключать δρ₁, нужно заранее определить приемлемое изменение результата и невязки."''')
display_replace(14,1,'''f"Пульсовые приращения рассчитывались без ограничения суммарного сопротивления исходным диапазоном. "
    f"Число фаз, в которых базовое ρ₂ вместе с δρ₂ выходит за его пределы: **вдох — {outside['Вдох']}, выдох — {outside['Выдох']} из 200**. "
    "На вдохе базовое ρ₂ уже находится на верхней границе, поэтому положительное пульсовое приращение выводит его за неё. "
    "Это ещё одна причина осторожно интерпретировать лёгочную оценку на вдохе. "
    "Обрезание кривой по границе скрывает проблему, но не подтверждает физиологическую достоверность результата."''')
display_replace(16,2,'''f"В основной CEM-модели число обусловленности для относительных изменений сопротивлений равно "
    f"**{number(ai.condition_log_parameters,2)} на вдохе** и **{number(ae.condition_log_parameters,2)} на выдохе**. "
    "Это означает, что измерения хорошо различают одни сочетания изменений тканей и значительно хуже — другие. "
    "Поэтому похожие импедансы могут соответствовать различающимся парам ρ₁ и ρ₂. "
    "Само число не говорит, на сколько процентов ошибочна каждая оценка: для этого нужно описать ошибки входных данных и модели."''')

def computation(source):
    tree=ast.parse(source)
    class RemoveProse(ast.NodeTransformer):
        def visit_Expr(self,node):
            n=node.value
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='display' and len(n.args)==1:
                a=n.args[0]
                if isinstance(a,ast.Call) and isinstance(a.func,ast.Name) and a.func.id=='Markdown':return None
            return self.generic_visit(node)
    return ast.dump(RemoveProse().visit(tree),include_attributes=False)

assert len(before.cells)==len(nb.cells)
changed=[]
for i,(a,b) in enumerate(zip(before.cells,nb.cells)):
    assert (a.id,a.cell_type,a.metadata)==(b.id,b.cell_type,b.metadata)
    if a.source!=b.source:changed.append(i)
    if a.cell_type=='code':
        assert a.execution_count==b.execution_count
        assert computation(a.source)==computation(b.source),f'Computation changed: {i}'
        assert len(a.outputs)==len(b.outputs)
        for oa,ob in zip(a.outputs,b.outputs):
            if 'text/markdown' in oa.get('data',{}):
                xa,xb=copy.deepcopy(oa),copy.deepcopy(ob)
                xa.data.pop('text/markdown');xb.data.pop('text/markdown')
                assert xa==xb
            else:assert oa==ob,f'Nonprose output changed: {i}'
    else:
        import re
        assert re.findall(r'\$\$.*?\$\$',a.source,re.S)==re.findall(r'\$\$.*?\$\$',b.source,re.S)
        if '**Рисунок ' in a.source:assert a.source.split('\n\n')[0]==b.source.split('\n\n')[0]
assert frozen=={name:sha(OUT/name) for name in frozen}
ast.parse(gen)
GEN.write_text(gen,encoding='utf-8')
nb.metadata.study.generator_sha256=sha(GEN)
nb.metadata.study.editorial_revision=dict(date='2026-09-16',scope='conclusions_only',cells_reexecuted=False)
nbformat.validate(nb)
nbformat.write(nb,NB)
exporter=HTMLExporter();exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
html,_=exporter.from_notebook_node(nb)
NB.with_suffix('.html').write_text(html,encoding='utf-8')
reader=[]
for c in nb.cells:
    if c.cell_type=='markdown':reader.append(c.source)
    else:
        for o in c.outputs:
            d=o.get('data',{})
            if 'text/markdown' in d:reader.append(d['text/markdown'])
            elif 'text/html' in d and 'plotly' not in d['text/html'].lower():reader.append(d['text/html'])
(OUT/'reader_33.08.md').write_text('\n\n'.join(reader),encoding='utf-8')
report=dict(date_utc=datetime.now(timezone.utc).isoformat(),scope='conclusions_only',changed_cells=changed,
    computation_ast_unchanged=True,nonprose_outputs_unchanged=True,formulae_unchanged=True,
    figure_captions_unchanged=True,cell_ids_and_execution_counts_unchanged=True,cells_reexecuted=False,
    all_frozen_inputs_unchanged=True,sha256={p.name:sha(p) for p in [NB,NB.with_suffix('.html'),GEN]},
    prior_execution_manifest_unchanged_sha256=sha(OUT/'execution_33.08.json'))
(OUT/'editorial_verification_20260916.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
