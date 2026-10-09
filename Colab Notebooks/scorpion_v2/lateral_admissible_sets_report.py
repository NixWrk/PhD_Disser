"""Reader figures for bounded scalar families and cross-record diagnostics."""
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lateral_pilot import ROOT, read, sha
from lateral_identifiability import SIZES
from build_lateral_identifiability_report import layout, russian_number as num, STATES


def verify_additional(out):
    out=Path(out);manifest=read(out/'manifest.local.json');contract=read(out/'input_contract.local.json')
    producer=Path(__file__).with_name('lateral_admissible_sets.py')
    registry=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    if sha(producer)!=manifest['code_sha256'] or sha(registry)!=contract['literature_registry_sha256']:
        raise ValueError('Additional diagnostics input changed')
    for name,expected in manifest['outputs'].items():
        if sha(out/name)!=expected:raise ValueError('Additional diagnostics output changed: '+name)


def bounded_cloud(case, out):
    with np.load(Path(out)/(case['key']+'.npz')) as data:
        a=data['cloud']; h=data['h']; r1=data['rho1']; r2=data['rho2']
    if len(a)>6000:
        a=a[np.linspace(0,len(a)-1,6000,dtype=int)]
    f=go.Figure()
    if len(a):
        f.add_trace(go.Scatter3d(x=a[:,0],y=a[:,1],z=a[:,2],mode='markers',
            marker=dict(size=2,opacity=.45,color=100*(a[:,3]/case['fit']['rss']-1),
                        colorscale='Viridis',colorbar=dict(title='Прирост<br>RSS, %')),
            name='RSS ≤110% ограниченного минимума'))
    f.add_trace(go.Scatter3d(x=r1,y=r2,z=h,mode='lines',name='Оптимум при каждой h',line=dict(color='#555')))
    p=case['fit']['parameters']
    f.add_trace(go.Scatter3d(x=[p[0]],y=[p[1]],z=[p[2]],mode='markers',
        marker=dict(size=7,color='#c43131'),name='Минимум в литературной области'))
    f.update_layout(scene=dict(xaxis=dict(title='ρ₁, Ом·м',type='log'),
        yaxis=dict(title='ρ₂, Ом·м',type='log'),zaxis=dict(title='h, мм',type='log'),
        uirevision='bounded-family-camera'))
    return layout(f,f"Литературная область: канал {case['channel']}, {STATES[case['state']].lower()}",610)


def assembly_families(case, out):
    with np.load(Path(out)/(case['key']+'.npz')) as d:
        h=d['h']; bands=d['assembly_bands']; lines=d['conditional_lines']
        fixed=[d['fixed_rho2'],d['fixed_rho1']]
    titles=[f'L = {int(size)} мм: {rho}' for size in SIZES for rho in ('ρ₁(h)','ρ₂(h)')]
    f=make_subplots(rows=5,cols=2,subplot_titles=titles,vertical_spacing=.045,horizontal_spacing=.09)
    colors=['#247bb5','#cc792e','#27875b']
    for i in range(5):
        for k in range(2):
            for b,fill in [(0,None),(1,'tonexty')]:
                f.add_trace(go.Scatter(x=h,y=bands[i,2*k+b],mode='lines',
                    line=dict(color='rgba(90,90,90,.3)',width=1),fill=fill,
                    fillcolor='rgba(100,100,100,.16)',name='Оболочка выборки точных пар',
                    showlegend=i==0 and k==0 and b==1),row=i+1,col=k+1)
            for j in range(3):
                label=f"ρ{['₂','₁'][k]} = {num(fixed[k][j],5)} Ом·м"
                f.add_trace(go.Scatter(x=h,y=lines[i,k,j],mode='lines',name=label,
                    legendgroup=f'{k}-{j}',showlegend=i==0,line=dict(color=colors[j],dash='solid' if k==0 else 'dash'),
                    connectgaps=False),row=i+1,col=k+1)
    h_ticks=[.5,1,2,5,10,20,50,100]
    rho_ticks=[1,2,5,10,20,50]
    f.update_xaxes(type='log',title_text='h, мм',tickvals=h_ticks,ticktext=[num(v) for v in h_ticks])
    f.update_yaxes(type='log',title_text='Удельное сопротивление, Ом·м',
                   tickvals=rho_ticks,ticktext=[num(v) for v in rho_ticks])
    layout(f,f"Каждая сборка отдельно: канал {case['channel']}, {STATES[case['state']].lower()}",1570)
    f.update_layout(legend=dict(orientation='h',y=-.065,font=dict(size=11)),margin=dict(b=170))
    return f


def bounded_table(cases):
    return pd.DataFrame([{'Канал':c['channel'],'Состояние':STATES[c['state']],
        'ρ₁, Ом·м':c['fit']['parameters'][0],'ρ₂, Ом·м':c['fit']['parameters'][1],
        'h, мм':c['fit']['parameters'][2],'Невязка, %':c['fit']['residual_percent'],
        'Граница':', '.join(n for n,b in zip(['ρ₁','ρ₂','h'],c['fit']['boundary_hits']) if b) or 'Нет'} for c in cases])


def cross_table(rows):
    return pd.DataFrame([{'Область':{'technical':'Техническая','literature':'Литературная'}[r['area']],
        'Канал':r['channel'],'Состояние':STATES[r['state']],'Запись 140 мм':r['record_140'],
        'ρ₁, Ом·м':r['fit']['parameters'][0],'ρ₂, Ом·м':r['fit']['parameters'][1],
        'h, мм':r['fit']['parameters'][2],
        'Z₁₄₀, Ом':r['fit']['observations_ohm'][-1],
        'Невязка пяти сборок, %':r['fit']['residual_percent'],
        'Прогноз 140 по 60–120 мм, Ом':r['four_size_prediction_140_ohm'],
        'Остаток прогноза 140, Ом':r['held_out_140_error_ohm']} for r in rows])


def cross_plot(rows):
    f=make_subplots(rows=2,cols=2,subplot_titles=('Канал 1: вдох','Канал 2: вдох','Канал 1: выдох','Канал 2: выдох'))
    for r in rows:
        if r['area']!='technical':continue
        row=1 if r['state']=='inhale' else 2;col=r['channel'];record=r['record_140']
        color='#247bb5' if record=='004' else '#cc792e'
        f.add_trace(go.Scatter(x=SIZES,y=r['fit']['observations_ohm'],mode='markers',name=f'Уровни: {record}',
            marker=dict(color=color,size=7),showlegend=row==col==1),row=row,col=col)
        f.add_trace(go.Scatter(x=SIZES,y=r['fit']['prediction_ohm'],mode='lines',name=f'Подгонка: {record}',
            line=dict(color=color),showlegend=row==col==1),row=row,col=col)
        if record=='004':
            f.add_trace(go.Scatter(x=[140],y=[r['four_size_prediction_140_ohm']],mode='markers',
                marker=dict(symbol='x',size=12,color='#222'),name='Прогноз по 60–120 мм',showlegend=row==col==1),row=row,col=col)
    f.update_xaxes(title_text='Токовая база L, мм');f.update_yaxes(title_text='Z, Ом')
    return layout(f,'140 мм: отдельная запись 004 и горизонтальная ось креста 006',790)


def all_140_table(out):
    a=pd.read_csv(Path(out)/'observations.local.csv')
    a=a[a.record_id.isin(['SCV2-G-'+s for s in ('002','003','004','005','006')])]
    return a[['record_id','channel','state','baseline_raw_median_ohm']].rename(columns={
        'record_id':'Запись','channel':'Канал','state':'Состояние','baseline_raw_median_ohm':'Z, Ом'}).replace(STATES)


def append_review(cells, out):
    out=Path(out);verify_additional(out);relative=out.relative_to(ROOT).as_posix()
    md=lambda s:cells.append(nbf.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbf.v4.new_code_cell(s))
    md('## 6.1. Отдельное множество в физиологически мотивированной области\n\n'
       'Дополнение от 9 октября 2026 года отвечает на вопрос о решениях в физиологических пределах. '
       'В доступном реестре нет валидированного индивидуального интервала для Георгия. Поэтому показана '
       'отдельная литературная область: ρ₁ = 1,3774–41,6899 Ом·м и ρ₂ = 3,6817–24,0741 Ом·м. '
       'Источник — профиль screening_profiles/100000 общего реестра тканей: эффективные мягкие ткани '
       'и лёгкое без уточнения состояния. Он объединяет разнородные данные, включая перенос низкочастотных '
       'границ; эти числа не являются установленной нормой при 100 кГц. Толщина остаётся в области 0,5–100 мм. '
       'Основной поиск в пределах 0,1–100 Ом·м сохранён без изменений.\n\n'
       'Рисунки 9–12 показывают отдельную выборку пар сопротивлений и толщин с RSS не более 110% '
       'минимума внутри литературной области. Это критерий близости подгонки, а не доверительная область '
       'и не множество доказанно физиологических параметров. Серая линия при каждой h использует '
       'заново оптимизированные ρ₁ и ρ₂. Таблица указывает точную тройку ограниченного минимума. '
       'Достижение границы ρ₁ для первого канала и ρ₁, ρ₂ для второго не подтверждает устойчивость оценки.')
    code("from lateral_pilot import ROOT\nfrom lateral_admissible_sets_report import bounded_cloud,assembly_families,bounded_table,cross_table,cross_plot,all_140_table,verify_additional\n"
         f"review_out=ROOT/{relative!r}\nverify_additional(review_out)\nbounded=read(review_out/'results.local.json')\ncomparisons=read(review_out/'cross_comparison.local.json')\n"
         "display(reader_table(bounded_table(bounded)))\nfor c in bounded: show(bounded_cloud(c,review_out))")
    md('## 6.2. Сопротивления как функции h для каждой сборки\n\n'
       'Одна сборка даёт одно скалярное уравнение для двух сопротивлений при заданной h. '
       'Поэтому единственной пары ρ₁(h), ρ₂(h) для отдельной сборки нет. На рисунках 13–16 '
       'каждая строка относится к одному размеру, левый столбец — к ρ₁, правый — к ρ₂. '
       'Серая область показывает оболочку дискретной выборки допустимых пар, точно воспроизводящих '
       'наблюдаемый уровень в принятой литературной области; промежутки между узлами не проверены.\n\n'
       'Цветные кривые — условные решения: слева фиксировано ρ₂, справа фиксировано ρ₁. '
       'В легенде явно указано фиксированное сопротивление. Для каждого сопротивления использованы '
       'нижняя и верхняя границы области и частотное опорное значение Gabriel: '
       '2,7636 Ом·м для мышцы и 9,3153 Ом·м для расправленного лёгкого при 100 кГц. '
       'Мышца служит опорой для сценария стенки, но не представляет весь её тканевый состав. '
       'Отсутствие линии означает отсутствие допустимого корня при данном фиксированном сопротивлении. '
       'Точность скалярного решения проверяется подстановкой в прямую модель. '
       'Эти семейства объясняют, почему график одного оптимума не заменяет анализ идентифицируемости.')
    code('for c in bounded: show(assembly_families(c,review_out))')
    md('## 6.3. Особенность 140 мм и горизонтальная ось креста\n\n'
       'Сравниваются отдельная горизонтальная сборка 140 мм в записи 004 и горизонтальная ось '
       '140 мм в записи креста 006. Для каждого сценария сохраняется один числовой боковой канал '
       'на весь размерный ряд. Замена не принята автоматически. Остальные четыре записи 60–120 мм '
       'в обеих подгонках одинаковы.\n\n'
       'Резкий рост относится прежде всего к гипотезе канала 2: на вдохе Z увеличивается '
       'от примерно 64,7 Ом при 120 мм до 79,6 Ом при 140 мм. В канале 1 аналогичного скачка нет. '
       'Резкое изменение восстановленных сопротивлений усиливается слабой обусловленностью обратной задачи: '
       'параметры компенсируют друг друга и могут достигать границ поиска. Это математическое объяснение '
       'чувствительности оценки; физическая причина различия самих записей пока не выделена.\n\n'
       'Кроме повторной подгонки пяти размеров использована проверка прогноза: модель подогнана '
       'только по 60–120 мм, затем одно и то же предсказание 140 мм сопоставлено с 004 и 006. '
       'Остаток определён как прогноз минус наблюдение. При технических пределах для канала 1 '
       'модуль остатка снижается с 12,34 до 2,54 Ом на вдохе и с 12,86 до 2,58 Ом на выдохе. '
       'Для канала 2 он увеличивается с 16,97 до 18,89 Ом и с 17,67 до 22,66 Ом соответственно. '
       'Литературно ограниченная подгонка сохраняет это направление различий.\n\n'
       '**Рисунок 17.** Наблюдения, отдельные минимумы подгонки и прогноз исключённой сборки 140 мм. '
       'Каждая линия использует собственную тройку ρ₁, ρ₂, h из следующей таблицы '
       'для соответствующих канала, дыхания и записи в технической области. '
       'Эта тройка постоянна вдоль линии. Таблица показывает все 16 сравнений, включая обе области поиска.')
    code('show(cross_plot(comparisons))\ndisplay(reader_table(cross_table(comparisons)))')
    md('Дополнительно приведены уровни записей 002–006 с размером 140 мм. '
       'Они относятся к разным монтажам и не считаются одинаковыми повторностями. '
       'При съёме с креста меняются соседние электроды и электрическая схема; '
       'возможны также различия контакта, времени и состояния испытуемого. '
       'Поэтому улучшение согласования в одном назначении канала не подтверждает его роль '
       'и не доказывает межканального механизма. Выбор 004 либо 006 требует проверки монтажа '
       'и индивидуальной прямой модели.')
    code('display(reader_table(all_140_table(review_out)))')
