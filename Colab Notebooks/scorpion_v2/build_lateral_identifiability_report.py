"""Scientific notebook and code-free reader export for 33.21."""
from pathlib import Path
import base64
import json

import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nbconvert import HTMLExporter

from lateral_pilot import ROOT, read, sha, save
from lateral_identifiability import SIZES

STEM='33.21_Множество_решений_двуслойной_модели_Scorpion_v2'
STATES={'inhale':'Задержка на вдохе','exhale':'Задержка на выдохе'}
PARAMETERS=['ρ₁, Ом·м','ρ₂, Ом·м','h, мм']


def russian_number(value,digits=3):
    result=f'{value:.{digits}g}'
    if 'e' in result:
        mantissa,exponent=result.split('e')
        result=mantissa+'·10'+str(int(exponent)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return result.replace('.',',')


def reader_table(frame):
    return frame.style.hide(axis='index').format({c:lambda x:russian_number(x,5)
        for c in frame.select_dtypes(include='floating').columns})


def layout(fig,title,height=480):
    fig.update_layout(template='plotly_white',title=title,height=height,
                      margin=dict(l=60,r=40,t=75,b=65),legend=dict(orientation='h',y=-.18))
    return fig


def cloud_figure(case,out):
    with np.load(Path(out)/case['cloud_file']) as data:
        cloud=data['cloud'];h=data['h'];r1=data['h_profile_rho1'];r2=data['h_profile_rho2']
    if len(cloud)>6000:
        cloud=cloud[np.linspace(0,len(cloud)-1,6000,dtype=int)]
    fig=go.Figure()
    if len(cloud):
        fig.add_trace(go.Scatter3d(x=cloud[:,0],y=cloud[:,1],z=cloud[:,2],mode='markers',
            marker=dict(size=2,opacity=.45,color=100*(cloud[:,3]/case['fit']['sse']-1),
                        colorscale='Viridis',colorbar=dict(title='Прирост<br>RSS, %')),
            name='RSS не более 110% минимума',
            hovertemplate='ρ₁=%{x:.4g} Ом·м<br>ρ₂=%{y:.4g} Ом·м<br>h=%{z:.4g} мм<extra></extra>'))
    fig.add_trace(go.Scatter3d(x=r1,y=r2,z=h,mode='lines',line=dict(color='#555',width=3),
                               name='Лучшие параметры при каждой h'))
    p=case['fit']['parameters']
    fig.add_trace(go.Scatter3d(x=[p[0]],y=[p[1]],z=[p[2]],mode='markers',
                               marker=dict(size=7,color='#c43131'),name='Минимум невязки'))
    fig.update_layout(scene=dict(xaxis=dict(title=PARAMETERS[0],type='log'),
        yaxis=dict(title=PARAMETERS[1],type='log'),zaxis=dict(title=PARAMETERS[2],type='log'),
        uirevision='preserve-camera-33.21'))
    return layout(fig,f"Множество решений: канал {case['channel']}; {STATES[case['state']].lower()}",610)


def profile_figure(cases,out):
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Канал 1: профиль невязки','Канал 2: профиль невязки',
        'Канал 1: семейство параметров','Канал 2: семейство параметров'),vertical_spacing=.15)
    colors={'inhale':'#247bb5','exhale':'#dc792e'}
    for case in cases:
        with np.load(Path(out)/case['cloud_file']) as d:
            h,sse,r1,r2=d['h'],d['h_profile_sse'],d['h_profile_rho1'],d['h_profile_rho2']
        ch=case['channel'];state=case['state']
        fig.add_trace(go.Scatter(x=h,y=100*np.sqrt(sse)/np.linalg.norm(case['fit']['observed_ohm']),
            name=STATES[state],legendgroup=state,showlegend=ch==1,line=dict(color=colors[state])),row=1,col=ch)
        for rho,label,dash in [(r1,'ρ₁','solid'),(r2,'ρ₂','dash')]:
            fig.add_trace(go.Scatter(x=h,y=rho,name=f'{label}: {STATES[state].lower()}',
                showlegend=False,line=dict(color=colors[state],dash=dash)),row=2,col=ch)
    fig.update_xaxes(title_text='h, мм',type='log')
    fig.update_yaxes(title_text='Относительная невязка, %',row=1)
    fig.update_yaxes(title_text='Удельное сопротивление, Ом·м',type='log',row=2)
    return layout(fig,'Профили по толщине и изменение оптимальных сопротивлений',780)


def observation_figure(cases):
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Канал 1: вдох','Канал 2: вдох','Канал 1: выдох','Канал 2: выдох'))
    for case in cases:
        row=1 if case['state']=='inhale' else 2;col=case['channel']
        y=np.array(case['fit']['observed_ohm']);quant=np.array(case['bootstrap']['observation_p05_p95'])
        fig.add_trace(go.Scatter(x=SIZES,y=y,mode='markers',name='Измерения',showlegend=row==col==1,
            error_y=dict(type='data',array=np.maximum(0,quant[:,1]-y),arrayminus=np.maximum(0,y-quant[:,0])),
            marker=dict(color='#222',size=8)),row=row,col=col)
        fig.add_trace(go.Scatter(x=SIZES,y=case['fit']['predicted_all_sizes_ohm'],mode='lines+markers',
            name='Двуслойная модель',showlegend=row==col==1,line=dict(color='#247bb5')),row=row,col=col)
    fig.update_xaxes(title_text='Токовая база L, мм');fig.update_yaxes(title_text='Z, Ом')
    return layout(fig,'Базовые уровни: измерения и минимум модельной невязки',760)


def stability_figure(cases):
    fig=make_subplots(rows=3,cols=2,subplot_titles=('Канал 1','Канал 2'),vertical_spacing=.09)
    for case in cases:
        color='#247bb5' if case['state']=='inhale' else '#dc792e'
        estimates=np.array([case['fit']['parameters']]+[r['parameters'] for r in case['leave_one_out']])
        labels=['Все размеры']+[f'Без {r["omitted_size_mm"]} мм' for r in case['leave_one_out']]
        for k in range(3):
            fig.add_trace(go.Scatter(x=labels,y=estimates[:,k],mode='lines+markers',name=STATES[case['state']],
                legendgroup=case['state'],showlegend=k==0 and case['channel']==1,
                line=dict(color=color)),row=k+1,col=case['channel'])
    for k,label in enumerate(PARAMETERS):
        fig.update_yaxes(title_text=label,type='log',row=k+1)
    return layout(fig,'Устойчивость к исключению каждого размера',960)


def resistance_figure(cases,out):
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Канал 1: профиль ρ₁','Канал 2: профиль ρ₁',
        'Канал 1: профиль ρ₂','Канал 2: профиль ρ₂'))
    for c in cases:
        with np.load(Path(out)/c['cloud_file']) as d:
            rho=d['profile_rho_axis']
            for k in (1,2):
                residual=100*np.sqrt(d[f'profile_rho{k}_sse'])/np.linalg.norm(c['fit']['observed_ohm'])
                fig.add_trace(go.Scatter(x=rho,y=residual,name=STATES[c['state']],
                    showlegend=k==1 and c['channel']==1,
                    line=dict(color='#247bb5' if c['state']=='inhale' else '#dc792e')),
                    row=k,col=c['channel'])
    fig.update_xaxes(title_text='Удельное сопротивление, Ом·м',type='log')
    fig.update_yaxes(title_text='Относительная невязка, %')
    return layout(fig,'Профили сопротивлений при свободных остальных параметрах',780)


def summary_table(cases):
    rows=[]
    for c in cases:
        f=c['fit'];r1,r2,h=f['parameters']
        rows.append({'Канал':c['channel'],'Состояние':STATES[c['state']],
            'ρ₁, Ом·м':r1,'ρ₂, Ом·м':r2,'h, мм':h,'Невязка, %':f['relative_residual_percent'],
            'κ(J)':f['condition_log_parameters'],
            'Граница поиска':', '.join(p for p,b in zip(['ρ₁','ρ₂','h'],f['boundary_hits']) if b) or 'Нет'})
    return pd.DataFrame(rows)


def build(out,observations,cases,contract,verification):
    out=Path(out);folder=Path(__file__).parent
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for case,ax in zip(cases,axes.ravel()):
        with np.load(out/case['cloud_file']) as d:
            h,sse,r1,r2=d['h'],d['h_profile_sse'],d['h_profile_rho1'],d['h_profile_rho2']
        ax.plot(h,100*np.sqrt(sse)/np.linalg.norm(case['fit']['observed_ohm']),color='#247bb5')
        ax.set(xscale='log',xlabel='Эквивалентная толщина h, мм',ylabel='Относительная невязка, %',
               title=f"Канал {case['channel']}; {STATES[case['state']].lower()}")
        ax.grid(alpha=.25)
        p=case['fit']['parameters'];ax.plot(p[2],case['fit']['relative_residual_percent'],'o',color='#c43131')
    fig.savefig(out/'profiles.png',dpi=180);plt.close(fig)
    # Freeze the scientific narrative and generated results in the notebook.
    cells=[]
    md=lambda s:cells.append(nbf.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbf.v4.new_code_cell(s))
    md('# 33.21. Множество решений двуслойной модели для боковых сборок Георгия\n\n'
       '**Статус исследования: разработка методики.** Оценки условны относительно геометрии, '
       'назначения каналов и предварительной разметки. Независимая точность параметров не установлена.\n\n'
       '## 1. Цель и последовательность исследования\n\n'
       'Цель — проверить, позволяют ли пять боковых сборок определить эффективные удельные сопротивления '
       'грудной стенки и лёгкого и эквивалентную толщину стенки. Применимость модели и устойчивость '
       'параметров рассматриваются раздельно: малый остаток подгонки может сочетаться с широким множеством решений. '
       'Обратная задача сопоставляется с индивидуальной конечноэлементной постановкой Георгия.\n\n'
       'Последовательность: исходные уровни → прямая модель → множество близких решений → профили невязки '
       '→ исключение отдельных сборок → оценка межцикловой и межзаписной неопределённости → КТ/FEM-проверка. '
       'Здесь выполнены численные проверки двуслойной постановки и условный анализ внутризаписной вариабельности.\n\n'
       '## 2. Экспериментальные входы и происхождение данных\n\n'
       'Использованы горизонтальные сборки 60, 80, 100, 120 и 140 мм из серии Scorpion v2, '
       'зарегистрированной 5 октября 2026 года при зондировании 100 кГц. По подтверждению автора '
       'они имели общий центр на правой боковой поверхности грудной клетки в области рёбер. '
       'Точные координаты центра и площадь электродного контакта не установлены.\n\n'
       'Базовый уровень определяется как медиана общего измеренного BEI внутри кандидатного окна '
       'задержки дыхания. Базовая и пульсовая составляющие получены из одного сигнала. '
       'Границы дыхания и события ЭКГ наследуются из этапа 11.20; их статус принятия не изменён. '
       'Сопоставимость последовательных записей разных размеров является допущением.\n\n'
       'Два числовых назначения каналов сохраняются для всего размерного ряда. Ни одна запись '
       'не получает отдельного назначения по результату подгонки. Динамическая КТ относится к другой сессии. '
       'Обхват 37 см сообщён автором 8 октября 2026 года для '
       'обеих рук на уровне токовых электродов. По решению автора руки моделируются цилиндрами. '
       'Дата измерения и его применимость к сессии КТ отдельно не установлены. '
       'Эта величина относится к последующей модели монтажа на руках.\n\n'
       '## 3. Модель и область поиска\n\n'
       'Модель представляет изотропное двуслойное полупространство: первый слой — эффективная грудная '
       'стенка, второй — лёгкое. Кривизна поверхности, отдельные рёбра и жир этой моделью не представлены. '
       'Толщина h является параметром электрического эквивалента; правило её связи с КТ-анатомией пока не проверено.\n\n'
       'Токовые и измерительные контакты имеют конечную площадь кругов диаметром 5 мм. '
       'Приняты равномерная плотность тока и усреднение потенциала по измерительной площадке. '
       'Это расчётное допущение; эквипотенциальность контакта и контактный импеданс требуют отдельной CEM-проверки.\n\n'
       'Технические пределы обоих сопротивлений составляют 0,1–100 Ом·м согласно принятому решению автора. '
       'Автоматического расширения нет. Толщина исследуется в вычислительной области 0,5–100 мм. '
       'Границы этой области не являются анатомическим или доверительным интервалом. '
       'Литературные диапазоны не ограничивают подгонку и используются для последующего сопоставления. '
       'Предыдущий проход 33.20 использовал литературные ограничения и сетку 5–60 мм; его результаты '
       'сохраняются как отдельный сценарий и не заменяют настоящий поиск.\n\n'
       'Критерий подгонки задаётся выражением\n\n'
       '$$RSS(\\rho_1,\\rho_2,h)=\\sum_{L}[Z_{\\mathrm{mod},L}(\\rho_1,\\rho_2,h)-Z_{\\mathrm{obs},L}]^2,$$\n\n'
       'где RSS — сумма квадратов остатков, Ом²; L — токовая база сборки, мм; '
       'Zmod и Zobs — модельный и наблюдаемый импедансы, Ом; ρ₁ и ρ₂ — удельные сопротивления '
       'первого и второго слоёв, Ом·м; h — эквивалентная толщина первого слоя, мм. '
       'Все размеры имеют одинаковый вес; распределение полной ошибки эксперимента неизвестно.\n\n'
       'Для каждой пары отношения сопротивлений и толщины общий масштаб сопротивлений минимизируется аналитически. '
       'Лучший сеточный кандидат уточняется непрерывной оптимизацией трёх параметров. '
       'Вдох и выдох сначала рассматриваются независимо; свободная разность найденных h '
       'не объявляется измеренным дыхательным изменением геометрии.')
    code("from pathlib import Path\nimport sys\nimport numpy as np\nfrom IPython.display import display, HTML, Image\n"
         "here=Path.cwd()\nmodule_dir=next(p for p in [here,here/'scorpion_v2',here/'Colab Notebooks/scorpion_v2'] if (p/'lateral_identifiability.py').exists())\n"
         "sys.path.insert(0,str(module_dir.resolve()))\nfrom lateral_pilot import EXPERIMENT,read\n"
         "from build_lateral_identifiability_report import cloud_figure,profile_figure,resistance_figure,observation_figure,stability_figure,summary_table,reader_table\n"
         "out=EXPERIMENT/'derived/exploratory/33.21_identifiability'\nresults=read(out/'results.local.json')\n"
         "plotly_loaded=False\ndef show(fig):\n    global plotly_loaded\n    display(HTML(fig.to_html(full_html=False,include_plotlyjs=not plotly_loaded,config={'displaylogo':False})))\n    plotly_loaded=True\n")
    md('## 4. Измеренные уровни и качество согласования\n\n'
       'На рисунке 1 показаны исходные уровни и предсказания в минимуме невязки. '
       'Вертикальные отрезки характеризуют 5-й и 95-й процентили медианы при повторной выборке секундных блоков '
       'внутри той же задержки. Они отражают только внутризаписную вариабельность. Ошибки контакта, '
       'повторной установки и различия физиологического состояния между записями в них не включены.\n\n'
       '**Рисунок 1.** Базовые импедансы пяти сборок на вдохе и выдохе при двух назначениях бокового канала.')
    code('display(reader_table(summary_table(results)))\nshow(observation_figure(results))')
    md('## 5. Множество близких решений\n\n'
       'Каждая точка рисунков 2–5 представляет тройку сопротивлений и толщины, '
       'для которой RSS не превышает 110% минимального значения. Цвет показывает относительное '
       'увеличение RSS. Дополнительно сохраняются области с допусками 1 и 5%. '
       'Допуски служат сценариями чтения поверхности невязки и не являются доверительными областями. '
       'Для каждой толщины и отношения сопротивлений вычисляется весь допустимый интервал масштаба; '
       'он представлен девятью точками. Для отображения число маркеров ограничено 6000.\n\n'
       'Серая линия соединяет лучшие сопротивления при каждой заданной толщине. Протяжённая область '
       'или длинная линия с близкой невязкой указывает на взаимную компенсацию параметров. '
       'Компактность области необходимо оценивать совместно с остатками модели и устойчивостью к изменению данных.')
    for i,case in enumerate(cases,2):
        md(f'**Рисунок {i}.** Множество решений: канал {case["channel"]}, {STATES[case["state"]].lower()}. '
           'По трём осям отложены ρ₁, ρ₂ и h; шкалы логарифмические. Красным отмечен минимум невязки.')
        code(f'show(cloud_figure(results[{i-2}],out))')
    md('## 6. Профили параметров и устойчивость\n\n'
       'Профиль по h строится минимизацией RSS по обоим сопротивлениям при каждой заданной толщине. '
       'На рисунке 6 верхние панели показывают относительную норму остатков, нижние — соответствующие '
       'сопротивления. Сплошная линия обозначает ρ₁, штриховая — ρ₂. Выход минимума на границу '
       'или слабое изменение невязки на широком интервале требуют явного ограничения интерпретации.\n\n'
       '**Рисунок 6.** Профили невязки и семейства оптимальных сопротивлений.')
    code('show(profile_figure(results,out))')
    md('**Рисунок 7.** Профили сопротивлений: для каждого фиксированного ρ₁ или ρ₂ '
       'остальные параметры минимизируются на сетке. Использована логарифмическая ось сопротивлений '
       'из 61 значения; её дискретность учитывается при чтении малых различий. '
       'Продолжение снижения невязки до технической границы показывает, что конечный параметр '
       'внутри исследованной области не найден.')
    code('show(resistance_figure(results,out))')
    md('На рисунке 8 расчёт повторяется после исключения каждого из пяти размеров. Это проверка '
       'чувствительности к составу наблюдений, без автоматического удаления записи. Остаток для '
       'исключённого размера сохраняется отдельно. Сильное изменение параметров при близком качестве '
       'подгонки показывает недостаточную устойчивость оценки.\n\n'
       '**Рисунок 8.** Изменение сопротивлений и толщины при исключении отдельных сборок.')
    code('show(stability_figure(results))')
    md('Дополнительно выполнены 120 повторных выборок секундных блоков для каждого состояния и канала. '
       'Оценки этих выборок получены на сетке параметров; их процентили описывают условную '
       'внутризаписную устойчивость. Полная неопределённость и статистическое покрытие не установлены. '
       'Узкие или совпадающие процентильные границы могут отражать дискретность сетки; '
       'они не устраняют протяжённого множества близких решений. '
       'Число обусловленности матрицы производных рассчитывается по логарифмам трёх параметров, '
       'чтобы сравнение не зависело от выбора миллиметров или метров. В таблице J обозначает '
       'матрицу производных импедансов по логарифмам ρ₁, ρ₂ и h; κ(J) — отношение '
       'её наибольшего и наименьшего сингулярных чисел. Эта локальная диагностика '
       'рассматривается совместно с нелинейным множеством решений.')
    code("import pandas as pd\nrows=[]\nfor c in results:\n    q=np.array(c['bootstrap']['p05_p95_parameters'])\n    for j,name in enumerate(['ρ₁, Ом·м','ρ₂, Ом·м','h, мм']):\n        rows.append({'Канал':c['channel'],'Состояние':c['state'],'Параметр':name,'5-й процентиль':q[0,j],'95-й процентиль':q[1,j]})\ndisplay(reader_table(pd.DataFrame(rows).replace({'inhale':'Вдох','exhale':'Выдох'})))")
    md('## 7. Численная проверка и границы вывода\n\n'
       'При обоих назначениях канала совместная оценка трёх параметров не продемонстрировала '
       'устойчивости. Относительная невязка лучших вариантов составляет 10,20–11,55%. '
       'В первом назначении минимум h достигает нижней границы 0,5 мм; '
       'во втором ρ₂ достигает верхней границы 100 Ом·м. Эти граничные числа '
       'не принимаются как индивидуальные свойства тканей. Числа обусловленности '
       'составляют приблизительно 1,3·10⁵–1,6·10⁶.\n\n'
       'Для первого канала на вдохе допуск 5% к минимальному RSS допускает '
       'ρ₁ примерно от 0,31 до 5,61 Ом·м и h от 0,5 до 9,22 мм. '
       'После исключения 60-миллиметровой сборки оценка ρ₁ достигает 100 Ом·м. '
       'Это наблюдаемая чувствительность к набору записей; причина различия '
       'между записями и вклад модельного расхождения отдельно не установлены. '
       'Предварительные статусы входов ограничивают вывод текущим исследовательским сценарием.\n\n'
       f'Синтетический пример с известными параметрами восстановлен с максимальной относительной ошибкой '
       f'{russian_number(verification["synthetic_relative_parameter_error"])}. При увеличении порядка интегрирования '
       f'по контактам и числа членов ряда максимальное различие импедансов составило '
       f'{russian_number(verification["maximum_abs_difference_ohm"],6)} Ом. Эти проверки подтверждают численную '
       'согласованность реализации в исследованных точках. Адекватность реальной анатомии ими не проверяется.\n\n'
       'Отсутствие устойчивой оценки фиксируется по протяжённому множеству близких решений, '
       'слабым профилям параметров, достижению технических границ и изменению результата при исключении сборок. '
       'Несоответствие модели данным оценивается по структуре остатков и известным составляющим ошибки. '
       'При неизвестной полной ошибке формальное статистическое отвержение модели не выполняется. '
       'Выход на край поиска сам по себе недостаточен для такого отвержения.\n\n'
       'Для дальнейшего использования тканевых параметров необходима индивидуальная КТ/FEM-проверка. '
       'Она должна включать сердце, лёгкие, кости, жир и оставшийся мягкотканный фон. '
       'В этой постановке ρ₁ обозначает сопротивление фона после выделения жира, '
       'ρ₂ — лёгкого, а сопротивление жира является третьим самостоятельным параметром. '
       'Эти величины не тождественны эффективным сопротивлениям двуслойного полупространства.\n\n'
       'Общий центр и ориентация боковых сборок восстанавливаются совместно с сопротивлениями '
       'либо задаются явными геометрическими сценариями. Кости и сердце сохраняются как отдельные материалы '
       'с проверяемыми опорными свойствами. При свободном сопротивлении жира требуется новая проверка '
       'идентифицируемости трёх тканевых параметров. FEM повышает анатомическую полноту, '
       'но сама по себе не гарантирует единственность обратной задачи.\n\n'
       '## 8. Передача результата в методику\n\n'
       'Передаваемый результат этого этапа — поверхности невязки, множества близких решений '
       'и диагностика устойчивости. Единственная физиологическая тройка параметров не утверждается. '
       'Коррекция ТТРКГ, восстановление сердечного объёма, УО и ФВ здесь не выполнялись. '
       'Дальнейший расчёт принимает тканевые параметры вместе с определением модели и их неопределённостью.\n\n'
       'Входные хэши, предварительные статусы разметки, версия реестра тканей, параметры интегрирования '
       'и полный численный результат сохранены в локальном паспорте расчёта. Исходные сигналы и КТ '
       'не изменены. Для повторного расчёта используется функция run из lateral_identifiability.py. '
       'Научная постановка и сохранённые результаты представлены в настоящем ноутбуке; '
       'вычислительное ядро находится в связанном Python-модуле. '
       'HTML является читательским экспортом без исходного кода.')
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')))
    path=folder/(STEM+'.ipynb')
    nbf.write(nb,path)
    # Execute result-reading cells, not another inverse run. Preserve saved outputs.
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(folder)}}).execute()
    nbf.write(nb,path)
    exporter=HTMLExporter()
    exporter.exclude_input=True;exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb)
    (folder/(STEM+'.html')).write_text(html,encoding='utf-8')
    reader='\n\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
    (folder/(STEM+'.md')).write_text(reader+'\n',encoding='utf-8')
    save(out/'reader_artifacts.local.json',dict(notebook_sha256=sha(path),
        html_sha256=sha(folder/(STEM+'.html')),reader_text_sha256=sha(folder/(STEM+'.md')),
        executed_code_cells=sum(c.cell_type=='code' for c in nb.cells),code_excluded_from_html=True))
    print(path,flush=True)
