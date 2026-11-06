"""Three individual Scorpion rho(L) plots from the uniform-rho CT/CEM run.

No signal processing or FEM is performed by this reader. Raw source paths and
per-record local provenance remain in the ignored calculation directory.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json


NAMES={'georg':'Георгий','nik':'Ник','yaroslav':'Ярослав'}
ORDER=('georg','nik','yaroslav')
FORMATS={'1x1':('#2563eb','o'),'2x2':('#d97706','s'),'1x4':('#15803d','^')}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(x,d=2):return f'{x:.{d}f}'.replace('.',',')


def render(plan_file,results_file,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    plan_file=Path(plan_file);results_file=Path(results_file);out=Path(out)
    plan=read(plan_file);result=read(results_file)
    if result['schema']!='scorpion_uniform_rho_results_v1' or result['input_sha256']!=sha(plan_file):
        raise ValueError('The FEM result does not belong to the supplied input plan')
    observations={(s['subject_alias'],o['record_id']):o for s in plan['subjects'] for o in s['observations']}
    rows=[]
    for a in result['answers']:
        o=observations[a['subject_alias'],a['record_id']]
        if a['observed_Z_ohm']!=o['Z_ohm'] or abs(a['residual_ohm'])>a['tolerance_ohm']:
            raise ValueError('Observation binding or forward reproduction failed')
        rows.append(dict(subject_alias=a['subject_alias'],record_id=a['record_id'],
            size_mm=a['size_mm'],ttrkg_variant=a['ttrkg_variant'],channel=o['channel'],
            Z_inspiration_ohm=a['observed_Z_ohm'],rho_apparent_ohm_m=a['rho_apparent_ohm_m'],
            unit_rho_Z_ohm=a['Z_unit_rho_ohm'],linear_ratio_rho_ohm_m=a['linear_ratio_rho_ohm_m'],
            fixed_contact_correction_relative=a['fixed_contact_correction_relative'],
            reproduced_Z_ohm=a['predicted_Z_ohm'],residual_ohm=a['residual_ohm'],
            fit_tolerance_ohm=a['tolerance_ohm'],raw_q05_ohm=o['raw_q05_ohm'],raw_q95_ohm=o['raw_q95_ohm'],
            window_start_sample=o['window_samples_half_open'][0],window_end_sample=o['window_samples_half_open'][1],
            fs_hz=o['fs_hz'],annotation_status=o['annotation_status'],
            result_status=a['status']))
    if len(rows)!=len(observations) or len({(r['subject_alias'],r['record_id']) for r in rows})!=len(rows):
        raise ValueError('Missing or duplicated observations')
    out.mkdir(parents=True,exist_ok=False)
    rows.sort(key=lambda r:(ORDER.index(r['subject_alias']),r['size_mm'],r['ttrkg_variant']))
    with (out/'rho_apparent.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    for subject in ORDER:
        selected=[r for r in rows if r['subject_alias']==subject]
        primary='1x1' if subject=='georg' else '2x2'
        fig,ax=plt.subplots(figsize=(8.2,5.4));fig.subplots_adjust(left=.12,right=.96,bottom=.25,top=.79)
        for variant in (primary,*[v for v in FORMATS if v!=primary]):
            subset=[r for r in selected if r['ttrkg_variant']==variant]
            if not subset:continue
            color,marker=FORMATS[variant]
            ax.plot([r['size_mm'] for r in subset],[r['rho_apparent_ohm_m'] for r in subset],
                color=color,marker=marker,markersize=7.5,markeredgewidth=1.6,
                markerfacecolor=color if variant==primary else 'white',
                linestyle='-' if variant==primary and len(subset)>1 else 'none',linewidth=1.8,
                label='ТТРКГ '+variant.replace('x','×'),zorder=3)
            for r in subset:
                dy=12 if variant==primary else (-18 if variant=='1x4' or subject!='georg' else 27)
                dx=-15 if r['size_mm']==max(x['size_mm'] for x in selected) else (12 if r['size_mm']==min(x['size_mm'] for x in selected) else 0)
                ax.annotate(number(r['rho_apparent_ohm_m']), (r['size_mm'],r['rho_apparent_ohm_m']),
                    xytext=(dx,dy),textcoords='offset points',ha='center',va='center',color=color,fontsize=10)
        ticks=sorted({r['size_mm'] for r in selected});ax.set_xticks(ticks);ax.set_xlim(min(ticks)-10,max(ticks)+10)
        ax.set_ylim(0,max(r['rho_apparent_ohm_m'] for r in selected)*1.24)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'.replace('.',',')))
        ax.set_xlabel('Размер боковой сборки L — токовая база, мм',labelpad=10)
        ax.set_ylabel('Кажущееся удельное сопротивление ρ, Ом·м',labelpad=10)
        ax.grid(True,alpha=.20,zorder=0);ax.legend(loc='best',frameon=False,fontsize=10)
        fig.suptitle(f'{NAMES[subject]} · индивидуальная КТ-геометрия',x=.12,y=.96,ha='left',fontsize=15,fontweight='bold')
        fig.text(.12,.875,'Scorpion, 100 кГц · задержка на глубоком вдохе',fontsize=11,color='#374151')
        fig.text(.12,.065,'Весь объём однороден; конечные контакты учтены в CEM.\nРазметка дыхания кандидатная; положение электродов задано модельным сценарием.',fontsize=8.5,color='#4b5563',linespacing=1.6)
        fig.savefig(out/f'{subject}_rho_apparent.png',dpi=220,facecolor='white')
        fig.savefig(out/f'{subject}_rho_apparent.svg',facecolor='white');plt.close(fig)
    check=dict(schema='scorpion_uniform_rho_reader_v1',observations=len(rows),subjects=list(ORDER),
        input_plan_sha256=sha(plan_file),results_sha256=sha(results_file),
        max_absolute_reproduction_error_ohm=max(abs(r['residual_ohm']) for r in rows),
        max_absolute_fixed_contact_correction_relative=max(abs(r['fixed_contact_correction_relative']) for r in rows),
        subject_checks=result['subject_checks'],all_source_medians_recomputed=True,
        all_forward_reproductions_passed=True,breathing_annotation_accepted=False,
        global_mesh_convergence_proven=False,result_status='exploratory_hypothesis_not_validated',
        figure_files={p.name:sha(p) for p in sorted(out.glob('*_rho_apparent.*'))})
    (out/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    text=make_report(rows,check)
    (out/'README.md').write_text(text,encoding='utf-8')
    body=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Scorpion: кажущееся ρ по индивидуальной анатомии</title>',
          '<style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#172033}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 12px;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600}a{color:#2563eb}</style>',
          '<h1>Кажущееся ρ по индивидуальной анатомии</h1>',
          '<p>Scorpion, задержка на глубоком вдохе, 100 кГц. Каждому измерению соответствует собственная однородная КТ/FEM-модель добровольца.</p>',
          '<p>Результат условен: дыхательная разметка кандидатная, координаты электродов модельные; глобальная сеточная сходимость не установлена.</p>']
    for s in ORDER:
        body += [f'<h2>{NAMES[s]}</h2><img src="{s}_rho_apparent.png" alt="Кажущееся ρ от размера сборки: {NAMES[s]}">',
                 '<details><summary>Таблица. Импедансы и кажущееся ρ</summary><table><tr><th>Запись</th><th>L, мм</th><th>ТТРКГ</th><th>Z, Ом</th><th>ρ, Ом·м</th></tr>']
        for r in rows:
            if r['subject_alias']==s:body.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['record_id'],r['size_mm'],r['ttrkg_variant'].replace('x','×'),number(r['Z_inspiration_ohm'],3),number(r['rho_apparent_ohm_m'],4)])+'</tr>')
        body.append('</table></details>')
    body.append('<p><a href="README.md">Методика, формулы, допущения и результаты</a> · <a href="rho_apparent.csv">Расчётная таблица CSV</a></p></html>')
    (out/'index.html').write_text('\n'.join(body),encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False),flush=True)


def make_report(rows,check):
    text='''# Кажущееся удельное сопротивление по индивидуальной анатомии

Расчёт от 06.11.2026. Прибор Scorpion, 100 кГц, задержка на глубоком вдохе.

## Постановка и исходные данные

Определяется зависимость кажущегося удельного сопротивления ρ от номинальной токовой базы L боковой четырёхэлектродной сборки. Для каждого добровольца использована его собственная подготовленная КТ/FEM-геометрия из исследования `fem_smart_20261026`; перенос оператора другого человека не выполнялся. Всему объёму тела, включая лёгкие, кости, сердце и жир, присвоено одно неизвестное положительное ρ. Исходная геометрия разделов сохранена в сетке, электрические различия между ними устранены.

Для Георгия рассматриваются пять размеров: 60, 80, 100, 120 и 140 мм. Для Ника и Ярослава — три: 60, 100 и 140 мм. Номинальные позиции A–M–N–B вдоль поверхности соответствуют −L/2, −L/4, L/4, L/2; потенциальная база равна L/2. В моделях сохранены конечные контакты номинальным диаметром 5 мм, собственные координаты боковой установки и ранее подготовленные цилиндрические продолжения рук. Эти продолжения и расположение контактов являются модельными допущениями; точные координаты наклейки не измерены.

Уровень Z для каждой записи заново вычислен как медиана исходного поля BEI соответствующего канала внутри прежнего внутреннего окна вдоха. SHA-256 первичных файлов сверены с паспортами. Все 15 медиан совпали с сохранёнными уровнями с абсолютным расхождением не более 10⁻⁸ Ом. Использованы отсчёты с полуоткрытыми границами [начало, конец); циклический COUNT не заменяет время. В расчёт входит скалярное поле импеданса BEI, комплексная фаза не восстанавливается.

Разные размеры зарегистрированы последовательно. Различия между ними могут включать изменение дыхательного состояния, времени, контакта и повторной установки электродов. Разметка задержки на вдохе имеет статус кандидатной и не принята автором как окончательная. Смена размера не является одновременным многоканальным измерением.

## Однородная прямая модель и пересчёт

Внутри индивидуальной расчётной области решается квазистатическая задача

\\[
\\nabla\\!\\cdot(\\rho^{-1}\\nabla u)=0.
\\]

Здесь u — электрический потенциал внутри тела. На конечных контактных площадках используется полная электродная модель CEM. Для электрода e задаются

\\[
u+z_e\\rho^{-1}\\frac{\\partial u}{\\partial n}=U_e,
\\qquad \\int_{E_e}\\rho^{-1}\\frac{\\partial u}{\\partial n}\\,dS=I_e.
\\]

Здесь Uₑ — потенциал электрода, Eₑ — его контактная площадка, Iₑ — полный ток, n — внешняя нормаль к поверхности тела. Вне электродов нормальный поток равен нулю. Боковой источник имеет токи +1 и −1 А на A и B; M и N измеряют разность потенциалов при нулевых внешних токах. Физически присутствующие пассивные площадки и соединения ТТРКГ сохранены в соответствующем CEM-сценарии. Возбуждение ТТРКГ с боковым источником не суммируется: применяется подтверждённый режим раздельной работы каналов. Нормировка на 1 А не меняет измеренный Z и не вводит коэффициента усиления.

Для однородного объёма используется сумма пяти подготовленных материальных матриц:

\\[
K(\\rho)=\\frac{1}{\\rho}K_0,
\\qquad K_0=\\sum_{k=1}^{5}K_k^{(\\sigma_k=1)}.
\\]

Координаты и операторы электрической задачи выражены в СИ. Контактное удельное сопротивление zₑ = 0,01 Ом·м² сохранено из исходного CEM-сценария и не измерено независимо. Оно не изменяется вместе с ρ. Поэтому полный оператор имеет вид H(ρ) = C + diag(K₀/ρ, 0), где C — контактная матрица; пропорциональность Z(ρ) = ρZ(1) не принимается как точная.

Для каждой записи решено отдельное скалярное уравнение

\\[
Z_{\\mathrm{FEM}}(\\rho_{\\mathrm{каж}};\\Omega_{\\mathrm{КТ}},L,z_e)
=Z_{\\mathrm{вдох}}.
\\]

Отношение Zвдох/ZFEM(1 Ом·м) используется только как начальное приближение. Далее ρ уточняется методом Ньютона по lnρ с производной канонического CEM-ядра. Критерий завершения: |ZFEM − Zвдох| ≤ 10⁻⁵ Ом + 10⁻⁷|Zвдох|. Это численный допуск воспроизведения, а не метрологическая погрешность прибора. Новая постановка не подбирает положение электродов и не восстанавливает отдельные свойства тканей.

## Результаты по добровольцам

Линиями соединены только точки с одинаковым вариантом ТТРКГ: 1×1 у Георгия и 2×2 у Ника и Ярослава. Другие варианты показаны отдельными маркерами и не усредняются. Линии служат для чтения последовательности размеров; значения между точками не измерены.

'''
    for s in ORDER:
        r=[x for x in rows if x['subject_alias']==s]
        text+=f'### {NAMES[s]}\n\n![Кажущееся удельное сопротивление: {NAMES[s]}]({s}_rho_apparent.png)\n\n'
        text+='| Запись | L, мм | ТТРКГ | Z вдоха, Ом | ρ кажущееся, Ом·м |\n|---|---:|---|---:|---:|\n'
        for x in r:text+=f"| {x['record_id']} | {x['size_mm']} | {x['ttrkg_variant'].replace('x','×')} | {number(x['Z_inspiration_ohm'],3)} | {number(x['rho_apparent_ohm_m'],4)} |\n"
        primary=[x for x in r if x['ttrkg_variant']==('1x1' if s=='georg' else '2x2')]
        lo=min(primary,key=lambda x:x['rho_apparent_ohm_m']);hi=max(primary,key=lambda x:x['rho_apparent_ohm_m'])
        text+=f"\nДля основного варианта подключения диапазон рассчитанных значений составляет {number(lo['rho_apparent_ohm_m'])}–{number(hi['rho_apparent_ohm_m'])} Ом·м. Минимум получен при L = {lo['size_mm']} мм, максимум — при L = {hi['size_mm']} мм. Это диапазон точек данного протокола, не доверительный интервал и не диапазон свойств отдельной ткани.\n\n"
    text+=f'''## Численные проверки и границы интерпретации

Во всех 15 точках повторный прямой расчёт удовлетворяет заданному критерию; максимальная абсолютная невязка равна {number(check['max_absolute_reproduction_error_ohm'],9)} Ом. Каждое решение прошло контроль относительной невязки полной CEM-системы < 10⁻⁷ и нарушения взаимности < 10⁻⁷ Ом. Суммы материальных матриц при исходных правилах q1024 и q4096 совпали с относительным расхождением < 10⁻¹²; проверена также симметрия K₀. При одинаковом ρ способ интегрирования тканевых долей не должен менять полный однородный оператор. Эта проверка не заменяет исследование сходимости при изменении сетки.

Максимальная относительная разность между окончательным ρ и начальным пересчётом простым отношением импедансов составляет {number(100*check['max_absolute_fixed_contact_correction_relative'],4)} %. В таблице CSV сохранены оба значения. Окончательные графики используют решение с фиксированным контактным сопротивлением.

Глобальная сеточная сходимость данных индивидуальных моделей не установлена. Точность анатомического положения электродов, межсессионная воспроизводимость вдоха и метрологическая неопределённость не определены. Квантили 5–95 % отсчётов внутри окна приведены в CSV как характеристика разброса сигнала; доверительными границами ρ они не являются. Число десятичных знаков в таблицах отражает сохранённый расчёт, а не точность физиологической оценки.

Получены три зависимости эквивалентного однородного ρ от размера сборки с учётом индивидуальной формы расчётного тела. Различие ρ между размерами означает, что один однородный параметр не описывает все эти измерения при фиксированных прочих допущениях. Оно само по себе не отделяет неоднородность тканей от ошибок установки, контакта или повторения дыхательной пробы. Результаты пригодны для сравнения размеров внутри явно заданного сценария; они не являются оценками удельного сопротивления лёгкого, мягких тканей или крови по отдельности.

## Файлы и воспроизведение

В каталоге находятся три PNG и три SVG, полная [таблица CSV](rho_apparent.csv), [результаты проверок](verification.json) и [HTML для просмотра](index.html). Сырые сигналы, локальные пути, SHA-256 входных матриц, выбранные отсчёты и итерации сохранены в локальном каталоге расчёта `MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_apparent_20261106/`.

Подготовка и запуск выполняются скриптом `MATLAB_TRKG4_real_subjects/tools/scorpion_apparent_resistivity.py`, физический расчёт — профилем `scorpion_apparent_resistivity` единого `run_pipeline.m`. Читатель графиков — `Colab Notebooks/scorpion_v2/plot_apparent_resistivity.py`. Использованы готовые матрицы EIDORS и неизменённое ядро `trkg4_lateral_cem_response`; отдельный электрический решатель не создавался. Существующие расчёты и одобренный Word не изменялись.
'''
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.plan,a.results,a.out)
