"""Georg: extend the finite-disk thickness profile to 120 mm, without changing history."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_pilot import ROOT,EXPERIMENT,prepare,DiskModel,fit_state,invert_pulse,read,save,sha,STATUS
from build_georg_fem_reports import new_cells,export,table

OUT=EXPERIMENT/'derived/exploratory/33.26_thickness_120mm_20261012'
STEM='33.26_Толщина_до_120_мм_и_проверка_прогноза_Георгия_Scorpion_v2'
MODEL=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_ct_model_20261011'
H=np.arange(5.,121.,5.)
SCENARIOS={'wide':('Профиль 5–120 мм',5.,120.),
    'ct_profile':('Сценарий 40–120 мм',40.,120.),
    'deep_band':('Сценарий 100–120 мм',100.,120.),
    'h100':('Задано h = 100 мм',100.,100.),
    'h110':('Задано h = 110 мм',110.,110.),
    'h120':('Задано h = 120 мм',120.,120.)}


class SelectedResponse:
    """Select measured sizes from the unchanged full finite-disk response."""
    def __init__(self,model,indices):self.model,self.indices=model,indices
    def evaluate(self,rho):
        z,j=self.model.evaluate(rho)
        return z[self.indices],j[self.indices]


def profile(models,observed,bounds,use):
    rows=[]
    for h,model in models.items():
        selected=SelectedResponse(model,use)
        fits={state:fit_state(selected,y[use],bounds) for state,y in observed.items()}
        rows.append(dict(h_mm=h,fits=fits,score=sum(f['rmse_ohm']**2 for f in fits.values())))
    return rows


def choose(rows,limits):
    low,high=limits
    candidates=[r for r in rows if low<=r['h_mm']<=high]
    if not candidates:raise ValueError('Thickness scenario is outside the computed profile')
    return min(candidates,key=lambda r:r['score'])


def checked_ct_profile():
    item=read(MODEL/'lung_depth_profile.local.json')
    if sha(MODEL/'body_ct_surface.stl')!=item['surface_sha256']:
        raise ValueError('CT depth profile refers to a different skin surface')
    import sys
    sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
    from georg_model_contract import resolve
    _,current=resolve()
    lung=ROOT/'MATLAB_TRKG4_real_subjects'/current['segmentation_directory']/'lung.nii.gz'
    if sha(lung)!=item['lung_mask_sha256']:raise ValueError('CT lung mask changed')
    return item


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    ct=checked_ct_profile();frame,ensembles,inputs=prepare(EXPERIMENT)
    registry=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    compartments=read(registry)['screening_profiles']['100000']['compartments']
    bounds=np.array([compartments[k]['rho_bounds_ohm_m'] for k in ['soft','lung_unspecified']])
    sizes=np.array([60.,80.,100.,120.,140.]);all_sizes=np.arange(5)
    models={float(h):DiskModel(sizes,h) for h in H}
    results=[]
    for reference in ['002','003','004']:
        ids=['SCV2-G-'+s for s in ['010','009','008','007',reference]];channels=[1,1,1,1,2]
        observed={};waves={}
        for state in ['inhale','exhale']:
            observed[state]=np.array([frame[(frame.record_id==rid)&(frame.channel==ch)&(frame.state==state)].baseline_raw_median_ohm.iloc[0]
                for rid,ch in zip(ids,channels)])
            items=[ensembles[rid]['windows'][state]['channels'][str(ch)] for rid,ch in zip(ids,channels)]
            if not all(item['n_cycles']>=3 for item in items):raise ValueError('Insufficient candidate cycles')
            waves[state]=np.array([item['median_ohm'] for item in items])
        profiles=profile(models,observed,bounds,all_sizes);scenarios={}
        for key,(_,low,high) in SCENARIOS.items():
            best=choose(profiles,(low,high));model=models[best['h_mm']]
            scenarios[key]=dict(h_limits_mm=[low,high],best=best,
                pulse={state:invert_pulse(model,best['fits'][state],wave) for state,wave in waves.items()},withheld=[])
        for omitted,size in enumerate(sizes):
            use=np.delete(all_sizes,omitted)
            training_profile=profile(models,observed,bounds,use)
            for key,(_,low,high) in SCENARIOS.items():
                best=choose(training_profile,(low,high));model=models[best['h_mm']]
                selected=SelectedResponse(model,use)
                for state,wave in waves.items():
                    base=best['fits'][state];rho=np.array([base['rho1_ohm_m'],base['rho2_ohm_m']])
                    z,j=model.evaluate(rho);pulse=invert_pulse(selected,base,wave[use])
                    fraction=np.array(pulse['delta_rho_ohm_m'])/rho[:,None]
                    predicted=j[omitted]@fraction;actual=wave[omitted]
                    scenarios[key]['withheld'].append(dict(size_mm=float(size),state=state,h_mm=best['h_mm'],
                        rho_ohm_m=rho.tolist(),training_sizes_mm=sizes[use].tolist(),
                        baseline_observed_ohm=float(observed[state][omitted]),baseline_predicted_ohm=float(z[omitted]),
                        pulse_observed_ohm=actual.tolist(),pulse_predicted_ohm=predicted.tolist(),
                        pulse_fractional_changes=fraction.tolist(),
                        pulse_relative_error_percent=float(100*np.linalg.norm(predicted-actual)/np.linalg.norm(actual)),
                        training_nonpositive_phase_count=pulse['nonpositive_resistivity_phase_count'],
                        training_max_abs_fractional_change=pulse['max_abs_fractional_change']))
        results.append(dict(reference_140_record=reference,record_ids=ids,channels=channels,
            sizes_mm=sizes.tolist(),profiles=profiles,scenarios=scenarios))
        print('Extended thickness and excluded-size forecasts:',reference,flush=True)
    save(OUT/'results.local.json',results)
    save(OUT/'contract.local.json',dict(status=STATUS,owner='33.26',input_contract=inputs,
        producer_sha256=sha(__file__),kernel_sha256=sha(Path(__file__).with_name('lateral_pilot.py')),
        registry_sha256=sha(registry),output_sha256=sha(OUT/'results.local.json'),h_grid_mm=H.tolist(),
        scenarios={key:list(value) for key,value in SCENARIOS.items()},bounds_ohm_m=bounds.tolist(),
        ct_profile_sha256=sha(MODEL/'lung_depth_profile.local.json'),
        ct_nominal_depth_summary_mm=[float(np.min(ct['depth_mm'])),float(np.median(ct['depth_mm'])),float(np.max(ct['depth_mm']))],
        author_upper_depth_mm=[100,120],author_depth_source='direct clarification 2026-10-12',
        h_is_not_validated_CT_reduction=True,annotations_accepted=False,
        unchanged_contact_diameter_mm=5,unchanged_resistivity_bounds=True,
        withheld_rule='Exclude size from both baseline fits and pulse fits, including joint selection of h from training states',
        historical_33_20_and_33_24_outputs_changed=False))
    return results


def verify():
    c=read(OUT/'contract.local.json')
    for path,key in [(Path(__file__),'producer_sha256'),(Path(__file__).with_name('lateral_pilot.py'),'kernel_sha256'),
            (ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json','registry_sha256'),
            (OUT/'results.local.json','output_sha256'),(MODEL/'lung_depth_profile.local.json','ct_profile_sha256')]:
        if sha(path)!=c[key]:raise ValueError('Thickness extension input changed: '+str(path))
    for name,key in [('derived/scorpion_review/summary.local.json','review_summary_sha256'),
            ('annotations/scorpion_review/annotations_candidates.local.json','annotations_sha256')]:
        if sha(EXPERIMENT/name)!=c['input_contract'][key]:raise ValueError('Candidate signal input changed')
    checked_ct_profile()
    return c


def summary_table():
    verify();rows=[]
    for item in read(OUT/'results.local.json'):
        for key,scenario in item['scenarios'].items():
            for state,fit in scenario['best']['fits'].items():
                pulse=scenario['pulse'][state]
                rows.append({'140 мм: запись':item['reference_140_record'],'Сценарий':SCENARIOS[key][0],
                    'Дыхание':'Вдох' if state=='inhale' else 'Выдох','h, мм':scenario['best']['h_mm'],
                    'ρ₁, Ом·м':fit['rho1_ohm_m'],'ρ₂, Ом·м':fit['rho2_ohm_m'],
                    'Базовый остаток, Ом':fit['rmse_ohm'],'Край диапазона ρ':fit['bound_hit'],
                    'Пульсовый остаток, %':pulse['residual_percent'],'κ(J)':pulse['condition'],
                    'Максимум |δρ/ρ|':pulse['max_abs_fractional_change'],
                    'Фазы с неположительным ρ':pulse['nonpositive_resistivity_phase_count']})
    return pd.DataFrame(rows)


def forecast_table():
    verify();rows=[]
    for item in read(OUT/'results.local.json'):
        for key,scenario in item['scenarios'].items():
            for fold in scenario['withheld']:
                rows.append({'140 мм: запись':item['reference_140_record'],'Сценарий':SCENARIOS[key][0],
                    'Дыхание':'Вдох' if fold['state']=='inhale' else 'Выдох','Исключённый L, мм':fold['size_mm'],
                    'h по остальным, мм':fold['h_mm'],
                    'Базовый остаток, Ом':fold['baseline_predicted_ohm']-fold['baseline_observed_ohm'],
                    'Пульсовая ошибка, %':fold['pulse_relative_error_percent'],
                    'Фазы с неположительным ρ в подгонке':fold['training_nonpositive_phase_count']})
    return pd.DataFrame(rows)


def context_figure():
    verify();ct=checked_ct_profile()
    fig=go.Figure(go.Scatter(x=ct['s_mm'],y=ct['depth_mm'],name='КТ: глубина до первого вокселя лёгкого'))
    for h in [100,120]:fig.add_hline(y=h,line_dash='dot',annotation_text=f'{h} мм')
    fig.update_layout(template='plotly_white',height=390,xaxis_title='Координата вдоль номинальной сборки, мм',
        yaxis_title='Глубина до лёгкого по +X LPS, мм')
    return fig


def profiles_figure():
    verify();fig=make_subplots(rows=1,cols=2,subplot_titles=['Базовая невязка','Обусловленность пульсового оператора: 004'])
    results=read(OUT/'results.local.json')
    for item in results:
        fig.add_trace(go.Scatter(x=[r['h_mm'] for r in item['profiles']],y=[r['score'] for r in item['profiles']],
            name='140 мм: '+item['reference_140_record']),row=1,col=1)
    main=next(r for r in results if r['reference_140_record']=='004')
    for state,label in [('inhale','Вдох'),('exhale','Выдох')]:
        fig.add_trace(go.Scatter(x=[r['h_mm'] for r in main['profiles']],
            y=[r['fits'][state]['condition'] for r in main['profiles']],name=label),row=1,col=2)
    fig.update_xaxes(title_text='Заданная эффективная h, мм')
    fig.update_yaxes(title_text='Сумма средних квадратов ошибок, Ом²',row=1,col=1)
    fig.update_yaxes(title_text='κ(J)',type='log',row=1,col=2)
    fig.update_layout(template='plotly_white',height=440,legend=dict(orientation='h'))
    return fig


def forecasts_figure():
    verify();main=next(r for r in read(OUT/'results.local.json') if r['reference_140_record']=='004')
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Глубокий вдох','Глубокий выдох'])
    for key in ['wide','ct_profile','deep_band','h100','h110','h120']:
        for col,state in enumerate(['inhale','exhale'],1):
            folds=[r for r in main['scenarios'][key]['withheld'] if r['state']==state]
            fig.add_trace(go.Scatter(x=[r['size_mm'] for r in folds],y=[r['pulse_relative_error_percent'] for r in folds],
                mode='lines+markers',name=SCENARIOS[key][0],legendgroup=key,showlegend=col==1),row=1,col=col)
    fig.update_xaxes(title_text='Исключённый размер, мм')
    fig.update_yaxes(title_text='Ошибка пульсовой формы, %',type='log')
    fig.update_layout(template='plotly_white',height=450,legend=dict(orientation='h',y=-.25))
    return fig


def cells():
    verify();cc,md,code=new_cells()
    md('''# Толщина до 120 мм: пересчёт боковых сборок Георгия

По уточнению автора от 12.10.2026 глубина тканей до лёгкого у Георгия
может достигать 100–120 мм. Профиль эффективной h расширен до 120 мм.
Отдельно рассчитаны диапазоны 40–120 и 100–120 мм, а также фиксированные
h = 100, 110 и 120 мм. Прежний диапазон 5–60 мм остаётся в исторических
33.20 и 33.24; их результаты сохранены.

Здесь используется прежняя плоская двуслойная модель с дисками 5 мм
и теми же границами сопротивлений при 100 кГц. Поэтому сравнение
показывает влияние выбора h в этой постановке. Обработка сигналов,
каналы и кандидатная разметка сохранены: 140 мм — BEI2, 120–60 мм — BEI1.
Повторные 140 мм при ТТРКГ 1×1, 2×2 и 1×4 рассмотрены раздельно.
Поздний вариант ТТРКГ по-прежнему не установлен однозначно.''')
    code('from IPython.display import display, HTML\nfrom lateral_thickness_extension import *\nverify()')
    md('''## Анатомический профиль и эффективная h

Сохранённый профиль КТ для номинальной сборки 140 мм даёт 41,5–100,5 мм,
медиана — 71 мм. Луч идёт от наружной поверхности до первого вокселя
лёгкого по +X LPS, при y = 0 и центре z = −210 мм. Это иллюстративное
положение, а не восстановленный фактический монтаж. Этот профиль
не устанавливает максимальную глубину во всех возможных положениях.

Неоднородная глубина по КТ и постоянная h плоского слоя имеют разный
смысл. Правило их сведения ещё не валидировано. Диапазон 40–120 мм
охватывает порядок глубин показанного профиля и авторское верхнее
значение; диапазон 100–120 мм и три фиксированные h — отдельные
сценарии больших толщин. Они не являются доверительными интервалами
или автоматически найденными индивидуальными параметрами.''')
    code('display(HTML(context_figure().to_html(full_html=False,include_plotlyjs=True)))')
    md('''## Базовая подгонка и чувствительность при увеличении h

В полном профиле h задана от 5 до 120 мм с шагом 5 мм. При каждой h
сопротивления на вдохе и выдохе подбираются отдельно. Общую h внутри
сценария выбирает минимум суммы квадратов двух среднеквадратичных ошибок.
Для фиксированных 100, 110 и 120 мм h не подбирается.

Нижние значения полного профиля позволяют сопоставить результат
с прежним расчётом. Если минимум остаётся малым, это не устанавливает
анатомическую толщину: отдельно оцениваются остаток модели, ограничения
сопротивлений и сопоставимость записей. κ(J) описывает локальную
обусловленность двух относительных тканевых изменений при фиксированной h.
Большая κ(J) означает усиление влияния ошибок; без модели ошибок она
не задаёт доверительные интервалы. Рост относительных δρ и появление
неположительных сопротивлений проверяются отдельно.''')
    code('display(HTML(profiles_figure().to_html(full_html=False,include_plotlyjs=False)))\ndisplay(table(summary_table()))')
    md('''## Прогноз размера, исключённого из подгонки

Для каждого сценария один размер полностью исключён из базовой и
пульсовой подгонки. По оставшимся четырём заново выбираются допустимая
h и сопротивления; две пульсовые функции прогнозируют исключённый размер.
При заданной h она остаётся фиксированной во всех шагах. Данные
исключённого размера не участвуют в выборе h.

Ошибка формы — норма остатка, делённая на норму исключённой медианной
кривой, в процентах. На графике показан основной ряд с записью 004;
таблица сохраняет все три варианта 140 мм. Кандидатные окна и QRS,
последовательная регистрация и перенос плоской модели сохраняют
неопределённость. Успех подгонки не валидирует тканевые источники.

Расчёт расширяет проверку толщины в аналитической ветке. В индивидуальной
FEM толщина уже задана пространственной геометрией КТ. Её отрицательные
численные проверки из 33.25 этим пересчётом не отменяются; совместные
ρ₁–ρ₄ и сердечная сфера здесь не определяются.''')
    code('display(HTML(forecasts_figure().to_html(full_html=False,include_plotlyjs=False)))\ndisplay(table(forecast_table()))')
    md('''## Результат расширения диапазона

В основном ряду с записью 004 свободный профиль 5–120 мм сохраняет
минимум при 5 мм. В сценариях 40–120 и 100–120 мм минимум находится
соответственно на нижних границах 40 и 100 мм. Эти минимумы не
заменяют геометрический профиль КТ.

При заданных 100–120 мм базовый остаток составляет примерно
17,6–18,7 Ом, а κ(J) возрастает примерно от 148 до 242. Оценка ρ₂
достигает верхней границы принятого литературного профиля. Поэтому
граничное число не следует считать определённым сопротивлением лёгкого.
Ошибка прогноза исключённой пульсовой кривой при этих фиксированных
толщинах составляет примерно 25–327%, в зависимости от размера и дыхания.

Большие толщины включены в расчёт, но не устранили несогласие этой
плоской постановки с размерным рядом. Полученный малый минимум
нельзя использовать как доказательство тонкой грудной стенки Георгия.
Следующий переход требует сопоставления неоднородной КТ-геометрии
и редуцированной модели с раздельной оценкой ошибок состояния,
разметки, положения и принятых сопротивлений.''')
    return cc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--calculate-only',action='store_true')
    args=parser.parse_args()
    if not args.export_only:run()
    if not args.calculate_only:export(STEM,cells(),OUT)
