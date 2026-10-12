"""Corrected measured channels; three 140-mm controls and withheld-size checks."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lateral_pilot import (ROOT,EXPERIMENT,prepare,DiskModel,fit_state,fit_shared,
    invert_pulse,save,read,sha,STATUS)
from scorpion_protocol import channel_assignment

OUT=EXPERIMENT/'derived/exploratory/33.24_corrected_lateral_20261012'
STEM='33.24_Пересчёт_боковых_сборок_и_проверка_прогноза_Георгия_Scorpion_v2'


def fit_grid(observed,sizes,bounds,h_grid):
    profiles=[];best=None;best_score=np.inf
    for h in h_grid:
        model=DiskModel(sizes,float(h))
        fits={state:fit_state(model,y,bounds) for state,y in observed.items()}
        score=sum(f['rmse_ohm']**2 for f in fits.values())
        row=dict(h_mm=float(h),fits=fits,score=score)
        profiles.append(row)
        if score<best_score:best_score=score;best=row
    return profiles,best


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,ensembles,inputs=prepare(EXPERIMENT)
    registry_path=ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    registry=read(registry_path);profile=registry['screening_profiles']['100000']
    bounds=np.array([profile['compartments'][name]['rho_bounds_ohm_m'] for name in ('soft','lung_unspecified')])
    h_grid=np.arange(5.,61.,5.)
    results=[]
    for reference in ('002','003','004'):
        ids=['SCV2-G-'+r for r in ('010','009','008','007',reference)]
        sizes=np.array([60.,80.,100.,120.,140.])
        channels=[1,1,1,1,2]
        observed={};waves={};counts={}
        for state in ('inhale','exhale'):
            observed[state]=np.array([frame[(frame.record_id==rid)&(frame.state==state)&(frame.channel==ch)].baseline_raw_median_ohm.iloc[0]
                for rid,ch in zip(ids,channels)])
            source=[ensembles[rid]['windows'][state]['channels'][str(ch)] for rid,ch in zip(ids,channels)]
            counts[state]=[e['n_cycles'] for e in source]
            waves[state]=np.array([e['median_ohm'] for e in source]) if all(e['n_cycles']>=3 for e in source) else None
        profiles,best=fit_grid(observed,sizes,bounds,h_grid)
        model=DiskModel(sizes,best['h_mm'])
        shared=fit_shared(model,observed['inhale'],observed['exhale'],bounds)
        pulses={state:invert_pulse(model,best['fits'][state],wave) for state,wave in waves.items() if wave is not None}
        withheld=[]
        for i,size in enumerate(sizes):
            use=np.arange(len(sizes))!=i
            _,fold=fit_grid({state:y[use] for state,y in observed.items()},sizes[use],bounds,h_grid)
            all_model=DiskModel(sizes,fold['h_mm'])
            train_model=DiskModel(sizes[use],fold['h_mm'])
            for state,wave in waves.items():
                base=fold['fits'][state];rho=np.array([base['rho1_ohm_m'],base['rho2_ohm_m']])
                prediction,j=all_model.evaluate(rho)
                row=dict(size_mm=float(size),record_id=ids[i],state=state,h_mm=fold['h_mm'],
                    rho_ohm_m=rho.tolist(),
                    training_sizes_mm=sizes[use].tolist(),baseline_observed_ohm=float(observed[state][i]),
                    baseline_predicted_ohm=float(prediction[i]),baseline_residual_ohm=float(prediction[i]-observed[state][i]),
                    training_bound_hit=base['bound_hit'])
                if wave is not None:
                    pulse=invert_pulse(train_model,base,wave[use])
                    fraction=np.array(pulse['delta_rho_ohm_m'])/rho[:,None]
                    forecast=j[i]@fraction;truth=wave[i]
                    row.update(pulse_observed_ohm=truth.tolist(),pulse_predicted_ohm=forecast.tolist(),
                        pulse_fractional_changes=fraction.tolist(),
                        pulse_relative_error_percent=float(100*np.linalg.norm(forecast-truth)/np.linalg.norm(truth)) if np.linalg.norm(truth) else None)
                withheld.append(row)
        results.append(dict(reference_140_record=reference,record_ids=ids,channels=channels,sizes_mm=sizes.tolist(),
            profiles=profiles,best_on_grid=best,shared_rho1=shared,pulse=pulses,n_cycles=counts,withheld=withheld))
        print('Completed corrected row with 140-mm record',reference,flush=True)
    save(OUT/'results.local.json',results)
    save(OUT/'contract.local.json',dict(status=STATUS,owner='33.24',input_contract=inputs,
        producer_sha256=sha(__file__),kernel_sha256=sha(Path(__file__).with_name('lateral_pilot.py')),
        protocol_sha256=sha(Path(__file__).with_name('scorpion_protocol.py')),registry_sha256=sha(registry_path),
        output_sha256=sha(OUT/'results.local.json'),h_grid_mm=h_grid.tolist(),bounds_ohm_m=bounds.tolist(),
        corrected_channels={'140':2,'120':1,'100':1,'80':1,'60':1},
        ttrkg_variants_do_not_enter_planar_operator=True,late_ttrkg_variants=['1x1','1x4'],
        individual_CT_FEM_used=False,annotations_accepted=False,
        withheld_rule='Refit h and baseline resistivities using four sizes; fit pulse functions using only those four; predict withheld fifth size',
        is_independent_external_validation=False))
    return results


def verify():
    c=read(OUT/'contract.local.json')
    for path,key in [(Path(__file__),'producer_sha256'),(Path(__file__).with_name('lateral_pilot.py'),'kernel_sha256'),
            (Path(__file__).with_name('scorpion_protocol.py'),'protocol_sha256'),
            (ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json','registry_sha256'),
            (OUT/'results.local.json','output_sha256')]:
        if sha(path)!=c[key]:raise ValueError('Corrected pilot dependency changed: '+str(path))
    for path,key in [(EXPERIMENT/'derived/scorpion_review/summary.local.json','review_summary_sha256'),
                     (EXPERIMENT/'annotations/scorpion_review/annotations_candidates.local.json','annotations_sha256')]:
        if sha(path)!=c['input_contract'][key]:raise ValueError('Corrected pilot annotation input changed')
    return c


def summary_table():
    verify();rows=[]
    for item in read(OUT/'results.local.json'):
        best=item['best_on_grid']
        for state,base in best['fits'].items():
            pulse=item['pulse'].get(state,{})
            rows.append({'140 мм: запись':item['reference_140_record'],'Состояние':'Вдох' if state=='inhale' else 'Выдох',
                'h сетки, мм':best['h_mm'],'ρ₁, Ом·м':base['rho1_ohm_m'],'ρ₂, Ом·м':base['rho2_ohm_m'],
                'Базовая ошибка, Ом':base['rmse_ohm'],'Край диапазона ρ':base['bound_hit'],
                'Пульсовый остаток, %':pulse.get('residual_percent'),
                'Ошибка линеаризации, %':pulse.get('linearization_relative_error_percent'),
                'Фазы с неположительным ρ':pulse.get('nonpositive_resistivity_phase_count'),
                'Максимум |δρ/ρ|':pulse.get('max_abs_fractional_change'),
                'Обусловленность J':pulse.get('condition')})
    return pd.DataFrame(rows)


def profile_figure():
    verify();fig=go.Figure()
    for item in read(OUT/'results.local.json'):
        fig.add_trace(go.Scatter(x=[p['h_mm'] for p in item['profiles']],y=[p['score'] for p in item['profiles']],
            mode='lines+markers',name='140 мм: '+item['reference_140_record']))
    fig.update_layout(template='plotly_white',height=450,xaxis_title='Заданная эффективная h, мм',
        yaxis_title='Сумма квадратов двух среднеквадратичных ошибок, Ом²')
    return fig


def pulse_figure(withheld=False):
    verify();item=next(r for r in read(OUT/'results.local.json') if r['reference_140_record']=='004')
    fig=make_subplots(rows=2,cols=5,subplot_titles=[f'{state} · {s:g} мм' for state in ('Вдох','Выдох') for s in item['sizes_mm']])
    for row,state in enumerate(('inhale','exhale'),1):
        for col,size in enumerate(item['sizes_mm'],1):
            if withheld:
                fold=next(r for r in item['withheld'] if r['state']==state and r['size_mm']==size)
                if 'pulse_predicted_ohm' not in fold:continue
                observed,forecast=[fold[key] for key in ('pulse_observed_ohm','pulse_predicted_ohm')]
            else:
                pulse=item['pulse'].get(state)
                if not pulse:continue
                observed,forecast=[pulse[key][col-1] for key in ('observed_ohm','predicted_ohm')]
            for name,y,color,dash in [('Измеренная медиана',observed,'#176aaf','solid'),('Прогноз исключённого размера' if withheld else 'Подгонка всех пяти',forecast,'#d97706','dash')]:
                fig.add_trace(go.Scatter(x=np.arange(200)/200,y=1000*np.array(y),name=name,legendgroup=name,
                    showlegend=row==col==1,line=dict(color=color,dash=dash)),row=row,col=col)
    fig.update_layout(template='plotly_white',height=670,margin=dict(t=80),legend=dict(orientation='h'))
    fig.update_xaxes(title_text='Фаза QRS–QRS');fig.update_yaxes(title_text='мОм')
    return fig


def withheld_table():
    verify();rows=[]
    for item in read(OUT/'results.local.json'):
        for fold in item['withheld']:
            rows.append({'140 мм: запись':item['reference_140_record'],'Исключённый L, мм':fold['size_mm'],
                'Состояние':'Вдох' if fold['state']=='inhale' else 'Выдох',
                'h по остальным, мм':fold['h_mm'],'Базовый остаток, Ом':fold['baseline_residual_ohm'],
                'Пульсовая ошибка, %':fold.get('pulse_relative_error_percent')})
    return pd.DataFrame(rows)


def cells():
    from build_georg_fem_reports import new_cells
    cells,md,code=new_cells()
    md('''# Исправленный расчёт боковых сборок Георгия и проверка исключённого размера

Исследовательский пересчёт после уточнения каналов. Для 140 мм используется
BEI2, для 120–60 мм — BEI1. Три записи 140 мм рассматриваются раздельно:
при ТТРКГ 1×1, 2×2 и 1×4. Это проверка межзаписной устойчивости, а не
три новых размера. Нумерация каналов подтверждена для начала ряда и 007;
для 008–010 сохраняется допущение продолжения схемы.

Использована прежняя плоская двухслойная модель с дисками 5 мм,
равномерным током и усреднённым потенциалом. Электроды ТТРКГ в её оператор
не входят: варианты 1×1 и 1×4 позднего блока не меняют этот прямой расчёт.
Их влияние требует отдельного FEM с полным набором контактов.
Дыхательные окна и QRS взяты из 11.20 без повторной разметки и принятия.

**Численный результат.** Во всех трёх вариантах 140 мм минимум попадает
на нижнюю границу h = 5 мм. Среднеквадратичный базовый остаток составляет
около 10 Ом, пульсовый — 43–48%. В основном ряду ошибка прогноза
исключённой пульсовой кривой составляет примерно 39–304%, в зависимости
от размера и состояния. Поэтому найденные ρ и h нельзя передавать
как определённые параметры в коррекцию ТТРКГ. Это отрицательная
диагностика устойчивости данной постановки; вклад ошибок разметки,
сопоставимости состояния и расхождения плоской модели ещё не разделён.

[Пошаговое пояснение модели и пульсовой инверсии](33.20_Пояснение_расчёта_и_изменений_12_10_2026.md).''')
    code('from IPython.display import display, HTML\nfrom lateral_pilot_revision import *\nfrom build_georg_fem_reports import table\nverify()\ndisplay(table(summary_table()))')
    md('''## Профиль эффективной толщины

Для каждого h = 5–60 мм отдельно подбираются сопротивления на вдохе
и выдохе. Лучший h минимизирует сумму квадратов двух среднеквадратичных
остатков на данной сетке. Литературные границы 100 кГц остаются сценариями;
граничные значения и минимум h не объявляются индивидуальной оценкой.''')
    code("display(HTML(profile_figure().to_html(full_html=False,include_plotlyjs=True)))")
    md('''## Пульсовые кривые при подгонке пяти размеров

Ниже показан ряд с основной записью 140 мм при ТТРКГ 1×4. На каждой
фазе общие относительные изменения ρ₁ и ρ₂ определены по всем пяти
кривым. Это воспроизведение входов подгонки; малый остаток не является
независимой проверкой источников. Базовая точка, h, контакт и геометрия
остаются условными. Сводная таблица также сохраняет проверку
положительности сопротивлений и нелинейного отклика.''')
    code('display(HTML(pulse_figure().to_html(full_html=False,include_plotlyjs=False)))')
    md('''## Прогноз для исключённого размера

Каждый размер последовательно исключён из базовой и пульсовой подгонки.
По остальным четырём заново выбран h и найдены базовые сопротивления;
затем определены две пульсовые функции. Они прогнозируют пятый размер,
который не участвовал в этих оценках. Ошибка пульсовой формы — отношение
нормы остатка к норме исключённой медианной кривой, в процентах.

Эта проверка строже сравнения данных подгонки. Она оценивает перенос
внутри последовательного ряда одного испытуемого и сохраняет зависимость
от общего физиологического состояния, выбора ЭКГ и кандидатных окон.
Независимая анатомическая или экспериментальная валидация не выполнена.''')
    code('display(HTML(pulse_figure(withheld=True).to_html(full_html=False,include_plotlyjs=False)))\ndisplay(table(withheld_table()))')
    return cells


if __name__=='__main__':
    import argparse
    from build_georg_fem_reports import export
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calculate-only',action='store_true')
    args=parser.parse_args()
    run()
    if not args.calculate_only:export(STEM,cells(),OUT)
