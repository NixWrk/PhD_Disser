"""Графики пульсового Δρкаж Георгия и Ника; читатель готовых расчётов МГТУ."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
from bmstu_apparent_resistivity import read, sha, write_new
from bmstu_pulse_resistivity import pulse_statistics

NAMES={'georg':'Георгий','nik':'Ник'}
COLORS={'georg':'#2563eb','nik':'#d97706'}


def ru(x,n=3):
    return f'{x:.{n}f}'.replace('.',',')


def csv_file(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def build(pulse_path,check_paths,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    pulse_path=Path(pulse_path).resolve();p=read(pulse_path)
    if p['schema']!='bmstu_uniform_pulse_v1' or p['experiment_id']!='exp02':
        raise ValueError('Ожидается только пульсовой расчёт МГТУ')
    checks={read(cp)['subject_alias']:(Path(cp).resolve(),read(cp)) for cp in check_paths}
    if set(checks)!=set(NAMES) or [s['subject_alias'] for s in p['subjects']]!=list(NAMES):
        raise ValueError('Нужны Георгий и Ник')
    for item in p['protected_inputs']:
        if sha(item['path'])!=item['sha256']:
            raise ValueError('Изменён защищённый вход: '+item['path'])
    rows=[];cycle_rows=[];trace_rows=[];numerical=[]
    for subject in p['subjects']:
        alias=subject['subject_alias'];cp,check=checks[alias]
        if (check['schema']!='bmstu_uniform_pulse_checks_v1' or
            check['pulse_plan_sha256']!=sha(pulse_path) or
            check['baseline_sha256']!=subject['baseline_sha256']):
            raise ValueError('Численная проверка относится к другой версии пульса или базы')
        baseline=read(subject['baseline_path'])
        if check['identity']['mesh_sha256']!=baseline['identity']['mesh_sha256']:
            raise ValueError('Численная проверка использует другую анатомическую сетку')
        if len(check['answers'])!=len(subject['records']):
            raise ValueError('Неполная численная проверка')
        answers={a['record_id']:a for a in check['answers']}
        for r in subject['records']:
            a=answers[r['record_id']]
            if (not a['passed'] or a['size_mm']!=r['size_mm'] or
                abs(a['max_observed_abs_delta_Z_ohm']-r['max_abs_delta_Z_ohm'])>1e-12 or
                a['rho0_ohm_m']!=r['rho0_ohm_m']):
                raise ValueError('Линейное приближение не проверено для данного диапазона')
            stats=pulse_statistics([c['centered_native_mohm'] for c in r['cycles']],
                [c['candidate_ecg_baseline_flag'] for c in r['cycles']],r['rho0_ohm_m'],r['dZ_dlogrho_ohm'])
            for key in ('delta_rho_median_ohm_m','delta_rho_q25_ohm_m','delta_rho_q75_ohm_m',
                        'ensemble_peak_to_peak_ohm_m','max_abs_delta_Z_ohm'):
                if abs(stats[key]-r[key])>1e-14:
                    raise ValueError('Не воспроизводится пульсовая статистика')
            alt=r['without_candidate_ecg_flags'];num=r['delta_rho_median_ohm_m']
            rows.append(dict(subject_alias=alias,record_id=r['record_id'],size_mm=r['size_mm'],
                delta_rho_median_ohm_m=num,delta_rho_q25_ohm_m=r['delta_rho_q25_ohm_m'],
                delta_rho_q75_ohm_m=r['delta_rho_q75_ohm_m'],
                ensemble_peak_to_peak_ohm_m=r['ensemble_peak_to_peak_ohm_m'],
                delta_Z_median_ohm=r['delta_Z_median_ohm'],rho0_ohm_m=r['rho0_ohm_m'],
                Z0_ohm=r['Z0_ohm'],dZ_dlogrho_ohm=r['dZ_dlogrho_ohm'],
                n_cycles=r['n_cycles'],n_candidate_ecg_flags=r['n_candidate_ecg_flags'],
                n_excluded_plateau=r['n_excluded_plateau'],rr_median_s=r['rr_median_s'],
                window_start_s=r['window_s'][0],window_stop_s=r['window_s'][1],
                relative_peak_to_peak_percent=100*num/r['rho0_ohm_m'],
                without_ecg_flags_delta_rho_ohm_m=alt['delta_rho_median_ohm_m'] if alt else '',
                without_ecg_flags_difference_percent=100*(alt['delta_rho_median_ohm_m']/num-1) if alt else '',
                linearity_error_relative_to_max_pulse=a['max_linearity_error_relative_to_max_pulse'],
                finite_difference_relative_error=a['finite_difference_relative_error'],
                status=p['status']))
            for c,amplitude in zip(r['cycles'],r['cycle_peak_to_peak_ohm_m']):
                cycle_rows.append(dict(subject_alias=alias,record_id=r['record_id'],size_mm=r['size_mm'],
                    start_r_s=c['start_r_s'],stop_r_s=c['stop_r_s'],rr_s=c['rr_s'],
                    candidate_ecg_baseline_flag=c['candidate_ecg_baseline_flag'],
                    native_sample_peak_to_peak_mohm=c['native_sample_peak_to_peak_mohm'],
                    normalized_peak_to_peak_mohm=c['normalized_peak_to_peak_mohm'],
                    delta_rho_peak_to_peak_ohm_m=amplitude))
            for i,phase in enumerate(p['phase_rr']):
                trace_rows.append(dict(subject_alias=alias,record_id=r['record_id'],size_mm=r['size_mm'],
                    phase_RR=phase,median_delta_rho_ohm_m=r['median_wave_ohm_m'][i],
                    q25_delta_rho_ohm_m=r['q25_wave_ohm_m'][i],q75_delta_rho_ohm_m=r['q75_wave_ohm_m'][i]))
            numerical.append(a)
    if len(rows)!=19 or len(cycle_rows)!=sum(r['n_cycles'] for r in rows):
        raise ValueError('Неполный результат')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    csv_file(out/'delta_rho_bmstu.csv',rows);csv_file(out/'cycles_bmstu.csv',cycle_rows)
    csv_file(out/'median_waveforms_bmstu.csv',trace_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    limit=max(r['delta_rho_q75_ohm_m'] for r in rows)*1000*1.24
    for alias in NAMES:
        selected=[r for r in rows if r['subject_alias']==alias]
        color=COLORS[alias];fig,ax=plt.subplots(figsize=(9.4,6.6))
        fig.subplots_adjust(left=.12,right=.97,bottom=.29,top=.78)
        x=np.array([r['size_mm'] for r in selected]);y=np.array([r['delta_rho_median_ohm_m'] for r in selected])*1000
        low=np.array([r['delta_rho_q25_ohm_m'] for r in selected])*1000
        high=np.array([r['delta_rho_q75_ohm_m'] for r in selected])*1000
        ax.errorbar(x,y,yerr=[y-low,high-y],fmt='o-',lw=1.8,color=color,ms=6,
            capsize=4,label='Медиана размахов; квартили 25–75 %',zorder=3)
        flagged=[r for r in selected if r['n_candidate_ecg_flags'] and r['without_ecg_flags_delta_rho_ohm_m']!='']
        if flagged:
            ax.scatter([r['size_mm']+1.2 for r in flagged],
                [r['without_ecg_flags_delta_rho_ohm_m']*1000 for r in flagged],
                marker='s',s=48,facecolors='white',edgecolors='#b91c1c',zorder=4,
                label='Без циклов с кандидатной меткой ЭКГ')
        for i,(xx,yy) in enumerate(zip(x,y)):
            ax.annotate(ru(yy,2),(xx,high[i]),xytext=(0,8),textcoords='offset points',
                ha='center',fontsize=9,color=color)
        ax.set(xlim=(44,146),ylim=(0,limit),xticks=list(range(50,141,10)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'.replace('.',',')))
        ax.set_xlabel('Размер боковой сборки L, мм',labelpad=10)
        ax.set_ylabel('Пульсовой размах Δρкаж, 10⁻³ Ом·м',labelpad=10)
        ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9,frameon=False)
        fig.suptitle(NAMES[alias]+' · реокардиомонитор МГТУ',x=.12,y=.955,ha='left',fontsize=16,fontweight='bold')
        fig.text(.12,.885,'Задержка глубокого вдоха · индивидуальная КТ · однородная область',fontsize=10.5)
        footer='Полный размах max − min; квартили описывают разброс циклов одной записи.\nКаждый размер записан отдельно; положение сборки задано модельным сценарием.'
        footer+='\n100 мм исключены: копия записи 90 мм.' if alias=='nik' else '\nКандидатные метки ЭКГ проверены отдельным сравнением; исходная разметка сохранена.'
        fig.text(.12,.05,footer,fontsize=8.5,color='#4b5563',linespacing=1.6)
        for ext in ('png','svg'):
            fig.savefig(out/f'{alias}_delta_rho_bmstu.{ext}',dpi=220,facecolor='white')
        plt.close(fig)

        records=next(s['records'] for s in p['subjects'] if s['subject_alias']==alias)
        fig,axes=plt.subplots(2,5,figsize=(16,7.8),squeeze=False,sharex=True)
        fig.subplots_adjust(left=.06,right=.98,bottom=.16,top=.83,hspace=.4,wspace=.3)
        phase=np.array(p['phase_rr'])*100
        for ax,r in zip(axes.flat,records):
            factor=r['conversion_ohm_m_per_mohm']*1000
            for c in r['cycles']:
                ax.plot(phase,np.array(c['centered_native_mohm'])*factor,color='#9ca3af',alpha=.23,lw=.6)
            ax.fill_between(phase,np.array(r['q25_wave_ohm_m'])*1000,np.array(r['q75_wave_ohm_m'])*1000,color=color,alpha=.18)
            ax.plot(phase,np.array(r['median_wave_ohm_m'])*1000,color=color,lw=2)
            if r['n_candidate_ecg_flags'] and r['without_candidate_ecg_flags']:
                ax.plot(phase,np.array(r['without_candidate_ecg_flags']['median_wave_ohm_m'])*1000,
                    color='#b91c1c',ls='--',lw=1.2)
            ax.axhline(0,color='#6b7280',lw=.6);ax.grid(alpha=.15)
            ax.set_title(f"L = {r['size_mm']} мм; n = {r['n_cycles']}",fontsize=10)
            ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.tick_params(labelsize=8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'.replace('.',',')))
        for ax in list(axes.flat)[len(records):]:ax.set_visible(False)
        fig.suptitle(NAMES[alias]+': знаковое пульсовое δρкаж на глубоком вдохе',x=.06,y=.97,ha='left',fontsize=16)
        fig.text(.06,.895,'Серый — отдельные циклы; цвет — медиана и квартили; пунктир — без кандидатных меток ЭКГ.',fontsize=10)
        fig.text(.5,.09,'Фаза сердечного цикла от R до следующего R, %',ha='center',fontsize=11)
        fig.text(.013,.49,'δρкаж, 10⁻³ Ом·м',rotation=90,va='center',fontsize=11)
        fig.text(.06,.025,'Вычтена медиана каждого цикла. Сохранён знак RHEO₂; физический знак и передаточная характеристика тракта независимо не проверены.',fontsize=9,color='#4b5563')
        for ext in ('png','svg'):
            fig.savefig(out/f'{alias}_pulse_cycles_bmstu.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    verification=dict(status=p['status'],records=19,cycles=len(cycle_rows),
        pulse_plan_sha256=sha(pulse_path),protected_input_count=len(p['protected_inputs']),
        protected_inputs_unchanged=True,
        checks=[dict(subject_alias=k,path=str(v[0]),sha256=sha(v[0])) for k,v in checks.items()],
        max_linearity_error_relative_to_max_pulse=max(a['max_linearity_error_relative_to_max_pulse'] for a in numerical),
        max_finite_difference_relative_error=max(a['finite_difference_relative_error'] for a in numerical),
        max_33_03_reproduction_error_mohm=max(r['reproduction_33_03_max_error_mohm'] for s in p['subjects'] for r in s['records']),
        global_mesh_convergence_proven=False,physical_validation_completed=False,
        producer_sha256=sha(__file__))
    verification['artifacts']=[dict(name=f.name,sha256=sha(f)) for f in sorted(out.iterdir())]
    write_new(out/'verification.json',verification)
    (out/'README.md').write_text(report(rows,verification),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ('checks','artifacts')},ensure_ascii=False),flush=True)


def report(rows,v):
    lines=['# Пульсовой размах кажущегося ρ: МГТУ, Георгий и Ник','',
        'Рассчитан полный размах эквивалентного однородного удельного сопротивления '+
        'на задержке глубокого вдоха. Для каждого участника и размера использованы '+
        'собственная КТ/FEM-геометрия, базовый ρ₀ и индивидуальная чувствительность. '+
        'Результаты условны относительно геометрического и приборного сценариев; '+
        'свойства отдельных тканей и изменение объёма крови не восстанавливаются.','',
        '## Метод и единицы','',
        'Сигнал RHEO₂ переведён из мОм в Ом без подбора усиления или изменения знака. '+
        'Использованы принятые дыхательные интервалы и R-метки. Полные циклы взяты '+
        'из 12.01 по сохранённому предварительному решению автора для 33.03: '+
        'циклы с экстремальными плато исключаются; кандидатные метки ЭКГ остаются '+
        'в основном наборе и проверяются отдельным сравнением. QC 12.01 остаётся '+
        'непринятым окончательно.','',
        'Каждый цикл интерполирован на 200 точек интервала R–R без включения следующего '+
        'R-зубца. Вычтена постоянная медиана; дополнительная фильтрация и удаление '+
        'наклонного тренда не выполнялись. Формы независимо воспроизведены по 33.03.','',
        r'\[\delta\rho_i(\phi)=\rho_0\,\delta Z_i(\phi)/S_{\ln\rho},\qquad S_{\ln\rho}=\partial Z_{\mathrm{CEM}}/\partial\ln\rho.\]','',
        r'\[A_i=\max_\phi\delta\rho_i-\min_\phi\delta\rho_i,\qquad\Delta\rho_{\mathrm{каж,пульс}}=\operatorname{median}_i A_i.\]','',
        'Основной результат — медиана размахов отдельных циклов. Квартили описывают '+
        'внутрикривую изменчивость; это не доверительный интервал полной погрешности. '+
        'Размах медианной формы сохранён отдельно в CSV.','',
        '## Результаты','',
        '| Доброволец | L, мм | Циклов | С меткой ЭКГ | Δρ, 10⁻³ Ом·м | Квартили 25–75 %, 10⁻³ Ом·м |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {NAMES[r['subject_alias']]} | {r['size_mm']} | {r['n_cycles']} | {r['n_candidate_ecg_flags']} | {ru(r['delta_rho_median_ohm_m']*1000)} | {ru(r['delta_rho_q25_ohm_m']*1000)}–{ru(r['delta_rho_q75_ohm_m']*1000)} |")
    lines+=['',f"Всего 19 независимых записей и {v['cycles']} полных циклов. "+
        'У Ника 100 мм исключены как копия 90 мм. В принятых для вдоха циклах '+
        '12.01 не отмечает перекрытия экстремальных плато. Это не снимает остальные ограничения сигнала.','']
    for alias in NAMES:
        selected=[r for r in rows if r['subject_alias']==alias]
        lo=min(r['delta_rho_median_ohm_m'] for r in selected)*1000
        hi=max(r['delta_rho_median_ohm_m'] for r in selected)*1000
        peak=max(selected,key=lambda r:r['delta_rho_median_ohm_m'])
        lines+=[f"Участник {NAMES[alias]}: медианный пульсовой размах составляет {ru(lo)}–{ru(hi)} × 10⁻³ Ом·м. "+
            f"Наибольшее значение в имеющемся ряду получено при L = {peak['size_mm']} мм. "+
            'Оно не устанавливает оптимальный размер сборки.','',
            f"![{NAMES[alias]}: зависимость от размера]({alias}_delta_rho_bmstu.png)",'',
            f"![{NAMES[alias]}: знаковые циклы]({alias}_pulse_cycles_bmstu.png)",'']
    flagged=[r for r in rows if r['n_candidate_ecg_flags']]
    worst=max(flagged,key=lambda r:abs(r['without_ecg_flags_difference_percent']))
    lines+=['## Численные проверки и ограничения','',
        f"Прямая CEM-проверка выполнена для всех 19 записей в двух точках, "+
        'ограничивающих линейно оценённый диапазон пульсового ρ. '+
        f"Максимальное расхождение линейного и прямого отклика — {v['max_linearity_error_relative_to_max_pulse']*100:.6g} % "+
        'наибольшего |δZ|; допустимое значение — 0,1 %. '+
        f"Максимальное расхождение центральной разности и сопряжённой производной — {v['max_finite_difference_relative_error']*100:.6g} % "+
        '(допуск 0,01 %). Численные критерии выполнены; они не подтверждают физическую точность.','',
        f"Наибольшее изменение медианного размаха при исключении кандидатных меток ЭКГ: "+
        f"{NAMES[worst['subject_alias']]}, {worst['size_mm']} мм, {ru(worst['without_ecg_flags_difference_percent'],2)} %. "+
        'Это чувствительность к отбору циклов, а не оценка полной погрешности.','',
        'Положение сборки у Ника реконструировано по центру и углу C01 на его собственной КТ. '+
        'У Георгия использовано прежнее модельное место на собственной КТ согласно решению автора. '+
        'Фактическое положение наклейки не подтверждено. Контактный параметр '+
        'и диаметр конечных дисков 5 мм являются модельным сценарием; '+
        'межканальное влияние не моделируется. Глобальная сеточная сходимость не установлена.','',
        'Каждый размер записан последовательно, поэтому зависимость от L содержит также '+
        'межзаписную изменчивость. Передаточная характеристика тракта, физический знак RHEO₂ '+
        'и отсутствие приборного ЭКГ-синхронного вклада независимо не проверены. '+
        'При фиксированной геометрии электрический эффект движения может отразиться '+
        'в кажущемся ρ. Эти результаты не являются оценками УО, ФВ или сопротивления отдельной ткани.','',
        '## Сохранённые материалы','',
        '- [Основная таблица](delta_rho_bmstu.csv): значения в Ом·м, квартили, чувствительности, базовые уровни и численные проверки.',
        '- [Все циклы](cycles_bmstu.csv): границы, R–R, метки и размахи исходных/нормированных сигналов.',
        '- [Медианные формы](median_waveforms_bmstu.csv): фаза R–R, медиана и квартили в Ом·м.',
        '- [Паспорт проверки](verification.json); рисунки сохранены в PNG и SVG.','',
        'Полные знаковые формы отдельных циклов и защищённые пути к источникам находятся '+
        'в локальном `MATLAB_TRKG4_real_subjects/output/exploratory/bmstu_pulse_20261107/inputs/pulse.local.json`. '+
        'Там же, в каталогах `georg_check` и `nik_check`, находятся протоколы прямого CEM-расчёта. '+
        'Первичные сигналы, разметка и прежний базовый расчёт не изменены. '+
        'Метод и команды воспроизведения приведены в `docs/BMSTU_PULSE_RESISTIVITY_METHOD.md`.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pulse',type=Path,required=True)
    parser.add_argument('--checks',type=Path,nargs=2,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();build(args.pulse,args.checks,args.out)
