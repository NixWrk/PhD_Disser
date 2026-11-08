"""Графики и таблицы пульсового кажущегося ρ для трёх участников серии «Скорпион»."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'MATLAB_TRKG4_real_subjects/tools'))
from scorpion_apparent_resistivity import read, sha

NAMES = {'georg': 'Георгий', 'nik': 'Ник', 'yaroslav': 'Ярослав'}
PRIMARY = {'georg': '1x1', 'nik': '2x2', 'yaroslav': '2x2'}
VARIANTS = {'1x1': ('o','#2563eb'), '2x2': ('s','#be5908'), '1x4': ('^','#168361')}
F1_LIMIT = .65


def weak_qrs(record):
    return record['in_window_QRS_pair_match']['F1'] < F1_LIMIT


def save_figure(fig, out, stem):
    for ext in ('png','svg'):
        fig.savefig(out / f'{stem}.{ext}', dpi=195, facecolor='white')
    plt.close(fig)


def decimal(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def plot_subject(alias, records, phase, upper, out):
    fig, ax = plt.subplots(figsize=(9.4,5.9))
    fig.subplots_adjust(left=.13,right=.965,top=.79,bottom=.25)
    primary = [r for r in records if r['ttrkg_variant']==PRIMARY[alias]]
    ax.plot([r['size_mm'] for r in primary],
            [np.nan if weak_qrs(r) else 1000*r['delta_rho_cycle_q25_median_q75_ohm_m'][1] for r in primary],
            color=VARIANTS[PRIMARY[alias]][1],lw=1.5,zorder=2)
    handles = []
    for variant, (marker, color) in VARIANTS.items():
        series = [r for r in records if r['ttrkg_variant']==variant]
        if not series:
            continue
        handles.append(Line2D([],[],color=color,marker=marker,ls='none',label='ТТРКГ '+variant.replace('x','×')))
        for r in series:
            lo, med, hi = np.array(r['delta_rho_cycle_q25_median_q75_ohm_m'])*1000
            x = r['size_mm']
            is_weak = weak_qrs(r)
            ax.errorbar(x,med,yerr=[[med-lo],[hi-med]],fmt=marker,color=color,
                        markerfacecolor='white' if is_weak else color,markersize=7,
                        markeredgewidth=1.5,capsize=4,lw=1.35,zorder=3)
            offsets = {'1x1':(0,8),'2x2':(0,8),'1x4':(-14,-15)}
            if len([s for s in records if s['size_mm']==x])>1:
                offsets = {'1x1':(15,-14),'2x2':(-13,10),'1x4':(-14,-14)}
                if alias=='nik' and variant=='1x1':
                    offsets['1x1']=(16,9)
                if alias=='georg' and variant=='1x1':
                    offsets['1x1']=(13,6)
            ax.annotate(decimal(med),(x,hi if offsets[variant][1]>0 else lo),xytext=offsets[variant],
                        textcoords='offset points',ha='center',color=color,fontsize=9)
    handles.append(Line2D([],[],color='#4b5563',marker='o',markerfacecolor='white',ls='none',label='Слабое согласие QRS'))
    ax.legend(handles=handles,loc='upper center',frameon=False,ncol=2,fontsize=9)
    ax.set(xlim=(52,151),ylim=(0,upper),xticks=sorted({r['size_mm'] for r in records}))
    ax.set_xlabel('Размер боковой сборки L, мм')
    ax.set_ylabel(r'Пульсовой размах $\Delta\rho_{\mathrm{каж}}$, 10⁻³ Ом·м',labelpad=9)
    ax.grid(alpha=.19)
    fig.suptitle(NAMES[alias]+' · «Скорпион», 100 кГц',x=.13,y=.95,ha='left',fontsize=16,fontweight='bold')
    fig.text(.13,.855,'Задержка глубокого вдоха по кандидатной разметке · индивидуальная КТ',fontsize=10.5)
    fig.text(.13,.047,'Точки — медианы полных размахов, отрезки — квартили 25 и 75 %.\n'
             'Пустой маркер означает F₁ < 0,65 для совпадения QRS двух каналов в окне вдоха.\n'
             'Один однородный параметр ρ для всего объёма, отдельная запись каждого размера.',
             fontsize=8.7,color='#4b5563',linespacing=1.55)
    save_figure(fig,out,alias+'_delta_rho_scorpion')

    columns=min(len(records),4)
    nrows=int(np.ceil(len(records)/columns))
    fig, axes=plt.subplots(nrows,columns,figsize=(3.45*columns,3.1*nrows+1.2),squeeze=False,sharex=True)
    fig.subplots_adjust(left=.074,right=.981,top=.79 if nrows==1 else .83,bottom=.18 if nrows==1 else .14,
                        hspace=.45,wspace=.30)
    for ax,r in zip(axes.flat,records):
        color=VARIANTS[r['ttrkg_variant']][1]
        factor=1000*r['rho0_ohm_m']/r['dZ_dlogrho_ohm']
        for c in r['cycles']:
            ax.plot(phase,np.array(c['centered_Z_ohm'])*factor,color='#939ca8',alpha=.32,lw=.62)
        med=np.array(r['median_wave_ohm_m'])*1000
        ax.fill_between(phase,np.array(r['q25_wave_ohm_m'])*1000,np.array(r['q75_wave_ohm_m'])*1000,color=color,alpha=.18)
        ax.plot(phase,med,color=color,lw=1.8)
        ax.axhline(0,color='#687385',lw=.65)
        label=f"{r['size_mm']} мм · {r['ttrkg_variant'].replace('x','×')} · n = {r['n_cycles']}"
        ax.set_title(label + ('  *' if weak_qrs(r) else ''),fontsize=9.7)
        ax.set(xlim=(0,100),xticks=(0,50,100))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f'{y:g}'.replace('.',',')))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.14)
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    fig.suptitle(NAMES[alias]+' · знаковые пульсовые кривые «Скорпиона»',x=.074,y=.96,ha='left',fontsize=15,fontweight='bold')
    fig.text(.074,.885,'Серые линии — отдельные циклы. Цветом показаны медиана и квартили.',fontsize=9.8)
    fig.supxlabel('Фаза интервала между соседними QRS, %',y=.085,fontsize=10)
    fig.supylabel(r'$\delta\rho_{\mathrm{каж}}$, 10⁻³ Ом·м',x=.014,fontsize=10)
    fig.text(.074,.027,'* Согласие QRS двух каналов F₁ < 0,65. Дыхательные окна и QRS сохраняют кандидатный статус.',fontsize=8.3)
    save_figure(fig,out,alias+'_pulse_cycles_scorpion')


def write_csv(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(pulse_path, check_path, out):
    pulse, checks = read(pulse_path), read(check_path)
    if (pulse['schema']!='scorpion_uniform_pulse_v1' or checks['schema']!='scorpion_uniform_pulse_checks_v1' or
        checks['pulse_plan_sha256']!=sha(pulse_path) or checks['baseline_sha256']!=pulse['baseline_sha256']):
        raise ValueError('Чужая численная проверка')
    if any(sha(p)!=h for p,h in pulse['protected_inputs'].items()):
        raise ValueError('Изменились защищённые входы')
    by_id = {a['record_id']:a for a in checks['answers']}
    records = sorted(pulse['records'],key=lambda r:(list(NAMES).index(r['subject_alias']),r['size_mm'],r['ttrkg_variant']))
    if len(records)!=15 or set(by_id)!={r['record_id'] for r in records}:
        raise ValueError('Неполный размерный ряд')
    rows, cycles, waveforms = [], [], []
    for r in records:
        rid, alias, size, variant = (r[k] for k in ('record_id','subject_alias','size_mm','ttrkg_variant'))
        q = by_id[rid]
        if (not q['passed'] or q['case_id']!=r['case_id'] or q['subject_alias']!=alias or
            abs(q['max_observed_abs_delta_Z_ohm']-r['max_abs_delta_Z_ohm'])>1e-12):
            raise ValueError('Нет численной проверки для '+rid)
        lo, med, hi = r['delta_rho_cycle_q25_median_q75_ohm_m']
        factor = r['rho0_ohm_m']/r['dZ_dlogrho_ohm']
        peak_to_peak = [float(np.ptp(c['centered_Z_ohm'])*factor) for c in r['cycles']]
        np.testing.assert_allclose(np.quantile(peak_to_peak,[.25,.5,.75]),[lo,med,hi],atol=1e-14,rtol=0)
        rows.append(dict(subject_alias=alias,record_id=rid,size_mm=size,ttrkg_variant=variant,
            delta_rho_median_ohm_m=med,delta_rho_q25_ohm_m=lo,delta_rho_q75_ohm_m=hi,
            median_wave_peak_to_peak_ohm_m=r['delta_rho_median_wave_span_ohm_m'],
            delta_Z_median_ohm=r['delta_Z_cycle_median_ohm'],Z0_ohm=r['Z_ohm'],rho0_ohm_m=r['rho0_ohm_m'],
            dZ_dlogrho_ohm=r['dZ_dlogrho_ohm'],n_cycles=r['n_cycles'],n_excluded=r['n_excluded'],
            BEI_channel=r['channel'],ECG_channel=r['ecg_reference_channel'],fs_hz=r['fs_hz'],
            window_start_sample=r['window_samples_half_open'][0],window_stop_sample=r['window_samples_half_open'][1],
            QRS_agreement_F1=r['in_window_QRS_pair_match']['F1'],weak_QRS_agreement=weak_qrs(r),
            linearity_error_relative=q['max_linearity_error_relative_to_max_pulse'],
            derivative_error_relative=q['finite_difference_relative_error'],
            breathing_status=r['annotation_status'],QRS_status=r['qrs_status'],status=pulse['status']))
        for c,p2p in zip(r['cycles'],peak_to_peak):
            cycles.append(dict(record_id=rid,start_sample=c['start_sample'],stop_sample=c['stop_sample'],rr_s=c['rr_s'],
                native_peak_to_peak_Z_ohm=c['native_peak_to_peak_Z_ohm'],delta_rho_peak_to_peak_ohm_m=p2p))
        for i,phase in enumerate(pulse['phase_rr']):
            waveforms.append(dict(record_id=rid,phase_RR=phase,delta_rho_median_ohm_m=r['median_wave_ohm_m'][i],
                delta_rho_q25_ohm_m=r['q25_wave_ohm_m'][i],delta_rho_q75_ohm_m=r['q75_wave_ohm_m'][i]))
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'delta_rho_scorpion.csv',rows)
    write_csv(out/'cycles_scorpion.csv',cycles)
    write_csv(out/'median_waveforms_scorpion.csv',waveforms)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    upper=1250*max(r['delta_rho_q75_ohm_m'] for r in rows)
    for alias in NAMES:
        plot_subject(alias,[r for r in records if r['subject_alias']==alias],np.array(pulse['phase_rr'])*100,upper,out)
    audit = dict(status=pulse['status'],records=len(rows),cycles=len(cycles),
        weak_qrs_records=[r['record_id'] for r in rows if r['weak_QRS_agreement']],
        pulse_sha256=sha(pulse_path),checks_sha256=sha(check_path),protected_inputs_unchanged=len(pulse['protected_inputs']),
        max_linearity_error_relative=max(r['linearity_error_relative'] for r in rows),
        max_derivative_error_relative=max(r['derivative_error_relative'] for r in rows),
        max_filter_reproduction_error_ohm=max(r['filter_reproduction_max_error_ohm'] for r in records),
        artifacts={f.name:sha(f) for f in sorted(out.iterdir())})
    (out/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    report = ['# Пульсовое кажущееся Δρ по «Скорпиону»','',
        'Задержка глубокого вдоха по кандидатной разметке. Боковой BEI в полосе '
        '0,7–12 Гц пересчитан по индивидуальной однородной КТ-модели каждого добровольца.', '',
        f"Обработано {audit['cycles']} полных циклов в {audit['records']} записях. "
        'Основная величина — медиана полных размахов отдельных циклов.', '',
        '| Доброволец | L, мм | ТТРКГ | Циклов | Δρ, 10⁻³ Ом·м | Квартили 25–75 %, 10⁻³ Ом·м | F₁ ЭКГ |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lo, med, hi = [decimal(1000*r['delta_rho_'+key+'_ohm_m']) for key in ('q25','median','q75')]
        agreement = decimal(r['QRS_agreement_F1']) + (' *' if r['weak_QRS_agreement'] else '')
        report.append(f"| {NAMES[r['subject_alias']]} | {r['size_mm']} | {r['ttrkg_variant'].replace('x','×')} | "
                      f"{r['n_cycles']} | {med} | {lo}–{hi} | {agreement} |")
    georg60 = next(r for r in rows if r['subject_alias']=='georg' and r['size_mm']==60)
    ratio = 100*georg60['median_wave_peak_to_peak_ohm_m']/georg60['delta_rho_median_ohm_m']
    report.extend(['', '* F₁ < 0,65 в окне вдоха. Для Георгия при 60 мм размах '
        f'медианной формы составляет {decimal(ratio,1)} % медианы отдельных размахов. '
        'Эта точка сохраняет статус диагностической оценки.', '',
        'Линейный пересчёт проверен двумя прямыми решениями для каждой записи. '
        'Положение контактов задано моделью. Квартили описывают изменчивость '
        'циклов в пределах записи. Источники пульсового отклика остаются неразделёнными.', ''])
    for alias,name in NAMES.items():
        report.extend([f"- {name} — [размах от размера]({alias}_delta_rho_scorpion.png) и "
                       f"[знаковые циклы]({alias}_pulse_cycles_scorpion.png)."])
    report.extend(['', '[Таблица размеров](delta_rho_scorpion.csv) · [циклы](cycles_scorpion.csv) · '
        '[медианные формы](median_waveforms_scorpion.csv) · [протокол проверок](verification.json).', ''])
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='artifacts'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pulse',type=Path,required=True)
    parser.add_argument('--checks',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build(args.pulse,args.checks,args.out)
