"""Critical channel-assignment diagnostics; never rewrites accepted input roles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bei_breathing_overview import OUT as BEI, verify as verify_bei
from lateral_pilot import EXPERIMENT, read, save, sha
from build_georg_fem_reports import export, table

FOLDER = Path(__file__).parent
OUT = BEI.parent/'10.30_channel_assignment_20261009'
STEM = '10.30_Критическая_проверка_назначения_каналов_Scorpion_v2'
MODES = ['calm_breathing','inspiration_hold','enhanced_breathing','expiration_hold']
LABELS = ['Спокойное дыхание','Задержка на вдохе','Усиленное дыхание','Задержка на выдохе']
ROW = ['004','007','008','009','010']
BIN = EXPERIMENT/'derived/scorpion_bin_review/txt_binary_matches.local.json'
SUMMARY = EXPERIMENT/'derived/scorpion_review/summary.local.json'


def values(records, rid, channel, scope='analysis'):
    record = next(r for r in records if r['record_id'][-3:] == rid)
    parts = next(c for c in record['channels'] if c['channel'] == channel)['phases']
    return {p['mode']: p[scope]['stats']['median'] for p in parts if p[scope] is not None}


def channel_at(rid, early, late):
    return early if rid == '004' else late


def hypotheses(records, scope='analysis'):
    output = []
    for early, late in [(1,1),(2,2),(2,1),(1,2)]:
        for mode in ('inspiration_hold','expiration_hold'):
            lateral = np.array([values(records,rid,channel_at(rid,early,late),scope)[mode] for rid in ROW])
            ttrkg = np.array([values(records,rid,3-channel_at(rid,early,late),scope)[mode] for rid in ROW])
            output.append(dict(early=early,late=late,mode=mode,
                author_consistent=(early,late)==(2,1),constant_number=early==late,lateral=lateral.tolist(),ttrkg=ttrkg.tolist(),
                ttrkg_range_ohm=float(np.ptp(ttrkg)),ttrkg_median_ohm=float(np.median(ttrkg)),
                ttrkg_range_percent=float(100*np.ptp(ttrkg)/np.median(ttrkg)),
                ttrkg_004_to_007_ohm=float(ttrkg[1]-ttrkg[0]),
                lateral_80_to_60_ohm=float(lateral[-1]-lateral[-2])))
    return output


def run():
    verify_bei();OUT.mkdir(parents=True,exist_ok=True)
    records=read(BEI/'display.local.json')['records']
    levels=[]
    for r in records:
        rid=r['record_id'][-3:]
        for ch in (1,2):
            full=values(records,rid,ch,'full');analysis=values(records,rid,ch)
            levels.append(dict(record=rid,montage=r['label'].split(' · ',1)[1],channel=ch,
                full=full,analysis=analysis,
                contrast_ohm=analysis.get('inspiration_hold',0)-analysis.get('expiration_hold',0) if analysis else None))
    matches=read(BIN)['matches']
    later=[m for m in matches if int(m['record_id'][-3:])>=5]
    assert len(later)==14 and len({m['bin_sha256'] for m in later})==1
    later.sort(key=lambda m:m['start_data_index'])
    assert [m['record_id'][-3:] for m in later]==[f'{i:03d}' for i in range(5,19)]
    assert all(a['end_data_index_exclusive']<=b['start_data_index'] for a,b in zip(later[:-1],later[1:]))
    settings=[]
    for m in matches:
        assert len(m['settings_states_in_record'])==1
        c=m['settings_states_in_record'][0]['control']
        settings.append(dict(record=m['record_id'][-3:],bin=m['bin_id'],
            fields=[c[k] for k in ('u16_at_2','u16_at_4','u16_at_6','u16_at_8','u16_at_10','byte_12','byte_13')]))
    ecg=[];summary=read(SUMMARY)
    for r in summary['records']:
        rid=r['record_id'][-3:];fs=r['fs_hz_scenario']
        if fs!=(400 if rid in ('005','006') else 1000):continue
        with np.load(SUMMARY.parent/(r['key']+'.npz')) as a:
            for ch in (1,2):
                q=r['channels'][str(ch)]
                span=float(np.ptp(np.quantile(a[f'ecg_filtered_{ch}'],[.01,.99])))
                ecg.append(dict(record=rid,channel=ch,preferred_11_20=r['selected_ecg_channel_candidate'],
                    mains_50_mV2=q['mains_notch_comparison']['50']['raw_band_power_mV2'],
                    filtered_p01_p99_span_mV=span,candidates=q['candidate_count'],
                    RR_cv=q['RR_candidate_cv'],short_RR_fraction=q['short_RR_fraction_below_0_45_s'],
                    maximum_gap_s=q['max_candidate_gap_s_including_record_edges']))
    result=dict(levels=levels,hypotheses=hypotheses(records),full_stage_hypotheses=hypotheses(records,'full'),bin_settings=settings,ecg_quality=ecg)
    save(OUT/'results.local.json',result)
    save(OUT/'contract.local.json',dict(producer_sha256=sha(__file__),
        bei_display_sha256=sha(BEI/'display.local.json'),bei_contract_sha256=sha(BEI/'contract.local.json'),
        role_source='author confirmation 2026-10-09: 004/005 TTRKG1; 006 horizontal1 vertical2; 007 TTRKG2',annotations_changed=False,
        physiological_constant_ttrkg_is_a_hypothesis=True,record_order_is_export_order=True,
        uncertainty_model_established=False,bin_matches_sha256=sha(BIN),
        ECG_and_BEI_same_index_share_U_author_confirmation='2026-10-09'))
    save(OUT/'manifest.local.json',dict(outputs={name:sha(OUT/name) for name in ('results.local.json','contract.local.json')}))
    return result


def verify():
    verify_bei()
    c=read(OUT/'contract.local.json')
    for name,digest in read(OUT/'manifest.local.json')['outputs'].items():
        if sha(OUT/name)!=digest:raise ValueError('Channel diagnostic changed')
    if sha(__file__)!=c['producer_sha256'] or sha(BEI/'display.local.json')!=c['bei_display_sha256'] or sha(BEI/'contract.local.json')!=c['bei_contract_sha256'] or sha(BIN)!=c['bin_matches_sha256']:
        raise ValueError('Channel diagnostic provenance changed')
    result=read(OUT/'results.local.json')
    assert len(result['levels'])==36 and len(result['hypotheses'])==8
    return result


def levels_table():
    result=verify();rows=[]
    for rid in [f'{i:03d}' for i in range(1,19)]:
        pair=[next(l for l in result['levels'] if l['record']==rid and l['channel']==ch) for ch in (1,2)]
        row={'Запись':rid,'Монтаж':pair[0]['montage']}
        for ch,p in enumerate(pair,1):
            row[f'BEI{ch}, вдох, Ом']=p['analysis'].get('inspiration_hold')
            row[f'BEI{ch}, выдох, Ом']=p['analysis'].get('expiration_hold')
            row[f'ΔBEI{ch}, Ом']=p['contrast_ohm']
        rows.append(row)
    return pd.DataFrame(rows).fillna('—')


def hypothesis_table(scope='hypotheses'):
    return pd.DataFrame([{'Боковой канал в 004':h['early'],'Боковой канал в 007–010':h['late'],
        'Соответствует подтверждению автора':'Да' if h['author_consistent'] else 'Нет',
        'Смена номера бокового канала':'Нет' if h['constant_number'] else 'Да',
        'Дыхание':'Вдох' if h['mode']=='inspiration_hold' else 'Выдох',
        'Размах условной ТТРКГ, Ом':h['ttrkg_range_ohm'],'Размах / медиана, %':h['ttrkg_range_percent'],
        'Переход ТТРКГ 004 → 007, Ом':h['ttrkg_004_to_007_ohm'],
        'Переход боковой 80 → 60 мм, Ом':h['lateral_80_to_60_ohm']}
        for h in verify()[scope]])


def settings_table():
    columns=['u16 +2','u16 +4','u16 +6','u16 +8','u16 +10','Байт +12','Байт +13']
    return pd.DataFrame([{'Запись':s['record'],'Поток BIN':s['bin'],**dict(zip(columns,s['fields']))}
        for s in verify()['bin_settings'] if s['record'] in ('003','004','005','006','007','011','012','018')])


def ecg_table():
    rows=[];qualities=verify()['ecg_quality']
    for rid in [f'{i:03d}' for i in range(1,19)]:
        pair=[q for q in qualities if q['record']==rid]
        row={'Запись':rid,'Кандидат предпочтительного канала 11.20':pair[0]['preferred_11_20'] or '—'}
        for q in pair:
            ch=q['channel'];row.update({f'ЭКГ{ch}: мощность 49–51 Гц, мВ²':q['mains_50_mV2'],
                f'ЭКГ{ch}: CV RR':q['RR_cv'],f'ЭКГ{ch}: RR < 0,45 с, %':100*q['short_RR_fraction'],
                f'ЭКГ{ch}: наибольший пропуск, с':q['maximum_gap_s']})
        rows.append(row)
    return pd.DataFrame(rows)


def ecg_plot():
    # Same full five-second samples as 10.00; no new QRS detection or annotation.
    from build_scorpion_experiment_viewer import OUT as VIEW, verify as verify_viewer
    verify_viewer();shown=read(VIEW/'display_signals.local.json')
    summary=read(SUMMARY);series=[]
    for r in shown:
        s=next(s for s in summary['records'] if s['record_id']==r['record_id'] and s['fs_hz_scenario']==r['fs'])
        x=r['detail_t'];data=[]
        with np.load(SUMMARY.parent/(s['key']+'.npz')) as a:
            for ch in (1,2):
                raw=np.array(r['ecg_variants']['00'][ch-1]['detail_y'])
                filtered=r['ecg_variants']['11'][ch-1]['detail_y']
                peaks=a[f'qrs_samples_{ch}'].astype(int)
                peaks=peaks[(peaks/r['fs']>=x[0])&(peaks/r['fs']<=x[-1])]
                data.extend([(x,(raw-np.median(raw)).tolist()),(x,filtered),
                    ((peaks/r['fs']).tolist(),a[f'ecg_filtered_{ch}'][peaks].tolist())])
        series.append((r['label'],data))
    fig=make_subplots(rows=2,cols=2,shared_xaxes=True,subplot_titles=['ЭКГ1, исходная без постоянного уровня','ЭКГ2, исходная без постоянного уровня','ЭКГ1, одинаковые фильтры','ЭКГ2, одинаковые фильтры'])
    selected=3
    for ch,color in [(1,'#176aaf'),(2,'#e47823')]:
        for j,(x,y) in enumerate(series[selected][1][3*(ch-1):3*ch]):
            label=['исходная без медианы','режекторы и полоса','кандидаты QRS'][j]
            fig.add_trace(go.Scattergl(x=x,y=y,name=f'ЭКГ{ch}: {label}',mode='markers' if j==2 else 'lines',
                line=dict(color=color),marker=dict(color='#b12237',size=7,symbol='circle-open') if j==2 else None),row=1 if j==0 else 2,col=ch)
    buttons=[dict(label=label,method='update',args=[{'x':[x for x,y in data],'y':[y for x,y in data]},
        {'title.text':'Рисунок 3. Сравнение ЭКГ · '+label,**{f'xaxis{suffix}.range':[data[0][0][0],data[0][0][-1]] for suffix in ('','2','3','4')}}]) for label,data in series]
    fig.update_xaxes(title='t записи, с');fig.update_yaxes(title='ЭКГ, мВ')
    fig.update_yaxes(matches='y',row=1,col=2);fig.update_yaxes(matches='y3',row=2,col=2)
    fig.update_layout(template='plotly_white',height=740,title='Рисунок 3. Сравнение ЭКГ · '+series[selected][0],
        updatemenus=[dict(buttons=buttons,active=selected,x=0,y=1.18)],legend=dict(orientation='h',y=-.19),margin=dict(t=140,b=140))
    return fig


def levels_plot():
    result=verify();fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.15,
        subplot_titles=['Уровни на вдохе и выдохе','Дыхательная разность: вдох минус выдох'])
    for ch,color in [(1,'#176aaf'),(2,'#e47823')]:
        entries=[l for l in result['levels'] if l['channel']==ch]
        for mode,label,dash in [('inspiration_hold','вдох','solid'),('expiration_hold','выдох','dash')]:
            fig.add_trace(go.Scatter(x=[l['record'] for l in entries],y=[l['analysis'].get(mode) for l in entries],
                name=f'BEI{ch}, {label}',mode='lines+markers',line=dict(color=color,dash=dash),connectgaps=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=[l['record'] for l in entries],y=[l['contrast_ohm'] for l in entries],
            name=f'ΔBEI{ch}',mode='lines+markers',line=dict(color=color),connectgaps=False),row=2,col=1)
    fig.add_hline(y=0,line_dash='dot',line_color='#555',row=2,col=1)
    fig.update_xaxes(type='category');fig.update_xaxes(title='Запись в порядке экспорта',row=2,col=1)
    fig.update_yaxes(title='BEI, Ом',row=1,col=1);fig.update_yaxes(title='ΔBEI, Ом',row=2,col=1)
    fig.update_layout(template='plotly_white',height=690,title='Рисунок 1. Приборные каналы без назначения монтажам',legend=dict(orientation='h',y=-.16))
    return fig


def hypothesis_plot():
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Боковая сборка: вдох','Условная ТТРКГ: вдох','Боковая сборка: выдох','Условная ТТРКГ: выдох'])
    colors={(1,1):'#176aaf',(2,2):'#e47823',(2,1):'#168d58',(1,2):'#a0479c'}
    for h in verify()['hypotheses']:
        row=1 if h['mode']=='inspiration_hold' else 2
        name=f"Боковой: {h['early']} в 004 → {h['late']} в 007–010"
        for col,field in [(1,'lateral'),(2,'ttrkg')]:
            fig.add_trace(go.Scatter(x=[140,120,100,80,60],y=h[field],name=name,legendgroup=name,
                showlegend=row==1 and col==1,mode='lines+markers',
                line=dict(color=colors[(h['early'],h['late'])],dash='solid' if h['constant_number'] else 'dash')),row=row,col=col)
    fig.update_xaxes(title='Номинал боковой сборки, мм',autorange='reversed');fig.update_yaxes(title='BEI, Ом')
    fig.update_layout(template='plotly_white',height=760,title='Рисунок 2. Четыре диагностических истории подключения',legend=dict(orientation='h',y=-.20))
    return fig


def cells():
    import nbformat as nbf
    cc=[]
    def md(s):cc.append(nbf.v4.new_markdown_cell(s))
    def code(s):cc.append(nbf.v4.new_code_cell(s))
    md('''# Критическая проверка назначения импедансных каналов

**Эксперимент:** Георг, Scorpion v2, МГТУ, 05.10.2026, 100 кГц.
**Статус:** диагностическое сопоставление измерений с сообщённой последовательностью подключения.
Назначение 004–007 уточнено автором 09.10.2026; обратная задача 33.21
пересчитана с изменением номера бокового поля в размерном ряду.

[Обзор монтажей и сигналов](10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2.html)
показывает подтверждённую последовательность и прежние альтернативы.
Лучшее численное согласие само по себе не назначает монтаж измеренному полю.

## Подтверждение последовательности подключения

По окончательному уточнению автора от 09.10.2026 в 004 и 005 ТТРКГ
занимает канал 1. В 004 горизонтальная боковая сборка — канал 2;
в 005 на канале 2 она вертикальная. При переходе 005 → 006 вертикальная
сборка остаётся на месте и канале 2, а канал 1 ТТРКГ переносится на
горизонтальную боковую ось. В 007 канал 2 становится ТТРКГ,
горизонтальная боковая 120 мм записывается каналом 1.
Это подтверждение автора по памяти, а не назначение по величине BEI.
Продолжение этой схемы в 008–010 и переход 010 → 011 приняты по
ранее сообщённому постоянству монтажа внутри позднего блока.
Назначение осей креста наследуется из предыдущей записи; качество
ЭКГ самого креста для этого не используется. EMG1/EMG2 сняты с U соответствующих BEI1/BEI2.
Назначение 001–003 и ТЕПС/АЕПС этим уточнением не установлено.


Тест 001 выполнен с 1×1 на руках и горизонтальной боковой 140 мм.
Порядок 001–018 соответствует экспорту файлов из ZIP. BIN подтверждает
порядок участков 005–018; восстановление монтажей получено от автора.

## Уровни и дыхательные разности

Используются исходные BEI в омах, без фильтрации и подбора масштаба.
Медианы задержек взяты из тех же обрезанных кандидатных интервалов 11.20,
что и уровни 33.21. От неоднозначных границ отступают на 2 с, от края
записи — на 5 с. Границы ещё не приняты автором. Дыхательная разность
ΔBEI определена как медиана на вдохе минус медиана на выдохе, в омах.
Она зависит от выбранных интервалов и не является оценкой изменения
сопротивления одного компартмента. У теста 001 задержек нет;
медианы спокойного дыхания составляют 72,770 и 81,331 Ом для BEI1 и BEI2.

**Таблица 1.** Все приборные поля. «—» означает отсутствие соответствующей
задержки, а не нулевой импеданс. Роли каналов в таблице не назначены.''')
    code('from IPython.display import display, HTML\nfrom build_channel_assignment_report import *\nresult = verify()\ndisplay(table(levels_table()))')
    md('''**Рисунок 1.** Сплошные линии верхней панели — вдох, штриховые — выдох.
Синий цвет обозначает BEI1, оранжевый — BEI2; точки — медианы выбранных
интервалов. Нижняя панель показывает ΔBEI; точечная горизонталь — нулевую
разность. Соединения соседних записей помогают видеть переходы и не
означают непрерывного измерения во время перестановки. Любую кривую можно
скрыть щелчком по легенде; масштаб меняется средствами графика.''')
    code('display(HTML(levels_plot().to_html(full_html=False, include_plotlyjs=True)))')
    md('''## Настройки и порядок найденных участков BIN

**Таблица 2.** Расшифрованные числовые поля служебного сообщения.
Подписи u16 +2 и аналогичные обозначают двухбайтовые числа со смещением
от начала декодированного сообщения; знак «+» относится к смещению,
а не арифметической операции. Значение u16 +2 трактуется как период
в микросекундах в рабочей временной шкале 10.20. Для остальных показанных
полей физический смысл не установлен.

Между 006 и 007 меняются не только период 2500 → 1000, но и поля
u16 +4: 1000 → 450, u16 +10: 1300 → 550. Поэтому возможное влияние
аппаратной настройки нельзя свести к смене частоты отсчётов. Однако
все показанные поля 004 и 007 совпадают: они не объясняют различие
между этими двумя горизонтальными записями сами по себе. Постоянные
поля также не исключают физическую перестановку кабелей в гнёздах.
Формат не содержит расшифрованного назначения электродной сборки.
Порядок 005–018 проверен по неперекрывающимся индексам пакетов одного BIN.''')
    code('display(table(settings_table()))')
    md('''## Противоречия в горизонтальном ряду

В 002–004 канал 2 на вдохе меняется лишь от 79,427 до 80,953 Ом,
хотя число контактов на руках меняется 1×1 → 2×2 → 1×4.
Канал 1 в этих же записях меняется от 63,730 до 72,480 Ом,
хотя номинальная боковая сборка остаётся 140 мм. При варианте «боковая
на канале 1» это означает изменчивость предположительно неизменной
боковой области. При варианте «боковая на канале 2» это допускает влияние
контактов на руках на канал 1. Оба объяснения остаются гипотезами:
одновременно могли измениться контакт, состояние и межканальное влияние.

В 007–010 канал 2 на вдохе занимает узкую область 64,054–66,325 Ом,
а канал 1 — 85,702–127,833 Ом. Поэтому для этого блока вариант
«ТТРКГ на канале 2, боковая на канале 1» согласуется с ожиданием
приблизительно постоянного уровня неизменной ТТРКГ. Само это ожидание
не проверено: изменение другой подключённой сборки может влиять на результат.

Главное противоречие возникает между 004 и 007. Если боковая сборка всегда
на канале 1, условная ТТРКГ на вдохе меняется с 79,595 до 64,707 Ом
при одном обозначении 1×4. Если боковая всегда на канале 2, условная ТТРКГ
меняется с 67,018 до 86,240 Ом и затем до 127,833 Ом в 009.
Ни одна постоянная нумерация не даёт близкого постоянства ТТРКГ всего ряда.

Последовательность «боковая на канале 2 в 004, на канале 1 в 007–010»
согласуется с окончательным уточнением автора. Численная близость
ТТРКГ служит дополнительным наблюдением, а не основанием назначения.

## Сравнение четырёх историй подключения

Одна история соответствует сообщённому переходу, остальные три
сохранены как прежние диагностические альтернативы.
Для пяти горизонтальных записей 004, 007–010 вычислен размах условной ТТРКГ
R = max Z − min Z, в омах. Относительный размах равен 100R / median Z,
в процентах. Это описательные показатели межзаписной изменчивости,
а не погрешность прибора, статистический критерий или вероятность назначения.
Здесь нет свободной калибровки и подгонки к модели тканей.

**Таблица 3.** Размах и переходы для кандидатных расчётных интервалов.''')
    code('display(table(hypothesis_table()))')
    md('''**Рисунок 2.** Каждый цвет соответствует одной истории подключения.
Сплошные линии — постоянная нумерация, штриховые — диагностическая перестановка.
Точки — измеренные медианы, которым условно назначена роль. По горизонтали
показан размер боковой сборки, в том числе в панелях ТТРКГ: её собственный
монтаж 1×4 во всех пяти записях считается неизменным. Щелчок по легенде
скрывает соответствующую историю во всех панелях.

Даже при перестановке уровень боковой сборки 60 мм ниже уровня 80 мм.
Следовательно, перестановка не устраняет все несогласования и не делает
ряд подтверждённым измерением одной неизменной физиологической системы.''')
    code('display(HTML(hypothesis_plot().to_html(full_html=False, include_plotlyjs=True)))')
    md('''**Таблица 4.** Та же диагностика по полным кандидатным задержкам.
Она проверяет зависимость наблюдаемого разрыва от обрезки интервалов;
сама кандидатная разметка в этой проверке остаётся общей.''')
    code("display(table(hypothesis_table('full_stage_hypotheses')))")
    md('''## Качество ЭКГ как дополнительный признак подключения

По предложению автора более отчётливая ЭКГ рассматривается как признак
вероятного монтажа ТТРКГ. Автор 09.10.2026 подтвердил, что EMG1 и EMG2
снимались с тех же U, что соответственно BEI1 и BEI2. Поэтому показатели
ЭКГ данного номера относятся к измерительным контактам того же монтажа.
Это подтверждение автора, а не независимая аппаратная проверка.
Амплитуда ЭКГ зависит от отведения;
большая амплитуда сама по себе не устанавливает качество или монтаж.

**Таблица 5.** Сопоставлены прежние показатели 11.20: мощность исходной ЭКГ
в полосе 49–51 Гц, коэффициент вариации кандидатных RR, доля RR короче
0,45 с и наибольший промежуток без кандидата, включая края записи.
RR — интервал между соседними автоматически найденными событиями;
CV RR — отношение стандартного отклонения к среднему, безразмерное.
Широкополосная помеха, ошибки детектора и действительная вариабельность
ритма этими числами не разделены. QRS-кандидаты не приняты как R-зубцы.

Предпочтительный канал перенесён из прежнего алгоритма 11.20 без изменения.
Алгоритм допускает канал при числе кандидатов не менее 10, CV RR < 0,35
и наибольшем пропуске < 3 с; среди допустимых минимизирует
CV RR + 2p, где p — доля RR короче 0,45 с. Этот эвристический выбор
не является клинической оценкой качества и не задаёт вероятность ТТРКГ.''')
    code('display(table(ecg_table()))')
    md('''**Рисунок 3.** Для выбранной записи сравниваются оба канала в одном
пятисекундном фрагменте, без прореживания. Верхние панели показывают
исходную ЭКГ за вычетом медианы фрагмента; это удаление постоянного уровня,
а не фильтрация. Нижние панели используют одинаковые режекторы
50/100/150 Гц и полосу 0,5–40 Гц из 11.20. Цвет соответствует каналу;
красные незакрашенные круги — прежние QRS-кандидаты. Щелчок по легенде
скрывает отдельную кривую или метки. Фрагмент совпадает с 10.00 и не
представляет качество всей записи; общую ЭКГ и спектр следует смотреть там.''')
    code('display(HTML(ecg_plot().to_html(full_html=False, include_plotlyjs=True)))')
    md('''В 003 и 004 прежний алгоритм предпочитает ЭКГ1, в 007–010 — ЭКГ2.
У 004 мощность 49–51 Гц составляет 0,01709 мВ² в ЭКГ1 против
0,11071 мВ² в ЭКГ2. У 007 соотношение обратное: 0,22648 против
0,003936 мВ²; у 010 — 0,09068 против 0,000386 мВ².
В ЭКГ2 записей 007–010 также нет кандидатных RR короче 0,45 с,
тогда как в ЭКГ1 их доля составляет около 1–18%.
С учётом подтверждённой связи ЭКГ и U это дополнительное наблюдение
в пользу смены роли каналов около 006 → 007.

Однако тест 001 и запись 002 дают предпочтение ЭКГ2, хотя 003–004 — ЭКГ1.
В крестах показатели таблицы характеризуют только пригодность ЭКГ для
временной разметки; они не назначают боковые оси. В ТЕПС канал 1
предпочтителен не во всех записях: у 013 выбран канал 2. Следовательно,
качество ЭКГ полезно как условное подтверждение численной гипотезы,
но не как самостоятельное правило назначения ТТРКГ.''')
    md('''## Кресты, вертикальная запись и ТЕПС

Последовательность через 005 установлена автором: ТТРКГ1 и вертикальная2
в 005; горизонтальная1 и сохранённая вертикальная2 в 006; боковая1 и
ТТРКГ2 в 007. Качество ЭКГ креста в этом назначении не участвует.
Для 011 назначение наследуется из 010 по протоколу позднего блока.

В сохранённой вертикальной BEI2 между 005 и 006 медиана меняется
с 75,980 до 81,515 Ом на вдохе и с 80,802 до 84,086 Ом на выдохе.
Разности равны +5,535 и +3,284 Ом. Сборка при этом не переносилась;
контакт, состояние и влияние второй оси остаются возможными причинами.
Знак дыхательной разности вертикальной оси в обеих записях отрицательный.

Для горизонтальной сборки 140 мм BEI2(004) и BEI1(006) равны
79,595 и 76,818 Ом на вдохе, 79,095 и 75,932 Ом на выдохе.
Разности составляют −2,778 и −3,163 Ом. Это разные пробы, а не повтор
одного отсчёта; в кресте дополнительно подключена вертикальная ось.
В 010 → 011 сохранённая горизонтальная BEI1 меняется
с 115,325 до 97,172 Ом на вдохе. Это заметная межзаписная изменчивость,
которую назначение каналов не устраняет.

В ТЕПС 012–017 канал 1 на вдохе занимает 51,231–60,934 Ом, канал 2 —
1,818–57,808 Ом. В частности, у ТЕПС-7 BEI2 равен лишь 1,818 Ом.
Это наблюдение о передаточном импедансе. Оно не доказывает отключение,
насыщение или ошибку: величина зависит от геометрии измерения и трактов.
Гипотеза «ТЕПС менялась на канале 2, на канале 1 сохранялся другой монтаж»
требует подтверждения обоих подключений и координат. Изменчивость канала 1
около 10 Ом также не позволяет назвать его неизменным эталоном.
Для АЕПС-5 канал 2 почти не меняется между задержками, но это не определяет
его назначение и не позволяет переносить схему ТЕПС на руку.

## Решение для дальнейшей методики

В 33.21 добавлен пересчёт по сообщённому переходу:
BEI2 для 140 мм в 004, BEI1 для 120–60 мм в 007–010;
горизонтальная ось креста 006 берётся из BEI1, вертикальная — из BEI2.
Прежние постоянные сценарии оставлены для оценки влияния ошибочного
назначения. BEI1 записи 005 — ТТРКГ и в боковые решения не включается.

Дальнейшие нерешённые вопросы: номера 001–003, оба монтажа ТЕПС/АЕПС,
фактические координаты, принятая дыхательная разметка и причины
межзаписной изменчивости. Уточнение по памяти является подтверждением
автора; схема, фотография или расшифрованная настройка могут дать
независимую проверку. Уровни BEI сами по себе не разделяют контакт,
межканальное влияние и изменение физиологического состояния.''')
    return cc


if __name__=='__main__':
    run();export(STEM,cells(),OUT)
