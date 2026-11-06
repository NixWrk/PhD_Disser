"""Два графика кажущегося ρ для реокардиомонитора МГТУ; читатель готовых FEM."""
from pathlib import Path
import argparse,csv,hashlib,json

NAMES={'georg':'Георгий','nik':'Ник'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ru(x,n=2):return f'{x:.{n}f}'.replace('.',',')
def plural(n,forms):
    n=int(n)
    return forms[2] if 11<=n%100<=14 else forms[0] if n%10==1 else forms[1] if n%10 in (2,3,4) else forms[2]
def count(n):return f'{int(n):,}'.replace(',',' ')


def build(observation_file,geometry_files,result_files,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    data=read(observation_file);geometries={read(p)['subject_alias']:(Path(p),read(p)) for p in geometry_files}
    results={read(p)['subject_alias']:(Path(p),read(p)) for p in result_files};out=Path(out)
    if set(results)!=set(NAMES) or set(geometries)!=set(NAMES):raise ValueError('Нужны собственные результаты обоих добровольцев')
    sources={(s['subject_alias'],o['record_id']):o for s in data['subjects'] for o in s['observations']};rows=[]
    for alias,(p,r) in results.items():
        gp,g=geometries[alias]
        request_file=p.parent.parent/(p.parent.name+'_input.local.json');request=read(request_file)
        if (sha(request_file)!=r['identity']['geometry_plan_sha256'] or
            request['geometry_source_sha256']!=sha(gp) or
            Path(request['geometry_source_path']).resolve()!=gp.resolve() or r['placement']!=g['placement']):
            raise ValueError('Решение относится к другой версии геометрии или положения сборки')
        if r['identity']['mesh_sha256']!=g['mesh_sha256'] or sha(observation_file)!=g['observations_plan_sha256']:
            raise ValueError('Не совпадают геометрия и происхождение наблюдений')
        for a in r['answers']:
            o=sources[alias,a['record_id']]
            if a['size_mm']!=o['size_mm'] or abs(a['observed_Z_ohm']-o['Z_ohm'])>1e-10 or abs(a['residual_ohm'])>a['tolerance_ohm']:
                raise ValueError('Не воспроизведено связанное с записью наблюдение')
            rows.append(dict(subject_alias=alias,record_id=a['record_id'],size_mm=a['size_mm'],
                Z_inspiration_ohm=o['Z_ohm'],rho_apparent_ohm_m=a['rho_apparent_ohm_m'],
                predicted_Z_ohm=a['predicted_Z_ohm'],residual_ohm=a['residual_ohm'],tolerance_ohm=a['tolerance_ohm'],
                Z_unit_rho_ohm=a['Z_unit_rho_ohm'],linear_ratio_rho_ohm_m=a['linear_ratio_rho_ohm_m'],
                fixed_contact_correction_relative=a['fixed_contact_correction_relative'],
                raw_q05_ohm=o['raw_q05_ohm'],raw_q95_ohm=o['raw_q95_ohm'],
                window_start_s=o['selected_window_s'][0],window_end_s=o['selected_window_s'][1],
                annotation_status=o['annotation_status'],channel=2,result_status=a['status']))
    if len(rows)!=19 or len({(r['subject_alias'],r['size_mm']) for r in rows})!=19:raise ValueError('Нужны ровно 19 независимых наблюдений')
    if {(r['subject_alias'],r['record_id']) for r in rows}!=set(sources):raise ValueError('Неполное соответствие исходным записям')
    rows.sort(key=lambda r:(list(NAMES).index(r['subject_alias']),r['size_mm']))
    out.mkdir(parents=True,exist_ok=False)
    with (out/'rho_apparent_bmstu.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    for alias in NAMES:
        selected=[r for r in rows if r['subject_alias']==alias];color='#2563eb' if alias=='georg' else '#d97706'
        fig,ax=plt.subplots(figsize=(8.8,5.6));fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.79)
        ax.plot([r['size_mm'] for r in selected],[r['rho_apparent_ohm_m'] for r in selected],'-o',color=color,lw=1.8,ms=6,zorder=3)
        for i,r in enumerate(selected):
            dx=9 if i==0 else (-9 if i==len(selected)-1 else 0)
            ax.annotate(ru(r['rho_apparent_ohm_m']),(r['size_mm'],r['rho_apparent_ohm_m']),
                xytext=(dx,11),textcoords='offset points',ha='center',fontsize=9.5,color=color)
        ax.set_xticks(list(range(50,141,10)));ax.set_xlim(44,146);ax.set_ylim(0,max(r['rho_apparent_ohm_m'] for r in selected)*1.22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'.replace('.',',')))
        ax.grid(alpha=.20,zorder=0)
        ax.set_xlabel('Размер боковой сборки L — номинальная токовая база, мм',labelpad=10)
        ax.set_ylabel('Кажущееся удельное сопротивление ρ, Ом·м',labelpad=10)
        fig.suptitle(NAMES[alias]+' · реокардиомонитор МГТУ',x=.12,y=.955,ha='left',fontsize=15,fontweight='bold')
        fig.text(.12,.87,'Индивидуальная КТ-геометрия; весь объём однороден',fontsize=11,color='#374151')
        footer='BASE₂ на задержке глубокого вдоха; принятая экспертная разметка.\nПоложение сборки — модельный сценарий; межканальное влияние не моделируется.'
        footer+='\n100 мм исключены: копия записи 90 мм.' if alias=='nik' else '\n90 и 100 мм: аномалия канала ТТРКГ; использован только боковой BASE₂.'
        fig.text(.12,.045,footer,fontsize=8.5,color='#4b5563',linespacing=1.55)
        for ext in ['png','svg']:fig.savefig(out/f'{alias}_rho_bmstu.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    summary=dict(status='exploratory_hypothesis_not_validated',points=19,
        max_absolute_forward_error_ohm=max(abs(r['residual_ohm']) for r in rows),
        max_fixed_contact_correction_relative=max(abs(r['fixed_contact_correction_relative']) for r in rows),
        observations_sha256=sha(observation_file),geometry_sha256={k:sha(v[0]) for k,v in geometries.items()},
        result_sha256={k:sha(v[0]) for k,v in results.items()},
        figures={p.name:sha(p) for p in out.glob('*_rho_bmstu.*')},
        global_mesh_convergence_proven=False,breathing_annotation_accepted=True,
        source_record_global_QC_completed=False,placement_is_measured=False,TTRKG_coupling_modelled=False)
    (out/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text=report(rows,geometries,results,summary)
    (out/'README.md').write_text(text,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)


def report(rows,geometries,results,summary):
    text='''# Кажущееся ρ боковых сборок реокардиомонитора МГТУ

Пересчёт от 07.11.2026 по данным эксперимента 2. Для каждого добровольца использована собственная КТ/FEM-геометрия. Весь расчётный объём однороден, включая области лёгких, сердца, костей и жира; измерения берутся из бокового канала BASE₂ в Ом.

## Измерения и дыхательная разметка

Исходные CSV и SHA-256 сверены с принятыми дыхательными аннотациями и снимком анализа 33.06. Для каждого размера заново вычислена медиана BASE₂ внутри принятой задержки вдоха с отступом 0,5 с от обеих границ; границы отбора по времени включены. Все 19 уровней воспроизведены с расхождением не более 10⁻⁸ Ом. Никакие коэффициенты усиления, смещения или исправления знака не подбирались.

У Георгия имеется десять размеров 50–140 мм, у Ника — девять. Файл Ника 100 мм побайтно совпал с 90 мм и исключён как дубликат. У Георгия 90 и 100 мм аномальны в канале 1 ТТРКГ; их боковой BASE₂ не исключается только по аномалии другого канала. Общий ручной контроль качества записей остаётся незавершённым. Дополнительные записи с неизвестным размером не включены.

Принятая разметка является экспертной ретроспективной реконструкцией; приборные отметки команд отсутствуют. Интервал «задержка вдоха» используется как глубокий вдох по условию пересчёта; глубина вдоха и совпадение с состоянием КТ не измерены независимо. Разные размеры записаны последовательно. Их различия могут включать повторение вдоха, ручную наклейку, контакт, время записи и особенности измерительного тракта. Этот расчёт не оценивает отдельно их неопределённость.

## Однородная КТ-модель и конечные контакты

Использованы индивидуальные КТ-модели текущей подготовки TotalSegmentator и сохранённые цилиндрические продолжения рук. Для Ника известные центр C01 и угол 14° заново заданы на актуальной поверхности: центр проецируется на собственную сетку, локальный касательный базис строится заново. Полные прежние массивы геометрии C01 недоступны; новый сценарий не тождествен старому набору координат и не является измерением наклейки. Номинальные смещения A–M–N–B равны −L/2, −L/4, L/4, L/2 в касательной плоскости с последующей проекцией на кожу.

Для Георгия использованы сохранённые правила прежнего модельного места на его собственной КТ: общий центр, продольная дуга сечения кожи и те же номинальные смещения по длине дуги. Координаты ранее рассчитанных размеров воспроизведены отдельно по каноническому алгоритму. Совпадение места установки между МГТУ и Scorpion не измерено и не считается установленным фактом.

Каждый монтаж включает четыре конечных контакта диаметром 5 мм. Контактное удельное сопротивление zₑ = 0,00015915494309189532 Ом·м² взято из прежнего сценария боковых сборок МГТУ; оно не измерено независимо. Контактные матрицы построены заново, готовые матрицы Scorpion не перенесены. Одновременно присутствующим набором всех размеров тело не покрывается. Возбуждается собственная пара A–B, измеряется M–N. Физически присутствующие площадки ТТРКГ и ток канала 1 в этом сценарии не моделируются; временное разделение каналов Scorpion на реокардиомонитор МГТУ не переносится. Влияние этой идеализации на измеренный BASE₂ не оценено.

Внутри тела решается квазистатическая скалярная задача ∇·(ρ⁻¹∇u) = 0. В CEM на площадке Eₑ выполняются u + zₑρ⁻¹∂u/∂n = Uₑ и ∫Eₑρ⁻¹∂u/∂n dS = Iₑ; вне контактов поток равен нулю. Здесь u — потенциал тела, Uₑ — потенциал электрода, Iₑ — полный ток, n — внешняя нормаль. Нормировка токов A/B на +1/−1 А даёт импеданс без изменения измерительной шкалы.

Объёмный оператор имеет вид K(ρ) = K₀/ρ. При изменении сетки K₀ заново собран каноническим EIDORS в СИ; геометрия тела при вложенном сгущении сохранена. Контактная матрица C фиксирована, полный оператор H(ρ) = C + diag(K₀/ρ, 0). Для каждой записи отдельно решено уравнение

\\[
Z_{\\mathrm{CEM}}(\\rho_{\\mathrm{каж}};\\Omega_{\\mathrm{КТ}},L,z_e)=Z_{\\mathrm{вдох}}.
\\]

Отношение Zвдох/ZCEM(1 Ом·м) используется как начальное приближение; затем применяется метод Ньютона по lnρ с канонической производной. Контактный параметр не масштабируется вместе с ρ. Допуск |ZCEM − Zвдох| ≤ 10⁻⁵ Ом + 10⁻⁷|Zвдох| относится к численному воспроизведению. В модели используется действительная скалярная проводимость; комплексная часть и частотный перенос между приборами не восстанавливаются. Историческая расчётная опора МГТУ — 50 кГц; частота данной регистрации независимо здесь не подтверждалась.

## Два индивидуальных результата

Линии соединяют отдельные последовательно записанные размеры и не задают измерений между ними. Пропущенная точка Ника 100 мм не интерполируется как новое наблюдение.

'''
    for alias in NAMES:
        selected=[r for r in rows if r['subject_alias']==alias];g=geometries[alias][1];r=results[alias][1]
        text+=f"### {NAMES[alias]}\n\n![Кажущееся ρ: {NAMES[alias]}]({alias}_rho_bmstu.png)\n\n"
        text+='| L, мм | Z вдоха, Ом | ρ кажущееся, Ом·м |\n|---:|---:|---:|\n'
        for row in selected:text+=f"| {row['size_mm']} | {ru(row['Z_inspiration_ohm'],3)} | {ru(row['rho_apparent_ohm_m'],4)} |\n"
        low=min(selected,key=lambda x:x['rho_apparent_ohm_m']);high=max(selected,key=lambda x:x['rho_apparent_ohm_m'])
        text+=f"\nПолучены значения {ru(low['rho_apparent_ohm_m'])}–{ru(high['rho_apparent_ohm_m'])} Ом·м; минимум при {low['size_mm']} мм, максимум при {high['size_mm']} мм. Это диапазон точек одного условного расчёта. Сетка содержит {count(r['mesh_nodes'])} {plural(r['mesh_nodes'],('узел','узла','узлов'))} и {count(r['mesh_elements'])} {plural(r['mesh_elements'],('тетраэдр','тетраэдра','тетраэдров'))}; геометрические и контактные проверки пройдены.\n\n"
    text+=f'''## Проверки и ограничения

Проверены принадлежность сеток участникам, положительность тетраэдров, замкнутая согласованная граница, сохранение исходных узлов и объёма при сгущении. Форма элементов проверяется при неизменных пределах: безразмерный показатель формы тетраэдра mean ratio не меньше 0,01, максимальная обусловленность рёбер 10 000, не более 1 000 000 узлов. Контакты прошли проверки непересечения, точного соответствия индуцированных граней, отклонения площади ≤10 %, центроида ≤0,75 мм и максимального удаления узла ≤3,5 мм от центра. Отказавшие подготовки не используются, допуски после отказа не ослаблялись.

Все 19 окончательных точек удовлетворяют допуску воспроизведения Z; максимальная невязка составляет {ru(summary['max_absolute_forward_error_ohm'],9)} Ом. Относительная невязка полной CEM-системы <10⁻⁷, нарушение взаимности <10⁻⁷ Ом. Максимальная поправка к начальному отношению импедансов при фиксированных контактах равна {ru(100*summary['max_fixed_contact_correction_relative'],4)} %.

Глобальная сеточная сходимость не установлена. Для Ника использованы два уровня сгущения; цель отбора рёбер 0,5 мм при этом ограничении не достигнута. Пройденные проверки качества сетки и контактов не устанавливают численную погрешность ρ. Разброс отсчётов 5–95 % в CSV не является доверительным интервалом ρ. Точность постановки электродов, контактный параметр, воспроизводимость дыхания и межканальное влияние не оценены независимо.

Получены зависимости эквивалентного однородного ρ от размера сборки на собственной анатомии каждого добровольца. Неопределённость монтажа, состояния добровольца и прибора ограничивает интерпретацию различий между точками. По этим графикам нельзя отдельно восстановить свойства лёгкого или мягких тканей либо установить причину различий с Scorpion.

## Сохранённые материалы

[Полная таблица CSV](rho_apparent_bmstu.csv) содержит уровни, выбранные окна, квантили и невязки; [паспорт проверки](verification.json) связывает данные, геометрию и результаты. Доступны два PNG и два SVG. Первичные CSV и аннотации не изменены. Локальные планы, геометрические проверки, отклонённые подготовки и итерации находятся в `MATLAB_TRKG4_real_subjects/output/exploratory/bmstu_apparent_20261107/`.
'''
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--geometry',type=Path,nargs=2,required=True);p.add_argument('--results',type=Path,nargs=2,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.observations,a.geometry,a.results,a.out)
