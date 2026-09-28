"""Update 40.23 from a completed, independently sealed five-material run."""
from pathlib import Path
import argparse
import json
import shutil
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'Colab Notebooks/cardiac_4d_segmentation/results/nix_cardiac_cycle_fat_20260916'
TITLE='40.23_Изменение_импеданса_TEPC_по_4D_КТ_Nix'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def label(m):return 'ТТРКГ' if m=='ttrkg' else m.replace('_','-').upper()
def fmt(x,n=3):return f'{x:.{n}f}'.replace('.',',')
def table(headers,rows):
    headers=[str(h).replace('|','&#124;') for h in headers]
    rows=[[str(v).replace('|','&#124;') for v in row] for row in rows]
    display(Markdown('| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)))
def save(fig,name):
    fig.savefig(RESULTS/name,dpi=150,bbox_inches='tight');plt.close(fig);display(Image(filename=str(RESULTS/name)))
def load():return read(RESULTS/'summary.json')

def intro():
    d=load()
    display(Markdown('## Цель и расчётная постановка\n\nРассчитано изменение передаточного импеданса TEPC-2–TEPC-7 и ТТРКГ при пофазном изменении наблюдаемой маски сердца Nix. Новая постановка добавляет статическую маску жира и использует явно заданные литературные сценарии. Сравниваются четыре сочетания свойств лёгкого и сердечной области на одной геометрии. Это исследование влияния модельных допущений, а не восстановление ударного объёма и не полная оценка чувствительности во всём литературном диапазоне.'))
    table(['Материал','ρ, Ом·м','Основание'],[
      ['Мышечный фон','5,000','Принято для этого сравнения внутри литературного диапазона; фон не является чистой мышцей'],
      ['Жир','23,109','Gabriel / IT’IS, 50 кГц'],['Компактная кость','48,445','Gabriel / IT’IS, 50 кГц'],
      ['Лёгкое, сценарий G','9,741','Gabriel, Lung (Inflated), 50 кГц'],
      ['Лёгкое, сценарий L','17,158','1 / средней σ низкочастотной сводки IT’IS; перенос на 50 кГц — допущение'],
      ['Сердечная область, сценарий M','5,117','Свойства миокарда назначены всей наблюдаемой маске'],
      ['Сердечная область, сценарий B','1,427','Свойства крови назначены той же маске; геометрия камер не выделена']])
    display(Markdown('**Почему для лёгкого два значения.** Частотная модель Gabriel задаёт воспроизводимую опору на 50 кГц. Низкочастотная сводка объединяет другие измерения и условия; её средняя проводимость 0,058283069 См/м даёт ρ = 17,157641 Ом·м. Обратная величина средней проводимости не является средним удельным сопротивлением. Сопоставление двух опор показывает зависимость расчёта от выбора источника; ни одна из них не установлена как индивидуально верное значение Nix.\n\nИсточники: [частотная модель IT’IS](https://itis.swiss/virtual-population/tissue-properties/database/dielectric-properties), [низкочастотная сводка](https://itis.swiss/virtual-population/tissue-properties/database/low-frequency-conductivity), [справочник проекта](40.08_Литературные_диапазоны_электрических_свойств_тканей.md).'))
    display(Markdown('Для мышечного фона принято ρ = 5,0 Ом·м, то есть σ = 0,2 См/м. Это выбранный параметр сценария внутри рабочего литературного диапазона примерно 1,38–10,0 Ом·м; он не является средним по базе или измерением Nix. Его значение в 3,50 раза выше сопротивления крови в данном расчёте. Реальная скелетная мышца анизотропна; здесь её свойства представлены одним скалярным значением. Остаточный фон включает также ткани, не выделенные отдельными масками, поэтому название «мышечный фон» обозначает электрическое допущение.'))
    display(Markdown('При выбранных ρ фона 5,0 Ом·м и миокарда 5,117 Ом·м электрический контраст между ними мал. Замещение участка такого фона сердечной областью может давать небольшой сигнал. Для крови ρ = 1,427 Ом·м контраст значительно больше. Однако отклик определяется также тем, какие части лёгкого, жира и костей перекрывает сердечная маска в каждой фазе; одной разности сопротивлений для предсказания сигнала недостаточно.'))
    display(Markdown('## Метод расчёта\n\nИспользована квазистатическая модель стационарной проводимости: ∇·(σ∇φ) = 0, где σ — проводимость ткани, См/м; φ — электрический потенциал, В. Значения σ = 1/ρ назначены по картам материалов. Ёмкостная составляющая в этом сравнении не учитывается: результат представляет действительный передаточный импеданс при проводимостях, выбранных для номинальной частоты 50 кГц. Для низкочастотной опоры лёгкого перенос на эту частоту принят отдельно.\n\nКонтакты заданы полной электродной моделью EIDORS с конечной площадью и контактным импедансом. Передаточный импеданс определяется как Z = (U₊ − U₋)/I, где U₊ и U₋ — потенциалы измерительных электродов, В; I — ток возбуждающей пары, А. Поэтому Z зависит от обеих пар электродов и не совпадает с сопротивлением между токовыми контактами. Для каждой фазы и сборки решается своя система; сравнение включает 12 фаз × 7 сборок × 4 сценария = 336 решений.'))
    display(Markdown('## Геометрия и временные данные\n\nИспользованы прежние 12 автоматических масок целого сердца, FEM-сетка и семь монтажей с конечными контактами диаметром 5 мм. Метки фаз 14–121% R–R сохраняются без искусственного замыкания. Основной график ограничен интервалом 822 мс от первой доступной фазы; это окно длительностью одного исходного R–R, а не полностью измеренный цикл от R до R. Размах рассчитывается по 11 доступным фазам внутри этого окна. Последняя, двенадцатая фаза 121% R–R находится за его пределами; её решение также сохранено в таблице результатов. Временное разрешение реконструкции составляет 140 мс, поэтому более частое расположение фазовых точек не означает столь же высокую независимую временную детализацию. Маски достигают верхней границы поля КТ, поэтому изменение их объёма нельзя непосредственно интерпретировать как ударный объём.\n\nЖир — объединение масок подкожного жира (subcutaneous_fat) и жира туловища (torso_fat) из TotalSegmentator v2 по полной статической КТ серии 301. Исходные лёгкие и кости сохранены; жир выделен внутри прежнего мягкотканного фона. Текущая сердечная область имеет приоритет. Освобождаемый ею участок возвращает свой фоновый материал. Это правило назначения проводимости, без механического перемещения окружающих тканей. Статическая и 4D-КТ совмещены по исходным медицинским координатам с согласованием ориентации осей. Независимая оценка ошибки этого совмещения не выполнена. В областях вне статической КТ жир не достраивается.'))
    table(['Геометрический показатель','Значение'],[
      ['Жир в исходной автоматической сегментации, мл',fmt(d['fat_geometry']['segmented_volume_ml'])],
      ['Жир, перенесённый в мягкотканный фон FEM, мл',fmt(d['fat_geometry']['background_volume_ml'])],
      ['Объём FEM с центрами элементов вне статической КТ, мл',fmt(d['fat_geometry']['outside_ct_centroid_volume_ml'])]])
    display(Image(filename=str(RESULTS/'fat_ct_overlay.png')))
    display(Markdown('**Рисунок 1.** Три ортогональных сечения статической КТ через центр первой сердечной маски. Оранжевым показан автоматический жир, красным — контур наблюдаемой сердечной области. Изображение позволяет визуально оценить пространственное совмещение; оно не заменяет независимую ручную верификацию сегментации. Объём всей исходной маски жира и объём жира внутри прежнего FEM-фона могут различаться вследствие границ модели и приоритета других материалов.'))
    display(Image(filename=str(RESULTS/'geometry.png')))
    display(Markdown('**Рисунок 2.** Объём наблюдаемой маски и смещение её центра сохранены из исходной геометрической подготовки: эти входы не менялись при добавлении жира и смене проводимостей. Пунктир соответствует исходной фазовой отметке 100% R–R. На электрических графиках ниже используется другая ось — время от первой доступной фазы. Одновременное изменение объёма и координат не позволяет приписывать весь электрический отклик только объёму.'))
    display(Image(filename=str(RESULTS/'montages.png')))
    table(['Сборка','Фактические площади четырёх контактов, мм²'],[[label(c['id']),'; '.join(fmt(a,2) for a in c['areas_mm2'])] for c in d['contacts']])
    display(Markdown('Для всех контактов сохранён удельный контактный импеданс **0,000159155 Ом·м²**. Он унаследован из прежней постановки и не измерен индивидуально. Размер диска 5 мм и фактическая площадь его дискретного представления указаны раздельно. Изменение контактов не является фактором этого сравнения.'))
    display(Markdown('**Рисунок 3.** Положения семи электродных сборок. Красным показана токовая пара, синим — измерительная. ТТРКГ здесь — конкретный ранее выбранный монтаж на руках: измерительные электроды расположены в 20 мм, токовые — в 80 мм от модельной границы туловища и руки; промежуток между ними равен 60 мм. Соответствие этих координат фактической экспериментальной наклейке отдельно не установлено.'))

def results():
    d=load();states=d['geometry'];scenarios=d['scenarios'];montages=d['montages']
    lookup={(r['scenario'],r['id'],r['montage']):r['Z_ohm'] for r in d['solutions']}
    Z=np.array([[[lookup[s['id'],p['id'],m] for p in states] for m in montages] for s in scenarios])
    time=np.array([p['derived_time_from_R0_ms'] for p in states]);time-=time[0]
    rr=states[0]['rr_interval_ms'];window=time<=rr;amplitude=np.ptp(Z[:,:,window],axis=2)
    assert np.all(amplitude>1e-12), 'A zero phase range needs explicit interpretation before ratio reporting'
    titles=['G–M','G–B','L–M','L–B']
    display(Markdown('## Изменение импеданса\n\nОбозначения: G — лёгкое по Gabriel; L — лёгкое по низкочастотной сводке; M — вся сердечная маска со свойствами миокарда; B — та же маска со свойствами крови. В пределах каждого сценария проводимости фиксированы между фазами. Поэтому пофазный сигнал возникает вследствие изменения геометрии наблюдаемой сердечной области.'))
    for delta,filename in [(False,'absolute_cycle.png'),(True,'delta_cycle.png')]:
        fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,constrained_layout=True)
        for j,m in enumerate(montages):
            ax=axes.ravel()[j]
            for k,title in enumerate(titles):
                values=1000*(Z[k,j]-Z[k,j,0]) if delta else Z[k,j]
                ax.plot(time,values,'o-',ms=3,label=title)
            ax.set(title=label(m),xlim=(0,rr),xlabel='Время от первой фазы, мс',ylabel='ΔZ, мОм' if delta else 'Z, Ом');ax.grid(alpha=.25);ax.tick_params(axis='x',labelbottom=True)
        axes.ravel()[-1].axis('off');axes.ravel()[0].legend(ncol=2)
        save(fig,filename)
    display(Markdown('**Рисунки 4–5.** Абсолютный импеданс и его изменение относительно первой фазы, ΔZ = Z(t) − Z(t₀). Вертикальное смещение абсолютных кривых характеризует исходную проводимость всей модели. Расстояние между кривыми ΔZ показывает, как электрическая трактовка тканей меняет прогнозируемый отклик на одну и ту же динамику геометрии. Большой базовый импеданс сам по себе не означает большой отклик сердца. Точки соответствуют рассчитанным фазам; соединяющие их отрезки служат для чтения графика и не повышают временное разрешение исходной КТ.'))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,values,title,unit in [(axes[0],Z[:,:,0].T,'Первая доступная фаза','Ом'),(axes[1],1000*amplitude.T,'Размах в наблюдаемом окне','мОм')]:
        im=ax.imshow(values,aspect='auto',cmap='YlOrRd');fig.colorbar(im,ax=ax,label=unit)
        ax.set_xticks(range(4),titles);ax.set_yticks(range(7),[label(m) for m in montages]);ax.set_title(title)
        for i in range(7):
            for j in range(4):ax.text(j,i,fmt(values[i,j],2),ha='center',va='center',fontsize=8,color='white' if values[i,j]>.65*np.max(values) else 'black')
    save(fig,'scenario_matrix.png')
    display(Markdown('**Рисунок 6.** Матрицы абсолютного импеданса и размаха ΔZ с числовыми подписями. Цветовые шкалы независимы. Размах определяется как максимум минус минимум среди доступных фаз внутри указанного окна. Это показатель модельного отклика на совместное изменение объёма, положения и формы; он не является отдельной производной по объёму или мерой точности его восстановления.'))
    table(['Сборка']+[t+': размах, мОм' for t in titles],[[label(m)]+[fmt(1000*amplitude[k,j]) for k in range(4)] for j,m in enumerate(montages)])
    table(['Сборка']+[t+': размах / |Z₀|, %' for t in titles],[[label(m)]+[fmt(100*amplitude[k,j]/abs(Z[k,j,0])) if abs(Z[k,j,0])>1e-12 else 'не определено' for k in range(4)] for j,m in enumerate(montages)])
    display(Markdown('Относительный размах делит изменение сигнала на модуль исходного импеданса. Он может быть большим у сборки с малым Z₀ даже при небольшом абсолютном отклике. Поэтому относительный размах не является избирательностью к сердцу и не заменяет оценку ошибки восстановления объёма.'))
    display(Markdown('### Различие TEPC-6 и TEPC-7'))
    table(['Сценарий','Сборка','U₊, мВ','U₋, мВ','U₊ − U₋, мВ при 1 мА'],[
        [titles[k],label(m),fmt(d['first_phase_potentials_mV_at_1mA'][s['id']][m][1]),fmt(d['first_phase_potentials_mV_at_1mA'][s['id']][m][2]),fmt(Z[k,montages.index(m),0])]
        for k,s in enumerate(scenarios) for m in ['tepc_6','tepc_7']])
    display(Markdown('Потенциалы первой фазы 14% R–R приведены для условного тока 1 мА и одной опорной точки модели. Их абсолютный уровень зависит от выбора этой точки; измеряемая разность от него не зависит. При токе 1 мА численное значение разности в мВ равно передаточному импедансу в Ом. Малый импеданс означает небольшую разность потенциалов измерительной пары. Он не означает, что ткани около электродов имеют столь же малое сопротивление. Изменение взаимного положения токовых и измерительных электродов меняет распределение потенциала и разность на контактах, поэтому близкие по размеру сборки могут давать существенно разные значения.'))
    display(Markdown('### Влияние выбранных свойств лёгкого и сердца'))
    for heart_index,heart_name in [(0,'миокарда'),(1,'крови')]:
        base_change=100*(Z[heart_index+2,:,0]-Z[heart_index,:,0])/abs(Z[heart_index,:,0])
        amplitude_change=100*(amplitude[heart_index+2]-amplitude[heart_index])/amplitude[heart_index]
        display(Markdown(f'При свойствах **{heart_name}** в сердечной области переход от лёгкого 9,741 к 17,158 Ом·м изменяет исходный импеданс сборок на **{fmt(base_change.min(),2)}…{fmt(base_change.max(),2)}%**, а размах сигнала — на **{fmt(amplitude_change.min(),2)}…{fmt(amplitude_change.max(),2)}%**. Диапазон здесь относится к семи сборкам, а не к статистической неопределённости.'))
    for offset,lung_name in [(0,'G'),(2,'L')]:
        ratio=amplitude[offset+1]/amplitude[offset]
        display(Markdown(f'При лёгком **{lung_name}** отношение размаха в сценарии крови к размаху в сценарии миокарда составляет **{fmt(ratio.min(),2)}–{fmt(ratio.max(),2)}** по семи сборкам. Оно описывает смену электрических свойств одной и той же маски, а не различие реальной динамики объёма крови и миокарда.'))
    sentences=[]
    for k,title in enumerate(titles):
        order=np.argsort(amplitude[k]);hi=int(order[-1]);lo=int(order[0])
        sentences.append(f'В сценарии **{title}** наибольший размах имеет **{label(montages[hi])}** ({fmt(1000*amplitude[k,hi])} мОм), наименьший — **{label(montages[lo])}** ({fmt(1000*amplitude[k,lo])} мОм).')
    display(Markdown('## Результат и интерпретация\n\n'+'\n\n'.join(sentences)))
    stable=len(set(int(np.argmax(amplitude[k])) for k in range(4)))==1
    display(Markdown(('Сборка с наибольшим размахом одинакова во всех четырёх сценариях.' if stable else 'Сборка с наибольшим размахом зависит от выбранных свойств тканей.')+' Это вывод об устойчивости данного модельного сравнения. Рекомендация по восстановлению объёма требует отдельной проверки разделимости влияний объёма и координат, ошибок сегментации и измерительного шума.'))
    tj=montages.index('ttrkg')
    display(Markdown(f'Для **ТТРКГ** исходный импеданс составляет **{fmt(Z[:,tj,0].min(),2)}–{fmt(Z[:,tj,0].max(),2)} Ом**. При свойствах миокарда размах равен **{fmt(1000*amplitude[[0,2],tj].min(),2)}–{fmt(1000*amplitude[[0,2],tj].max(),2)} мОм**, при свойствах крови — **{fmt(1000*amplitude[[1,3],tj].min(),2)}–{fmt(1000*amplitude[[1,3],tj].max(),2)} мОм**. Эти интервалы сравнивают два выбранных свойства лёгкого и не являются доверительными интервалами. Даже при неизменной динамике маски электрическое представление сердца существенно меняет прогнозируемый сигнал.'))
    falls=[]
    for k,name in [(0,'миокарда'),(1,'крови')]:
        for j,m in enumerate(montages):
            change=100*(amplitude[k+2,j]/amplitude[k,j]-1)
            if change<0:falls.append(f'{label(m)} при свойствах {name}: {fmt(change,2)}%')
    if falls:
        display(Markdown('Увеличение удельного сопротивления лёгкого не обязательно увеличивает размах сердечного отклика. В этом сравнении размах уменьшился для следующих случаев: **'+'; '.join(falls)+'**. Следовательно, выбор более высокого сопротивления ткани нельзя использовать как универсальный способ усиления сигнала: он изменяет пространственное распределение тока и влияние движения сердечной области.'))
    old=read(ROOT/'Colab Notebooks/cardiac_4d_segmentation/results/nix_cardiac_cycle_20260916/summary.json')
    old_lookup={(r['id'],r['montage']):r['Z_ohm'] for r in old['solutions']}
    table(['Сборка','Прежнее Z₀, Ом']+[t+': новое Z₀, Ом' for t in titles],[[label(m),fmt(old_lookup[('phase_00',m)])]+[fmt(Z[k,j,0]) for k in range(4)] for j,m in enumerate(montages)])
    display(Markdown('Сравнение с прежним расчётом показывает совокупный эффект нового набора сопротивлений и выделения жира. Приписывать всё различие только жиру нельзя: одновременно изменены свойства фоновых мягких тканей, лёгкого и сердца. Отдельная проверка матриц при одинаковой проводимости жира и фона подтверждает корректность добавления материала, но не измеряет его физиологический вклад.'))

def checks():
    d=load();qc=d['checks']
    display(Markdown('## Численные проверки и ограничения'))
    table(['Проверка','Результат'],[
        ['Решения',str(d['completed'])+' / '+str(d['required'])],
        ['Наибольшая относительная невязка',f"{qc['max_residual']:.3g}"],
        ['Сборка пяти блоков против EIDORS',f"{qc['max_matrix_error']:.3g}"],
        ['Совпадение с прежними блоками при равных свойствах жира и фона',f"{qc['max_fat_equal_error']:.3g}"],
        ['Изменение интеграла жира между 4096 и 16384 точками, мл',fmt(qc['fat_quadrature_volume_difference_ml'])]])
    display(Markdown('Все решения проверяются по невязке, восстановлению токов и взаимности. Доли материалов интегрируются общими точками внутри каждого тетраэдра: пересечение жира с сердцем вычисляется совместно, а не произведением независимо усреднённых долей. Совпадение интегралов и матриц не доказывает пространственную сходимость FEM или физическую точность.\n\nОсновные ограничения: автоматическая и обрезанная сердечная маска; статический жир; неизменяемый фон вместо механики тканей; мышечные свойства всего остаточного фона; два условных представления всей сердечной области; действительная скалярная проводимость; фиксированные контакты. Эти расчёты не отделяют изменение объёма от перемещения и деформации и не дают независимо валидированного ударного объёма.'))

def publish(run):
    run=Path(run);p=read(run/'electrical/plan.json');summary=read(run/'electrical/summary.json');geo=read(run/'materials/geometry.json')
    assert summary['status']=='complete' and summary['completed']==summary['required']==336
    pairs={(r['scenario'],r['id'],r['montage']) for r in summary['solutions']};assert len(pairs)==336
    assert summary['plan_sha256']==hashlib.sha256((run/'electrical/plan.json').read_bytes()).hexdigest()
    exports=[read(run/'electrical'/s['id']/'export.json') for s in geo['states']]
    import sys
    sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
    import cardiac_cycle_fat_electrical as electrical
    import heart_replacement_electrical_pilot as core
    electrical.plan_at(run/'electrical');core.verify_files(p['inputs'])
    for row in summary['solutions']:
        assert electrical.completed(run/'electrical',p,row['scenario'],row['id'],row['montage'])==row
    for phase,e in zip(geo['states'],exports):
        assert e['status']=='passed' and e['plan_sha256']==summary['plan_sha256']
        assert core.sha(run/'electrical'/phase['id']/'blocks.mat')==e['blocks_sha256']
    RESULTS.mkdir(parents=True,exist_ok=True)
    keys=['id','rr_percent','derived_time_from_R0_ms','rr_interval_ms','mask_volume_ml','material_volume_ml','center_fem_mm','fat_volume_ml']
    d=dict(summary,geometry=[{k:s[k] for k in keys} for s in geo['states']],montages=[c['id'] for c in p['contacts']],
        registry_id=p['registry_id'],registry_sha256=p['registry_sha256'],physical_validation=False,spatial_convergence=False,
        contact_parameters=p['contact_parameters'],contacts=[{k:c[k] for k in ['id','areas_mm2','actual_centres_xyz_mm']} for c in p['contacts']],
        fat_geometry={'segmented_volume_ml':geo['fat_source']['segmented_volume_ml'],'background_volume_ml':geo['fat_background_volume_ml'],'outside_ct_centroid_volume_ml':geo['element_centroid_outside_static_ct_volume_ml']},
        checks={'max_residual':max(r['max_relative_residual'] for r in summary['solutions']),
                'max_matrix_error':max(e['heterogeneous_error'] for e in exports),
                'max_fat_equal_error':max(max(e['fat_equal_old_blocks_relative_errors']) for e in exports),
                'fat_quadrature_volume_difference_ml':abs(np.diff(geo['fat_background_quadrature_volumes_ml'])[0])})
    d['first_phase_potentials_mV_at_1mA']={}
    for scenario in p['scenarios']:
        values={}
        for montage in ['tepc_6','tepc_7']:
            with np.load(run/'electrical/solutions'/scenario['id']/'phase_00'/montage/'fields.npz') as fields:
                u=fields['potentials_ohm'][-4:,0]
            expected=next(r['Z_ohm'] for r in summary['solutions'] if (r['scenario'],r['id'],r['montage'])==(scenario['id'],'phase_00',montage))
            assert abs((u[1]-u[2])-expected)<1e-8
            values[montage]=u.tolist()
        d['first_phase_potentials_mV_at_1mA'][scenario['id']]=values
    d['parameter_selection_note']='The author requested a higher muscle resistivity; the assistant selected 5 ohm m within the registered interval. The sealed plan wording User-selected refers to the requested change, not an author-specified numerical value.'
    (RESULTS/'summary.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import csv
    z0={(r['scenario'],r['montage']):r['Z_ohm'] for r in summary['solutions'] if r['id']=='phase_00'}
    with (RESULTS/'impedance_by_phase.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['scenario','phase','montage','rr_percent','Z_ohm','delta_Z_from_phase00_ohm','max_relative_residual']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in summary['solutions']:
            writer.writerow(dict(scenario=r['scenario'],phase=r['id'],montage=r['montage'],rr_percent=r['rr_percent'],
                Z_ohm=r['Z_ohm'],delta_Z_from_phase00_ohm=r['Z_ohm']-z0[(r['scenario'],r['montage'])],max_relative_residual=r['max_relative_residual']))
    shutil.copy2(ROOT/'Colab Notebooks/cardiac_4d_segmentation/results/nix_cardiac_cycle_20260916/montages.png',RESULTS/'montages.png')
    shutil.copy2(ROOT/'Colab Notebooks/cardiac_4d_segmentation/results/nix_cardiac_cycle_20260916/geometry.png',RESULTS/'geometry.png')
    ct_overlay(run,geo)
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    folder=ROOT/'Colab Notebooks';archive=folder/'archive/private/20260916_4023_before_fat';archive.mkdir(parents=True,exist_ok=True)
    for suffix in ['.ipynb','.html','.language-review.json']:
        old=folder/(TITLE+suffix);dest=archive/old.name
        if old.exists() and not dest.exists():shutil.copy2(old,dest)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Изменение импеданса TEPC и ТТРКГ по 4D-КТ Nix: литературные сценарии и отдельный жир'),
        nbformat.v4.new_code_cell('from cardiac_4d_segmentation import heart_cardiac_cycle_fat_report as report\nreport.intro()'),
        nbformat.v4.new_code_cell('report.results()'),nbformat.v4.new_code_cell('report.checks()')],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(nb,timeout=300,resources={'metadata':{'path':str(folder)}}).execute()
    nbformat.write(nb,folder/(TITLE+'.ipynb'))
    html,_=HTMLExporter(exclude_input=True,exclude_input_prompt=True,exclude_output_prompt=True).from_notebook_node(nb);(folder/(TITLE+'.html')).write_text(html,encoding='utf-8')
    review={'status':'pending_full_text_model_review','physical_validation':False,'artifacts':{suffix:hashlib.sha256((folder/(TITLE+suffix)).read_bytes()).hexdigest() for suffix in ['.ipynb','.html']}}
    (folder/(TITLE+'.language-review.json')).write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')

def ct_overlay(run,geo):
    import sys
    sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
    import heart_replacement_electrical_pilot as core
    core.configure(Path(read(run/'config.local.json')['deps']),1)
    import nibabel as nib
    from scipy.ndimage import map_coordinates
    cfg=read(run/'config.local.json');seg=Path(cfg['fat_segmentation'])
    ct=nib.load(seg/'static_ct.nii.gz');fat=nib.load(seg/'fat.nii.gz')
    heart_path=next(v['path'] for v in geo['inputs'] if Path(v['path']).name=='whole_heart.nii.gz' and Path(v['path']).parent.name=='phase_00')
    heart=nib.load(heart_path);h=np.asanyarray(heart.dataobj)>0
    centre=nib.affines.apply_affine(np.linalg.inv(ct.affine),nib.affines.apply_affine(heart.affine,np.column_stack(np.where(h)).mean(axis=0)))
    density=np.asanyarray(ct.dataobj);fm=np.asanyarray(fat.dataobj)>0
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for axis,ax in enumerate(axes):
        idx=int(round(centre[axis]));others=[k for k in range(3) if k!=axis]
        aa,bb=np.meshgrid(np.arange(ct.shape[others[0]]),np.arange(ct.shape[others[1]]),indexing='ij')
        grid=np.empty(aa.shape+(3,));grid[...,axis]=idx;grid[...,others[0]]=aa;grid[...,others[1]]=bb
        hp=nib.affines.apply_affine(np.linalg.inv(heart.affine)@ct.affine,grid)
        hs=map_coordinates(h.astype(float),hp.reshape(-1,3).T,order=0,mode='constant',cval=0).reshape(aa.shape)
        sl=np.take(density,idx,axis=axis).T;fs=np.take(fm,idx,axis=axis).T
        spacing=nib.affines.voxel_sizes(ct.affine)[others]
        extent=[0,sl.shape[1]*spacing[0],0,sl.shape[0]*spacing[1]]
        ax.imshow(sl,cmap='gray',vmin=-180,vmax=250,origin='lower',extent=extent)
        rgba=np.zeros(fs.shape+(4,));rgba[fs]=[1,.55,0,.5];ax.imshow(rgba,origin='lower',extent=extent)
        ax.contour(hs.T,levels=[.5],colors=['red'],linewidths=.8,origin='lower',extent=extent)
        orientation=nib.aff2axcodes(ct.affine)[axis]
        title={'A':'Фронтальное сечение','P':'Фронтальное сечение','R':'Сагиттальное сечение','L':'Сагиттальное сечение','S':'Поперечное сечение','I':'Поперечное сечение'}[orientation]
        ax.set(title=title,xlabel='мм',ylabel='мм')
    fig.savefig(RESULTS/'fat_ct_overlay.png',dpi=150,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);publish(ap.parse_args().run)
