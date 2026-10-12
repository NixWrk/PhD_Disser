"""Paired Georg CEM, four material derivatives and an explicit inversion gate."""
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from build_georg_fem_reports import ROOT,new_cells,export,table
from lateral_pilot import read,sha
PIPE=ROOT/'MATLAB_TRKG4_real_subjects'
sys.path.insert(0,str(PIPE/'tools'))
from georg_model_contract import resolve
OUT=PIPE/'output/exploratory/georg_paired_montages_20261012'
RUN=OUT/'h6'
STEM='33.25_Совместные_монтажи_и_готовность_подбора_тканей_Георгия_Scorpion_v2'
INDICES=[0,1,3,4]


def montage_label(item):
    number=int(item['record_id'][-3:])
    variant=item['variant'].replace('x','×')+('?' if 7<=number<=10 else '')
    companion=('ТЕПС-'+str(number-10) if item['family']=='TEPC' else
        'боковая '+str({2:140,3:140,4:140,7:120,8:100,9:80,10:60}[number])+' мм')
    return f'{number:03d} · {companion} + ТТРКГ {variant}'


def verify():
    _,current=resolve();plan=read(OUT/'plan.local.json');forward=read(RUN/'forward_results.local.json')
    if len(forward['cases'])!=len(plan['scenarios']):raise ValueError('Incomplete paired series')
    for source in plan['sources']:
        if sha(source['path'])!=source['sha256']:raise ValueError('Paired input changed')
    identity=forward['identity']
    if identity['input_sha256']!=sha(OUT/'plan.local.json'):raise ValueError('Paired plan changed')
    for key,path in [('worker_sha256',PIPE/'src/run_georg_paired_montages.m'),
            ('CEM_sha256',PIPE/'src/trkg4_cem_contact_matrix.m'),
            ('patch_builder_sha256',PIPE/'vendor_stl_eidors/electrode_faces_by_area.m'),
            ('response_sha256',PIPE/'src/trkg4_lateral_cem_response.m')]:
        if sha(path)!=identity[key]:raise ValueError('Executed CEM code changed')
    if any(r['identity']!=identity for r in forward['cases']):raise ValueError('Mixed CEM versions')
    return plan,forward,current


def comparison_table():
    _,forward,_=verify();observed=pd.read_csv(OUT/'observations.local.csv');rows=[]
    for item in forward['cases']:
        if not item['contact_qc_passed']:continue
        T=np.array(item['port_transfer_ohm']);delta=np.array(item['quadrature_difference_ohm'])
        first=1 if int(item['record_id'][-3:])<=4 or item['family']=='TEPC' else 2
        for lead,ch,index in [('ТТРКГ',first,(1,0)),('ТЕПС' if item['family']=='TEPC' else 'Боковая',3-first,(3,2))]:
            target=observed[(observed.record_id==item['record_id'])&(observed.channel==ch)]
            values={r['state']:r['baseline_raw_median_ohm'] for r in target.to_dict('records')}
            rows.append({'Монтаж':montage_label(item),'Отведение':lead,'BEI-канал':ch,
                'Вдох: измерено, Ом':values.get('inhale'),'Выдох: измерено, Ом':values.get('exhale'),
                'FEM: опорный сценарий, Ом':float(T[index]),'Разность интегрирования, Ом':float(delta[index]),
                'Проверка двух собственных передач':item['own_source_quadrature_passed']})
    return pd.DataFrame(rows)


def materials_table():
    _,forward,_=verify();rho=forward['cases'][0]['rho_ohm_m']
    names=['Мягкотканный фон','Лёгкие','Кости','Сердце целиком','Жир']
    symbols=['ρ₁','ρ₂','ρ костей','ρ₃','ρ₄']
    return pd.DataFrame([{'Материал':name,'Параметр':symbol,'Опорное ρ, Ом·м':rho[index],
        'Роль в будущем подборе':'Фиксированный материал' if index==2 else 'Свободный параметр'}
        for index,(name,symbol) in enumerate(zip(names,symbols))])


def derivatives_table():
    _,forward,_=verify();rows=[]
    for item in forward['cases']:
        if not item['contact_qc_passed']:continue
        J=np.array(item['J_logrho_port_ohm'])
        for lead,index in [('ТТРКГ',(1,0)),('ТЕПС' if item['family']=='TEPC' else 'Боковая',(3,2))]:
            rows.append({'Монтаж':montage_label(item),'Отведение':lead,
                **dict(zip(['J₁: фон, Ом','J₂: лёгкие, Ом','J₃: сердце, Ом','J₄: жир, Ом'],J[index][INDICES]))})
    return pd.DataFrame(rows)


def rank_table():
    _,forward,_=verify();rows=[]
    for variant in ['1x1','1x4']:
        matrices=[]
        for item in forward['cases']:
            if not item['contact_qc_passed']:continue
            if 7<=int(item['record_id'][-3:])<=10 and item['variant']!=variant:continue
            J=np.array(item['J_logrho_port_ohm']);matrices.extend([J[1,0,INDICES],J[3,2,INDICES]])
        matrix=np.array(matrices);sv=np.linalg.svd(matrix,compute_uv=False)
        rows.append({'Поздняя ТТРКГ':variant.replace('x','×'),'Число передач':len(matrix),
            'Локальный ранг':int(np.linalg.matrix_rank(matrix)),'Обусловленность':float(sv[0]/sv[-1]),
            **{f'σ{i+1}, Ом':float(s) for i,s in enumerate(sv)}})
    return pd.DataFrame(rows)


def derivatives_figure():
    _,forward,_=verify();fig=go.Figure()
    items=[r for r in forward['cases'] if r['family']=='TEPC' and r['contact_qc_passed']]
    for material,label,color in [(0,'Фон','#d97706'),(1,'Лёгкие','#248ca6'),(3,'Сердце','#bd3c54'),(4,'Жир','#9b8435')]:
        fig.add_trace(go.Bar(x=['ТЕПС-'+str(int(r['record_id'][-3:])-10) for r in items],
            y=[np.array(r['J_logrho_port_ohm'])[3,2,material] for r in items],name=label,marker_color=color))
    fig.update_layout(template='plotly_white',barmode='group',height=470,
        title='Производные ТЕПС для подготовки подбора: литературные опорные ρ',yaxis_title='∂Z / ∂lnρ, Ом')
    return fig


def mesh_control_table():
    _,forward,_=verify()
    fine=read(OUT/'h4_control/002_1x1.local.json')
    coarse=next(r for r in forward['cases'] if r['id']=='002_1x1')
    pairs=[(6,4,coarse,fine)]
    third=OUT/'h3_control/002_1x1.local.json'
    if third.is_file():
        pairs.append((4,3,fine,read(third)))
    rows=[]
    for low,high,left,right in pairs:
        for key in ['input_sha256','worker_sha256','CEM_sha256','patch_builder_sha256',
                'response_sha256','inactive_arm_pads_included']:
            if right['identity'][key]!=left['identity'][key]:
                raise ValueError('Mesh comparison used different paired operators')
        if not left['contact_qc_passed'] or not right['contact_qc_passed']:
            raise ValueError('Mesh comparison has failed finite-contact QC')
        for label,index in [('ТТРКГ',(1,0)),('Боковая 140 мм',(3,2))]:
            a=np.array(left['port_transfer_ohm'])[index];b=np.array(right['port_transfer_ohm'])[index]
            tolerance=.02+.005*abs(b)
            rows.append({'Отведение':label,'Сетки, мм':f'{low} → {high}',
                'Первая сетка: Z, Ом':a,'Следующая сетка: Z, Ом':b,
                'Разность, Ом':abs(b-a),'Допуск, Ом':tolerance,'Критерий пройден':bool(abs(b-a)<=tolerance)})
    return pd.DataFrame(rows)


def disconnected_pads_table():
    _,forward,_=verify()
    connected=read(OUT/'h6_connected_only/002_1x1.local.json')
    floating=next(r for r in forward['cases'] if r['id']=='002_1x1')
    for key in ['input_sha256','worker_sha256','prepared_identity','CEM_sha256',
            'patch_builder_sha256','response_sha256']:
        if connected['identity'][key]!=floating['identity'][key]:
            raise ValueError('Disconnected-pad control changed more than the contact set')
    assert not connected['identity']['inactive_arm_pads_included']
    assert floating['identity']['inactive_arm_pads_included']
    return pd.DataFrame([{'Отведение':label,
        'С плавающими площадками, Ом':np.array(floating['port_transfer_ohm'])[index],
        'Только подключённые, Ом':np.array(connected['port_transfer_ohm'])[index],
        'Изменение, Ом':np.array(connected['port_transfer_ohm'])[index]-np.array(floating['port_transfer_ohm'])[index]}
        for label,index in [('ТТРКГ',(1,0)),('Боковая 140 мм',(3,2))]])


def cells():
    verify();cc,md,code=new_cells()
    md('''# Совместные монтажи Георгия и готовность подбора четырёх сопротивлений

Рассчитаны полные контактные наборы: ТТРКГ с боковой сборкой или ТЕПС.
Для ТЕПС 2–7 использована реконструкция Георгия и одновременная ТТРКГ 1×1.
В позднем боковом блоке отдельно сохранены 1×1 и 1×4. Основной оператор,
как у Ника по указанию автора, использует собственную токовую пару
соответствующего отведения.

**Статус.** Выполнена прямая подготовка обратной задачи. Опорные ρ взяты
из литературного сценария 100 кГц и не подбирались к данным. Общая модель
ещё не прошла пространственную проверку точности. Количественный подбор,
передача его параметров в пульсовую коррекцию и сердечную сферу пока
остановлены численным контрактом.''')
    code('from IPython.display import display, HTML\nfrom build_georg_paired_preflight_report import *\nplan, forward, current = verify()')
    md('''## Материалы и контактные группы

Свободные параметры будущего подбора: ρ₁ — фон, ρ₂ — лёгкие,
ρ₃ — эффективная область всего сердца, ρ₄ — жир. Кости представлены
отдельной маской и фиксированным сопротивлением. В FEM-массивах порядок
иной: фон, лёгкие, кости, сердце, жир; производные явно приведены
к четырём свободным параметрам.

Контакты моделируются дисками 5 мм; соединённые площадки образуют
эквипотенциальный терминал CEM. Неиспользуемые контакты неподвижных
электродов на руках в этом контроле сохранены как отдельные плавающие
терминалы с нулевым суммарным током. Их присутствие во всех ранних
записях не установлено отдельно: это сценарий для сопоставления
с вариантом только подключённых контактов. Площадки проверены по площади,
центру, радиусу и отсутствию перекрытия. Такая проверка не устанавливает
точность ручного восстановления установки.

Для бокового ряда здесь сохранён иллюстративный центр 20.20;
он не заменяет совместный поиск положения и ρ. Вертикальные и
перекрёстные установки остаются в локализации 33.23. Для АЕПС
фактические координаты отсутствуют. Эти записи доступны в полном
сравнении 10.00 и не исключены из экспериментального реестра.

Входы — медианы общего BEI в кандидатных дыхательных окнах 11.20.
КТ получена на глубоком вдохе в отдельной сессии. Перенос одной геометрии
на выдох и сопоставимость последовательных записей остаются допущениями.
Около 7 мА относятся к каждому каналу по автору; точная амплитудная
конвенция не задана. Единичный ток в FEM служит нормировкой U/I.''')
    md('''Опорные числа ниже — литературные сценарии общей модели при 100 кГц.
Фон приближается мышечной тканью, кости — кортикальным материалом,
сердце целиком — материалом миокарда. Это рабочие назначения для прямой
проверки, а не измеренные индивидуальные параметры или установленная
электрическая смесь крови и миокарда.''')
    code('display(table(materials_table()))\ndisplay(table(comparison_table()))')
    md('''## Контроль неподключённых площадок

Для монтажа 002 выполнен второй расчёт на той же сетке 6 мм:
оставлены только подключённые контакты. Исключение плавающих площадок
меняет ТТРКГ примерно на 0,013 Ом, боковую 140 мм — на 0,0006 Ом.
В этом опорном сценарии эффект меньше найденного расхождения между
сетками. Проверка не устанавливает фактическое наличие площадок и
не переносится автоматически на другие соединения или сопротивления.''')
    code('display(table(disconnected_pads_table()))')
    md('''## Производные для подготовки совместного подбора

Сопряжённый расчёт на Георгии даёт Jᵢ = ∂Z/∂lnρᵢ в Ом при фиксированных
контактах и масках. Эти операторы нужны для обратной задачи и не являются
долями зарегистрированного пульса. В первом приближении вклад изменения
ткани ΔZᵢ ≈ Jᵢ δρᵢ/ρᵢ; амплитуда δρᵢ также должна быть обоснована.

Ниже сохранены производные обоих отведений и локальный ранг четырёх
параметров при каждой гипотезе поздней ТТРКГ. Ранг в опорной точке
не доказывает глобальную идентифицируемость, единственность позы
или устойчивость по всему диапазону ρ. Требуемое исследование ТЕПС
в согласованных с экспериментом базовых точках выполняется после
успешного подбора сопротивлений. В этой опорной точке обусловленность
составляет примерно 335–359: разные сочетания относительных изменений
параметров существенно различаются по чувствительности. Без модели
ошибок этот показатель не задаёт доверительных интервалов ρ.''')
    code('display(table(derivatives_table()))\ndisplay(table(rank_table()))\ndisplay(HTML(derivatives_figure().to_html(full_html=False,include_plotlyjs=True)))')
    md('''## Численное условие продолжения

Квадратуры сравниваются для двух собственных передач при неизменной сетке.
Допуск: 0,02 Ом + 0,1% модуля опорного импеданса. Отдельно сохраняется
отрицательное сравнение пространственных сеток общей модели. Допуски
после отказа не расширяются. Проверки системы и взаимности подтверждают
внутреннее решение, но не заменяют эти проверки точности и анатомическую
валидацию.

Будущий подбор должен совместно учитывать боковые и ТТРКГ-сигналы,
позу боковых матриц, дыхательные состояния и четыре ρ. Для 007–010
оба варианта соединения сохраняются; физические позиции на руках
остаются общими. Вместе с параметрами нужны профили невязки, сравнимые
минимумы, влияние фиксированных костей и неопределённость установки.

После успешного согласования базовых точек отдельно выполняются
чувствительности ТЕПС и подбор сердечной сферы. Сфера сравнивается
с сердечной маской Георгия при тех же контактах, фоне и материалах.
Преобразование её электрического объёма в УО/ФВ требует независимой
проверки. Эти последующие расчёты здесь не объявляются выполненными.''')
    md('''Дополнительный контроль того же нового монтажа 002 на сетках 6 и
4 мм также не прошёл критерий 0,02 Ом + 0,5% опорного Z. Для ТТРКГ
разность составляет около 1,88 Ом при допуске около 0,50 Ом; для боковой
140 мм — около 0,78 Ом при допуске около 0,33 Ом. Это отказ проверки
точности текущей дискретизации, независимо от будущей подгонки ρ.''')
    code('display(table(mesh_control_table()))')
    if (OUT/'h3_control/002_1x1.local.json').is_file():
        md('''Контроль дополнен независимо построенной сеткой 3 мм с тем же
наружным контуром, тканевыми масками и новым парным монтажом 002.
Переход 4 → 3 мм также не прошёл прежний допуск: разность ТТРКГ
составляет около 1,38 Ом при допуске около 0,50 Ом, боковой 140 мм —
около 1,23 Ом при допуске около 0,34 Ом. Граница, положительность
элементов, конечные контакты, линейная система и сравнение квадратур
проверены отдельно. Увеличение числа тетраэдров до 9,1 млн пока не
устранило пространственное расхождение. Это одна контрольная
конфигурация; статус общей модели и запрет количественной инверсии
сохранены.''')
    code("try:\n    resolve(require_converged=True)\n    gate = 'Численный контракт общей модели разрешает инверсию'\nexcept ValueError:\n    gate = 'Инверсия остановлена: точность общей FEM-модели ещё не подтверждена'\ndisplay(HTML('<p><strong>'+gate+'</strong></p>'))")
    return cc


if __name__=='__main__':export(STEM,cells(),OUT)
