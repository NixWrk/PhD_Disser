"""Executed experiment overview: nominal CT montages linked to cached signals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import trimesh
from plotly.offline import get_plotlyjs
from lateral_pilot import ROOT, EXPERIMENT, read, save, sha
from build_georg_fem_reports import export, table
from build_georg_geometry_revision_report import OUT as GEOMETRY, SOURCE, verify as verify_geometry

FOLDER=Path(__file__).parent
OUT=EXPERIMENT/'derived/exploratory/10.00_experiment_view_20261009'
STEM='10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2'
PRIMARY={'002':140,'003':140,'004':140,'007':120,'008':100,'009':80,'010':60}


def extrema_indices(values,buckets=900):
    """Keep chronological endpoints and extrema of every display bucket."""
    values=np.asarray(values)
    edges=np.linspace(0,len(values),min(buckets,len(values))+1,dtype=int)
    indices=[0,len(values)-1]
    for start,end in zip(edges[:-1],edges[1:]):
        segment=values[start:end]
        indices.extend([start+int(np.argmin(segment)),start+int(np.argmax(segment))])
    return np.unique(indices)


def transverse_centres(source,size,y,z):
    origins=np.array([[-600.,y+d,z] for d in [-size/2,-size/4,size/4,size/2]])
    points,rays,_=source.ray.intersects_location(origins,np.tile([1.,0.,0.],(4,1)))
    result=[]
    for i in range(4):
        hits=points[rays==i]
        if len(hits)==0:raise ValueError('Perpendicular montage ray misses body')
        result.append(hits[np.argmin(hits[:,0])].tolist())
    return result


def montage(record,contacts,source):
    """Channel numbers are assigned in the viewer under one global hypothesis."""
    rid=record['record_id'][-3:]
    result={'lateral':[],'perpendicular':[],'arm':[],'mapping':'unknown'}
    sizes=contacts['sizes_mm']
    if rid in PRIMARY or rid in ('005','006','011'):
        size=PRIMARY.get(rid,60 if rid=='011' else 140)
        result['lateral']=contacts['lateral']['centres_by_size_mm'][sizes.index(size)]
        if rid in ('006','011'):
            result['perpendicular']=transverse_centres(source,size,contacts['lateral']['y_mm'],contacts['lateral']['z_mm'])
            result['mapping']='cross'
        else:
            result['arm']=contacts['ttrkg_2x2' if rid=='003' else 'ttrkg_1x4']
            result['mapping']='horizontal' if rid!='005' else 'unknown_vertical'
            if rid=='002':
                # A displayed single-potential scenario, not recovered acquisition coordinates.
                result['arm']=[g[2:3] if i in (1,2) else g[:1] for i,g in enumerate(result['arm'])]
            if rid=='005':
                result['perpendicular']=transverse_centres(source,size,contacts['lateral']['y_mm'],contacts['lateral']['z_mm'])
                result['lateral']=[]
    return result


def run(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);verify_geometry()
    review=EXPERIMENT/'derived/scorpion_review';summary_path=review/'summary.local.json'
    annotation_path=EXPERIMENT/'annotations/scorpion_review/annotations_candidates.local.json'
    summary=read(summary_path);annotations=read(annotation_path)
    if summary['code_sha256']!=sha(FOLDER/'scorpion_qc.py'):
        raise ValueError('Signal cache producer changed')
    source=trimesh.load(SOURCE,force='mesh');contacts=read(GEOMETRY/'contacts.local.json')
    body=trimesh.load(GEOMETRY/'body_embedded_arms_370mm.stl',force='mesh')
    import pymeshlab
    ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(body.vertices,body.faces))
    ms.apply_filter('meshing_decimation_quadric_edge_collapse',targetfacenum=40000,preservenormal=True,preserveboundary=True)
    mesh=ms.current_mesh();np.savez_compressed(out/'display_geometry.local.npz',vertices=mesh.vertex_matrix(),faces=mesh.face_matrix())
    records=[];sources=[]
    for record in summary['records']:
        rid=record['record_id'][-3:];fs=record['fs_hz_scenario']
        if fs!=(400 if rid in ('005','006') else 1000):continue
        raw_path=EXPERIMENT/record['source_relative_path']
        if sha(raw_path)!=record['source_sha256']:raise ValueError('Changed recording: '+rid)
        annotation=next(a for a in annotations['records'] if a['record_id']==record['record_id'] and a['fs_hz_scenario']==fs)
        if annotation['source_sha256']!=record['source_sha256']:raise ValueError('Annotation source mismatch')
        cache_path=review/(record['key']+'.npz')
        with np.load(cache_path) as a:
            t=a['time_s'];traces=[]
            for row,kind,scale in [(1,'bei_raw',1),(1,'bei_slow',1),(2,'bei_pulse_band',1000),(3,'ecg_filtered',1)]:
                for channel in (1,2):
                    y=a[f'{kind}_{channel}']*scale;idx=extrema_indices(y)
                    traces.append(dict(row=row,channel=channel,kind=kind,t=np.round(t[idx],6).tolist(),y=np.round(y[idx],7).tolist()))
            breathing=annotation['breathing'];boundaries=breathing.get('boundaries_samples',[])
            start=(boundaries[0]/fs+6) if boundaries else max(0.,float(t[-1])/2-2.5)
            start=min(start,max(0.,float(t[-1])-5));first=int(round(start*fs));last=min(len(t),first+int(5*fs))
            detail=[]
            for channel in (1,2):
                for kind in ('ecg_raw','ecg_filtered'):
                    detail.append(dict(channel=channel,kind=kind,t=np.round(t[first:last],6).tolist(),y=np.round(a[f'{kind}_{channel}'][first:last],7).tolist()))
        records.append(dict(record_id=record['record_id'],label=rid+' · '+record['montage_label'],fs=fs,
            duration=float(t[-1]),timebase_status=record['timebase_status'],traces=traces,detail=detail,
            breathing={k:breathing.get(k) for k in ('boundaries_samples','boundary_uncertainty_ranges_samples','single_mode','author_accepted')},
            montage=montage(record,contacts,source)))
        sources.append(dict(record_id=record['record_id'],source_sha256=record['source_sha256'],cache_sha256=sha(cache_path)))
    if len(records)!=18:raise ValueError('Expected eighteen independent recordings')
    save(out/'display_signals.local.json',records)
    save(out/'contract.local.json',dict(producer_sha256=sha(__file__),summary_sha256=sha(summary_path),
        annotations_sha256=sha(annotation_path),geometry_contacts_sha256=sha(GEOMETRY/'contacts.local.json'),
        geometry_surface_sha256=sha(GEOMETRY/'body_embedded_arms_370mm.stl'),sources=sources,
        decimation='900 chronological buckets per overview trace; minima and maxima retained; 5 s ECG detail unthinned',
        filters=summary['filters'],coordinates_are_scenarios=True,actual_electrode_localisation_recovered=False,
        numerical_channel_assignment_confirmed=False,annotations_modified=False,annotations_accepted=False,
        CT_is_separate_session=True,display_mesh_is_not_FEM_mesh=True,display_faces=len(mesh.face_matrix())))
    save(out/'manifest.local.json',dict(outputs={name:sha(out/name) for name in
        ['display_geometry.local.npz','display_signals.local.json','contract.local.json']}))
    return out


def verify(out=OUT):
    out=Path(out);c=read(out/'contract.local.json')
    for name,digest in read(out/'manifest.local.json')['outputs'].items():
        if sha(out/name)!=digest:raise ValueError('Experiment viewer input changed: '+name)
    if sha(__file__)!=c['producer_sha256']:raise ValueError('Viewer producer changed')
    for path,key in [(EXPERIMENT/'derived/scorpion_review/summary.local.json','summary_sha256'),
        (EXPERIMENT/'annotations/scorpion_review/annotations_candidates.local.json','annotations_sha256'),
        (GEOMETRY/'contacts.local.json','geometry_contacts_sha256'),(GEOMETRY/'body_embedded_arms_370mm.stl','geometry_surface_sha256')]:
        if sha(path)!=c[key]:raise ValueError('Viewer reference changed')
    return c


def widget(out=OUT):
    verify(out);out=Path(out)
    with np.load(out/'display_geometry.local.npz') as a:
        payload={'vertices':np.round(a['vertices'],5).tolist(),'faces':a['faces'].tolist()}
    payload['records']=read(out/'display_signals.local.json')
    ui='''<div id="experiment-viewer" style="border:1px solid #aaa;padding:12px">
<p><label>Запись: <select id="experiment-record"></select></label></p>
<p><label>Вариант нумерации подключения: <select id="experiment-channel"><option value="1">Основная боковая ось — BEI1; перпендикулярная ось креста — BEI2</option><option value="2">Основная боковая ось — BEI2; перпендикулярная ось креста — BEI1</option></select></label></p>
<p id="experiment-description"></p><p id="experiment-localisation" style="color:#904018"></p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px"><div id="experiment-ct"></div><div id="experiment-signals"></div></div>
<div id="experiment-ecg-detail"></div></div>'''
    return ui+'<script>'+get_plotlyjs()+'</script><script>const experimentPayload='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';'+(FOLDER/'experiment_view.js').read_text(encoding='utf-8')+'</script>'


def recording_table(out=OUT):
    return pd.DataFrame([{'Запись':r['record_id'][-3:],'Монтаж по имени файла':r['label'].split(' · ',1)[1],
        'Рабочая частота отсчётов, Гц':r['fs'],'Последний отсчёт, с':r['duration'],
        'Размещение на КТ':('Сценарий; координаты не измерены' if r['montage']['mapping']!='unknown' else 'Не локализовано')}
        for r in read(Path(out)/'display_signals.local.json')])


def cells():
    import nbformat as nbf
    cc=[]
    def md(s):cc.append(nbf.v4.new_markdown_cell(s))
    def code(s):cc.append(nbf.v4.new_code_cell(s))
    md('''# Электродные монтажи и сигналы эксперимента Scorpion v2

**Участник:** Георг, псевдоним. **Регистрация:** МГТУ, 05.10.2026.
**Прибор:** Scorpion v2; частота зондирования 100 кГц.
**Статус:** обзор измерений и геометрических сценариев на этапе разработки методики.

Этот ноутбук — начало последовательного разбора: он связывает запись,
номинальный монтаж и сигналы. Далее рассматриваются [настройки прибора](10.20_Настройки_и_бинарные_потоки_Scorpion_v2.md),
[разметка и обработка](11.20_Просмотр_и_разметка_Scorpion_v2.md),
[двуслойная обратная задача](33.21_Множество_решений_двуслойной_модели_Scorpion_v2.html)
и [индивидуальная КТ/FEM-геометрия](20.20_Индивидуальная_КТ_FEM_модель_Георгия_Scorpion_v2.html).

## Что измерено и что показано на КТ

Измерены два общих импедансных сигнала BEI в Ом и два поля EMG в мВ,
которые автор определил как ЭКГ. На Scorpion базовая и пульсовая составляющие
не записываются отдельно. Монтаж и последовательность проб известны из
сообщений автора и имён файлов; фотографии данной установки отсутствуют.
Фактические координаты электродов и номера каналов их подключения пока
не установлены.

КТ Георга получена в отдельной сессии на глубоком вдохе с задержкой дыхания.
Показанная наружная поверхность служит анатомической основой сценария,
а не реконструкцией позы во время регистрации Scorpion. Для отображения
сетка прорежена до 40 тысяч граней; расчётная FEM-сетка этим не изменена.

По подтверждению автора центр всех размеров и осей креста один, на правой
боковой поверхности грудной клетки. В сценарии он задан на уровне z = −210 мм
при y = 0 мм в системе LPS: x направлен влево, y — назад, z — к голове.
Основная ось располагается вдоль тела, перпендикулярная — по y.
Это выбранные координаты для показа; общий центр ещё не восстановлен из КТ.
Буквы A, B обозначают токовые, M, N — измерительные контакты. Маркеры показывают
центры, их экранный размер не соответствует физической площади электрода.

Руки представлены цилиндрами радиуса 58,887 мм по сообщённому обхвату 37 см
на уровне токовых электродов обеих рук. Полное сечение цилиндров погружено
в тело. Для 1×4 показан один верхний токовый контакт и четыре измерительных
контакта через 90° на каждой руке. Для 2×2 добавлен нижний токовый контакт,
измерительная пара остаётся спереди и сзади как рабочая гипотеза.
В варианте 1×1 показан один измерительный контакт спереди — его угловое
положение не подтверждено. Осевые расстояния перенесены из прежнего сценария
геометрии, а не измерены в новой сессии; площадь контактов на руках неизвестна.

В записях креста оба импедансных канала боковые. Для основной размерной серии
сохранены два варианта нумерации каналов: BEI1 — основная ось, BEI2 —
перпендикулярная ось креста, либо обратное назначение. Эти варианты
обозначают неизвестную нумерацию подключения, а не разные физиологические
механизмы. Назначение сохраняется во всём основном размерном ряду.
Переключатель меняет
условные подписи и цвета электродов, а сами сигналы каждого поля сохраняются.
Для теста, отдельной записи «верт» и ТЕПС/АЕПС это подтверждение нумерации
не распространяется. В «верт» показывается только номинальное направление
сборки без назначения её каналу.

ТЕПС — торсовые электродные конфигурации, АЕПС — электродная конфигурация
на руке по Metshein et al. (EMBC 2021). Схемы источника проверены в
[протоколе](../../Experiments/scorpion_v2_georg/PROTOCOL.md), однако они не
задают координаты данной сессии. Для этих записей доступен просмотр сигналов;
электроды на КТ не нанесены до восстановления индивидуальной установки.
Для такого восстановления подготовлен [инструмент ручного размещения ТЕПС](../../MATLAB_TRKG4_real_subjects/output/exploratory/georg_tepc_manual_20261009/georg_tepc_picker.html).
Он позволяет отдельно разметить ТЕПС 2–7 на КТ Георгия и сохранить координаты
с их происхождением в JSON и CSV. Ручной экспорт ещё не является
подтверждением точности установки или готовым FEM-расчётом.''')
    code('from IPython.display import display, HTML\nfrom build_scorpion_experiment_viewer import verify, widget, recording_table\nfrom build_georg_fem_reports import table\ncontract = verify()')
    md('''## Совместный просмотр КТ и зарегистрированных сигналов

**Рисунок 1.** Выбор одной из 18 записей одновременно обновляет монтаж
и графики. Синий цвет соответствует каналу 1, оранжевый — каналу 2;
в геометрии неизвестного подключения используются серые маркеры.
Ракурс, масштаб и смещение КТ сохраняются при переключении записей и гипотез.

Справа показаны исходный BEI и его медленная составляющая ниже 0,5 Гц,
полосовая составляющая BEI 0,7–12 Гц в мОм и ЭКГ после режекторов
50/100/150 Гц и полосового фильтра 0,5–40 Гц. Это частотное разделение,
которое само по себе не выделяет сердце, лёгкие или мягкие ткани.
На обзорных линиях сохраняются минимум и максимум каждого из 900 временных
интервалов; пульсовую форму следует проверять в полном просмотре этапа 11.20.
Нижний график содержит пятисекундный фрагмент исходной и обработанной ЭКГ
без прореживания. При наличии кандидатной первой границы его начало выбрано
через 6 с после неё, иначе — около середины записи.

Пунктирные границы и серые полосы — ранее подготовленная кандидатная
реконструкция дыхательного протокола и неоднозначности её границ из 11.20.
Она не принята автором и не является зарегистрированными командами прибора.
Распознавание зубцов и новая разметка в этом ноутбуке не выполняются.

Рабочая шкала времени использует 400 Гц для записей «140 верт» и «140 крест»,
1000 Гц — для остальных. Основание — разбор настроек BIN; для первых двух
записей сохранено допущение о настройке по умолчанию. Шкала времени
и частотная обработка зависят от этого допущения.''')
    code('display(HTML(widget()))')
    md('''## Состав доступных записей

**Таблица 1.** Записи, рабочая временная шкала и статус анатомического размещения.
Названия монтажей сохранены по файлам; историческая метка «горизонтальный»
не задаёт ось сценария КТ, которая по уточнению автора проходит вдоль тела.''')
    code('display(table(recording_table()))')
    md('''## Использование обзора в методике

Сначала проверяются соответствие записи протоколу и качество сигналов,
затем принимаются интервалы дыхания и ЭКГ. После этого можно сопоставлять
уровни и амплитуды разных сборок при явно указанной гипотезе каналов.
Для основного ряда последовательно записанные размеры считаются
сопоставимыми по физиологическому состоянию; это рабочее допущение.
Контакт, повторная установка и изменения состояния между записями пока
не разделены количественно.

Размещение на КТ позволяет проверить пространственный охват сборок
и планировать расчёты чувствительности. Восстановление действительного
центра, осей и площадей контактов должно предшествовать сравнению
индивидуального моделирования с измерениями, особенно для ТЕПС и АЕПС.
Просмотр сигналов не подтверждает параметры тканей и не даёт УО или ФВ.''')
    return cc


if __name__=='__main__':
    run();export(STEM,cells(),OUT)
