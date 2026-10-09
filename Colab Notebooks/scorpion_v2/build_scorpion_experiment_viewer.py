"""Executed experiment overview: nominal CT montages linked to cached signals."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import trimesh
from plotly.offline import get_plotlyjs
from lateral_pilot import ROOT, EXPERIMENT, read, save, sha
from build_georg_fem_reports import export, table
from build_georg_geometry_revision_report import OUT as GEOMETRY, SOURCE, verify as verify_geometry
from experiment_signal_display import ecg_variants, spectrum, phase_intervals

FOLDER=Path(__file__).parent
OUT=EXPERIMENT/'derived/exploratory/10.00_experiment_view_20261009'
STEM='10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2'
PRIMARY={'002':140,'003':140,'004':140,'007':120,'008':100,'009':80,'010':60}
SEGMENTATION=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_fem_20261008/segmentation'
TISSUES={'fat':('Жир','#dba832',.20,12000),'bone':('Кости','#c8b69b',.85,12000),
        'lung':('Лёгкие','#49aebc',.35,10000),'heart':('Сердце','#c52b47',.90,6000)}


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
    def decimate(mesh, count):
        ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(mesh.vertices,mesh.faces))
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',targetfacenum=count,preservenormal=True,preserveboundary=True)
        m=ms.current_mesh()
        return m.vertex_matrix(),m.face_matrix()
    vertices,faces=decimate(body,40000);display=dict(vertices=vertices,faces=faces)
    sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
    from build_georg_tepc_picker import mask_surface
    import nibabel as nib
    ct=nib.load(SEGMENTATION/'static_ct.nii.gz');mask_hashes={}
    for name,(_,_,_,count) in TISSUES.items():
        mask=SEGMENTATION/(name+'.nii.gz');image=nib.load(mask)
        if image.shape!=ct.shape or not np.allclose(image.affine,ct.affine,atol=1e-5,rtol=0):
            raise ValueError('Tissue mask and CT coordinate frames differ')
        target=out/(name+'_lps.local.stl');mask_surface(mask,target)
        v,f=decimate(trimesh.load(target,force='mesh'),count)
        display[name+'_vertices']=v;display[name+'_faces']=f;mask_hashes[name]=sha(mask)
    np.savez_compressed(out/'display_geometry.local.npz',**display)
    section=GEOMETRY/'ct_longitudinal_sections.png'
    if not section.exists():raise ValueError('Build the 20.20 CT section before the viewer')
    section_provenance=read(GEOMETRY/'section_provenance.local.json')
    if (section_provenance['mask_sha256']!=mask_hashes or
        section_provenance['ct_sha256']!=sha(SEGMENTATION/'static_ct.nii.gz') or
        section_provenance['plane_y_lps_mm']!=contacts['lateral']['y_mm'] or
        section_provenance['builder_sha256']!=sha(FOLDER/'build_georg_geometry_revision_report.py')):
        raise ValueError('20.20 section anatomy or plane is stale')
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
            for row,kind,scale in [(1,'bei_raw',1),(2,'bei_slow',1),(3,'bei_pulse_band',1000)]:
                for channel in (1,2):
                    y=a[f'{kind}_{channel}']*scale;idx=extrema_indices(y)
                    traces.append(dict(row=row,channel=channel,kind=kind,t=np.round(t[idx],6).tolist(),y=np.round(y[idx],7).tolist()))
            breathing=annotation['breathing'];boundaries=breathing.get('boundaries_samples',[])
            start=(boundaries[0]/fs+6) if boundaries else max(0.,float(t[-1])/2-2.5)
            start=min(start,max(0.,float(t[-1])-5));first=int(round(start*fs));last=min(len(t),first+int(5*fs))
            variants={key:[] for key in ('00','10','01','11')}
            for channel in (1,2):
                full=ecg_variants(a[f'ecg_raw_{channel}'],fs)
                np.testing.assert_allclose(full['11'],a[f'ecg_filtered_{channel}'],rtol=1e-10,atol=1e-10)
                for key,y in full.items():
                    idx=extrema_indices(y);freq,psd=spectrum(y,fs)
                    variants[key].append(dict(channel=channel,t=np.round(t[idx],6).tolist(),y=np.round(y[idx],7).tolist(),
                        detail_y=np.round(y[first:last],7).tolist(),psd=psd.tolist()))
        records.append(dict(record_id=record['record_id'],label=rid+' · '+record['montage_label'],fs=fs,
            duration=float(t[-1]),timebase_status=record['timebase_status'],traces=traces,ecg_variants=variants,
            detail_t=np.round(t[first:last],6).tolist(),psd_frequency_hz=freq.tolist(),
            breathing={**{k:breathing.get(k) for k in ('boundaries_samples','boundary_uncertainty_ranges_samples','single_mode','author_accepted')},
                'phases':phase_intervals(breathing,fs,float(t[-1]))},
            montage=montage(record,contacts,source)))
        sources.append(dict(record_id=record['record_id'],source_sha256=record['source_sha256'],cache_sha256=sha(cache_path)))
    if len(records)!=18:raise ValueError('Expected eighteen independent recordings')
    save(out/'display_signals.local.json',records)
    save(out/'contract.local.json',dict(producer_sha256=sha(__file__),summary_sha256=sha(summary_path),
        annotations_sha256=sha(annotation_path),geometry_contacts_sha256=sha(GEOMETRY/'contacts.local.json'),
        geometry_surface_sha256=sha(GEOMETRY/'body_embedded_arms_370mm.stl'),sources=sources,
        decimation='900 chronological buckets per overview trace; minima and maxima retained; 5 s ECG detail unthinned',
        filters=summary['filters'],coordinates_are_scenarios=True,actual_electrode_localisation_recovered=False,
        signal_display_producer_sha256=sha(FOLDER/'experiment_signal_display.py'),
        viewer_js_sha256=sha(FOLDER/'experiment_view.js'),
        tissue_surface_producer_sha256=sha(ROOT/'MATLAB_TRKG4_real_subjects/tools/build_georg_tepc_picker.py'),
        tissue_mask_sha256=mask_hashes,tissue_display_faces={name:len(display[name+'_faces']) for name in TISSUES},
        ct_sha256=sha(SEGMENTATION/'static_ct.nii.gz'),
        ct_section_sha256=sha(section),ct_section_plane_lps_y_mm=contacts['lateral']['y_mm'],
        ct_section_provenance_sha256=sha(GEOMETRY/'section_provenance.local.json'),
        ecg_filter_variants='00 raw; 10 notches; 01 bandpass; 11 notches then bandpass; whole-record filtering before display sampling',
        spectrum='Welch PSD, Hann 8 s windows (short records: whole record), 50% overlap, constant detrend; mV²/Hz; full Nyquist range',
        numerical_channel_assignment_confirmed=False,annotations_modified=False,annotations_accepted=False,
        CT_is_separate_session=True,display_mesh_is_not_FEM_mesh=True,display_faces=len(faces)))
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
    for path,key in [(FOLDER/'experiment_signal_display.py','signal_display_producer_sha256'),
        (FOLDER/'experiment_view.js','viewer_js_sha256'),
        (ROOT/'MATLAB_TRKG4_real_subjects/tools/build_georg_tepc_picker.py','tissue_surface_producer_sha256'),
        (SEGMENTATION/'static_ct.nii.gz','ct_sha256'),(GEOMETRY/'ct_longitudinal_sections.png','ct_section_sha256'),
        (GEOMETRY/'section_provenance.local.json','ct_section_provenance_sha256')]:
        if sha(path)!=c[key]:raise ValueError('Viewer anatomy or filter dependency changed')
    for name,digest in c['tissue_mask_sha256'].items():
        if sha(SEGMENTATION/(name+'.nii.gz'))!=digest:raise ValueError('Viewer mask changed: '+name)
    return c


def widget(out=OUT):
    verify(out);out=Path(out)
    with np.load(out/'display_geometry.local.npz') as a:
        payload={'vertices':np.round(a['vertices'],5).tolist(),'faces':a['faces'].tolist()}
        payload['tissues']=[dict(name=label,color=color,opacity=opacity,
            vertices=np.round(a[name+'_vertices'],5).tolist(),faces=a[name+'_faces'].tolist())
            for name,(label,color,opacity,_) in TISSUES.items()]
    payload['records']=read(out/'display_signals.local.json')
    ui='''<div id="experiment-viewer" style="border:1px solid #aaa;padding:12px">
<p><label>Запись: <select id="experiment-record"></select></label></p>
<p><label>Вариант нумерации подключения: <select id="experiment-channel"><option value="1">Основная боковая ось — BEI1; перпендикулярная ось креста — BEI2</option><option value="2">Основная боковая ось — BEI2; перпендикулярная ось креста — BEI1</option></select></label></p>
<p id="experiment-description"></p><p id="experiment-localisation" style="color:#904018"></p>
<fieldset><legend>Фильтры ЭКГ — независимое включение этапов</legend>
<label><input type="checkbox" id="experiment-notch" checked> Режекторы 50/100/150 Гц, Q = 30</label><br>
<label><input type="checkbox" id="experiment-band" checked> Полосовой фильтр Баттерворта (N = 4), 0,5–40 Гц</label>
<p>Обработка всей записи в прямом и обратном направлениях, без фазового сдвига. Режекторы подавляют сетевые частоты; полосовой фильтр ослабляет медленное смещение и высокочастотные компоненты. Оба отключены — исходная ЭКГ.</p>
<p id="experiment-filter-description"></p></fieldset>
<p id="experiment-phase-description"></p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px"><div id="experiment-ct"></div><div id="experiment-signals"></div></div>
<p>В легенде КТ можно независимо включать жир, кости, лёгкие и сердце. Маски получены автоматически; точность границ не подтверждена.</p>
<div id="experiment-ecg-detail"></div><div id="experiment-ecg-spectrum"></div></div>'''
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
а не реконструкцией позы во время регистрации Scorpion. Добавлены
автоматические маски жира, костей, лёгких и сердца из TotalSegmentator;
независимая точность их границ не установлена. Все поверхности приведены
к одной системе координат КТ. Для отображения наружная сетка прорежена
до 40 тысяч граней, тканевые — до 6–18 тысяч; эти сетки не используются
для FEM-расчёта.

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
контакта через 90° на каждой руке. Для 2×2 обе пары — токовая I и
измерительная U — расположены сверху и снизу: по уточнению автора
от 09.10.2026 измерительная пара также повёрнута на 90°.
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
Ракурс, масштаб и смещение КТ сохраняются при переключении записей,
вариантов подключения, фильтров и видимости тканевых слоёв.

Справа расположены четыре отдельных графика: общий BEI в Ом,
расчётный базовый импеданс в Ом, пульсовая полоса в мОм и ЭКГ в мВ.
Базовый импеданс здесь определён как результат фильтра нижних частот
Баттерворта с частотой среза 0,5 Гц; пульсовая полоса — как результат
полосовой фильтрации 0,7–12 Гц. Параметр порядка N = 4, обработка
выполняется в прямом и обратном направлениях с дополнением краёв на 5 с.
Базовая и пульсовая кривые не составляют полного разложения общего BEI:
остаток, равный общему сигналу за вычетом этих двух составляющих,
сохранён в обработке 11.20. Такое частотное разделение само по себе
не выделяет сердце, лёгкие или мягкие ткани; пульсовая полоса может
содержать и помехи.

Два переключателя ЭКГ независимо включают группу режекторных фильтров
50/100/150 Гц с добротностью Q = 30 и полосу Баттерворта 0,5–40 Гц
с параметром порядка N = 4. Для полосового фильтра N = 4 соответствует
восьмому порядку одного прохода; прямой и обратный проходы изменяют
амплитудную характеристику и устраняют фазовый сдвиг. При включении обоих
этапов сначала применяются режекторы, затем полоса. Отключение обоих
возвращает исходную ЭКГ. Каждый вариант рассчитан по всей записи до
выбора фрагмента, с дополнением краёв на 3 с; возможные краевые эффекты
следует учитывать. Переключение не изменяет исходные данные и разметку.
На обзорных линиях сохраняются минимум и максимум каждого из 900 временных
интервалов; пульсовую форму следует проверять в полном просмотре этапа 11.20.
Нижний временной график содержит пятисекундный фрагмент исходной и выбранной ЭКГ
без прореживания. При наличии кандидатной первой границы его начало выбрано
через 6 с после неё, иначе — около середины записи.

Отдельный спектральный график сравнивает исходную ЭКГ и выбранный вариант
за всю запись. Спектральная плотность мощности S в мВ²/Гц оценена методом
Уэлча: окно Ханна длиной 8 с, перекрытие 50%, удаление среднего в каждом
окне. Частота f показана от нуля до половины частоты отсчётов; меньший
диапазон можно выбрать масштабированием. Спектр не является амплитудным
спектром одного сердечного цикла.

Цветные интервалы на всех четырёх обзорных графиках и фрагменте ЭКГ
обозначают спокойное дыхание, задержку на вдохе, усиленное дыхание и
задержку на выдохе. Для тестовой записи показано только спокойное дыхание.
Пунктирные границы и серые полосы — ранее подготовленная кандидатная
реконструкция протокола и неоднозначности её границ из 11.20. Названия
режимов следуют сообщённому автором порядку проб. Эта разметка не принята
автором и не является зарегистрированными командами прибора.
Распознавание зубцов и новая разметка в этом ноутбуке не выполняются.

Рабочая шкала времени использует 400 Гц для записей «140 верт» и «140 крест»,
1000 Гц — для остальных. Основание — разбор настроек BIN; для первых двух
записей сохранено допущение о настройке по умолчанию. Шкала времени
и частотная обработка зависят от этого допущения.''')
    code('display(HTML(widget()))')
    md('''## Сечение КТ в плоскости бокового монтажа

**Рисунок 2.** То же сечение, что в 20.20: плоскость y LPS = 0 мм
содержит продольную ось боковых сборок. Слева показано полное сечение,
справа — область сборки 140 мм с токовыми A, B и измерительными M, N.
Цветом отмечены лёгкие, кости, сердце и жир; белая линия показывает
наружную поверхность, голубая — первую границу маски лёгкого.
Все размеры сборок имеют один номинальный центр. Здесь показан сценарий
его положения; это сечение не задаёт плоскости ТЕПС/АЕПС и не заменяет
восстановление фактического монтажа. Расстояние до лёгкого и эквивалентная
толщина двуслойной модели рассматриваются раздельно в 20.20 и 33.21.''')
    code("from IPython.display import Image\nfrom build_scorpion_experiment_viewer import GEOMETRY\ndisplay(Image(filename=str(GEOMETRY/'ct_longitudinal_sections.png')))")
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
