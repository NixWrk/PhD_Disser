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
from build_georg_geometry_revision_report import OUT as GEOMETRY, SEGMENTATION, SOURCE, verify as verify_geometry
from experiment_signal_display import ecg_variants, spectrum, phase_intervals, pack_full_samples
from scorpion_protocol import channel_assignment, ttrkg_configuration, arm_groups

FOLDER=Path(__file__).parent
OUT=EXPERIMENT/'derived/exploratory/10.00_experiment_view_20261009'
STEM='10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2'
PRIMARY={'001':140,'002':140,'003':140,'004':140,'007':120,'008':100,'009':80,'010':60}
TISSUES={'fat':('Жир','#dba832',.20,None),'bone':('Кости','#c8b69b',.85,12000),
        'lung':('Лёгкие','#49aebc',.35,10000),'heart':('Сердце','#c52b47',.90,6000)}


def decimate_display(mesh,count):
    import pymeshlab
    ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(mesh.vertices,mesh.faces))
    options=dict(targetfacenum=count,preservenormal=True,preserveboundary=True)
    ms.apply_filter('meshing_decimation_quadric_edge_collapse',**options)
    m=ms.current_mesh()
    return m.vertex_matrix(),m.face_matrix()


def check_surface_samples(source,vertices,faces,tolerance_mm=3.):
    """Reject spurious interior sheets; sampled distance is not a Hausdorff bound."""
    from scipy.spatial import cKDTree
    tri=vertices[faces]
    points=np.concatenate([vertices,tri.mean(axis=1),(tri[:,0]+tri[:,1])/2,
        (tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2])
    distance=cKDTree(source.vertices).query(points)[0]
    maximum=float(distance.max())
    if maximum>tolerance_mm:
        raise ValueError(f'Display surface departs from source samples: {maximum:.3f} mm > {tolerance_mm} mm')
    return dict(samples=len(points),maximum_distance_to_source_vertices_mm=maximum,
        p95_distance_to_source_vertices_mm=float(np.quantile(distance,.95)),
        tolerance_mm=tolerance_mm,checked_points='vertices, face centroids and three edge midpoints',
        is_hausdorff_bound=False,is_segmentation_accuracy=False)


def fat_display_surface(mask_path,source):
    """Use a spatial CT grid instead of collapsing a thin anatomical shell."""
    import nibabel as nib
    from skimage.measure import marching_cubes
    image=nib.load(mask_path)
    vertices,faces,_,_=marching_cubes((np.asanyarray(image.dataobj)>0).astype(np.uint8),
        level=.5,step_size=3)
    vertices=nib.affines.apply_affine(image.affine,vertices)*[-1.,-1.,1.]
    qc=check_surface_samples(source,vertices,faces)
    qc.update(method='marching_cubes on original mask, no subsequent triangle collapse',
        grid_step_voxels=3,grid_step_mm=(3*np.array(image.header.get_zooms())).tolist(),
        faces=len(faces),maximum_edge_mm=float(np.linalg.norm(
            vertices[faces[:,[1,2,0]]]-vertices[faces],axis=2).max()),
        topology_preservation_claimed=False)
    return vertices,faces,qc


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
    """Nominal geometry is independent of the record's physical channel roles."""
    rid=record['record_id'][-3:]
    configuration=ttrkg_configuration(record['record_id'])
    result={'lateral':[],'perpendicular':[],'arm':arm_groups(contacts,configuration),
        'tepc':[],'mapping':'unknown','ttrkg_configuration':configuration,
        'single_U_angle_author_confirmed':True}
    if configuration and configuration['status']=='unresolved_hypothesis':
        result['arm_alternatives']={variant:arm_groups(contacts,ttrkg_configuration(record['record_id'],variant))
            for variant in ('1x1','1x4')}
    if 12<=int(rid)<=17:
        path=GEOMETRY/'landmarks'/f'georg_tepc_{int(rid)-10}.json'
        data=read(path)
        if data['measurement_context']['recording']['id']!=rid:
            raise ValueError('TEPC recording mapping changed')
        result.update(tepc=[data['preliminary_electrode_centers_xyz_mm'][k]
            for k in ['I_plus','V_plus','V_minus','I_minus']],mapping='tepc',
            landmark_sha256=sha(path),author_reconstruction=True,
            new_ct_projection=True,placement_accuracy_mm=None)
    sizes=contacts['sizes_mm']
    if rid in PRIMARY or rid in ('005','006','011'):
        size=PRIMARY.get(rid,60 if rid=='011' else 140)
        result['lateral']=contacts['lateral']['centres_by_size_mm'][sizes.index(size)]
        if rid in ('006','011'):
            result['perpendicular']=transverse_centres(source,size,contacts['lateral']['y_mm'],contacts['lateral']['z_mm'])
            result['mapping']='cross'
        else:
            result['mapping']='horizontal' if rid not in ('001','005') else ('test_horizontal' if rid=='001' else 'vertical')
            if rid=='005':
                result['perpendicular']=transverse_centres(source,size,contacts['lateral']['y_mm'],contacts['lateral']['z_mm'])
                result['lateral']=[]
    if rid=='018':result['mapping']='aepc_unlocalized'
    return result


def verify_montages(records,contacts):
    """Verify every displayed nominal montage against its geometry, not acquisition."""
    body=trimesh.load(GEOMETRY/'body_embedded_arms_370mm.stl',force='mesh')
    maximum=0.;points=0
    for record in records:
        rid=record['record_id'][-3:];m=record['montage']
        expected_arm=arm_groups(contacts,ttrkg_configuration(record['record_id']))
        assert len(m['arm'])==len(expected_arm)
        for a,b in zip(m['arm'],expected_arm):np.testing.assert_allclose(a,b,atol=1e-12)
        if rid in PRIMARY or rid in ('006','011'):
            size=PRIMARY.get(rid,60 if rid=='011' else 140)
            np.testing.assert_allclose(m['lateral'],contacts['lateral']['centres_by_size_mm'][contacts['sizes_mm'].index(size)],atol=1e-12)
            np.testing.assert_allclose(np.array(m['lateral'])[:,2],contacts['lateral']['z_mm']+np.array([-size/2,-size/4,size/4,size/2]),atol=1e-8)
        if 12<=int(rid)<=17:
            assert not m['lateral'] and not m['perpendicular'] and m['mapping']=='tepc'
            path=GEOMETRY/'landmarks'/f'georg_tepc_{int(rid)-10}.json'
            assert sha(path)==m['landmark_sha256']
            expected=read(path)['preliminary_electrode_centers_xyz_mm']
            np.testing.assert_allclose(m['tepc'],[expected[k] for k in
                ['I_plus','V_plus','V_minus','I_minus']],atol=1e-12)
        elif int(rid)==18:assert m['mapping']=='aepc_unlocalized' and not m['tepc']
        p=m['lateral']+m['perpendicular']+m['tepc']+[p for group in m['arm'] for p in group]
        if p:
            _,distance,_=body.nearest.on_surface(np.array(p));maximum=max(maximum,float(distance.max()));points+=len(p)
    if maximum>.02:raise ValueError('Displayed contact is not on the corrected exterior')
    return dict(records_checked=len(records),contact_centres_checked=points,
        maximum_contact_distance_mm=maximum,actual_localisation_validated=False)


def run(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);verify_geometry()
    review=EXPERIMENT/'derived/scorpion_review';summary_path=review/'summary.local.json'
    annotation_path=EXPERIMENT/'annotations/scorpion_review/annotations_candidates.local.json'
    summary=read(summary_path);annotations=read(annotation_path)
    if summary['code_sha256']!=sha(FOLDER/'scorpion_qc.py'):
        raise ValueError('Signal cache producer changed')
    source=trimesh.load(SOURCE,force='mesh');contacts=read(GEOMETRY/'contacts.local.json')
    body=trimesh.load(GEOMETRY/'body_embedded_arms_370mm.stl',force='mesh')
    vertices,faces=decimate_display(body,40000);display=dict(vertices=vertices,faces=faces)
    sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
    from build_georg_tepc_picker import mask_surface
    import nibabel as nib
    ct=nib.load(SEGMENTATION/'static_ct.nii.gz');mask_hashes={};surface_checks={}
    for name,(_,_,_,count) in TISSUES.items():
        mask=SEGMENTATION/(name+'.nii.gz');image=nib.load(mask)
        if image.shape!=ct.shape or not np.allclose(image.affine,ct.affine,atol=1e-5,rtol=0):
            raise ValueError('Tissue mask and CT coordinate frames differ')
        target=out/(name+'_lps.local.stl');mask_surface(mask,target)
        original=trimesh.load(target,force='mesh')
        if name=='fat':v,f,surface_checks[name]=fat_display_surface(mask,original)
        else:v,f=decimate_display(original,count)
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
            t=a['time_s'];traces=[];full_fields={'time_s':t}
            for row,kind,scale in [(1,'bei_raw',1),(2,'bei_slow',1),(3,'bei_pulse_band',1000)]:
                for channel in (1,2):
                    y=a[f'{kind}_{channel}']*scale;idx=extrema_indices(y)
                    full_fields[f'{kind}_{channel}']=y
                    traces.append(dict(row=row,channel=channel,kind=kind,t=np.round(t[idx],6).tolist(),y=np.round(y[idx],7).tolist()))
            breathing=annotation['breathing'];boundaries=breathing.get('boundaries_samples',[])
            start=(boundaries[0]/fs+6) if boundaries else max(0.,float(t[-1])/2-2.5)
            start=min(start,max(0.,float(t[-1])-5));first=int(round(start*fs));last=min(len(t),first+int(5*fs))
            variants={key:[] for key in ('00','10','01','11')}
            for channel in (1,2):
                full=ecg_variants(a[f'ecg_raw_{channel}'],fs)
                np.testing.assert_allclose(full['11'],a[f'ecg_filtered_{channel}'],rtol=1e-10,atol=1e-10)
                for key,y in full.items():
                    full_fields[f'ecg_{key}_{channel}']=y
                    idx=extrema_indices(y);freq,psd=spectrum(y,fs)
                    variants[key].append(dict(channel=channel,t=np.round(t[idx],6).tolist(),y=np.round(y[idx],7).tolist(),
                        detail_y=np.round(y[first:last],7).tolist(),psd=psd.tolist()))
        label=rid+' · '+record['montage_label']
        if rid in ('007','008','009','010'):
            label=rid+' · боковая '+str(PRIMARY[rid])+' мм · ТТРКГ 1×1? / 1×4?'
        records.append(dict(record_id=record['record_id'],label=label,source_montage_label=record['montage_label'],fs=fs,
            duration=float(t[-1]),timebase_status=record['timebase_status'],traces=traces,ecg_variants=variants,
            detail_t=np.round(t[first:last],6).tolist(),psd_frequency_hz=freq.tolist(),
            full_samples=pack_full_samples(full_fields),detail_indices=[first,last],
            channel_assignment=channel_assignment(record['record_id']),
            breathing={**{k:breathing.get(k) for k in ('boundaries_samples','boundary_uncertainty_ranges_samples','single_mode','author_accepted')},
                'phases':phase_intervals(breathing,fs,float(t[-1]))},
            montage=montage(record,contacts,source)))
        sources.append(dict(record_id=record['record_id'],source_sha256=record['source_sha256'],cache_sha256=sha(cache_path)))
    if len(records)!=18:raise ValueError('Expected eighteen independent recordings')
    montage_checks=verify_montages(records,contacts)
    save(out/'display_signals.local.json',records)
    save(out/'contract.local.json',dict(producer_sha256=sha(__file__),summary_sha256=sha(summary_path),
        annotations_sha256=sha(annotation_path),geometry_contacts_sha256=sha(GEOMETRY/'contacts.local.json'),
        geometry_surface_sha256=sha(GEOMETRY/'body_embedded_arms_370mm.stl'),sources=sources,
        decimation='Default: every original sample, lossless gzip float64; optional 900-bucket min/max overview; ECG detail always unthinned',
        filters=summary['filters'],coordinates_are_scenarios=True,actual_electrode_localisation_recovered=False,
        signal_display_producer_sha256=sha(FOLDER/'experiment_signal_display.py'),
        viewer_js_sha256=sha(FOLDER/'experiment_view.js'),
        tissue_surface_producer_sha256=sha(ROOT/'MATLAB_TRKG4_real_subjects/tools/build_georg_tepc_picker.py'),
        tissue_mask_sha256=mask_hashes,tissue_display_faces={name:len(display[name+'_faces']) for name in TISSUES},
        tissue_surface_checks=surface_checks,
        fat_display_decimation='marching cubes, step 3 voxels; original mask unchanged; no triangle collapse',
        ct_sha256=sha(SEGMENTATION/'static_ct.nii.gz'),
        ct_section_sha256=sha(section),ct_section_plane_lps_y_mm=contacts['lateral']['y_mm'],
        ct_section_provenance_sha256=sha(GEOMETRY/'section_provenance.local.json'),
        ecg_filter_variants='00 raw; 10 notches; 01 bandpass; 11 notches then bandpass; whole-record filtering before display sampling',
        spectrum='Welch PSD, Hann 8 s windows (short records: whole record), 50% overlap, constant detrend; mV²/Hz; full Nyquist range',
        channel_roles_author_confirmed_records=['001','002','003','004','005','006','007','012','013','014','015','016','017','018'],
        channel_roles_still_unknown_records=[],
        protocol_revision_sha256=sha(FOLDER/'scorpion_protocol.py'),
        late_block_ttrkg_variant_unresolved=True,late_block_ttrkg_scenarios=['1x1','1x4'],
        late_block_continuity_is_protocol_assumption=True,annotations_modified=False,annotations_accepted=False,
        montage_checks=montage_checks,single_U_angle_author_confirmed=True,
        ECG_and_BEI_same_index_share_U_author_confirmation='2026-10-09',
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
        (FOLDER/'scorpion_protocol.py','protocol_revision_sha256'),
        (ROOT/'MATLAB_TRKG4_real_subjects/tools/build_georg_tepc_picker.py','tissue_surface_producer_sha256'),
        (SEGMENTATION/'static_ct.nii.gz','ct_sha256'),(GEOMETRY/'ct_longitudinal_sections.png','ct_section_sha256'),
        (GEOMETRY/'section_provenance.local.json','ct_section_provenance_sha256')]:
        if sha(path)!=c[key]:raise ValueError('Viewer anatomy or filter dependency changed')
    for name,digest in c['tissue_mask_sha256'].items():
        if sha(SEGMENTATION/(name+'.nii.gz'))!=digest:raise ValueError('Viewer mask changed: '+name)
    if c['tissue_surface_checks']['fat']['maximum_distance_to_source_vertices_mm']>3.:
        raise ValueError('Fat display surface has failed geometric QC')
    verify_montages(read(out/'display_signals.local.json'),read(GEOMETRY/'contacts.local.json'))
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
<p><label>История подключения: <select id="experiment-channel"><option value="0">Текущее назначение: подтверждения автора и допущения продолжения</option><option value="1">Прежний сценарий: постоянно боковая BEI1</option><option value="2">Прежний сценарий: постоянно боковая BEI2</option><option value="3">Диагностика: боковая BEI2 в 004–005 → BEI1 с 006</option><option value="4">Диагностика: боковая BEI1 в 004–005 → BEI2 с 006</option></select></label></p>
<p id="experiment-description"></p><p id="experiment-localisation" style="color:#904018"></p>
<p><label>ТТРКГ в 007–010: <select id="experiment-ttrkg-variant"><option value="1x1">1×1 — гипотеза автора</option><option value="1x4">1×4 — прежняя метка файлов</option></select></label> Вариант меняет только показ электродов; измеренные сигналы сохраняются.</p>
<p><label>Отсчёты обзорных графиков: <select id="experiment-sampling"><option value="full">Все отсчёты — без прореживания</option><option value="extrema">Быстрый обзор — минимумы и максимумы 900 интервалов</option></select></label></p>
<fieldset><legend>Фильтры ЭКГ — независимое включение этапов</legend>
<label><input type="checkbox" id="experiment-notch" checked> Режекторы 50/100/150 Гц, Q = 30</label><br>
<label><input type="checkbox" id="experiment-band" checked> Полосовой фильтр Баттерворта (N = 4), 0,5–40 Гц</label>
<p>Обработка всей записи в прямом и обратном направлениях, без фазового сдвига. Режекторы подавляют сетевые частоты; полосовой фильтр ослабляет медленное смещение и высокочастотные компоненты. Оба отключены — исходная ЭКГ.</p>
<p id="experiment-filter-description"></p></fieldset>
<fieldset><legend>Видимость сигналов</legend>
<label><input type="checkbox" id="experiment-show-channel-1" checked> Канал 1 на всех графиках сигналов</label>
<label><input type="checkbox" id="experiment-show-channel-2" checked> Канал 2 на всех графиках сигналов</label>
<p>Ниже можно отдельно скрыть каждый график и каждую его кривую. Выбор сохраняется при смене записи и фильтров. Эти настройки не назначают электродам номера каналов.</p>
<div id="experiment-visibility-controls" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px"></div>
</fieldset>
<p id="experiment-phase-description"></p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:8px"><div id="experiment-ct"></div><div id="experiment-signals"><div id="experiment-bei-raw"></div><div id="experiment-bei-slow"></div><div id="experiment-bei-pulse"></div><div id="experiment-ecg-overview"></div></div></div>
<p>В легенде КТ можно независимо включать жир, кости, лёгкие и сердце. Маски получены автоматически; точность границ не подтверждена.</p>
<div id="experiment-ecg-detail"></div><div id="experiment-ecg-spectrum"></div></div>'''
    return ui+'<script>'+get_plotlyjs()+'</script><script>const experimentPayload='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';'+(FOLDER/'experiment_view.js').read_text(encoding='utf-8')+'</script>'


def recording_table(out=OUT):
    return pd.DataFrame([{'Запись':r['record_id'][-3:],'Исходная метка файла':r['source_montage_label'],
        'Монтаж с уточнениями':r['label'].split(' · ',1)[1],
        'Рабочая частота отсчётов, Гц':r['fs'],'Последний отсчёт, с':r['duration'],
        'Размещение на КТ':('Авторская реконструкция; проекция на новый контур' if r['montage']['mapping']=='tepc'
            else 'Сценарий; координаты не измерены' if r['montage']['mapping']!='unknown' else 'Не локализовано')}
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
Для ТЕПС 2–7 автор передал реконструированные координаты четырёх контактов.
Они перенесены на исправленный наружный контур ближайшей проекцией в
исходной системе LPS; первоначальные координаты сохранены. Точность
восстановления установки неизвестна. Для остальных монтажей показаны
явно заданные геометрические сценарии.
По уточнению автора от 09.10.2026 EMG1 и EMG2 снимались с тех же
измерительных электродов U, что соответственно BEI1 и BEI2.
Качество ЭКГ поэтому может служить дополнительным признаком монтажа;
его диагностическая пригодность проверяется в 10.30.

КТ Георга получена в отдельной сессии на глубоком вдохе с задержкой дыхания.
Наружная поверхность заново построена по полному КТ и используется как
общий геометрический вход последующих расчётов. Поза при регистрации
Scorpion отдельно не восстановлена. Добавлены
автоматические маски жира, костей, лёгких и сердца из TotalSegmentator;
независимая точность их границ не установлена. Все поверхности приведены
к одной системе координат КТ. Для отображения наружная сетка прорежена
до 40 тысяч граней, сетки костей, лёгких и сердца — до 6–12 тысяч.
Эти сетки не используются для FEM-расчёта.

В прежнем 3D-изображении жира упрощение поверхности создало ложные
треугольные перегородки с рёбрами до 283 мм. В исходной поверхности
маски таких перегородок не было. Теперь поверхность жира строится прямо
из исходной маски с шагом три вокселя по каждой оси, без последующего
схлопывания треугольников. Шаг равен 2,637 × 2,637 × 3,75 мм в координатах
КТ; поверхность содержит около 547 тысяч граней. Мелкие детали и отдельные
тонкие участки при таком отображении могут исчезать; их необходимо
проверять на исходных сечениях. Дополнительно проверены вершины,
центры граней и середины рёбер. Расстояние каждой проверенной точки
до ближайшей вершины исходной поверхности не должно превышать 3 мм.
Этот технический допуск относится к отображению; он не задаёт точность
сегментации и не является оценкой максимальной ошибки всей поверхности.
Исходная маска и сечение КТ не изменены.

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
Для 001 и 002 показан монтаж 1×1: единственный U находится на верхней
стороне цилиндра, на той же угловой оси, что I. Это исправленный сценарий;
угол U у 1×1 подтверждён автором 12.10.2026. Тест 001 по уточнению
от 09.10.2026 выполнен с 1×1 и горизонтальной боковой сборкой 140 мм.
Осевые расстояния перенесены из прежнего сценария
геометрии, а не измерены в новой сессии; площадь контактов на руках неизвестна.

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
Отдельным уточнением автора подтверждено: в 001–003 ТТРКГ — канал 1,
боковая горизонтальная сборка — канал 2. По подтверждению автора 12.10.2026
в ТЕПС и АЕПС ТТРКГ 1×1 записана каналом 1; второй канал относится
соответственно к ТЕПС и АЕПС.


В просмотре по умолчанию используется это назначение. Прежние варианты
доступны как диагностические сценарии и не заменяют подтверждение автора.
Переключатель меняет диагностическое назначение сигналов; исходные поля
сохраняются. Цвет задан по отведению и сохраняется при смене номера канала:
ТТРКГ — синий, боковая горизонтальная сборка — оранжевый, вертикальная —
фиолетовый. Электроды на КТ окрашены по типу номинального монтажа;
его геометрия не зависит от гипотезы о номере гнезда прибора. В 005 показаны вертикальная BEI2 и контакты ТТРКГ1.
Сопоставление сигналов приведено в
[10.30](10.30_Критическая_проверка_назначения_каналов_Scorpion_v2.html).

ТЕПС — торсовые электродные конфигурации, АЕПС — электродная конфигурация
на руке по Metshein et al. (EMBC 2021). Схемы источника проверены в
[протоколе](../../Experiments/scorpion_v2_georg/PROTOCOL.md), однако они не
задают координаты данной сессии. Для записей 012–017 на КТ показана авторская
реконструкция ТЕПС 2–7 из шести переданных JSON. Соответствие записей
установлено по именам файлов и подтверждено автором. Во всех этих записях
добавлены четыре верхних контакта ТТРКГ 1×1 на руках. Для АЕПС 018 индивидуальных
координат нет.
[Инструмент размещения ТЕПС](20.21_Размещение_ТЕПС_Георгия_Scorpion_v2.html)
открывает эти конфигурации на новом контуре и позволяет уточнить центры.
Новый ручной экспорт потребует отдельной проверки и пересчёта контактов;
сохранённые FEM-результаты относятся к версии координат, описанной в 20.20.''')
    code('from IPython.display import display, HTML\nfrom build_scorpion_experiment_viewer import verify, widget, recording_table\nfrom build_georg_fem_reports import table\ncontract = verify()')
    md('''## Совместный просмотр КТ и зарегистрированных сигналов

**Рисунок 1.** Выбор одной из 18 записей одновременно обновляет монтаж
и графики. Цвет соответствует отведению: ТТРКГ — синий, боковая
горизонтальная сборка — оранжевый, вертикальная — фиолетовый.
Центры авторской реконструкции ТЕПС показаны зелёным.
ТЕПС на канале 2 обозначена зелёным, АЕПС — коричневым; одновременная
ТТРКГ 1×1 на канале 1 — синим. Номер канала указан отдельно.
Назначение креста 006: горизонтальная ось — канал 1, сохранённая
вертикальная — канал 2, по уточнению автора о переходе через 005.
Для 011 назначение перенесено из 010 по постоянству позднего блока;
качество ЭКГ креста для назначения не используется.
Ракурс, масштаб и смещение КТ сохраняются при переключении записей,
вариантов подключения, фильтров и видимости тканевых слоёв.

Справа расположены четыре отдельных графика: общий BEI в Ом,
расчётный базовый импеданс в Ом, пульсовая полоса в мОм и ЭКГ в мВ.
В блоке видимости можно отключить канал сразу на всех графиках сигналов,
а также отдельно каждый график и каждую его кривую. Канальный переключатель
скрывает и исходную, и выбранную ЭКГ данного канала в её фрагменте и спектре.
После повторного включения сохраняется отдельный выбор кривых. Выбор
видимости и ракурс КТ сохраняются при смене записи и фильтров; скрытие
сигнала не меняет гипотезу подключения или исходные отсчёты.
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
По умолчанию обзорные линии содержат все отсчёты на исходной временной
сетке, без прореживания, интерполяции и дополнительного сглаживания.
Полные массивы сохранены в HTML без потерь в формате float64 с gzip-сжатием;
при открытии записи проверяется их контрольная сумма. Рабочая частота
составляет 1000 Гц, для 005–006 — 400 Гц в явно указанном сценарии.
Отдельный быстрый режим сохраняет минимум и максимум каждого из 900
интервалов. Он уменьшает число отображаемых точек; увеличение масштаба
в нём не восстанавливает пропущенные отсчёты. Для проверки формы
сигнала следует выбрать режим всех отсчётов. Фрагмент ЭКГ 5 с всегда
показывается без прореживания. Плотная линия за всю запись сама по себе
не характеризует качество; форму нужно оценивать при увеличении.
Нижний временной график содержит пятисекундный фрагмент исходной и выбранной ЭКГ
без прореживания. При наличии кандидатной первой границы его начало выбрано
через 6 с после неё, иначе — около середины записи.

Отдельный спектральный график сравнивает исходную ЭКГ и выбранный вариант
за всю запись. Спектральная плотность мощности S в мВ²/Гц оценена методом
Уэлча: окно Ханна длиной 8 с, перекрытие 50%, удаление среднего в каждом
окне. Частота f показана от нуля до половины частоты отсчётов; меньший
диапазон можно выбрать масштабированием. Спектр не является амплитудным
спектром одного сердечного цикла.
В сравнении ЭКГ и спектров сплошная линия соответствует исходной ЭКГ,
штриховая — выбранным фильтрам; цвет обозначает отведение, номер канала
указан в подписи каждой кривой.

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
    md('''## Сравнение ТТРКГ и боковых сборок

Для каждой записи отдельно показаны базовая кривая и медианы общего BEI
на задержках после глубокого вдоха и выдоха. Медианы рассчитаны по всем
исходным отсчётам в кандидатных окнах 11.20: от неоднозначных границ
сделан отступ 2 с, от краёв записи — 5 с. Разметка остаётся непринятой;
подписи дыхания обозначают её текущую реконструкцию. Тест 001 не имеет
глубоких задержек, поэтому входит только в обзор базовых кривых.
Для сравнительного обзора медленная составляющая отображена с частотой
20 Гц; расчёт медиан не использует это прореживание. Время каждой записи
отсчитывается от собственного начала: наложение кривых не означает
синхронность дыхательных команд или разных сессий.

**Сравнение ТТРКГ.** 002 — 1×1, 003 — 2×2, 004 — 1×4 при боковой
140 мм, по подтверждению автора. В 005 вариант 1×4 предполагается по
последовательности. В 007–010 прежние имена файлов содержат 1×4, автор
предполагает 1×1; знак вопроса сохраняет эту неопределённость. В 012–018
подтверждены 1×1 и канал 1. Электроды на руках оставались в своих местах;
менялись подключённые группы. Их отображение в 007–010 можно переключить
между двумя гипотезами, не меняя измерений.

**Сравнение боковых сборок.** Показаны все размеры, повторные 140 мм,
поперечные установки и обе оси креста. Историческая метка «горизонтальная»
соответствует продольной оси показанного монтажа. Повторные записи одного
размера не образуют дополнительные глубины чувствительности. Межзаписные
различия включают состояние, контакт и другое одновременно подключённое
отведение; неизменность геометрии не устраняет эти различия.''')
    code("from scorpion_comparisons import comparison_figure, levels_table, ecg_figure\ndisplay(HTML(comparison_figure('ttrkg').to_html(full_html=False, include_plotlyjs=False)))\ndisplay(table(levels_table('ttrkg')))")
    code("display(HTML(comparison_figure('lateral').to_html(full_html=False, include_plotlyjs=False)))\ndisplay(table(levels_table('lateral')))")
    md('''## Проверка варианта ТТРКГ по ЭКГ

Показана медианная форма ЭКГ на канале ТТРКГ в кандидатной задержке
на вдохе. Используются прежние события QRS, полоса 0,5–40 Гц и режекторы
50/100/150 Гц; полные интервалы QRS–QRS приведены к 200 фазам,
из каждого удалено среднее. Слева сохранена шкала мВ. Справа каждая
медианная кривая разделена на максимум своего модуля только для
сопоставления формы; усиления измерений и временные сдвиги не подбирались.

Поздний блок 007–010 имеет более близкую форму к ранним 2×2 и 1×4,
чем к ранней 1×1: корреляция без сдвига на вдохе составляет примерно
0,99 против 0,93–0,94. Этот признак не определяет конфигурацию:
положение и контакт, длительность циклов и состояние испытуемого
также влияют на ЭКГ. Подтверждённые поздние записи 1×1 дают проверку
вариабельности того же признака. Ни ЭКГ, ни базовый BEI пока не
подтверждают исправление 007–010 на 1×1 однозначно; для нового FEM
сохраняются оба сценария.

Частота зондирования 100 кГц подтверждена автором. Ток около 7000 мкА
сообщён по памяти как приблизительный; автор подтвердил, что число
относится к каждому каналу. Амплитудное или действующее значение
не уточнено. Основной сценарий, как в расчёте Ника по указанию автора,
использует передачу от собственного источника при полном наборе контактов.
В линейной модели передаточный импеданс определяется как U/I,
поэтому численное возбуждение 1 А служит нормировкой и не означает
экспериментальный ток 1 А.''')
    code("display(HTML(ecg_figure().to_html(full_html=False, include_plotlyjs=False)))")
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
Исходная метка файла и монтаж с последующими уточнениями показаны отдельно.
Историческая метка «горизонтальный»
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
    from scorpion_comparisons import run as run_comparisons
    run_comparisons();run();export(STEM,cells(),OUT)
    from build_scorpion_supervisor import build
    build()
