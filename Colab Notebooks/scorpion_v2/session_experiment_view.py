"""Subject data adapter for the joint CT/signal reader used in Georg's 33.27."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import trimesh
from plotly.offline import get_plotlyjs

import session_report_figures as F
import subject_report_figures as R
from audit_scorpion_session import read,sha
from build_session_placement_3d import mask_mesh,reduce_mesh
from experiment_signal_display import ecg_variants,spectrum,pack_full_samples

FOLDER=Path(__file__).resolve().parent
LAYERS={'body':('Поверхность Inobitec и руки','#b9b0a8',.16,35000),
        'lung':('Лёгкие','#419aba',.45,28000),
        'heart':('Целое сердце','#b91c1c',.65,16000),
        'bone':('Кости','#b6a57f',.6,38000),
        'fat':('Жир','#eab308',.2,28000)}


def extrema_indices(values,buckets=900):
    """The same display-bucket rule as 33.27, without importing Georg's model."""
    values=np.asarray(values)
    edges=np.linspace(0,len(values),min(buckets,len(values))+1,dtype=int)
    indices=[0,len(values)-1]
    for start,end in zip(edges[:-1],edges[1:]):
        segment=values[start:end]
        indices.extend([start+int(np.argmin(segment)),start+int(np.argmax(segment))])
    return np.unique(indices)


def geometry(subject):
    cfg,plan,q=R.geometry(subject);root=R.model_root(subject)
    passport=read(root/'surface/preparation.local.json')
    products={}
    for filename in ('exterior.stl','contacts.local.json'):
        path=root/'surface'/filename
        if sha(path)!=passport['products'][filename]:raise ValueError('Prepared geometry changed: '+filename)
        products[filename]=path
    exterior=trimesh.load(products['exterior.stl'],force='mesh')
    layers=[];sources=[]
    for kind,(label,color,opacity,limit) in LAYERS.items():
        source=products['exterior.stl'] if kind=='body' else Path(cfg['inputs']['masks'][kind])
        digest=sha(source)
        if kind!='body' and digest!=passport['sources'][str(source.resolve())]:
            raise ValueError('Own tissue mask changed: '+kind)
        mesh=reduce_mesh(exterior.copy(),limit) if kind=='body' else mask_mesh(source,limit)
        layers.append(dict(name=label,color=color,opacity=opacity,
            vertices=np.round(mesh.vertices,5).tolist(),faces=mesh.faces.tolist()))
        sources.append(dict(role=kind,path=str(source),sha256=digest,display_faces=len(mesh.faces)))
    contacts={c['id']:c for c in read(products['contacts.local.json'])['physical_contacts']}
    body=layers[0];bounds=exterior.bounds;pad=np.ptp(bounds,axis=0)*.06
    payload=dict(vertices=body['vertices'],faces=body['faces'],tissues=layers[1:],
        scene_ranges=[[float(bounds[0,i]-pad[i]),float(bounds[1,i]+pad[i])] for i in range(3)])
    audit=dict(subject=subject,sources=sources,contacts_sha256=sha(products['contacts.local.json']),
        exterior_sha256=sha(products['exterior.stl']),coordinates='LPS_mm',display_only=True,
        uses_prepared_projected_contact_centres=True,circumference_mm=370,delta_Y_scenario_mm=-15,
        interactive_camera_verified=False)
    return payload,contacts,audit


def montage(record,contacts):
    context=record['context'];variant=context['ttrkg_variant'];size=context['lateral_size_mm']
    kind=context['record_kind'];all_arm=[k for k in contacts if k.startswith('arm')]
    selected=[k for k in all_arm if variant=='2x2' or k.endswith('_top')]
    current=[k for k in selected if '_I_' in k];potential=[k for k in selected if '_U_' in k]
    if kind=='reciprocity':current,potential=potential,current
    side=[f'side{size}_{role}' for role in ('A','M','N','B')]
    inactive=[k for k in all_arm if k not in selected]
    if kind=='lateral_only':inactive=all_arm;current=[];potential=[]
    if kind=='ttrkg_only':inactive+=side;side=[]
    def points(ids):return [contacts[k]['projected_lps_mm'] for k in ids]
    description=(f'Боковая {size} мм; ТТРКГ {variant.replace("x","×")}. '
        'Собственный наружный STL Inobitec; цилиндры рук с обхватом 37 см; '
        'рабочее смещение бокового центра −15 мм по Y. Координаты установки являются сценарием.')
    if kind in ('ttrkg_only','lateral_only'):
        description+=' Разъём второй сборки вынут из прибора; оставшиеся контакты показаны серым как допущение.'
    if variant=='1x1':description+=' Для 1×1 используются верхние контакты; оставленные нижние площадки — допущение.'
    if kind=='reciprocity':description+=' Токовая и измерительная пары ТТРКГ переставлены местами; координаты не менялись.'
    if kind=='switching':description+=' Подключение менялось в ходе записи; физические позиции электродов сохранялись.'
    return dict(lateral=points(side),current=points(current),potential=points(potential),
        inactive=points(inactive),contact_ids=dict(lateral=side,current=current,potential=potential,inactive=inactive),
        description=description)


def breathing(annotation,record):
    duration=(record['n']-1)/record['fs'];fs=record['fs']
    bands=annotation['boundary_uncertainty_ranges_s']
    if record['context']['record_kind']=='switching':
        phases=[dict(start=0.,end=duration,label='Глубокий вдох с задержкой')]
        note='Глубокий вдох на протяжении пробы подтверждён автором. Окна состояния разъёмов — кандидатная реконструкция.'
    else:
        # Keep ambiguity intervals as gaps; do not invent point events at their midpoints.
        spans=[(0,bands[0][0]),(bands[0][1],bands[1][0]),(bands[1][1],bands[2][0]),(bands[2][1],duration)]
        labels=['Спокойное дыхание','Задержка на глубоком вдохе','Усиленное дыхание','Задержка на глубоком выдохе']
        phases=[dict(start=float(a),end=float(b),label=label) for (a,b),label in zip(spans,labels)]
        note='Кандидатная реконструкция. Серые полосы — неоднозначные границы; точные моменты команд не назначены.'
    return dict(phases=phases,boundaries_samples=[],
        boundary_uncertainty_ranges_samples=[] if not bands else (np.asarray(bands)*fs).tolist(),note=note)


def record_payload(record,annotation,contacts):
    if annotation['source_sha256']!=record['source_sha256'] or annotation['fs_hz']!=record['fs']:
        raise ValueError('Annotation does not belong to this recording')
    a=F.arrays(record['record_id']);fs=record['fs'];t=np.arange(record['n'])/fs
    fields={'time_s':t};traces=[];variants={key:[] for key in ('00','10','01','11')}
    start=record['windows'].get('inhale',[int(max(0,t[-1]/2-2.5)*fs)])[0]
    first=min(start,max(0,len(t)-int(5*fs)));last=min(len(t),first+int(5*fs))
    for kind,key,scale in [('bei_raw','raw',1),('bei_slow','slow',1),('bei_pulse_band','pulse',1000)]:
        for ch in (1,2):
            y=a[f'{key}_{ch}']*scale;fields[f'{kind}_{ch}']=y;idx=extrema_indices(y)
            traces.append(dict(channel=ch,kind=kind,t=t[idx].tolist(),y=y[idx].tolist()))
    for ch in (1,2):
        ecg=ecg_variants(a[f'ecg_raw_{ch}'],fs)
        np.testing.assert_allclose(ecg['11'],a[f'ecg_{ch}'],rtol=1e-10,atol=1e-10)
        for key,y in ecg.items():
            fields[f'ecg_{key}_{ch}']=y;idx=extrema_indices(y);freq,psd=spectrum(y,fs)
            variants[key].append(dict(t=t[idx].tolist(),y=y[idx].tolist(),detail_y=y[first:last].tolist(),psd=psd.tolist()))
    roles={str(ch):('lateral_horizontal' if role=='lateral' else role) for role,ch in record['physical_channel_roles'].items()}
    switch=record['context']['record_kind']=='switching'
    active=None if switch else [('lateral_horizontal' if role=='lateral' else role) for role in record['active_roles']]
    notes={} if switch else {role:' · отключена: технические числа' for role in ('ttrkg','lateral_horizontal') if role not in active}
    return dict(record_id=record['record_id'],label=record['record_id'][-3:]+' · '+F.assembly_label(record),
        fs=fs,duration=float(t[-1]),traces=traces,full_samples=pack_full_samples(fields),
        ecg_variants=variants,psd_frequency_hz=freq.tolist(),detail_t=t[first:last].tolist(),detail_indices=[first,last],
        channel_assignment=dict(physical_channel_roles=roles),lateral_channel=record['physical_channel_roles']['lateral'],
        active_leads=active,lead_notes=notes,breathing=breathing(annotation,record),montage=montage(record,contacts))


def reader_script():
    """Use Georg's plotting/visibility/filter engine, adapting only subject data."""
    script=(FOLDER/'integrated_experiment_view.js').read_text(encoding='utf8')
    adapter=(FOLDER/'session_experiment_view.js').read_text(encoding='utf8')
    geometry_js,text_js=adapter.split('// SUBJECT_TEXT\n',1)
    replacements=[('let ri=3,history=0','let ri=0,history=0'),
        ("range:[-430,430]","range:D.scene_ranges[0]"),
        ("range:[-180,180]","range:D.scene_ranges[1]"),
        ("range:[-380,0]","range:D.scene_ranges[2]"),
        ("'Поверхность КТ и цилиндры рук'","'Поверхность Inobitec и руки'"),
        ("unknown:'#667085'","inactive:'#94a3b8',unknown:'#667085'"),
        ("fillcolor:phaseColors[i]","fillcolor:r.breathing.phases.length===1?'#d4e5ff':phaseColors[i]"),
        ("+(D.physical_lead_labels?'':' · канал '+ch);","+(D.physical_lead_labels?'':' · канал '+ch)+(r.lead_notes?.[physicalRole(ch)]||'');"),
        ("line:{color:leadColors[physicalRole(ch)],width:","line:{color:r.active_leads&&!r.active_leads.includes(physicalRole(ch))?'#94a3b8':leadColors[physicalRole(ch)],width:")]
    for old,new in replacements:
        if script.count(old)!=1:raise ValueError('Georg reader adapter marker changed: '+old)
        script=script.replace(old,new,1)
    for pattern,new in [
        (r'const lateralChannel=\(\)=>\{.*?\n\};',"const lateralChannel=()=>D.records[ri].lateral_channel;"),
        (r'function physicalRole\(ch\)\{.*?\n\}',"function physicalRole(ch){return D.records[ri].channel_assignment.physical_channel_roles[String(ch)];}"),
        (r'function geometry\(\)\{.*?\n\}',geometry_js.strip()),
        (r'function text\(\)\{.*?\n\}',text_js.strip())]:
        script,count=re.subn(pattern,lambda m:new,script,count=1,flags=re.S)
        if count!=1:raise ValueError('Georg reader function changed: '+pattern)
    return script


def widget(subject):
    payload,contacts,audit=geometry(subject);S,_,_=F.results()
    annotations=read(F.EXPERIMENT/'annotations/session_1140/candidate_annotations.local.json')
    by_id={a['record_id']:a for a in annotations['records']}
    records=[r for r in S['records'] if r['subject']==subject]
    output=R.model_root(subject)/'reader_figures/integrated_view';output.mkdir(parents=True,exist_ok=True)
    # Cache only display products, keyed by the frozen inputs and their producer.
    key=hashlib.sha256(json.dumps(dict(subject=subject,records=[r['arrays_sha256'] for r in records],
        annotations_sha256=sha(F.EXPERIMENT/'annotations/session_1140/candidate_annotations.local.json'),
        contacts_sha256=audit['contacts_sha256'],producer=sha(__file__),
        labels_producer=sha(F.__file__),filter_source_sha256=S['filter_source_sha256'],
        filter_helper=sha(FOLDER/'experiment_signal_display.py')),sort_keys=True).encode()).hexdigest()
    cache=output/'records.local.json';key_path=output/'records_key.local.json'
    if cache.exists() and key_path.exists() and read(key_path)['key']==key:
        display_records=read(cache)
    else:
        display_records=[record_payload(r,by_id[r['record_id']],contacts) for r in records]
        cache.write_text(json.dumps(display_records,separators=(',',':'),allow_nan=False),encoding='utf8')
        key_path.write_text(json.dumps(dict(key=key)),encoding='utf8')
    payload.update(records=display_records,physical_lead_labels=True)
    payload_text=json.dumps(payload,separators=(',',':'),allow_nan=False)
    payload_path=output/'payload.local.json';payload_path.write_text(payload_text,encoding='utf8')
    audit.update(producer_sha256=sha(__file__),shared_reader_sha256=sha(FOLDER/'integrated_experiment_view.js'),
        display_payload_sha256=sha(payload_path),
        source_summary_sha256=sha(F.OUT/'summary.local.json'),source_arrays=[r['arrays_sha256'] for r in records],
        shared_georg_reader=True,iframe=False,external_geometry_dependency=False)
    (output/'manifest.local.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    ui=(FOLDER/'session_experiment_view.html').read_text(encoding='utf8')
    return ui+'<script>'+get_plotlyjs()+'</script><script>const experimentPayload='+payload_text.replace('</','<\\/')+';'+reader_script()+'</script>'
