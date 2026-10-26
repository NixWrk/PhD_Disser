"""Subject-local reader figures; no meshing, FEM launch, fitting or acceptance."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import session_report_figures as F
from audit_scorpion_session import read,sha

RUN=F.EXPERIMENT/'derived/exploratory/manual_stl_fem_20261026'


def model_root(subject):return RUN/(subject+'_v3')


def model_state(subject):
    root=model_root(subject);rows=[]
    for stage,label in [('surface','Наружная поверхность и руки'),('mesh','Объёмная сетка'),
        ('materials','Тканевые доли 1024/4096'),('cem_contacts_input','Вход конечных контактов')]:
        done=root/stage/'preparation.local.json';failed=root/stage/'preparation_failed.local.json'
        status='Подготовлено' if done.exists() else 'Отказ' if failed.exists() else 'Выполняется' if (root/(stage+'_started.local.json')).exists() else 'Ожидает входов'
        reason=read(failed)['error'] if failed.exists() else ''
        rows.append({'Этап':label,'Состояние':status,'Причина отказа':reason})
    for name,label,done,failed in [('cem','Конечные CEM-площадки','cem_contacts_manifest.local.json','cem_contacts_failed.local.json'),
        ('electrical','Прямой FEM','forward.local.json','failed.local.json')]:
        path=root/name;status='Выполнено' if (path/done).exists() else 'Отказ' if (path/failed).exists() else 'Выполняется' if path.exists() else 'Ожидает входов'
        if name=='electrical' and (path/'resource_wait.local.json').exists() and not (path/'resource_acquired.local.json').exists():status='Ожидает ресурсов'
        reason=read(path/failed).get('message','') if (path/failed).exists() else ''
        rows.append({'Этап':label,'Состояние':status,'Причина отказа':reason})
    return pd.DataFrame(rows)


def geometry(subject):
    root=model_root(subject)
    cfg=read(root/'configuration.local.json');plan=read(root/'placement_plan.local.json')
    q=read(root/'surface/surface_qc.local.json')
    if sha(cfg['body']['source_stl'])!=cfg['body']['source_stl_sha256']:raise ValueError('Own manual STL changed')
    return cfg,plan,q


def geometry_image(subject):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    root=model_root(subject);cfg,plan,q=geometry(subject)
    mesh=trimesh.load(root/'surface/exterior.stl',force='mesh')
    display_mesh=mesh.simplify_quadric_decimation(face_count=24000)
    v=display_mesh.vertices;f=display_mesh.faces
    fig=plt.figure(figsize=(13,7),layout='constrained');ax=fig.add_subplot(projection='3d')
    ax.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=f,color='#cfc8bf',alpha=.4,linewidth=0,shade=True)
    for name,value in cfg['contacts']['physical_centres_lps_mm'].items():
        x,y,z=value['raw_lps_mm'];current=('_I_' in name) or name.endswith(('_A','_B'))
        ax.scatter(x,y,z,s=17,c='#dc2626' if current else '#1d4ed8',depthshade=False)
    ax.set(xlabel='X LPS, мм',ylabel='Y LPS, мм',zlabel='Z LPS, мм',
        title=F.SUBJECT_NAMES[subject]+': собственная поверхность Inobitec и руки, обхват 370 мм')
    ax.set_box_aspect(np.ptp(mesh.bounds,axis=0));ax.view_init(elev=13,azim=-116)
    folder=root/'reader_figures';folder.mkdir(exist_ok=True)
    path=folder/'manual_body_and_contacts.png';fig.savefig(path,dpi=140);plt.close(fig)
    return path


def geometry_widget(subject):
    from IPython.display import HTML,display
    from subject_geometry_3d import widget
    display(HTML(widget(subject)))


def literal_tissues(subject):
    root=model_root(subject);ref=read(root/'reference_materials.local.json')
    labels=['Мягкотканный фон: мышца','Лёгкое','Кость: кортикальная','Целое сердце: миокардиальный сценарий','Жир']
    return pd.DataFrame({'Компартмент':labels,'Вдох, Ом·м':ref['inhale_rho_ohm_m'],'Выдох, Ом·м':ref['exhale_rho_ohm_m']})


def ct_profile(subject):
    p=read(model_root(subject)/'ct_profile.local.json')
    for ref in p['sources']:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Manual profile source changed')
    arc=np.array(p['arc_offset_mm']);select=abs(arc)<=70
    fig=go.Figure(go.Scatter(x=arc[select],y=np.asarray(p['skin_to_first_lung_mm'],dtype=float)[select],
        mode='lines',name='Кожа → первое правое лёгкое',line=dict(color='#d97706')))
    for offset,label in [(-70,'A'),(-35,'M'),(35,'N'),(70,'B')]:fig.add_vline(x=offset,line_dash='dot',annotation_text=label)
    fig.update_xaxes(title_text='Расстояние от центра по кожной дуге, мм')
    fig.update_yaxes(title_text='Длина луча по LPS +X, мм',rangemode='tozero')
    return F.style(fig,450)


def forward(subject):
    root=model_root(subject);p=root/'electrical/forward.local.json'
    if not p.exists():return None
    value=read(p)
    if value['subject_alias']!=subject:raise ValueError('Wrong subject FEM')
    if value['identity']['reference_sha256']!=sha(root/'reference_materials.local.json'):
        raise ValueError('FEM reference scenario changed')
    return value


def forward_figure(subject):
    value=forward(subject)
    if value is None:raise ValueError('No completed FEM result')
    fig=make_subplots(rows=2,cols=2,subplot_titles=['Боковая · вдох','Боковая · выдох','ТТРКГ · вдох','ТТРКГ · выдох'])
    for row,role in enumerate(('lateral','ttrkg'),1):
        for col,state in enumerate(('inhale','exhale'),1):
            rows=[v for v in value['comparison'] if v['role']==role and v['state']==state]
            rows=sorted(rows,key=lambda v:(int(v['case_id'].split('_')[0][1:]),v['case_id']))
            labels=[v['case_id'].replace('TT','ТТРКГ ').replace('x','×').replace('L','').replace('_',' мм · ') for v in rows]
            for key,label,color in [('observed_Z_ohm','Scorpion','#2563eb'),('fine_Z_ohm','КТ/FEM · литературный сценарий','#d97706')]:
                fig.add_trace(go.Scatter(x=labels,y=[v[key] for v in rows],name=label,legendgroup=key,
                    showlegend=row==col==1,mode='lines+markers',line=dict(color=color)),row=row,col=col)
            fig.update_yaxes(title_text='Импеданс, Ом',row=row,col=col)
    return F.style(fig,780)


def forward_table(subject):
    value=forward(subject)
    def label(case):
        size,tt=case.split('_')
        return 'Боковая '+size[1:]+' мм + ТТРКГ '+tt[2:].replace('x','×')
    return pd.DataFrame([{'Сборки':label(v['case_id']),'Отведение':F.ROLE[v['role']],'Дыхание':F.STATE[v['state']],
        'Измерено, Ом':v['observed_Z_ohm'],'FEM 4096, Ом':v['fine_Z_ohm'],'Невязка, Ом':v['residual_ohm'],
        'Различие 1024/4096, Ом':v['difference_ohm'],'Допуск интегрирования, Ом':v['integration_tolerance_ohm'],
        'Допуск выдержан':v['integration_passed']} for v in value['comparison']])
