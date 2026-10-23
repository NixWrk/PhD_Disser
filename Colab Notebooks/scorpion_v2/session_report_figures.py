"""Reader figures and collapsed tables for the independent 22 October report."""
from functools import lru_cache
from pathlib import Path
import html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display

from audit_scorpion_session import read, sha
from session_20261022 import EXPERIMENT, ROOT, SUBJECT_NAMES, STATUS

OUT=EXPERIMENT/'derived/exploratory/session_3340'
STATE={'inhale':'Глубокий вдох','exhale':'Глубокий выдох'}
ROLE={'ttrkg':'ТТРКГ','lateral':'Боковая сборка'}
COLORS={60:'#c2410c',100:'#15803d',140:'#1d4ed8'}


@lru_cache(maxsize=1)
def results():
    s=read(OUT/'summary.local.json'); m=read(OUT/'two_layer.local.json'); e=read(OUT/'equipment.local.json')
    for name,key in [('session_20261022.py',s['producer_sha256']),
                     ('scorpion_qc.py',s['filter_source_sha256']),
                     ('lateral_pilot.py',s['ensemble_source_sha256']),
                     ('session_two_layer.py',m['producer_sha256']),
                     ('session_equipment.py',e['producer_sha256'])]:
        if sha(Path(__file__).with_name(name))!=key:
            raise ValueError('Changed producer: '+name)
    if m['summary_sha256']!=sha(OUT/'summary.local.json'):
        raise ValueError('Layer fit uses another signal summary')
    if m['registry_sha256']!=sha(ROOT/'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'):
        raise ValueError('Tissue literature registry changed')
    if read(EXPERIMENT/'metadata/inventory.local.json')['frozen_record_identity_sha256']!=sha(EXPERIMENT/'record_identity.json'):
        raise ValueError('Frozen record identities changed')
    for p,digest in [(EXPERIMENT/'metadata/inventory.local.json',s['inventory_sha256']),
        (EXPERIMENT/'derived/scorpion_bin_review/txt_binary_matches.local.json',s['matches_sha256']),
        (EXPERIMENT/'annotations/session_1140/candidate_annotations.local.json',s['annotations_sha256']),
        (EXPERIMENT/'metadata/author_confirmations_20261023.local.json',s['author_context_sha256'])]:
        if sha(p)!=digest:raise ValueError('Candidate source changed: '+p.name)
    config_path=Path(os.environ.get('KALMYKOV_EXP02_CONFIG',ROOT/'Colab Notebooks/config/exp02_paths.local.json'))
    config=read(config_path);derived=Path(config['derived_root'])/'exp02'
    for p,digest in [(config_path,e['config_sha256']),
        (derived/'qc/10.01_record_manifest.candidate.json',e['manifest_sha256']),
        (derived/'qc/12.01_signal_qc.candidate.json',e['qc_sha256'])]:
        if sha(p)!=digest:raise ValueError('Equipment input changed: '+p.name)
    for p in e['provenance']:
        if sha(p['raw_path'])!=p['sha256']:raise ValueError('Equipment raw changed')
        for kind,key in [('breathing','breathing_sha256'),('ecg','ecg_sha256')]:
            if sha(derived/'annotations'/kind/(p['sha256'][:16]+'.json'))!=p[key]:
                raise ValueError('Equipment annotation changed')
    for r in s['records']:
        if sha(EXPERIMENT/r['source_relative_path'])!=r['source_sha256']:
            raise ValueError('Raw source changed: '+r['record_id'])
        if sha(OUT/(r['record_id']+'.local.npz'))!=r['arrays_sha256']:
            raise ValueError('Signal cache changed: '+r['record_id'])
    return s,m,e


@lru_cache(maxsize=15)
def arrays(rid):
    with np.load(OUT/(rid+'.local.npz')) as a:
        return {k:a[k] for k in a.files}


def show_table(frame,caption):
    table=frame if isinstance(frame,pd.DataFrame) else pd.DataFrame(frame)
    display(HTML('<details><summary>'+html.escape(caption)+'</summary>'+
                 table.to_html(index=False,escape=True,na_rep='—',float_format=lambda v:f'{v:.5g}')+'</details>'))


def show(fig):
    display(HTML('<div class="scientific-figure" tabindex="0" role="region" '
        'aria-label="График с горизонтальной прокруткой на узком экране" '
        'style="overflow-x:auto;max-width:100%">'+
        fig.to_html(full_html=False,include_plotlyjs=False,
                    config={'responsive':True,'displaylogo':False})+'</div>'))


def style(fig,height=520):
    fig.update_layout(template='plotly_white',height=height,font=dict(size=14),
        margin=dict(l=65,r=28,t=80,b=100),hovermode='x unified',
        legend=dict(orientation='h',y=-.17,x=0),uirevision='scorpion-session-20261022')
    return fig


def assembly_label(r):
    c=r['context'];variant=(c['ttrkg_variant'] or '?').replace('x','×')
    kind=c['record_kind']
    if kind=='ttrkg_only':return f'Только ТТРКГ {variant}'
    if kind=='lateral_only':return f'Только боковая {c["lateral_size_mm"]} мм'
    if kind=='switching':return 'Попеременное отключение: 140 мм и ТТРКГ 2×2'
    suffix=' · переставлены пары ТТРКГ' if kind=='reciprocity' else ''
    return f'Боковая {c["lateral_size_mm"]} мм + ТТРКГ {variant}'+suffix


def envelope_indices(v,budget=2400):
    """Ordered min/max per bucket for viewing only; no analysis uses these."""
    if len(v)<=budget:return np.arange(len(v))
    step=int(np.ceil(len(v)/(budget//2)))
    indices=[0,len(v)-1]
    for start in range(0,len(v),step):
        local=v[start:start+step]
        indices.extend([start+int(np.argmin(local)),start+int(np.argmax(local))])
    return np.unique(indices)


def overview(subject):
    records=[r for r in results()[0]['records'] if r['subject']==subject]
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.07,
        subplot_titles=['ТТРКГ: общий импеданс','Боковая сборка: общий импеданс',
                        'ЭКГ выбранной подключённой сборки','Пульсовая полоса подключённых сборок'])
    blocks=[]
    for k,r in enumerate(records):
        a=arrays(r['record_id']);begin=len(fig.data);switch=r['context']['record_kind']=='switching'
        for role,row,color in [('ttrkg',1,'#7c3aed'),('lateral',2,'#0369a1')]:
            c=r['physical_channel_roles'][role];v=a[f'raw_{c}'];ii=envelope_indices(v)
            active=role in r['active_roles']
            label=ROLE[role]+(' · отключена, технические числа' if not active and not switch else '')
            fig.add_trace(go.Scatter(x=ii/r['fs'],y=v[ii],name=label,
                line=dict(color=color if active or switch else '#94a3b8',width=1),visible=k==0),row=row,col=1)
            if active and not switch:
                ii=np.arange(0,r['n'],50)
                fig.add_trace(go.Scatter(x=ii/r['fs'],y=a[f'slow_{c}'][ii],name=ROLE[role]+' · медленно',
                    line=dict(color='#f59e0b',width=2),visible=k==0),row=row,col=1)
                v=a[f'pulse_{c}']*1000;ii=envelope_indices(v)
                fig.add_trace(go.Scatter(x=ii/r['fs'],y=v[ii],name=ROLE[role]+' · 0,7–12 Гц',
                    line=dict(color=color,width=1),visible=k==0),row=4,col=1)
        c=r['ecg_reference'];v=a[f'ecg_raw_{c}'] if switch else a[f'ecg_{c}'];ii=envelope_indices(v)
        fig.add_trace(go.Scatter(x=ii/r['fs'],y=v[ii],name='ЭКГ · технический обзор' if switch else 'ЭКГ 0,5–40 Гц',
            line=dict(color='#475569',width=.8),visible=k==0),row=3,col=1)
        for state,(l,h) in r['windows'].items():
            for row in (1,2):
                # Separate traces let each record keep its own reconstructed windows.
                c=r['physical_channel_roles']['ttrkg' if row==1 else 'lateral']
                v=a[f'raw_{c}'];height=float(np.quantile(v,.99));low=float(np.quantile(v,.01))
                fig.add_trace(go.Scatter(x=[l/r['fs'],h/r['fs'],h/r['fs'],l/r['fs'],l/r['fs']],
                    y=[low,low,height,height,low],fill='toself',fillcolor='rgba(34,197,94,.1)' if state=='inhale' else 'rgba(59,130,246,.1)',
                    line=dict(width=0),name=STATE[state]+' · внутреннее окно',showlegend=row==1,
                    hoverinfo='name',visible=k==0),row=row,col=1)
        blocks.append((begin,len(fig.data)))
    buttons=[]
    for r,(a,b) in zip(records,blocks):
        visible=[a<=i<b for i in range(len(fig.data))]
        buttons.append(dict(label=assembly_label(r),method='update',args=[dict(visible=visible),
                            dict(title=SUBJECT_NAMES[subject]+' · '+assembly_label(r))]))
    fig.update_layout(updatemenus=[dict(buttons=buttons,x=0,y=1.12,xanchor='left')],
                      title=SUBJECT_NAMES[subject]+' · '+assembly_label(records[0]))
    for row in (1,2):fig.update_yaxes(title_text='Ом',row=row,col=1)
    fig.update_yaxes(title_text='мВ',row=3,col=1);fig.update_yaxes(title_text='мОм',row=4,col=1)
    fig.update_xaxes(title_text='Время от начала записи, с',row=4,col=1)
    style(fig,1080)
    fig.update_layout(margin=dict(l=65,r=28,t=160,b=180),title=dict(y=.99))
    return fig


def levels(subject,role):
    data=[r for r in results()[0]['levels'] if r['subject']==subject and r['kind']=='both' and r['role']==role]
    fig=make_subplots(rows=1,cols=2,subplot_titles=list(STATE.values()))
    for col,state in enumerate(STATE,1):
        for variant,color in [('2x2','#1d4ed8'),('1x1','#c2410c')]:
            rows=sorted([r for r in data if r['state']==state and r['variant']==variant],key=lambda r:r['size_mm'])
            fig.add_trace(go.Scatter(x=[r['size_mm'] for r in rows],y=[r['baseline_ohm'] for r in rows],
                mode='lines+markers',name='ТТРКГ '+variant.replace('x','×'),showlegend=col==1,
                line=dict(color=color),marker=dict(size=9),
                text=[f"{r['start_s']:.1f}–{r['end_s']:.1f} с" for r in rows],
                hovertemplate='%{x} мм: %{y:.3f} Ом<br>Окно %{text}<extra>%{fullData.name}</extra>'),row=1,col=col)
        fig.update_xaxes(title_text='Сопутствующая боковая сборка, мм' if role=='ttrkg' else 'Размер боковой сборки, мм',
                         tickvals=[60,100,140],row=1,col=col)
        fig.update_yaxes(title_text=ROLE[role]+', Ом',row=1,col=col)
    return style(fig)


def cycle_plot(subject,role,variants=False):
    records=[r for r in results()[0]['records'] if r['subject']==subject and r['context']['record_kind']=='both'
        and (r['context']['lateral_size_mm']==60 if variants else r['context']['ttrkg_variant']=='2x2')]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.15,
                      subplot_titles=[ROLE[role]+' · четыре кандидатных цикла на вдохе','ЭКГ и границы циклов'])
    for k,r in enumerate(records):
        selection=r['four_cycles']
        if selection is None:continue
        a=arrays(r['record_id']);p=np.array(selection['events_samples']);ii=np.arange(p[0],p[-1]+1)
        c=r['physical_channel_roles'][role];ec=r['ecg_reference'];color=(['#1d4ed8','#c2410c'][k] if variants else COLORS[r['context']['lateral_size_mm']])
        label='ТТРКГ '+r['context']['ttrkg_variant'].replace('x','×') if variants else str(r['context']['lateral_size_mm'])+' мм'
        t=(ii-p[0])/r['fs']
        fig.add_trace(go.Scatter(x=t,y=a[f'pulse_{c}'][ii]*1000,name=label,line=dict(color=color,width=1.8)),row=1,col=1)
        fig.add_trace(go.Scatter(x=t,y=a[f'ecg_{ec}'][ii],name=label+' · ЭКГ',showlegend=False,line=dict(color=color,width=1.2)),row=2,col=1)
        fig.add_trace(go.Scatter(x=(p-p[0])/r['fs'],y=a[f'ecg_{ec}'][p],mode='markers',showlegend=False,
                                marker=dict(color=color,size=8),name=label+' · кандидаты QRS'),row=2,col=1)
    fig.update_yaxes(title_text='мОм',row=1,col=1);fig.update_yaxes(title_text='мВ',row=2,col=1)
    fig.update_xaxes(title_text='Время от первого выбранного QRS, с',row=2,col=1)
    return style(fig,650)


def ensemble_plot(subject):
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Боковые сборки при ТТРКГ 2×2','ТТРКГ 1×1 и 2×2 при боковой 60 мм'])
    for r in results()[0]['records']:
        if r['subject']!=subject or r['context']['record_kind']!='both':continue
        c=r['context'];en=r['ensembles']['inhale']
        if c['ttrkg_variant']=='2x2':
            v=en['lateral'];fig.add_trace(go.Scatter(x=v['phase_rr'],y=np.array(v['median_ohm'])*1000,
                name=str(c['lateral_size_mm'])+' мм',line=dict(color=COLORS[c['lateral_size_mm']])),row=1,col=1)
        if c['lateral_size_mm']==60:
            v=en['ttrkg'];fig.add_trace(go.Scatter(x=v['phase_rr'],y=np.array(v['median_ohm'])*1000,
                name='ТТРКГ '+c['ttrkg_variant'].replace('x','×')),row=1,col=2)
    fig.update_xaxes(title_text='Доля интервала QRS–QRS');fig.update_yaxes(title_text='мОм')
    return style(fig)


def two_layer_profile(subject):
    fig=make_subplots(rows=1,cols=3,subplot_titles=['Остаток по трём размерам','Сопротивление наружного слоя','Сопротивление внутреннего слоя'])
    for state,color in [('inhale','#c2410c'),('exhale','#1d4ed8')]:
        rows=[r for r in results()[1]['profiles'] if r['subject']==subject and r['state']==state]
        for col,key,index in [(1,'rmse_ohm',None),(2,'rho_ohm_m',0),(3,'rho_ohm_m',1)]:
            fig.add_trace(go.Scatter(x=[r['h_mm'] for r in rows],
                y=[r[key] if index is None else r[key][index] for r in rows],
                name=STATE[state],showlegend=col==1,line=dict(color=color),mode='lines+markers'),row=1,col=col)
        fig.update_yaxes(title_text='RMSE, Ом',row=1,col=1)
        for col in (2,3):fig.update_yaxes(title_text='Ом·м',type='log',row=1,col=col)
    fig.update_xaxes(title_text='Фиксированная h в каждом подборе, мм')
    return style(fig,530)


def pulse_prediction(subject,h=120):
    r=next(p for p in results()[1]['profiles'] if p['subject']==subject and p['state']=='inhale' and p['h_mm']==h)
    fig=make_subplots(rows=1,cols=3,subplot_titles=[f'{s} мм · оставлен для проверки' for s in r['sizes_mm']])
    if r['pulse'] is None:raise ValueError('Missing independent pulse cycles')
    p=r['pulse']
    for col,(size,test,obs) in enumerate(zip(r['sizes_mm'],p['left_out'],p['observed_mohm']),1):
        fig.add_trace(go.Scatter(x=p['phase_rr'],y=obs,name='Медианный кандидатный цикл',showlegend=col==1,
                                line=dict(color='#1d4ed8')),row=1,col=col)
        fig.add_trace(go.Scatter(x=p['phase_rr'],y=test['predicted_mohm'],name='Прогноз по двум другим размерам',showlegend=col==1,
                                line=dict(color='#c2410c',dash='dash')),row=1,col=col)
    fig.update_xaxes(title_text='Доля интервала QRS–QRS');fig.update_yaxes(title_text='мОм')
    return style(fig,530)


def equipment_levels():
    s,m,e=results();fig=make_subplots(rows=2,cols=2,subplot_titles=['Боковая · вдох','Боковая · выдох','ТТРКГ · вдох','ТТРКГ · выдох'])
    for row,role in enumerate(['lateral','ttrkg'],1):
        for col,state in enumerate(STATE,1):
            for device,color in [('mgtu','#c2410c'),('scorpion','#1d4ed8')]:
                if device=='mgtu':v=[v for v in e['levels'] if v['role']==role and v['state']==state and v['usable']]
                else:v=[v for v in s['levels'] if v['subject']=='nik' and v['role']==role and v['state']==state and v['kind']=='both' and v['variant']=='2x2' and v['size_mm'] in(60,140)]
                v=sorted(v,key=lambda v:v['size_mm'])
                fig.add_trace(go.Scatter(x=[v['size_mm'] for v in v],y=[v['baseline_ohm'] for v in v],
                    name='МГТУ · 29.04.2026' if device=='mgtu' else 'Scorpion · 22.10.2026',
                    showlegend=row==col==1,line=dict(color=color),mode='lines+markers'),row=row,col=col)
            fig.update_xaxes(title_text='Боковая сборка, мм',tickvals=[60,140],row=row,col=col)
            fig.update_yaxes(title_text='Ом',row=row,col=col)
    return style(fig,720)


def equipment_pulse(role):
    s,m,e=results();fig=make_subplots(rows=2,cols=2,shared_xaxes=True,vertical_spacing=.16,
        subplot_titles=['60 мм · '+ROLE[role],'140 мм · '+ROLE[role],'ЭКГ · 60 мм','ЭКГ · 140 мм'])
    for col,size in enumerate([60,140],1):
        legacy=next(v for v in e['segments'] if v['role']==role and v['size_mm']==size)
        r=next(r for r in s['records'] if r['subject']=='nik' and r['context']['record_kind']=='both' and
               r['context']['lateral_size_mm']==size and r['context']['ttrkg_variant']=='2x2')
        a=arrays(r['record_id']);p=np.array(r['four_cycles']['events_samples']);ii=np.arange(p[0],p[-1]+1)
        new=dict(time_s=(ii-p[0])/r['fs'],pulse_mohm=a[f'pulse_{r["physical_channel_roles"][role]}'][ii]*1000,
                 ecg_mV=a[f'ecg_{r["ecg_reference"]}'][ii])
        for v,label,color in [(legacy,'МГТУ','#c2410c'),(new,'Scorpion','#1d4ed8')]:
            fig.add_trace(go.Scatter(x=v['time_s'],y=v['pulse_mohm'],name=label,showlegend=col==1,line=dict(color=color)),row=1,col=col)
            fig.add_trace(go.Scatter(x=v['time_s'],y=v['ecg_mV'],name=label+' · ЭКГ',showlegend=False,line=dict(color=color)),row=2,col=col)
        fig.update_yaxes(title_text='мОм',row=1,col=col);fig.update_yaxes(title_text='мВ',row=2,col=col)
        fig.update_xaxes(title_text='Время от первого QRS, с',row=2,col=col)
    return style(fig,680)


def reciprocity():
    rec=[r for r in results()[0]['records'] if r['record_id'] in ['SCV2-N-007','SCV2-N-008']]
    fig=make_subplots(rows=1,cols=2,subplot_titles=['Общий импеданс ТТРКГ','Медианный кандидатный цикл ТТРКГ на вдохе'])
    for r,label,color in zip(rec,['Обычное включение','Пары ТТРКГ переставлены'],['#1d4ed8','#c2410c']):
        v=[v for v in results()[0]['levels'] if v['record_id']==r['record_id'] and v['role']=='ttrkg']
        fig.add_trace(go.Scatter(x=[STATE[v['state']] for v in v],y=[v['baseline_ohm'] for v in v],
            name=label,line=dict(color=color),mode='lines+markers'),row=1,col=1)
        en=r['ensembles']['inhale']['ttrkg']
        fig.add_trace(go.Scatter(x=en['phase_rr'],y=np.array(en['median_ohm'])*1000,
            name=label,showlegend=False,line=dict(color=color)),row=1,col=2)
    fig.update_yaxes(title_text='Ом',row=1,col=1);fig.update_yaxes(title_text='мОм',row=1,col=2)
    fig.update_xaxes(title_text='Доля интервала QRS–QRS',row=1,col=2)
    return style(fig)
