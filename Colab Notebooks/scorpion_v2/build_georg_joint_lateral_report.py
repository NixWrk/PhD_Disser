"""Read saved inputs and numerical audits of the joint Georg lateral problem."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh

from build_georg_fem_reports import ROOT,new_cells,export,read,sha,table,number,write

sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
from georg_model_contract import resolve
OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_lateral_inverse_20261011'
STEM='33.23_Совместная_локализация_боковых_сборок_Георгия_Scorpion_v2'
STATES={'inhale':'Задержка на вдохе','exhale':'Задержка на выдохе','quiet_breathing':'Спокойное дыхание, тест'}
SOURCE_AUDIT=OUT/'source_audit_all_h6_v1/source_audit.local.json'
NIX_AUDIT=OUT/'nix_level_audit_v7/nix_level.local.json'
SEARCH_MODEL=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_ct_model_20261011/lateral_search_h6_q4096_v2'
JOINT_RUN=OUT/'joint_own_drive_nix_search_resume_v1'
AUTHOR_CENTRE=OUT/'author_centre_20261012.local.json'


def verify():
    model,current=resolve();inputs=read(OUT/'inputs.local.json');grid=read(OUT/'pose_grid.local.json')
    for source in inputs['sources']:
        if sha(source['path'])!=source['sha256']:raise ValueError('Inverse source changed')
    for name,digest in inputs['outputs_sha256'].items():
        if sha(OUT/name)!=digest:raise ValueError('Inverse observations changed')
    if grid['surface_sha256']!=sha(model/'body_ct_surface.stl'):raise ValueError('Search surface changed')
    audit=read(OUT/'homogeneous_mesh_audit.local.json')
    if not audit['complete']:raise ValueError('Homogeneous audit incomplete')
    return model,current,inputs,grid,audit


def observations_figure():
    frame=pd.read_csv(OUT/'observations.local.csv');fig=go.Figure()
    colors={'longitudinal':'#2673b8','perpendicular':'#dc7c20'}
    for (_,ch),group in frame.groupby(['record_id','channel'],sort=True):
        group=group.copy();group['state']=group.state.map(STATES)
        fig.add_trace(go.Scatter(x=group.state,y=group.Z_ohm,mode='lines+markers',
            name=group.display_name.iloc[0],line=dict(color=colors[group.axis.iloc[0]]),
            customdata=np.c_[group.slow_p05_ohm,group.slow_p95_ohm,group.start_s,group.end_s],
            hovertemplate='%{y:.5f} Ом<br>5–95% медленного сигнала: %{customdata[0]:.3f}–%{customdata[1]:.3f} Ом<br>%{customdata[2]:.2f}–%{customdata[3]:.2f} с<extra>%{fullData.name}</extra>'))
    fig.update_layout(template='plotly_white',height=680,title='Боковые отведения: уровни по дыхательным состояниям',
        yaxis_title='Медиана BEI, Ом',legend=dict(orientation='h',y=-.20),margin=dict(b=180),
        xaxis=dict(categoryorder='array',categoryarray=list(STATES.values())))
    return fig


def cross_figure():
    frame=pd.read_csv(OUT/'cross_changes.local.csv');fig=go.Figure()
    labels={'004':'140 мм · продольная','005':'140 мм · поперечная','010':'60 мм · продольная'}
    for state,color in [('inhale','#2673b8'),('exhale','#dc7c20')]:
        g=frame[frame.state==state]
        fig.add_trace(go.Bar(x=g.previous_record.map(lambda x:labels[f'{int(x):03d}']),
            y=g.difference_ohm,name=STATES[state],marker_color=color))
    fig.update_layout(template='plotly_white',height=450,title='Изменение уровня после перехода к кресту',
        yaxis_title='ΔBEI, Ом',barmode='group')
    return fig


def pose_figure():
    model,_,_,grid,_=verify()
    surface=trimesh.load(model/'body_ct_surface.stl',force='mesh')
    surface=surface.simplify_quadric_decimation(face_count=20000)
    v=surface.vertices;f=surface.faces;centres=np.array([p['centre_lps_mm'] for p in grid['poses']])
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        color='#c8c2b9',opacity=.17,name='Кожа КТ',hoverinfo='skip',showlegend=True))
    fig.add_trace(go.Scatter3d(x=centres[:,0],y=centres[:,1],z=centres[:,2],mode='markers',
        marker=dict(size=3,color='#2673b8'),name='Предварительные центры',
        hovertemplate='x=%{x:.1f}; y=%{y:.1f}; z=%{z:.1f} мм<extra>Кандидат, без подгонки</extra>'))
    p=read(AUTHOR_CENTRE)['centre_lps_mm']
    fig.add_trace(go.Scatter3d(x=[p[0]],y=[p[1]],z=[p[2]],mode='markers',
        name='Центр по воспоминанию автора',marker=dict(size=7,color='#b44841',symbol='diamond'),
        hovertemplate='x=%{x:.3f}; y=%{y:.3f}; z=%{z:.3f} мм<extra>Ориентировочная реконструкция, не измерение</extra>'))
    fig.update_layout(template='plotly_white',height=700,showlegend=True,title='Область предварительного поиска на правой поверхности КТ',
        scene=dict(aspectmode='data',uirevision='georg-joint-lateral-fixed',
            camera=dict(eye=dict(x=-1.05,y=.35,z=.15),up=dict(x=0,y=0,z=1)),
            xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм'))
    return fig


def numerical_table():
    model,_,_,_,_=verify();rows=[]
    for h in [8,6,4]:
        for case in ['reference_60_lateral','reference_140_lateral']:
            r=read(model/f'fem_h{h}/scenarios/{case}.local.json')
            rows.append(dict(Сетка=f'Исходная, номинал {h} мм',Размер=int(r['montage']),
                Импеданс=r['Z_ohm'],Интегрирование=r['quadrature_difference_ohm']))
    folders={3:'lateral_nested_h3_r20_v2',2:'lateral_nested_h2_r20',1.5:'lateral_nested_h1p5_r20',1.25:'lateral_nested_h1p25_r20'}
    for h,name in folders.items():
        folder=model/name
        if not (folder/'forward_results.local.json').is_file():continue
        plan=read(folder/'plan.local.json')
        if plan['code_sha256']!=sha(ROOT/'MATLAB_TRKG4_real_subjects/tools/georg_lateral_refinement.py'):
            raise ValueError('Executed nested preparation source changed')
        for source in plan['sources']:
            if sha(source['path'])!=source['sha256']:raise ValueError('Nested source changed')
        for r in read(folder/'forward_results.local.json'):
            if r['identity']['plan_sha256']!=sha(folder/'plan.local.json'):raise ValueError('Nested FEM result changed')
            rows.append(dict(Сетка=f'Локально {number(h)} мм, радиус 20 мм',Размер=int(r['montage']),
                Импеданс=r['Z_ohm'],Интегрирование=r['quadrature_difference_ohm']))
    return pd.DataFrame(rows).rename(columns={'Размер':'L, мм','Импеданс':'Z, Ом','Интегрирование':'Разность интегрирования, Ом'})


def source_rows():
    audit=read(SOURCE_AUDIT)
    if not audit['all_contacts_passed']:raise ValueError('Source contacts unresolved')
    rows=[]
    for r in audit['cases']:
        ident=r['identity']
        if ident['worker_sha256']!=sha(ROOT/'MATLAB_TRKG4_real_subjects/src/run_georg_lateral_source_audit.m'):
            raise ValueError('Executed source audit changed')
        T=np.array(r['port_transfer_ohm'])
        axes=['longitudinal','perpendicular'] if r['is_cross'] else [
            'perpendicular' if r['record_id'].endswith('005') else 'longitudinal']
        for axis in axes:
            i,j,c=(1,0,2) if r['is_cross'] and axis=='longitudinal' else (3,2,0)
            own=float(T[i,j]);counter=float(T[i,c])
            rows.append(dict(record_id=r['record_id'],axis=axis,size_mm=r['size_mm'],
                             own=own,counter=counter,in_phase=own+counter,
                             opposite_phase=own-counter,quadrature=float(np.hypot(own,counter))))
    return pd.DataFrame(rows)


def nix_comparison_table():
    rows=[]
    for folder,points,mesh in [('nix_level_audit_v3','256 / 1024','Контрольный центр'),
                             ('nix_level_audit_v4','1024 / 4096','Контрольный центр'),
                             ('nix_level_audit_v7','1024 / 4096','Вся область поиска')]:
        path=OUT/folder/'nix_level.local.json'
        if not path.is_file():continue
        audit=read(path);digest=audit['identity']['worker_sha256']
        current=ROOT/'MATLAB_TRKG4_real_subjects/src/run_georg_lateral_nix_audit.m'
        frozen=OUT/'source_snapshots'/(current.stem+'_'+digest+current.suffix)
        older=OUT/'source_snapshots'/(digest+'.m')
        if sha(current)!=digest and not any(p.is_file() and sha(p)==digest for p in [frozen,older]):
            raise ValueError('Executed numerical audit source unavailable')
        for case in audit['cases']:
            delta=np.array(case['quadrature_difference_ohm']);tol=np.array(case['quadrature_tolerance_ohm'])
            rows.append({'Запись':case['record_id'][-3:],'Сетка':mesh,'Число точек':points,
                'Максимальная разность, Ом':float(delta.max()),
                'Максимальное отношение к допуску':float((delta/tol).max()),
                'Собственные передачи':bool(np.all(delta[[0,2]]<=tol[[0,2]])),
                'Перекрёстные передачи':bool(np.all(delta[[1,3]]<=tol[[1,3]]))})
    return pd.DataFrame(rows)


def current_audit_text():
    audit=read(NIX_AUDIT)
    if not audit['complete']:raise ValueError('Search mesh audit incomplete')
    if audit['identity']['plan_sha256']!=sha(SEARCH_MODEL/'plan.local.json'):
        raise ValueError('Numerical audit belongs to a different search mesh')
    own=np.array([np.array(c['quadrature_difference_ohm'])[[0,2]] for c in audit['cases']])
    own_tolerance=np.array([np.array(c['quadrature_tolerance_ohm'])[[0,2]] for c in audit['cases']])
    counter=np.array([np.array(c['quadrature_difference_ohm'])[[1,3]] for c in audit['cases']])
    counter_tolerance=np.array([np.array(c['quadrature_tolerance_ohm'])[[1,3]] for c in audit['cases']])
    derivative=max(max(c['relative_errors']) for c in audit['derivatives'])
    own_result='пройден' if np.all(own<=own_tolerance) else 'не пройден'
    counter_result='пройден' if np.all(counter<=counter_tolerance) else 'не пройден'
    return (f'На сетке всей области поиска критерий собственных передач {own_result}; '
            f'максимальная разность составляет {number(own.max(),5)} Ом, '
            f'максимальное отношение к допуску — {number((own/own_tolerance).max(),5)}. '
            f'Критерий перекрёстных передач {counter_result}; максимальное отношение к допуску — '
            f'{number((counter/counter_tolerance).max(),5)}. '
            f'Максимальная относительная ошибка сопряжённых производных равна {number(derivative,5)}. '
            'Эти проверки относятся к заданным опорным сопротивлениям и контрольному центру. '
            'Они не доказывают пространственную сходимость или точность каждого будущего положения.')


def joint_result():
    path=JOINT_RUN/'execution.local.json'
    if not path.is_file():return None
    result=read(path)
    if result['identity']['worker_sha256']!=sha(ROOT/'MATLAB_TRKG4_real_subjects/src/run_georg_lateral_inverse.m'):
        raise ValueError('Executed joint inverse worker changed')
    if result['identity']['model_plan_sha256']!=sha(SEARCH_MODEL/'plan.local.json'):
        raise ValueError('Joint result belongs to a different mesh')
    if result['identity']['numerical_audit_sha256']!=sha(NIX_AUDIT):
        raise ValueError('Joint numerical audit changed')
    for filename,key in [('inputs.local.json','input_sha256'),
                         ('pose_grid.local.json','grid_sha256'),
                         ('observations.local.csv','observations_sha256')]:
        if sha(OUT/filename)!=result['identity'][key]:
            raise ValueError('Joint input identity changed: '+filename)
    replay=read(JOINT_RUN/'replay_sources.local.json')
    if sha(JOINT_RUN/'replay_sources.local.json')!=result['identity']['resume_manifest_sha256']:
        raise ValueError('Replay source inventory changed')
    for source in replay['sources']:
        if sha(source['path'])!=source['sha256']:
            raise ValueError('Replayed checkpoint changed')
    precision=read(JOINT_RUN/'fit_precision.local.json')
    if precision['identity']!=result['identity']:
        raise ValueError('Final precision check belongs to another run')
    if not np.allclose(precision['fine_Z_ohm'],result['best']['predicted_Z_ohm'],rtol=0,atol=1e-10):
        raise ValueError('Final precision check belongs to another candidate')
    observations=pd.read_csv(OUT/'observations.local.csv')
    target=observations.loc[observations.state!='quiet_breathing','Z_ohm'].to_numpy()
    if len(target)!=24 or not np.allclose(target,result['best']['target_Z_ohm'],rtol=0,atol=1e-10):
        raise ValueError('Joint observation order changed')
    residual=np.asarray(result['best']['predicted_Z_ohm'])-target
    if not np.allclose(residual,result['best']['residual_ohm'],rtol=0,atol=1e-10):
        raise ValueError('Joint residual convention changed')
    if not np.isclose(np.mean(residual**2),result['best_mean_square_ohm2'],rtol=1e-12):
        raise ValueError('Joint objective does not match the residuals')
    return result


def joint_history():
    result=joint_result();rows=[]
    for i in range(1,result['search']['evaluations']+1):
        row=read(JOINT_RUN/f'evaluation_{i:04d}.local.json')
        if row['evaluation']!=i:raise ValueError('Joint checkpoint order changed')
        valid=row['status']=='numerical_candidate_not_validated'
        if not valid and row['status']!='invalid_geometry':raise ValueError('Unfinished checkpoint')
        rows.append(dict(evaluation=i,valid=valid,rmse=np.sqrt(row['mean_square_ohm2']) if valid else np.nan))
    frame=pd.DataFrame(rows)
    frame['best_rmse']=frame.rmse.cummin().ffill()
    if not np.isclose(frame.rmse.min(),np.sqrt(result['best_mean_square_ohm2']),rtol=1e-12):
        raise ValueError('Saved best is not the best completed checkpoint')
    return frame


def joint_frame():
    result=joint_result();frame=pd.read_csv(OUT/'observations.local.csv')
    frame=frame[frame.state!='quiet_breathing'].copy()
    frame['prediction']=result['best']['predicted_Z_ohm']
    frame['residual']=result['best']['residual_ohm']
    frame['label']=[f"{r.record_id[-3:]} · {int(r.size_mm)} мм · "+
                    ('продольная' if r.axis=='longitudinal' else 'поперечная') for r in frame.itertuples()]
    precision=read(JOINT_RUN/'fit_precision.local.json')
    frame['quad_difference']=precision['difference_ohm']
    frame['quad_tolerance']=precision['tolerance_ohm']
    frame['quad_ratio']=frame.quad_difference/frame.quad_tolerance
    frame['quad_pass']=frame.quad_ratio<=1
    return frame


def parameter_table():
    best=joint_result()['best'];initial=read(JOINT_RUN/'evaluation_0001.local.json');rows=[]
    for state in ['inhale','exhale']:
        for j,name in enumerate(['ρ₁, мягкотканный фон','ρ₂, лёгкое','ρ₃, жир']):
            before=initial[f'rho_{state}_ohm_m'][j];after=best[f'rho_{state}_ohm_m'][j]
            unchanged=np.isclose(before,after,rtol=1e-12,atol=0)
            rows.append({'Состояние':STATES[state],'Параметр':name,'Старт, Ом·м':before,
                         'Лучший кандидат, Ом·м':after,'Изменение, %':100*(after/before-1),
                         'Сравнение со стартом':'Сохранился старт' if unchanged else 'Изменился'})
    return pd.DataFrame(rows)


def electrode_table():
    rows=[]
    for e in joint_result()['best']['pose']['electrodes']:
        for contact,point in zip(['I+','U+','U−','I−'],e['centres_lps_mm']):
            rows.append({'L, мм':int(e['size_mm']),'Ось':'Продольная' if e['axis']=='longitudinal' else 'Поперечная',
                         'Контакт':contact,'x LPS, мм':point[0],'y LPS, мм':point[1],'z LPS, мм':point[2]})
    return pd.DataFrame(rows)


def history_figure():
    from plotly.subplots import make_subplots
    frame=joint_history();fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.78,.22],vertical_spacing=.08)
    fig.add_trace(go.Scatter(x=frame.evaluation,y=frame.rmse,mode='lines+markers',
                            name='RMSE каждого допустимого кандидата',line=dict(color='#a5b9cf'),
                            marker=dict(size=5),connectgaps=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=frame.evaluation,y=frame.best_rmse,mode='lines',
                            name='Лучший RMSE к данной оценке',line=dict(color='#2673b8',shape='hv',dash='dash')),row=1,col=1)
    fig.add_trace(go.Scatter(x=[81],y=[frame.rmse.min()],mode='markers',name='Повторный расчёт лучшего кандидата',
                            marker=dict(color='#389058',symbol='diamond',size=10)),row=1,col=1)
    bad=frame[~frame.valid]
    fig.add_trace(go.Scatter(x=bad.evaluation,y=['Отказ геометрии']*len(bad),mode='markers',
                            name='FEM не выполнялся: геометрический отказ',
                            marker=dict(color='#b44841',symbol='x',size=10)),row=2,col=1)
    fig.update_yaxes(title_text='RMSE по 24 уровням, Ом',row=1,col=1)
    fig.update_xaxes(title_text='Номер оценки; 81 — контроль вне бюджета',row=2,col=1)
    fig.update_layout(template='plotly_white',height=640,title='Ход ограниченного совместного поиска',
                      legend=dict(orientation='h',y=-.22),margin=dict(b=160))
    return fig


def residual_figure():
    frame=joint_frame();fig=go.Figure()
    for state,color in [('inhale','#2673b8'),('exhale','#dc7c20')]:
        part=frame[frame.state==state]
        fig.add_trace(go.Bar(x=part.label,y=part.residual,name=STATES[state],marker_color=color))
    fig.add_trace(go.Scatter(x=[frame.label.iloc[0],frame.label.iloc[-1]],y=[0,0],mode='lines',
                            name='Нулевая невязка',line=dict(color='#555',dash='dash')))
    fig.update_layout(template='plotly_white',height=560,title='Невязки лучшего кандидата по каждому отведению',
                      yaxis_title='FEM − BEI, Ом',xaxis=dict(tickangle=-45),barmode='group',margin=dict(b=150))
    return fig


def precision_figure():
    frame=joint_frame();fig=go.Figure()
    for state,color in [('inhale','#2673b8'),('exhale','#dc7c20')]:
        part=frame[frame.state==state]
        fig.add_trace(go.Bar(x=part.label,y=part.quad_ratio,name=STATES[state],marker_color=color,
                      customdata=np.c_[part.quad_difference,part.quad_tolerance],
                      hovertemplate='δ=%{customdata[0]:.6f} Ом<br>Допуск=%{customdata[1]:.6f} Ом<br>δ/допуск=%{y:.4f}<extra>%{fullData.name}</extra>'))
    fig.add_trace(go.Scatter(x=[frame.label.iloc[0],frame.label.iloc[-1]],y=[1,1],mode='lines',
                            name='Граница допуска',line=dict(color='#b44841',dash='dash')))
    fig.update_layout(template='plotly_white',height=580,title='Интегрирование в лучшем кандидате: 1024 и 4096 точек',
                      yaxis_title='Разность / допуск',xaxis=dict(tickangle=-45),barmode='group',margin=dict(b=150))
    return fig


def joint_fit_figure():
    result=joint_result()
    if result is None:raise ValueError('Joint search is not complete')
    row=result['best'];frame=pd.read_csv(OUT/'observations.local.csv')
    frame=frame[frame.state!='quiet_breathing'].copy()
    frame['prediction']=row['predicted_Z_ohm']
    frame['label']=[f"{r.record_id[-3:]} · {int(r.size_mm)} мм · "+
                    ('продольная' if r.axis=='longitudinal' else 'поперечная') for r in frame.itertuples()]
    fig=go.Figure()
    for state,color in [('inhale','#2673b8'),('exhale','#dc7c20')]:
        part=frame[frame.state==state]
        for field,kind,symbol in [('Z_ohm','BEI','diamond'),('prediction','FEM','x')]:
            fig.add_trace(go.Scatter(x=part.label,y=part[field],mode='markers',
                          name=STATES[state]+' · '+kind,marker=dict(color=color,symbol=symbol,size=9)))
    fig.update_layout(template='plotly_white',height=620,title='Согласование боковых отведений в лучшем кандидате',
                      xaxis=dict(tickangle=-45),yaxis_title='Импеданс, Ом',
                      legend=dict(orientation='h',y=-.40),margin=dict(b=250))
    return fig


def joint_pose_figure():
    result=joint_result()
    if result is None:raise ValueError('Joint search is not complete')
    model,_,_,_,_=verify();surface=trimesh.load(model/'body_ct_surface.stl',force='mesh')
    surface=surface.simplify_quadric_decimation(face_count=20000);v=surface.vertices;f=surface.faces
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
                          color='#c8c2b9',opacity=.17,name='Кожа КТ',hoverinfo='skip',showlegend=True))
    pose=result['best']['pose']
    for electrode in pose['electrodes']:
        points=np.array(electrode['centres_lps_mm']);longitudinal=electrode['axis']=='longitudinal'
        name=f"{int(electrode['size_mm'])} мм · "+('продольная' if longitudinal else 'поперечная')
        fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='lines+markers',
                      name=name,line=dict(color='#2673b8' if longitudinal else '#dc7c20',width=4),
                      marker=dict(size=4),customdata=['I+','U+','U−','I−'],
                      hovertemplate='%{customdata}<br>x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra>%{fullData.name}</extra>'))
    centre=pose['centre_lps_mm']
    fig.add_trace(go.Scatter3d(x=[centre[0]],y=[centre[1]],z=[centre[2]],mode='markers',
                  name='Общий центр лучшего кандидата',marker=dict(size=7,color='#389058')))
    p=read(AUTHOR_CENTRE)['centre_lps_mm']
    fig.add_trace(go.Scatter3d(x=[p[0]],y=[p[1]],z=[p[2]],mode='markers',
        name='Центр по воспоминанию автора',marker=dict(size=7,color='#b44841',symbol='diamond'),
        hovertemplate='x=%{x:.3f}; y=%{y:.3f}; z=%{z:.3f} мм<extra>Ориентировочная реконструкция, не измерение</extra>'))
    fig.add_trace(go.Scatter3d(x=[p[0],centre[0]],y=[p[1],centre[1]],z=[p[2],centre[2]],mode='lines',
        name='Прямая между центрами',line=dict(color='#777',width=3,dash='dash')))
    fig.update_layout(template='plotly_white',height=700,title='Положение сборок в лучшем кандидате',
                      scene=dict(aspectmode='data',uirevision='georg-joint-fitted-pose',
                                 camera=dict(eye=dict(x=-1.05,y=.35,z=.15),up=dict(x=0,y=0,z=1)),
                                 xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм'))
    return fig


def source_figure():
    rows=source_rows();obs=pd.read_csv(OUT/'observations.local.csv');fig=go.Figure()
    labels=[r.record_id[-3:]+' · '+str(int(r.size_mm))+' мм · '+
            ('продольная' if r.axis=='longitudinal' else 'поперечная') for r in rows.itertuples()]
    for key,name,color,dash in [('own','Отклик на собственный ток','#b44841','solid'),
        ('counter','Отклик на второй ток','#9361a6','dot'),
        ('in_phase','Одинаковые токи в фазе','#dd8c2d','solid'),
        ('opposite_phase','Одинаковые токи в противофазе','#217d85','solid'),
        ('quadrature','Квадратура: модуль напряжения','#468549','dash')]:
        fig.add_trace(go.Scatter(x=labels,y=rows[key],mode='lines+markers',name=name,
                      line=dict(color=color,dash=dash)))
    for state,color,symbol in [('inhale','#212121','diamond'),('exhale','#777777','square')]:
        values=[obs[(obs.record_id==r.record_id)&(obs.axis==r.axis)&(obs.state==state)].Z_ohm.iloc[0]
                for r in rows.itertuples()]
        fig.add_trace(go.Scatter(x=labels,y=values,mode='markers',name='Запись: '+STATES[state].lower(),
                               marker=dict(color=color,symbol=symbol,size=9)))
    fig.update_layout(template='plotly_white',height=690,title='Явные сценарии двух источников: прямой контроль',
                      yaxis_title='Отклик или измеренный BEI, Ом',xaxis=dict(tickangle=-45),
                      legend=dict(orientation='h',y=-.36),margin=dict(b=250))
    return fig


def source_matrix_figure():
    audit=read(SOURCE_AUDIT);fig=go.Figure();buttons=[]
    for k,r in enumerate(audit['cases']):
        T=np.array(r['port_transfer_ohm']);Z=T[np.ix_([1,3],[0,2])]
        names=['Продольная','Поперечная'] if r['is_cross'] else ['ТТРКГ','Боковая']
        fig.add_trace(go.Heatmap(z=Z,x=names,y=names,visible=k==0,
            colorscale='RdBu_r',zmid=0,text=Z,texttemplate='%{z:.3f} Ом',
            colorbar=dict(title='Ом'),hovertemplate='Ток: %{x}<br>Напряжение: %{y}<br>%{z:.5f} Ом<extra></extra>'))
        title=f"Матрица передачи: запись {r['record_id'][-3:]} · {int(r['size_mm'])} мм"
        buttons.append(dict(label=r['record_id'][-3:]+(' · крест' if r['is_cross'] else ''),method='update',
                            args=[dict(visible=[j==k for j in range(len(audit['cases']))]),dict(title=title)]))
    fig.update_layout(template='plotly_white',height=520,title='Матрица передачи: запись 002 · 140 мм',
                      xaxis_title='Отведение источника тока',yaxis_title='Отведение измеряемого напряжения',
                      updatemenus=[dict(buttons=buttons,x=0,y=1.15)])
    return fig


def main():
    model,current,inputs,grid,audit=verify()
    result=joint_result()
    if result is None:raise ValueError('Completed joint search is required for this report')
    best=result['best'];search=result['search'];pose=best['pose']
    history=joint_history();frame=joint_frame();precision=read(JOINT_RUN/'fit_precision.local.json')
    first=history.rmse.iloc[0];rmse=np.sqrt(best['mean_square_ohm2'])
    best_evaluation=int(history.loc[history.rmse.idxmin(),'evaluation'])
    plan=read(JOINT_RUN/'plan.local.json')
    author=read(AUTHOR_CENTRE)
    distance=np.linalg.norm(np.array(author['centre_lps_mm'])-np.array(pose['centre_lps_mm']))
    summary=dict(source_directory=JOINT_RUN.relative_to(ROOT).as_posix(),
                 source_sha256={name:sha(JOINT_RUN/name) for name in
                                ['execution.local.json','fit_precision.local.json','plan.local.json','replay_sources.local.json']},
                 budget=search['evaluations'],fem_candidates=int(history.valid.sum()),
                 invalid_geometry=int((~history.valid).sum()),best_optimization_evaluation=best_evaluation,
                 fresh_restore_evaluation=best['evaluation'],initial_RMSE_ohm=first,best_RMSE_ohm=rmse,
                 RMSE_reduction_percent=100*(1-rmse/first),quadrature_failures=int((~frame.quad_pass).sum()),
                 max_quadrature_difference_ohm=float(frame.quad_difference.max()),
                 max_quadrature_ratio=float(frame.quad_ratio.max()),resolution_reached=search['resolution_reached'],
                 author_centre_sha256=sha(AUTHOR_CENTRE),best_author_straight_distance_mm=float(distance),
                 centre_v_at_upper_boundary=bool(np.isclose(best['parameters'][7],plan['upper'][7])),
                 quantitative_result_accepted=result['quantitative_result_accepted'],physical_validation=result['physical_validation'])
    write(OUT/'completed_search_report_summary.local.json',summary)
    parameter_table().to_csv(OUT/'completed_search_parameters.local.csv',index=False)
    electrode_table().to_csv(OUT/'completed_search_electrodes.local.csv',index=False)
    frame.to_csv(OUT/'completed_search_fit_and_quadrature.local.csv',index=False)
    numerical=numerical_table();numerical.to_csv(OUT/'numerical_comparison.local.csv',index=False)
    cells,md,code=new_cells()
    md(f'''# Совместная локализация боковых сборок и сопротивления тканей Георгия

**Отчёт по расчёту, завершённому 12.10.2026.** В совместном поиске использованы все боковые размеры и обе оси крестов: 24 уровня BEI на задержках дыхания. Среднеквадратическая невязка снижена с {number(first,4)} до {number(rmse,4)} Ом. Бюджет 80 оценок исчерпан до достижения заданного разрешения поиска. В лучшем кандидате изменилось только сопротивление мягкотканного фона на вдохе; сопротивления лёгкого и жира сохранили стартовые значения. Центр достиг границы области поиска и отличается от позднее указанной автором ориентировочной точки на {number(distance,5)} мм по прямой. Итоговая проверка интегрирования не пройдена для 8 из 24 уровней; пространственная сходимость не подтверждена. Найденные параметры пока не принимаются для коррекции ТТРКГ и расчёта УО/ФВ.

Ноутбук отображает сохранённые результаты MATLAB/FEM и не запускает новый подбор. Порядок изложения: происхождение наблюдений → физическая постановка → численные проверки и связь с расчётом Ника → ход поиска → параметры, координаты и невязки → итоговая проверка и ограничения.

## 1. Входные данные и назначение отведений

Эксперимент выполнен 05.10.2026 на Scorpion v2. BEI содержит базовую и пульсовую составляющие одного импедансного сигнала. Здесь используются медианные уровни исходного BEI в кандидатных задержках дыхания. Входом служит разметка 11.20; её принятие автором этим расчётом не заменяется.

Подтверждение автора устанавливает боковой канал 2 в записях 001–005. В 005 боковая сборка поперечная. Крест 006 содержит продольную ось в канале 1 и сохранённую поперечную ось в канале 2. С 007 продольная боковая сборка занимает канал 1; продолжение в 008–010 и распределение осей 011 приняты по постоянству позднего блока. В этом отчёте экспериментальное название «горизонтальная» соответствует продольному направлению по телу, оси z LPS; перпендикулярная ось соответствует направлению y LPS.

Для задержек подготовлено 24 наблюдения: по 12 на вдохе и выдохе. К ним относятся повторные продольные 140 мм, поперечная 140 мм, обе оси крестов 140 и 60 мм и продольные размеры 120, 100, 80 и 60 мм. Короткая запись 001, сборка 140 мм и монтаж рук 1×1, остаётся самостоятельной проверкой при спокойном дыхании. Задержки ей не приписаны.

По согласованному с автором допущению последовательные записи разных сборок объединяются как наблюдения одного физиологического состояния. Фактическая воспроизводимость дыхания, состояние контактов и межзаписные изменения этим допущением не измерены. Один центр боковых и крестообразных сборок подтверждён автором; его координаты на КТ не были измерены во время эксперимента.

После завершения поиска автор указал приблизительную область центра на изображении КТ. Координаты показанной точки: (−204,346; −20,215; −207,500) мм LPS. Это реконструкция по воспоминанию, независимая от критерия невязки; она не использовалась в завершённом подборе. Радиус неопределённости положения и ошибка переноса между датами КТ и регистрации не установлены. Точка показана отдельно для проверки найденного положения.
''')
    code("from pathlib import Path\nimport sys\nfrom IPython.display import display, HTML\nhere=Path.cwd()\nsys.path.insert(0,str(here))\nimport build_georg_joint_lateral_report as R\nR.verify()\ndisplay(HTML(R.observations_figure().to_html(full_html=False,include_plotlyjs=True)))")
    md('''**Рисунок 1.** Медианы исходного BEI для каждого бокового отведения. Синие линии относятся к продольным, оранжевые — к поперечным осям; точки соответствуют отдельно выбранным дыхательным состояниям. Соединяющие линии связывают состояния одной записи и не задают непрерывное изменение между ними. Каждое отведение можно отключить в легенде. При наведении показаны интервал времени и квантили медленной составляющей.

Квантили описывают изменчивость внутри интервала. Они не являются погрешностью прибора, межзаписной ошибкой или доверительным интервалом сопротивления. Медианы последовательных пятисекундных блоков сохранены отдельно; блоки не считаются независимыми повторениями эксперимента. Для 005–006 используется рабочая частота 400 Гц, для остальных записей — 1000 Гц, как в текущем просмотре 11.20.
''')
    code("import pandas as pd\na=pd.read_csv(R.OUT/'observations.local.csv')\na['state']=a.state.map(R.STATES)\ndisplay(R.table(a[['display_name','state','start_s','end_s','Z_ohm']].rename(columns={'display_name':'Отведение','state':'Состояние','start_s':'Начало, с','end_s':'Конец, с','Z_ohm':'BEI, Ом'})))\ndisplay(HTML(R.cross_figure().to_html(full_html=False,include_plotlyjs=False)))")
    md('''**Рисунок 2.** Разность уровней ΔBEI после перехода к кресту для сохранённой боковой оси: из уровня креста вычитается уровень предыдущего бокового монтажа. Каждая пара столбцов соответствует одному переходу: 004→006, 005→006 или 010→011; цвет обозначает дыхательное состояние. Столбец ниже нуля означает уменьшение уровня в кресте. В легенде можно отключить одно из состояний.

На вдохе продольная сборка 60 мм меняется с 115,33 до 97,17 Ом, а продольная 140 мм — с 79,60 до 76,82 Ом. Поперечная 140 мм меняется с 75,98 до 81,51 Ом. Наблюдение требует проверки полного монтажа, состояния и межканального влияния; причина разностей пока не установлена. Эти уровни не усредняются как повторения одного неизменного электрического опыта.

## 2. Постановка совместной обратной задачи

Конечноэлементная модель (FEM) задаёт обратную задачу для удельных сопротивлений мягкотканного фона ρ₁, лёгкого ρ₂ и жира ρ₃, а также общего поверхностного центра боковых сборок. Сердце и кости остаются отдельными материалами с фиксированными литературными свойствами при 100 кГц. Геометрия, четыре автоматические тканевые маски и цилиндрические руки обхватом 37 см берутся из общей модели 20.20. Автоматические маски жира, костей, сердца и лёгких не прошли независимую анатомическую валидацию. Остальной объём считается мягкотканным фоном со свойствами мышцы. Всему костному объёму приписаны свойства кортикальной кости, всему сердцу — миокарда. Удельные сопротивления считаются вещественными, скалярными и изотропными; это модельные допущения.

Вдох и выдох имеют отдельные тройки сопротивлений при одном положении электродов. Неизменная геометрия КТ на глубоком вдохе для обоих состояний является рабочим допущением. Она не описывает непосредственно измеренное движение грудной клетки при выдохе. Всего в основной постановке девять свободных параметров: две поверхностные координаты центра, общий угол сборок и шесть сопротивлений. Диапазон угла ±20° является вычислительным сценарием поиска; фактическая ошибка ориентации в эксперименте не измерена.

Основной поиск ограничивается физиологически мотивированным профилем литературного справочника при 100 кГц: фон 1,377–10,000 Ом·м; обобщённое лёгкое 3,682–24,074 Ом·м; жир 4,646–41,690 Ом·м. Для фона используется диапазон мышцы, поскольку жир выделен отдельно. Эти литературные границы не являются проверенными индивидуальными нормами Георгия. Выход на границу требует профиля невязки и не выдаётся за определённое сопротивление; автоматическое расширение диапазонов отключено.

Положение, угол и обе тройки сопротивлений совместно изменяются по всем 24 наблюдениям. Критерий — средний квадрат разности между измеренным и FEM-импедансом в Ом². Каждое отведение имеет одинаковый вес; неизвестная полная модель ошибки пока не позволяет считать критерий статистическим правдоподобием. Логарифмы сопротивлений сохраняют положительность параметров. Конечные шаги по координатам и углу учитывают дискретное изменение контактных площадок на сетке.

Полная модель электродов (CEM) учитывает каждый физически присутствующий контакт. При записи с ТТРКГ сохраняются контакты на руках соответствующего монтажа; в кресте учитываются восемь боковых контактов. Диск диаметром 5 мм и контактное сопротивление являются сценариями проекта; размер фактической площадки в этой сессии не измерен. Отключённые и отсутствующие контакты нельзя добавлять произвольно.

Номинальная площадь одного диска равна 19,635 мм². Удельное контактное сопротивление в прямых контролях составляет 1,59155·10⁻⁴ Ом·м²; ток нормирован на 1 А. Несколько контактов одной группы на руке соединяются в один эквипотенциальный электрический узел, сохраняя свои раздельные площадки.

Координаты контактов на цилиндрах рук задаются геометрическим сценарием 20.20. Для монтажа 1×1 верхнее положение единственного измерительного контакта U на каждой руке пока не подтверждено автором.

Координаты модели представлены в системе LPS: x направлена влево, y — кзади, z — вверх. Все координаты и расстояния в этой системе приведены в миллиметрах.

Оценка по собственной токовой паре предполагает нулевой суммарный ток остальных групп. Если прибор возбуждает обе оси одновременно, дополнительно нужны перекрёстные передаточные импедансы, относительные фазы и правила демодуляции. Режим возбуждения Scorpion пока не установлен. Произвольные коэффициенты масштаба, сдвиги и перестановки каналов в подбор не вводятся.
''')
    code('display(HTML(R.pose_figure().to_html(full_html=False,include_plotlyjs=False)))')
    md(f'''**Рисунок 3.** Кожа КТ показана полупрозрачной серой поверхностью; {grid['admissible_count']} синих точек — предварительные центры, для которых лучевая проекция помещает все семь сборок на правую половину сканированного тела. Красный ромб отмечает ориентировочный центр по воспоминанию автора. Шаг сетки центров равен 10 мм. Ни цвет, ни размер точки не показывают невязку: изображён геометрический отбор кандидатов и независимый ориентир. Поверхность и группы точек отключаются отдельно через легенду. Найденное положение показано на рисунке 5. Поверхность сокращена только для отображения; расчётный контур сохранён.

В каждом центре исходные координаты контактов задаются смещениями −L/2, −L/4, +L/4 и +L/2 вдоль выбранной оси и проецируются на поверхность первым пересечением луча по +x. Для поперечной оси кривизна может заметно увеличить расстояния по коже относительно номинала. Эта лучевая схема является предварительной параметризацией; фактические расстояния после наклейки и её соответствие поверхности ещё требуют проверки. Проверка конечных площадок проводится отдельно. Найденные геометрические кандидаты не считаются восстановленными координатами установки.

В совместной оптимизации используется пространственная схема Ника: центр смещается в касательной плоскости, обе ортогональные оси поворачиваются на общий угол, затем точки контактов проецируются на ближайшие треугольники FEM-поверхности. Номинальные расстояния задаются в касательной плоскости; длины по коже непосредственно не измерены. Отдельная проверка удерживает боковые контакты на исходной коже туловища и исключает их переход на искусственные цилиндры рук. Отступ от верхнего и нижнего срезов КТ должен превышать 3,7 мм — разрешённую максимальную протяжённость контактной площадки от центра. Это условие исключает контакт с искусственными торцевыми замыканиями тома. Точки рисунка задают предварительную область поиска, а оптимизация уточняет непрерывные координаты в этой области.

## 3. Проверка точности прямого FEM

Предыдущая проверка полной модели не выдержала критерий 0,02 Ом + 0,5% от модуля импеданса уточнённой сетки. Для контроля реализации дополнительно рассчитаны те же сборки 60 и 140 мм при одинаковом сопротивлении всех пяти материалов, равном 1 Ом·м. Так устраняются тканевые границы при сохранении кожи, рук и конечных контактов.

Для неоднородных контролей в следующей таблице при 100 кГц задано: ρ₁ фона = 2,76359 Ом·м; ρ₂ лёгкого = 9,31525 Ом·м; ρ₃ жира = 23,04061 Ом·м; сопротивление костей = 48,09767 Ом·м; сердца = 4,64881 Ом·м. Это опорные значения модели Gabriel из справочника IT’IS, а не оценки по записям Георгия.
''')
    code("h=pd.read_csv(R.OUT/'homogeneous_mesh_audit.local.csv')\nh['case_id']=h.case_id.str.extract(r'reference_(\\d+)_lateral')[0]+' мм, продольная'\ndisplay(R.table(h[['mesh_mm','case_id','Z_ohm','relative_system_residual']].rename(columns={'mesh_mm':'Номинал сетки, мм','case_id':'Контрольная сборка','Z_ohm':'Z, Ом','relative_system_residual':'Относительная невязка системы'})))\ndisplay(R.table(R.numerical_table()))")
    a=read(model/'lateral_nested_h2_r20/forward_results.local.json');b=read(model/'lateral_nested_h1p5_r20/forward_results.local.json')
    pairs=[]
    for x,y in zip(sorted(a,key=lambda r:r['id']),sorted(b,key=lambda r:r['id'])):
        assert x['id']==y['id'];difference=abs(x['Z_ohm']-y['Z_ohm']);tol=.02+.005*abs(y['Z_ohm'])
        pairs.append(dict(size=int(y['montage']),difference=difference,tolerance=tol,passed=difference<=tol))
    write(OUT/'nested_precision_2_to_1p5.local.json',dict(cases=pairs,all_passed=all(p['passed'] for p in pairs),
        checked_reference_centre_lps_yz_mm=[0,-210],radius_mm=20,global_candidate_convergence_proven=False))
    results='; '.join(f"{p['size']} мм: разность {number(p['difference'],5)} Ом при допуске {number(p['tolerance'],5)} Ом" for p in pairs)
    md(f'''В однородном контроле переход 6→4 мм изменил результат 60 мм на 0,11124 Ом, примерно 0,74%; допуск для этого контрольного импеданса равен 0,09563 Ом. Следовательно, сеточное расхождение сохраняется и без тканевых границ. Невязка линейной системы проверяет решение дискретной задачи, но не точность её приближения к непрерывной модели.

Дополнительно выполнены три локальных сгущения исходной сетки 6 мм. Максимальная длина рёбер в объёмных окрестностях боковых контактов ограничена 3, 2 либо 1,5 мм; радиус окрестностей равен 20 мм. Все исходные вершины и полигональная кожа сохраняются, грани делятся согласованно, инвертированные элементы не допускаются. Материальные доли пересчитаны по исходным маскам с 256 и 1024 точками интегрирования. В таблице разность интегрирования относится к этим двум правилам, а не к сравнению пространственных сеток.

Последнее сравнение локальных сгущений, 2→1,5 мм: {results}. Оно относится только к прежнему иллюстративному центру y=0, z=−210 мм и не подтверждает точность по всей области поиска. Значения в таблице рассчитаны при литературной опорной тройке, а не при восстановленных свойствах Георгия.

Критерий сравнения выполнен для 140 мм и не выполнен для 60 мм. Проверка 256→1024 точек интегрирования на обоих локальных сгущениях выполняется по отдельному допуску 0,02 Ом + 0,1% от модуля Z. Её прохождение не заменяет пространственной проверки.

## 4. Численные проверки по схеме расчёта Ника

Ранний расчёт Ника совместно подбирал два сопротивления, две координаты центра и угол сборок. Следующий полный перебор положений использовал фиксированную найденную пару сопротивлений. Позднее в 33.09 добавлялся жир при ранее выбранном фиксированном центре. Для Георгия перенесены пространственная схема раннего совместного расчёта, полная модель конечных электродов и сопряжённые производные поздней тканевой задачи. Найденные у Ника сопротивления, положение и ограничения глубины 7–30 мм не переносятся на Георгия.

У Ника полная сеточная сходимость также не была доказана. По отдельному указанию автора предварительный поиск для Георгия допускается после воспроизведения соответствующих численных проверок. Строгий режим сохраняет прежнее блокирование, отрицательные сравнения сеток не отменяются. Относительная невязка системы должна быть менее 10⁻⁷, абсолютная ошибка взаимности — менее 10⁻⁷ Ом; разность интегрирования — не более 0,02 Ом + 0,1% модуля передаточного импеданса; относительная ошибка сопряжённых производных — менее 10⁻⁴.

Первоначальная пара 256/1024 точек не прошла проверку одного перекрёстного передаточного импеданса креста 60 мм: разность 0,03461 Ом при допуске 0,02189 Ом. Собственные передачи этого креста прошли проверку. Как в уточнённом расчёте Ника, добавлены 4096 точек. Прежние 1024 точки и их материальные доли воспроизводятся точно; сетка, кожа и маски не изменяются. Следующая таблица сохраняет оба сравнения для каждого монтажа. Проверяются четыре передачи между токовыми и измерительными парами; максимальное отношение разности к её допуску учитывает различие абсолютных уровней. Значение больше единицы означает отказ хотя бы одной передачи.
''')
    code('display(R.table(R.nix_comparison_table()))')
    search_plan=read(SEARCH_MODEL/'plan.local.json')
    md(f'''На сетке контрольного центра при 1024/4096 точках собственные передачи всех монтажей прошли критерий Ника. Максимальная разность равна 0,07097 Ом. Для одной перекрёстной передачи креста 60 мм сохранился отказ: 0,02410 Ом при допуске 0,02061 Ом. Допуск не изменён.

Первая попытка совместного поиска на этой сетке завершилась геометрическими отказами всех 16 начальных кандидатов. Количественные оценки целевой функции не получены: контактные площадки при новых проекциях не выдержали проверку площади или формы либо вышли на искусственные руки. Уточнение только вокруг прежнего контрольного центра оказалось недостаточным для поиска положения.

Для следующего расчёта рёбра правой боковой поверхности сканированного туловища согласованно разделены до 1 мм, как в контактной сетке Ника. Получено {format(search_plan['nodes'],',').replace(',',' ')} узлов и {format(search_plan['tetrahedra'],',').replace(',',' ')} тетраэдров. Исходные вершины, полигональный наружный контур, объём и соединения рук сохранены; положительность объёмов и согласованность границы проверены. Плоские замыкания на верхней и нижней границах КТ сохраняются в объёме, но исключены из области контактного сгущения. Материальные доли новых дочерних элементов пересчитаны по исходным маскам с 1024 и 4096 точками; неизменённые элементы сохраняют прежние доли. Сравнение правил интегрирования выполняется заново на одной и той же новой сетке.

{current_audit_text()}

Предварительный подбор использует собственные передачи; альтернативные сценарии одновременных токов требуют отдельного прохождения проверки перекрёстных передач. Сам численный расчёт не устанавливает фактический режим возбуждения прибора.

После прохождения проверок для собственного источника выполнен предварительный совместный поиск. Восемь начальных положений распределены по области КТ, каждому соответствуют два начальных набора сопротивлений. Бюджет — 80 оценок целевой функции. Не прошедшие проверку конечные площадки сохранены как геометрические отказы. Этот бюджет и стартовая сетка не доказывают глобальность минимума.

Окончательные индивидуальные оценки требуют завершённого поиска, профилей невязки и проверки сравнимых минимумов. На конечных кандидатах повторяются проверки интегрирования и пространственной сетки, изучается влияние фиксированных свойств костей и сердца, контактов и межзаписных изменений. Граничный минимум не становится определённым сопротивлением ткани; неопределённость положения передаётся совместно с неопределённостью сопротивлений. Режим возбуждения двух каналов остаётся отдельным нерешённым вопросом; предварительный основной сценарий использует собственную токовую пару каждого отведения.

Историческая инверсия 33.22 и иллюстративный центр 20.20 не подставляются вместо этого результата. Передача сопротивлений в коррекцию ТТРКГ и расчёт УО/ФВ остаётся последующим этапом.
''')
    md(f'''## 5. Ход совместного поиска и причина остановки

Расчёт завершён по лимиту {search['evaluations']} оценок после {search['sweeps']} проходов алгоритма. Из них {summary['fem_candidates']} кандидатов получили FEM-невязку; {summary['invalid_geometry']} отклонены по геометрии до FEM. Первые 16 оценок относятся к начальным наборам, следующие — к направленному поиску. После остановки первого процесса 77 завершённых оценок воспроизведены из проверенных сохранённых результатов; оценки 78–80 рассчитаны при возобновлении. Это продолжение одного поиска. Лучший кандидат впервые получен на оценке {best_evaluation}; затем отдельно повторно рассчитан без использования сохранённого FEM-результата, под номером {best['evaluation']}, вне бюджета оптимизации.

Среднеквадратическая невязка RMSE — квадратный корень из среднего квадрата 24 разностей FEM − BEI — снизилась с {number(first,7)} до {number(rmse,7)} Ом, на {number(summary['RMSE_reduction_percent'],4)}%. Средний квадрат невязки равен {number(best['mean_square_ohm2'],8)} Ом². Это описательная мера согласования модели и данных; её статистическая погрешность не определена.

Заданное разрешение поиска не достигнуто. Конечные шаги равны 0,18; 0,25; 0,30 по натуральным логарифмам сопротивлений каждого состояния, 10 мм по обеим касательным координатам и 5° по углу. Они характеризуют незавершённую оптимизацию и не являются неопределённостями параметров. Полный перебор области и доказательство глобальности минимума отсутствуют.
''')
    code('display(HTML(R.history_figure().to_html(full_html=False,include_plotlyjs=False)))')
    md('''**Рисунок 4.** Светлая сплошная линия с круглыми метками показывает RMSE каждого допустимого кандидата; разрывы соответствуют геометрическим отказам. Синяя штриховая ступенчатая линия показывает лучший RMSE, достигнутый к данному номеру. Зелёный ромб, номер 81, обозначает отдельный повторный расчёт лучшего кандидата. Красные крестики в нижней панели отмечают геометрические отказы; у них нет вычисленного RMSE. Ряды отключаются независимо через легенду. Соединяющие линии показывают последовательность проверок, а не траекторию физического изменения тканей.

## 6. Сопротивления и положение лучшего кандидата

В следующей таблице лучший найденный набор сопоставлен с первым литературным стартом. Из шести сопротивлений изменилось только ρ₁ на вдохе: с 2,76359 до 2,30834 Ом·м, на −16,47%. Остальные пять значений совпадают со стартом. Они оставались свободными переменными задачи, но до остановки поиска другие значения не вошли в лучший набор. Такое совпадение не доказывает индивидуальную достоверность литературных сопротивлений лёгкого и жира и не позволяет считать их восстановленными. Равенство ρ₂ и ρ₃ на вдохе и выдохе также не устанавливает отсутствия дыхательных изменений этих тканей.
''')
    code('display(R.table(R.parameter_table()))')
    md(f'''Общий центр лучшего кандидата в системе LPS: **({number(pose['centre_lps_mm'][0],5)}; {number(pose['centre_lps_mm'][1],4)}; {number(pose['centre_lps_mm'][2],5)}) мм**. Угол относительно исходной продольной касательной равен {number(pose['angle_deg'],4)}°. Смещения в касательной плоскости: u={number(pose['offsets_uv_mm'][0],4)}, v={number(pose['offsets_uv_mm'][1],4)} мм. Они заданы относительно иллюстративного центра (−206,98; 0; −210) мм и его локального касательного базиса; u и v не совпадают с абсолютными координатами LPS.

Поиск ограничен диапазонами u от −80 до 120 мм, v от −80 до 90 мм и углом ±20°. Значение v=90 мм достигает верхней границы. Поэтому положение может быть ограничено выбранной областью, а минимум требует проверки за её пределами с сохранением анатомических и контактных ограничений. Ни эти координаты, ни угол пока не являются реконструкцией фактической установки с установленной точностью. Округление в отчёте не задаёт точность восстановления.

Авторская ориентировочная точка расположена близко к исходному иллюстративному центру: расстояние между ними около 20,54 мм. От лучшего кандидата она удалена на **{number(distance,5)} мм по прямой**. Это евклидово расстояние в LPS, не расстояние по коже и не измеренная ошибка установки. Существенное расхождение с независимым воспоминанием автора дополнительно ограничивает интерпретацию найденных координат. Для следующего расчёта следует отдельно оценить согласование всех сборок возле указанной области; точное положение и радиус поиска нельзя вывести из одного изображения.
''')
    code('display(HTML(R.joint_pose_figure().to_html(full_html=False,include_plotlyjs=False)))\ndisplay(R.table(R.electrode_table()))')
    md('''**Рисунок 5.** Полупрозрачная серая поверхность показывает кожу КТ. Зелёная точка обозначает общий центр в лучшем численном кандидате; красный ромб — центр по воспоминанию автора. Серая штриховая прямая соединяет эти два центра и показывает их пространственное расхождение, а не путь по коже или траекторию поиска. Синие сплошные линии и точки относятся к продольным сборкам 60–140 мм, оранжевые — к поперечным сборкам 60 и 140 мм. Точки обозначают центры I+, U+, U−, I−; соединяющие прямые показывают порядок контактов и не заменяют длины по криволинейной коже. Обозначения «+» и «−» задают полярность пары в модели. Конечные площадки дисков рассчитываются отдельно. Поверхность, каждая сборка, оба центра и линия сравнения отключаются независимо через легенду; выбранный ракурс сохраняется.

Таблица содержит координаты всех 28 центров контактов в лучшем кандидате. Продольные сборки повторяются в разных записях при одном центре; кресты используют соответствующие продольную и поперечную оси. Это рассчитанные координаты, а не измеренные положения электродов.

## 7. Согласование с каждым боковым отведением
''')
    code('display(HTML(R.joint_fit_figure().to_html(full_html=False,include_plotlyjs=False)))')
    md('''**Рисунок 6.** Ромбы показывают измеренные медианы BEI, крестики — FEM-импедансы лучшего кандидата. Синие метки относятся к задержке на вдохе, оранжевые — к задержке на выдохе. Каждая позиция горизонтальной оси соответствует одной записи и боковой оси; все четыре ряда отключаются независимо через легенду. Метки не соединены линиями, поскольку сборки записаны последовательно.
''')
    code("a=R.joint_frame();a['state']=a.state.map(R.STATES)\ndisplay(R.table(a[['display_name','state','Z_ohm','prediction','residual']].rename(columns={'display_name':'Отведение','state':'Состояние','Z_ohm':'BEI, Ом','prediction':'FEM, Ом','residual':'FEM − BEI, Ом'})))\ndisplay(HTML(R.residual_figure().to_html(full_html=False,include_plotlyjs=False)))")
    md('''**Рисунок 7.** Столбцы показывают разность FEM − BEI для каждого отведения: синие — на вдохе, оранжевые — на выдохе. Серая штриховая линия обозначает нулевую разность. Положительный столбец означает завышение модельного импеданса, отрицательный — занижение. Оба дыхательных состояния и линия нуля отключаются независимо через легенду.

Расхождение имеет структуру по сборкам: модель занижает уровни продольных 140 мм примерно на 17–27 Ом и завышает уровни креста 60 мм. Максимальная абсолютная невязка составляет 60,43 Ом для поперечной оси креста 60 мм на выдохе. При неизменном общем центре повторные продольные 140 мм в модели близки, но реальные уровни и изменения после перехода к кресту воспроизводятся неполностью. Эти наблюдения описывают несогласованность данного кандидата с данными; они не устанавливают дефект прибора или ошибочность всей FEM-постановки.

## 8. Итоговая численная проверка в лучшем кандидате

После повторного прямого расчёта лучшего набора сопоставили две версии материальных долей. Для вновь разделённых тетраэдров использованы 1024 и 4096 точек интегрирования; неизменённые элементы в обоих вариантах сохраняют прежние доли. Сетка, конечные площадки, сопротивления и положение в этом сравнении одинаковы. Контролируется разность δ=|Z₄₀₉₆−Z₁₀₂₄|; индексы обозначают варианты интегрирования. Допуск равен 0,02 Ом + 0,001|Z₄₀₉₆|. Это проверка интегрирования тканевых долей, отдельно от пространственного уточнения сетки и экспериментальной невязки.
''')
    code("a=R.joint_frame();a['state']=a.state.map(R.STATES);a['result']=a.quad_pass.map({True:'Пройден',False:'Не пройден'})\ndisplay(R.table(a[['display_name','state','quad_difference','quad_tolerance','quad_ratio','result']].rename(columns={'display_name':'Отведение','state':'Состояние','quad_difference':'δ, Ом','quad_tolerance':'Допуск, Ом','quad_ratio':'δ / допуск','result':'Критерий'})))\ndisplay(HTML(R.precision_figure().to_html(full_html=False,include_plotlyjs=False)))")
    md(f'''**Рисунок 8.** Столбцы показывают отношение разности интегрирования к её допуску: синие — на вдохе, оранжевые — на выдохе. Красная штриховая линия на уровне 1 обозначает границу допуска. Столбец выше линии означает нарушение критерия. Все ряды, включая линию допуска, можно независимо отключить в легенде; при наведении показаны δ и допуск в омах.

Критерий нарушен для **{summary['quadrature_failures']} из 24 уровней**: продольная сборка 140 мм в записях 002, 003, 004 и 006, на вдохе и выдохе. Максимальная разность равна {number(summary['max_quadrature_difference_ohm'],6)} Ом, максимальное отношение к допуску — {number(summary['max_quadrature_ratio'],6)}. Допуск сохранён. Прохождение проверки в исходном контрольном центре не переносится автоматически на найденное положение и новые сопротивления.

Относительная невязка линейной системы в повторном расчёте лучшего кандидата равна {number(best['maximum_system_residual'],3)}, ошибка взаимности — {number(best['maximum_reciprocity_ohm'],3)} Ом; оба внутренних критерия пройдены. Они подтверждают решение дискретной электрической задачи, но не устраняют отказ интегрирования или отсутствие пространственной сходимости.

## 9. Интерпретация результата и последовательность дальнейшей проверки

Расчёт показал возможность оценивать все боковые отведения на общей исправленной геометрии и совместно менять сопротивления, центр и угол. Он также выявил самостоятельные ограничения: поиск завершён по бюджету, итоговый численный контроль не пройден, экспериментальное расхождение остаётся большим. Критерий приемлемой точности согласования с измерениями ещё не задан. Разность интегрирования до 0,094 Ом значительно меньше RMSE 26,59 Ом, поэтому она сама по себе не объясняет наблюдаемое расхождение. Влияние пространственной дискретизации на этом положении ещё не определено.

На основании этого ограниченного поиска нельзя заключить, что физиологичного решения не существует, что модель принципиально неидентифицируема или что значения литературного старта верны для Георгия. Сохранённый Якобиан содержит производные по шести логарифмам сопротивлений при фиксированном положении. Он не охватывает три пространственных параметра и не задаёт совместную неопределённость всей задачи.

Дальнейшая проверка имеет следующую последовательность:

1. Повторить интегрирование и пространственное уточнение в найденном положении с неизменными критериями. Определить численную погрешность до принятия количественных параметров.
2. Продолжить совместный поиск всех шести сопротивлений, центра и угла. В первую очередь оценить область, указанную автором, затем сопоставить дополнительные старты и результат у границы v=90 мм. Условия пространственного ограничения задавать явно, не выдавая их за измеренную точность установки. Сопоставить несколько минимумов и построить профили невязки.
3. Раздельно проверить влияние автоматической сегментации, фиксированных свойств костей и сердца, размеров контактов и контактного сопротивления. Уточнить фактическое возбуждение и демодуляцию обоих каналов прибора; альтернативные сценарии требуют прошедшего контроля перекрёстных передач.
4. Сопоставить абсолютные уровни и парный дыхательный контраст, оценить влияние кандидатной разметки и межзаписной изменчивости. Только после этого оценивать идентифицируемость и неопределённость индивидуальных сопротивлений и координат.

**Статус для общей методики.** Лучшая найденная тройка на вдохе — (2,30834; 9,31525; 23,04061) Ом·м, на выдохе — (2,76359; 9,31525; 23,04061) Ом·м. Это непринятый численный кандидат. Исправленная общая геометрия сохраняется; сопротивления и координаты этого поиска пока не становятся входом коррекции ТТРКГ, оценки ударного объёма или фракции выброса.
''')
    export(STEM,cells,out=OUT)


if __name__=='__main__':main()
