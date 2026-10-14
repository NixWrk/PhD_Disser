"""Reader of the completed ten-hour Georg FEM run; never starts a solver."""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh

from build_georg_fem_reports import ROOT, new_cells, export, read, sha, table, number, write

PIPE = ROOT / 'MATLAB_TRKG4_real_subjects'
BASE = PIPE / 'output/exploratory/georg_lateral_inverse_20261011'
JOB = BASE / 'author_local_pipeline_10hour_v1'
INPUT = BASE / 'author_inputs_20261012'
RUN = INPUT / 'joint_own_drive_author_local30_10hour_v1'
REPORT = JOB / 'report_20261013'
STEM = '33.28_Совместный_FEM_подбор_за_10_часов_Георгия_Scorpion_v2'
AXES = {'longitudinal': 'Продольная', 'perpendicular': 'Поперечная'}
STATES = {'inhale': 'Вдох', 'exhale': 'Выдох'}
MATERIALS = ['Мягкотканный фон', 'Лёгкое', 'Жир']
COLORS = {'inhale': '#2673b8', 'exhale': '#c46b24'}


@lru_cache(maxsize=1)
def load_run():
    config, state = read(JOB/'configuration.local.json'), read(JOB/'state.local.json')
    execution, validation = read(RUN/'execution.local.json'), read(JOB/'numerical_validation.local.json')
    for source in config['code']:
        if sha(PIPE/source['snapshot']) != source['sha256']:
            raise ValueError('Executed code snapshot changed')
    for name, key in [('inputs.local.json', 'input_sha256'), ('observations.local.csv', 'observations_sha256'),
                      ('pose_grid.local.json', 'grid_sha256'), ('author_centre_20261012.local.json', 'author_centre_sha256')]:
        if sha(INPUT/name) != execution['identity'][key]:
            raise ValueError('Saved inverse input changed: '+name)
    if sha(JOB/'selected_candidate.local.json') != validation['selected_candidate_sha256']:
        raise ValueError('Selected numerical-check candidate changed')
    if read(JOB/'selected_candidate.local.json') != execution:
        raise ValueError('Checks do not use the completed search candidate')
    replay = read(RUN/'replay_sources.local.json')
    if sha(RUN/'replay_sources.local.json') != execution['identity']['resume_manifest_sha256']:
        raise ValueError('Checkpoint provenance changed')
    for source in replay['sources']:
        if sha(source['path']) != source['sha256']:
            raise ValueError('Original replayed checkpoint changed')
    obs = pd.read_csv(INPUT/'observations.local.csv')
    obs = obs[obs.state != 'quiet_breathing'].reset_index(drop=True)
    best = execution['best']
    np.testing.assert_allclose(obs.Z_ohm, best['target_Z_ohm'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.array(best['predicted_Z_ohm'])-obs.Z_ohm, best['residual_ohm'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.mean(np.square(best['residual_ohm'])), execution['best_mean_square_ohm2'], rtol=1e-12)
    for result in validation['integration_checks']+validation['mesh_checks']:
        if result['identity']['checked_candidate_sha256'] != validation['selected_candidate_sha256']:
            raise ValueError('Fixed check belongs to another candidate')
        np.testing.assert_array_equal(result['parameters'], best['parameters'])
        if result['pose'] != best['pose']:
            raise ValueError('Electrode centres changed in a fixed numerical check')
    history = [read(f) for f in sorted(RUN.glob('evaluation_*.local.json'))]
    author = read(INPUT/'author_centre_20261012.local.json')
    precision = read(RUN/'fit_precision.local.json')
    if precision['identity'] != execution['identity']:
        raise ValueError('Original precision check belongs to another search')
    reference_plan = read(PIPE/config['search_model']/'plan.local.json')
    if sha(PIPE/config['search_model']/'plan.local.json') != execution['identity']['model_plan_sha256']:
        raise ValueError('Search mesh plan changed')
    model = PIPE/'output/exploratory/georg_ct_model_20261011'
    ref = read(PIPE/config['search_model']/'reference_materials.local.json')
    return dict(config=config, state=state, execution=execution, best=best, validation=validation,
                obs=obs, history=history, author=author, precision=precision, model=model, ref=ref,
                reference_plan=reference_plan)


def lead_label(row):
    cross = 'Крест ' if bool(row.is_cross) else ''
    return f'{cross}{int(row.size_mm)} мм · {AXES[row.axis].lower()} · {row.record_id[-3:]}'


def observations():
    d=load_run();f=d['obs'].copy();f['lead']=f.apply(lead_label,axis=1)
    f['predicted']=d['best']['predicted_Z_ohm'];f['residual']=d['best']['residual_ohm']
    f['SSE_percent']=100*f.residual**2/(f.residual**2).sum()
    return f


def output_table():
    f=observations()
    return f[['lead','channel','state','Z_ohm','predicted','residual','SSE_percent']].assign(
        state=f.state.map(STATES)).rename(columns={'lead':'Отведение и запись','channel':'Канал BEI',
        'state':'Дыхание','Z_ohm':'Измерено, Ом','predicted':'Модель, Ом',
        'residual':'Модель − измерение, Ом','SSE_percent':'Доля квадратичной ошибки, %'})


def parameters():
    d=load_run();bounds=read(INPUT/'inputs.local.json')['search_bounds_ohm_m'];first=d['history'][0]
    rows=[]
    for state in ['inhale','exhale']:
        for k,name in enumerate(MATERIALS):
            value=d['best']['rho_'+state+'_ohm_m'][k]
            rows.append({'Ткань':name,'Дыхание':STATES[state], 'Старт, Ом·м':first['rho_'+state+'_ohm_m'][k],
                         'Лучший кандидат, Ом·м':value,'Нижняя граница, Ом·м':bounds[k][0],
                         'Верхняя граница, Ом·м':bounds[k][1],
                         'Изменилось относительно старта':not np.isclose(value,first['rho_'+state+'_ohm_m'][k])})
    return pd.DataFrame(rows)


def window_table():
    f=observations()
    return f[['lead','state','fs_hz','start_s','end_s','slow_p05_ohm','slow_p95_ohm']].assign(
        state=f.state.map(STATES)).rename(columns={'lead':'Отведение и запись','state':'Дыхание',
        'fs_hz':'Сценарий частоты отсчётов, Гц','start_s':'Начало, с','end_s':'Конец, с',
        'slow_p05_ohm':'5-й процентиль медленной компоненты, Ом',
        'slow_p95_ohm':'95-й процентиль медленной компоненты, Ом'})


def contacts():
    rows=[]
    for e in load_run()['best']['pose']['electrodes']:
        xyz=np.array(e['centres_lps_mm'])
        for label,p in zip(['I+','U+','U−','I−'],xyz):
            rows.append({'L, мм':e['size_mm'],'Ось':AXES[e['axis']],'Контакт':label,
                         'x LPS, мм':p[0],'y LPS, мм':p[1],'z LPS, мм':p[2]})
    return pd.DataFrame(rows)


def spacing_table():
    rows=[]
    centre=np.array(load_run()['best']['pose']['centre_lps_mm'])
    for e in load_run()['best']['pose']['electrodes']:
        xyz=np.array(e['centres_lps_mm']);L=e['size_mm']
        rows.append({'L, мм':L,'Ось':AXES[e['axis']], 'I+–I− по прямой, мм':float(np.linalg.norm(xyz[0]-xyz[3])),
                     'U+–U− по прямой, мм':float(np.linalg.norm(xyz[1]-xyz[2])),
                     'Номинальная база U, мм':L/2,
                     'Смещение середины I от центра, мм':float(np.linalg.norm((xyz[0]+xyz[3])/2-centre)),
                     'Минимум промежутка соседей, мм':float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).min())})
    return pd.DataFrame(rows)


def combined_states():
    d=load_run();pose=d['best']['pose'];f=d['obs']
    rows=[r for r in d['history'] if r['status']=='numerical_candidate_not_validated' and r['pose']==pose]
    chosen=[];predicted=np.empty(24);q=np.array(d['best']['parameters']).copy()
    for k,state in enumerate(['inhale','exhale']):
        ids=np.flatnonzero(f.state==state)
        best=min(rows,key=lambda r:np.sum(np.square(np.array(r['residual_ohm'])[ids])))
        predicted[ids]=np.array(best['predicted_Z_ohm'])[ids]
        q[3*k:3*k+3]=np.array(best['parameters'])[3*k:3*k+3]
        chosen.append(best['evaluation'])
    residual=predicted-f.Z_ohm.to_numpy()
    return dict(source_evaluations=chosen,parameters=q.tolist(),predicted_Z_ohm=predicted.tolist(),
                residual_ohm=residual.tolist(),RMSE_ohm=float(np.sqrt(np.mean(residual**2))),
                separate_states_at_identical_pose=True,new_fem_performed=False,
                independent_numerical_validation_performed=False,quantitative_result_accepted=False)


def material_diagnostics():
    d=load_run();J=np.array(d['best']['jacobian_logrho']);e=np.array(d['best']['residual_ohm'])
    ids=np.flatnonzero(d['obs'].state=='inhale');A=J[ids,:3];norms=np.linalg.norm(A,axis=0)
    gradient=2/24*J.T@e
    return dict(MSE_gradient_logrho=gradient.tolist(),inhale_column_norms_ohm=norms.tolist(),
                soft_fat_cosine=float(A[:,0]@A[:,2]/norms[0]/norms[2]),
                lung_one_percent_linear_response_norm_ohm=float(norms[1]*.01),
                grid_change_inhale_norm_ohm=float(np.linalg.norm(np.array(d['validation']['spatial_comparisons'][0]['difference_ohm'])[ids])))


def combination_table():
    d=load_run();c=combined_states();rows=[]
    for k,state in enumerate(['inhale','exhale']):
        rows.append({'Дыхание':STATES[state],'Источник сочетания: оценка':c['source_evaluations'][k],
                     'ρ₁, Ом·м':np.exp(c['parameters'][3*k]),'ρ₂, Ом·м':np.exp(c['parameters'][3*k+1]),
                     'ρ₃, Ом·м':np.exp(c['parameters'][3*k+2])})
    return pd.DataFrame(rows)


def contrast_diagnostics():
    d=load_run();obs=d['obs'];target=obs.Z_ohm.to_numpy()
    inhale=np.flatnonzero(obs.state=='inhale');exhale=np.flatnonzero(obs.state=='exhale')
    np.testing.assert_array_equal(obs.iloc[inhale].record_id,obs.iloc[exhale].record_id)
    np.testing.assert_array_equal(obs.iloc[inhale].channel,obs.iloc[exhale].channel)
    rows=[]
    for name,pred in [('Сохранённый кандидат',d['best']['predicted_Z_ohm']),
                      ('Сочетание сохранённых фаз',combined_states()['predicted_Z_ohm'])]:
        pred=np.array(pred);mean_error=(pred[inhale]+pred[exhale]-target[inhale]-target[exhale])/2
        delta_error=pred[exhale]-pred[inhale]-(target[exhale]-target[inhale])
        mean_MSE=float(np.mean(mean_error**2));delta_MSE=float(np.mean(delta_error**2))
        total=float(np.mean((pred-target)**2));np.testing.assert_allclose(mean_MSE+.25*delta_MSE,total,rtol=1e-12)
        rows.append(dict(name=name,mean_MSE_ohm2=mean_MSE,contrast_MSE_ohm2=delta_MSE,
                         contrast_RMSE_ohm=float(np.sqrt(delta_MSE)),
                         weighted_contrast_percent=100*.25*delta_MSE/total))
    return rows


def prepare_ct_planes():
    import nibabel as nib
    from scipy.ndimage import map_coordinates
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    seg=PIPE/'output/exploratory/scorpion_georg_fem_20261008/segmentation'
    ct=nib.load(seg/'static_ct.nii.gz');hu=np.asanyarray(ct.dataobj)
    labels=np.zeros(ct.shape,np.uint8)
    for tissue,k in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        mask=nib.load(seg/(tissue+'.nii.gz'))
        np.testing.assert_allclose(mask.affine,ct.affine,rtol=0,atol=1e-5)
        labels[np.asanyarray(mask.dataobj)>0]=k
    pose=load_run()['best']['pose'];centre=np.array(pose['centre_lps_mm'])
    inward=-np.array(pose['normal_unit_xyz']);transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    sx=np.linspace(-85,85,481);depth=np.linspace(-20,180,501);ss,dd=np.meshgrid(sx,depth)
    fig,axes=plt.subplots(1,2,figsize=(13,7),constrained_layout=True)
    cmap=ListedColormap(['#68b2cd','#e2d3ac','#b65362','#ccad43'])
    offplanes=[]
    for ax,axis,key in zip(axes,['longitudinal','perpendicular'],['primary_unit_xyz','perpendicular_unit_xyz']):
        direction=np.array(pose[key]);xyz=centre+ss[...,None]*direction+dd[...,None]*inward
        ijk=nib.affines.apply_affine(transform,xyz.reshape(-1,3))
        image=map_coordinates(hu,ijk.T,order=1,mode='constant',cval=-1024).reshape(ss.shape)
        mask=map_coordinates(labels,ijk.T,order=0,mode='constant',cval=0).reshape(ss.shape)
        ax.imshow(image,extent=[sx[0],sx[-1],depth[-1],depth[0]],cmap='gray',vmin=-750,vmax=300,aspect='equal')
        ax.imshow(np.ma.masked_where(mask==0,mask),extent=[sx[0],sx[-1],depth[-1],depth[0]],
                  cmap=cmap,vmin=1,vmax=4,alpha=.48,interpolation='nearest',aspect='equal')
        e=next(e for e in pose['electrodes'] if e['size_mm']==140 and e['axis']==axis)
        delta=np.array(e['centres_lps_mm'])-centre
        s=delta@direction;d=delta@inward
        for label,x,y in zip(['I+','U+','U−','I−'],s,d):
            ax.plot(x,y,'o' if label.startswith('I') else 's',color='#e87f28' if label.startswith('I') else '#337dc6',markeredgecolor='white',markersize=7)
            ax.annotate(label,(x,y),xytext=(3,-12),textcoords='offset points',color='white',fontsize=10)
        ax.plot(0,0,'+',color='#83e19e',markersize=11,markeredgewidth=2)
        offplanes.append(float(np.max(np.abs(delta@np.cross(direction,inward)))))
        ax.set(title=AXES[axis]+' ось, 140 мм',xlabel='Координата вдоль касательной оси, мм',ylabel='Координата внутрь по нормали, мм')
    fig.legend(handles=[Patch(color=cmap(i),label=n) for i,n in enumerate(['Лёгкое','Кость','Сердце','Жир'])],loc='outside lower center',ncol=4)
    path=REPORT/'ct_planes_140.local.png';fig.savefig(path,dpi=150);plt.close(fig)
    write(REPORT/'ct_planes.local.json',dict(centre_lps_mm=centre.tolist(),
          maximum_contact_distance_from_plane_mm=offplanes,
          geometric_projection_only=True,effective_h_estimated=False,
          sources=[dict(name=n,sha256=sha(seg/(n+'.nii.gz'))) for n in ['static_ct','fat','heart','bone','lung']]))


def plot_layout(fig,title,y,height=480):
    if fig.data and isinstance(fig.data[0].x[0],str):
        height=max(height,700);bottom=280;legend_y=-.65
    else:
        bottom=160;legend_y=-.24
    fig.update_layout(template='plotly_white',height=height,title=title,yaxis_title=y,
                      legend=dict(orientation='h',y=legend_y),margin=dict(b=bottom),
                      uirevision='georg-tenhour-fixed',hovermode='closest')
    return fig


def fit_figure():
    f=observations();fig=go.Figure()
    for state in STATES:
        g=f[f.state==state];c=COLORS[state]
        fig.add_trace(go.Scatter(x=g.lead,y=g.Z_ohm,name=STATES[state]+': измерение',mode='markers',
                                marker=dict(color=c,size=9,symbol='circle')))
        fig.add_trace(go.Scatter(x=g.lead,y=g.predicted,name=STATES[state]+': модель',mode='lines+markers',
                                line=dict(color=c,dash='solid' if state=='inhale' else 'dash'),
                                marker=dict(symbol='diamond',size=6)))
    return plot_layout(fig,'Измеренные уровни и лучший численный кандидат','Импеданс, Ом',620)


def residual_figure():
    f=observations();fig=go.Figure()
    for state in STATES:
        g=f[f.state==state];fig.add_trace(go.Bar(x=g.lead,y=g.residual,
            name=STATES[state]+': модель − измерение',marker_color=COLORS[state],
            customdata=g.SSE_percent,hovertemplate='%{y:.3f} Ом<br>Доля ошибки: %{customdata:.2f}%<extra>%{fullData.name}</extra>'))
    fig.add_hline(y=0,line_color='#555',line_dash='dot')
    return plot_layout(fig,'Распределение невязок по отведениям','Невязка, Ом',590)


def history_figure():
    d=load_run();fig=go.Figure();valid=[r for r in d['history'] if r['status']=='numerical_candidate_not_validated']
    for resumed,color,symbol,name in [(True,'#898989','circle-open','Повторно использованный кандидат'),
        (False,'#2673b8','circle','Новое FEM-вычисление')]:
        a=[r for r in valid if ('source_evaluation' in r)==resumed]
        fig.add_trace(go.Scatter(x=[r['evaluation'] for r in a],y=[np.sqrt(r['mean_square_ohm2']) for r in a],
            mode='markers',name=name,marker=dict(color=color,symbol=symbol,size=8)))
    fig.add_trace(go.Scatter(x=[r['evaluation'] for r in valid],
        y=np.minimum.accumulate([np.sqrt(r['mean_square_ohm2']) for r in valid]),
        mode='lines',name='Лучший достигнутый уровень',line=dict(color='#172d3e',width=3)))
    fig.add_vline(x=6.5,line_dash='dash',line_color='#777')
    fig.add_trace(go.Scatter(x=[37],y=[np.sqrt(d['execution']['best_mean_square_ohm2'])],
        mode='markers',name='Независимое повторение лучшего кандидата',marker=dict(color='#bb493c',symbol='star',size=12)))
    fig.update_xaxes(title='Номер оценки целевой функции')
    return plot_layout(fig,'Ход ограниченного совместного поиска','Среднеквадратическая невязка, Ом')


def rho_figure(k):
    fig=go.Figure();rows=[r for r in load_run()['history'] if r['status']=='numerical_candidate_not_validated']
    for state in STATES:
        fig.add_trace(go.Scatter(x=[r['evaluation'] for r in rows],y=[r['rho_'+state+'_ohm_m'][k] for r in rows],
            mode='markers',name=STATES[state]+': проверенный кандидат',
            marker=dict(color=COLORS[state],symbol='circle' if state=='inhale' else 'x',size=7)))
    best=np.array([r['mean_square_ohm2'] for r in rows]);minimum=np.minimum.accumulate(best)
    selected=[];i=0
    for n in range(len(rows)):
        if best[n]<=minimum[n]:i=n
        selected.append(rows[i]['rho_inhale_ohm_m'][k])
    fig.add_trace(go.Scatter(x=[r['evaluation'] for r in rows],y=selected,mode='lines',
        name='Параметр лучшего кандидата на вдохе',line=dict(color='#222',width=2)))
    fig.update_xaxes(title='Номер оценки целевой функции')
    return plot_layout(fig,MATERIALS[k]+': проверенные сопротивления','Удельное сопротивление, Ом·м',400)


def breathing_figure():
    f=observations();f['combined']=combined_states()['predicted_Z_ohm'];rows=[]
    for _,g in f.groupby(['record_id','channel'],sort=False):
        a=g.set_index('state');rows.append((g.lead.iloc[0],a.loc['exhale','Z_ohm']-a.loc['inhale','Z_ohm'],
                                          a.loc['exhale','predicted']-a.loc['inhale','predicted'],
                                          a.loc['exhale','combined']-a.loc['inhale','combined']))
    fig=go.Figure(go.Bar(x=[r[0] for r in rows],y=[r[1] for r in rows],name='Измеренный контраст',marker_color='#2673b8'))
    fig.add_trace(go.Scatter(x=[r[0] for r in rows],y=[r[2] for r in rows],mode='lines+markers',
        name='Контраст лучшего кандидата',line=dict(color='#c46b24'),marker=dict(symbol='diamond')))
    fig.add_trace(go.Scatter(x=[r[0] for r in rows],y=[r[3] for r in rows],mode='lines+markers',
        name='Контраст сочетания сохранённых фаз',line=dict(color='#7862aa',dash='dot'),
        marker=dict(symbol='diamond-open')))
    return plot_layout(fig,'Парный дыхательный контраст: выдох − вдох','Разность импедансов, Ом',560)


def precision_figure():
    d=load_run();f=observations();fig=go.Figure()
    checks=[('Исходная сетка: 1024/4096',d['precision']),
            ('Исходная сетка: 4096/16384',d['validation']['integration_checks'][0]),
            ('Локально 3 мм: 4096/16384',d['validation']['mesh_checks'][0])]
    for name,r in checks:
        fig.add_trace(go.Scatter(x=np.arange(1,25),y=np.array(r['difference_ohm'])/r['tolerance_ohm'],
            mode='lines+markers',name=name,customdata=np.c_[f.lead,f.state.map(STATES),r['difference_ohm']],
            hovertemplate='%{customdata[0]} · %{customdata[1]}<br>Разность: %{customdata[2]:.6f} Ом<br>Разность/допуск: %{y:.4f}<extra>%{fullData.name}</extra>'))
    fig.add_hline(y=1,line_dash='dash',line_color='#b44841')
    fig.update_xaxes(title='Номер наблюдения в таблице 24 уровней',dtick=1)
    return plot_layout(fig,'Контроль интегрирования при неизменных параметрах','Разность / допуск')


def mesh_figure():
    d=load_run();f=observations();s=d['validation']['spatial_comparisons'][0];fig=go.Figure()
    for state in STATES:
        ids=np.flatnonzero(f.state==state)
        fig.add_trace(go.Bar(x=f.loc[ids,'lead'],y=np.array(s['difference_ohm'])[ids]/np.array(s['tolerance_ohm'])[ids],
            name=STATES[state]+': разность / допуск',marker_color=COLORS[state],
            customdata=np.c_[np.array(s['difference_ohm'])[ids],np.array(s['tolerance_ohm'])[ids]],
            hovertemplate='%{y:.3f}<br>Разность: %{customdata[0]:.5f} Ом<br>Допуск: %{customdata[1]:.5f} Ом<extra>%{fullData.name}</extra>'))
    fig.add_hline(y=1,line_dash='dash',line_color='#b44841')
    return plot_layout(fig,'Изменение FEM-результата после объёмного сгущения до 3 мм','Разность / допуск',590)


def jacobian_figure():
    d=load_run();f=observations();ids=np.flatnonzero(f.state=='inhale')
    J=np.array(d['best']['jacobian_logrho'])[ids,:3]
    fig=go.Figure(go.Heatmap(z=J,x=MATERIALS,y=f.loc[ids,'lead'],colorscale='RdBu',zmid=0,
        colorbar=dict(title='Ом'),hovertemplate='%{y}<br>%{x}<br>∂Z/∂lnρ = %{z:.4f} Ом<extra></extra>'))
    return plot_layout(fig,'Локальная чувствительность при фиксированном положении','Отведение',580)


def geometry_figure():
    d=load_run();fig=go.Figure()
    surfaces=[('body_ct_surface.stl','Кожа КТ','#b8ada1',.18,True),
              ('picker/lung_lps.stl','Лёгкие','#529cba',.35,'legendonly'),
              ('picker/fat_lps.stl','Жир','#c6a33c',.22,'legendonly'),
              ('picker/bone_lps.stl','Кости','#e5d8bc',.65,'legendonly'),
              ('picker/heart_lps.stl','Сердце','#bc5e64',.65,'legendonly')]
    for path,name,color,opacity,visible in surfaces:
        mesh=trimesh.load(d['model']/path,force='mesh')
        if len(mesh.faces)>20000:mesh=mesh.simplify_quadric_decimation(face_count=20000)
        v,f=mesh.vertices,mesh.faces
        fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
            name=name,color=color,opacity=opacity,visible=visible,showlegend=True,hoverinfo='skip'))
    author=np.array(d['author']['centre_lps_mm']);best=np.array(d['best']['pose']['centre_lps_mm'])
    fig.add_trace(go.Scatter3d(x=[author[0]],y=[author[1]],z=[author[2]],mode='markers',
        name='Ориентировочный центр автора',marker=dict(color='#bb493c',symbol='diamond',size=7)))
    fig.add_trace(go.Scatter3d(x=[best[0]],y=[best[1]],z=[best[2]],mode='markers',
        name='Центр лучшего кандидата',marker=dict(color='#3f875a',size=7)))
    fig.add_trace(go.Scatter3d(x=[author[0],best[0]],y=[author[1],best[1]],z=[author[2],best[2]],
        mode='lines',name='Расстояние между центрами',line=dict(color='#666',dash='dash',width=3)))
    for e in d['best']['pose']['electrodes']:
        xyz=np.array(e['centres_lps_mm']);color='#2673b8' if e['axis']=='longitudinal' else '#c46b24'
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines+markers',
            name=f"{int(e['size_mm'])} мм · {AXES[e['axis']].lower()}",line=dict(color=color,width=3),
            marker=dict(color=color,size=5,symbol=['circle','square','square','circle']),
            text=['I+','U+','U−','I−'],hovertemplate='%{text}<br>x=%{x:.3f}; y=%{y:.3f}; z=%{z:.3f} мм<extra>%{fullData.name}</extra>'))
    fig.update_layout(template='plotly_white',height=780,title='Общий центр и семь геометрий боковых сборок',
        uirevision='georg-tenhour-geometry',legend=dict(orientation='h',y=-.05),
        scene=dict(aspectmode='data',uirevision='georg-tenhour-geometry',xaxis_title='x LPS, мм',
                   yaxis_title='y LPS, мм',zaxis_title='z LPS, мм',
                   camera=dict(eye=dict(x=-1.2,y=.2,z=.15),up=dict(x=0,y=0,z=1))))
    return fig


def stage_table():
    labels={'search_30':'Поиск и итоговое повторение', 'prepare_quadrature_4096_16384':'Подготовка интегрирования 4096/16384',
            'check_author_q16384_v1':'Контроль интегрирования', 'prepare_local_volume_h3p0':'Подготовка сетки 3 мм',
            'check_author_local_volume_h3p0_v1':'FEM на сетке 3 мм', 'prepare_local_volume_h2p0':'Подготовка сетки 2 мм'}
    rows=[]
    for s in load_run()['state']['stages']:
        seconds=(pd.Timestamp(s['finished_utc'])-pd.Timestamp(s['started_utc'])).total_seconds()
        rows.append({'Стадия':labels[s['label']],'Время, мин':seconds/60,
                     'Выполнение':'Завершено' if s['status']=='complete' else 'Прервано по времени'})
    return pd.DataFrame(rows)


def audit_table():
    return pd.DataFrame(read(REPORT/'astra_review.local.json')['reader_findings'])


def main():
    REPORT.mkdir(exist_ok=True)
    d=load_run();b=d['best'];v=d['validation'];f=observations();p=parameters()
    initial=np.sqrt(d['history'][0]['mean_square_ohm2']);rmse=np.sqrt(b['mean_square_ohm2'])
    distance=np.linalg.norm(np.array(b['pose']['centre_lps_mm'])-d['author']['centre_lps_mm'])
    rows=[r for r in d['history'] if r['evaluation']<=36]
    geometry_failures=sum(r['status']=='invalid_geometry' for r in rows)
    bad=(f.record_id=='SCV2-G-011')&(f.axis=='perpendicular')
    share=float(f.loc[bad,'SSE_percent'].sum());without=float(np.sqrt(np.mean(f.loc[~bad,'residual']**2)))
    c=v['integration_checks'][0]['patch_qc']['contacts'];m=v['mesh_checks'][0]['patch_qc']['contacts']
    contact_change={key:max(abs(a[key]-z[key]) for a,z in zip(c,m)) for key in ['area_error','centroid_offset_mm','extent_mm']}
    if [(x['row'],x['contact']) for x in c]!=[(x['row'],x['contact']) for x in m]:raise ValueError('Contact QC rows are not aligned')
    combo=combined_states();diag=material_diagnostics()
    write(REPORT/'combined_states.local.json',combo);write(REPORT/'material_diagnostics.local.json',diag)
    contrast=contrast_diagnostics();write(REPORT/'contrast_diagnostics.local.json',contrast)
    prepare_ct_planes()
    summary=dict(initial_RMSE_ohm=initial,best_RMSE_ohm=rmse,author_distance_mm=distance,
                 combined_saved_states_RMSE_ohm=combo['RMSE_ohm'],
                 search_evaluations=36,valid_search_candidates=36-geometry_failures,
                 replayed_candidates=sum('source_evaluation' in r for r in rows),
                 independent_best_restore=b['evaluation'],bad_cross_SSE_percent=share,
                 fixed_parameter_RMSE_without_bad_cross_ohm=without,contact_qc_changes=contact_change,
                 integration_passed=True,mesh_comparison_passed=False,
                 quantitative_result_accepted=False,builder_sha256=sha(__file__),
                 source_hashes={str(path.relative_to(ROOT)):sha(path) for path in [RUN/'execution.local.json',
                    RUN/'plan.local.json',JOB/'numerical_validation.local.json',JOB/'selected_candidate.local.json',
                    JOB/'state.local.json',REPORT/'astra_review.local.json']})
    write(REPORT/'summary.local.json',summary)
    output_table().to_csv(REPORT/'fit.local.csv',index=False)
    contacts().to_csv(REPORT/'contacts.local.csv',index=False)
    cells,md,code=new_cells()
    md(f'''# Совместный подбор сопротивлений тканей и положения боковых сборок Георгия

**Расчёт с общим ограничением 10 часов, завершённый 13 октября 2026 года.** Среднеквадратическая невязка (RMSE) уменьшилась с {number(initial)} до {number(rmse)} Ом. Проверка интегрирования пройдена. Объёмное сгущение до 3 мм изменило все 24 расчётных импеданса сверх допуска; подготовка уровня 2 мм прервана по времени. Поиск исчерпал бюджет до достижения заданного разрешения. Найденный набор сопротивлений и координат остаётся непринятым кандидатом.

Отчёт показывает сохранённые результаты, их происхождение и критический разбор постановки с участием модели Astra. Ячейки читают завершённые вычисления; новый FEM и повторный подбор не выполняются. Предшествующий широкий поиск сохранён в [33.23](33.23_Совместная_локализация_боковых_сборок_Георгия_Scorpion_v2.ipynb); описание регистрации находится в [10.00](10.00_Эксперимент_и_сигналы_Георгия_Scorpion_v2.ipynb).

## 1. Экспериментальные наблюдения

Scorpion v2 регистрировал общий импеданс BEI; базовая и пульсовая составляющие не записывались отдельными файлами. В обратную задачу включены медианы исходного BEI в кандидатных интервалах задержек дыхания: 12 отведений на вдохе и те же 12 на выдохе. Короткая тестовая запись при спокойном дыхании в целевую функцию не включена. Дыхательные границы реконструированы ретроспективно и остаются непринятыми вручную.

Боковая сборка занимает канал 2 в записях 002–005; в 005 она поперечная. В кресте 006 канал 1 соответствует продольной оси, канал 2 — поперечной. В 007 продольная боковая сборка находится в канале 1; продолжение схемы в 008–010 и распределение осей креста 011 приняты по постоянству позднего блока. Назначение каналов не подбиралось по импедансу. Экспериментальное название «горизонтальная» в этой модели соответствует продольной оси тела.

Разные размеры зарегистрированы последовательно. Их сопоставимость как одного физиологического состояния принята по указанию автора; межзаписная изменчивость, контакт и повторная установка могут нарушать это допущение. В записях 007–010 использован сценарий контактов ТТРКГ 1×4; вариант 1×1 остаётся открытой альтернативой. Столбец «канал BEI» описывает поле записи, а не цвет физического отведения.
''')
    code("from pathlib import Path\nimport sys\nfrom IPython.display import display, HTML\nhere=Path.cwd()\nmodule_dir=next(p for p in [here,here/'scorpion_v2',here/'Colab Notebooks/scorpion_v2'] if (p/'build_georg_tenhour_report.py').exists())\nsys.path.insert(0,str(module_dir.resolve()))\nfrom build_georg_tenhour_report import *\nd=load_run()\nplotly_loaded=False\ndef show(fig):\n    global plotly_loaded\n    fig.update_layout(width=1100)\n    display(HTML('<div style=\"overflow-x:auto;max-width:100%\">'+fig.to_html(full_html=False,include_plotlyjs=not plotly_loaded,config={'displaylogo':False,'responsive':False})+'</div>'))\n    plotly_loaded=True\ndisplay(table(output_table()[['Отведение и запись','Канал BEI','Дыхание','Измерено, Ом']]))")
    md('''**Таблица 1.** Все 24 уровня, использованные в подборе. Номер записи различает повторные сборки и кресты одного номинала. Это медианы экспериментальных интервалов; доверительные интервалы измерительной ошибки не установлены.

Дыхательные окна перенесены из [кандидатной разметки 11.20](11.20_Просмотр_и_разметка_Scorpion_v2.ipynb). Неоднозначные границы сокращены на 2 с; для границ самой записи применяется отступ 5 с. Время и фактически использованный сценарий частоты отсчётов показаны в продолжении таблицы. Частота отсчётов 400 либо 1000 Гц относится к временной шкале файла и отличается от частоты электрического зондирования 100 кГц.
''')
    code("display(table(window_table()))")
    md(r'''**Продолжение таблицы 1.** Границы интервалов и 5-й/95-й процентили медленной компоненты показывают, какие участки вошли в обработку и насколько изменялся уровень внутри окна. Эти процентили являются описанием временной вариабельности, без интерпретации как доверительных интервалов или измерительной погрешности. Медиана для подбора рассчитана по исходному BEI, а процентили — по сохранённой медленной компоненте.

## 2. Физическая и математическая постановка

Расчёт выполнен методом конечных элементов (FEM) с полной электродной моделью (CEM). Геометрия построена по индивидуальной КТ на глубоком вдохе с задержкой дыхания. Включены мягкотканный фон, лёгкие, жир, кости и сердце. Четыре тканевые маски имеют автоматическое происхождение; независимая анатомическая валидация не завершена. Вся маска сердца получает свойства миокарда; отдельная проводимость крови не задаётся. Кости и сердце фиксированы, сопротивления фона, лёгких и жира свободны. Та же геометрия используется на вдохе и выдохе; дыхательная деформация не моделируется.

Для каждого отведения используется собственная токовая пара и полный набор контактов одновременно установленного монтажа. Контакты соседнего отведения остаются в полной электродной модели с нулевым суммарным током. Это заданный сценарий электрического возбуждения. Реальная проводимость входов и фазовая схема прибора отдельно не установлены; произвольная поправка шкалы к измерениям не вводилась.

Контакты — диски номинальным диаметром 5 мм, площадью 19,635 мм²; фактическая контактная площадь при регистрации неизвестна. Руки моделируются цилиндрами по авторскому обхвату 37 см на уровне токовых электродов. Контактные группы соответствуют монтажам 1×1, 2×2 и 1×4. Частота расчётного сценария — 100 кГц. Используется вещественная изотропная проводимость. Номинальный порядок вдоль боковой сборки: I+, U+, U−, I−; внешний токовый размер L, потенциальная база L/2. Крест содержит восемь контактов.

Целевая функция имеет вид

$$F(\mathbf q)=\frac{1}{24}\sum_{i=1}^{24}\left[Z_i^{\mathrm{FEM}}(\mathbf q)-Z_i^{\mathrm{BEI}}\right]^2.$$

где $Z_i^{\mathrm{BEI}}$ — медиана соответствующего измеренного интервала, $Z_i^{\mathrm{FEM}}$ — передаточный импеданс полной электродной модели, Ом; $F$ измеряется в Ом²; RMSE равна квадратному корню из F, Ом. Вектор $\mathbf q$ содержит шесть логарифмов сопротивлений: по три отдельно для вдоха и выдоха, два смещения общего центра в касательной плоскости, мм, и угол ориентации, градусы. Все наблюдения имеют одинаковый вес. Общий центр и угол связывают все размеры и обе оси крестов.

Ни равенство сопротивлений двух дыхательных состояний, ни неизменность сопротивления фона не навязывались. Полученное совпадение трёх значений на вдохе и выдохе относится только к лучшему кандидату ограниченного поиска. Жёсткие границы сопротивлений взяты из действующего литературного профиля при 100 кГц по позднему указанию автора; это диапазоны поиска, без индивидуальной валидации для Георгия. Происхождение профиля описано в [справочнике свойств тканей](../40.08_Литературные_диапазоны_электрических_свойств_тканей.md).
''')
    code("display(table(parameters()))\nref=d['ref'];display(table(pd.DataFrame({'Фиксированный материал':['Кости','Сердце как миокард'],'ρ, Ом·м':[ref['rho_ohm_m'][2],ref['rho_ohm_m'][3]]})))")
    md(f'''В полной электродной модели удельный контактный импеданс задан равным {number(d['reference_plan']['contact_impedance_ohm_m2'],7)} Ом·м². Он фиксирован при поиске и обеих численных проверках. Это допущение расчёта; контактные свойства кожи Георгия отдельно не измерялись и не восстанавливались в данной задаче.
''')
    md('''**Таблицы 2–3.** Начальные и лучшие найденные сопротивления, границы поиска и фиксированные свойства. Неизменившиеся стартовые значения не являются восстановленными индивидуальными параметрами.

## 3. Общий центр и геометрия контактов
''')
    md(f'''Ориентировочная точка автора в координатах LPS: (−204,346; −20,215; −207,500) мм. Это реконструкция по воспоминанию, а не измерение положения. Поиск ограничен окном ±30 мм по двум касательным координатам и ±20° по углу. Проекция исходной точки на поверхность практически совпадает с ней.

Центр лучшего кандидата: ({'; '.join(number(x,7) for x in b['pose']['centre_lps_mm'])}) мм LPS. Угол — 0°, смещения в касательной плоскости — (0; 20) мм. Расстояние от авторской точки по прямой равно {number(distance,5)} мм. Кандидат не достиг границы окна. Проекция четырёх номинальных центров на КТ-поверхность не сохраняет автоматически дуговые расстояния по коже; фактические базы по прямой приведены отдельно.
''')
    code("show(geometry_figure())\ndisplay(table(spacing_table()))\ndisplay(table(contacts()))")
    md('''**Рисунок 1; таблицы 4–5.** В 3D-сцене красный ромб обозначает ориентировочную точку автора, зелёная точка — лучший кандидат; штриховая серая линия соединяет их. Синие линии показывают продольные сборки, оранжевые — поперечные. Кружки соответствуют I, квадраты — U. Линии соединяют центры для просмотра порядка и не изображают провода или геодезическую траекторию. Легенда позволяет независимо скрывать сборки, кожу и четыре тканевых поверхности. Точные 28 координат приведены в таблице. Тканевые поверхности упрощены только для показа; расчётные маски этим не изменяются.

''')
    code("from IPython.display import Image\ndisplay(Image(filename=str(REPORT/'ct_planes_140.local.png')))\ndisplay(table(pd.DataFrame({'Ось':['Продольная','Поперечная'],'Максимальное отклонение контакта от плоскости, мм':read(REPORT/'ct_planes.local.json')['maximum_contact_distance_from_plane_mm']})))")
    md('''**Рисунок 2; таблица 6.** Каждая плоскость проходит через лучший центр и натянута на соответствующую касательную ось и внутреннюю нормаль. Показаны КТ и четыре исходные тканевые маски. I обозначены кружками, U — квадратами; зелёный крест — центр. Положения контактов ортогонально проецируются на эту плоскость для просмотра. Их ненулевое отклонение от плоскости приведено в таблице и исключает интерпретацию меток как точных точек сечения. Реальная кривизна кожи сохранена. Электрическая эквивалентная толщина h здесь не рассчитывается.

## 4. Бюджет и ход оптимизации
''')
    md(f'''После сокращения бюджета использованы шесть стартов: пять ранее рассчитанных положений при первой тройке сопротивлений и вторая тройка в центре автора. Сохранённые пять кандидатов повторно использованы без нового FEM. В бюджете 36 оценок содержатся {36-geometry_failures} допустимых кандидатов и {geometry_failures} геометрический отказ. После поиска выполнено отдельное повторение лучшего кандидата под номером 37.

Метод координатного прямого поиска перенесён из ранней совместной постановки Ника. Он проверяет ограниченные изменения всех девяти параметров. Выполнены два прохода; заданное разрешение не достигнуто, шаги не уменьшены. Начальные шаги логарифмов сопротивлений — 0,18; 0,25; 0,30 для каждого дыхательного состояния; смещения центра — 10 мм, угол — 5°. Нахождение минимума и совместная неопределённость девяти параметров этим запуском не подтверждены.
''')
    code("show(history_figure())\nshow(rho_figure(0))\nshow(rho_figure(1))\nshow(rho_figure(2))")
    md('''**Рисунки 3–6.** На истории RMSE серые пустые кружки обозначают повторно использованные точки, синие — новые вычисления; тёмная сплошная линия показывает лучший уровень среди завершённых точек. Красная звезда — отдельное повторение результата. Вертикальная штриховая линия отделяет старты от координатных проверок. На трёх графиках сопротивлений кружки соответствуют вдоху, крестики — выдоху, тёмная линия — параметру лучшего достигнутого кандидата на вдохе. Соединительные линии показывают порядок поиска, без интерполяции физических состояний. Сопротивления фона и лёгкого проверялись при иных значениях, но в итоговом кандидате остались стартовыми.

Шесть стартов здесь означают шесть исходных оценок, из которых выбирается один лучший кандидат. Они не являются шестью независимо доведёнными до сходимости оптимизациями. На каждом проходе принимается одно лучшее изменение координаты; второй проход прерван по бюджету.

### Сочетание уже рассчитанных дыхательных состояний

При одинаковом положении электроды и модель вдоха не зависят от сопротивлений выдоха, и наоборот. Поэтому допустимо собрать прогноз из уже сохранённого вдоха оценки 25 и выдоха оценки 17: геометрия этих двух оценок в точности совпадает с итоговой. Такой контроль не требует нового решения FEM.
''')
    code("display(table(combination_table()))\ndisplay(table(pd.DataFrame({'Вариант':['Сохранённый итог поиска','Сочетание ранее рассчитанных фаз'],'RMSE, Ом':[np.sqrt(d['execution']['best_mean_square_ohm2']),combined_states()['RMSE_ohm']]})))")
    md(f'''**Таблицы 7–8.** RMSE составляет {number(combo['RMSE_ohm'],6)} Ом, меньше сохранённого результата {number(rmse,6)} Ом. Это прямое свидетельство незавершённости оптимизации при той же геометрии. Сочетание не подменяет исходный итог; независимые сеточные проверки для него не выполнены. Более низкая невязка не делает эти сопротивления принятыми.

## 5. Соответствие расчёта эксперименту
''')
    code("show(fit_figure())\nshow(residual_figure())\ndisplay(table(output_table()))")
    md(f'''**Рисунки 7–8; таблица 9.** На сравнении уровней круги показывают измерение, ромбы и соединительные линии — модель. Синий цвет соответствует вдоху, оранжевый — выдоху; сплошная модельная линия относится к вдоху, штриховая — к выдоху. На диаграмме невязок столбцы показывают модель минус измерение, точечная горизонтальная линия — нулевую невязку. Порядок отведений одинаков в обоих рисунках. Все серии можно отключать независимо.

Поперечное отведение креста 60 мм даёт {number(share,4)}% суммы квадратов невязок. На выдохе измерено 67,05 Ом, модель даёт 152,11 Ом. Продольная ось того же креста также завышена: около 139 Ом вместо 97 Ом. Расхождение нельзя приписать лишь нескольким миллиомам погрешности интегрирования.

Для диагностики при тех же сопротивлениях и координатах RMSE оставшихся 22 строк составляет {number(without,4)} Ом. Это пересчёт показателя по сохранённым невязкам, без исключения креста из подбора и без нового оптимального решения. Все 24 наблюдения сохраняются в основном результате.

## 6. Парный дыхательный контраст

Парный контраст определяется как разность медиан на выдохе и вдохе для одного отведения. Он отделяет изменение между двумя выбранными интервалами от абсолютного уровня, но зависит от корректности дыхательной разметки и сопоставимости состояния. В лучшем кандидате обе тройки сопротивлений совпали, геометрия общая, поэтому расчётный контраст равен нулю.
''')
    code("show(breathing_figure())")
    md(r'''**Рисунок 9.** Синие столбцы — измеренный контраст, оранжевая сплошная линия с ромбами — контраст исходного лучшего кандидата; фиолетовая точечная линия с пустыми ромбами — контраст сочетания сохранённых фаз. Нулевая оранжевая линия является результатом данного ограниченного поиска; она не подтверждает отсутствие дыхательного эффекта или необходимость равенства сопротивлений. Геометрическое изменение грудной клетки здесь не отделено от изменения её электрических свойств.

Для равных весов двух состояний целевая функция точно разделяется:

$$F=\frac{1}{12}\sum_{j=1}^{12}e_{\mathrm{mean},j}^{2}+\frac{1}{48}\sum_{j=1}^{12}e_{\Delta,j}^{2}.$$

где $e_{\mathrm{mean},j}$ — ошибка среднего из уровней вдоха и выдоха одного отведения, $e_{\Delta,j}$ — ошибка контраста выдох минус вдох, Ом. Это арифметическое разложение той же целевой функции, без изменения её весов.
''')
    md(f'''Для исходного кандидата взвешенная доля контраста в F составляет только {number(contrast[0]['weighted_contrast_percent'],4)}%. Ошибка дыхательного контраста по 12 отведениям, выраженная как RMSE, равна {number(contrast[0]['contrast_RMSE_ohm'],4)} Ом. Для сочетания сохранённых фаз эта ошибка возрастает до {number(contrast[1]['contrast_RMSE_ohm'],4)} Ом, хотя общая RMSE снижается до {number(combo['RMSE_ohm'],5)} Ом. Следовательно, улучшение общего согласования не означает улучшения дыхательного контраста. Это существенное ограничение текущего критерия; оба результата следует анализировать отдельно. Новые веса нельзя назначать произвольно без модели ошибок и проверки дыхательных интервалов.

''')
    md(r'''## 7. Интегрирование и пространственная дискретизация

Проверки выполнены при неизменных сопротивлениях, общем центре, ориентации и номинальных центрах всех контактов. Вложенные правила интегрирования определяют долю каждой КТ-маски в тетраэдре; их сравнение оценивает погрешность присвоения свойств внутри данной сетки. Пространственное сгущение меняет саму конечноэлементную аппроксимацию. Эти два источника ошибки проверяются раздельно.

Для интегрирования критерий имеет вид

$$|Z_{\mathrm{fine}}-Z_{\mathrm{coarse}}|\leq 0{,}02+0{,}001|Z_{\mathrm{fine}}|.$$

где $Z_{\mathrm{fine}}$ и $Z_{\mathrm{coarse}}$ — импедансы более плотного и менее плотного правил, Ом. Для пространственного сравнения коэффициент относительного члена равен 0,005. Допуски после получения результата не менялись.
''')
    code("show(precision_figure())\nshow(mesh_figure())")
    md(f'''**Рисунки 10–11.** На первом рисунке цветные линии с точками показывают отношение различия интегрирования к его допуску для трёх сравнений. На втором столбцы показывают то же отношение для пространственного сгущения, отдельно для вдоха и выдоха. Штриховая горизонтальная линия на обоих рисунках — граница 1: значение выше неё означает отказ соответствующего критерия. Численные различия и допускаемые значения доступны при наведении.

Интегрирование 1024/4096 в конце поиска прошло. Дополнительное сравнение 4096/16384 на той же сетке также прошло: максимальная разность — {number(max(v['integration_checks'][0]['difference_ohm']),5)} Ом. После объёмного сгущения до 3 мм интегрирование 4096/16384 прошло вновь; максимальная разность — {number(max(v['mesh_checks'][0]['difference_ohm']),5)} Ом.

Пространственное сгущение изменило Z на 2,13–5,01 Ом. Все 24 наблюдения не выдержали пространственный допуск, наибольшее отношение к нему — {number(v['spatial_comparisons'][0]['maximum_ratio'],4)}. Показатели площади, смещения центра и размаха площадки совпали для всех {len(c)} записей контактов. Это исключает изменение этих трёх показателей как объяснение различия, но не является независимым доказательством полной эквивалентности контактных операторов.

Подготовка сетки 2 мм была остановлена на этапе расчёта материальных долей. Для этого уровня нет завершённого паспорта и прямого FEM-результата; сравнение 3/2 мм не выполнено. Пространственная сходимость остаётся неподтверждённой. Одно отказавшее сравнение не устанавливает окончательную погрешность предельного решения.

## 8. Локальная чувствительность и идентифицируемость

Сохранённый Якобиан содержит производные по шести логарифмам сопротивлений при фиксированном положении. Производная определяет первый порядок изменения Z при малом относительном изменении одного сопротивления. Она не является долей полного импеданса, ткани или объёма чувствительности. Три пространственных параметра в этот Якобиан не входят.
''')
    code("show(jacobian_figure())\nsv=np.array(d['best']['singular_values']);display(table(pd.DataFrame({'Номер сингулярного значения':np.arange(1,7),'Значение, Ом':sv})))")
    md(f'''**Рисунок 12; таблица 10.** Цвет каждой ячейки показывает сохранённую производную ∂Z/∂lnρ на вдохе, Ом. Для лучшего кандидата дыхательные тройки одинаковы. Шесть сингулярных значений получены для общего материального Якобиана 24×6; отношение наибольшего к наименьшему равно {number(max(b['singular_values'])/min(b['singular_values']),5)}. Это локальная обусловленность материальной части при фиксированной геометрии и равных весах. Она не задаёт доверительные интервалы, не учитывает ошибку модели и не доказывает идентифицируемость совместной задачи по девяти параметрам.

''')
    code("a=material_diagnostics();display(table(pd.DataFrame({'Материал':MATERIALS,'Норма столбца чувствительности на вдохе, Ом':a['inhale_column_norms_ohm']})))\ndisplay(table(pd.DataFrame({'Параметр':['lnρ₁ вдох','lnρ₂ вдох','lnρ₃ вдох','lnρ₁ выдох','lnρ₂ выдох','lnρ₃ выдох'],'Производная F по lnρ, Ом²':a['MSE_gradient_logrho']})))")
    md(f'''**Таблицы 11–12.** Нормы столбцов на вдохе составляют 154,39 Ом для фона, 3,51 Ом для лёгкого и 172,65 Ом для жира. Косинус между столбцами фона и жира равен {number(diag['soft_fat_cosine'],5)}: их локальные изменения Z близки по направлению. Производная целевой функции по логарифмам сопротивлений рассчитана как (2/24) Jᵀr, где r — вектор модель минус измерение. Большой ненулевой градиент в итоговой внутренней точке подтверждает отсутствие стационарности.

В локальном линейном приближении увеличение сопротивления лёгкого на 1% даёт норму изменения 12 прогнозов вдоха около {number(diag['lung_one_percent_linear_response_norm_ohm'],4)} Ом. Норма изменения тех же 12 прогнозов при сгущении до 3 мм равна {number(diag['grid_change_inhale_norm_ohm'],4)} Ом. В данном состоянии численная пространственная неопределённость значительно превышает малый отклик лёгкого. Это ограничивает интерпретацию ρ₂; вывод о принципиальной неидентифицируемости по этому сравнению не делается.

## 9. Вычислительные затраты и ограничение времени
''')
    code("display(table(stage_table()))\nplans=[d['reference_plan'],read(JOB/'local_volume_h3p0/plan.local.json')];display(table(pd.DataFrame({'Вариант':['Сетка поиска','Локальное объёмное сгущение 3 мм'],'Узлов':[x['nodes'] for x in plans],'Тетраэдров':[x['tetrahedra'] for x in plans]})))")
    md('''**Таблицы 13–14.** Продолжительность фактически выполненных стадий и размер сеток. Завершение стадии означает получение файлов, а прохождение критерия оценивается отдельно. Локальное сгущение увеличило число тетраэдров примерно с 6,72 до 29,90 млн. Новые элементы пересечённых областей повторно размечались по КТ-маскам. Эта подготовка и сборка операторов являются существенной частью затрат, помимо самого поиска.

Ограничение 10 часов отсчитывалось с момента сокращения бюджета автором, включая подготовку перезапуска. Фоновый процесс остановлен около 05:54 МСК 13 октября. В этом запуске не выполнялись расширение окна до ±60 мм, интегрирование 65536 точками и полное сгущение остального объёма. Незавершённая стадия не считается прошедшей проверкой.

## 10. Независимый разбор постановки с участием Astra

Модель Astra получила исходную постановку, завершённые результаты, журналы, сохранённые версии кода и методику Ника. Задача разбора — отличить установленную ошибку от гипотезы и определить проверки, способные разделить причины расхождения. Новые FEM-расчёты этим разбором не выполнялись. Ниже приведены проверенные основным анализом замечания; основания и локаторы кода сохранены вместе с отчётом.
''')
    code("display(table(audit_table()))")
    md(read(REPORT/'astra_review.local.json')['reader_discussion'])
    md('''## 11. Последовательность дальнейших проверок

Сначала необходимо сверить геометрию и электрическую схему каждого отведения с подтверждённым протоколом: фактические базы после проекции, оси обоих крестов, порядок I и U, роли каналов и сценарий пассивных контактов. Эти проверки выполняются по сохранённым координатам и операторам и не требуют повторения всего поиска. Поперечный крест 60 мм сохраняется в основном наборе как диагностически важное наблюдение.

Затем следует локализовать источник сеточной разности при фиксированном кандидате. Нужны завершённое сравнение 3/2 мм, контроль качества дочерних тетраэдров и отдельное сравнение контактов и присвоения материалов. Однородный контроль на тех же сетках поможет отделить геометрическую аппроксимацию от ошибки представления тканевых границ. Пройденное интегрирование на каждой сетке сохраняется как самостоятельный результат.

После контроля прямой задачи целесообразно проверить более эффективный совместный подбор: внутреннюю оптимизацию шести сопротивлений при заданном положении с использованием уже рассчитываемых производных и внешнее уточнение двух координат и угла. Такой переход требует нового паспорта и сравнения с текущим прямым поиском. Необходимо проверять несколько стартов и профили невязки, раздельно показывать абсолютные уровни и дыхательный контраст. Сокращение числа размеров или исключение неудобного отведения до этих проверок не обосновано.

## 12. Передача результата в методику

Этот запуск подтвердил возможность совместного вычислительного поиска и прохождение контроля интегрирования в найденном положении. Он также установил отказ пространственного допуска и значительное экспериментальное расхождение. Найденные ρ₁, ρ₂, ρ₃ и центр не передаются как принятые входы коррекции ТТРКГ. Оценка УО и ФВ в этом расчёте не выполнялась.

Область применимости, источники неопределённости и допустимая ошибка согласования с измерениями остаются предметом разработки. Ограниченный бюджет не позволяет заключить, что физиологичное решение отсутствует или что совместная задача принципиально неидентифицируема. Сохранённые отрицательные результаты определяют следующие проверки, а не завершают разработку методики.
''')
    export(STEM,cells,out=REPORT)


if __name__=='__main__':
    main()
