"""CT geometry and conditional tissue inversion: executed, code-free reports."""
from pathlib import Path
import argparse
import hashlib
import json

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh
from scipy.ndimage import map_coordinates

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_fem_20261008'
FOLDER=Path(__file__).parent
GEOMETRY='20.20_Индивидуальная_КТ_FEM_модель_Георгия_Scorpion_v2'
INVERSE='33.22_Сопротивления_тканей_в_КТ_FEM_модели_Георгия_Scorpion_v2'
NAMES=['Мягкотканный фон','Лёгкое','Кости','Сердце','Жир']
STATES={'inhale':'Вдох','exhale':'Выдох'}
COLORS=['#9773a5','#4097dd','#dbc790','#e76b6b','#e0ab28']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def number(v,digits=4):
    s=f'{float(v):.{digits}g}'
    if 'e' in s:
        m,e=s.split('e');s=m+'·10'+str(int(e)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return s.replace('.',',').replace('-','−')


def table(frame):
    columns={c:lambda x:number(x,5) for c in frame.select_dtypes(include='floating').columns}
    columns.update({c:lambda x:'Да' if x else 'Нет' for c in frame.select_dtypes(include='bool').columns})
    return frame.style.hide(axis='index').format(columns)


def require_current_geometry(out,allow_superseded=False):
    historical=Path(out).resolve()==OUT.resolve() or (Path(out)/'superseded_geometry.local.json').exists()
    if historical and not allow_superseded:
        raise ValueError('Historical geometry was superseded on 2026-10-09; recompute FEM for the corrected montage')


def verify_inputs(out=OUT,allow_superseded=False):
    require_current_geometry(out,allow_superseded)
    p=read(out/'plan.local.json')
    for entry in p['sources']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed input: '+entry['path'])
    expected=sha(out/'plan.local.json')
    for i in (1,2,3):
        f=read(out/f'forward_pose{i}.local.json')
        if f['identity']['plan_sha256']!=expected:raise ValueError('Stale forward result')
        if not f['quadrature_within_scenario_tolerance']:raise ValueError('Volume integration check failed')
        if f.get('quadrature_rule_pair')!=[1024,4096]:raise ValueError('Finer integration validation missing')
    return p


def prepare_anatomy_assets(out=OUT):
    """Explicit transverse ray profiles; no reduction to effective h."""
    seg=out/'segmentation';ct=nib.load(seg/'static_ct.nii.gz')
    hu=np.asanyarray(ct.dataobj);labels=np.zeros(ct.shape,dtype=np.uint8)
    for name,value in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        mask=nib.load(seg/(name+'.nii.gz'))
        np.testing.assert_allclose(ct.affine,mask.affine,rtol=0,atol=1e-5)
        labels[np.asanyarray(mask.dataobj)>0]=value
    tr=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    contacts=read(out/'contacts.local.json');distances=np.arange(0.,150.25,.5)
    records=[]
    for pose in contacts['poses']:
        xyz=np.array(pose['lateral_centres_mm'])
        for i,start in enumerate(xyz):
            points=start+distances[:,None]*[1.,0.,0.]
            ijk=nib.affines.apply_affine(tr,points)
            valid=np.all((ijk>=-.5)&(ijk<np.asarray(ct.shape)-.5),axis=1)
            lab=map_coordinates(labels,ijk.T,order=0,mode='constant',cval=0)
            attenuation=map_coordinates(hu,ijk.T,order=0,mode='constant',cval=-1024)
            lung=np.flatnonzero((lab==1)&valid)
            first=int(lung[0]) if len(lung) else None
            wall=lab[:first] if first is not None else lab
            row=dict(pose=pose['id'],z_mm=pose['z_mm'],L_mm=contacts['sizes_mm'][i//4],
                     contact=['A','M','N','B'][i%4],start_lps_mm=start.tolist(),
                     direction_lps=[1,0,0],sampling_step_mm=.5,
                     first_lung_distance_mm=float(distances[first]) if first is not None else None,
                     tissue_lengths_before_lung_mm=[float(np.count_nonzero(wall==j)*.5) for j in range(5)],
                     outside_or_air_before_lung_mm=float(np.count_nonzero((~valid|(attenuation< -700))[:first]& (lab[:first]!=1))*.5),
                     labels_to_150mm=lab.astype(int).tolist())
            records.append(row)
    write(out/'transverse_profiles.local.json',dict(records=records,
        effective_h_estimated=False,profile_is_surface_normal=False,
        ct_sha256=sha(seg/'static_ct.nii.gz'),code_sha256=sha(__file__),
        rule='positive LPS X ray from each scenario contact to first segmented lung; not an electrical equivalent'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    x=np.linspace(-225,225,512);y=np.linspace(-225,225,512);xx,yy=np.meshgrid(x,y)
    fig,axes=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
    for ax,pose in zip(axes,contacts['poses']):
        xyz=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,pose['z_mm'])]
        ijk=nib.affines.apply_affine(tr,xyz)
        image=map_coordinates(hu,ijk.T,order=0,mode='constant',cval=-1024).reshape(xx.shape)
        mask=map_coordinates(labels,ijk.T,order=0,mode='constant',cval=0).reshape(xx.shape)
        ax.imshow(image,extent=[-225,225,-225,225],origin='lower',cmap='gray',vmin=-700,vmax=300)
        ax.imshow(np.ma.masked_where(mask==0,mask),extent=[-225,225,-225,225],origin='lower',
                  cmap=ListedColormap(COLORS[1:]),vmin=1,vmax=4,alpha=.55,interpolation='nearest')
        xy=np.array(pose['lateral_centres_mm'])[[16,17,18,19],:2]
        ax.plot(xy[:,0],xy[:,1],'o',color='#fff',markeredgecolor='#111',markersize=5)
        ax.set(title=f"Сценарий {int(pose['id'][1:])+1}; z = {number(pose['z_mm'])} мм",
               xlabel='x LPS, мм',ylabel='y LPS, мм',xlim=(-225,225),ylim=(-170,170))
    fig.legend(handles=[Patch(color=COLORS[j],label=NAMES[j]) for j in (1,2,3,4)],
               loc='lower center',ncol=4,bbox_to_anchor=(.5,-.03))
    fig.savefig(out/'ct_masks_and_poses.png',dpi=170,bbox_inches='tight');plt.close(fig)
    return records


def scene(out=OUT):
    surface=trimesh.load(out/'body_voxel2mm_hxt_checked_boundary.stl',force='mesh')
    v=surface.vertices;f=surface.faces
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
                         opacity=.22,color='#bbb',name='Расчётная поверхность',hoverinfo='skip'))
    c=read(out/'contacts.local.json')
    for i,pose in enumerate(c['poses'],1):
        xyz=np.array(pose['lateral_centres_mm'])
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
                                 name=f'Боковой сценарий {i}',marker=dict(size=3),
                                 hovertemplate='x=%{x:.1f}; y=%{y:.1f}; z=%{z:.1f} мм<extra></extra>'))
    for i,xyz in enumerate(c['arm_groups_mm']):
        xyz=np.array(xyz)
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
            marker=dict(size=5,color='#333'),name=['Токовые контакты','Измерительные контакты'][i in (1,2)],
            showlegend=i<2))
    fig.update_layout(template='plotly_white',height=650,title='Цилиндры рук и геометрические сценарии боковых сборок',
        scene=dict(aspectmode='data',xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм',
                   uirevision='georg-scorpion-preserve-camera'),legend=dict(orientation='h',y=-.05))
    return fig


def profiles(out=OUT):
    records=read(out/'transverse_profiles.local.json')['records'];fig=go.Figure()
    chosen=[r for r in records if r['L_mm']==140]
    labels=[f"Сценарий {int(r['pose'][1:])+1}, {r['contact']}" for r in chosen]
    for j in (4,0,2,3):
        fig.add_trace(go.Bar(y=labels,x=[r['tissue_lengths_before_lung_mm'][j] for r in chosen],
            orientation='h',marker_color=COLORS[j],name=NAMES[j]))
    fig.update_layout(template='plotly_white',height=580,barmode='stack',title='Тканевый состав пути до первого лёгочного вокселя',
        xaxis_title='Суммарная длина участков вдоль +x LPS, мм',yaxis=dict(autorange='reversed'),
        legend=dict(orientation='h',y=-.15),margin=dict(l=140,r=40,t=70,b=70))
    return fig


def forward_plot(out=OUT):
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Сценарий: боковой канал 1','Сценарий: боковой канал 2'))
    obs=pd.read_csv(out/'observations.csv');c=read(out/'contacts.local.json')
    for channel in (1,2):
        for state,color in [('inhale','#333'),('exhale','#777')]:
            o=obs.query('channel==@channel and state==@state').sort_values('L_mm')
            fig.add_trace(go.Scatter(x=o.L_mm,y=o.Z_ohm,mode='markers',name=f'Наблюдения: {STATES[state].lower()}',
                                    showlegend=channel==1,marker=dict(color=color)),row=1,col=channel)
        for pose in (1,2,3):
            f=read(out/f'forward_pose{pose}.local.json')
            fig.add_trace(go.Scatter(x=c['sizes_mm'],y=f['Z_ohm'],mode='lines+markers',name=f'КТ-сценарий {pose}',
                                    showlegend=channel==1),row=1,col=channel)
    fig.update_layout(template='plotly_white',title='Прямой расчёт при литературном опорном состоянии',height=520,
                      legend=dict(orientation='h',y=-.2),margin=dict(t=90,b=100))
    fig.update_xaxes(title_text='Токовая база L, мм');fig.update_yaxes(title_text='Z, Ом')
    return fig


def sensitivity_plot(out=OUT):
    fig=make_subplots(rows=1,cols=3,subplot_titles=tuple(f'КТ-сценарий {p}' for p in (1,2,3)))
    sizes=read(out/'contacts.local.json')['sizes_mm']
    for pose in (1,2,3):
        f=read(out/f'forward_pose{pose}.local.json');J=np.array(f['J_logrho_ohm'])
        for j,name,color in [(0,'Фон',COLORS[0]),(1,'Лёгкое',COLORS[1]),(2,'Жир',COLORS[4])]:
            fig.add_trace(go.Scatter(x=sizes,y=J[:,j]/np.array(f['Z_ohm']),mode='lines+markers',
                name=name,legendgroup=name,showlegend=pose==1,line=dict(color=color)),row=1,col=pose)
    fig.update_layout(template='plotly_white',title='Относительная чувствительность к трём свободным сопротивлениям',height=500,
                      legend=dict(orientation='h',y=-.2),margin=dict(t=90,b=90))
    fig.update_xaxes(title_text='L, мм');fig.update_yaxes(title_text='∂ln Z / ∂ln ρ')
    return fig


def cases(out=OUT,allow_superseded=False):
    require_current_geometry(out,allow_superseded)
    p=read(out/'plan.local.json');rows=[]
    validation=read(out/'integration_validation.local.json')
    if validation['identity']['plan_sha256']!=sha(out/'plan.local.json'):raise ValueError('Stale validation')
    checks={(c['pose'],c['channel'],c['state']):c for c in validation['cases']}
    for pose in (1,2,3):
        for channel in (1,2):
            for state in ('inhale','exhale'):
                c=read(out/f'inverse_pose{pose}_ch{channel}_{state}.local.json')
                if c['identity']['plan_sha256']!=sha(out/'plan.local.json'):raise ValueError('Stale inverse result')
                if c['identity']['worker_sha256']!=sha(ROOT/'MATLAB_TRKG4_real_subjects/src/run_scorpion_georg_fem.m'):
                    raise ValueError('Inverse worker changed')
                check=checks[pose,channel,state]
                if check['case_sha256']!=sha(out/f'inverse_pose{pose}_ch{channel}_{state}.local.json'):
                    raise ValueError('Case changed after integration check')
                c['fine_integration_validation']=check
                rows.append(c)
    return rows


def inverse_table(rows):
    return pd.DataFrame([{'Сценарий КТ':c['pose'],'Канал':c['channel'],'Дыхание':STATES[c['state']],
        'ρ₁, Ом·м':c['rho_ohm_m'][0],'ρ₂, Ом·м':c['rho_ohm_m'][1],'ρж, Ом·м':c['rho_ohm_m'][4],
        'Невязка, %':c['relative_residual_percent'],'κ(J)':c['condition_log_parameters'],
        'Граничные параметры':', '.join(n for n,b in zip(['ρ₁','ρ₂','ρж'],c['boundary_hits']) if b) or 'Нет',
        'Оптимизация завершена':c['exitflag']>0,'Проверка 256→1024':c['quadrature_within_scenario_tolerance'],
        'Проверка 1024→4096':c['fine_integration_validation']['passed']}
        for c in rows])


def fitted_plot(rows,out=OUT):
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Канал 1: вдох','Канал 2: вдох','Канал 1: выдох','Канал 2: выдох'))
    sizes=read(out/'contacts.local.json')['sizes_mm'];colours=['#277da8','#c17b24','#55a35b']
    for case in rows:
        row=1 if case['state']=='inhale' else 2;col=case['channel'];pose=case['pose']
        if pose==1:
            fig.add_trace(go.Scatter(x=sizes,y=case['Z_observed_ohm'],mode='markers',name='Наблюдения',
                showlegend=row==col==1,marker=dict(color='#222',size=8)),row=row,col=col)
        fig.add_trace(go.Scatter(x=sizes,y=case['Z_predicted_ohm'],mode='lines+markers',name=f'КТ-сценарий {pose}',
            legendgroup=str(pose),showlegend=row==col==1,line=dict(color=colours[pose-1])),row=row,col=col)
    fig.update_layout(template='plotly_white',title='Пять размеров: наблюдения и условная FEM-подгонка',height=750,
                      legend=dict(orientation='h',y=-.14),margin=dict(t=90,b=100))
    fig.update_xaxes(title_text='L, мм');fig.update_yaxes(title_text='Z, Ом')
    return fig


def parameter_plot(rows):
    fig=make_subplots(rows=3,cols=2,subplot_titles=('Канал 1','Канал 2'),vertical_spacing=.10)
    for channel in (1,2):
        for state,color in [('inhale','#277da8'),('exhale','#c17b24')]:
            cc=sorted([c for c in rows if c['channel']==channel and c['state']==state],key=lambda c:c['pose'])
            for j,index in enumerate([0,1,4]):
                fig.add_trace(go.Scatter(x=[c['pose'] for c in cc],y=[c['rho_ohm_m'][index] for c in cc],
                    mode='lines+markers',name=STATES[state],showlegend=channel==1 and j==0,
                    legendgroup=state,line=dict(color=color),
                    hovertemplate='Сценарий %{x}: %{y:.5g} Ом·м<extra>%{fullData.name}</extra>'),row=j+1,col=channel)
    for row,name in enumerate(['ρ₁, Ом·м','ρ₂, Ом·м','ρж, Ом·м'],1):
        fig.update_yaxes(title_text=name,type='log',range=[-1,2],
                         tickvals=[.1,1,10,100],ticktext=['0,1','1','10','100'],row=row)
    fig.update_xaxes(title_text='Номер геометрического сценария',tickvals=[1,2,3])
    fig.update_layout(template='plotly_white',title='Зависимость оценок сопротивлений от места монтажа',height=900,
                      legend=dict(orientation='h',y=-.14),margin=dict(t=90,b=100))
    return fig


def weak_family_plot(out=OUT):
    r=read(out/'weak_direction_family.local.json');a=np.array(r['family'])
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'xy'}]],
        subplot_titles=('Семейство сопротивлений','Невязка вдоль слабого направления'))
    mask=(a[:,4]<=10)&(a[:,6]==1)
    fig.add_trace(go.Scatter3d(x=a[:,0],y=a[:,1],z=a[:,2],mode='lines',line=dict(color='#999'),
                              name='Проверенное семейство'),row=1,col=1)
    fig.add_trace(go.Scatter3d(x=a[mask,0],y=a[mask,1],z=a[mask,2],mode='markers',
        marker=dict(size=5,color=a[mask,4],colorscale='Viridis'),name='Прирост RSS ≤10%; интегрирование прошло'),row=1,col=1)
    fig.add_trace(go.Scatter(x=a[:,1],y=1+a[:,4]/100,mode='lines+markers',name='RSS / RSS₀',showlegend=False),row=1,col=2)
    for threshold,dash,label in [(1.05,'dash','Допуск 5%'),(1.10,'dot','Допуск 10%')]:
        fig.add_trace(go.Scatter(x=[a[:,1].min(),a[:,1].max()],y=[threshold,threshold],mode='lines',
            line=dict(color='#888',dash=dash),name=label),row=1,col=2)
    fig.update_layout(template='plotly_white',height=630,title='КТ-сценарий 2, канал 1, вдох: прямые расчёты семейства',
        scene=dict(xaxis=dict(title='ρ₁, Ом·м',type='log'),yaxis=dict(title='ρ₂, Ом·м',type='log'),
                   zaxis=dict(title='ρж, Ом·м',type='log'),uirevision='preserve-weak-family-camera'),
        legend=dict(orientation='h',y=-.14),margin=dict(t=110,b=90))
    fig.update_xaxes(title_text='ρ₂, Ом·м',type='log',row=1,col=2)
    fig.update_yaxes(title_text='RSS / RSS₀',row=1,col=2)
    return fig


def export(stem,cells,out=OUT):
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')))
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(FOLDER)}}).execute()
    nbf.write(nb,FOLDER/(stem+'.ipynb'))
    exp=HTMLExporter();exp.exclude_input=True;exp.exclude_input_prompt=True;exp.exclude_output_prompt=True
    html,_=exp.from_notebook_node(nb);(FOLDER/(stem+'.html')).write_text(html,encoding='utf-8')
    (FOLDER/(stem+'.md')).write_text('\n\n'.join(c.source for c in cells if c.cell_type=='markdown')+'\n',encoding='utf-8')
    reader_out=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_geometry_20261009' if Path(out).resolve()==OUT.resolve() else out
    write(reader_out/(stem[:5]+'_reader_artifacts.local.json'),dict(files=[dict(name=stem+ext,sha256=sha(FOLDER/(stem+ext)))
        for ext in ('.ipynb','.html','.md')],executed_cells=sum(c.cell_type=='code' for c in nb.cells),code_excluded=True))


def new_cells():
    import nbformat as nbf
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    return cells,md,code


INITIAL="""from pathlib import Path
import sys
from IPython.display import display, HTML, Image
here=Path.cwd()
module_dir=next(p for p in [here,here/'scorpion_v2',here/'Colab Notebooks/scorpion_v2'] if (p/'build_georg_fem_reports.py').exists())
sys.path.insert(0,str(module_dir.resolve()))
from build_georg_fem_reports import *
plotly_loaded=False
def show(fig):
    global plotly_loaded
    display(HTML(fig.to_html(full_html=False,include_plotlyjs=not plotly_loaded,config={'displaylogo':False})))
    plotly_loaded=True
plan=verify_inputs()
"""


def initialization(out,allow_superseded=False):
    relative=Path(out).resolve().relative_to(ROOT).as_posix()
    return INITIAL.replace('plan=verify_inputs()',f'OUT=ROOT/{relative!r}\nplan=verify_inputs(OUT,allow_superseded={allow_superseded!r})')


def build_geometry(out=OUT):
    if Path(out).resolve()==OUT.resolve() or (Path(out)/'superseded_geometry.local.json').exists():
        from build_georg_geometry_revision_report import build
        return build()
    plan=verify_inputs(out);records=prepare_anatomy_assets(out)
    prep=read(out/'preparation_qc.local.json');meshqc=read(out/'body_voxel2mm_hxt_qc.local.json')
    change=read(out/'voxel_geometry_comparison.local.json')
    cells,md,code=new_cells()
    md('# 20.20. Индивидуальная КТ/FEM-модель Георгия для Scorpion v2\n\n'
       '**Статус: разработка методики; геометрические и электрические сценарии.** '
       'Положение бокового монтажа не измерено, автоматические маски не прошли независимую анатомическую проверку.\n\n'
       '## 1. Назначение и последовательность\n\n'
       'В двуслойном расчёте 33.21 совместная оценка сопротивлений и эквивалентной толщины оказалась неустойчивой. '
       'Настоящий этап строит индивидуальную анатомическую модель методом конечных элементов (FEM), чтобы исследовать причины расхождения '
       'и определить чувствительности к отдельным тканям. Последовательность: полный том КТ → тканевые маски '
       '→ наружная поверхность и цилиндры рук → геометрические сценарии контактов → объёмная сетка '
       '→ распределение материалов → конечная электродная модель → прямой расчёт и проверка производных. '
       'Условная обратная задача описана в 33.22.\n\n'
       '## 2. КТ и сегментация\n\n'
       'Использована полная статическая серия 301, содержащая 280 срезов. Размер тома — 512 × 512 × 280 вокселей; '
       'шаг — 0,879 × 0,879 × 1,25 мм. По подтверждению автора КТ снята на глубоком вдохе с задержкой дыхания. '
       'Она относится к другой сессии, чем Scorpion. Кардиальные реконструкции с меньшим полем зрения '
       'не использованы для полной карты жира грудной стенки.\n\n'
       'Маски сердца, лёгких, костей и жира построены автоматически в TotalSegmentator 2.18.0. '
       'Результаты автоматической сегментации не заменяют авторскую ручную сегментацию. '
       'При пересечениях принят порядок: лёгкое, кости, сердце, жир, оставшийся мягкотканный фон. '
       'Все маски проверены на совпадение формы и аффинного преобразования с исходным томом. '
       'При переносе координат из NIfTI RAS в DICOM LPS меняются знаки x и y.\n\n'
       '**Рисунок 1.** Три поперечных сечения с четырьмя тканевыми масками; белыми маркерами показана сборка 140 мм. '
       'Показанные уровни являются заданными геометрическими сценариями.')
    code(initialization(out)+'\ndisplay(Image(filename=str(OUT/"ct_masks_and_poses.png")))')
    code("inventory=pd.read_csv(OUT/'material_inventory.csv')\ninventory=inventory.query('quadrature_points==1024').copy()\ninventory['material']=NAMES\ndisplay(table(inventory[['material','volume_ml']].rename(columns={'material':'Материал','volume_ml':'Объём в FEM, мл'})))")
    md('## 3. Наружная геометрия и руки\n\n'
       'Наружная поверхность туловища основана на имеющейся модели Георгия из авторской сегментации Inobitec. '
       'Её ранее выполненное заполнение и обработка с замыканием радиусом 10 мм сохраняются как источник '
       'геометрической неопределённости. Новые руки добавлены существующим алгоритмом создания моделей. '
       'По уточнению автора обхват обеих рук на уровне токовых электродов равен 37 см; руки принимаются цилиндрами. '
       'Соответствующий радиус — 58,89 мм. Дата измерения и погрешность обхвата неизвестны; '
       'его применимость к дате КТ отдельно не установлена.\n\n'
       'Для получения пригодной объёмной сетки поверхность повторно представлена на решётке с шагом 2 мм '
       'без дополнительного морфологического замыкания. '
       f'Изменение объёма относительно модели с новыми цилиндрами составляет {number(change["relative_volume_change"]*100)}%. '
       f'Максимальные расстояния в двух направлениях для вершин и центров граней составляют '
       f'{number(change["original_to_voxel"]["maximum_mm"])} и {number(change["voxel_to_original"]["maximum_mm"])} мм. '
       'Это выборочные оценки расстояния до поверхности, а не сертифицированное расстояние Хаусдорфа.\n\n'
       'Из сетки удалён один машинно вырожденный тетраэдр. После удаления проверены замкнутость, отсутствие '
       'самопересечений границы, совпадение внешних граней с границей тетраэдров и сохранение объёма. '
       f'Изменение триангуляции границы относительно воксельной поверхности достигает {number(meshqc["max_boundary_distance_mm"])} мм. '
       'Эта дополнительная аппроксимация явно включена в геометрический сценарий.\n\n'
       '**Рисунок 2.** Модель рук и контактов в координатах LPS. Показаны три альтернативных положения бокового монтажа; '
       'в отдельном расчёте присутствует одно положение и один размер сборки.')
    code('show(scene(OUT))')
    md('## 4. Монтаж и тканевый профиль\n\n'
       'Для правого бока заданы уровни z = −180, −210 и −240 мм при y = 0. '
       'Правая сторона соответствует отрицательному x в LPS. Для каждого сценария все размеры сохраняют один центр. '
       'Горизонтальная ось направлена вдоль y; номинальные положения A–M–N–B составляют −L/2, −L/4, L/4, L/2. '
       'Контакты проецируются на поверхность лучами вдоль x. Эти координаты не считаются восстановленной установкой.\n\n'
       'Для каждой контактной позиции исследован путь вдоль положительного x до первого вокселя лёгкого '
       'с шагом 0,5 мм. Это поперечный пространственный профиль, который может включать жир, фон и ребро; '
       'он не совпадает с нормалью к поверхности и не определяет электрическую эквивалентную h. '
       'Воздух по КТ и выход за поле зрения учитываются отдельно в локальном паспорте. '
       'Правило сведения таких профилей к эффективной толщине остаётся нерешённым.\n\n'
       '**Рисунок 3.** Длины тканевых участков до лёгкого для четырёх контактов сборки 140 мм. '
       'Столбцы показывают суммарную длину каждого материала; порядок слоёв в них не воспроизводится. '
       'Полные последовательности для всех пяти размеров сохранены отдельно.')
    code('show(profiles(OUT))')
    maximum_air=max(r['outside_or_air_before_lung_mm'] for r in records)
    md(f'Максимальная длина участка с воздухом по КТ или вне тома до лёгкого составляет {number(maximum_air)} мм. '
       'Такие участки указывают на возможное различие обработанной наружной поверхности и КТ; '
       'их влияние на электрический расчёт ещё не выделено.\n\n'
       'На руках измерительные контакты расположены на 20 мм дистальнее среза модели плеча, '
       'токовые — ещё на 100 мм дальше. В конфигурации 1×4 четыре измерительных контакта через 90° '
       'объединены в один электрический узел; токовый контакт на каждой руке один. '
       'Поворот вокруг цилиндра принят равным нулю как сценарий. В каждой модели присутствуют '
       '14 физических контактов и 8 электрических узлов: монтаж ТТРКГ 1×4 и одна боковая сборка. '
       'Остальные размеры в тот же расчёт не включены.\n\n'
       'Все контакты представлены круглыми площадками номинального диаметра 5 мм и площади 19,635 мм². '
       'Это стандарт проекта для новой расчётной постановки; фактическая проводящая площадь Scorpion не установлена. '
       'Принята полная электродная модель CEM с эквипотенциальными группами и контактным импедансом '
       '1,592·10⁻⁴ Ом·м² из существующего расчётного сценария проекта. Это значение не является измеренной '
       'характеристикой контакта нового прибора. Источники включаются поочерёдно; остальные контакты пассивны. '
       'Реальный порядок возбуждения каналов прибором не подтверждён.\n\n'
       'Квазистатическая прямая задача определяется уравнениями\n\n'
       '$$\\nabla\\!\\cdot(\\sigma\\nabla u)=0,\\qquad '
       'u+z_e\\sigma\\frac{\\partial u}{\\partial n}=U_e\\quad\\text{на }\\Gamma_e,\\qquad '
       '\\int_{\\Gamma_e}\\sigma\\frac{\\partial u}{\\partial n}\\,dS=I_e,$$\n\n'
       'где σ — проводимость материала, См/м; u — потенциал в ткани, В; '
       'n — внешняя нормаль; $\\Gamma_e$ — совокупность контактных площадок электрической группы e; '
       '$U_e$ — общий потенциал группы, В; $z_e$ — удельный контактный импеданс, Ом·м²; '
       '$I_e$ — заданный суммарный входящий ток группы, А; dS — элемент площади поверхности, м². '
       'На остальной границе ток равен нулю; сумма $I_e$ по группам равна нулю. '
       'Группа измерительных контактов имеет нулевой суммарный ток, но токи отдельных площадок '
       'могут перераспределяться внутри группы. Это свойство принятой CEM-постановки.\n\n'
       'В боковом расчёте ток подаётся между A и B, измеряется разность потенциалов M и N. '
       'За счёт линейности нормирующий ток принят равным 1 А; импеданс получается как '
       'отношение разности потенциалов к этому току. Монтаж на руках остаётся пассивным '
       'в том же решении. Потенциал одного узла фиксируется для устранения произвольной '
       'аддитивной константы. Координаты перед сборкой матрицы переведены из миллиметров в метры.\n\n'
       'Проводимость принята вещественной и скалярной: анизотропия и реактивная составляющая '
       'в этом проходе не представлены. Частота 100 кГц определяет выбор опорных свойств тканей.\n\n'
       '## 5. Материалы и численный контроль\n\n'
       f'Сетка содержит {format(plan["nodes"],",").replace(","," ")} узлов и '
       f'{format(plan["tetrahedra"],",").replace(","," ")} тетраэдров. '+
       'У контактов выполнено согласованное сгущение до ребра не более 1 мм. '
       'В каждом тетраэдре учитываются доли пяти материалов, а проводимость усредняется с этими долями '
       'для линейных конечных элементов. Доли определены вложенными выборками Соболя из 256 и 1024 точек. '
       'Дополнительная независимая реализация с 4096 точками точно воспроизвела первые 1024 точки '
       'и использована для более строгой проверки интегрирования. '
       'Материальные границы представлены внутри элементов; сходимость по размеру всей объёмной сетки не установлена.\n\n'
       'Элементная проводимость рассчитывается по формуле\n\n'
       '$$\\sigma_T=\\sum_{j=1}^{5}\\frac{f_{T,j}}{\\rho_j},\\qquad\\sum_{j=1}^{5}f_{T,j}=1,$$\n\n'
       'где T — тетраэдр; j — номер материала; $f_{T,j}$ — его объёмная доля внутри элемента; '
       '$\\rho_j$ — удельное сопротивление материала, Ом·м; $\\sigma_T$ — проводимость, используемая в элементной матрице, См/м. '
       'Усредняется проводимость, а не сопротивление. Для линейного элемента градиент потенциала постоянен; '
       'поэтому доли задают интеграл проводимости в матрице жёсткости. Это численная аппроксимация '
       'неоднородной области, точность которой требует проверки по сетке.\n\n'
       f'Максимальное отличие фактической площади контакта от номинальной равно {number(prep["max_area_relative_error"]*100)}%; '
       f'смещение центра площадки — {number(prep["max_centroid_offset_mm"])} мм. '
       'Эти геометрические проверки не подтверждают точность электродных координат эксперимента. '
       f'Объём вне поля КТ — {number(plan["geometry"]["outside_ct_fem_volume_ml"])} мл; '
       'он отнесён к мягкотканному фону. В основном это искусственные продолжения рук, '
       'тканевый состав которых неизвестен.\n\n'
       'Опорные сопротивления взяты из частотных значений Gabriel в общем реестре проекта на 100 кГц. '
       'Для фона выбран параметр мышцы, для лёгкого — состояние наполнения воздухом; '
       'весь объём сердца моделируется как миокард, все кости — как кортикальная кость. '
       'Последние два назначения являются допущениями: кровь в полостях сердца и губчатое вещество костей '
       'в этом первом проходе не выделены. Литературные параметры задают прямое опорное состояние '
       'и старт обратного поиска; они не являются индивидуальными измерениями.\n\n'
       '**Рисунок 4.** Прямой расчёт на индивидуальной геометрии при опорных сопротивлениях. '
       'Наблюдения двух каналов сохраняются как общие для серии гипотезы назначения бокового канала.')
    code("ref=read(OUT/'reference_materials.local.json')\ndisplay(table(pd.DataFrame({'Материал':NAMES,'ρ, Ом·м':[ref['soft'],ref['lung_inflated'],ref['bone'],ref['heart'],ref['fat']]})))\nshow(forward_plot(OUT))")
    md('## 6. Чувствительность и переход к обратной задаче\n\n'
       'В обратном расчёте свободны сопротивления фона ρ₁, лёгкого ρ₂ и жира ρж. '
       'Сопротивления сердца и костей пока фиксированы. На рисунке 5 показана относительная производная '
       'импеданса по каждому свободному сопротивлению при опорном состоянии. '
       'Она характеризует реакцию модели на малое изменение материала и не является долей физиологического '
       'пульсового источника. Малый или близкий к другим вектор чувствительности ограничивает разделимость параметров.\n\n'
       '**Рисунок 5.** Относительные чувствительности к фону, лёгкому и жиру для трёх мест монтажа.')
    code('show(sensitivity_plot(OUT))')
    fwd=[read(out/f'forward_pose{p}.local.json') for p in (1,2,3)]
    md('Прямые решения проверены по невязке системы и взаимности источника и измерения. '
       f'Максимальная относительная ошибка сопряжённой производной при сравнении с центральными разностями '
       f'равна {number(max(f["jacobian_finite_difference_relative_error"] for f in fwd))}. '
       f'Максимальное отличие импеданса при 1024 и 4096 точках составляет '
       f'{number(max(f["quadrature_max_error_ohm"] for f in fwd))} Ом. '
       'Для этого сценария заранее принят численный критерий 0,02 Ом + 0,1% от |Z|; '
       'он выполнен в проверенных опорных состояниях. Этот критерий не задаёт экспериментальную погрешность.\n\n'
       'Предварительное сравнение 256 и 1024 точек не прошло критерий для части состояний. '
       'Оно сохранено отдельно; результат проверки не был повышен до успешного изменением допуска. '
       'Выполнено более подробное интегрирование. Весь прямой и обратный расчёт использует 1024 точки, '
       'а 4096 точек служат независимым численным сопоставлением.\n\n'
       'Индивидуальная геометрия, конечные контакты и тканевые маски позволяют перейти к условному поиску '
       'трёх сопротивлений. Их физическая интерпретация требует проверки влияния положения, площади, '
       'контактного импеданса, состава сердца и костей и ошибок сегментации. '
       'В 33.22 эти ограничения сохраняются; автоматическая передача найденных чисел в коррекцию ТТРКГ не выполняется. '
       'УО и ФВ этим этапом не рассчитываются.\n\n'
       '## 7. Воспроизводимость\n\n'
       'Вычислительные владельцы — scorpion_georg_fem_prepare.py и run_scorpion_georg_fem.m. '
       'Паспорт фиксирует исходные хэши, сетку, маски, контактные группы, материалы и проверки. '
       'Настоящий ноутбук сохраняет рисунки и результаты; его HTML-экспорт исключает исходный код. '
       'Исходные КТ и STL остаются в локальном хранилище. '
       'Отдельные исследования увеличения токовой и измерительной площади, восстановления креста на КТ '
       'и сопоставления ТЕПС с записями должны использовать эту же индивидуальную модель после проверки параметров.')
    export(GEOMETRY,cells,out)


def build_inverse(out=OUT):
    rows=cases(out,allow_superseded=True);cells,md,code=new_cells();residuals=[c['relative_residual_percent'] for c in rows]
    md('# 33.22. Удельные сопротивления тканей в индивидуальной КТ/FEM-модели Георгия\n\n'
       '**Статус: исторический расчёт геометрии от 8 октября 2026 года.** '
       '9 октября автор уточнил монтаж: боковые сборки направлены вдоль тела, токовые контакты '
       'находятся сверху, а соединение цилиндров рук должно охватывать полное сечение. '
       'Старая геометрия этим условиям не соответствовала. Все таблицы и рисунки этого ноутбука '
       'сохраняют прежние численные результаты; они не относятся к исправленной геометрии 20.20 '
       'и исключены из дальнейшей коррекции ТТРКГ. Новая сетка, полный поиск положения и инверсия '
       'ещё не выполнены. Ни место монтажа, ни точность тканевых параметров '
       'не подтверждены независимыми данными.\n\n'
       '## 1. Что проверяется и почему\n\n'
       'Этап 33.21 показал неустойчивость совместной двуслойной оценки ρ₁, ρ₂ и h. '
       'Здесь плоская модель заменена прежней КТ/FEM-моделью редакции 20.20 от 8 октября, содержащей сердце, '
       'лёгкие, кости, жир и оставшийся мягкотканный фон. Цель — найти условные сопротивления трёх '
       'свободных материалов и проверить, снимает ли анатомическая модель прежнюю неоднозначность. '
       'Конечноэлементное решение само по себе не подтверждает идентифицируемость.\n\n'
       'Последовательность: пять уровней общего BEI → проверенная прямая CEM-модель → ограниченная '
       'минимизация невязки → повторение из двух стартов → сравнение геометрических сценариев '
       '→ локальная чувствительность и численная проверка → решение о допустимости переноса.\n\n'
       '## 2. Входы и допущения\n\n'
       'Использованы те же медианы BEI в кандидатных задержках, что в 33.21; сигналы не масштабируются '
       'для согласования с моделью. Разметка 11.20 остаётся предварительной. '
       'Все пять размеров 60–140 мм сохраняются, исключение отдельных записей здесь не выполняется. '
       'Два назначения бокового канала применяются ко всему ряду. '
       'Сопоставимость последовательных записей является допущением; полная экспериментальная ошибка неизвестна.\n\n'
       'Геометрия, маски, цилиндрические руки с обхватом 37 см, контакты 5 мм, группировка 1×4 '
       'и контактный импеданс заданы в прежней редакции 20.20. Боковые контакты были направлены '
       'по y LPS, токовые — вперёд; соединение рук имело недостаточное сечение. '
       'Расчёт повторён для трёх предполагаемых положений '
       'общего центра: z = −180, −210 и −240 мм. '
       'Это проверка чувствительности к трём положениям, а не полный поиск установки на поверхности. '
       'Для вдоха и выдоха используется одна КТ-геометрия глубокого вдоха. '
       'Найденная разность сопротивлений поэтому может включать неучтённое изменение геометрии.\n\n'
       '## 3. Параметры и критерий\n\n'
       'Свободны ρ₁ фона, ρ₂ лёгкого и ρж жира, каждое в технических пределах 0,1–100 Ом·м. '
       'Литературные диапазоны не ограничивают поиск; автоматического расширения границ нет. '
       'ρ₁ после выделения жира не тождественно эффективному сопротивлению грудной стенки в 33.21. '
       'h здесь не оптимизируется: неоднородная анатомия представлена пространственными масками. '
       'Электрически эквивалентная h из этой модели пока не определена.\n\n'
       'Сопротивления сердца и костей фиксированы по опорным значениям прежней редакции 20.20: '
       f'{number(rows[0]["rho_ohm_m"][3],7)} и {number(rows[0]["rho_ohm_m"][2],7)} Ом·м соответственно. '
       f'Удельный контактный импеданс равен {number(read(out/"plan.local.json")["contact_impedance_ohm_m2"],7)} Ом·м². '
       'Все пять размеров имеют одинаковый вес. Критерий задаётся выражением\n\n'
       '$$RSS(\\rho_1,\\rho_2,\\rho_{\\mathrm{ж}};P)=\\sum_L[Z_{\\mathrm{FEM},L}(\\rho_1,\\rho_2,\\rho_{\\mathrm{ж}};P)-Z_{\\mathrm{obs},L}]^2,$$\n\n'
       'где RSS — сумма квадратов остатков, Ом²; P — заданное положение и ориентация монтажа; '
       'L — токовая база, мм; ZFEM и Zobs — модельный и наблюдаемый импедансы, Ом; '
       'ρ₁, ρ₂ и ρж — сопротивления трёх свободных материалов, Ом·м. '
       'Минимизация выполняется по логарифмам сопротивлений с сопряжёнными производными. '
       'Использованы два старта: частотные опорные значения и тройка 1, 30, 80 Ом·м. '
       'Выбирается меньший RSS; два старта не доказывают глобальность минимума.\n\n'
       'Производная вычисляется без повторной подгонки по формуле\n\n'
       '$$J_{L,j}=\\frac{\\partial Z_L}{\\partial\\ln\\rho_j}='
       '\\frac{(u_L^{\\mathrm{rec}})^\\mathsf{T}K_j u_L^{\\mathrm{dir}}}{I_{\\mathrm{dir}}I_{\\mathrm{rec}}\\rho_j},$$\n\n'
       'где $J_{L,j}$ — производная импеданса сборки L по логарифму сопротивления материала j, Ом; '
       '$u_L^{\\mathrm{dir}}$ и $u_L^{\\mathrm{rec}}$ — векторы узловых потенциалов, В, при прямом и взаимном возбуждениях; '
       '$K_j$ — геометрическая матрица материала с его объёмными долями и без множителя проводимости, м; '
       '$I_{\\mathrm{dir}}$ и $I_{\\mathrm{rec}}$ — токи соответствующих возбуждений, А, в расчёте оба равны 1 А; '
       'верхний знак T обозначает транспонирование. Прямой источник подключён к A–B, '
       'взаимный — к M–N. Контактный импеданс при дифференцировании фиксирован. '
       'Эта формула дополнительно проверена прямым изменением каждого сопротивления.')
    code(initialization(out,allow_superseded=True)+'\nresults=cases(OUT,allow_superseded=True)\ndisplay(table(inverse_table(results)))')
    md('## 4. Согласование с наблюдениями\n\n'
       f'Относительная норма остатков в двенадцати задачах составляет {number(min(residuals))}–{number(max(residuals))}%. '
       'На рисунке 1 показаны все сохранённые варианты. Различие кривых при перемещении центра '
       'обусловлено изменением анатомии в объёме чувствительности. '
       'Сравнение остатков между каналами не устанавливает их фактические роли без протокола.\n\n'
       '**Рисунок 1.** Наблюдаемые импедансы и FEM-предсказания после подгонки в каждом положении.')
    code('show(fitted_plot(results,OUT))')
    md('## 5. Идентифицируемость и геометрическая зависимость\n\n'
       'В таблице κ(J) — отношение наибольшего и наименьшего сингулярных чисел матрицы J '
       'производных пяти импедансов по логарифмам ρ₁, ρ₂ и ρж. '
       'Эта локальная характеристика не охватывает неопределённость положения, контакта, масок и фиксированных материалов. '
       'Достижение границы означает ограниченный технической областью результат, а не подтверждённое свойство ткани.\n\n'
       '**Рисунок 2.** Условные сопротивления при трёх местах общего центра. '
       'Линии связывают сценарии и не обозначают измеренное движение монтажей. '
       'Для всех сопротивлений показаны одни технические пределы 0,1–100 Ом·м; '
       'округление не превращает численное приближение к границе в различие тканей.')
    code('show(parameter_plot(results))')
    family=read(out/'weak_direction_family.local.json');points=np.array(family['family'])
    near=points[(points[:,4]<=5)&(points[:,6]==1)]
    md('Дополнительно построено семейство вдоль слабого сингулярного направления в логарифмах '
       'трёх сопротивлений для заранее выбранного примера: положение 2, канал 1, вдох. '
       'Направление продолжено до технических границ; каждая точка проверена новым прямым решением '
       'при 1024 и 4096 точках интегрирования. Остальные сопротивления после перемещения '
       'не переоптимизируются. Это проверенное подмножество вариантов, а не полный нелинейный профиль '
       'и не доверительная область. RSS₀ обозначает невязку выбранного локального минимума этой задачи.\n\n'
       f'Среди проверенных точек с приростом RSS не более 5% сопротивление лёгкого лежит от '
       f'{number(near[:,1].min())} до {number(near[:,1].max())} Ом·м. '
       'Допуск относится к RSS и не равен 5%-ной погрешности импеданса или сопротивлений.\n\n'
       '**Рисунок 3.** Семейство сопротивлений и изменение невязки вдоль слабого направления. '
       'Цветные точки удовлетворяют допуску 10% и численному критерию интегрирования; '
       'серая линия соединяет все вычисленные варианты.')
    code('show(weak_family_plot(OUT))')
    code("start_rows=[]\nfor c in results:\n    for i,s in enumerate(c['multiple_starts'],1):\n        start_rows.append({'Сценарий КТ':c['pose'],'Канал':c['channel'],'Дыхание':STATES[c['state']],'Старт':i,'ρ₁, Ом·м':s['rho_ohm_m'][0],'ρ₂, Ом·м':s['rho_ohm_m'][1],'ρж, Ом·м':s['rho_ohm_m'][2],'RSS, Ом²':s['sse_ohm2'],'Завершена':s['exitflag']>0})\ndisplay(table(pd.DataFrame(start_rows)))")
    bounds=sum(any(c['boundary_hits']) for c in rows)
    qpassed=sum(c['fine_integration_validation']['passed'] for c in rows)
    converged=sum(c['exitflag']>0 for c in rows)
    lung_upper=sum(c['rho_ohm_m'][1]>99.99 for c in rows)
    channel_one=[c for c in rows if c['channel']==1]
    rho1=[c['rho_ohm_m'][0] for c in channel_one];fat=[c['rho_ohm_m'][4] for c in channel_one]
    md(f'В {bounds} из 12 задач хотя бы одно сопротивление достигает технической границы; '
       f'численная остановка оптимизации с положительным кодом получена в {converged} задачах. '
       'Изменение параметров между местами и различие решений двух стартов должны учитываться '
       'вместе с остатками. Даже совпадение двух стартов в фиксированном месте не подтверждает '
       'единственную индивидуальную тройку при неизвестных координатах установки.\n\n'
       f'Верхней границы 100 Ом·м сопротивление лёгкого достигает в {lung_upper} из 12 задач. '
       'Совпадение таких граничных значений между вдохом и выдохом не доказывает неизменности лёгкого: '
       'дыхательное изменение параметра ограничено самой областью поиска. '
       f'Для гипотезы бокового канала 1 условные оценки фона лежат от {number(min(rho1))} '
       f'до {number(max(rho1))} Ом·м, жира — от {number(min(fat))} до {number(max(fat))} Ом·м. '
       'Это разброс между заданными положениями и состояниями, а не индивидуальный доверительный интервал.\n\n'
       '## 6. Численная проверка и ограничения\n\n'
       'До инверсии сопряжённые производные проверены центральными разностями в опорном состоянии '
       'каждого положения. При всех решениях контролируются невязка линейной системы и взаимность. '
       f'Проверка интегрирования при конечных оценках выполнена в {qpassed} из 12 задач '
       'по критерию 0,02 Ом + 0,1% от |Z| для вложенных выборок из 1024 и 4096 точек. '
       'Предварительное сравнение 256 и 1024 точек показано отдельно и не заменяет этой проверки. '
       'При передаче результата используется уточнённая проверка 1024→4096; её непрохождение '
       'исключает принятие соответствующей оценки. '
       'Сходимость по размеру всей сетки пока не установлена.\n\n'
       'Статистические доверительные интервалы не рассчитывались: ошибки повторной установки, '
       'контакта, межзаписного состояния и расхождения модели ещё не оценены. '
       'Неизвестные компоненты не заменены одним числом. Структура остатков сохраняется; '
       'формальное статистическое отвержение анатомической модели при неизвестной полной ошибке не выполняется.\n\n'
       '## 7. Решение о дальнейшем использовании\n\n'
       'Получены воспроизводимые условные тройки для двенадцати явно заданных задач. '
       'Они служат историческим материалом для проверки реализации. После уточнения геометрии '
       'их перенос запрещён; индивидуальные ρ₁, ρ₂ и ρж для окончательной '
       'коррекции ТТРКГ этим расчётом не утверждаются. Для выбора рабочей оценки требуются '
       'принятая разметка, уточнение положения и контактов, проверка сеточной сходимости '
       'и неопределённости фиксированных тканей. Полный многоразмерный ряд сохраняется.\n\n'
       'Далее базовые уровни вдоха и выдоха проверяются отдельно от парного дыхательного контраста. '
       'Пульсовые изменения сопротивлений оцениваются после принятия ЭКГ-разметки и проверки '
       'разделимости тканевых источников. Затем их неопределённость передаётся в модель ТТРКГ. '
       'Восстановление движения и объёма эквивалентной сердцу фигуры, УО и ФВ остаётся следующим звеном. '
       'Точность этой цепочки ещё не установлена.\n\n'
       'Ноутбук содержит сохранённые численные выходы, HTML исключает исходный код. '
       'Входы, версии вычислительных модулей, параметры оптимизации и два старта каждой задачи '
       'связаны хэшами в локальном паспорте.')
    export(INVERSE,cells,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    build_geometry(args.out);build_inverse(args.out)
