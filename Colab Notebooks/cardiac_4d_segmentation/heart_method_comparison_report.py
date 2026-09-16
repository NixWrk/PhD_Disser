"""Reader plots for the explicit anatomical/project/library comparison."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,Image

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
WORK=ROOT/'cardiac_4d_segmentation/results/reduced_heart_geometry_20260914/exploratory'
FEM=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_replacement_electrical_20260915'
AUDIT=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_geometry_perturbations_20260915'
OUT=WORK/'method_reader_01'
SUBJECTS=('adam','nix','georg')
NAMES=('Участник 1','Участник 2','Участник 3')
LABELS={'T0':'Маска целевой фазы','I1':'Индивидуальная форма','I2':'Индивидуальная + поворот','I3':'Индивидуальная + три масштаба',
'G1':'Г1 · равный объём','G2':'Г2 · второй момент','G3a':'Г3а · поверхность, центр задан',
'G3b':'Г3б · поверхность, центр свободен','G3c':'Г3в · поверхность, объём задан',
'G4a':'Г4а · перекрытие, объём задан','G4b':'Г4б · перекрытие, объём свободен','G5':'Г5 · алгебраический критерий',
'L1':'Л1 · моменты и объём','L2':'Л2 · моменты','L3a':'Л3а','L3b':'Л3б','L3c':'Л3в','L3d':'Л3г','L4a':'Л4а','L4b':'Л4б'}

INPUT_HASHES={}
def read(p):
    p=Path(p);data=p.read_bytes()
    INPUT_HASHES[p.resolve().relative_to(REPO.resolve()).as_posix()]=hashlib.sha256(data).hexdigest()
    return json.loads(data.decode('utf-8-sig'))

def write_input_provenance():
    p=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/tepc_preparation_20260911/report/figures/03_montages.png'
    INPUT_HASHES[p.relative_to(REPO).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'input_provenance.json').write_text(json.dumps(INPUT_HASHES,ensure_ascii=False,indent=2),encoding='utf-8')

def load():
    rows=[]
    for folder in ('method_comparison_01','method_overlap_sphere_01','method_ellipsoid_01'):
        path=WORK/folder/'results.json'
        if not path.exists():raise FileNotFoundError(path)
        rows.extend(read(path)['rows'])
    # Explicit later revisions; preceding attempts remain in their own outputs.
    revisions=('method_comparison_final','method_ellipsoid_surface_02','method_ellipsoid_overlap_03','method_overlap_sphere_georg_G4b_retry_30m','method_surface_full')
    used=[]
    for revision in revisions:
        path=WORK/revision/'results.json'
        if not path.exists():raise FileNotFoundError('Required final calculation revision: '+str(path))
        replacement=read(path)['rows'];keys={(r['case_id'],r['method_id']) for r in replacement}
        rows=[r for r in rows if (r['case_id'],r['method_id']) not in keys]+replacement
        used.append(revision)
    order=['T0','I1','I2','I3','G1','G2','G3a','G3b','G3c','G4a','G4b','G5','L1','L2','L3a','L3b','L3c','L3d','L4a','L4b']
    rows.sort(key=lambda r:(order.index(r['method_id']),SUBJECTS.index(r['subject'])))
    return rows

def show(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/name,dpi=155,bbox_inches='tight',facecolor='white')
    display(Image(filename=str(OUT/name)));plt.close(fig)

def heat(ax,values,rowlabels,collabels,title,cmap='viridis',limits=None,fmt='.3f'):
    arr=np.array(values,dtype=float);kw={} if limits is None else dict(vmin=limits[0],vmax=limits[1])
    im=ax.imshow(arr,aspect='auto',cmap=cmap,**kw)
    ax.set_yticks(range(len(rowlabels)),rowlabels);ax.set_xticks(range(len(collabels)),collabels)
    ax.set_title(title,pad=13)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v=arr[i,j];v=0. if np.isfinite(v) and abs(v)<1e-10 else v;t='нет оценки' if not np.isfinite(v) else format(v,fmt)
            rgba=im.cmap(im.norm(v)) if np.isfinite(v) else (1,1,1,1)
            lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,t,ha='center',va='center',fontsize=10,color='black' if lum>.5 else 'white')
    return im

def core():
    rows=load();lookup={(r['subject'],r['method_id']):r for r in rows}
    ids=['T0','I1','G1','L1']
    values=[[lookup[(s,m)]['metrics']['dice'] for s in SUBJECTS] for m in ids]
    fig,ax=plt.subplots(figsize=(9,3.7));im=heat(ax,values,[LABELS[m] for m in ids],NAMES,'Перекрытие с маской целевой фазы',limits=(0,1))
    fig.colorbar(im,ax=ax,label='Dice, от 0 до 1');show(fig,'01_anatomical_controls.png')

def method_tradeoff(prefix):
    rows=[r for r in load() if r['method_id'].startswith(prefix)]
    methods=list(dict.fromkeys(r['method_id'] for r in rows))
    lookup={(r['subject'],r['method_id']):r for r in rows}
    fig,axes=plt.subplots(1,2,figsize=(13,6.4),gridspec_kw={'width_ratios':[1,1]})
    for ax,key,title in zip(axes,['dice','volume_error_percent'],['Перекрытие с маской','Ошибка аналитического объёма, %']):
        values=[[(lookup[(s,m)].get('metrics') or {}).get(key,np.nan) for s in SUBJECTS] for m in methods]
        if key=='dice':kwargs=dict(cmap='viridis',limits=(.6,.9),fmt='.3f')
        else:
            mx=max(1.,float(np.nanmax(np.abs(values))));kwargs=dict(cmap='RdBu_r',limits=(-mx,mx),fmt='+.1f')
        im=heat(ax,values,[LABELS.get(m,m) if key=='dice' else LABELS.get(m,m).split(' · ')[0] for m in methods],NAMES,title,**kwargs)
        ax.tick_params(axis='y',labelsize=9);fig.colorbar(im,ax=ax,fraction=.047,pad=.04)
    fig.tight_layout(w_pad=3);show(fig,'02_sphere_criteria.png' if prefix=='G' else '03_ellipsoid_criteria.png')

def table():
    rows=load();lines=['| Метод | Участник 1: Dice / ошибка V | Участник 2 | Участник 3 |','|---|---|---|---|']
    methods=list(dict.fromkeys(r['method_id'] for r in rows))
    for method in methods:
        fields=[]
        for s in SUBJECTS:
            r=next(x for x in rows if x['subject']==s and x['method_id']==method);m=r.get('metrics')
            fields.append(f"{m['dice']:.3f} / {m['volume_error_percent']:+.2f}%" if m else 'Нет сошедшегося решения')
        lines.append('| '+LABELS.get(method,method)+' | '+' | '.join(fields)+' |')
    display(Markdown('\n'.join(lines)))

def electrical():
    data=read(FEM/'summary.json');rows=data['rows'] if isinstance(data,dict) else data
    if len(rows)!=18:raise ValueError('Electrical pilot incomplete')
    lookup={(r['geometry'],r['montage']):r for r in rows};montages=[f'tepc_{i}' for i in range(2,8)]
    fig,axes=plt.subplots(2,1,figsize=(12,6.4))
    for ax,key,title,unit in zip(axes,['delta_Z_vs_nominal_ohm','delta_Z_vs_nominal_percent'],
        ['Изменение импеданса при замещении исходной формы','То же изменение относительно исходного импеданса'],['Ом','%']):
        values=[[lookup[(g,m)][key] for m in montages] for g in ('sphere','ellipsoid')];mx=np.max(np.abs(values))
        im=heat(ax,values,['Сфера равного объёма','Эллипсоид равного объёма'],[m.upper().replace('_','-') for m in montages],title,
            cmap='RdBu_r',limits=(-mx,mx),fmt='+.3f' if unit=='Ом' else '+.2f')
        fig.colorbar(im,ax=ax,label=unit)
    fig.tight_layout(h_pad=2);show(fig,'04_electrical_replacement.png')
    lines=['| Сборка | Исходная форма, Ом | Сфера, Ом | Эллипсоид, Ом |','|---|---:|---:|---:|']
    for m in montages:lines.append('| '+m.upper().replace('_','-')+' | '+' | '.join(f"{lookup[(g,m)]['Z_ohm']:.4f}" for g in ('nominal','sphere','ellipsoid'))+' |')
    display(Markdown('\n'.join(lines)))

def electrodes():
    display(Image(filename=str(REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/tepc_preparation_20260911/report/figures/03_montages.png')))

def material_resolution():
    rows=read(AUDIT/'results.json')['derivatives'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for g,label,color in [('sphere','Сфера','#b96c00'),('ellipsoid','Эллипсоид','#9657af')]:
        r=[x for x in rows if x['geometry']==g and x['parameter']=='volume_fraction']
        axes[0].plot([100*x['step'] for x in r],[x['material_volume_derivative'] for x in r],'o-',label=label,color=color)
    axes[0].axhline(1,color='black',ls='--',label='Точное геометрическое значение')
    axes[0].set(xlabel='Полуширина изменения объёма, %',ylabel='Производная объёма: дискретный / заданный',title='Передача изменения объёма в материальную карту')
    for g,style in [('sphere','-'),('ellipsoid','--')]:
        for p,c in [('x_mm','#4477aa'),('y_mm','#ee7733'),('z_mm','#228833')]:
            r=[x for x in rows if x['geometry']==g and x['parameter']==p]
            axes[1].plot([x['step'] for x in r],[x['material_volume_derivative'] for x in r],style,marker='o',color=c,
                label=('Сфера' if g=='sphere' else 'Эллипсоид')+' · '+p[0])
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Полуширина перемещения, мм',ylabel='Ложное изменение объёма, мл/мм',title='При переносе объём должен сохраняться')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();show(fig,'05_material_resolution.png')

def physical_settings():
    p=read(FEM/'plan.json');rho=p['rho_ohm_m']
    display(Markdown('| Область | ρ, Ом·м |\n|---|---:|\n'+'\n'.join('| '+label+f" | {rho[k]:.4f} |" for k,label in [('soft','Мягкие ткани'),('heart','Эффективное целое сердце'),('lung','Лёгкие'),('bone','Кости')])))


def fractional_resolution():
    folder=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_fractional_materials_20260915'
    data=read(folder/'results.json');summary=data['level_metrics']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title,ylabel in [(axes[0],'max_abs_volume_derivative_minus_one','Ошибка производной объёма','Максимальное |dVдиск/dVзад − 1|'),
             (axes[1],'max_abs_translation_dV_ml_per_mm','Ложное изменение объёма при переносе','Максимальное |dVдиск/dx|, мл/мм')]:
        ax.loglog([r['points_per_tet'] for r in summary],[r[key] for r in summary],'o-',color='#246b83')
        for r in summary:ax.annotate(f"{r[key]:.4g}",(r['points_per_tet'],r[key]),xytext=(5,5),textcoords='offset points')
        ax.set(title=title,xlabel='Точки интегрирования в граничном элементе',ylabel=ylabel);ax.set_xticks([64,512,4096],['64','512','4096']);ax.grid(which='both',alpha=.2)
    fig.tight_layout();show(fig,'06_fractional_resolution.png')
    display(Markdown('| Точек в элементе | Максимальная ошибка объёма, % | Максимальная ошибка производной объёма, % |\n|---:|---:|---:|\n'+
        '\n'.join(f"| {r['points_per_tet']} | {r['max_abs_volume_error_percent']:.4f} | {100*r['max_abs_volume_derivative_minus_one']:.3f} |" for r in summary)))

def libraries():
    folder=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison'
    rows=[read(p) for p in folder.glob('*__32768.json')]
    if len(rows)!=27:raise ValueError('Expected 27 evaluated library fits')
    names=['М1 · BOOK','М2 · TAUB','М3 · HES','М4 · SOD','М5 · FC','М6 · 2-NORM','М7 · ellipsoid_fit: сфера','М8 · ellipsoid_fit: эллипсоид','М9 · SphereFit']
    fig,axes=plt.subplots(1,2,figsize=(13,6.4))
    for ax,key,title in zip(axes,['dice','volume_error_percent'],['Перекрытие с маской','Ошибка аналитического объёма, %']):
        values=[]
        for i in range(1,10):
            values.append([next(r for r in rows if r['case_id'].startswith(subject+'_') and r['method_id']==f'M{i}')['metrics'][key] for subject in SUBJECTS])
        mx=max(1.,float(np.nanmax(np.abs(values))))
        im=heat(ax,values,names if key=='dice' else [x.split(' · ')[0] for x in names],NAMES,title,
                cmap='viridis' if key=='dice' else 'RdBu_r',limits=(.6,.9) if key=='dice' else (-mx,mx),fmt='.3f' if key=='dice' else '+.1f')
        ax.tick_params(axis='y',labelsize=9);fig.colorbar(im,ax=ax,fraction=.047,pad=.04)
    fig.tight_layout(w_pad=3);show(fig,'07_library_criteria.png')

def phase_stability():
    rows=[r for r in read(WORK/'method_phase_01/results.json')['rows'] if not r['is_reference_phase']]
    fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
    for ax,s,name in zip(axes,SUBJECTS,NAMES):
        for i,method in enumerate(('G3c','G5')):
            values=[r['metrics']['volume_error_percent'] for r in rows if r['subject']==s and r['method_id']==method]
            ax.scatter(i+np.linspace(-.12,.12,len(values)),values,s=24,color=('#d49532','#4574a3')[i])
        ax.set_xticks([0,1],['Г3в: объём задан','Г5: объём свободен'],rotation=15);ax.axhline(0,color='gray',lw=.7);ax.set_title(name);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Ошибка аналитического объёма, %');fig.tight_layout();show(fig,'08_phase_stability.png')


def volume_sensitivity():
    folder=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_fractional_electrical_20260915'
    data=read(folder/'summary.json')
    if data['completed']!=13:raise ValueError('Fractional electrical pilot incomplete')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    for g,label,color in [('sphere','Сфера','#b96c00'),('ellipsoid','Эллипсоид','#9657af')]:
        states=sorted([r for r in data['states'] if r['geometry']==g and r['points_per_tet']==4096],key=lambda r:r['volume_fraction'])
        base=next(r for r in states if r['volume_fraction']==0)
        axes[0].plot([r['analytic_volume_ml']-base['analytic_volume_ml'] for r in states],
                     [1000*(r['Z_ohm']-base['Z_ohm']) for r in states],'o-',color=color,label=label)
        deriv=sorted([r for r in data['derivatives'] if r['geometry']==g and r['points_per_tet']==4096],key=lambda r:r['step_fraction'])
        axes[1].plot([100*r['step_fraction'] for r in deriv],[1000*abs(r['dZ_dV_analytic_ohm_per_ml']) for r in deriv],'o-',color=color,label=label+' · 4096 точек')
    extra=next(r for r in data['derivatives'] if r['points_per_tet']==512)
    axes[1].scatter([100*extra['step_fraction']],[1000*abs(extra['dZ_dV_analytic_ohm_per_ml'])],marker='x',s=80,color='black',label='Сфера · 512 точек',zorder=5)
    axes[0].set(xlabel='Заданное изменение объёма, мл',ylabel='Изменение импеданса, мОм',title='TEPC-2: изменение базового импеданса')
    axes[1].set(xlabel='Полуширина изменения объёма, %',ylabel='Модуль производной, мОм/мл',title='Проверка шага производной')
    axes[1].set_ylim(bottom=0)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();show(fig,'09_volume_sensitivity.png')
    lines=['| Фигура | Точек в элементе | Полуширина, % | dZ/dV, мОм/мл | Относительная чувствительность |','|---|---:|---:|---:|---:|']
    for r in data['derivatives']:
        volume=next(s['analytic_volume_ml'] for s in data['states'] if s['geometry']==r['geometry'] and s['volume_fraction']==0 and s['points_per_tet']==r['points_per_tet'])
        derivative=r['dZ_dV_analytic_ohm_per_ml'];elasticity=volume*derivative/r['Z_base_ohm']
        lines.append(f"| {'Сфера' if r['geometry']=='sphere' else 'Эллипсоид'} | {r['points_per_tet']} | {100*r['step_fraction']:.1f} | {1000*derivative:.4f} | {elasticity:.5f} |")
    display(Markdown('\n'.join(lines)))


def ellipsoid_surface_validation():
    rows=read(WORK/'method_ellipsoid_surface_02/results.json')['rows']
    methods=['L3a','L3b','L3c','L3d']
    values=[[next(r for r in rows if r['subject']==subject and r['method_id']==m)['metrics']['target_surface_to_candidate_RMS_mm'] for subject in SUBJECTS] for m in methods]
    fig,ax=plt.subplots(figsize=(9,3.7));im=heat(ax,values,[LABELS[m] for m in methods],NAMES,
        'Проверочная поверхностная RMS: 4096 точек подгонки, 32768 точек проверки',cmap='magma_r',fmt='.3f')
    fig.colorbar(im,ax=ax,label='RMS, мм; меньше — лучше');show(fig,'03b_ellipsoid_surface_RMS.png')


def individual_volume_sensitivity():
    """Original-mesh comparison only; spatial convergence stays separate."""
    folder=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_individual_stage1_20260916/comparison'
    data=read(folder/'comparison.json')
    if data['status']!='calculated_on_original_mesh':
        raise ValueError('Expected completed original-mesh comparison')
    names={'individual':'Индивидуальная форма','sphere':'Сфера','ellipsoid':'Эллипсоид'}
    colors={'individual':'#26705e','sphere':'#b96c00','ellipsoid':'#9657af'}
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    for geometry in names:
        states=sorted([r for r in data['states'] if r['geometry']==geometry],
                      key=lambda r:r['volume_fraction'])
        baseline=next(r for r in states if r['volume_fraction']==0)
        axes[0].plot([r['analytic_volume_ml']-baseline['analytic_volume_ml'] for r in states],
                     [1000*(r['Z_ohm']-baseline['Z_ohm']) for r in states],
                     'o-',label=names[geometry],color=colors[geometry])
        ds=sorted([r for r in data['derivatives'] if r['geometry']==geometry],
                  key=lambda r:r['step_fraction'])
        axes[1].plot([100*r['step_fraction'] for r in ds],
                     [1000*r['dZ_dV_ohm_per_ml'] for r in ds],
                     'o-',label=names[geometry],color=colors[geometry])
    checks=data['individual_material_derivative_checks']
    for step in sorted({r['step_fraction'] for r in checks}):
        values=sorted([r for r in checks if r['step_fraction']==step],
                      key=lambda r:r['points_per_tet'])
        axes[2].plot([r['points_per_tet'] for r in values],
                     [100*r['error_from_one'] for r in values],
                     'o-',label=f'Полуширина ±{100*step:g}%')
    axes[0].set(xlabel='Заданное изменение объёма, мл',
                ylabel='Изменение импеданса, мОм',title='Отклик трёх форм · TEPC-2')
    axes[1].set(xlabel='Полуширина возмущения объёма, %',
                ylabel='Центральная оценка, мОм/мл',title='Зависимость от шага')
    axes[2].axhline(0,color='black',lw=.7)
    axes[2].set(xlabel='Точек интегрирования в элементе',
                ylabel='Ошибка производной объёма, %',title='Точность изменения объёма')
    axes[2].set_xscale('log',base=2)
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();show(fig,'10_individual_volume_sensitivity.png')
    lines=['| Представление | Полуширина, % | dZ/dV, мОм/мл | Отличие производной от индивидуальной формы, % |',
           '|---|---:|---:|---:|']
    for r in data['derivatives']:
        error='—' if r['derivative_relative_error'] is None else f"{100*r['derivative_relative_error']:.2f}"
        lines.append(f"| {names[r['geometry']]} | {100*r['step_fraction']:g} | {1000*r['dZ_dV_ohm_per_ml']:.4f} | {error} |")
    display(Markdown('\n'.join(lines)))


def individual_quadrature_refinement():
    """Separate integration convergence from FEM-space convergence."""
    root=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_individual_stage1_20260916'
    coarse=read(root/'comparison/comparison.json')
    fine=read(root/'comparison_q16384/comparison.json')
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.4))
    colors={.005:'#26705e',.01:'#754ca3'}
    for step,color in colors.items():
        rows=sorted([r for r in fine['individual_material_derivative_checks'] if r['step_fraction']==step],key=lambda r:r['points_per_tet'])
        label=f'Полуширина ±{100*step:g}%'
        axes[0].plot([r['points_per_tet'] for r in rows],[100*r['error_from_one'] for r in rows],'o-',color=color,label=label)
        values=[next(r for r in d['derivatives'] if r['geometry']=='individual' and r['step_fraction']==step) for d in (coarse,fine)]
        axes[1].plot([r['points_per_tet'] for r in values],[1000*r['dZ_dV_ohm_per_ml'] for r in values],'o-',color=color,label=label)
    axes[0].axhline(0,color='black',lw=.7)
    axes[0].set(xlabel='Точек интегрирования в элементе',ylabel='Ошибка производной объёма, %',title='Воспроизведение заданного приращения')
    axes[1].set(xlabel='Точек интегрирования в элементе',ylabel='Центральная оценка, мОм/мл',title='Чувствительность индивидуальной формы')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([4096,8192,16384],['4096','8192','16384']);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();show(fig,'11_individual_quadrature_refinement.png')
    lines=['| Представление | Точек в элементе | Полуширина, % | dZ/dV, мОм/мл | Отличие от индивидуальной формы, % |','|---|---:|---:|---:|---:|']
    names={'individual':'Индивидуальная форма','sphere':'Сфера','ellipsoid':'Эллипсоид'}
    for r in fine['derivatives']:
        lines.append(f"| {names[r['geometry']]} | {r['points_per_tet']} | {100*r['step_fraction']:g} | {1000*r['dZ_dV_ohm_per_ml']:.4f} | {100*r['derivative_relative_error']:.2f} |")
    display(Markdown('\n'.join(lines)))


def heart_directional_response():
    root=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_individual_stage1_20260916'
    data=read(root/'comparison_q16384/comparison.json')
    records={r['geometry']:r for r in data['derivatives'] if r['step_fraction']==.005}
    names=['Индивидуальная форма','Сфера','Эллипсоид'];geometries=['individual','sphere','ellipsoid']
    fields=['decrease_slope_ohm_per_ml','dZ_dV_ohm_per_ml','increase_slope_ohm_per_ml']
    labels=['Уменьшение объёма','Центральная оценка','Увеличение объёма']
    colors=['#3c789a','#8c9299','#b0713f']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for i,(field,label,color) in enumerate(zip(fields,labels,colors)):
        values=[abs(records[g][field])*1000 for g in geometries]
        bars=axes[0].bar(np.arange(3)+(i-1)*.24,values,width=.23,color=color,label=label)
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
        errors=[100*abs(records[g][field]/records['individual'][field]-1) for g in ['sphere','ellipsoid']]
        bars=axes[1].bar(np.arange(2)+(i-1)*.24,errors,width=.23,color=color,label=label)
        axes[1].bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    axes[0].set_xticks(range(3),names);axes[1].set_xticks(range(2),['Сфера','Эллипсоид'])
    axes[0].set(ylabel='Модуль разностной оценки, мОм/мл',title='Чувствительность зависит от направления',ylim=(0,24))
    axes[1].set(ylabel='Отличие от индивидуальной формы, %',title='Ошибка для того же направления',ylim=(0,72))
    for ax in axes:ax.legend(fontsize=8,loc='upper right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();show(fig,'12_heart_directional_response.png')
    exchange=read(root/'material_exchange_diagnostic.json')
    lines=['| Изменение объёма | В исходных мягких тканях, мл | В исходном лёгком, мл | В исходной области сердца, мл | Всего, мл |',
           '|---|---:|---:|---:|---:|']
    for r in exchange['rows']:
        if abs(r['volume_fraction'])!=.005:continue
        v=r['changes_by_original_material_ml']
        lines.append(f"| {100*r['volume_fraction']:+g}% | {v['1']:+.3f} | {v['2']:+.3f} | {v['3']:+.3f} | {r['delta_heart_volume_ml']:+.3f} |")
    display(Markdown('\n'.join(lines)))


def heart_spatial_baseline():
    root=REPO/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_individual_stage1_20260916'
    result=read(root/'L1_baseline/result.json')
    seal=read(root/'L1_baseline/completion.json')
    if hashlib.sha256((root/'L1_baseline/result.json').read_bytes()).hexdigest()!=seal['hashes']['result.json']:
        raise ValueError('Baseline comparison result changed')
    export=read(root/'L1_export/heart_refinement_export.json')
    if hashlib.sha256((root/'L1_export/heart_refinement_export.json').read_bytes()).hexdigest()!=result['refined_export_sha256']:
        raise ValueError('Refined export changed')
    mesh=read(root/'L1_interior_collar0/refinement_report.json')
    if hashlib.sha256((root/'L1_interior_collar0/refinement_report.json').read_bytes()).hexdigest()!=export['refinement_report_sha256']:
        raise ValueError('Refined geometry QC changed')
    if not all(mesh['checks'].values()) or not export['contact_faces_preserved']:
        raise ValueError('Refined geometry or contact QC incomplete')
    volumes=mesh['metrics']['original_heart_volume_m3']
    rows=[('Узлы',f"{result['coarse_nodes']:,}",f"{result['fine_nodes']:,}"),
          ('Тетраэдры',f"{int(result['coarse_elements']):,}",f"{int(result['fine_elements']):,}"),
          ('Наружные треугольники',f"{mesh['mesh']['boundary_triangles']:,}",f"{mesh['mesh']['boundary_triangles']:,}"),
          ('Объём исходной сердечной области, мл',f"{1e6*volumes['before']:.6f}",f"{1e6*volumes['after']:.6f}"),
          ('Базовый импеданс TEPC-2, Ом',f"{result['coarse_Z_ohm']:.6f}",f"{result['fine_Z_ohm']:.6f}")]
    lines=['| Показатель | Исходная сетка | Уточнённая сетка |','|---|---:|---:|']
    lines += [f"| {name} | {coarse.replace(',', ' ').replace('.', ',')} | {fine.replace(',', ' ').replace('.', ',')} |" for name,coarse,fine in rows]
    display(Markdown('\n'.join(lines)))
