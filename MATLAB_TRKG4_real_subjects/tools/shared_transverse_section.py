"""Shared C01 section for report 20.10, using the same results as 20.16."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from transverse_geometry_study import ROOT,read_surface
from transverse_geometry_analysis import analytic,direct,LABELS
OUT=ROOT/'output/exploratory/transverse_consistency_20260914'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(out=OUT):
    out=Path(out);dest=out/'shared_section';dest.mkdir(exist_ok=True)
    g=json.loads((out/'geometry.json').read_text(encoding='utf8'))
    contract=json.loads((out/'input_contract.json').read_text(encoding='utf8'))
    profile=pd.read_csv(out/'central_profile.csv');lib=pd.read_csv(out/'pem_library.csv')
    cem=pd.read_csv(out/'cem_baseline.csv');sizes=np.sort(lib.L_mm.unique())
    assert contract['candidate']=='C01'
    rho=np.array([4.,16.]);zref,_=direct(lib,'reference',rho)
    palette={'planar':'#ad6b12','averaged':'#8667a7','uniform_transverse':'#3184b6','variable_transverse':'#c84f40','reference':'#1c2630'}
    rows=[]
    for model in ['planar','averaged','uniform_transverse','variable_transverse','reference']:
        z=rho[0]*analytic(rho[1]/rho[0],sizes,profile,g['h_centre_mm'],model)[0] if model in ['planar','averaged'] else direct(lib,model,rho)[0]
        for L,Z,R in zip(sizes,z,zref):rows.append(dict(model=model,L_mm=L,rho1=rho[0],rho2=rho[1],Z_ohm=Z,reference_ohm=R,residual_ohm=Z-R,contact='PEM'))
    data=pd.DataFrame(rows);data.to_csv(dest/'direct_comparison.csv',index=False)
    metrics=data.groupby('model').residual_ohm.agg(rmse_ohm=lambda a:np.sqrt(np.mean(a*a)),max_abs_ohm=lambda a:np.max(np.abs(a))).reset_index()
    metrics.to_csv(dest/'direct_metrics.csv',index=False)
    # Exact STL-plane intersections in the SAME orthonormal frame as FEM preparation.
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    centre=np.array(contract['centre_mm']);frame=np.array(contract['frame_columns_s_t_d'])
    def contours(filename):
        poly=read_surface(ROOT/'data/nik/stl'/filename,centre,frame)
        plane=vtk.vtkPlane();plane.SetOrigin(0,0,0);plane.SetNormal(0,1,0)
        cutter=vtk.vtkCutter();cutter.SetInputData(poly);cutter.SetCutFunction(plane);cutter.Update()
        stripper=vtk.vtkStripper();stripper.SetInputConnection(cutter.GetOutputPort());stripper.Update();cut=stripper.GetOutput()
        result=[]
        for i in range(cut.GetNumberOfCells()):
            points=vtk_to_numpy(cut.GetCell(i).GetPoints().GetData());result.append(points[:,[0,2]])
        return result
    body=contours('body_solid_v3_1mm_r10_volume_fill.stl');lung=contours('lungs_solid_v3_volume_fill.stl')
    inp=loadmat(out/'geometry_input.mat',simplify_cells=True)
    contact=pd.read_csv(out/'contacts.csv').query('L_mm==140')
    points=(inp['nodes_mm'][contact.point_node.to_numpy(dtype=int)-1]-centre)@frame
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    for k,(segments,color,label) in enumerate([(body,'#9a6827','Кожа'),(lung,'#188fa8','Граница лёгкого')]):
        for j,seg in enumerate(segments):axes[0].plot(seg[:,0],seg[:,1],color=color,lw=1.6,label=label if j==0 else None)
    axes[0].axhline(0,color='0.4',ls=':',label='Касательная плоскость')
    axes[0].scatter(points[:,0],points[:,2],c='#202d3a',s=28,zorder=5,label='Проекции PEM-контактов, L=140 мм')
    for point,label in zip(points,['I+','V+','V−','I−']):axes[0].annotate(label,(point[0],point[2]),xytext=(2,-13),textcoords='offset points')
    for ss in [-60,-30,0,30,60]:
        r=profile.loc[profile.s_mm==ss].iloc[0]
        if np.isfinite(r.h_mm):axes[0].plot([ss,ss],[r.skin_d_mm,r.skin_d_mm+r.h_mm],color='#589445',lw=1.3)
    axes[0].set(xlim=(-85,85),ylim=(230,-20),xlabel='s вдоль сборки, мм',ylabel='d от касательной плоскости, мм',title='КТ-срез в общей плоскости s–d, центр C01')
    axes[0].legend(fontsize=8,loc='lower left');axes[0].grid(alpha=.22)
    for _,r in profile.iterrows():
        for depth in json.loads(r.lung_crossings_d_mm):axes[1].plot(r.s_mm,depth-r.skin_d_mm,'.',color='#188fa8',ms=1.8)
    finite=np.isfinite(profile.h_mm)
    axes[1].plot(profile.s_mm,np.where(finite,profile.h_mm,np.nan),color='#2054a0',lw=2,label='Первое пересечение: h(s)')
    axes[1].axhline(0,color='#9a6827',lw=1.5,label='Кожа после визуального выпрямления')
    axes[1].axhline(g['h_centre_mm'],color='#a9670b',ls='--',label=f'М1: h(0) = {g["h_centre_mm"]:.2f} мм')
    for x in profile.loc[~finite,'s_mm']:axes[1].axvspan(x-.5,x+.5,color='#e59e98',alpha=.35,lw=0)
    axes[1].set(xlim=(-85,85),ylim=(250,-20),xlabel='s вдоль сборки, мм',ylabel='Расстояние от кожи вдоль d, мм',title='Все пересечения после вычитания профиля кожи')
    axes[1].legend(fontsize=8,loc='lower left');axes[1].grid(alpha=.22)
    fig.savefig(dest/'c01_anatomical_conditions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for model in ['planar','averaged','uniform_transverse','variable_transverse','reference']:
        d=data[data.model==model];axes[0].plot(d.L_mm,d.Z_ohm,'o-',ms=3,color=palette[model],label=LABELS[model]);axes[1 if model in ['planar','averaged'] else 2].plot(d.L_mm,d.residual_ohm,'o-',ms=3,color=palette[model],label=LABELS[model])
    axes[0].set(xlabel='Номинальный размер L, мм',ylabel='Передаточное сопротивление, Ом',title='Одинаковые ρ₁ = 4 и ρ₂ = 16 Ом·м')
    axes[1].set(xlabel='Номинальный размер L, мм',ylabel='Z модели − Z КТ/PEM, Ом',title='Аналитические модели: разность с КТ/PEM')
    axes[2].set(xlabel='Номинальный размер L, мм',ylabel='Z модели − Z КТ/PEM, Ом',title='Пространственные модели: разность с КТ/PEM')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for ax in axes[1:]:ax.legend(fontsize=8);ax.axhline(0,color='0.3',lw=.7)
    fig.savefig(dest/'c01_direct_comparison.png',dpi=160);plt.close(fig)
    sourcefiles=['input_contract.json','geometry.json','central_profile.csv','geometry_masks.mat','source_electrodes.csv','contacts.csv','pem_library.csv','cem_baseline.csv']
    record={'candidate':'C01','rho1_ohm_m':4,'rho2_ohm_m':16,'h_centre_mm':g['h_centre_mm'],'same_source_as_2016':True,'sources':{n:sha(out/n) for n in sourcefiles},'source_sha256':sha(__file__),'artifacts':{p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.csv','.png']},'scope':'Repeated synthetic calculation, not measurement of tissue properties'}
    (dest/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(metrics.to_string(index=False))

def shared_cells(out=OUT):
    # One result directory feeds both notebooks; labels explicitly distinguish historical runs.
    code=r'''from pathlib import Path
import sys
_shared_project = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p/'MATLAB_TRKG4_real_subjects').is_dir())
if str(_shared_project/'MATLAB_TRKG4_real_subjects/tools') not in sys.path:
    sys.path.insert(0,str(_shared_project/'MATLAB_TRKG4_real_subjects/tools'))
from shared_transverse_section import display_geometry, display_direct
display_geometry()'''
    md=r'''### 11.3.1. Согласованная геометрическая постановка для 20.16

**Пересчёт от 14.09.2026.** Для текущего сравнения используется кандидат C01 из поиска раздела 20: центр и весь локальный базис совпадают с 20.16. Это ранее выбранное автором положение; его совпадение с фактической наклейкой электродов и совместная оптимальность по двум сопротивлениям не установлены. Ранний пример другого положения перенесён в раздел 11.5 и не используется как геометрический вход 20.16.

Используются одна исходная КТ на вдохе, одна сетка тела и одни контактные площадки. Ось $s$ направлена вдоль сборки, $t$ — поперёк в касательной плоскости, $d$ — внутрь по нормали в центре. Направленная толщина $h(s)$ измеряется от пересечения кожи до первого пересечения лёгкого на луче вдоль $d$. Рёбра, сердце и поддиафрагмальные ткани входят в единый фон с сопротивлением $\rho_1$; лёгкое имеет $\rho_2$. Два тканевых параметра не означают, что любой луч встречает ровно две пространственные области.'''
    caption=r'''**Рисунок 11.3а. Исходный срез и представление всех пересечений в единой геометрии C01.** Слева показано пересечение исходных STL-поверхностей плоскостью $t=0$; точки электродов спроецированы на эту плоскость. Зелёные отрезки показывают несколько направленных расстояний кожа–первое лёгкое. Справа из глубин всех лёгочных пересечений вычтен профиль кожи: синяя линия выделяет первое пересечение, бирюзовые точки сохраняют остальные. Красный фон отмечает координаты, где луч не встречает лёгкое.

Сопоставлять следует изменение расстояния вдоль $s$, исчезновение лёгочного пересечения и наличие дополнительных границ. Одно $h$ теряет первое различие; даже полный профиль первого пересечения теряет остальные границы. Визуальное выпрямление справа не преобразует электрическую задачу: без изменения коэффициентов проводимости и контактных условий это только изображение расстояний. Нижний край рисунка не задаёт плоский слой ткани под лёгким. По этому срезу нельзя определить распределение тока или поперечную форму; поэтому далее сравниваются электрические операторы, а в 20.16 — объёмные модели и ошибка обратной задачи.'''
    direct_md=r'''### 11.4. Прямое сравнение при одинаковых сопротивлениях и исходных данных

Для воспроизводимого синтетического контроля заданы $\rho_1=4$ и $\rho_2=16$ Ом·м — внутри рабочих диапазонов 20.16. Это заданные числа вычислительного опыта, а не измеренные свойства добровольца. Сопротивления не подгоняются к эксперименту. Девять размеров и единая геометрическая постановка совпадают с 20.16.

М1 использует центральную толщину. М2 использует вторую схему раздела 11.3: усредняется само локальное плоское ядро. В него подставляется табличный профиль первого пересечения по STL без полиномиальной аппроксимации; при отсутствии лёгкого сохраняется однородное ядро. Подстановка одной средней толщины в ядро относится к другой эвристике и в текущую М2 не входит. М3 и М4 решают пространственную задачу в теле с общей и переменной поперечной формой соответственно. Референс использует исходную полную КТ-маску лёгкого. М1 и М2 имеют номинальные точечные электроды на плоскости; М3, М4 и КТ — одинаковые узлы PEM на реальном теле. Следовательно, отличие аналитических моделей включает замену кожи и контактов; чистое влияние геометрии лёгкого устанавливается сравнением пространственных моделей между собой.'''
    caption2=r'''**Рисунок 11.4а. Прямые расчёты пяти операторов для одного синтетического состояния.** Слева представлены импедансы, в центре — знаковые разности аналитических моделей с КТ/PEM, справа — соответствующие разности пространственных моделей. Раздельные вертикальные масштабы позволяют увидеть малые расхождения М3 и М4; их нельзя сравнивать по высоте линий без чтения осей. Близость к нулю означает лучшее воспроизведение референса в этом состоянии. Нулевая линия КТ является вычитанием результата из самого себя, а не нулевой физической погрешностью FEM.

**Результат для этого состояния.** RMSE относительно КТ/PEM составляет 11,248 Ом для М1, 12,767 Ом для М2, 0,0540 Ом для М3 и 0,0405 Ом для М4. Переменная пространственная форма даёт наименьший RMSE среди кандидатов по девяти размерам, но не минимальную ошибку для каждого размера отдельно. При этом добавление профиля глубины в М2 ухудшило прямой прогноз относительно М1: больше геометрических сведений внутри эвристического усреднения не гарантирует более точного электрического поля.

Эта проверка отвечает, насколько упрощение меняет импеданс при известных сопротивлениях. Она не устанавливает точность восстановления неизвестных $\rho_1,\rho_2$: для этого нужны независимые изменения обоих параметров и обратный расчёт 20.16. Площадки CEM проверяются отдельно при тех же заданных сопротивлениях; нельзя смешивать невязки относительно PEM и CEM или приписывать их только форме лёгкого. Новые варианты включения в полупространстве здесь ещё не рассчитаны.'''
    return [nbf.v4.new_markdown_cell(md),nbf.v4.new_code_cell(code),nbf.v4.new_markdown_cell(caption),nbf.v4.new_markdown_cell(direct_md),nbf.v4.new_code_cell('display_direct()'),nbf.v4.new_markdown_cell(caption2)]

def checked(out=OUT):
    m=json.loads((out/'shared_section/manifest.json').read_text(encoding='utf8'))
    for name,digest in m['sources'].items():assert sha(out/name)==digest,name
    for name,digest in m['artifacts'].items():assert sha(out/'shared_section'/name)==digest,name
    return m

def display_geometry(out=OUT):
    from IPython.display import display,Markdown,Image
    m=checked(out);c=json.loads((out/'input_contract.json').read_text(encoding='utf8'))
    display(Markdown(f"Центр: **({c['centre_mm'][0]:.3f}; {c['centre_mm'][1]:.3f}; {c['centre_mm'][2]:.3f}) мм**. Направленная центральная толщина: **{m['h_centre_mm']:.2f} мм**. Эти значения читаются из того же расчётного набора, что и 20.16."))
    display(Image(filename=str(out/'shared_section/c01_anatomical_conditions.png')))

def display_direct(out=OUT):
    from IPython.display import display,Markdown,Image,HTML
    checked(out);d=pd.read_csv(out/'shared_section/direct_metrics.csv')
    d['model']=d.model.replace(LABELS);d=d.rename(columns={'model':'Модель','rmse_ohm':'RMSE к КТ/PEM, Ом','max_abs_ohm':'Макс. |разность|, Ом'})
    display(Image(filename=str(out/'shared_section/c01_direct_comparison.png')))
    display(HTML(d.to_html(index=False,float_format=lambda x:f'{x:.5f}')))
    cem=pd.read_csv(out/'cem_baseline.csv');ref=cem[cem.model=='reference'].set_index('L_mm').Z
    rows=[]
    for model in ['uniform_transverse','variable_transverse']:
        err=cem[cem.model==model].set_index('L_mm').Z-ref
        rows.append({'Модель':LABELS[model],'RMSE к КТ/CEM, Ом':np.sqrt(np.mean(err**2)),'Макс. |разность|, Ом':np.max(abs(err))})
    display(Markdown('**Отдельный контроль CEM:** одинаковые конечные площадки и контактное сопротивление во всех трёх пространственных моделях.'))
    display(HTML(pd.DataFrame(rows).to_html(index=False,float_format=lambda x:f'{x:.5f}')))

if __name__=='__main__':prepare()
