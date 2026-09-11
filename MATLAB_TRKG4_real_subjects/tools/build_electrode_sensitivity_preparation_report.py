"""Build the expanded scientific report 40.06 from frozen numerical results."""
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'Colab Notebooks/40.06_Карты_чувствительности_и_подготовка_новых_комбинаций.ipynb'
TEXT = Path(__file__).with_name('electrode_sensitivity_report_ru.md')
BLOCKS = {}
BLOCKS['load'] = r'''
from pathlib import Path
import sys, json, hashlib
ROOT=Path.cwd()
if ROOT.name=='Colab Notebooks': ROOT=ROOT.parent
BASE=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/electrode_sensitivity_20260910'
OLD=BASE.parent/'arm_sigma_20260908'
sys.path.insert(0,str(OLD/'python_solver_deps'))
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import display, Markdown
RUN=BASE/'pilot_combinations_verified'
CHECKS=BASE/'checks_combinations_final'
REV=BASE/'report_revision_20260910'
FIGURES=REV/'figures'; FIGURES.mkdir(exist_ok=True)
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
for folder,manifest in [(RUN,'completion.json'),(CHECKS,'checks.json')]:
    passport=json.loads((folder/manifest).read_text(encoding='utf-8'))
    for name,expected in passport['output_sha256'].items(): assert sha(folder/name)==expected,name
assert sha(RUN/'completion.json')==(RUN/'completion.sha256').read_text().strip()
channels=pd.read_csv(RUN/'channel_results.csv')
summary=pd.read_csv(RUN/'compartment_maps.csv')
probes=pd.read_csv(CHECKS/'equal_volume_probes.csv')
fd=pd.read_csv(CHECKS/'local_finite_differences.csv')
verification=pd.read_csv(CHECKS/'verification.csv'); assert verification.passed.all()
study=json.loads((BASE/'pilot_combinations.json').read_text(encoding='utf-8'))
old_manifest=json.loads((OLD/'result_integrity.json').read_text(encoding='utf-8'))
assert sha(OLD/'prepared_fem.mat')==study['source']['prepared_sha256']
for filename in ['fixed_parameters.json','in20_out80_circumferential_ring_patches.csv']:
    assert sha(OLD/filename)==old_manifest['files'][filename]
fixed=json.loads((OLD/'fixed_parameters.json').read_text(encoding='utf-8'))
patches=pd.read_csv(OLD/'in20_out80_circumferential_ring_patches.csv')
with np.load(RUN/'map_geometry.npz') as f: geometry={k:f[k] for k in f.files}
row=channels[channels.channel=='standard'].iloc[0]
with np.load(RUN/f'{row.montage}__{row.state}__standard_sensitivity.npz') as f:
    density=f['density']/1e6
xyz=geometry['centroid_m']*1000
volumes=geometry['volume_m3']; tissue_ids=geometry['tissue_id']
order=['soft_tissue','lungs','heart','bones']
labels={'soft_tissue':'Мягкотканный фон','lungs':'Лёгкие','heart':'Сердечная область','bones':'Кости'}
short=['Мягкие ткани','Лёгкие','Сердце','Кости']
colours={'soft_tissue':'#6585a1','lungs':'#368579','heart':'#bb5a53','bones':'#a78a55'}
parameters={'soft_tissue':'soft','lungs':'lung','heart':'heart','bones':'bone'}
sigma={k:study['states'][0]['conductivity'][parameters[k]] for k in order}
s=summary[summary.channel=='standard'].set_index('tissue').loc[order].copy()
p=probes[probes.channel=='standard'].set_index('tissue').loc[order[:3]].copy()
channel_names={'standard':'А','adjacent':'Б','cross':'В'}
plt.rcParams.update({'figure.dpi':115,'savefig.dpi':180,'font.size':11,'axes.titlesize':12,
                     'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
def ru(x,d=3): return f'{float(x):.{d}f}'.replace('.',',').replace('-','−')
def sci(x): return f'{float(x):.3e}'.replace('.',',').replace('-','−')
def table(df,formats=None):
    styler=df.style.hide(axis='index')
    if formats: styler=styler.format(formats,na_rep='—')
    display(styler)
def finish(n,fig):
    fig.savefig(FIGURES/f'figure_{n:02d}.png',dpi=180,bbox_inches='tight')
    plt.show(); plt.close(fig)
def bars(ax,values,title,xlabel,log=False):
    ys=np.arange(len(values)); b=ax.barh(ys,values,color=[colours[t] for t in order[:len(values)]],height=.63)
    ax.set(yticks=ys,yticklabels=short[:len(values)],title=title,xlabel=xlabel)
    ax.invert_yaxis()
    if log: ax.set_xscale('log'); ax.set_xlim(min(values)*.45,max(values)*6)
    else: ax.set_xlim(0,max(values)*1.25)
    ax.bar_label(b,labels=[ru(x,5 if 0<abs(x)<.1 else (3 if abs(x)<10 else 2)) for x in values],padding=4,fontsize=9)
    ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True)
'''
BLOCKS['baseline'] = r'''
df=pd.DataFrame({'Компартмент':[labels[t] for t in order],
 'Объём, л':s.volume_m3.to_numpy()*1000,'Проводимость, См/м':[sigma[t] for t in order],
 'Статус параметра':['Исследуемый','Исследуемый','Исследуемый','Фиксирован']})
table(df,{'Объём, л':lambda x:ru(x,3),'Проводимость, См/м':lambda x:ru(x,6)})
'''
BLOCKS['montages'] = r'''
fig,axes=plt.subplots(3,1,figsize=(12,7.5),constrained_layout=True)
xx=[-3,-1.8,1.8,3]
for ax,c in zip(axes,study['montages'][0]['channels']):
    ax.set(xlim=(-4,4),ylim=(-.9,1));ax.axis('off')
    ax.add_patch(Rectangle((-1.1,-.35),2.2,.7,facecolor='#eef0f2',edgecolor='#a5adb4'))
    ax.text(0,0,'Туловище',ha='center',va='center',fontsize=12)
    ax.plot([-3.5,-1.1],[0,0],lw=13,color='#dde1e5',zorder=0)
    ax.plot([1.1,3.5],[0,0],lw=13,color='#dde1e5',zorder=0)
    for i,x in enumerate(xx):
        drive=c['drive'][i]; measure=c['measure'][i]
        colour='#b44d43' if drive else '#3d70a1';role=('I+' if drive>0 else 'I−') if drive else ('U+' if measure>0 else 'U−')
        ax.add_patch(Circle((x,0),.17,facecolor=colour,edgecolor='white',lw=1.2,zorder=3))
        ax.text(x,.3,f'E{i+1}\n{role}',ha='center',va='bottom',color=colour,fontweight='bold')
    ax.text(-2.4,-.65,'Правая рука модели',ha='center',fontsize=10)
    ax.text(2.4,-.65,'Левая рука модели',ha='center',fontsize=10)
    ax.text(-3.8,.82,'Комбинация '+channel_names[c['id']],fontweight='bold')
finish(1,fig)
'''
BLOCKS['channel_table'] = r'''
rows=[]
for c in study['montages'][0]['channels']:
    d=np.asarray(c['drive']);m=np.asarray(c['measure'])
    rows.append({'Комбинация':channel_names[c['id']],
      'Ток: ввод → вывод':f'E{np.argmax(d)+1} → E{np.argmin(d)+1}',
      'Напряжение':f'U(E{np.argmax(m)+1}) − U(E{np.argmin(m)+1})'})
table(pd.DataFrame(rows))
'''
BLOCKS['anatomy'] = r'''
fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
for ax,(a,b) in zip(axes,[(0,2),(0,1)]):
    for tissue,label in zip(order,[1,2,3,4]):
        ids=np.flatnonzero(tissue_ids==label)
        maximum=22000 if label==1 else 6000
        ids=ids[::max(1,int(np.ceil(len(ids)/maximum)))]
        ax.scatter(xyz[ids,a],xyz[ids,b],s=1.5 if label==1 else 2,
          alpha=.12 if label==1 else .35,color=colours[tissue],label=labels[tissue],rasterized=True)
    coords=patches[['x_mm','y_mm','z_mm']].to_numpy()
    ax.scatter(coords[:,a],coords[:,b],s=60,facecolors='white',edgecolors='black',zorder=5)
    for i,point in enumerate(coords):ax.annotate(f'E{i+1}',(point[a],point[b]),xytext=(0,9),textcoords='offset points',ha='center',fontweight='bold')
    ax.set(xlabel='XYZ'[a]+', мм',ylabel='XYZ'[b]+', мм',title='Проекция '+ 'XYZ'[a]+'–'+'XYZ'[b]);ax.set_aspect('equal')
handles,names=axes[0].get_legend_handles_labels()
fig.legend(handles,names,loc='outside lower center',ncol=4,markerscale=3,fontsize=9)
finish(2,fig)
'''
BLOCKS['global_metrics'] = r'''
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
bars(axes[0,0],s.volume_m3.to_numpy()*1000,'а. Объём компартмента','л')
bars(axes[0,1],np.array([sigma[t] for t in order]),'б. Исходная проводимость','См/м')
bars(axes[1,0],abs(s.dZ_dsigma.to_numpy()),'в. Модуль интегральной производной','Ом/(См/м), логарифмическая шкала',True)
bars(axes[1,1],abs(s.mean_density.to_numpy())/1e6,'г. Модуль средней плотности','Ом/[(См/м)·мл], логарифмическая шкала',True)
finish(3,fig)
'''
BLOCKS['integrals'] = r'''
df=pd.DataFrame({'Компартмент':[labels[t] for t in order],'D, Ом/(См/м)':s.dZ_dsigma.to_numpy(),
 'Средняя плотность, Ом/[(См/м)·мл]':s.mean_density.to_numpy()/1e6,
 'D+, Ом/(См/м)':s.positive_integral.to_numpy(),'D−, Ом/(См/м)':s.negative_integral.to_numpy(),
 'Компенсация, %':s.cancellation_ratio.to_numpy()*100})
table(df,{c:lambda x:ru(x,6) for c in df.columns[1:]})
'''
BLOCKS['global_interpretation'] = r'''
ratio_volume=s.loc['soft_tissue','volume_m3']/s.loc['heart','volume_m3']
ratio_D=abs(s.loc['soft_tissue','dZ_dsigma']/s.loc['heart','dZ_dsigma'])
ratio_mean=abs(s.loc['soft_tissue','mean_density']/s.loc['heart','mean_density'])
display(Markdown(f'Объём мягкотканного фона больше объёма сердечной области в **{ru(ratio_volume,1)} раза**. Модуль интегральной производной больше в **{ru(ratio_D,1)} раза**, а модуль средней плотности — в **{ru(ratio_mean,1)} раза**. Эти отношения связаны равенством из формулы (5): {ru(ratio_volume,1)} × {ru(ratio_mean,1)} ≈ {ru(ratio_D,1)}. Следовательно, полный объём объясняет часть различия; оставшаяся часть отражена в средней плотности данной модели.'))
'''
BLOCKS['whole_perturbations'] = r'''
D=s.dZ_dsigma.to_numpy();sig=np.array([sigma[t] for t in order])
fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
bars(axes[0],abs(.01*sig*D)*1000,'а. Весь компартмент: +1%','|ΔZ|, мОм, логарифмическая шкала',True)
bars(axes[1],abs(.001*D)*1000,'б. Весь компартмент: +0,001 См/м','|ΔZ|, мОм, логарифмическая шкала',True)
finish(4,fig)
'''
BLOCKS['percent_interpretation'] = r'''
soft_R=float(row.R_1percent_soft_ohm)*1000;heart_R=float(row.R_1percent_heart_ohm)*1000;lung_R=float(row.R_1percent_lung_ohm)*1000
display(Markdown(f'При условном увеличении проводимости всего компартмента на 1% прогноз составляет **{ru(soft_R,2)} мОм** для мягкотканного фона, **{ru(heart_R,3)} мОм** для сердечной области и **{ru(lung_R,3)} мОм** для лёгких. Сам шаг проводимости равен соответственно {ru(.01*sigma["soft_tissue"],6)}, {ru(.01*sigma["heart"],6)} и {ru(.01*sigma["lungs"],6)} См/м. Поэтому одинаковый процент в этих случаях означает разные абсолютные изменения параметра.'))
'''
BLOCKS['maps'] = r'''
heart=tissue_ids==3;centre=np.average(xyz[heart],axis=0,weights=volumes[heart])
fig,axes=plt.subplots(1,3,figsize=(14,5.5),constrained_layout=True)
for ax,(a,b,c) in zip(axes,[(0,1,2),(0,2,1),(1,2,0)]):
    ids=np.flatnonzero(abs(xyz[:,c]-centre[c])<=3);ids=ids[::max(1,int(np.ceil(len(ids)/30000)))]
    dots=ax.scatter(xyz[ids,a],xyz[ids,b],c=density[ids],s=3,cmap='RdBu_r',norm=SymLogNorm(linthresh=.001,vmin=-1,vmax=1),rasterized=True)
    hid=ids[tissue_ids[ids]==3]
    ax.scatter(xyz[hid,a],xyz[hid,b],facecolors='none',edgecolors='black',s=11,linewidths=.35)
    ax.set(xlabel='XYZ'[a]+', мм',ylabel='XYZ'[b]+', мм',title=f'{"XYZ"[c]} = {ru(centre[c],1)} ± 3 мм');ax.set_aspect('equal')
fig.colorbar(dots,ax=axes,label='w, Ом/[(См/м)·мл]; линейная зона |w| ≤ 0,001',shrink=.82,extend='both')
finish(5,fig)
'''
BLOCKS['distributions'] = r'''
max_abs=float(np.max(abs(density))); edge=max(1,10**np.ceil(np.log10(max_abs)))
positive=np.geomspace(1e-7,edge,42);bins=np.r_[-positive[::-1],0,positive]
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for ax,t,label in zip(axes.ravel(),order,[1,2,3,4]):
    mask=tissue_ids==label
    counts,_=np.histogram(density[mask],bins=bins,weights=volumes[mask])
    pct=100*counts/volumes[mask].sum();assert abs(pct.sum()-100)<1e-7
    ax.stairs(pct,bins,fill=True,color=colours[t],alpha=.72)
    ax.axvline(0,color='black',lw=.8)
    ax.axvline(s.loc[t,'mean_density']/1e6,color='black',ls='--',lw=1,label='Средняя плотность')
    ax.set_xscale('symlog',linthresh=1e-4)
    ticks=[-1e3,-1e1,-1e-1,-1e-3,0,1e-3,1e-1,1e1,1e3]
    ax.set_xticks(ticks)
    ax.tick_params(axis='x',labelsize=9)
    ax.set(title=labels[t],xlabel='w, Ом/[(См/м)·мл]',ylabel='Доля объёма в интервале, %')
    ax.legend(fontsize=9);ax.grid(alpha=.15)
finish(6,fig)
'''
BLOCKS['probe_geometry'] = r'''
probe_shapes=[]
with h5py.File(OLD/'prepared_fem.mat','r') as f:
    for t,label in zip(order[:3],[1,2,3]):
        with np.load(CHECKS/f'probe_tissue_{label}.npz') as a:roi={k:a[k] for k in a.files}
        ids=roi['indices'];permutation=np.argsort(ids)
        tetra_sorted=f['fmdl_mm/elems'][:,ids[permutation]].T.astype(int)-1
        tetra=tetra_sorted[np.argsort(permutation)]
        used=np.unique(tetra)
        points=f['fmdl_mm/nodes'][:,used].T
        coordinates=points[np.searchsorted(used,tetra)]-roi['centre_m']*1000
        probe_shapes.append((t,coordinates,roi['weights']))
limit=np.ceil(max(float(np.max(abs(x[1]))) for x in probe_shapes)/5)*5
fig=plt.figure(figsize=(14,5.3),layout='constrained')
faces_idx=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
for i,(t,coordinates,weights) in enumerate(probe_shapes):
    ax=fig.add_subplot(1,3,i+1,projection='3d')
    for vertices,w in zip(coordinates,weights):
        collection=Poly3DCollection(vertices[faces_idx],facecolors=colours[t],edgecolors='#3c4650',linewidths=.32,alpha=.12+.35*float(w))
        ax.add_collection3d(collection)
    ax.scatter([0],[0],[0],color='black',marker='+',s=55)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),zlim=(-limit,limit),xlabel='ΔX, мм',ylabel='ΔY, мм',zlabel='',
           title=f'{labels[t]}\n{len(weights)} элементов; полный объём {ru(p.loc[t,"support_volume_ml"],3)} мл')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-58);ax.tick_params(labelsize=8)
    ax.text2D(.89,.70,'ΔZ, мм',transform=ax.transAxes,rotation=90,fontsize=10)
finish(7,fig)
'''
BLOCKS['probe_table'] = r'''
df=pd.DataFrame({'Компартмент':[labels[t] for t in order[:3]],'Эффективный объём, мл':p.effective_volume_ml.to_numpy(),
 'Полный объём, мл':p.support_volume_ml.to_numpy(),'Элементов':p.element_count.to_numpy(),
 'Вес последнего элемента':p.terminal_weight.to_numpy(),
 'Δσ/σ, %':[.001/sigma[t]*100 for t in order[:3]],'Прогноз ΔZ, мОм':p.predicted_delta_Z_mohm.to_numpy()})
table(df,{c:lambda x:ru(x,4) for c in df.columns if c not in ['Компартмент','Элементов']})
'''
BLOCKS['probe_response'] = r'''
actual=p.predicted_delta_Z_mohm.to_numpy()
mean_reference=s.loc[order[:3],'mean_density'].to_numpy()/1e6*10*.001*1000
fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
ys=np.arange(3)
b=axes[0].barh(ys,actual,color=[colours[t] for t in order[:3]])
axes[0].set(yticks=ys,yticklabels=short[:3],xlabel='Прогноз ΔZ, мОм',title='а. Выбранные локальные пробы')
axes[0].bar_label(b,labels=[ru(v,4) for v in actual],padding=4,fontsize=10)
axes[0].set_xlim(min(actual)*1.42,0);axes[0].invert_yaxis()
axes[1].barh(ys-.17,abs(actual),height=.3,color='#557fa0',label='Выбранная проба')
axes[1].barh(ys+.17,abs(mean_reference),height=.3,color='#c4ccd4',label='Оценка по средней плотности')
axes[1].set(yticks=ys,yticklabels=short[:3],xlabel='|ΔZ|, мОм',title='б. Тот же объём и электрический масштаб')
axes[1].invert_yaxis();axes[1].legend(fontsize=9,loc='lower right')
for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
finish(8,fig)
'''
BLOCKS['probe_interpretation'] = r'''
soft=float(p.loc['soft_tissue','predicted_delta_Z_mohm']);heart=float(p.loc['heart','predicted_delta_Z_mohm']);lung=float(p.loc['lungs','predicted_delta_Z_mohm'])
display(Markdown(f'Для выбранных областей изменение импеданса равно **{ru(soft,4)} мОм** в мягкотканном фоне, **{ru(heart,4)} мОм** в сердечной области и **{ru(lung,4)} мОм** в лёгком. Лёгочная проба даёт по модулю в **{ru(abs(lung/heart),2)} раза** больший отклик, чем сердечная. Отношение мягкотканной пробы к сердечной составляет лишь **{ru(abs(soft/heart),2)}**, хотя отношение интегральных производных целых компартментов равно приблизительно **{ru(ratio_D,0)}**. Противоречия нет: размеры изменяющихся областей и способ пространственного усреднения различаются.'))
display(Markdown(f'Оценка для 10 мл по средней плотности всего компартмента дала бы {ru(mean_reference[0],4)}, {ru(mean_reference[1],4)} и {ru(mean_reference[2],4)} мОм соответственно для мягкотканного фона, лёгких и сердечной области. Отклик выбранной мягкотканной пробы составляет около **{ru(abs(soft/mean_reference[0])*100,1)}%** соответствующего среднего ориентира. Этот процент характеризует выбранное положение относительно среднего по модели; он не является долей ткани в физиологическом сигнале.'))
'''
BLOCKS['finite_difference'] = r'''
f=fd.sort_values('delta_sigma_S_per_m').copy()
x=f.delta_sigma_S_per_m.to_numpy();y=f.delta_Z_ohm.to_numpy();lin=f.linear_delta_Z_ohm.to_numpy()
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
axes[0].plot(x,lin*1e6,color='#202f3d',label='Линейный прогноз',lw=1.8)
axes[0].scatter(x,y*1e6,s=65,facecolors='white',edgecolors='#b6544b',linewidths=1.8,label='Прямое решение',zorder=3)
axes[0].set(xlabel='Δσ, См/м',ylabel='ΔZ, мкОм',title='а. Полный локальный отклик');axes[0].legend(fontsize=10)
axes[1].axhline(0,color='black',lw=.8)
axes[1].scatter(x,(y-lin)*1e9,s=55,color='#b6544b')
axes[1].set(xlabel='Δσ, См/м',ylabel='ΔZ прямой − ΔZ линейный, нОм',title='б. Увеличенный масштаб расхождений')
for ax in axes:
    ax.set_xticks([-.001,-.0005,0,.0005,.001]);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:ru(v,4)));ax.tick_params(axis='x',labelsize=9);ax.grid(alpha=.2)
finish(9,fig)
'''
BLOCKS['fd_table'] = r'''
df=pd.DataFrame({'Δσ, См/м':f.delta_sigma_S_per_m,'Прямой ΔZ, мкОм':f.delta_Z_ohm*1e6,
 'Линейный ΔZ, мкОм':f.linear_delta_Z_ohm*1e6,'|Расхождение|, нОм':abs(f.delta_Z_ohm-f.linear_delta_Z_ohm)*1e9,
 '|Расхождение| / |линейный ΔZ|, %':abs((f.delta_Z_ohm-f.linear_delta_Z_ohm)/f.linear_delta_Z_ohm)*100,
 'Относительная невязка системы':f.relative_residual})
formats={c:lambda x:ru(x,4) for c in df.columns};formats['Относительная невязка системы']=sci
table(df,formats)
'''
BLOCKS['fd_interpretation'] = r'''
percent_error=abs((f.delta_Z_ohm-f.linear_delta_Z_ohm)/f.linear_delta_Z_ohm)*100
local=verification[verification.check=='local_central_difference'].copy()
display(Markdown(f'На четырёх проверенных возмущениях абсолютное расхождение с линейным прогнозом составляет от **{ru(abs(f.delta_Z_ohm-f.linear_delta_Z_ohm).min()*1e9,2)} до {ru(abs(f.delta_Z_ohm-f.linear_delta_Z_ohm).max()*1e9,2)} нОм**, или от **{ru(percent_error.min(),3)} до {ru(percent_error.max(),3)}%** модуля малого линейного отклика. Обе центральные разности удовлетворяют заданному критерию. Эти проценты относятся к прогнозу данной локальной пробы; невязка системы в последнем столбце таблицы — другой показатель.'))
display(Markdown('Центральные разности при шагах 0,001 и 0,0005 См/м дают соответственно '+', '.join(ru(x,7) for x in local.actual)+' Ом/(См/м), тогда как производная карты равна '+ru(float(local.reference.iloc[0]),7)+' Ом/(См/м).'))
'''
BLOCKS['channel_results'] = r'''
c=channels.set_index('channel').loc[['standard','adjacent','cross']]
values=c.Z_ohm.to_numpy();names=['А','Б','В']
fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
col=['#6585a1','#ba6b55','#448c82']
b=axes[0].bar(names,values,color=col);axes[0].set(ylabel='Z, Ом',title='а. Знаковый передаточный импеданс',ylim=(-15,230))
axes[0].bar_label(b,labels=[ru(values[0],3),'≈ −0,000000440',ru(values[2],3)],padding=5,fontsize=9)
axes[1].bar(names,abs(values),color=col);axes[1].set_yscale('log');axes[1].set(ylabel='|Z|, Ом',title='б. Модули на логарифмической шкале',ylim=(1e-8,1e3))
axes[1].axhline(float(c.normalization_floor_ohm.iloc[0]),ls='--',color='black',lw=1,label='Порог относительной нормировки')
axes[1].legend(fontsize=9)
for ax in axes:ax.set_xlabel('Комбинация');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
finish(10,fig)
'''
BLOCKS['channel_interpretation'] = r'''
z=channels.set_index('channel').Z_ohm
identity_error=abs(z['cross']-z['standard']-z['adjacent'])
display(Markdown(f'Расчётный остаток тождества (10) составляет **{ru(identity_error*1e9,3)} нОм**. Модуль импеданса комбинации Б равен **{ru(abs(z["adjacent"])*1e6,4)} мкОм**. Близость А и В поэтому согласуется с малостью добавочного канала Б.'))
adj=summary[(summary.channel=='adjacent')&(summary.tissue=='soft_tissue')].iloc[0]
display(Markdown(f'При этом малый интегральный отклик не тождествен отсутствию чувствительности в каждой точке. В мягкотканном фоне комбинации Б показатель знаковой компенсации составляет **{ru(adj.cancellation_ratio*100,2)}%**. Интеграл модуля равен **{sci(adj.absolute_integral)} Ом/(См/м)**, а знаковый интеграл — **{sci(adj.dZ_dsigma)} Ом/(См/м)**. Здесь существенная часть противоположных вкладов погашается при равномерном изменении. Это дополнительная причина сохранять знаковую карту даже для малого итогового канала.'))
'''
BLOCKS['qc'] = r'''
qc=verification.groupby('check',as_index=False).agg(Число=('passed','size'),Выполнены=('passed','all'))
names={'map_integral':'Интеграл карты и матричная производная','frozen_40_05_regression':'Воспроизведение опорного расчёта 40.05','four_electrode_channel_identity':'Тождество каналов для поэлементных производных','local_matrix_vs_map':'Независимая локальная матрица и карта','local_central_difference':'Центральная конечная разность'}
qc['Проверка']=qc['check'].map(names);qc['Выполнены']=qc['Выполнены'].map({True:'Да',False:'Нет'})
table(qc[['Проверка','Число','Выполнены']])
'''
BLOCKS['provenance'] = r'''
display(Markdown('**Паспорт этой редакции.** Использованы сохранённые данные трёх комбинаций, '+f'{len(volumes):,}'.replace(',',' ')+' элементов и '+str(len(verification))+' численных сопоставлений основной серии. Проверка контрольных сумм пройдена; все десять рисунков построены из указанных данных или явно обозначенных схем.'))
'''

def build():
    text=TEXT.read_text(encoding='utf-8-sig')
    segments=re.split(r'<!-- CODE:(\w+) -->',text)
    cells=[]
    for i,part in enumerate(segments):
        if i%2:
            if part not in BLOCKS:raise ValueError(f'Unknown report block: {part}')
            cells.append(nbf.v4.new_code_cell(BLOCKS[part].strip(),metadata={'report_block':part}))
        elif part.strip():cells.append(nbf.v4.new_markdown_cell(part.strip()))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'report_revision':'scientific_expansion_20260910'})
    nbf.write(nb,TARGET)
    print(TARGET)

if __name__=='__main__':build()
