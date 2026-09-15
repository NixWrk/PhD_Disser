"""Reader figures for 20.17, each built from explicitly named numerical fields."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from halfspace_study import OUT,SOURCE
from halfspace_25d import load_contours

ORDER=['reference','uniform_transverse','variable_transverse','m3h','m4h','planar_projected','planar_nominal']
LABEL={'reference':'КТ/FEM','uniform_transverse':'М3, тело по КТ','variable_transverse':'М4, тело по КТ','m3h':'М3-H','m4h':'М4-H','planar_projected':'Плоская, проекция','planar_nominal':'Плоская, номинал'}
COLORS=['#263846','#6884a9','#159c8a','#8054ac','#d47b30','#ad4848','#888888']
def read(name):return pd.read_csv(OUT/(name+'.csv'))
def labels(ax,order=ORDER,rotation=25):
    ax.set_xticks(range(len(order)),[LABEL[x] for x in order],rotation=rotation,ha='right')
def clean(ax):
    ax.grid(alpha=.20,axis='y');ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
def paired(ax,frame,a,b,order=ORDER,ylabel='Ошибка, %'):
    x=np.arange(len(order));d=frame.reindex(order)
    ax.bar(x-.18,d[a],.36,label='ρ₁',color='#4178a5');ax.bar(x+.18,d[b],.36,label='ρ₂',color='#dc913a')
    labels(ax,order);ax.set_ylabel(ylabel);clean(ax);ax.legend(frameon=False)

def geometry():
    fig=plt.figure(figsize=(16,8),layout='constrained')
    p=read('electrodes_projected');ec=p[p.L_mm==140]
    for k in [0,1]:
        ax=fig.add_subplot(1,2,k+1,projection='3d')
        if k==0:
            polys=[]
            for c in load_contours(OUT/'geometry_contours.npz'):
                c=c*1000
                if np.linalg.norm(c[0]-c[-1])<1e-8:c=c[:-1]
                for a,b in zip(c,np.roll(c,-1,axis=0)):polys.append([[a[0],-100,a[1]],[b[0],-100,b[1]],[b[0],100,b[1]],[a[0],100,a[1]]])
            colour='#7c64ae';lims=((-85,175),(-110,110),(0,250));title='М3-H: одинаковое сечение\nПоказано окно по t, торцевых границ нет'
        else:
            z=np.load(OUT/'m4h_field025_surface.npz');v=z['vertices']*1000;polys=v[z['faces']]
            colour='#229fa8';lims=((-100,205),(-180,80),(0,310));title='М4-H: переменное сечение\nПолная замкнутая лёгочная область'
        ax.add_collection3d(Poly3DCollection(polys,facecolor=colour,edgecolor='none',alpha=.20))
        (a,b),(c,d),_=lims
        ax.add_collection3d(Poly3DCollection([[[a,c,0],[b,c,0],[b,d,0],[a,d,0]]],facecolor='#bec5ca',alpha=.13,edgecolor='#8b949e',linewidth=.7))
        ax.scatter(p.s_mm,p.t_mm,np.zeros(len(p)),s=9,c='#64707a',alpha=.55,depthshade=False)
        for j,(x,y) in enumerate(ec[['s_mm','t_mm']].to_numpy()):
            col=['#d84040','#d4a000','#229953','#2461bd'][j];lab=['I+','V+','V−','I−'][j]
            ax.scatter([x],[y],[0],s=38,c=col,depthshade=False,zorder=50,label=lab)
            ax.text(x,y,-10,lab,color=col,fontsize=10,zorder=51)
        ax.plot([0,0],[0,0],[0,67],'--',c='#333333',lw=1.3)
        ax.text(3,0,40,'h(0) ≈ 67 мм',fontsize=9,color='#222222')
        ax.set(xlim=lims[0],ylim=lims[1],zlim=lims[2],xlabel='s, мм',ylabel='t, мм',zlabel='d, мм (внутрь)')
        ax.invert_zaxis();ax.set_box_aspect([b-a,d-c,lims[2][1]]);ax.view_init(elev=20,azim=-60);ax.set_title(title,pad=18)
        ax.tick_params(labelsize=8)
    fig.suptitle('Рисунок 1. Геометрия двух моделей и плоская наружная граница',fontsize=15)
    return fig

def convergence():
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');df=read('fourier_convergence');base=df[(df.segments==360)&(df.k_nodes==64)].set_index(['ratio','L_mm'])
    for n,nk in [(120,32),(240,48)]:
        d=df[(df.segments==n)&(df.k_nodes==nk)&(df.ratio!=1)].set_index(['ratio','L_mm']);b=base.loc[d.index]
        for j,col in enumerate(['f','df_dlogq']):
            val=4*abs(d[col]-b[col]) if j==0 else 100*abs(d[col]/b[col]-1)
            v=val.groupby(level='L_mm').max();axs[0,j].plot(v.index,v,'o-',label=f'{n} отрезков, {nk} узла')
    extra=read('fourier_independent_qc');d=extra[extra.control=='segment_refinement_at_fixed_k']
    for j in [0,1]:
        val=4*abs(d.delta_f_vs_primary_360_64) if j==0 else 100*abs(d.delta_df_dlogq_vs_primary_360_64/d.primary_360_64_df_dlogq)
        v=val.groupby(d.L_mm).max();axs[0,j].plot(v.index,v,'o-',label='480 отрезков, 64 узла')
    d=read('field_resolution_convergence');a=d[d.tolerance==.25].set_index(['ratio','L_mm']);b=d[d.tolerance==.15].set_index(['ratio','L_mm'])
    for j,col in enumerate(['f','df_dlogq']):
        val=4*abs(a[col]-b[col]) if j==0 else 100*abs(a[col]/b[col]-1)
        v=val.groupby(level='L_mm').max();axs[1,j].plot(v.index,v,'o-',c='#db8a36',label='4284 и 7520 треугольников')
    for row,name in enumerate(['М3-H: относительно 360 отрезков / 64 узлов','М4-H: уточнение разбиения поверхности']):
        for j in [0,1]:
            axs[row,j].set_title(name);axs[row,j].set_xlabel('Размер сборки L, мм')
            axs[row,j].set_ylabel('Максимальное |ΔZ| при ρ₁=4, Ом' if j==0 else 'Максимальное расхождение J₂, %')
            axs[row,j].legend(fontsize=9,frameon=False);clean(axs[row,j])
    fig.suptitle('Рисунок 2. Численная устойчивость Z и чувствительности к ρ₂ при q=4 и q=12',fontsize=14)
    return fig

def homogeneous():
    d=read('homogeneous_body_control');fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    ax.plot(d.L_mm,d.relative_error_pct,'o-',c='#8054ac');ax.axhline(0,color='#666',lw=1)
    for row in d.itertuples():ax.annotate(f'{row.relative_error_pct:.1f}%',(row.L_mm,row.relative_error_pct),xytext=(0,8),textcoords='offset points',ha='center',fontsize=10)
    ax.set(xlabel='Размер сборки L, мм',ylabel='(Z полупространства / Z КТ-FEM − 1), %',ylim=(-25,3))
    clean(ax);fig.suptitle('Рисунок 3. Однородный контроль: ρ₁=ρ₂, электрической границы лёгкого нет')
    return fig

def inverse():
    d=read('inverse_recovery');fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    true=d[d.model=='reference'].sort_values('case');xlabels=[f'{r.rho1_true:g}/{r.rho2_true:g}' for r in true.itertuples()]
    for j in [0,1]:
        z=d.pivot(index='model',columns='case',values=f'error{j+1}_pct').reindex(ORDER).abs();im=axs[0,j].imshow(z,aspect='auto',cmap='magma',vmin=0)
        axs[0,j].set_yticks(range(len(ORDER)),[LABEL[n] for n in ORDER]);axs[0,j].set_xticks(range(16),xlabels,rotation=55,ha='right')
        axs[0,j].set_xlabel('Заданные ρ₁/ρ₂, Ом·м');axs[0,j].set_title(f'Абсолютная относительная ошибка ρ{j+1}, %');fig.colorbar(im,ax=axs[0,j],shrink=.8)
    agg=d.groupby('model').agg(e1=('error1_pct',lambda x:np.median(abs(x))),e2=('error2_pct',lambda x:np.median(abs(x))),bound=('on_bound','sum'))
    paired(axs[1,0],agg,'e1','e2');axs[1,0].set_title('Медианная ошибка в 16 состояниях')
    axs[1,1].bar(range(len(ORDER)),agg.reindex(ORDER).bound,color=COLORS);labels(axs[1,1]);axs[1,1].set(ylabel='Число решений на границе',ylim=(0,18),title='Достижение границы: диагностический признак');clean(axs[1,1])
    fig.suptitle('Рисунок 4. Точность восстановления и ограничения параметров',fontsize=15)
    return fig

def pulse():
    fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained');d=read('jacobian_comparison')
    use=['reference','variable_transverse','m3h','m4h','planar_projected']
    for j,key in enumerate(['known_rho_J_relative_error','estimated_rho_J_relative_error']):
        a=d.groupby(['model','tissue'])[key].median().unstack()*100;a.columns=['e1','e2']
        paired(axs[0,j],a,'e1','e2',use,'Относительное расхождение столбца J, %')
        axs[0,j].set_title('Чувствительности при известных ρ' if j==0 else 'Чувствительности при восстановленных ρ')
    p=read('fem_pulse_nonlinearity')
    names=['delta_rho1_plus_0p0004','delta_rho2_plus_0p016','both_plus','both_minus','enlarged_control_10x']
    v=[100*max(abs(p[p.scenario==x].nonlinear_minus_linear))/max(abs(p[p.scenario==x].nonlinear_response)) for x in names]
    axs[1,0].bar(range(5),v,color='#d98a34')
    axs[1,0].set_xticks(range(5),['Только ρ₁','Только ρ₂','Оба +','Оба −','Амплитуда ×10'],rotation=20,ha='right')
    axs[1,0].set(ylabel='Ошибка линейного отклика, %',title='Независимый нелинейный FEM-контроль')
    for i,y in enumerate(v):axs[1,0].text(i,y,f'{y:.3f}%',ha='center',va='bottom',fontsize=9)
    clean(axs[1,0]);p=read('pulse_nonlinear_inverse');a=p[p.scenario=='both_plus'].set_index('model')
    paired(axs[1,1],a,'error1_pct','error2_pct',use,'Знаковая ошибка пульсового Δρ, %');axs[1,1].set_title('Оба тканевых пульсовых вклада восстанавливаются')
    fig.suptitle('Рисунок 5. Проверка якобиана и перехода к пульсовому восстановлению',fontsize=15)
    return fig

def scenarios():
    fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    use=['reference','variable_transverse','m3h','m4h'];d=read('placement_inverse')
    scenarios=['s_minus_2mm','s_plus_2mm','t_minus_2mm','t_plus_2mm']
    z=d[d.scenario.isin(scenarios)].pivot(index='model',columns='scenario',values='error2_pct').reindex(index=use,columns=scenarios)
    im=axs[0,0].imshow(z,aspect='auto',cmap='RdBu_r',vmin=-60,vmax=60)
    axs[0,0].set_yticks(range(4),[LABEL[x] for x in use]);axs[0,0].set_xticks(range(4),['s: −2 мм','s: +2 мм','t: −2 мм','t: +2 мм'],rotation=20)
    for a in range(4):
        for b in range(4):axs[0,0].text(b,a,f'{z.iloc[a,b]:.1f}%',ha='center',va='center',color='white' if abs(z.iloc[a,b])>35 else 'black')
    axs[0,0].set_title('Ошибка ρ₂ при неизвестном сдвиге точечных контактов');fig.colorbar(im,ax=axs[0,0],shrink=.8)
    d=read('linked_scenarios');sc=['h_minus2_deep','fixed_deep','h_plus2_deep','h_minus1_quiet','fixed_quiet','h_plus1_quiet']
    names=['Глубокое: −2','Глубокое: 0','Глубокое: +2','Обычное: −1','Обычное: 0','Обычное: +1']
    for n in use:
        q=d[(d.model==n)].set_index('scenario').reindex(sc)
        axs[0,1].plot(range(6),q.delta2_error_pct,'o-',label=LABEL[n])
    axs[0,1].set_xticks(range(6),names,rotation=25,ha='right');axs[0,1].set(ylabel='Ошибка дыхательного Δρ₂, %',title='Изменение h задано в FEM и игнорируется при инверсии');axs[0,1].legend(fontsize=9,frameon=False);clean(axs[0,1])
    x=np.arange(len(use))
    for k,(s,label) in enumerate([('fixed_deep','ρ₁ действительно постоянна'),('rho1_deep','ρ₁ увеличена на 1%')]):
        q=d[d.scenario==s].set_index('model').reindex(use)
        axs[1,0].bar(x+(k-.5)*.35,q.shared_delta2_error_pct,.35,label=label)
    labels(axs[1,0],use);axs[1,0].set(ylabel='Ошибка дыхательного Δρ₂, %',title='Принудительно общая ρ₁ двух дыхательных состояний');axs[1,0].legend(fontsize=9,frameon=False);clean(axs[1,0])
    d=read('perturbations');g=d.groupby('model').agg(e1=('change1_pct',lambda x:max(abs(x))),e2=('change2_pct',lambda x:max(abs(x))))
    paired(axs[1,1],g,'e1','e2',use,'Максимальное изменение оценки, %');axs[1,1].set_title('Возмущение одного импеданса на ±1%')
    fig.suptitle('Рисунок 6. Ошибки геометрии, дыхательные допущения и устойчивость оценок',fontsize=15)
    return fig

def timing():
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');d=read('direct_timings');f=read('fem_benchmark')
    cols=list(f.columns);key='factor_solve_derivatives_seconds'
    if key not in cols:key=next(c for c in cols if 'factor_solve_derivatives' in c)
    vals=[d[d.model==m].seconds.median() for m in ['m3h','m4h']]+[f[key].median()]
    axs[0,0].bar(range(3),vals,color=[COLORS[3],COLORS[4],COLORS[0]])
    axs[0,0].set_xticks(range(3),['М3-H','М4-H','КТ/FEM']);axs[0,0].set(ylabel='Время, с',title='Девять Z и якобиан: новая факторизация')
    for i,y in enumerate(vals):axs[0,0].text(i,y,f'{y:.3f}',ha='center',va='bottom')
    prep=d.groupby('model').matrix_preparation_seconds.first()
    axs[0,1].bar([0,1],prep.reindex(['m3h','m4h']),color=[COLORS[3],COLORS[4]])
    axs[0,1].set_xticks([0,1],['М3-H','М4-H']);axs[0,1].set(ylabel='Время, с',title='Подготовка BEM-оператора для заданной геометрии')
    axs[0,1].text(.5,.94,'Построение исходных матриц FEM здесь не измерялось',transform=axs[0,1].transAxes,ha='center',va='top',fontsize=9)
    d=read('surrogate_timings');use=['reference','variable_transverse','m3h','m4h','planar_projected']
    for j,(task,scale,label) in enumerate([('field_and_jacobian',1e6,'Время, мкс'),('inverse',1e3,'Время, мс')]):
        v=d[d.task==task].groupby('model').seconds.median().reindex(use)*scale
        axs[1,j].bar(range(len(use)),v,color=[COLORS[ORDER.index(n)] for n in use]);labels(axs[1,j],use);axs[1,j].set_ylabel(label)
        axs[1,j].set_title('Z и якобиан из готовой библиотеки' if j==0 else 'Обратное восстановление из готовой библиотеки')
    for ax in axs.flat:clean(ax)
    fig.suptitle('Рисунок 7. Раздельная стоимость подготовки, прямого и обратного расчёта',fontsize=15)
    return fig

def cem():
    d=read('cem_baseline_check');ct=pd.read_csv(SOURCE/'cem_baseline.csv');ct=ct[ct.model=='reference'].sort_values('L_mm')
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    axs[0].plot(ct.L_mm,ct.Z,'ko-',label='КТ/FEM, исходные площадки')
    for ring in [1,2,3]:
        a=d[d.rings==ring].sort_values('L_mm');n=int(a.panels_per_electrode.iloc[0])
        axs[0].plot(a.L_mm,a.Z,'o-',label=f'М4-H, {n} панелей на электрод')
        axs[1].plot(a.L_mm,100*(a.Z/ct.Z.to_numpy()-1),'o-',label=f'{n} панелей')
    axs[0].set(ylabel='Передаточный импеданс Z, Ом',title='Известное состояние ρ₁=4, ρ₂=16 Ом·м')
    axs[1].set(ylabel='Расхождение с КТ/FEM, %',title='Полупространство с равноплощадными дисками')
    for ax in axs:ax.set_xlabel('Размер сборки L, мм');ax.legend(frameon=False,fontsize=9);clean(ax)
    fig.suptitle('Рисунок 8. Отдельная проверка полной электродной модели с конечными контактами',fontsize=14)
    return fig

def show(number):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11})
    fig=[geometry,convergence,homogeneous,inverse,pulse,scenarios,timing,cem][number-1]()
    fig.savefig(OUT/f'render_figure_{number}.png',dpi=150,bbox_inches='tight')
    plt.show();plt.close(fig)
