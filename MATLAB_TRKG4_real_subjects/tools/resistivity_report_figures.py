"""Reader tables and fourteen figures for report 40.07; no FEM execution."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.colors import TwoSlopeNorm
from IPython.display import display,Markdown,Image,HTML
P=['soft','heart','lung']
L={'soft':'Мягкие ткани','heart':'Сердечная область','lung':'Лёгкие'}
C={'soft':'#526f94','heart':'#b85656','lung':'#278a80'}
K={'point_disc_5mm':'Диск 5','circumferential_ring':'Манжета 5','wide_cuff_equivalent_area':'Манжета 15','outer_planes_inner_rings':'Плоскости + манжеты'}
KC=dict(zip(K,['#7e65ad','#3978ad','#369380','#c78031']))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ru(x,n=3):return f'{float(x):.{n}f}'.replace('.',',').replace('-','−')
def md(s):display(Markdown(s))
def table(x):display(HTML(x.style.hide(axis='index').set_properties(**{'text-align':'left'}).to_html()))
def load_report(root):
    b=root/'MATLAB_TRKG4_real_subjects/output/exploratory/resistivity_sensitivity_20260910';a=b/'analysis'
    summary=json.loads((a/'analysis_summary.json').read_text(encoding='utf-8'))
    for name,h in summary['output_sha256'].items():assert sha(a/name)==h
    profile=json.loads((b/'literature/frozen/rho_profile.json').read_text(encoding='utf-8'))
    assert sha(b/'literature/frozen/rho_profile.json')==summary['profile_sha256']
    r=SimpleNamespace(root=root,base=b,summary=summary,profile=profile,data=pd.read_csv(a/'states_rho.csv'),cards=pd.read_csv(a/'assembly_scorecards.csv'),audit=pd.read_csv(a/'coverage_audit.csv'),end=pd.read_csv(a/'endpoint_excursions.csv'))
    r.ids=sorted(r.data.montage.unique());r.names={};r.kind={}
    for mid in r.ids:
        d,l,k=re.fullmatch(r'in(\d+)_out(\d+)_(.*)',mid).groups();r.names[mid]=f'{K[k]}; {d}/{l}';r.kind[mid]=k
    r.rep='in20_out80_circumferential_ring';r.figures=b/'report/figures';r.figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'figure.dpi':115,'axes.spines.top':False,'axes.spines.right':False,'axes.formatter.useoffset':False})
    return r
def finish(r,n,f):
    for ext in ['png','svg']:f.savefig(r.figures/f'figure_{n:02d}.{ext}',dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(f);display(Image(filename=str(r.figures/f'figure_{n:02d}.png')))
def abstract(r):
    s=r.summary; verdict='не прошла заданный контроль' if not s['interior_check_pass'] else 'прошла контроль в выбранных центрах'
    md(f"**Основной результат.** Проанализированы {s['coarse_states']} исходных и {s['new_fem_states']} новых FEM-состояния. Исходная сетка {verdict}: максимальная относительная ошибка интерполяции производной составила {ru(100*s['max_derivative_relative_error'],1)}%. Численные характеристики сборок получены, но их непрерывные диапазоны и устойчивость окончательного выбора ещё не подтверждены. При принятом литературном масштабе преобладает чувствительность к объединённому мягкотканному фону.")
def provenance(r):
    table(pd.DataFrame([
      ['Анатомия Nix','КТ на вдохе и ручная сегментация','Перенесена из серии 40.05'],
      ['Пространственная сетка','752 827 узлов; 4 358 825 тетраэдров','Одна фиксированная сетка'],
      ['Продолжения рук','Цилиндры радиусом 30 мм','Геометрическое допущение'],
      ['Мягкотканный фон','Маска фона, включая модельные руки','Один изменяемый параметр'],
      ['Сердечная область','Целая маска сердца','Эффективная смесь крови и миокарда'],
      ['Лёгкие','Маска лёгких','Один изменяемый параметр'],
      ['Кости',f"ρ = {ru(r.profile['bone_fixed_rho_ohm_m'])} Ом·м",'Фиксированы'],
      ['Контакт','Удельный импеданс 1,59155·10⁻⁴ Ом·м²','Одинаковый фиксированный параметр'],
      ['Частота','50 кГц','Действительная скалярная проводимость']],columns=['Вход','Значение или происхождение','Статус']))
def montages(r):
    f,axs=plt.subplots(4,1,figsize=(11,8),layout='constrained')
    for ax,(kind,label) in zip(axs,K.items()):
        ax.axis('off');ax.set(xlim=(-5,5),ylim=(-.8,1))
        ax.add_patch(Rectangle((-1.05,-.32),2.1,.64,facecolor='#e4e8ed',edgecolor='#9aa7b3'));ax.text(0,0,'Туловище',ha='center',va='center')
        for x in [-4.45,1.05]:ax.add_patch(Rectangle((x,-.12),3.4,.24,color='#dce2e8'))
        for j,x in enumerate([-3.9,-2,2,3.9]):
            if kind=='point_disc_5mm':ax.add_patch(Circle((x,.12),.095,color=KC[kind]))
            elif kind=='outer_planes_inner_rings' and j in [0,3]:ax.plot([x,x],[-.3,.3],lw=4,color=KC[kind])
            else:
                w=.32 if kind=='wide_cuff_equivalent_area' else .11;ax.add_patch(Rectangle((x-w/2,-.24),w,.48,color=KC[kind]))
            ax.text(x,.53,['I+','V+','V−','I−'][j],ha='center',fontweight='bold')
        ax.text(-4.8,.87,label,fontweight='bold',color=KC[kind])
    for start,end,text in [(1.05,2,'d = 20 или 40 мм'),(2,3.9,'l = 80 или 100 мм')]:
        axs[-1].annotate('',xy=(end,-.46),xytext=(start,-.46),arrowprops={'arrowstyle':'<->','color':'#444'});axs[-1].text((start+end)/2,-.72,text,ha='center',fontsize=8)
    finish(r,1,f)
def ranges(r):
    refs={'soft':'Мышца — жир','heart':'Кровь — миокард','lung':'Спавшееся — наполненное лёгкое'}
    table(pd.DataFrame([[L[k],ru(v[0]),ru(v[1]),ru(v[1]-v[0]),refs[k]] for k,v in r.profile['bounds'].items()],columns=['Компартмент','ρ min, Ом·м','ρ max, Ом·м','Δρ, Ом·м','Ориентиры']))
def range_figure(r):
    f,ax=plt.subplots(figsize=(10,3.7),layout='constrained')
    for j,k in enumerate(P):
        lo,hi=r.profile['bounds'][k];ax.plot([lo,hi],[j,j],lw=7,solid_capstyle='round',color=C[k])
        for v in [lo,hi]:ax.text(v,j-.18,ru(v,2),ha='center',fontsize=9)
    bone=r.profile['bone_fixed_rho_ohm_m'];ax.scatter([bone],[3],s=80,c='#9b8158',marker='D');ax.text(bone,2.8,ru(bone,2),ha='center')
    ax.set(yticks=range(4),yticklabels=[L[k] for k in P]+['Кости (фиксированы)'],xlabel='Удельное сопротивление ρ, Ом·м',ylim=(3.5,-.6),xlim=(0,54));ax.grid(axis='x',alpha=.2);finish(r,2,f)
def axes(r):
    f,axs=plt.subplots(3,1,figsize=(10,5.8),layout='constrained')
    for ax,k in zip(axs,P):
        x=np.array(r.profile['original_rho_axes'][k]);mid=(x[:-1]+x[1:])/2
        ax.plot(x,np.zeros(len(x)),color='#adb8c3');ax.scatter(x,np.zeros(len(x)),s=60,label='Исходные узлы',color='#3978ad');ax.scatter(mid,np.full(len(mid),-.3),marker='D',s=50,label='Середины интервалов',color='#d47e27')
        for v in x:ax.text(v,.13,ru(v),ha='center',fontsize=9)
        for v in mid:ax.text(v,-.51,ru(v),ha='center',fontsize=9)
        ax.set(title=L[k],yticks=[],ylim=(-.75,.45),xlabel='ρ, Ом·м');ax.spines['left'].set_visible(False)
    f.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,fontsize=9);finish(r,3,f)
def grid(r):
    f,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    low,high=r.profile['bounds']['lung'];middle=(low+high)/2
    data=r.data[r.data.montage==r.rep]
    for ax,rho in zip(axs,[low,middle,high]):
        ischeck=np.isclose(rho,middle);color='#d47e27' if ischeck else '#3978ad'
        g=data[np.isclose(data.rho_lung,rho)]
        ax.scatter(g.rho_soft,g.rho_heart,c=color,s=65,marker='D' if ischeck else 'o')
        for x in r.profile['original_rho_axes']['soft']:ax.axvline(x,color='#ccd5dd',lw=.7,zorder=0)
        for y in r.profile['original_rho_axes']['heart']:ax.axhline(y,color='#ccd5dd',lw=.7,zorder=0)
        ax.set(title=f'ρ лёгких = {ru(rho,3)} Ом·м'+('\n4 независимые проверки' if ischeck else '\n9 исходных состояний'),xlabel='ρ мягких тканей, Ом·м',ylabel='ρ сердца, Ом·м',xlim=(1.5,24.5),ylim=(1.15,5.4))
    finish(r,4,f)
def z_curves(r):
    import itertools
    from matplotlib.lines import Line2D
    f,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');coarse=r.data[r.data['sample']=='coarse']
    for ax,(h,l) in zip(axs.flat,itertools.product(r.profile['bounds']['heart'],r.profile['bounds']['lung'])):
        for mid,g in coarse.groupby('montage'):
            v=g[np.isclose(g.rho_heart,h)&np.isclose(g.rho_lung,l)].sort_values('rho_soft');ax.plot(v.rho_soft,v.Z_ohm,'o-',markersize=3,lw=1,color=KC[r.kind[mid]],alpha=.8)
        ax.set(title=f'ρ сердца = {ru(h,2)}; ρ лёгких = {ru(l,2)} Ом·м',xlabel='ρ мягких тканей, Ом·м',ylabel='Z, Ом');ax.grid(alpha=.2)
    f.legend([Line2D([0],[0],color=KC[k]) for k in K],list(K.values()),loc='outside lower center',ncol=4,fontsize=9);finish(r,5,f)
def derivative_curves(r):
    f,axs=plt.subplots(1,3,figsize=(14,5.2),layout='constrained');g=r.data[(r.data.montage==r.rep)&(r.data['sample']=='coarse')]
    for ax,k in zip(axs,P):
        others=[j for j in P if j!=k]
        for background,h in g.groupby(['rho_'+j for j in others]):
            h=h.sort_values('rho_'+k);ax.plot(h['rho_'+k],h['S_'+k],'o-',lw=1,markersize=3,label='/'.join(ru(v,1) for v in background))
        ax.set(title=L[k],xlabel='Собственное ρ, Ом·м',ylabel='∂Z/∂ρ, м⁻¹');ax.grid(alpha=.18)
        ax.legend(title='Фон: '+('/'.join('мягк.' if j=='soft' else 'серд.' if j=='heart' else 'лёгк.' for j in others)),fontsize=7,title_fontsize=8,loc='upper right')
    finish(r,6,f)
def intervals(r,ax,k,low,high,xlabel,labels=True,scale=1):
    g=r.cards[r.cards.compartment==k].set_index('montage').loc[r.ids]
    for j,mid in enumerate(r.ids):
        vals=g.loc[mid,[low,high]].to_numpy(float)*scale;ax.plot(vals,[j,j],lw=2.4,color=KC[r.kind[mid]]);ax.scatter(vals,[j,j],s=10,color=KC[r.kind[mid]])
    ax.set(yticks=range(len(r.ids)),yticklabels=[r.names[m] for m in r.ids] if labels else [],xlabel=xlabel,title=L[k]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',labelsize=8)
def s_ranges(r):
    f,axs=plt.subplots(1,3,figsize=(13,7.8),layout='constrained')
    for j,(ax,k) in enumerate(zip(axs,P)):intervals(r,ax,k,'S_min','S_max','∂Z/∂ρ, м⁻¹',j==0)
    finish(r,7,f)
def result_summary(r):
    rows=[]
    for k in P:
        c=r.cards[r.cards.compartment==k];rows.append([L[k],f'{ru(c.S_min.min(),4)}–{ru(c.S_max.max(),4)}',f'{ru(100*c.Q_min.min(),3)}–{ru(100*c.Q_max.max(),3)}',f'{ru(c.E_min.min(),3)}–{ru(c.E_max.max(),3)}'])
    table(pd.DataFrame(rows,columns=['Компартмент','∂Z/∂ρ, м⁻¹','Q, %','E, Ом']))
    assert (r.cards.S_min>0).all()
    md('**Результат модели.** Таблица объединяет все сборки и рассчитанные состояния. Во всех найденных узлах производные трёх компартментов положительны; это наблюдение по узлам, а не доказательство монотонности на всей непрерывной области. Мягкотканный фон имеет наибольшую чувствительность при принятом определении маски и литературном масштабе.')
def bq_ranges(r):
    f,axs=plt.subplots(2,3,figsize=(14,13),layout='constrained')
    for j,k in enumerate(P):
        intervals(r,axs[0,j],k,'B_min','B_max','B, Ом',j==0);intervals(r,axs[1,j],k,'Q_min','Q_max','Q, %',j==0,100)
    finish(r,8,f)
def score_table(r):
    rows=[]
    for i,mid in enumerate(r.ids,1):
        c=r.cards[(r.cards.montage==mid)&(r.cards.compartment=='heart')].iloc[0];rows.append([i,r.names[mid],f'{ru(c.S_min,4)}–{ru(c.S_max,4)}',ru(100*c.Q_min,4),ru(c.nuisance_to_target_max,1),f'{ru(c.E_min,3)}–{ru(c.E_max,3)}'])
    table(pd.DataFrame(rows,columns=['№','Сборка; d/l, мм','∂Z/∂ρ сердца, м⁻¹','min Q, %','max N','E сердца, Ом']))
def endpoints(r):
    f,axs=plt.subplots(1,3,figsize=(13,7.8),layout='constrained')
    for j,(ax,k) in enumerate(zip(axs,P)):intervals(r,ax,k,'E_min','E_max','Конечный отклик E, Ом',j==0)
    finish(r,9,f)
def finite_vs_local(r):
    f,axs=plt.subplots(1,3,figsize=(13,4.6),layout='constrained');data=r.data[(r.data.montage==r.rep)&(r.data['sample']=='coarse')]
    for ax,k in zip(axs,P):
        e=r.end[(r.end.montage==r.rep)&(r.end.compartment==k)]
        for i,(_,row) in enumerate(e.iterrows(),1):
            g=data
            for other in P:
                if other!=k:g=g[np.isclose(g['rho_'+other],row['background_rho_'+other])]
            g=g.sort_values('rho_'+k);value=row.excursion_ohm
            for color,idx,label in [('#3978ad',0,'Нижняя граница ρ'),('#d47e27',-1,'Верхняя граница ρ')]:ax.scatter(i,g.iloc[idx]['B_'+k]/value if value else np.nan,color=color,s=30,label=label if i==1 else None)
        ax.axhline(1,color='#555',ls='--',lw=1);ax.set(title=L[k],xlabel='Номер фонового сочетания',ylabel='B / E');ax.grid(alpha=.18)
    axs[0].legend(fontsize=8);finish(r,10,f)
def audit_summary(r):
    s=r.summary
    md(f"**Результат независимой проверки.** Из {s['derivative_checks']} проверок производной не прошли критерий {s['failed_derivative_checks']}; проверок избирательности с превышением допуска — {s['failed_selectivity_checks']}. Максимальная относительная ошибка производной — {ru(100*s['max_derivative_relative_error'],2)}%, базового импеданса — {ru(100*s['max_Z_relative_error'],2)}%; максимальная абсолютная ошибка избирательности — {ru(100*s['max_Q_absolute_error'],3)} процентного пункта.")
    table(pd.DataFrame([[L[k],ru(100*g.relative_error.max(),2),int((~g.derivative_pass).sum()),len(g),ru(100*g.Q_absolute_error.max(),3)] for k,g in r.audit.groupby('compartment',sort=False)],columns=['Компартмент','max ошибка S, %','Не прошли','Всего проверок','max ошибка Q, п.п.']))
    if not s['interior_check_pass']:md('**Вывод о покрытии.** Исходная сетка недостаточна для принятой точности описания производных между узлами. Сохранённые значения в самих узлах от этого не становятся неверными. Недостаточность относится к восстановлению зависимости по редкой сетке и к претензии на устойчивый выбор сборок во всём диапазоне. Требуется сгущение параметрической сетки; объявлять метод полностью проверенным или сборку окончательно оптимальной нельзя.')
def audit_error(r):
    f,axs=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,k in zip(axs,P):
        g=r.audit[r.audit.compartment==k].sort_values(['montage','state']);ax.scatter(np.arange(len(g))+1,100*g.relative_error,s=20,color=C[k]);ax.set(title=L[k],xlabel='Проверочное состояние, №',ylabel='Относительная ошибка S, %');ax.grid(alpha=.2);ax.set_title(L[k]+'\nМакс. ошибка '+ru(100*g.relative_error.max(),2)+'%')
    finish(r,11,f)
def audit_heat(r):
    f,axs=plt.subplots(1,3,figsize=(13,8),layout='constrained');norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(1.01,r.audit.error_over_tolerance.max()))
    for j,(ax,k) in enumerate(zip(axs,P)):
        grid=r.audit[r.audit.compartment==k].pivot(index='montage',columns='state',values='error_over_tolerance').loc[r.ids];im=ax.imshow(grid.to_numpy(),aspect='auto',cmap='coolwarm',norm=norm)
        yy,xx=np.where(grid.to_numpy()>1);ax.scatter(xx,yy,marker='x',c='#222',s=18);ax.set(xticks=range(4),xticklabels=range(1,5),xlabel='Центр ячейки, №',yticks=range(16),yticklabels=[r.names[m] for m in r.ids] if j==0 else [],title=L[k]);ax.tick_params(axis='y',labelsize=8)
    f.colorbar(im,ax=axs,label='Ошибка / допуск; × — превышение',shrink=.7);finish(r,12,f)
def audit_selectivity(r):
    f,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for k in P:
        g=r.audit[r.audit.compartment==k].sort_values(['montage','state']);axs[0].scatter(np.arange(len(g))+1,100*g.Q_absolute_error,s=18,label=L[k],color=C[k])
    axs[0].axhline(.5,color='#555',ls='--',label='Допуск 0,5 п.п.');axs[0].set(ylabel='Абсолютная ошибка Q, п.п.',xlabel='Проверочное состояние, №');axs[0].legend(fontsize=8)
    g=r.audit[r.audit.compartment=='soft'].sort_values(['montage','state']);axs[1].scatter(np.arange(len(g))+1,100*g.Z_relative_error,s=20,c='#526f94');axs[1].set(ylabel='Относительная ошибка Z, %',xlabel='Проверочное состояние, №')
    for ax in axs:ax.grid(alpha=.2)
    finish(r,13,f)
def pareto(r):
    from matplotlib.ticker import MaxNLocator,FuncFormatter
    f,axs=plt.subplots(1,3,figsize=(14,5.8),layout='constrained')
    for ax,k in zip(axs,P):
        g=r.cards[r.cards.compartment==k].set_index('montage');points=[]
        for n,mid in enumerate(r.ids,1):
            c=g.loc[mid];ax.scatter(c.B_min,100*c.Q_min,s=38,c=KC[r.kind[mid]]);points.append([c.B_min,100*c.Q_min,n])
        ax.set(title=L[k],xlabel='Наименьшее найденное B, Ом',ylabel='Наименьшее найденное Q, %');ax.grid(alpha=.2);ax.margins(.18)
        xlim=ax.get_xlim();ylim=ax.get_ylim();groups=[]
        for x,y,n in points:
            pos=np.array([(x-xlim[0])/np.ptp(xlim),(y-ylim[0])/np.ptp(ylim)])
            for group in groups:
                if np.max(abs(pos-group['pos']))<.04:
                    group['ids'].append(n);break
            else:groups.append({'pos':pos,'xy':(x,y),'ids':[n]})
        groups.sort(key=lambda a:a['pos'][1]);heights=[]
        for group in groups:heights.append(max(group['pos'][1]+.035,heights[-1]+.06 if heights else .05))
        if heights[-1]>.96:
            heights[-1]=.96
            for i in range(len(heights)-2,-1,-1):heights[i]=min(heights[i],heights[i+1]-.06)
        for group,ytext in zip(groups,heights):
            x=group['pos'][0];right=x<.55
            ax.annotate(', '.join(map(str,group['ids'])),group['xy'],xytext=(x+(.025 if right else -.025),ytext),textcoords='axes fraction',ha='left' if right else 'right',fontsize=8,arrowprops={'arrowstyle':'-','lw':.5,'color':'#999'})
        digits={'soft':0,'heart':5,'lung':4}[k];ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos,d=digits:ru(v,d)))
    finish(r,14,f)
def comparison(r):
    rows=[]
    for k in P:
        g=r.cards[r.cards.compartment==k];b=g.loc[g.B_min.idxmax()];q=g.loc[g.Q_min.idxmax()];rows.append([L[k],r.names[b.montage],ru(b.B_min),r.names[q.montage],ru(100*q.Q_min,4)])
    table(pd.DataFrame(rows,columns=['Цель','Наибольшее min B: сборка','min B, Ом','Наибольшее min Q: сборка','min Q, %']))
    md('**Условное сравнение.** Таблица указывает лидеров по двум раздельным сеточным критериям. Различие названий в двух столбцах означает зависимость выбора от задачи. Даже совпадение лидеров не доказывает общего оптимума: набор сборок ограничен, а непрерывное покрытие не сертифицировано. Числа служат исходными показателями для уточнения.')
    spread=pd.read_csv(r.base/'analysis/same_state_assembly_spread.csv')
    parts=[]
    for k in P:
        a=spread[spread.compartment==k].relative_spread
        parts.append(f"{L[k].lower()} — {ru(100*a.min(),2)}–{ru(100*a.max(),2)}%")
    md('**Различие при одинаковом фоне.** Размах производной между всеми 16 сборками, отнесённый к её наибольшему модулю в том же состоянии, составляет: '+ '; '.join(parts)+'. Для сердца и лёгких эти различия малы по сравнению с установленной ошибкой описания между узлами. Однако ошибки разных сборок могут быть связаны: их нельзя автоматически считать погрешностью разности между сборками. Для проверки предпочтения нужно контролировать и сами парные различия при сгущении. Лидирующее место в таблице поэтому не следует трактовать как доказанное практическое преимущество. Избирательность может улучшаться за счёт уменьшения мягкотканного влияния при почти неизменной абсолютной чувствительности к сердцу или лёгким.')

def validation(r):
    s=r.summary
    table(pd.DataFrame([
      ['Происхождение результатов','Проверены контрольные суммы исходных таблиц и всех 16 новых запусков'],
      ['Новые прямые решения',f"64 состояния; max относительная невязка {s['max_new_system_relative_residual']:.3e}"],
      ['Взаимность',f"max |Z − Z взаимн.| = {s['max_new_reciprocity_absolute_ohm']:.3e} Ом"],
      ['Переход σ → ρ','Правило цепочки; аналитические тесты последовательных и параллельных сопротивлений'],
      ['Плотность параметрической сетки','Недостаточна по проверке производных' if not s['interior_check_pass'] else 'Проверочные центры прошли; сплошной гарантии нет'],
      ['Пространственная сходимость','В этой серии не исследована'],
      ['Альтернативные тканевые профили','Расхождение отмечено; отдельная FEM-матрица не выполнена'],
      ['Физическая и экспериментальная валидация','В этой серии не выполнена']],columns=['Проверка','Результат или ограничение']))
def reproduction(r):
    md('''Воспроизводимые материалы находятся в папке проекта `MATLAB_TRKG4_real_subjects`. Профиль диапазонов: `output/exploratory/resistivity_sensitivity_20260910/literature/frozen/rho_profile.json`. Конфигурации проверочных состояний находятся в соседней папке `studies`, результаты — в `fem`, сводки — в `analysis`. Скрипт `tools/analyze_resistivity_coverage.py` проверяет завершение всех запусков и строит таблицы. Ноутбук читает готовые результаты и не запускает FEM автоматически.

Для новой геометрии сначала подготавливается совместимая конфигурация контактов. Из корня MATLAB-проекта пример генерации полной сетки:

```text
python tools/configure_resistivity_grid.py --template new_assembly.json --profile output/exploratory/resistivity_sensitivity_20260910/literature/frozen/rho_profile.json --levels 5 --output new_assembly_rho5.json
```

Затем в MATLAB вызывается `run_pipeline` с профилем `electrode_sensitivity`, созданным файлом в `StudyFile`, новой папкой `OutputDirectory`, локальными путями к Python и зависимостям. Модуль `resistivity_sensitivity_metrics.py` предоставляет `conductivity_to_resistivity` и `range_scaled_metrics`; порядок столбцов параметров должен совпадать с порядком ширин интервалов. Сводки формируются отдельно для каждой сборки, канала и целевого компартмента.

Текущие расчёты повторяются через `tools/run_resistivity_coverage.m`, если переменная `rho_python_executable` содержит путь установленного Python. Завершённые совпадающие результаты проверяются и используются повторно; незавершённая папка не принимается за готовый расчёт. HTML экспортируется из исполненного ноутбука с удалением программных входов из разметки. Исходный код остаётся в ноутбуке и файлах проекта.''')
def conclusion(r):
    s=r.summary;verdict='Исходная сетка недостаточна по заданному критерию.' if not s['interior_check_pass'] else 'Проверочные центры прошли критерий, но непрерывная область не сертифицирована.'
    md(f'''Получено количественное описание 16 сборок через производные по ρ целых компартментов, показатели при общем литературном масштабе и реальные разности импеданса между границами диапазонов. Объём компартмента учитывается полевым расчётом; дополнительное деление основной производной на объём не требуется.

В принятом профиле и рассмотренной геометрии преобладает чувствительность к объединённому мягкотканному фону. Этот вывод относится к модели исходных свойств и не устанавливает состав пульсового или дыхательного сигнала.

Независимые промежуточные решения показали максимальную ошибку интерполяции производной {ru(100*s['max_derivative_relative_error'],1)}%. {verdict} Следующий необходимый расчёт для устойчивого выбора — уточнение совместной параметрической сетки новых сборок с повторным контролем производных. Метод задан и реализован как исследовательская процедура; достаточность покрытия, физиологическая полнота интервалов и оптимальность сборки пока не доказаны.''')
