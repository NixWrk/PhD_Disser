"""Build the preparation notebook and its code-free HTML; never solve FEM."""
from pathlib import Path
import argparse,base64,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASE=ROOT/'output/exploratory/tepc_preparation_20260911'
NAME='40.09_Подготовка_чувствительности_сборок_TEPC'
ROLES=['I_plus','V_plus','V_minus','I_minus']
LABELS=['I+','V+','V−','I−']
PARAMS=['soft','heart','lung'];RUS=['Мягкие ткани','Сердце целиком','Лёгкие']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ru(value,spec='g'):return format(value,spec).replace('.',',')
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def pic(p):return f'![{p.stem}](attachment:{p.name})'

def build(base):
    sys.path.insert(0,str(ROOT/'output/exploratory/arm_sigma_20260908/python_solver_deps'))
    import numpy as np,pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nbformat
    from nbconvert import HTMLExporter
    from prepare_surface_sensitivity import verify_manifest,make_states
    manifest=verify_manifest(base);profile=read(base/'rho_profile.json')
    axes,states,pilot=make_states(profile)
    images={};figdir=base/'report/figures';figdir.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        p=figdir/(name+'.png');fig.savefig(p,dpi=115,bbox_inches='tight',facecolor='white');plt.close(fig);images[p.name]=p;return pic(p)
    coords={i['id']:np.array([read(base/i['json'])['preliminary_electrode_centers_xyz_mm'][k] for k in ROLES]) for i in manifest['inputs']}
    dt=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    verts=np.fromfile(base/manifest['source']['surface'],dtype=dt,offset=84)['v'].reshape(-1,3)
    from matplotlib.collections import PolyCollection
    triangles=verts.reshape(-1,3,3)
    middle=triangles.mean(1)
    selected=np.flatnonzero((middle[:,1]<0)&(abs(middle[:,0])<235))
    selected=selected[np.argsort(middle[selected,1])[::-1]]
    front=triangles[selected][:,:,[0,2]]
    shade=.70+.22*(middle[selected,1]-middle[selected,1].min())/np.ptp(middle[selected,1])
    facecolors=np.column_stack([shade,shade,shade,np.ones(len(shade))])
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,(mid,c) in zip(axs.flat,coords.items()):
        ax.add_collection(PolyCollection(front,facecolors=facecolors,edgecolors='none',antialiased=False,rasterized=True))
        for ids,color in [([0,3],'#c04c42'),([1,2],'#246aa2')]:ax.plot(c[ids,0],c[ids,2],color=color,lw=1.4,alpha=.8)
        for j,color in enumerate(['#c04c42','#246aa2','#246aa2','#c04c42']):
            ax.scatter(c[j,0],c[j,2],s=48,color=color,edgecolors='white',linewidths=.7,zorder=5)
            ax.annotate(LABELS[j],(c[j,0],c[j,2]),xytext=(-8,-2) if j in [0,3] else (8,0),ha='right' if j in [0,3] else 'left',va='center',textcoords='offset points',fontsize=9,color=color)
        ax.set(xlim=(-225,225),ylim=(-280,45),xlabel='X, мм',ylabel='Z, мм',title=mid.replace('_','-').upper());ax.set_aspect('equal');ax.grid(alpha=.15)
    montage_figure=save(fig,'03_montages')
    for n in ['article_scheme_1.png','article_scheme_2.png']:
        if (base/n).exists():images[n]=base/n
    article_figures='\n\n'.join(pic(images[n]) for n in ['article_scheme_1.png','article_scheme_2.png'] if n in images)
    if not article_figures:article_figures='[Опубликованные рисунки 3–4](https://www.paperhost.org/proceedings/embs/EMBC21/files/1232.pdf#page=3).'
    rows=[]
    for mid,c in coords.items():rows += [[mid.replace('_','-').upper(),LABELS[j],*[ru(v,'.3f') for v in point]] for j,point in enumerate(c)]
    coordtable=table(['Сборка','Роль','X, мм','Y, мм','Z, мм'],rows)
    ranges=[]
    for k,label in zip(PARAMS,RUS):
        lo,hi=profile['bounds'][k];ranges.append([label,ru(lo,'.4f'),ru(hi,'.4f'),ru((hi-lo)/4,'.4f'),5])
    b=profile['bone_fixed_rho_ohm_m'];ranges.append(['Кости',ru(b,'.4f'),ru(b,'.4f'),'—','фиксировано'])
    rangetable=table(['Компартмент','Нижняя граница','Верхняя граница','Шаг по ρ','Узлов'],ranges)
    fig,axs=plt.subplots(3,1,figsize=(10,5.7),constrained_layout=True)
    for ax,k,label in zip(axs,PARAMS,RUS):
        a=np.array(axes[k]);m=(a[:-1]+a[1:])/2
        ax.plot(a,np.ones(5),'o-',color='#246aa2',label='Основные узлы');ax.scatter(m,np.full(4,.93),marker='D',color='#d5923a',s=35,label='Промежуточные значения')
        ax.set(title=label,xlabel='ρ, Ом·м',ylim=(.85,1.15),yticks=[],xticks=a);ax.grid(axis='x',alpha=.2)
    axs[0].legend(loc='upper right',fontsize=8,ncol=2);axesfigure=save(fig,'05_rho_axes')
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
    for check,color,marker,label in [(False,'#246aa2','o','125 узлов'),(True,'#d5923a','D','64 центра ячеек')]:
        a=np.array([[rho[k] for k in PARAMS] for role,_,rho in states if (role=='check5')==check]);ax.scatter(a[:,0],a[:,1],a[:,2],s=18,color=color,marker=marker,alpha=.75,label=label)
    ax.set(xlabel='ρ мягких тканей, Ом·м',ylabel='ρ сердца, Ом·м',zlabel='ρ лёгких, Ом·м');ax.legend(loc='upper left');gridfigure=save(fig,'06_full_rho_grid')
    prepfile=base/'contacts/preparation.json';prep=read(prepfile) if prepfile.exists() else None
    contactfigure='График будет сформирован после геометрической подготовки.'
    status='исходные данные и программа перебора подготовлены; контактная проверка ещё не завершена'
    contactstatus='Геометрические показатели ещё не рассчитаны. Этот статус запрещает формирование готовых файлов основной серии.'
    if prep:
        status='геометрическая и матричная подготовка выполнена; контрольная и основная серии ещё не рассчитаны' if prep['status']=='passed' else 'геометрическая или матричная подготовка выявила препятствие для запуска'
        rows=[];frames=[]
        for m in prep['montages']:
            p=base/m.get('diagnostics_file','')
            if p.is_file() and p.suffix=='.csv':
                f=pd.read_csv(p);f['montage']=m['id'];frames.append(f)
            rows.append([m['id'].replace('_','-').upper(),'пройден' if m['status']=='passed' else 'не пройден',m.get('error_message','') or '—'])
        contactstatus=table(['Сборка','Геометрический и матричный контроль','Причина отказа'],rows)
        if prep['status']=='passed':contactstatus+=f"\n\nМаксимальная относительная норма расхождения с матрицей EIDORS — {ru(max(m['matrix_error'] for m in prep['montages']),'.3g')}; установленный предел — 10⁻¹²."
        if prep.get('error_message'):contactstatus+='\n\nОбщее препятствие: '+prep['error_message']
        if frames:
            f=pd.concat(frames,ignore_index=True);fig,axs=plt.subplots(2,1,figsize=(13,6),constrained_layout=True)
            ix=np.arange(len(f));labels=[m.replace('tepc_','')+' '+dict(zip(ROLES,LABELS))[r] for m,r in zip(f.montage,f.label)]
            axs[0].bar(ix,f.patch_area_mm2,color='#246aa2');axs[0].axhline(manifest['contact']['area_mm2'],color='black',ls='--',label='Целевая площадь');axs[0].set(ylabel='Площадь, мм²',xticks=ix,xticklabels=[]);axs[0].legend()
            axs[1].bar(ix,f.patch_centroid_offset_mm,color='#d5923a');axs[1].axhline(2.5,color='#c04c42',ls='--',label='Порог 2,5 мм');axs[1].set(ylabel='Смещение центра, мм',xticks=ix,xticklabels=labels);axs[1].tick_params(axis='x',rotation=70,labelsize=8);axs[1].legend()
            contactfigure=save(fig,'04_contact_qc')+'\n\n**Рисунок 4а.** Фактические площади и смещения центров контактных площадок. Пороговые линии относятся только к техническому контролю текущей дискретизации.'
            from matplotlib.collections import PolyCollection
            from matplotlib.patches import Circle
            detail,da=plt.subplots(len(prep['montages']),4,figsize=(12,2.6*len(prep['montages'])),squeeze=False,constrained_layout=True)
            for row,m in enumerate(prep['montages']):
                geo=read(base/m['geometry_file'])
                for col,e in enumerate(geo['electrodes']):
                    ax=da[row,col];c=np.asarray(geo['requested_centres_xyz_mm'][col]);patch=e['patch'];v=np.asarray(patch['coords_mm']);tri=np.asarray(patch['faces_1based'],int)-1
                    _,_,basis=np.linalg.svd(v-v.mean(0),full_matrices=False);uv=(v-c)@basis[:2].T
                    ax.add_collection(PolyCollection(uv[tri],facecolors='#c04c42' if col in [0,3] else '#246aa2',edgecolors='white',linewidths=.2,alpha=.9))
                    near=e['neighbourhood'];nv=np.asarray(near['coords_mm']);nt=np.asarray(near['faces_1based'],int)-1;nu=(nv-c)@basis[:2].T
                    ax.add_collection(PolyCollection(nu[nt],facecolors='none',edgecolors='#888888',linewidths=.25))
                    ax.add_patch(Circle((0,0),2.5,fill=False,ls='--',edgecolor='black',linewidth=.8));ax.plot(0,0,'k+',ms=7)
                    ax.set(xlim=(-4,4),ylim=(-4,4),title=m['id'].replace('_','-').upper()+' · '+LABELS[col],xlabel='мм',ylabel='мм');ax.set_aspect('equal')
            contactfigure+='\n\n'+save(detail,'04b_real_contact_patches')+'\n\n**Рисунок 4б.** Реальные контактные пятна на расчётной границе в локальной плоской проекции. Крест — исходный центр; пунктир — окружность радиусом 2,5 мм; окрашены фактически включённые в CEM треугольники. Проекция показывает форму дискретизации и не изменяет геометрию при решении.'

            contactstatus+=f'\n\nПо {len(f)} контактам: площадь {ru(f.patch_area_mm2.min(),'.3f')}–{ru(f.patch_area_mm2.max(),'.3f')} мм²; наибольшее смещение центра площади {ru(f.patch_centroid_offset_mm.max(),'.3f')} мм; наибольшее расстояние исходной точки до границы {ru(f.nearest_boundary_distance_mm.max(),'.3f')} мм.'
    planfile=base/'surface_plan.json';plan=read(planfile) if planfile.exists() else None
    pilotfile=base/'pilot_checks.json';pil=read(pilotfile) if pilotfile.exists() else None
    summaryfile=base/'analysis/summary.json'
    if not summaryfile.exists():summaryfile=base/'analysis/surface_summary.json'
    summary=read(summaryfile) if summaryfile.exists() else None
    results='### Результаты основной серии\n\nОсновной перебор пока не выполнен. Числа чувствительности и предпочтительные сборки появятся после контрольной серии, полного перебора и проверки достаточности покрытия.'
    if summary:
        from run_surface_sensitivity import configure,check_inputs,verified_pilot,load_frames
        configure(ROOT/'output/exploratory/arm_sigma_20260908/python_solver_deps')
        checked_plan,_,studies=check_inputs(base)
        verified_pilot(base,checked_plan,studies)
        _,completions=load_frames(base,checked_plan,studies,checked_plan['batches'])
        if sha(summaryfile)!=(summaryfile.parent/'summary.sha256').read_text().strip() or summary['plan_sha256']!=sha(planfile) or summary['input_completion_sha256']!=completions:
            raise ValueError('Analysis is not bound to the current completed study')
        for name,digest in summary['output_sha256'].items():
            if sha(summaryfile.parent/name)!=digest:raise ValueError('Changed analysis table '+name)
        status='полный перебор и проверка покрытия завершены; '+('критерии выполнены на проверенных состояниях' if summary['adequate_on_tested_states'] else 'покрытие требует уточнения из-за невыполненных критериев')
        names={'complete_matrix':'Полнота матрицы','derivative_interpolation':'Интерполяция производных','selectivity_interpolation':'Интерполяция долей','pair_S_accuracy':'Парные разности чувствительности','pair_Q_accuracy':'Парные разности долей','pair_preferences':'Устойчивость знаков парных различий','score_stability':'Устойчивость минимумов','preferred_sets_stable':'Устойчивость предпочтительных множеств'}
        results='### Результаты основной серии\n\n'+table(['Критерий','Результат'],[[names[k],'выполнен' if v else 'не выполнен'] for k,v in summary['criteria_decisions'].items()])
        cards=pd.read_csv(summaryfile.parent/'scorecards.csv');final=cards[cards.stage=='grid5_plus_checks'].copy()
        montage_order=sorted(final.montage.unique(),key=lambda value:int(value.rsplit('_',1)[1]))
        fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for j,(k,label) in enumerate(zip(PARAMS,RUS)):
            f=final[final.compartment==k].set_index('montage').loc[montage_order].reset_index()
            for row,field,ylabel in [(0,'min_B','Минимальный B, Ом'),(1,'min_Q','Минимальная доля Q')]:
                axs[row,j].bar(f.montage.str.replace('tepc_',''),f[field],color='#246aa2' if row==0 else '#d5923a');axs[row,j].set(title=label if row==0 else '',xlabel='Номер TEPC',ylabel=ylabel);axs[row,j].grid(axis='y',alpha=.2)
        results+='\n\n'+save(fig,'07_target_scorecards')+'\n\n**Рисунок 7.** Минимальные показатели по 125 узлам и 64 проверочным состояниям каждой сборки. Это минимумы конечной выборки состояний, а не доказанные нижние границы на непрерывной области. B характеризует абсолютный масштаб, Q — долю целевой чувствительности.'
        results+='\n\n'+table(['Сборка','Компартмент','Минимальный B, Ом','Минимальный Q'],[[r.montage.replace('_','-').upper(),dict(zip(PARAMS,RUS))[r.compartment],ru(r.min_B,'.6g'),ru(r.min_Q,'.6g')] for r in final.itertuples()])

        columns=[(k,field,label+' · '+metric) for k,label in zip(PARAMS,RUS) for field,metric in [('min_B','B'),('min_Q','Q')]]
        raw=np.array([[float(final[(final.montage==montage)&(final.compartment==k)][field].iloc[0]) for k,field,_ in columns] for montage in montage_order])
        normalized=raw/raw.max(axis=0,keepdims=True)
        fig,ax=plt.subplots(figsize=(13,5.5),constrained_layout=True)
        view=ax.imshow(normalized,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
        ax.set(xticks=np.arange(len(columns)),xticklabels=[label for _,_,label in columns],yticks=np.arange(len(montage_order)),yticklabels=[m.replace('_','-').upper() for m in montage_order])
        ax.tick_params(axis='x',rotation=35,labelsize=9)
        for row in range(raw.shape[0]):
            for col in range(raw.shape[1]):
                value=ru(raw[row,col],'.3g')
                ax.text(col,row,value+'\n'+ru(100*normalized[row,col],'.0f')+'%',ha='center',va='center',fontsize=8,color='white' if normalized[row,col]>.58 else '#222222')
        bar=fig.colorbar(view,ax=ax,fraction=.025,pad=.02);bar.set_label('Доля от лучшего значения в столбце')
        results+='\n\n'+save(fig,'08_sensitivity_matrix')+'\n\n**Рисунок 8.** Матрица минимальной чувствительности с цветовым градиентом. Верхняя строка каждой ячейки — исходное значение B, Ом, либо Q; нижняя — процент от наибольшего значения в том же столбце. Тёмный цвет означает большее значение для выбранного компартмента и показателя. Нормирование выполнено отдельно по столбцам, поэтому цвет B нельзя количественно сравнивать с цветом Q или с другим компартментом.'

        def ordered(compartment,field):
            return final[final.compartment==compartment].sort_values(field,ascending=False)
        soft_b,soft_q=ordered('soft','min_B'),ordered('soft','min_Q')
        heart_b,heart_q=ordered('heart','min_B'),ordered('heart','min_Q')
        lung_b,lung_q=ordered('lung','min_B'),ordered('lung','min_Q')
        heart_gap=100*(heart_b.iloc[0].min_B-heart_b.iloc[1].min_B)/heart_b.iloc[0].min_B
        results+='\n\n### Предварительная практическая интерпретация\n\n'
        results+=f'''- **Мягкие ткани.** TEPC-2 имеет наибольшие значения минимальных $B$ ({ru(soft_b.iloc[0].min_B,'.3g')} Ом) и $Q$ ({ru(soft_q.iloc[0].min_Q,'.3f')}). Если цель состоит в регистрации изменения сопротивления всего мягкотканного компартмента, TEPC-2 является первым кандидатом среди шести рассчитанных вариантов.
- **Сердечная область: абсолютный отклик.** Наибольший минимальный $B$ получен у TEPC-4 ({ru(heart_b.iloc[0].min_B,'.3g')} Ом). TEPC-5 почти совпадает с ним; разница составляет около {ru(heart_gap,'.2f')}% от результата TEPC-4. Поскольку часть критериев попарной точности не выполнена, эти две сборки следует считать близкими кандидатами до дополнительного сгущения по ρ и пространственной проверки.
- **Сердечная область: избирательность.** Наибольший минимальный $Q$ получен у TEPC-7 ({ru(heart_q.iloc[0].min_Q,'.3f')}), но её сердечный $B$ равен только {ru(heart_q.iloc[0].min_B,'.3g')} Ом. Высокая доля возникла при малом абсолютном отклике всех компартментов; одна эта доля не делает TEPC-7 лучшей для измерения сердца.
- **Лёгкие: абсолютный отклик.** Наибольший минимальный $B$ получен у TEPC-2 ({ru(lung_b.iloc[0].min_B,'.3g')} Ом); второе значение — у TEPC-3 ({ru(lung_b.iloc[1].min_B,'.3g')} Ом). TEPC-2 является кандидатом для максимального модельного отклика лёгких.
- **Лёгкие: избирательность.** TEPC-7 имеет наибольший минимальный $Q$ ({ru(lung_q.iloc[0].min_Q,'.3f')}), но её лёгочный $B$ равен {ru(lung_q.iloc[0].min_B,'.3g')} Ом. TEPC-3 сочетает существенно больший $B$ ({ru(float(lung_q[lung_q.montage=='tepc_3'].min_B.iloc[0]),'.3g')} Ом) с меньшей, но сравнительно высокой долей Q; поэтому она представляет более содержательный компромисс для последующей проверки отношения сигнал/шум.'''
        results+='\n\nВ пределах этих шести вариантов широкая диагональная трасса TEPC-2 связана с большим абсолютным вкладом мягких тканей и лёгких, а центральная продольная TEPC-4 — с большим абсолютным вкладом сердечной области. Поперечная TEPC-7 повышает относительную долю сердца и лёгких главным образом за счёт снижения общего абсолютного отклика. Это наблюдение модели, а не доказанный общий закон геометрии электродов.'
        results+='\n\nИз этой серии нельзя вывести универсальное правило «сдвигать электроды к плечам», «сближать их» или «увеличивать площадь». Координаты не изменялись систематически по одному фактору, а площадь всех контактов была одинаковой. Увеличение площади обычно уменьшает контактную составляющую импеданса и меняет распределение плотности тока, но не обязано увеличивать тканевую производную или её избирательность. Эти факторы требуется проверить отдельным вычислительным планом при фиксированных остальных параметрах.'
        results+='\n\n'+('По заранее заданным численным критериям покрытие достаточно на проверенных состояниях. Это допускает модельное сравнение в пределах настоящей постановки; пространственная сходимость и физическая валидация остаются отдельными задачами.' if summary['adequate_on_tested_states'] else 'Не все критерии достаточности покрытия выполнены. Указанные предпочтения являются предварительными; окончательное различение близких сборок требует анализа проверок с невыполненными критериями, дополнительного сгущения по ρ и оценки пространственной неопределённости.')
    fig,ax=plt.subplots(figsize=(13,4.8),constrained_layout=True);ax.set_axis_off()
    boxes=[(.08,.72,'КТ\nвоксели 1 мм'),(.28,.72,'Ручная\nсегментация'),(.48,.72,'Треугольные\nповерхности'),(.68,.72,'Тетраэдры\nи CEM'),(.88,.72,'Результаты\nZ, S, B, Q')]
    for x,y,label in boxes:
        ax.text(x,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.55',facecolor='#dceaf5',edgecolor='#246aa2',linewidth=1.2),transform=ax.transAxes)
    for (x1,y1,_),(x2,y2,_) in zip(boxes[:-1],boxes[1:]):
        ax.annotate('',xy=(x2-.075,y2),xytext=(x1+.075,y1),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',color='#555555',lw=1.4))
    ax.text(.59,.22,'Пространственная сходимость\nменяем только сетку;\nсравниваем Z, S, B, Q',ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.55',facecolor='#f7e5c8',edgecolor='#d5923a',linewidth=1.2),transform=ax.transAxes)
    ax.text(.87,.22,'Физическая валидация\nфантом и прибор;\nсравниваем расчёт с измерением',ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.55',facecolor='#f3d8d5',edgecolor='#c04c42',linewidth=1.2),transform=ax.transAxes)
    ax.annotate('',xy=(.68,.58),xytext=(.59,.36),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',color='#d5923a',lw=1.5))
    ax.annotate('',xy=(.88,.58),xytext=(.87,.36),xycoords=ax.transAxes,arrowprops=dict(arrowstyle='->',color='#c04c42',lw=1.5))
    ax.text(.08,.46,'Размер вокселя относится\nтолько к исходному изображению',ha='center',va='center',fontsize=9,color='#246aa2',transform=ax.transAxes)
    validation_figure=save(fig,'09_validation_logic')+'\n\n**Рисунок 9.** Место двух валидаций в расчётной цепочке. Размер вокселя относится к исходному изображению; после него возникают отдельные этапы сегментации, построения поверхностей, тетраэдрализации и контактной модели. Пространственная проверка сравнивает решения на разных сетках, а физическая — расчёт с независимым измерением.'
    spatial_result='**Текущий статус проверки контактной дискретизации:** результат ещё не получен. Полная пространственная сходимость и физическая валидация не установлены.'
    spatial_base=ROOT/'output/exploratory/tepc_spatial_convergence_20260914'
    spatial_summary_path=spatial_base/'spatial_analysis/summary.json'
    spatial_status_path=spatial_base/'spatial_orchestrator_status.json'
    if spatial_summary_path.exists():
        spatial_summary=read(spatial_summary_path)
        if spatial_summary.get('status')!='completed_contact_local_two_level_mesh_screen':raise ValueError('Unexpected spatial-screen status')
        for name,digest in spatial_summary['outputs_sha256'].items():
            candidate=spatial_summary_path.parent/name
            if sha(candidate)!=digest:raise ValueError('Changed spatial-screen output '+name)
        mesh_comparison=pd.read_csv(spatial_summary_path.parent/'mesh_metric_comparison.csv')
        mesh_comparison['criterion_ratio']=mesh_comparison.absolute_change/mesh_comparison.tolerance
        metric_ratios=mesh_comparison.groupby('metric').criterion_ratio.max().reindex(['Z','S','B','Q'])
        montage_ratios=mesh_comparison.groupby(['montage','metric']).criterion_ratio.max().unstack().reindex(index=sorted(mesh_comparison.montage.unique(),key=lambda value:int(value.rsplit('_',1)[1])),columns=['Z','S','B','Q'])
        fig,ax=plt.subplots(figsize=(8.5,5),constrained_layout=True)
        view=ax.imshow(np.minimum(montage_ratios.to_numpy(),2),cmap='RdYlGn_r',vmin=0,vmax=2,aspect='auto')
        ax.set(xticks=np.arange(4),xticklabels=['Z','S','B','Q'],yticks=np.arange(len(montage_ratios)),yticklabels=[value.replace('_','-').upper() for value in montage_ratios.index])
        for row in range(len(montage_ratios)):
            for col in range(4):
                ratio=montage_ratios.iloc[row,col]
                ax.text(col,row,ru(ratio,'.2f'),ha='center',va='center',fontsize=9,color='white' if ratio>1.25 else '#222222')
        bar=fig.colorbar(view,ax=ax,fraction=.04,pad=.03);bar.set_label('Ошибка / допустимая ошибка; 1 — граница')
        spatial_result=table(['Показатель предварительной проверки','Результат'],[
            ['Сопоставленные состояния',spatial_summary['paired_states']],
            ['Проверок метрик с превышением допуска',spatial_summary['failed_metric_checks']],
            ['Максимальное относительное изменение Z',ru(100*spatial_summary['max_relative_Z_change'],'.4g')+'%'],
            ['Наибольшее отношение ошибки S к допуску',ru(metric_ratios['S'],'.4g')],
            ['Наибольшее отношение ошибки B к допуску',ru(metric_ratios['B'],'.4g')],
            ['Максимальное абсолютное изменение Q',ru(spatial_summary['max_absolute_Q_change'],'.4g')],
            ['Сборки с максимальными показателями сохранены','да' if spatial_summary['stable_top_choices'] else 'нет'],
            ['Полный порядок шести сборок сохранён','да' if spatial_summary['stable_complete_orders'] else 'нет'],
            ['Решение по локальной проверке','пройден' if spatial_summary['contact_discretisation_screen_passed'] else 'не пройден']])
        spatial_result+='\n\n'+save(fig,'10_contact_mesh_convergence')+'\n\n**Рисунок 10.** Максимальное по 78 состояниям отношение изменения показателя при переходе 1,0→0,5 мм к его рабочему допуску. Значение 1 соответствует границе критерия; значения больше 1 отмечают требование дополнительного уточнения. Цвет сравнивает ошибку с допуском внутри каждого показателя и не является картой физиологической чувствительности.'
        spatial_result+='\n\n'+('**Результат математической проверки:** все 780 сопоставлений прошли критерии; локальная контактная дискретизация прошла двухуровневую проверку на 78 состояниях, а полный порядок сборок сохранился. Этот результат не устанавливает сходимость объёмной сетки или границ компартментов.' if spatial_summary['contact_discretisation_screen_passed'] else '**Результат математической проверки:** локальная проверка контактной дискретизации не пройдена. До анализа объёмной сходимости требуется рассмотреть нарушенные метрики и следующий уровень локального уточнения.')
    elif spatial_status_path.exists():
        spatial_status=read(spatial_status_path)
        stages={'refine_contact_neighbourhood':'уточнение контактной сетки','export_fine_fem_model':'экспорт уточнённой модели EIDORS','configure_fine_studies':'подготовка исследований','prepare_contacts':'повторный контроль контактов','seal_pilot':'фиксация входов','run_pilot_78_states':'расчёт 78 состояний','compare_1mm_to_0p5mm':'сопоставление двух сеток'}
        if spatial_status.get('status')=='running':spatial_result='**Текущий статус проверки контактной дискретизации:** выполняется этап «'+stages.get(spatial_status.get('stage'),spatial_status.get('stage','неизвестно'))+'». Численный результат ещё не сформирован.'
        elif spatial_status.get('status')=='failed':spatial_result='**Текущий статус проверки контактной дискретизации:** выполнение остановлено на этапе «'+stages.get(spatial_status.get('stage'),spatial_status.get('stage','неизвестно'))+'». Причина сохранена в локальном паспорте запуска; вывод о сходимости отсутствует.'
    readiness=table(['Этап','Текущее состояние'],[
        ['Исходные координаты',f'{len(coords)} файлов, 24 центра; целостность проверена'],
        ['Геометрия и CEM','проверены' if prep and prep['status']=='passed' else 'не допущены к расчёту'],
        ['Основные файлы запуска',f"{len(plan['batches'])} блоков; {plan['total_required_states']} состояния" if plan else 'ожидают успешной контактной проверки'],
        ['Контроль производных','результат сохранён; см. отдельный протокол' if pil else 'не выполнен'],
        ['Основная серия','анализ сохранён' if summary else 'не выполнена'],
        ['Полная пространственная сходимость и физическая валидация','не выполнены']])
    caveat='Сходство с рисунками установлено на уровне назначения пар. Метрическое и анатомическое совпадение с публикацией не подтверждено. В частности, диагональ TEPC-2 и сторона токовой пары TEPC-7 зависят от принятого соглашения о сторонах модели. Токовые точки TEPC-3 находятся у верхней границы усечённой модели; принадлежность расчётной границе не доказывает соответствие кожной поверхности плеча. Координаты и роли не отражались и не переставлялись автоматически: расчёт относится к переданным JSON, а не к восстановленным по рисунку положениям.'
    replacements=dict(STATUS=status,RESULT_STATUS='ещё не получено' if not summary else 'представлено в разделе результатов с отдельным заключением о покрытии',ARTICLE_FIGURES=article_figures,MONTAGE_FIGURE=montage_figure,GEOMETRY_CAVEAT=caveat,COORDINATES=coordtable,CONTACT_STATUS=contactstatus,CONTACT_FIGURE=contactfigure,RANGES=rangetable,AXES_FIGURE=axesfigure,GRID_FIGURE=gridfigure,READINESS=readiness,RESULTS=results,VALIDATION_FIGURE=validation_figure,SPATIAL_RESULT=spatial_result)
    source=Path(__file__).with_name('surface_sensitivity_report_ru.md').read_text(encoding='utf-8')
    for k,v in replacements.items():source=source.replace('@@'+k+'@@',v)
    if '@@' in source:raise ValueError('Unresolved report token')
    nb=nbformat.v4.new_notebook();nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),study_status='exploratory_hypothesis_not_validated',manifest_sha256=sha(base/'manifest.json'))
    for section in source.split('<!-- cell -->'):
        cell=nbformat.v4.new_markdown_cell(section.strip());cell['attachments']={n:{'image/png':base64.b64encode(p.read_bytes()).decode('ascii')} for n,p in images.items() if 'attachment:'+n in section};nb.cells.append(cell)
        if '## 9.' in section:
            code='''from pathlib import Path
import os, shutil, subprocess, sys, json

# Только проверка по умолчанию. После неё: pilot → full → analyze.
MODE = "check"
repo = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "MATLAB_TRKG4_real_subjects").is_dir())
project = repo / "MATLAB_TRKG4_real_subjects"
base = project / "output/exploratory/tepc_preparation_20260911"
deps = project / "output/exploratory/arm_sigma_20260908/python_solver_deps"
runtime_path = base / "local_runtime.json"
runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
matlab = os.environ.get("MATLAB_EXECUTABLE") or runtime.get("matlab_executable") or shutil.which("matlab")
python = runtime.get("python_executable") or sys.executable
args = [python, "-X", "utf8", str(project / "tools/run_surface_sensitivity.py"),
        "--base", str(base), "--deps", str(deps), "--" + MODE]
if matlab:
    args += ["--matlab", matlab]
if MODE not in {"check", "pilot", "full", "analyze"}:
    raise ValueError("Выберите check, pilot, full или analyze")
if MODE in {"pilot", "full"} and not matlab:
    raise RuntimeError("Задайте MATLAB_EXECUTABLE для установленного MATLAB")
subprocess.run(args, cwd=project, check=True)
'''
            nb.cells.append(nbformat.v4.new_code_cell(code))
    target=ROOT.parent/'Colab Notebooks'/(NAME+'.ipynb');nbformat.write(nb,target)
    exp=HTMLExporter();exp.exclude_input=True;exp.exclude_input_prompt=True;exp.exclude_output_prompt=True
    html,_=exp.from_notebook_node(nb);target.with_suffix('.html').write_text(html,encoding='utf-8')
    (base/'report/reader_text.md').write_text(source,encoding='utf-8')
    artifacts={p.name:sha(p) for p in [target,target.with_suffix('.html'),base/'report/reader_text.md']}
    (base/'report/build.json').write_text(json.dumps(dict(artifacts=artifacts,figures=list(images),notebook_cells=len(nb.cells),fem_executed=False,reader_proofreading='pending_full_text_model_review'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(notebook=str(target),figures=len(images),cells=len(nb.cells),status=status),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=DEFAULT_BASE);a=p.parse_args();build(a.base.resolve())
