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
    for mid,c in coords.items():rows += [[mid.replace('_','-').upper(),LABELS[j],*[f'{v:.3f}' for v in point]] for j,point in enumerate(c)]
    coordtable=table(['Сборка','Роль','X, мм','Y, мм','Z, мм'],rows)
    ranges=[]
    for k,label in zip(PARAMS,RUS):
        lo,hi=profile['bounds'][k];ranges.append([label,f'{lo:.4f}',f'{hi:.4f}',f'{(hi-lo)/4:.4f}',5])
    b=profile['bone_fixed_rho_ohm_m'];ranges.append(['Кости',f'{b:.4f}',f'{b:.4f}','—','фиксировано'])
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
        if prep['status']=='passed':contactstatus+=f"\n\nМаксимальная относительная норма расхождения с матрицей EIDORS — {max(m['matrix_error'] for m in prep['montages']):.3g}; установленный предел — 10⁻¹²."
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

            contactstatus+=f'\n\nПо {len(f)} контактам: площадь {f.patch_area_mm2.min():.3f}–{f.patch_area_mm2.max():.3f} мм²; наибольшее смещение центра площади {f.patch_centroid_offset_mm.max():.3f} мм; наибольшее расстояние исходной точки до границы {f.nearest_boundary_distance_mm.max():.3f} мм.'
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
        status='полный перебор и проверка покрытия завершены; '+('критерии выполнены на проверенных состояниях' if summary['adequate_on_tested_states'] else 'требуется дальнейшая проверка покрытия по не прошедшим критериям')
        names={'complete_matrix':'Полнота матрицы','derivative_interpolation':'Интерполяция производных','selectivity_interpolation':'Интерполяция долей','pair_S_accuracy':'Парные разности чувствительности','pair_Q_accuracy':'Парные разности долей','pair_preferences':'Устойчивость знаков парных различий','score_stability':'Устойчивость минимумов','preferred_sets_stable':'Устойчивость предпочтительных множеств'}
        results='### Результаты основной серии\n\n'+table(['Критерий','Результат'],[[names[k],'выполнен' if v else 'не выполнен'] for k,v in summary['criteria_decisions'].items()])
        cards=pd.read_csv(summaryfile.parent/'scorecards.csv');final=cards[cards.stage=='grid5_plus_checks']
        fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for j,(k,label) in enumerate(zip(PARAMS,RUS)):
            f=final[final.compartment==k]
            for row,field,ylabel in [(0,'min_B','Минимальный B, Ом'),(1,'min_Q','Минимальная доля Q')]:
                axs[row,j].bar(f.montage.str.replace('tepc_',''),f[field],color='#246aa2' if row==0 else '#d5923a');axs[row,j].set(title=label if row==0 else '',xlabel='Номер TEPC',ylabel=ylabel);axs[row,j].grid(axis='y',alpha=.2)
        results+='\n\n'+save(fig,'07_target_scorecards')+'\n\n**Рисунок 7.** Минимальные показатели по 125 узлам и 64 проверочным состояниям каждой сборки. Это минимумы конечной выборки состояний, а не доказанные нижние границы на непрерывной области. B характеризует абсолютный масштаб, Q — долю целевой чувствительности.'
        results+='\n\n'+table(['Сборка','Компартмент','Минимальный B, Ом','Минимальный Q'],[[r.montage.replace('_','-').upper(),dict(zip(PARAMS,RUS))[r.compartment],f'{r.min_B:.6g}',f'{r.min_Q:.6g}'] for r in final.itertuples()])
        results+='\n\n'+('По заранее заданным численным критериям покрытие достаточно на проверенных состояниях. Это допускает модельное сравнение в пределах настоящей постановки; пространственная сходимость и физическая валидация остаются отдельными задачами.' if summary['adequate_on_tested_states'] else 'Не все критерии достаточности покрытия выполнены. Предпочтения из этой таблицы предварительны; использовать их для окончательного выбора сборки до разбора не прошедших проверок нельзя.')
    readiness=table(['Этап','Текущее состояние'],[
        ['Исходные координаты',f'{len(coords)} файлов, 24 центра; целостность проверена'],
        ['Геометрия и CEM','проверены' if prep and prep['status']=='passed' else 'не допущены к расчёту'],
        ['Основные файлы запуска',f"{len(plan['batches'])} блоков; {plan['total_required_states']} состояний" if plan else 'ожидают успешной контактной проверки'],
        ['Контроль производных','результат сохранён; см. отдельный протокол' if pil else 'не выполнен'],
        ['Основная серия','анализ сохранён' if summary else 'не выполнена'],
        ['Пространственная сходимость и физическая валидация','для новых сборок не выполнены']])
    caveat='Сходство с рисунками установлено на уровне назначения пар. Метрическое и анатомическое совпадение с публикацией не подтверждено. В частности, диагональ TEPC-2 и сторона токовой пары TEPC-7 зависят от принятого соглашения о сторонах модели. Токовые точки TEPC-3 находятся у верхней границы усечённой модели; принадлежность расчётной границе не доказывает соответствие кожной поверхности плеча. Координаты и роли не отражались и не переставлялись автоматически: расчёт относится к переданным JSON, а не к восстановленным по рисунку положениям.'
    replacements=dict(STATUS=status,RESULT_STATUS='ещё не получено' if not summary else 'представлено в разделе результатов с отдельным заключением о покрытии',ARTICLE_FIGURES=article_figures,MONTAGE_FIGURE=montage_figure,GEOMETRY_CAVEAT=caveat,COORDINATES=coordtable,CONTACT_STATUS=contactstatus,CONTACT_FIGURE=contactfigure,RANGES=rangetable,AXES_FIGURE=axesfigure,GRID_FIGURE=gridfigure,READINESS=readiness,RESULTS=results)
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
