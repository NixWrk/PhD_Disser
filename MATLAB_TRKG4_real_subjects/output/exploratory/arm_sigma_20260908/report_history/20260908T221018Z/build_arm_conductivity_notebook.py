"""Build notebook 40.05; execution reads verified MATLAB outputs only."""
from pathlib import Path
import hashlib
import json
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
NAME='40.05_Чувствительность_ТТРКГ_к_проводимости_тканей.ipynb'

def build():
    out=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/arm_sigma_20260908'
    completion=out/'completion.json'
    manifest_file=out/'result_integrity.json'
    if completion.exists() and json.loads(completion.read_text(encoding='utf-8'))['status']=='matrix_completed_numerical_checks_passed' and not manifest_file.exists():
        files=[]
        for pattern in ['in*_results.csv','in*_patches.csv','in*_derivative_qc.csv','in*_contact.mat']:
            files.extend(out.glob(pattern))
        files.extend(out/name for name in ['identity.json','python_identity.json','independent_solver_qc.json','tissue_inventory.csv','fixed_parameters.json','historical_40_02_summary.csv','historical_reference_identity.json','source_used_matrix.zip'])
        manifest={'files':{file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(files)}}
        manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
    md(r'''
# 40.05 Чувствительность ТТРКГ к проводимости тканей при разных монтажах на руках

**Статус:** прямой вычислительный эксперимент на индивидуальной КТ/FEM-модели Nix в рамках разработки методики. Входной физиологический сигнал не используется. Числа характеризуют модель при фиксированной анатомии, а не измеренную функцию сердца.

Задача — сравнить четыре варианта контакта при расстояниях внутренних электродов от модельной плоскости туловища 20 и 40 мм и расстояниях наружных электродов от внутренних 80 и 100 мм. Для каждого из 16 монтажей совместно варьируются проводимости мягкотканного фона, эффективной сердечной области и лёгких: три уровня каждой величины образуют полную матрицу из 27 сочетаний.

**Исправление постановки, подтверждённое автором 08.09.2026:** поперечными плоскостями могут быть только наружные токовые электроды I+ и I−. В этом варианте внутренние измерительные электроды V+ и V− являются поверхностными манжетами шириной 5 мм. Исторические результаты четырёхплоскостного монтажа из 40.02 исключены из настоящего сравнения.

**Допущение, принятое автором:** вся имеющаяся маска сердца представляет эффективную смесь крови и миокарда. Отдельной маски крови у Nix нет; изменение её проводимости независимо от миокарда здесь не моделируется.
''')
    code(r'''
from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown

ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'MATLAB_TRKG4_real_subjects').is_dir())
MODEL=ROOT/'MATLAB_TRKG4_real_subjects'
OUT=MODEL/'output'/'exploratory'/'arm_sigma_20260908'
identity=json.loads((OUT/'identity.json').read_text(encoding='utf-8-sig'))
study=identity['study']
python_identity=json.loads((OUT/'python_identity.json').read_text(encoding='utf-8'))
assert python_identity['matlab_identity_sha256']==hashlib.sha256((OUT/'identity.json').read_bytes()).hexdigest()
completion=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
assert completion['status']=='matrix_completed_numerical_checks_passed'
manifest=json.loads((OUT/'result_integrity.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest, f'Изменён файл результата: {name}'
assert study['status']=='exploratory_hypothesis_not_validated'
changed_working_sources=[]
with zipfile.ZipFile(OUT/'source_used_matrix.zip') as used_sources:
    assert python_identity['source_sha256']==hashlib.sha256(used_sources.read('tools/solve_arm_conductivity_matrix.py')).hexdigest()
    for item in identity['sources']:
        assert hashlib.sha256(used_sources.read(item['path'])).hexdigest()==item['sha256'], f"Повреждён архив исходников: {item['path']}"
        live=MODEL/item['path']
        if not live.exists() or hashlib.sha256(live.read_bytes()).hexdigest()!=item['sha256']:
            changed_working_sources.append(item['path'])
if changed_working_sources:
    display(Markdown('**Версия кода:** происхождение результатов проверено по архиву `source_used_matrix.zip`. Рабочая версия следующих файлов уже отличается от использованной в расчёте: '+', '.join('`'+name+'`' for name in changed_working_sources)+'. Архивные исходники не подменяют текущий рабочий код.'))
acceptance=json.loads((MODEL/'output'/'accepted_arm_parameter_mesh_acceptance.json').read_text(encoding='utf-8'))
assert acceptance['status']=='accepted_for_parameterized_arm_electrode_method_development'
assert all(acceptance['checks'].values())
with (MODEL/'output'/'accepted_arm_parameter_mesh.msh').open('rb') as f:
    mesh_hash=hashlib.file_digest(f,'sha256').hexdigest()
assert mesh_hash==identity['mesh_sha256']==acceptance['accepted_mesh_sha256']
files=sorted(OUT.glob('in*_results.csv'))
df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
assert len(df)==432 and df['montage'].nunique()==16
assert not df.duplicated(['montage','case_index']).any()
assert (df.groupby('montage')['case_index'].nunique()==27).all()
assert set(df.electrode_kind)==set(study['montage_kinds'])
assert np.isfinite(df.select_dtypes('number').drop(columns=['solver_warning','solver_warning_message'],errors='ignore')).all().all()
assert np.allclose(df[['sigma_soft','sigma_heart','sigma_lung']].to_numpy(),np.asarray(study['parameter_matrix'])[df.case_index.to_numpy(dtype=int)-1],rtol=1e-12,atol=1e-14)
assert (df.reciprocity_relative_error<1e-8).all()
assert (df.relative_residual<1e-7).all()
assert (df.matrix_relative_error<1e-12).all()
patch_tables=[]
for p in sorted(OUT.glob('in*_patches.csv')):
    tab=pd.read_csv(p);tab['montage']=p.name.removesuffix('_patches.csv');patch_tables.append(tab)
patches=pd.concat(patch_tables,ignore_index=True)
assert len(patches)==64 and (patches.groupby('montage').size()==4).all()
assert (patches.pass_clearance==1).all()
assert (patches.face_count>0).all() and (patches.node_count>0).all()
is_plane=patches.contact_geometry=='internal_plane'
assert is_plane.sum()==8
assert set(patches.loc[is_plane,'label'])=={'I_plus','I_minus'}
mixed=patches.electrode_kind=='outer_planes_inner_rings'
assert (patches.loc[mixed&~is_plane,'contact_geometry']=='surface_cuff').all()
assert (patches.loc[mixed&~is_plane,'axial_width_mm']==5).all()
assert patches.loc[~is_plane,'area_ratio'].between(.95,1.05).all()
assert patches.loc[is_plane,'projected_cross_section_area_ratio'].between(.95,1.05).all()
assert ((patches.loc[is_plane,'patch_x_max_mm']-patches.loc[is_plane,'patch_x_min_mm'])<=3).all()
assert ((patches.patch_x_centroid_mm-patches.x_mm).abs()<=.5).all()
fd=pd.concat([pd.read_csv(p).assign(montage=p.name.removesuffix('_derivative_qc.csv')) for p in OUT.glob('*_derivative_qc.csv')],ignore_index=True)
assert len(fd)==24 and fd.montage.nunique()==4
assert (fd.relative_error<.002).all()
labels={'point_disc_5mm':'Диски 5 мм','circumferential_ring':'Кольца 5 мм',
        'wide_cuff_equivalent_area':'Манжеты 15 мм','outer_planes_inner_rings':'Наружные плоскости + внутренние кольца'}
short={'point_disc_5mm':'Диски','circumferential_ring':'Кольца',
       'wide_cuff_equivalent_area':'Широкие манжеты','outer_planes_inner_rings':'Плоскости + кольца'}
tissues=['soft','heart','lung']
tissue_labels=['Мягкотканный фон','Сердечная область','Лёгкие']
baseline=np.asarray(study['baseline'])
base=df[np.isclose(df[['sigma_soft','sigma_heart','sigma_lung']],baseline).all(axis=1)].sort_values('montage_index').copy()
assert len(base)==16
df.sort_values(['montage_index','case_index']).to_csv(OUT/'matrix_results.csv',index=False)
base.to_csv(OUT/'baseline_sensitivity.csv',index=False)
old=pd.read_csv(OUT/'historical_40_02_summary.csv')
old_identity=json.loads((OUT/'historical_reference_identity.json').read_text(encoding='utf-8'))
assert hashlib.sha256((OUT/'historical_40_02_summary.csv').read_bytes()).hexdigest()==old_identity['sha256']
regression=base.merge(old,left_on='montage',right_on='scenario_id',suffixes=('','_old'))
assert len(regression)==12
regression['relative_difference']=(regression.Z_direct_ohm-regression.Z_direct_real_ohm).abs()/regression.Z_direct_real_ohm.abs()
assert (regression.relative_difference<1e-8).all()
base['label']=[f"{short[k]} · {i}/{o}" for k,i,o in zip(base.electrode_kind,base.inner_mm,base.outer_mm)]
pd.set_option('display.max_rows',40)
plt.rcParams.update({'figure.dpi':120,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
display(Markdown(f"Проверены результаты для **{len(df)} состояний и {len(base)} монтажей**. Полнота матрицы, хэш сетки, исходный код и геометрический контроль входов проверены при выполнении этой ячейки."))
''')
    md(r'''
## Геометрия сравниваемых контактов

На схеме показана одна модельная рука для положения 20/80 мм: начало отсчёта — плоскость соединения с туловищем; центр V+ находится на 20 мм, I+ — ещё на 80 мм дальше. На другой руке задана соответствующая пара V−/I−. Это поясняющая схема; фактические поверхности контактов строятся на тетраэдральной сетке и проходят отдельный QC.
''')
    code(r'''
from matplotlib.patches import Rectangle, Circle
fig,axes=plt.subplots(2,2,figsize=(10,5.6),constrained_layout=True)
for ax,kind in zip(axes.flat,study['montage_kinds']):
    ax.add_patch(Rectangle((0,-30),168,60,facecolor='#e9edf0',edgecolor='#81919b'))
    ax.axvline(0,c='#435460',ls='--',lw=1)
    for x,color,role in [(20,'#2679a5','V+'),(100,'#be4e43','I+')]:
        plane=kind=='outer_planes_inner_rings' and role=='I+'
        if kind=='point_disc_5mm':
            ax.add_patch(Circle((x,30),2.5,facecolor=color,edgecolor=color))
        elif plane:
            ax.plot([x,x],[-30,30],c=color,lw=3)
        else:
            width=15 if kind=='wide_cuff_equivalent_area' else 5
            ax.add_patch(Rectangle((x-width/2,-30),width,60,facecolor='none',edgecolor=color,lw=2))
        ax.text(x,36,role,ha='center',color=color,fontweight='bold')
    ax.set_xlim(-8,176);ax.set_ylim(-36,50);ax.set_aspect('equal')
    ax.set_yticks([]);ax.set_xticks([0,20,100,168]);ax.set_title('Плоскость I + манжета V' if kind=='outer_planes_inner_rings' else labels[kind],fontsize=10,loc='left')
fig.supxlabel('Расстояние от плоскости соединения с туловищем, мм',fontsize=10)
plt.show()
''')
    md(r'''
## Проводимости: источник и границы переноса

На всех этапах частота составляет 50 кГц; проводимости действительные, скалярные, в См/м. Основные тканевые ориентиры вычислены функцией `itis_conductivity` по коэффициентам **IT’IS Tissue Properties Database V5.0 / Gabriel**. Это значения частотной модели, не индивидуальные измерения Nix. [Описание частотной модели IT’IS](https://itis.swiss/virtual-population/tissue-properties/database/dielectric-properties), [версия базы V5.0](https://itis.swiss/virtual-population/tissue-properties/downloads/database-v5-0).

Мягкотканный фон включает ткани, не получившие отдельной STL-маски, и искусственные руки. Нижний и верхний уровни соответствуют жировой ткани и мышце в принятой скалярной модели; средний уровень сохраняет прежний параметр проекта. Такие пределы задают широкий диапазон возможного тканевого состава. Они не означают, что одна и та же ткань физиологически меняется от жира до мышцы. В исследовании десяти добровольцев Kangasmaa и соавт. показаны зависимость оценок от направления мышечных волокон и существенная неопределённость проводимости жира. Поэтому выбранный интервал не объявляется универсальным физиологическим или доверительным диапазоном человека. [Kangasmaa et al., 2025](https://doi.org/10.1002/bem.22541).

Для сердечной области крайние уровни соответствуют миокарду и крови, средний — их арифметической середине как отдельному сценарию. Опорная точка сохраняет прежнюю проводимость миокарда на нижней границе интервала для сопоставимости с 40.02; она не считается независимо установленной проводимостью смеси. Это допущение об эффективном параметре цельной маски, не измеренная доля крови и не модель изменения объёма камер.

Для лёгких сохранён прежний низкий параметр проекта и добавлены ориентиры IT’IS для наполненного воздухом и ненаполненного лёгкого. Геометрия КТ на вдохе во всех сценариях остаётся неизменной: моделируется только изменение проводимости. Полностью спавшееся лёгкое не приравнивается к обычному выдоху. Измерения Gabriel, Peyman и Grant относятся к тканям свиньи, а жидкостей — к измерениям вне организма; их межвидовой перенос требует оговорки. [Gabriel et al., 2009](https://doi.org/10.1088/0031-9155/54/16/002).

В Zotero проверена тематическая коллекция картирования/FEM и выполнен поиск по всей библиотеке. Количественные частотные ориентиры для этого расчёта взяты из указанных внешних источников. Никакое распределение вероятностей и независимость физиологических изменений не задавались: полная комбинационная сетка является планом вычислительного эксперимента.
''')
    code(r'''
range_table=pd.DataFrame({
 'Область':tissue_labels,
 'Три уровня σ, См/м':['; '.join(f'{x:.6f}' for x in study[k]) for k in ['soft_levels','heart_levels','lung_levels']],
 'Опорная σ, См/м':baseline,
 'Основание':['Жир — прежний фон — мышца','Миокард — условная середина — кровь','Прежний параметр — наполненное воздухом — ненаполненное лёгкое'],
})
display(range_table)
fixed=json.loads((OUT/'fixed_parameters.json').read_text(encoding='utf-8'))
assert len(set(fixed['z_contact_ohm_m2']))==1
display(Markdown(f"Фиксированы: σ кости = {fixed['bone_conductivity_S_m']:.6f} См/м; удельный контактный импеданс {fixed['z_contact_ohm_m2'][0]:.9f} Ом·м²; радиус искусственных рук {fixed['arm_radius_mm']:.0f} мм. Это параметры модели, а не новые измерения контакта или анатомии."))
inventory=pd.read_csv(OUT/'tissue_inventory.csv')
inventory.tissue=inventory.tissue.map({'soft_tissue':'Мягкотканный фон, включая руки','lungs':'Лёгкие','heart':'Цельная сердечная область','bones':'Кости'})
inventory.columns=['Тканевая область модели','Тетраэдры','Объём модели, мл']
display(inventory.round(2))
display(Markdown('Объёмы рассчитаны по назначенным тетраэдрам и не являются новым независимым измерением анатомических объёмов. Отдельной кровяной области в этом разбиении нет.'))
''')
    md(r'''
## Прямая задача и определение чувствительности

Проводимости назначаются по центроидам тетраэдров внутри ручных STL-масок. Кость сохраняет прежнюю проводимость. Геометрия, контактный импеданс и положения электродов при изменении тканей фиксированы. Ток 1 А служит нормировкой линейной задачи.

После фиксации одного узла система имеет вид
\[
K(\boldsymbol\sigma)u=b_I,\qquad K(\boldsymbol\sigma)v=b_V,\qquad Z=b_V^T u.
\]
Здесь $K$ — матрица проводимости FEM с полной электродной моделью; $u$ и $v$ — потенциалы прямой и взаимной задач, нормированные на ток (Ом); $b_I$ и $b_V$ — безразмерные векторы соответствующих электродных пар с элементами +1 и −1; $Z$ — передаточный импеданс, Ом. Взаимная задача использует те же контактные поверхности.

Матрицы объёмных тканей собираются из оператора EIDORS один раз. Полные системы решаются прямым методом Intel oneMKL PARDISO в двойной точности после диагонального масштабирования; контроль невязки выполняется в исходной немасштабированной системе. Геометрия и контактные матрицы экспортируются MATLAB. Версии библиотек и хэши записаны в `python_identity.json`. Затем для каждого состояния точно формируется
\[
K=\sigma_sK_s+\sigma_hK_h+\sigma_lK_l+\sigma_bK_b+K_c,
\]
где индексы $s,h,l,b,c$ обозначают мягкотканный фон, сердечную область, лёгкие, кость и контакты. Для каждого из 432 состояний выполняется новое прямое решение; результаты не заменяются линейной экстраполяцией или интерполяцией таблиц. Контактная матрица интегрирует линейные функции на треугольных гранях и проверяется относительно полной сборки EIDORS для каждого монтажа.

При фиксированных контактах точная производная дискретной задачи равна
\[
\frac{\partial Z}{\partial\sigma_t}=-v^TK_tu,\qquad
E_t=\frac{\sigma_t}{Z}\frac{\partial Z}{\partial\sigma_t}.
\]
Здесь $t$ — изменяемая ткань, $E_t$ — безразмерная относительная чувствительность. При малом изменении проводимости на 1% относительное изменение $Z$ приближённо составляет $E_t$ процентов. Знак сохраняется. Значение $E_t$ не является долей ткани в полном импедансе.

Производные проверяются центральными конечными разностями с шагами 1% и 0,5% в опорной точке для всех четырёх типов контакта при положении 20/80 мм. Эти шаги являются численными параметрами проверки, а не физиологическими амплитудами. Критерии: ошибка взаимности <10⁻⁸, относительная невязка системы <10⁻⁷, расхождение матриц с EIDORS <10⁻¹², отличие производной от конечной разности <0,2%.

В смешанной схеме наружные плоскости сохраняют конечный контактный импеданс. Эквипотенциален идеальный электрод; потенциал прилегающей ткани не обязан быть одинаковым на всём сечении. Поверхность сечения аппроксимирована гранями сетки и не является идеально плоской.
''')
    code(r'''
independent=json.loads((OUT/'independent_solver_qc.json').read_text(encoding='utf-8'))
assert independent['relative_difference']<1e-8
display(Markdown(f"Независимая проверка опорного смешанного монтажа: PARDISO и сопряжённые градиенты с многосеточным предобусловливанием дали относительное расхождение Z **{independent['relative_difference']:.2g}**. Обе проверки решают одну дискретную FEM-задачу и не являются внешней физической валидацией."))
quality=acceptance['mesh_shape_quality']
basic=acceptance['mesh_basic_quality']
display(Markdown(f"Сетка содержит {basic['nodes']:,} узлов и {basic['tetrahedra']:,} тетраэдров. Вырожденных и инвертированных элементов нет, но имеются **{quality['count_le_0_001']} элементов с SICN ≤ 0,001**; минимум качества {quality['minimum']:.3g}. Численный QC фиксированной сетки пройден, сеточная сходимость тканевых чувствительностей не проверена. Старый паспорт приёмки относится к прежнему набору контактов; все 64 контакта текущей серии проверены отдельно."))
qc=pd.DataFrame({
 'Проверка':['Состояний','Контактных поверхностей','Максимальная ошибка взаимности','Максимальная относительная невязка','Максимальное расхождение с матрицей EIDORS','Максимальная относительная ошибка производной'],
 'Результат':[len(df),len(patches),df.reciprocity_relative_error.max(),df.relative_residual.max(),df.matrix_relative_error.max(),fd.relative_error.max()],
})
display(qc)
display(Markdown(f"Двенадцать неизменённых поверхностных монтажей воспроизвели исходные опорные значения 40.02; максимальное относительное отличие **{regression.relative_difference.max():.2g}**. Исторические четыре плоскости исключены из этой проверки и из матрицы чувствительности."))
plane_area=patches.loc[is_plane,'area_ratio']
plane_projection=patches.loc[is_plane,'projected_cross_section_area_ratio']
display(Markdown(f'Для наружных плоскостей фактическая площадь граней составляет **{plane_area.min():.3f}–{plane_area.max():.3f}** номинальной площади сечения, а площадь проекции — **{plane_projection.min():.3f}–{plane_projection.max():.3f}**. Контактная матрица интегрируется по фактическим граням. При фиксированном удельном контактном импедансе это меняет интегральную контактную проводимость; компенсация площади не вводилась. Следовательно, результат относится к данной дискретной CEM-идеализации.'))
geometry=patches.groupby(['electrode_kind','contact_geometry'],as_index=False).agg(contacts=('label','size'),area_ratio_min=('area_ratio','min'),area_ratio_max=('area_ratio','max'))
geometry.electrode_kind=geometry.electrode_kind.map(short)
geometry.contact_geometry=geometry.contact_geometry.replace({**short,'surface_cuff':'Поверхностная манжета','internal_plane':'Внутренняя поверхность сечения'})
geometry.columns=['Схема','Контактная поверхность','Контактов','Мин. A/Aном','Макс. A/Aном']
display(geometry.round(4))
warning_rows=df[df.solver_warning.notna()|df.solver_warning_message.notna()]
if len(warning_rows):
    display(Markdown('**Численные предупреждения:** приведены ниже; выполнение критериев невязки и взаимности не отменяет их рассмотрения.'))
    display(warning_rows[['montage','case_index','solver_warning','solver_warning_message']].drop_duplicates())
show=base[['electrode_kind','inner_mm','outer_mm','Z_direct_ohm','elasticity_soft','elasticity_heart','elasticity_lung']].copy()
show.electrode_kind=show.electrode_kind.map(labels)
show.columns=['Контакт','От туловища, мм','Между электродами, мм','Z, Ом','E мягкие ткани','E сердце','E лёгкие']
display(show.round(6))
''')
    md(r'''
## Локальное сравнение монтажей в опорной точке

Ниже сопоставлены абсолютный отклик в мОм на одинаковое малое относительное изменение проводимости (+1%) и безразмерная чувствительность. Первая величина полезна для будущего сопоставления с измерительной погрешностью; вторая показывает масштаб отклика относительно базового импеданса. Погрешность реального прибора здесь не задана, поэтому обнаружимость физиологических изменений не устанавливается.
''')
    code(r'''
E=base[[f'elasticity_{t}' for t in tissues]].to_numpy()
dZ=base[[f'dZ_dsigma_{t}' for t in tissues]].to_numpy()*baseline[None,:]*.01*1000
fig,axes=plt.subplots(1,2,figsize=(13,7.2),constrained_layout=True)
for ax,values,title,fmt in zip(axes,[dZ,E],['Линеаризованный отклик на +1% σ, мОм','Относительная чувствительность E'],['.1f','.4f']):
    vmax=np.abs(values).max();im=ax.imshow(values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
    ax.set_xticks(range(3),['Мягкие ткани','Сердце','Лёгкие']);ax.set_yticks(range(16),base.label)
    ax.set_title(title)
    if ax is axes[1]:ax.set_yticklabels([])
    for i in range(16):
        for j in range(3):ax.text(j,i,format(values[i,j],fmt),ha='center',va='center',fontsize=8,color='white' if abs(values[i,j])>vmax*.65 else 'black')
    fig.colorbar(im,ax=ax,shrink=.75)
plt.show()
# All background, distal artificial-arm background and torso background.
arm_response=base.dZ_dsigma_soft_distal.to_numpy()*baseline[0]*.01*1000
torso_response=base.dZ_dsigma_soft_torso.to_numpy()*baseline[0]*.01*1000
fig,ax=plt.subplots(figsize=(11,6.8),constrained_layout=True)
y=np.arange(16)
ax.barh(y-.16,arm_response,height=.30,label='Фон дистальнее плоскостей стыка рук')
ax.barh(y+.16,torso_response,height=.30,label='Остальной мягкотканный фон')
ax.set_yticks(y,base.label);ax.invert_yaxis();ax.set_xlabel('Линеаризованный отклик на +1% σ фона, мОм');ax.legend(fontsize=8,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False);ax.grid(axis='x',alpha=.2)
plt.show()
''')
    md(r'''
## Нелинейные отклики в выбранном диапазоне

Кривые для положения 20/80 мм построены по новым прямым решениям. В каждой панели изменяется один параметр при двух остальных, равных опорным. Три точки не задают физиологическую траекторию и не доказывают монотонность между узлами сетки. Полная комбинационная матрица ниже позволяет проверить, сохраняется ли сравнение контактов при совместном изменении тканей.
''')
    code(r'''
fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
for t,ax in enumerate(axes):
    other=[i for i in range(3) if i!=t]
    sel=(df.inner_mm==20)&(df.outer_mm==80)
    for j in other:sel &= np.isclose(df[f'sigma_{tissues[j]}'],baseline[j])
    for kind in study['montage_kinds']:
        g=df[sel&(df.electrode_kind==kind)].sort_values(f'sigma_{tissues[t]}')
        ax.plot(g[f'sigma_{tissues[t]}'],g.Z_direct_ohm,'o-',label=short[kind])
    ax.set_xlabel('Проводимость, См/м');ax.set_title(tissue_labels[t]);ax.grid(alpha=.2)
axes[0].set_ylabel('Передаточный импеданс, Ом');axes[-1].legend(fontsize=8)
plt.show()
# Selectivity is a diagnostic ratio of derivative magnitudes, not a tissue
# fraction in Z or a probability of identifying the source.
es=df[[f'elasticity_{t}' for t in tissues]].abs()
df['heart_selectivity_equal_fraction']=es.elasticity_heart/es.sum(axis=1)
robust=df.groupby(['montage_index','montage','electrode_kind'],as_index=False).agg(
    heart_min=('heart_selectivity_equal_fraction','min'),heart_median=('heart_selectivity_equal_fraction','median'),heart_max=('heart_selectivity_equal_fraction','max'),
    heart_elasticity_min=('elasticity_heart','min'),heart_elasticity_max=('elasticity_heart','max'),Z_min=('Z_direct_ohm','min'),Z_max=('Z_direct_ohm','max'))
robust=robust.sort_values('montage_index')
fig,ax=plt.subplots(figsize=(10,6.5),constrained_layout=True)
y=np.arange(len(robust))
ax.hlines(y,robust.heart_min,robust.heart_max,color='#7c98b3',lw=4)
ax.scatter(robust.heart_median,y,c='#234b70',zorder=3)
ax.set_yticks(y,base.label);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
ax.set_xlabel('Отношение |E сердца| к сумме |E| трёх областей');ax.set_title('Минимум, медиана и максимум по 27 заданным состояниям')
plt.show()
display(Markdown('Это отношение сравнивает три **равных относительных возмущения** проводимости. Оно не учитывает различные физиологические амплитуды тканей, шум или изменение геометрии и не является долей сердечного сигнала. Медиана относится к выбранной сетке, не к популяции.'))
''')
    md(r'''
## Совместное изменение трёх тканевых параметров

Следующие панели показывают все 27 сочетаний для каждого типа контакта в положении 20/80 мм. Цвет и подписи соответствуют относительной чувствительности сердечной области $E_h$; единая шкала позволяет сравнивать панели. Все значения получены из отдельных полных FEM-решений. Остальные сочетания расстояний включены в сводки всех 16 монтажей.
''')
    code(r'''
grid=df[(df.inner_mm==20)&(df.outer_mm==80)]
vmin=grid.elasticity_heart.min();vmax=grid.elasticity_heart.max()
fig,axes=plt.subplots(4,3,figsize=(11.5,10),constrained_layout=True)
for i,kind in enumerate(study['montage_kinds']):
    for j,heart_sigma in enumerate(study['heart_levels']):
        ax=axes[i,j]
        g=grid[(grid.electrode_kind==kind)&np.isclose(grid.sigma_heart,heart_sigma)]
        values=g.pivot(index='sigma_lung',columns='sigma_soft',values='elasticity_heart').sort_index().sort_index(axis=1)
        assert values.shape==(3,3) and values.notna().all().all()
        im=ax.imshow(values.to_numpy(),origin='lower',cmap='viridis',vmin=vmin,vmax=vmax,aspect='auto')
        ax.set_xticks(range(3),[f'{x:.3f}' for x in values.columns]);ax.set_yticks(range(3),[f'{x:.3f}' for x in values.index])
        if i==0:ax.set_title(f'σ сердца = {heart_sigma:.3f} См/м')
        if j==0:ax.set_ylabel(short[kind]+'\nσ лёгких, См/м')
        if i==3:ax.set_xlabel('σ мягких тканей, См/м')
        for y in range(3):
            for x in range(3):
                val=values.iloc[y,x]
                ax.text(x,y,f'{val:.4f}',ha='center',va='center',fontsize=8,color='white' if val<(vmin+vmax)/2 else 'black')
fig.colorbar(im,ax=axes,label='E сердечной области',shrink=.75)
plt.show()
''')
    md(r'''
## Совместная информативность нескольких монтажей

Один монтаж даёт одно значение импеданса при трёх неизвестных проводимостях. Локальную разделимость можно предварительно рассмотреть только для набора монтажей при допущении неизменного состояния между измерениями. Ниже выполнено сингулярное разложение матрицы чувствительности к равным относительным изменениям проводимости. Оно характеризует данную модель и выбранное масштабирование; межсессионные ошибки и шум не включены.
''')
    code(r'''
J=base[[f'dZ_dsigma_{t}' for t in tissues]].to_numpy()*baseline
singular=np.linalg.svd(J,compute_uv=False)
display(pd.DataFrame({'Сингулярное значение':singular,'Единицы':['Ом']*3}))
condition=singular[0]/singular[-1]
display(Markdown(f'Число обусловленности матрицы всех 16 монтажей: **{condition:.3g}**. Большое отношение крайних сингулярных значений указывает на плохую локальную обусловленность: некоторые сочетанные изменения трёх проводимостей меняют набор импедансов гораздо слабее, чем другие. Поэтому наличие 16 монтажей само по себе не гарантирует устойчивое разделение трёх параметров. Вывод относится к опорной точке и принятому масштабированию; практическую идентифицируемость без модели погрешности он не устанавливает.'))
''')
    md(r'''
## Полная матрица и проверяемые выводы

[Полная численная матрица с размерными производными (CSV)](../MATLAB_TRKG4_real_subjects/output/exploratory/arm_sigma_20260908/matrix_results.csv). Ниже значения продублированы в читательской таблице; размерные производные доступны также в коде и CSV.
''')
    code(r'''
cols=['montage_index','electrode_kind','inner_mm','outer_mm','case_index','sigma_soft','sigma_heart','sigma_lung','Z_direct_ohm','elasticity_soft','elasticity_heart','elasticity_lung']
full=df.sort_values(['montage_index','case_index'])[cols].copy()
full.electrode_kind=full.electrode_kind.map(short)
full.columns=['Монтаж','Контакт','Внутр., мм','Разнос, мм','Состояние','σ фон','σ сердце','σ лёгкие','Z, Ом','E фон','E сердце','E лёгкие']
display(HTML('<details><summary>Все 432 состояния: проводимости, импеданс и относительные чувствительности</summary>'+full.to_html(index=False,float_format=lambda x:f'{x:.6g}',border=0)+'</details>'))
b=base.iloc[np.argmax(np.abs(base.elasticity_heart))]
r=robust.iloc[np.argmax(robust.heart_min)]
arm_fraction=np.abs(base.dZ_dsigma_soft_distal/base.dZ_dsigma_soft)
heart_response=base.dZ_dsigma_heart*baseline[1]*.01*1000
geometry_change=base[base.inner_mm==20].merge(base[base.inner_mm==40],on=['electrode_kind','outer_mm'],suffixes=('_20','_40'))
geometry_delta_Z=geometry_change.Z_direct_ohm_40-geometry_change.Z_direct_ohm_20
geometry_delta_heart=(geometry_change.dZ_dsigma_heart_40-geometry_change.dZ_dsigma_heart_20)*baseline[1]*.01*1000
comparison=df[df.electrode_kind=='outer_planes_inner_rings'].merge(df[df.electrode_kind=='circumferential_ring'],on=['inner_mm','outer_mm','case_index'],suffixes=('_mixed','_ring'))
assert len(comparison)==108
mixed_ring_Z_difference=(comparison.Z_direct_ohm_mixed-comparison.Z_direct_ohm_ring).abs().max()
mixed_ring_E_difference=(comparison.elasticity_heart_mixed-comparison.elasticity_heart_ring).abs().max()
display(Markdown(f"""
**Результаты математической модели.** В опорной точке максимальный модуль относительной чувствительности сердечной области получен для варианта **{labels[b.electrode_kind]}**, {int(b.inner_mm)}/{int(b.outer_mm)} мм: $E_h={b.elasticity_heart:.6f}$. Во всей комбинационной матрице $Z$ лежит между {df.Z_direct_ohm.min():.3f} и {df.Z_direct_ohm.max():.3f} Ом.

Для одинакового малого увеличения проводимости сердечной области на 1% линеаризованный отклик всех 16 монтажей находится в диапазоне **{heart_response.min():.4f}–{heart_response.max():.4f} мОм**. Сравнение этого абсолютного отклика с $E_h$ необходимо: увеличение относительной чувствительности может быть связано с уменьшением базового $Z$.

При переходе от 20 к 40 мм от стыка до внутренней пары при неизменном расстоянии I–V базовый импеданс меняется на **{geometry_delta_Z.min():.3f}–{geometry_delta_Z.max():.3f} Ом**, тогда как максимальное по модулю изменение сердечного отклика на +1% проводимости составляет **{geometry_delta_heart.abs().max():.6f} мОм**. В этих геометрических сравнениях наружные электроды также смещаются вместе с внутренними.

На всех 108 попарно совпадающих тканевых состояниях и расстояниях максимальное различие смешанного и полностью кольцевого монтажей составляет **{mixed_ring_Z_difference:.6g} Ом** по $Z$ и **{mixed_ring_E_difference:.3g}** по $E_h$. Это сравнение фиксированной модели; физическая эквивалентность контактов из него не следует.

По минимальному значению диагностического отношения $|E_h|/(|E_s|+|E_h|+|E_l|)$ на выбранных 27 состояниях первым оказался монтаж **{labels[r.electrode_kind]}**, {int(base.loc[base.montage_index==r.montage_index,'inner_mm'].iloc[0])}/{int(base.loc[base.montage_index==r.montage_index,'outer_mm'].iloc[0])} мм; значение критерия — {r.heart_min:.6f}. Это условное ранжирование на заданной сетке параметров, не установленный оптимум для реального прибора.

Модуль отношения производной дистального мягкотканного фона к производной всего фона в опорной точке составляет {arm_fraction.min():.3f}–{arm_fraction.max():.3f}. Эта геометрическая диагностика показывает, какую часть изменения фонового параметра создают модельные участки за плоскостями стыка рук. Она не является оценкой кровенаполнения реальных рук.
"""))
''')
    md(r'''
## Ограничения и воспроизведение

1. Положения заданы от плоскостей соединения КТ-туловища с искусственными цилиндрами радиуса 30 мм. Связь с реальными анатомическими ориентирами и электродами экспериментов 2–3 не проверена.
2. КТ снята на вдохе; дыхательное изменение формы, смещение сердца, сосудистая динамика и объёмы крови не менялись. Производная по проводимости не равна производной по объёму крови.
3. Маска сердца единая, а мягкотканный фон неоднороден анатомически, но однороден электрически. Анизотропия, диэлектрическая составляющая и отдельная кожа не моделируются.
4. Плоскостные токовые контакты внутри объёма цилиндров — идеализированное граничное условие. Их поверхность сеточно-зависима. Выполненные проверки относятся к фиксированной сетке и не заменяют исследование сеточной сходимости.
5. Удельный контактный импеданс фиксирован. Неопределённость проводимостей, контактов, ручной установки, КТ-сегментации и несоответствие модели реальной анатомии не объединялись в один случайный шум.
6. Ранжирование монтажей требует дальнейшей проверки относительно целей коррекции ТТРКГ, реальных амплитуд тканевых изменений и измерительной погрешности. Клиническая или экспериментальная валидация не заявляется.

Канонический профиль из папки MATLAB-моделирования: `run_pipeline('Profile','arm_conductivity_matrix')`. Точная использованная версия сохранена в `source_used_matrix.zip`; её соответствие паспорту проверено по хэшам. Для точного воспроизведения нужна эта версия кода в отдельной копии рабочей области. Если рабочие исходники уже изменены, новый расчёт запускается с отдельным `OutputSubdirectory`; продолжение старой серии обновлённым кодом блокируется. Код находится в `src/run_trkg4_arm_conductivity_matrix.m`; уровни — в `src/trkg4_arm_conductivity_design.m`; смешанные контакты — в `src/trkg4_build_mixed_arm_electrodes.m`. Результаты и паспорт с хэшами сохраняются в `output/exploratory/arm_sigma_20260908`. Ноутбук не запускает FEM повторно, а читает и проверяет полный результат. Для воспроизведения нужны локальные STL/MSH, MATLAB/EIDORS и Python с NumPy, SciPy, h5py и PyPardiso/oneMKL. Переменная `TRKG4_PYTHON` позволяет указать интерпретатор для единой MATLAB-команды; при её отсутствии используется `python` из PATH. Исходные медицинские данные не входят в Git.

Соседний `40.04` сохраняет отдельную задачу согласованного сравнения экспериментальных ТТРКГ и боковых сборок. Настоящий параметрический эксперимент её не подменяет.
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'research_status':'exploratory_hypothesis_not_validated'})
    path=ROOT/'Colab Notebooks'/NAME
    nbf.write(nb,path)
    return path

if __name__=='__main__':print(build())
