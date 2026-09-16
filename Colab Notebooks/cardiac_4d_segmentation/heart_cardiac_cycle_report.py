"""Figures and reader report for the observed Nix 4D-mask CEM pilot."""
from pathlib import Path
import json
import base64
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'MATLAB_TRKG4_real_subjects/output/exploratory/nix_cardiac_cycle_20260916'
RESULTS = ROOT / 'Colab Notebooks/cardiac_4d_segmentation/results/nix_cardiac_cycle_20260916'
TITLE = '40.23_Изменение_импеданса_TEPC_по_4D_КТ_Nix'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def fmt(x, digits=3): return f'{x:.{digits}f}'.replace('.', ',')
def table(headers, rows):
    display(Markdown('| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---']*len(headers)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)))
def save(fig, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / name, dpi=155, bbox_inches='tight'); plt.close(fig)
    display(Image(filename=str(RESULTS / name)))


def validate_summary(d):
    expected={(f'phase_{j:02d}',f'tepc_{m}') for j in range(12) for m in range(2,8)}
    rows=d.get('solutions',d.get('states',[]))
    pairs=[(r['id'],r['montage']) for r in rows]
    if d.get('completed')!=72 or d.get('required')!=72 or len(rows)!=72 or len(set(pairs))!=72 or set(pairs)!=expected:
        raise ValueError('Exactly 72 unique phase/montage solutions are required')
    if not all(np.isfinite(r['Z_ohm']) for r in rows): raise ValueError('Nonfinite impedance')


def load():
    # Public numerical summary contains no external medical file paths.
    d = read(RESULTS / 'summary.json')
    validate_summary(d)
    return d


def geometry_figures(d):
    s = d['geometry']; phase = [x['rr_percent'] for x in s]
    v = np.array([x['mask_volume_ml'] for x in s]); c = np.array([x['center_fem_mm'] for x in s])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(phase, v, 'o-', color='#ad1457'); axes[0].set(ylabel='Объём наблюдаемой маски, мл')
    for j, label in enumerate(('x: влево', 'y: кзади', 'z: к голове')):
        axes[1].plot(phase, c[:, j]-c[0, j], 'o-', label=label)
    axes[1].legend(ncol=3); axes[1].set(xlabel='Исходная метка фазы, % R–R', ylabel='Смещение центра от первой фазы, мм')
    for ax in axes: ax.axvline(100, ls='--', color='gray'); ax.grid(alpha=.25)
    save(fig, 'geometry.png')


def montage_figure(d):
    points = np.asarray(d['body_outline_xz_mm']); fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, contact in zip(axes.ravel(), d['contacts']):
        ax.scatter(points[:, 0], points[:, 1], s=.25, color='#bbb', alpha=.25, rasterized=True)
        for poly in d['heart_projection_polygons_xz_mm']:
            poly=np.asarray(poly);ax.fill(poly[:,0],poly[:,1],facecolor='#c2185b',edgecolor='#c2185b',alpha=.18,lw=.6)
        p = np.array(contact['actual_centres_xyz_mm'])
        ax.plot(p[[0, 3], 0], p[[0, 3], 2], color='#c62828', lw=1, alpha=.7)
        ax.plot(p[[1, 2], 0], p[[1, 2], 2], color='#1565c0', lw=1, alpha=.7)
        for j, label in enumerate(('I+', 'V+', 'V−', 'I−')):
            ax.scatter(p[j, 0], p[j, 2], s=35, c='#c62828' if j in (0, 3) else '#1565c0')
            ax.annotate(label, p[j, [0, 2]], xytext=(4, 5), textcoords='offset points', fontsize=9)
        ax.set(title=contact['id'].replace('_', '-').upper(), xlabel='x, мм', ylabel='z, мм', xlim=(-210, 210)); ax.set_aspect('equal')
    save(fig, 'montages.png')


def impedance_figures(d):
    states = d['geometry']; phase = np.array([s['rr_percent'] for s in states])
    t = np.array([s['derived_time_from_R0_ms'] for s in states]); t -= t[0]
    lookup = {(r['id'], r['montage']): r['Z_ohm'] for r in d['solutions']}
    Z = np.array([[lookup[s['id'], f'tepc_{m}'] for s in states] for m in range(2, 8)])
    delta = 1000*(Z-Z[:, :1]); rr = states[0]['rr_interval_ms']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
    for k, ax in enumerate(axes.ravel()):
        ax.plot(t, delta[k], 'o-', color=f'C{k}'); ax.axhline(0, color='gray', lw=.6)
        ax.set(title=f'TEPC-{k+2}', xlabel='Время от первой доступной фазы, мс', ylabel='ΔZ, мОм', xlim=(0, rr)); ax.grid(alpha=.25)
    save(fig, 'one_cycle.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for k in range(6):
        axes[0].plot(phase, Z[k], 'o-', label=f'TEPC-{k+2}')
        axes[1].plot(phase, delta[k], 'o-', label=f'TEPC-{k+2}')
    axes[0].set(ylabel='Z, Ом'); axes[1].set(ylabel='ΔZ, мОм')
    for ax in axes: ax.set(xlabel='Исходная метка фазы, % R–R'); ax.axvline(100, color='gray', ls='--'); ax.grid(alpha=.25)
    axes[1].legend(ncol=2); save(fig, 'all_phases.png')
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    limit = abs(delta).max(); im = ax.imshow(delta, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set(xticks=range(len(states)), xticklabels=[fmt(p, 0) for p in phase], yticks=range(6), yticklabels=[f'TEPC-{m}' for m in range(2, 8)], xlabel='Исходная метка фазы, % R–R')
    for k in range(6):
        for j in range(len(states)): ax.text(j, k, fmt(delta[k, j], 1), ha='center', va='center', fontsize=8, color='white' if abs(delta[k, j]) > .65*limit else 'black')
    fig.colorbar(im, ax=ax, label='ΔZ, мОм'); save(fig, 'heatmap.png')
    # Table uses directly sampled first-cycle states only: no interpolated extrema.
    mask = t <= rr
    table(['Сборка', 'Z опорной фазы, Ом', 'Размах ΔZ в окне R–R, мОм', 'Размах / модуль Z опорной фазы, %'],
          [[f'TEPC-{k+2}', fmt(Z[k, 0], 6), fmt(np.ptp(delta[k, mask])), fmt(np.ptp(delta[k, mask])/abs(Z[k, 0])/10)] for k in range(6)])


def numerical_table(d):
    table(['Фаза, % R–R', 'Объём маски, мл', 'Объём материала FEM, мл', 'Расхождение, %', 'Маска касается границы КТ'],
          [[fmt(s['rr_percent'],0), fmt(s['mask_volume_ml']), fmt(s['material_volume_ml']), fmt(s['volume_error_percent'],5), 'да' if s['touches_source_boundary'] else 'нет'] for s in d['geometry']])


def input_figure():
    import nibabel as nib
    import trimesh
    cfg=read(RUN/'config.local.json'); root=Path(cfg['derived_subject'])
    ct=nib.as_closest_canonical(nib.load(root/'phases/phase_00.nii.gz'))
    mask=nib.as_closest_canonical(nib.load(root/'products_v2/phase_00/whole_heart.nii.gz'))
    data=np.asanyarray(ct.dataobj); heart=np.asanyarray(mask.dataobj)>0
    affine=ct.affine; center=np.column_stack(np.where(heart)).mean(axis=0)
    meshpath=ROOT/'MATLAB_TRKG4_real_subjects/data/nik/stl/heart.stl'
    mesh=trimesh.load(meshpath,process=False); mesh.apply_transform(np.diag([-1.,-1.,1.,1.]))
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,fixed,axis,label in zip(axes,(1,0),(0,1),('Фронтальное сечение','Сагиттальное сечение')):
        index=int(round(center[fixed])); arr=np.take(data,index,axis=fixed); hh=np.take(heart,index,axis=fixed)
        extent=[affine[axis,3]-.5*affine[axis,axis],affine[axis,3]+(data.shape[axis]-.5)*affine[axis,axis],affine[2,3]-.5*affine[2,2],affine[2,3]+(data.shape[2]-.5)*affine[2,2]]
        ax.imshow(arr.T,origin='lower',extent=extent,cmap='gray',vmin=-200,vmax=300)
        xx=affine[axis,3]+np.arange(data.shape[axis])*affine[axis,axis]
        zz=affine[2,3]+np.arange(data.shape[2])*affine[2,2]
        ax.contour(xx,zz,hh.T,levels=[.5],colors=['#00e5ff'],linewidths=1)
        origin=np.zeros(3);origin[fixed]=affine[fixed,3]+index*affine[fixed,fixed];normal=np.eye(3)[fixed]
        section=mesh.section(plane_origin=origin,plane_normal=normal)
        if section is not None:
            for line in section.discrete:
                ax.plot(line[:,axis],np.where(line[:,2]<=extent[3],line[:,2],np.nan),color='#ffbb33',lw=1)
                ax.plot(line[:,axis],np.where(line[:,2]>extent[3],line[:,2],np.nan),color='#888888',lw=1,ls='--')
        ax.axhline(extent[3],color='#ff1744',ls='--',lw=1.5,label='Граница 4D-КТ')
        ax.plot([],[],color='#00e5ff',label='Маска фазы 14% R–R');ax.plot([],[],color='#ffbb33',label='Статическая ручная поверхность')
        ax.set(title=label,xlabel=('x' if axis==0 else 'y')+' RAS, мм',ylabel='z, мм',xlim=(-115,115),ylim=(-245,-10));ax.legend(fontsize=8,loc='upper left')
    save(fig,'ct_overlay.png')
    provenance={'ct_sha256':hashlib.sha256((root/'phases/phase_00.nii.gz').read_bytes()).hexdigest(),'static_heart_sha256':hashlib.sha256(meshpath.read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((root/'products_v2/phase_00/whole_heart.nii.gz').read_bytes()).hexdigest(),'no_additional_registration':True}
    (RESULTS/'ct_overlay_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')


def export_summary():
    import sys
    sys.path.insert(0, str(ROOT / 'MATLAB_TRKG4_real_subjects/tools'))
    import heart_replacement_electrical_pilot as core
    geo = read(RUN / 'materials/geometry.json'); p = read(RUN / 'electrical/plan.json'); r = read(RUN / 'electrical/summary.json')
    if r.get('status')!='complete': raise ValueError('Electrical run is incomplete')
    validate_summary(r)
    if geo['source_prepared_sha256']!=p['source_prepared_sha256']: raise ValueError('Geometry/plan mesh mismatch')
    if core.sha(RUN/'electrical/plan.json')!=(RUN/'electrical/plan.sha256').read_text().strip(): raise ValueError('Plan seal changed')
    if any(row['plan_sha256']!=core.sha(RUN/'electrical/plan.json') for row in r['states']): raise ValueError('Results use another plan')
    runner = core.configure(Path(p['runtime']['deps']), 1)
    with runner.h5py.File(p['source_prepared']) as f:
        nodes = np.asarray(f['fmdl_m/nodes']).T*1000; bd = np.asarray(f['fmdl_m/boundary']).astype(int)-1
    v = nodes[np.unique(bd)][::20]
    contacts = [{k:c[k] for k in ('id','areas_mm2','actual_centres_xyz_mm')} for c in p['contacts']]
    import nibabel as nib
    cfg=read(RUN/'config.local.json');maskpath=Path(cfg['derived_subject'])/'products_v2/phase_00/whole_heart.nii.gz'
    if core.sha(maskpath)!=geo['states'][0]['mask_sha256']: raise ValueError('Montage projection mask changed')
    im=nib.as_closest_canonical(nib.load(maskpath));projection=np.any(np.asanyarray(im.dataobj)>0,axis=1)
    xx=-(im.affine[0,3]+np.arange(im.shape[0])*im.affine[0,0]);zz=im.affine[2,3]+np.arange(im.shape[2])*im.affine[2,2]
    fig,ax=plt.subplots();contours=ax.contour(xx,zz,projection.T,levels=[.5]);heart_polygons=[seg.tolist() for seg in contours.allsegs[0]];plt.close(fig)
    solutions = [{k:row[k] for k in ('id','montage','Z_ohm','max_relative_residual','max_current_error_A','reciprocity_absolute_ohm')} for row in r['states']]
    data = dict(completed=len(solutions), required=72, geometry=geo['states'], contacts=contacts,
                solutions=solutions, coordinate_qc=geo['coordinate_qc'], rho_ohm_m=p['rho_ohm_m'],
                body_outline_xz_mm=v[:,[0,2]].tolist(), heart_projection_polygons_xz_mm=heart_polygons, plan_sha256=core.sha(RUN/'electrical/plan.json'),
                geometry_sha256=core.sha(RUN/'materials/geometry.json'), summary_sha256=core.sha(RUN/'electrical/summary.json'),
                spatial_convergence=False, physical_validation=False)
    RESULTS.mkdir(parents=True,exist_ok=True); core.write(RESULTS/'summary.json',data)
    import csv
    lookup={(row['id'],row['montage']):row['Z_ohm'] for row in solutions}
    with (RESULTS/'impedance_by_phase.csv').open('w',encoding='utf8',newline='') as stream:
        columns=['phase','rr_percent','time_from_first_phase_ms','mask_volume_ml']
        columns += [f'tepc_{m}_{kind}' for m in range(2,8) for kind in ('Z_ohm','delta_Z_mohm')]
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for state in geo['states']:
            row=dict(phase=state['id'],rr_percent=state['rr_percent'],time_from_first_phase_ms=state['derived_time_from_R0_ms']-geo['states'][0]['derived_time_from_R0_ms'],mask_volume_ml=state['mask_volume_ml'])
            for m in range(2,8):
                z=lookup[state['id'],f'tepc_{m}']; base=lookup['phase_00',f'tepc_{m}']
                row[f'tepc_{m}_Z_ohm']=z;row[f'tepc_{m}_delta_Z_mohm']=1000*(z-base)
            writer.writerow(row)
    return data


def build():
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    d = export_summary(); input_figure()
    if read(RESULTS/'ct_overlay_provenance.json')['mask_sha256']!=d['geometry'][0]['mask_sha256']:
        raise ValueError('CT illustration mask differs from calculated phase')
    s = d['geometry']; first = s[0]; v=[x['mask_volume_ml'] for x in s]
    c=np.array([x['center_fem_mm'] for x in s]); maxerr=max(abs(x['volume_error_percent']) for x in s)
    rows=d['solutions']; residual=max(x['max_relative_residual'] for x in rows)
    lookup={(r['id'],r['montage']):r['Z_ohm'] for r in rows}
    included=[x for x in s if x['derived_time_from_R0_ms']-first['derived_time_from_R0_ms']<=first['rr_interval_ms']]
    amplitudes={m:1000*np.ptp([lookup[x['id'],f'tepc_{m}'] for x in included]) for m in range(2,8)}
    ordered=sorted(amplitudes,key=amplitudes.get,reverse=True)
    amplitude_text='; '.join(f'TEPC-{m}: {fmt(amplitudes[m])} мОм' for m in ordered)
    relative={m:amplitudes[m]/abs(lookup['phase_00',f'tepc_{m}'])/10 for m in ordered}
    relative_best=max(relative,key=relative.get)
    peak_groups={}
    for m in range(2,8):
        peak=max(included,key=lambda state:lookup[state['id'],f'tepc_{m}'])['rr_percent']
        peak_groups.setdefault(peak,[]).append(f'TEPC-{m}')
    peak_text='; '.join(', '.join(names)+' — '+fmt(phase,0)+'% R–R' for phase,names in peak_groups.items())
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('# Изменение импеданса сборок TEPC-2–TEPC-7 при динамике сердца по 4D-КТ Nix\n\n'
       '**Статус: вычислительный пилот на автоматических масках. Независимая анатомическая и физическая валидация не выполнена.**\n\n'
       'Цель опыта — получить сопоставимые кривые изменения импеданса шести заданных электродных сборок при пофазном изменении анатомической маски сердца. '
       'Этот расчёт выполняет первые два этапа плана 40.19: проверку координат и построение общего анатомического ориентира. '
       'Приоритет здесь имеет непосредственно рассчитанное изменение сигнала, а не производная в одной центральной точке.\n\n'
       '**Главное ограничение:** маски доходят до верхней границы поля обзора 4D-КТ. Поэтому моделируется наблюдаемая часть автоматической маски. '
       'Динамика за пределами поля обзора неизвестна и не достраивается. '
       'Само касание границы не определяет, какая именно сердечная или сосудистая структура обрезана; для этого нужен отдельный анатомический разбор. '
       'Полученный сигнал нельзя считать подтверждённым сигналом полного сердца.')
    code("from cardiac_4d_segmentation import heart_cardiac_cycle_report as report\ndata = report.load()")
    md('## Исходные данные и совмещение\n\n'
       'Использованы 12 масок TotalSegmentator v2 одного участника и существующая статическая модель его грудной клетки, построенная методом конечных элементов (FEM). '
       'Использована независимая маска целого органа из задачи total; она не получена объединением масок камер и миокарда. '
       'Маски являются вычислительным анатомическим ориентиром; их точность относительно независимой разметки пока неизвестна. '
       'Все 12 состояний данного пилота относятся к разработке методики и не являются отложенной проверочной выборкой для будущей настройки модели. '
       'Объём этой маски не равен объёму отдельной камеры или ударному объёму.\n\n'
       'Статическая и динамическая серии имеют одинаковый идентификатор системы координат пациента. '
       'Координаты NIfTI RAS преобразованы в координаты FEM изменением знаков x и y и переводом миллиметров в метры. '
       'Для всех фаз применено одно преобразование; отдельная подгонка сердца каждой фазы не выполнялась. '
       'По общим костным поверхностям медиана расстояния составила 0,79 мм, 95-й процентиль — 4,39 мм. '
       'По внешней поверхности тела в общей центральной области соответствующие значения составили 0,56 и 1,41 мм. '
       'Это поддерживает прямой перенос координат для пилота, но не определяет независимую погрешность регистрации. '
       'Расстояния включают различия сегментации и дискретизации поверхностей.')
    code("display(report.Image(filename=str(report.RESULTS / 'ct_overlay.png')))")
    md('**Рисунок 1.** Фазовая маска на КТ и сечение статической ручной поверхности сердца в общей системе координат. '
       'Красная линия обозначает верхнюю границу динамической серии. Серый пунктир выше неё показывает только продолжение статической поверхности за пределами обзора; '
       'анатомическое соответствие этой части по 4D-КТ не проверяется. Различие контуров включает различия фазы, поля обзора и способов сегментации. '
       'Совпадение центра сердца специально не навязывалось: такая подгонка могла бы скрыть искомое перемещение.')
    md('## Временная привязка и определение одного цикла\n\n'
       f'Исходные метки фаз: {", ".join(fmt(x["rr_percent"],0) for x in s)}% R–R. Метки выше 100% относятся к следующему интервалу. '
       f'Длительность первого интервала по существующей ЭКГ-привязке равна {fmt(first["rr_interval_ms"],1)} мс, следующего — {fmt(s[-1]["rr_interval_ms"],1)} мс. '
       'Время между точками восстановлено по исходным меткам и длительностям соответствующих R–R. Основной график охватывает окно длительностью первого интервала от первой доступной фазы. Точки сохраняют исходный порядок и время; '
       'последняя и первая точки искусственно не приравниваются. Начало полного интервала от R-зубца до 14% отсутствует, '
       'поэтому окно длиной R–R не следует считать полностью наблюдённым циклом между двумя R-зубцами. '
       'Перенос фаз следующего сокращения в начало первого потребовал бы отдельного допущения о повторяемости. '
       'Соединяющие отрезки служат только для чтения графика. '
       'Значение на границе окна между двумя рассчитанными фазами не является отдельным решением FEM.\n\n'
       f'Временное разрешение реконструкции в таблице составляет {fmt(first["temporal_resolution_ms"],0)} мс. '
       'Поэтому близость фазовых меток не означает соответствующего независимого временного разрешения. '
       'Здесь фазовая информация используется для прямого расчёта; возможность восстановления объёма без ЭКГ этим опытом не проверяется.')
    code('report.geometry_figures(data)')
    md(f'**Рисунок 2.** Изменение объёма наблюдаемой маски и координат её объёмного центра. '
       f'Во всех доступных фазах объём лежит в диапазоне {fmt(min(v))}–{fmt(max(v))} мл. '
       f'Размахи координат x, y, z составляют соответственно {"; ".join(fmt(x) for x in np.ptp(c,axis=0))} мм. '
       'Эти величины характеризуют данную автоматическую сегментацию. Они содержат как возможные реальные изменения анатомии, '
       'так и вариабельность границ маски; эти составляющие ещё не разделены.')
    md('## Электрическая модель и заданные сборки\n\n'
       'Расчёт выполнен на одной сетке: 791 141 узел и 4 448 426 тетраэдров. '
       'Используется квазистатическая резистивная модель на частоте 50 кГц; ёмкостная составляющая не моделируется. '
       'Электроды представлены полной моделью конечного контакта (CEM), номинальный диаметр диска — 5 мм. '
       'Удельный контактный импеданс принят равным 0,000159155 Ом·м². Эти параметры и сопротивления тканей — условия модели, а не индивидуальные измерения.\n\n'
       'Исходное сердце заменено мягкими тканями, а лёгкие и кости сохранены. Каждая фазовая маска накладывается заново на этот неподвижный фон. '
       'Внутри маски назначаются свойства эффективного сердца, после её ухода восстанавливается фоновый материал. '
       'Неизвестная сердечная область за пределами 4D-КТ не сохраняется в качестве придуманной динамической части. '
       'Поэтому абсолютный импеданс также зависит от принятого ограничения поля обзора. '
       'Вычитание опорного Z не гарантирует устранения этой ошибки: исключение сердечной или сосудистой области за пределами обзора меняет распределение тока '
       'и тем самым может изменить отклик на движение оставшейся области.')
    code("report.table(['Ткань','ρ, Ом·м'], [[label,report.fmt(data['rho_ohm_m'][key],6)] for key,label in [('soft','Мягкие ткани'),('lung','Лёгкие'),('heart','Эффективное сердце'),('bone','Кости')]])\nreport.montage_figure(data)\nreport.table(['Сборка','Фактические площади I+, V+, V−, I−, мм²'], [[c['id'].replace('_','-').upper(), '; '.join(report.fmt(x,3) for x in c['areas_mm2'])] for c in data['contacts']])")
    md('**Рисунок 3.** Фактические расчётные центры электродов в проекции на фронтальную плоскость. '
       'Красным показана токовая пара I+/I−, синим — измерительная V+/V−, розовым — проекция маски первой фазы. Линии соединяют пары условно и не обозначают путь тока. '
       'Каждая панель соответствует отдельной четырёхэлектродной модели; совместное подключение всех сборок не моделируется.\n\n'
       'Для чтения рисунка: TEPC-2 — диагональное расположение; TEPC-3 — верхний поперечный монтаж; '
       'TEPC-4 — протяжённый вертикальный монтаж; TEPC-5 — более короткий центральный вертикальный монтаж; '
       'TEPC-6 — две горизонтальные пары на разных уровнях; TEPC-7 — две разнесённые вертикальные пары.')
    md('В каждом тетраэдре рассчитана доля объёма фазовой маски. Для проверки интегрирования использованы вложенные наборы из 4096 и 16 384 точек; '
       'в электрический расчёт переданы доли по 16 384 точкам. '
       'Для линейных конечных элементов средняя проводимость элемента равна σₑ = (1 − fₑ)σфон,ₑ + fₑσсердце, '
       'где fₑ — доля сердечного материала в элементе, σфон,ₑ — проводимость его неизменного фонового материала, σсердце — принятая проводимость эффективного сердца. '
       'При этом ρ = 1/σ; усреднение сопротивлений вместо проводимостей не выполняется. '
       'Это способ интегрирования материалов внутри конечного элемента, а не новая физиологическая смесь тканей. '
       'Материальные матрицы EIDORS собраны заново для каждой фазы. '
       'Такое интегрирование уменьшает ступенчатость замены материала, но не уточняет пространственную аппроксимацию электрического поля. '
       'У разных сборок использованы свои контактные матрицы и отдельные факторизации системы.')
    md('## Изменение импеданса\n\n'
       'Передаточный импеданс определён как Z = (U(V+) − U(V−)) / I. Изменение ΔZ = Z(t) − Z(t₀) отсчитывается '
       'от первой доступной фазы, соответствующей 14% R–R. Положительное ΔZ означает увеличение передаточного импеданса при зафиксированном порядке электродов. '
       'Знаки сохраняются во всех таблицах и рисунках. Размах определяется как максимум минус минимум рассчитанных значений внутри выбранного временного окна.')
    code('report.impedance_figures(data)')
    md(f'**Численный результат.** Размахи в окне одного R–R: {amplitude_text}. '
       f'В этом опыте наибольший размах получен для TEPC-{ordered[0]}, наименьший — для TEPC-{ordered[-1]}. '
       'Это ранжирование относится к совместному изменению доступной геометрии; оно пока не определяет точность восстановления объёма.\n\n'
       f'По относительному размаху наибольшее значение имеет TEPC-{relative_best} — около {fmt(relative[relative_best],2)}% исходного Z. '
       f'У TEPC-{ordered[0]} наибольший абсолютный размах, а относительный составляет около {fmt(relative[ordered[0]],2)}%. Различие связано с нормированием на собственный исходный импеданс каждой сборки. '
       'Нормирование меняет порядок сборок и не превращает относительную амплитуду в оценку избирательности.\n\n'
       f'Фазы максимумов среди рассчитанных состояний: {peak_text}. Таким образом, монтаж влияет на форму отклика, а не только на его масштаб. '
       'Этот сдвиг относится к дискретным фазам; временное разрешение КТ не позволяет трактовать его как точную физиологическую задержку.\n\n'
       '**Рисунки 4–6.** Первый рисунок показывает по одной кривой для каждой сборки в окне длительностью одного R–R; '
       'вертикальные шкалы индивидуальны, чтобы сохранить видимость формы каждого сигнала. Второй рисунок показывает все 12 исходных фаз '
       'и позволяет сравнить амплитуды по общей шкале. Третий представляет те же изменения цветом: красный — увеличение, синий — уменьшение, '
       'белый — значение около опорного. Цветовая шкала едина для всех сборок.\n\n'
       'Таблица под рисунками показывает размах среди непосредственно рассчитанных точек, попавших в окно одного R–R. '
       'Он может недооценивать истинный размах непрерывной кривой из-за редкой фазовой выборки. '
       'Процент относительно исходного Z — относительная амплитуда данного сигнала; он не является производной по объёму или показателем избирательности к сердцу.')
    md('## Проверка расчёта и ограничения интерпретации\n\n'
       f'Максимальное расхождение объёма маски и её материального представления в FEM составило {fmt(maxerr,5)}%. '
       f'Максимальная относительная невязка линейной системы среди 72 решений составила {residual:.3g}. '
       'Дополнительно проверены восстановление токов на контактах и взаимность передаточного импеданса. '
       f'При увеличении числа точек интегрирования с 4096 до 16 384 максимальное изменение полного материального объёма составило {fmt(max(abs(x["quadrature_volumes_ml"][1]-x["quadrature_volumes_ml"][0]) for x in s),4)} мл. '
       f'При этом максимальная сумма абсолютных изменений по отдельным тетраэдрам равна {fmt(max(x["quadrature_weighted_l1_ml"] for x in s),4)} мл. '
       'Различие этих двух показателей означает, что локальные ошибки могут компенсироваться в суммарном объёме. Для оценки их влияния на Z требуется отдельный электрический пересчёт. '
       'Эти проверки подтверждают согласованность численного решения принятой системы. Они не доказывают её соответствие реальному измеренному сигналу.')
    code('report.numerical_table(data)')
    md('Изменение геометрии объединяет изменение объёма, положения и формы. Поэтому большой размах ΔZ не доказывает высокую чувствительность именно к объёму '
       'и не определяет лучшую сборку для восстановления объёма. Для такого выбора нужно отдельно менять объём и координаты, '
       'затем проверить их совместное восстановление на состояниях, которые не использовались для настройки.\n\n'
       'Основные ограничения этого опыта: обрезание маски полем обзора; отсутствие независимой оценки точности автоматической сегментации; '
       'один участник и одна серия; фиксированные свойства тканей и дыхательный фон; непроверенная пространственная сходимость для этих фаз; '
       'отсутствие модели измерительного шума и ошибок установки электродов. Совпадение координатных систем само по себе не устраняет межсерийное движение или дыхательные различия.\n\n'
       '## Вывод и следующий расчёт\n\n'
       'Получены шесть сопоставимых кривых импеданса для доступной динамической анатомии Nix. '
       'Они позволяют увидеть, как одна и та же последовательность масок проявляется в разных монтажах. '
       'На данном этапе корректно сравнивать наблюдаемые формы, знаки и амплитуды отклика при заданных условиях. '
       'Рекомендацию по выбору сборки для оценки объёма давать преждевременно.\n\n'
       'Следующий содержательный шаг — выяснить, какая часть верхнего контура отсутствует в 4D-КТ, и проверить влияние способов обращения с этой областью. '
       'Затем на выбранных фазах следует проверить интегрирование и пространственную сходимость электрического отклика. '
       'После этого можно сопоставлять анатомические маски с индивидуальной формой, сферой и эллипсоидом и разделять вклады объёма и перемещения. '
       'Независимые ручные маски будут использоваться для проверки автоматической сегментации, а не как обязательная замена входов рабочего метода.\n\n'
       '## Воспроизводимость\n\n'
       'Код подготовки материалов и расчёта сохранён в проекте, происхождение входов и контрольные суммы — в паспортах прогона. '
       'Данный ноутбук воспроизводит таблицы и рисунки по сохранённой обезличенной численной сводке. '
       'Повторное решение прямой задачи требует локальных исходных масок, FEM-сетки, MATLAB/EIDORS и контактных матриц; медицинское сырьё в Git не включается. '
       'Постановка и дальнейшая последовательность: [план 40.19](40.19_План_выбора_электродов_и_модели_для_оценки_объёма.md).')
    nb=nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    dest=ROOT/'Colab Notebooks'/f'{TITLE}.ipynb'
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(dest.parent)}}).execute()
    nbf.write(nb,dest)
    exporter=HTMLExporter(); exporter.exclude_input=True; exporter.exclude_input_prompt=True; exporter.exclude_output_prompt=True
    html,_=exporter.from_notebook_node(nb); dest.with_suffix('.html').write_text(html,encoding='utf8',newline='\n')
    print(dest)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--build',action='store_true'); args=parser.parse_args()
    if args.build: build()
