"""Corrected geometry reader; deliberately performs no FEM inversion."""
from pathlib import Path
import argparse
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from scipy.ndimage import map_coordinates

from build_georg_fem_reports import ROOT, OUT as OLD, GEOMETRY, new_cells, export, number, table, read, sha, write

sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
from georg_model_contract import resolve as resolve_shared_model
OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/georg_ct_model_20261011'
SEGMENTATION=OLD/'segmentation'
if (ROOT/'MATLAB_TRKG4_real_subjects/output/georg_current_model.local.json').is_file():
    OUT,current_contract=resolve_shared_model()
    SEGMENTATION=ROOT/'MATLAB_TRKG4_real_subjects'/current_contract['segmentation_directory']
SOURCE=OUT/'body_ct_surface.stl'


def verify(out=OUT):
    out=Path(out);g=read(out/'geometry.local.json')
    current,_=resolve_shared_model()
    if out.resolve()!=current.resolve():raise ValueError('Reader must use the current checked Georg model')
    if sha(SOURCE)!=g['source_sha256'] or sha(out/'body_embedded_arms_370mm.stl')!=g['surface_sha256']:
        raise ValueError('Geometry input hash mismatch')
    producer=ROOT/'MATLAB_TRKG4_real_subjects/tools/scorpion_georg_geometry_revision.py'
    if sha(producer)!=g['producer_sha256']:raise ValueError('Geometry producer changed')
    if not g['watertight'] or g['connected_components']!=1 or g['self_intersecting_faces']:
        raise ValueError('Geometry quality check failed')
    return g


def join_table(g):
    return pd.DataFrame([{'Рука':{'left':'Правая (x < 0)','right':'Левая (x > 0)'}[key],
        'Заглубление, мм':j['inset_mm'],'Сдвиг y, мм':j['centre_shift_yz_mm'][0],
        'Сдвиг z, мм':j['centre_shift_yz_mm'][1],
        'Площадь сечения цилиндра, мм²':j['analytic_section_area_mm2'],
        'Площадь многоугольника, мм²':j['polygon_area_mm2'],
        'Доля сечения внутри тела, %':100*j['cross_section_covered_fraction']}
        for key,j in g['joins'].items()])


def contour_table(out=OUT):
    q=read(Path(out)/'contour_qc.local.json')
    return pd.DataFrame([{'Показатель':name,'Значение':value} for name,value in [
        ('Объём тела до добавления цилиндров, мл',q['native_volume_ml']),
        ('Объём прежней поверхности, мл',q['legacy_volume_ml']),
        ('Выборочное расстояние прежняя поверхность → новый КТ-контур, максимум, мм',q['legacy_to_ct']['max_mm']),
        ('Выборочное расстояние новый КТ-контур → прежняя поверхность, максимум, мм',q['ct_to_legacy']['max_mm']),
        ('Выборочное расстояние исходный контур → полигональная сетка, максимум, мм',q['simplification']['native_to_surface']['max_mm']),
        ('Выборочное расстояние полигональная сетка → исходный контур, максимум, мм',q['simplification']['surface_to_native']['max_mm'])]])


def scene(out=OUT):
    out=Path(out);surface=trimesh.load(out/'body_embedded_arms_370mm.stl',force='mesh')
    v=surface.vertices;f=surface.faces
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        opacity=.25,color='#bbb',name='Исправленная поверхность',hoverinfo='skip'))
    contacts=read(out/'contacts.local.json')
    for i,size in enumerate(contacts['sizes_mm']):
        xyz=np.array(contacts['lateral']['centres_by_size_mm'][i])
        fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines+markers',
            name=f'Боковая сборка {size} мм',marker=dict(size=3),visible=True if size==140 else 'legendonly'))
    for montage in ('1x4','2x2'):
        groups=contacts['ttrkg_'+montage]
        for i,points in enumerate(groups):
            xyz=np.array(points);current=i in (0,3)
            fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
                name=f"{montage.replace('x','×')}: {'токовые' if current else 'измерительные'}",
                marker=dict(size=6 if current else 4,color='#c43131' if current else '#247bb5'),
                showlegend=i in (0,1),legendgroup=f'{montage}-{current}',
                visible=True if montage=='1x4' else 'legendonly',
                hovertemplate='x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra></extra>'))
    fig.update_layout(template='plotly_white',height=680,title='Продольные боковые сборки и верхние/нижние токовые контакты',
        scene=dict(aspectmode='data',xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм',
            camera=dict(eye=dict(x=-1.3,y=-1.8,z=.9)),uirevision='corrected-georg-camera'),
        legend=dict(orientation='h',y=-.05),margin=dict(b=110))
    return fig


def prepare_sections(out=OUT):
    out=Path(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    # The CT affine is RAS; displayed geometry is LPS.
    ct=nib.load(SEGMENTATION/'static_ct.nii.gz');hu=np.asanyarray(ct.dataobj)
    labels=np.zeros(ct.shape,dtype=np.uint8)
    colours=['#4097dd','#dbc790','#e76b6b','#e0ab28']
    for name,value in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=nib.load(SEGMENTATION/(name+'.nii.gz'))
        np.testing.assert_allclose(image.affine,ct.affine,rtol=0,atol=1e-5)
        labels[np.asanyarray(image.dataobj)>0]=value
    contacts=read(out/'contacts.local.json');y=contacts['lateral']['y_mm']
    x=np.linspace(-225,225,650);z=np.linspace(-371.5,-19.5,600);xx,zz=np.meshgrid(x,z)
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    xyz=np.c_[xx.ravel(),np.full(xx.size,y),zz.ravel()]
    ijk=nib.affines.apply_affine(transform,xyz)
    image=map_coordinates(hu,ijk.T,order=0,mode='constant',cval=-1024).reshape(xx.shape)
    mask=map_coordinates(labels,ijk.T,order=0,mode='constant',cval=0).reshape(xx.shape)
    from lung_depth_profile import profile,skin_aligned_section
    source=trimesh.load(SOURCE,force='mesh')
    lung=nib.load(SEGMENTATION/'lung.nii.gz')
    depth=profile(source,np.asanyarray(lung.dataobj)>0,lung.affine,y,
        contacts['lateral']['z_mm'])
    depth.update(surface_sha256=sha(SOURCE),lung_mask_sha256=sha(SEGMENTATION/'lung.nii.gz'),
        producer_sha256=sha(Path(__file__).with_name('lung_depth_profile.py')),
        builder_sha256=sha(__file__))
    write(out/'lung_depth_profile.local.json',depth)
    landmarks=[dict(label=label,s_mm=offset,depth_mm=depth['depth_mm'][depth['s_mm'].index(offset)],colour=colour)
        for offset,label,colour in [(-70,'A','#b12a90'),(-35,'M','#4575b4'),(0,'Центр','#30343b'),
            (35,'N','#19955b'),(70,'B','#d26b13')]]
    write(out/'lung_depth_landmarks.local.json',landmarks)
    pd.DataFrame({k:depth[k] for k in ['s_mm','z_lps_mm','skin_x_lps_mm','lung_x_lps_mm','depth_mm','flags']}).to_csv(
        out/'lung_depth_profile.local.csv',index=False)
    points=np.array(contacts['lateral']['centres_by_size_mm'][contacts['sizes_mm'].index(140)])
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for i,ax in enumerate(axes):
        extent=[x[0],x[-1],z[0],z[-1]]
        ax.imshow(image,extent=extent,origin='lower',cmap='gray',vmin=-700,vmax=300)
        ax.imshow(np.ma.masked_where(mask==0,mask),extent=extent,origin='lower',
            cmap=ListedColormap(colours),vmin=1,vmax=4,alpha=.5,interpolation='nearest')
        if i:
            ax.plot(points[:,0],points[:,2],'o-',color='white',markeredgecolor='black',markersize=6)
            for name,p in zip(['A','M','N','B'],points):
                ax.annotate(name,(p[0],p[2]),xytext=(-16,3),textcoords='offset points',color='white',
                    bbox=dict(facecolor='black',alpha=.5,pad=1))
            ax.plot(depth['skin_x_lps_mm'],depth['z_lps_mm'],color='white',lw=1,label='Наружная поверхность')
            ax.plot(depth['lung_x_lps_mm'],depth['z_lps_mm'],color='#00efff',lw=2,label='Первая граница лёгкого')
            for offset in [-70,-35,0,35,70]:
                j=depth['s_mm'].index(offset)
                ax.plot([depth['skin_x_lps_mm'][j],depth['lung_x_lps_mm'][j]],
                    [depth['z_lps_mm'][j]]*2,color='#00efff',lw=1,ls='--')
            ax.set(xlim=(-230,0),ylim=(-300,-120),title='Сборка 140 мм; общий центр s = 0')
        else:ax.set(title=f'Полное сечение y LPS = {number(y)} мм')
        ax.set(xlabel='x LPS, мм',ylabel='z LPS, мм');ax.set_aspect('equal')
    fig.legend(handles=[Patch(color=c,label=n) for c,n in zip(colours,['Лёгкие','Кости','Сердце','Жир'])],
        loc='lower center',ncol=4,bbox_to_anchor=(.5,-.02))
    fig.savefig(out/'ct_longitudinal_sections.png',dpi=180,bbox_inches='tight');plt.close(fig)
    aligned=skin_aligned_section(depth,hu,labels,ct.affine,ray_step_mm=.5)
    np.savez_compressed(out/'ct_skin_aligned_section.local.npz',**aligned)
    fig,ax=plt.subplots(figsize=(11,7.5),constrained_layout=True)
    s,d=aligned['s_mm'],aligned['depth_mm'];valid=aligned['valid'].T
    extent=[s[0]-.5,s[-1]+.5,d[-1]+.25,-.25]
    ax.set_facecolor('#e5e7eb')
    ax.imshow(np.ma.masked_where(~valid,aligned['hu'].T),extent=extent,origin='upper',
        cmap='gray',vmin=-700,vmax=300,aspect='equal',interpolation='nearest')
    ax.imshow(np.ma.masked_where(~valid|(aligned['labels'].T==0),aligned['labels'].T),
        extent=extent,origin='upper',cmap=ListedColormap(colours),vmin=1,vmax=4,
        alpha=.5,aspect='equal',interpolation='nearest')
    ax.axhline(0,color='white',lw=2)
    ax.plot(s,depth['depth_mm'],color='#00efff',lw=1.8,label='Первая граница лёгкого')
    for item in landmarks:
        offset,label,value,colour=[item[k] for k in ('s_mm','label','depth_mm','colour')]
        ax.axvline(offset,color='white',alpha=.55,lw=.8,ls='--')
        ax.plot(offset,0,'o',color='white',markeredgecolor='black',clip_on=False)
        ax.annotate(label,(offset,0),xytext=(0,9),textcoords='offset points',ha='center',clip_on=False)
        if value is not None:
            ax.axhline(value,color=colour,lw=1,ls=(0,(6,4)))
            ax.plot(offset,value,'o',mfc='white',mec=colour,ms=5)
            ax.text(1.02,value,f'{label}: {number(value)} мм',transform=ax.get_yaxis_transform(),
                va='center',color=colour,fontsize=10,clip_on=False)
    ax.set(xlim=(-74,74),ylim=(d[-1],0),xlabel='s — смещение вдоль оси сборки, мм',
        ylabel='d — глубина от наружной поверхности вдоль +x LPS, мм',
        title='Сборка 140 мм: КТ с выровненной наружной поверхностью\nРасстояния по глубине сохранены')
    ax.set_title(ax.get_title(),pad=34)
    fig.legend(handles=[Patch(color=c,label=n) for c,n in zip(colours,['Лёгкие','Кости','Сердце','Жир'])],
        loc='lower center',ncol=4,bbox_to_anchor=(.5,-.105))
    fig.savefig(out/'ct_skin_aligned_section.png',dpi=180,bbox_inches='tight');plt.close(fig)
    write(out/'ct_skin_aligned_section_provenance.local.json',dict(
        ct_sha256=sha(SEGMENTATION/'static_ct.nii.gz'),
        mask_sha256={n:sha(SEGMENTATION/(n+'.nii.gz')) for n in ['heart','lung','bone','fat']},
        surface_sha256=sha(SOURCE),depth_profile_sha256=sha(out/'lung_depth_profile.local.json'),
        producer_sha256=sha(Path(__file__).with_name('lung_depth_profile.py')),builder_sha256=sha(__file__),
        image_sha256=sha(out/'ct_skin_aligned_section.png'),plane_y_lps_mm=y,
        mapping='d=x_LPS-x_skin(s); independent translation of each +X ray, no scaling of depths or s',
        ray_step_mm=.5,axis_step_mm=depth['axis_step_mm'],limit='right half-space x_LPS<=0; missing CT not labelled as background',
        uses_automatic_masks=True,anatomical_accuracy_validated=False,is_electrical_equivalent_h=False))
    fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
    ax.plot(depth['s_mm'],depth['depth_mm'],color='#247bb5',lw=2)
    for item in landmarks:
        offset,label,value,colour=[item[k] for k in ('s_mm','label','depth_mm','colour')]
        ax.axvline(offset,color='#777',lw=.8,ls='--')
        if value is not None:
            ax.axhline(value,color=colour,lw=1,ls=(0,(6,4)))
            ax.plot(offset,value,'o',color=colour)
            ax.text(1.02,value,f'{label}: {number(value)} мм',transform=ax.get_yaxis_transform(),
                va='center',color=colour,fontsize=10,clip_on=False)
    ax.set(xlabel='s — смещение вдоль оси сборки от общего центра, мм',
        ylabel='d — расстояние до маски лёгкого\nвдоль +x LPS, мм',xlim=(-74,74))
    ax.margins(y=.12)
    ax.grid(alpha=.25)
    fig.savefig(out/'lung_depth_along_montage.png',dpi=180);plt.close(fig)
    # Actual sections of the unextended surface at the chosen buried interfaces.
    import sys
    tool_dir=ROOT/'MATLAB_TRKG4_real_subjects/tools';sys.path.insert(0,str(tool_dir))
    from scorpion_georg_geometry_revision import section_polygon, RADIUS
    from shapely.geometry import Point
    source=trimesh.load(SOURCE,force='mesh');g=verify(out)
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    for ax,(key,j) in zip(axes,g['joins'].items()):
        polygon=section_polygon(source,j['x_mm'])
        parts=polygon.geoms if polygon.geom_type=='MultiPolygon' else [polygon]
        for part in parts:
            xy=np.array(part.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='#ddd',edgecolor='#555')
        disk=Point(j['centre_yz_mm']).buffer(RADIUS,quad_segs=64);xy=np.array(disk.exterior.coords)
        ax.fill(xy[:,0],xy[:,1],color='#247bb5',alpha=.5,edgecolor='#15527b')
        ax.set(title=f"{'Правая' if key=='left' else 'Левая'} рука; x = {number(j['x_mm'])} мм",
            xlabel='y LPS, мм',ylabel='z LPS, мм');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.savefig(out/'full_cylinder_joins.png',dpi=180);plt.close(fig)
    write(out/'section_provenance.local.json',dict(ct_sha256=sha(SEGMENTATION/'static_ct.nii.gz'),
        mask_sha256={n:sha(SEGMENTATION/(n+'.nii.gz')) for n in ['heart','lung','bone','fat']},
        plane_y_lps_mm=y,all_contacts_in_display_plane=True,poses_are_examples=True,
        interpolation='nearest-neighbour CT and masks',builder_sha256=sha(__file__)))


def montage_name(case):
    if case.startswith('TEPC'):return case.split('_')[0].replace('TEPC','ТЕПС ')
    if case.startswith('TTRKG_'):return 'ТТРКГ '+case.split('_')[1].replace('x','×')
    if case.startswith('reference_'):
        _,size,lead=case.split('_')
        return (f'Боковая {size} мм (ТТРКГ подключена)' if lead=='lateral' else
            f'ТТРКГ 1×4 (боковая {size} мм подключена)')
    raise ValueError('Unknown montage label: '+case)


def model_results(out=OUT):
    _,contract=resolve_shared_model()
    fem=ROOT/'MATLAB_TRKG4_real_subjects'/contract['fem_directory']
    rows=read(fem/'forward_results.local.json')
    nominal=[r for r in rows if r['perturbation']=='nominal']
    return pd.DataFrame([{'Монтаж':montage_name(r['id']),
        'Z, Ом':r['Z_ohm'],'Невязка системы':r['linear_system_relative_residual'],
        'Ошибка взаимности, Ом':r['reciprocity_error_ohm'],
        'Разность интегрирования, Ом':r['quadrature_difference_ohm']} for r in nominal])


def numerical_tables(out=OUT):
    _,contract=resolve_shared_model();out=Path(out);sizes=contract['mesh_comparison_mm']
    if contract.get('additional_mesh_comparison_mm'):
        sizes=list(dict.fromkeys(sizes+contract['additional_mesh_comparison_mm']))
    plans=[read(out/f'fem_h{h:g}'/'plan.local.json') for h in sizes]
    mesh=pd.DataFrame([{'Ограничение объёма элемента, мм³':h**3/6,
        'Узлы':p['nodes'],'Тетраэдры':p['tetrahedra'],
        'Минимальный объём, мм³':p['min_tetrahedron_volume_mm3'],
        'Объём модели, мл':p['volume_ml']} for h,p in zip(sizes,plans)])
    refs=read(ROOT/'MATLAB_TRKG4_real_subjects'/contract['fem_directory']/'reference_materials.local.json')
    material=pd.DataFrame({'Область':['Мягкотканный фон','Лёгкие','Кости','Сердце','Жир'],
        'Опорное сопротивление, Ом·м':refs['rho_ohm_m']})
    cases=read(out/'mesh_convergence.local.json')['cases']
    convergence=pd.DataFrame([{'Монтаж':montage_name(r['id']),'Разность сеток, Ом':r['difference_ohm'],
        'Принятый численный допуск, Ом':r['tolerance_ohm'],
        'Критерий выполнен':'Да' if r['passed'] else 'Нет'} for r in cases])
    frames=[mesh,material,convergence]
    if contract.get('additional_mesh_comparison_mm'):
        cases=read(out/'additional_mesh_comparison.local.json')['cases']
        frames.append(pd.DataFrame([{'Монтаж':montage_name(r['id']),
            'Сетка 6 мм — Z, Ом':r['coarse_ohm'],'Сетка 4 мм — Z, Ом':r['fine_ohm'],
            'Разность сеток, Ом':r['difference_ohm'],
            'Принятый численный допуск, Ом':r['tolerance_ohm'],
            'Критерий выполнен':'Да' if r['passed'] else 'Нет'} for r in cases]))
    return frames


def mesh_history_table(out=OUT):
    rows=read(Path(out)/'mesh_h8_h6_negative_comparison.local.json')
    return pd.DataFrame([{'Монтаж':montage_name(r['id']),'Первая сетка, Ом':r['coarse_ohm'],
        'Вторая сетка, Ом':r['fine_ohm'],'Разность, Ом':r['difference_ohm'],
        'Численный допуск, Ом':r['tolerance_ohm'],
        'Критерий выполнен':'Да' if r['passed'] else 'Нет'} for r in rows])


def centre_transfer_table(out=OUT):
    rows=read(Path(out)/'centre_transfer.local.json')
    return pd.DataFrame([{'Монтаж':r['montage'].replace('TEPC','ТЕПС '),
        'Контакт':r['contact'].replace('plus','+').replace('minus','−').replace('_',''),
        'Поправка проекции, мм':r['displacement_mm']} for r in rows])


def sensitivity_table(out=OUT):
    _,contract=resolve_shared_model()
    rows=read(ROOT/'MATLAB_TRKG4_real_subjects'/contract['fem_directory']/'forward_results.local.json')
    names=['Фон','Лёгкие','Кости','Сердце','Жир']
    return pd.DataFrame([dict({'Монтаж':r['id'].split('_')[0].replace('TEPC','ТЕПС ')},
        **{name+' — J, Ом':value for name,value in zip(names,r['J_logrho_ohm'])})
        for r in rows if r['perturbation']=='nominal'])


def individual_position_table(out=OUT):
    rows=read(Path(out)/'individual_position_uncertainty.local.json')
    return pd.DataFrame([{'Монтаж':f'ТЕПС {r["montage"]}',
        'Смещённый контакт':r['contact'].replace('plus','+').replace('minus','−').replace('_',''),
        'Смещение по касательной, мм':r['shift_mm'],
        'Изменение Z, Ом':r['difference_from_nominal_ohm']} for r in rows])


def position_plot(out=OUT):
    rows=read(Path(out)/'position_uncertainty.local.json');fig=go.Figure()
    for n in range(2,8):
        rr=[r for r in rows if r['montage']==n]
        fig.add_trace(go.Scatter(x=[r['amplitude_mm'] for r in rr],
            y=[r['max_abs_change_ohm'] for r in rr],mode='lines+markers',name=f'ТЕПС {n}',
            hovertemplate='Смещение %{x} мм: |ΔZ| ≤ %{y:.4g} Ом в проверенных направлениях<extra>%{fullData.name}</extra>'))
    fig.update_layout(template='plotly_white',height=520,title='Влияние совместных смещений контактов',
        xaxis_title='Заданная амплитуда смещения вдоль касательной, мм',
        yaxis_title='Максимальное |ΔZ| среди четырёх направлений, Ом')
    return fig


def patch_scene(out=OUT):
    _,contract=resolve_shared_model()
    fem=ROOT/'MATLAB_TRKG4_real_subjects'/contract['fem_directory']
    mesh=trimesh.load(Path(out)/'body_embedded_arms_370mm.stl',force='mesh')
    v,f=mesh.vertices,mesh.faces
    fig=go.Figure(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
        color='#c7c8c9',opacity=.2,name='Тело',hoverinfo='skip'))
    buttons=[]
    for n in range(2,8):
        patches=read(fem/'scenarios'/f'TEPC{n}_nominal_patches.local.json')
        for p,label in zip(patches,['I+','U+','U−','I−']):
            v=np.asarray(p['vertices_lps_mm']);f=np.asarray(p['faces'])-1
            fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
                name=f'ТЕПС {n}: {label}',color='#e77d29' if label.startswith('I') else '#287fb8',
                opacity=1.,visible=n==2,showlegend=True))
        mask=[True]+[case==n for case in range(2,8) for _ in range(4)]
        buttons.append(dict(label=f'ТЕПС {n}',method='restyle',args=[{'visible':mask}]))
    fig.update_layout(template='plotly_white',height=680,title='Расчётные контактные площадки ТЕПС',
        scene=dict(aspectmode='data',xaxis_title='x LPS, мм',yaxis_title='y LPS, мм',zaxis_title='z LPS, мм',
            uirevision='shared-georg-patches'),uirevision='shared-georg-patches',
        updatemenus=[dict(buttons=buttons,direction='right',x=0,y=1.1)],legend=dict(orientation='h',y=-.05))
    return fig


def prepare_contour_image(out=OUT):
    import matplotlib.pyplot as plt
    ct=nib.load(SEGMENTATION/'static_ct.nii.gz');hu=np.asanyarray(ct.dataobj)
    body=np.asanyarray(nib.load(Path(out)/'body_mask.nii.gz').dataobj)>0
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    transfer=read(Path(out)/'centre_transfer.local.json')
    for n,ax in zip(range(2,8),axes.flat):
        point=next(r for r in transfer if r['montage']==f'TEPC{n}' and r['contact']=='V_plus')
        xyz=np.array([point['original_lps_mm'],point['projected_lps_mm']])
        vox=nib.affines.apply_affine(np.linalg.inv(ct.affine),xyz*[-1,-1,1]);k=int(round(vox[1,2]))
        ax.imshow(hu[:,:,k],cmap='gray',vmin=-800,vmax=300)
        ax.contour(body[:,:,k],levels=[.5],colors=['#12b6ac'],linewidths=.8)
        ax.scatter(vox[0,1],vox[0,0],s=25,c='#ef7b25',label='Исходный центр U+')
        ax.scatter(vox[1,1],vox[1,0],s=35,marker='+',c='#2366dc',label='Проекция на контур')
        ax.set_title(f'ТЕПС {n}, срез через U+');ax.axis('off')
    axes[0,0].legend(loc='lower left',fontsize=7);fig.tight_layout()
    fig.savefig(Path(out)/'ct_contour_and_transfers.png',dpi=170);plt.close(fig)


def build(out=OUT):
    out=Path(out);g=verify(out);prepare_sections(out);prepare_contour_image(out)
    _,model_contract=resolve_shared_model()
    selected_plan=read(ROOT/'MATLAB_TRKG4_real_subjects'/model_contract['fem_directory']/'plan.local.json')
    additional=model_contract.get('additional_mesh_comparison_mm')
    additional_note=''
    if additional:
        comparison=read(out/'additional_mesh_comparison.local.json')['cases']
        failed=sum(not r['passed'] for r in comparison)
        largest=max(comparison,key=lambda r:r['difference_ohm']/abs(r['fine_ohm']))
        additional_note=(f' Дополнительная проверка сеток с параметрами {number(additional[0],0)} '
            f'и {number(additional[1],0)} мм также не выполнила допуск в {failed} из {len(comparison)} '
            f'случаев. Наибольшее относительное расхождение — '
            f'{number(100*largest["difference_ohm"]/abs(largest["fine_ohm"]),2)}% '
            f'для монтажа «{montage_name(largest["id"])}». '
            'Сценарии смещения на третьей сетке не пересчитывались.')
    precision_note=('Численные критерии проверенных случаев выполнены. '
        if model_contract['status']=='numerical_reference_checked' else
        'Геометрические и алгебраические проверки выполнены; критерий сеточной точности не выполнен. '
        'Геометрия закреплена как общий вход. Абсолютные FEM-импедансы остаются исследовательскими; '
        'они не разрешают количественную инверсию сопротивлений. ')
    cells,md,code=new_cells()
    md('# 20.20. Исправленная индивидуальная геометрия КТ/FEM Георгия\n\n'
       '**Редакция от 11 октября 2026 года. Наружный контур заново построен по КТ; выполнен новый прямой FEM-расчёт.** '
       'Автор уточнил ориентацию боковых сборок и токовых контактов и потребовал полного сечения '
       'соединения цилиндров рук с телом. Предыдущая геометрия этим условиям не соответствовала. '
       'Численные результаты 33.22 от 8 октября сохранены как исторический расчёт и не относятся '
       'к настоящей геометрии. '+precision_note+'\n\n'
       '## 1. Входы и последовательность\n\n'
       'Для подготовки конечноэлементной модели (FEM) использованы новая наружная поверхность '
       'полного статического КТ-тома и четыре маски: '
       'сердце, лёгкие, кости и жир. Остальной объём представляет мягкотканный фон. '
       'Маски получены автоматически TotalSegmentator 2.18; независимая точность границ не установлена. '
       'КТ снята на глубоком вдохе с задержкой дыхания и относится к другой сессии, чем Scorpion v2. '
       'Динамическая КТ сердца не подменяет полный том грудной клетки.\n\n'
       'Контур выделен из КТ по порогу −450 HU. Тонкие внешние фрагменты удалены двумерным '
       'размыканием радиусом два вокселя; затем в каждом срезе выбрана основная компонента, '
       'выполнено замыкание радиусом один воксель и заполнение внутренних полостей. '
       'Контур визуально сопоставлен с КТ на всех 280 аксиальных срезах: стол отделён от тела. '
       'На отдельных срезах сохраняются небольшие поверхностные выступы, происхождение которых '
       'не установлено; их влияние на локальный контакт и расчёт пока не отделено от других '
       'источников погрешности. Это автоматическая реконструкция с геометрическим контролем; '
       'независимая анатомическая точность не установлена. Прежнее замыкание радиусом 10 мм '
       'не применяется. Четыре внутренние маски сохранены из полной автоматической сегментации.\n\n'
       'Порядок данного этапа: проверить сечения исходной поверхности → подобрать заглубление цилиндров '
       '→ объединить объёмы и проверить поверхность → перенести центры и построить контакты '
       '→ построить новые объёмные сетки → выполнить прямые FEM-расчёты и сценарии смещения. '
       'Выполнены геометрические этапы, новая сетка, прямой расчёт и сценарии ошибки установки ТЕПС. '
       'Полный поиск общего центра боковых сборок и новая инверсия сопротивлений ещё не выполнены.')
    relative=out.relative_to(ROOT).as_posix()
    code("from pathlib import Path\nimport sys\nfrom IPython.display import display,HTML,Image\nhere=Path.cwd()\n"
         "module_dir=next(p for p in [here,here/'scorpion_v2',here/'Colab Notebooks/scorpion_v2'] if (p/'build_georg_geometry_revision_report.py').exists())\n"
         "sys.path.insert(0,str(module_dir.resolve()))\nfrom build_georg_geometry_revision_report import ROOT,verify,join_table,contour_table,scene,table,model_results,numerical_tables,mesh_history_table,centre_transfer_table,sensitivity_table,individual_position_table,position_plot,patch_scene\n"
         f"OUT=ROOT/{relative!r}\ng=verify(OUT)\n"
         "def show(fig): display(HTML(fig.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False})))\n")
    md('**Таблица 1.** Изменение наружного контура и влияние сокращения числа граней. '
       'Расстояния вычислены по 15 тысячам воспроизводимо выбранных точек каждой поверхности. '
       'Их максимум не является гарантированной максимальной ошибкой всей поверхности. '
       'Опорная поверхность построена по исходной КТ-маске; упрощённая поверхность '
       'используется сеточным генератором. После экспорта STL выполнена ограниченная '
       'локальная коррекция пересекающихся граней с сохранением топологии. '
       'Окончательный скрипт повторно дал побайтно совпадающие маску и поверхности.')
    code('display(table(contour_table(OUT)))')
    md('## 2. Полное соединение цилиндров рук\n\n'
       'Автор сообщил обхват 37 см для обеих рук на уровне токовых электродов и принял цилиндрическую '
       'модель. Радиус равен 370/(2π) = 58,8873 мм; площадь поперечного сечения — 10 894,16 мм². '
       'Прежние срезы наружной поверхности имели площади около 2721 и 3676 мм². '
       'Присоединение большого цилиндра только через эти срезы не обеспечивало требуемого полного контакта.\n\n'
       'Новый подбор рассматривает плоскости, смещённые внутрь исходной поверхности на 0,5–109,5 мм '
       'с шагом 1 мм. В каждой плоскости центр круга выбирается в области, где всё сечение цилиндра '
       'помещается внутри тела, с геометрическим запасом 0,15 мм. Из допустимых кандидатов выбирается '
       'минимум суммы квадратов заглубления и сдвига центра относительно исходного среза. '
       'Это геометрическое правило, а не восстановление анатомического положения плеча.\n\n'
       '**Таблица 2.** Заглубления и сдвиги центров цилиндров относительно прежних срезов. '
       'Полное сечение находится внутри исходного объёма. Круг представлен 256-сторонним многоугольником; '
       'отличие его площади от площади круга составляет около 0,010%. При объединении добавлен внутренний '
       'нахлёст 0,5 мм. Цилиндры не выходят за исходные пределы по y и z; их продолжение по x '
       'является заданной моделью рук.\n\n'
       f'Итоговая поверхность замкнута, имеет одну связную компоненту и положительный объём '
       f'{number(g["volume_ml"],6)} мл. Самопересекающихся граней не обнаружено. '
       'Эти проверки подтверждают целостность поверхности и полноту выбранных сечений; '
       'точность реальной формы рук ими не установлена.')
    code('display(table(join_table(g)))')
    md('## 3. Контакты и продольные сборки\n\n'
       'По уточнению автора в монтаже 1×4 единственный токовый контакт каждой руки перенесён '
       'с передней поверхности наверх, поворотом на 90°. В монтаже 2×2 добавлен диаметрально '
       'противоположный нижний контакт. Четыре измерительных контакта 1×4 расположены через 90°; '
       'измерительная пара 2×2 также повёрнута на 90° и расположена сверху и снизу '
       'по уточнению автора от 09.10.2026. '
       'В полной электродной модели (CEM) контакты одной электрической группы '
       'должны иметь общий потенциал.\n\n'
       'Продольные расстояния контактов рук сохранены: 20 мм для измерительных и 120 мм для токовых '
       'от прежней плоскости обрезки КТ. Начало отсчёта не перенесено на заглублённое соединение. '
       'Эта плоскость является модельным ориентиром, а не измеренной анатомической точкой. '
       'Номинальные центры проверены на конечной поверхности с допуском 0,02 мм.\n\n'
       'В системе координат LPS положительные оси x, y и z направлены соответственно влево, '
       'назад и к голове относительно испытуемого; их начало перенесено из КТ. '
       'Боковые сборки теперь ориентированы по z LPS — вдоль тела. '
       'Для всех размеров сохранён один иллюстративный центр y = 0, z = −210 мм; '
       'его фактическая локализация по эксперименту пока неизвестна. Центры A, M, N, B '
       'задаются на уровнях −L/2, −L/4, L/4, L/2 относительно общего центра, '
       'где L — номинальное расстояние между токовыми центрами в миллиметрах. Они проецируются '
       'на правую поверхность лучами вдоль +x LPS. Отрицательная сторона x соответствует правой стороне человека. '
       'Это геометрический пример, используемый для проверки направления; тройка прежних выбранных точек '
       'не используется вместо полного поиска.\n\n'
       '**Рисунок 1.** Исправленная наружная поверхность и центры контактов. Монтаж 2×2 и другие размеры '
       'можно включить через легенду. Маркеры изображают центры, а не дискретизированные FEM-площадки. '
       'Для новой CEM-постановки принят номинальный диаметр контакта 5 мм и площадь 19,635 мм²; '
       'фактическая проводящая площадь Scorpion ещё не установлена.')
    code('show(scene(OUT))')
    md('## 4. КТ-сечение по оси сборок\n\n'
       '**Рисунок 2.** Продольное КТ-сечение y LPS = 0 мм: полный том и сборка 140 мм. '
       'Все размеры имеют один центр; меньшие сборки занимают внутренние участки той же оси. '
       'Метки A и B обозначают токовые центры, M и N — измерительные. Показаны маски лёгких, костей, '
       'сердца и жира; отсутствие ткани в данном сечении не означает отсутствия её маски в объёме. '
       'Использована интерполяция ближайшего соседа. Геометрическая поверхность и КТ могут различаться '
       'из-за обработки наружного контура; выборочные расстояния поверхностей приведены в таблице 1. '
       'Голубая линия отмечает первое пересечение лучей вдоль +x LPS с маской правого лёгкого; '
       'пунктиром показаны расстояния от наружной поверхности до этой линии. '
       'Направление выбрано для сопоставления с принятой проекцией электродов. '
       'Оно не является локальной нормалью к коже или кратчайшим расстоянием до лёгкого.')
    code("display(Image(filename=str(OUT/'ct_longitudinal_sections.png')))")
    md('**Рисунок 2, дополнительная панель.** КТ и те же тканевые маски после выравнивания '
       'наружной поверхности. Для каждого смещения s начало луча перенесено на наружную '
       'STL-поверхность, а глубина d равна разности координаты x LPS и координаты поверхности '
       'на этом уровне. Перенос не растягивает луч: расстояния между границами и толщины '
       'анатомических элементов вдоль +x LPS сохранены с шагом 0,5 мм. '
       'Ось s, шаг 1 мм и общий центр совпадают с рисунком 3. Белая горизонтальная линия '
       'соответствует нулевой глубине, голубая — первой границе маски правого лёгкого. '
       'Вертикальные штриховые линии задают положения A, M, центра, N и B вдоль оси сборки. '
       'Цветные горизонтальные штриховые линии показывают глубину лёгкого в этих пяти положениях; '
       'подписи справа содержат обозначение и расстояние в миллиметрах. Кружки на голубой границе '
       'соединяют каждое положение с соответствующим уровнем глубины. '
       'Серые области вне доступного КТ или за срединной плоскостью x LPS = 0 не интерпретируются '
       'как мягкие ткани. Маски остаются автоматическими; расстояния зависят от их точности '
       'и обработки наружной поверхности. Это выровненное изображение реальной неоднородной '
       'анатомии в выбранном сценарии, а не плоская двуслойная модель или оценка электрической h.')
    code("display(Image(filename=str(OUT/'ct_skin_aligned_section.png')))")
    md('**Рисунок 3.** Профиль анатомического расстояния до маски лёгкого вдоль общей оси сборок. '
       'Смещение s отсчитывается вдоль z LPS от иллюстративного центра на уровне −210 мм: '
       'от −70 до +70 мм с шагом 1 мм. Положительное направление s соответствует направлению к голове. '
       'A, M, N и B — центры контактов сборки 140 мм. Расстояние d вычислено от наружной STL-поверхности '
       'до первого ненулевого вокселя маски лёгкого вдоль +x LPS; луч заканчивается на плоскости x LPS = 0, '
       'чтобы не подменить правое лёгкое левым. Шаг вдоль луча составляет 0,5 мм; '
       'ошибка такого поиска границы не превышает одного шага относительно границы дискретной маски. '
       'Погрешности сегментации, обработки контура тела и восстановления положения этим пределом не охвачены. '
       'Если лёгкое на луче отсутствует или поле КТ неполно, расстояние не назначается.\n\n'
       'Профиль построен независимо от импеданса и сопротивлений. Он показывает пространственную '
       'неоднородность анатомии при данном примере положения. Электрически эквивалентная толщина h '
       'двуслойной модели пока не определена через этот профиль: способ такого сведения требуется '
       'проверить по индивидуальной конечноэлементной модели. Сплошная синяя линия представляет '
       'профиль d(s); вертикальные серые штриховые линии — положения электродов и центра. '
       'Горизонтальные цветные штриховые линии и кружки отмечают значения d в этих положениях; '
       'подписи справа показывают их в миллиметрах. При неопределённой глубине соответствующий '
       'горизонтальный уровень не проводится.')
    code("display(Image(filename=str(OUT/'lung_depth_along_montage.png')))")
    md('## 5. Полный перерасчёт положения и сопротивлений\n\n'
       'Следующий этап — поиск общего центра по всей допустимой правой поверхности грудной клетки '
       'с проверкой размещения всех пяти размеров. Направление вдоль тела задаёт номинальную ориентацию; '
       'ошибку установки необходимо исследовать отдельно. Для каждого положения сопротивления '
       'мягкотканного фона, лёгкого и жира следует оптимизировать совместно либо вложенным поиском. '
       'Поиск положения при одной фиксированной тройке сопротивлений не заменяет эту обратную задачу. '
       'Цель — карта невязки и допустимых тканевых параметров по всей области, с оценкой '
       'геометрической и электрической неоднозначности. Это план следующего расчёта, а не выполненный результат.\n\n'
       'Для новой поверхности построены новые объёмные сетки и материальные матрицы четырёх '
       'масок и фона. Контактные площадки и CEM-матрицы собраны заново для каждого проверяемого '
       'положения. Матрицы и оценки 8 октября не использованы. Проверки двух сеток, интегрирования, '
       'взаимности, сборки EIDORS и производных представлены ниже. Новая инверсия ещё требуется. '
       'Для двуслойного поиска в 33.21 приняты физиологически мотивированные литературные интервалы '
       'при 100 кГц. Их перенос на мягкотканный фон, лёгкое и жир FEM требует отдельного обоснования '
       'для каждого компартмента; прежние технические пределы 0,1–100 Ом·м не задают физиологический диапазон.\n\n'
       'До этого пересчёта индивидуальные ρ₁, ρ₂ и ρж не выбраны для коррекции ТТРКГ. '
       'Влияние площади контактов на сигнал и чувствительность каждого компартмента следует '
       'сравнивать после проверки геометрии и независимо для токовых и измерительных площадок. '
       'Расчёт УО и ФВ здесь не выполнялся. Хэши поверхности, кода, КТ и масок сохранены '
       'в локальном паспорте; первичные массивы не включены в читательский экспорт.')
    md('## 6. Перенос авторских центров на новый контур\n\n'
       'Исходные 24 центра ТЕПС 2–7 сохранены. Для расчёта использованы их ближайшие проекции '
       'на новый КТ-контур в той же системе LPS. Поправка проекции достигает 4,54 мм; '
       'она не измеряет ошибку ручной реконструкции. Соответствие файлов записям 012–017 '
       'подтверждено автором. Номер BEI и физическая полярность пока не подтверждены.\n\n'
       '**Рисунок 4.** КТ-срезы через проекцию центра U+ каждого монтажа. Бирюзовая линия — '
       'граница новой объёмной маски тела. Оранжевые кружки — исходные центры, синие крестики — '
       'их проекции; обе точки показаны в координатах выбранного среза. Это проверка автоматического '
       'переноса, а не доказательство точного воспроизведения установки во время записи.')
    code("display(Image(filename=str(OUT/'ct_contour_and_transfers.png')))\ndisplay(table(centre_transfer_table(OUT)))")
    md('## 7. Новый прямой FEM-расчёт\n\n'
       'Геометрия и тканевое разбиение закреплены общим паспортом модели Георгия. Ноутбук читает '
       'готовые результаты и не выполняет FEM. Все расчёты используют сердце, лёгкие, кости, '
       'жир и оставшийся мягкотканный фон. Для искусственных продолжений рук принят мягкотканный '
       'фон: их состав по КТ неизвестен. КТ соответствует глубокому вдоху.\n\n'
       'Опорные сопротивления взяты из частотных значений Габриэля при 100 кГц в общем реестре: '
       'фон заменён мышечной тканью, лёгкое соответствует вдоху, вся маска сердца — миокарду, '
       'вся костная маска — кортикальной кости. Это литературные допущения прямого расчёта; '
       'индивидуальные сопротивления Георгия ещё не восстановлены. Проводимость задана '
       'скалярной действительной величиной; анизотропия и ёмкостная составляющая тканей '
       'в этой постановке не учитываются.\n\n'
       'Объёмные сетки построены TetGen по одной и той же проверенной полигональной границе. '
       'Ограничения максимального объёма элементов указаны в таблице; '
       'это параметры сеточного генератора, а не равномерный шаг сетки. Вокруг всех '
       'проверяемых центров рёбра границы разделены до длины не более 0,7 мм. '
       'Координаты исходной границы при этом сохранены. Тканевые доли каждого элемента '
       'получены по 256 и 1024 точкам последовательности Соболя; их сумма равна единице. '
       'Пересечения автоматических масок разрешены в порядке приоритета: лёгкие, кости, '
       'сердце, жир, фон. Это правило назначения материала сохранено в паспорте.\n\n'
       'Для каждого физического контакта принят диск диаметром 5 мм, площадь 19,635 мм². '
       'Фактическая площадь неизвестна. Удельное контактное сопротивление CEM равно '
       f'{number(selected_plan["contact_impedance_ohm_m2"],9)} Ом·м². Это прежний расчётный '
       'сценарий проекта; параметр не измерен на Scorpion. Контакты одной электрической группы '
       'эквипотенциальны. Для ТЕПС в модель включены четыре размеченных контакта. Неизвестный '
       'дополнительный одновременный монтаж не добавлен произвольно; его присутствие потребует '
       'отдельной CEM-постановки. Расчётный знак соответствует I+ → I− и U+ − U−. '
       'В сценариях совместного подключения боковой сборки и ТТРКГ возбуждается одна '
       'токовая пара; остальным электрическим группам задан нулевой суммарный ток. '
       'Это постановка прямого расчёта. Её соответствие одновременному зондированию '
       'двух каналов Scorpion отдельно не установлено.\n\n'
       '**Таблица 3.** Импедансы шести ручных реконструкций в литературном сценарии и численные '
       'проверки. Невязка относится к линейной системе FEM; ошибка взаимности — к перестановке '
       'токовой и измерительной пар; разность интегрирования — к двум правилам отображения масок. '
       'Эти проверки не подтверждают физическую адекватность тканей или приборной схемы.')
    code("display(table(model_results(OUT)))")
    md('**Таблица 4.** Сетки после локального сгущения, фактические опорные сопротивления '
       'и сравнение импедансов. Принятый допуск сравнения сеток равен 0,02 Ом + 0,5% '
       'от модуля импеданса более плотной сетки. Допуск сравнения интегрирования равен '
       '0,02 Ом + 0,1%. Это критерии численной проверки данной постановки. '
       'Сравнение двух сеток охватывает исходные ТЕПС, монтажи рук и иллюстративные '
       'боковые сборки; оно не доказывает сходимость для любого нового положения.')
    code("for frame in numerical_tables(OUT): display(table(frame))")
    md('Первая пара сеток, с ограничениями объёма элемента 85,333 и 36 мм³, '
       'не удовлетворила этому допуску в 17 из 19 исходных случаев. Наибольшее '
       'относительное расхождение составило около 3,48% для ТЕПС 2. Площадки '
       'контактов на двух сетках совпали: расхождение возникло при объёмной '
       'дискретизации и отображении тканей. Этот отрицательный результат сохранён '
       'в таблице 4; первая сетка не принята численным референсом.'+additional_note+'\n\n'
       'На выбранной сетке выполнены 163 сценария: 150 для шести монтажей ТЕПС, '
       'три монтажа на руках и десять сценариев пяти боковых размеров с ТТРКГ. '
       'Для всех этих случаев ошибка площади каждого дискретного контакта не превышает 1,51%, '
       'смещение центроида относительно заданного центра — 0,80 мм. '
       'На выбранной сетке критерий сравнения интегрирования выполнен для всех исходных случаев; '
       'на первой сетке его не выполнил ТЕПС 5. Алгебраические проверки не устраняют '
       'выявленное расхождение объёмной дискретизации.')
    if additional:
        md('После локального сгущения третья сетка содержит 120 элементов объёмом менее '
           '10⁻¹² мм³. Для каждого знак и объём повторно проверены по точным значениям '
           'записанных координат с 80 десятичными знаками. Все объёмы положительны; '
           'наибольшее относительное расхождение с вычислением двойной точности — '
           '7,89·10⁻¹². Элементы сохранены, общий объём не изменился. Эта проверка '
           'исключает ошибочное удаление малых положительных элементов, но не устанавливает '
           'точность FEM-импеданса; она оценивается отдельными сеточными сравнениями.')
    if model_contract['mesh_comparison_mm']!=[8,6]:code("display(table(mesh_history_table(OUT)))")
    md('**Таблица 5.** Локальные чувствительности к свойствам пяти областей: '
       'Jᵢ = ∂Z/∂ln ρᵢ, в Ом. При малом относительном изменении сопротивления '
       'δρᵢ/ρᵢ ожидаемое изменение импеданса равно Jᵢ·δρᵢ/ρᵢ. Производные '
       'вычислены при неизменных геометрии и свойствах остальных областей; '
       'для одного исходного монтажа все пять производных проверены центральными '
       'конечными разностями. Эти величины характеризуют выбранную прямую модель; '
       'они не являются долями объёма или измеренными пульсовыми вкладами тканей.')
    code("display(table(sensitivity_table(OUT)))")
    condition=read(out/'material_sensitivity.local.json')['condition_number']
    md('Для шести исходных ТЕПС матрица чувствительности к логарифмам сопротивлений '
       f'фона, лёгких и жира имеет число обусловленности {number(condition,2)}. '
       'Это локальная невзвешенная оценка при фиксированных положении, костях и сердце. '
       'Она не доказывает идентифицируемость сопротивлений при неизвестной установке, '
       'погрешности измерений и расхождении анатомической модели.')
    md('**Рисунок 5.** Фактические граничные треугольники конечных контактов. Оранжевые площадки — '
       'токовые I+ и I−, синие — измерительные U+ и U−. Кнопки выбирают монтаж; ракурс сохраняется. '
       'Это площадки, использованные при сборке CEM, а не условные маркеры центров.')
    code("show(patch_scene(OUT))")
    md('## 8. Неопределённость установки\n\n'
       'Для ТЕПС проверены совместные смещения всех четырёх контактов на 1, 3, 5 и 10 мм '
       'в двух локальных касательных направлениях с обоими знаками. Каждый контакт повторно '
       'спроецирован на поверхность; площадки и контактные матрицы пересобраны. Дополнительно '
       'проверены отдельные продольные смещения каждого контакта на ±5 мм. Заданные амплитуды '
       'не являются доверительными интервалами неизвестной погрешности ручной установки.\n\n'
       'Все показанные смещения относятся к сетке, указанной общим паспортом модели. '
       'Сходимость каждого смещённого положения на независимой сетке отдельно не доказана; '
       'при незавершённой сеточной проверке эти зависимости показывают исследовательские '
       'сценарии, а не гарантированный предел погрешности.\n\n'
       'Для ТЕПС 5 наименьшее расстояние между исходными центрами равно 12,8 мм. '
       'Отдельный сдвиг центра I− на +5 мм с повторной проекцией изменил Z '
       'с 155,26 до 224,87 Ом, то есть на 69,62 Ом. Фактическое перемещение '
       'спроецированного центра составило 5,20 мм. Этот пример показывает '
       'необходимость учитывать ошибку отдельных контактов; величина не является '
       'измеренной ошибкой установки или доказанным расхождением с записью.\n\n'
       '**Рисунок 6.** Каждая цветная линия соответствует одному монтажу ТЕПС. Кружки показывают '
       'наибольшее абсолютное изменение импеданса среди четырёх совместных направлений при '
       'данной амплитуде; соединяющие сплошные отрезки служат для чтения дискретных результатов. '
       'Между отмеченными амплитудами расчёты не выполнены. Легенда позволяет отключать монтажи. '
       'Это чувствительность прямой модели при фиксированных литературных сопротивлениях; '
       'она не заменяет совместную обратную задачу положения и свойств тканей.\n\n'
       'Новая геометрия является общим входом дальнейших расчётов. Статус численной точности '
       'сохраняется отдельно и проверяется до количественной инверсии. Полная локализация '
       'боковых сборок, идентификация сопротивлений, сопоставление с принятыми фазами записей '
       'и независимая проверка УО/ФВ остаются отдельными этапами.')
    code("show(position_plot(OUT))")
    md('**Таблица 6.** Разность импеданса при отдельном смещении каждого контакта '
       'на ±5 мм относительно исходного монтажа. Смещение задаётся вдоль проекции '
       'оси z LPS на локальную касательную плоскость, после чего центр возвращается '
       'ближайшей проекцией на поверхность. Знак разности показывает направление '
       'изменения Z. Совместный перенос и ошибка одного контакта проверяются раздельно: '
       'они меняют взаимные расстояния и поле тока различным образом.')
    code("display(table(individual_position_table(OUT)))")
    export(GEOMETRY,cells,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    build(parser.parse_args().out)
