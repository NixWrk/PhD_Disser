"""Уровни BASE2 на вдохе и собственные КТ-модели серии МГТУ.

Первичные CSV и принятая разметка доступны только для чтения. Локальные
планы содержат пути к медицинским данным и не предназначены для Git.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
ALIASES=('georg','nik')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def collect(data_root,derived_root,model_root,out):
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT/'Colab Notebooks'))
    from exploratory_analysis import interval_median
    data_root=Path(data_root).resolve();derived_root=Path(derived_root).resolve();out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    snapshot=derived_root/'exp02/exploratory/33.06_side_arrays_exploratory.json'
    d=read(snapshot);provenance={(r['subject_id'],r['size_mm']):r for r in d['input_provenance']}
    subjects=[]
    for alias in ALIASES:
        sid='exp02_'+alias;rows=[]
        for row in sorted([r for r in d['static_observations'] if r['subject_id']==sid],key=lambda x:x['size_mm']):
            size=row['size_mm'];p=provenance[sid,size]
            annotation_path=derived_root/'exp02/annotations/breathing'/(p['record_id']+'.json')
            if sha(annotation_path)!=p['breathing_sidecar_sha256']:raise ValueError('Изменилась принятая разметка')
            annotation=read(annotation_path)
            if annotation['qc']['status']!='accepted' or annotation['subject_id']!=sid or annotation['size_mm']!=size:
                raise ValueError('Непринятая или чужая разметка дыхания')
            source=(data_root/annotation['input']['relative_path']).resolve();source.relative_to(data_root)
            if sha(source)!=annotation['input']['sha256'] or sha(source)!=p['source_csv_sha256']:
                raise ValueError('Исходная запись не соответствует разметке')
            frame=pd.read_csv(source,usecols=['TIME_s','BASE_2_Ω'])
            t=frame['TIME_s'].to_numpy(float);z=frame['BASE_2_Ω'].to_numpy(float)
            if not np.all(np.isfinite(t)) or not np.all(np.diff(t)>0):raise ValueError('Некорректное время записи')
            interval=annotation['accepted_modes']['задержка_вдох'];median=interval_median(t,z,interval,margin_s=.5)
            if abs(median-row['z_inhale_ohm'])>1e-8:raise ValueError(f'Не воспроизведена медиана {sid} {size}: {median}')
            mask=(t>=interval[0]+.5)&(t<=interval[1]-.5);indices=np.flatnonzero(mask)
            q05,q95=map(float,np.quantile(z[mask],[.05,.95]))
            rows.append(dict(record_id=f'BMSTU-{alias[0].upper()}-{size:03d}',source_record_id=p['record_id'],
                subject_alias=alias,size_mm=size,Z_ohm=median,raw_q05_ohm=q05,raw_q95_ohm=q95,
                channel=2,column='BASE_2_Ω',annotation_status='accepted',manual_annotation_accepted=True,
                source_path=str(source),source_sha256=sha(source),annotation_path=str(annotation_path),
                annotation_sha256=sha(annotation_path),accepted_hold_s=interval,hold_margin_s=.5,
                selected_window_s=[float(t[indices[0]]),float(t[indices[-1]])],
                window_samples_half_open=[int(indices[0]),int(indices[-1]+1)],
                fs_hz=float(1/np.median(np.diff(t))),source_qc_status='machine_QC_pending_manual_review',
                channel1_anomaly_does_not_exclude_BASE2=alias=='georg' and size in (90,100)))
            print(f'{alias} L={size} мм Z={median:.8f} Ом: медиана подтверждена',flush=True)
        expected=list(range(50,141,10));expected.remove(100) if alias=='nik' else None
        if [r['size_mm'] for r in rows]!=expected:raise ValueError('Неверный независимый размерный ряд')
        if len({r['source_sha256'] for r in rows})!=len(rows):raise ValueError('Дублированная запись в размерном ряду')
        subjects.append(dict(subject_alias=alias,model_directory=str((Path(model_root)/alias).resolve()),observations=rows))
    nik90=next(r for s in subjects if s['subject_alias']=='nik' for r in s['observations'] if r['size_mm']==90)
    duplicate=Path(nik90['source_path']).with_name('100nik.csv')
    if not duplicate.exists() or sha(duplicate)!=nik90['source_sha256']:raise ValueError('Не подтверждён известный дубликат Ника 100/90 мм')
    plan=dict(schema='bmstu_uniform_rho_observations_v1',experiment_id='exp02',instrument='BMSTU_reocardiomonitor',
        subjects=subjects,static_snapshot_path=str(snapshot),static_snapshot_sha256=sha(snapshot),
        excluded=[dict(subject_alias='nik',size_mm=100,duplicate_of_size_mm=90,source_sha256=sha(duplicate))],
        scalar_impedance_column='BASE_2_Ω',calibration_gain_applied=1,offset_ohm=0,
        status='exploratory_hypothesis_not_validated',breathing_annotation_status='accepted',
        global_record_QC_status='pending_manual_review',producer_sha256=sha(__file__))
    (out/'data_worker_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_new(out/'observations.local.json',plan)
    return out/'observations.local.json'


def make_geometry(observations,alias,out,*,target_edge_mm=0,refinement_levels=2):
    """Новая контактная сетка на собственной поверхности; геометрия не сглаживается."""
    import numpy as np
    import trimesh
    from scipy.io import loadmat,savemat
    sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
    import ct_fem_core as core
    import refine_surface_contact_mesh as refinement
    from georg_quality_selected_refinement import select_parents,refine_screened
    import ct_fem_contact_placement as placement_kernel
    refinement.np=np
    if not np.isfinite(target_edge_mm) or target_edge_mm<0 or refinement_levels not in range(7):
        raise ValueError('Нужны неотрицательная цель ребра и от нуля до шести уровней')
    data=read(observations);subject=next(s for s in data['subjects'] if s['subject_alias']==alias)
    model=Path(subject['model_directory']);config=read(model/'configuration.local.json')
    source_mesh=model/'materials/mesh.mat';parent=loadmat(source_mesh)
    nodes=np.asarray(parent['nodes_mm'],float);old_nodes=nodes.copy()
    tets=np.asarray(parent['elems'],np.int32)-1;boundary=np.asarray(parent['boundary'],np.int32)-1
    native=Path(config['body']['source_stl'])
    if sha(native)!=config['body']['source_stl_sha256']:raise ValueError('Изменилась собственная поверхность КТ')
    surface=trimesh.load(native,force='mesh',process=False)
    fem_surface=trimesh.Trimesh(vertices=nodes,faces=boundary,process=False)
    sizes=[o['size_mm'] for o in subject['observations']];contacts=[]
    c01=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/lateral_array_design_20260916/plan.json'
    source_plan=read(c01);z_contact=float(source_plan['contact_ohm_m2'])
    if source_plan['diameter_mm']!=5:raise ValueError('Неожиданный контактный сценарий C01')
    if alias=='nik':
        anchor=np.asarray(source_plan['centre_mm'],float)
        centre,distance,face=fem_surface.nearest.on_surface(anchor[None,:])
        centre=centre[0];normal=fem_surface.face_normals[face[0]]
        axis=np.array([0.,0.,1.])-normal[2]*normal;axis/=np.linalg.norm(axis)
        side=np.cross(normal,axis);side/=np.linalg.norm(side)
        direction=np.cos(np.deg2rad(14))*axis+np.sin(np.deg2rad(14))*side
        placement=dict(kind='C01_anchor_reprojected_to_current_own_CT_mesh',anchor_lps_mm=anchor.tolist(),
            centre_lps_mm=centre.tolist(),anchor_projection_mm=float(distance[0]),angle_deg=14.,
            direction_unit=direction.tolist(),normal_unit=normal.tolist(),
            nominal_distance_definition='tangent_offsets_then_triangle_projection',
            original_C01_basis_and_contact_coordinates_reused=False,
            reason='Historical complete C01 geometry arrays are unavailable; known C01 centre and 14-degree angle define an explicit reconstructed scenario on the current own CT.',
            measured_installation=False)
        for size in sizes:
            raw=centre+np.array([-size/2,-size/4,size/4,size/2])[:,None]*direction
            projected,dist,_=fem_surface.nearest.on_surface(raw)
            contacts.append(dict(size_mm=size,centres_lps_mm=projected.tolist(),
                tangent_targets_lps_mm=raw.tolist(),tangent_to_skin_projection_mm=dist.tolist()))
    else:
        old_path=model/'cem_contacts_input/cem_contacts_input.local.json';old=read(old_path)
        placement_file=native.parent/'contact_placement.local.json';rules=read(placement_file)['rules']
        skin_rules=[r for r in rules if r['kind']=='skin_arc']
        anchors={(r['y_mm'],r['z_mm']) for r in skin_rules}
        if len(anchors)!=1:raise ValueError('У прежних сборок Георгия нет общего центра')
        y,z=next(iter(anchors));centre=placement_kernel.skin_arc(surface,y,z,[0])[0]
        maximum_previous_misfit=0.
        catalog={p['id']:p for p in old['contact_catalog']}
        for rule in skin_rules:
            recreated=placement_kernel.skin_arc(surface,y,z,rule['offsets_mm'])
            expected=np.array([catalog[k]['raw_lps_mm'] for k in rule['ids']])
            maximum_previous_misfit=max(maximum_previous_misfit,float(np.max(np.linalg.norm(recreated-expected,axis=1))))
        if maximum_previous_misfit>.001:raise ValueError('Не воспроизведены координаты прежнего собственного сценария')
        placement=dict(kind='previous_own_CT_skin_path_as_unmeasured_BMSTU_scenario',centre_lps_mm=centre.tolist(),
            parent_contact_input=str(old_path),parent_contact_sha256=sha(old_path),
            original_placement_file=str(placement_file),original_placement_sha256=sha(placement_file),
            nominal_distance_definition='skin_arc',old_contact_reproduction_maximum_error_mm=maximum_previous_misfit,
            y_mm=y,z_mm=z,measured_installation=False,
            BMSTU_Scorpion_installations_known_equal=False)
        for size in sizes:
            raw=placement_kernel.skin_arc(surface,y,z,[-size/2,-size/4,size/4,size/2])
            projected,dist,_=fem_surface.nearest.on_surface(raw)
            if np.max(dist)>1:raise ValueError('Проекция собственного пути превысила 1 мм')
            contacts.append(dict(size_mm=size,centres_lps_mm=projected.tolist(),
                native_path_targets_lps_mm=raw.tolist(),native_to_FEM_projection_mm=dist.tolist()))
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    centres=np.unique(np.vstack([c['centres_lps_mm'] for c in contacts]),axis=0)
    before=refinement.volumes(nodes,tets)
    if np.any(before<=0):raise ValueError('Исходные тетраэдры неположительны')
    original_volume=float(before.sum());history=[]
    if target_edge_mm:
        for level in range(refinement_levels):
            if not len(refinement.selected_edges(nodes,boundary,centres,5.,target_edge_mm)):break
            selected=select_parents(nodes,tets,centres,5.)
            step=out/f'refinement_{level+1}';step.mkdir()
            refined,quality,_,attempts=refine_screened(nodes,tets,boundary,selected,step,1000000,
                promotion_rounds=2,max_promoted=10000)
            nodes,tets,boundary=refined[:3]
            facts=dict(level=level+1,algorithm='canonical_screened_red_green',nodes=len(nodes),
                tetrahedra=len(tets),quality=quality,attempts=len(attempts),target_edge_mm=target_edge_mm)
            history.append(facts);print(alias,'сгущение',level+1,len(nodes),len(tets),flush=True)
    volumes=refinement.volumes(nodes,tets)
    if np.any(volumes<=0) or not np.array_equal(nodes[:len(old_nodes)],old_nodes):raise ValueError('Изменилась геометрия или ориентация элементов')
    if abs(volumes.sum()/original_volume-1)>1e-10:raise ValueError('Изменился объём тела')
    print(alias,'проверка полной границы и качества',flush=True)
    boundary,qc=core.mesh_qc(nodes,tets,boundary)
    meshfile=out/'mesh.mat';savemat(meshfile,dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    plan=dict(schema='bmstu_uniform_rho_geometry_v1',subject_alias=alias,observations=subject['observations'],
        observations_plan=str(Path(observations).resolve()),observations_plan_sha256=sha(observations),
        parent_mesh=str(source_mesh),parent_mesh_sha256=sha(source_mesh),native_body=str(native),native_body_sha256=sha(native),
        mesh_path=str(meshfile),mesh_sha256=sha(meshfile),mesh_QC=qc,refinement_history=history,
        refinement_level_limit=refinement_levels,refinement_selection_edge_target_mm=target_edge_mm,
        refinement_edge_target_achieved=not bool(target_edge_mm and len(refinement.selected_edges(nodes,boundary,centres,5.,target_edge_mm))),
        volume_before_mm3=original_volume,volume_after_mm3=float(volumes.sum()),original_vertices_unchanged=True,
        parent_configuration_path=str(model/'configuration.local.json'),parent_configuration_sha256=sha(model/'configuration.local.json'),
        placement=placement,contacts=contacts,nominal_diameter_mm=5.,nominal_area_mm2=float(np.pi*2.5**2),
        z_contact_ohm_m2=z_contact,contact_parameter_source=str(c01),contact_parameter_source_sha256=sha(c01),
        four_lateral_contacts_only=True,TTRKG_channel_excitation_or_passive_pads_modelled=False,
        own_source_response_is_assumption=True,source_multiplexing_known=False,
        model_reference_frequency_hz=50000,acquisition_frequency_verified=False,
        contact_area_relative_error_max=.10,contact_centroid_error_mm_max=.75,contact_extent_mm_max=3.5,
        system_relative_residual_max=1e-7,reciprocity_ohm_max=1e-7,
        global_mesh_convergence_proven=False,status='exploratory_hypothesis_not_validated',producer_sha256=sha(__file__))
    write_new(out/'geometry.local.json',plan)
    (out/'geometry_worker_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(alias,'геометрия подготовлена',len(nodes),len(tets),flush=True)
    return out/'geometry.local.json'


def run(geometry,out,matlab):
    """Единая точка MATLAB; настройки и планы нового запуска сохраняются отдельно."""
    geometry=Path(geometry).resolve();out=Path(out).resolve();d=read(geometry)
    prepared=Path(d['parent_mesh']).parents[1]/'electrical/prepared_materials.mat'
    if not prepared.is_file():raise FileNotFoundError('Нет собственного родительского оператора EIDORS')
    targets=[out,out.parent/(out.name+'.log'),out.parent/(out.name+'_input.local.json'),
        out.parent/(out.name+'.preferences'),out.parent/(out.name+'.temp')]
    if any(p.exists() for p in targets):raise FileExistsError('Все выходы запуска должны быть новыми')
    out.parent.mkdir(parents=True,exist_ok=True)
    d.update(geometry_source_path=str(geometry),geometry_source_sha256=sha(geometry),
        parent_prepared_materials_path=str(prepared),parent_prepared_materials_sha256=sha(prepared))
    inputfile=targets[2];write_new(inputfile,d);targets[3].mkdir();targets[4].mkdir()
    env=os.environ.copy();env.update(MATLAB_PREFDIR=str(targets[3]),TEMP=str(targets[4]),TMP=str(targets[4]))
    quote=lambda p:str(p).replace("'","''")
    script=f"cd('{quote(ROOT/'MATLAB_TRKG4_real_subjects')}');run_pipeline('Profile','bmstu_apparent_resistivity','InputFile','{quote(inputfile)}','OutputDir','{quote(out)}');"
    with targets[1].open('x',encoding='utf-8') as f:
        process=subprocess.run([str(matlab),'-batch',script],env=env,stdout=f,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode:raise RuntimeError(f'MATLAB завершился с кодом {process.returncode}; журнал: {targets[1]}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    p1=s.add_parser('data');p1.add_argument('--data-root',type=Path,required=True)
    p1.add_argument('--derived-root',type=Path,required=True);p1.add_argument('--model-root',type=Path,required=True);p1.add_argument('--out',type=Path,required=True)
    p1=s.add_parser('geometry');p1.add_argument('--observations',type=Path,required=True);p1.add_argument('--subject',choices=ALIASES,required=True)
    p1.add_argument('--out',type=Path,required=True);p1.add_argument('--target-edge-mm',type=float,default=0)
    p1.add_argument('--refinement-levels',type=int,default=2)
    p1=s.add_parser('run');p1.add_argument('--geometry',type=Path,required=True)
    p1.add_argument('--out',type=Path,required=True);p1.add_argument('--matlab',type=Path,required=True)
    a=p.parse_args()
    if a.command=='data':print(collect(a.data_root,a.derived_root,a.model_root,a.out),flush=True)
    elif a.command=='geometry':print(make_geometry(a.observations,a.subject,a.out,target_edge_mm=a.target_edge_mm,refinement_levels=a.refinement_levels),flush=True)
    else:run(a.geometry,a.out,a.matlab)
