"""Prepare own manual-STL models through the common CT/FEM geometry stages."""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import ct_fem_core as core


def skin_curve(surface,y,z):
    heights=z+np.arange(-100,100.01,.5)
    starts=np.column_stack([np.full(len(heights),surface.bounds[0,0]-100),np.full(len(heights),y),heights])
    locations,rays,_=surface.ray.intersects_location(starts,np.tile([1.,0.,0.],(len(starts),1)))
    points={int(i):locations[rays==i][np.argmin(locations[rays==i,0])] for i in np.unique(rays)}
    origin=len(heights)//2
    if origin not in points:raise ValueError('Lateral centre ray misses own manual skin')
    first=last=origin
    while first-1 in points:first-=1
    while last+1 in points:last+=1
    trace=np.array([points[i] for i in range(first,last+1)])
    arc=np.r_[0.,np.linalg.norm(np.diff(trace,axis=0),axis=1).cumsum()];arc-=arc[origin-first]
    if arc[0]>-70 or arc[-1]<70:raise ValueError('All lateral sizes must fit one continuous own-skin path')
    return trace,arc


def create_config(subject,sources,preflight,placement,out):
    source=sources[subject];surface=trimesh.load(source['source_stl']['path'],force='mesh')
    core.surface_qc(surface)
    arms=copy.deepcopy(source['cylinders']);joins=[core.full_section_join(surface,a) for a in arms]
    cfg=core.read(preflight['subjects'][subject]['config']['path'])
    # Resolve the existing local input paths before moving the configuration.
    original=core.load_config(preflight['subjects'][subject]['config']['path'])
    cfg['inputs']=original['inputs'];cfg['subject_alias']=subject
    audit=core.read(out.parent/(subject+'_coverage_audit.local.json'))
    limits={k:(0 if k in ('lung','heart') else int(np.floor(v['total']*.05))) for k,v in audit.items()}
    cfg['body']=dict(strategy='author_STL_LPS',source_stl=source['source_stl']['path'],
        source_stl_sha256=source['source_stl']['sha256'],units='mm',frame='LPS',
        provenance='Own Inobitec outer solid selected by the author on 2026-10-26; not an accepted clinical model',
        maximum_outside_body_voxels=limits,
        coverage_limits_basis='New author-selected manual computational domain. Lung and heart must be fully covered. Up to 5% of candidate bone/fat voxels may remain outside the domain as an explicit screening scenario, not segmentation accuracy or FEM error; original masks are unchanged. Observed exclusion is reported separately.')
    cfg['synthetic_cylinders']=arms
    cfg['surface_qc']['manual_STL_export_maximum_sampled_distance_mm']=.001
    cfg['surface_qc']['sample_count']=15000
    physical={};area=float(np.pi*2.5**2)
    armgroups={}
    for side,(a,j) in enumerate(zip(arms,joins)):
        for role,distance in [('I',120.),('U',20.)]:
            ids=[]
            for location,sign in [('top',1),('bottom',-1)]:
                name=f'arm{side}_{role}_{location}'
                point=[a['cut_x_mm']+a['direction']*distance,j['centre_yz_mm'][0],j['centre_yz_mm'][1]+sign*a['radius_mm']]
                physical[name]=dict(raw_lps_mm=point,nominal_area_mm2=area,
                    provenance='Author top/bottom scheme; distances from own original CT cap proxy, not measured shoulder; rebuilt 370mm circumference')
                ids.append(name)
            armgroups[f'TT_{role}{side}']=ids
    base=placement['subjects'][subject]['scenarios']['-15']
    center=base['center_lps_mm']
    trace,arc=skin_curve(surface,center[1],center[2])
    for length in (60,100,140):
        offsets=np.array([-length/2,-length/4,length/4,length/2])
        points=np.column_stack([np.interp(offsets,arc,trace[:,i]) for i in range(3)])
        projected,dist,_=surface.nearest.on_surface(points)
        if np.max(dist)>.1:raise ValueError('Interpolated lateral path departed from manual skin')
        for name,point in zip(('A','M','N','B'),projected):
            physical[f'side{length}_{name}']=dict(raw_lps_mm=point.tolist(),nominal_area_mm2=area,
                provenance='Working -15mm Y scenario transferred onto own Inobitec skin; nominal size uses skin arc, not measured installation geometry')
    scenarios=[]
    for length,variant in [(60,'2x2'),(100,'2x2'),(140,'2x2'),(60,'1x1')]:
        groups={k:(v if variant=='2x2' else v[:1]) for k,v in armgroups.items()}
        groups.update({name:[f'side{length}_{name}'] for name in ('A','M','N','B')})
        present=[name for name in physical if name.startswith('arm') or name.startswith(f'side{length}_')]
        for role,drive,measure in [('ttrkg',['TT_I0','TT_I1'],['TT_U0','TT_U1']),('lateral',['A','B'],['M','N'])]:
            scenarios.append(dict(id=f'L{length}_TT{variant}_{role}',present_contact_ids=present,groups=groups,
                drive_groups=drive,measure_groups=measure,inactive_contact_model='individual_floating_CEM_assumption'))
    cfg['contacts']=dict(maximum_projection_mm=1.,physical_centres_lps_mm=physical,scenarios=scenarios)
    cfg['cem_contacts']['physical_parameters']=[dict(id=name,z_contact_ohm_m2=.01,
        z_contact_source='Fixed finite-CEM working scenario; not independently measured') for name in physical]
    cfg['remeshing']['selection']['physical_contact_ids']=list(physical)
    cfg['remeshing']['selection']['guides_lps_mm']=[]
    cfg['remeshing']['global_edge_mm']=5.
    cfg['remeshing']['buffer_edge_mm']=2.0
    cfg['remeshing']['contact_edge_mm']=.8
    cfg['mesh']['size_mm']=8.
    cfg['electrical_context']=dict(frequency_hz=100000,current_per_source_A=.007,
        frequency_provenance='Author Scorpion session confirmation; binary settings consistent',
        source_rule='own_drive',source_relative_phase='unknown',contact_impedance_is_measured=False,
        absent_bottom_pads_for_1x1=False,bottom_pads_presence_is_working_assumption=True)
    cfg['research_status']='exploratory_hypothesis_not_validated'
    cfg['manual_domain_audit']=audit
    out.mkdir(parents=True,exist_ok=False)
    path=out/'configuration.local.json';core.write_new(path,cfg)
    core.write_new(out/'placement_plan.local.json',dict(subject=subject,circumference_mm=370.,
        radius_mm=float(370/(2*np.pi)),joins=joins,lateral_center_lps_mm=trace[np.argmin(abs(arc))].tolist(),
        nominal_distance_definition='skin_arc',delta_Y_scenario_mm=-15,
        placement_is_measured=False,arm_distance_origin='own CT crop cap, shoulder proxy',
        segmentation='manual outer STL; own automatic lung/bone/whole-heart/fat masks',
        old_subject_matrices_used=False,geometry_accepted_for_FEM=False))
    return path


def run(args):
    out=Path(args.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    sources=core.read(args.sources);preflight=core.read(args.preflight);placement=core.read(args.placement)
    if not (out/'configuration.local.json').exists():
        config=create_config(args.subject,sources,preflight,placement,out)
    else:
        config=out/'configuration.local.json'
    # The standard mesh stage accepts an unmodified surface parent. Avoid
    # lossy STL conditioning; TetGen may subdivide the original PLC boundary.
    stages=['surface','mesh','materials','cem_contacts_input']
    parent=None
    for stage in stages:
        target=out/stage
        if (target/'preparation.local.json').exists():
            core.verify_artifact(target);parent=target;continue
        core.write_new(out/(stage+'_started.local.json'),dict(stage=stage,started_unix_s=time.time(),config_sha256=core.sha(config),FEM_was_run=False))
        print(args.subject,stage,'started',flush=True)
        manifest=core.prepare(stage,config,target,parent)
        print(args.subject,stage,manifest['status'],flush=True)
        parent=target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','preflight','placement','out','subject'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    runtime=Path(args.out).resolve().parent/'runtime'
    if runtime.is_dir():sys.path.insert(0,str(runtime))
    run(args)
