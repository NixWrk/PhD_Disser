"""Build a new quality-screened candidate; never promotes a shared model."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pymeshlab
import trimesh
from scipy.io import loadmat, savemat

from georg_ct_model import PIPE, sampled_distance, repair_surface
from scorpion_georg_fem_prepare import read, write, sha
from georg_mesh_quality import (quality, orient, boundary_faces, refine_eight,
                                refine_local, volumes, QUALITY_LIMITS)

SOURCE = PIPE / 'output/exploratory/georg_ct_model_20261011'
OUT = PIPE / 'output/exploratory/georg_quality_rebuild_20261013/mesh_v1'


def selection(mesh, contacts, fine):
    points = mesh.triangles_center
    if fine:
        lateral = (points[:,0] < -140) & (points[:,1] > -75) & (points[:,1] < 40) & (points[:,2] > -325) & (points[:,2] < -95)
        radius = 10.
    else:
        lateral = (points[:,0] < -120) & (points[:,1] > -100) & (points[:,1] < 70) & (points[:,2] > -340) & (points[:,2] < -75)
        radius = 16.
    for centre in contacts:
        lateral |= np.linalg.norm(points-centre,axis=1) < radius
    return lateral


def surface(out, source=SOURCE, scheme='creases'):
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(global_edge_mm=5., buffer_edge_mm=2.8, contact_edge_mm=1.4,
                    local_operation_surface_tolerance_mm=.2, sampled_surface_tolerance_mm=.75,
                    relative_volume_tolerance=.003, quality_limits=QUALITY_LIMITS,
                    minimum_dihedral_deg=8., radius_edge_ratio=1.5,
                    maximum_coarse_tetrahedra=600000, maximum_fine_nodes=1000000,
                    settings_declared_before_meshing=True, scheme=scheme,
                    smooth_vertices=False,feature_angle_deg=170.)
    file = out/'settings.local.json'
    if file.exists() and read(file)!=settings:
        raise ValueError('Use a new candidate label for different mesh settings')
    write(file, settings)
    original = trimesh.load(source/'body_embedded_arms_370mm.stl', force='mesh')
    if (source/'fem_contacts.local.json').exists():
        descriptions = read(source/'fem_contacts.local.json')
        xyz = np.vstack([np.vstack(s['groups_mm']) for s in descriptions['scenarios'] if s['id'] in ['TTRKG_1x4','TTRKG_2x2']])
    else:
        descriptions=read(source/'contacts.local.json')
        xyz=np.vstack([np.vstack(descriptions[k]) for k in ['ttrkg_1x4','ttrkg_2x2']])
    contacts = np.unique(np.round(xyz,8),axis=0)
    ms = pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(original.vertices,original.faces.astype(np.int32)))
    def remesh(target, selected=False):
        print('Surface remeshing',target,'mm, selected',selected,flush=True)
        ms.meshing_isotropic_explicit_remeshing(iterations=6, adaptive=False,
            targetlen=pymeshlab.PureValue(target), selectedonly=selected,
            featuredeg=settings['feature_angle_deg'], checksurfdist=True, maxsurfdist=pymeshlab.PureValue(settings['local_operation_surface_tolerance_mm']),
            splitflag=True,collapseflag=True,swapflag=True,smoothflag=settings['smooth_vertices'],reprojectflag=True)
    remesh(settings['global_edge_mm'])
    for fine, target in [(False,settings['buffer_edge_mm']),(True,settings['contact_edge_mm'])]:
        m=ms.current_mesh();current=trimesh.Trimesh(m.vertex_matrix(),m.face_matrix(),process=False)
        selected=selection(current,contacts,fine)
        ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(current.vertices,current.faces.astype(np.int32),f_scalar_array=selected.astype(float)))
        ms.compute_selection_by_condition_per_face(condselect='fq > 0.5')
        remesh(target,True)
    m=ms.current_mesh();candidate=trimesh.Trimesh(m.vertex_matrix(),m.face_matrix(),process=True)
    candidate.fix_normals();candidate.export(out/'surface_before_repair.stl')
    candidate=trimesh.load(out/'surface_before_repair.stl',force='mesh')
    write(out/'source_identity.local.json',dict(producer_sha256=sha(__file__),
        source_surface_sha256=sha(source/'body_embedded_arms_370mm.stl')))
    candidate,repair=repair_surface(candidate)
    candidate.export(out/'body_meshing.stl')
    candidate=trimesh.load(out/'body_meshing.stl',force='mesh')
    distances={'original_to_candidate':sampled_distance(original,candidate),
               'candidate_to_original':sampled_distance(candidate,original)}
    displacement=np.linalg.norm(candidate.nearest.on_surface(contacts)[0]-contacts,axis=1)
    volume_error=abs(candidate.volume/original.volume-1)
    passed=candidate.is_volume and candidate.is_watertight and len(candidate.split())==1 and candidate.euler_number==original.euler_number
    passed &= max(x['max_mm'] for x in distances.values())<=settings['sampled_surface_tolerance_mm']
    passed &= volume_error<=settings['relative_volume_tolerance'] and displacement.max()<1
    result=dict(source_surface_sha256=sha(source/'body_embedded_arms_370mm.stl'),
                candidate_surface_sha256=sha(out/'body_meshing.stl'),sampled_distances=distances,
                relative_volume_error=volume_error,contact_projection_displacement_mm=displacement.tolist(),
                repair=repair,faces=len(candidate.faces),vertices=len(candidate.vertices),
                watertight=candidate.is_watertight,passed=bool(passed),surface_euler=candidate.euler_number,
                source_directory=str(source.resolve()),
                independent_anatomical_validation=False,sampled_distance_is_not_certified_global_bound=True)
    write(out/'surface_qc.local.json',result);print(json.dumps(result),flush=True)
    if not passed:raise ValueError('Candidate remeshing did not preserve the checked exterior')


def coarse(out,size=8.):
    if not read(out/'surface_qc.local.json')['passed']:raise ValueError('Surface QC failed')
    sys.path.insert(0,str(SOURCE/'python_mesh_deps'));import tetgen
    mesh=trimesh.load(out/'body_meshing.stl',force='mesh');engine=tetgen.TetGen(mesh.vertices,mesh.faces.astype(np.int32))
    start=time.monotonic()
    print('TetGen quality candidate',flush=True)
    nodes,tets,*_=engine.tetrahedralize(plc=True,quality=True,minratio=1.5,mindihedral=8.,
        fixedvolume=True,maxvolume=size**3/6,nobisect=False,steinerleft=500000,verbose=1,
        opt_scheme=1,smooth_maxiter=0,nomergefacet=True,epsilon=1e-10)
    tets=orient(nodes,tets);boundary=boundary_faces(tets)
    qc=quality(nodes,tets);qc['relative_volume_error']=abs(qc['volume_ml']*1000/mesh.volume-1)
    qc['passed'] &= qc['relative_volume_error']<1e-8
    settings=read(out/'settings.local.json')
    if 'maximum_coarse_nodes' in settings:
        qc['within_predeclared_size_limit']=len(nodes)<=settings['maximum_coarse_nodes']
    else:
        qc['within_predeclared_size_limit']=len(tets)<=settings['maximum_coarse_tetrahedra']
    qc['passed'] &= qc['within_predeclared_size_limit']
    qc.update(tetgen_version=tetgen.__version__,seconds=time.monotonic()-start,
              maximum_volume_mm3=size**3/6,source_surface_sha256=sha(out/'body_meshing.stl'))
    target=out/'coarse';target.mkdir(exist_ok=True)
    savemat(target/'mesh.mat',dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    qc['passed']=bool(qc['passed']);qc['mesh_sha256']=sha(target/'mesh.mat');write(target/'mesh_quality.local.json',qc)
    print(json.dumps(qc),flush=True)
    if not qc['passed']:raise ValueError('TetGen candidate failed predeclared shape/resource QC')


def fine(out, local=False):
    if not read(out/'coarse/mesh_quality.local.json')['passed']:raise ValueError('Parent shape QC failed')
    m=loadmat(out/'coarse/mesh.mat');nodes=m['nodes_mm'];tets=m['elems'].astype(np.int32)-1;boundary=m['boundary'].astype(np.int32)-1
    centres_parent=np.empty(0,np.int32);centre_weights=np.empty((0,4))
    selection_record={}
    if local:
        surface=trimesh.Trimesh(nodes,boundary,process=False)
        root=PIPE/'output/exploratory/georg_quality_rebuild_20261013'
        source=read(out/'surface_qc.local.json')['source_directory']
        contact=read(Path(source)/'contacts.local.json')
        arms=np.unique(np.round(np.vstack([np.vstack(contact[k]) for k in ['ttrkg_1x4','ttrkg_2x2']]),8),axis=0)
        centres=[np.array([-204.346,-20.215,-207.5]),np.array([-207.861328125,-.402093132,-205.339110431])]
        guides=[]
        for c in centres:
            c,_,face=surface.nearest.on_surface(c[None,:]);c=c[0];normal=surface.face_normals[face[0]]
            u=np.array([0.,0.,1.]);u-=u.dot(normal)*normal;u/=np.linalg.norm(u);v=np.cross(normal,u)
            for direction,sizes in [(u,[60,80,100,120,140]),(v,[60,140])]:
                for size in sizes:guides.extend(c+np.array([-size/2,-size/4,size/4,size/2])[:,None]*direction)
        guides=surface.nearest.on_surface(np.vstack([guides,arms]))[0]
        radius=12.;selected=np.zeros(len(tets),bool)
        for start in range(0,len(tets),100000):
            p=nodes[tets[start:start+100000]];lo=p.min(1);hi=p.max(1);hit=np.zeros(len(p),bool)
            for c in guides:
                d=np.maximum(np.maximum(lo-c,c-hi),0);hit|=np.einsum('ij,ij->i',d,d)<=radius**2
            selected[start:start+len(p)]=hit
        selection_record=dict(method='tetrahedron bounding boxes within declared radius of electrode guides',
                              radius_mm=radius,selected_parent_tetrahedra=int(selected.sum()),
                              guide_centres_lps_mm=guides.tolist(),guides_are_not_measured_positions=True,
                              local_check_is_not_global_uniform_refinement=True)
        print('Local nested parents',int(selected.sum()),'/',len(tets),flush=True)
        fn,ft,fb,parent,face_parent,edges,centres_parent,centre_weights=refine_local(nodes,tets,boundary,selected)
    else:
        fn,ft,fb,parent,face_parent,edges=refine_eight(nodes,tets,boundary)
    qc=quality(fn,ft);qc['relative_volume_difference']=abs(qc['volume_ml']/read(out/'coarse/mesh_quality.local.json')['volume_ml']-1)
    qc['original_nodes_unchanged']=bool(np.array_equal(fn[:len(nodes)],nodes))
    actual=boundary_faces(ft)
    qc['conforming_boundary_verified']=np.array_equal(np.unique(np.sort(actual,axis=1),axis=0),np.unique(np.sort(fb,axis=1),axis=0))
    qc['passed'] &= qc['relative_volume_difference']<1e-10 and qc['original_nodes_unchanged'] and qc['conforming_boundary_verified']
    qc['within_predeclared_size_limit']=len(fn)<=read(out/'settings.local.json')['maximum_fine_nodes']
    qc['passed'] &= qc['within_predeclared_size_limit']
    target=out/'fine';target.mkdir(exist_ok=True)
    savemat(target/'mesh.mat',dict(nodes_mm=fn,elems=ft+1,boundary=fb+1),do_compression=False)
    savemat(target/'parent_maps.mat',dict(element_parent=parent+1,boundary_parent=face_parent+1,
        midpoint_edges=edges+1,parent_nodes=len(nodes),center_parent_ids=centres_parent+1,
        center_barycentric_weights=centre_weights,
        boundary_child_counts=np.bincount(face_parent,minlength=len(boundary))),do_compression=False)
    qc.update(mesh_sha256=sha(target/'mesh.mat'),parent_maps_sha256=sha(target/'parent_maps.mat'),
              source_mesh_sha256=sha(out/'coarse/mesh.mat'),refinement='nested 1-to-8 with quality-selected octahedron diagonal')
    qc['local_selection']=selection_record;qc['passed']=bool(qc['passed'])
    qc['refinement']='nested red8 and incenter star-green transitions' if local else qc['refinement']
    write(target/'mesh_quality.local.json',qc);print(json.dumps({k:v for k,v in qc.items() if k!='local_selection'}),flush=True)
    if not qc['passed']:raise ValueError('Nested refinement failed predeclared shape/resource QC')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--scheme',default='creases')
    p.add_argument('--local',action='store_true')
    p.add_argument('--stage',choices=['surface','coarse','fine'],required=True);p.add_argument('--size',type=float,default=8.)
    a=p.parse_args()
    if a.stage=='surface':surface(a.out,source=a.source,scheme=a.scheme)
    elif a.stage=='coarse':coarse(a.out,a.size)
    else:fine(a.out,local=a.local)
