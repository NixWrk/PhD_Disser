"""Individual full-static CT material fractions and finite-contact geometry.

Three right-lateral poses are declared scenarios, not recovered installation.
The author-specified 370 mm cylindrical arms are included. Raw data stay local.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import gmsh
import nibabel as nib
import numpy as np
import pandas as pd
import pymeshlab
import trimesh
from scipy.io import savemat
from scipy.spatial import cKDTree
from scipy.stats import qmc

import refine_surface_contact_mesh as refine
refine.np=np

PIPE=Path(__file__).resolve().parents[1]
ROOT=PIPE.parent
OUT=PIPE/'output/exploratory/scorpion_georg_fem_20261008'
MATERIALS=['soft_background','lung','bone','heart','fat']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_mesh(path,source_path=None):
    # HXT's MSH2 export can include infinite-cell sentinels with node zero.
    # Never feed those to a solver. A filtered candidate is acceptable only
    # after its entire exterior is proved to match the source polyhedron.
    with path.open() as stream:
        for line in stream:
            if line.strip()=='$Nodes':break
        count=int(next(stream));rows=np.array([np.fromstring(next(stream),sep=' ') for _ in range(count)])
        tags=rows[:,0].astype(np.int64);order=np.argsort(tags)
        sorted_tags=tags[order];nodes=rows[order,1:]
        for line in stream:
            if line.strip()=='$Elements':break
        count=int(next(stream));tet=[];sentinels=[]
        for _ in range(count):
            row=np.fromstring(next(stream),sep=' ',dtype=np.int64)
            if row[1]!=4:continue
            vertices=row[3+row[2]:]
            if len(vertices)!=4:raise ValueError('Nonlinear or malformed volume elements')
            if np.any(vertices==0):sentinels.append(row.tolist())
            else:tet.append(vertices)
    tets=np.searchsorted(sorted_tags,np.array(tet))
    initial_volume=refine.volumes(nodes,tets)
    flat=np.abs(initial_volume)<1e-10
    removed_flat=[dict(nodes=t.tolist(),signed_volume_mm3=float(v))
                  for t,v in zip(tets[flat],initial_volume[flat])]
    # A coplanar boundary tetrahedron may encode a diagonal flip. Remove only
    # machine-zero volumes, then re-prove closure, source surface and volume.
    tets=tets[~flat]
    used=np.unique(tets);remap=np.full(len(nodes),-1,dtype=np.int64);remap[used]=np.arange(len(used))
    nodes=nodes[used];tets=remap[tets]
    faces=np.concatenate([tets[:,ix] for ix in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]]);faces.sort(axis=1)
    unique,counts=np.unique(faces,axis=0,return_counts=True)
    if not np.isin(counts,[1,2]).all():raise ValueError('Nonmanifold volume faces')
    boundary=unique[counts==1]
    source=trimesh.load(source_path or path.parent/'body_arm_surface_4mm.stl',force='mesh')
    query=trimesh.proximity.ProximityQuery(source)
    samples=np.vstack([nodes[np.unique(boundary)],nodes[boundary].mean(axis=1)])
    maximum=0.
    for i in range(0,len(samples),2048):
        _,distance,_=query.on_surface(samples[i:i+2048]);maximum=max(maximum,float(distance.max()))
    v=refine.volumes(nodes,tets)
    mismatch=abs(float(np.abs(v).sum())/source.volume-1)
    candidate=trimesh.Trimesh(nodes,boundary,process=False)
    boundary_tolerance=1.0 if removed_flat else 1e-3
    checker=pymeshlab.MeshSet()
    checker.add_mesh(pymeshlab.Mesh(nodes,boundary.astype(np.int32)))
    checker.compute_selection_by_self_intersections_per_face()
    intersections=checker.current_mesh().selected_face_number()
    if maximum>boundary_tolerance or mismatch>1e-6 or not candidate.is_watertight or intersections:
        write(path.with_name(path.stem+'_qc.local.json'),dict(accepted=False,node_zero_elements=sentinels,
            removed_machine_zero_tetrahedra=removed_flat,
            self_intersecting_boundary_faces=intersections,
            max_boundary_distance_mm=maximum,relative_volume_mismatch=mismatch,watertight=candidate.is_watertight))
        raise ValueError('Filtered HXT candidate does not reproduce the source boundary')
    write(path.with_name(path.stem+'_qc.local.json'),dict(accepted=True,node_zero_elements=sentinels,
        removed_machine_zero_tetrahedra=removed_flat,
        self_intersecting_boundary_faces=intersections,boundary_scenario_tolerance_mm=boundary_tolerance,
        removed_infinite_cell_sentinels=len(sentinels),unused_nodes_removed=len(tags)-len(nodes),
        max_boundary_distance_mm=maximum,relative_volume_mismatch=mismatch,watertight=True,
        rule='no node-zero or machine-zero cell solved; exterior, closure and volume checked against source',
        geometry_preserved_exactly=not removed_flat and maximum<1e-3))
    candidate.fix_normals()
    candidate.export(path.with_name(path.stem+'_checked_boundary.stl'))
    v=refine.volumes(nodes,tets)
    if np.any(abs(v)<1e-12):raise ValueError('Degenerate tetrahedra')
    reverse=v<0;tets[reverse]=tets[reverse][:,[1,0,2,3]]
    refine.verify_boundary(tets,boundary)
    return nodes,tets,boundary


def lateral_centres(surface,z):
    sizes=np.array([60.,80.,100.,120.,140.])
    points=[]
    for size in sizes:
        targets=np.array([[-600.,d,z] for d in [-size/2,-size/4,size/4,size/2]])
        locations,rays,_=surface.ray.intersects_location(targets,np.tile([1.,0.,0.],(4,1)))
        row=[]
        for j in range(4):
            hits=locations[rays==j]
            if not len(hits):raise ValueError('Lateral scenario misses body')
            row.append(hits[np.argmin(hits[:,0])])
        points.extend(row)
    return np.array(points)


def arm_centres(surface,arms):
    groups=[]
    for name in ('left','right'):
        a=arms['arm_extension'][name];direction=-1 if a['direction']=='negative_axis' else 1
        for distance,angles in [(120.,[0.]),(20.,[0.,np.pi/2,np.pi,3*np.pi/2])]:
            centres=np.array([[a['cut_coordinate']+direction*distance,
                              a['center_y']+a['radius']*np.cos(theta),
                              a['center_z']+a['radius']*np.sin(theta)] for theta in angles])
            projected,_,_=surface.nearest.on_surface(centres)
            if max(np.linalg.norm(projected-centres,axis=1))>1:
                raise ValueError('Arm centre projection exceeds 1 mm')
            groups.append(projected)
    # electrical order: current negative-X, potential negative-X,
    # potential positive-X, current positive-X.
    return [groups[0],groups[1],groups[3],groups[2]]


def map_fractions(nodes,tets,seg,out):
    ct=nib.load(seg/'static_ct.nii.gz');labels=np.zeros(ct.shape,dtype=np.uint8)
    overlap=[]
    for name,label in [('fat',4),('heart',3),('bone',2),('lung',1)]:
        image=nib.load(seg/(name+'.nii.gz'))
        if image.shape!=ct.shape:raise ValueError('Mask shape mismatch')
        np.testing.assert_allclose(image.affine,ct.affine,atol=1e-5,rtol=0)
        mask=np.asanyarray(image.dataobj)>0
        overlap.append(dict(tissue=name,voxels=int(mask.sum()),overlap_with_lower_priority=int(np.count_nonzero(mask&(labels>0)))))
        labels[mask]=label
    # DICOM/FEM LPS mm -> NIfTI RAS -> voxel coordinates.
    transform=np.linalg.inv(ct.affine)@np.diag([-1.,-1.,1.,1.])
    vox=nib.affines.apply_affine(transform,nodes)
    cut=np.sort(qmc.Sobol(3,scramble=True,seed=20261008).random_base2(10),axis=1)
    bary=np.diff(np.c_[np.zeros(1024),cut,np.ones(1024)],axis=1)
    fractions={q:np.zeros((len(tets),5),dtype=np.float64) for q in (256,1024)}
    outside=np.zeros(len(tets))
    for start in range(0,len(tets),1024):
        stop=min(start+1024,len(tets))
        xyz=np.einsum('qj,bjk->bqk',bary,vox[tets[start:stop]],optimize=True)
        index=np.floor(xyz+.5).astype(np.int32)
        valid=np.all((index>=0)&(index<np.asarray(labels.shape)),axis=2)
        sampled=np.zeros(valid.shape,dtype=np.uint8)
        ijk=index[valid];sampled[valid]=labels[ijk[:,0],ijk[:,1],ijk[:,2]]
        outside[start:stop]=1-valid.mean(axis=1)
        for q,array in fractions.items():
            for material in range(5):array[start:stop,material]=np.mean(sampled[:,:q]==material,axis=1)
        if start%102400==0:print('Material mapping',stop,'/',len(tets),flush=True)
    volume=refine.volumes(nodes,tets)
    inventory=[]
    for q,f in fractions.items():
        if not np.array_equal(f.sum(axis=1),np.ones(len(f))):raise ValueError('Fractions do not sum to one')
        savemat(out/f'fractions_q{q}.mat',dict(fractions=f),do_compression=False)
        for j,name in enumerate(MATERIALS):
            inventory.append(dict(quadrature_points=q,material=name,volume_ml=float(volume@f[:,j]/1000),
                                  nonzero_elements=int(np.count_nonzero(f[:,j]))))
    pd.DataFrame(inventory).to_csv(out/'material_inventory.csv',index=False)
    pd.DataFrame(overlap).to_csv(out/'mask_overlaps.csv',index=False)
    return dict(ras_to_lps=np.diag([-1.,-1.,1.,1.]).tolist(),
                full_ct_shape=list(ct.shape),ct_spacing_mm=list(map(float,ct.header.get_zooms())),
                outside_ct_fem_volume_ml=float(volume@outside/1000),
                outside_ct_material='soft background scenario for artificial arms and padded scan boundary',
                masks_priority=['lung','bone','heart','fat','soft background'])


def prepare(out=OUT):
    out=Path(out);seg=out/'segmentation'
    metadata=read(seg/'segmentation.json')
    if not metadata['complete']:raise ValueError('Segmentation is incomplete')
    for name,p in metadata['products'].items():
        if sha(seg/(name+'.nii.gz'))!=p['sha256']:raise ValueError('Mask changed: '+name)
    meshpath=out/'body_voxel2mm_hxt.msh'
    surface_path=out/'body_voxel2mm.stl'
    nodes,tets,boundary=load_mesh(meshpath,surface_path)
    surface=trimesh.Trimesh(nodes,boundary,process=False)
    arms=read(out/'arms_370mm.json')
    if any(abs(a['radius']-370/(2*np.pi))>1e-5 for a in arms['arm_extension'].values()):
        raise ValueError('Cylindrical arms do not match author measurement')
    poses=[dict(id=f'p{j}',z_mm=z,lateral_centres_mm=lateral_centres(surface,z))
           for j,z in enumerate([-180.,-210.,-240.])]
    arm=arm_centres(surface,arms)
    centres=np.unique(np.vstack([*(p['lateral_centres_mm'] for p in poses),*arm]),axis=0)
    # Preserve the polyhedral surface while resolving every finite contact.
    nodes,tets,boundary,_,_,history=refine.refine(nodes,tets,boundary,
        np.zeros(len(tets),dtype=np.uint8),np.zeros(len(tets),dtype=np.uint8),centres,
        radius=5.,target=1.,max_rounds=14)
    volume=refine.volumes(nodes,tets)
    reverse=volume<0;tets[reverse]=tets[reverse][:,[1,0,2,3]]
    volume=refine.volumes(nodes,tets)
    if np.any(volume<=1e-12):raise ValueError('Invalid refined volume elements')
    refine.verify_boundary(tets,boundary)
    for p in poses:
        p['lateral_centres_mm']=p['lateral_centres_mm'].tolist()
    savemat(out/'mesh.mat',dict(nodes_mm=nodes,elems=tets+1,boundary=boundary+1),do_compression=False)
    write(out/'contacts.local.json',dict(poses=poses,arm_groups_mm=[g.tolist() for g in arm],
         sizes_mm=[60,80,100,120,140],diameter_mm=5.,nominal_area_mm2=float(np.pi*2.5**2),
         electric_nodes=8,physical_contacts_per_montage=14,
         group_order=['arm I+','arm V+ (four contacts)','arm V- (four contacts)','arm I-',
                      'lateral A','lateral M','lateral N','lateral B'],
         simultaneous_contacts='TTRKG 1x4 and one lateral size; other lateral sizes absent',
         arm_rotation_rad=0.,arm_rotation_is_scenario=True,
         lateral_pose_recovered=False,
         lateral_geometry='common centre per pose; horizontal axis parallel to LPS Y; ray projection at fixed Y,Z',
         stimulation_scenario='one source at a time; other electrical groups passive, equipotential CEM nodes'))
    geo=map_fractions(nodes,tets,seg,out)
    registry_path=PIPE/'configs/tissue_literature_v2.json';registry=read(registry_path)
    profile=registry['screening_profiles']['100000']
    references={key:registry['tissues'][name]['gabriel_at_frequency']['100000']['rho_ohm_m']
                for key,name in [('soft','Muscle'),('lung_inflated','Lung (Inflated)'),
                                 ('lung_deflated','Lung (Deflated)'),('bone','Bone (Cortical)'),
                                 ('heart','Heart Muscle'),('fat','Fat')]}
    references.update(frequency_hz=100000,source='Gabriel frequency-specific entries in project registry',
                      registry_sha256=sha(registry_path),
                      whole_heart_as_heart_muscle_is_model_assumption=True,
                      whole_bone_as_cortical_bone_is_model_assumption=True)
    write(out/'reference_materials.local.json',references)
    # Reference-specific values come from the registry, not range midpoints.
    print('Profile compartments:',{k:v.get('reference_values_ohm_m',v.get('rho_bounds_ohm_m')) for k,v in profile['compartments'].items()},flush=True)
    experiment=ROOT/'Experiments/scorpion_v2_georg'
    pilot=experiment/'derived/exploratory/33.21_identifiability/results.local.json'
    observations=[]
    for c in read(pilot):
        for size,z in zip([60,80,100,120,140],c['fit']['observed_ohm']):
            observations.append(dict(channel=c['channel'],state=c['state'],L_mm=size,Z_ohm=z))
    pd.DataFrame(observations).to_csv(out/'observations.csv',index=False)
    files=[meshpath,surface_path,out/'mesh.mat',out/'contacts.local.json',registry_path,
           out/'reference_materials.local.json',out/'voxel_geometry_comparison.local.json',
           out/'fractions_q256.mat',out/'fractions_q1024.mat',out/'observations.csv',
           out/'arms_370mm.json',out/'body_arm_surface_4mm_qc.json',
           meshpath.with_name(meshpath.stem+'_qc.local.json'),seg/'conversion.json',seg/'segmentation.json',
           *(seg/(name+'.nii.gz') for name in ('heart','lung','bone','fat')),
           experiment/'metadata/anthropometry.local.json']
    plan=dict(status='exploratory_hypothesis_not_validated',subject_alias='georg',experiment='Scorpion v2',
              frequency_hz=100000,nodes=len(nodes),tetrahedra=len(tets),
              material_order=MATERIALS,geometry=geo,contact_diameter_mm=5.,
              outer_boundary_approximation='2 mm voxelization of author-circumference geometry; no additional closing',
              contact_impedance_ohm_m2=2.5e-7*.05/(np.pi*.01**2/4),
              contact_impedance_source='existing project calculation scenario, not measured Scorpion contact',
              resistivity_search_bounds_ohm_m=[.1,100.],literature_ranges_are_inverse_bounds=False,
              poses_are_installation_measurements=False,
              arm_circumference_mm=370.,measurement_level='current electrodes, both arms',
              circular_cylindrical_arms='author-confirmed modelling assumption',
              mesh_qc=dict(inverted_tetrahedra=0,degenerate_tetrahedra=0,boundary_matches_tetrahedra=True,
                           volume_ml=float(volume.sum()/1000),refinement_history=history),
              registry_id=registry['registry_id'],registry_sha256=sha(registry_path),
              sources=[dict(path=str(p.resolve()),sha256=sha(p)) for p in files],
              code_sha256=sha(__file__),global_spatial_convergence_verified=False,
              volume_integration_points=[256,1024],
              limitations=['full static inspiratory CT used for both electrical states',
                           'automatic masks not independently validated',
                           'three declared lateral poses; global localization not performed',
                           'real scalar conductivity and fixed contact impedance',
                           'artificial-arm regions outside CT have unknown tissue composition'])
    write(out/'plan.local.json',plan)
    print('Prepared',len(nodes),'nodes',len(tets),'tetrahedra',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    prepare(p.parse_args().out)
