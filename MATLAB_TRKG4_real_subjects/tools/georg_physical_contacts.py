"""Separate permanent physical arm patches from electrical connections."""
from pathlib import Path
import numpy as np
import trimesh
from scorpion_georg_fem_prepare import read, write, sha


def prepare(geometry, mesh_surface, out):
    geometry,mesh_surface,out=Path(geometry),Path(mesh_surface),Path(out)
    source=read(geometry/'contacts.local.json')
    groups4=[np.asarray(g,float).reshape(-1,3) for g in source['ttrkg_1x4']]
    groups2=[np.asarray(g,float).reshape(-1,3) for g in source['ttrkg_2x2']]
    raw=np.unique(np.round(np.vstack(groups4+groups2),8),axis=0)
    if len(raw)!=12:raise ValueError('Expected the twelve author-confirmed permanent arm patches')
    surface=trimesh.load(mesh_surface,force='mesh')
    centres,distance,_=surface.nearest.on_surface(raw)
    if np.max(distance)>=1:raise ValueError('Artificial-arm patch projection exceeds 1 mm')
    def ids(groups):
        answer=[]
        for g in groups:
            positions=np.argmin(np.linalg.norm(g[:,None,:]-raw[None,:,:],axis=2),axis=1)
            if np.max(np.linalg.norm(g-raw[positions],axis=1))>1e-7:raise ValueError('Unknown physical patch')
            answer.append(positions.tolist())
        return answer
    group_indices={'1x4':ids(groups4),'2x2':ids(groups2)}
    group_indices['1x1']=[group_indices['1x4'][0],group_indices['1x4'][1][1:2],
                          group_indices['1x4'][2][1:2],group_indices['1x4'][3]]
    rows=[]
    for name,groups in group_indices.items():
        used=[i for group in groups for i in group]
        if len(set(used))!=len(used):raise ValueError('A physical patch belongs to two active groups')
        rows.append(dict(id='TTRKG_'+name,family='TTRKG',montage=name,
                         groups_mm=[centres[i].tolist() for i in groups],drive=[1,4],measure=[2,3],
                         active_patch_ids=groups,unconnected_patch_ids=sorted(set(range(12))-set(used))))
    description=dict(schema='georg_constant_physical_arm_contacts_v1',
        diameter_mm=5.,nominal_area_mm2=float(np.pi*2.5**2),actual_contact_diameter_is_known=False,
        arm_physical_centres_lps_mm=centres.tolist(),previous_derived_centres_lps_mm=raw.tolist(),
        projection_displacement_mm=distance.tolist(),scenarios=rows,
        permanent_physical_patch_count=12,
        physical_patch_presence_origin='author confirmation: unchanged patches, switched connections',
        unconnected_patch_model='individual floating CEM electrode with zero net external current',
        unconnected_patch_model_is_assumption=True,
        cross_arms_are_individual_floating_contacts=True,
        cross_physical_presence_is_continuation_assumption=True,
        artificial_arm_geometry_is_measured=False,
        source_geometry_sha256=sha(geometry/'geometry.local.json'),surface_sha256=sha(mesh_surface),
        producer_sha256=sha(__file__))
    write(out,description)
    return description
