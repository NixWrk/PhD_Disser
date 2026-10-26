"""Geometrical skin-to-lung rays on the selected own manual-STL montage."""
from pathlib import Path
import argparse
import sys
import numpy as np
import nibabel as nib
import trimesh

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'Colab Notebooks/scorpion_v2'))
import ct_fem_core as core
from session_ct_geometry import interval_material_lengths
from scorpion_session_prepare import skin_curve


def calculate(model):
    model=Path(model).resolve();cfg=core.load_config(model/'configuration.local.json')
    plan=core.read(model/'placement_plan.local.json')
    mesh=trimesh.load(cfg['body']['source_stl'],force='mesh')
    center=plan['lateral_center_lps_mm'];trace,arc=skin_curve(mesh,center[1],center[2])
    images={k:nib.as_closest_canonical(nib.load(v)) for k,v in cfg['inputs']['masks'].items()}
    reference=images['lung'];affine=reference.affine
    if any(i.shape!=reference.shape or not np.array_equal(i.affine,affine) for i in images.values()):
        raise ValueError('Own mask grids differ')
    if not np.allclose(affine[:3,:3],np.diag(np.diag(affine[:3,:3])),atol=1e-5):raise ValueError('Axis-aligned CT required')
    labels=np.zeros(reference.shape,np.uint8)
    for name,index in [('fat',4),('heart',3),('bone',2),('lung',1)]:labels[np.asarray(images[name].dataobj)>0]=index
    x,y,z=[affine[a,3]+np.arange(reference.shape[a])*affine[a,a] for a in range(3)]
    dx=float(affine[0,0]);xmid=-float(mesh.bounds[:,0].mean());depth=[];parts=[]
    for point in trace:
        j=int(np.argmin(abs(y+point[1])));k=int(np.argmin(abs(z-point[2])))
        lung=np.flatnonzero((labels[:,j,k]==1)&(x>xmid))
        if not len(lung):depth.append(None);parts.append(None);continue
        inner=float(x[lung[-1]]+.51*dx);outer=-float(point[0])
        if outer<=inner:raise ValueError('Manual skin intersects the first-lung ray boundary')
        d=interval_material_lengths(x,labels[:,j,k],inner,outer,dx)
        depth.append(outer-inner);parts.append(d.tolist())
    summaries=[]
    for length in (60,100,140):
        selected=abs(arc)<=length/2
        if any(depth[i] is None for i in np.flatnonzero(selected)):raise ValueError('Lateral montage leaves the lung-supported region')
        values=np.array([depth[i] for i in np.flatnonzero(selected)])
        summaries.append(dict(size_mm=length,ray_count=len(values),minimum_depth_mm=float(values.min()),
            median_depth_mm=float(np.median(values)),maximum_depth_mm=float(values.max())))
    sources=([dict(path=cfg['body']['source_stl'],sha256=cfg['body']['source_stl_sha256'])]+
        [dict(path=p,sha256=core.sha(p)) for p in cfg['inputs']['masks'].values()])
    core.write_new(model/'ct_profile.local.json',dict(subject_alias=cfg['subject_alias'],sources=sources,
        producer_sha256=core.sha(__file__),arc_offset_mm=arc.tolist(),skin_to_first_lung_mm=depth,
        material_lengths_mm=parts,summaries=summaries,selected_delta_y_mm=-15,
        frame='LPS_mm',ray_direction='positive_X',lung_binary_boundary_level=.49,
        anatomical_depth_is_effective_h=False,electrical_sensitivity_calculated=False,
        manual_source_skin_is_not_voxelised_for_ray_start=True))
    print(cfg['subject_alias'],summaries,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True)
    calculate(p.parse_args().model)
