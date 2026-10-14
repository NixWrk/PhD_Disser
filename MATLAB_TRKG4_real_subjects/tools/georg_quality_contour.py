"""Bounded 3D topology regularization and a non-tangent artificial arm join."""
from pathlib import Path
import json
import shutil

import nibabel as nib
import numpy as np
import pymeshlab
import trimesh
from scipy import ndimage as ndi
from skimage.measure import euler_number

from georg_ct_model import PIPE, surface_from_mask, sampled_distance, repair_surface
from scorpion_georg_fem_prepare import read, write, sha
from scorpion_georg_geometry_revision import run as join_arms

SOURCE=PIPE/'output/exploratory/georg_ct_model_20261011'
OUT=PIPE/'output/exploratory/georg_quality_rebuild_20261013/geometry_v5'
SEG=PIPE/'output/exploratory/scorpion_georg_fem_20261008/segmentation'


def run(out=OUT):
    out.mkdir(parents=True,exist_ok=False)
    settings=dict(body_closing='one padded 6-neighbour 3D iteration',
                  maximum_relative_changed_body_voxels=.001,
                  candidate_to_native_sample_tolerance_mm=1.25,
                  arm_section_clearance_mm=2.,arm_circumference_mm=370.,
                  topology_is_a_geometric_regularization_scenario=True,
                  independent_anatomical_validation=False,
                  binary_surface_level=.49, avoids_exact_binary_saddle_level=True,
                  pre_union_decimation=False)
    write(out/'settings.local.json',settings)
    image=nib.load(SOURCE/'body_mask.nii.gz');original=np.asanyarray(image.dataobj)>0
    mask=ndi.binary_closing(np.pad(original,2),structure=ndi.generate_binary_structure(3,1))[2:-2,2:-2,2:-2]
    change=mask!=original;ratio=float(change.sum()/original.sum())
    topology={str(c):int(euler_number(mask,connectivity=c)) for c in (1,3)}
    if ratio>settings['maximum_relative_changed_body_voxels'] or set(topology.values())!={1}:
        raise ValueError('Bounded body topology regularization failed')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8),image.affine),out/'body_mask.nii.gz')
    raw=surface_from_mask(mask,image.affine,level=settings['binary_surface_level'])
    write(out/'raw_topology.local.json',dict(surface_euler=raw.euler_number,
          voxel_euler=topology,producer_sha256=sha(__file__),
          surface_helper_sha256=sha(PIPE/'tools/georg_ct_model.py')))
    if raw.euler_number!=2:raise ValueError('Marching-cubes exterior retains a handle')
    raw.export(out/'body_ct_native.stl')
    surface=trimesh.load(out/'body_ct_native.stl',force='mesh')
    if not surface.is_volume or surface.euler_number!=2:raise ValueError('Native binary-STL topology changed')
    ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(surface.vertices,surface.faces.astype(np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    if ms.current_mesh().selected_face_number():raise ValueError('Native CT surface has intersections')
    repair=dict(removed_intersecting_faces=0,remaining_intersecting_faces=0,no_pre_union_decimation=True)
    surface.export(out/'body_ct_surface.stl')
    distances=dict(native_to_surface=sampled_distance(raw,surface),surface_to_native=sampled_distance(surface,raw))
    if max(x['max_mm'] for x in distances.values())>settings['candidate_to_native_sample_tolerance_mm']:
        raise ValueError('Surface simplification failed unchanged CT-native tolerance')
    mask_checks={}
    for name in ('lung','bone','heart','fat'):
        tissue=np.asanyarray(nib.load(SEG/(name+'.nii.gz')).dataobj)>0
        mask_checks[name]=dict(outside_old_body_voxels=int(np.count_nonzero(tissue & ~original)),
                              outside_new_body_voxels=int(np.count_nonzero(tissue & ~mask)))
        if mask_checks[name]['outside_new_body_voxels']>mask_checks[name]['outside_old_body_voxels']:
            raise ValueError('Topology regularization removed covered tissue')
    # Author's admissible centre window and both axes; this is not a confidence interval.
    xyz=nib.affines.apply_affine(image.affine,np.argwhere(change))*[-1,-1,1]
    lateral=(xyz[:,0]<-130)&(xyz[:,1]>-120)&(xyz[:,1]<90)&(xyz[:,2]>-320)&(xyz[:,2]<-95)
    qc=dict(source_body_mask_sha256=sha(SOURCE/'body_mask.nii.gz'),ct_sha256=sha(SEG/'static_ct.nii.gz'),
            changed_voxels=int(change.sum()),added_voxels=int(np.count_nonzero(mask & ~original)),
            removed_voxels=int(np.count_nonzero(original & ~mask)),relative_changed_voxels=ratio,
            changed_voxel_volume_ml=float(change.sum()*np.prod(image.header.get_zooms())/1000),
            source_voxel_euler={str(c):int(euler_number(original,connectivity=c)) for c in (1,3)},
            candidate_voxel_euler=topology,candidate_surface_euler=surface.euler_number,
            changed_voxels_in_lateral_search_region=int(lateral.sum()),mask_coverage=mask_checks,
            simplification=distances,repair=repair,producer_sha256=sha(__file__),
            skin_is_automatically_reconstructed=True,independent_anatomical_validation=False,
            body_correction_is_a_bounded_numerical_regularization=True)
    write(out/'contour_qc.local.json',qc)
    draw_changes(image,original,mask,change,out)
    join_arms(out/'body_ct_surface.stl',out,clearance_mm=settings['arm_section_clearance_mm'])
    print(json.dumps(qc),flush=True)


def draw_changes(image,old,new,change,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ct=np.asanyarray(nib.load(SEG/'static_ct.nii.gz').dataobj)
    labels,num=ndi.label(change);counts=np.bincount(labels.ravel());counts[0]=0
    keys=np.argsort(counts)[-12:][::-1]
    fig,axes=plt.subplots(3,4,figsize=(15,10))
    for ax,key in zip(axes.flat,keys):
        indices=np.argwhere(labels==key);centre=np.median(indices,axis=0).astype(int)
        lo=np.maximum(indices.min(0)-10,0);hi=np.minimum(indices.max(0)+11,old.shape);k=centre[2]
        area=(slice(lo[0],hi[0]),slice(lo[1],hi[1]),k)
        ax.imshow(ct[area].T,origin='lower',cmap='gray',vmin=-600,vmax=300)
        ax.contour(old[area].T,levels=[.5],colors=['#3374bb'],linewidths=1)
        ax.contour(new[area].T,levels=[.5],colors=['#c6532c'],linewidths=1)
        z=nib.affines.apply_affine(image.affine,centre)[2]
        ax.set_title(f'z = {z:.2f} мм; изменено {counts[key]} вокселей',fontsize=9);ax.set_axis_off()
    fig.suptitle('КТ: исходный контур — синий; ограниченное 3D-замыкание — оранжевый',fontsize=13)
    fig.tight_layout();fig.savefig(out/'ct_topology_regularization.local.png',dpi=150);plt.close(fig)


if __name__=='__main__':run()
