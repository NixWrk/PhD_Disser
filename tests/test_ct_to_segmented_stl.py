"""Native anatomical STL export preserves components, cavities and coordinates."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from segment_ct_to_stl import export_mask
import ct_fem_core as core


def inputs(tmp_path):
    affine=np.array([[0.,-.8,0.,12.3],[1.1,0.,0.,-13.4],[0.,0.,1.3,-14.5],[0.,0.,0.,1.]])
    data=np.zeros((24,24,24),np.uint8)
    data[3:15,3:15,3:15]=1;data[17:21,17:21,17:21]=1;data[8:10,8:10,8:10]=0
    paths=[]
    for name,values in [('ct',np.zeros(data.shape,np.int16)),('mask',data)]:
        path=tmp_path/(name+'.nii.gz');image=nib.Nifti1Image(values,affine)
        image.header.set_xyzt_units('mm');nib.save(image,path);paths.append(path)
    return paths,affine


def test_all_components_and_internal_cavity_survive_STL_export(tmp_path):
    (ct,mask),affine=inputs(tmp_path);digest=core.sha(mask);target=tmp_path/'tissue.stl'
    qc=export_mask(mask,ct,target)
    assert qc['connected_surface_components']==3
    assert qc['all_components_and_cavity_interfaces_preserved']
    assert qc['watertight'] and qc['winding_consistent']
    assert qc['ordered_triangles_checked'] and qc['maximum_export_vertex_displacement_mm']<.001
    assert core.sha(mask)==digest
    mesh=trimesh.load(target,force='mesh')
    voxel_points=np.array([[5.,5.,5.],[18.,18.,18.],[8.5,8.5,8.5],[1.,1.,1.]])
    world=nib.affines.apply_affine(affine,voxel_points)*[-1.,-1.,1.]
    np.testing.assert_array_equal(mesh.contains(world),[True,True,False,False])


@pytest.mark.parametrize('problem',['affine','nonbinary'])
def test_incompatible_segmentation_is_not_silently_corrected(tmp_path,problem):
    (ct,mask),_=inputs(tmp_path);image=nib.load(mask)
    data=np.asanyarray(image.dataobj).copy();affine=image.affine.copy()
    if problem=='affine':affine[0,3]+=.1
    else:data[5,5,5]=2
    image=nib.Nifti1Image(data,affine);image.header.set_xyzt_units('mm');nib.save(image,mask)
    target=tmp_path/'rejected.stl'
    with pytest.raises(ValueError):export_mask(mask,ct,target)
    assert not target.exists()
