import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))
from heart_ct_overlays import candidate_membership, plane_grid, sample_ct, voxel_membership


def test_ct_linear_field_with_permuted_negative_anisotropic_axes():
    affine=np.array([[0,-2,0,12],[-3,0,0,15],[0,0,.5,-2],[0,0,0,1.]])
    ijk=np.stack(np.meshgrid(np.arange(7),np.arange(6),np.arange(8),indexing='ij'),axis=-1)
    world=ijk@affine[:3,:3].T+affine[:3,3]
    data=2*world[...,0]-world[...,1]+3*world[...,2]
    points=np.array([[8.2,6.4,-.7],[6.1,4.1,.3]])
    np.testing.assert_allclose(sample_ct(data,affine,points),2*points[:,0]-points[:,1]+3*points[:,2],atol=1e-12)
    assert np.isnan(sample_ct(data,affine,np.array([[1000.,0,0]]))).all()


def test_world_planes_have_explicit_anatomical_directions():
    center=np.array([10.,20,30]);bounds=np.array([[0,20],[10,30],[20,40.]])
    for horizontal,vertical,fixed in ((0,1,2),(0,2,1),(1,2,0)):
        world,u,v=plane_grid(center,bounds,horizontal,vertical,fixed,pixels=3)
        np.testing.assert_array_equal(world[...,fixed],np.full((3,3),center[fixed]))
        assert np.all(np.diff(u)<0) and np.all(np.diff(v)>0)
        assert world[0,0,horizontal]==bounds[horizontal,1]
        assert world[-1,0,vertical]==bounds[vertical,1]


def test_individual_uses_saved_world_transform_and_voxel_affine():
    mask=np.zeros((3,3,3),bool);mask[1,1,1]=True
    affine=np.diag([2.,3,4,1]);affine[:3,3]=[5,-7,10]
    transform=np.diag([1.2,1.2,1.2,1]);transform[:3,3]=[10,20,-5]
    p={'kind':'individual_isotropic_volume_centroid','reference_affine':affine,'transform_world':transform}
    combined=transform@affine
    point=combined[:3,:3]@np.ones(3)+combined[:3,3]
    points=np.array([point,point+combined[:3,0]*.49,point+combined[:3,0]*.51])
    np.testing.assert_array_equal(candidate_membership(p,points,mask),[True,True,False])
    np.testing.assert_array_equal(voxel_membership(mask,combined,points),[True,True,False])


def test_ellipsoid_axes_are_columns_in_physical_space():
    axes=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    p={'kind':'ellipsoid_volume_scaled','center_mm':[10,20,30],'axes_matrix':axes,'semi_axes_mm':[5,2,1]}
    center=np.array(p['center_mm'])
    points=np.array([center,center+4.9*axes[:,0],center+5.1*axes[:,0],center+2.1*axes[:,1]])
    np.testing.assert_array_equal(candidate_membership(p,points),[True,True,False,False])
