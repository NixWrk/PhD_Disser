import numpy as np
import trimesh
from lung_depth_profile import profile


def phantom():
    surface=trimesh.creation.box(extents=[40,20,20])
    # Voxel i has LPS x=i-25; lung starts at voxel x=-10, boundary x=-10.5.
    affine=np.diag([-1.,-1.,1.,1.]);affine[:3,3]=[25,15,-15]
    lung=np.zeros((51,31,31),np.uint8);lung[15:25,5:26,5:26]=1
    return surface,lung,affine


def test_directional_depth_and_half_space():
    surface,lung,affine=phantom()
    p=profile(surface,lung,affine,0,0,length_mm=10,ray_step_mm=.25)
    # Nearest-neighbour tie convention can shift the boundary by one step.
    assert np.all(np.abs(np.array(p['depth_mm'])-9.5)<=.25+1e-8)
    assert set(p['flags'])=={'sampled_lung_boundary'}
    lung[:25]=0;lung[30:40]=1
    assert all(v is None for v in profile(surface,lung,affine,0,0,length_mm=10)['depth_mm'])


def test_incomplete_field_is_not_zero_depth():
    surface,lung,affine=phantom()
    # Crop off the exterior wall while preserving physical lung coordinates.
    affine[:3,3]+=affine[:3,0]*10
    p=profile(surface,lung[10:],affine,0,0,length_mm=10)
    assert set(p['flags'])=={'incomplete_ct_field'}
    assert all(v is None for v in p['depth_mm'])
