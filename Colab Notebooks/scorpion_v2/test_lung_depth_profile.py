import numpy as np
import trimesh
from lung_depth_profile import profile,skin_aligned_section


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


def test_flattening_translates_depth_without_stretching_anatomical_layers():
    _,lung,affine=phantom();labels=np.zeros_like(lung)
    labels[15:19]=4  # LPS interval [-10.5,-6.5): thickness 4 mm.
    depth=dict(s_mm=[-1,0,1],z_lps_mm=[-1,0,1],y_lps_mm=0,
               skin_x_lps_mm=[-20,-17,-14])
    aligned=skin_aligned_section(depth,np.zeros_like(lung,dtype=np.int16),labels,affine,ray_step_mm=.25)
    for i,skin in enumerate(depth['skin_x_lps_mm']):
        np.testing.assert_allclose(aligned['x_lps_mm'][i]-skin,aligned['depth_mm'])
        d=aligned['depth_mm'][(aligned['labels'][i]==4)&aligned['valid'][i]]
        assert abs(len(d)*.25-4.)<=.25
        assert abs(d[0]-(-10.5-skin))<=.25
    np.testing.assert_allclose(np.diff([aligned['depth_mm'][np.flatnonzero(r==4)[0]]
        for r in aligned['labels']]),[-3,-3])


def test_flattened_image_does_not_invent_tissue_outside_CT_or_right_half_space():
    surface,lung,affine=phantom();depth=profile(surface,lung,affine,0,0,length_mm=10)
    labels=np.full_like(lung,4)
    aligned=skin_aligned_section(depth,np.zeros_like(lung),labels,affine,ray_step_mm=.5)
    assert np.all(aligned['x_lps_mm'][aligned['valid']]<=0)
    depth['skin_x_lps_mm'][0]=None
    missing=skin_aligned_section(depth,np.zeros_like(lung),labels,affine)
    assert not missing['valid'][0].any()
    affine[:3,3]+=affine[:3,0]*10
    cropped=skin_aligned_section(depth,np.zeros_like(lung[10:]),labels[10:],affine)
    assert not cropped['valid'][1:,0].any()
