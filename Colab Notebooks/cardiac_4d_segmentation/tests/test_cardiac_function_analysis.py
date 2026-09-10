import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cardiac_function_analysis import geometry, scenario_volumes, extrema, estimates, changes, verify_grid
from analyze_subtraction_and_rr import periodic_cycle_profile, assess_cycle_merge


def rows_for_cycle():
    return [dict(subject='synthetic',chamber='heart_ventricle_left',phase_id=f'p{i:02d}',
                 cycle_index=i//10,phase_percent_within_cycle=(i%10)*10,rr_percent=i*10,
                 derived_time_from_R0_ms=i*100,image_heart_rate_bpm=60,
                 volume_ml=100+30*np.cos(i*np.pi/5),clipped_volume_ml=100+30*np.cos(i*np.pi/5),
                 surface_area_mm2=1000+i*10,elongation=1,centroid_x_mm=0,centroid_y_mm=0,centroid_z_mm=0)
            for i in range(12)]


def test_voxel_volume_uses_affine_and_includes_all_components():
    mask=np.zeros((20,20,20),dtype=np.uint8);mask[1:5,2:7,3:9]=1;mask[15,15,15]=1
    a=np.diag([2.,3.,4.,1.]);g=geometry(mask,a)
    assert g['volume_ml']==pytest.approx(121*24/1000)
    assert g['connected_components_26']==2
    assert g['largest_component_fraction']==pytest.approx(120/121)
    assert not g['touches_image_boundary']


def test_geometry_is_rigid_invariant_and_scales_dimensionally():
    mask=np.zeros((25,25,25),dtype=bool);mask[2:15,4:12,5:10]=1
    base=geometry(mask,np.eye(4));angle=.63;c,s=np.cos(angle),np.sin(angle)
    a=np.array([[c,-s,0,20],[s,c,0,-3],[0,0,1,7],[0,0,0,1.]])
    rotated=geometry(mask,a)
    for key in ('volume_ml','surface_area_mm2','principal_long_mm','elongation'):
        assert rotated[key]==pytest.approx(base[key],rel=1e-6)
    scaled=geometry(mask,np.diag([2.,2.,2.,1.]))
    assert scaled['volume_ml']==pytest.approx(8*base['volume_ml'])
    assert scaled['surface_area_mm2']==pytest.approx(4*base['surface_area_mm2'])
    assert scaled['principal_long_mm']==pytest.approx(2*base['principal_long_mm'])


@pytest.mark.parametrize('values',[[0,2],[1,float('nan')],[1],[-1,2]])
def test_invalid_volumes_block_function(values):
    with pytest.raises(ValueError):extrema(values,list(range(len(values))))


def test_mask_must_be_binary_and_nonempty():
    with pytest.raises(ValueError):geometry(np.zeros((3,3,3)),np.eye(4))
    with pytest.raises(ValueError):geometry(np.full((3,3,3),2),np.eye(4))


def test_grid_requires_provenance_and_ct_registration():
    a=np.diag([.7,.8,1.,1.]);img=nib.Nifti1Image(np.zeros((3,4,5)),a)
    info=dict(shape=[3,4,5],voxel_spacing_mm=[.7,.8,1],affine_coordinate_system='NIfTI RAS+')
    verify_grid(img,info,img)
    with pytest.raises(ValueError):verify_grid(img,{**info,'voxel_spacing_mm':[1,1,1]})
    shifted=a.copy();shifted[0,3]=1
    with pytest.raises(ValueError):verify_grid(img,info,nib.Nifti1Image(np.zeros((3,4,5)),shifted))


def test_offset_and_scale_laws_hold_for_observed_and_fitted_cycle():
    result,_=estimates(rows_for_cycle(),periodic_cycle_profile,assess_cycle_merge)
    for estimator in ('observed_global_range','periodic_composite'):
        by_id={r['scenario']:r for r in result if r['estimator']==estimator}
        assert by_id['offset_10ml']['sv_ml']==pytest.approx(by_id['raw']['sv_ml'])
        assert by_id['offset_10ml']['edv_ml']==pytest.approx(by_id['raw']['edv_ml']-10)
        for sid,factor in [('scale_090',.9),('scale_110',1.1)]:
            assert by_id[sid]['ef_percent']==pytest.approx(by_id['raw']['ef_percent'])
            assert by_id[sid]['sv_ml']==pytest.approx(by_id['raw']['sv_ml']*factor)


def test_shell_uses_area_units_and_rejects_impossible_volume():
    rows=[dict(volume_ml=100,surface_area_mm2=20000)]
    assert scenario_volumes(rows,'shell_mm',1)==pytest.approx([80])
    with pytest.raises(ValueError):scenario_volumes(rows,'shell_mm',5)


def test_rejected_cycle_merge_never_returns_composite_estimate():
    rows=rows_for_cycle()
    for row in rows:
        if row['cycle_index']==1:row['image_heart_rate_bpm']=120
    result,profiles=estimates(rows,periodic_cycle_profile,assess_cycle_merge)
    composite=[r for r in result if r['estimator']=='periodic_composite']
    assert composite and all('sv_ml' not in r for r in composite)
    assert not profiles
    assert not any(r['estimator']=='model_sensitivity' for r in result)


def test_observed_extrema_preserve_cycle_and_temporal_order():
    rows=rows_for_cycle();rows[-1]['volume_ml']=160
    result,_=estimates(rows,periodic_cycle_profile,assess_cycle_merge)
    raw=next(r for r in result if r['scenario']=='raw' and r['estimator']=='observed_global_range')
    assert raw['ed_phase_id']=='p11' and raw['ed_cycle_index']==1
    assert raw['es_phase_id']=='p05' and raw['es_cycle_index']==0
    assert raw['ed_precedes_es_in_recording'] is False
    delta=changes(rows)
    assert sum(r['cycle_boundary_crossed'] for r in delta)==1
    rows[1]['derived_time_from_R0_ms']=rows[0]['derived_time_from_R0_ms']
    with pytest.raises(ValueError):changes(rows)
