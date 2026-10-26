"""Provenance and native-grid contracts; neural inference is not mocked as validation."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ct_fem_core as core
import ct_fem_ct_pipeline as pipeline
from test_ct_fem_preparation import fixture_config

POLICY = dict(include_TS_tissues=True,maximum_added_relative=.001,maximum_disconnected_relative=.001,
              maximum_internal_cavity_relative=.001)


def cached_segmentation(tmp_path, ct):
    folder = tmp_path/'seg';folder.mkdir()
    image = nib.load(ct);data = np.zeros(image.shape,np.uint8);data[3:21,3:21,3:21] = 1
    mask = nib.Nifti1Image(data,image.affine);mask.header.set_xyzt_units('mm')
    nib.save(mask,folder/'body.nii.gz');nib.save(mask,folder/'body_trunc.nii.gz')
    value = dict(complete=True,backend='TotalSegmentator',version='2.18.0',ct_sha256=core.sha(ct),
        products=dict(body=dict(sha256=core.sha(folder/'body.nii.gz'),
            source_masks=[dict(file='body_trunc.nii.gz',sha256=core.sha(folder/'body_trunc.nii.gz'))])))
    core.write_new(folder/'segmentation.local.json',value)
    return folder


def test_changed_source_class_and_wrong_CT_block_cached_TS(tmp_path):
    path,cfg = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct'])
    _,bound = pipeline.verify_segmentation(folder,cfg['inputs']['ct'],roles=['body'])
    assert str((folder/'body_trunc.nii.gz').resolve()) in bound
    with (folder/'body_trunc.nii.gz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='Changed source class'):
        pipeline.verify_segmentation(folder,cfg['inputs']['ct'],roles=['body'])
    other = tmp_path/'different_CT.nii.gz';other.write_bytes(b'another CT')
    with pytest.raises(ValueError,match='exact CT'):
        pipeline.verify_segmentation(folder,other,roles=['body'])


def test_TS_v1_and_incomplete_passports_are_rejected(tmp_path):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct'])
    passport = folder/'segmentation.local.json'; value = core.read(passport)
    for change in [dict(version='1.5.7'),dict(complete=False),dict(backend='manual')]:
        passport.write_text(__import__('json').dumps(dict(value,**change)),encoding='utf-8')
        with pytest.raises(ValueError,match='TotalSegmentator v2'):
            pipeline.verify_segmentation(folder,cfg['inputs']['ct'],roles=['body'])


def test_TS_body_native_affine_STL_and_no_morphology(tmp_path):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct']); out = tmp_path/'export';out.mkdir()
    original = core.sha(folder/'body.nii.gz')
    source = pipeline.surface_from_body(folder,cfg,out,POLICY)
    qc = core.read(out/'body_STL_QC.local.json')
    assert source.is_file() and qc['surface_QC']['passed']
    assert qc['added_TS_tissue_voxels'] == 0
    assert not qc['HU_threshold_fallback'] and not qc['manual_STL_fallback'] and not qc['morphology_applied']
    assert original == core.sha(folder/'body.nii.gz')
    np.testing.assert_array_equal(nib.load(out/'computational_body_mask.nii.gz').affine,nib.load(cfg['inputs']['ct']).affine)


def test_incompatible_TS_body_grid_cannot_be_resampled_silently(tmp_path):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct']);image = nib.load(folder/'body.nii.gz')
    affine = image.affine.copy();affine[0,3] += .5
    wrong = nib.Nifti1Image(np.asanyarray(image.dataobj),affine);wrong.header.set_xyzt_units('mm');nib.save(wrong,folder/'body.nii.gz')
    with pytest.raises(ValueError,match='native CT grid'):
        pipeline.surface_from_body(folder,cfg,tmp_path,POLICY)


def test_excessive_body_tissue_disagreement_stops_before_STL(tmp_path):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct'])
    image = nib.load(cfg['inputs']['masks']['fat']);data = np.ones(image.shape,np.uint8)
    nib.save(nib.Nifti1Image(data,image.affine),cfg['inputs']['masks']['fat'])
    with pytest.raises(ValueError,match='disagreement'):
        pipeline.surface_from_body(folder,cfg,tmp_path,POLICY)


def test_missing_placement_rule_cannot_silently_reuse_old_centres(tmp_path):
    from ct_fem_contact_placement import apply
    import trimesh
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    with pytest.raises(ValueError,match='Every physical contact'):
        apply(cfg,trimesh.creation.box(),[])


def test_lateral_distance_follows_the_actual_section_arc():
    import trimesh
    from ct_fem_contact_placement import skin_arc
    sphere = trimesh.creation.icosphere(subdivisions=5,radius=40.)
    points = skin_arc(sphere,0.,0.,[-20.,0.,20.])
    expected = [[-40*np.cos(.5),0.,-40*np.sin(.5)],[-40.,0.,0.],[-40*np.cos(.5),0.,40*np.sin(.5)]]
    np.testing.assert_allclose(points,expected,atol=.03)
    # The 40mm path has a shorter chord; confusing these changes the montage.
    assert np.linalg.norm(points[-1]-points[0]) < 39.


def test_isolated_fat_is_preserved_as_diagnostic_and_counted(tmp_path):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct'])
    source = Path(cfg['inputs']['masks']['fat']);image = nib.load(source)
    data = np.asanyarray(image.dataobj).copy();data[1,1,1] = 1
    image = nib.Nifti1Image(data,image.affine);image.header.set_xyzt_units('mm');nib.save(image,source)
    digest = core.sha(source);out = tmp_path/'export';out.mkdir()
    pipeline.surface_from_body(folder,cfg,out,POLICY)
    qc = core.read(out/'body_STL_QC.local.json')
    assert qc['excluded_material_voxels']['fat'] == qc['isolated_TS_component_voxels'] == 1
    assert qc['anatomical_disagreement_pending_review']
    assert core.sha(source) == digest
    assert nib.load(out/'isolated_TS_components.nii.gz').get_fdata().sum() == 1
    assert nib.load(cfg['inputs']['masks']['fat']).get_fdata()[1,1,1] == 0


def test_even_one_disconnected_heart_voxel_blocks_the_domain(tmp_path):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct'])
    source = Path(cfg['inputs']['masks']['heart']);image = nib.load(source)
    data = np.asanyarray(image.dataobj).copy();data[1,1,1] = 1
    nib.save(nib.Nifti1Image(data,image.affine),source)
    with pytest.raises(ValueError,match='include heart/lung'):
        pipeline.surface_from_body(folder,cfg,tmp_path,POLICY)


@pytest.mark.parametrize('width,accepted',[(1,True),(3,False)])
def test_internal_cavity_surfaces_have_a_separate_bounded_solid_assumption(tmp_path,width,accepted):
    path,_ = fixture_config(tmp_path);cfg = core.load_config(path)
    folder = cached_segmentation(tmp_path,cfg['inputs']['ct'])
    source = folder/'body.nii.gz';image = nib.load(source);data = np.asanyarray(image.dataobj).copy()
    data[10:10+width,10:10+width,10:10+width] = 0
    image = nib.Nifti1Image(data,image.affine);image.header.set_xyzt_units('mm');nib.save(image,source)
    digest = core.sha(source);out = tmp_path/'export';out.mkdir()
    if accepted:
        pipeline.surface_from_body(folder,cfg,out,POLICY)
        qc = core.read(out/'body_STL_QC.local.json')
        assert qc['internal_cavity_components'] == 1
        assert qc['internal_cavity_relative_volume'] < .001
        assert (out/'native_TS_all_interfaces.stl').is_file()
        assert qc['anatomical_disagreement_pending_review']
    else:
        with pytest.raises(ValueError,match='cavity interfaces'):
            pipeline.surface_from_body(folder,cfg,out,POLICY)
    assert digest == core.sha(source)
