"""Tiny real surface transfer and negative provenance tests; no FEM."""
import copy
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
import georg_quality_geometry_bundle as tool


def put(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')


def test_exact_original_transfer_definition_loads_without_mesher_modules():
    names=('gmsh','pymeshlab','georg_ct_model','scorpion_georg_fem_prepare')
    before={name:name in sys.modules for name in names}
    function=tool.load_transfer()
    source=tool.ROOT/'tools/georg_ct_model.py'
    original=next(n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='transfer')
    expected=compile(ast.Module(body=[original],type_ignores=[]),str(source),'exec')
    expected_code=next(c for c in expected.co_consts if hasattr(c,'co_name') and c.co_name=='transfer')
    assert function.__code__==expected_code
    assert {name:name in sys.modules for name in names}==before


def inputs(tmp_path):
    geometry=tmp_path/'geometry';raw=tmp_path/'originals';normal=tmp_path/'normalised'
    for p in (geometry,raw,normal):p.mkdir()
    trimesh.creation.box(extents=[9,9,9]).export(geometry/'body_ct_native.stl')
    (geometry/'body_ct_surface.stl').write_bytes((geometry/'body_ct_native.stl').read_bytes())
    trimesh.creation.box(extents=[10,10,10]).export(geometry/'body_embedded_arms_370mm.stl')
    put(geometry/'geometry.local.json',dict(surface_sha256=tool.sha(geometry/'body_embedded_arms_370mm.stl'),
        source_sha256=tool.sha(geometry/'body_ct_surface.stl'),status='historical_geometry_status_unchanged'))
    put(geometry/'contour_qc.local.json',dict(ct_sha256='a'*64))
    (geometry/'extra_geometry_asset.bin').write_bytes(b'geometry asset, copied exactly')
    for n in range(2,8):
        name=f'georg_tepc_{n}.json'
        ctx=dict(subject_pseudonym='Georg',experiment='scorpion_v2_georg',experiment_date='2026-10-05',
            electrode_configuration_family='TEPC',physical_coordinate_frame='LPS',units='mm',
            body_sha256='b'*64,ct_sha256='a'*64,tissue_mask_sha256=dict(fat='c'*64),bei_channel=None,
            recording=dict(id='012',montage_id='georg_tepc_2',label='Исходная метка автора'),
            status='manual_reconstruction_pending_confirmation',reconstruction_notes='Авторские точки')
        payload=dict(schema='trkg4_ttrkg_surface_v3',modality='TTRKG',montage='four_electrode_surface',
            placement_mode='explicit_points',coordinate_system='surface_path_v1',units='mm',symmetry=None,
            source_basename='body_embedded_arms_370mm.stl',electrode_order=tool.ROLES,
            positive_guide_points_xyz_mm=[],negative_guide_points_xyz_mm=[],
            reference_definition='user_selected_centres_on_external_body_surface',
            electrode_centers_status='preliminary_surface_selection; exact_centres_resolved_by_MATLAB',
            preliminary_electrode_centers_xyz_mm={k:[5.5,float(y),n*.01] for k,y in zip(tool.ROLES,[-2,-1,1,2])},
            measurement_context=ctx)
        # BOM/newlines/non-ASCII make preservation a byte test, not JSON equality.
        (raw/name).write_bytes(b'\xef\xbb\xbf'+json.dumps(payload,ensure_ascii=False,indent=3).encode()+b'\r\n')
        normalised=copy.deepcopy(payload)
        normalised['measurement_context']['recording']=dict(id=f'{n+10:03d}',montage_id=f'georg_tepc_{n}',label=f'ТЕПС {n}')
        normalised['measurement_context']['status']='manual_reconstruction_surface_uncertainty_unquantified'
        normalised['import_provenance']=dict(original_sha256=tool.sha(raw/name),selected_coordinates_changed=False,
            original_recording=copy.deepcopy(ctx['recording']),mapping_source='direct_author_confirmation_2026-10-11',
            anatomical_accuracy_validated=False,signs_and_bei_channel_confirmed=False)
        put(normal/name,normalised)
    return geometry,raw,normal


def test_bundle_uses_real_transfer_only_on_copy_and_keeps_all_sources(tmp_path,monkeypatch):
    geometry,raw,normal=inputs(tmp_path);out=tmp_path/'bundle'
    before={p:p.read_bytes() for directory in (geometry,raw,normal) for p in directory.iterdir()}
    original_transfer=tool.load_transfer();calls=[]
    def spy(target,source):
        assert target==out.resolve() and target!=geometry
        assert source==out/'landmark_sources/normalised'
        assert (target/'body_embedded_arms_370mm.stl').read_bytes()==before[geometry/'body_embedded_arms_370mm.stl']
        calls.append((target,source));return original_transfer(target,source)
    monkeypatch.setattr(tool,'load_transfer',lambda:spy)
    manifest=tool.prepare(geometry,raw,normal,out,filename_mapping_confirmed=True)
    assert len(calls)==1 and manifest['status']=='prepared_not_selected'
    assert all(p.read_bytes()==value for p,value in before.items())
    assert len(manifest['landmarks'])==6 and len(tool.read(out/'centre_transfer.local.json'))==24
    for record in manifest['copies']:
        assert (out/record['path']).read_bytes()==Path(record['source_path']).read_bytes()
        assert record['byte_identical_verified'] and tool.sha(out/record['path'])==record['sha256']
    for n in range(2,8):
        name=f'georg_tepc_{n}.json';original=tool.read(raw/name);source=tool.read(normal/name)
        derived=tool.read(out/'landmarks'/name);prov=derived['ct_transfer_provenance']
        assert (out/'landmark_sources/originals'/name).read_bytes()==before[raw/name]
        assert (out/'landmark_sources/normalised'/name).read_bytes()==before[normal/name]
        assert prov['raw_source_measurement_context']==original['measurement_context']
        assert prov['normalised_source_measurement_context']==source['measurement_context']
        assert prov['raw_original_sha256']==tool.sha(raw/name)
        assert prov['normalised_sha256']==tool.sha(normal/name)
        assert prov['old_body_sha256']=='b'*64
        assert prov['new_body_sha256']==tool.sha(out/'body_embedded_arms_370mm.stl')
        assert derived['measurement_context']['recording']['id']==f'{n+10:03d}'
        assert original['measurement_context']['recording']['id']=='012'
        assert not prov['is_new_manual_placement'] and not prov['displacement_is_placement_error']
        assert prov['projection_is_candidate'] and prov['placement_accuracy_mm'] is None
        np.testing.assert_allclose(prov['displacement_mm'],[.5]*4)
        np.testing.assert_allclose(np.array(prov['projected_centres_lps_mm'])[:,0],5.)
    for key in ('global_model_pointer_changed','common_model_selected','mesh_convergence_passed',
                'integration_comparison_passed','quantitative_inversion_allowed','global_mesh_convergence_proven',
                'physiological_validation','placement_accuracy_is_known','FEM_was_run','legacy_worker_routing_changed'):
        assert manifest[key] is False
    assert all(tool.sha(r['path'])==r['sha256'] for r in manifest['sources'])
    assert all(tool.sha(out/r['path'])==r['sha256'] for r in manifest['products'])
    assert tool.read(out/'geometry.local.json')==tool.read(geometry/'geometry.local.json')
    assert not (out/'georg_current_model.local.json').exists()
    assert not (out/'georg_tepc_picker.html').exists()
    with pytest.raises(ValueError,match='fresh'):
        tool.prepare(geometry,raw,normal,out,filename_mapping_confirmed=True)


@pytest.mark.parametrize('defect',['coordinates','previous_projection','LPS','mm','CT','body',
                                 'recording','original_SHA','import_changed','missing_file'])
def test_invalid_sources_rejected_before_creating_bundle(tmp_path,defect):
    geometry,raw,normal=inputs(tmp_path);file=normal/'georg_tepc_3.json';value=tool.read(file)
    if defect=='coordinates':value['preliminary_electrode_centers_xyz_mm']['I_plus'][0]+=.01
    elif defect=='previous_projection':value['ct_transfer_provenance']={'is_new_manual_placement':False}
    elif defect=='LPS':value['measurement_context']['physical_coordinate_frame']='RAS'
    elif defect=='mm':value['measurement_context']['units']='m'
    elif defect=='CT':value['measurement_context']['ct_sha256']='d'*64
    elif defect=='body':value['measurement_context']['body_sha256']='d'*64
    elif defect=='recording':value['measurement_context']['recording']['id']='012'
    elif defect=='original_SHA':value['import_provenance']['original_sha256']='d'*64
    elif defect=='import_changed':value['import_provenance']['selected_coordinates_changed']=True
    if defect=='missing_file':file.unlink()
    else:put(file,value)
    out=tmp_path/'bundle'
    with pytest.raises(ValueError):tool.prepare(geometry,raw,normal,out,filename_mapping_confirmed=True)
    assert not out.exists()


def test_mapping_confirmation_is_required(tmp_path):
    geometry,raw,normal=inputs(tmp_path)
    with pytest.raises(ValueError,match='author confirmation'):
        tool.prepare(geometry,raw,normal,tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


@pytest.mark.parametrize('which',[0,1,2])
def test_bundle_cannot_be_nested_in_sources(tmp_path,which):
    args=inputs(tmp_path)
    with pytest.raises(ValueError,match='fresh'):
        tool.prepare(*args,args[which]/'bundle',filename_mapping_confirmed=True)


def test_refuse_to_clone_existing_transfer_products(tmp_path):
    geometry,raw,normal=inputs(tmp_path)
    put(geometry/'centre_transfer.local.json',[])
    with pytest.raises(ValueError,match='previous transfer'):
        tool.prepare(geometry,raw,normal,tmp_path/'bundle',filename_mapping_confirmed=True)


def test_geometry_hash_mismatch_rejected(tmp_path):
    geometry,raw,normal=inputs(tmp_path)
    (geometry/'body_embedded_arms_370mm.stl').write_bytes(b'changed after QC')
    with pytest.raises(ValueError,match='passport differs'):
        tool.prepare(geometry,raw,normal,tmp_path/'bundle',filename_mapping_confirmed=True)


def fem_reference(tmp_path,geometry):
    surface=tmp_path/'actual_fem_boundary.stl'
    trimesh.creation.box(extents=[8,8,8]).export(surface)
    passport=tmp_path/'surface_qc.local.json'
    put(passport,dict(source_surface_sha256=tool.sha(geometry/'body_embedded_arms_370mm.stl'),
        candidate_surface_sha256=tool.sha(surface),passed=True,actual_solved_boundary_is_fixed_hierarchy_reference=True))
    return surface,passport


@pytest.mark.parametrize('include_surface',[True,False])
def test_fem_reference_never_replaces_picker_surface(tmp_path,include_surface):
    geometry,raw,normal=inputs(tmp_path);surface,passport=fem_reference(tmp_path,geometry)
    out=tmp_path/'bundle'
    result=tool.prepare(geometry,raw,normal,out,filename_mapping_confirmed=True,
        fem_surface=surface if include_surface else None,fem_surface_passport=passport)
    ref=result['geometry_roles']['FEM_boundary_reference']
    assert ref['supplied']==include_surface and ref['passport_surface_bytes_bound']==include_surface
    assert not ref['used_as_picker_geometry'] and not ref['used_for_landmark_projection']
    assert (out/'body_embedded_arms_370mm.stl').read_bytes()==(geometry/'body_embedded_arms_370mm.stl').read_bytes()
    assert (out/'fem_reference/surface_passport.local.json').read_bytes()==passport.read_bytes()
    if include_surface:
        assert (out/ref['path']).read_bytes()==surface.read_bytes()
        assert ref['sha256']!=result['geometry_roles']['body_union_picker_surface']['sha256']
    prov=result['landmarks'][0]
    np.testing.assert_allclose(np.array(prov['projected_centres_lps_mm'])[:,0],5.)  # Not the FEM box at x=4.


@pytest.mark.parametrize('field',['candidate_surface_sha256','source_surface_sha256'])
def test_wrong_fem_passport_binding_fails_before_copy(tmp_path,field):
    geometry,raw,normal=inputs(tmp_path);surface,passport=fem_reference(tmp_path,geometry)
    value=tool.read(passport);value[field]='d'*64;put(passport,value)
    with pytest.raises(ValueError):
        tool.prepare(geometry,raw,normal,tmp_path/'bundle',filename_mapping_confirmed=True,
                     fem_surface=surface,fem_surface_passport=passport)
    assert not (tmp_path/'bundle').exists()


def test_mid_preparation_source_change_leaves_no_completed_manifest(tmp_path,monkeypatch):
    geometry,raw,normal=inputs(tmp_path)
    transfer=tool.load_transfer()
    def change_source(out,source):
        transfer(out,source)
        (normal/'georg_tepc_2.json').write_bytes(b'changed concurrently')
    monkeypatch.setattr(tool,'load_transfer',lambda:change_source)
    out=tmp_path/'bundle'
    with pytest.raises(ValueError,match='changed during'):
        tool.prepare(geometry,raw,normal,out,filename_mapping_confirmed=True)
    assert not (out/'geometry_bundle.local.json').exists()
