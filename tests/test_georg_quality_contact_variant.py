"""Small provenance/contact-isolation tests; no FEM or production mesh loading."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
import georg_quality_contact_variant as tool


def put(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def toy_base(tmp_path):
    base=tmp_path/'fine2';base.mkdir()
    for name in ('mesh.mat','fractions_q1024.mat','fractions_q4096.mat'):
        (base/name).write_bytes(('immutable toy '+name).encode())
    tool.write_new(base/'mesh_quality.local.json',dict(passed=True,mesh_sha256=tool.sha(base/'mesh.mat')))
    tool.write_new(base/'reference_materials.local.json',dict(materials=tool.MATERIALS,
        frequency_hz=100000,rho_ohm_m=[2.,9.,48.,4.6,17.]))
    pose=tmp_path/'original_pose.mat';pose.write_bytes(b'original pose geometry')
    master=tmp_path/'fine1_master.mat';master.write_bytes(b'older contact triangulation')
    geometry=tmp_path/'geometry_to_original.mat';geometry.write_bytes(b'geometry boundary map')
    contactmap=tmp_path/'contact_to_fine1.mat';contactmap.write_bytes(b'contact boundary map')
    contacts=tmp_path/'contacts.local.json';tool.write_new(contacts,dict(centres=[[k,0,0] for k in range(12)]))
    paths=[*base.iterdir(),pose,master,geometry,contactmap,contacts]
    plan=dict(schema='georg_shared_ct_fem_v1',frequency_hz=100000,
        material_order=tool.MATERIALS,numerical_quadrature_points=list(tool.POINTS),
        integration_rules=[256,1024],constant_physical_contact_layout=True,
        contacts_file=str(contacts),pose_reference_mesh=str(pose),exact_contact_parent_mesh=str(master),
        boundary_parent_map=str(contactmap),geometry_boundary_parent_map=str(geometry),
        fraction_origin='CT mask integration',contact_impedance_ohm_m2=.00015915494309189532,
        source_rule='own_drive',geometry_accuracy_is_independently_validated=False,
        prepared_file='stale_prepared.mat',prepared_sha256='f'*64,
        sources=[dict(path=str(p),sha256=tool.sha(p)) for p in paths])
    tool.write_new(base/'plan.local.json',plan)
    return base


def test_prepare_is_bulk_identical_and_preserves_independent_geometry(tmp_path):
    base=toy_base(tmp_path);before={p:tool.sha(p) for p in base.iterdir()}
    (base/'prepared_material_old.mat').write_bytes(b'never copied or relabelled')
    out=tmp_path/'direct';facts=tool.prepare(base,out);plan=tool.read(out/'plan.local.json')
    assert {p:tool.sha(p) for p in before}==before
    assert not list(out.glob('prepared_material_*.mat'))
    assert all(tool.sha(base/n)==tool.sha(out/n)==facts['bulk_file_sha256'][n] for n in tool.FILES)
    original=tool.read(base/'plan.local.json')
    for key in ('pose_reference_mesh','geometry_boundary_parent_map','contacts_file',
                'constant_physical_contact_layout','contact_impedance_ohm_m2','source_rule',
                'numerical_quadrature_points','integration_rules'):
        assert plan[key]==original[key]
    assert plan['parent_directory']==str(base.resolve())
    assert plan['exact_contact_parent_mesh']==str(out.resolve()/'mesh.mat')
    assert plan['boundary_parent_map']==''
    assert plan['contact_parent_mesh_sha256']==tool.sha(base/'mesh.mat')
    assert 'prepared_file' not in plan and 'prepared_sha256' not in plan
    pinned={Path(s['path']):s['sha256'] for s in plan['sources']}
    assert pinned[base/'plan.local.json']==tool.sha(base/'plan.local.json')
    assert all(pinned[Path(s['path'])]==s['sha256'] for s in original['sources'])
    assert all(tool.sha(path)==digest for path,digest in pinned.items())
    assert facts['material_cache_policy']['expected_reused_rule_rows']==[1,2]
    assert not facts['material_cache_policy']['actual_reuse_verified_here']
    assert not facts['contact_patch_change_verified'] and not facts['FEM_was_run']
    assert plan['contact_patch_QC_status']=='not_evaluated'
    assert not plan['quantitative_inversion_allowed']
    assert not plan['global_mesh_convergence_proven']
    with pytest.raises(ValueError,match='fresh'):
        tool.prepare(base,out)


@pytest.mark.parametrize('where',['base','child','ancestor'])
def test_reject_overlapping_output(tmp_path,where):
    base=toy_base(tmp_path)
    out={'base':base,'child':base/'child','ancestor':tmp_path}[where]
    with pytest.raises(ValueError,match='fresh'):
        tool.prepare(base,out)
    assert not (base/'child').exists()


def test_changed_pinned_source_fails_before_output(tmp_path):
    base=toy_base(tmp_path);(base/'fractions_q4096.mat').write_bytes(b'changed')
    out=tmp_path/'direct'
    with pytest.raises(ValueError,match='source changed'):
        tool.prepare(base,out)
    assert not out.exists()


def test_invalid_qc_fails_before_output(tmp_path):
    base=toy_base(tmp_path);qc=tool.read(base/'mesh_quality.local.json');qc['passed']=False
    put(base/'mesh_quality.local.json',qc)
    plan=tool.read(base/'plan.local.json')
    for row in plan['sources']:
        row['sha256']=tool.sha(row['path'])
    put(base/'plan.local.json',plan)
    with pytest.raises(ValueError,match='shape QC'):
        tool.prepare(base,tmp_path/'direct')
    assert not (tmp_path/'direct').exists()


def test_legacy_coupled_geometry_map_is_preserved(tmp_path):
    base=toy_base(tmp_path);plan=tool.read(base/'plan.local.json')
    plan['pose_reference_mesh']=plan['exact_contact_parent_mesh']
    del plan['geometry_boundary_parent_map'];put(base/'plan.local.json',plan)
    out=tmp_path/'direct';tool.prepare(base,out)
    assert tool.read(out/'plan.local.json')['geometry_boundary_parent_map']==plan['boundary_parent_map']


def test_independent_pose_cannot_guess_missing_geometry_map(tmp_path):
    base=toy_base(tmp_path);plan=tool.read(base/'plan.local.json')
    del plan['geometry_boundary_parent_map'];put(base/'plan.local.json',plan)
    with pytest.raises(ValueError,match='Independent pose reference'):
        tool.prepare(base,tmp_path/'direct')
    assert not (tmp_path/'direct').exists()


def test_direct_base_identity_maps_remain_compatible(tmp_path):
    base=toy_base(tmp_path);plan=tool.read(base/'plan.local.json')
    plan.update(pose_reference_mesh=str(base/'mesh.mat'),exact_contact_parent_mesh=str(base/'mesh.mat'),
                boundary_parent_map='');del plan['geometry_boundary_parent_map']
    put(base/'plan.local.json',plan)
    out=tmp_path/'direct';tool.prepare(base,out)
    assert tool.read(out/'plan.local.json')['geometry_boundary_parent_map']==''


def test_signed_delta_retains_direction_and_direct_tau():
    result=tool.contact_difference([100.,100.,100.],[100.1,99.,102.])
    np.testing.assert_allclose(result['signed_difference_ohm'],[.1,-1.,2.])
    np.testing.assert_allclose(result['tolerance_ohm'],.02+.005*np.array([100.1,99.,102.]))
    np.testing.assert_allclose(result['signed_ratio'],np.array([.1,-1,2])/np.array(result['tolerance_ohm']))
    assert result['failed_values']==2 and not result['passed']


@pytest.mark.parametrize('a,b',[([],[]),([1],[1,2]),([[1]],[[1]]),([1],[float('nan')]),
    ([float('inf')],[1]),([-1.7e308],[1.7e308])])
def test_invalid_difference_vectors_fail(a,b):
    with pytest.raises(ValueError):tool.contact_difference(a,b)


def cache_identity(directory,parent_hash='c'*64):
    return dict(plan_sha256=tool.sha(directory/'plan.local.json'),mesh_sha256=tool.sha(directory/'mesh.mat'),
        parent_prepared_sha256=parent_hash,field_builder_sha256='a'*64,
        worker_sha256=tool.sha(tool.ROOT/'src/trkg4_georg_lateral_materials.m'),
        rule_helper_sha256=tool.sha(tool.ROOT/'src/trkg4_georg_material_rules.m'),quadrature_points=list(tool.POINTS),
        material_rules=dict(length_unit='mm',scale_nodes_to_si=True,node_scale_to_si=.001,
            scale_helper_sha256=tool.sha(tool.ROOT/'src/trkg4_scale_fmdl_to_si.m'),quadrature_points=list(tool.POINTS),
            fractions_sha256=[tool.sha(directory/f'fractions_q{q}.mat') for q in tool.POINTS]))


def saved_cache(file,identity,delta=1e-16):
    savemat(file,dict(identity=identity,homogeneous_bulk_relative_difference=delta,
                     parts=np.eye(4),mm=dict(nodes=np.ones((4,3)))))


def toy_controls(tmp_path):
    base=toy_base(tmp_path);variant=tmp_path/'direct';tool.prepare(base,variant)
    parent=base/'prepared_material_parent.mat';direct=variant/'prepared_material_direct.mat'
    saved_cache(parent,cache_identity(base));saved_cache(direct,cache_identity(variant,tool.sha(parent)))
    rows=[]
    for directory,cache,z in ((base,parent,[100.,101.,120.,121.]),(variant,direct,[100.1,100.,122.,121.])):
        plan=tool.read(directory/'plan.local.json')
        row=dict(identity=dict(worker_sha256=tool.sha(tool.ROOT/'src/run_georg_quality_forward.m'),
            plan_sha256=tool.sha(directory/'plan.local.json'),prepared_sha256=tool.sha(cache),
            physical_contacts_sha256=tool.sha(plan['contacts_file']),reference_sha256='e'*64,
            observations_sha256='d'*64,physical_montage_helper_sha256=tool.sha(tool.ROOT/'src/trkg4_georg_physical_montages.m'),
            contact_inheritance=dict(contact_parent_mesh_sha256=tool.sha(plan['exact_contact_parent_mesh']),
                current_mesh_sha256=tool.sha(directory/'mesh.mat'),
                boundary_parent_map_sha256=tool.sha(plan['boundary_parent_map']) if plan['boundary_parent_map'] else '',
                mapping_helper_sha256=tool.sha(tool.ROOT/'src/trkg4_georg_contact_inheritance.m'))),
            parameters=[1.,2.,3.,4.,5.,6.,0.,0.,0.],pose=dict(centre_lps_mm=[-204.,-20.,-207.]),
            record_ids=['r1','r1','r2','r2'],axes=['long']*4,states=['inhale','exhale']*2,
            target_Z_ohm=[99.,100.,119.,120.],source_rule='own_drive',fine_Z_ohm=z,
            patch_qc=dict(passed=True),system_checks_passed=True,quadrature_passed=True,
            patches=[dict(record_id='r1',xyz=[[0.,0.,0.],[1.,0.,0.]],gids=[1,2],electrical_groups=2,
                physical_patches=2,faces=[[1,2,3]],areas=[1.,1.])])
        rows.append(row)
    # Different areas/faces are the deliberately isolated contact approximation.
    rows[1]['patches'][0].update(faces=[[1,2,4],[2,3,4]],areas=[.98,1.01])
    paths=[tmp_path/'inherited.local.json',tmp_path/'direct.local.json']
    for file,row in zip(paths,rows):put(file,row)
    return base,variant,parent,direct,paths


def test_real_mat_v5_metadata_is_filtered_and_ancestry_verified(tmp_path,monkeypatch):
    import scipy.io
    base,variant,parent,direct,paths=toy_controls(tmp_path)
    original=scipy.io.loadmat;calls=[]
    def filtered(*args,**kwargs):
        calls.append(kwargs['variable_names'])
        return original(*args,**kwargs)
    monkeypatch.setattr(scipy.io,'loadmat',filtered)
    result=tool.compare_saved_controls(base,variant,*paths)
    assert calls==[['identity','homogeneous_bulk_relative_difference']]*2
    assert result['actual_parent_prepared_ancestry_verified']
    assert result['units_and_material_cache_identity_verified_here']
    assert result['reused_rule_rows']==[1,2] and not result['matrix_entries_compared']
    assert result['parent_prepared_sha256']==tool.sha(parent)
    assert result['direct_prepared_sha256']==tool.sha(direct)
    np.testing.assert_allclose(result['signed_difference_ohm'],[.1,-1,2,0])
    assert not result['passed'] and result['failed_values']==2
    assert not result['quantitative_result_accepted']
    assert not result['final_joint_or_selected_validation_complete']
    assert all(tool.sha(s['path'])==s['sha256'] for s in result['sources'])


@pytest.mark.parametrize('defect',['parent','units','fractions','points','field_builder','helper','plan','homogeneous'])
def test_ancestry_rejects_wrong_parent_units_rules_or_operators(tmp_path,defect):
    base,variant,parent,direct,paths=toy_controls(tmp_path)
    identity=cache_identity(variant,tool.sha(parent));delta=1e-16
    if defect=='parent':identity['parent_prepared_sha256']='b'*64
    elif defect=='units':identity['material_rules']['node_scale_to_si']=1.
    elif defect=='fractions':identity['material_rules']['fractions_sha256'][1]='b'*64
    elif defect=='points':identity['quadrature_points']=[256,1024]
    elif defect=='field_builder':identity['field_builder_sha256']='b'*64
    elif defect=='helper':identity['rule_helper_sha256']='b'*64
    elif defect=='plan':identity['plan_sha256']='b'*64
    elif defect=='homogeneous':delta=1e-8
    saved_cache(direct,identity,delta)
    with pytest.raises(ValueError):tool.verify_reused_ancestry(base,variant,parent,direct)


@pytest.mark.parametrize('defect',['rho','pose','reference','inventory','groups','master','QC','bulk','worker'])
def test_contact_comparison_rejects_nonisolated_or_unqualified_results(tmp_path,defect):
    base,variant,parent,direct,paths=toy_controls(tmp_path);row=tool.read(paths[1])
    if defect=='rho':row['parameters'][0]+=1
    elif defect=='pose':row['pose']['centre_lps_mm'][0]+=1
    elif defect=='reference':row['identity']['reference_sha256']='b'*64
    elif defect=='inventory':row['identity']['physical_contacts_sha256']='b'*64
    elif defect=='groups':row['patches'][0]['gids']=[2,1]
    elif defect=='master':row['identity']['contact_inheritance']['contact_parent_mesh_sha256']='b'*64
    elif defect=='QC':row['patch_qc']['passed']=False
    elif defect=='worker':row['identity']['worker_sha256']='b'*64
    elif defect=='bulk':(variant/'mesh.mat').write_bytes(b'changed mesh')
    put(paths[1],row)
    with pytest.raises(ValueError):tool.compare_saved_controls(base,variant,*paths)


def test_actual_parent_replacement_invalidates_ancestry(tmp_path):
    base,variant,parent,direct,paths=toy_controls(tmp_path)
    identity=cache_identity(base);identity['field_builder_sha256']='b'*64
    saved_cache(parent,identity)
    with pytest.raises(ValueError,match='actual parent'):
        tool.verify_reused_ancestry(base,variant,parent,direct)


def test_failed_quadrature_never_appears_as_passed_contact_control(tmp_path):
    base,variant,parent,direct,paths=toy_controls(tmp_path)
    row=tool.read(paths[1]);row['fine_Z_ohm']=tool.read(paths[0])['fine_Z_ohm']
    row['quadrature_passed']=False;put(paths[1],row)
    result=tool.compare_saved_controls(base,variant,*paths)
    assert result['contact_difference_passed']
    assert not result['quadrature_checks_passed'] and not result['passed']


def test_missing_native_metadata_backend_fails_explicitly(tmp_path):
    with pytest.raises(ValueError,match='needs h5py or'):
        tool._read_native_hdf_metadata(tmp_path/'missing.mat',None)
