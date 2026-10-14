"""Tiny synthetic provenance files only; no FEM, CT decoding or real pointer."""
import importlib.util
import json
from pathlib import Path
import pytest

MODULE = Path(__file__).parents[1]/'MATLAB_TRKG4_real_subjects/tools/georg_quality_shared_selection.py'
spec = importlib.util.spec_from_file_location('shared_selection', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def case(tmp_path):
    def write(rel, value):
        p = tmp_path/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value) if not isinstance(value, str) else value, encoding='utf-8')
        return p
    def row(p):
        return dict(path=str(p), sha256=m.sha(p))
    write('output/georg_current_model.local.json', {'old': True})
    b, f, seg = [tmp_path/n for n in ('bundle','fine2','seg')]
    roles = {}
    for role,name in [('native_CT_contour','body_ct_native.stl'),
                      ('corrected_CT_skin','body_ct_surface.stl'),
                      ('body_union_picker_surface','body_embedded_arms_370mm.stl')]:
        p = write('bundle/'+name, name)
        roles[role] = dict(path=name,sha256=m.sha(p))
    surface = write('bundle/fem.stl','FEM surface')
    passport = write('bundle/surface.json',dict(passed=True,
        candidate_surface_sha256=m.sha(surface),
        source_surface_sha256=roles['body_union_picker_surface']['sha256']))
    roles['FEM_boundary_reference'] = dict(path='fem.stl',sha256=m.sha(surface),
        passport_path='surface.json',passport_sha256=m.sha(passport),supplied=True,
        passport_surface_bytes_bound=True)
    ct = write('seg/static_ct.nii.gz','CT bytes')
    write('bundle/contour_qc.local.json',dict(ct_sha256=m.sha(ct)))
    geometry = write('bundle/geometry.local.json',dict(watertight=True,positive_volume=True,
        connected_components=1,self_intersecting_faces=0,
        surface_sha256=roles['body_union_picker_surface']['sha256'],
        joins={'left':{'cross_section_covered_fraction':1}}))
    landmark = write('bundle/landmarks/a.json',{'raw':[1,2,3]})
    write('bundle/geometry_bundle.local.json',dict(schema='georg_quality_geometry_bundle_v1',
        status='prepared_not_selected',ct_sha256=m.sha(ct), geometry_roles=roles,
        checks={k:True for k in ['byte_copies_verified','raw_normalised_centres_equal',
            'source_contexts_preserved','all_24_transfer_records_verified','source_hashes_rechecked']},
        sources=[row(geometry)],copies=[],products=[dict(path='landmarks/a.json',sha256=m.sha(landmark))]))
    config = write('source.json',dict(segmentation_directory='seg'))
    mesh = write('fine2/mesh.mat','mesh bytes')
    plan = dict(geometry_directory='bundle',shape_QC_passed=True,
                numerical_quadrature_points=[1024,4096])
    qc = dict(passed=True,all_positive=True,mesh_sha256=m.sha(mesh))
    for key,qkey in [('pose_reference_mesh','pose_reference_mesh_sha256'),
        ('exact_contact_parent_mesh','contact_master_mesh_sha256'),
        ('boundary_parent_map','contact_parent_maps_sha256'),
        ('geometry_boundary_parent_map','geometry_parent_maps_sha256')]:
        p=write('fine2/'+key,'bytes '+key);plan[key]=str(p);qc[qkey]=m.sha(p)
    write('fine2/mesh_quality.local.json',qc)
    sources = [row(write('seg/'+n+'.nii.gz', n)) for n in ('lung','bone','heart','fat')]
    sources += [row(write(f'fine2/fractions_q{q}.mat',str(q))) for q in (1024,4096)]
    plan['sources']=sources
    write('fine2/plan.local.json',plan)
    write('fine2/reference_materials.local.json',{})
    check=write('negative.json',dict(scope='selected pose only; contact discretization',
                                   passed=False,sources=[row(mesh)]))
    return tmp_path,b,f,config,tmp_path/'candidate.json',check,write,row


def prepared(case):
    root,b,f,c,out,check,_,_=case
    return m.prepare(root,b,f,c,out,[check])


def test_prepare_keeps_negative_and_pointer(case):
    root,_,_,_,out,_,_,_=case
    old=(root/'output/georg_current_model.local.json').read_bytes()
    c=prepared(case)
    assert c['status']=='geometry_reference_checked'
    assert c['local_numerical_checks'][0]['outcome'] is False
    assert all(c[k] is False for k in m.FALSE_FLAGS)
    assert c['segmentation_directory']=='seg'
    assert (root/'output/georg_current_model.local.json').read_bytes()==old
    with pytest.raises(ValueError,match='already exists'):prepared(case)


@pytest.mark.parametrize('file', ['seg/static_ct.nii.gz','seg/fat.nii.gz',
    'bundle/landmarks/a.json','bundle/fem.stl','fine2/mesh.mat','fine2/fractions_q4096.mat',
    'fine2/geometry_boundary_parent_map','fine2/exact_contact_parent_mesh'])
def test_changed_inputs_refused(case,file):
    case[6](file,'modified')
    with pytest.raises(ValueError,match='SHA'):prepared(case)
    assert not case[4].exists()


def receipt(case):
    root,_,_,_,out,_,write,row=case
    code=write('fixed_resolver.py','fixed source')
    report=write('guards_test.json',{'negative_guards_passed':True})
    return write('guards.json',dict(schema='georg_shared_selection_guards_v1',
        candidate_sha256=m.sha(out),boolean_guards_passed=True,legacy_routing_blocked=True,
        sources=[row(code)],tests=[row(report)]))


def test_select_only_synthetic_pointer_and_backup(case):
    prepared(case);g=receipt(case);root=case[0]
    pointer=root/'output/georg_current_model.local.json';old=pointer.read_bytes()
    c=m.select(root,case[4],g)
    assert c['selection_state']=='selected_geometry_only'
    assert c['quantitative_inversion_allowed'] is False
    assert list(pointer.parent.glob('*.bak'))[0].read_bytes()==old


@pytest.mark.parametrize('change',['pointer','source','guard','elevation'])
def test_select_fails_closed(case,change):
    prepared(case);g=receipt(case);root=case[0]
    if change=='pointer':case[6]('output/georg_current_model.local.json',{'new':True})
    elif change=='source':case[6]('fixed_resolver.py','changed')
    elif change=='guard':
        d=m.read(g);d['boolean_guards_passed']='true';case[6]('guards.json',d)
    else:
        d=m.read(case[4]);d['quantitative_inversion_allowed']=True;case[6]('candidate.json',d)
    before=(root/'output/georg_current_model.local.json').read_bytes()
    with pytest.raises(ValueError):m.select(root,case[4],g)
    assert (root/'output/georg_current_model.local.json').read_bytes()==before


@pytest.mark.parametrize('kind',['scope','plan','segmentation','boolean'])
def test_incomplete_or_wrong_metadata_rejected(case,kind):
    root,b,f,config,out,check,write,_=case
    if kind in ('scope','plan'):
        d=m.read(check)
        if kind=='scope':d.pop('scope')
        else:d['model_plan_sha256']='0'*64
        write('negative.json',d)
    elif kind=='segmentation':
        write('other/static_ct.nii.gz','CT bytes')
        write('source.json',{'segmentation_directory':'other'})
    else:
        d=m.read(f/'mesh_quality.local.json');d['passed']='true'
        write('fine2/mesh_quality.local.json',d)
    with pytest.raises(ValueError):prepared(case)
    assert not out.exists()
