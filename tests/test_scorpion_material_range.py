"""Bounded plans, immutable provenance and small canonical numerical checks."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('material_range',ROOT/'MATLAB_TRKG4_real_subjects/tools/scorpion_material_range.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def fixture(tmp_path):
    base=tmp_path/'base';base.mkdir()
    for name in ['materials/mesh.mat','cem/cem_contacts.mat','electrical/prepared_materials.mat','registry.json']:
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic fixture, not FEM')
    ref=dict(frequency_hz=100000,material_order=module.ORDER,inhale_rho_ohm_m=[3.,15.,20.,2.,25.],
        registry=module.pin(base/'registry.json'))
    write(base/'reference_materials.local.json',ref)
    write(base/'electrical_inputs.local.json',dict(subject_alias='nik',source_files=[],observations=[
        dict(status='eligible_conditional',case_id='L140_TT2x2',record_id='synthetic',role='ttrkg',
             state='inhale',Z_ohm=10.,annotation_status='pending_manual_review')]))
    identity=dict(mesh_sha256=module.sha(base/'materials/mesh.mat'),cem_sha256=module.sha(base/'cem/cem_contacts.mat'),
        reference_sha256=module.sha(base/'reference_materials.local.json'),inputs_sha256=module.sha(base/'electrical_inputs.local.json'),
        field_kernel_sha256='f'*64,scale_kernel_sha256=module.sha(ROOT/'MATLAB_TRKG4_real_subjects/src/trkg4_scale_fmdl_to_si.m'),
        response_kernel_sha256=module.sha(ROOT/'MATLAB_TRKG4_real_subjects/src/trkg4_lateral_cem_response.m'))
    baseline=dict(subject_alias='nik',FEM_was_run=True,limited_run=False,frequency_hz=100000,
        derivative_checks=dict(performed=True,passed=True,active_material_indices=[1,2,4,5]),
        system_checks=[dict(system_relative_residual=1e-12,reciprocity_ohm=1e-12)],identity=identity,
        answers=[dict(case_id=c,state='inhale',rule=q) for c in ['L140_TT2x2','L60_TT2x2'] for q in [1024,4096]])
    write(base/'electrical/forward.local.json',baseline)
    return base,ref


def test_grid_exact_scope_and_blood_is_explicit():
    rows=module.rows_for_scan(dict(inhale_rho_ohm_m=[9.,17.,23.,4.,28.]))
    fine=[r for r in rows if r['rule']==4096];coarse=[r for r in rows if r['rule']==1024]
    assert len(rows)==20 and len(fine)==15 and len(coarse)==5
    assert {(r['rho_ohm_m'][0],r['rho_ohm_m'][1]) for r in coarse}==module.COARSE
    assert len({tuple(r['rho_ohm_m'][:2]) for r in fine})==15
    assert sum(r['finite_difference'] for r in rows)==1 and rows[0]['finite_difference']
    assert all(r['rho_ohm_m'][2:]==[23.,1.35,28.] for r in rows)
    assert all(r['parts_row']==(1 if r['rule']==1024 else 2) for r in rows)


def test_prepare_scan_pins_sources_and_never_changes_baseline(tmp_path):
    base,_=fixture(tmp_path);before=module.sha(base/'electrical/forward.local.json')
    out=tmp_path/'new';p=module.prepare(base,'scan',out)
    assert len(p['rows'])==20 and len(p['case_ids'])==2
    assert p['wholeheart_is_author_blood_replacement'] and p['fat_and_bone_fixed_registry_are_assumptions']
    assert p['soft_background_is_effective_unlabelled_mixture'] and not p['blood_replacement_is_tissue_validation']
    assert not p['current_is_Z_multiplier'] and p['source_relative_phase']=='unknown'
    assert not p['individual_fitted'] and not p['clinical_validation']
    assert before==module.sha(base/'electrical/forward.local.json')
    for source in p['source_files']:module.check_pin(source)
    with pytest.raises(ValueError):module.prepare(base,'scan',out)


@pytest.mark.parametrize('change',['mesh','reference','FD','cases'])
def test_bad_baseline_or_case_rejected_before_output(tmp_path,change):
    base,_=fixture(tmp_path);cases=None
    if change=='mesh':(base/'materials/mesh.mat').write_bytes(b'changed')
    if change=='reference':(base/'reference_materials.local.json').write_text('{}',encoding='utf8')
    if change=='FD':
        b=module.read(base/'electrical/forward.local.json');b['derivative_checks']['passed']=False;write(base/'electrical/forward.local.json',b)
    if change=='cases':cases='absent'
    with pytest.raises((ValueError,KeyError)):module.prepare(base,'scan',tmp_path/'out',cases)
    assert not (tmp_path/'out').exists()


def heart_fixture(base):
    b=module.read(base/'electrical/forward.local.json');variants=[]
    for g in ['wholeheart_ct','sphere_v095','sphere_v100','sphere_v105']:
        for q in [1024,4096]:
            v=base/f'heart_pilot/{g}_q{q}.mat';v.parent.mkdir(exist_ok=True);v.write_bytes(b'fractions fixture')
            variants.append(dict(id=f'{g}_q{q}',geometry=g,q=q,file=str(v),sha256=module.sha(v)))
    plan=dict(subject_alias='nik',mesh_sha256=b['identity']['mesh_sha256'],material_order=module.ORDER,variants=variants)
    p=base/'heart_pilot/plan.local.json';write(p,plan)
    for v in variants:
        directory=p.parent/('assemble_'+v['id']);directory.mkdir();parts=directory/'parts.mat';parts.write_bytes(b'parts fixture')
        report=dict(status='canonical_weighted_assembly_checked',identity=dict(subject_alias='nik',
            mesh_sha256=b['identity']['mesh_sha256'],plan_sha256=module.sha(p),
            **{k:b['identity'][k] for k in ('field_kernel_sha256','scale_kernel_sha256','reference_sha256')}),parts_sha256=module.sha(parts),
            homogeneous_error=1e-15,heterogeneous_error=1e-15)
        write(directory/'report.local.json',report)


def test_heart_eight_sources_primary_same_blood_and_no_invented_FD(tmp_path):
    base,_=fixture(tmp_path);heart_fixture(base)
    p=module.prepare(base,'heart',tmp_path/'heart')
    assert len(p['rows'])==8 and len(p['operator_sources'])==8 and p['case_ids']==[module.PRIMARY]
    assert all(r['parts_row']==1 and not r['finite_difference'] for r in p['rows'])
    assert all(o['layout']=='1x5' for o in p['operator_sources'])
    assert len({tuple(r['rho_ohm_m']) for r in p['rows']})==1
    assert p['FD_scope']=='not_run_for_heart'
    parts=base/'heart_pilot/assemble_wholeheart_ct_q1024/parts.mat';parts.write_bytes(b'corrupt')
    with pytest.raises(ValueError):module.prepare(base,'heart',tmp_path/'corrupt')


@pytest.mark.parametrize('subject,number_of_cases',[('nik',4),('yaroslav',4),('georg',7)])
def test_prepare_from_real_complete_baseline_without_solving(tmp_path,subject,number_of_cases):
    base=ROOT/'Experiments/scorpion_20261022/derived/exploratory/fem_smart_20261026'/subject
    if not (base/'electrical/forward.local.json').exists():pytest.skip('Local baseline unavailable')
    before=module.sha(base/'electrical/forward.local.json')
    plan=module.prepare(base,'scan',tmp_path/subject)
    assert len(plan['rows'])==20 and len(plan['case_ids'])==number_of_cases
    assert len(plan['operator_sources'])==1 and plan['operator_sources'][0]['layout']=='2x5'
    assert plan['subject_alias']==subject and plan['identity']['mesh_sha256']==module.sha(base/'materials/mesh.mat')
    assert module.sha(base/'electrical/forward.local.json')==before


def test_matlab_small_contract(tmp_path):
    if os.environ.get('SCORPION_RANGE_MATLAB_TEST')!='1':pytest.skip('Explicit small MATLAB check')
    matlab=shutil.which('matlab')
    if not matlab:pytest.skip('MATLAB unavailable')
    source=(ROOT/'MATLAB_TRKG4_real_subjects/src/run_scorpion_material_range.m').read_text(encoding='utf8')
    helpers=source[source.index('function validate_plan'):]
    base,_=fixture(tmp_path);module.prepare(base,'scan',tmp_path/'plan')
    script=r'''function material_range_smoke
assert(nargin('run_scorpion_material_range')==-1);
file=fullfile(fileparts(mfilename('fullpath')),'plan','plan.local.json');p=jsondecode(fileread(file));validate_plan(p);
bad=p;bad.rows(1).rho_ohm_m(1)=4;rejected=false;try,validate_plan(bad);catch,rejected=true;end;assert(rejected);
g=struct('id',{'TT_I0','TT_U0','TT_I1','TT_U1','A','M','N','B'});P=ports(g,'nik');
assert(isequal(P(:,1),[1;0;-1;0;0;0;0;0]));
Q=sparse(8*eye(8)-ones(8));parts=cell(1,5);for k=1:5,parts{k}=Q*k/15;end
rho=[2.5;7;20;1.35;25];C=[speye(8),-speye(8);-speye(8),speye(8)];
[T,J,qc]=trkg4_lateral_cem_response(bulk(parts,rho),C,parts,rho,P,[1 2 4 5],1);
fd=finite_difference(parts,rho,C,P,J);assert(fd.passed&&fd.performed);
a=struct('point_id','p','geometry','wholeheart_ct','case_id','L140_TT2x2','rule',4096,...
 'ttrkg_Z_ohm',10,'lateral_Z_ohm',20,'rho_ohm_m',rho','transfer_matrix_ohm',T);
q=quadrature_checks(a);assert(~q.evaluated&&~q.passed);
a(2)=a;a(2).rule=1024;a(2).ttrkg_Z_ohm=10.001;q=quadrature_checks(a);assert(q.evaluated&&q.passed);
a(3)=a(1);a(3).geometry='sphere_v100';a(3).transfer_matrix_ohm=T+eye(4);a(3).ttrkg_Z_ohm=11;
d=heart_differences(a,'heart');assert(d.delta_ttrkg_Z_ohm==1&&isequal(d.delta_transfer_ohm,eye(4)));
% End-to-end generic runner on eight synthetic algebraic nodes, under the same OS lock.
nodes_mm=zeros(8,3);save(p.mesh_mat,'nodes_mm');
scenarios=cell(numel(p.case_ids),1);
for k=1:numel(scenarios)
 scenarios{k}=struct('id',[p.case_ids{k} '_ttrkg'],'ports',g,'grouped_contact_matrix_SI',C);
end
result=struct('nodes_lps_m',nodes_mm/1000,'scenarios',{scenarios});save(p.cem_mat,'result');
p.identity.mesh_sha256=trkg4_file_sha256(p.mesh_mat);p.identity.cem_sha256=trkg4_file_sha256(p.cem_mat);
b=jsondecode(fileread(p.baseline_forward));b.identity.mesh_sha256=p.identity.mesh_sha256;b.identity.cem_sha256=p.identity.cem_sha256;
rewrite_fixture(p.baseline_forward,b);
identity=b.identity;identity.subject_alias='nik';
parts=repmat(parts,2,1);save(p.operator_sources.path,'parts','identity');
p.operator_sources.sha256=trkg4_file_sha256(p.operator_sources.path);
for k=1:numel(p.source_files),p.source_files(k).sha256=trkg4_file_sha256(p.source_files(k).path);end
rewrite_fixture(file,p);
eidors=getenv('EIDORS_STARTUP');if isfile(eidors),addpath(genpath(fileparts(eidors)));end
actual=run_scorpion_material_range('PlanFile',file);
assert(numel(actual.answers)==40&&actual.derivative_checks.passed&&~actual.mesh_convergence);
assert(sum([actual.q_checks.evaluated])==10&&sum(~[actual.q_checks.evaluated])==20);
fprintf('MATERIAL_RANGE_SMOKE_PASSED\n');
end
function rewrite_fixture(path,v)
fid=fopen(path,'w','n','UTF-8');fprintf(fid,'%s',jsonencode(v));fclose(fid);
end
'''
    (tmp_path/'material_range_smoke.m').write_text(script+helpers,encoding='utf8')
    prefs=tmp_path/'prefs';prefs.mkdir();quote=lambda p:str(p).replace("'","''")
    command=f"addpath('{quote(ROOT/'MATLAB_TRKG4_real_subjects/src')}');addpath('{quote(tmp_path)}');material_range_smoke;"
    result=subprocess.run([matlab,'-batch',command],env=dict(os.environ,MATLAB_PREFDIR=str(prefs)),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',timeout=150)
    assert result.returncode==0,result.stdout
    assert 'MATERIAL_RANGE_SMOKE_PASSED' in result.stdout
