"""Проверки происхождения наблюдений, исключений и контракта baseline FEM."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('scorpion_baseline_inputs', ROOT /
    'MATLAB_TRKG4_real_subjects/tools/scorpion_session_electrical_inputs.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')


def session_fixture(tmp_path):
    experiment = tmp_path / 'experiment'
    raw = experiment / 'raw/record.txt'; raw.parent.mkdir(parents=True); raw.write_text('record', encoding='utf8')
    record = dict(record_id='S-001', subject='nik', source_relative_path='raw/record.txt',
        source_sha256=adapter.core.sha(raw), context=dict(record_kind='both', lateral_size_mm=140, ttrkg_variant='2x2'),
        active_roles=['ttrkg', 'lateral'], physical_channel_roles=dict(ttrkg=2, lateral=1),
        roles_evidence='Роли заданы протоколом', ecg_qc={'1': {'events_are_confirmed_R_waves': False}})
    ann = experiment / 'annotations/session_1140/candidate_annotations.local.json'
    write(ann, dict(records=[dict(record_id='S-001', source_sha256=record['source_sha256'],
        status='pending_manual_review', boundary_uncertainty_ranges_s=[[1, 2], [3, 4], [5, 6]])]))
    summary = dict(records=[record], levels=[dict(record_id='S-001', role='ttrkg', state='inhale',
        baseline_ohm=17.25, start_s=2, end_s=3)], annotations_sha256=adapter.core.sha(ann),
        sampling=dict(working_fs_hz=1000), status='exploratory_hypothesis_not_validated')
    path = experiment / 'derived/exploratory/session_3340/summary.local.json'; write(path, summary)
    return experiment, summary, path


def test_roles_are_taken_from_protocol_and_missing_pairs_are_explicit(tmp_path):
    _, summary, path = session_fixture(tmp_path)
    rows, sources, _ = adapter.session_observations('nik', summary, path)
    assert len(rows) == 4
    measured = next(r for r in rows if r['status'] == 'eligible_conditional')
    assert measured['channel'] == 2 and measured['Z_ohm'] == 17.25
    assert measured['case_id'] == 'L140_TT2x2'
    assert measured['annotation_status'] == 'pending_manual_review'
    assert not measured['manual_annotation_accepted']
    assert all(r['exclusion_reason'] for r in rows if r['status'] == 'excluded')
    assert all(s['sha256'] == adapter.core.sha(s['path']) for s in sources)


@pytest.mark.parametrize('kind', ['ttrkg_only', 'lateral_only', 'reciprocity', 'switching'])
def test_controls_do_not_silently_use_both_sources_CEM(tmp_path, kind):
    _, summary, path = session_fixture(tmp_path)
    summary['records'][0]['context']['record_kind'] = kind
    rows, _, _ = adapter.session_observations('nik', summary, path)
    assert all(r['status'] == 'excluded' and r['exclusion_reason'] for r in rows)


@pytest.mark.parametrize('changed', ['raw', 'annotation'])
def test_changed_sources_are_rejected(tmp_path, changed):
    experiment, summary, path = session_fixture(tmp_path)
    target = experiment / ('raw/record.txt' if changed == 'raw' else
        'annotations/session_1140/candidate_annotations.local.json')
    target.write_text('{}', encoding='utf8')
    with pytest.raises(ValueError):
        adapter.session_observations('nik', summary, path)


def test_materials_are_exact_frequency_entries_and_bone_is_fixed():
    registry = ROOT / 'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    ref = adapter.reference_materials(registry); source = adapter.core.read(registry)
    assert ref['inhale_rho_ohm_m'][1] == source['tissues']['Lung (Inflated)']['gabriel_at_frequency']['100000']['rho_ohm_m']
    assert ref['exhale_rho_ohm_m'][1] == source['tissues']['Lung (Deflated)']['gabriel_at_frequency']['100000']['rho_ohm_m']
    assert ref['inhale_rho_ohm_m'][2] == ref['exhale_rho_ohm_m'][2]
    assert not ref['individual_fitted'] and not ref['mesh_convergence']
    assert not ref['geometric_breathing_motion_modelled']
    assert ref['registry_sha256'] == adapter.core.sha(registry)


def test_build_is_immutable_and_does_not_create_partial_reference_on_bad_input(tmp_path):
    _, summary, path = session_fixture(tmp_path); model = tmp_path / 'model'; model.mkdir()
    bad = dict(summary, annotations_sha256='0' * 64); write(path, bad)
    with pytest.raises(ValueError): adapter.build('nik', model, path)
    assert not list(model.iterdir())
    write(path, summary)
    result = adapter.build('nik', model, path)
    assert result['subject_alias'] == 'nik' and not result['fitted_gain']
    assert not result['individual_fitted'] and not result['mesh_convergence']
    digest = adapter.core.sha(model / 'electrical_inputs.local.json')
    with pytest.raises(FileExistsError): adapter.build('nik', model, path)
    assert digest == adapter.core.sha(model / 'electrical_inputs.local.json')


@pytest.mark.parametrize('subject,expected', [('nik', 16), ('yaroslav', 16), ('georg', 28)])
def test_existing_observation_sources_read_only(subject, expected):
    path = ROOT / ('Experiments/scorpion_v2_georg/derived/scorpion_review/summary.local.json' if subject == 'georg'
        else 'Experiments/scorpion_20261022/derived/exploratory/session_3340/summary.local.json')
    if not path.exists(): pytest.skip('Локальные наблюдения отсутствуют')
    summary = adapter.core.read(path)
    if subject == 'georg': rows, sources, _ = adapter.georg_observations(summary, path)
    else: rows, sources, _ = adapter.session_observations(subject, summary, path)
    assert sum(r['status'] == 'eligible_conditional' for r in rows) == expected
    assert len({(r['record_id'], r['role'], r['channel'], r['state']) for r in rows}) == len(rows)
    assert all(r['status'] == 'eligible_conditional' or r['exclusion_reason'] for r in rows)
    assert all(not r['manual_annotation_accepted'] for r in rows)
    assert len(sources) > 1
    if subject == 'georg':
        late = [r for r in rows if r['record_id'][-3:] in ('008', '009', '010') and r['status'] == 'eligible_conditional']
        assert late and all(r['case_id'].endswith('TT1x1') for r in late)
        assert all(r['role_evidence']['status'] == 'late_block_continuity_protocol_assumption' for r in late)
        assert any('large_high_frequency_ecg_interference_channel_1' in r['source_qc']['flags'] for r in late)


def test_matlab_lightweight_contract_and_canonical_derivatives(tmp_path):
    if os.environ.get('SCORPION_BASELINE_MATLAB_TEST') != '1':
        pytest.skip('Лёгкая MATLAB-проверка запускается отдельным явным выбором')
    matlab = shutil.which('matlab')
    if not matlab: pytest.skip('MATLAB отсутствует')
    source = (ROOT / 'MATLAB_TRKG4_real_subjects/src/run_scorpion_session_fem.m').read_text(encoding='utf8')
    helpers = source[source.index('function P=make_ports'):]
    # Synthetic algebra only; use the unchanged canonical response kernel.
    script = r"""function scorpion_baseline_smoke
assert(nargin('run_scorpion_session_fem')==-1);
source=which('run_scorpion_session_fem');digest=trkg4_file_sha256(source);
mapfile=fullfile(fileparts(mfilename('fullpath')),'source_map.local.json');
fid=fopen(mapfile,'w','n','UTF-8');fprintf(fid,'{"sources":{%s:"%s"}}',jsonencode(source),digest);fclose(fid);
verify_source_map(mapfile);
fid=fopen(mapfile,'w','n','UTF-8');fprintf(fid,'{"sources":{%s:"%s"}}',jsonencode(source),repmat('0',1,64));fclose(fid);
rejected=false;try,verify_source_map(mapfile);catch,rejected=true;end;assert(rejected);
g=struct('id',{'TT_Ileft','TT_Uleft','TT_Uright','TT_Iright','A','M','N','B'});
P=make_ports(g,'georg');assert(isequal(P(:,1),[1;0;0;-1;0;0;0;0]));
assert(isequal(P(:,2),[0;1;-1;0;0;0;0;0]));assert(all(sum(P)==0));
g(1).id='TT_I0';g(2).id='TT_U0';g(3).id='TT_U1';g(4).id='TT_I1';
assert(isequal(P,make_ports(g,'nik')));
Q=sparse(8*eye(8)-ones(8));parts=cell(1,5);
for k=1:5,parts{k}=Q*k/15;end
rho=[3;15;20;2;25];K=sparse(8,8);
for k=1:5,K=K+parts{k}/rho(k);end
C=[speye(8),-speye(8);-speye(8),speye(8)];active=[1 2 4 5];
[T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rho,P,active,1);
assert(qc.system_relative_residual<1e-7&&qc.reciprocity_ohm<1e-7);
for k=1:4
 rp=rho;rm=rho;h=1e-3;rp(active(k))=rp(active(k))*exp(h);rm(active(k))=rm(active(k))*exp(-h);
 kp=sparse(8,8);km=kp;
 for j=1:5,kp=kp+parts{j}/rp(j);km=km+parts{j}/rm(j);end
 tp=trkg4_lateral_cem_response(kp,C,parts,rp,P,[],1);tm=trkg4_lateral_cem_response(km,C,parts,rm,P,[],1);
 fd=(tp-tm)/(2*h);assert(norm(fd-J(:,:,k),'fro')/norm(fd,'fro')<1e-4);
end
o=struct('record_id','r','case_id','L140_TT2x2','state','inhale','role','ttrkg',...
 'Z_ohm',10,'status','eligible_conditional','exclusion_reason','','annotation_status','pending_manual_review','source_sha256','a');
a=struct('case_id',o.case_id,'state','inhale','rule',1024,'ttrkg_Z_ohm',11);
a(2)=a;a(2).rule=4096;a(2).ttrkg_Z_ohm=11.01;
r=compare_observations(o,a,[1024 4096]);assert(strcmp(r.status,'predicted')&&r.integration_passed);
assert(abs(r.residual_ohm-1.01)<1e-12);
r0=compare_observations(o,a(1),[1024 4096]);assert(~r0.integration_evaluated&&~r0.integration_passed&&isempty(r0.fine_Z_ohm));
o(2)=o;o(2).state='exhale';o(2).Z_ohm=9;
a(3)=a(1);a(3).state='exhale';a(3).ttrkg_Z_ohm=10;
a(4)=a(2);a(4).state='exhale';a(4).ttrkg_Z_ohm=10.01;
r=compare_observations(o,a,[1024 4096]);d=compare_breathing(r);
assert(d.observed_delta_Z_ohm==-1&&abs(d.predicted_delta_Z_ohm+1)<1e-12);
o(1).status='excluded';o(1).exclusion_reason='control';r=compare_observations(o,a,[1024 4096]);
assert(strcmp(r(1).exclusion_reason,'control'));
o(1).status='eligible_conditional';o(1).case_id='absent';r=compare_observations(o,a,[1024 4096]);
assert(strcmp(r(1).status,'excluded')&&~isempty(r(1).exclusion_reason));
fprintf('SCORPION_BASELINE_SMOKE_PASSED\n');
end
"""
    script_file = tmp_path / 'scorpion_baseline_smoke.m'
    script_file.write_text(script + helpers, encoding='utf8')
    quote = lambda p: str(p).replace("'", "''")
    command = f"addpath('{quote(ROOT / 'MATLAB_TRKG4_real_subjects/src')}');addpath('{quote(tmp_path)}');scorpion_baseline_smoke;"
    preferences = tmp_path / 'matlab_preferences'; preferences.mkdir()
    environment = dict(os.environ, MATLAB_PREFDIR=str(preferences))
    run = subprocess.run([matlab, '-batch', command], text=True, encoding='utf8', errors='replace',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150, env=environment)
    assert run.returncode == 0, run.stdout
    assert 'SCORPION_BASELINE_SMOKE_PASSED' in run.stdout
