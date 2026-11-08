function report=run_scorpion_apparent_resistivity(varargin)
% Uniform-rho inversion on each participant's own prepared CT/CEM operator.
% Contact impedance stays fixed: Z(rho) is NOT assumed exactly rho*Z(1).
p=inputParser;addParameter(p,'InputFile','');addParameter(p,'OutputDir','');addParameter(p,'PulsePlan','');
parse(p,varargin{:});a=p.Results;
plan=jsondecode(fileread(a.InputFile));
assert(strcmp(plan.schema,'scorpion_uniform_rho_input_v1'));
pulse_mode=~isempty(a.PulsePlan);
if pulse_mode
    pulse=jsondecode(fileread(a.PulsePlan));assert(strcmp(pulse.schema,'scorpion_uniform_pulse_v1'));
    assert(strcmp(pulse.input_sha256,trkg4_file_sha256(a.InputFile)));
    assert(strcmp(pulse.baseline_sha256,trkg4_file_sha256(pulse.baseline_path)));
    baseline=jsondecode(fileread(pulse.baseline_path));
    assert(strcmp(baseline.input_sha256,pulse.input_sha256));
    assert(strcmp(baseline.response_kernel_sha256,trkg4_file_sha256(which('trkg4_lateral_cem_response'))));
    assert(numel(pulse.records)==numel(baseline.answers));
end
out=char(a.OutputDir);assert(~isfolder(out),'Use a fresh FEM output directory');mkdir(out);
maxNumCompThreads(6);clock=tic;answers=struct([]);checks=struct([]);
for index=1:numel(plan.subjects)
    subject=plan.subjects(index);base=subject.model_directory;
    fprintf('UNIFORM_RHO subject=%s loading own prepared FEM\n',subject.subject_alias);
    prepared=fullfile(base,'electrical','prepared_materials.mat');
    cemfile=fullfile(base,'cem','cem_contacts.mat');meshfile=fullfile(base,'materials','mesh.mat');
    assert(strcmp(trkg4_file_sha256(prepared),subject.prepared_sha256));
    assert(strcmp(trkg4_file_sha256(cemfile),subject.cem_sha256));
    assert(strcmp(trkg4_file_sha256(meshfile),subject.mesh_sha256));
    S=load(prepared,'parts','identity','mm','rules');C=load(cemfile,'result');C=C.result;
    assert(strcmp(S.identity.subject_alias,subject.subject_alias));
    assert(strcmp(S.identity.cem_sha256,subject.cem_sha256));
    assert(strcmp(S.identity.mesh_sha256,subject.mesh_sha256));
    assert(strcmp(S.identity.response_kernel_sha256,trkg4_file_sha256(which('trkg4_lateral_cem_response'))));
    assert(strcmp(S.identity.bulk_matrix_units,'S for sigma=1 S/m'));
    assert(isequal(size(S.parts),[2,5])&&isequal(S.rules(:),[1024;4096]));
    assert(isequal(size(S.mm.nodes),size(C.nodes_lps_m)));
    assert(max(abs(S.mm.nodes(:)/1000-C.nodes_lps_m(:)))<1e-12);
    n=size(S.mm.nodes,1);coarse=sparse(n,n);bulk=sparse(n,n);
    for k=1:5,coarse=coarse+S.parts{1,k};bulk=bulk+S.parts{2,k};end
    partition_error=norm(bulk-coarse,'fro')/norm(bulk,'fro');
    symmetry_error=norm(bulk-bulk','fro')/norm(bulk,'fro');
    assert(partition_error<1e-12&&symmetry_error<1e-12);
    mesh_nodes=n;mesh_elements=size(S.mm.elems,1);clear S coarse;
    ids=cellfun(@(s)s.id,C.scenarios,'UniformOutput',false);
    for oi=1:numel(subject.observations)
        o=subject.observations(oi);start=tic;
        si=find(strcmp(ids,[o.case_id '_lateral']));assert(isscalar(si));s=C.scenarios{si};
        groups={s.ports.id};ports=zeros(numel(groups),2);
        [found,ix]=ismember({'A','B','M','N'},groups);assert(all(found));
        ports(ix(1:2),1)=[1;-1];ports(ix(3:4),2)=[1;-1];
        if pulse_mode
            entry=pulse.records(strcmp({pulse.records.record_id},o.record_id));assert(isscalar(entry));
            b=baseline.answers(strcmp({baseline.answers.record_id},o.record_id));assert(isscalar(b));
            assert(strcmp(b.subject_alias,subject.subject_alias)&&strcmp(b.case_id,o.case_id));
            assert(strcmp(entry.source_sha256,o.source_sha256)&&strcmp(entry.case_id,o.case_id));
            assert(b.observed_Z_ohm==o.Z_ohm&&b.rho_apparent_ohm_m==entry.rho0_ohm_m);
            assert(b.dZ_dlogrho_ohm==entry.dZ_dlogrho_ohm);
            fprintf('SCORPION PULSE %s max|delta Z|=%.8g Ohm\n',o.record_id,entry.max_abs_delta_Z_ohm);
            answer=trkg4_uniform_pulse_check(bulk,s.grouped_contact_matrix_SI,b,entry.max_abs_delta_Z_ohm,ports);
            answer.record_id=o.record_id;answer.subject_alias=subject.subject_alias;answer.case_id=o.case_id;
            answers=append(answers,answer);
            write_json(fullfile(out,[o.record_id '.local.json']),answer);
            assert(answer.passed,'Pulse linearisation failed: use nonlinear inversion');
            continue;
        end
        fprintf('UNIFORM_RHO %s %s unit response\n',subject.subject_alias,o.record_id);
        [T,~,unit_qc]=trkg4_lateral_cem_response(bulk,s.grouped_contact_matrix_SI,{bulk},1,ports,[],1);
        zunit=T(2,1);assert(zunit>0&&o.Z_ohm>0);
        initial=o.Z_ohm/zunit;rho=initial;history=struct([]);converged=false;
        tolerance=1e-5+1e-7*abs(o.Z_ohm);
        for iteration=1:6
            [T,J,qc]=trkg4_lateral_cem_response(bulk/rho,s.grouped_contact_matrix_SI,{bulk},rho,ports,1,1);
            z=T(2,1);slope=J(2,1,1);assert(isfinite(slope)&&slope>0);
            residual=z-o.Z_ohm;
            step=struct('rho_ohm_m',rho,'Z_ohm',z,'residual_ohm',residual,...
                'dZ_dlogrho_ohm',slope,'system_relative_residual',qc.system_relative_residual,...
                'reciprocity_ohm',qc.reciprocity_ohm);
            history=append(history,step);
            fprintf('UNIFORM_RHO %s %s rho=%.9g residual=%.6g Ohm\n',subject.subject_alias,o.record_id,rho,residual);
            if abs(residual)<=tolerance,converged=true;break;end
            rho=rho*exp(max(-1,min(1,-residual/slope)));
        end
        assert(converged,'Uniform-rho inversion did not meet the predeclared impedance tolerance');
        answer=struct('subject_alias',subject.subject_alias,'record_id',o.record_id,...
            'case_id',o.case_id,'size_mm',o.size_mm,'ttrkg_variant',o.ttrkg_variant,...
            'observed_Z_ohm',o.Z_ohm,'rho_apparent_ohm_m',rho,'Z_unit_rho_ohm',zunit,...
            'unit_rho_ohm_m',1,'linear_ratio_rho_ohm_m',initial,...
            'fixed_contact_correction_relative',(rho-initial)/initial,...
            'predicted_Z_ohm',z,'residual_ohm',residual,'tolerance_ohm',tolerance,...
            'dZ_dlogrho_ohm',slope,'iterations',{history},'unit_response_qc',unit_qc,...
            'elapsed_seconds',toc(start),'annotation_status',o.annotation_status,...
            'status','exploratory_hypothesis_not_validated','uniform_all_five_materials',true,...
            'contact_impedance_fixed',true,'individual_anatomy',true,...
            'mesh_convergence_proven',false,'electrode_positions_independently_measured',false);
        answers=append(answers,answer);
        write_json(fullfile(out,[o.record_id '.local.json']),answer);
    end
    checks=append(checks,struct('subject_alias',subject.subject_alias,'mesh_nodes',mesh_nodes,...
        'mesh_elements',mesh_elements,'partition_q1024_q4096_relative_error',partition_error,...
        'bulk_symmetry_relative_error',symmetry_error,'mesh_sha256',subject.mesh_sha256,...
        'cem_sha256',subject.cem_sha256,'prepared_sha256',subject.prepared_sha256));
    clear bulk C;
end
report=struct('schema','scorpion_uniform_rho_results_v1','answers',{answers},'subject_checks',{checks},...
    'input_sha256',trkg4_file_sha256(a.InputFile),...
    'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
    'response_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
    'elapsed_seconds',toc(clock),'matlab_version',version,'frequency_hz',100000,...
    'excitation','lateral_own_drive','quantitative_result_accepted',false,...
    'status','exploratory_hypothesis_not_validated');
if pulse_mode
    report.schema='scorpion_uniform_pulse_checks_v1';
    report.pulse_plan_sha256=trkg4_file_sha256(a.PulsePlan);
    report.baseline_sha256=pulse.baseline_sha256;
    report.pulse_check_kernel_sha256=trkg4_file_sha256(which('trkg4_uniform_pulse_check'));
    write_json(fullfile(out,'pulse_checks.local.json'),report);
    fprintf('SCORPION PULSE COMPLETE %d observations, %.1f seconds\n',numel(answers),toc(clock));return;
end
write_json(fullfile(out,'results.local.json'),report);
fprintf('UNIFORM_RHO COMPLETE %d observations, %.1f seconds\n',numel(answers),toc(clock));
end

function a=append(a,b)
if isempty(a),a=b;else,a(end+1)=b;end
end
function write_json(path,value)
assert(~isfile(path));f=fopen(path,'w','n','UTF-8');assert(f>0);guard=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value,PrettyPrint=true));
end
