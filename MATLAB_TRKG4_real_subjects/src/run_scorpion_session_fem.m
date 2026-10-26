function report=run_scorpion_session_fem(varargin)
% Own subject mesh/CEM and canonical EIDORS bulk/response kernels.
p=inputParser;addParameter(p,'Subject','');addParameter(p,'ModelDir','');
addParameter(p,'OutputDir','');addParameter(p,'PrepareOnly',false);
addParameter(p,'PreparedMaterialsFile','');addParameter(p,'CaseIds',{});
addParameter(p,'States',{'inhale','exhale'});addParameter(p,'Rules',[1024,4096]);
addParameter(p,'RunFiniteDifference',true);addParameter(p,'FiniteDifferenceCaseId','L140_TT2x2');
parse(p,varargin{:});a=p.Results;
assert(any(strcmp(a.Subject,{'georg','nik','yaroslav'})));
assert(islogical(a.PrepareOnly)&&isscalar(a.PrepareOnly));
assert(islogical(a.RunFiniteDifference)&&isscalar(a.RunFiniteDifference));
assert(~isempty(a.Rules)&&all(ismember(a.Rules,[1024,4096]))&&numel(unique(a.Rules))==numel(a.Rules));
if ischar(a.States)||isstring(a.States),a.States=cellstr(a.States);end
if ischar(a.CaseIds)||isstring(a.CaseIds),a.CaseIds=cellstr(a.CaseIds);end
assert(~isempty(a.States)&&all(ismember(a.States,{'inhale','exhale'}))&&numel(unique(a.States))==numel(a.States));
cfg=trkg4_config(a.Subject);cfg.frequency_hz=100000;
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
root=cfg.project_root;addpath(fullfile(root,'tools'));
base=char(a.ModelDir);out=char(a.OutputDir);
assert(isfolder(base)&&~isfolder(out),'Use an existing own model and a fresh electrical output');
mkdir(out);
try
 % Sparse direct factorizations share one resource slot on this host.
 lockdir=fullfile(root,'output','exploratory','scorpion_session_runtime_20261026');
 if ~isfolder(lockdir),mkdir(lockdir);end
 write_json(fullfile(out,'resource_wait.local.json'),struct('subject_alias',a.Subject,'state','waiting_for_sparse_solver_slot'));
 lockfile=javaObject('java.io.RandomAccessFile',fullfile(lockdir,'sparse_solver.lock'),'rw');
 channel=lockfile.getChannel();lease=channel.lock();
 resource_guard=onCleanup(@()release_resource(lease,channel,lockfile));
 write_json(fullfile(out,'resource_acquired.local.json'),struct('subject_alias',a.Subject,'state','resource_slot_acquired'));
 input=read_json(fullfile(base,'electrical_inputs.local.json'));
 assert(strcmp(input.subject_alias,a.Subject),'Wrong subject observations');
 assert(strcmp(trkg4_file_sha256(input.summary_source.path),input.summary_source.sha256));
 if isfield(input,'source_files'),verify_products(input.source_files);end
 ref_file=fullfile(base,'reference_materials.local.json');ref=read_json(ref_file);
 assert(strcmp(trkg4_file_sha256(ref_file),input.reference_materials_sha256)&&ref.frequency_hz==100000);
 assert(isequal(ref.material_order(:),{'soft_background';'lung';'bone';'heart';'fat'}));
 assert(isequal(size(ref.inhale_rho_ohm_m),size(ref.exhale_rho_ohm_m))&&numel(ref.inhale_rho_ohm_m)==5);
 assert(all(isfinite([ref.inhale_rho_ohm_m;ref.exhale_rho_ohm_m]),'all')&&all([ref.inhale_rho_ohm_m;ref.exhale_rho_ohm_m]>0,'all'));
 assert(ref.inhale_rho_ohm_m(3)==ref.exhale_rho_ohm_m(3),'Bone must remain fixed');
 if isfield(ref,'registry'),verify_products(ref.registry);end
 manifest=read_json(fullfile(base,'materials','preparation.local.json'));
 assert(strcmp(manifest.subject_alias,a.Subject)&&strcmp(manifest.stage,'materials'));
 verify_products(manifest.product_files);
 verify_source_map(fullfile(base,'materials','preparation.local.json'));
 fprintf('SCORPION %s: finite CEM contact preparation\n',a.Subject);
 cemdir=fullfile(base,'cem');
 if ~isfolder(cemdir)
  run_pipeline('Profile','ct_fem_contacts_prepare','InputFile',fullfile(base,'cem_contacts_input','cem_contacts_input.local.json'),'OutputDir',cemdir);
 end
 cm=read_json(fullfile(cemdir,'cem_contacts_manifest.local.json'));
 assert(strcmp(cm.subject_alias,a.Subject)&&cm.CEM_contacts_QC_passed);verify_products(cm.products);verify_products(cm.source_files);
 c=load(fullfile(cemdir,'cem_contacts.mat'),'result');c=c.result;
 M=load(fullfile(base,'materials','mesh.mat'));
 mm=eidors_obj('fwd_model',[a.Subject ' Scorpion own CT model']);
 mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
 mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
 mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;
 mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;clear M;
 assert(isequal(size(mm.nodes),size(c.nodes_lps_m))&&max(abs(mm.nodes(:)/1000-c.nodes_lps_m(:)))<1e-12);
 rules=[1024,4096];active=[1,2,4,5];
 identity=struct('subject_alias',a.Subject,'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
  'mesh_sha256',trkg4_file_sha256(fullfile(base,'materials','mesh.mat')),...
  'cem_sha256',trkg4_file_sha256(fullfile(cemdir,'cem_contacts.mat')),...
  'reference_sha256',trkg4_file_sha256(ref_file),'inputs_sha256',trkg4_file_sha256(fullfile(base,'electrical_inputs.local.json')),...
  'response_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
  'field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields')),...
  'scale_kernel_sha256',trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')),...
  'materials_manifest_sha256',trkg4_file_sha256(fullfile(base,'materials','preparation.local.json')),...
  'cem_manifest_sha256',trkg4_file_sha256(fullfile(cemdir,'cem_contacts_manifest.local.json')),...
  'rules',rules,'active_material_indices',active,'material_order',{ref.material_order},...
  'bulk_matrix_units','S for sigma=1 S/m','mm_nodes_units','mm',...
  'mesh_convergence',false,'individual_fitted',false);
 if isfield(ref,'registry_sha256'),identity.registry_sha256=ref.registry_sha256;end
 prepared_path=char(a.PreparedMaterialsFile);
 if isempty(prepared_path)
 F=system_mat_fields(trkg4_scale_fmdl_to_si(mm,cfg));parts=cell(2,5);
 for rule=1:2
  f=load(fullfile(base,'materials',sprintf('fractions_q%d.mat',rules(rule))));
  assert(isequal(size(f.fractions),[size(mm.elems,1),5])&&all(isfinite(f.fractions),'all')&&...
    all(f.fractions>=0,'all')&&all(f.fractions<=1,'all')&&all(sum(f.fractions,2)==1));
  for k=1:5
   w=repelem(f.fractions(:,k),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
   parts{rule,k}=B'*B;fprintf('SCORPION %s: material q%d component%d\n',a.Subject,rules(rule),k);
  end
 end
 clear F B w f;
 prepared_path=fullfile(out,'prepared_materials.mat');
 reference_materials=ref;port_order={'TT_drive','TT_measure','lateral_drive','lateral_measure'};
 save(prepared_path,'parts','mm','identity','rules','active','reference_materials','port_order','-v7.3');
 else
  prepared=load(prepared_path,'parts','mm','identity');
  for key={'subject_alias','mesh_sha256','cem_sha256','reference_sha256','inputs_sha256',...
      'response_kernel_sha256','field_kernel_sha256','scale_kernel_sha256','materials_manifest_sha256','cem_manifest_sha256'}
   assert(isequal(prepared.identity.(key{1}),identity.(key{1})),['Prepared identity mismatch: ' key{1}]);
  end
  assert(isequal(size(prepared.parts),[2,5])&&isequal(prepared.mm.nodes,mm.nodes)&&isequal(prepared.mm.elems,mm.elems));
  parts=prepared.parts;clear prepared;
 end
 identity.prepared_materials_path=prepared_path;
 if a.PrepareOnly
  verify_products(manifest.product_files);verify_source_map(fullfile(base,'materials','preparation.local.json'));
  verify_products(cm.products);verify_products(cm.source_files);
  if isfield(input,'source_files'),verify_products(input.source_files);end
  assert(strcmp(trkg4_file_sha256(ref_file),identity.reference_sha256));
  report=struct('schema','scorpion_prepared_materials_v2','identity',identity,'FEM_was_run',false,...
   'mesh_convergence',false,'individual_fitted',false);
  write_json(fullfile(out,'prepared.local.json'),report);return;
 end
 states=a.States;answers=struct([]);checks=struct([]);
 derivative_checks=struct('performed',false,'passed',false,'reason','not_requested_or_check_case_not_in_limited_run');
 scenarios=c.scenarios;ids=cellfun(@(s)s.id,scenarios,'UniformOutput',false);
 cases=cellfun(@(s)regexprep(s,'_(ttrkg|lateral)$',''),ids,'UniformOutput',false);
 [unique_cases,first]=unique(cases,'stable');
 assert(all(ismember(a.CaseIds,unique_cases)),'Requested CEM case is absent');
 if ~isempty(a.CaseIds),keep=ismember(unique_cases,a.CaseIds);unique_cases=unique_cases(keep);first=first(keep);end
 primary=find(strcmp(unique_cases,a.FiniteDifferenceCaseId));
 order=[primary(:)',setdiff(1:numel(first),primary,'stable')];
 unique_cases=unique_cases(order);first=first(order);
 % Paired role scenarios must describe the same physical CEM operator.
 for i=1:numel(first)
  same=find(strcmp(cases,unique_cases{i}));s=scenarios{first(i)};
  for j=same(:)'
   assert(isequal(s.ports,scenarios{j}.ports)&&isequal(s.grouped_contact_matrix_SI,scenarios{j}.grouped_contact_matrix_SI),...
    'Role scenarios do not share one physical CEM montage');
  end
 end
 for state=1:numel(states)
  rho=ref.([states{state} '_rho_ohm_m'])(:);
  % Fine primary inhale/FD is available early; every full-run baseline is solved once.
  for rule=[2,1]
   if ~ismember(rules(rule),a.Rules),continue;end
   K=sparse(size(mm.nodes,1),size(mm.nodes,1));for k=1:5,K=K+parts{rule,k}/rho(k);end
   for index=1:numel(first)
    s=scenarios{first(index)};ports=make_ports(s.ports,a.Subject);
    fprintf('SCORPION %s: FEM q%d %s %s\n',a.Subject,rules(rule),states{state},unique_cases{index});
    [T,J,qc]=trkg4_lateral_cem_response(K,s.grouped_contact_matrix_SI,parts(rule,:),rho,ports,active,1);
    entry=struct('state',states{state},'rule',rules(rule),'case_id',unique_cases{index},...
      'rho_ohm_m',rho','transfer_matrix_ohm',T,'jacobian_logrho',J,...
      'active_material_indices',active,'port_order',{{'TT_drive','TT_measure','lateral_drive','lateral_measure'}},...
      'excitation_scenario','own_drive','geometry_breathing_motion_modelled',false,...
      'exhale_on_fixed_CT_is_conditional',strcmp(states{state},'exhale'),...
      'ttrkg_Z_ohm',T(2,1),'lateral_Z_ohm',T(4,3),...
      'ttrkg_from_lateral_current_ohm',T(2,3),'lateral_from_ttrkg_current_ohm',T(4,1));
    answers=append(answers,entry);qc.case_id=unique_cases{index};qc.rule=rules(rule);qc.state=states{state};checks=append(checks,qc);
    write_json(fullfile(out,sprintf('answer_%s_q%d_%s.local.json',unique_cases{index},rules(rule),states{state})),entry);
    if a.RunFiniteDifference&&strcmp(states{state},'inhale')&&rule==2&&strcmp(unique_cases{index},a.FiniteDifferenceCaseId)
     errors=zeros(1,numel(active));transfer_errors=zeros(4,numel(active));h=1e-3;
     transfer_indices=sub2ind([4,4],[2,4,2,4],[1,3,3,1]);
     for j=1:numel(active)
      plus=rho;minus=rho;plus(active(j))=plus(active(j))*exp(h);minus(active(j))=minus(active(j))*exp(-h);
      kp=sparse(size(K,1),size(K,2));km=kp;
      for k=1:5,kp=kp+parts{rule,k}/plus(k);km=km+parts{rule,k}/minus(k);end
      tp=trkg4_lateral_cem_response(kp,s.grouped_contact_matrix_SI,parts(rule,:),plus,ports,[],1);
      tm=trkg4_lateral_cem_response(km,s.grouped_contact_matrix_SI,parts(rule,:),minus,ports,[],1);
      fd=(tp-tm)/(2*h);analytic=J(:,:,j);
      errors(j)=norm(fd-analytic,'fro')/max(norm(fd,'fro'),realmin);
      transfer_errors(:,j)=abs(fd(transfer_indices)-analytic(transfer_indices))'./...
        (1e-6+1e-4*max(abs(fd(transfer_indices)),abs(analytic(transfer_indices))))';
     end
     derivative_checks=struct('performed',true,'case_id',unique_cases{index},'state',states{state},'rule',rules(rule),'log_step',h,...
       'active_material_indices',active,'relative_errors',errors,'tolerance',1e-4,...
       'transfer_absolute_tolerance_ohm',1e-6,'transfer_scaled_errors',transfer_errors,...
       'passed',all(errors<1e-4)&&all(transfer_errors<=1,'all'));
     write_json(fullfile(out,'derivative_checks.local.json'),derivative_checks);
     assert(derivative_checks.passed,'Own geometry derivative check failed');
     checkpoint=struct('schema','scorpion_own_CT_forward_v2','identity',identity,'subject_alias',a.Subject,...
       'frequency_hz',100000,'reference_materials',ref,'answers',answers,'system_checks',checks,...
       'derivative_checks',derivative_checks,'active_material_indices',active,...
       'prepared_materials_path',prepared_path,'FEM_was_run',true,'limited_run',true,...
       'own_drive_is_working_assumption',true,'source_relative_phase','unknown',...
       'mesh_convergence',false,'mesh_convergence_proven',false,'individual_fitted',false,...
       'fitted_gain',false,'quantitative_result_accepted',false,'clinical_validation',false);
     write_json(fullfile(out,'primary_forward.local.json'),checkpoint);
    end
   end
  end
 end
 comparisons=compare_observations(input.observations,answers,rules);
 breathing_differences=compare_breathing(comparisons);
 verify_products(manifest.product_files);verify_source_map(fullfile(base,'materials','preparation.local.json'));
 verify_products(cm.products);verify_products(cm.source_files);
 assert(strcmp(trkg4_file_sha256(input.summary_source.path),input.summary_source.sha256));
 if isfield(input,'source_files'),verify_products(input.source_files);end
 if isfield(ref,'registry'),verify_products(ref.registry);end
 assert(strcmp(trkg4_file_sha256(ref_file),identity.reference_sha256));
 assert(strcmp(trkg4_file_sha256([mfilename('fullpath') '.m']),identity.worker_sha256));
 assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),identity.response_kernel_sha256));
 assert(strcmp(trkg4_file_sha256(which('system_mat_fields')),identity.field_kernel_sha256));
 assert(strcmp(trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')),identity.scale_kernel_sha256));
 compared=comparisons(strcmp({comparisons.status},'predicted'));
 report=struct('schema','scorpion_own_CT_forward_v2','identity',identity,'subject_alias',a.Subject,...
  'frequency_hz',100000,'current_per_source_A',.007,'rules',rules,'reference_materials',ref,...
  'answers',answers,'comparison',comparisons,'breathing_differences',breathing_differences,...
  'observations',input.observations,'system_checks',checks,'derivative_checks',derivative_checks,...
  'active_material_indices',active,'prepared_materials_path',prepared_path,...
  'FEM_was_run',true,'own_drive_is_working_assumption',true,'source_relative_phase','unknown',...
  'integration_passed',~isempty(compared)&&all([compared.integration_passed]),'mesh_convergence_proven',false,...
  'mesh_convergence',false,'individual_fitted',false,'fitted_gain',false,...
  'integration_is_global_mesh_convergence',false,'coverage_complete',numel(comparisons)==numel(input.observations),...
  'limited_run',~isempty(a.CaseIds)||numel(a.Rules)<2||numel(a.States)<2,...
  'quantitative_result_accepted',false,'tissue_parameters_fitted',false,'clinical_validation',false);
 write_json(fullfile(out,'forward.local.json'),report);
catch exception
 write_json(fullfile(out,'failed.local.json'),struct('identifier',exception.identifier,'message',exception.message,...
  'FEM_forward_completed',false,'subject_alias',a.Subject));rethrow(exception);
end
end

function P=make_ports(groups,subject)
if strcmp(subject,'georg')
 pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else
 pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};
end
names={groups.id};assert(numel(unique(names))==numel(names),'Duplicate group id');
P=zeros(numel(groups),4);
for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end

function rows=compare_observations(observations,answers,rules)
rows=struct([]);
for i=1:numel(observations)
 o=observations(i);
 row=struct('record_id',o.record_id,'case_id',o.case_id,'state',o.state,'role',o.role,...
  'observed_Z_ohm',o.Z_ohm,'status','excluded','exclusion_reason','',...
  'coarse_Z_ohm',[],'fine_Z_ohm',[],'difference_ohm',[],'integration_tolerance_ohm',[],...
  'integration_evaluated',false,'integration_passed',false,'residual_ohm',[],...
  'prediction_rule',[],'annotation_status',o.annotation_status,'source_sha256',o.source_sha256);
 if isfield(o,'status')&&strcmp(o.status,'excluded')
  row.exclusion_reason=o.exclusion_reason;
 elseif ~any(strcmp(o.role,{'lateral','ttrkg'}))
  row.exclusion_reason='unsupported_physical_role';
 else
  ix=find(strcmp({answers.case_id},o.case_id)&strcmp({answers.state},o.state));
  if isempty(ix)
   row.exclusion_reason='case_or_state_absent_from_prepared_CEM_or_limited_run';
  else
   row.status='predicted';z=nan(1,2);
   for j=ix(:)',v=answers(j);z(rules==v.rule)=v.([o.role '_Z_ohm']);end
   if isfinite(z(1)),row.coarse_Z_ohm=z(1);end
   if isfinite(z(2)),row.fine_Z_ohm=z(2);end
   available=find(isfinite(z),1,'last');row.prediction_rule=rules(available);
   row.residual_ohm=z(available)-o.Z_ohm;
   if all(isfinite(z))
    row.difference_ohm=abs(diff(z));row.integration_tolerance_ohm=.02+.001*abs(z(2));
    row.integration_evaluated=true;row.integration_passed=row.difference_ohm<=row.integration_tolerance_ohm;
   end
  end
 end
 rows=append(rows,row);
end
end

function differences=compare_breathing(rows)
differences=struct([]);
for i=1:numel(rows)
 r=rows(i);
 if ~strcmp(r.state,'inhale')||~strcmp(r.status,'predicted'),continue;end
 j=find(strcmp({rows.record_id},r.record_id)&strcmp({rows.case_id},r.case_id)&...
  strcmp({rows.role},r.role)&strcmp({rows.state},'exhale')&strcmp({rows.status},'predicted'));
 if isempty(j),continue;end
 assert(numel(j)==1);e=rows(j);
 if isempty(r.fine_Z_ohm)||isempty(e.fine_Z_ohm),continue;end
 measured=e.observed_Z_ohm-r.observed_Z_ohm;predicted=e.fine_Z_ohm-r.fine_Z_ohm;
 differences=append(differences,struct('record_id',r.record_id,'case_id',r.case_id,'role',r.role,...
  'definition','exhale_minus_inhale','rule',4096,'observed_delta_Z_ohm',measured,...
  'predicted_delta_Z_ohm',predicted,'delta_residual_ohm',predicted-measured,...
  'fixed_inspiration_CT_conditional_scenario',true,'manual_annotation_accepted',false));
end
end
function value=read_json(path),value=jsondecode(fileread(path));end
function verify_products(items)
for k=1:numel(items),assert(strcmp(trkg4_file_sha256(items(k).path),items(k).sha256),'Input product changed');end
end
function verify_source_map(path)
% jsondecode sanitizes path-shaped object keys. Decode each quoted path separately.
quoted='"(?:\\.|[^"\\])*"';
pattern=['"sources"\s*:\s*\{\s*((?:' quoted '\s*:\s*"[0-9a-fA-F]{64}"\s*,?\s*)*)\}'];
object=regexp(fileread(path),pattern,'tokens');assert(numel(object)==1,'Invalid source hash map');
pairs=regexp(object{1}{1},['(' quoted ')\s*:\s*"([0-9a-fA-F]{64})"'],'tokens');
assert(~isempty(pairs),'Empty preparation source map');
for k=1:numel(pairs)
 source=jsondecode(pairs{k}{1});assert(strcmpi(trkg4_file_sha256(source),pairs{k}{2}),'Preparation source changed');
end
end
function result=append(result,value)
if isempty(result),result=value;else,result(end+1)=value;end
end
function write_json(path,value)
assert(~isfile(path),'Use a fresh output file');fid=fopen(path,'w','n','UTF-8');assert(fid>0);
guard=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
function release_resource(lease,channel,lockfile)
lease.release();channel.close();lockfile.close();
end
