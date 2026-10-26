function report=run_scorpion_session_fem(varargin)
% Own subject mesh/CEM and canonical EIDORS bulk/response kernels.
p=inputParser;addParameter(p,'Subject','');addParameter(p,'ModelDir','');
addParameter(p,'OutputDir','');parse(p,varargin{:});a=p.Results;
assert(any(strcmp(a.Subject,{'nik','yaroslav'})));
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
 ref_file=fullfile(base,'reference_materials.local.json');ref=read_json(ref_file);
 assert(strcmp(trkg4_file_sha256(ref_file),input.reference_materials_sha256)&&ref.frequency_hz==100000);
 manifest=read_json(fullfile(base,'materials','preparation.local.json'));
 assert(strcmp(manifest.subject_alias,a.Subject)&&strcmp(manifest.stage,'materials'));
 verify_products(manifest.product_files);
 fprintf('SCORPION %s: finite CEM contact preparation\n',a.Subject);
 cemdir=fullfile(base,'cem');
 if ~isfolder(cemdir)
  run_pipeline('Profile','ct_fem_contacts_prepare','InputFile',fullfile(base,'cem_contacts_input','cem_contacts_input.local.json'),'OutputDir',cemdir);
 end
 cm=read_json(fullfile(cemdir,'cem_contacts_manifest.local.json'));
 assert(strcmp(cm.subject_alias,a.Subject)&&cm.CEM_contacts_QC_passed);verify_products(cm.products);
 c=load(fullfile(cemdir,'cem_contacts.mat'),'result');c=c.result;
 M=load(fullfile(base,'materials','mesh.mat'));
 mm=eidors_obj('fwd_model',[a.Subject ' Scorpion own manual CT model']);
 mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
 mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
 mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;
 mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;clear M;
 assert(isequal(size(mm.nodes),size(c.nodes_lps_m))&&max(abs(mm.nodes(:)/1000-c.nodes_lps_m(:)))<1e-12);
 F=system_mat_fields(trkg4_scale_fmdl_to_si(mm,cfg));parts=cell(2,5);rules=[1024,4096];
 for rule=1:2
  f=load(fullfile(base,'materials',sprintf('fractions_q%d.mat',rules(rule))));
  assert(size(f.fractions,1)==size(mm.elems,1)&&all(sum(f.fractions,2)==1));
  for k=1:5
   w=repelem(f.fractions(:,k),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
   parts{rule,k}=B'*B;fprintf('SCORPION %s: material q%d component%d\n',a.Subject,rules(rule),k);
  end
 end
 clear F B w f;
 identity=struct('subject_alias',a.Subject,'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
  'mesh_sha256',trkg4_file_sha256(fullfile(base,'materials','mesh.mat')),...
  'cem_sha256',trkg4_file_sha256(fullfile(cemdir,'cem_contacts.mat')),...
  'reference_sha256',trkg4_file_sha256(ref_file),'inputs_sha256',trkg4_file_sha256(fullfile(base,'electrical_inputs.local.json')),...
  'response_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
  'field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields')));
 save(fullfile(out,'prepared_materials.mat'),'parts','mm','identity','-v7.3');
 states={'inhale','exhale'};answers=struct([]);checks=struct([]);derivative_checks=struct([]);
 scenarios=c.scenarios;ids=cellfun(@(s)s.id,scenarios,'UniformOutput',false);
 cases=cellfun(@(s)regexprep(s,'_(ttrkg|lateral)$',''),ids,'UniformOutput',false);
 [unique_cases,first]=unique(cases,'stable');active=[1,2,4,5];
 for state=1:2
  rho=ref.([states{state} '_rho_ohm_m'])(:);
  for rule=1:2
   K=sparse(size(mm.nodes,1),size(mm.nodes,1));for k=1:5,K=K+parts{rule,k}/rho(k);end
   for index=1:numel(first)
    s=scenarios{first(index)};ports=make_ports(s.ports);
    fprintf('SCORPION %s: FEM q%d %s %s\n',a.Subject,rules(rule),states{state},unique_cases{index});
    [T,J,qc]=trkg4_lateral_cem_response(K,s.grouped_contact_matrix_SI,parts(rule,:),rho,ports,active,1);
    entry=struct('state',states{state},'rule',rules(rule),'case_id',unique_cases{index},...
      'rho_ohm_m',rho','transfer_matrix_ohm',T,'jacobian_logrho',J,...
      'ttrkg_Z_ohm',T(2,1),'lateral_Z_ohm',T(4,3),...
      'ttrkg_from_lateral_current_ohm',T(2,3),'lateral_from_ttrkg_current_ohm',T(4,1));
    answers=append(answers,entry);checks=append(checks,qc);
    if state==1&&rule==2&&index==1
     errors=zeros(1,numel(active));h=1e-3;
     for j=1:numel(active)
      plus=rho;minus=rho;plus(active(j))=plus(active(j))*exp(h);minus(active(j))=minus(active(j))*exp(-h);
      kp=sparse(size(K,1),size(K,2));km=kp;
      for k=1:5,kp=kp+parts{rule,k}/plus(k);km=km+parts{rule,k}/minus(k);end
      tp=trkg4_lateral_cem_response(kp,s.grouped_contact_matrix_SI,parts(rule,:),plus,ports,[],1);
      tm=trkg4_lateral_cem_response(km,s.grouped_contact_matrix_SI,parts(rule,:),minus,ports,[],1);
      fd=(tp-tm)/(2*h);errors(j)=norm(fd-J(:,:,j),'fro')/norm(fd,'fro');
     end
     derivative_checks=struct('case_id',unique_cases{index},'state',states{state},'log_step',h,...
       'active_material_indices',active,'relative_errors',errors,'tolerance',1e-4,'passed',all(errors<1e-4));
     assert(derivative_checks.passed,'Own geometry derivative check failed');
    end
   end
  end
 end
 comparisons=struct([]);
 for i=1:numel(input.observations)
  observed=input.observations(i);ix=find(strcmp({answers.case_id},observed.case_id)&strcmp({answers.state},observed.state));
  assert(numel(ix)==2);z=zeros(1,2);
  for j=1:2,row=answers(ix(j));z(rules==row.rule)=row.([observed.role '_Z_ohm']);end
  tolerance=.02+.001*abs(z(2));
  comparisons=append(comparisons,struct('record_id',observed.record_id,'case_id',observed.case_id,...
   'state',observed.state,'role',observed.role,'observed_Z_ohm',observed.Z_ohm,...
   'coarse_Z_ohm',z(1),'fine_Z_ohm',z(2),'difference_ohm',abs(diff(z)),...
   'integration_tolerance_ohm',tolerance,'integration_passed',abs(diff(z))<=tolerance,...
   'residual_ohm',z(2)-observed.Z_ohm));
 end
 verify_products(manifest.product_files);verify_products(cm.products);verify_products(cm.source_files);
 assert(strcmp(trkg4_file_sha256(input.summary_source.path),input.summary_source.sha256));
 assert(strcmp(trkg4_file_sha256(ref_file),identity.reference_sha256));
 assert(strcmp(trkg4_file_sha256([mfilename('fullpath') '.m']),identity.worker_sha256));
 assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),identity.response_kernel_sha256));
 assert(strcmp(trkg4_file_sha256(which('system_mat_fields')),identity.field_kernel_sha256));
 report=struct('schema','scorpion_own_CT_forward_v1','identity',identity,'subject_alias',a.Subject,...
  'frequency_hz',100000,'current_per_source_A',.007,'rules',rules,'reference_materials',ref,...
  'answers',answers,'comparison',comparisons,'system_checks',checks,'derivative_checks',derivative_checks,...
  'FEM_was_run',true,'own_drive_is_working_assumption',true,'source_relative_phase','unknown',...
  'integration_passed',all([comparisons.integration_passed]),'mesh_convergence_proven',false,...
  'quantitative_result_accepted',false,'tissue_parameters_fitted',false,'clinical_validation',false);
 write_json(fullfile(out,'forward.local.json'),report);
catch exception
 write_json(fullfile(out,'failed.local.json'),struct('identifier',exception.identifier,'message',exception.message,...
  'FEM_forward_completed',false,'subject_alias',a.Subject));rethrow(exception);
end
end

function P=make_ports(groups)
names={groups.id};pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};
P=zeros(numel(groups),4);
for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
function value=read_json(path),value=jsondecode(fileread(path));end
function verify_products(items)
for k=1:numel(items),assert(strcmp(trkg4_file_sha256(items(k).path),items(k).sha256),'Input product changed');end
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
