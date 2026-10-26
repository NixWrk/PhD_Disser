function report=run_scorpion_heart_pilot(plan_file,action,variant)
% Five-material own-CT pilot; canonical EIDORS assembly and full CEM response.
% Parent schedules ONE variant/q per call. No optimization or volume inversion.
if nargin<3,variant='';end
plan=jsondecode(fileread(plan_file));base=fileparts(plan_file);
assert(strcmp(plan.schema,'scorpion_heart_pilot_plan_v1')&&plan.frequency_hz==100000);
assert(isequal(cellstr(plan.material_order(:)),{'soft_background';'lung';'bone';'heart';'fat'}));
assert(isequal(plan.active_material_indices_one_based(:)',[1 2 4 5]));
assert(any(strcmp(action,{'assemble','solve','breath'})),'Unknown action');
assert(plan.threads>=1&&plan.threads<=6&&plan.threads==fix(plan.threads));
% Runtime settings only; no fallback to another participant's registry/data.
root=fileparts(fileparts(mfilename('fullpath')));
cfg=struct('project_root',root,'length_unit','mm','scale_nodes_to_si',true);
trkg4_prepare_runtime(cfg);maxNumCompThreads(plan.threads);
if exist('gcp','file'),assert(isempty(gcp('nocreate')),'No parallel pool');end
out=fullfile(base,[action '_' char(variant)]);
assert(~isfolder(out),'Inspect existing attempt; use a fresh plan/output');mkdir(out);
plan_sha=trkg4_file_sha256(plan_file);
worker_sha=trkg4_file_sha256([mfilename('fullpath') '.m']);
try
 verify(plan.source_files);check_hash(plan.mesh_mat,plan.mesh_sha256);
 % Same advisory OS lock as the baseline worker, including assembly memory.
 lockdir=fullfile(root,'output','exploratory','scorpion_session_runtime_20261026');
 if ~isfolder(lockdir),mkdir(lockdir);end
 lockfile=javaObject('java.io.RandomAccessFile',fullfile(lockdir,'sparse_solver.lock'),'rw');
 channel=lockfile.getChannel();lease=channel.lock();
 guard=onCleanup(@()release_resource(lease,channel,lockfile)); %#ok<NASGU>
 ref=jsondecode(fileread(plan.reference_materials));
 assert(ref.frequency_hz==100000);
 rho=double(ref.inhale_rho_ohm_m(:));
 assert(numel(rho)==5&&all(isfinite(rho))&&all(rho>0));
 identity=struct('subject_alias',plan.subject_alias,'plan_sha256',plan_sha,...
  'worker_sha256',worker_sha,'mesh_sha256',plan.mesh_sha256,...
  'reference_sha256',trkg4_file_sha256(plan.reference_materials),...
  'field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields')),...
  'response_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
  'scale_kernel_sha256',trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')));
 if strcmp(action,'assemble')
  entry=select_variant(plan,variant);check_hash(entry.file,entry.sha256);
  M=load(plan.mesh_mat,'nodes_mm','elems','boundary');
  mm=eidors_obj('fwd_model',[plan.subject_alias ' own-CT whole-heart scenario']);
  mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
  mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
  mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;
  mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;clear M;
  data=load(entry.file,'fractions','element_ids');ne=size(mm.elems,1);nn=size(mm.nodes,1);
  f=double(data.fractions);assert(isequal(double(data.element_ids(:)),(1:ne)'));
  assert(isequal(size(f),[ne 5])&&all(isfinite(f),'all')&&all(f>=0,'all')&&all(sum(f,2)==1));
  si=trkg4_scale_fmdl_to_si(mm,cfg);clear mm data;
  F=system_mat_fields(si);assert(isequal(size(F),[3*ne nn]));
  parts=cell(1,5);sigma_effective=zeros(ne,1);tic;
  for k=1:5
   w=f(:,k);ids=find(w>0);rows=reshape((3*(ids-1)+(1:3)).',[],1);
   G=spdiags(repelem(sqrt(w(ids)),3),0,numel(rows),numel(rows))*F(rows,:);
   parts{k}=G'*G;sigma_effective=sigma_effective+w/rho(k);
   fprintf('HEART %s %s: fresh component %d/5\n',plan.subject_alias,variant,k);
  end
  clear G w ids rows f;
  homogeneous=sparse(nn,nn);K=homogeneous;
  for k=1:5,homogeneous=homogeneous+parts{k};K=K+parts{k}/rho(k);end
  target=F'*F;clear F;
  homogeneous_error=norm(homogeneous-target,'fro')/norm(target,'fro');clear homogeneous target;
  image=mk_image(si,1);image.elem_data=sigma_effective;
  independent=calc_system_mat(image);
  heterogeneous_error=norm(K-independent.E,'fro')/norm(independent.E,'fro');
  assert(isfinite(homogeneous_error)&&homogeneous_error<1e-12,'Homogeneous assembly check failed');
  assert(isfinite(heterogeneous_error)&&heterogeneous_error<1e-12,'Canonical heterogeneous assembly check failed');
  clear K independent image si sigma_effective;
  save(fullfile(out,'parts.mat'),'parts','identity','entry','-v7.3');
  report=struct('status','canonical_weighted_assembly_checked','identity',identity,...
   'variant',entry,'parts_sha256',trkg4_file_sha256(fullfile(out,'parts.mat')),...
   'homogeneous_error',homogeneous_error,'heterogeneous_error',heterogeneous_error,...
   'seconds',toc,'FEM_was_run',false,'quantitative_result_accepted',false);
  check_hash(entry.file,entry.sha256);
 else
  baseline=jsondecode(fileread(plan.baseline_forward));check_baseline(baseline,plan,identity);
  identity.baseline_sha256=trkg4_file_sha256(plan.baseline_forward);
  identity.baseline_prepared_sha256=trkg4_file_sha256(plan.baseline_prepared);
  identity.cem_sha256=trkg4_file_sha256(plan.cem_mat);
  c=load(plan.cem_mat,'result');c=c.result;
  M=load(plan.mesh_mat,'nodes_mm');
  assert(isequal(size(M.nodes_mm),size(c.nodes_lps_m))&&max(abs(M.nodes_mm(:)/1000-c.nodes_lps_m(:)))<1e-12);clear M;
  if strcmp(action,'solve')
   entry=select_variant(plan,variant);assembly=fullfile(base,['assemble_' char(variant)]);
   a=jsondecode(fileread(fullfile(assembly,'report.local.json')));
   assert(strcmp(a.status,'canonical_weighted_assembly_checked'));
   assert(strcmp(a.identity.plan_sha256,plan_sha)&&strcmp(a.identity.worker_sha256,worker_sha));
   check_hash(fullfile(assembly,'parts.mat'),a.parts_sha256);
   data=load(fullfile(assembly,'parts.mat'),'parts');parts=data.parts;clear data;
   states=struct('id','inhale','rho',rho);rule=entry.q;
   selected=plan.case_ids;
  else
   assert(any(strcmp(variant,{'q1024','q4096'})));
   rule=str2double(variant(2:end));index=find([1024 4096]==rule);
   data=load(plan.baseline_prepared,'parts','identity');
   assert(strcmp(data.identity.mesh_sha256,plan.mesh_sha256));
   parts=data.parts(index,:);clear data;
   exhale=double(ref.exhale_rho_ohm_m(:));
   assert(numel(exhale)==5&&all(isfinite(exhale))&&all(exhale>0)&&rho(3)==exhale(3),'Fixed bone required');
   lung=rho;lung(2)=exhale(2);background=rho;background([1 5])=exhale([1 5]);
   states=struct('id',{'lung_only','background_only'},'rho',{lung,background});
   selected={}; % All baseline cases for conditional breath predictions.
  end
  assert(numel(parts)==5);n=size(parts{1},1);
  scenarios=c.scenarios;if ~iscell(scenarios),scenarios=num2cell(scenarios);end
  names=cellfun(@(s)regexprep(s.id,'_(ttrkg|lateral)$',''),scenarios,'UniformOutput',false);
  [case_names,first]=unique(names,'stable');answers=struct([]);
  if ~isempty(selected),assert(all(ismember(cellstr(selected),case_names)),'Unknown requested case');end
  for h=1:numel(states)
   r=states(h).rho;K=sparse(n,n);for k=1:5,K=K+parts{k}/r(k);end
   for j=1:numel(first)
    id=case_names{j};if ~isempty(selected)&&~any(strcmp(id,selected)),continue;end
    s=scenarios{first(j)};P=ports(s.ports);
    [T,J,qc]=trkg4_lateral_cem_response(K,s.grouped_contact_matrix_SI,parts,r,P,[1 2 4 5],1);
    b=baseline.answers(strcmp({baseline.answers.case_id},id)&strcmp({baseline.answers.state},'inhale')&[baseline.answers.rule]==rule);
    assert(numel(b)==1,'Missing/duplicate own baseline case');
    row=struct('case_id',id,'state',states(h).id,'rule',rule,'rho_ohm_m',r',...
     'transfer_matrix_ohm',T,'jacobian_logrho',J,'ttrkg_Z_ohm',T(2,1),'lateral_Z_ohm',T(4,3),...
     'delta_from_wholeheart_inhale_ohm',T-b.transfer_matrix_ohm,...
     'baseline_wholeheart_transfer_ohm',b.transfer_matrix_ohm,'system_checks',qc);
    if isempty(answers),answers=row;else,answers(end+1)=row;end %#ok<AGROW>
   end
  end
  report=struct('schema','scorpion_heart_forward_v1','action',action,'variant',variant,...
   'identity',identity,'subject_alias',plan.subject_alias,'frequency_hz',100000,...
   'answers',answers,'FEM_was_run',true,'own_drive_is_working_assumption',true,...
   'source_relative_phase','unknown','active_material_indices',[1 2 4 5],...
   'fixed_inhale_CT',true,'mesh_convergence_proven',false,'quantitative_result_accepted',false,...
   'shape_or_position_optimized',false,'clinical_validation',false,'stroke_volume_estimated',false,...
   'reader_tables_default_open',false);
  check_hash(plan.baseline_forward,identity.baseline_sha256);
  check_hash(plan.baseline_prepared,identity.baseline_prepared_sha256);check_hash(plan.cem_mat,identity.cem_sha256);
 end
 verify(plan.source_files);check_hash(plan_file,plan_sha);
 check_hash([mfilename('fullpath') '.m'],worker_sha);
 check_hash(which('system_mat_fields'),identity.field_kernel_sha256);
 check_hash(which('trkg4_lateral_cem_response'),identity.response_kernel_sha256);
 check_hash(which('trkg4_scale_fmdl_to_si'),identity.scale_kernel_sha256);
 report.matlab_version=version;write_json(fullfile(out,'report.local.json'),report);
catch failure
 write_json(fullfile(out,'failed.local.json'),struct('identifier',failure.identifier,...
  'reason',failure.message,'quantitative_result_accepted',false));rethrow(failure);
end
end

function entry=select_variant(plan,variant)
idx=find(strcmp({plan.variants.id},variant));assert(numel(idx)==1,'Unknown material variant');entry=plan.variants(idx);
end
function P=ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft'))
 pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else
 pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};
end
P=zeros(numel(groups),4);
for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
function check_baseline(b,plan,identity)
assert(strcmp(b.subject_alias,plan.subject_alias)&&b.frequency_hz==100000&&b.FEM_was_run);
assert(strcmp(b.identity.mesh_sha256,plan.mesh_sha256)&&strcmp(b.identity.reference_sha256,identity.reference_sha256));
assert(strcmp(b.identity.cem_sha256,trkg4_file_sha256(plan.cem_mat)));
assert(b.derivative_checks.passed,'Baseline finite-difference check not passed');
assert(isequal(b.derivative_checks.active_material_indices(:)',[1 2 4 5]));
assert(all([b.system_checks.system_relative_residual]<1e-7)&&all([b.system_checks.reciprocity_ohm]<1e-7));
% Quadrature and global mesh convergence retain separate statuses in reports.
end
function verify(items)
for k=1:numel(items),check_hash(items(k).path,items(k).sha256);end
end
function check_hash(path,expected)
assert(strcmp(trkg4_file_sha256(path),expected),'Input/code changed: %s',path);
end
function write_json(path,value)
assert(~isfile(path),'Never overwrite attempt results');fid=fopen(path,'w','n','UTF-8');assert(fid>0);
closer=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
function release_resource(lease,channel,lockfile)
lease.release();channel.close();lockfile.close();
end
