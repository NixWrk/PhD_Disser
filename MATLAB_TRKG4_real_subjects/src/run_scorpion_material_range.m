function report=run_scorpion_material_range(varargin)
% Fresh bounded scenarios on existing canonical bulk operators; no reassembly.
p=inputParser;addParameter(p,'PlanFile','');parse(p,varargin{:});file=char(p.Results.PlanFile);
plan=jsondecode(fileread(file));validate_plan(plan);
root=fileparts(fileparts(mfilename('fullpath')));
cfg=struct('project_root',root,'length_unit','mm','scale_nodes_to_si',true);
trkg4_prepare_runtime(cfg);maxNumCompThreads(plan.threads);
if exist('gcp','file'),assert(isempty(gcp('nocreate')),'No parallel pool');end
out=fullfile(fileparts(file),'results');assert(~isfolder(out),'Use a fresh result directory');mkdir(out);
plan_sha=trkg4_file_sha256(file);worker=[mfilename('fullpath') '.m'];worker_sha=trkg4_file_sha256(worker);
try
 verify(plan.source_files);
 lockdir=fullfile(root,'output','exploratory','scorpion_session_runtime_20261026');
 if ~isfolder(lockdir),mkdir(lockdir);end
 write_json(fullfile(out,'resource_wait.local.json'),struct('state','waiting_for_sparse_solver_slot'));
 lockfile=javaObject('java.io.RandomAccessFile',fullfile(lockdir,'sparse_solver.lock'),'rw');
 channel=lockfile.getChannel();lease=channel.lock();
 guard=onCleanup(@()release_resource(lease,channel,lockfile)); %#ok<NASGU>
 verify(plan.source_files);
 baseline=jsondecode(fileread(plan.baseline_forward));ref=jsondecode(fileread(plan.reference_materials));
 assert(strcmp(baseline.subject_alias,plan.subject_alias)&&baseline.derivative_checks.passed&&~baseline.limited_run);
 assert(strcmp(baseline.identity.mesh_sha256,plan.identity.mesh_sha256)&&...
  strcmp(baseline.identity.cem_sha256,plan.identity.cem_sha256));
 assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),baseline.identity.response_kernel_sha256),...
  'Canonical response kernel differs from own baseline');
 for key={'field_kernel_sha256','scale_kernel_sha256','reference_sha256','response_kernel_sha256'}
  assert(strcmp(plan.identity.(key{1}),baseline.identity.(key{1})),'Plan canonical binding changed');
 end
 assert(all([baseline.system_checks.system_relative_residual]<1e-7)&&all([baseline.system_checks.reciprocity_ohm]<1e-7));
 assert(strcmp(trkg4_file_sha256(plan.cem_mat),plan.identity.cem_sha256));
 assert(strcmp(trkg4_file_sha256(plan.mesh_mat),plan.identity.mesh_sha256));
 assert(strcmp(trkg4_file_sha256(plan.reference_materials),plan.identity.reference_sha256));
 c=load(plan.cem_mat,'result');c=c.result;M=load(plan.mesh_mat,'nodes_mm');
 assert(isequal(size(M.nodes_mm),size(c.nodes_lps_m))&&max(abs(M.nodes_mm(:)/1000-c.nodes_lps_m(:)))<1e-12);clear M;
 scenarios=c.scenarios;if ~iscell(scenarios),scenarios=num2cell(scenarios);end
 names=cellfun(@(s)regexprep(s.id,'_(ttrkg|lateral)$',''),scenarios,'UniformOutput',false);
 selected=cellstr(plan.case_ids);assert(all(ismember(selected,names))&&any(strcmp(selected,plan.primary_case)),...
  'Requested subset must contain the declared primary case');
 selected=[{plan.primary_case};selected(~strcmp(selected,plan.primary_case))];
 for j=1:numel(selected)
  ix=find(strcmp(names,selected{j}));s=scenarios{ix(1)};
  for k=ix(:)'
   assert(isequal(s.ports,scenarios{k}.ports)&&isequal(s.grouped_contact_matrix_SI,scenarios{k}.grouped_contact_matrix_SI));
  end
 end
 identity=plan.identity;identity.plan_sha256=plan_sha;identity.worker_sha256=worker_sha;
 identity.response_kernel_sha256=trkg4_file_sha256(which('trkg4_lateral_cem_response'));
 observations=jsondecode(fileread(plan.observations_file));
 answers=struct([]);checks=struct([]);comparisons=struct([]);
 fd=struct('performed',false,'passed',false,'scope',plan.FD_scope);
 loaded='';data=[];n=size(c.nodes_lps_m,1);
 for h=1:numel(plan.rows)
  row=plan.rows(h);rho=double(row.rho_ohm_m(:));
  assert(rho(3)==ref.inhale_rho_ohm_m(3)&&rho(5)==ref.inhale_rho_ohm_m(5),'Fixed bone/fat changed');
  oi=find(strcmp({plan.operator_sources.id},row.operator_id));assert(numel(oi)==1);operator=plan.operator_sources(oi);
  if ~strcmp(loaded,operator.id)
   clear data parts K;
   assert(strcmp(trkg4_file_sha256(operator.path),operator.sha256),'Operator source changed');
   data=load(operator.path,'parts','identity');
   assert(strcmp(data.identity.subject_alias,plan.subject_alias)&&strcmp(data.identity.mesh_sha256,plan.identity.mesh_sha256));
   for key={'field_kernel_sha256','scale_kernel_sha256','reference_sha256'}
    assert(strcmp(data.identity.(key{1}),baseline.identity.(key{1})),'Operator canonical binding changed');
   end
   if strcmp(operator.layout,'2x5')
    assert(isequal(size(data.parts),[2,5])&&strcmp(data.identity.cem_sha256,plan.identity.cem_sha256));
   else
    assert(strcmp(operator.layout,'1x5')&&isequal(size(data.parts),[1,5]));
   end
   assert(all(cellfun(@(a)issparse(a)&&isequal(size(a),[n,n])&&all(isfinite(nonzeros(a))),data.parts),'all'));
   loaded=operator.id;
  end
  parts=data.parts(row.parts_row,:);
  K=bulk(parts,rho);
  for j=1:numel(selected)
   case_id=selected{j};ix=find(strcmp(names,case_id),1);s=scenarios{ix};P=ports(s.ports,plan.subject_alias);
   fprintf('MATERIAL_RANGE %s %s %s\n',plan.subject_alias,row.id,case_id);
   [T,J,qc]=trkg4_lateral_cem_response(K,s.grouped_contact_matrix_SI,parts,rho,P,[1 2 4 5],1);
   entry=struct('row_id',row.id,'point_id',row.point_id,'geometry',row.geometry,'case_id',case_id,...
    'operator_id',row.operator_id,'rule',row.rule,'rho_ohm_m',rho','transfer_matrix_ohm',T,...
    'jacobian_logrho',J,'active_material_indices',[1 2 4 5],...
    'port_order',{{'TT_drive','TT_measure','lateral_drive','lateral_measure'}},...
    'ttrkg_Z_ohm',T(2,1),'lateral_Z_ohm',T(4,3),...
    'ttrkg_from_lateral_current_ohm',T(2,3),'lateral_from_ttrkg_current_ohm',T(4,1));
   answers=append(answers,entry);qc.row_id=row.id;qc.case_id=case_id;checks=append(checks,qc);
   write_json(fullfile(out,['answer_' row.id '_' case_id '.local.json']),entry);
   for z=1:numel(observations.observations)
    o=observations.observations(z);
    if ~strcmp(o.status,'eligible_conditional')||~strcmp(o.case_id,case_id),continue;end
    predicted=entry.([o.role '_Z_ohm']);
    comparisons=append(comparisons,struct('row_id',row.id,'geometry',row.geometry,'rule',row.rule,...
     'record_id',o.record_id,'role',o.role,'observed_state',o.state,'case_id',case_id,...
     'observed_Z_ohm',o.Z_ohm,'predicted_Z_ohm',predicted,'residual_ohm',predicted-o.Z_ohm,...
     'annotation_status',o.annotation_status,'state_mapping','grid_point_has_no_assigned_breath_state'));
   end
   if row.finite_difference&&strcmp(case_id,plan.primary_case)
    fd=finite_difference(parts,rho,s.grouped_contact_matrix_SI,P,J);
    fd.row_id=row.id;fd.case_id=case_id;fd.rule=row.rule;fd.scope=plan.FD_scope;
    write_json(fullfile(out,'derivative_checks.local.json'),fd);assert(fd.passed,'Material-range FD failed');
   end
  end
 end
 q_checks=quadrature_checks(answers);heart_comparison=heart_differences(answers,plan.action);
 verify(plan.source_files);assert(strcmp(trkg4_file_sha256(file),plan_sha));
 assert(strcmp(trkg4_file_sha256(worker),worker_sha));
 assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),identity.response_kernel_sha256));
 report=struct('schema','scorpion_material_range_forward_v1','action',plan.action,'identity',identity,...
  'subject_alias',plan.subject_alias,'frequency_hz',100000,'answers',answers,'comparison',comparisons,...
  'system_checks',checks,'derivative_checks',fd,'q_checks',q_checks,'q_scope',plan.q_scope,...
  'heart_comparison',heart_comparison,'FEM_was_run',true,'geometry','fixed_deep_inspiration_CT',...
  'wholeheart_is_author_blood_replacement',true,'rho_wholeheart_blood_ohm_m',1.35,...
  'soft_background_is_effective_unlabelled_mixture',true,'blood_replacement_is_tissue_validation',false,...
  'fat_and_bone_fixed_registry_are_assumptions',true,'active_material_indices',[1 2 4 5],...
  'current_per_source_A',.007,'current_is_Z_multiplier',false,'excitation_scenario','own_drive',...
  'source_relative_phase','unknown','mesh_convergence',false,'mesh_convergence_proven',false,...
  'individual_fitted',false,'fitted_gain',false,'tissue_parameters_fitted',false,...
  'quantitative_result_accepted',false,'clinical_validation',false,'stroke_volume_estimated',false,...
  'shape_or_position_optimized',false,'reader_tables_default_open',false);
 write_json(fullfile(out,'forward.local.json'),report);
catch exception
 write_json(fullfile(out,'failed.local.json'),struct('identifier',exception.identifier,...
  'message',exception.message,'quantitative_result_accepted',false));rethrow(exception);
end
end

function validate_plan(p)
assert(strcmp(p.schema,'scorpion_material_range_plan_v1')&&any(strcmp(p.action,{'scan','heart'})));
assert(p.frequency_hz==100000&&isequal(cellstr(p.material_order(:)),{'soft_background';'lung';'bone';'heart';'fat'}));
assert(isequal(p.active_material_indices(:)',[1 2 4 5])&&p.threads>=1&&p.threads<=6&&fix(p.threads)==p.threads);
assert(any(strcmp(p.subject_alias,{'nik','georg','yaroslav'}))&&strcmp(p.primary_case,'L140_TT2x2'));
assert(p.soft_background_is_effective_unlabelled_mixture&&~p.blood_replacement_is_tissue_validation);
assert(numel(unique({p.rows.id}))==numel(p.rows));
for k=1:numel(p.rows)
 r=p.rows(k);v=r.rho_ohm_m(:);
 assert(numel(v)==5&&all(isfinite(v))&&all(v>0)&&ismember(v(1),[2 2.5 3])&&ismember(v(2),[2 4 7 10 12])&&v(4)==1.35);
 assert(ismember(r.rule,[1024 4096])&&ismember(r.parts_row,[1 2]));
 if strcmp(p.action,'scan')
  assert(strcmp(r.geometry,'wholeheart_ct')&&strcmp(r.operator_id,'baseline'));
  assert(r.parts_row==1+(r.rule==4096));
  if r.rule==1024,assert((ismember(v(1),[2 3])&&ismember(v(2),[2 12]))||(v(1)==2.5&&v(2)==7));end
  assert(r.finite_difference==(r.rule==4096&&v(1)==2.5&&v(2)==7));
 else
  assert(v(1)==2.5&&v(2)==7&&r.parts_row==1&&~r.finite_difference);
 end
end
if strcmp(p.action,'scan')
 assert(numel(p.rows)==20&&sum([p.rows.rule]==4096)==15&&sum([p.rows.finite_difference])==1);
 fine=p.rows([p.rows.rule]==4096);coarse=p.rows([p.rows.rule]==1024);
 fine_points=cell2mat(arrayfun(@(r)r.rho_ohm_m([1 2])',fine(:),'UniformOutput',false));
 coarse_points=cell2mat(arrayfun(@(r)r.rho_ohm_m([1 2])',coarse(:),'UniformOutput',false));
 assert(size(unique(fine_points,'rows'),1)==15&&size(unique(coarse_points,'rows'),1)==5);
else
 assert(numel(p.rows)==8&&sum([p.rows.rule]==4096)==4);
end
end
function K=bulk(parts,rho)
n=size(parts{1},1);K=sparse(n,n);for k=1:5,K=K+parts{k}/rho(k);end
end
function P=ports(groups,subject)
if strcmp(subject,'georg'),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
names={groups.id};assert(numel(unique(names))==numel(names));P=zeros(numel(groups),4);
for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
function result=finite_difference(parts,rho,C,P,J)
active=[1 2 4 5];h=1e-3;errors=zeros(1,4);scaled=zeros(4,4);
ix=sub2ind([4 4],[2 4 2 4],[1 3 3 1]);
for k=1:4
 rp=rho;rm=rho;rp(active(k))=rp(active(k))*exp(h);rm(active(k))=rm(active(k))*exp(-h);
 tp=trkg4_lateral_cem_response(bulk(parts,rp),C,parts,rp,P,[],1);
 tm=trkg4_lateral_cem_response(bulk(parts,rm),C,parts,rm,P,[],1);
 f=(tp-tm)/(2*h);a=J(:,:,k);errors(k)=norm(f-a,'fro')/max(norm(f,'fro'),realmin);
 scaled(:,k)=abs(f(ix)-a(ix))'./(1e-6+1e-4*max(abs(f(ix)),abs(a(ix))))';
end
result=struct('performed',true,'passed',all(errors<1e-4)&&all(scaled<=1,'all'),...
 'log_step',h,'active_material_indices',active,'relative_errors',errors,'transfer_scaled_errors',scaled,...
 'relative_tolerance',1e-4,'transfer_absolute_tolerance_ohm',1e-6);
end
function result=quadrature_checks(answers)
result=struct([]);
for k=1:numel(answers)
 a=answers(k);if a.rule~=4096,continue;end
 ix=find(strcmp({answers.point_id},a.point_id)&strcmp({answers.geometry},a.geometry)&...
  strcmp({answers.case_id},a.case_id)&[answers.rule]==1024);
 row=struct('point_id',a.point_id,'geometry',a.geometry,'case_id',a.case_id,...
  'evaluated',false,'passed',false,'reason','q1024_not_in_declared_scope',...
  'ttrkg_difference_ohm',[],'lateral_difference_ohm',[],'ttrkg_tolerance_ohm',[],'lateral_tolerance_ohm',[]);
 if ~isempty(ix)
  assert(numel(ix)==1);b=answers(ix);row.evaluated=true;row.reason='';
  row.ttrkg_difference_ohm=abs(a.ttrkg_Z_ohm-b.ttrkg_Z_ohm);row.lateral_difference_ohm=abs(a.lateral_Z_ohm-b.lateral_Z_ohm);
  row.ttrkg_tolerance_ohm=.02+.001*abs(a.ttrkg_Z_ohm);row.lateral_tolerance_ohm=.02+.001*abs(a.lateral_Z_ohm);
  row.passed=row.ttrkg_difference_ohm<=row.ttrkg_tolerance_ohm&&row.lateral_difference_ohm<=row.lateral_tolerance_ohm;
 end
 result=append(result,row);
end
end
function result=heart_differences(answers,action)
result=struct([]);if ~strcmp(action,'heart'),return;end
for k=1:numel(answers)
 a=answers(k);if strcmp(a.geometry,'wholeheart_ct'),continue;end
 ix=find(strcmp({answers.geometry},'wholeheart_ct')&strcmp({answers.case_id},a.case_id)&[answers.rule]==a.rule);
 assert(numel(ix)==1);b=answers(ix);assert(isequal(a.rho_ohm_m,b.rho_ohm_m));
 result=append(result,struct('geometry',a.geometry,'case_id',a.case_id,'rule',a.rule,...
  'definition','sphere_blood_minus_wholeheart_blood','delta_transfer_ohm',a.transfer_matrix_ohm-b.transfer_matrix_ohm,...
  'delta_ttrkg_Z_ohm',a.ttrkg_Z_ohm-b.ttrkg_Z_ohm,'delta_lateral_Z_ohm',a.lateral_Z_ohm-b.lateral_Z_ohm));
end
end
function verify(items)
for k=1:numel(items),assert(strcmp(trkg4_file_sha256(items(k).path),items(k).sha256),'Input changed: %s',items(k).path);end
end
function r=append(r,v)
if isempty(r),r=v;else,r(end+1)=v;end
end
function write_json(path,v)
assert(~isfile(path));fid=fopen(path,'w','n','UTF-8');assert(fid>0);g=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(v,PrettyPrint=true));
end
function release_resource(lease,channel,lockfile)
lease.release();channel.close();lockfile.close();
end
