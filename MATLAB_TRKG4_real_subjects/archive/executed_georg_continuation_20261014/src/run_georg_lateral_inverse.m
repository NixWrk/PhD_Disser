function report=run_georg_lateral_inverse(varargin)
% Nix joint tissue/centre/angle search, or a discrete tissue-profile scan.
% Default: converged FEM. Explicit Nix-level preliminary mode retains failed
% mesh checks and requires an independently saved numerical audit.
p=inputParser;
addParameter(p,'OutputDir','output/exploratory/georg_lateral_inverse_20261011');
addParameter(p,'PoseIds',{});
addParameter(p,'SearchMode','nix_joint',@(x)any(strcmp(x,{'nix_joint','profile_grid'})));
addParameter(p,'MaxEvaluations',80,@(x)isscalar(x)&&x>=16&&x==fix(x));
addParameter(p,'NumericalPolicy','strict',@(x)any(strcmp(x,{'strict','nix_preliminary'})));
addParameter(p,'NumericalAuditFile','');
addParameter(p,'ModelDir','');
addParameter(p,'ModelContractFile',fullfile('output','georg_current_model.local.json'));
addParameter(p,'RunLabel','',@(x)isempty(x)||~isempty(regexp(char(x),'^[A-Za-z0-9_]+$','once')));
addParameter(p,'ResumeFrom','',@(x)isempty(x)||~isempty(regexp(char(x),'^[A-Za-z0-9_]+$','once')));
addParameter(p,'CentreFile','');
addParameter(p,'CentreHalfWidthMM',[30,30],@(x)isnumeric(x)&&numel(x)==2&&all(isfinite(x))&&all(x>0));
addParameter(p,'AngleHalfWidthDeg',20,@(x)isscalar(x)&&isfinite(x)&&x>0&&x<=45);
addParameter(p,'CheckCandidateFile','');
addParameter(p,'WarmStartFile','');
addParameter(p,'StartIndices',[],@(x)isnumeric(x)&&isvector(x)&&all(x>=1)&&all(x==fix(x))||isempty(x));
addParameter(p,'ExcitationScenario','own_drive',@(x)any(strcmp(x,{'own_drive','in_phase','opposite_phase','quadrature_magnitude'})));
parse(p,varargin{:});a=p.Results;
forward_check=~isempty(a.CheckCandidateFile);
assert(~forward_check||isempty(a.ResumeFrom),'A fixed forward check cannot resume an inverse search');
assert(isempty(a.CentreFile)||strcmp(a.SearchMode,'nix_joint'),'CentreFile requires the joint search');
assert(isempty(a.ResumeFrom)||strcmp(a.SearchMode,'nix_joint'),'ResumeFrom requires the joint search');
cfg=trkg4_config('georg');root=cfg.project_root;
cfg.frequency_hz=100000;
% Author decision 11.10.2026: reach Nix's numerical checks, then launch the
% preliminary inverse. This explicit policy never changes the common pointer.
strict=strcmp(a.NumericalPolicy,'strict')&&~forward_check;
current=trkg4_georg_current_model(root,strict,a.ModelContractFile);
out=fullfile(root,char(a.OutputDir));
inputfile=fullfile(out,'inputs.local.json');inputs=jsondecode(fileread(inputfile));
assert(inputs.confirmed_channel_roles_not_fitted && ~inputs.annotation_acceptance_changed);
for j=1:numel(inputs.sources)
 assert(strcmp(trkg4_file_sha256(inputs.sources(j).path),inputs.sources(j).sha256),'Inverse input changed');
end
obsfile=fullfile(out,'observations.local.csv');O=readtable(obsfile,'TextType','string');
assert(strcmp(trkg4_file_sha256(obsfile),inputs.outputs_sha256.observations_local_csv));
O=O(O.state~="quiet_breathing",:);assert(height(O)==24);
O.is_cross=lower(string(O.is_cross))=="true" | string(O.is_cross)=="1";
gridfile=fullfile(out,'pose_grid.local.json');grid=jsondecode(fileread(gridfile));
assert(strcmp(grid.surface_sha256,trkg4_file_sha256(fullfile(root,current.model_directory,'body_ct_surface.stl'))),...
 'Candidate coordinates refer to a different CT surface');
modeldir=char(a.ModelDir);if isempty(modeldir),modeldir=current.fem_directory;end
base=fullfile(root,modeldir);planfile=fullfile(base,'plan.local.json');plan=jsondecode(fileread(planfile));
% A variant's forward-control list does not define the arm montages.
contactfile=fullfile(root,current.model_directory,'fem_contacts.local.json');
contacts=jsondecode(fileread(contactfile));
auditfile=char(a.NumericalAuditFile);
if ~strict&&~forward_check
 assert(~isempty(auditfile),'Nix preliminary mode requires a numerical audit');
 auditfile=fullfile(root,auditfile);audit=jsondecode(fileread(auditfile));
 if isfield(plan,'constant_physical_contact_layout')&&plan.constant_physical_contact_layout
  assert(strcmp(audit.schema,'georg_quality_separated_numerics_v1')&&audit.complete);
  assert(audit.shape_QC_passed&&audit.energy_inheritance_passed&&audit.quadrature_passed&&audit.derivatives_passed&&audit.system_checks_passed,...
   'New physical-layout numerical prerequisites failed');
  assert(strcmp(a.ExcitationScenario,'own_drive'),'Other excitation scenarios need a separate audit');
  assert(strcmp(audit.model_plan_sha256,trkg4_file_sha256(planfile))&&...
   strcmp(audit.physical_contacts_sha256,trkg4_file_sha256(contactfile)),'Numerical controls refer to a different model');
  assert(strcmp(audit.physical_montage_helper_sha256,trkg4_file_sha256(which('trkg4_georg_physical_montages'))));
  for j=1:numel(audit.sources)
   assert(strcmp(trkg4_file_sha256(audit.sources(j).path),audit.sources(j).sha256),'Separated numerical audit dependency changed');
  end
  scope_passed=audit.quadrature_passed&&audit.derivatives_passed&&audit.system_checks_passed;
 else
 assert(strcmp(audit.schema,'georg_nix_numerical_level_v1')&&audit.complete,...
  'Nix-level numerical audit is incomplete');
 % Nix's inverse uses the own-source transfer. Preserve failures of extra,
 % counter-source transfers; they gate the other excitation scenarios.
 scope_passed=audit.all_checks_passed;
 if strcmp(a.ExcitationScenario,'own_drive')
  scope_passed=all([audit.derivatives.passed]);
  for j=1:numel(audit.cases)
   checked=audit.cases(j);assert(isequal(checked.readout_port_indices(:)',[2,10,12,4]));
   own=[1,3];scope_passed=scope_passed&&all(checked.quadrature_difference_ohm(own)<=checked.quadrature_tolerance_ohm(own));
   scope_passed=scope_passed&&checked.fine_qc.system_relative_residual<1e-7&&checked.coarse_qc.system_relative_residual<1e-7;
   scope_passed=scope_passed&&checked.fine_qc.reciprocity_ohm<1e-7&&checked.coarse_qc.reciprocity_ohm<1e-7;
  end
 end
 assert(scope_passed,'Numerical checks failed for the requested excitation scenario');
 assert(strcmp(audit.model_directory,modeldir)&&strcmp(audit.identity.plan_sha256,trkg4_file_sha256(planfile)),...
  'Numerical audit belongs to another mesh');
 assert(strcmp(audit.identity.current_pointer_sha256,trkg4_file_sha256(fullfile(root,'output/georg_current_model.local.json'))),...
  'Common geometry changed after the audit');
 assert(strcmp(audit.identity.worker_sha256,trkg4_file_sha256(fullfile(root,'src/run_georg_lateral_nix_audit.m'))));
 assert(strcmp(audit.identity.cem_response_sha256,trkg4_file_sha256(fullfile(root,'src/trkg4_lateral_cem_response.m'))));
 assert(strcmp(audit.identity.material_cache_helper_sha256,trkg4_file_sha256(fullfile(root,'src/trkg4_georg_lateral_materials.m'))));
 assert(strcmp(audit.identity.local_patch_builder_sha256,trkg4_file_sha256(fullfile(root,'tools/trkg4_local_exact_patches.m'))));
 assert(strcmp(audit.identity.local_patch_context_sha256,trkg4_file_sha256(fullfile(root,'tools/trkg4_local_patch_context.m'))));
 end
end
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
if forward_check
 [cache,prepared]=trkg4_georg_check_materials(cfg,base);
else
 [cache,prepared]=trkg4_georg_lateral_materials(cfg,base);
end
if ~strict&&~forward_check
 if isfield(audit,'prepared_sha256'),expected=audit.prepared_sha256;else,expected=audit.identity.prepared_sha256;end
 assert(strcmp(expected,trkg4_file_sha256(cache)),'Audited matrices changed');
end
S=load(cache);addpath(fullfile(root,'tools'));
% Existing certified local wrapper calls the unchanged canonical builder;
% omitted faces cannot affect growth, ownership, ties or induced areas.
fprintf('JOINT building exact local contact context\n');
context=trkg4_local_patch_context(S.mm);
physical_inheritance=[];pose_mm=S.mm;
if isfield(plan,'constant_physical_contact_layout')&&plan.constant_physical_contact_layout
 M=load(plan.pose_reference_mesh);pose_mm.nodes=M.nodes_mm;pose_mm.elems=double(M.elems);pose_mm.boundary=double(M.boundary);clear M;
 [physical_inheritance,contact_inheritance_identity]=trkg4_georg_contact_inheritance(S.mm,plan,fullfile(base,'mesh.mat'));
end
arm_names={'1x1','1x4','2x2'};arms=cell(1,3);
for j=1:3
 r=contacts.scenarios(strcmp({contacts.scenarios.id},['TTRKG_' arm_names{j}]));groups=r.groups_mm;
 if isnumeric(groups),groups=num2cell(groups,[2,3]);end
 arms{j}=groups(:);
end
ref=jsondecode(fileread(fullfile(base,'reference_materials.local.json')));
fixed=ref.rho_ohm_m(:);active=[1,2,5];
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'cem_response_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
 'readout_sha256',trkg4_file_sha256(which('trkg4_two_source_readout')),...
 'nix_method_source_sha256',trkg4_file_sha256(fullfile(root,'src','run_lateral_fat_inverse.m')),...
 'nix_joint_pose_source_sha256',trkg4_file_sha256(fullfile(root,'src','run_trkg4_inverse_inhale.m')),...
 'surface_pose_sha256',trkg4_file_sha256(fullfile(root,'src','trkg4_lateral_surface_pose.m')),...
 'direct_search_sha256',trkg4_file_sha256(fullfile(root,'src','trkg4_bounded_pattern_search.m')),...
 'input_sha256',trkg4_file_sha256(inputfile),'grid_sha256',trkg4_file_sha256(gridfile),...
 'observations_sha256',trkg4_file_sha256(obsfile),'prepared_sha256',trkg4_file_sha256(cache));
identity.excitation_scenario=char(a.ExcitationScenario);
identity.search_mode=char(a.SearchMode);
identity.numerical_policy=char(a.NumericalPolicy);
identity.model_plan_sha256=trkg4_file_sha256(planfile);
identity.contact_file_sha256=trkg4_file_sha256(contactfile);
contract_file=char(a.ModelContractFile);
if isempty(regexp(contract_file,'^([A-Za-z]:[\\/]|[\\/])','once')),contract_file=fullfile(root,contract_file);end
identity.model_contract_sha256=trkg4_file_sha256(contract_file);
if isfield(plan,'constant_physical_contact_layout')&&plan.constant_physical_contact_layout
 identity.physical_montage_helper_sha256=trkg4_file_sha256(which('trkg4_georg_physical_montages'));
 identity.pose_reference_mesh_sha256=trkg4_file_sha256(plan.pose_reference_mesh);
 identity.contact_inheritance=contact_inheritance_identity;
 if ~isempty(plan.boundary_parent_map),identity.boundary_parent_map_sha256=trkg4_file_sha256(plan.boundary_parent_map);end
end
identity.material_cache_helper_sha256=trkg4_file_sha256(which('trkg4_georg_lateral_materials'));
if forward_check,identity.material_cache_helper_sha256=trkg4_file_sha256(which('trkg4_georg_check_materials'));end
identity.patch_builder_sha256=trkg4_file_sha256(which('electrode_faces_by_area'));
identity.local_patch_builder_sha256=trkg4_file_sha256(which('trkg4_local_exact_patches'));
identity.local_patch_context_sha256=trkg4_file_sha256(which('trkg4_local_patch_context'));
identity.local_subset_builder_sha256=trkg4_file_sha256(which('trkg4_local_cached_subset'));
if ~strict&&~forward_check
 identity.numerical_audit_sha256=trkg4_file_sha256(auditfile);
 identity.numerical_audit_scope=char(a.ExcitationScenario);
 identity.all_transfer_checks_passed=audit.all_checks_passed;
 identity.requested_readout_checks_passed=scope_passed;
end
starts=[log(fixed(active))',log(fixed(active))';log([4,15,40,4,15,40])];
bounds=inputs.search_bounds_ohm_m;
assert(isequal(size(bounds),[3,2])&&all(bounds(:,1)>0)&&all(bounds(:,2)>bounds(:,1)));
lb=repmat(log(bounds(:,1))',1,2);ub=repmat(log(bounds(:,2))',1,2);
starts=min(max(starts,lb),ub);
% Shared closure variables for both the joint search and fixed-pose profiles.
C={};drive={};measure={};lastx=[];lastZ=[];lastJ=[];lastOwn=[];lastCounter=[];
lastres=0;lastrec=0;evaluations=0;
state_cache={[],[]};lastpose=[];lastpatchqc=[];
if forward_check
 report=check_candidate();return;
end
if strcmp(a.SearchMode,'nix_joint')
 assert(isempty(a.PoseIds),'PoseIds belongs to the discrete profile mode');
 report=joint_search();return;
end
opt=optimoptions('lsqnonlin','SpecifyObjectiveGradient',true,'Display','off',...
 'FunctionTolerance',1e-8,'StepTolerance',1e-6,'OptimalityTolerance',1e-6,...
 'MaxIterations',60,'MaxFunctionEvaluations',100);
dest=fullfile(out,['fits_' char(a.ExcitationScenario)]);if ~isfolder(dest),mkdir(dest);end
rows=struct([]);geometry_failures=struct([]);
for k=1:numel(grid.poses)
 pose=grid.poses(k);
 if ~isempty(a.PoseIds)&&~any(strcmp(a.PoseIds,pose.pose_id)),continue;end
 file=fullfile(dest,[pose.pose_id '.local.json']);
 if isfile(file)
  prior=jsondecode(fileread(file));assert(isequal(prior.identity,identity),'Stale pose fit');
  rows=append(rows,prior);continue;
 end
 [C,drive,measure,patchqc]=montages(pose,S.mm,arms,arm_names,O,contacts,plan,cfg,context,physical_inheritance);
 if ~patchqc.passed
  geometry_failures=append(geometry_failures,struct('pose_id',pose.pose_id,'qc',patchqc));continue;
 end
 lastx=[];lastZ=[];lastJ=[];lastOwn=[];lastCounter=[];lastres=0;lastrec=0;evaluations=0;
 state_cache={[],[]};
 solutions=zeros(2,6);values=zeros(2,1);flags=zeros(2,1);
 for start=1:2
  [x,value,~,flag]=lsqnonlin(@objective,starts(start,:),lb,ub,opt);
  solutions(start,:)=x;values(start)=value;flags(start)=flag;
 end
 [~,best]=min(values);[residual,J]=objective(solutions(best,:));sv=svd(J);
 boundary=any(abs(solutions(best,:)'-[lb;ub]')<1e-4,2);
 row=struct('pose_id',pose.pose_id,'centre_lps_mm',pose.centre_lps_mm,...
  'electrodes',pose.electrodes,'rho_inhale_ohm_m',exp(solutions(best,1:3)),...
  'rho_exhale_ohm_m',exp(solutions(best,4:6)),'rho_bone_ohm_m',fixed(3),...
  'rho_heart_ohm_m',fixed(4),'RSS_ohm2',sum(residual.^2),'RMSE_ohm',sqrt(mean(residual.^2)),...
  'target_Z_ohm',O.Z_ohm,'predicted_Z_ohm',lastZ,'residual_ohm',residual,...
  'own_drive_transfer_ohm',lastOwn,'counter_drive_transfer_ohm',lastCounter,...
  'jacobian_logrho',J,'singular_values',sv,'condition_number',sv(1)/sv(end),...
  'bound_parameters',boundary,'start_log_solutions',solutions,'start_RSS',values,...
  'start_exitflags',flags,'all_starts_converged',all(flags>0),...
  'maximum_system_residual',lastres,'maximum_reciprocity_ohm',lastrec,...
  'evaluations',evaluations,'patch_qc',patchqc,'identity',identity,...
  'status','candidate_requires_pose_specific_mesh_and_quadrature_check',...
  'physical_validation',false,'source_rule',char(a.ExcitationScenario),...
  'excitation_scenario_is_measured',false,'equal_source_current_amplitudes_are_assumed',~strcmp(a.ExcitationScenario,'own_drive'));
 write_json(file,row);rows=append(rows,row);
 fprintf('POSE %s centre=%s RMSE=%.6g rho=%s\n',pose.pose_id,mat2str(pose.centre_lps_mm,5),row.RMSE_ohm,mat2str(exp(solutions(best,:)),5));
end
report=struct('identity',identity,'poses_requested',numel(grid.poses),...
 'method_reference','Nix 33.09 section 5.6: full CEM, adjoint derivatives, bounded log-resistivity least squares, two starts',...
 'pose_method_reference','Nix run_trkg4_inverse_inhale: joint tissue/centre/angle fitting; current Georg implementation profiles tissues over a discrete centre grid at the confirmed nominal angle',...
 'poses_solved',numel(rows),'geometry_failures',geometry_failures,...
 'full_declared_grid_complete',isempty(a.PoseIds)&&numel(rows)+numel(geometry_failures)==numel(grid.poses),...
 'global_continuous_optimum_certified',false,'quantitative_result_accepted',false,...
 'annotation_acceptance_changed',false,'confidence_intervals_calculated',false,...
 'required_next_check','pose-specific integration, both meshes, source cross-transfer, residual profiles and position refinement');
write_json(fullfile(out,['inverse_execution_' char(a.ExcitationScenario) '.local.json']),report);

 function [residual,J]=objective(x,rule)
  if nargin<2,rule=2;end
  x=x(:);if rule==2&&isequal(x,lastx),residual=lastZ-O.Z_ohm;J=lastJ;return;end
  Z=zeros(height(O),1);J=zeros(height(O),6);Own=Z;Counter=Z;
  for state=1:2
   state_x=x((state-1)*3+(1:3));rho=fixed;rho(active)=exp(state_x);
   n=size(S.mm.nodes,1);K=sparse(n,n);
   states=["inhale","exhale"];ids=find(O.state==states(state));
   hit=[];
   for previous=1:2
    saved=state_cache{previous};
    if ~isempty(saved)&&saved.rule==rule&&isequal(saved.logrho,state_x)&&...
     isequal(saved.records,O.record_id(ids))&&isequal(saved.axes,O.axis(ids)),hit=saved;break;end
   end
   if ~isempty(hit)
    state_cache{state}=hit;
    Z(ids)=hit.Z;J(ids,(state-1)*3+(1:3))=hit.J;Own(ids)=hit.Own;Counter(ids)=hit.Counter;
    lastres=max(lastres,hit.res);lastrec=max(lastrec,hit.rec);continue;
   end
   for m=1:5,K=K+S.parts{rule,m}/rho(m);end
   % Nix's full CEM/adjoint kernel; both cross axes use the same solve.
   records=unique(O.record_id(ids),'stable');
   for record=records'
    indices=ids(O.record_id(ids)==record);index=indices(1);
    pairs={[1,4],[2,3],drive{index},measure{index}};
    if O.is_cross(index),pairs={[1,4],[2,3],[5,8],[6,7]};end
    ports=zeros(size(C{index},1)-n,4);
    for p=1:4,ports(pairs{p},p)=[1;-1];end
    fprintf('JOINT forward state=%s record=%s integration=%d\n',states(state),record,rule);
    [T,Jport,checks]=trkg4_lateral_cem_response(K,C{index},S.parts(rule,:),rho,ports,active,1);
    lastres=max(lastres,checks.system_relative_residual);
    lastrec=max(lastrec,checks.reciprocity_ohm);
    for index=indices'
     d=3;if isequal(drive{index},[1,4]),d=1;end
     m=d+1;c=4-d;
     assert(isequal(measure{index},pairs{m}));
     own=T(m,d);cross=T(m,c);Own(index)=own;Counter(index)=cross;
     Jown=reshape(Jport(m,d,:),1,3);Jcross=reshape(Jport(m,c,:),1,3);
     [Z(index),J(index,(state-1)*3+(1:3))]=trkg4_two_source_readout(own,cross,Jown,Jcross,char(a.ExcitationScenario));
    end
   end
   state_cache{state}=struct('rule',rule,'logrho',state_x,'records',O.record_id(ids),'axes',O.axis(ids),...
    'Z',Z(ids),'J',J(ids,(state-1)*3+(1:3)),'Own',Own(ids),'Counter',Counter(ids),'res',lastres,'rec',lastrec);
  end
  lastx=x;lastZ=Z;lastJ=J;lastOwn=Own;lastCounter=Counter;evaluations=evaluations+1;residual=Z-O.Z_ohm;
  if rule~=2,lastx=[];end
 end

 function result=joint_search()
  addpath(fullfile(root,'tools','full_pose_scan'));
  projection=full_scan_context(pose_mm);
  identity.projection_sha256=trkg4_file_sha256(which('full_scan_project'));
  identity.projection_mex_sha256=trkg4_file_sha256(which('full_scan_project_mex'));
  identity.projection_context_sha256=trkg4_file_sha256(which('full_scan_context'));
  % jsondecode returns each three-coordinate JSON vector as a column.
  centres=reshape([grid.poses.centre_lps_mm],3,[])';
  assert(size(centres,1)==numel(grid.poses)&&size(centres,2)==3);
  author=struct([]);
  author=struct([]);
  if isempty(a.CentreFile)
   yz=centres(:,2:3);[~,refindex]=min(sum((yz-[0,-210]).^2,2));
   reference=centres(refindex,:);[c0,n0]=full_scan_project(projection,reference);
   u0=[0,0,1]-n0(3)*n0;u0=u0/norm(u0);v0=cross(n0,u0);v0=v0/norm(v0);
   uv=[(centres-c0)*u0',(centres-c0)*v0'];
   parameter_lower=[lb,min(uv,[],1),-20];parameter_upper=[ub,max(uv,[],1),20];
   chosen=refindex;
   while numel(chosen)<8
    distance=Inf(size(uv,1),1);
    for j=chosen(:)',distance=min(distance,sum((uv-uv(j,:)).^2,2));end
    [~,j]=max(distance);chosen(end+1)=j; %#ok<AGROW>
   end
   seeds=[repmat(starts(1,:),8,1),uv(chosen,:),zeros(8,1)];
   seeds=[seeds;repmat(starts(2,:),8,1),uv(chosen,:),zeros(8,1)];
  else
   centre_file=fullfile(root,char(a.CentreFile));author=jsondecode(fileread(centre_file));
   assert(~author.independently_measured,'This scenario must preserve the approximate-centre provenance');
   reference=reshape(author.centre_lps_mm,1,3);assert(all(isfinite(reference)));
   [c0,n0]=full_scan_project(projection,reference);
   identity.author_centre_sha256=trkg4_file_sha256(centre_file);
   identity.local_search_helper_sha256=trkg4_file_sha256(which('trkg4_georg_local_search_parameters'));
   [parameter_lower,parameter_upper,seeds]=trkg4_georg_local_search_parameters(starts,lb,ub,...
    a.CentreHalfWidthMM,a.AngleHalfWidthDeg);
   if ~isempty(a.WarmStartFile)
    warm_file=fullfile(root,char(a.WarmStartFile));warm=jsondecode(fileread(warm_file));
    if isfield(warm,'best'),warm=warm.best;end
    assert(strcmp(warm.identity.author_centre_sha256,identity.author_centre_sha256)&&...
     strcmp(warm.identity.observations_sha256,identity.observations_sha256),'Warm-start reference changed');
    q=reshape(warm.parameters,1,9);
    assert(all(q>=parameter_lower)&&all(q<=parameter_upper),'Warm start is outside the declared domain');
    seeds=[q;seeds];identity.warm_start_sha256=trkg4_file_sha256(warm_file);
   end
  end
  full_seeds=seeds;
  if ~isempty(a.StartIndices)
   assert(numel(unique(a.StartIndices))==numel(a.StartIndices)&&max(a.StartIndices)<=size(seeds,1),'Invalid start subset');
   seeds=seeds(a.StartIndices,:);
  end
  suffix=char(a.RunLabel);if ~isempty(suffix),suffix=['_' suffix];end
  dest=fullfile(out,['joint_' char(a.ExcitationScenario) suffix]);
  assert(~isfolder(dest),'Use a fresh output directory for the joint search');mkdir(dest);
  surface=stlread(fullfile(root,current.model_directory,'body_ct_surface.stl'));
  % The common FEM also contains artificial arms; lateral contacts must stay
  % on the original scanned torso, including after nearest-triangle projection.
  skinmodel=struct('nodes',surface.Points,'boundary',surface.ConnectivityList);
  skinpc=electrode_patch_context(skinmodel);
  skincontext=struct('nodes',surface.Points,'boundary',surface.ConnectivityList,...
   'normals',vertexNormal(surface),'boundary_nodes',unique(surface.ConnectivityList(:)),...
   'node_face_list',double(skinpc.node_face_list),'node_face_start',double(skinpc.node_face_start));
  skincontext.tree=KDTreeSearcher(skincontext.nodes(skincontext.boundary_nodes,:));
  clear skinpc skinmodel;
  midx=(min(surface.Points(:,1))+max(surface.Points(:,1)))/2;
  ct_z_limits=[min(surface.Points(:,3)),max(surface.Points(:,3))];
  replay={};force_fresh=false;
  if ~isempty(a.ResumeFrom)
   previous_directory=fullfile(out,char(a.ResumeFrom));
   previous_plan_file=fullfile(previous_directory,'plan.local.json');
   previous_plan=jsondecode(fileread(previous_plan_file));
   previous_identity=previous_plan.identity;names=fieldnames(previous_identity);
   for key=1:numel(names)
    name=names{key};if strcmp(name,'worker_sha256')||startsWith(name,'resume_'),continue;end
    assert(isfield(identity,name)&&isequal(identity.(name),previous_identity.(name)),...
     'trkg4:staleResumeInput',['Resume dependency changed: ' name]);
   end
   previous_starts=previous_plan.starts;
   if ~isempty(a.StartIndices)&&isequal(previous_starts,full_seeds)
    previous_starts=previous_starts(a.StartIndices,:);
   end
   assert(max(abs(previous_plan.lower(:)-parameter_lower(:)))<1e-12&&...
    max(abs(previous_plan.upper(:)-parameter_upper(:)))<1e-12&&...
    isequal(previous_starts,seeds),'trkg4:staleResumeSearch','Resume search bounds or start subset changed');
   archived_worker=fullfile(out,'source_snapshots',['run_georg_lateral_inverse_' previous_identity.worker_sha256 '.m']);
   assert(strcmp(trkg4_file_sha256(archived_worker),previous_identity.worker_sha256),...
    'trkg4:missingResumeSource','Original executed worker must be preserved');
   files=dir(fullfile(previous_directory,'evaluation_*.local.json'));sources=struct([]);
   for key=1:numel(files)
    source_file=fullfile(files(key).folder,files(key).name);saved=jsondecode(fileread(source_file));
    if ~any(strcmp(saved.status,{'numerical_candidate_not_validated','invalid_geometry'})),continue;end
    assert(isequal(saved.identity,previous_identity),'trkg4:staleResumeInput','Checkpoint identity differs from its run');
    replay{end+1}=saved; %#ok<AGROW>
    sources=append(sources,struct('path',source_file,'sha256',trkg4_file_sha256(source_file)));
   end
   assert(~isempty(replay),'trkg4:noCompletedResumePoints','No completed checkpoints found');
   manifest_file=fullfile(dest,'replay_sources.local.json');
   write_json(manifest_file,struct('previous_plan_sha256',trkg4_file_sha256(previous_plan_file),...
    'archived_worker_sha256',previous_identity.worker_sha256,'sources',sources,...
    'changed_worker_scope','checkpoint orchestration only; physical and search helpers unchanged'));
   identity.resume_manifest_sha256=trkg4_file_sha256(manifest_file);
   identity.resume_directory=char(a.ResumeFrom);
   identity.resume_lookup_sha256=trkg4_file_sha256(which('trkg4_replay_search_point'));
   fprintf('JOINT resume loaded %d completed checkpoints; interrupted points will be recomputed\n',numel(replay));
  end
  search_plan=struct('identity',identity,...
   'reference_centre_lps_mm',reference,'reference_is_measured',false,...
   'parameter_order',{{'lnrho1_inhale','lnrho2_inhale','lnrho3_inhale',...
    'lnrho1_exhale','lnrho2_exhale','lnrho3_exhale','centre_u_mm','centre_v_mm','angle_deg'}},...
   'lower',parameter_lower,'upper',parameter_upper,'starts',seeds,'budget',a.MaxEvaluations,...
   'objective','equal-weight mean squared residual of all 24 hold levels',...
   'numerical_policy',char(a.NumericalPolicy),'common_model_status',current.status,...
   'mesh_convergence_passed',current.mesh_convergence_passed,...
   'nix_depth_prior_transferred',false,'nix_slope_penalty_transferred',false,...
   'ct_cut_planes_z_mm',ct_z_limits,'lateral_cap_clearance_mm',3.7,...
   'angle_bounds_are_search_scenario',true,'confidence_interval',false);
  search_plan.author_centre=author;search_plan.projected_reference_centre_lps_mm=c0;
  search_plan.reference_projection_distance_mm=norm(c0-reference);
  search_plan.local_author_centre_search=~isempty(a.CentreFile);
  search_plan.included_start_indices=a.StartIndices;
  search_plan.full_original_starts=full_seeds;
  write_json(fullfile(dest,'plan.local.json'),search_plan);
  counter=0;bestvalue=Inf;bestrow=struct([]);
  [best,value,info]=trkg4_bounded_pattern_search(@joint_objective,seeds,parameter_lower,parameter_upper,...
   [.18,.25,.30,.18,.25,.30,10,10,5],[.005,.01,.01,.005,.01,.01,.5,.5,.5],a.MaxEvaluations);
  % Restore the best contact pose, then check quadrature at its fitted values.
  force_fresh=true;state_cache={[],[]};lastx=[];lastres=0;lastrec=0;
  value=joint_objective(best);
  assert(isfinite(value),'trkg4:resumeBestGeometryChanged','Best point must pass a fresh FEM evaluation');
  fine=lastZ;
  lastx=[];[coarse_residual,~]=objective(best(1:6),1);coarse=coarse_residual+O.Z_ohm;
  delta=abs(fine-coarse);tol=.02+.001*abs(fine);
  precision=struct('identity',identity,'best_parameters',best,'fine_Z_ohm',fine,'coarse_Z_ohm',coarse,...
   'difference_ohm',delta,'tolerance_ohm',tol,'quadrature_passed',all(delta<=tol),...
   'global_mesh_convergence',false,'quantitative_result_accepted',false);
  write_json(fullfile(dest,'fit_precision.local.json'),precision);
  result=struct('identity',identity,'best_parameters',best,'best_mean_square_ohm2',value,...
   'best',bestrow,'search',info,'full_declared_grid_complete',false,...
   'quantitative_result_accepted',false,'physical_validation',false,...
   'quadrature_passed_at_best',precision.quadrature_passed,'common_model_status',current.status,...
   'required_next_check','material and pose profiles, additional starts, pose-specific mesh and quadrature comparisons');
  write_json(fullfile(dest,'execution.local.json'),result);

  function value=joint_objective(q)
   counter=counter+1;file=fullfile(dest,sprintf('evaluation_%04d.local.json',counter));
   if ~force_fresh
    [saved,value,hit]=trkg4_replay_search_point(replay,q);
    if hit
     saved.source_evaluation=saved.evaluation;saved.evaluation=counter;
     saved.resume_orchestration_identity=identity;write_json(file,saved);
     if value<bestvalue,bestvalue=value;bestrow=saved;write_json(fullfile(dest,'best.local.json'),saved);end
     fprintf('JOINT replay evaluation=%d source=%d objective=%.8g\n',counter,saved.source_evaluation,value);
     return;
    end
   end
   row=struct('evaluation',counter,'parameters',q,'identity',identity,...
    'status','invalid_geometry','mean_square_ohm2',[],'pose',[],'patch_qc',[]);
   try
    candidate=trkg4_lateral_surface_pose(projection,reference,q(7:8),q(9));row.pose=candidate;
    points=vertcat(candidate.electrodes.centres_lps_mm);
    assert(all(points(:,1)<midx),'trkg4:poseOutsideRightTorso','A side contact exits the right CT surface');
    % The flat end closures are artificial surfaces. The existing 3.7 mm
    % maximum patch extent must fit entirely between the CT cut planes.
    row.clearance_from_CT_cut_planes_mm=min(points(:,3)-ct_z_limits(1),ct_z_limits(2)-points(:,3));
    assert(min(row.clearance_from_CT_cut_planes_mm)>3.7,...
     'trkg4:poseOnCTCap','A lateral patch can touch an artificial CT end closure');
    skinpoints=full_scan_project(skincontext,points);
    row.distance_to_CT_skin_mm=vecnorm(points-skinpoints,2,2);
    assert(max(row.distance_to_CT_skin_mm)<1,'trkg4:poseOutsideCTTorso','A lateral contact leaves the scanned torso');
    if isempty(lastpose)||~isequal(lastpose,q(7:9))
     [C,drive,measure,lastpatchqc]=montages(candidate,S.mm,arms,arm_names,O,contacts,plan,cfg,context,physical_inheritance);
     lastpose=q(7:9);state_cache={[],[]};
    end
    row.patch_qc=lastpatchqc;
    if ~lastpatchqc.passed,value=Inf;write_json(file,row);fprintf('JOINT evaluation=%d contact QC failed\n',counter);return;end
   catch exception
    if ~any(strcmp(exception.identifier,{'trkg4:poseOutsideRightTorso','trkg4:poseOutsideCTTorso','trkg4:poseOnCTCap','trkg4:invalidSurfaceNormal',...
      'trkg4:patchOverlap','electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall'}))
     rethrow(exception);
    end
    row.failure_code=exception.identifier;value=Inf;write_json(file,row);return;
   end
   row.status='forward_calculation_running';write_json(file,row);
   fprintf('JOINT candidate=%d pose=%s rho=%s\n',counter,mat2str(q(7:9),5),mat2str(exp(q(1:6)),5));
   lastx=[];[r,J]=objective(q(1:6));value=mean(r.^2);
   row.status='numerical_candidate_not_validated';row.mean_square_ohm2=value;
   row.rho_inhale_ohm_m=exp(q(1:3));row.rho_exhale_ohm_m=exp(q(4:6));
   row.predicted_Z_ohm=lastZ;row.target_Z_ohm=O.Z_ohm;row.residual_ohm=r;
   row.jacobian_logrho=J;row.singular_values=svd(J);
   row.own_transfer_ohm=lastOwn;row.counter_transfer_ohm=lastCounter;
   row.maximum_system_residual=lastres;row.maximum_reciprocity_ohm=lastrec;
   write_json(file,row);
   if value<bestvalue||force_fresh,bestvalue=value;bestrow=row;write_json(fullfile(dest,'best.local.json'),row);end
   fprintf('JOINT evaluation=%d RMSE=%.8g centre=%s angle=%.4g rho=%s\n',...
    counter,sqrt(value),mat2str(candidate.centre_lps_mm,5),q(9),mat2str(exp(q(1:6)),5));
  end
 end

 function result=check_candidate()
  source_file=fullfile(root,char(a.CheckCandidateFile));saved=jsondecode(fileread(source_file));
  if isfield(saved,'best'),saved=saved.best;end
  assert(strcmp(saved.identity.observations_sha256,identity.observations_sha256),'Forward-check observations changed');
  for name={'cem_response_sha256','readout_sha256','contact_file_sha256'}
   assert(strcmp(saved.identity.(name{1}),identity.(name{1})),'Forward-check physics changed');
  end
  assert(strcmp(saved.identity.excitation_scenario,char(a.ExcitationScenario)),'Forward-check excitation changed');
  candidate=saved.pose;
  identity.checked_candidate_sha256=trkg4_file_sha256(source_file);
  identity.check_material_helper_sha256=trkg4_file_sha256(which('trkg4_georg_check_materials'));
  [C,drive,measure,qc]=montages(candidate,S.mm,arms,arm_names,O,contacts,plan,cfg,context,physical_inheritance);
  suffix=char(a.RunLabel);assert(~isempty(suffix),'Fixed checks require a unique RunLabel');
  file=fullfile(out,['forward_check_' char(a.ExcitationScenario) '_' suffix '.local.json']);
  assert(~isfile(file),'Use a fresh label for a forward check');
  result=struct('identity',identity,'pose',candidate,'parameters',saved.parameters,'patch_qc',qc,...
   'numerical_quadrature_points',plan.numerical_quadrature_points,'quantitative_result_accepted',false,...
   'global_mesh_convergence',false,'status','contact_geometry_failed');
  if ~qc.passed,write_json(file,result);return;end
  [rf,J]=objective(saved.parameters(1:6),2);fine=lastZ;
  lastx=[];[rc,~]=objective(saved.parameters(1:6),1);coarse=lastZ;
  result.status='fixed_parameter_numerical_check';
  result.fine_Z_ohm=fine;result.coarse_Z_ohm=coarse;
  result.target_Z_ohm=O.Z_ohm;result.residual_ohm=rf;
  result.mean_square_ohm2=mean(rf.^2);result.jacobian_logrho=J;
  result.difference_ohm=abs(fine-coarse);result.tolerance_ohm=.02+.001*abs(fine);
  result.quadrature_passed=all(result.difference_ohm<=result.tolerance_ohm);
  result.maximum_system_residual=lastres;result.maximum_reciprocity_ohm=lastrec;
  result.system_checks_passed=lastres<1e-7&&lastrec<1e-7;
  result.shared_geometry_changed=false;result.parameters_optimized=false;
  write_json(file,result);
 end
end

function [C,drive,measure,qc]=montages(pose,mm,arms,arm_names,O,contacts,plan,cfg,context,inheritance)
if nargin<10,inheritance=[];end
if isfield(contacts,'arm_physical_centres_lps_mm')
 [C,drive,measure,qc]=trkg4_georg_physical_montages(pose,mm,contacts,O,plan,cfg,context,inheritance);
 return;
end
C=cell(height(O),1);drive=C;measure=C;rows=struct([]);patches=struct([]);
for i=1:height(O)
 first=5;if O.is_cross(i)&&O.axis(i)=="longitudinal",first=1;end
 previous=find(O.record_id(1:i-1)==O.record_id(i),1);
 if ~isempty(previous)
  C{i}=C{previous};drive{i}=first+[0,3];measure{i}=first+[1,2];continue;
 end
 points=pose.electrodes([pose.electrodes.size_mm]==O.size_mm(i)&strcmp({pose.electrodes.axis},O.axis(i))).centres_lps_mm;
 if O.is_cross(i)
  primary=pose.electrodes([pose.electrodes.size_mm]==O.size_mm(i)&strcmp({pose.electrodes.axis},'longitudinal')).centres_lps_mm;
  transverse=pose.electrodes([pose.electrodes.size_mm]==O.size_mm(i)&strcmp({pose.electrodes.axis},'perpendicular')).centres_lps_mm;
  groups=num2cell([primary;transverse],2);first=1+4*(O.axis(i)=="perpendicular");
 else
  groups=[arms{strcmp(arm_names,O.arm_montage(i))};num2cell(points,2)];first=5;
 end
 xyz=[];gids=[];
 for g=1:numel(groups)
  p=reshape(groups{g},[],3);xyz=[xyz;p];gids=[gids;repmat(g,size(p,1),1)]; %#ok<AGROW>
 end
 [el,faces,certificate]=trkg4_local_exact_patches(context,xyz,contacts.nominal_area_mm2,plan.contact_impedance_ohm_m2);
 patches=append(patches,struct('row',i,'certificate',certificate));
 grouped=struct([]);
 for e=1:numel(el)
  f=mm.boundary(faces{e},:);v=mm.nodes;
  area=vecnorm(cross(v(f(:,2),:)-v(f(:,1),:),v(f(:,3),:)-v(f(:,1),:),2),2,2)/2;
  centroid=sum((v(f(:,1),:)+v(f(:,2),:)+v(f(:,3),:))/3.*area,1)/sum(area);
  row=struct('row',i,'contact',e,'area_error',abs(sum(area)/contacts.nominal_area_mm2-1),...
   'centroid_offset_mm',norm(centroid-xyz(e,:)),'extent_mm',max(vecnorm(v(el(e).nodes,:)-xyz(e,:),2,2)));
  rows=append(rows,row);
 end
 for g=1:numel(groups)
  ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
  grouped(g).faces=mm.boundary(vertcat(faces{ix}),:);grouped(g).z_contact=plan.contact_impedance_ohm_m2;
 end
 assert(numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes)),...
  'trkg4:patchOverlap','Physical electrode contacts overlap');
 temp=mm;temp.electrode=grouped;C{i}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(temp,cfg));
 drive{i}=first+[0,3];measure{i}=first+[1,2];
end
qc=struct('passed',max([rows.area_error])<.12&&max([rows.centroid_offset_mm])<1&&max([rows.extent_mm])<3.7,...
 'contacts',rows,'exact_local_patch_certificates',patches);
end

function rows=append(rows,row)
if isempty(rows),rows=row;else,rows(end+1)=row;end
end

function write_json(file,value)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
