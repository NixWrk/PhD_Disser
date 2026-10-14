function report=run_georg_quality_forward(varargin)
% Fixed-parameter FEM controls, with permanent physical contacts and exact
% parent-face inheritance. Does not fit parameters or accept a model.
p=inputParser;
addParameter(p,'ModelDir','');addParameter(p,'InputDir','');
addParameter(p,'OutputFile','');addParameter(p,'ReferenceFile','');
addParameter(p,'AuthorCentreFile','');addParameter(p,'PreviousFitFile','');
addParameter(p,'CheckDerivatives',false);
addParameter(p,'Stage','forward');parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');cfg.frequency_hz=100000;root=cfg.project_root;
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
addpath(fullfile(root,'tools'));addpath(fullfile(root,'tools','full_pose_scan'));
base=resolve(a.ModelDir);planfile=fullfile(base,'plan.local.json');plan=jsondecode(fileread(planfile));
assert(plan.shape_QC_passed&&plan.frequency_hz==100000);
q=plan.numerical_quadrature_points(:)';
if isequal(q,[1024,4096])
 [cache,prepared]=trkg4_georg_lateral_materials(cfg,base);
else
 [cache,prepared]=trkg4_georg_check_materials(cfg,base);
end
report=struct('prepared',prepared,'prepared_file',cache);
if strcmp(a.Stage,'materials'),return;end
assert(any(strcmp(a.Stage,{'forward','contacts'})));
S=load(cache);inputdir=resolve(a.InputDir);observationfile=fullfile(inputdir,'observations.local.csv');
O=readtable(observationfile,'TextType','string');O=O(O.state~="quiet_breathing",:);assert(height(O)==24);
O.is_cross=lower(string(O.is_cross))=="true"|string(O.is_cross)=="1";
contactfile=plan.contacts_file;contacts=jsondecode(fileread(contactfile));
referencefile=resolve(a.ReferenceFile);
reference_mm=S.mm;M=load(plan.pose_reference_mesh);reference_mm.nodes=M.nodes_mm;
reference_mm.elems=double(M.elems);reference_mm.boundary=double(M.boundary);clear M;
if isfile(referencefile)
 reference=jsondecode(fileread(referencefile));
 assert(strcmp(reference.pose_reference_mesh_sha256,trkg4_file_sha256(plan.pose_reference_mesh)));
else
 authorfile=resolve(a.AuthorCentreFile);author=jsondecode(fileread(authorfile));
 previousfile=resolve(a.PreviousFitFile);prior=jsondecode(fileread(previousfile));
 if isfield(prior,'best'),prior=prior.best;end
 x=reshape(prior.parameters,1,9);projection=full_scan_context(reference_mm);
 pose=trkg4_lateral_surface_pose(projection,author.centre_lps_mm,x(7:8),x(9));
 reference=struct('pose',pose,'parameters',x,'author_centre_sha256',trkg4_file_sha256(authorfile),...
  'previous_fit_sha256',trkg4_file_sha256(previousfile),...
  'pose_reference_mesh_sha256',trkg4_file_sha256(plan.pose_reference_mesh),...
  'observations_sha256',trkg4_file_sha256(observationfile),...
  'common_parameters_not_fitted_here',true,'independent_placement_measurement',false);
 write_json(referencefile,reference);
end
assert(strcmp(reference.observations_sha256,trkg4_file_sha256(observationfile)));
context=trkg4_local_patch_context(S.mm);
[inherited,contact_identity]=trkg4_georg_contact_inheritance(S.mm,plan,fullfile(base,'mesh.mat'));
[C,drive,measure,patchqc,patches]=trkg4_georg_physical_montages(reference.pose,S.mm,contacts,O,plan,cfg,context,inherited);
if strcmp(a.Stage,'contacts')
 report=struct('patch_qc',patchqc,'patches',patches,'pose',reference.pose,'parameters',reference.parameters);
 write_json(resolve(a.OutputFile),report);return;
end
assert(patchqc.passed,'Fixed physical contact QC failed');
ref=jsondecode(fileread(fullfile(base,'reference_materials.local.json')));
active=[1,2,5];states=["inhale","exhale"];
predictions=zeros(height(O),2);J=zeros(height(O),6);allchecks=struct([]);derivatives=struct([]);
for rule=1:2
 for state=1:2
  if state==2&&isequal(reference.parameters(1:3),reference.parameters(4:6))
   for i=find(O.state==states(2))'
    previous=find(O.state==states(1)&O.record_id==O.record_id(i)&O.axis==O.axis(i));assert(isscalar(previous));
    predictions(i,rule)=predictions(previous,rule);
    if rule==2,J(i,4:6)=J(previous,1:3);end
   end
   fprintf('QUALITY equal phase conductivities: exact reference state reuse\n');continue;
  end
  rho=ref.rho_ohm_m(:);rho(active)=exp(reference.parameters((state-1)*3+(1:3)));
  n=size(S.mm.nodes,1);K=sparse(n,n);for m=1:5,K=K+S.parts{rule,m}/rho(m);end
  ids=find(O.state==states(state));records=unique(O.record_id(ids),'stable');
  for record=records'
   ix=ids(O.record_id(ids)==record);first=ix(1);groups=size(C{first},1)-n;
   pairs={[1,4],[2,3],drive{first},measure{first}};
   if O.is_cross(first),pairs={[1,4],[2,3],[5,8],[6,7]};end
   ports=zeros(groups,4);for k=1:4,ports(pairs{k},k)=[1;-1];end
   fprintf('QUALITY forward q=%d state=%s record=%s groups=%d\n',q(rule),states(state),record,groups);
   [T,derivative,check]=trkg4_lateral_cem_response(K,C{first},S.parts(rule,:),rho,ports,active,1);
   for i=ix'
    source=3;if O.is_cross(i)&&O.axis(i)=="longitudinal",source=1;end
    predictions(i,rule)=T(source+1,source);
    if rule==2,J(i,(state-1)*3+(1:3))=reshape(derivative(source+1,source,:),1,3);end
   end
   allchecks=append(allchecks,struct('rule',q(rule),'state',char(states(state)),...
    'record_id',char(record),'check',check));
   if a.CheckDerivatives&&rule==2&&state==1&&any(record==["SCV2-G-006","SCV2-G-011"])
    errors=zeros(1,3);indices=sub2ind([4,4],[2,4],[1,3]);h=1e-3;
    for k=1:3
     rp=rho;rm=rho;rp(active(k))=rp(active(k))*exp(h);rm(active(k))=rm(active(k))*exp(-h);
     kp=sparse(n,n);km=kp;
     for j=1:5,kp=kp+S.parts{rule,j}/rp(j);km=km+S.parts{rule,j}/rm(j);end
     Tp=trkg4_lateral_cem_response(kp,C{first},S.parts(rule,:),rp,ports,[],1);
     Tm=trkg4_lateral_cem_response(km,C{first},S.parts(rule,:),rm,ports,[],1);
     fd=(Tp-Tm)/(2*h);adj=derivative(:,:,k);
     errors(k)=norm(fd(indices)-adj(indices))/norm(fd(indices));
     clear kp km;
    end
    derivatives=append(derivatives,struct('record_id',char(record),'relative_errors',errors,...
     'tolerance',1e-4,'passed',all(errors<1e-4),'log_step',h,'material_indices',active));
   end
  end
 end
end
delta=abs(predictions(:,2)-predictions(:,1));tolerance=.02+.001*abs(predictions(:,2));
checks=[allchecks.check];
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'plan_sha256',trkg4_file_sha256(planfile),'prepared_sha256',trkg4_file_sha256(cache),...
 'physical_contacts_sha256',trkg4_file_sha256(contactfile),...
 'contact_inheritance',contact_identity,...
 'reference_sha256',trkg4_file_sha256(referencefile),'observations_sha256',trkg4_file_sha256(observationfile),...
 'physical_montage_helper_sha256',trkg4_file_sha256(which('trkg4_georg_physical_montages')));
report=struct('identity',identity,'parameters',reference.parameters,'pose',reference.pose,...
 'record_ids',O.record_id,'axes',O.axis,'states',O.state,'target_Z_ohm',O.Z_ohm,...
 'coarse_Z_ohm',predictions(:,1),'fine_Z_ohm',predictions(:,2),...
 'difference_ohm',delta,'tolerance_ohm',tolerance,'quadrature_passed',all(delta<=tolerance),...
 'jacobian_logrho',J,'patch_qc',patchqc,'patches',patches,'system_checks',allchecks,'derivatives',derivatives,...
 'maximum_system_residual',max([checks.system_relative_residual]),...
 'maximum_reciprocity_ohm',max([checks.reciprocity_ohm]),...
 'mean_square_ohm2',mean((predictions(:,2)-O.Z_ohm).^2),...
 'parameters_optimized',false,'quantitative_result_accepted',false,...
 'source_rule','own_drive','source_rule_is_experimental_working_assumption',true);
report.system_checks_passed=report.maximum_system_residual<1e-7&&report.maximum_reciprocity_ohm<1e-7;
write_json(resolve(a.OutputFile),report);

 function answer=resolve(value)
  answer=char(value);
  if isempty(regexp(answer,'^([A-Za-z]:[\\/]|[\\/])','once')),answer=fullfile(root,answer);end
 end
end

function answer=append(answer,row)
if isempty(answer),answer=row;else,answer(end+1)=row;end
end
function write_json(path,value)
assert(~isfile(path),'Use a fresh immutable forward-control output');
fid=fopen(path,'w','n','UTF-8');assert(fid>0);guard=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
