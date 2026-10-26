function report=run_scorpion_contact_sensitivity(varargin)
% New explicit +/-5 mm LPS-Y CEM placements; parent schedules every solve.
% prepare: canonical finite patches only. solve: two canonical full CEM solves.
% No remeshing, material reassembly, threshold relaxation or new PDE kernel.
p=inputParser;
addParameter(p,'Action','prepare');addParameter(p,'ModelRoot','');
addParameter(p,'CEMInputFile','');addParameter(p,'CaseID','L140_TT2x2');
addParameter(p,'PreparedDir','');addParameter(p,'OutputDir','');
addParameter(p,'SolverLockFile','');
parse(p,varargin{:});a=p.Results;
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'tools'),fullfile(root,'vendor_stl_eidors'));
base=canonical(a.ModelRoot);out=canonical(a.OutputDir);
assert(isfolder(base)&&~isfolder(out)&&~isfile(out),'scorpion:sensitivityOutput','Use a fresh output directory');
assert(strcmp(a.CaseID,'L140_TT2x2'),'scorpion:sensitivityCase','Limited pilot requires L140_TT2x2');
assert(any(strcmp(a.Action,{'prepare','solve'})),'scorpion:sensitivityAction','Use prepare or solve');
if strcmp(a.Action,'solve')
 % Acquire before loading large matrices; refuse a busy slot, never queue a
 % second unscheduled task behind the parent. All callers use the same lock.
 lockpath=char(a.SolverLockFile);
 if isempty(lockpath)
  lockpath=fullfile(root,'output','exploratory','scorpion_session_runtime_20261026','sparse_solver.lock');
 end
 assert(isfolder(fileparts(lockpath)),'scorpion:solverLock','Parent must establish the shared sparse-solver slot');
 lockfile=javaObject('java.io.RandomAccessFile',lockpath,'rw');channel=lockfile.getChannel();
 try,lease=channel.tryLock();catch exception,channel.close();lockfile.close();rethrow(exception);end
 if isempty(lease),channel.close();lockfile.close();error('scorpion:solverBusy','Parent sparse-solver slot is busy');end
 resource_guard=onCleanup(@()release_resource(lease,channel,lockfile)); %#ok<NASGU>
end
mkdir(out);
try
 if strcmp(a.Action,'prepare'),report=prepare(base,out,a);else,report=solve(base,out,a);end
catch exception
 write_json(fullfile(out,'failed.local.json'),struct('action',a.Action,'identifier',exception.identifier,...
  'message',exception.message,'FEM_was_run',false,'quantitative_result_accepted',false));
 rethrow(exception);
end
end

function plan=prepare(base,out,a)
cemfile=fullfile(base,'cem','cem_contacts.mat');cemsha=trkg4_file_sha256(cemfile);
manifestfile=fullfile(base,'cem','cem_contacts_manifest.local.json');manifestsha=trkg4_file_sha256(manifestfile);manifest=read_json(manifestfile);
assert(manifest.CEM_contacts_QC_passed,'scorpion:cemQC','Baseline CEM must pass QC');
verify_files(manifest.products);verify_files(manifest.source_files);
inputfile=char(a.CEMInputFile);if isempty(inputfile),inputfile=manifest.input_json;end
inputfile=canonical(inputfile);inputsha=trkg4_file_sha256(inputfile);source=read_json(inputfile);
assert(strcmp(inputsha,manifest.input_sha256),'scorpion:cemIdentity','Baseline input identity mismatch');
data=load(cemfile,'result');c=data.result;clear data;
assert(strcmp(source.subject_alias,manifest.subject_alias)&&strcmp(source.subject_alias,c.input_contract.subject_alias),...
 'scorpion:subject','CEM subject mismatch');
verify_files(source.source_files);verify_files(source.input_products);
scenario_id=[a.CaseID '_lateral'];sindex=find(strcmp({source.scenarios.id},scenario_id));
assert(numel(sindex)==1,'scorpion:scenario','Exactly one primary L140_TT2x2 lateral scenario required');
oldindex=find(cellfun(@(s)strcmp(s.id,scenario_id),c.scenarios));
assert(numel(oldindex)==1,'scorpion:scenario','Prepared primary scenario missing');
old=c.scenarios{oldindex};
mm=struct('nodes',c.nodes_lps_m*1000,'elems',c.elems,'boundary',c.boundary);
ctx=trkg4_local_patch_context(mm);allfaces=(1:size(mm.boundary,1))';
labels={'A','M','N','B'};moved_ids=cell(4,1);
for k=1:4
 group=find(strcmp({old.ports.id},labels{k}));assert(numel(group)==1);
 members=string_list(old.ports(group).contact_ids);assert(numel(members)==1,'scorpion:lateralPad','One physical pad per lateral port required');
 moved_ids{k}=members{1};
end
assert(numel(unique(moved_ids))==4,'scorpion:lateralPad','Distinct lateral pads required');
before=zeros(4,3);normal=zeros(4,3);oldfaces=cell(4,1);
for k=1:4
 q=find(cellfun(@(v)strcmp(v.id,moved_ids{k}),old.physical_contact_QC));assert(numel(q)==1);
 row=old.physical_contact_QC{q};before(k,:)=row.fem_projected_lps_mm;
 oldfaces{k}=row.boundary_face_ids;
 [distance,~,face]=trkg4_local_triangle_distance(ctx,before(k,:),allfaces);
 assert(distance<1e-8,'scorpion:surface','Baseline centre is not on the same FEM surface');
 f=mm.boundary(face,:);n=cross(mm.nodes(f(2),:)-mm.nodes(f(1),:),mm.nodes(f(3),:)-mm.nodes(f(1),:));
 normal(k,:)=n/norm(n);
end
outputs=struct([]);signs=[-1,1];
for direction=1:2
 delta=[0,5*signs(direction),0];tag=sprintf('Y_%s5mm',choose(signs(direction)<0,'minus','plus'));
 contract=source;contract.scenarios=source.scenarios(sindex);contract.scenarios.id=[a.CaseID '_' tag '_lateral'];
 selected=string_list(contract.scenarios.present_contact_ids);
 contract.contact_catalog=source.contact_catalog(ismember({source.contact_catalog.id},selected));
 shifts=struct([]);after=zeros(4,3);
 for k=1:4
  ix=find(strcmp({contract.contact_catalog.id},moved_ids{k}));assert(numel(ix)==1);
  q=contract.contact_catalog(ix);assert(abs(q.nominal_area_mm2-pi*2.5^2)<1e-12,'scorpion:area','Only 5mm disks allowed');
  raw=before(k,:)+delta;
  [distance,point,face]=trkg4_local_triangle_distance(ctx,raw,allfaces);
  after(k,:)=point;actual=point-before(k,:);nv=dot(actual,normal(k,:))*normal(k,:);
  shifts=append(shifts,struct('id',q.id,'baseline_fem_lps_mm',before(k,:),'requested_LPS_shift_mm',delta,...
   'raw_scenario_lps_mm',raw,'new_projected_lps_mm',point,'projection_distance_mm',distance,...
   'projected_boundary_face_id',face,'actual_shift_mm',actual,'actual_distance_mm',norm(actual),...
   'baseline_face_unit_normal',normal(k,:),'normal_shift_mm',nv,'signed_normal_component_mm',dot(actual,normal(k,:)),...
   'tangential_shift_mm',actual-nv,'tangential_distance_mm',norm(actual-nv),...
   'baseline_raw_lps_mm',q.raw_lps_mm,'baseline_source_projected_lps_mm',q.source_projected_lps_mm));
  q.raw_lps_mm=raw;q.source_projected_lps_mm=point;
  % original_projected_lps_mm remains the historical source and is explicitly
  % distinguished from this scenario's raw and newly projected centres.
  q.provenance=[char(q.provenance) '; placement scenario: new explicit LPS-Y shift, projected on unchanged FEM surface; not measured installation error'];
  contract.contact_catalog(ix)=q;
 end
 contract.reconfiguration_provenance=struct('schema','scorpion_placement_reconfiguration_v1',...
  'baseline_cem_input_path',inputfile,'baseline_cem_input_sha256',inputsha,...
  'baseline_cem_mat_path',cemfile,'baseline_cem_mat_sha256',cemsha,...
  'baseline_cem_manifest_path',manifestfile,'baseline_cem_manifest_sha256',manifestsha,...
  'unchanged_CT_mesh_materials',true,'requested_axis_LPS',[0,1,0],'requested_offset_mm',5*signs(direction),...
  'offset_is_confidence_interval',false,'surface_arc_length_is_not_prescribed',true,...
  'shift_details',shifts,'baseline_distance_matrix_mm',distances(before),...
  'new_distance_matrix_mm',distances(after),'distance_matrix_change_mm',distances(after)-distances(before));
 contract.source_files=append(contract.source_files,struct('path',inputfile,'sha256',inputsha));
 contract.source_files=append(contract.source_files,struct('path',cemfile,'sha256',cemsha));
 contract.source_files=append(contract.source_files,struct('path',manifestfile,'sha256',manifestsha));
 inputdir=fullfile(out,[tag '_input']);mkdir(inputdir);newinput=fullfile(inputdir,'cem_contacts_input.local.json');
 write_json(newinput,contract);
 write_json(fullfile(inputdir,'preparation.local.json'),struct('schema','ct_fem_preparation_artifact_v1',...
  'stage','cem_contacts_input','status','prepared_not_validated','subject_alias',source.subject_alias,...
  'product_files',struct('path',newinput,'sha256',trkg4_file_sha256(newinput)),...
  'reconfiguration_provenance',contract.reconfiguration_provenance,'FEM_was_run',false));
 cemout=fullfile(out,[tag '_cem']);passed=false;failure=struct();
 try
  run_ct_fem_contacts_prepare(newinput,cemout);
  fresh=load(fullfile(cemout,'cem_contacts.mat'),'result');fresh=fresh.result.scenarios{1};
  for k=1:4
   ix=find(cellfun(@(v)strcmp(v.id,moved_ids{k}),fresh.physical_contact_QC));
   assert(numel(ix)==1&&~isequal(oldfaces{k}(:),fresh.physical_contact_QC{ix}.boundary_face_ids(:)),...
    'scorpion:footprintUnchanged','Shifted pad reused the baseline footprint; refine rather than relabel');
  end
  passed=true;
 catch exception
  failure=struct('identifier',exception.identifier,'message',exception.message,...
   'refine_or_change_scenario_required',true,'tolerances_relaxed',false);
 end
 entry=struct('scenario_id',contract.scenarios.id,'input_json',newinput,'input_sha256',trkg4_file_sha256(newinput),...
  'cem_directory',cemout,'CEM_contacts_QC_passed',passed,'failure',failure,...
  'placement',contract.reconfiguration_provenance);
 if passed
  entry.cem_sha256=trkg4_file_sha256(fullfile(cemout,'cem_contacts.mat'));
 else,entry.cem_sha256='';end
 outputs=append(outputs,entry);
end
assert(strcmp(trkg4_file_sha256(cemfile),cemsha)&&strcmp(trkg4_file_sha256(inputfile),inputsha));
assert(strcmp(trkg4_file_sha256(manifestfile),manifestsha));
plan=struct('schema','scorpion_contact_sensitivity_plan_v1','subject_alias',source.subject_alias,...
 'case_id',a.CaseID,'model_root',base,'baseline_cem_mat_sha256',cemsha,...
 'baseline_cem_input_sha256',inputsha,'scenarios',outputs,'FEM_was_run',false,...
 'CEM_contacts_QC_passed',all([outputs.CEM_contacts_QC_passed]),'quantitative_result_accepted',false,...
 'offsets_are_confidence_intervals',false,'mesh_materials_unchanged',true,...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
write_json(fullfile(out,'contact_plan.local.json'),plan);
end

function report=solve(base,out,a)
planfile=fullfile(canonical(a.PreparedDir),'contact_plan.local.json');plansha=trkg4_file_sha256(planfile);plan=read_json(planfile);
assert(strcmp(plan.schema,'scorpion_contact_sensitivity_plan_v1')&&plan.CEM_contacts_QC_passed&&numel(plan.scenarios)==2,...
 'scorpion:placementQC','Both new placements must pass unchanged finite-contact QC');
assert(strcmp(trkg4_file_sha256([mfilename('fullpath') '.m']),plan.worker_sha256),...
 'scorpion:workerIdentity','Placement worker changed after preparation');
assert(strcmp(canonical(plan.model_root),base)&&strcmp(plan.case_id,a.CaseID));
assert(strcmp(trkg4_file_sha256(fullfile(base,'cem','cem_contacts.mat')),plan.baseline_cem_mat_sha256));
forwardfile=fullfile(base,'electrical','forward.local.json');forwardsha=trkg4_file_sha256(forwardfile);forward=read_json(forwardfile);
assert(forward.FEM_was_run&&forward.frequency_hz==100000&&strcmp(forward.subject_alias,plan.subject_alias));
assert(strcmp(forward.source_relative_phase,'unknown')&&forward.own_drive_is_working_assumption);
ix=find(strcmp({forward.answers.case_id},a.CaseID)&strcmp({forward.answers.state},'inhale')&[forward.answers.rule]==4096);
assert(numel(ix)==1,'scorpion:baseline','One q4096 inhale baseline required');baseline=forward.answers(ix);
assert(isequal(double(forward.derivative_checks.active_material_indices(:)'),[1,2,4,5]));
cfg=trkg4_config(plan.subject_alias);cfg.frequency_hz=100000;trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
materialfile=fullfile(base,'electrical','prepared_materials.mat');materialsha=trkg4_file_sha256(materialfile);
data=load(materialfile,'parts','mm','identity');parts=data.parts;mm=data.mm;
assert(isequal(size(parts),[2,5])&&strcmp(data.identity.subject_alias,plan.subject_alias));
assert(strcmp(data.identity.cem_sha256,plan.baseline_cem_mat_sha256),'scorpion:materialsIdentity','Material preparation is bound to another CEM');
for field={'mesh_sha256','reference_sha256','inputs_sha256','response_kernel_sha256',...
 'field_kernel_sha256','scale_kernel_sha256','materials_manifest_sha256','cem_manifest_sha256'}
 if isfield(forward.identity,field{1})
  assert(isfield(data.identity,field{1})&&strcmp(data.identity.(field{1}),forward.identity.(field{1})),...
   'scorpion:materialsIdentity','Prepared input provenance differs from baseline');
 end
end
assert(strcmp(trkg4_file_sha256(fullfile(base,'reference_materials.local.json')),forward.identity.reference_sha256));
if isfield(forward.identity,'materials_manifest_sha256')
 assert(strcmp(trkg4_file_sha256(fullfile(base,'materials','preparation.local.json')),forward.identity.materials_manifest_sha256));
end
if isfield(forward.identity,'cem_manifest_sha256')
 assert(strcmp(trkg4_file_sha256(fullfile(base,'cem','cem_contacts_manifest.local.json')),forward.identity.cem_manifest_sha256));
end
assert(strcmp(trkg4_file_sha256(which('system_mat_fields')),forward.identity.field_kernel_sha256));
if isfield(forward.identity,'scale_kernel_sha256')
 assert(strcmp(trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')),forward.identity.scale_kernel_sha256));
end
response_sha=trkg4_file_sha256(which('trkg4_lateral_cem_response'));
assert(strcmp(response_sha,forward.identity.response_kernel_sha256));
rho=baseline.rho_ohm_m(:);K=sparse(size(mm.nodes,1),size(mm.nodes,1));
for k=1:5,K=K+parts{2,k}/rho(k);end
answers=struct([]);
for index=1:2
 scenario=plan.scenarios(index);assert(scenario.CEM_contacts_QC_passed);
 assert(strcmp(trkg4_file_sha256(scenario.input_json),scenario.input_sha256));
 cemfile=fullfile(scenario.cem_directory,'cem_contacts.mat');assert(strcmp(trkg4_file_sha256(cemfile),scenario.cem_sha256));
 cm=read_json(fullfile(scenario.cem_directory,'cem_contacts_manifest.local.json'));verify_files(cm.products);verify_files(cm.source_files);
 contact=load(cemfile,'result');contact=contact.result;
 assert(strcmp(contact.input_contract.subject_alias,plan.subject_alias)&&contact.qc.CEM_contacts_QC_passed);
 assert(isequal(size(mm.nodes),size(contact.nodes_lps_m))&&max(abs(mm.nodes(:)/1000-contact.nodes_lps_m(:)))<1e-12);
 assert(isequal(mm.elems,contact.elems)&&isequal(mm.boundary,contact.boundary),'scorpion:meshIdentity','Placement mesh changed');
 s=contact.scenarios{1};ports=make_ports(s.ports);
 [T,J,qc]=trkg4_lateral_cem_response(K,s.grouped_contact_matrix_SI,parts(2,:),rho,ports,[1,2,4,5],1);
 answers=append(answers,struct('scenario_id',scenario.scenario_id,'state','inhale','rule',4096,...
  'rho_ohm_m',rho','transfer_matrix_ohm',T,'jacobian_logrho',J,'system_checks',qc,...
  'placement',scenario.placement,'new_cem_mat_sha256',scenario.cem_sha256));
end
assert(strcmp(trkg4_file_sha256(materialfile),materialsha)&&strcmp(trkg4_file_sha256(forwardfile),forwardsha));
assert(strcmp(trkg4_file_sha256(planfile),plansha)&&strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),response_sha));
report=struct('schema','scorpion_contact_sensitivity_v1','subject_alias',plan.subject_alias,'case_id',a.CaseID,...
 'baseline_forward_sha256',forwardsha,'prepared_materials_sha256',materialsha,'plan_sha256',plansha,...
 'baseline',baseline,'answers',answers,'FEM_was_run',true,'frequency_hz',100000,...
 'source_rule','own_drive','source_relative_phase','unknown','active_material_indices',[1,2,4,5],...
 'bone_fixed',true,'CT_mesh_materials_unchanged',true,'offsets_are_confidence_intervals',false,...
 'numerical_mesh_convergence_proven',false,'quantitative_result_accepted',false,'clinical_validation',false);
write_json(fullfile(out,'contact_sensitivity.local.json'),report);
end

function P=make_ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft'))
 pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else
 pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};
end
P=zeros(numel(groups),4);
for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
function value=distances(points)
value=zeros(size(points,1));for k=1:size(points,1),value(k,:)=vecnorm(points-points(k,:),2,2)';end
end
function value=choose(test,yes,no),if test,value=yes;else,value=no;end,end
function value=canonical(path),assert(~isempty(path),'scorpion:path','Explicit local path required');value=char(java.io.File(char(path)).getCanonicalPath());end
function value=read_json(path)
fid=fopen(path,'rb');assert(fid>=0);guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
value=jsondecode(native2unicode(fread(fid,inf,'*uint8')','UTF-8'));
end
function verify_files(items)
for k=1:numel(items),assert(strcmp(trkg4_file_sha256(items(k).path),items(k).sha256),'scorpion:sourceChanged','Source/product changed');end
end
function values=string_list(value)
if ischar(value),values={value};elseif isstring(value),values=cellstr(value);else,values=value;end
values=values(:);
end
function value=append(value,row),if isempty(value),value=row;else,value(end+1)=row;end,end
function write_json(path,value)
assert(~isfile(path));fid=fopen(path,'w','n','UTF-8');assert(fid>=0);guard=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
function release_resource(lease,channel,lockfile),lease.release();channel.close();lockfile.close();end
