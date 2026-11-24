function run_extended(plan_file)
% Local orchestration of the canonical weighted EIDORS assembly and CEM kernel.
% Same algorithms as run_scorpion_heart_pilot; no root-level lock/output writes.
plan=jsondecode(fileread(plan_file));here=fileparts(mfilename('fullpath'));
jobs=jsondecode(fileread(fullfile(here,'exploratory','jobs.json')));
root=jobs.root;addpath(fullfile(root,'MATLAB_TRKG4_real_subjects','src'));
cfg=struct('project_root',fullfile(root,'MATLAB_TRKG4_real_subjects'),'length_unit','mm','scale_nodes_to_si',true);
% Startup uses cd and a cache under the installation. Load the same installed
% kernels read-only, with their documented defaults and a local cache instead.
startup=getenv('EIDORS_STARTUP');if isempty(startup),startup=winqueryreg('HKEY_CURRENT_USER','Environment','EIDORS_STARTUP');end
eidors_dir=fileparts(startup);assert(isfolder(eidors_dir));
for folder={'','tools','models','solvers','solvers/forward','arch/matlab','overloads'}
 addpath(fullfile(eidors_dir,folder{1}));
end
eidors_cache('init');eidors_cache('cache_size',512*1024^2);eidors_cache('boost_priority',0);
cache=fullfile(here,'exploratory','eidors_cache');if ~isfolder(cache),mkdir(cache);end
eidors_cache('cache_path',cache);eidors_cache('eidors_path',eidors_dir);
eidors_default('set','fwd_solve','fwd_solve_1st_order');eidors_default('set','calc_system_mat','system_mat_1st_order');
eidors_default('set','calc_jacobian','jacobian_adjoint');eidors_default('set','mdl_normalize',@(x)0);
maxNumCompThreads(2);
job=jobs.jobs(strcmp({jobs.jobs.subject},plan.subject_alias)&cellfun(@(r)abs(r(4)-1.35)<1e-9,{jobs.jobs.rho_fixed_ohm_m}));
assert(numel(job)==1);baseline_file=fullfile(jobs.output,[job.id '.json']);
baseline=jsondecode(fileread(baseline_file));rho=baseline.fit.rho_ohm_m(:);
source=load(job.cem_file,'result');sc=source.result.scenarios;clear source;
if ~iscell(sc),sc=num2cell(sc);end
idx=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_ids{1}),sc),1);
C=sc{idx}.grouped_contact_matrix_SI;P=ports(sc{idx}.ports);clear sc;
assert(strcmp(trkg4_file_sha256(plan.mesh_mat),plan.mesh_sha256));
M=load(plan.mesh_mat,'nodes_mm','elems','boundary');
mm=eidors_obj('fwd_model',[plan.subject_alias ' extended sphere on unchanged mesh']);
mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);clear M;
mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;
mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;
si=trkg4_scale_fmdl_to_si(mm,cfg);clear mm;ne=size(si.elems,1);nn=size(si.nodes,1);
F=system_mat_fields(si);assert(isequal(size(F),[3*ne nn]));
identity=struct('plan_sha256',trkg4_file_sha256(plan_file),'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'mesh_sha256',plan.mesh_sha256,'fit_sha256',trkg4_file_sha256(baseline_file),...
 'field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields')),...
 'scale_kernel_sha256',trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')),...
 'response_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),'matlab_version',version);
for vi=1:numel(plan.variants)
 v=plan.variants(vi);if v.q~=4096,continue;end
 output=fullfile(fileparts(plan_file),['checked_' v.id]);if ~isfolder(output),mkdir(output);end
 target=fullfile(output,'result.json');if isfile(target),old=jsondecode(fileread(target));assert(isequal(old.identity,identity));continue;end
 assert(strcmp(trkg4_file_sha256(v.file),v.sha256));data=load(v.file,'fractions','element_ids');f=double(data.fractions);
 assert(isequal(double(data.element_ids(:)),(1:ne)')&&isequal(size(f),[ne 5])&&all(f>=0,'all')&&all(sum(f,2)==1));clear data;
 parts=cell(1,5);sigma_effective=zeros(ne,1);clock=tic;
 for k=1:5
  w=f(:,k);ids=find(w>0);rr=reshape((3*(ids-1)+(1:3)).',[],1);
  G=spdiags(repelem(sqrt(w(ids)),3),0,numel(rr),numel(rr))*F(rr,:);
  parts{k}=G'*G;sigma_effective=sigma_effective+w/rho(k);
 end
 clear G w ids rr f;homogeneous=sparse(nn,nn);K=homogeneous;
 for k=1:5,homogeneous=homogeneous+parts{k};K=K+parts{k}/rho(k);end
 E=F'*F;hom_error=norm(homogeneous-E,'fro')/norm(E,'fro');clear homogeneous E;
 img=mk_image(si,1);img.elem_data=sigma_effective;ind=calc_system_mat(img);
 het_error=norm(K-ind.E,'fro')/norm(ind.E,'fro');clear img ind sigma_effective;
 assert(hom_error<1e-12&&het_error<1e-12);
 save(fullfile(output,'parts.mat'),'parts','identity','v','-v7.3');
 answer=struct([]);
 % Baseline and the two prescribed interval endpoints. CT targets never enter.
 for phase=[0 .1 .4]
  rr=rho;
  if phase~=0,[~,ii]=min(abs(baseline.phase_rr-phase));rr(1:2)=rr(1:2).*exp(baseline.dlogrho_FEM(:,ii));end
  K=sparse(nn,nn);for k=1:5,K=K+parts{k}/rr(k);end
  [T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rr,P,[1 2 4 5],1);
  row=struct('phase_rr',phase,'rho_ohm_m',rr','Z_t_ohm',T(2,1),'Z_b_ohm',T(4,3),...
   'transfer_matrix_ohm',T,'J_t_logrho',reshape(J(2,1,:),1,[]),'J_b_logrho',reshape(J(4,3,:),1,[]),'system_checks',qc);
  answer=[answer;row]; %#ok<AGROW>
 end
 result=struct('identity',identity,'variant',v,'job_id',job.id,'case_id',job.case_ids{1},...
  'homogeneous_assembly_error',hom_error,'heterogeneous_assembly_error',het_error,...
  'parts_sha256',trkg4_file_sha256(fullfile(output,'parts.mat')),'answers',answer,...
  'elapsed_s',toc(clock),'mesh_unchanged',true,'global_mesh_convergence_proven',false);
 fid=fopen(target,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));fclose(fid);
 fprintf('EXTENDED %s %s seconds=%.1f\n',plan.subject_alias,v.id,result.elapsed_s);clear parts K;
end
end
function P=ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
P=zeros(numel(groups),4);for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
