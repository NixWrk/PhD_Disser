function assemble_ecg_fractions(plan_file,root)
% Reuse canonical EIDORS weighted assembly; generate no new mesh or physical solver.
plan=jsondecode(fileread(plan_file));here=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'MATLAB_TRKG4_real_subjects','src'));
cfg=struct('project_root',fullfile(root,'MATLAB_TRKG4_real_subjects'),'length_unit','mm','scale_nodes_to_si',true);
startup=getenv('EIDORS_STARTUP');if isempty(startup),startup=winqueryreg('HKEY_CURRENT_USER','Environment','EIDORS_STARTUP');end
eidors_dir=fileparts(startup);assert(isfolder(eidors_dir));
for folder={'','tools','models','solvers','solvers/forward','arch/matlab','overloads'},addpath(fullfile(eidors_dir,folder{1}));end
eidors_cache('init');eidors_cache('cache_size',512*1024^2);eidors_cache('boost_priority',0);
cache=fullfile(here,'exploratory','eidors_cache');if ~isfolder(cache),mkdir(cache);end
eidors_cache('cache_path',cache);eidors_cache('eidors_path',eidors_dir);
eidors_default('set','fwd_solve','fwd_solve_1st_order');eidors_default('set','calc_system_mat','system_mat_1st_order');
eidors_default('set','calc_jacobian','jacobian_adjoint');eidors_default('set','mdl_normalize',@(x)0);maxNumCompThreads(2);
assert(strcmp(trkg4_file_sha256(plan.mesh_mat),plan.mesh_sha256));M=load(plan.mesh_mat,'nodes_mm','elems','boundary');
mm=eidors_obj('fwd_model','ECG endpoint continuation on unchanged CT mesh');
mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);clear M;
mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;
si=trkg4_scale_fmdl_to_si(mm,cfg);clear mm;ne=size(si.elems,1);nn=size(si.nodes,1);F=system_mat_fields(si);
assert(isequal(size(F),[3*ne nn]));
identity=struct('plan_sha256',trkg4_file_sha256(plan_file),'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields')),'mesh_sha256',plan.mesh_sha256,'matlab_version',version);
for vi=1:numel(plan.variants)
 v=plan.variants(vi);if v.q~=4096,continue;end
 output=fullfile(fileparts(plan_file),['assembled_' v.id]);if ~isfolder(output),mkdir(output);end
 target=fullfile(output,'assembly.json');
 if isfile(target),old=jsondecode(fileread(target));assert(isequal(old.identity,identity));continue;end
 assert(strcmp(trkg4_file_sha256(v.file),v.sha256));data=load(v.file,'fractions','element_ids');f=double(data.fractions);
 assert(isequal(double(data.element_ids(:)),(1:ne)')&&isequal(size(f),[ne 5])&&all(f>=0,'all')&&all(sum(f,2)==1));clear data;
 parts=cell(1,5);
 for k=1:5
  w=f(:,k);ids=find(w>0);rr=reshape((3*(ids-1)+(1:3)).',[],1);
  G=spdiags(repelem(sqrt(w(ids)),3),0,numel(rr),numel(rr))*F(rr,:);parts{k}=G'*G;
 end
 % Independent assembly check at declared diagnostic conductivities, not patient estimates.
 test_sigma=[1;.5;.1;2;.2];K=sparse(nn,nn);hom=K;
 for k=1:5,K=K+parts{k}*test_sigma(k);hom=hom+parts{k};end
 E=F'*F;hom_error=norm(hom-E,'fro')/norm(E,'fro');clear E hom;
 img=mk_image(si,1);img.elem_data=f*test_sigma;ind=calc_system_mat(img);
 het_error=norm(K-ind.E,'fro')/norm(ind.E,'fro');assert(hom_error<1e-12&&het_error<1e-12);
 save(fullfile(output,'parts.mat'),'parts','identity','v','-v7.3');
 result=struct('identity',identity,'variant',v,'parts_sha256',trkg4_file_sha256(fullfile(output,'parts.mat')),...
  'homogeneous_error',hom_error,'heterogeneous_error',het_error,'test_conductivities_S_m',test_sigma);
 fid=fopen(target,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(result,'PrettyPrint',true));fclose(fid);
 fprintf('ASSEMBLED %s %.3g %.3g\n',v.id,hom_error,het_error);clear parts f G K img ind;
end
end
