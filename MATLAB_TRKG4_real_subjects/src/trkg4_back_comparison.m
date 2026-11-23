function trkg4_back_comparison(planfile,stage)
% Fixed posterior arrays: canonical finite CEM contacts and full FEM responses.
% New output only. All five electrical tissues, fixed geometry, no heart volume.
if nargin<2,stage='all';end
p=jsondecode(fileread(planfile));addpath(fullfile(p.root,'MATLAB_TRKG4_real_subjects','src'),...
 fullfile(p.root,'MATLAB_TRKG4_real_subjects','tools'));
maxNumCompThreads(4);work=p.work;ports=[1 0;0 1;0 -1;-1 0];
assert(strcmp(trkg4_file_sha256(p.mesh_path),p.mesh_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),p.canonical_kernel_sha256));
plan_sha256=trkg4_file_sha256(planfile);cache=fullfile(work,'contacts.mat');
if isfile(cache)
 d=load(cache,'C','plan_sha256');assert(strcmp(d.plan_sha256,plan_sha256));C=d.C;clear d;
else
 M=load(p.mesh_path);cfg=trkg4_config(p.subject);
 mm=struct('name','posterior fixed arrays','nodes',M.nodes_mm,'elems',double(M.elems),'boundary',double(M.boundary));clear M;
 ctx=trkg4_local_patch_context(mm);C=cell(numel(p.contacts),1);checks=struct([]);passed=true;
 for c=1:numel(p.contacts)
  cc=p.contacts(c);entry=struct('id',cc.id,'side',cc.side,'size_mm',cc.size_mm,'passed',false);
  try
   [el,faces]=trkg4_local_exact_patches(ctx,cc.centres_lps_mm,p.nominal_area_mm2,p.z_contact_ohm_m2);
   current=struct([]);
   for e=1:4
    el(e).faces=mm.boundary(faces{e},:);area=ctx.canonical.face_area(faces{e});actual=sum(area);
    centroid=sum(ctx.face_centroid(faces{e},:).*area,1)/actual;
    offset=norm(centroid-cc.centres_lps_mm(e,:));extent=max(vecnorm(mm.nodes(el(e).nodes,:)-cc.centres_lps_mm(e,:),2,2));
    current=[current;struct('role',cc.roles{e},'actual_area_mm2',actual,...
       'area_relative_error',abs(actual/p.nominal_area_mm2-1),'centroid_error_mm',offset,...
       'extent_mm',extent,'actual_centroid_lps_mm',centroid,'boundary_faces',numel(faces{e}))]; %#ok<AGROW>
   end
   entry.contacts=current;
   entry.passed=all([current.area_relative_error]<=p.contact_area_relative_error_max)&...
    all([current.centroid_error_mm]<=p.contact_centroid_error_mm_max)&all([current.extent_mm]<=p.contact_extent_mm_max);
   if entry.passed,mm.electrode=el;C{c}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(mm,cfg));end
   entry.failure='';
  catch exception
   entry.contacts=struct([]);entry.failure=exception.message;
  end
  passed=passed&&entry.passed;checks=[checks;entry]; %#ok<AGROW>
  fprintf('CONTACT %s passed %d\n',cc.id,entry.passed);
 end
 writejson(fullfile(work,'contact_qc.local.json'),struct('passed',passed,'rows',checks,'plan_sha256',plan_sha256));
 assert(passed,'Posterior finite-contact QC failed; see contact_qc.local.json');
 save(cache,'C','plan_sha256','-v7.3');clear mm ctx;
end
if strcmp(stage,'contacts'),return;end
if ~isfile(p.shared_file)
 assert(isfield(p,'fraction_files'),'No prepared material fractions');
 cfg=trkg4_config(p.subject);trkg4_prepare_runtime(cfg);M=load(p.mesh_path);
 f=eidors_obj('fwd_model','posterior locally refined own CT');
 f.nodes=M.nodes_mm;f.elems=double(M.elems);f.boundary=double(M.boundary);f.gnd_node=1;
 f.electrode=struct([]);f.stimulation=struct([]);f.normalize_measurements=0;
 f.solve=@fwd_solve_1st_order;f.system_mat=@system_mat_1st_order;f.jacobian=@jacobian_adjoint;clear M;
 F=system_mat_fields(trkg4_scale_fmdl_to_si(f,cfg));clear f;
 parts=cell(2,5);assembly_qc=struct([]);
 for q=1:2
  ref=p.fraction_files(q);assert(strcmp(trkg4_file_sha256(ref.path),ref.sha256));
  d=load(ref.path,'fractions');assert(all(d.fractions>=0,'all')&&all(sum(d.fractions,2)==1));
  for k=1:5
   w=repelem(d.fractions(:,k),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
   parts{q,k}=B'*B;
  end
  K0=F'*F;total=parts{q,1};for k=2:5,total=total+parts{q,k};end
  partition_error=norm(total-K0,'fro')/norm(K0,'fro');assert(partition_error<1e-12);
  assembly_qc=[assembly_qc;struct('points',ref.points,'partition_relative_error',partition_error)]; %#ok<AGROW>
  fprintf('MATERIAL MATRICES q%d complete\n',ref.points);clear d B w K0 total;
 end
 clear F;
 identity=struct('mesh_sha256',p.mesh_sha256,'plan_sha256',plan_sha256,...
  'field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields')));
 save(p.shared_file,'parts','identity','-v7.3');
 writejson(fullfile(work,'assembly_qc.local.json'),struct('rows',assembly_qc,'identity',identity));clear parts;
elseif ~isempty(p.shared_sha256)
 assert(strcmp(trkg4_file_sha256(p.shared_file),p.shared_sha256));
end
data=load(p.shared_file,'parts','identity');assert(strcmp(data.identity.mesh_sha256,p.mesh_sha256));
if isfield(p,'fraction_files'),assert(strcmp(data.identity.plan_sha256,plan_sha256));end
parts=data.parts(2,:);clear data;
for s=1:numel(p.scenarios)
 scenario=p.scenarios(s);rho=scenario.rho_ohm_m(:)';K=parts{1}/rho(1);
 for k=2:5,K=K+parts{k}/rho(k);end
 rows=struct([]);
 for c=1:numel(p.contacts)
  cc=p.contacts(c);output=fullfile(work,sprintf('%s_%s.local.json',scenario.name,cc.id));
  if isfile(output)
   row=jsondecode(fileread(output));assert(strcmp(row.plan_sha256,plan_sha256));
  else
   timer=tic;[T,J,qc]=trkg4_lateral_cem_response(K,C{c},parts,rho,ports,1:5,1);
   row=struct('id',cc.id,'side',cc.side,'size_mm',cc.size_mm,'scenario',scenario.name,...
       'rho_ohm_m',rho,'Z_ohm',T(2,1),'J_logrho_ohm',reshape(J(2,1,:),1,[]),...
       'qc',qc,'seconds',toc(timer),'plan_sha256',plan_sha256,...
       'canonical_kernel_sha256',p.canonical_kernel_sha256,'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
   writejson(output,row);
   fprintf('SOLVED %s %s Z %.8g sec %.1f\n',scenario.name,cc.id,row.Z_ohm,row.seconds);
  end
  rows=[rows;row]; %#ok<AGROW>
 end
 writejson(fullfile(work,[scenario.name '.local.json']),struct('scenario',scenario,'rows',rows,'plan_sha256',plan_sha256));
 clear K;
end
writejson(fullfile(work,'complete.local.json'),struct('completed',true,'plan_sha256',plan_sha256,...
 'cases',numel(p.contacts)*numel(p.scenarios),'status','exploratory_hypothesis_not_validated'));
end

function writejson(path,d)
temp=[path '.writing'];f=fopen(temp,'w','n','UTF-8');assert(f>=0);
fprintf(f,'%s\n',jsonencode(d));fclose(f);movefile(temp,path,'f');
end
