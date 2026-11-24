function spatial_control(manifestfile)
% Exactly two full solutions per participant; split element operators only.
m=jsondecode(fileread(manifestfile));
addpath(fullfile(m.root,'MATLAB_TRKG4_real_subjects','src'));
assert(strcmp(trkg4_file_sha256(which('spatial_control')),m.matlab_code_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),m.kernel_sha256));
maxNumCompThreads(4);trkg4_prepare_runtime(struct());eidors_cache('off');
for ji=1:numel(m.jobs)
 job=m.jobs(ji);folder=fullfile(m.work,job.participant);started=tic;
 names=fieldnames(job.paths);
 for k=1:numel(names),assert(strcmp(trkg4_file_sha256(job.paths.(names{k})),job.hashes.(names{k})));end
 d=load(job.paths.shared,'parts','identity');parts=d.parts(2,:);identity=d.identity;clear d;
 assert(strcmp(identity.mesh_sha256,job.hashes.mesh));
 assert(strcmp(identity.plan_sha256,job.original_plan_sha256));
 assert(strcmp(identity.field_kernel_sha256,trkg4_file_sha256(which('system_mat_fields'))));
 cache=fullfile(folder,'operators.local.mat');
 if isfile(cache)
  d=load(cache,'right','left','remainder','manifest_sha256');
  assert(strcmp(d.manifest_sha256,trkg4_file_sha256(manifestfile)));
  right=d.right;left=d.left;remainder=d.remainder;clear d;
 else
  mesh=load(job.paths.mesh,'nodes_mm','elems');
  d=load(job.paths.fractions,'fractions');soft=d.fractions(:,1);clear d;
  d=load(job.paths.selection,'region_label');labels=d.region_label;clear d;
  n=size(mesh.nodes_mm,1);ne=size(mesh.elems,1);assert(ne==numel(soft)&&ne==numel(labels));
  whole=sparse(n,n);right=sparse(n,n);left=sparse(n,n);
  for first=1:m.element_block_size:ne
   last=min(ne,first+m.element_block_size-1);ids=first:last;
   [used,~,local]=unique(mesh.elems(ids,:));
   f=eidors_obj('fwd_model','unchanged element block');
   f.nodes=mesh.nodes_mm(used,:)*1e-3;f.elems=reshape(double(local),numel(ids),4);
   f.boundary=zeros(0,3);f.electrode=struct([]);f.stimulation=struct([]);
   f.gnd_node=1;f.normalize_measurements=0;
   F=system_mat_fields(f);assert(size(F,1)==3*numel(ids));
   w=repelem(soft(ids),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
   whole=whole+scatter(B'*B,used,n);
   for side=1:2
    if ~any(labels(ids)==side),continue;end
    wr=w.*repelem(labels(ids)==side,3);
    B=spdiags(sqrt(wr),0,numel(wr),numel(wr))*F;
    if side==1,right=right+scatter(B'*B,used,n);else,left=left+scatter(B'*B,used,n);end
   end
   clear F B f w wr local used;
   if mod(first-1,5*m.element_block_size)==0 || last==ne
    fprintf('ASSEMBLE %s %d/%d elements elapsed %.1fs\n',job.participant,last,ne,toc(started));
   end
  end
  reconstruction=norm(whole-parts{1},'fro')/norm(parts{1},'fro');
  assert(reconstruction<1e-11,'Element/fraction reconstruction differs from original soft K');
  remainder=parts{1}-right-left;
  partition=norm(right+left+remainder-parts{1},'fro')/norm(parts{1},'fro');
  assert(partition<1e-13);
  one=ones(n,1);row_sum=norm((right+left)*one)/max(norm(right+left,'fro'),eps);
  assert(row_sum<1e-11,'Regional element matrix does not preserve constants');
  writejson(fullfile(folder,'assembly.local.json'),struct(...
   'soft_operator_reconstruction_relative_error',reconstruction,...
   'operator_partition_relative_error',partition,'region_constant_null_relative_error',row_sum,...
   'assembly_seconds',toc(started),'elements',ne,'nodes',n,...
   'selected_right_elements',sum(labels==1),'selected_left_elements',sum(labels==2),...
   'element_field_kernel_sha256',trkg4_file_sha256(which('system_mat_fields'))));
  manifest_sha256=trkg4_file_sha256(manifestfile); %#ok<NASGU>
  save(cache,'right','left','remainder','manifest_sha256','-v7.3');
  clear mesh labels soft whole one;
 end
 rho=m.rho(:)';K=parts{1}/rho(1);for k=2:5,K=K+parts{k}/rho(k);end
 % Extra parts are derivative directions only; forward K remains five-tissue.
 expanded=[parts {right,left,remainder}];extended_rho=[rho rho(1)*ones(1,3)];
 clear parts right left remainder;
 for lead={'arm','posterior'}
  name=lead{1};output=fullfile(folder,[name '.local.json']);
  if isfile(output)
   old=jsondecode(fileread(output));
   assert(strcmp(old.manifest_sha256,trkg4_file_sha256(manifestfile)));
   assert(old.reproduction_passed,'Prior solve failed reproduction; do not spend extra solutions');continue;
  end
  planfile=job.paths.([name '_plan']);plan=jsondecode(fileread(planfile));
  d=load(job.paths.([name '_contacts']),'C','plan_sha256');
  assert(strcmp(d.plan_sha256,trkg4_file_sha256(planfile)));
  if strcmp(name,'arm'),id='arm_arm_top_1x1_step100';else,id='right_L100';end
  ci=find(strcmp({plan.contacts.id},id));assert(isscalar(ci));C=d.C{ci};clear d;
  timer=tic;[T,J,qc]=trkg4_lateral_cem_response(K,C,expanded,extended_rho,[1 0;0 1;0 -1;-1 0],1:8,1);
  jj=reshape(J(2,1,:),1,[]);ref=jsondecode(fileread(job.paths.([name '_baseline'])));
  ze=T(2,1)-ref.Z_ohm;je=max(abs(jj(1:5)-ref.J_logrho_ohm(:)'));
  pe=abs(sum(jj(6:8))-jj(1));
  passed=abs(ze)<1e-8&&je<1e-8&&pe<1e-8&&qc.system_relative_residual<1e-10&&qc.reciprocity_ohm<1e-8;
  row=struct('Z_ohm',T(2,1),'J_soft_ohm',jj(1),'J_distal_right_ohm',jj(6),...
   'J_distal_left_ohm',jj(7),'J_distal_union_ohm',sum(jj(6:7)),'J_remainder_ohm',jj(8),...
   'original_Z_error_ohm',ze,'original_max_J_error_ohm',je,'J_partition_abs_error_ohm',pe,...
   'qc',qc,'seconds',toc(timer),'reproduction_passed',passed,...
   'manifest_sha256',trkg4_file_sha256(manifestfile));
  writejson(output,row);assert(passed,'Original Z/J or regional partition failed; result not accepted');
  fprintf('SOLVED %s %s Z %.10g Jsoft %.10g Jdistal %.10g Jrest %.10g dJ %.3g sec %.1f\n',...
   job.participant,name,T(2,1),jj(1),sum(jj(6:7)),jj(8),je,row.seconds);
 end
 clear expanded K;
 fprintf('COMPLETE %s elapsed %.1fs\n',job.participant,toc(started));
end
end

function A=scatter(local,used,n)
[i,j,v]=find(local);A=sparse(double(used(i)),double(used(j)),v,n,n);
end

function writejson(path,data)
tmp=[path '.writing'];f=fopen(tmp,'w','n','UTF-8');assert(f>=0);
fprintf(f,'%s\n',jsonencode(data));fclose(f);movefile(tmp,path,'f');
end
