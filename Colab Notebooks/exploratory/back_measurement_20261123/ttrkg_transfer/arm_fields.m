function arm_fields(manifestfile)
% Bounded replay and arm-only cases through unchanged canonical FEM kernels.
m=jsondecode(fileread(manifestfile));
addpath(fullfile(m.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(4);
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),m.kernel_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_back_comparison')),m.canonical_driver_sha256));
for j=1:numel(m.jobs)
 job=m.jobs(j);p=jsondecode(fileread(job.plan));
 fprintf('START %s original-heart arm fields\n',job.participant);
 names=fieldnames(job.paths);
 for k=1:numel(names)
  assert(strcmp(trkg4_file_sha256(job.paths.(names{k})),job.hashes.(names{k})));
 end
 legacyfile=fullfile(p.work,'legacy_reproduction.local.json');
 if ~isfile(legacyfile)
  data=load(job.paths.legacy_materials,'parts','identity');parts=data.parts(2,:);identity=data.identity;clear data;
  data=load(job.paths.legacy_cem,'result');sc=data.result.scenarios;
  if ~iscell(sc),sc=num2cell(sc);end
  index=find(cellfun(@(s)strcmp(s.id,'L60_TT1x1_ttrkg'),sc));assert(numel(index)==1);
  c=sc{index};C=c.grouped_contact_matrix_SI;groups={c.ports.id};
  ports=zeros(numel(groups),2);
  pairs={job.drive_groups,job.measure_groups};
  for k=1:2,[ok,idx]=ismember(pairs{k},groups);assert(all(ok));ports(idx,k)=[1;-1];end
  clear data sc c;
  ref=jsondecode(fileread(job.paths.legacy_answer));rho=ref.rho_ohm_m(:)';
  K=parts{1}/rho(1);for k=2:5,K=K+parts{k}/rho(k);end
  [T,J,qc]=trkg4_lateral_cem_response(K,C,parts,rho,ports,1:5,1);
  jj=reshape(J(2,1,:),1,[]);oldj=reshape(ref.jacobian_logrho(2,1,:),1,[]);
  ze=T(2,1)-ref.ttrkg_Z_ohm;je=max(abs(jj(ref.active_material_indices)-oldj));
  assert(abs(ze)<1e-8&&je<1e-8,'Historical whole-heart CT replay failed');
  writejson(legacyfile,struct('Z_ohm',T(2,1),'J_logrho_ohm',jj,'rho_ohm_m',rho,...
    'Z_error_ohm',ze,'max_J_error_ohm',je,'qc',qc,'mesh_sha256',identity.mesh_sha256));
  fprintf('LEGACY %s Z %.12g dZ %.3g dJ %.3g\n',job.participant,T(2,1),ze,je);
  clear parts identity C ports K T J;
 end
 assert(strcmp(trkg4_file_sha256(p.shared_file),p.shared_sha256));
 d=load(p.shared_file,'identity');assert(strcmp(d.identity.mesh_sha256,p.mesh_sha256));
 assert(strcmp(d.identity.plan_sha256,job.posterior_source_plan_sha256));clear d;
 trkg4_back_comparison(job.plan,'all');
 fprintf('COMPLETE %s arm fields\n',job.participant);
end
end

function writejson(path,value)
f=fopen(path,'w','n','UTF-8');assert(f>=0);guard=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value));
end
