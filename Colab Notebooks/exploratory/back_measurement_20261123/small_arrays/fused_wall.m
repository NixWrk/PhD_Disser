function fused_wall(manifestfile)
% Full canonical CEM solves; one common absolute rho in soft tissue AND fat.
% No surrogate, ROM, mesh alteration, or equal-log change about separate bases.
m=jsondecode(fileread(manifestfile));
src=fullfile(m.root,'MATLAB_TRKG4_real_subjects','src');addpath(src);
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),m.kernel_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_back_comparison')),m.driver_sha256));
assert(strcmp(trkg4_file_sha256(which('fused_wall')),m.matlab_code_sha256));
maxNumCompThreads(4);
for ji=1:numel(m.jobs)
 runjob(m.jobs(ji),m);
end
end

function runjob(job,m)
started=tic;p=jsondecode(fileread(job.source_plan));
assert(strcmp(trkg4_file_sha256(job.source_plan),job.source_plan_sha256));
assert(strcmp(trkg4_file_sha256(p.mesh_path),p.mesh_sha256));
assert(strcmp(trkg4_file_sha256(p.shared_file),job.shared_sha256));
assert(strcmp(trkg4_file_sha256(job.contacts_file),job.contacts_sha256));
d=load(p.shared_file,'parts','identity');
assert(strcmp(d.identity.mesh_sha256,p.mesh_sha256));
assert(strcmp(d.identity.plan_sha256,job.source_plan_sha256));
parts=d.parts(2,:);clear d;
d=load(job.contacts_file,'C','plan_sha256');
assert(strcmp(d.plan_sha256,job.source_plan_sha256));C=d.C;clear d;
assert(p.nominal_diameter_mm==5 && numel(parts)==5);
folder=fullfile(m.work,job.participant);if ~isfolder(folder),mkdir(folder);end
ports=[1 0;0 1;0 -1;-1 0];
% A fresh 100-kHz full solve is compared with Sagan's independent invocation.
base=response(job.rho_proxy,'right_L100','baseline_proxy');
reference=jsondecode(fileread(job.baseline_file));
assert(strcmp(trkg4_file_sha256(job.baseline_file),job.baseline_sha256));
base_error=base.Z_ohm-reference.Z_ohm;
base_J_error=max(abs(base.J_logrho_ohm(:)-reference.J_logrho_ohm(:)));
assert(abs(base_error)<1e-8 && base_J_error<1e-8,'Baseline reproduction failed');
writejson(fullfile(folder,'baseline_check.local.json'),struct(...
 'Z_error_ohm',base_error,'max_J_error_ohm',base_J_error,...
 'source_contact_operators_used_without_modification',true));
for fi=1:numel(job.targets)
 target=job.targets(fi);fitid=target.posture;path=fullfile(folder,['fit_' fitid '.local.json']);
 if isfile(path)
  previous=jsondecode(fileread(path));
  assert(strcmp(previous.manifest_sha256,trkg4_file_sha256(manifest_path(m))));
  if previous.complete,fprintf('CACHED FIT %s %s\n',job.participant,fitid);continue;end
 end
 x=log(m.initial_rho_ohm_m);lower=-Inf;upper=Inf;history=struct([]);converged=false;
 for iteration=1:m.root_iterations_max
  rho=job.rho_proxy;rho([1 5])=exp(x);
  r=response(rho,'right_L100',['calibrate_' fitid]);
  residual=r.Z_ohm-target.Z_ohm;J=sum(r.J_logrho_ohm([1 5]));
  history=[history;struct('iteration',iteration,'rho_common_ohm_m',exp(x),...
   'Z_ohm',r.Z_ohm,'target_Z_ohm',target.Z_ohm,'residual_ohm',residual,'J_common_ohm',J)]; %#ok<AGROW>
  fprintf('FIT %s %s iter %d rho %.9g Z %.9g residual %.6g J %.9g\n',...
   job.participant,fitid,iteration,exp(x),r.Z_ohm,residual,J);
  assert(J>0 && isfinite(J),'Nonpositive derivative; monotone root assumption fails');
  if abs(residual)<m.root_tolerance_ohm,converged=true;break;end
  if residual<0,lower=x;else,upper=x;end
  step=max(-log(m.log_step_factor_limit),min(log(m.log_step_factor_limit),-residual/J));candidate=x+step;
  if candidate<=lower || candidate>=upper,candidate=(lower+upper)/2;end
  assert(isfinite(candidate),'Nonfinite unconstrained log-rho step');x=candidate;
 end
 result=struct('participant',job.participant,'posture',fitid,'complete',false,...
  'root_converged',converged,'rho_common_ohm_m',exp(x),'target_Z_ohm',target.Z_ohm,...
  'history',history,'manifest_sha256',trkg4_file_sha256(manifest_path(m)));
 writejson(path,result);
 if ~converged,fprintf('FAILED FIT %s %s after 12 full solves\n',job.participant,fitid);continue;end
 rho=job.rho_proxy;rho([1 5])=exp(x);predictions=struct([]);
 for ci=1:numel(job.contact_ids)
  r=response(rho,job.contact_ids{ci},['predict_' fitid]);
  predictions=[predictions;r]; %#ok<AGROW>
 end
 % Central simultaneous perturbation tests J_common at the fitted common base.
 rp=rho;rm=rho;rp([1 5])=rho(1)*exp(m.fd_log_step);rm([1 5])=rho(1)*exp(-m.fd_log_step);
 plus=response(rp,'right_L100',['fd_plus_' fitid]);
 minus=response(rm,'right_L100',['fd_minus_' fitid]);
 jfd=(plus.Z_ohm-minus.Z_ohm)/(2*m.fd_log_step);
 janalytic=history(end).J_common_ohm;
 fd_error=abs(jfd-janalytic)/abs(janalytic);
 assert(fd_error<1e-4,'Common-rho derivative failed finite-difference check');
 result.predictions=predictions;result.J_common_fd_ohm=jfd;
 result.J_common_fd_relative_error=fd_error;result.complete=true;
 writejson(path,result);
 fprintf('COMPLETE FIT %s %s rho %.9g FD_relative %.3g elapsed %.1fs\n',...
  job.participant,fitid,exp(x),fd_error,toc(started));
end
writejson(fullfile(folder,'complete.local.json'),struct('completed',true,'elapsed_s',toc(started)));

 function row=response(rho,id,purpose)
  ci=find(strcmp({p.contacts.id},id));assert(isscalar(ci));
  assert(isequal(p.contacts(ci).roles(:),{'I+';'U+';'U-';'I-'}));
  assert(max(abs(p.contacts(ci).arc_offsets_mm(:)-[-.5;-.25;.25;.5]*p.contacts(ci).size_mm))<1e-9);
  if rho(1)==rho(5),tag=strrep(sprintf('%.17g',rho(1)),'.','d');else,tag='proxy';end
  output=fullfile(folder,['solve_' tag '_' id '.local.json']);
  if isfile(output)
   row=jsondecode(fileread(output));
   assert(max(abs(row.rho_ohm_m(:)-rho(:)))<1e-13);
   assert(strcmp(row.manifest_sha256,trkg4_file_sha256(manifest_path(m))));return;
  end
  timer=tic;K=parts{1}/rho(1);for k=2:5,K=K+parts{k}/rho(k);end
  [T,J,qc]=trkg4_lateral_cem_response(K,C{ci},parts,rho,ports,1:5,1);clear K;
  assert(qc.system_relative_residual<1e-10 && qc.reciprocity_ohm<1e-8);
  row=struct('participant',job.participant,'id',id,'side',p.contacts(ci).side,...
   'size_mm',p.contacts(ci).size_mm,'rho_ohm_m',rho,'Z_ohm',T(2,1),...
   'J_logrho_ohm',reshape(J(2,1,:),1,[]),'qc',qc,'seconds',toc(timer),...
   'purpose',purpose,'manifest_sha256',trkg4_file_sha256(manifest_path(m)));
  writejson(output,row);
  fprintf('SOLVED %s %s %s rho1 %.8g rho5 %.8g Z %.9g seconds %.1f\n',...
   job.participant,purpose,id,rho(1),rho(5),row.Z_ohm,row.seconds);
 end
end

function p=manifest_path(m)
p=fullfile(m.work,'manifest.local.json');
end

function writejson(path,data)
tmp=[path '.writing'];f=fopen(tmp,'w','n','UTF-8');assert(f>=0);
fprintf(f,'%s\n',jsonencode(data));fclose(f);movefile(tmp,path,'f');
end
