function fem60(manifestfile)
% Scalar interpolation proposes points; every accepted value uses full CEM.
m=jsondecode(fileread(manifestfile));
addpath(fullfile(m.root,'MATLAB_TRKG4_real_subjects','src'));maxNumCompThreads(4);
m.manifest_sha256=trkg4_file_sha256(manifestfile);
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),m.kernel_sha256));
assert(strcmp(trkg4_file_sha256(which('fem60')),m.matlab_sha256));
for k=1:numel(m.jobs),runjob(m.jobs(k),m);end
end

function runjob(job,m)
folder=fullfile(m.work,job.participant);if ~isfolder(folder),mkdir(folder);end
proxy=m.rho_proxy(:)';ports=[1 0;0 1;0 -1;-1 0];
cachefile=fullfile(folder,'cache.local.json');known=struct([]);newcount=0;
if isfile(cachefile)
 data=jsondecode(fileread(cachefile));assert(strcmp(data.manifest_sha256,m.manifest_sha256));
 known=data.rows(:);newcount=data.new_full_solves;clear data;
 for ii=1:numel(known)
  assert(strcmp(trkg4_file_sha256(known(ii).source),known(ii).source_sha256));
  known(ii).rho_ohm_m=known(ii).rho_ohm_m(:)';known(ii).J_logrho_ohm=known(ii).J_logrho_ohm(:)';
 end
else
 for ii=1:numel(job.imports)
  q=job.imports(ii);assert(strcmp(trkg4_file_sha256(q.source),q.source_sha256));
  r=blankrow();r.family=q.family;r.id=q.id;r.size_mm=sscanf(q.id,'right_L%d');
  r.rho_ohm_m=q.rho_ohm_m(:)';r.Z_ohm=q.Z_ohm;r.J_logrho_ohm=q.J_logrho_ohm(:)';
  r.qc=q.qc;r.provenance=q.provenance;r.source=q.source;r.source_sha256=q.source_sha256;
  known=[known;r]; %#ok<AGROW>
 end
 savecache();
end
family='small';[parts,C,p]=loadfamily(job.sources.small);started=tic;
old=response(proxy,'right_L60','reference',false);
fresh=response(proxy,'right_L60','fresh_reference',true);
assert(abs(old.Z_ohm-fresh.Z_ohm)<1e-8 && max(abs(old.J_logrho_ohm(:)-fresh.J_logrho_ohm(:)))<1e-8);
writejson(fullfile(folder,'baseline_small_check.local.json'),struct('Z_error_ohm',fresh.Z_ohm-old.Z_ohm,...
 'max_J_error_ohm',max(abs(fresh.J_logrho_ohm(:)-old.J_logrho_ohm(:)))));
for si=1:numel(m.scenarios)
 name=m.scenarios{si};file=fullfile(folder,[name '.local.json']);
 if isfile(file)
  saved=jsondecode(fileread(file));assert(strcmp(saved.manifest_sha256,m.manifest_sha256));
  if saved.small_complete,fprintf('CACHED SMALL %s %s\n',job.participant,name);continue;end
 end
 fprintf('SCENARIO %s %s\n',job.participant,name);
 [lower,upper]=bounds(name);n=curve(name);
 if isempty(n)
  if strcmp(name,'common_absolute'),q=5;else,q=1;end
  response(decode(name,q),'right_L60',['seed_' name],false);n=curve(name);
 end
 % Fat-only receives one upper-end check so a descending high-rho branch is
 % not silently excluded just because all measured targets lie below base.
 if strcmp(name,'fat_only')
  response(decode(name,upper),'right_L60','fat_upper_envelope_check',false);n=curve(name);
 end
 targets=[job.targets.measured_Z_ohm];margin=max(.05,.05*(max(targets)-min(targets)));
 for expansion=1:m.expansion_evaluations_max
  needlower=min(targets)<min(n(:,2))+margin;
  needupper=max(targets)>max(n(:,2))-margin;
  if needlower && n(1,1)>log(lower)+1e-12
   dx=-m.equal_log_step_limit;
   if n(1,3)>m.derivative_screen_ohm
    dx=max(-m.equal_log_step_limit,min(-.1,(min(targets)-margin-n(1,2))/n(1,3)));
   end
   x=max(log(lower),n(1,1)+dx);
  elseif needupper && n(end,1)<log(upper)-1e-12
   dx=m.equal_log_step_limit;
   if n(end,3)>m.derivative_screen_ohm
    dx=min(m.equal_log_step_limit,max(.1,(max(targets)+margin-n(end,2))/n(end,3)));
   end
   x=min(log(upper),n(end,1)+dx);
  else,break;
  end
  response(decode(name,exp(x)),'right_L60',['bracket_' name],false);n=curve(name);
 end
 fits=cell(numel(job.targets),1);didfd=false;
 for ti=1:numel(job.targets)
  target=job.targets(ti);y=target.measured_Z_ohm;history=struct([]);n=curve(name);
  fit=struct('posture',target.posture,'status','not_yet_solved','sampled_crossing_count',0,...
   'uniqueness_status','not_established','predictions',struct([]),'root_iterations',0);
  [~,nearest]=min(abs(n(:,2)-y));fit.closest_checked_parameter=exp(n(nearest,1));fit.closest_checked_Z60_ohm=n(nearest,2);
  intervals=crossings(n,y,m.root_tolerance_ohm);fit.sampled_crossing_count=size(intervals,1);
  if isempty(intervals)
   fit.status='target_not_attained_in_checked_range';fit.uniqueness_status='no_sampled_bracket_no_extrapolation';
   fprintf('NO ROOT %s %s %s target %.9g checked rho %.6g..%.6g Z %.6g..%.6g\n',...
    job.participant,name,target.posture,y,exp(n(1,1)),exp(n(end,1)),min(n(:,2)),max(n(:,2)));
  elseif size(intervals,1)>1
   fit.status='multiple_root_brackets';fit.uniqueness_status='multiple_sign_changes_no_unique_estimate';
   fit.brackets_parameter=exp(n(intervals,1));
  else
   converged=false;
   for iteration=1:m.root_iterations_max
    n=curve(name);intervals=crossings(n,y,m.root_tolerance_ohm);
    if size(intervals,1)~=1,fit.status='multiple_root_brackets';break;end
    ix=intervals(1,:);lo=n(ix(1),:);hi=n(ix(2),:);
    if ix(1)==ix(2),x=lo(1);else,x=hermite_root(lo,hi,y);end
    rr=response(decode(name,exp(x)),'right_L60',['root_' name '_' target.posture],false);
    residual=rr.Z_ohm-y;J=scalarJ(rr.J_logrho_ohm,name);
    history=[history;struct('parameter',exp(x),'Z_ohm',rr.Z_ohm,'residual_ohm',residual,'J_parameter_ohm',J)]; %#ok<AGROW>
    fprintf('ROOT %s %s %s q %.10g residual %.4g J %.6g\n',job.participant,name,target.posture,exp(x),residual,J);
    if abs(residual)<=m.root_tolerance_ohm,converged=true;break;end
   end
   fit.root_iterations=numel(history);fit.history=history;
   if converged
    fit.status='converged_full_FEM';fit.parameter=exp(x);fit.rho_ohm_m=rr.rho_ohm_m;
    fit.residual_ohm=residual;fit.J_parameter_ohm=J;fit.relative_condition=abs(rr.Z_ohm/J);
    fit.ill_conditioned=abs(J)<m.derivative_screen_ohm || fit.relative_condition>m.condition_screen;
    fit.uniqueness_status='one_sampled_crossing_global_uniqueness_not_proven';
    pred=rr;
    for spacing=[100,140]
     pred=[pred;response(rr.rho_ohm_m,sprintf('right_L%d',spacing),['holdout_' name '_' target.posture],false)]; %#ok<AGROW>
    end
    fit.predictions=pred;
    if ~didfd && x-m.fd_log_step>log(lower) && x+m.fd_log_step<log(upper)
     a=response(decode(name,exp(x+m.fd_log_step)),'right_L60',['FDplus_' name],false);
     b=response(decode(name,exp(x-m.fd_log_step)),'right_L60',['FDminus_' name],false);
     fd=(a.Z_ohm-b.Z_ohm)/(2*m.fd_log_step);err=abs(fd-J)/max(abs(J),1e-12);
     assert(err<1e-4,'Scenario log derivative failed finite differences');
     fit.FD_relative_error=err;fit.FD_J_parameter_ohm=fd;didfd=true;
    end
   elseif strcmp(fit.status,'not_yet_solved')
    fit.status='root_iteration_limit';fit.uniqueness_status='bracket_found_final_FEM_tolerance_not_met';
   end
  end
  fits{ti}=fit;
  savestudy(name,fits,false);
 end
 savestudy(name,fits,true);
end
% Release the small-mesh matrices before loading the independent extended mesh.
parts={};C={};p=struct();family='extended';[parts,C,p]=loadfamily(job.sources.extended);
old=response(proxy,'right_L200','reference',false);fresh=response(proxy,'right_L200','fresh_reference',true);
assert(abs(old.Z_ohm-fresh.Z_ohm)<1e-8 && max(abs(old.J_logrho_ohm(:)-fresh.J_logrho_ohm(:)))<1e-8);
writejson(fullfile(folder,'baseline_extended_check.local.json'),struct('Z_error_ohm',fresh.Z_ohm-old.Z_ohm,...
 'max_J_error_ohm',max(abs(fresh.J_logrho_ohm(:)-old.J_logrho_ohm(:)))));
for si=1:numel(m.scenarios)
 name=m.scenarios{si};file=fullfile(folder,[name '.local.json']);study=jsondecode(fileread(file));
 fits=study.fits;if ~iscell(fits),fits=num2cell(fits);end
 for ti=1:numel(fits)
  f=fits{ti};if ~strcmp(f.status,'converged_full_FEM'),continue;end
  pred=response(f.rho_ohm_m,'right_L200',['holdout200_' name '_' f.posture],false);
  previous=f.predictions;if iscell(previous),previous=[previous{:}];end
  if ~any([previous.size_mm]==200),f.predictions=[previous;pred];end
  fits{ti}=f;
 end
 study.fits=fits;study.extended_complete=true;writejson(file,study);
end
writejson(fullfile(folder,'complete.local.json'),struct('completed',true,'new_full_solves',newcount,'seconds',toc(started)));
fprintf('COMPLETE %s new full solves %d elapsed %.1fs\n',job.participant,newcount,toc(started));

 function [parts,C,p]=loadfamily(source)
  assert(strcmp(trkg4_file_sha256(source.plan),source.plan_sha256));
  p=jsondecode(fileread(source.plan));
  assert(strcmp(trkg4_file_sha256(p.mesh_path),source.mesh_sha256));
  assert(strcmp(trkg4_file_sha256(source.shared_file),source.shared_sha256));
  assert(strcmp(trkg4_file_sha256(source.contacts_file),source.contacts_sha256));
  d=load(source.shared_file,'parts','identity');
  assert(strcmp(d.identity.mesh_sha256,source.mesh_sha256)&&strcmp(d.identity.plan_sha256,source.plan_sha256));
  parts=d.parts(2,:);clear d;
  d=load(source.contacts_file,'C','plan_sha256');assert(strcmp(d.plan_sha256,source.plan_sha256));C=d.C;
 end

 function r=response(rho,id,purpose,force)
  rho=rho(:)';assert(all(rho>0)&&all(isfinite(rho)));
  if ~force
   for kk=numel(known):-1:1
    z=known(kk);
    if strcmp(z.family,family)&&strcmp(z.id,id)&&max(abs(z.rho_ohm_m(:)'-rho))<1e-13
     r=z;return;
    end
   end
  end
  ci=find(strcmp({p.contacts.id},id));assert(isscalar(ci));
  timer=tic;K=parts{1}/rho(1);for kk=2:5,K=K+parts{kk}/rho(kk);end
  [responseT,responseJ,responseQC]=trkg4_lateral_cem_response(K,C{ci},parts,rho,ports,1:5,1);clear K;
  assert(responseQC.system_relative_residual<1e-10&&responseQC.reciprocity_ohm<1e-8);
  r=blankrow();r.family=family;r.id=id;r.size_mm=p.contacts(ci).size_mm;r.rho_ohm_m=rho;
  r.Z_ohm=responseT(2,1);r.J_logrho_ohm=reshape(responseJ(2,1,:),1,[]);r.qc=responseQC;r.seconds=toc(timer);
  r.purpose=purpose;r.provenance='new_full_canonical_CEM';newcount=newcount+1;
  r.source=fullfile(folder,sprintf('solve_%04d.local.json',newcount));writejson(r.source,r);
  r.source_sha256=trkg4_file_sha256(r.source);known=[known;r];savecache();
  fprintf('SOLVE %s %s %s rhoS %.8g rhoF %.8g Z %.9g sec %.1f total %d\n',...
   job.participant,purpose,id,rho(1),rho(5),r.Z_ohm,r.seconds,newcount);
 end

 function savecache()
  writejson(cachefile,struct('manifest_sha256',m.manifest_sha256,'new_full_solves',newcount,'rows',known));
 end

 function rho=decode(name,q)
  rho=proxy;
  switch name
   case 'soft_only',rho(1)=q;
   case 'fat_only',rho(5)=q;
   case 'common_absolute',rho([1 5])=q;
   case 'common_multiplier',rho([1 5])=proxy([1 5])*q;
   otherwise,error('Unknown scalar hypothesis');
  end
 end

 function [lo,hi]=bounds(name)
  lo=m.parameter_absolute_envelope_ohm_m(1);hi=m.parameter_absolute_envelope_ohm_m(2);
  if strcmp(name,'common_multiplier'),lo=max(lo./proxy([1 5]));hi=min(hi./proxy([1 5]));end
 end

 function n=curve(name)
  n=zeros(0,3);[lo,hi]=bounds(name);
  for kk=1:numel(known)
   r=known(kk);if ~strcmp(r.family,'small')||~strcmp(r.id,'right_L60'),continue;end
   rho=r.rho_ohm_m(:)';
   if strcmp(name,'fat_only'),q=rho(5);elseif strcmp(name,'common_multiplier'),q=rho(1)/proxy(1);else,q=rho(1);end
   if q<lo-1e-12||q>hi+1e-12||max(abs(rho-decode(name,q)))>1e-11,continue;end
   n=[n;log(q),r.Z_ohm,scalarJ(r.J_logrho_ohm,name)]; %#ok<AGROW>
  end
  if ~isempty(n),[~,idx]=unique(n(:,1),'last');n=sortrows(n(idx,:),1);end
 end

 function savestudy(name,fits,complete)
  n=curve(name);value=struct('scenario',name,'fits',{fits},'manifest_sha256',m.manifest_sha256,...
   'small_complete',complete,'extended_complete',false,'checked_parameter_min',exp(n(1,1)),...
   'checked_parameter_max',exp(n(end,1)),'checked_Z_min_ohm',min(n(:,2)),'checked_Z_max_ohm',max(n(:,2)),...
   'sampled_J_parameter_min_ohm',min(n(:,3)),'sampled_J_parameter_max_ohm',max(n(:,3)),...
   'checked_nodes',size(n,1),'curve_logparameter_Z_J',n,'global_uniqueness_proven',false);
  writejson(fullfile(folder,[name '.local.json']),value);
 end
end

function J=scalarJ(j,name)
if strcmp(name,'soft_only'),J=j(1);elseif strcmp(name,'fat_only'),J=j(5);else,J=j(1)+j(5);end
end

function out=crossings(n,target,tol)
f=n(:,2)-target;out=zeros(0,2);
for k=1:numel(f)
 if abs(f(k))<=tol,out=[out;k k];end %#ok<AGROW>
 if k<numel(f)&&abs(f(k))>tol&&abs(f(k+1))>tol&&f(k)*f(k+1)<0,out=[out;k k+1];end %#ok<AGROW>
end
end

function x=hermite_root(a,b,target)
h=b(1)-a(1);assert(h>0);
fun=@(t)(2*t.^3-3*t.^2+1)*a(2)+(t.^3-2*t.^2+t)*h*a(3)+...
 (-2*t.^3+3*t.^2)*b(2)+(t.^3-t.^2)*h*b(3)-target;
assert(fun(0)*fun(1)<0);
t=fzero(fun,[0 1]);x=a(1)+t*h;
if x<=a(1)+1e-12||x>=b(1)-1e-12,x=(a(1)+b(1))/2;end
end

function r=blankrow()
r=struct('family','','id','','size_mm',0,'rho_ohm_m',zeros(1,5),'Z_ohm',0,...
 'J_logrho_ohm',zeros(1,5),'qc',struct(),'provenance','','source','','source_sha256','',...
 'seconds',0,'purpose','');
end

function writejson(path,value)
temp=[path '.writing'];f=fopen(temp,'w','n','UTF-8');assert(f>=0);
fprintf(f,'%s\n',jsonencode(value));fclose(f);movefile(temp,path,'f');
end
