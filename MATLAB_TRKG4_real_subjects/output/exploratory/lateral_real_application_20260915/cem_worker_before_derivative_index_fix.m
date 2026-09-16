function report=run_lateral_real_application(varargin)
% Measured inspiration/expiration: selected model 4 and full CT, finite electrodes.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_real_application_20260915');parse(p,varargin{:});
cfg=trkg4_config('nik');root=cfg.project_root;out=char(p.Results.OutputDir);
if ~isfile(fullfile(out,'input_contract.json')),out=fullfile(root,out);end
assert(isfile(fullfile(out,'input_contract.json')),'Run Python prepare first.');
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
source=fullfile(root,'output','exploratory','transverse_consistency_20260914');
G=jsondecode(fileread(fullfile(source,'geometry.json')));E=jsondecode(fileread(fullfile(source,'input_contract.json')));
A=jsondecode(fileread(fullfile(out,'input_contract.json')));
assert(strcmp(A.candidate,'C01')&&strcmp(A.subject_id,'exp02_nik'));
assert(norm(A.centre_mm(:)-E.centre_mm(:))<1e-10&&norm(A.bounds_ohm_m-G.bounds_ohm_m,'fro')<1e-12);
prepared=fullfile(source,'prepared_fem.mat');S=load(prepared,'Kall','Klung','C','free','sizes','identity');
assert(strcmp(S.identity.geometry_sha256,trkg4_file_sha256(fullfile(source,'geometry.json'))));
assert(strcmp(G.masks_sha256,trkg4_file_sha256(fullfile(source,'geometry_masks.mat'))));
names=string(G.model_names);assert(names(1)=="reference"&&names(3)=="variable_transverse");
sizes=double(S.sizes(:));ns=numel(sizes);n=size(S.Kall,1);assert(ns==9&&isequal(sizes,A.sizes_mm(:)));
T=readtable(fullfile(out,'observations.csv'));assert(isequal(sizes,T.L_mm));
assert(strcmp(A.prepared_sha256.observations_csv,trkg4_file_sha256(fullfile(out,'observations.csv'))));
y=[T.Z_inhale_hold_ohm,T.Z_exhale_hold_ohm];assert(all(isfinite(y),'all'));
P=readtable(fullfile(out,'pem_fits.csv'),'TextType','string');
identity=struct('prepared_sha256',trkg4_file_sha256(prepared),'geometry_sha256',trkg4_file_sha256(fullfile(source,'geometry.json')),...
 'input_contract_sha256',trkg4_file_sha256(fullfile(out,'input_contract.json')),'observations_sha256',trkg4_file_sha256(fullfile(out,'observations.csv')),...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),'contact_ohm_m2',E.contact_ohm_m2,'current_ampere',1,...
 'requested_area_mm2',E.requested_area_mm2,'sizes_mm',sizes.','signal_units_accepted',true);
idfile=fullfile(out,'cem_identity.json');
if isfile(idfile),assert(isequal(jsondecode(fileread(idfile)),jsondecode(jsonencode(identity))),'Run identity changed; use a new output directory.');else,local_json(idfile,identity);end
allfree=[S.free(:);n+(1:4)'];rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];
cache=containers.Map('KeyType','char','ValueType','any');trace=struct([]);qc=struct([]);
lb=log(G.bounds_ohm_m(:,1));ub=log(G.bounds_ohm_m(:,2));
opts=optimoptions('lsqnonlin','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true,'Display','iter',...
 'MaxIterations',35,'OptimalityTolerance',1e-7,'FunctionTolerance',1e-11,'StepTolerance',1e-9);
baseline=readtable(fullfile(source,'cem_baseline.csv'),'TextType','string');
for mi=[3 1]
 model=names(mi);
 [z,J]=forward(log([4;16]));control=sortrows(baseline(baseline.model==model,:),'L_mm');
 assert(isequal(control.L_mm,sizes));Jcontrol=[control.dZ_drho1,control.dZ_drho2].*[4 16];
 dz=max(abs(z-control.Z));dj=norm(J-Jcontrol,'fro')/norm(Jcontrol,'fro');assert(dz<1e-7&&dj<1e-7);
 checkrow=struct('model',model,'baseline_Z_max_error_ohm',dz,'baseline_J_relative_error',dj);
 if isempty(qc),qc=checkrow;else,qc(end+1)=checkrow;end
 for state=["inhale","exhale"]
  scenario=state;iy=1+double(state=="exhale");b=P(P.model==model & P.state==state,:);assert(height(b)==1);
  fit([log([b.rho1;b.rho2]),(lb+ub)/2],lb,ub);
 end
 if model=="variable_transverse"
  scenario="shared";iy=0;b=jsondecode(fileread(fullfile(out,'pem_shared.json')));
  fit([log(b.rho(:)),(lb([1 2 2])+ub([1 2 2]))/2],lb([1 2 2]),ub([1 2 2]));
  b=jsondecode(fileread(fullfile(out,'cem_variable_transverse_inhale.json')));x=log(b.best.rho(:));[~,J]=forward(x);fd=zeros(ns,2);
  for k=1:2,dx=zeros(2,1);dx(k)=1e-4;zp=forward(x+dx);zm=forward(x-dx);fd(:,k)=(zp-zm)/(2e-4);end
  err=norm(fd-J,'fro')/norm(J,'fro');assert(err<1e-5,'CEM derivative finite difference failed.');
  local_json(fullfile(out,'cem_derivative_qc.json'),struct('model',model,'state','inhale','relative_error',err,'step_logrho',1e-4));
 end
end
report=struct('complete',true,'status','exploratory_hypothesis_not_validated','identity',identity,...
 'baseline_reproduction',qc,'new_cached_parameter_evaluations',cache.Count,'global_optimum_proved',false,...
 'geometry_fixed',true,'new_mesh',false,'independent_tissue_validation',false);
local_json(fullfile(out,'cem_execution.json'),report);

 function fit(starts,lo,hi)
  stem=fullfile(out,char("cem_"+model+"_"+scenario));
  if isfile([stem '.json']),old=jsondecode(fileread([stem '.json']));assert(isequal(old.identity,jsondecode(jsonencode(identity)))&&old.complete);return;end
  solutions=struct([]);
  for si=1:size(starts,2)
   initial=min(max(starts(:,si),lo),hi);
   [x,resnorm,~,flag,info]=lsqnonlin(@objective,initial,lo,hi,opts);assert(flag>0,'CEM optimization did not converge.');
   [r,J]=objective(x);sv=svd(J);
   fitrow=struct('start',si,'rho',exp(x).','rmse_ohm',sqrt(resnorm/numel(r)),'exitflag',flag,...
    'iterations',info.iterations,'function_count',info.funcCount,'condition_log_parameters',sv(1)/sv(end),...
    'at_bound',(abs(x-lo)<1e-6 | abs(x-hi)<1e-6).','residual_gradient_log',(2*J'*r/numel(r)).'); %#ok<AGROW>
   if isempty(solutions),solutions=fitrow;else,solutions(end+1)=fitrow;end
   local_json([stem '_starts.json'],solutions);
  end
  [~,bestindex]=min([solutions.rmse_ohm]);best=solutions(bestindex);x=log(best.rho(:));[res,~]=objective(x);
  if scenario=="shared"
   [zin,Jin]=forward(x([1 2]));[zex,Jex]=forward(x([1 3]));
   result=table([sizes;sizes],[repmat("inhale",ns,1);repmat("exhale",ns,1)],y(:),[zin;zex],...
    [Jin(:,1);Jex(:,1)],[Jin(:,2);Jex(:,2)],'VariableNames',{'L_mm','state','observed_ohm','predicted_ohm','dZ_dlogrho1','dZ_dlogrho2'});
  else
   [z,Js]=forward(x);result=table(sizes,y(:,iy),z,Js(:,1),Js(:,2),...
    'VariableNames',{'L_mm','observed_ohm','predicted_ohm','dZ_dlogrho1','dZ_dlogrho2'});
  end
  writetable(result,[stem '_predictions.csv']);
  local_json([stem '.json'],struct('complete',true,'identity',identity,'model',model,'scenario',scenario,'best',best,...
   'starts',solutions,'start_rmse_spread',max([solutions.rmse_ohm])-min([solutions.rmse_ohm]),...
   'residual_rms_recomputed',sqrt(mean(res.^2)),'global_minimum_proven',false));
 end
 function [r,J]=objective(x)
  if scenario=="shared"
   [za,Ja]=forward(x([1 2]));[zb,Jb]=forward(x([1 3]));r=[za-y(:,1);zb-y(:,2)];
   J=[Ja(:,1),Ja(:,2),zeros(ns,1);Jb(:,1),zeros(ns,1),Jb(:,2)];
  else,[z,J]=forward(x);r=z-y(:,iy);end
 end
 function [z,J]=forward(x)
  key=sprintf('%d_%.17g_%.17g',mi,x(1),x(2));if isKey(cache,key),v=cache(key);z=v.z;J=v.J;return;end
  t0=tic;rho=exp(x);Kl=S.Klung{mi};Ks=S.Kall-Kl;K=Ks/rho(1)+Kl/rho(2);z=zeros(ns,1);J=zeros(ns,2);maxres=0;maxrec=0;
  for k=1:ns
   H=S.C{k};H(1:n,1:n)=H(1:n,1:n)+K;B=H(allfree,allfree);assert(norm(B-B','fro')/norm(B,'fro')<1e-12);
   U=zeros(n+4,2);U(allfree,:)=decomposition((B+B')/2,'chol')\rhs(allfree,:);
   res=norm(H(allfree,:)*U-rhs(allfree,:),'fro')/norm(rhs(allfree,:),'fro');
   z(k)=rhs(:,2)'*U(:,1);rec=abs(z(k)-rhs(:,1)'*U(:,2));assert(res<1e-7&&rec<1e-7&&isfinite(z(k)));
   maxres=max(maxres,res);maxrec=max(maxrec,rec);
   J(k,:)=[U(1:n,2)'*(Ks*U(1:n,1))/rho(1),U(1:n,2)'*(Kl*U(1:n,1))/rho(2)];
  end
  cache(key)=struct('z',z,'J',J);row=struct('model',names(mi),'evaluation',cache.Count,'rho1',rho(1),'rho2',rho(2),...
   'relative_residual',maxres,'reciprocity_abs_ohm',maxrec,'elapsed_seconds',toc(t0));
  if isempty(trace),trace=row;else,trace(end+1)=row;end
  writetable(struct2table(trace),fullfile(out,'cem_trace.csv'));
  fprintf('Real CEM %s eval=%d rho=[%.8g %.8g] %.1fs\n',names(mi),cache.Count,rho(1),rho(2),row.elapsed_seconds);
 end
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
