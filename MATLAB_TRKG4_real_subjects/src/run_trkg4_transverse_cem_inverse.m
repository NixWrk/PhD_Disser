function report=run_trkg4_transverse_cem_inverse(varargin)
% Exact bounded CEM inversion of synthetic reference data, study 20.16.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/transverse_geometry_20260911');parse(p,varargin{:});
cfg=trkg4_config('nik');root=cfg.project_root;out=char(p.Results.OutputDir);if ~isfolder(out),out=fullfile(root,out);end
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
G=jsondecode(fileread(fullfile(out,'geometry.json')));M=load(fullfile(out,'geometry_masks.mat'));
load(fullfile(out,'prepared_fem.mat'),'Kall','Klung','C','free','sizes','identity');
assert(strcmp(identity.geometry_sha256,trkg4_file_sha256(fullfile(out,'geometry.json'))));
T=readtable(fullfile(out,'cem_baseline.csv'),'TextType','string');T=T(T.model=="reference",:);T=sortrows(T,'L_mm');
assert(isequal(sizes,T.L_mm));y=T.Z;Jtruth=[T.dZ_drho1 T.dZ_drho2];
I=readtable(fullfile(out,'cem_initial.csv'),'TextType','string');
lb=log(G.bounds_ohm_m(:,1));ub=log(G.bounds_ohm_m(:,2));ns=numel(sizes);n=size(Kall,1);
allfree=[free;n+(1:4)'];q=sparse(n+4,2);q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];
cache=containers.Map('KeyType','char','ValueType','any');trace=struct([]);solutions=struct([]);predictions=struct([]);
opts=optimoptions('lsqnonlin','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true,'Display','iter', ...
 'MaxIterations',30,'OptimalityTolerance',1e-8,'FunctionTolerance',1e-11,'StepTolerance',1e-9);
for mi=2:3
 initial=I(I.model==string(M.model_names{mi}),:);starts=[log([initial.rho1;initial.rho2]),(lb+ub)/2];starts=min(max(starts,lb),ub);
 for si=1:size(starts,2)
  [x,resnorm,~,flag,info]=lsqnonlin(@residual_objective,starts(:,si),lb,ub,opts);loss=resnorm/ns;
  [~,~,z,Jlog]=objective(x);rho=exp(x);sv=svd(Jlog);J=Jlog./rho';
  truthPulse=[4e-4;16e-3];pulse=J\(Jtruth*truthPulse);
  row=struct('model',string(M.model_names{mi}),'start_index',si,'rho1',rho(1),'rho2',rho(2), ...
    'error1_pct',100*(rho(1)/4-1),'error2_pct',100*(rho(2)/16-1),'rmse_ohm',sqrt(loss),'exitflag',flag, ...
    'iterations',info.iterations,'function_count',info.funcCount,'condition_log',sv(1)/sv(end), ...
    'pulse1_error_pct',100*(pulse(1)/truthPulse(1)-1),'pulse2_error_pct',100*(pulse(2)/truthPulse(2)-1));
  if isempty(solutions),solutions=row;else,solutions(end+1)=row;end
  writetable(struct2table(solutions),fullfile(out,'cem_inverse.csv'));
  for k=1:ns
   r=struct('model',string(M.model_names{mi}),'start_index',si,'L_mm',sizes(k),'observed',y(k),'predicted',z(k), ...
     'dZ_drho1',J(k,1),'dZ_drho2',J(k,2));
   if isempty(predictions),predictions=r;else,predictions(end+1)=r;end
  end
  writetable(struct2table(predictions),fullfile(out,'cem_inverse_predictions.csv'));
 end
end
report=struct('completed',true,'scope','Exact CEM bounded inverse at one synthetic baseline rho=[4,16], two starts per spatial approximation.', ...
  'unique_evaluations',cache.Count,'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']), ...
  'geometry_sha256',identity.geometry_sha256,'global_minimum_proven',false);
local_json(fullfile(out,'cem_inverse_summary.json'),report);
 function [r,J]=residual_objective(x)
  [~,~,z,J]=objective(x);r=z-y;
 end
 function [loss,grad,z,J]=objective(x)
  key=sprintf('%d_%.17g_%.17g',mi,x(1),x(2));
  if isKey(cache,key),v=cache(key);loss=v.loss;grad=v.grad;z=v.z;J=v.J;return;end
  started=tic;rho=exp(x);Kl=Klung{mi};Ks=Kall-Kl;K=Ks/rho(1)+Kl/rho(2);z=zeros(ns,1);J=zeros(ns,2);resmax=0;recmax=0;
  for k=1:ns
   H=C{k};H(1:n,1:n)=H(1:n,1:n)+K;A=H(allfree,allfree);assert(norm(A-A','fro')/norm(A,'fro')<1e-12);
   U=zeros(n+4,2);U(allfree,:)=decomposition((A+A')/2,'chol')\q(allfree,:);
   res=norm(H(allfree,:)*U-q(allfree,:),'fro')/norm(q(allfree,:),'fro');rec=abs(q(:,2)'*U(:,1)-q(:,1)'*U(:,2));
   assert(res<1e-7 && rec<1e-7);resmax=max(resmax,res);recmax=max(recmax,rec);z(k)=q(:,2)'*U(:,1);
   J(k,:)=[U(1:n,2)'*(Ks*U(1:n,1))/rho(1),U(1:n,2)'*(Kl*U(1:n,1))/rho(2)];
  end
  residual=z-y;loss=mean(residual.^2);grad=2*J'*residual/ns;cache(key)=struct('loss',loss,'grad',grad,'z',z,'J',J);
  row=struct('model',string(M.model_names{mi}),'evaluation',cache.Count,'rho1',rho(1),'rho2',rho(2),'rmse_ohm',sqrt(loss), ...
    'relative_residual',resmax,'reciprocity_abs',recmax,'elapsed_seconds',toc(started));
  if isempty(trace),trace=row;else,trace(end+1)=row;end
  writetable(struct2table(trace),fullfile(out,'cem_inverse_trace.csv'));
  fprintf('CEM inverse %s eval=%d rho=%.6g %.6g RMSE=%.6g %.1fs\n',M.model_names{mi},cache.Count,rho(1),rho(2),sqrt(loss),row.elapsed_seconds);
 end
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
