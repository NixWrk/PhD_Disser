function report=run_trkg4_geometry_validation_cem(varargin)
% Exact finite-contact fit to measured inhale data at the frozen CT pose.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/geometry_validation_20260910');
parse(p,varargin{:});cfg=trkg4_config('nik');root=cfg.project_root;out=char(p.Results.OutputDir);
if ~isfolder(out),out=fullfile(root,out);end
planfile=fullfile(out,'plan.json');plan=jsondecode(fileread(planfile));
source=fullfile(root,'output','exploratory','geometry_reduction_20260910');
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
assert(exist('fmincon','file')==2,'Optimization Toolbox is required');
prepared=fullfile(source,'prepared_fem.mat');load(prepared,'Kall','Klung','C','free','sizes');
old=jsondecode(fileread(fullfile(out,'fem_identity.json')));
assert(strcmp(trkg4_file_sha256(prepared),old.prepared_sha256),'Prepared model changed');
T=readtable(fullfile(out,'experimental_observations.csv'));
assert(isequal(T.L_mm,sizes));y=T.Z_inhale_hold_ohm;
n=size(Kall,1);ns=numel(sizes);Ksoft=Kall-Klung{1};Klung=Klung{1};
q=sparse(n+4,2);q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];allfree=[free;n+(1:4)'];
lb=log(plan.bounds_ohm_m(:,1));ub=log(plan.bounds_ohm_m(:,2));
F=readtable(fullfile(out,'experimental_fits.csv'),'TextType','string');r=F(F.model=="reference",:);
anchors=plan.anchors;mi=find(strcmp({anchors.tissue},'Muscle'));li=find(strcmp({anchors.tissue},'Lung (Inflated)'));
starts=[log([r.rho1;r.rho2]),(lb+ub)/2,log([anchors(mi).resistivity_50khz_ohm_m;anchors(li).resistivity_50khz_ohm_m])];
starts=min(max(starts,lb),ub);
cache=containers.Map('KeyType','char','ValueType','any');trace=struct([]);solutions=struct([]);
opts=optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true, ...
 'Display','iter','MaxIterations',35,'OptimalityTolerance',1e-6,'StepTolerance',1e-9);
for si=1:size(starts,2)
 [x,loss,flag,info]=fmincon(@objective,starts(:,si),[],[],[],[],lb,ub,[],opts);
 [~,~,z,J]=objective(x);sv=svd(J);rho=exp(x);
 result=struct('start_index',si,'rho1',rho(1),'rho2',rho(2),'rmse_ohm',sqrt(loss), ...
  'exitflag',flag,'iterations',info.iterations,'function_count',info.funcCount, ...
  'condition_log_parameters',sv(1)/sv(end));
 if isempty(solutions),solutions=result;else,solutions(end+1)=result;end %#ok<AGROW>
 writetable(struct2table(solutions),fullfile(out,'cem_multistart.csv'));
end
[~,bi]=min([solutions.rmse_ohm]);best=solutions(bi);[~,~,z,J]=objective(log([best.rho1;best.rho2]));
writetable(table(sizes,y,z,z-y,J(:,1),J(:,2),'VariableNames', ...
 {'L_mm','observed_ohm','predicted_ohm','residual_ohm','dZ_dlogrho1','dZ_dlogrho2'}),fullfile(out,'cem_experimental_predictions.csv'));
report=struct('status','exploratory_hypothesis_not_validated','best',best, ...
 'plan_sha256',trkg4_file_sha256(planfile),'prepared_sha256',trkg4_file_sha256(prepared), ...
 'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']), ...
 'observations_sha256',trkg4_file_sha256(fullfile(out,'experimental_observations.csv')), ...
 'pose_fixed_from_full_data',true,'independent_end_to_end_validation',false, ...
 'global_optimality_proved',false,'unique_field_evaluations',cache.Count);
local_json(fullfile(out,'cem_fit_summary.json'),report);

 function [loss,gradient,z,J]=objective(x)
  key=sprintf('%.17g_',x);if isKey(cache,key),v=cache(key);loss=v.loss;gradient=v.gradient;z=v.z;J=v.J;return;end
  ticid=tic;rho=exp(x);K=Ksoft/rho(1)+Klung/rho(2);z=zeros(ns,1);J=zeros(ns,2);maxres=0;maxrec=0;
  for k=1:ns
   H=C{k};H(1:n,1:n)=H(1:n,1:n)+K;
   V=zeros(n+4,2);A=H(allfree,allfree);V(allfree,:)=decomposition((A+A')/2,'chol')\q(allfree,:);
   res=norm(H(allfree,:)*V-q(allfree,:),'fro')/norm(q(allfree,:),'fro');
   z(k)=q(:,2)'*V(:,1);zr=q(:,1)'*V(:,2);
   assert(isfinite(z(k))&&res<1e-7&&abs(z(k)-zr)<1e-7,'CEM solve failed');
   maxres=max(maxres,res);maxrec=max(maxrec,abs(z(k)-zr));
   J(k,1)=V(1:n,2)'*(Ksoft*V(1:n,1))/rho(1);
   J(k,2)=V(1:n,2)'*(Klung*V(1:n,1))/rho(2);
  end
  residual=z-y;loss=mean(residual.^2);gradient=2*(J'*residual)/ns;
  cache(key)=struct('loss',loss,'gradient',gradient,'z',z,'J',J);
  row=struct('evaluation',cache.Count,'rho1',rho(1),'rho2',rho(2),'rmse_ohm',sqrt(loss), ...
   'relative_residual',maxres,'reciprocity_abs_ohm',maxrec,'elapsed_seconds',toc(ticid));
  if isempty(trace),trace=row;else,trace(end+1)=row;end %#ok<AGROW>
  writetable(struct2table(trace),fullfile(out,'cem_evaluation_trace.csv'));
  fprintf('CEM evaluation %d: rho1 %.7g rho2 %.7g RMSE %.7g (%.1fs)\n',cache.Count,rho(1),rho(2),sqrt(loss),toc(ticid));
 end
end

function local_json(p,value)
f=fopen(p,'w');assert(f>0);c=onCleanup(@()fclose(f));fprintf(f,'%s',jsonencode(value,'PrettyPrint',true));
end
