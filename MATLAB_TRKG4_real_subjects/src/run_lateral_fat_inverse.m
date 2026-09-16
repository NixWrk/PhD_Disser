function report=run_lateral_fat_inverse(varargin)
% Experimental inverse problem with fixed full-static CT, C01, finite 5 mm CEM.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_fat_inverse_20260916/capped100');
addParameter(p,'State','inhale');parse(p,varargin{:});state=string(p.Results.State);
assert(any(state==["inhale","exhale"]));cfg=trkg4_config('nik');root=cfg.project_root;
out=fullfile(root,char(p.Results.OutputDir));if ~isfolder(out),mkdir(out);end
trkg4_prepare_runtime(cfg);maxNumCompThreads(5);
base=fullfile(root,'output/exploratory/lateral_array_design_20260916');
fat=fullfile(root,'output/exploratory/lateral_fat_20260916/refined');
obsfile=fullfile(root,'output/exploratory/lateral_real_application_20260915/observations.csv');
S=load(fullfile(base,'prepared_fem.mat'));M=load(fullfile(fat,'matrices_q4096.mat'));S.Kparts=M.Kparts;
Coarse=load(fullfile(fat,'matrices_q1024.mat'));fineparts=S.Kparts;
P=jsondecode(fileread(fullfile(fat,'plan.json')));E=jsondecode(fileread(fullfile(fat,'execution.json')));
assert(isequal(M.identity,E.identity));assert(strcmp(E.identity.plan_sha256,trkg4_file_sha256(fullfile(fat,'plan.json'))));
for ii=1:numel(P.sources),assert(strcmp(trkg4_file_sha256(fullfile(root,P.sources(ii).path)),P.sources(ii).sha256));end
O=sortrows(readtable(obsfile),'L_mm');assert(isequal(O.L_mm,S.sizes));
lit=readtable(fullfile(fat,'literature_values.csv'),'TextType','string');
if state=="inhale",observed=O.Z_inhale_hold_ohm;else,observed=O.Z_exhale_hold_ohm;end
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'matrix_sha256',trkg4_file_sha256(fullfile(fat,'matrices_q4096.mat')),...
 'base_sha256',trkg4_file_sha256(fullfile(base,'prepared_fem.mat')),...
 'coarse_matrix_sha256',trkg4_file_sha256(fullfile(fat,'matrices_q1024.mat')),...
 'forward_plan_sha256',trkg4_file_sha256(fullfile(fat,'plan.json')),...
 'observations_sha256',trkg4_file_sha256(obsfile),...
 'literature_sha256',trkg4_file_sha256(fullfile(fat,'literature_values.csv')));
file=fullfile(out,'fits_'+state+'.csv');predfile=fullfile(out,'predictions_'+state+'.csv');jacfile=fullfile(out,'jacobian_'+state+'.csv');
rows=struct([]);pred=struct([]);jrows=struct([]);
if isfile(file)
 assert(isequal(jsondecode(fileread(file+'.identity.json')),jsondecode(jsonencode(identity))));
 rows=table2struct(readtable(file,'TextType','string'));pred=table2struct(readtable(predfile,'TextType','string'));
 jrows=table2struct(readtable(jacfile,'TextType','string'));
else,local_json(file+'.identity.json',identity);end
% Columns: frequency, fixed fat multiplier, number of free parameters, synthetic.
cases=[50000,1,2,0;100000,1,2,0;50000,.75,2,0;50000,1.25,2,0;50000,1,3,0];
if state=="inhale",cases=[cases;50000,1,3,1];end
opt=optimoptions('lsqnonlin','Display','off','SpecifyObjectiveGradient',true,...
 'FunctionTolerance',1e-9,'StepTolerance',1e-7,'OptimalityTolerance',1e-7,'MaxIterations',45,'MaxFunctionEvaluations',70);
lastx=[];lastz=[];lastJ=[];rho=[];active=[];target=[];evals=0;maxres=0;maxrec=0;
for ci=1:size(cases,1)
 f=cases(ci,1);factor=cases(ci,2);npar=cases(ci,3);synthetic=logical(cases(ci,4));
 id=string(sprintf('%s_f%d_fat%03d_p%d_s%d',state,f,round(100*factor),npar,synthetic));
 if ~isempty(rows)&&any(string({rows.case_id})==id),continue;end
 a=lit(lit.frequency_hz==f,:);getrho=@(name)a.rho_ohm_m(a.tissue==name);
 lung="Lung (Inflated)";if state=="exhale",lung="Lung (Deflated)";end
 rho=[getrho("Muscle"),getrho(lung),getrho("Bone (Cortical)"),getrho("Heart Muscle"),getrho("Fat")*factor];
 active=[1,2];if npar==3,active=[1,2,5];end
 limits=repmat([.1,100],3,1); % User-selected numerical interval; literature is assessed afterward.
 limits=limits(1:npar,:);lb=log(limits(:,1))';ub=log(limits(:,2))';target=observed;
 if synthetic
  truth=rho;truth([1,2,5])=[4,12,getrho("Fat")];target=vector(truth);
 end
 % Two widely separated numerical starts; no physiological interval constrains the search.
 seeds=[3,5,10;8,80,60];starts=log(seeds(:,1:npar));solutions=zeros(2,npar);values=zeros(2,1);flags=zeros(2,1);iters=zeros(2,1);
 lastx=[];evals=0;maxres=0;maxrec=0;t=tic;
 history=struct([]);stopreason="interior_stationary_point";
 for stage=0:0
  for st=1:2
   [x,value,~,flag,info]=lsqnonlin(@residual,min(max(starts(st,:),lb),ub),lb,ub,opt);
   solutions(st,:)=x;values(st)=value;flags(st)=flag;iters(st)=info.iterations;
   fprintf('INVERSE START %s stage=%d start=%d rho=%s rmse=%.7g exit=%d\n',id,stage,st,mat2str(exp(x),6),sqrt(value/9),flag);
  end
  [~,best]=min(values);x=solutions(best,:);[rr,J]=residual(x);rmse=sqrt(mean(rr.^2));
  assert(all(flags>0),'At least one numerical start failed to converge');
  lower=abs(x-lb)<1e-4;upper=abs(x-ub)<1e-4;boundary=lower|upper;
  hrow=struct('stage',stage,'bounds_ohm_m',[exp(lb(:)),exp(ub(:))],'start_log_solutions',solutions,...
   'start_objectives',values,'start_exitflags',flags,'best_rmse_ohm',rmse,'best_Z',lastz);
  history=append(history,hrow);
  if any(boundary),stopreason="user_numeric_boundary_reached";end
 end
 rho(active)=exp(x);limits=[exp(lb(:)),exp(ub(:))];
 S.Kparts=Coarse.Kparts;zc=vector(rho);S.Kparts=fineparts;qd=lastz-zc;
 sv=svd(J);
 row=struct('case_id',id,'state',state,'frequency_hz',f,'fat_factor',factor,'free_parameters',npar,'synthetic',synthetic,...
 'rho1',rho(1),'rho2',rho(2),'rho3_fat',rho(5),'rho_bone',rho(3),'rho_heart',rho(4),...
 'rmse_ohm',sqrt(mean(rr.^2)),'max_residual_ohm',max(abs(rr)),'bound_rho1',boundary(1),'bound_rho2',boundary(2),...
 'bound_rho3',npar==3&&boundary(end),'condition_number',sv(1)/sv(end),'smin_ohm',sv(end),...
 'quadrature_max_difference_ohm',max(abs(qd)),'quadrature_within_tolerance',all(abs(qd)<=.02+.001*abs(lastz)),...
 'bound_expansions',stage,'stop_reason',stopreason,...
 'two_start_log_difference',max(abs(solutions(1,:)-solutions(2,:))),'exitflag',flags(best),...
 'iterations',iters(best),'vector_evaluations',evals,'CEM_solves',9*evals,'seconds',toc(t),...
 'max_relative_residual',maxres,'max_reciprocity_ohm',maxrec);
 for k=1:numel(S.sizes)
  pr=struct('case_id',id,'L_mm',S.sizes(k),'target_Z',target(k),'predicted_Z',lastz(k),'residual_Z',rr(k),'coarse_Z',zc(k));
  jr=struct('case_id',id,'L_mm',S.sizes(k),'dZ_dlnrho1',lastJ(k,1),'dZ_dlnrho2',lastJ(k,2),'dZ_dlnrho3',lastJ(k,3));
  pred=append(pred,pr);jrows=append(jrows,jr);
 end
 rows=append(rows,row);writetable(struct2table(rows),file);writetable(struct2table(pred),predfile);writetable(struct2table(jrows),jacfile);
 local_json(fullfile(out,id+'.json'),struct('identity',identity,'bounds_ohm_m',limits,'start_log_solutions',solutions,...
  'start_objectives',values,'start_exitflags',flags,'singular_values',sv,'fitted_rho_all',rho,'synthetic',synthetic,...
  'history',history,'stop_reason',stopreason,'physiological_bounds_used',false));
 fprintf('FAT INVERSE %s rho=[%.7g %.7g %.7g] RMSE=%.7g boundary=%s %ds\n',id,rho(1),rho(2),rho(5),row.rmse_ohm,mat2str(boundary),round(row.seconds));
end
% Profile the lung parameter inside the user-selected interval; each point refits rho1.
profilefile=fullfile(out,'profile_'+state+'.csv');profile_rows=struct([]);
if isfile(profilefile),profile_rows=table2struct(readtable(profilefile,'TextType','string'));end
profile_values=[.1,1,5,10,20,40,60,100];a=lit(lit.frequency_hz==50000,:);getrho=@(name)a.rho_ohm_m(a.tissue==name);
rho=[getrho("Muscle"),10,getrho("Bone (Cortical)"),getrho("Heart Muscle"),getrho("Fat")];active=1;target=observed;
for fixed=profile_values
 if ~isempty(profile_rows)&&any(abs([profile_rows.rho2]-fixed)<1e-10),continue;end
 rho(2)=fixed;lastx=[];evals=0;maxres=0;maxrec=0;t=tic;
 [px,value,~,flag]=lsqnonlin(@residual,log(4),log(.1),log(100),opt);assert(flag>0);
 [pr,pj]=residual(px);rho(1)=exp(px);
 S.Kparts=Coarse.Kparts;pc=vector(rho);S.Kparts=fineparts;
 row=struct('state',state,'rho2',fixed,'rho1',rho(1),'rho3_fat',rho(5),'rmse_ohm',sqrt(value/9),...
  'exitflag',flag,'CEM_solves',9*(evals+1),'seconds',toc(t),...
  'quadrature_max_difference_ohm',max(abs(lastz-pc)),'quadrature_within_tolerance',all(abs(lastz-pc)<=.02+.001*abs(lastz)));
 profile_rows=append(profile_rows,row);writetable(struct2table(profile_rows),profilefile);
 local_json(fullfile(out,sprintf('profile_%s_rho2_%g.json',state,fixed)),struct('rho',[rho(1),fixed,rho(5)],'Z',lastz,'coarse_Z',pc,'residual',pr,'dZ_dlnrho1',pj));
 fprintf('PROFILE %s rho2=%g rho1=%g rmse=%g\n',state,fixed,rho(1),row.rmse_ohm);
end

% Independent central finite differences of all three log-resistivity derivatives.
if state=="inhale" && ~isfile(fullfile(out,'derivative_qc.json'))
 rho=[3,10,48.445126062,5.11678442,23.108608];[z0,J0]=vector(rho);h=1e-4;fd=zeros(size(J0));
 for j=1:3
  which=[1,2,5];rp=rho;rm=rho;rp(which(j))=rp(which(j))*exp(h);rm(which(j))=rm(which(j))*exp(-h);
  fd(:,j)=(vector(rp)-vector(rm))/(2*h);
 end
 error=sqrt(sum((fd-J0).^2,1))./sqrt(sum(fd.^2,1));assert(all(error<1e-4));
 local_json(fullfile(out,'derivative_qc.json'),struct('complete',true,'identity',identity,'rho',rho,'step_log',h,...
  'relative_column_errors',error,'analytic',J0,'finite_difference',fd,'Z',z0,'CEM_solves',63));
end
report=struct('complete',true,'identity',identity,'state',state,'fits',numel(rows),'global_optimum_certified',false,...
 'no_new_localization',true,'same_inspiratory_geometry',true,'rho3_scenario_is_not_population_interval',true,'physiological_bounds_used',false,...
 'CEM_solves_profile',sum([profile_rows.CEM_solves]),'profile_sha256',trkg4_file_sha256(profilefile),...
 'CEM_solves_optimization',sum([rows.CEM_solves]),'fit_sha256',trkg4_file_sha256(file),...
 'predictions_sha256',trkg4_file_sha256(predfile),'jacobian_sha256',trkg4_file_sha256(jacfile));
local_json(fullfile(out,'execution_'+state+'.json'),report);

function [r,J]=residual(x)
 if isempty(lastx)||~isequal(x,lastx)
  rhos=rho;rhos(active)=exp(x);[lastz,lastJ]=vector(rhos);lastx=x;evals=evals+1;
 end
 r=lastz-target;if numel(active)==1,ix=1;elseif numel(active)==2,ix=[1,2];else,ix=[1,2,3];end;J=lastJ(:,ix);
end
function [z,J]=vector(r)
 z=zeros(numel(S.sizes),1);J=zeros(numel(S.sizes),3);
 K=sparse(size(S.Kall,1),size(S.Kall,2));for jj=1:5,K=K+S.Kparts{jj}/r(jj);end;n=size(K,1);
 free=[S.free(:);n+(1:4)'];rhs=sparse(n+4,2);rhs(n+[1,4],1)=[1;-1];rhs(n+[2,3],2)=[1;-1];
 for kk=1:numel(S.sizes)
  H=S.C{kk};H(1:n,1:n)=H(1:n,1:n)+K;B=H(free,free);U=zeros(n+4,2);
  U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
  res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');z(kk)=rhs(:,2)'*U(:,1);
  rec=abs(z(kk)-rhs(:,1)'*U(:,2));assert(res<1e-7&&rec<1e-7&&isfinite(z(kk)));
  maxres=max(maxres,res);maxrec=max(maxrec,rec);
  which=[1,2,5];for jj=1:3,J(kk,jj)=U(1:n,2)'*(S.Kparts{which(jj)}*U(1:n,1))/r(which(jj));end
 end
end
end
function x=append(x,row)
if isempty(x),x=row;else,x(end+1)=row;end
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
