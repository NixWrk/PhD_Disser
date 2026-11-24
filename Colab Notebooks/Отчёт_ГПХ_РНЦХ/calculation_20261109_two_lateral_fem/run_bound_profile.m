function run_bound_profile(config_file)
% Conditional local profile: fix lung rho and re-estimate soft rho using two Zb.
cfg=jsondecode(fileread(config_file));job=cfg.job;maxNumCompThreads(2);
addpath(fullfile(cfg.root,'MATLAB_TRKG4_real_subjects','src'));
baseline=jsondecode(fileread(cfg.baseline_file));rho=baseline.fit.rho_ohm_m(:);
block=job.material_blocks(2);assert(strcmp(trkg4_file_sha256(block.parts_file),block.sha256));
data=load(block.parts_file,'parts');parts=data.parts;clear data;
data=load(job.cem_file,'result');sc=data.result.scenarios;clear data;
if ~iscell(sc),sc=num2cell(sc);end
Cs=cell(2,1);Ps=Cs;
for i=1:2
 ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),job.case_ids{i}),sc),1);
 assert(~isempty(ix));Cs{i}=sc{ix}.grouped_contact_matrix_SI;Ps{i}=ports(sc{ix}.ports);
end
clear sc;last_x=[];last_f=[];last_j=[];history=struct([]);fixed_lung=NaN;result=struct([]);
options=optimoptions('lsqnonlin','SpecifyObjectiveGradient',true,'Display','iter',...
 'MaxFunctionEvaluations',12,'MaxIterations',10,'FunctionTolerance',1e-7,'StepTolerance',1e-6,'OptimalityTolerance',1e-5);
for lung=cfg.lung_nodes_ohm_m(:)'
 fixed_lung=lung;last_x=[];history=struct([]);
 [x,resnorm,residual,exitflag,output]=lsqnonlin(@objective,log(rho(1)),log(.1),log(100),options);
 result=[result;struct('rho_lung_ohm_m',lung,'rho_soft_ohm_m',exp(x),'rmse_ohm',sqrt(mean(residual.^2)),...
  'residual_ohm',residual,'resnorm',resnorm,'exitflag',exitflag,'output',output,'history',history)];
 save_result();fprintf('PROFILE %s lung=%.6g soft=%.8g RMSE=%.8g\n',job.id,lung,exp(x),sqrt(mean(residual.^2)));
end
 function [f,j]=objective(x)
  if ~isequal(x,last_x)
   rr=rho;rr(1)=exp(x);rr(2)=fixed_lung;n=size(parts{1},1);K=sparse(n,n);
   for m=1:5,K=K+parts{m}/rr(m);end
   z=zeros(2,1);J=zeros(2,1);
   for c=1:2
    [T,JJ,~]=trkg4_lateral_cem_response(K,Cs{c},parts,rr,Ps{c},[1 2 4 5],1);
    z(c)=T(4,3);J(c)=JJ(4,3,1);
   end
   last_x=x;last_f=z-job.observed_lateral_ohm(:);last_j=J;
   history=[history;struct('rho_soft_ohm_m',rr(1),'rho_lung_ohm_m',rr(2),'Zb_FEM_ohm',z,'residual_ohm',last_f)];
  end
  f=last_f;j=last_j;
 end
 function save_result()
  d=struct('participant',job.participant_code,'profile',result,'baseline_rho_ohm_m',rho',...
   'baseline_rmse_ohm',baseline.fit.rmse_lateral_ohm,'profile_local_not_global',true,...
   'soft_bounds_ohm_m',[.1 100],'CT_targets_used',false,'baseline_sha256',trkg4_file_sha256(cfg.baseline_file),...
   'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
  fid=fopen(cfg.output,'w','n','UTF-8');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(d,'PrettyPrint',true));fclose(fid);
 end
end
function P=ports(groups)
names={groups.id};
if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
P=zeros(numel(groups),4);for k=1:4,[found,ix]=ismember(pairs{k},names);assert(all(found));P(ix,k)=[1;-1];end
end
