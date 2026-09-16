function report=run_lateral_tissue_inverse_probe(varargin)
% Nonlinear synthetic check of organ-to-background replacement; fixed C01.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_tissue_ablation_20260916');parse(p,varargin{:});
cfg=trkg4_config('nik');root=cfg.project_root;out=fullfile(root,char(p.Results.OutputDir));
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
base=fullfile(root,'output/exploratory/lateral_array_design_20260916');S=load(fullfile(base,'prepared_fem.mat'));
T=load(fullfile(out,'material_matrices.mat'));S.Kparts=T.Kparts;
Q=readtable(fullfile(out,'requests.csv'),'TextType','string');D=readtable(fullfile(out,'responses.csv'),'TextType','string');
E=jsondecode(fileread(fullfile(out,'execution.json')));
assert(E.complete&&strcmp(E.responses_sha256,trkg4_file_sha256(fullfile(out,'responses.csv'))));
assert(strcmp(E.identity.worker_sha256,trkg4_file_sha256(fullfile(root,'src/run_lateral_tissue_ablation.m'))));
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
    'forward_execution_sha256',trkg4_file_sha256(fullfile(out,'execution.json')),...
    'prepared_sha256',trkg4_file_sha256(fullfile(out,'material_matrices.mat')),...
    'base_prepared_sha256',trkg4_file_sha256(fullfile(base,'prepared_fem.mat')));
targetfile=fullfile(out,'synthetic_inverse.csv');predfile=fullfile(out,'synthetic_inverse_predictions.csv');
rows=struct([]);predictions=struct([]);evaluations=0;lastx=[];lastz=[];lastJ=[];rho_fixed=[];model="";target=[];
if isfile(targetfile)
    assert(isequal(jsondecode(fileread([targetfile '.identity.json'])),jsondecode(jsonencode(identity))));
    rows=table2struct(readtable(targetfile,'TextType','string'));
    predictions=table2struct(readtable(predfile,'TextType','string'));
else,local_json([targetfile '.identity.json'],identity);end
pairs=["two_tissue","bones","omit_bone";"bones","bones_heart","omit_heart_after_bone";"two_tissue","bones_heart","omit_both"];
% The interval contains the selected ITIS inputs. It is a scenario bound, not an individual norm.
bounds=[1.3774104683195594,10;3.8172830517819216,24.074073806584366];
opt=optimoptions('lsqnonlin','Display','off','SpecifyObjectiveGradient',true,...
    'FunctionTolerance',1e-10,'StepTolerance',1e-8,'OptimalityTolerance',1e-8,'MaxIterations',35,'MaxFunctionEvaluations',60);
for state=["inhale","exhale"]
  rq=find(Q.frequency_hz==50000&Q.state==state&Q.model=="bones_heart",1);
  truth=[Q.rho1(rq),Q.rho2(rq)];rho_fixed=[truth,Q.rho_bone(rq),Q.rho_heart(rq)];
  for i=1:size(pairs,1)
    model=pairs(i,1);targetmodel=pairs(i,2);label=pairs(i,3);
    if ~isempty(rows)&&any(string({rows.state})==state & string({rows.comparison})==label),continue;end
    g=sortrows(D(D.frequency_hz==50000&D.state==state&D.model==targetmodel&D.kind=="literature",:),'L_mm');
    assert(isequal(g.L_mm,S.sizes));target=g.Z;lastx=[];evaluations=0;t0=tic;
    % Start at known synthetic truth. A second start checks local dependence.
    starts=[log(truth);log([sqrt(prod(bounds(1,:))),sqrt(prod(bounds(2,:)))])];
    solutions=zeros(2,2);objectives=zeros(2,1);exits=zeros(2,1);iters=zeros(2,1);
    for st=1:2
      [x,value,~,flag,output]=lsqnonlin(@residual,starts(st,:),log(bounds(:,1))',log(bounds(:,2))',opt);
      solutions(st,:)=x;objectives(st)=value;exits(st)=flag;iters(st)=output.iterations;
    end
    [~,best]=min(objectives);x=solutions(best,:);[rr,JJ]=residual(x);estimated=exp(x);
    assert(all(exits>0),'Synthetic fit did not converge');
    agreement=max(abs(solutions(1,:)-solutions(2,:)));
    row=struct('frequency_hz',50000,'state',state,'comparison',label,'reduced_model',model,'target_model',targetmodel,...
      'truth_rho1',truth(1),'truth_rho2',truth(2),'estimated_rho1',estimated(1),'estimated_rho2',estimated(2),...
      'error_rho1_pct',100*(estimated(1)/truth(1)-1),'error_rho2_pct',100*(estimated(2)/truth(2)-1),...
      'fit_rmse_ohm',sqrt(mean(rr.^2)),'max_residual_ohm',max(abs(rr)),...
      'bound_hit',any(abs(x-log(bounds(:,1))')<1e-5|abs(x-log(bounds(:,2))')<1e-5),...
      'two_start_log_agreement',agreement,'exitflag',exits(best),'iterations',iters(best),...
      'forward_vector_evaluations',evaluations,'CEM_solves',9*evaluations,'seconds',toc(t0),'condition_number',cond(JJ));
    for k=1:numel(S.sizes)
      pr=struct('state',state,'comparison',label,'L_mm',S.sizes(k),'target_Z',target(k),'predicted_Z',target(k)+rr(k),'residual_Z',rr(k));
      if isempty(predictions),predictions=pr;else,predictions(end+1)=pr;end
    end
    if isempty(rows),rows=row;else,rows(end+1)=row;end
    writetable(struct2table(rows),targetfile);writetable(struct2table(predictions),predfile);
    fprintf('INVERSE %s %s rho=[%.6g %.6g] errors=[%.4g%% %.4g%%] rmse=%.6g %g evaluations %.1fs\n',...
      state,label,estimated(1),estimated(2),row.error_rho1_pct,row.error_rho2_pct,row.fit_rmse_ohm,evaluations,row.seconds);
  end
end
report=struct('complete',true,'identity',identity,'fits',numel(rows),'bounds_ohm_m',bounds,...
  'starts_per_fit',2,'global_optimum_certified',false,'synthetic_only',true,...
  'source_frequency_hz',50000,'CEM_solves',sum([rows.CEM_solves]),...
  'fits_sha256',trkg4_file_sha256(targetfile),'predictions_sha256',trkg4_file_sha256(predfile));
local_json(fullfile(out,'synthetic_inverse_qc.json'),report);

function [r,J]=residual(x)
    if isempty(lastx)||~isequal(x,lastx)
        lastz=zeros(numel(S.sizes),1);lastJ=zeros(numel(S.sizes),2);rho=rho_fixed;rho(1:2)=exp(x);
        for kk=1:numel(S.sizes),[lastz(kk),lastJ(kk,:)]=local_forward(S,kk,rho,model);end
        lastx=x;evaluations=evaluations+1;
    end
    r=lastz-target;J=lastJ;
end
end
function [z,J]=local_forward(S,k,r,model)
Kb=S.Kparts{3};Kh=S.Kparts{4};Kl=S.Kparts{2};Ks=S.Kparts{1};
if model=="two_tissue",Ks=Ks+Kb+Kh;Kb=sparse(size(Ks,1),size(Ks,2));
elseif model=="bones",Ks=Ks+Kh;
else,error('Unsupported reduced model');end
K=Ks/r(1)+Kl/r(2)+Kb/r(3);n=size(K,1);
H=S.C{k};H(1:n,1:n)=H(1:n,1:n)+K;free=[S.free(:);n+(1:4)'];
rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];
B=H(free,free);U=zeros(n+4,2);U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
z=rhs(:,2)'*U(:,1);assert(res<1e-7&&abs(z-rhs(:,1)'*U(:,2))<1e-7);
J=[U(1:n,2)'*(Ks*U(1:n,1))/r(1),U(1:n,2)'*(Kl*U(1:n,1))/r(2)];
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end

