function report=run_c01_baseline_models(varargin)
% Bounded C01 baseline worker: direct PEM first, then reference CEM.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/c01_baseline_models_20260914');
addParameter(p,'Stage','pem');parse(p,varargin{:});opt=p.Results;
cfg=trkg4_config('nik');root=cfg.project_root;out=char(opt.OutputDir);
if ~isfolder(out),out=fullfile(root,out);end
if ~isfolder(out),mkdir(out);end
stage=lower(char(opt.Stage));assert(any(strcmp(stage,{'pem','cem','cem_baseline'})),...
 'Stage must be pem, cem, or cem_baseline.');
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
geometry_file=fullfile(out,'geometry.json');masks_file=fullfile(out,'geometry_masks.mat');
source=fullfile(root,'output','exploratory','transverse_consistency_20260914');prepared=fullfile(source,'prepared_fem.mat');
assert(isfile(geometry_file)&&isfile(masks_file)&&isfile(prepared),...
 'C01 geometry and immutable transverse prepared inputs are required.');
G=jsondecode(fileread(geometry_file));mask_sha=trkg4_file_sha256(masks_file);source_sha=trkg4_file_sha256(prepared);
assert(isfield(G,'masks_sha256')&&string(G.masks_sha256)==string(mask_sha),'geometry masks hash mismatch.');
assert(string(local_prepared_hash(G))==string(source_sha),'geometry prepared_fem input hash mismatch.');M=load(masks_file);
assert(isfield(G,'mesh_sha256')&&isfield(G,'bounds_ohm_m')&&isfield(G,'truth'));
assert(isequal(size(G.bounds_ohm_m),[2 2])&&isequal(size(G.truth),[65 2]));
names=cellstr(string(M.model_names(:)));expected={'plane';'ellipsoid';'ellipsoid_depth';'extended_lung'};
assert(isequal(names,expected),'Unexpected C01 mask names.');masks=logical(M.masks);ratios=double(M.ratios(:));truth=double(M.truth);
assert(isequal(size(masks,2),4)&&isequal(size(truth),[65 2])&&numel(ratios)>=20);
assert(all(isfinite(ratios))&&all(ratios>0)&&numel(unique(ratios))==numel(ratios));
assert(isequal(size(M.bounds),[2 2])&&norm(double(M.bounds)-double(G.bounds_ohm_m),'fro')<1e-12);
assert(string(M.mesh_sha256)==string(G.mesh_sha256));
S=load(prepared,'Kall','Klung','P','free','C','mm','sizes');
Kall=S.Kall;Kref=S.Klung{1};P=S.P;free=S.free(:);C=S.C;mm=S.mm;sizes=double(S.sizes(:));
n=size(Kall,1);ns=numel(sizes);assert(ns==9&&size(P,2)==18&&size(P,1)==n&&size(masks,1)==size(mm.elems,1));
sys_source=which('system_mat_fields');assert(~isempty(sys_source));
baseid=struct('geometry_sha256',trkg4_file_sha256(geometry_file),'masks_sha256',mask_sha,...
 'source_prepared_sha256',source_sha,'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'system_mat_fields_sha256',trkg4_file_sha256(sys_source),'mesh_sha256',char(string(G.mesh_sha256)),...
 'model_names',{names(:)'},'ratios',ratios(:).','sizes_mm',sizes(:).');runid=baseid;
if any(strcmp(stage,{'pem','cem_baseline'}))
 extfile=fullfile(out,'prepared_extensions.mat');
 if isfile(extfile)
  E=load(extfile,'Klung','identity','kall_relative_error');assert(isfield(E,'Klung')&&isfield(E,'identity'));
  local_assert_identity(E.identity,baseid);assert(E.kall_relative_error<1e-12);Klung=E.Klung;extid=E.identity;
 else
  si=trkg4_scale_fmdl_to_si(mm,cfg);si.electrode=struct([]);si.stimulation=struct([]);F=system_mat_fields(si);
  assert(size(F,2)==n&&size(F,1)==3*size(mm.elems,1));Kcheck=F'*F;err=norm(Kcheck-Kall,'fro')/norm(Kall,'fro');
  assert(isfinite(err)&&err<1e-12,'system_mat_fields Kall disagrees with immutable prepared_fem.');Klung=cell(4,1);
  for j=1:4,rr=reshape((3*(find(masks(:,j))-1)+(1:3))',[],1);B=F(rr,:);Klung{j}=B'*B;end
  clear F B rr Kcheck;extid=baseid;extid.kall_relative_error=err;kall_relative_error=err;identity=extid;model_names=names;
  save(extfile,'Klung','identity','kall_relative_error','model_names','ratios','truth','-v7.3');
 end
 assert(numel(Klung)==4);runid=extid;runid.prepared_extensions_sha256=trkg4_file_sha256(extfile);
 idfile=fullfile(out,'pem_identity.json');if isfile(idfile),local_assert_identity(jsondecode(fileread(idfile)),runid);else,local_json(idfile,runid);end
end
if strcmp(stage,'pem')
 target=fullfile(out,'pem_library.csv');rows=local_read_rows(target,...
  {'model_index','model','ratio','L_mm','f','df_dlogq','reciprocity_abs','relative_residual','elapsed_seconds'});
 for j=1:4
  for q=ratios'
   fprintf('C01 PEM model=%s q=%.12g\n',names{j},q);hit=local_match(rows,j,q,'model_index');if local_complete(rows,hit,sizes),continue;end
   if any(hit),rows=rows(~hit);end
   t0=tic;K=Kall+(1/q-1)*Klung{j};U=zeros(n,2*ns);U(free,:)=decomposition(K(free,free),'chol')\P(free,:);
   residual=norm(K(free,:)*U-P(free,:),'fro')/norm(P(free,:),'fro');assert(isfinite(residual)&&residual<1e-7,'PEM residual failed.');
   W=Klung{j}*U(:,1:ns);elapsed=toc(t0);
   for k=1:ns
    z=P(:,ns+k)'*U(:,k);zr=P(:,k)'*U(:,ns+k);rec=abs(z-zr);assert(isfinite(z)&&rec<1e-7,'PEM reciprocity failed.');
    row=struct('model_index',j,'model',string(names{j}),'ratio',q,'L_mm',sizes(k),'f',z,...
     'df_dlogq',U(:,ns+k)'*W(:,k)/q,'reciprocity_abs',rec,'relative_residual',residual,'elapsed_seconds',elapsed);
    if isempty(rows),rows=row;else,rows(end+1)=row;end
   end
   writetable(struct2table(rows),target);
  end
 end
 htarget=fullfile(out,'reference_holdouts.csv');hrows=local_read_rows(htarget,...
  {'ratio','L_mm','f','df_dlogq','residual','reciprocity','elapsed_seconds'});
 for q=unique([truth(:,2)./truth(:,1);4])'
  fprintf('C01 reference holdout q=%.12g\n',q);hit=local_match(hrows,0,q,'ratio');if local_complete(hrows,hit,sizes),continue;end
  if any(hit),hrows=hrows(~hit);end;t0=tic;K=Kall+(1/q-1)*Kref;U=zeros(n,2*ns);U(free,:)=decomposition(K(free,free),'chol')\P(free,:);
  residual=norm(K(free,:)*U-P(free,:),'fro')/norm(P(free,:),'fro');assert(isfinite(residual)&&residual<1e-7,'Reference residual failed.');W=Kref*U(:,1:ns);elapsed=toc(t0);
  for k=1:ns
   z=P(:,ns+k)'*U(:,k);zr=P(:,k)'*U(:,ns+k);rec=abs(z-zr);assert(isfinite(z)&&rec<1e-7,'Reference reciprocity failed.');
   row=struct('ratio',q,'L_mm',sizes(k),'f',z,'df_dlogq',U(:,ns+k)'*W(:,k)/q,'residual',residual,'reciprocity',rec,'elapsed_seconds',elapsed);
   if isempty(hrows),hrows=row;else,hrows(end+1)=row;end
  end
  writetable(struct2table(hrows),htarget);
 end
 report=struct('status','exploratory_hypothesis_not_validated','stage',stage,'identity',runid,...
  'pem_complete',height(readtable(target))==4*numel(ratios)*ns,'reference_holdouts_complete',...
  height(readtable(htarget))==numel(unique([truth(:,2)./truth(:,1);4]))*ns,...
  'pem_library_sha256',trkg4_file_sha256(target),'reference_holdouts_sha256',trkg4_file_sha256(htarget));
 local_json(fullfile(out,'execution_pem.json'),report);return;
end
if strcmp(stage,'cem_baseline')
 target=fullfile(out,'cem_baseline.csv');rows=local_read_rows(target,...
  {'model_index','model','L_mm','rho1','rho2','Z','dZ_drho1','dZ_drho2','relative_residual','reciprocity_abs','elapsed_seconds'});
 for j=1:4
  for k=1:ns
   if ~isempty(rows)&&any([rows.model_index]==j & abs([rows.L_mm]-sizes(k))<1e-9),continue;end
   t0=tic;[z,J,res,rec]=local_cem(Kall,Klung{j},C{k},free,[4 16]);
   row=struct('model_index',j,'model',string(names{j}),'L_mm',sizes(k),'rho1',4,'rho2',16,'Z',z,...
    'dZ_drho1',J(1),'dZ_drho2',J(2),'relative_residual',res,'reciprocity_abs',rec,'elapsed_seconds',toc(t0));
   if isempty(rows),rows=row;else,rows(end+1)=row;end;writetable(struct2table(rows),target);
  end
 end
 report=struct('status','exploratory_hypothesis_not_validated','stage',stage,'identity',runid,'complete',height(readtable(target))==4*ns);
 local_json(fullfile(out,'execution_cem_baseline.json'),report);return;
end
assert(exist('fmincon','file')==2,'Optimization Toolbox is required.');
T=readtable(fullfile(out,'experimental_observations.csv'),'TextType','string');
assert(isequal(T.Properties.VariableNames,{'L_mm','Z_inhale_hold_ohm'})&&height(T)==ns);
assert(all(abs(double(T.L_mm)-sizes)<1e-9));y=double(T.Z_inhale_hold_ohm);assert(all(isfinite(y)));
Ffit=readtable(fullfile(out,'experimental_fits.csv'),'TextType','string');assert(any(strcmp(Ffit.Properties.VariableNames,'model')));
r=Ffit(Ffit.model=="reference",:);assert(height(r)==1&&all(ismember({'rho1','rho2'},Ffit.Properties.VariableNames)));
lb=log(double(G.bounds_ohm_m(:,1)));ub=log(double(G.bounds_ohm_m(:,2)));assert(all(isfinite([lb;ub]))&&all(lb<ub));
starts=[log([double(r.rho1(1));double(r.rho2(1))]),(lb+ub)/2];starts=min(max(starts,lb),ub);
allfree=[free;n+(1:4)'];rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];Ksoft=Kall-Kref;
cache=containers.Map('KeyType','char','ValueType','any');trace=struct([]);solutions=struct([]);
opts=optimoptions('fmincon','Algorithm','sqp','SpecifyObjectiveGradient',true,'Display','iter',...
 'MaxIterations',35,'OptimalityTolerance',1e-6,'StepTolerance',1e-9);
for si=1:2
 [x,loss,flag,info]=fmincon(@objective,starts(:,si),[],[],[],[],lb,ub,[],opts);assert(flag>0,'CEM start did not converge.');
 [~,~,z,J]=objective(x);rho=exp(x);sv=svd(J);result=struct('start_index',si,'rho1',rho(1),'rho2',rho(2),...
  'rmse_ohm',sqrt(loss),'exitflag',flag,'iterations',info.iterations,'function_count',info.funcCount,...
  'condition_log_parameters',sv(1)/max(sv(end),eps));
 if isempty(solutions),solutions=result;else,solutions(end+1)=result;end;writetable(struct2table(solutions),fullfile(out,'cem_multistart.csv'));
end
[~,bi]=min([solutions.rmse_ohm]);best=solutions(bi);[~,~,z,J]=objective(log([best.rho1;best.rho2]));
writetable(table(sizes,y,z,z-y,J(:,1),J(:,2),'VariableNames',...
 {'L_mm','observed_ohm','predicted_ohm','residual_ohm','dZ_dlogrho1','dZ_dlogrho2'}),fullfile(out,'cem_experimental_predictions.csv'));
report=struct('status','exploratory_hypothesis_not_validated','stage',stage,'best',best,...
 'geometry_sha256',baseid.geometry_sha256,'prepared_sha256',baseid.source_prepared_sha256,...
 'source_sha256',baseid.worker_sha256,'observations_sha256',trkg4_file_sha256(fullfile(out,'experimental_observations.csv')),...
 'experimental_fits_sha256',trkg4_file_sha256(fullfile(out,'experimental_fits.csv')),...
 'pose_fixed_from_full_data',true,'independent_end_to_end_validation',false,'global_optimality_proved',false,...
 'unique_evaluations',cache.Count,'starts',2);
local_json(fullfile(out,'cem_fit_summary.json'),report);
 function [loss,gradient,z,J]=objective(x)
  key=sprintf('%.17g_%.17g',x(1),x(2));if isKey(cache,key),v=cache(key);loss=v.loss;gradient=v.gradient;z=v.z;J=v.J;return;end
  t0=tic;rho=exp(x);K=Ksoft/rho(1)+Kref/rho(2);z=zeros(ns,1);J=zeros(ns,2);maxres=0;maxrec=0;
  for k=1:ns
   H=C{k};H(1:n,1:n)=H(1:n,1:n)+K;A=H(allfree,allfree);assert(norm(A-A','fro')/norm(A,'fro')<1e-12);
   U=zeros(n+4,2);U(allfree,:)=decomposition((A+A')/2,'chol')\rhs(allfree,:);
   res=norm(H(allfree,:)*U-rhs(allfree,:),'fro')/norm(rhs(allfree,:),'fro');z(k)=rhs(:,2)'*U(:,1);rec=abs(z(k)-rhs(:,1)'*U(:,2));
   assert(isfinite(z(k))&&res<1e-7&&rec<1e-7,'CEM solve failed.');maxres=max(maxres,res);maxrec=max(maxrec,rec);
   J(k,1)=U(1:n,2)'*(Ksoft*U(1:n,1))/rho(1);J(k,2)=U(1:n,2)'*(Kref*U(1:n,1))/rho(2);
  end
  residual=z-y;loss=mean(residual.^2);gradient=2*(J'*residual)/ns;cache(key)=struct('loss',loss,'gradient',gradient,'z',z,'J',J);
  row=struct('evaluation',cache.Count,'rho1',rho(1),'rho2',rho(2),'rmse_ohm',sqrt(loss),...
   'relative_residual',maxres,'reciprocity_abs_ohm',maxrec,'elapsed_seconds',toc(t0));
  if isempty(trace),trace=row;else,trace(end+1)=row;end;writetable(struct2table(trace),fullfile(out,'cem_evaluation_trace.csv'));
  fprintf('C01 CEM evaluation=%d rho=[%.8g %.8g] RMSE=%.8g\n',cache.Count,rho(1),rho(2),sqrt(loss));
 end
end

function [z,J,res,rec]=local_cem(Kall,Kl,C,free,rho)
n=size(Kall,1);K=Kall/rho(1)+(1/rho(2)-1/rho(1))*Kl;H=C;H(1:n,1:n)=H(1:n,1:n)+K;
allfree=[free;n+(1:4)'];rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];A=H(allfree,allfree);
assert(norm(A-A','fro')/norm(A,'fro')<1e-12);U=zeros(n+4,2);U(allfree,:)=decomposition((A+A')/2,'chol')\rhs(allfree,:);
z=rhs(:,2)'*U(:,1);rec=abs(z-rhs(:,1)'*U(:,2));res=norm(H(allfree,:)*U-rhs(allfree,:),'fro')/norm(rhs(allfree,:),'fro');
assert(isfinite(z)&&res<1e-7&&rec<1e-7,'CEM baseline solve failed.');J=[U(1:n,2)'*((Kall-Kl)*U(1:n,1))/rho(1)^2,U(1:n,2)'*(Kl*U(1:n,1))/rho(2)^2];
end
function rows=local_read_rows(path,expected)
rows=struct([]);if ~isfile(path),return;end;T=readtable(path,'TextType','string');assert(isequal(T.Properties.VariableNames,expected),...
 'Existing CSV has an incompatible schema.');rows=table2struct(T);
end
function hit=local_match(rows,j,q,field)
hit=false(size(rows));if isempty(rows),return;end;if j==0,hit=abs([rows.(field)]-q)<1e-12;else,hit=[rows.(field)]==j & abs([rows.ratio]-q)<1e-12;end
end
function tf=local_complete(rows,hit,sizes)
tf=nnz(hit)==numel(sizes);if ~tf,return;end;tf=all(abs(sort([rows(hit).L_mm])-sort(sizes(:)'))<1e-9);
end
function local_assert_identity(a,b)
for f={'geometry_sha256','masks_sha256','source_prepared_sha256','worker_sha256','system_mat_fields_sha256','mesh_sha256','prepared_extensions_sha256'}
 if isfield(b,f{1}),assert(isfield(a,f{1})&&string(a.(f{1}))==string(b.(f{1})),['Identity mismatch: ' f{1}]);end
end
assert(isequal(string(a.model_names(:)),string(b.model_names(:)))&&all(abs(double(a.ratios(:))-double(b.ratios(:)))<1e-12)&&...
 all(abs(double(a.sizes_mm(:))-double(b.sizes_mm(:)))<1e-12),'Identity vectors mismatch.');
end
function h=local_prepared_hash(G)
if isfield(G,'source_prepared_sha256'),h=char(string(G.source_prepared_sha256));return;end
if isfield(G,'prepared_sha256'),h=char(string(G.prepared_sha256));return;end
assert(isfield(G,'input_sha256'),'geometry.json has no prepared_fem input hash.');f=fieldnames(G.input_sha256);
hit=contains(lower(string(f)),'prepared_fem.mat');assert(nnz(hit)==1,'prepared_fem hash metadata is ambiguous.');h=char(string(G.input_sha256.(f{hit})));
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
