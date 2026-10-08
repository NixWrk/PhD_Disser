function report=run_scorpion_georg_fem_integration_check(varargin)
% Independent finer material integration; preserve the original comparison.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/scorpion_georg_fem_20261008');
addParameter(p,'ForwardOnly',false);parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');out=char(a.OutputDir);
if ~isfile(fullfile(out,'plan.local.json')),out=fullfile(cfg.project_root,out);end
plan=jsondecode(fileread(fullfile(out,'plan.local.json')));
for i=1:numel(plan.sources),assert(strcmp(trkg4_file_sha256(plan.sources(i).path),plan.sources(i).sha256));end
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
hashplan=trkg4_file_sha256(fullfile(out,'plan.local.json'));
hashworker=trkg4_file_sha256(fullfile(cfg.src_dir,'run_scorpion_georg_fem.m'));
S=load(fullfile(out,['prepared_fem_' hashplan(1:8) '_' hashworker(1:8) '.mat']));
proof=jsondecode(fileread(fullfile(out,'fraction_refinement.local.json')));
assert(strcmp(proof.plan_sha256,hashplan)&&proof.original_fractions_reproduced_exactly);
assert(strcmp(proof.refined_fractions_sha256,trkg4_file_sha256(fullfile(out,'fractions_q4096.mat'))));
identity=struct('plan_sha256',hashplan,'original_worker_sha256',hashworker,...
 'validator_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'fraction_refinement_sha256',trkg4_file_sha256(fullfile(out,'fraction_refinement.local.json')));
cache=fullfile(out,['integration_4096_' identity.fraction_refinement_sha256(1:8) '_' identity.validator_sha256(1:8) '.mat']);
if isfile(cache)
 V=load(cache);assert(isequal(V.identity,identity));parts=V.parts;
else
 F=system_mat_fields(trkg4_scale_fmdl_to_si(S.mm,cfg));Q=load(fullfile(out,'fractions_q4096.mat'));parts=cell(5,1);
 for j=1:5
  w=repelem(Q.fractions(:,j),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;parts{j}=B'*B;
 end
 save(cache,'parts','identity','-v7.3');clear F Q B w;
end
rows=struct([]);
for pose=1:3
 path=fullfile(out,sprintf('forward_pose%d.local.json',pose));f=jsondecode(fileread(path));
 assert(strcmp(f.identity.plan_sha256,hashplan)&&strcmp(f.identity.worker_sha256,hashworker));
 z=solve_vector(S,parts,pose,f.rho);difference=abs(z-f.Z_ohm(:));passed=all(difference<=.02+.001*abs(f.Z_ohm(:)));
 if ~isfield(f,'original_quadrature_check')
  f.original_quadrature_check=struct('rules',[256,1024],'max_error_ohm',f.quadrature_max_error_ohm,...
   'passed',f.quadrature_within_scenario_tolerance);
 end
 f.quadrature_within_scenario_tolerance=passed;f.quadrature_max_error_ohm=max(difference);
 f.quadrature_rule_pair=[1024,4096];f.quadrature_validation_identity=identity;
 local_json(path,f);
end
if ~a.ForwardOnly
 for pose=1:3
  for channel=1:2
   for state={'inhale','exhale'}
    path=fullfile(out,sprintf('inverse_pose%d_ch%d_%s.local.json',pose,channel,state{1}));c=jsondecode(fileread(path));
    assert(strcmp(c.identity.worker_sha256,hashworker)&&strcmp(c.identity.plan_sha256,hashplan));
    z=solve_vector(S,parts,pose,c.rho_ohm_m);delta=abs(z-c.Z_predicted_ohm(:));
    r=struct('pose',pose,'channel',channel,'state',state{1},'case_sha256',trkg4_file_sha256(path),...
     'rules',[1024,4096],'Z_refined_ohm',z,'max_error_ohm',max(delta),...
     'passed',all(delta<=.02+.001*abs(c.Z_predicted_ohm(:))));
    if isempty(rows),rows=r;else,rows(end+1)=r;end
   end
  end
 end
 local_json(fullfile(out,'integration_validation.local.json'),struct('identity',identity,'cases',rows,...
  'tolerance_absolute_ohm',.02,'tolerance_relative',.001,'spatial_mesh_convergence_verified',false));
 % A predeclared diagnostic family, not a fitted role or a global profile.
 c=jsondecode(fileread(fullfile(out,'inverse_pose2_ch1_inhale.local.json')));
 active=[1,2,5];x=log(c.rho_ohm_m(active));x=x(:);[~,~,V]=svd(c.J_logrho_ohm,0);direction=V(:,end);
 lo=-inf;hi=inf;
 for j=1:3
  if abs(direction(j))<1e-12,continue;end
  limits=sort(([log(.1);log(100)]-x(j))/direction(j));lo=max(lo,limits(1));hi=min(hi,limits(2));
 end
 tau=unique([linspace(lo,hi,15),0]);family=zeros(numel(tau),7);
 for j=1:numel(tau)
  rho=c.rho_ohm_m;rho(active)=exp(x+tau(j)*direction);
  z=solve_vector(S,S.Kparts,2,rho);rss=norm(z-c.Z_observed_ohm(:))^2;
  refined=solve_vector(S,parts,2,rho);delta=abs(refined-z);
  family(j,:)=[reshape(rho(active),1,[]),rss,100*(rss/c.sse_ohm2-1),max(delta),all(delta<=.02+.001*abs(z))];
 end
 local_json(fullfile(out,'weak_direction_family.local.json'),struct('identity',identity,...
  'case_sha256',trkg4_file_sha256(fullfile(out,'inverse_pose2_ch1_inhale.local.json')),...
  'case',struct('pose',2,'channel',1,'state','inhale'),'direction_logrho',direction,...
  'tau',tau,'columns',{{'rho1','rho2','rho_fat','rss_ohm2','rss_increase_percent','integration_max_error_ohm','integration_pass'}},...
  'family',family,'others_reoptimized',false,'confidence_region',false,'global_profile',false));
end
report=rows;
end

function z=solve_vector(S,parts,pose,rho)
n=size(parts{1},1);K=sparse(n,n);for j=1:5,K=K+parts{j}/rho(j);end
free=[S.free;n+(1:8)'];rhs=sparse(n+8,2);rhs(n+[5,8],1)=[1;-1];rhs(n+[6,7],2)=[1;-1];
z=zeros(numel(S.sizes),1);
for k=1:numel(S.sizes)
 H=S.C{pose,k};H(1:n,1:n)=H(1:n,1:n)+K;B=H(free,free);U=zeros(n+8,2);
 U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
 assert(norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro')<1e-7);
 z(k)=rhs(:,2)'*U(:,1);assert(abs(z(k)-rhs(:,1)'*U(:,2))<1e-7);
end
fprintf('4096-point validation pose%d Z=%s\n',pose,mat2str(z',6));
end

function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
