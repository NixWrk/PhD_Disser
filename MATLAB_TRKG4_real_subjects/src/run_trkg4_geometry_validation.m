function report=run_trkg4_geometry_validation(varargin)
% Independently solved targets for the approved literature-bounded tests.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/geometry_validation_20260910');
parse(p,varargin{:});opt=p.Results;
cfg=trkg4_config('nik');root=cfg.project_root;out=char(opt.OutputDir);
if ~isfolder(out),out=fullfile(root,out);end
planfile=fullfile(out,'plan.json');assert(isfile(planfile),'Prepare the Python test plan first');
plan=jsondecode(fileread(planfile));
source=fullfile(root,'output','exploratory','geometry_reduction_20260910');
prepared=fullfile(source,'prepared_fem.mat');
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
identity=struct('plan_sha256',trkg4_file_sha256(planfile), ...
 'prepared_sha256',trkg4_file_sha256(prepared), ...
 'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
idfile=fullfile(out,'fem_identity.json');
if isfile(idfile)
 assert(isequaln(jsondecode(fileread(idfile)),identity),'Changed inputs or code: use a fresh output folder');
else,local_json(idfile,identity);end
load(prepared,'Kall','Klung','P','free','sizes');
n=size(Kall,1);ns=numel(sizes);assert(isequal(sizes(:),plan.sizes_mm(:)));
target=fullfile(out,'reference_holdouts.csv');rows=struct([]);
if isfile(target),rows=table2struct(readtable(target));end
for ci=1:numel(plan.synthetic_cases)
 c=plan.synthetic_cases(ci);
 if ~isempty(rows)&&nnz([rows.case_index]==c.case_index)==ns,continue;end
 started=tic;K=Kall+(1/c.ratio-1)*Klung{1};
 U=zeros(n,2*ns);U(free,:)=decomposition(K(free,free),'chol')\P(free,:);
 residual=norm(K(free,:)*U-P(free,:),'fro')/norm(P(free,:),'fro');
 assert(isfinite(residual)&&residual<1e-7,'Independent target linear solve failed');
 for k=1:ns
  z=c.rho1*(P(:,ns+k)'*U(:,k));zr=c.rho1*(P(:,k)'*U(:,ns+k));
  assert(isfinite(z)&&abs(z-zr)<1e-7,'Independent target reciprocity failed');
  row=struct('case_index',c.case_index,'rho1',c.rho1,'rho2',c.rho2, ...
   'ratio',c.ratio,'L_mm',sizes(k),'Z_ohm',z,'reciprocity_abs_ohm',abs(z-zr), ...
   'relative_residual',residual,'elapsed_all_sizes_seconds',toc(started));
  if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
 end
 writetable(struct2table(rows),target);
 fprintf('Independent CT target %d/%d, rho1=%.6g rho2=%.6g: %.2fs\n',ci,numel(plan.synthetic_cases),c.rho1,c.rho2,toc(started));
end
report=identity;report.status='exploratory_hypothesis_not_validated';
report.complete=height(readtable(target))==ns*numel(plan.synthetic_cases);
report.ct_anatomy_accepted=true;report.in_vivo_rhos_known=false;
local_json(fullfile(out,'fem_execution.json'),report);
end

function local_json(p,value)
f=fopen(p,'w');assert(f>0);c=onCleanup(@()fclose(f));fprintf(f,'%s',jsonencode(value,'PrettyPrint',true));
end
