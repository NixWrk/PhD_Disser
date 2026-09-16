function report=run_lateral_resistivity_range(varargin)
% Direct CEM solves on frozen matrices; two independent partition files.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_resistivity_range_20260916');
addParameter(p,'RequestsFile','points.csv');addParameter(p,'WorkerIndex',1);
addParameter(p,'NumWorkers',2);addParameter(p,'Threads',4);parse(p,varargin{:});
root=fileparts(fileparts(mfilename('fullpath')));out=char(p.Results.OutputDir);
if ~isfile(fullfile(out,'plan.json')),out=fullfile(root,out);end
planfile=fullfile(out,'plan.json');A=jsondecode(fileread(planfile));
assert(A.finite_electrodes&&A.diameter_mm==5&&A.electrode_count_per_measurement==4);
assert(~A.simultaneous_all_sizes&&strcmp(A.model,'variable_transverse'));
for f=string(fieldnames(A.sources))'
 z=A.sources.(f);assert(strcmp(trkg4_file_sha256(fullfile(root,z.path)),z.sha256),'Source identity mismatch.');
end
S=load(fullfile(root,A.sources.prepared_fem.path));
assert(strcmp(S.identity.plan_sha256,A.sources.base_plan.sha256));
assert(strcmp(S.identity.worker_sha256,trkg4_file_sha256(fullfile(root,'src','run_lateral_array_design.m'))));
assert(S.identity.diameter_mm==5&&isequal(S.sizes(:),A.sizes_mm(:)));
mi=find(S.names=="variable_transverse");assert(isscalar(mi));
n=size(S.Kall,1);Kl=S.Klung{mi};Ks=S.Kall-Kl;
maxNumCompThreads(p.Results.Threads);
qpath=fullfile(out,char(p.Results.RequestsFile));assert(isfile(qpath));
if strcmp(p.Results.RequestsFile,'points.csv'),assert(strcmp(trkg4_file_sha256(qpath),A.points_sha256));end
wi=p.Results.WorkerIndex;nw=p.Results.NumWorkers;assert(wi>=1&&wi<=nw&&mod(wi,1)==0&&mod(nw,1)==0);
Q=readtable(qpath,'TextType','string');Q=Q(mod((0:height(Q)-1)',nw)==wi-1,:);
[~,stem]=fileparts(qpath);target=fullfile(out,sprintf('%s_w%02dof%02d.csv',stem,wi,nw));
identity=struct('plan_sha256',trkg4_file_sha256(planfile),...
 'prepared_sha256',A.sources.prepared_fem.sha256,'requests_sha256',trkg4_file_sha256(qpath),...
 'worker_source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'worker_index',wi,'num_workers',nw,'finite_diameter_mm',5);
rows=struct([]);idfile=[target '.identity.json'];
if isfile(target)
 assert(isfile(idfile)&&isequal(jsondecode(fileread(idfile)),jsondecode(jsonencode(identity))),'Checkpoint identity mismatch.');
 rows=table2struct(readtable(target,'TextType','string'));
else,local_json(idfile,identity);end
old=readtable(fullfile(root,A.sources.base_responses.path),'TextType','string');expected=0;
for qi=1:height(Q)
 rho=[Q.rho1(qi),Q.rho2(qi)];assert(all(isfinite(rho))&&all(rho>0));
 selected=1:numel(S.sizes);
 if ismember('L_mm',Q.Properties.VariableNames)&&isfinite(Q.L_mm(qi))
  selected=find(S.sizes==Q.L_mm(qi));assert(isscalar(selected));
 end
 expected=expected+numel(selected);
 if ~isempty(rows)
  hit=string({rows.point_id})==Q.point_id(qi);
  if any(hit)
   assert(nnz(hit)==numel(selected)&&isequal(sort([rows(hit).L_mm])',sort(S.sizes(selected))));
   assert(max(abs([rows(hit).rho1]-rho(1)))<1e-10&&max(abs([rows(hit).rho2]-rho(2)))<1e-10);continue;
  end
 end
 t0=tic;batch=struct([]);K=Ks/rho(1)+Kl/rho(2);
 for k=selected(:)'
  H=S.C{k};H(1:n,1:n)=H(1:n,1:n)+K;free=[S.free(:);n+(1:4)'];
  rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];
  B=H(free,free);assert(norm(B-B','fro')/norm(B,'fro')<1e-12);U=zeros(n+4,2);
  U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
  res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
  z=rhs(:,2)'*U(:,1);rec=abs(z-rhs(:,1)'*U(:,2));
  assert(res<A.acceptance.system_relative_residual_max&&rec<A.acceptance.reciprocity_abs_ohm_max&&isfinite(z));
  J=[U(1:n,2)'*(Ks*U(1:n,1))/rho(1),U(1:n,2)'*(Kl*U(1:n,1))/rho(2)];
  if Q.kind(qi)=="anchor"
   h=old.model=="variable_transverse"&abs(old.rho1-rho(1))<1e-10&abs(old.rho2-rho(2))<1e-10&old.L_mm==S.sizes(k);
   assert(nnz(h)==1&&abs(z-old.Z(h))<A.acceptance.anchor_Z_abs_error_ohm_max,'Old-model anchor failed.');
  end
  row=struct('point_id',Q.point_id(qi),'kind',Q.kind(qi),'model',"variable_transverse",...
   'rho1',rho(1),'rho2',rho(2),'L_mm',S.sizes(k),'Z',z,'dZ_dlogrho1',J(1),'dZ_dlogrho2',J(2),...
   'relative_residual',res,'reciprocity_abs',rec,'elapsed_seconds',toc(t0));
  if isempty(batch),batch=row;else,batch(end+1)=row;end
 end
 if isempty(rows),rows=batch(:);else,rows=[rows(:);batch(:)];end
 temporary=[target '.tmp'];writetable(struct2table(rows),temporary,'FileType','text','Delimiter',',');
 movefile(temporary,target,'f');
 fprintf('RANGE w%d %d/%d %s rho=[%.6g %.6g] %.2fs\n',wi,qi,height(Q),Q.point_id(qi),rho(1),rho(2),toc(t0));
end
assert(numel(rows)==expected);
report=struct('complete',true,'identity',identity,'responses_sha256',trkg4_file_sha256(target),...
 'rows',numel(rows),'finite_diameter_mm',5,'new_direct_CEM_solves',true,...
 'status','exploratory_hypothesis_not_validated');
local_json([target '.execution.json'],report);
end

function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
