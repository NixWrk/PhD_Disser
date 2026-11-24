function trkg4_bmstu_inverse_worker(planfile)
% Persistent FEM worker. Reduced matrices are Galerkin projections of the
% same finite-contact system, enriched only with full-system solutions.
p=jsondecode(fileread(planfile));addpath(fullfile(p.root,'MATLAB_TRKG4_real_subjects','src'),...
 fullfile(p.root,'MATLAB_TRKG4_real_subjects','tools'));
maxNumCompThreads(4);work=p.work;ports=[1 0;0 1;0 -1;-1 0];
g=jsondecode(fileread(p.geometry_path));ref=jsondecode(fileread(p.reference_path));
assert(strcmp(trkg4_file_sha256(p.geometry_path),p.geometry_sha256));
assert(strcmp(trkg4_file_sha256(p.reference_path),p.reference_sha256));
assert(strcmp(trkg4_file_sha256(g.mesh_path),p.mesh_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),p.canonical_kernel_sha256));
data=load(p.shared_file,'parts','identity');parts=data.parts(2,:);coarse=data.parts(1,:);
assert(strcmp(data.identity.mesh_sha256,p.mesh_sha256));clear data;
n=size(parts{1},1);ns=numel(p.sizes_mm);
cache=fullfile(work,'contacts.mat');
if isfile(cache)
 s=load(cache,'C','plan_sha256');assert(strcmp(s.plan_sha256,trkg4_file_sha256(planfile)));C=s.C;clear s;
else
 cfg=trkg4_config(p.subject_alias);M=load(g.mesh_path);
 mm=struct('name','BMSTU inverse study','nodes',M.nodes_mm,'elems',double(M.elems),'boundary',double(M.boundary));clear M;
 ctx=trkg4_local_patch_context(mm);C=cell(ns,1);
 for c=1:ns
  L=p.sizes_mm(c);cc=g.contacts([g.contacts.size_mm]==L);
  [el,faces]=trkg4_local_exact_patches(ctx,cc.centres_lps_mm,g.nominal_area_mm2,g.z_contact_ohm_m2);
  cert=jsondecode(fileread(fullfile(fileparts(p.reference_path),sprintf('contacts_L%d.local.json',L))));
  for e=1:4
   el(e).faces=mm.boundary(faces{e},:);area=ctx.canonical.face_area(faces{e});
   centroid=sum(ctx.face_centroid(faces{e},:).*area,1)/sum(area);
   assert(numel(area)==cert.contacts(e).boundary_faces);
   assert(abs(sum(area)-cert.contacts(e).actual_area_mm2)<1e-10);
   assert(norm(centroid-cert.contacts(e).actual_centroid_lps_mm(:)')<1e-10);
  end
  mm.electrode=el;C{c}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(mm,cfg));
 end
 plan_sha256=trkg4_file_sha256(planfile);save(cache,'C','plan_sha256','-v7.3');clear ctx mm;
end
basis=cell(ns,1);reduced=cell(ns,1);statefile=fullfile(work,'basis.mat');count=0;
if isfile(statefile)
 s=load(statefile,'basis','reduced','count','plan_sha256');
 assert(strcmp(s.plan_sha256,trkg4_file_sha256(planfile)));
 basis=s.basis;reduced=s.reduced;count=s.count;clear s;
end
writejson(fullfile(work,'ready.local.json'),struct('ready',true,'nodes',n,'sizes',p.sizes_mm,'snapshot_count',count));
fprintf('READY %s, %d sizes, %d nodes\n',p.subject_alias,ns,n);
request=1;
while true
 input=fullfile(work,sprintf('request_%04d.local.json',request));
 output=fullfile(work,sprintf('answer_%04d.local.json',request));
 if isfile(output),request=request+1;continue;end
 if ~isfile(input),pause(.2);continue;end
 q=jsondecode(fileread(input));
 if strcmp(q.operation,'stop'),writejson(output,struct('stopped',true));break;end
 timer=tic;rho=q.rho_ohm_m(:)';assert(numel(rho)==5&&all(rho>0)&&all(isfinite(rho)));
 usecoarse=isfield(q,'coarse')&&q.coarse;
 pp=parts;if usecoarse,pp=coarse;end
 K=pp{1}/rho(1);for k=2:5,K=K+pp{k}/rho(k);end
 rows=struct([]);indices=1:ns;if isfield(q,'indices'),indices=q.indices(:)';end
 for c=indices
  started=tic;[z,j,U,qc]=full_state(K,C{c},pp,rho,ports);
  if isfield(q,'canonical_check')&&q.canonical_check
   [T,JJ,qq]=trkg4_lateral_cem_response(K,C{c},pp,rho,ports,1:5,1);
   qc.canonical_Z_error_ohm=z-T(2,1);qc.canonical_J_error_ohm=max(abs(j-reshape(JJ(2,1,:),1,[])));
   assert(abs(qc.canonical_Z_error_ohm)<1e-8&&qc.canonical_J_error_ohm<1e-8);
   assert(qq.system_relative_residual<1e-7);
  end
  if q.enrich && ~usecoarse
   [basis{c},reduced{c}]=enrich(basis{c},reduced{c},U,parts,C{c});
  end
  rows=[rows;struct('size_mm',p.sizes_mm(c),'Z_ohm',z,'J_logrho_ohm',j,'qc',qc,'seconds',toc(started))]; %#ok<AGROW>
  fprintf('Q%d L%d Z %.7g sec %.1f\n',request,p.sizes_mm(c),z,toc(started));
 end
 clear K U;
 if q.enrich && ~usecoarse
  count=count+1;plan_sha256=trkg4_file_sha256(planfile);
  save(statefile,'basis','reduced','count','plan_sha256','-v7.3');
  writejson(fullfile(work,'reduced.local.json'),struct('cases',{reduced},'snapshot_count',count,'sizes_mm',p.sizes_mm));
 end
 writejson(output,struct('rho_ohm_m',rho,'coarse',usecoarse,'rows',rows,'seconds',toc(timer),...
  'request_sha256',trkg4_file_sha256(input),'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
  'canonical_kernel_sha256',p.canonical_kernel_sha256,'status','exploratory_hypothesis_not_validated'));
 fprintf('ANSWER %d %.1f s\n',request,toc(timer));request=request+1;
end
end

function [z,j,U,qc]=full_state(K,C,parts,rho,ports)
% Identical full CEM equations and adjoint derivative to the canonical core;
% retain nodal fields to build a deterministic Galerkin subspace.
n=size(K,1);H=C+blkdiag(K,sparse(4,4));free=(2:n+4)';
rhs=sparse(n+4,2);rhs(n+(1:4),:)=ports;B=H(free,free);U=zeros(n+4,2);
U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
T=ports'*U(n+(1:4),:);rec=max(abs(T-T'),[],'all');
assert(res<1e-7&&rec<1e-7&&all(isfinite(T),'all'));
z=T(2,1);j=zeros(1,5);for k=1:5,j(k)=U(1:n,2)'*(parts{k}*U(1:n,1))/rho(k);end
qc=struct('system_relative_residual',res,'reciprocity_ohm',rec);
end

function [V,r]=enrich(V,r,U,parts,C)
n=size(parts{1},1);old=size(V,2);Q=[];
for k=1:2
 u=U(1:n,k);original=norm(u);
 for pass=1:2
  if ~isempty(V),u=u-V*(V'*u);end
  if ~isempty(Q),u=u-Q*(Q'*u);end
 end
 if norm(u)>1e-11*original,Q=[Q,u/norm(u)];end %#ok<AGROW>
end
if isempty(Q),return;end
Vnew=[V,Q];nv=size(Vnew,2);assert(nv<=100,'Reduced subspace limit reached');
A=cell(1,5);
for k=1:5
 PQ=parts{k}*Q;block=Q'*PQ;
 if old>0,cross=V'*PQ;A{k}=[r.parts{k},cross;cross',block];else,A{k}=block;end
 A{k}=(A{k}+A{k}')/2;
end
% Contact entries use only the surface nodes, avoiding a large dense product.
nodes=find(any(C(1:n,:),2));CB=C(nodes,nodes);B=C(nodes,n+(1:4));D=full(C(n+(1:4),n+(1:4)));
W=Vnew(nodes,:);Cr=[W'*CB*W,W'*B;B'*W,D];
r=struct('parts',{A},'contact',(Cr+Cr')/2,'rank',nv);V=Vnew;
end

function writejson(path,d)
temp=[path '.writing'];f=fopen(temp,'w','n','UTF-8');assert(f>=0);
fprintf(f,'%s\n',jsonencode(d));fclose(f);movefile(temp,path,'f');
end
