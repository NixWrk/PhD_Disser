function trkg4_scorpion_tissue_worker(planfile)
% Electrical-only study using frozen material matrices and recorded contacts.
% Floating non-lateral terminals are exactly eliminated by a Schur complement.
p=jsondecode(fileread(planfile));addpath(fullfile(p.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(2);work=p.work;ports=[1 0;0 1;0 -1;-1 0];
assert(strcmp(trkg4_file_sha256(p.parts_file),p.parts_sha256));
assert(strcmp(trkg4_file_sha256(p.cem_file),p.cem_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),p.canonical_kernel_sha256));
d=load(p.parts_file,'parts');parts=d.parts;clear d;n=size(parts{1},1);ns=numel(p.sizes_mm);
d=load(p.cem_file,'result');sc=d.result.scenarios;clear d;if ~iscell(sc),sc=num2cell(sc);end
C=cell(ns,1);Cfull=C;Pfull=C;
for c=1:ns
 ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),p.case_ids{c}),sc),1);assert(~isempty(ix));
 cc=sc{ix}.grouped_contact_matrix_SI;groups=sc{ix}.ports;names={groups.id};
 [found,active]=ismember({'A','M','N','B'},names);assert(all(found));
 omit=setdiff(1:numel(names),active);keep=[1:n,n+active];other=n+omit;
 C{c}=cc(keep,keep)-cc(keep,other)*(cc(other,other)\cc(other,keep));
 C{c}=(C{c}+C{c}')/2;Cfull{c}=cc;
 if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
 else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
 P=zeros(numel(names),4);for k=1:4,[ok,ix]=ismember(pairs{k},names);assert(all(ok));P(ix,k)=[1;-1];end
 Pfull{c}=P;
end
clear sc cc;
basis=cell(ns,1);reduced=cell(ns,1);count=0;statefile=fullfile(work,'basis.mat');
if isfile(statefile)
 s=load(statefile);assert(strcmp(s.plan_sha256,trkg4_file_sha256(planfile)));
 basis=s.basis;reduced=s.reduced;count=s.count;clear s;
end
writejson(fullfile(work,'ready.local.json'),struct('ready',true,'nodes',n,'sizes',p.sizes_mm));
fprintf('READY Scorpion %s %d nodes\n',p.subject_alias,n);number=1;
while true
 input=fullfile(work,sprintf('request_%04d.local.json',number));output=fullfile(work,sprintf('answer_%04d.local.json',number));
 if isfile(output),number=number+1;continue;end
 if ~isfile(input),pause(.2);continue;end
 q=jsondecode(fileread(input));if strcmp(q.operation,'stop'),writejson(output,struct('stopped',true));break;end
 assert(~q.coarse,'No alternative quadrature block in this frozen Scorpion study');
 timer=tic;rho=q.rho_ohm_m(:)';assert(numel(rho)==5&&all(rho>0)&&all(isfinite(rho)));
 K=parts{1}/rho(1);for k=2:5,K=K+parts{k}/rho(k);end
 indices=1:ns;if isfield(q,'indices'),indices=q.indices(:)';end
 rows=struct([]);
 for c=indices
  started=tic;[z,j,U,qc]=full_state(K,C{c},parts,rho,ports);
  if q.canonical_check
   [T,J,qq]=trkg4_lateral_cem_response(K,Cfull{c},parts,rho,Pfull{c},1:5,1);
   qc.original_full_contact_Z_error_ohm=z-T(4,3);
   qc.original_full_contact_J_error_ohm=max(abs(j-reshape(J(4,3,:),1,[])));
   assert(abs(qc.original_full_contact_Z_error_ohm)<1e-7&&qc.original_full_contact_J_error_ohm<1e-7);
   assert(qq.system_relative_residual<1e-7);
  end
  if q.enrich,[basis{c},reduced{c}]=enrich(basis{c},reduced{c},U,parts,C{c});end
  rows=[rows;struct('size_mm',p.sizes_mm(c),'Z_ohm',z,'J_logrho_ohm',j,'qc',qc,'seconds',toc(started))]; %#ok<AGROW>
  fprintf('Q%d L%d Z %.8g sec %.1f\n',number,p.sizes_mm(c),z,toc(started));
 end
 clear K U;
 if q.enrich
  count=count+1;plan_sha256=trkg4_file_sha256(planfile);save(statefile,'basis','reduced','count','plan_sha256','-v7.3');
  writejson(fullfile(work,'reduced.local.json'),struct('cases',{reduced},'snapshot_count',count,'sizes_mm',p.sizes_mm));
 end
 writejson(output,struct('rho_ohm_m',rho,'coarse',false,'rows',rows,'seconds',toc(timer),...
  'request_sha256',trkg4_file_sha256(input),'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
  'canonical_kernel_sha256',p.canonical_kernel_sha256,'status','exploratory_hypothesis_not_validated'));
 fprintf('ANSWER %d %.1f s\n',number,toc(timer));number=number+1;
end
end

function [z,j,U,qc]=full_state(K,C,parts,rho,ports)
n=size(K,1);H=C+blkdiag(K,sparse(4,4));free=(2:n+4)';rhs=sparse(n+4,2);rhs(n+(1:4),:)=ports;
B=H(free,free);U=zeros(n+4,2);U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');T=ports'*U(n+(1:4),:);rec=max(abs(T-T'),[],'all');
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
Vnew=[V,Q];nv=size(Vnew,2);assert(nv<=100);A=cell(1,5);
for k=1:5
 PQ=parts{k}*Q;block=Q'*PQ;
 if old>0,cross=V'*PQ;A{k}=[r.parts{k},cross;cross',block];else,A{k}=block;end
 A{k}=(A{k}+A{k}')/2;
end
nodes=find(any(C(1:n,:),2));CB=C(nodes,nodes);B=C(nodes,n+(1:4));D=full(C(n+(1:4),n+(1:4)));
W=Vnew(nodes,:);Cr=[W'*CB*W,W'*B;B'*W,D];r=struct('parts',{A},'contact',(Cr+Cr')/2,'rank',nv);V=Vnew;
end

function writejson(path,d)
tmp=[path '.writing'];f=fopen(tmp,'w','n','UTF-8');assert(f>=0);fprintf(f,'%s\n',jsonencode(d));fclose(f);movefile(tmp,path,'f');
end
