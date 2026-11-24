function trkg4_requested_tissue_checks(jobfile)
% Full canonical CEM checks of new tissue-only inverse candidates.
j=jsondecode(fileread(jobfile));p=jsondecode(fileread(j.source_plan));
assert(strcmp(trkg4_file_sha256(j.source_plan),j.source_plan_sha256));
addpath(fullfile(j.root,'MATLAB_TRKG4_real_subjects','src'));maxNumCompThreads(3);
if ~isfolder(j.work),mkdir(j.work);end
ports=[1 0;0 1;0 -1;-1 0];ns=numel(p.sizes_mm);C=cell(ns,1);P=C;
if strcmp(j.series,'bmstu')
 d=load(p.shared_file,'parts');parts=d.parts(2,:);clear d;
 d=load(fullfile(p.work,'contacts.mat'),'C','plan_sha256');assert(strcmp(d.plan_sha256,j.source_plan_sha256));C=d.C;
 for c=1:ns,P{c}=ports;end
else
 assert(strcmp(trkg4_file_sha256(p.parts_file),p.parts_sha256));assert(strcmp(trkg4_file_sha256(p.cem_file),p.cem_sha256));
 d=load(p.parts_file,'parts');parts=d.parts;d=load(p.cem_file,'result');sc=d.result.scenarios;
 if ~iscell(sc),sc=num2cell(sc);end
 for c=1:ns
  ix=find(cellfun(@(s)strcmp(regexprep(s.id,'_(ttrkg|lateral)$',''),p.case_ids{c}),sc),1);assert(~isempty(ix));
  C{c}=sc{ix}.grouped_contact_matrix_SI;names={sc{ix}.ports.id};
  if any(strcmp(names,'TT_Ileft')),pairs={{'TT_Ileft','TT_Iright'},{'TT_Uleft','TT_Uright'},{'A','B'},{'M','N'}};
  else,pairs={{'TT_I0','TT_I1'},{'TT_U0','TT_U1'},{'A','B'},{'M','N'}};end
  pp=zeros(numel(names),4);for k=1:4,[ok,ix]=ismember(pairs{k},names);assert(all(ok));pp(ix,k)=[1;-1];end
  P{c}=pp;
 end
 clear sc;
end
clear d;
for s=1:numel(j.requests)
 q=j.requests(s);file=fullfile(j.work,[q.id '.local.json']);if isfile(file),continue;end
 rho=q.rho_ohm_m(:)';K=parts{1}/rho(1);for k=2:5,K=K+parts{k}/rho(k);end
 rows=struct([]);timer=tic;
 for c=q.indices(:)'
  [T,J,qc]=trkg4_lateral_cem_response(K,C{c},parts,rho,P{c},1:5,1);
  if strcmp(j.series,'bmstu'),a=2;b=1;else,a=4;b=3;end
  row=struct('size_mm',p.sizes_mm(c),'Z_ohm',T(a,b),'J_logrho_ohm',reshape(J(a,b,:),1,[]),'qc',qc);
  if strcmp(j.series,'scorpion'),row.TTRKG_Z_ohm=T(2,1);row.TTRKG_J_logrho_ohm=reshape(J(2,1,:),1,[]);end
  rows=[rows;row]; %#ok<AGROW>
 end
 writejson(file,struct('id',q.id,'rho_ohm_m',rho,'rows',rows,'seconds',toc(timer),...
  'source_plan_sha256',j.source_plan_sha256,'job_sha256',trkg4_file_sha256(jobfile),...
  'canonical_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
  'producer_sha256',trkg4_file_sha256([mfilename('fullpath') '.m'])));
 fprintf('FULL %s %s %s %.1f s\n',j.series,p.subject_alias,q.id,toc(timer));clear K;
end
writejson(fullfile(j.work,'complete.local.json'),struct('completed',true,'requests',numel(j.requests)));
end
function writejson(path,d)
t=[path '.writing'];f=fopen(t,'w','n','UTF-8');assert(f>=0);fprintf(f,'%s\n',jsonencode(d));fclose(f);movefile(t,path,'f');
end
