function trkg4_back_uniform_flux(jobfile)
% Diagnostic finite uniform-current/area-average contacts on unchanged CT FEM.
% The reference remains the complete electrode model. No new anatomical solver.
j=jsondecode(fileread(jobfile));p=jsondecode(fileread(j.plan_file));
addpath(fullfile(p.root,'MATLAB_TRKG4_real_subjects','src'));
maxNumCompThreads(4);plan_hash=trkg4_file_sha256(j.plan_file);
assert(strcmp(plan_hash,j.plan_sha256));
d=load(p.shared_file,'parts','identity');assert(strcmp(d.identity.plan_sha256,plan_hash));
parts=d.parts(2,:);clear d;
d=load(fullfile(p.work,'contacts.mat'),'C','plan_sha256');assert(strcmp(d.plan_sha256,plan_hash));
C=d.C;clear d;n=size(parts{1},1);nc=numel(C);ports=[1 0;0 1;0 -1;-1 0];
Q=sparse(n,2*nc);area=zeros(nc,4);
for c=1:nc
 D=full(diag(C{c}(n+(1:4),n+(1:4))));
 W=-C{c}(1:n,n+(1:4))*spdiags(1./D,0,4,4);
 assert(max(abs(full(sum(W,1))-1))<1e-10);
 Q(:,2*c-1:2*c)=W*ports;area(c,:)=D*p.z_contact_ohm_m2;
end
clear C W D;
assert(max(abs(full(sum(Q,1))))<1e-10);free=2:n;
for s=1:numel(j.scenarios)
 if iscell(j.scenarios),scenario=j.scenarios{s};else,scenario=j.scenarios(s);end
 file=fullfile(j.work,[scenario.name '.local.json']);
 if isfile(file),continue;end
 timer=tic;rho=scenario.rho_ohm_m(:)';K=parts{1}/rho(1);
 for k=2:5,K=K+parts{k}/rho(k);end
 A=K(free,free);factor=decomposition((A+A')/2,'chol');clear A;
 U=zeros(n,2*nc);U(free,:)=factor\Q(free,:);clear factor;
 residual=norm(K(free,:)*U-Q(free,:),'fro')/norm(Q(free,:),'fro');
 T=full(Q'*U);reciprocity=max(abs(T-T'),[],'all');
 J=zeros(nc,5);
 for k=1:5
  V=parts{k}*U(:,1:2:end);J(:,k)=sum(U(:,2:2:end).*V,1)'/rho(k);
 end
 z=diag(T(2:2:end,1:2:end));homogeneous_identity=max(abs(sum(J,2)-z));
 assert(residual<1e-7&&reciprocity<1e-7&&homogeneous_identity<1e-7);
 rows=struct([]);
 for c=1:nc
  cc=p.contacts(c);rows=[rows;struct('id',cc.id,'side',cc.side,'size_mm',cc.size_mm,...
   'Z_ohm',z(c),'rho_ohm_m',rho,'J_logrho_ohm',J(c,:),'area_m2',area(c,:))]; %#ok<AGROW>
 end
 qc=struct('system_relative_residual',residual,'reciprocity_ohm',reciprocity,...
  'homogeneous_scaling_identity_error_ohm',homogeneous_identity);
 writejson(file,struct('scenario',scenario,'rows',rows,'qc',qc,'seconds',toc(timer),...
  'plan_sha256',plan_hash,'job_sha256',trkg4_file_sha256(jobfile),...
  'producer_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
  'electrode_model','uniform finite surface flux; area-average voltage; exact same CEM patches'));
 fprintf('UNIFORM %s %s %.1f s\n',p.subject,scenario.name,toc(timer));clear K U V T;
end
writejson(fullfile(j.work,'complete.local.json'),struct('completed',true,'scenarios',numel(j.scenarios),...
 'plan_sha256',plan_hash,'job_sha256',trkg4_file_sha256(jobfile)));
end

function writejson(file,d)
t=[file '.writing'];f=fopen(t,'w','n','UTF-8');assert(f>=0);fprintf(f,'%s\n',jsonencode(d));fclose(f);movefile(t,file,'f');
end
