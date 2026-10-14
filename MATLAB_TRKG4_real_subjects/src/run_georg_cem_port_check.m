function report=run_georg_cem_port_check(varargin)
% Synthetic algebra check and optional CT forward control; no inverse fitting.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/georg_lateral_inverse_20261011/port_check');
addParameter(p,'ModelDir','');parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');root=cfg.project_root;trkg4_prepare_runtime(cfg);
addpath(fullfile(root,'tools','full_pose_scan'));maxNumCompThreads(6);
out=fullfile(root,char(a.OutputDir));assert(~isfolder(out),'Use a new immutable check directory');mkdir(out);
rng(20261011);n=40;B=randn(n);K=B'*B;K=K-diag(sum(K,2));
% A complete graph Laplacian with positive conductances is a bulk operator.
A=rand(n);A=triu(A,1);A=A+A';K=diag(sum(A,2))-A;
contacts=cell(10,1);g=[1;1;2;2;3;4;5;6;7;8];
for j=1:10,R=chol(diag([2,3])+ones(2));contacts{j}=struct('nodes',(2*j-1:2*j)','R',R);end
I=zeros(8,4);I([5,8],1)=[1;-1];I([6,7],2)=[1;-1];I([1,4],3)=[1;-1];I([2,3],4)=[1;-1];
[T,meta,V]=trkg4_cem_port_schur(K,contacts,g,I,1);
[direct,fullV]=direct_synthetic(K,contacts,g,I,1);
err=max(abs(T-direct),[],'all');field_error=norm(V-fullV,'fro')/norm(fullV,'fro');
assert(err<1e-10&&field_error<1e-10);
% Include the grounded node within a physical contact (not discarded).
[T2]=trkg4_cem_port_schur(K,contacts,g,I,20);direct2=direct_synthetic(K,contacts,g,I,20);
assert(max(abs(T2-direct2),[],'all')<1e-10);
delta=1e-4;
Tp=trkg4_cem_port_schur(K*exp(-delta),contacts,g,I,1);
Tm=trkg4_cem_port_schur(K*exp(delta),contacts,g,I,1);
analytic=V'*K*V;derivative_error=norm((Tp-Tm)/(2*delta)-analytic,'fro')/norm(analytic,'fro');
assert(derivative_error<1e-7,'Port adjoint derivative failed');
report=struct('synthetic_transfer_max_error_ohm',err,'synthetic_field_relative_error',field_error,...
 'synthetic_grounded_contact_max_error_ohm',max(abs(T2-direct2),[],'all'),...
 'synthetic_passed',true,'synthetic_derivative_relative_error',derivative_error,...
 'ct_cases',struct([]),'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'operator_sha256',trkg4_file_sha256(which('trkg4_cem_port_schur')),'synthetic_meta',meta);
if ~isempty(a.ModelDir)
 current=trkg4_georg_current_model(root,false);
 base=fullfile(root,char(a.ModelDir));plan=jsondecode(fileread(fullfile(base,'plan.local.json')));
 assert(strcmp(trkg4_file_sha256(fullfile(root,current.model_directory,'body_embedded_arms_370mm.stl')),...
   plan.sources(7).sha256),'Geometry identity mismatch');
 id=run_georg_ct_model('OutputDir',base,'Stage','prepare');
 cache=fullfile(base,['prepared_' id.plan_sha256(1:8) '_' id.worker_sha256(1:8) '.mat']);S=load(cache);
 ref=jsondecode(fileread(fullfile(base,'reference_materials.local.json')));rho=ref.rho_ohm_m(:);
 K=sparse(size(S.mm.nodes,1),size(S.mm.nodes,1));for j=1:5,K=K+S.parts{2,j}/rho(j);end
 si=trkg4_scale_fmdl_to_si(S.mm,cfg);ctx=electrode_patch_context(S.mm);
 cs=jsondecode(fileread(plan.contacts_file));
 for caseid=1:numel(cs.scenarios)
  r=cs.scenarios(caseid);groups=r.groups_mm;if isnumeric(groups),groups=num2cell(groups,[2,3]);end
  xyz=[];g=[];for j=1:numel(groups),points=reshape(groups{j},[],3);xyz=[xyz;points];g=[g;repmat(j,size(points,1),1)];end %#ok<AGROW>
  [~,faces]=electrode_faces_by_area(S.mm,xyz,cs.nominal_area_mm2,plan.contact_impedance_ohm_m2,ctx);
  c=cellfun(@(f)full_scan_contact(si,f,plan.contact_impedance_ohm_m2),faces,'UniformOutput',false);
  ng=numel(groups);I=zeros(ng,4);I(r.drive,1)=[1;-1];I(r.measure,2)=[1;-1];
  I([1,4],3)=[1;-1];I([2,3],4)=[1;-1];
  [T,meta,V]=trkg4_cem_port_schur(K,c,g,I,1);[Td,Vd]=direct_synthetic(K,c,g,I,1);
  err=max(abs(T-Td),[],'all');fe=norm(V-Vd,'fro')/norm(Vd,'fro');
  assert(err<1e-5&&fe<1e-7,'Exact Schur and full CEM disagree');
  J=zeros(4,4,5);for j=1:5,J(:,:,j)=V'*(S.parts{2,j}*V)/rho(j);end
  row=struct('case_id',r.id,'rho_ohm_m',rho','port_transfer_ohm',T,...
    'J_logrho_port_ohm',J,'full_cem_max_error_ohm',err,'field_relative_error',fe,...
    'port_order',{{'lateral_drive','lateral_voltage','arm_drive','arm_voltage'}},...
    'own_drive_Z_ohm',T(2,1),'counter_drive_Z_ohm',T(2,3),...
    'in_phase_equal_current_Z_ohm',T(2,1)+T(2,3),...
    'opposite_phase_equal_current_Z_ohm',T(2,1)-T(2,3),...
    'quadrature_magnitude_equal_current_ohm',hypot(T(2,1),T(2,3)),...
    'actual_excitation_scenario_known',false,'meta',meta,'prepared_sha256',trkg4_file_sha256(cache));
  if isempty(report.ct_cases),report.ct_cases=row;else,report.ct_cases(end+1)=row;end
  fprintf('PORT %s own=%.8g counter=%.8g exact-error=%.3g\n',r.id,T(2,1),T(2,3),err);
 end
end
fid=fopen(fullfile(out,'port_check.local.json'),'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end

function [T,V]=direct_synthetic(K,c,g,I,ground)
n=size(K,1);ng=max(g);C=sparse(n+ng,n+ng);
for j=1:numel(c)
 ids=[c{j}.nodes;n+g(j)];D=c{j}.R'*c{j}.R;one=ones(size(D,1),1);
 C(ids,ids)=C(ids,ids)+[D,-D*one;-one'*D,one'*D*one];
end
C(1:n,1:n)=C(1:n,1:n)+K;free=(1:n+ng)';free(ground)=[];
rhs=sparse(n+ng,size(I,2));rhs(n+(1:ng),:)=I;U=zeros(size(rhs));
U(free,:)=decomposition((C(free,free)+C(free,free)')/2,'chol')\rhs(free,:);
assert(norm(C(free,:)*U-rhs(free,:),'fro')/norm(rhs,'fro')<1e-7);
T=I'*U(n+(1:ng),:);V=U(1:n,:);
end
