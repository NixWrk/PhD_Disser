function report=run_georg_ct_model(varargin)
% Shared CT-derived Georg model, finite-contact EIDORS assembly and scenarios.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/georg_ct_model_20261011/fem_h8');
addParameter(p,'Stage','forward');addParameter(p,'NominalOnly',false);parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');out=char(a.OutputDir);
if ~isfile(fullfile(out,'plan.local.json')),out=fullfile(cfg.project_root,out);end
plan=jsondecode(fileread(fullfile(out,'plan.local.json')));
assert(strcmp(plan.schema,'georg_shared_ct_fem_v1')&&plan.frequency_hz==100000);
if isfield(plan,'numerical_quadrature_points')
 assert(isequal(double(plan.numerical_quadrature_points(:))',[256,1024]),...
  'trkg4:legacyGeorgModel',...
  'This legacy worker requires 256/1024. Use georg_quality_forward with explicit model, inputs and reference for the new model.');
end
for j=1:numel(plan.sources)
 assert(strcmp(trkg4_file_sha256(plan.sources(j).path),plan.sources(j).sha256),'Shared model input changed');
end
cfg.frequency_hz=plan.frequency_hz;trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
contacts=jsondecode(fileread(plan.contacts_file));scenarios=contacts.scenarios;
identity=struct('plan_sha256',trkg4_file_sha256(fullfile(out,'plan.local.json')),...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'patch_builder_sha256',trkg4_file_sha256(which('electrode_faces_by_area')),...
 'cem_builder_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')),...
 'field_builder_sha256',trkg4_file_sha256(which('system_mat_fields')));
cache=fullfile(out,['prepared_' identity.plan_sha256(1:8) '_' identity.worker_sha256(1:8) '.mat']);
if ~isfile(cache)
 M=load(fullfile(out,'mesh.mat'));
 mm=eidors_obj('fwd_model','Georg shared CT-derived model');mm.nodes=M.nodes_mm;
 mm.elems=double(M.elems);mm.boundary=double(M.boundary);mm.gnd_node=1;
 mm.electrode=struct([]);mm.stimulation=struct([]);mm.solve=@fwd_solve_1st_order;
 mm.system_mat=@system_mat_1st_order;mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;
 si=trkg4_scale_fmdl_to_si(mm,cfg);F=system_mat_fields(si);parts=cell(2,5);
 for rule=1:2
  q=[256,1024];fraction=load(fullfile(out,sprintf('fractions_q%d.mat',q(rule))));
  assert(all(sum(fraction.fractions,2)==1));
  for j=1:5
   w=repelem(fraction.fractions(:,j),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
   parts{rule,j}=B'*B;
  end
  fprintf('Material matrices q%d ready\n',q(rule));
 end
 clear F B w fraction;
 save(cache,'parts','mm','identity','-v7.3');
end
S=load(cache);assert(isequal(S.identity,identity));
if strcmp(a.Stage,'prepare'),report=identity;return;end
assert(strcmp(a.Stage,'forward'));
ref=jsondecode(fileread(fullfile(out,'reference_materials.local.json')));rho=ref.rho_ohm_m(:);
n=size(S.mm.nodes,1);K=material_matrix(S.parts(2,:),rho);
context=electrode_patch_context(S.mm);results=struct([]);qc=struct([]);
dest=fullfile(out,'scenarios');if ~isfolder(dest),mkdir(dest);end
for s=1:numel(scenarios)
 r=scenarios(s);
 if a.NominalOnly&&~strcmp(r.perturbation,'nominal')&&~strcmp(r.family,'TTRKG')&&~strcmp(r.family,'reference'),continue;end
 file=fullfile(dest,[r.id '.local.json']);
 if isfile(file)
  prior=jsondecode(fileread(file));assert(isequal(prior.identity,identity),'Stale scenario result');
  if isempty(results),results=prior;else,results(end+1)=prior;end
  continue;
 end
 groups=r.groups_mm;if isnumeric(groups),groups=num2cell(groups,[2,3]);end
 if isstruct(groups),error('Unexpected contact group representation');end
 if iscell(groups)
  xyz=[];gids=[];
  for g=1:numel(groups)
   points=reshape(groups{g},[],3);xyz=[xyz;points];gids=[gids;repmat(g,size(points,1),1)]; %#ok<AGROW>
  end
 else
  error('Invalid grouped contacts');
 end
 [el,faces]=electrode_faces_by_area(S.mm,xyz,contacts.nominal_area_mm2,plan.contact_impedance_ohm_m2,context);
 assert(numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes)),'Overlapping contacts');
 grouped=struct([]);local_qc=struct([]);
 for e=1:numel(el)
  ff=S.mm.boundary(faces{e},:);v=S.mm.nodes;
  aa=vecnorm(cross(v(ff(:,2),:)-v(ff(:,1),:),v(ff(:,3),:)-v(ff(:,1),:),2),2,2)/2;
  cc=(v(ff(:,1),:)+v(ff(:,2),:)+v(ff(:,3),:))/3;
  centroid=sum(cc.*aa,1)/sum(aa);
  row=struct('scenario',r.id,'physical_contact',e,'group',gids(e),...
   'area_mm2',sum(aa),'relative_area_error',abs(sum(aa)/contacts.nominal_area_mm2-1),...
   'centroid_offset_mm',norm(centroid-xyz(e,:)),...
   'requested_centre_lps_mm',xyz(e,:),'centroid_lps_mm',centroid,...
   'radial_extent_mm',max(vecnorm(v(el(e).nodes,:)-xyz(e,:),2,2)));
  if isempty(local_qc),local_qc=row;else,local_qc(end+1)=row;end
 end
 if strcmp(r.perturbation,'nominal')
  pp=cell(numel(el),1);
  for e=1:numel(el)
   ff=S.mm.boundary(faces{e},:);[nn,~,ii]=unique(ff(:));
   pp{e}=struct('vertices_lps_mm',S.mm.nodes(nn,:),'faces',reshape(ii,size(ff)),...
       'contact',e,'requested_centre_lps_mm',xyz(e,:));
  end
  write_json(fullfile(dest,[r.id '_patches.local.json']),pp);
 end
 assert(max([local_qc.relative_area_error])<.12&&max([local_qc.centroid_offset_mm])<1&&...
  max([local_qc.radial_extent_mm])<3.7,'Finite contact geometry failed QC');
 for g=1:numel(groups)
  ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
  grouped(g).faces=S.mm.boundary(vertcat(faces{ix}),:);grouped(g).z_contact=plan.contact_impedance_ohm_m2;
 end
 tmp=S.mm;tmp.electrode=grouped;C=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(tmp,cfg));
 [z,J,res,rec,seconds]=solve(K,C,r.drive,r.measure,S.parts(2,:),rho);
 coarse=NaN;fd_error=NaN;assembly_error=NaN;
 if strcmp(r.perturbation,'nominal')||strcmp(r.family,'TTRKG')||strcmp(r.family,'reference')
  coarse=solve(material_matrix(S.parts(1,:),rho),C,r.drive,r.measure,S.parts(1,:),rho);
 end
 if strcmp(r.id,'TEPC4_nominal')
  fraction=load(fullfile(out,'fractions_q1024.mat'));
  check_model=tmp;check_model.electrode=rmfield(check_model.electrode,'faces');
  img=mk_image(trkg4_scale_fmdl_to_si(check_model,cfg),1);img.elem_data=fraction.fractions*(1./rho);
  standard=calc_system_mat(img);H=C;H(1:n,1:n)=H(1:n,1:n)+K;
  assembly_error=norm(H-standard.E,'fro')/norm(standard.E,'fro');
  assert(assembly_error<1e-12,'Independent EIDORS assembly verification failed');
  clear fraction standard img H;
  fd=zeros(1,5);h=1e-3;
  for j=1:5
   rp=rho;rm=rho;rp(j)=rho(j)*exp(h);rm(j)=rho(j)*exp(-h);
   zp=solve(material_matrix(S.parts(2,:),rp),C,r.drive,r.measure,S.parts(2,:),rp);
   zm=solve(material_matrix(S.parts(2,:),rm),C,r.drive,r.measure,S.parts(2,:),rm);fd(j)=(zp-zm)/(2*h);
  end
  fd_error=norm(fd-J)/norm(J);assert(fd_error<1e-5,'Sensitivity finite-difference check failed');
 end
 rr=struct('id',r.id,'family',r.family,'montage',r.montage,'perturbation',r.perturbation,...
  'shift_mm',r.shift_mm,'Z_ohm',z,'J_logrho_ohm',J,'rho_ohm_m',rho',...
  'linear_system_relative_residual',res,'reciprocity_error_ohm',rec,'solve_seconds',seconds,...
  'quadrature_difference_ohm',abs(z-coarse),'jacobian_relative_error',fd_error,...
  'eidors_assembly_relative_error',assembly_error,...
  'physical_validation',false,'identity',identity,'contact_qc',local_qc);
 write_json(file,rr);
 if isempty(results),results=rr;else,results(end+1)=rr;end
 if isempty(qc),qc=local_qc;else,qc=[qc,local_qc];end %#ok<AGROW>
 fprintf('%s Z=%.8g Ohm, reciprocity %.3g, %.2fs\n',r.id,z,rec,seconds);
end
write_json(fullfile(out,'forward_results.local.json'),results);
report=results;
end

function K=material_matrix(parts,rho)
n=size(parts{1},1);K=sparse(n,n);for j=1:5,K=K+parts{j}/rho(j);end
end

function [z,J,res,rec,seconds]=solve(K,C,drive,measure,parts,rho)
t=tic;n=size(K,1);ng=size(C,1)-n;H=C;H(1:n,1:n)=H(1:n,1:n)+K;
rhs=sparse(n+ng,2);rhs(n+drive,1)=[1;-1];rhs(n+measure,2)=[1;-1];
free=(2:n+ng)';B=H(free,free);U=zeros(n+ng,2);
U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
z=rhs(:,2)'*U(:,1);rec=abs(z-rhs(:,1)'*U(:,2));
assert(res<1e-7&&rec<1e-7&&isfinite(z),'CEM numerical solve failed');
J=zeros(1,5);for j=1:5,J(j)=U(1:n,2)'*(parts{j}*U(1:n,1))/rho(j);end
seconds=toc(t);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
