function report=run_trkg4_geometry_reduction(varargin)
% Geometry-only CT reduction, normalized two-material PEM response library.
% Exact CEM checkpoints retain the original finite contact model.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/geometry_reduction_20260910');
addParameter(p,'PrepareOnly',false);addParameter(p,'ModelIndices',1:4);
addParameter(p,'RatioIndices',[]);addParameter(p,'RunCEM',true);parse(p,varargin{:});opt=p.Results;
cfg=trkg4_config('nik');root=cfg.project_root;out=char(opt.OutputDir);
if ~isfolder(out),out=fullfile(root,out);end
assert(isfolder(out),'Run Python geometry preparation first');
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
G=jsondecode(fileread(fullfile(out,'geometry.json')));M=load(fullfile(out,'geometry_masks.mat'));
meshfile=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
assert(strcmp(strtrim(M.mesh_sha256),trkg4_file_sha256(meshfile)));
assert(strcmp(G.masks_sha256,trkg4_file_sha256(fullfile(out,'geometry_masks.mat'))));
identity=struct('mesh_sha256',G.mesh_sha256,'geometry_sha256',trkg4_file_sha256(fullfile(out,'geometry.json')), ...
 'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']), ...
 'contact_source_sha256',trkg4_file_sha256(fullfile(root,'src','trkg4_cem_contact_matrix.m')));
identity_file=fullfile(out,'fem_identity.json');
if isfile(identity_file)
 assert(isequaln(jsondecode(fileread(identity_file)),jsondecode(jsonencode(identity))),'Inputs changed: use a new study directory');
else,local_json(identity_file,identity);end
prepared=fullfile(out,'prepared_fem.mat');
if ~isfile(prepared)
 S=load(meshfile,'fmdl_mesh_units');mm=S.fmdl_mesh_units;model=trkg4_scale_fmdl_to_si(mm,cfg);
 model.electrode=struct([]);model.stimulation=struct([]);
 F=system_mat_fields(model);n=size(model.nodes,1);assert(size(F,2)==n);
 Kall=F'*F;Klung=cell(size(M.masks,2),1);
 for j=1:numel(Klung)
  rr=reshape((3*(find(M.masks(:,j))-1)+(1:3))',[],1);A=F(rr,:);Klung{j}=A'*A;
  fprintf('Prepared volume block %s\n',M.model_names{j});
 end
 clear F A rr;
 E=readtable(fullfile(root,'output','nik_trkg4_inverse_inhale_electrodes_auditfix_20260908.csv'),'TextType','string');
 assert(strcmp(trkg4_file_sha256(fullfile(root,'output','nik_trkg4_inverse_inhale_electrodes_auditfix_20260908.csv')),G.reference_geometry.electrodes_sha256));
 sizes=G.reference_geometry.L_mm(:);context=electrode_patch_context(mm);surface=unique(mm.boundary(:));
 cfg.electrode_diagnostics_file='';cfg.electrode_diagnostics_verbose=false;cfg.fail_on_electrode_diagnostics=true;
 C=cell(numel(sizes),1);electrodes=C;P=sparse(n,2*numel(sizes));Prequested=P;contact_rows=struct([]);
 for k=1:numel(sizes)
  rows=E(E.L_mm==sizes(k),:);xyz=[rows.requested_x_mm,rows.requested_y_mm,rows.requested_z_mm];
  assert(isequal(string(rows.electrode),["I_plus";"V_plus";"V_minus";"I_minus"]));
  [el,induced]=electrode_faces_by_area(mm,xyz,cfg.electrode_area,cfg.z_contact,context);
  tmp=mm;tmp.electrode=el;trkg4_electrode_diagnostics(tmp,xyz,string(cfg.electrode_order),cfg,xyz,induced);
  electrodes{k}=el;tmp=trkg4_scale_fmdl_to_si(tmp,cfg);C{k}=trkg4_cem_contact_matrix(tmp);
  nodes=zeros(4,1);requested_nodes=nodes;
  for e=1:4
   faces=mm.boundary(all(ismember(mm.boundary,el(e).nodes),2),:);
   a=mm.nodes(faces(:,2),:)-mm.nodes(faces(:,1),:);b=mm.nodes(faces(:,3),:)-mm.nodes(faces(:,1),:);
   area=vecnorm(cross(a,b,2),2,2)/2;
   cent=(mm.nodes(faces(:,1),:)+mm.nodes(faces(:,2),:)+mm.nodes(faces(:,3),:))/3;
   centroid=sum(cent.*area,1)/sum(area);
   [err,ii]=min(vecnorm(mm.nodes(surface,:)-centroid,2,2));nodes(e)=surface(ii);
   [requested_error,ii]=min(vecnorm(mm.nodes(surface,:)-xyz(e,:),2,2));requested_nodes(e)=surface(ii);
   row=struct('L_mm',sizes(k),'electrode',string(rows.electrode(e)),'area_mm2',sum(area), ...
    'centroid_x_mm',centroid(1),'centroid_y_mm',centroid(2),'centroid_z_mm',centroid(3), ...
    'point_node',nodes(e),'snap_to_centroid_mm',err,'requested_point_node',requested_nodes(e), ...
    'snap_to_request_mm',requested_error,'centroid_to_request_mm',norm(centroid-xyz(e,:)));
   if isempty(contact_rows),contact_rows=row;else,contact_rows(end+1)=row;end %#ok<AGROW>
  end
  assert(numel(unique(nodes))==4 && numel(unique(requested_nodes))==4);
  P(nodes([1 4]),k)=[1;-1];P(nodes([2 3]),numel(sizes)+k)=[1;-1];
  Prequested(requested_nodes([1 4]),k)=[1;-1];Prequested(requested_nodes([2 3]),numel(sizes)+k)=[1;-1];
 end
 free=setdiff((1:n)',model.gnd_node);writetable(struct2table(contact_rows),fullfile(out,'contacts.csv'));
 save(prepared,'mm','Kall','Klung','P','Prequested','free','C','electrodes','sizes','-v7.3');
 fprintf('Prepared study: %d nodes, %d tetrahedra\n',n,size(mm.elems,1));
end
if opt.PrepareOnly,report=identity;return;end
load(prepared,'mm','Kall','Klung','P','Prequested','free','C','electrodes','sizes');
n=size(Kall,1);ns=numel(sizes);ratios=M.ratios(:);
indices=opt.RatioIndices;if isempty(indices),indices=1:numel(ratios);end
for j=opt.ModelIndices
 target=fullfile(out,['pem_' M.model_names{j} '.csv']);rows=struct([]);
 if isfile(target),rows=table2struct(readtable(target));end
 for ri=indices
  if ~isempty(rows)&&nnz([rows.ratio_index]==ri)==ns,continue;end
  contrast=ratios(ri);started=tic;K=Kall+(1/contrast-1)*Klung{j};
  factor=decomposition(K(free,free),'chol');U=zeros(n,2*ns);U(free,:)=factor\P(free,:);
  residual=norm(K(free,:)*U-P(free,:),'fro')/norm(P(free,:),'fro');
  assert(residual<1e-7,'PEM linear solver failed');
  W=Klung{j}*U(:,1:ns);elapsed=toc(started);
  for k=1:ns
   z=P(:,ns+k)'*U(:,k);zr=P(:,k)'*U(:,ns+k);derivative=U(:,ns+k)'*W(:,k)/contrast;
   assert(isfinite(z)&&isfinite(derivative)&&abs(z-zr)<1e-7);
   row=struct('ratio_index',ri,'ratio',contrast,'L_mm',sizes(k),'f_ohm_per_ohm_m',z, ...
    'df_dlog_ratio',derivative,'reciprocity_abs',abs(z-zr),'relative_residual',residual,'elapsed_all_sizes_seconds',elapsed);
   if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
  end
  writetable(struct2table(rows),target);
  fprintf('PEM %s ratio %.6g (%d/%d): %.2fs, residual %.2g\n',M.model_names{j},contrast,ri,numel(ratios),elapsed,residual);
 end
end
if opt.RunCEM
 rho=[G.reference_geometry.rho_soft_ohm_m,G.reference_geometry.rho_lung_ohm_m];
 K=Kall/rho(1)+(1/rho(2)-1/rho(1))*Klung{1};factor=decomposition(K(free,free),'chol');
 requested=zeros(n,2*ns);requested(free,:)=factor\Prequested(free,:);
 target=fullfile(out,'cem_reference.csv');rows=struct([]);if isfile(target),rows=table2struct(readtable(target));end
 for k=1:ns
  if ~isempty(rows)&&any([rows.L_mm]==sizes(k)),continue;end
  started=tic;Kcem=C{k};Kcem(1:n,1:n)=Kcem(1:n,1:n)+K;
  q=sparse(n+4,2);q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];allfree=[free;n+(1:4)'];
  V=zeros(n+4,2);H=Kcem(allfree,allfree);
  assert(norm(H-H','fro')/norm(H,'fro')<1e-12,'CEM assembly is asymmetric');
  factor=decomposition((H+H')/2,'chol');V(allfree,:)=factor\q(allfree,:);
  z=q(:,2)'*V(:,1);zr=q(:,1)'*V(:,2);residual=norm(Kcem(allfree,:)*V-q(allfree,:),'fro')/norm(q(allfree,:),'fro');
  fsi=trkg4_scale_fmdl_to_si(mm,cfg);fsi.electrode=electrodes{k};
  [fsi.stimulation,fsi.meas_select]=trkg4_make_4electrode_stimulation(cfg);
  sigma=ones(size(M.masks,1),1)/rho(1);sigma(M.masks(:,1))=1/rho(2);
  exact=calc_system_mat(mk_image(fsi,sigma));matrix_error=norm(Kcem-exact.E,'fro')/norm(exact.E,'fro');
  assert(matrix_error<1e-12 && residual<1e-7 && abs(z-zr)<1e-7,'CEM check failed');
  row=struct('L_mm',sizes(k),'Z_cem_ohm',z,'Z_requested_pem_ohm',Prequested(:,ns+k)'*requested(:,k), ...
   'dZ_drho1',V(1:n,2)'*((Kall-Klung{1})*V(1:n,1))/rho(1)^2, ...
   'dZ_drho2',V(1:n,2)'*(Klung{1}*V(1:n,1))/rho(2)^2, ...
   'reciprocity_abs',abs(z-zr),'relative_residual',residual,'matrix_error',matrix_error,'elapsed_seconds',toc(started));
  if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
  writetable(struct2table(rows),target);eidors_cache('clear_all');
  fprintf('CEM L=%g: Z=%.9g, %.1fs\n',sizes(k),z,row.elapsed_seconds);
 end
end
qcfile=fullfile(out,'derivative_qc.csv');
if opt.RunCEM && ~isfile(qcfile)
 contrast=G.reference_geometry.rho_lung_ohm_m/G.reference_geometry.rho_soft_ohm_m;
 k=find(sizes==90,1);K=Kall+(1/contrast-1)*Klung{1};
 U=zeros(n,2);U(free,:)=decomposition(K(free,free),'chol')\P(free,[k,ns+k]);
 exact=U(:,2)'*(Klung{1}*U(:,1))/contrast;checks=struct([]);
 for step=[.001,.0005]
  z=zeros(1,2);
  for side=1:2
   ratio=contrast*exp((2*side-3)*step);K=Kall+(1/ratio-1)*Klung{1};
   u=decomposition(K(free,free),'chol')\P(free,k);z(side)=P(free,ns+k)'*u;
  end
  fd=diff(z)/(2*step);err=abs(fd-exact)/max(abs(exact),1e-12);
  assert(err<1e-4,'Adjoint derivative failed independent finite differences');
  row=struct('log_step',step,'adjoint',exact,'finite_difference',fd,'relative_error',err);
  if isempty(checks),checks=row;else,checks(end+1)=row;end %#ok<AGROW>
 end
 writetable(struct2table(checks),qcfile);
end
report=identity;report.status='exploratory_hypothesis_not_validated';report.matlab_version=version;
report.mesh_convergence_tested=false;report.ratio_count=numel(ratios);report.geometry_count=numel(Klung);
local_json(fullfile(out,'fem_execution.json'),report);
end

function local_json(file,data)
fid=fopen(file,'w');assert(fid>0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(data,'PrettyPrint',true));
end
