function report=run_trkg4_transverse_geometry(varargin)
% Study 20.16: fixed C01 pose, two tissues, common contacts and body mesh.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/transverse_geometry_20260911');
addParameter(p,'Stage','export');parse(p,varargin{:});opt=p.Results;
cfg=trkg4_config('nik');root=cfg.project_root;out=char(opt.OutputDir);
if ~isfolder(out),out=fullfile(root,out);end
if ~isfolder(out),mkdir(out);end
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
addpath(fullfile(root,'tools','full_pose_scan'));
meshfile=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
meshsha=trkg4_file_sha256(meshfile);
scan=fullfile(root,'output','exploratory','scan_localization_20260910');
record=jsondecode(fileread(fullfile(scan,'best_pose_review','minimum_certificate.json')));
assert(record.best.flat_index==432912 && ~record.rho.current_pose_is_joint_rho_pose_optimum);
report=struct('stage',opt.Stage,'output',out);
if strcmpi(opt.Stage,'export')
    S=load(meshfile,'fmdl_mesh_units');mm=S.fmdl_mesh_units;
    context=full_scan_context(mm);
    ref=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
    xyz=[ref.x_mm ref.y_mm ref.z_mm];r=xyz(4,:)-xyz(1,:);r=r/norm(r);
    [c0,n0]=full_scan_project(context,mean(xyz,1));u0=r-dot(r,n0)*n0;u0=u0/norm(u0);
    E=readtable(fullfile(scan,'candidate_electrodes.csv'),'TextType','string');E=E(E.candidate_id=="C01",:);
    A=readtable(fullfile(scan,'candidate_axis_geometry.csv'),'TextType','string');A=A(A.candidate_id=="C01" & A.axis_s_mm==0,:);
    centre=[record.best.x_mm record.best.y_mm record.best.z_mm];normal=[A.centre_nx A.centre_ny A.centre_nz];
    base=u0-dot(u0,normal)*normal;base=base/norm(base);side=cross(normal,base);side=side/norm(side);
    axis=cosd(record.best.phi_deg)*base+sind(record.best.phi_deg)*side;axis=axis/norm(axis);
    inward=-normal;transverse=cross(inward,axis);transverse=transverse/norm(transverse);
    frame=[axis(:) transverse(:) inward(:)];assert(norm(frame'*frame-eye(3))<1e-10 && det(frame)>0);
    nodes_mm=mm.nodes;boundary_faces=mm.boundary;elems=mm.elems;
    centres_mm=(nodes_mm(elems(:,1),:)+nodes_mm(elems(:,2),:)+nodes_mm(elems(:,3),:)+nodes_mm(elems(:,4),:))/4;
    v1=nodes_mm(elems(:,2),:)-nodes_mm(elems(:,1),:);v2=nodes_mm(elems(:,3),:)-nodes_mm(elems(:,1),:);v3=nodes_mm(elems(:,4),:)-nodes_mm(elems(:,1),:);
    volume_mm3=abs(sum(v1.*cross(v2,v3,2),2))/6;
    sizes=unique(E.L_mm);writetable(E,fullfile(out,'source_electrodes.csv'));
    save(fullfile(out,'geometry_input.mat'),'nodes_mm','boundary_faces','elems','centres_mm','volume_mm3','centre','frame','sizes','meshsha','-v7');
    local_json(fullfile(out,'input_contract.json'),struct('candidate','C01','centre_mm',centre,'axis',axis,'inward',inward, ...
      'frame_columns_s_t_d',frame,'mesh_sha256',meshsha,'electrodes_sha256',trkg4_file_sha256(fullfile(out,'source_electrodes.csv')), ...
      'pose_certificate_sha256',trkg4_file_sha256(fullfile(scan,'best_pose_review','minimum_certificate.json')), ...
      'localization_status','fixed_rho_discrete_candidate_not_validated','current_ampere',cfg.current_ampere, ...
      'contact_ohm_m2',cfg.z_contact,'requested_area_mm2',cfg.electrode_area));
    fprintf('Export C01: %d nodes, %d tetrahedra\n',size(nodes_mm,1),size(elems,1));return;
end
G=jsondecode(fileread(fullfile(out,'geometry.json')));M=load(fullfile(out,'geometry_masks.mat'));
assert(strcmp(meshsha,G.mesh_sha256));assert(strcmp(G.masks_sha256,trkg4_file_sha256(fullfile(out,'geometry_masks.mat'))));
identity=struct('geometry_sha256',trkg4_file_sha256(fullfile(out,'geometry.json')), ...
  'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']), ...
  'contact_source_sha256',trkg4_file_sha256(fullfile(root,'src','trkg4_cem_contact_matrix.m')),'runtime',version);
local_json(fullfile(out,'fem_identity.json'),identity);
prepared=fullfile(out,'prepared_fem.mat');
if ~isfile(prepared)
 S=load(meshfile,'fmdl_mesh_units');mm=S.fmdl_mesh_units;si=trkg4_scale_fmdl_to_si(mm,cfg);
 si.electrode=struct([]);si.stimulation=struct([]);F=system_mat_fields(si);Kall=F'*F;n=size(Kall,1);
 Klung=cell(size(M.masks,2),1);
 for j=1:numel(Klung)
  rr=reshape((3*(find(M.masks(:,j))-1)+(1:3))',[],1);B=F(rr,:);Klung{j}=B'*B;
  fprintf('Volume block %s prepared\n',M.model_names{j});
 end
 clear F B rr;
 E=readtable(fullfile(out,'source_electrodes.csv'),'TextType','string');sizes=unique(E.L_mm);ns=numel(sizes);
 context=electrode_patch_context(mm);surface=unique(mm.boundary(:));C=cell(ns,1);electrodes=C;P=sparse(n,2*ns);contact_rows=struct([]);
 cfg.electrode_diagnostics_file='';cfg.electrode_diagnostics_verbose=false;cfg.fail_on_electrode_diagnostics=true;
 for k=1:ns
  rows=E(E.L_mm==sizes(k),:);xyz=[rows.x_mm rows.y_mm rows.z_mm];
  assert(isequal(string(rows.electrode),["I+";"V+";"V-";"I-"]));
  [el,faces]=electrode_faces_by_area(mm,xyz,cfg.electrode_area,cfg.z_contact,context);
  tmp=mm;tmp.electrode=el;trkg4_electrode_diagnostics(tmp,xyz,string(cfg.electrode_order),cfg,xyz,faces);
  electrodes{k}=el;tmp=trkg4_scale_fmdl_to_si(tmp,cfg);C{k}=trkg4_cem_contact_matrix(tmp);
  point_nodes=zeros(4,1);
  for e=1:4
   ff=mm.boundary(all(ismember(mm.boundary,el(e).nodes),2),:);
   aa=vecnorm(cross(mm.nodes(ff(:,2),:)-mm.nodes(ff(:,1),:),mm.nodes(ff(:,3),:)-mm.nodes(ff(:,1),:),2),2,2)/2;
   cc=(mm.nodes(ff(:,1),:)+mm.nodes(ff(:,2),:)+mm.nodes(ff(:,3),:))/3;cent=sum(cc.*aa,1)/sum(aa);
   saved=[rows.patch_centroid_x_mm(e) rows.patch_centroid_y_mm(e) rows.patch_centroid_z_mm(e)];
   assert(norm(cent-saved)<1e-8 && abs(sum(aa)-rows.contact_area_mm2(e))<1e-8,'Contacts differ from saved C01');
   [err,ii]=min(vecnorm(mm.nodes(surface,:)-cent,2,2));point_nodes(e)=surface(ii);
   row=struct('L_mm',sizes(k),'electrode',rows.electrode(e),'area_mm2',sum(aa),'point_node',point_nodes(e), ...
     'snap_mm',err,'x_mm',cent(1),'y_mm',cent(2),'z_mm',cent(3));
   if isempty(contact_rows),contact_rows=row;else,contact_rows(end+1)=row;end
  end
  assert(numel(unique(point_nodes))==4);P(point_nodes([1 4]),k)=[1;-1];P(point_nodes([2 3]),ns+k)=[1;-1];
 end
 free=setdiff((1:n)',si.gnd_node);writetable(struct2table(contact_rows),fullfile(out,'contacts.csv'));
 save(prepared,'Kall','Klung','P','free','C','electrodes','mm','sizes','identity','-v7.3');
else
 saved=load(prepared,'identity');assert(isequal(saved.identity,identity),'Prepared matrices belong to another implementation');
end
load(prepared,'Kall','Klung','P','free','C','electrodes','mm','sizes');n=size(Kall,1);ns=numel(sizes);
if any(strcmpi(opt.Stage,{'solve','pem'}))
 rows=struct([]);target=fullfile(out,'pem_library.csv');if isfile(target),rows=table2struct(readtable(target));end
 for j=1:numel(Klung)
  if j<=3,ratios=M.ratios(:);else,ratios=M.extra_ratios(:);end
  for q=ratios'
   if ~isempty(rows)&&nnz([rows.model_index]==j & abs([rows.ratio]-q)<1e-12)==ns,continue;end
   tic;K=Kall+(1/q-1)*Klung{j};F=decomposition(K(free,free),'chol');U=zeros(n,2*ns);U(free,:)=F\P(free,:);
   res=norm(K(free,:)*U-P(free,:),'fro')/norm(P(free,:),'fro');assert(res<1e-7);W=Klung{j}*U(:,1:ns);elapsed=toc;
   for k=1:ns
    z=P(:,ns+k)'*U(:,k);zr=P(:,k)'*U(:,ns+k);assert(abs(z-zr)<1e-7);
    row=struct('model_index',j,'model',string(M.model_names{j}),'ratio',q,'L_mm',sizes(k),'f',z, ...
       'df_dlogq',U(:,ns+k)'*W(:,k)/q,'reciprocity_abs',abs(z-zr),'relative_residual',res,'elapsed_seconds',elapsed);
    if isempty(rows),rows=row;else,rows(end+1)=row;end
   end
   writetable(struct2table(rows),target);fprintf('PEM %s q=%.6g %.2fs\n',M.model_names{j},q,elapsed);
  end
 end
end
if any(strcmpi(opt.Stage,{'solve','cem'}))
 target=fullfile(out,'cem_baseline.csv');rows=struct([]);if isfile(target),rows=table2struct(readtable(target));end
 for j=1:3
  for k=1:ns
   if ~isempty(rows)&&any([rows.model_index]==j & [rows.L_mm]==sizes(k)),continue;end
   tic;[z,J,res,rec]=local_cem(Kall,Klung{j},C{k},free,[4 16]);
   if j==1 && k==ns
    f=trkg4_scale_fmdl_to_si(mm,cfg);f.electrode=electrodes{k};[f.stimulation,f.meas_select]=trkg4_make_4electrode_stimulation(cfg);
    sigma=ones(size(mm.elems,1),1)/4;sigma(M.masks(:,1))=1/16;E=calc_system_mat(mk_image(f,sigma));
    H=C{k};H(1:n,1:n)=H(1:n,1:n)+Kall/4+(1/16-1/4)*Klung{1};assert(norm(H-E.E,'fro')/norm(E.E,'fro')<1e-12);
   end
   row=struct('model_index',j,'model',string(M.model_names{j}),'L_mm',sizes(k),'rho1',4,'rho2',16,'Z',z, ...
       'dZ_drho1',J(1),'dZ_drho2',J(2),'relative_residual',res,'reciprocity_abs',rec,'elapsed_seconds',toc);
   if isempty(rows),rows=row;else,rows(end+1)=row;end
   writetable(struct2table(rows),target);fprintf('CEM %s L=%g %.2fs\n',M.model_names{j},sizes(k),row.elapsed_seconds);
  end
 end
end
local_json(fullfile(out,['execution_' char(opt.Stage) '.json']),struct('completed',true,'identity',identity,'stage',opt.Stage));
end

function [z,J,res,rec]=local_cem(Kall,Kl,C,free,rho)
n=size(Kall,1);K=Kall/rho(1)+(1/rho(2)-1/rho(1))*Kl;H=C;H(1:n,1:n)=H(1:n,1:n)+K;
allfree=[free;n+(1:4)'];q=sparse(n+4,2);q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];
F=decomposition(H(allfree,allfree),'chol');U=zeros(n+4,2);U(allfree,:)=F\q(allfree,:);
z=q(:,2)'*U(:,1);rec=abs(z-q(:,1)'*U(:,2));res=norm(H(allfree,:)*U-q(allfree,:),'fro')/norm(q(allfree,:),'fro');
assert(rec<1e-7 && res<1e-7);J=[U(1:n,2)'*((Kall-Kl)*U(1:n,1))/rho(1)^2,U(1:n,2)'*(Kl*U(1:n,1))/rho(2)^2];
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
