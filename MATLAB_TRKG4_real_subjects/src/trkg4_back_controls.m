function trkg4_back_controls(planfile)
% Quadrature comparison and same lateral contact operator on the refined bulk.
p=jsondecode(fileread(planfile));root=p.root;work=p.work;maxNumCompThreads(4);
addpath(fullfile(root,'MATLAB_TRKG4_real_subjects','src'),fullfile(root,'MATLAB_TRKG4_real_subjects','tools'));
cfg=trkg4_config(p.subject);
assert(strcmp(trkg4_file_sha256(p.mesh_path),p.mesh_sha256));
ph=trkg4_file_sha256(planfile);data=load(p.shared_file,'parts','identity');
assert(strcmp(data.identity.plan_sha256,ph));fine=data.parts(2,:);coarse=data.parts(1,:);clear data;
d=load(fullfile(work,'contacts.mat'),'C','plan_sha256');assert(strcmp(d.plan_sha256,ph));C=d.C;clear d;
rho=p.scenarios(strcmp({p.scenarios.name},'base')).rho_ohm_m(:)';ports=[1 0;0 1;0 -1;-1 0];
K=coarse{1}/rho(1);for k=2:5,K=K+coarse{k}/rho(k);end
quad=struct([]);
for c=1:numel(p.contacts)
 cc=p.contacts(c);if ~ismember(cc.size_mm,[50 140]),continue;end
 original=jsondecode(fileread(fullfile(work,['base_' cc.id '.local.json'])));
 [T,J,qc]=trkg4_lateral_cem_response(K,C{c},coarse,rho,ports,1:5,1);
 j=reshape(J(2,1,:),1,[]);j0=original.J_logrho_ohm(:)';
 row=struct('id',cc.id,'size_mm',cc.size_mm,'side',cc.side,'Z_q1024_ohm',T(2,1),...
  'Z_q4096_ohm',original.Z_ohm,'relative_Z_difference',abs(T(2,1)/original.Z_ohm-1),...
  'J_q1024_ohm',j,'J_q4096_ohm',j0,'relative_J_l2_difference',norm(j-j0)/norm(j0),...
  'absolute_J_difference_ohm',abs(j-j0),'qc',qc);
 quad=[quad;row]; %#ok<AGROW>
 fprintf('QUADRATURE %s dZ %.6g dJ %.6g\n',cc.id,row.relative_Z_difference,row.relative_J_l2_difference);
end
clear K coarse C;
oldfile=fullfile(root,'MATLAB_TRKG4_real_subjects','output','exploratory','bmstu_inverse_20261111',p.subject,'plan.local.json');
old=jsondecode(fileread(oldfile));g=jsondecode(fileread(old.geometry_path));
assert(strcmp(trkg4_file_sha256(old.geometry_path),old.geometry_sha256));
d=load(fullfile(old.work,'contacts.mat'),'C','plan_sha256');assert(strcmp(d.plan_sha256,trkg4_file_sha256(oldfile)));oldC=d.C;clear d;
M=load(p.mesh_path);mm=struct('name','same lateral contacts on refined bulk','nodes',M.nodes_mm,...
 'elems',double(M.elems),'boundary',double(M.boundary));clear M;
ctx=trkg4_local_patch_context(mm);n=size(fine{1},1);lateralC=cell(2,1);ci=0;
for size_mm=[50 140]
 ci=ci+1;cc=g.contacts([g.contacts.size_mm]==size_mm);
 [el,faces]=trkg4_local_exact_patches(ctx,cc.centres_lps_mm,g.nominal_area_mm2,g.z_contact_ohm_m2);
 for e=1:4,el(e).faces=mm.boundary(faces{e},:);end
 mm.electrode=el;lateralC{ci}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(mm,cfg));
 index=find(old.sizes_mm==size_mm);source=oldC{index};nold=size(source,1)-4;
 [ii,jj,vv]=find(source);ii(ii>nold)=n+ii(ii>nold)-nold;jj(jj>nold)=n+jj(jj>nold)-nold;
 embedded=sparse(ii,jj,vv,n+4,n+4);
 assert(norm(lateralC{ci}-embedded,'fro')/norm(embedded,'fro')<1e-12,'Lateral contact operator changed');
end
clear mm ctx oldC embedded source;
synfile=fullfile(old.work,'synthetic.local.json');syn=jsondecode(fileread(synfile));
base=syn.scenarios(strcmp({syn.scenarios.name},'base')&~[syn.scenarios.coarse]);
assert(numel(base)==1);assert(max(abs(base.truth_rho_ohm_m(:)-rho(:)))<1e-12);
K=fine{1}/rho(1);for k=2:5,K=K+fine{k}/rho(k);end
lateral=struct([]);ci=0;
for size_mm=[50 140]
 ci=ci+1;index=find(old.sizes_mm==size_mm);
 [T,J,qc]=trkg4_lateral_cem_response(K,lateralC{ci},fine,rho,ports,1:5,1);
 j=reshape(J(2,1,:),1,[]);j0=base.J_logrho_ohm(index,:);
 row=struct('size_mm',size_mm,'Z_original_mesh_ohm',base.Z_ohm(index),'Z_refined_mesh_ohm',T(2,1),...
  'relative_Z_difference',abs(T(2,1)/base.Z_ohm(index)-1),'J_original_mesh_ohm',j0,...
  'J_refined_mesh_ohm',j,'relative_J_l2_difference',norm(j-j0)/norm(j0),...
  'absolute_J_difference_ohm',abs(j-j0),'contact_operator_unchanged',true,'qc',qc);
 lateral=[lateral;row]; %#ok<AGROW>
 fprintf('LATERAL CONTROL L%d dZ %.6g dJ %.6g\n',size_mm,row.relative_Z_difference,row.relative_J_l2_difference);
end
writejson(fullfile(work,'numerical_controls.local.json'),struct('plan_sha256',ph,...
 'quadrature',quad,'lateral_original_vs_refined_bulk',lateral,...
 'old_plan_sha256',trkg4_file_sha256(oldfile),'old_synthetic_sha256',trkg4_file_sha256(synfile),...
 'driver_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'global_mesh_convergence_proven',false));
end

function writejson(path,d)
temp=[path '.writing'];f=fopen(temp,'w','n','UTF-8');assert(f>=0);
fprintf(f,'%s\n',jsonencode(d));fclose(f);movefile(temp,path,'f');
end
