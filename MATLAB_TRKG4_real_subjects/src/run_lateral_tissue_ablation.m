function report=run_lateral_tissue_ablation(varargin)
% Sequential restoration of bone and heart at fixed C01 and finite 5 mm CEM.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_tissue_ablation_20260916');
addParameter(p,'Stage','solve');parse(p,varargin{:});
cfg=trkg4_config('nik');root=cfg.project_root;out=fullfile(root,char(p.Results.OutputDir));
if ~isfolder(out),mkdir(out);end
source=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
if strcmpi(p.Results.Stage,'export')
    A=load(source,'tissue_id','tissue_names','fmdl_mesh_units');
    G=load(fullfile(root,'output/exploratory/transverse_consistency_20260914/geometry_input.mat'));
    assert(strcmp(trkg4_file_sha256(source),G.meshsha));
    assert(isequal(A.fmdl_mesh_units.nodes,G.nodes_mm)&&isequal(A.fmdl_mesh_units.elems,G.elems));
    tissue_id=A.tissue_id;tissue_names=A.tissue_names;mesh_sha256=G.meshsha;
    target=fullfile(out,'original_labels.mat');assert(~isfile(target),'Frozen export exists.');
    save(target,'tissue_id','tissue_names','mesh_sha256','-v7');
    report=struct('source_sha256',G.meshsha,'original_labels_sha256',trkg4_file_sha256(target));
    local_json(fullfile(out,'export.json'),report);return;
end
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
planfile=fullfile(out,'plan.json');P=jsondecode(fileread(planfile));
for i=1:numel(P.sources)
    assert(strcmp(trkg4_file_sha256(fullfile(root,P.sources(i).path)),P.sources(i).sha256),'Input hash changed.');
end
assert(strcmp(trkg4_file_sha256(fullfile(out,'requests.csv')),P.requests_sha256));
assert(strcmp(trkg4_file_sha256(fullfile(out,'material_masks.mat')),P.masks_sha256));
identity=struct('plan_sha256',trkg4_file_sha256(planfile),'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
base=fullfile(root,'output/exploratory/lateral_array_design_20260916');
S=load(fullfile(base,'prepared_fem.mat'));assert(isequal(S.sizes(:),P.sizes_mm(:)));
prepared=fullfile(out,'material_matrices.mat');
if ~isfile(prepared)
    M=load(fullfile(base,'refined_mesh.mat'));T=load(fullfile(out,'material_masks.mat'));
    old=load(source,'fmdl_mesh_units');mm=old.fmdl_mesh_units;
    mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
    mm.electrode=struct([]);mm.stimulation=struct([]);if isfield(mm,'mat_idx'),mm=rmfield(mm,'mat_idx');end
    si=trkg4_scale_fmdl_to_si(mm,cfg);F=system_mat_fields(si);
    assert(norm(F'*F-S.Kall,'fro')/norm(S.Kall,'fro')<1e-12,'Matrix scale or mesh changed.');
    Kparts=cell(4,1);masks={logical(T.soft_mask),logical(M.reference_mask),logical(T.bone_mask),logical(T.heart_mask)};
    assert(all(sum([masks{:}],2)==1));
    for j=1:4
        rr=reshape((3*(find(masks{j})-1)+(1:3))',[],1);B=F(rr,:);Kparts{j}=B'*B;
    end
    clear F B rr;
    total=Kparts{1}+Kparts{2}+Kparts{3}+Kparts{4};
    partition_error=norm(total-S.Kall,'fro')/norm(S.Kall,'fro');assert(partition_error<1e-12);
    refidx=find(S.names=="reference");lung_error=norm(Kparts{2}-S.Klung{refidx},'fro')/norm(Kparts{2},'fro');assert(lung_error<1e-12);
    save(prepared,'Kparts','identity','-v7.3');
    local_json(fullfile(out,'preparation_qc.json'),struct('identity',identity,'partition_relative_error',partition_error,...
        'lung_matrix_relative_error',lung_error,'same_mesh_and_contacts',true,'global_spatial_convergence',false));
end
T=load(prepared);assert(isequal(T.identity,identity));S.Kparts=T.Kparts;
if strcmpi(p.Results.Stage,'prepare'),report=identity;return;end
assert(strcmpi(p.Results.Stage,'solve'));
Q=readtable(fullfile(out,'requests.csv'),'TextType','string');target=fullfile(out,'responses.csv');rows=struct([]);
if isfile(target)
    assert(isequal(jsondecode(fileread([target '.identity.json'])),jsondecode(jsonencode(identity))));
    rows=table2struct(readtable(target,'TextType','string'));
else,local_json([target '.identity.json'],identity);end
for qi=1:height(Q)
    if ~isempty(rows)
        hit=string({rows.request_id})==Q.request_id(qi);
        if nnz(hit)==numel(S.sizes),assert(isequal(sort([rows(hit).L_mm])',S.sizes));continue;end
        assert(~any(hit),'Partial request must be reviewed.');
    end
    r=[Q.rho1(qi),Q.rho2(qi),Q.rho_bone(qi),Q.rho_heart(qi)];batch=struct([]);t=tic;
    for k=1:numel(S.sizes)
        [z,J,res,rec]=local_forward(S,k,r,Q.model(qi));
        row=struct('request_id',Q.request_id(qi),'kind',Q.kind(qi),'frequency_hz',Q.frequency_hz(qi),...
            'state',Q.state(qi),'model',Q.model(qi),'rho1',r(1),'rho2',r(2),'rho_bone',r(3),'rho_heart',r(4),...
            'L_mm',S.sizes(k),'Z',z,'dZ_dlogrho1',J(1),'dZ_dlogrho2',J(2),...
            'dZ_dlogrho_bone',J(3),'dZ_dlogrho_heart',J(4),'relative_residual',res,'reciprocity_abs',rec);
        if isempty(batch),batch=row;else,batch(end+1)=row;end
    end
    if isempty(rows),rows=batch(:);else,rows=[rows(:);batch(:)];end
    writetable(struct2table(rows),target);
    fprintf('TISSUES %s %s %.2fs\n',Q.request_id(qi),Q.model(qi),toc(t));
end
% Central log-parameter differences for every material, at a real four-material point.
qpath=fullfile(out,'derivative_qc.json');
if ~isfile(qpath)
    q=find(Q.kind=="literature" & Q.model=="bones_heart",1);r=[Q.rho1(q),Q.rho2(q),Q.rho_bone(q),Q.rho_heart(q)];
    [~,J]=local_forward(S,numel(S.sizes),r,"bones_heart");fd=zeros(1,4);h=1e-3;
    for j=1:4
        rp=r;rm=r;rp(j)=rp(j)*exp(h);rm(j)=rm(j)*exp(-h);
        fd(j)=(local_forward(S,numel(S.sizes),rp,"bones_heart")-local_forward(S,numel(S.sizes),rm,"bones_heart"))/(2*h);
    end
    err=abs(fd-J)./max(abs(J),1e-7);assert(max(err)<2e-3,'Derivative check failed.');
    local_json(qpath,struct('adjoint',J,'central_difference',fd,'relative_errors',err,'log_step',h,'L_mm',S.sizes(end)));
end
report=struct('complete',true,'identity',identity,'rows',numel(rows),'responses_sha256',trkg4_file_sha256(target),...
    'status','exploratory_hypothesis_not_validated','physical_validation',false,'global_spatial_convergence',false);
local_json(fullfile(out,'execution.json'),report);
end

function [z,J,res,rec]=local_forward(S,k,r,model)
Kb=S.Kparts{3};Kh=S.Kparts{4};Kl=S.Kparts{2};Ks=S.Kparts{1};
if model=="two_tissue",Ks=Ks+Kb+Kh;Kb=sparse(size(Ks,1),size(Ks,2));Kh=Kb;
elseif model=="bones",Ks=Ks+Kh;Kh=sparse(size(Ks,1),size(Ks,2));
else,assert(model=="bones_heart");end
A={Ks,Kl,Kb,Kh};K=Ks/r(1)+Kl/r(2)+Kb/r(3)+Kh/r(4);n=size(K,1);
H=S.C{k};H(1:n,1:n)=H(1:n,1:n)+K;free=[S.free(:);n+(1:4)'];
rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];
B=H(free,free);assert(norm(B-B','fro')/norm(B,'fro')<1e-12);U=zeros(n+4,2);
U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
z=rhs(:,2)'*U(:,1);rec=abs(z-rhs(:,1)'*U(:,2));assert(res<1e-7&&rec<1e-7&&isfinite(z));
J=zeros(1,4);for j=1:4,J(j)=U(1:n,2)'*(A{j}*U(1:n,1))/r(j);end
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
