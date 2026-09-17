function report=run_lateral_fat_forward(varargin)
% Fixed C01, finite 5 mm CEM, five full-static TotalSegmentator materials.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_fat_20260916');parse(p,varargin{:});
cfg=trkg4_config('nik');root=cfg.project_root;out=fullfile(root,char(p.Results.OutputDir));
P=jsondecode(fileread(fullfile(out,'plan.json')));trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
for i=1:numel(P.sources)
    assert(strcmp(trkg4_file_sha256(fullfile(root,P.sources(i).path)),P.sources(i).sha256),'Frozen source changed.');
end
assert(strcmp(trkg4_file_sha256(fullfile(out,'requests.csv')),P.requests_sha256));
identity=struct('plan_sha256',trkg4_file_sha256(fullfile(out,'plan.json')),...
    'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
base=fullfile(root,'output/exploratory/lateral_array_design_20260916');S=load(fullfile(base,'prepared_fem.mat'));
assert(isequal(S.sizes(:),P.sizes_mm(:)));Q=readtable(fullfile(out,'requests.csv'),'TextType','string');
target=fullfile(out,'responses.csv');rows=struct([]);
if isfile(target)
    assert(isequal(jsondecode(fileread([target '.identity.json'])),jsondecode(jsonencode(identity))));
    rows=table2struct(readtable(target,'TextType','string'));
else,local_json([target '.identity.json'],identity);end
for nq=[256 1024]
    fractionfile=fullfile(out,sprintf('fractions_q%d.mat',nq));
    field=sprintf('fractions_q%d_mat',nq);assert(strcmp(trkg4_file_sha256(fractionfile),P.fractions_sha256.(field)));
    prepared=fullfile(out,sprintf('matrices_q%d.mat',nq));
    if ~isfile(prepared)
        M=load(fullfile(base,'refined_mesh.mat'));T=load(fractionfile);
        assert(size(T.fractions,1)==size(M.elems,1) && size(T.fractions,2)==5);
        assert(all(abs(sum(T.fractions,2)-1)<1e-12) && all(T.fractions(:)>=0));
        old=load(fullfile(root,'output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'),'fmdl_mesh_units');mm=old.fmdl_mesh_units;
        mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
        mm.electrode=struct([]);mm.stimulation=struct([]);if isfield(mm,'mat_idx'),mm=rmfield(mm,'mat_idx');end
        si=trkg4_scale_fmdl_to_si(mm,cfg);F=system_mat_fields(si);
        assert(norm(F'*F-S.Kall,'fro')/norm(S.Kall,'fro')<1e-12,'Mesh or matrix units changed.');
        Kparts=cell(5,1);Ksum=sparse(size(S.Kall,1),size(S.Kall,2));nr=size(F,1);
        for j=1:5
            weight=repelem(T.fractions(:,j),3);Kparts{j}=F'*(spdiags(weight,0,nr,nr)*F);Ksum=Ksum+Kparts{j};
        end
        partition_error=norm(Ksum-S.Kall,'fro')/norm(S.Kall,'fro');assert(partition_error<1e-12);
        save(prepared,'Kparts','identity','-v7.3');clear F T M old Ksum weight;
        local_json(fullfile(out,sprintf('matrix_qc_q%d.json',nq)),struct('identity',identity,...
            'partition_relative_error',partition_error,'same_mesh_and_contacts',true,'quadrature_points',nq));
    end
    T=load(prepared);assert(isequal(T.identity,identity));S.Kparts=T.Kparts;clear T;
    for qi=find(Q.quadrature_points==nq)'
        if ~isempty(rows)
            hit=string({rows.request_id})==Q.request_id(qi);
            if nnz(hit)==numel(S.sizes),assert(isequal(sort([rows(hit).L_mm])',S.sizes));continue;end
            assert(~any(hit),'Partial request requires review.');
        end
        r=[Q.rho1(qi),Q.rho2(qi),Q.rho_bone(qi),Q.rho_heart(qi),Q.rho_fat(qi)];batch=struct([]);t=tic;
        for k=1:numel(S.sizes)
            [z,res,rec]=local_forward(S,k,r,Q.model(qi));
            row=struct('request_id',Q.request_id(qi),'kind',Q.kind(qi),'quadrature_points',nq,...
                'frequency_hz',Q.frequency_hz(qi),'state',Q.state(qi),'model',Q.model(qi),...
                'rho1',r(1),'rho2',r(2),'rho_bone',r(3),'rho_heart',r(4),'rho_fat',r(5),...
                'L_mm',S.sizes(k),'Z',z,'relative_residual',res,'reciprocity_abs',rec);
            if isempty(batch),batch=row;else,batch(end+1)=row;end
        end
        rows=[rows(:);batch(:)];writetable(struct2table(rows),target);
        fprintf('FAT %s %.2fs\n',Q.request_id(qi),toc(t));
    end
end
report=struct('complete',true,'identity',identity,'rows',numel(rows),'responses_sha256',trkg4_file_sha256(target),...
    'no_new_pose_search',true,'no_inverse_fit',true,'global_spatial_convergence',false,'physical_validation',false);
local_json(fullfile(out,'execution.json'),report);
end

function [z,res,rec]=local_forward(S,k,r,model)
A=S.Kparts;
if model=="ts_without_fat",A{1}=A{1}+A{5};A{5}=sparse(size(A{1},1),size(A{1},2));
elseif model=="homogeneous",A={S.Kall};r=r(1);
else,assert(model=="ts_with_fat");end
K=sparse(size(S.Kall,1),size(S.Kall,2));for j=1:numel(A),K=K+A{j}/r(j);end;n=size(K,1);
H=S.C{k};H(1:n,1:n)=H(1:n,1:n)+K;free=[S.free(:);n+(1:4)'];
rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];
B=H(free,free);assert(norm(B-B','fro')/norm(B,'fro')<1e-12);U=zeros(n+4,2);
U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');z=rhs(:,2)'*U(:,1);
rec=abs(z-rhs(:,1)'*U(:,2));assert(res<1e-7&&rec<1e-7&&isfinite(z));
end

function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
