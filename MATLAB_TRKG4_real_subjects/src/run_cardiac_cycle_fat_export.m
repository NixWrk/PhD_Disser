function run_cardiac_cycle_fat_export(plan_file)
% Five fresh EIDORS P1 blocks, with pointwise heart/fat partition from CT.
plan=jsondecode(fileread(plan_file));base=fileparts(plan_file);
assert(strcmp(plan.schema,'cardiac_cycle_fat_electrical_v1'));
maxNumCompThreads(plan.threads);assert(isempty(gcp('nocreate')));
local_hash(plan.source_prepared,plan.source_prepared_sha256);
for j=1:numel(plan.implementation),v=plan.implementation(j);local_hash(v.path,v.sha256);end
frozen=load(plan.source_prepared,'fmdl_m','tissue_id','coordinate_units');
assert(strcmp(frozen.coordinate_units,'m'));fmdl=frozen.fmdl_m;
nn=size(fmdl.nodes,1);ne=size(fmdl.elems,1);
local_hash(plan.background_path,plan.background_sha256);
b=load(plan.background_path,'background');bg=double(b.background(:));
expected=double(frozen.tissue_id(:));expected(expected==3)=1;assert(isequal(bg,expected));
clear b expected frozen;
F=system_mat_fields(fmdl);F=F(1:3*ne,:);volume=F'*F;C=trkg4_cem_contact_matrix(fmdl);
sigma=double(plan.scenarios(1).conductivity_by_material(:));
for j=1:numel(plan.states)
    s=plan.states(j);dest=fullfile(base,s.id);
    if isfile(fullfile(dest,'export.json'))
        prev=jsondecode(fileread(fullfile(dest,'export.json')));
        assert(strcmp(prev.plan_sha256,trkg4_file_sha256(plan_file)));
        local_hash(fullfile(dest,'blocks.mat'),prev.blocks_sha256);continue
    end
    assert(~isfolder(dest),'Incomplete export requires explicit inspection');
    local_hash(s.fractions_path,s.fraction_mat_sha256);
    a=load(s.fractions_path);h=double(a.heart_fraction(:));fat=double(a.fat_fraction(:));
    assert(numel(h)==ne&&all(isfinite(h))&&all(h>=0&h<=1));
    assert(numel(fat)==ne&&all(isfinite(fat))&&all(fat>=0&fat<=1-h));
    assert(all(fat(bg~=1)==0));
    fprintf('Five-material export %d/%d %s\n',j,numel(plan.states),s.id);tic;
    blocks=cell(5,1);weights=zeros(ne,1);sigma_eff=zeros(ne,1);
    for k=1:5
        if k==1,w=(1-h).*double(bg==1)-fat;
        elseif k==3,w=h;
        elseif k==5,w=fat;
        else,w=(1-h).*double(bg==k);end
        assert(all(w>=0));weights=weights+w;sigma_eff=sigma_eff+sigma(k)*w;
        ids=find(w>0);rows=reshape((3*(ids-1)+(1:3)).',[],1);
        fw=spdiags(repelem(sqrt(w(ids)),3),0,numel(rows),numel(rows))*F(rows,:);
        blocks{k}=fw'*fw;
    end
    clear fw w ids rows;assert(max(abs(weights-1))<1e-14);
    K=C;sumB=sparse(nn+4,nn+4);
    for k=1:5,K=K+sigma(k)*blocks{k};sumB=sumB+blocks{k};end
    homogeneous_error=norm(sumB-volume,'fro')/norm(volume,'fro');
    img=mk_image(fmdl,1);img.elem_data=sigma_eff;ref=calc_system_mat(img);
    heterogeneous_error=norm(K-ref.E,'fro')/norm(ref.E,'fro');
    % Equal fat and background conductivity must recover the sealed old blocks.
    local_hash(s.old_blocks_path,s.old_blocks_sha256);
    old=load(s.old_blocks_path,'blocks');fat_equal_error=zeros(1,4);
    for k=1:4
        candidate=blocks{k};if k==1,candidate=candidate+blocks{5};end
        fat_equal_error(k)=norm(candidate-old.blocks{k},'fro')/max(norm(old.blocks{k},'fro'),eps);
    end
    assert(homogeneous_error<1e-12&&heterogeneous_error<1e-12&&max(fat_equal_error)<1e-12);
    clear old candidate K sumB ref img;
    mkdir(dest);ground=fmdl.gnd_node;coordinate_units='m'; %#ok<NASGU>
    tissue_names={'muscle_background','lungs','heart_region','bones','fat'}; %#ok<NASGU>
    save(fullfile(dest,'blocks.mat'),'blocks','nn','ground','coordinate_units','tissue_names','-v7.3');
    report=struct('status','passed','plan_sha256',trkg4_file_sha256(plan_file), ...
        'blocks_sha256',trkg4_file_sha256(fullfile(dest,'blocks.mat')), ...
        'homogeneous_error',homogeneous_error,'heterogeneous_error',heterogeneous_error, ...
        'fat_equal_old_blocks_relative_errors',fat_equal_error,'seconds',toc, ...
        'phase',s.id,'fraction_mat_sha256',s.fraction_mat_sha256, ...
        'source_prepared_sha256',plan.source_prepared_sha256, ...
        'physical_validation',false,'spatial_convergence',false, ...
        'matlab_version',version,'system_mat_fields_sha256',trkg4_file_sha256(which('system_mat_fields')), ...
        'calc_system_mat_sha256',trkg4_file_sha256(which('calc_system_mat')));
    fid=fopen(fullfile(dest,'export.json'),'w','n','UTF-8');assert(fid>=0);
    fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));fclose(fid);
    clear blocks a h fat weights sigma_eff;
end
for j=1:numel(plan.implementation),v=plan.implementation(j);local_hash(v.path,v.sha256);end
end
function local_hash(path,expected)
assert(strcmp(trkg4_file_sha256(path),expected),'Changed input/code: %s',path);
end
