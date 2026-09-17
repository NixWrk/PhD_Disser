function run_heart_fractional_electrical_pilot(plan_file)
% New weighted P1 material blocks; original mesh and CEM geometry unchanged.
% No old material blocks are loaded; one geometric F shared across states.
plan=jsondecode(fileread(plan_file));base=fileparts(plan_file);
assert(strcmp(plan.schema,'heart_fractional_electrical_pilot_v1'));
maxNumCompThreads(plan.threads);assert(isempty(gcp('nocreate')),'No parpool');
local_hash(plan.source_prepared,plan.source_prepared_sha256);
local_hash(fullfile(base,'background.mat'),plan.background_mat_sha256);
for j=1:numel(plan.implementation),v=plan.implementation(j);local_hash(v.path,v.sha256);end
frozen=load(plan.source_prepared,'fmdl_m','tissue_id','coordinate_units');
assert(strcmp(frozen.coordinate_units,'m'));
fmdl=frozen.fmdl_m;ne=size(fmdl.elems,1);nn=size(fmdl.nodes,1);
b=load(fullfile(base,'background.mat'),'background');background=double(b.background(:));
expected=double(frozen.tissue_id(:));expected(expected==3)=1;
assert(isequal(background,expected));clear expected frozen b;
sigma=double(plan.conductivity_by_tissue_id(:));
assert(ne==plan.element_count&&nn==plan.node_count);
fprintf('Building fresh EIDORS geometry fields once (%d tetrahedra).\n',ne);
F=system_mat_fields(fmdl);F=F(1:3*ne,:);
assert(size(F,2)==nn+4&&isreal(F)&&all(isfinite(nonzeros(F))));
C=trkg4_cem_contact_matrix(fmdl);
geometry_volume=F'*F;
for j=1:numel(plan.states)
    state=plan.states(j);out=fullfile(base,state.id);
    if isfile(fullfile(out,'export.json'))
        prior=jsondecode(fileread(fullfile(out,'export.json')));
        assert(strcmp(prior.plan_sha256,trkg4_file_sha256(plan_file)));
        local_hash(fullfile(out,'blocks.mat'),prior.blocks_sha256);continue
    end
    assert(~isfolder(out),'Incomplete export must be inspected, not overwritten');
    local_hash(fullfile(base,state.fraction_mat),state.fraction_mat_sha256);
    data=load(fullfile(base,state.fraction_mat),'heart_fraction');h=double(data.heart_fraction(:));
    assert(numel(h)==ne&&all(isfinite(h))&&all(h>=0&h<=1));
    fprintf('Weighted blocks %d/%d: %s\n',j,numel(plan.states),state.id);tic;
    blocks=cell(4,1);sigma_eff=zeros(ne,1);sum_weights=zeros(ne,1);
    for k=1:4
        if k==3,w=h;else,w=(1-h).*double(background==k);end
        sum_weights=sum_weights+w;sigma_eff=sigma_eff+sigma(k)*w;
        ids=find(w>0);rows=reshape((3*(ids-1)+(1:3)).',[],1);
        fw=spdiags(repelem(sqrt(w(ids)),3),0,numel(rows),numel(rows))*F(rows,:);
        blocks{k}=fw'*fw;
    end
    clear fw w ids rows;
    assert(max(abs(sum_weights-1))<1e-14);
    homogeneous=sparse(nn+4,nn+4);K=C;
    for k=1:4,homogeneous=homogeneous+blocks{k};K=K+sigma(k)*blocks{k};end
    homogeneous_error=norm(homogeneous-geometry_volume,'fro')/norm(geometry_volume,'fro');
    img=mk_image(fmdl,1);img.elem_data=sigma_eff;ref=calc_system_mat(img);
    heterogeneous_error=norm(K-ref.E,'fro')/norm(ref.E,'fro');
    assert(isfinite(homogeneous_error)&&homogeneous_error<1e-12);
    assert(isfinite(heterogeneous_error)&&heterogeneous_error<1e-12);
    mkdir(out);coordinate_units='m';ground=fmdl.gnd_node; %#ok<NASGU>
    source_prepared_sha256=plan.source_prepared_sha256; %#ok<NASGU>
    tissue_names={'soft_tissue','lungs','heart','bones'}; %#ok<NASGU>
    save(fullfile(out,'blocks.mat'),'blocks','coordinate_units','ground','nn', ...
         'source_prepared_sha256','tissue_names','-v7.3');
    report=struct('status','weighted_P1_assembly_checks_passed','state_id',state.id, ...
        'plan_sha256',trkg4_file_sha256(plan_file),'source_prepared_sha256',plan.source_prepared_sha256, ...
        'fraction_mat_sha256',state.fraction_mat_sha256,'blocks_sha256',trkg4_file_sha256(fullfile(out,'blocks.mat')), ...
        'homogeneous_error',homogeneous_error,'heterogeneous_error',heterogeneous_error, ...
        'seconds',toc,'matlab_version',version,'fractional_materials',true, ...
        'old_material_blocks_loaded',false,'spatial_convergence',false, ...
        'system_mat_fields_sha256',trkg4_file_sha256(which('system_mat_fields')), ...
        'calc_system_mat_sha256',trkg4_file_sha256(which('calc_system_mat')));
    local_hash(fullfile(base,state.fraction_mat),state.fraction_mat_sha256);
    fid=fopen(fullfile(out,'export.json.tmp'),'w','n','UTF-8');assert(fid>=0);
    fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));fclose(fid);
    movefile(fullfile(out,'export.json.tmp'),fullfile(out,'export.json'));
    clear blocks K homogeneous ref img data h sum_weights sigma_eff;
end
local_hash(plan.source_prepared,plan.source_prepared_sha256);
for j=1:numel(plan.implementation),v=plan.implementation(j);local_hash(v.path,v.sha256);end
fprintf('All weighted exports complete.\n');
end
function local_hash(path,expected)
assert(strcmp(trkg4_file_sha256(path),expected),'Input/code hash changed: %s',path);
end
