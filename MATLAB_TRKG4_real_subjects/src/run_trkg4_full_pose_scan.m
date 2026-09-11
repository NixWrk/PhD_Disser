function report = run_trkg4_full_pose_scan(varargin)
%RUN_TRKG4_FULL_POSE_SCAN Exhaustive fixed-material CEM centre/angle scan.
% Every requested (u,v,phi) is retained. Geometry failures carry no impedance
% or numerical error score. Checkpoints are bound to inputs and source bytes.
p=inputParser;
addParameter(p,'Profile','full_pose_scan');
addParameter(p,'ResultTag','fullscan_20260909_v3');
addParameter(p,'SourceTag','auditfix_20260908');
addParameter(p,'Stage','all');
addParameter(p,'Workers',8);
addParameter(p,'CentreChunk',64);
addParameter(p,'GreenNodeLimit',10000);
addParameter(p,'UValues',-80:80);
addParameter(p,'VValues',-120:120);
addParameter(p,'PhiValues',-20:20);
addParameter(p,'Streaming',true);
addParameter(p,'StreamGroupBlocks',1);
addParameter(p,'MaxBlocks',Inf);
addParameter(p,'ReuseGeometryTag','fullscan_20260908_v2');
addParameter(p,'ReportPython','python');
addParameter(p,'WriteLiveReport',true);
addParameter(p,'PatchCacheTileMM',8);
addParameter(p,'PatchCacheLimit',64);
parse(p,varargin{:});opt=p.Results;
assert(~opt.Streaming||any(strcmp(opt.Stage,{'all','finalize'})), ...
    'full_scan:stage','Streaming supports Stage all or finalize; use Streaming=false for the legacy stage API');
validateattributes(opt.StreamGroupBlocks,{'numeric'},{'scalar','integer','positive'});
assert(isscalar(opt.MaxBlocks)&&(isinf(opt.MaxBlocks)&&opt.MaxBlocks>0 || ...
    isfinite(opt.MaxBlocks)&&opt.MaxBlocks>0&&opt.MaxBlocks==fix(opt.MaxBlocks)), 'Invalid MaxBlocks');
assert(isempty(opt.ReuseGeometryTag)||~isempty(regexp(char(opt.ReuseGeometryTag),'^[A-Za-z0-9_-]+$','once')),'Invalid reuse tag');
for tag={opt.ResultTag,opt.SourceTag}
    assert(~isempty(regexp(char(tag{1}),'^[A-Za-z0-9_-]+$','once')),'Invalid result tag');
end
assert(any(strcmp(opt.Stage,{'all','geometry','solve','finalize'})),'Invalid stage');
for values={opt.UValues,opt.VValues,opt.PhiValues}
    validateattributes(values{1},{'numeric'},{'vector','real','finite','nonempty'});
    assert(all(diff(values{1})>0),'Grid coordinates must increase strictly');
end
validateattributes(opt.Workers,{'numeric'},{'scalar','integer','positive'});
validateattributes(opt.CentreChunk,{'numeric'},{'scalar','integer','positive'});
validateattributes(opt.GreenNodeLimit,{'numeric'},{'scalar','integer','positive'});
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'tools','full_pose_scan'),fullfile(root,'tools'));
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
% Radius is a certified broad-phase acceleration, never a contact/QC limit.
cfg.local_patch_options=struct('initial_radius',6*sqrt(cfg.electrode_area/pi),'max_local_attempts',3);
if opt.Streaming
    cfg.local_patch_options.cache_tile_mm=opt.PatchCacheTileMM;
    cfg.local_patch_options.cache_limit=opt.PatchCacheLimit;
end
out=fullfile(root,'output',['full_pose_scan_' char(opt.ResultTag)]);
if ~isfolder(out),mkdir(out);end
source_contract_path=fullfile(root,'output',['nik_trkg4_inverse_inhale_contract_' char(opt.SourceTag) '.json']);
source_contract=jsondecode(fileread(source_contract_path));
assert(strcmp(source_contract.result_tag,opt.SourceTag) && source_contract.geometry_qc_passed);
% Validate the historical result against its own recorded source/input set.
% New scan entrypoints do not retroactively become inputs of that old run.
entries=[source_contract.inputs(:);source_contract.artifacts(:)];
% These arm-only entrypoints are not dependencies of the side-array scan.
% Their unrelated concurrent changes must not be reverted or imported here.
excluded_arm_sources={'src/run_trkg4_arm_electrode_tests.m','src/trkg4_arm_montage_spec.m'};
entries=entries(~ismember({entries.path},excluded_arm_sources));
% A separately verified shape-only correction is an explicit source transition.
% Preserve the old contract; bind this scan to the new builder and the proof
% that all nine published best contact node/face sets remain identical.
patch_shape_transition=struct();
patch_index=find(strcmp({entries.path},'vendor_stl_eidors/electrode_faces_by_area.m'));
assert(isscalar(patch_index),'Historical builder input is missing');
new_patch_hash=trkg4_file_sha256(fullfile(root,entries(patch_index).path));
if ~strcmp(new_patch_hash,entries(patch_index).sha256)
    relative_check='notebooks/full_scan_2026_09_08/patch_shape_compatibility.json';
    compatibility_path=fullfile(root,relative_check);
    compatibility=jsondecode(fileread(compatibility_path));
    assert(strcmp(compatibility.schema,'canonical_patch_shape_compatibility_v1') && ...
        compatibility.passed && compatibility.published_contacts_identical && ...
        strcmp(compatibility.historical_result_tag,opt.SourceTag) && ...
        strcmp(compatibility.old_builder_sha256,entries(patch_index).sha256) && ...
        strcmp(compatibility.new_builder_sha256,new_patch_hash), ...
        'Changed canonical builder requires verified shape compatibility');
    patch_shape_transition=struct('old_sha256',entries(patch_index).sha256, ...
        'new_sha256',new_patch_hash,'compatibility_path',relative_check, ...
        'compatibility_sha256',trkg4_file_sha256(compatibility_path));
    entries(patch_index).sha256=new_patch_hash;
    entries(end+1)=struct('name','patch_shape_compatibility.json','path',relative_check, ...
        'sha256',patch_shape_transition.compatibility_sha256);
    for dependency=reshape(compatibility.dependencies,1,[])
        assert(strcmp(trkg4_file_sha256(fullfile(root,dependency.path)),dependency.sha256), ...
            'Shape compatibility evidence changed');
        if ~any(strcmp({entries.path},dependency.path))
            [~,name,ext]=fileparts(dependency.path);
            entries(end+1)=struct('name',[name ext],'path',dependency.path,'sha256',dependency.sha256); %#ok<AGROW>
        end
    end
end
for k=1:numel(entries)
    assert(strcmp(trkg4_file_sha256(fullfile(root,entries(k).path)),entries(k).sha256), ...
        ['Changed historical input/artifact: ' entries(k).path]);
end
fit=readtable(fullfile(root,'output',['nik_trkg4_inverse_inhale_fit_summary_' char(opt.SourceTag) '.csv']));
fixed_rho=[fit.rho_soft_ohm_m(1),fit.rho_lungs_ohm_m(1)];
source_files=[dir(fullfile(root,'tools','full_pose_scan','full_scan*.m')); ...
    dir(fullfile(root,'tools','full_pose_scan','full_scan_project_mex.cpp')); ...
    dir(fullfile(root,'tools','full_pose_scan',['full_scan_project_mex.' mexext])); ...
    dir(fullfile(root,'tools','trkg4_local*.m')); ...
    dir(mfilename('fullpath') + ".m");dir(fullfile(root,'run_pipeline.m')); ...
    dir(fullfile(root,'notebooks','build_trkg4_full_scan_report.py'))];
producer_sources=struct('path',{},'sha256',{});
for k=1:numel(source_files)
    name=fullfile(source_files(k).folder,source_files(k).name);
    producer_sources(k)=struct('path',strrep(name,[root filesep],''),'sha256',trkg4_file_sha256(name));
end
physics=struct('schema','trkg4_full_pose_scan_v1','result_tag',opt.ResultTag, ...
    'source_tag',opt.SourceTag,'source_contract_sha256',trkg4_file_sha256(source_contract_path), ...
    'rho_ohm_m',fixed_rho,'u_mm',opt.UValues,'v_mm',opt.VValues,'phi_deg',opt.PhiValues, ...
    'centre_projection','continuous_triangle_same_16_node_neighbourhood_as_inverse', ...
    'contact_model','CEM_canonical_disjoint_induced_faces', ...
    'ranking','RMSE_ohm_equal_weights_all_nine_experimental_sizes_no_priors', ...
    'geometry_thresholds',source_contract.geometry_qc,'producer_sources',producer_sources, ...
    'runtime',source_contract.runtime,'matlab_version',version,'centre_chunk',opt.CentreChunk,'green_node_limit',opt.GreenNodeLimit, ...
    'verified_source_inputs_and_artifacts',entries,'excluded_unrelated_arm_sources',{excluded_arm_sources}, ...
    'canonical_patch_shape_transition',patch_shape_transition);
physics.acceleration=struct('streaming',logical(opt.Streaming),'patch_cache_tile_mm',opt.PatchCacheTileMM, ...
    'patch_cache_limit',opt.PatchCacheLimit,'stream_group_blocks',opt.StreamGroupBlocks, ...
    'reuse_geometry_tag',opt.ReuseGeometryTag,'search_domain','recorded_artificial_arms_and_extreme_z_caps_excluded');
if opt.Streaming
    physics.acceleration.anatomy_preparation_sha256=trkg4_file_sha256(fullfile(root,'output','nik_stl_preparation_report.json'));
    if ~isempty(opt.ReuseGeometryTag)
        physics.acceleration.reused_manifest_sha256=trkg4_file_sha256(fullfile(root,'output', ...
            ['full_pose_scan_' char(opt.ReuseGeometryTag)],'run_manifest.json'));
    end
end
fingerprint=local_text_sha(jsonencode(physics));
manifest_path=fullfile(out,'run_manifest.json');
if isfile(manifest_path)
    previous=jsondecode(fileread(manifest_path));
    assert(strcmp(previous.fingerprint,fingerprint),'Full scan inputs/code changed: use a new result tag');
else
    local_json(manifest_path,struct('fingerprint',fingerprint,'physics',physics));
end
S=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
f=S.fmdl_mesh_units;
[sizes,measured]=trkg4_load_nik_inhale_data(root,'manifest');
si=trkg4_scale_fmdl_to_si(f,cfg);
sigma=S.elem_sigma;
sigma(S.tissue_id==find(strcmp(S.tissue_names,'soft_tissue'),1))=1/fixed_rho(1);
sigma(S.tissue_id==find(strcmp(S.tissue_names,'lungs'),1))=1/fixed_rho(2);
context=full_scan_context(f);
ref=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
ref_xyz=[ref.x_mm ref.y_mm ref.z_mm];ref_axis=ref_xyz(4,:)-ref_xyz(1,:);ref_axis=ref_axis/norm(ref_axis);
[c0,n0]=full_scan_project(context,mean(ref_xyz,1));
u0=ref_axis-dot(ref_axis,n0)*n0;u0=u0/norm(u0);v0=cross(n0,u0);v0=v0/norm(v0);
[gu,gv]=ndgrid(opt.UValues,opt.VValues);
[P,N,F]=full_scan_project(context,c0+gu(:)*u0+gv(:)*v0);
lung=stlread(cfg.subject.stl.lungs);[~,h]=knnsearch(lung.Points,P);
nc=numel(gu);na=numel(opt.PhiValues);np=nc*na;nb=ceil(nc/opt.CentreChunk);
centres=table((1:nc)',gu(:),gv(:),P(:,1),P(:,2),P(:,3),h, ...
    'VariableNames',{'centre_index','u_mm','v_mm','x_mm','y_mm','z_mm','h_vertex_mm'});
centres_path=fullfile(out,'centres.csv');
if ~isfile(centres_path),writetable(centres,centres_path);else
    old=readtable(centres_path);assert(isequal(size(old),size(centres)) && max(abs(old{:,:}-centres{:,:}),[],'all')<1e-8);
end
surface_path=fullfile(out,'surface.mat');
if ~isfile(surface_path)
    boundary_ids=unique(f.boundary(:));nodes_mm=f.nodes(boundary_ids,:);[~,boundary_faces]=ismember(f.boundary,boundary_ids);
    save(surface_path,'nodes_mm','boundary_faces','-v7');
end
grid=struct('u',opt.UValues,'v',opt.VValues,'phi',opt.PhiValues);
fprintf('FULL CEM SCAN: %d centres x %d rotations = %d poses; rho %.9g, %.9g Ohm m.\n',nc,na,np,fixed_rho);
if opt.Workers>1 && ~strcmp(opt.Stage,'finalize')
    pool=gcp('nocreate');
    if isempty(pool),parpool('Threads',opt.Workers);end
end
parfor_limit=opt.Workers;if opt.Workers==1,parfor_limit=0;end
if opt.Streaming && ~strcmp(opt.Stage,'finalize')
    data=struct('root',root,'out',out,'f',f,'si',si,'sigma',sigma,'cfg',cfg,'opt',opt, ...
        'physics',physics,'fingerprint',fingerprint,'projection',context,'P',P,'N',N,'u0',u0, ...
        'sizes',sizes,'measured',measured,'centres',centres,'grid',grid);
    report=full_scan_stream(data);
    if ~report.complete,return;end
    opt.Stage='finalize';
end
clock_start=tic;total_valid=0;all_candidate_nodes=false(size(f.nodes,1),1);
do_geometry=any(strcmp(opt.Stage,{'all','geometry'}));
if do_geometry
    patch_context=full_scan_patch_context(f);
end
geo_dir=fullfile(out,'geometry');if ~isfolder(geo_dir),mkdir(geo_dir);end
solve_dir=fullfile(out,'solved');if ~isfolder(solve_dir),mkdir(solve_dir);end
for block_index=1:nb
    ids=(block_index-1)*opt.CentreChunk+1:min(block_index*opt.CentreChunk,nc);
    path=fullfile(geo_dir,sprintf('block_%05d.mat',block_index));
    if local_checkpoint_valid(path,fingerprint)
        block=load(path,'candidate_nodes','valid_rows');
    else
        assert(do_geometry,'Missing geometry checkpoint: run geometry stage');
        centre_ids=reshape(repelem(ids(:),na),[],1);phi_deg=repmat(opt.PhiValues(:),numel(ids),1);
        centre_points=P(centre_ids,:);centre_normals=N(centre_ids,:);
        base=u0-sum(u0.*centre_normals,2).*centre_normals;base=base./vecnorm(base,2,2);
        side=cross(centre_normals,base,2);side=side./vecnorm(side,2,2);
        axis=cos(phi_deg*pi/180).*base+sin(phi_deg*pi/180).*side;axis=axis./vecnorm(axis,2,2);
        offsets=[-sizes(:)'/2;-sizes(:)'/4;sizes(:)'/4;sizes(:)'/2];
        nrows=numel(centre_ids);raw=zeros(nrows*numel(offsets),3);
        for j=1:numel(offsets),rr=(j-1)*nrows+(1:nrows);raw(rr,:)=centre_points+offsets(j)*axis;end
        % MEX runs once on the client; MATLAB thread workers handle only the
        % exact patch builder/QC. MATLAB prohibits MEX on thread workers.
        [projected,~,projection_faces]=full_scan_project(context,raw);
        projected=reshape(projected,nrows,4,numel(sizes),3);
        projection_faces=reshape(projection_faces,nrows,4,numel(sizes));
        status_code=zeros(nrows,1,'uint8');failure_code=repmat({''},nrows,1);patch_parts=cell(nrows,1);
        parfor (k=1:nrows,parfor_limit)
            [status_code(k),failure_code{k},patch_parts{k}]=local_pose_geometry( ...
                f,patch_context,cfg,reshape(projected(k,:,:,:),4,numel(sizes),3), ...
                reshape(projection_faces(k,:,:),4,numel(sizes)),sizes);
        end
        patch_faces=cat(1,patch_parts{:});valid_rows=find(status_code==1);
        patch_faces=patch_faces(valid_rows,:,:);
        face_ids=unique(vertcat(patch_faces{:}));
        candidate_nodes=unique(reshape(f.boundary(face_ids,:),[],1));
        centre_ids=reshape(repelem(ids(:),na),[],1);phi_deg=repmat(opt.PhiValues(:),numel(ids),1);
        block=struct('centre_ids',centre_ids,'phi_deg',phi_deg,'status_code',status_code, ...
            'failure_code',{failure_code},'valid_rows',valid_rows,'patch_faces',{patch_faces}, ...
            'candidate_nodes',candidate_nodes,'fingerprint',fingerprint);
        local_checkpoint(path,block,fingerprint);
    end
    all_candidate_nodes(block.candidate_nodes)=true;total_valid=total_valid+numel(block.valid_rows);
    if do_geometry
        local_json(fullfile(out,'progress.json'),struct('stage','geometry','complete',false, ...
            'centres_done',ids(end),'centres_total',nc,'poses_checked',ids(end)*na, ...
            'poses_total',np,'geometry_valid',total_valid,'elapsed_seconds',toc(clock_start)));
        fprintf('Geometry %d/%d centres; %d/%d poses; valid %d; %.1f s\n',ids(end),nc,ids(end)*na,np,total_valid,toc(clock_start));
    end
end
if strcmp(opt.Stage,'geometry')
    report=struct('stage','geometry_complete','centres',nc,'poses',np,'geometry_valid',total_valid,'out_dir',out);
    local_json(fullfile(out,'geometry_summary.json'),report);return;
end
if any(strcmp(opt.Stage,{'all','solve'}))
    % Partition adjacent geometry blocks by their EXACT union of contact
    % nodes. This caps each dense Green matrix without dropping any pose or
    % replacing CEM with a point-electrode approximation.
    groups={};group_blocks=[];group_nodes=[];
    for bi=1:nb
        item=load(fullfile(geo_dir,sprintf('block_%05d.mat',bi)),'candidate_nodes');
        proposed=union(group_nodes,item.candidate_nodes);
        if numel(proposed)>opt.GreenNodeLimit && ~isempty(group_blocks)
            groups{end+1}=struct('blocks',group_blocks,'nodes',group_nodes); %#ok<AGROW>
            group_blocks=[];group_nodes=[];proposed=item.candidate_nodes;
        end
        if numel(proposed)>opt.GreenNodeLimit
            % An unusually broad block is solved in smaller exact unions of
            % whole poses. Its geometry checkpoint and requested grid stay intact.
            groups{end+1}=struct('blocks',bi,'nodes',[],'split',true); %#ok<AGROW>
            group_blocks=[];group_nodes=[];continue;
        end
        group_blocks(end+1)=bi;group_nodes=proposed; %#ok<AGROW>
    end
    if ~isempty(group_blocks),groups{end+1}=struct('blocks',group_blocks,'nodes',group_nodes);end
    group_meta=cell(numel(groups),1);green_dir=fullfile(out,'green');if ~isfolder(green_dir),mkdir(green_dir);end
    solved_valid=0;
    for group_index=1:numel(groups)
        group=groups{group_index};candidate_nodes=group.nodes;
        % Completed groups do not require reading or rebuilding their Green.
        done=true;
        for bi=group.blocks,done=done && local_checkpoint_valid(fullfile(solve_dir,sprintf('block_%05d.mat',bi)),fingerprint);end
        if done
            for bi=group.blocks,item=load(fullfile(solve_dir,sprintf('block_%05d.mat',bi)),'valid_count');solved_valid=solved_valid+item.valid_count;end
            group_meta{group_index}=struct('status','solved_checkpoints_reused','blocks',group.blocks);continue;
        end
        if isfield(group,'split') && group.split
            bi=group.blocks;b=load(fullfile(geo_dir,sprintf('block_%05d.mat',bi)));
            [solved,split_meta]=local_split_cem_block(b,si,sigma,cfg,measured,fingerprint, ...
                opt.GreenNodeLimit,parfor_limit,green_dir,group_index);
            local_checkpoint(fullfile(solve_dir,sprintf('block_%05d.mat',bi)),solved,fingerprint);
            solved_valid=solved_valid+solved.valid_count;group_meta{group_index}=split_meta;
            local_json(fullfile(out,'progress.json'),struct('stage','solve','complete',false,'blocks_done',bi, ...
                'blocks_total',nb,'poses_total',np,'valid_solved',solved_valid,'geometry_valid',total_valid));
            fprintf('CEM split block %d/%d; solved valid %d/%d\n',bi,nb,solved_valid,total_valid);
            continue;
        end
        local_json(fullfile(out,'progress.json'),struct('stage','green','complete',false,'group',group_index,'groups_total',numel(groups), ...
            'poses_total',np,'geometry_valid',total_valid,'candidate_nodes',numel(candidate_nodes)));
        if isempty(candidate_nodes)
            green=zeros(0);green_meta=struct('status','no_admissible_geometry');
        else
            fprintf('Green group %d/%d: %d nodes, %.3f GiB dense storage.\n',group_index,numel(groups),numel(candidate_nodes),8*numel(candidate_nodes)^2/2^30);
            [green,green_meta]=full_scan_green(si,sigma,candidate_nodes, ...
                fullfile(green_dir,sprintf('group_%04d.mat',group_index)),struct('scan_fingerprint',fingerprint));
        end
        global_to_green=zeros(size(f.nodes,1),1);global_to_green(candidate_nodes)=1:numel(candidate_nodes);
        assert(isequal(size(green),[numel(candidate_nodes),numel(candidate_nodes)]));
        group_meta{group_index}=green_meta;
    for block_index=group.blocks
        path=fullfile(solve_dir,sprintf('block_%05d.mat',block_index));
        geo_path=fullfile(geo_dir,sprintf('block_%05d.mat',block_index));
        if local_checkpoint_valid(path,fingerprint)
            solved=load(path,'valid_count');solved_valid=solved_valid+solved.valid_count;continue;
        end
        b=load(geo_path);nv=numel(b.valid_rows);zv=zeros(nv,numel(sizes));
        parfor (k=1:nv,parfor_limit)
            row=zeros(1,numel(sizes));
            for j=1:numel(sizes)
                row(j)=full_scan_cem_transfer(si,reshape(b.patch_faces(k,j,:),1,4),global_to_green,green,cfg.z_contact);
            end
            zv(k,:)=row;
        end
        assert(all(isfinite(zv),'all'),'Nonfinite CEM result is a technical error');
        z=nan(numel(b.centre_ids),numel(sizes));z(b.valid_rows,:)=zv;
        rmse=sqrt(mean((z-measured(:)').^2,2));mae=mean(abs(z-measured(:)'),2);
        solved=struct('centre_ids',b.centre_ids,'phi_deg',b.phi_deg,'status_code',b.status_code, ...
            'failure_code',{b.failure_code},'z',z,'rmse',rmse,'mae',mae,'valid_count',nv,'fingerprint',fingerprint);
        local_checkpoint(path,solved,fingerprint);solved_valid=solved_valid+nv;
        local_json(fullfile(out,'progress.json'),struct('stage','solve','complete',false,'blocks_done',block_index, ...
            'blocks_total',nb,'poses_total',np,'valid_solved',solved_valid,'geometry_valid',total_valid));
        fprintf('CEM block %d/%d; solved valid %d/%d\n',block_index,nb,solved_valid,total_valid);
    end
    clear green;
    % The committed solved blocks retain every Z and their checksums. Release
    % this reproducible working matrix to bound disk use as well as RAM.
    group_cache=fullfile(green_dir,sprintf('group_%04d.mat',group_index));
    if isfile(group_cache),delete(group_cache);end
    end
    local_json(fullfile(out,'green_summary.json'),group_meta);
end
% Publish the exhaustive rectangular table only after every checkpoint exists.
tmp=fullfile(out,'poses.tmp.csv');if isfile(tmp),delete(tmp);end
best_rmse=inf(nc,1);best_phi=nan(nc,1);valid_angle_count=zeros(nc,1);total_rows=0;total_valid=0;total_excluded=0;best=struct();global_best=Inf;
for block_index=1:nb
    path=fullfile(solve_dir,sprintf('block_%05d.mat',block_index));assert(local_checkpoint_valid(path,fingerprint),'Missing solved block');
    b=load(path);n=numel(b.centre_ids);assert(n==size(unique([b.centre_ids b.phi_deg],'rows'),1),'Duplicate pose');
    status=repmat("invalid_geometry",n,1);status(b.status_code==1)="evaluated";status(b.status_code==3)="outside_search_region";
    T=table(b.centre_ids,b.phi_deg,status,b.rmse,b.mae,string(b.failure_code), ...
        'VariableNames',{'centre_index','phi_deg','status','rmse_ohm','mae_ohm','failure_code'});
    for j=1:numel(sizes),T.(sprintf('Z_L%03d_ohm',sizes(j)))=b.z(:,j);end
    if block_index==1,writetable(T,tmp);else,writetable(T,tmp,'WriteMode','append','WriteVariableNames',false);end
    total_rows=total_rows+n;total_valid=total_valid+sum(b.status_code==1);total_excluded=total_excluded+sum(b.status_code==3);
    for ci=unique(b.centre_ids)'
        ii=find(b.centre_ids==ci);assert(numel(ii)==na && isequal(sort(b.phi_deg(ii)),opt.PhiValues(:)),'Missing angle');
        valid=ii(b.status_code(ii)==1);valid_angle_count(ci)=numel(valid);
        if isempty(valid),continue;end
        [value,j]=min(b.rmse(valid));best_rmse(ci)=value;best_phi(ci)=b.phi_deg(valid(j));
        if value<global_best
            global_best=value;best=table2struct(centres(ci,:));best.phi_deg=best_phi(ci);best.rmse_ohm=value;best.mae_ohm=b.mae(valid(j));best.z_ohm=b.z(valid(j),:);
        end
    end
end
assert(total_rows==np && total_valid==sum(valid_angle_count),'Incomplete full grid');
best_rmse(~isfinite(best_rmse))=NaN;
centre_results=centres;centre_results.valid_angle_count=valid_angle_count;centre_results.rmse_min_ohm=best_rmse;centre_results.best_phi_deg=best_phi;
writetable(centre_results,fullfile(out,'centre_results.csv'));
movefile(tmp,fullfile(out,'poses.csv'),'f');
report=struct('schema','trkg4_full_pose_scan_v1','result_tag',opt.ResultTag,'source_tag',opt.SourceTag, ...
    'complete',true,'status','numerical_scan_not_physical_validation','grid',grid,'rho1',fixed_rho(1),'rho2',fixed_rho(2), ...
    'sizes_mm',sizes(:)','experiment_z_ohm',measured(:)','best',best,'bestrecord',best, ...
    'experiment',struct('L_mm',sizes(:)','Z_ohm',measured(:)'), ...
    'geometry_qc_passed',true, ...
    'counts',struct('centres',nc,'angles',na,'poses',np,'evaluated',total_valid,'invalid_geometry',np-total_valid-total_excluded,'outside_search_region',total_excluded, ...
    'centres_with_valid_angles',sum(valid_angle_count>0)), ...
    'metric','equal_weight_RMSE_ohm_against_experiment','fingerprint',fingerprint,'out_dir',out);
for k=1:numel(producer_sources)
    assert(strcmp(producer_sources(k).sha256,trkg4_file_sha256(fullfile(root,producer_sources(k).path))), ...
        ['Source changed during full scan: ' producer_sources(k).path]);
end
for k=1:numel(entries)
    assert(strcmp(entries(k).sha256,trkg4_file_sha256(fullfile(root,entries(k).path))), ...
        ['Historical model input/artifact changed during scan: ' entries(k).path]);
end
if isempty(fieldnames(best)),report.bestrecord=NaN;end
local_json(fullfile(out,'summary.json'),report);
outputs={'centres.csv','centre_results.csv','poses.csv','surface.mat','summary.json'};
artifacts=struct('name',{},'path',{},'sha256',{});
for k=1:numel(outputs),artifacts(k)=struct('name',outputs{k},'path',outputs{k},'sha256',trkg4_file_sha256(fullfile(out,outputs{k})));end
input_records=struct('name',{},'path',{},'sha256',{});
combined=[rmfield(entries,'name');producer_sources(:)];
for k=1:numel(combined)
    [~,name,ext]=fileparts(combined(k).path);
    input_records(k)=struct('name',[name ext],'path',strrep(fullfile('..','..',combined(k).path),filesep,'/'),'sha256',combined(k).sha256);
end
local_json(fullfile(out,'result_contract.json'),struct('complete',true,'fingerprint',fingerprint,'artifacts',artifacts));
local_json(fullfile(out,'manifest.json'),struct('complete',true,'fingerprint',fingerprint,'inputs',input_records,'outputs',artifacts));
local_json(fullfile(out,'progress.json'),struct('stage','complete','complete',true,'counts',report.counts));
fprintf('FULL SCAN COMPLETE: %d poses, %d evaluated; minimum RMSE %.9g Ohm.\n',np,total_valid,global_best);
end


function [solved,meta]=local_split_cem_block(b,si,sigma,cfg,measured,fingerprint,node_limit,workers,green_dir,group_index)
nv=numel(b.valid_rows);ns=numel(measured);zv=zeros(nv,ns);first=1;part=0;parts={};
while first<=nv
    nodes=[];last=first-1;
    for k=first:nv
        row_faces=reshape(b.patch_faces(k,:,:),[],1);ids=unique(vertcat(row_faces{:}));
        proposed=union(nodes,reshape(si.boundary(ids,:),[],1));
        if numel(proposed)>node_limit
            assert(k>first,'One whole pose exceeds the configured Green node limit');break;
        end
        nodes=proposed;last=k;
    end
    part=part+1;cache=fullfile(green_dir,sprintf('group_%04d_part_%04d.mat',group_index,part));
    fprintf('Green split group %d part %d, valid rows %d:%d/%d, %d nodes.\n',group_index,part,first,last,nv,numel(nodes));
    [G,gm]=full_scan_green(si,sigma,nodes,cache,struct('scan_fingerprint',fingerprint));
    map=zeros(size(si.nodes,1),1);map(nodes)=1:numel(nodes);rows=first:last;zp=zeros(numel(rows),ns);
    parfor (k=1:numel(rows),workers)
        row=zeros(1,ns);
        for j=1:ns,row(j)=full_scan_cem_transfer(si,reshape(b.patch_faces(rows(k),j,:),1,4),map,G,cfg.z_contact);end
        zp(k,:)=row;
    end
    zv(rows,:)=zp;parts{part}=gm;clear G; %#ok<AGROW>
    if isfile(cache),delete(cache);end
    first=last+1;
end
assert(all(isfinite(zv),'all'),'Nonfinite CEM result is a technical error');
z=nan(numel(b.centre_ids),ns);z(b.valid_rows,:)=zv;
rmse=sqrt(mean((z-measured(:)').^2,2));mae=mean(abs(z-measured(:)'),2);
solved=struct('centre_ids',b.centre_ids,'phi_deg',b.phi_deg,'status_code',b.status_code, ...
    'failure_code',{b.failure_code},'z',z,'rmse',rmse,'mae',mae,'valid_count',nv,'fingerprint',fingerprint);
meta=struct('split',true,'parts',{parts},'valid_poses',nv);
end

function [status,failure,patches]=local_pose_geometry(f,pc,cfg,projected,projection_faces,sizes)
ns=numel(sizes);status=uint8(0);failure='';patches=cell(1,ns,4);
[~,size_order]=sort(sizes,'descend');
try
    for j=size_order(:)'
        centres=reshape(projected(:,j,:),4,3);
        [~,faces]=full_scan_patches(f,centres,cfg,pc,projection_faces(:,j));
        for e=1:4,patches{1,j,e}=uint32(faces{e}(:));end
    end
    status=uint8(1);
catch exception
    allowed={'trkg4:electrodeDiagnostics','electrode_faces_by_area:noSeed', ...
        'electrode_faces_by_area:patchTooSmall','trkg4:patchOverlap'};
    if ~any(strcmp(exception.identifier,allowed)),rethrow(exception);end
    status=uint8(2);failure=exception.identifier;patches(:)={[]};
end
end

function ok=local_checkpoint_valid(path,fingerprint)
ok=false;metadata=[path '.json'];
if ~isfile(path)&&~isfile(metadata),return;end
if ~isfile(path)||~isfile(metadata)
    % An uncommitted checkpoint is safe to recompute; never treat it as done.
    return;
end
m=jsondecode(fileread(metadata));
assert(strcmp(m.fingerprint,fingerprint),'Checkpoint belongs to another full scan');
assert(strcmp(m.sha256,trkg4_file_sha256(path)),'Checkpoint content changed');ok=true;
end

function local_checkpoint(path,s,fingerprint)
tmp=[path '.tmp.mat'];save(tmp,'-struct','s','-v7');movefile(tmp,path,'f');
local_json([path '.json'],struct('fingerprint',fingerprint,'sha256',trkg4_file_sha256(path)));
end

function local_json(path,value)
tmp=[path '.tmp'];fid=fopen(tmp,'w');assert(fid>0,'Cannot create result file');
cleaner=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));clear cleaner;
movefile(tmp,path,'f');
end

function value=local_text_sha(text)
md=java.security.MessageDigest.getInstance('SHA-256');md.update(uint8(unicode2native(text,'UTF-8')));
bytes=typecast(md.digest(),'uint8');value=lower(reshape(dec2hex(bytes,2)',1,[]));
end
