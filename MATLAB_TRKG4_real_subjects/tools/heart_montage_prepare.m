function report = heart_montage_prepare(configFile)
%HEART_MONTAGE_PREPARE Four-contact CEM geometries on an unchanged volume mesh.
% Geometry-only helper; no setup_pipeline, material assignment or FEM solve.
% Usage: addpath('MATLAB_TRKG4_real_subjects/tools');
%        report = heart_montage_prepare(runtime_config_json);
% The local JSON supplies source file paths, explicit 4-by-3 LPS-mm centres,
% contact names/basis and predeclared QC limits. Output MAT variables are
% f (EIDORS model in metres), faces (1-based boundary IDs), montage, provenance.
% The baseline remains the ORIGINAL source MAT (including its byte hash);
% only alternative montages are saved. Manifest model paths are relative
% to the manifest directory. Consumers must require status='ready'.
% A later run_pipeline caller can use each MAT as its ttrkg_model_file. No
% sigma, voltage, sensitivity or old solver cache is copied into these files.
% configFile must live in output_dir. A rerun verifies identical inputs and
% saved contacts; it never silently overwrites a different preparation.
root=fileparts(fileparts(mfilename('fullpath')));
oldpath=path; restore=onCleanup(@()path(oldpath));
addpath(fullfile(root,'src'),fullfile(root,'vendor_stl_eidors'),fullfile(root,'tools'));
cfg=jsondecode(fileread(configFile));
assert(strcmp(cfg.schema,'heart_montage_prepare_v1'),'Unsupported config schema');
assert(strcmp(canonical(fileparts(configFile)),canonical(cfg.output_dir)), ...
    'Runtime config must be inside output_dir');
assert(numel(cfg.montages)>=3 && numel(cfg.montages)<=4,'Expected 3 or 4 montages');
ids={cfg.montages.id};
assert(numel(unique(ids))==numel(ids) && all(cellfun(@(s) ...
    ~isempty(regexp(s,'^[A-Za-z][A-Za-z0-9_]*$','once')),ids)),'Invalid montage IDs');
assert(cfg.diameter_mm==5,'This preparation requires 5 mm disks');
qcfg=cfg.qc; qcfg.electrode_area=pi*(cfg.diameter_mm/2)^2;
qcfg.length_unit='mm'; qcfg.fail_on_electrode_diagnostics=true;
qcfg.electrode_diagnostics_verbose=false;
assert(~isfield(qcfg,'electrode_diagnostics_file'),'QC must not write outside output_dir');
limits=struct2cell(qcfg);
assert(all(cellfun(@(v)~isnumeric(v)||all(isfinite(v(:)) & v(:)>0),limits)), ...
    'QC limits must be finite and positive');
sourceFields={'ttrkg_model_file','eit_model_file','transfer_config_file', ...
    'mesh_file','mesh_contacts_file','mesh_quality_file'};
provenance=struct('schema','heart_montage_provenance_v1');
for k=1:numel(sourceFields)
    key=sourceFields{k};
    provenance.sources.(key)=struct('name',basename(cfg.(key)), ...
        'sha256',trkg4_file_sha256(cfg.(key)));
end
provenance.runtime_config_name=basename(configFile);
provenance.runtime_config_sha256=trkg4_file_sha256(configFile);
deps={'heart_montage_prepare','pulmovista_transfer_check','trkg4_local_patch_context', ...
    'trkg4_local_exact_patches','electrode_faces_by_area','electrode_patch_context', ...
    'trkg4_electrode_diagnostics','trkg4_file_sha256'};
for k=1:numel(deps)
    provenance.implementation.(deps{k})=trkg4_file_sha256(which(deps{k}));
end
provenance.generator_note=['Current generator source fingerprint; baseline ', ...
    'reproduction is checked against the saved model, not inferred from this hash.'];
report=struct('schema','heart_montage_contacts_v1','status','checking', ...
    'path_base','manifest_directory', ...
    'provenance',provenance,'coordinate_frame','LPS','model_units','m', ...
    'reported_coordinates_units','mm','qc_limits',qcfg, ...
    'FEM_was_run',false,'tissue_properties_selected',false, ...
    'mesh_convergence_proven',false,'anatomical_validation',false, ...
    'measured_electrode_positions',false,'montages',{{}});
if isfield(cfg,'rejected_preparations'), report.rejected_preparations=cfg.rejected_preparations; end
manifestFile=fullfile(cfg.output_dir,'manifest.local.json');
if isfile(manifestFile)
    prior=jsondecode(fileread(manifestFile));
    if strcmp(prior.status,'pending_qc')
        assert(strcmp(prior.runtime_config_sha256,provenance.runtime_config_sha256), ...
            'Pending interface describes another runtime config');
    else
        assert(isequaln(prior.provenance,provenance), ...
            'Existing preparation has different provenance; use a fresh output directory');
    end
end
try
    a=load(cfg.ttrkg_model_file,'f','faces'); base=a.f;
    e=load(cfg.eit_model_file,'fmdl');
    meshFields={'nodes','elems','boundary','mat_idx','mat_idx_labels','gnd_node'};
    assert(same_fields(base,e.fmdl,meshFields),'EIT/TTRKG mesh, labels or ground differ');
    clear e
    m=load(cfg.mesh_file,'nodes_mm','elems','boundary');
    assert(isequal(base.nodes,double(m.nodes_mm)*1e-3) && ...
        isequal(base.elems,double(m.elems)) && isequal(base.boundary,double(m.boundary)), ...
        'Raw mesh differs from the operational models, including index ordering');
    clear m
    mq=jsondecode(fileread(cfg.mesh_quality_file));
    assert(strcmp(mq.mesh_sha256,provenance.sources.mesh_file.sha256) && mq.passed && ...
        mq.all_positive && mq.actual_surface_qc.passed && mq.actual_surface_qc.watertight && ...
        mq.actual_surface_qc.connected_components==1 && mq.outward_boundary_orientation_checked, ...
        'Missing or failed hash-bound mesh/surface QC');
    tc=jsondecode(fileread(cfg.transfer_config_file));
    assert(strcmp(canonical(tc.patient_output_dir),canonical(fileparts(cfg.ttrkg_model_file))) && ...
        strcmp(canonical(tc.mesh_path),canonical(cfg.mesh_file)), ...
        'Transfer config does not describe the declared baseline/mesh');
    assert(tc.z_contact_ohm_m2>0 && tc.current_A>0 && ...
        isfinite(tc.z_contact_ohm_m2) && isfinite(tc.current_A),'Invalid inherited contact/current');
    assert(numel(base.electrode)==4 && isscalar(base.stimulation) && ...
        all([base.electrode.z_contact]==tc.z_contact_ohm_m2),'Baseline is not the declared four-contact CEM');
    stim=struct('stimulation','Amp','stim_pattern',sparse([1;0;0;-1])*tc.current_A, ...
        'meas_pattern',sparse([0 1 -1 0]));
    assert(isequaln(base.stimulation,stim),'Baseline stimulation or polarity differs');
    report.mesh=struct('node_count',size(base.nodes,1),'tetrahedron_count',size(base.elems,1), ...
        'boundary_face_count',size(base.boundary,1),'exact_EIT_TTRKG_raw_mesh_identity',true, ...
        'material_partition_unchanged',true,'ground_node',base.gnd_node, ...
        'nodes_sha256',array_hash(base.nodes),'elems_sha256',array_hash(base.elems), ...
        'boundary_sha256',array_hash(base.boundary), ...
        'surface_QC_inherited_from_hash_bound_unchanged_mesh',mq.actual_surface_qc);
    ctx=trkg4_local_patch_context(struct('nodes',base.nodes*1000,'boundary',base.boundary));
    anchors=jsondecode(fileread(cfg.mesh_contacts_file));
    anchors=anchors.physical_contacts;
    rows=cell(numel(cfg.montages),1); patches=cell(size(rows)); patchFaces=cell(size(rows));
    for k=1:numel(cfg.montages)
        job=cfg.montages(k); centres=double(job.centres_lps_mm);
        report.active_montage=job.id;
        assert(isequal(size(centres),[4 3]) && all(isfinite(centres),'all'),'Invalid centres');
        assert(numel(job.contact_names)==4 && numel(job.anchor_ids)==4,'Need four contact names/anchors');
        assert(isequal(size(job.anchor_offset_mm),[4 3]),'Need four explicit anchor offsets');
        anchorRaw=zeros(4,3); anchorProjected=anchorRaw;
        for j=1:4
            ai=find(strcmp({anchors.id},job.anchor_ids{j}));
            assert(isscalar(ai),'Unknown or duplicate current-mesh anchor');
            anchorRaw(j,:)=anchors(ai).raw_lps_mm(:)';
            anchorProjected(j,:)=anchors(ai).projected_lps_mm(:)';
        end
        assert(max(abs(centres-anchorRaw-job.anchor_offset_mm),[],'all')<1e-10, ...
            'Explicit centres disagree with current-mesh anchor provenance');
        [el,faces,buildInfo]=trkg4_local_exact_patches(ctx,centres,qcfg.electrode_area,tc.z_contact_ohm_m2);
        mm=ctx.fmdl; mm.electrode=el;
        diagnostics=trkg4_electrode_diagnostics(mm,centres,job.contact_names,qcfg,centres,faces);
        topology=contact_topology(ctx,el,faces,centres,qcfg);
        report.active_contact_QC=topology;
        assert(all(cellfun(@(q)q.passed,topology)), ...
            'heart_montage:diskQC','Disk shape/resolution QC failed; see active_contact_QC in manifest');
        reused=false;
        if k==1
            assert(isequal(centres,tc.ttrkg_centres_lps_mm),'First montage must reproduce baseline');
            assert(same_contacts(base.electrode,el) && same_faces(a.faces,faces), ...
                'Canonical builder did not reproduce saved baseline contacts');
            el=base.electrode; faces=a.faces; reused=true;
        end
        unchanged=all(centres==tc.ttrkg_centres_lps_mm,2);
        for j=find(unchanged)'
            assert(isequal(el(j).nodes(:),base.electrode(j).nodes(:)) && ...
                isequal(faces{j}(:),a.faces{j}(:)), ...
                'A fixed contact changed when other contacts moved');
        end
        row=struct('id',job.id,'basis',job.basis,'roles',{{'I+','U+','U-','I-'}}, ...
            'contact_names',{job.contact_names},'anchor_ids',{job.anchor_ids}, ...
            'anchor_raw_lps_mm',anchorRaw,'anchor_projected_lps_mm',anchorProjected, ...
            'anchor_offset_mm',job.anchor_offset_mm,'requested_centres_lps_mm',centres, ...
            'nearest_surface_centres_lps_mm',[diagnostics.nearest_x_mm diagnostics.nearest_y_mm diagnostics.nearest_z_mm], ...
            'area_centroids_lps_mm',[diagnostics.patch_centroid_x_mm diagnostics.patch_centroid_y_mm diagnostics.patch_centroid_z_mm], ...
            'diameter_mm',cfg.diameter_mm,'nominal_area_mm2',qcfg.electrode_area, ...
            'actual_area_mm2',diagnostics.patch_area_mm2,'node_counts',arrayfun(@(x)numel(x.nodes),el), ...
            'diagnostics',table2struct(diagnostics),'topology',{topology}, ...
            'axial_I_U_gap_mm',abs(centres([1 4],1)-centres([2 3],1)), ...
            'I_U_centre_distance_mm',vecnorm(centres([1 4],:)-centres([2 3],:),2,2), ...
            'CEM',true,'simultaneous_electrode_count',4,'z_contact_ohm_m2',tc.z_contact_ohm_m2, ...
            'z_contact_basis','Inherited transfer-config modelling assumption; not a measured interface impedance', ...
            'current_A',tc.current_A,'current_basis','Inherited baseline stimulation; no solve performed', ...
            'stim_pattern_A',full(stim.stim_pattern),'meas_pattern',full(stim.meas_pattern), ...
            'builder_info',buildInfo,'baseline_contacts_reused',reused, ...
            'fixed_contacts_identical_to_baseline',unchanged, ...
            'contact_discretization','Connected triangle approximation to nominal 5 mm disk; actual area/shape retained', ...
            'QC_passed',true,'uses_existing_model',reused,'ttrkg_model_file',[job.id,'.mat']);
        if reused
            sourcePath=java.io.File(canonical(cfg.ttrkg_model_file)).toPath();
            outputPath=java.io.File(canonical(cfg.output_dir)).toPath();
            row.ttrkg_model_file=strrep(char(outputPath.relativize(sourcePath).toString()),'\','/');
        end
        rows{k}=row; patches{k}=el; patchFaces{k}=faces;
        report.montages=rows(1:k);
        fprintf('CONTACTS_QC %s nodes=%s area_mm2=%s\n',job.id,mat2str(row.node_counts),mat2str(row.actual_area_mm2',6));
    end
    report.reuse_audit=struct('source','ttrkg_model_file','mesh_and_contacts_identical',true);
    if isfield(cfg,'reuse_model_files')
        audit=cell(numel(cfg.reuse_model_files),1);
        for k=1:numel(cfg.reuse_model_files)
            candidate=cfg.reuse_model_files{k}; c=load(candidate,'f','faces');
            eq=isfield(c,'f') && same_fields(base,c.f,meshFields);
            match=false(numel(rows),1);
            if eq && isfield(c,'faces')
                for j=1:numel(rows)
                    match(j)=same_contacts(c.f.electrode,patches{j}) && same_faces(c.faces,patchFaces{j}) && ...
                        isequaln(c.f.stimulation,stim);
                end
            end
            audit{k}=struct('runtime_config_index',k,'sha256',trkg4_file_sha256(candidate), ...
                'mesh_and_material_identity',eq,'matching_montages',find(match), ...
                'use','Prefer canonical baseline when a duplicate has identical contacts');
        end
        report.reuse_candidates=audit;
    end
    % No output model is accepted until every requested montage has passed.
    for k=1:numel(rows)
        if rows{k}.uses_existing_model
            rows{k}.model_sha256=provenance.sources.ttrkg_model_file.sha256;
            continue;
        end
        f=base; f.electrode=patches{k}; f.stimulation=stim;
        if isfield(f,'meas_select'), f=rmfield(f,'meas_select'); end
        faces=patchFaces{k}; montage=rows{k};
        assert(numel(f.electrode)==4 && same_fields(base,f,meshFields),'Model identity changed');
        target=fullfile(cfg.output_dir,montage.ttrkg_model_file);
        if isfile(target)
            saved=load(target,'f','faces','montage','provenance');
            assert(isequaln(saved.provenance,provenance) && isequaln(saved.montage,montage) && ...
                isequaln(saved.f,f) && same_faces(saved.faces,faces),'Saved model is stale or modified');
        else
            temporary=fullfile(cfg.output_dir,[montage.id,'.pending.mat']);
            save(temporary,'f','faces','montage','provenance','-v7');
            [ok,message]=movefile(temporary,target); assert(ok,'%s',message);
        end
        rows{k}.model_sha256=trkg4_file_sha256(target);
    end
    % Detect concurrent source changes before publishing the successful manifest.
    for k=1:numel(sourceFields)
        key=sourceFields{k};
        assert(strcmp(trkg4_file_sha256(cfg.(key)),provenance.sources.(key).sha256),'Input changed during preparation');
    end
    assert(strcmp(trkg4_file_sha256(configFile),provenance.runtime_config_sha256), ...
        'Runtime config changed during preparation');
    report.status='ready';
    report.validation_scope='four_contact_CEM_geometry_checked_no_electrical_solve';
    report=rmfield(report,{'active_montage','active_contact_QC'});
    report.montages=rows;
    report.next_step='Resolve each ttrkg_model_file relative to this manifest, then use heart_stl_electrical_check through run_pipeline.';
    write_json(manifestFile,report);
catch err
    report.status='failed'; report.error=struct('identifier',err.identifier,'message',err.message);
    write_json(manifestFile,report); rethrow(err);
end
end

function q=contact_topology(ctx,el,faces,centres,cfg)
% Node-only connectivity is insufficient: require an edge-connected disk,
% one boundary loop, no hole/nonmanifold edge, finite area and no shared node.
q=cell(4,1); seen=[];
for k=1:4
    ids=el(k).nodes(:); tris=ctx.fmdl.boundary(faces{k},:);
    assert(numel(unique(ids))==numel(ids) && isempty(intersect(ids,seen)), ...
        'Repeated or shared electrode nodes'); seen=[seen;ids];
    assert(numel(ids)>=cfg.min_node_count && size(tris,1)>=cfg.min_face_count, ...
        'Contact is point-like or underresolved');
    assert(isequal(sort(ids),unique(tris(:))),'Contact has nodes without a surface triangle');
    edges=sort([tris(:,[1 2]);tris(:,[2 3]);tris(:,[3 1])],2);
    [uniqueEdges,~,ei]=unique(edges,'rows'); count=accumarray(ei,1);
    assert(all(count<=2),'Nonmanifold contact edge');
    edgeFaces=repmat((1:size(tris,1))',3,1);
    [~,order]=sort(ei); sortedFaces=edgeFaces(order); starts=cumsum([1;count(1:end-1)]);
    paired=find(count==2);
    g=graph(sortedFaces(starts(paired)),sortedFaces(starts(paired)+1),[],size(tris,1));
    nComponents=max(conncomp(g));
    rim=uniqueEdges(count==1,:); [rimNodes,~,ri]=unique(rim);
    mappedRim=reshape(ri,size(rim));
    rimGraph=graph(mappedRim(:,1),mappedRim(:,2));
    loops=max(conncomp(rimGraph)); euler=numel(ids)-size(uniqueEdges,1)+size(tris,1);
    assert(nComponents==1 && loops==1 && all(degree(rimGraph)==2) && euler==1, ...
        'Contact is not a single topological disk');
    p=ctx.fmdl.nodes; radius=vecnorm(p(rimNodes,:)-centres(k,:),2,2);
    edgeLength=vecnorm(p(uniqueEdges(:,1),:)-p(uniqueEdges(:,2),:),2,2);
    allRadius=vecnorm(p(ids,:)-centres(k,:),2,2);
    normals=cross(p(tris(:,2),:)-p(tris(:,1),:),p(tris(:,3),:)-p(tris(:,1),:),2);
    n=sum(normals,1); n=n/norm(n);
    delta=p(rimNodes,:)-centres(k,:); tangent=delta-(delta*n')*n;
    tangentRadius=vecnorm(tangent,2,2);
    singular=svd(tangent-mean(tangent,1),0); axisRatio=singular(1)/singular(2);
    thickness=max(abs((p(ids,:)-centres(k,:))*n'));
    normalAngle=max(acosd(max(-1,min(1,(normals./vecnorm(normals,2,2))*n'))));
    passed=max(allRadius)<=cfg.max_patch_radius_mm && min(radius)>=cfg.min_rim_radius_mm && ...
        max(edgeLength)<=cfg.max_patch_edge_mm && axisRatio<=cfg.max_boundary_axis_ratio && ...
        min(tangentRadius)>=cfg.min_rim_radius_mm && thickness<=cfg.max_projection_thickness_mm && ...
        normalAngle<=cfg.max_normal_angle_deg;
    q{k}=struct('passed',passed,'edge_connected_components',nComponents,'boundary_loops',loops, ...
        'euler_characteristic',euler,'surface_only',true,'node_disjoint',true, ...
        'rim_radius_range_mm',[min(radius) max(radius)], ...
        'max_node_radius_mm',max(allRadius),'max_edge_length_mm',max(edgeLength), ...
        'tangent_rim_radius_range_mm',[min(tangentRadius) max(tangentRadius)], ...
        'boundary_axis_ratio',axisRatio,'max_normal_distance_mm',thickness, ...
        'max_face_normal_angle_deg',normalAngle, ...
        'node_ids_sha256',array_hash(ids),'boundary_face_ids_sha256',array_hash(faces{k}));
end
end

function yes=same_fields(a,b,fields)
yes=all(isfield(a,fields)) && all(isfield(b,fields));
for k=1:numel(fields), yes=yes && isequaln(a.(fields{k}),b.(fields{k})); end
end

function yes=same_contacts(a,b)
yes=numel(a)==4 && numel(b)==4;
if ~yes, return; end
for k=1:4
    yes=yes && isequal(a(k).nodes(:),b(k).nodes(:)) && ...
        isequal(a(k).z_contact,b(k).z_contact);
end
end

function yes=same_faces(a,b)
yes=numel(a)==numel(b);
if ~yes, return; end
for k=1:numel(a), yes=yes && isequal(sort(a{k}(:)),sort(b{k}(:))); end
end

function h=array_hash(x)
% SHA-256 of class, dimensions, then native column-major numeric bytes.
md=java.security.MessageDigest.getInstance('SHA-256');
[~,~,endian]=computer;
md.update(uint8([class(x),' ',mat2str(size(x)),' column-major ',endian]));
for k=1:1000000:numel(x)
    md.update(typecast(x(k:min(k+999999,numel(x))),'uint8'));
end
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function p=canonical(p)
p=char(java.io.File(p).getCanonicalPath());
end

function n=basename(p)
[~,n,e]=fileparts(p); n=[n,e];
end

function write_json(file,data)
[folder,name]=fileparts(file); temporary=fullfile(folder,[name,'.pending.json']);
fid=fopen(temporary,'w','n','UTF-8'); assert(fid>=0,'Cannot write local manifest');
close=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));
clear close
[ok,message]=movefile(temporary,file,'f'); assert(ok,'%s',message);
end
