function manifest = run_ct_fem_contacts_prepare(input_json, output_directory)
%RUN_CT_FEM_CONTACTS_PREPARE Sidecar finite-pad preparation, no FEM solve.
% Input: prepare_ct_fem.py cem_contacts_input output JSON, fresh output root.
% Explicit per-scenario presence; absent catalog candidates are not electrodes.
% Mixed areas use collective canonical scalar-area calls then select matching
% pads. This may conservatively reject a feasible placement. No new grower,
% threshold relaxation or automatic model promotion is added. The canonical
% run_pipeline profile ct_fem_contacts_prepare delegates preparation here.
% Group potentials are tied by congruence of the canonical per-physical CEM
% matrix, allowing different positive SI contact impedances within a group.

input_json = canonical_path(input_json);
output_directory = canonical_path(output_directory);
input_root = fileparts(input_json);
assert(~isfolder(output_directory) && ~isfile(output_directory), ...
    'ct_fem:freshOutput','Use a fresh output directory.');
assert(~within(output_directory,input_root) && ~within(input_root,output_directory), ...
    'ct_fem:outputScope','Input and output directories must be separate.');
input_digest = trkg4_file_sha256(input_json);
preparation_path=fullfile(input_root,'preparation.local.json');
preparation_digest=trkg4_file_sha256(preparation_path);
prepared=read_json(preparation_path);
assert(strcmp(prepared.schema,'ct_fem_preparation_artifact_v1') && ...
    strcmp(prepared.stage,'cem_contacts_input') && strcmp(prepared.status,'prepared_not_validated'), ...
    'ct_fem:stage','CEM input sidecar is incomplete.');
verify_files(prepared.product_files);
bound=find(strcmp({prepared.product_files.path},input_json));
assert(numel(bound)==1 && strcmp(prepared.product_files(bound).sha256,input_digest), ...
    'ct_fem:inputIdentity','CEM input JSON is not bound to the completed sidecar.');
c = read_json(input_json);
assert(strcmp(c.schema,'ct_fem_cem_contacts_input_v1') && ...
    strcmp(c.status,'input_prepared_MATLAB_QC_pending') && c.mesh_QC_passed && ...
    strcmp(c.geometry_frame,'LPS') && strcmp(c.length_unit,'mm') && c.matrix_index_base==1, ...
    'ct_fem:contract','Expected completed generic CEM input in LPS/mm.');
assert(c.mesh_QC_limits.minimum_mean_ratio==.01 && c.mesh_QC_limits.maximum_edge_condition==10000, ...
    'ct_fem:contract','Fixed mesh QC limits are required.');
assert(strcmp(c.patch_method, ...
    'collective_scalar_area_calls_select_matching_contacts_reject_mixed_overlap'), ...
    'ct_fem:contract','Unsupported patch method.');
verify_files(c.source_files);
verify_files(c.input_products);
assert(strcmp(trkg4_file_sha256(c.mesh_path),c.mesh_sha256) && ...
    strcmp(c.source_mesh_sha256,c.mesh_sha256), 'ct_fem:meshIdentity','Mesh SHA mismatch.');
assert(strcmp(trkg4_file_sha256(fullfile(c.source_artifact_directory,'preparation.local.json')), ...
    c.source_manifest_sha256),'ct_fem:parentIdentity','Parent manifest changed.');
% No path mutation: caller must expose canonical src/tools/vendor functions.
% Reject a shadowed helper rather than running an unpinned implementation.
for k=1:numel(c.kernel_files)
    p=c.kernel_files(k);
    assert(strcmpi(canonical_path(which(p.name)),canonical_path(p.path)), ...
        'ct_fem:kernelPath','Canonical helper is absent/shadowed: %s',p.name);
end
m = load(c.mesh_path,'nodes_mm','elems','boundary');
check_array(m.nodes_mm,c.mesh_arrays.nodes_mm,'double');
check_array(m.elems,c.mesh_arrays.elems,'int64');
check_array(m.boundary,c.mesh_arrays.boundary,'int64');
assert(size(m.nodes_mm,2)==3 && size(m.elems,2)==4 && size(m.boundary,2)==3 && ...
    all(isfinite(m.nodes_mm),'all') && all(m.elems>=1 & m.elems<=size(m.nodes_mm,1),'all') && ...
    all(m.boundary>=1 & m.boundary<=size(m.nodes_mm,1),'all'), 'ct_fem:mesh','Invalid bound mesh arrays.');
mm = struct('name','Generic CT finite-contact preparation','nodes',double(m.nodes_mm), ...
    'elems',double(m.elems),'boundary',double(m.boundary));
ctx=trkg4_local_patch_context(mm); % internally reuses electrode_patch_context
ids={c.contact_catalog.id};
assert(numel(unique(ids))==numel(ids),'ct_fem:catalog','Duplicate physical IDs.');
assert(isfinite(c.maximum_projection_mm) && c.maximum_projection_mm>0, ...
    'ct_fem:projection','Explicit positive projection limit required.');
assert(~isempty(c.scenarios) && numel(unique({c.scenarios.id}))==numel(c.scenarios), ...
    'ct_fem:scenarios','Distinct explicit scenarios required.');
[ok,message]=mkdir(output_directory);
assert(ok,'ct_fem:output','Cannot create output: %s',message);
try
    scenarios=cell(numel(c.scenarios),1); summaries=cell(size(scenarios));
    for s=1:numel(c.scenarios)
        [scenarios{s},summaries{s}]=prepare_scenario(c.scenarios(s),c.contact_catalog,ids,ctx,mm,c.maximum_projection_mm);
    end
    flags=struct('FEM_was_run',false,'global_mesh_convergence_proven',false, ...
        'quantitative_result_accepted',false,'independent_anatomical_validation',false,'automatic_publication',false);
    qc=struct('status','cem_contacts_prepared_not_validated','CEM_contacts_QC_passed',true, ...
        'mesh_sha256',c.mesh_sha256,'scenarios',{summaries},'flags',flags, ...
        'mixed_area_limit',c.mixed_area_limit,'contact_impedance_units','ohm m^2', ...
        'contact_QC_is_not_FEM_convergence',true,'shape_is_discrete_canonical_patch_not_exact_disk',true);
    flag_names=fieldnames(flags);
    for k=1:numel(flag_names), qc.(flag_names{k})=flags.(flag_names{k}); end
    si=trkg4_scale_fmdl_to_si(mm,struct('scale_nodes_to_si',true,'length_unit','mm'));
    result=struct('schema','ct_fem_cem_contacts_model_v1','nodes_lps_m',si.nodes, ...
        'elems',mm.elems,'boundary',mm.boundary,'scenarios',{scenarios},'qc',qc, ...
        'input_contract',c,'flags',flags);
    mat_path=fullfile(output_directory,'cem_contacts.mat');
    save(mat_path,'result','-v7.3');
    qc_path=fullfile(output_directory,'cem_contacts_qc.local.json');
    write_json_new(qc_path,qc);
    verify_files(c.source_files); verify_files(c.input_products);
    verify_files(prepared.product_files);
    assert(strcmp(trkg4_file_sha256(preparation_path),preparation_digest), ...
        'ct_fem:inputChanged','Completed sidecar manifest changed during preparation.');
    assert(strcmp(trkg4_file_sha256(input_json),input_digest), ...
        'ct_fem:inputChanged','CEM input contract changed during preparation.');
    manifest=struct('schema','ct_fem_cem_contacts_artifact_v1', ...
        'status','cem_contacts_prepared_not_validated','stage','cem_contacts', ...
        'subject_alias',c.subject_alias,'input_json',input_json,'input_sha256',input_digest, ...
        'input_preparation_manifest',preparation_path,'input_preparation_sha256',preparation_digest, ...
        'source_manifest_sha256',c.source_manifest_sha256,'mesh_sha256',c.mesh_sha256, ...
        'mesh_arrays',c.mesh_arrays,'source_files',c.source_files,'kernel_files',c.kernel_files, ...
        'input_products',c.input_products, ...
        'products',[struct('path',mat_path,'sha256',trkg4_file_sha256(mat_path)); ...
                    struct('path',qc_path,'sha256',trkg4_file_sha256(qc_path))], ...
        'CEM_contacts_QC_passed',true,'flags',flags,'matlab_version',version, ...
        'canonical_pipeline_routing_added',false);
    for k=1:numel(flag_names), manifest.(flag_names{k})=flags.(flag_names{k}); end
    write_json_new(fullfile(output_directory,'cem_contacts_manifest.local.json'),manifest);
catch exception
    write_json_new(fullfile(output_directory,'cem_contacts_failed.local.json'), ...
        struct('status','failed','identifier',exception.identifier,'message',exception.message, ...
        'CEM_contacts_QC_passed',false,'FEM_was_run',false,'quantitative_result_accepted',false));
    rethrow(exception);
end
end

function [r,q]=prepare_scenario(s,catalog,catalog_ids,ctx,mm,max_projection)
present=strings_list(s.present_contact_ids);
assert(~isempty(present) && numel(unique(present))==numel(present), ...
    'ct_fem:presence','Explicit distinct present IDs required.');
[known,index]=ismember(present,catalog_ids);
assert(all(known),'ct_fem:presence','Unknown present contact.');
p=catalog(index); n=numel(p);
centres=zeros(n,3); projection_face=zeros(n,1); projection=zeros(n,2);
all_faces=(1:size(mm.boundary,1))';
for k=1:n
    assert(isfinite(p(k).z_contact_ohm_m2) && p(k).z_contact_ohm_m2>0 && ...
        ~isempty(strtrim(p(k).z_contact_source)) && isfinite(p(k).nominal_area_mm2) && p(k).nominal_area_mm2>0, ...
        'ct_fem:contact','Explicit positive area/z_contact and source required.');
    raw=point(p(k).raw_lps_mm); old=point(p(k).source_projected_lps_mm);
    [distance,centres(k,:),projection_face(k)]=trkg4_local_triangle_distance(ctx,raw,all_faces);
    projection(k,:)=[distance,norm(centres(k,:)-old)];
    assert(all(projection(k,:)<=max_projection), ...
        'ct_fem:projection','Actual FEM projection exceeds limit for %s',p(k).id);
end
assert(size(unique(centres,'rows'),1)==n,'ct_fem:projection','Present contacts collapse after projection.');
areas=[p.nominal_area_mm2]; distinct=unique(areas,'stable');
el=[]; faces=cell(n,1); certificates=cell(numel(distinct),1);
for a=1:numel(distinct)
    % ALL and ONLY physically present pads participate in canonical ownership.
    [trial,trial_faces,certificates{a}]=trkg4_local_exact_patches(ctx,centres,distinct(a),1);
    if isempty(el), el=trial; end
    for k=find(areas==distinct(a))
        el(k)=trial(k); faces{k}=trial_faces{k};
    end
end
assert(numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes)), ...
    'ct_fem:patchOverlap','Selected physical patches overlap; refine/reconsider preparation, never relax QC.');
rows=cell(n,1);
for k=1:n
    induced=find(all(ismember(mm.boundary,el(k).nodes),2));
    assert(isequal(induced(:),faces{k}(:)), 'ct_fem:inducedFaces','Canonical induced-face identity mismatch.');
    f=mm.boundary(faces{k},:); v=mm.nodes;
    [~,~,edge]=unique(sort([f(:,[1,2]);f(:,[2,3]);f(:,[3,1])],2),'rows');
    incidence=sparse(repmat((1:size(f,1))',3,1),edge,1);
    component=conncomp(graph(spones(incidence*incidence'),'upper','omitselfloops'));
    assert(numel(unique(component))==1,'ct_fem:patchConnectivity','Induced patch is not edge-connected.');
    da=ctx.canonical.face_area(faces{k});
    total=sum(da); centroid=sum(ctx.face_centroid(faces{k},:).*da,1)/total;
    area_error=abs(total/areas(k)-1); offset=norm(centroid-centres(k,:));
    extent=max(vecnorm(v(el(k).nodes,:)-centres(k,:),2,2));
    limits=p(k).patch_qc;
    assert(isfinite(limits.maximum_relative_area_error) && limits.maximum_relative_area_error>0 && ...
        limits.maximum_relative_area_error<=.12 && isfinite(limits.maximum_centroid_offset_mm) && ...
        limits.maximum_centroid_offset_mm>0 && limits.maximum_centroid_offset_mm<=1 && ...
        isfinite(limits.maximum_extent_mm) && limits.maximum_extent_mm>0 && ...
        ~isempty(strtrim(limits.extent_justification)), 'ct_fem:limits','Invalid explicit patch QC.');
    if abs(areas(k)-pi*2.5^2)<=1e-12
        assert(limits.maximum_extent_mm<=3.7,'ct_fem:limits','5mm extent QC cannot be relaxed.');
    end
    assert(area_error<limits.maximum_relative_area_error && offset<limits.maximum_centroid_offset_mm && ...
        extent<limits.maximum_extent_mm, 'ct_fem:patchQC','Finite-patch QC failed for %s',p(k).id);
    el(k).z_contact=p(k).z_contact_ohm_m2;
    rows{k}=struct('id',p(k).id,'nominal_area_mm2',areas(k),'actual_area_mm2',total, ...
        'relative_area_error',area_error,'centroid_lps_mm',centroid,'centroid_offset_mm',offset, ...
        'extent_mm',extent,'boundary_face_ids',faces{k},'node_ids',el(k).nodes, ...
        'edge_connected',true, ...
        'raw_lps_mm',point(p(k).raw_lps_mm),'original_projected_lps_mm',point(p(k).original_projected_lps_mm), ...
        'source_projected_lps_mm',point(p(k).source_projected_lps_mm), ...
        'fem_projected_lps_mm',centres(k,:),'projection_boundary_face_id',projection_face(k), ...
        'raw_to_FEM_distance_mm',projection(k,1),'source_to_FEM_shift_mm',projection(k,2), ...
        'provenance',p(k).provenance,'z_contact_ohm_m2',p(k).z_contact_ohm_m2, ...
        'z_contact_source',p(k).z_contact_source,'limits',limits,'passed',true);
end
% Add explicit faces uniformly after selecting structs from canonical calls.
for k=1:n, el(k).faces=mm.boundary(faces{k},:); end
temp=mm; temp.electrode=el;
temp=trkg4_scale_fmdl_to_si(temp,struct('scale_nodes_to_si',true,'length_unit','mm'));
Kphysical=trkg4_cem_contact_matrix(temp);
[ports,pad_port,drive,measure,floating]=port_map(s,present);
nn=size(mm.nodes,1); np=numel(ports);
G=sparse((1:n)',pad_port,1,n,np);
T=blkdiag(speye(nn),G);
Kgrouped=T'*Kphysical*T;
% Matrix assembly controls only; no conductivity matrix or PDE solve.
symmetry=norm(Kgrouped-Kgrouped',1)/max(1,norm(Kgrouped,1));
constant=norm(Kgrouped*ones(nn+np,1),inf)/max(1,norm(Kgrouped,inf));
assert(symmetry<1e-12 && constant<1e-12 && all(isfinite(nonzeros(Kgrouped))), ...
    'ct_fem:matrixQC','CEM contact matrix algebra failed.');
drive_vector=sparse([drive(1),drive(2)],1,[1,-1],np,1);
measure_vector=sparse([measure(1),measure(2)],1,[1,-1],np,1);
r=struct('id',s.id,'present_contact_ids',{present},'ports',ports,'physical_contact_QC',{rows}, ...
    'physical_electrodes',el,'physical_to_port',G,'physical_contact_matrix_SI',Kphysical, ...
    'grouped_contact_matrix_SI',Kgrouped,'drive_ports',drive,'measure_ports',measure, ...
    'drive_port_pattern_per_unit_A',drive_vector,'measure_port_pattern',measure_vector, ...
    'floating_ports',floating,'floating_external_current_A',zeros(numel(floating),1), ...
    'exact_patch_certificates',{certificates},'matrix_index_base',1);
q=struct('id',s.id,'present_contact_ids',{present},'absent_candidates_not_modelled',true, ...
    'contacts',{rows},'ports',ports,'drive_ports',drive,'measure_ports',measure, ...
    'floating_ports',floating,'node_disjoint',true,'faces_induced_and_bound',true, ...
    'matrix_symmetry_relative',symmetry,'matrix_constant_null_relative',constant,'passed',true);
end

function [ports,assignment,drive,measure,floating]=port_map(s,present)
groups=s.groups; ng=numel(groups); assignment=zeros(numel(present),1);
group_ids={groups.id};
assert(ng>0 && numel(unique(group_ids))==ng,'ct_fem:groups','Distinct groups required.');
ports=struct('kind',{},'id',{},'contact_ids',{});
for g=1:ng
    members=strings_list(groups(g).contact_ids);
    [found,ix]=ismember(members,present);
    assert(~isempty(members) && all(found) && numel(unique(ix))==numel(ix) && all(assignment(ix)==0), ...
        'ct_fem:groups','Groups contain absent/duplicate physical pads.');
    assignment(ix)=g;
    ports(g)=struct('kind','group','id',groups(g).id,'contact_ids',{members});
end
floating=[]; actual_floating=present(assignment==0);
assert(isequal(actual_floating(:),strings_list(s.floating_contact_ids)), ...
    'ct_fem:floating','Floating list must equal unassigned PRESENT pads.');
assert(s.floating_external_current==0 && s.absent_catalog_contacts_are_not_modelled, ...
    'ct_fem:floating','Floating pad model must prescribe zero external current.');
for k=find(assignment==0)'
    g=numel(ports)+1; assignment(k)=g; floating(end+1)=g; %#ok<AGROW>
    ports(g)=struct('kind','floating_physical','id',present{k},'contact_ids',{{present{k}}});
end
[found,drive]=ismember(strings_list(s.drive_group_ids),group_ids);
assert(all(found) && numel(drive)==2 && drive(1)~=drive(2),'ct_fem:drive','Ordered drive group pair required.');
[found,measure]=ismember(strings_list(s.measure_group_ids),group_ids);
assert(all(found) && numel(measure)==2 && measure(1)~=measure(2),'ct_fem:measure','Ordered measure group pair required.');
drive=drive(:)'; measure=measure(:)';
end

function v=point(v)
v=double(v(:)');
assert(numel(v)==3 && all(isfinite(v)),'ct_fem:point','Finite XYZ point required.');
end

function values=strings_list(values)
if isempty(values), values=cell(0,1); return; end
if ischar(values), values={values}; elseif isstring(values), values=cellstr(values); end
assert(iscell(values) && all(cellfun(@(v)ischar(v) && ~isempty(v),values)), ...
    'ct_fem:IDs','Original nonempty string IDs required.');
values=values(:);
end

function verify_files(files)
for k=1:numel(files)
    assert(strcmp(canonical_path(files(k).path),files(k).path), 'ct_fem:path','Non-normalized source path.');
    assert(strcmp(trkg4_file_sha256(files(k).path),files(k).sha256), ...
        'ct_fem:sourceChanged','Source/product changed: %s',files(k).path);
end
end

function check_array(value,identity,type)
assert(isequal(double(size(value)),double(identity.shape(:)')), 'ct_fem:arrayShape','Array shape mismatch.');
assert(all(isfinite(value),'all'), 'ct_fem:arrayValue','Nonfinite mesh values.');
if strcmp(type,'int64')
    assert(all(value==fix(value),'all'),'ct_fem:arrayValue','Integer mesh indices required.');
    expected='row_major_<i8_little_endian';
else
    expected='row_major_<f8_little_endian';
end
assert(strcmp(identity.encoding,expected),'ct_fem:arrayEncoding','Unknown array encoding.');
a=cast(value,type); [~,~,endian]=computer;
if endian=='B', a=swapbytes(a); end
bytes=typecast(reshape(a.',[],1),'uint8');
digest=java.security.MessageDigest.getInstance('SHA-256'); digest.update(bytes);
hash=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
assert(strcmp(hash,identity.sha256),'ct_fem:arrayIdentity','Mesh array identity mismatch.');
end

function value=canonical_path(value)
assert(ischar(value) || (isstring(value) && isscalar(value)), 'ct_fem:path','Expected one path.');
value=char(java.io.File(char(value)).getCanonicalPath());
end

function value=within(child,parent)
value=strcmpi(child,parent) || startsWith(lower(child),[lower(parent),filesep]);
end

function write_json_new(path,value)
assert(~isfile(path),'ct_fem:output','Output already exists: %s',path);
[fid,message]=fopen(path,'w','n','UTF-8');
assert(fid>=0,'ct_fem:output','Cannot write output: %s',message);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end

function value=read_json(path)
fid=fopen(path,'rb');
assert(fid>=0,'ct_fem:input','Cannot read JSON: %s',path);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
value=jsondecode(native2unicode(fread(fid,inf,'*uint8')','UTF-8'));
end
