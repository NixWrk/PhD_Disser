function report = trkg4_prepare_surface_sensitivity(manifest_file)
%TRKG4_PREPARE_SURFACE_SENSITIVITY Build CEM montages on a frozen FEM.
%
% report = trkg4_prepare_surface_sensitivity(manifest_file)
% Manifest schema: surface_sensitivity_preparation_v1. Required fields:
% source.prepared_fem (relative to manifest), prepared_sha256, mesh_sha256,
% surface_basename; contact.area_mm2, z_contact_ohm_m2; qc with
% max_surface_distance_mm, max_centroid_offset_mm, min_area_fraction,
% max_area_ratio; one or more inputs with id, json (relative), sha256.
% JSON inputs must be v3 explicit_points in raw_json/, with the role order
% I_plus, V_plus, V_minus, I_minus. E1..E4 preserve exactly that order.
%
% The existing prepared_fem contains fmdl_m in metres, five volume blocks
% of size (nn+4)-by-(nn+4), tissue_id and tissue_names. Only the four surface
% electrodes are replaced. Millimetres are used for patch construction and
% diagnostics; the SI model retains the ORIGINAL node array without an
% mm->m round trip. z_contact is taken from the manifest in ohm*m^2.
% Legacy 40.05 MAT files instead store fmdl_mm with SI blocks. This requires
% source.prepared_format='legacy_40_05_mm_geometry_si_blocks' plus a verified
% source.model_identity binding the MAT, mesh and current producer code hash.
% Legacy patches use original mm nodes; CEM uses exactly nodes * 0.001.
% report.prepared_units records this explicit conversion and its evidence.
%
% Outputs: contacts/<id>_contact.mat, <id>_diagnostics.csv, <id>_geometry.json;
% contacts/preparation.json is written LAST. Only status='passed' authorises
% consumption of the complete set. QC/assembly/source failures return a
% failed report; malformed manifests and nonempty output directories throw.
% No existing run is overwritten or resumed, including a failed run.
%
% This performs matrix assembly, NOT factorisation or a conductivity sweep.
% Mesh convergence and anatomical/experimental validity remain untested.
% Optional source paths mesh, surface, mesh_acceptance and model_identity
% are hash-checked when supplied with their <name>_sha256 fields. Without
% source.mesh, mesh_sha256 remains a manifest provenance assertion only.

manifest_file = local_canonical(local_text(manifest_file));
manifest = jsondecode(fileread(manifest_file));
local_validate_manifest(manifest);
base = fileparts(manifest_file);
prepared_file = local_relative_file(base, manifest.source.prepared_fem);
[source_files, source_hashes, source_names] = local_source_files(base, manifest.source);
input_count = numel(manifest.inputs);
json_files = cell(input_count, 1);
for k = 1:input_count
    json_files{k} = local_relative_file(base, manifest.inputs(k).json);
    raw_root = [local_canonical(fullfile(base, 'raw_json')), filesep];
    if ~local_path_starts_with(json_files{k}, raw_root)
        error('trkg4:surfacePreparationManifest', ...
            'Every input JSON must resolve inside raw_json/ beside the manifest.');
    end
end

out = fullfile(base, 'contacts');
if isfolder(out)
    entries = dir(out);
    if any(~ismember({entries.name}, {'.', '..'}))
        error('trkg4:outputWouldOverwrite', ...
            'contacts/ is not empty. Use a new manifest directory for a new run.');
    end
elseif isfile(out)
    error('trkg4:outputWouldOverwrite', 'contacts exists as a file.');
else
    mkdir(out);
end
% An exclusive marker also excludes simultaneous writers in an empty folder.
lock_file = fullfile(out, 'preparation.lock');
if ~java.io.File(lock_file).createNewFile()
    error('trkg4:outputWouldOverwrite', 'Another preparation owns contacts/.');
end
lock_cleanup = onCleanup(@() delete(lock_file)); %#ok<NASGU>

roles = {'I_plus', 'V_plus', 'V_minus', 'I_minus'};
entry = struct('id', '', 'status', 'not_prepared', ...
    'contact_matrix', '', 'contact_sha256', '', 'matrix_error', [], ...
    'diagnostics_file', '', 'diagnostics_sha256', '', ...
    'geometry_file', '', 'geometry_sha256', '', ...
    'source_json', '', 'source_json_sha256', '', ...
    'requested_centres_xyz_mm', [], 'actual_centres_xyz_mm', [], ...
    'areas_mm2', [], 'face_counts', [], 'geometry_qc', [], ...
    'error_identifier', '', 'error_message', '');
report = struct('schema', 'surface_sensitivity_preparation_result_v1', ...
    'status', 'failed', 'manifest_sha256', trkg4_file_sha256(manifest_file), ...
    'prepared_sha256', lower(manifest.source.prepared_sha256), ...
    'mesh_sha256', lower(manifest.source.mesh_sha256), ...
    'mesh_hash_verification', 'manifest_assertion_only_no_mesh_file_in_contract', ...
    'surface_basename', manifest.source.surface_basename, ...
    'source', manifest.source, 'contact', manifest.contact, 'qc', manifest.qc, ...
    'electrode_ids', {{'E1', 'E2', 'E3', 'E4'}}, ...
    'electrode_order', {roles}, 'matrix_error_tolerance', 1e-12, ...
    'mesh_convergence', 'untested', 'factorisation_performed', false, ...
    'fem_sweep_performed', false, 'error_identifier', '', 'error_message', '', ...
    'montages', repmat(entry, input_count, 1));
report.warnings = { ...
    'Mesh convergence is untested; CT/STL voxel spacing is not FEM element size.', ...
    'Contact area and z_contact are model inputs, not measured contact properties.', ...
    'A passed report establishes geometry QC and algebraic assembly only.', ...
    'No minimum face-count threshold is imposed; triangle counts are reported.', ...
    'External means a boundary triangle with exactly one adjacent tetrahedron; anatomy is not revalidated.', ...
    'Actual centres are area-weighted patch centroids and may lie off a curved surface.'};
for k = 1:input_count
    report.montages(k).id = manifest.inputs(k).id;
    report.montages(k).source_json = manifest.inputs(k).json;
    report.montages(k).source_json_sha256 = lower(manifest.inputs(k).sha256);
end

try
    local_check_hash(prepared_file, report.prepared_sha256);
    for k = 1:numel(source_files)
        local_check_hash(source_files{k}, source_hashes{k});
    end
    report.verified_source_files = source_names;
    if isfield(manifest.source, 'mesh')
        report.mesh_hash_verification = 'file_sha256_verified';
    end
    available = whos('-file', prepared_file);
    wanted = {'fmdl_m', 'fmdl_mm', 'blocks', 'tissue_id', 'tissue_names', 'coordinate_units'};
    wanted = intersect(wanted, {available.name}, 'stable');
    frozen = load(prepared_file, wanted{:});
    [frozen, fmdl_mm, report.prepared_units, producer_file] = ...
        local_resolve_prepared(frozen, manifest.source, base);
    local_validate_prepared(frozen);
    required = {'electrode_faces_by_area', 'electrode_patch_context', ...
        'trkg4_load_surface_landmarks', 'trkg4_electrode_diagnostics', ...
        'trkg4_boundary_normals', 'trkg4_cem_contact_matrix', ...
        'mk_image', 'calc_system_mat'};
    for k = 1:numel(required)
        if exist(required{k}, 'file') == 0
            error('trkg4:surfacePreparationDependency', ...
                'Required function is not on the MATLAB path: %s.', required{k});
        end
    end
    project = fileparts(fileparts(mfilename('fullpath')));
    if ~strcmp(local_canonical(which('electrode_faces_by_area')), ...
            local_canonical(fullfile(project, 'vendor_stl_eidors', 'electrode_faces_by_area.m')))
        error('trkg4:surfacePreparationDependency', ...
            'electrode_faces_by_area must resolve to the canonical vendor_stl_eidors implementation.');
    end
    % Freeze the executed preparation code as well as its inputs.
    code_names = [{'trkg4_prepare_surface_sensitivity', 'trkg4_file_sha256'}, required];
    code_files = cellfun(@which, code_names, 'UniformOutput', false);
    if ~isempty(producer_file)
        code_names{end+1} = 'run_trkg4_arm_conductivity_matrix';
        code_files{end+1} = producer_file;
    end
    code_hashes = cellfun(@trkg4_file_sha256, code_files, 'UniformOutput', false);
    report.implementation = struct('function_name', code_names, 'sha256', code_hashes);
    report.counts = struct('nodes', size(frozen.fmdl_m.nodes, 1), ...
        'elements', size(frozen.fmdl_m.elems, 1), 'blocks', 5, ...
        'montages', input_count, 'electrodes_per_montage', 4);
    % Check all supplied boundary rows against their actual tetrahedral
    % owners once; tissue interfaces or duplicate/nonmanifold faces fail.
    trkg4_boundary_normals(fmdl_mm);
    report.boundary_qc = 'passed_single_adjacent_tetrahedron';
    context = electrode_patch_context(fmdl_mm);
    volume = sparse(size(frozen.blocks{1}, 1), size(frozen.blocks{1}, 2));
    for k = 1:5
        volume = volume + frozen.blocks{k};
    end
    % Drop extra copies of large blocks; no prepared model is generated.
    frozen = rmfield(frozen, 'blocks');
    for k = 1:input_count
        report.montages(k) = local_montage(report.montages(k), ...
            json_files{k}, manifest, frozen.fmdl_m, fmdl_mm, context, ...
            volume, base, report.prepared_sha256, report.manifest_sha256);
    end

    % Detect input mutation during preparation before issuing the final seal.
    local_check_hash(manifest_file, report.manifest_sha256);
    local_check_hash(prepared_file, report.prepared_sha256);
    for k = 1:numel(source_files)
        local_check_hash(source_files{k}, source_hashes{k});
    end
    for k = 1:numel(code_files)
        local_check_hash(code_files{k}, code_hashes{k});
    end
    for k = 1:input_count
        local_check_hash(json_files{k}, manifest.inputs(k).sha256);
    end
    if all(strcmp({report.montages.status}, 'passed'))
        report.status = 'passed';
    end
catch exception
    report.error_identifier = exception.identifier;
    report.error_message = exception.message;
end
local_write_json(fullfile(out, 'preparation.json'), report);
warning('trkg4:surfaceMeshConvergenceUntested', ...
    'Preparation status: %s. Mesh convergence has NOT been tested.', report.status);
end

function item = local_montage(item, json_file, manifest, fmdl_m, fmdl_mm, ...
        context, volume, base, prepared_sha256, manifest_sha256)
roles = {'I_plus', 'V_plus', 'V_minus', 'I_minus'};
geometry = struct('schema', 'surface_sensitivity_contact_geometry_v1', ...
    'id', item.id, 'status', 'failed', 'units', 'mm', 'index_base', 1, ...
    'manifest_sha256', manifest_sha256, 'prepared_sha256', prepared_sha256, ...
    'source_json_sha256', item.source_json_sha256, ...
    'mesh_sha256', manifest.source.mesh_sha256, ...
    'requested_centres_xyz_mm', [], 'actual_centres_xyz_mm', [], ...
    'diagnostics', [], 'electrodes', [], 'error_identifier', '', 'error_message', '');
try
    local_check_hash(json_file, item.source_json_sha256);
    landmarks = trkg4_load_surface_landmarks(json_file);
    if ~isfield(landmarks, 'schema') || ...
            ~strcmp(landmarks.schema, 'trkg4_ttrkg_surface_v3') || ...
            ~strcmp(landmarks.placement_mode, 'explicit_points') || ...
            ~isequal(cellstr(landmarks.electrode_order(:)), roles(:)) || ...
            ~strcmp(landmarks.source_basename, manifest.source.surface_basename)
        error('trkg4:surfacePreparationLandmarks', ...
            'Expected v3 explicit_points, matching surface_basename and I+/V+/V-/I- order.');
    end
    requested = landmarks.requested_centres_xyz_mm;
    item.requested_centres_xyz_mm = requested;
    geometry.requested_centres_xyz_mm = requested;
    [electrodes, induced] = electrode_faces_by_area(fmdl_mm, requested, ...
        manifest.contact.area_mm2, manifest.contact.z_contact_ohm_m2, context);
    fmdl_mm.electrode = electrodes; % Replace the entire old electrode array.
    cfg = struct('length_unit', 'mm', 'electrode_area', manifest.contact.area_mm2, ...
        'max_electrode_surface_distance_mm', manifest.qc.max_surface_distance_mm, ...
        'max_electrode_patch_centroid_offset_mm', manifest.qc.max_centroid_offset_mm, ...
        'min_electrode_patch_area_fraction', manifest.qc.min_area_fraction, ...
        'max_electrode_patch_area_ratio', manifest.qc.max_area_ratio, ...
        'fail_on_electrode_diagnostics', false, 'electrode_diagnostics_verbose', false);
    % Keep measured diagnostics even when a threshold fails. The canonical
    % diagnostic emits a warning; the explicit booleans below block export.
    d = trkg4_electrode_diagnostics(fmdl_mm, requested, roles, cfg, requested, induced);
    connected = false(4, 1);
    used = [];
    disjoint = true;
    patches = cell(4, 1);
    for j = 1:4
        faces = fmdl_mm.boundary(induced{j}, :);
        connected(j) = local_edge_connected(faces);
        disjoint = disjoint && isempty(intersect(used, electrodes(j).nodes));
        used = [used; electrodes(j).nodes(:)]; %#ok<AGROW>
        neighbours = find(any(ismember(fmdl_mm.boundary, electrodes(j).nodes), 2));
        patches{j} = struct('electrode_id', sprintf('E%d', j), 'role', roles{j}, ...
            'z_contact_ohm_m2', electrodes(j).z_contact, ...
            'patch', local_mesh_crop(fmdl_mm, induced{j}), ...
            'neighbourhood', local_mesh_crop(fmdl_mm, neighbours));
    end
    area_fraction = d.patch_area_mm2 / manifest.contact.area_mm2;
    d.pass_surface_distance = isfinite(d.nearest_boundary_distance_mm) & ...
        d.nearest_boundary_distance_mm <= manifest.qc.max_surface_distance_mm;
    d.pass_centroid_offset = isfinite(d.patch_centroid_offset_mm) & ...
        d.patch_centroid_offset_mm <= manifest.qc.max_centroid_offset_mm;
    d.area_fraction = area_fraction;
    d.pass_area = isfinite(area_fraction) & ...
        area_fraction >= manifest.qc.min_area_fraction & ...
        area_fraction <= manifest.qc.max_area_ratio;
    d.edge_connected = connected;
    d.external_boundary = true(4, 1); % Global single-owner check passed.
    d.node_disjoint = repmat(disjoint, 4, 1);
    d.pass_geometry = d.pass_surface_distance & d.pass_centroid_offset & ...
        d.pass_area & connected & d.node_disjoint & d.patch_face_count > 0;
    item.actual_centres_xyz_mm = [d.patch_centroid_x_mm, ...
        d.patch_centroid_y_mm, d.patch_centroid_z_mm];
    item.areas_mm2 = d.patch_area_mm2;
    item.face_counts = d.patch_face_count;
    item.geometry_qc = struct('passed', all(d.pass_geometry), ...
        'external', true, 'edge_connected', all(connected), 'node_disjoint', disjoint);
    geometry.actual_centres_xyz_mm = item.actual_centres_xyz_mm;
    geometry.diagnostics = table2struct(d);
    geometry.electrodes = vertcat(patches{:});
    item.diagnostics_file = ['contacts/', item.id, '_diagnostics.csv'];
    writetable(d, fullfile(base, item.diagnostics_file));
    item.diagnostics_sha256 = trkg4_file_sha256(fullfile(base, item.diagnostics_file));
    if ~all(d.pass_geometry)
        error('trkg4:surfacePreparationGeometry', ...
            'Contact geometry QC failed for %s. See per-electrode diagnostics.', item.id);
    end

    % Preserve exact original SI coordinates and all volume topology. Do NOT
    % multiply the old fmdl_m nodes, or scale a returned mm model by 1/1000.
    fmdl_m.electrode = electrodes;
    C = trkg4_cem_contact_matrix(fmdl_m);
    ground = fmdl_m.gnd_node;
    electrode_ids = {'E1', 'E2', 'E3', 'E4'};
    reference = calc_system_mat(mk_image(fmdl_m, 1));
    if ~isfield(reference, 'E') || ~isequal(size(reference.E), size(volume)) || ...
            ~isequal(size(C), size(volume)) || ...
            any(~isfinite(nonzeros(C))) || any(~isfinite(nonzeros(reference.E)))
        error('trkg4:surfacePreparationAssembly', 'Invalid CEM/EIDORS matrix shape or values.');
    end
    reference_norm = norm(reference.E, 'fro');
    matrix_error = norm(volume + C - reference.E, 'fro') / reference_norm;
    item.matrix_error = matrix_error;
    if ~isfinite(matrix_error) || matrix_error > 1e-12
        error('trkg4:surfacePreparationAssembly', ...
            'sum(existing blocks)+C differs from EIDORS: relative error %.6g.', matrix_error);
    end
    item.contact_matrix = ['contacts/', item.id, '_contact.mat'];
    save(fullfile(base, item.contact_matrix), 'C', 'ground', ...
        'prepared_sha256', 'electrode_ids', 'matrix_error', '-v7');
    item.contact_sha256 = trkg4_file_sha256(fullfile(base, item.contact_matrix));
    item.status = 'passed';
    geometry.status = 'passed';
catch exception
    item.status = 'failed';
    item.error_identifier = exception.identifier;
    item.error_message = exception.message;
    geometry.error_identifier = exception.identifier;
    geometry.error_message = exception.message;
end
geometry.matrix_error = item.matrix_error;
item.geometry_file = ['contacts/', item.id, '_geometry.json'];
local_write_json(fullfile(base, item.geometry_file), geometry);
item.geometry_sha256 = trkg4_file_sha256(fullfile(base, item.geometry_file));
end

function crop = local_mesh_crop(fmdl_mm, indices)
faces = fmdl_mm.boundary(indices, :);
[node_ids, ~, local_ids] = unique(faces(:));
crop = struct('nodes_global_1based', node_ids, ...
    'coords_mm', fmdl_mm.nodes(node_ids, :), ...
    'faces_1based', reshape(local_ids, size(faces)), ...
    'boundary_face_ids_1based', indices(:));
end

function connected = local_edge_connected(faces)
n = size(faces, 1);
if n == 0, connected = false; return; end
edges = sort([faces(:, [1 2]); faces(:, [2 3]); faces(:, [3 1])], 2);
[~, ~, group] = unique(edges, 'rows');
face_ids = repmat((1:n)', 3, 1);
counts = accumarray(group, 1);
if any(counts > 2), connected = false; return; end
first = accumarray(group, face_ids, [], @min);
last = accumarray(group, face_ids, [], @max);
shared = counts == 2;
components = conncomp(graph(first(shared), last(shared), [], n));
connected = numel(unique(components)) == 1;
end

function [p, mm, units, producer_file] = local_resolve_prepared(p, source, base)
% Never infer units from variable magnitudes or rename mm geometry as SI.
producer_file = '';
legacy_format = 'legacy_40_05_mm_geometry_si_blocks';
format = '';
if isfield(source, 'prepared_format'), format = local_text(source.prepared_format); end
if isfield(p, 'fmdl_m')
    if isfield(p, 'fmdl_mm') || strcmp(format, legacy_format)
        error('trkg4:surfacePreparationModel', 'Ambiguous geometry keys or conflicting prepared_format.');
    end
    if ~isfield(p, 'coordinate_units') || ~strcmp(p.coordinate_units, 'm')
        error('trkg4:surfacePreparationModel', 'SI prepared_fem requires coordinate_units=m.');
    end
    if ~isempty(format) && ~strcmp(format, 'si_geometry_si_blocks')
        error('trkg4:surfacePreparationModel', 'Unsupported SI prepared_format.');
    end
    if isfield(source, 'prepared_coordinate_units') && ...
            ~strcmp(source.prepared_coordinate_units, 'm')
        error('trkg4:surfacePreparationModel', 'Manifest coordinate units conflict with SI geometry.');
    end
    local_validate_prepared(p);
    mm = p.fmdl_m;
    mm.nodes = p.fmdl_m.nodes * 1000;
    units = struct('origin', 'prepared_mat_coordinate_units', ...
        'source_variable', 'fmdl_m', 'source_coordinate_units', 'm', ...
        'fem_coordinate_units', 'm', 'volume_block_units', 'SI', 'node_scale_to_m', 1);
    return;
end
if ~isfield(p, 'fmdl_mm') || ~strcmp(format, legacy_format)
    error('trkg4:surfacePreparationModel', ...
        'Missing fmdl_m. Legacy fmdl_mm requires explicit prepared_format=legacy_40_05_mm_geometry_si_blocks.');
end
if (isfield(p, 'coordinate_units') && ~strcmp(p.coordinate_units, 'mm')) || ...
        (isfield(source, 'prepared_coordinate_units') && ...
        ~strcmp(source.prepared_coordinate_units, 'mm'))
    error('trkg4:surfacePreparationModel', 'Legacy geometry must be declared in mm, never m.');
end
if ~all(isfield(source, {'model_identity', 'model_identity_sha256'}))
    error('trkg4:surfacePreparationLegacy', 'Legacy format requires a hash-bound model_identity.');
end
identity_file = local_relative_file(base, source.model_identity);
local_check_hash(identity_file, source.model_identity_sha256);
identity = jsondecode(fileread(identity_file));
if ~isstruct(identity) || ~isscalar(identity) || ...
        ~all(isfield(identity, {'prepared_fem_sha256', 'mesh_sha256', 'sources'})) || ...
        ~strcmpi(identity.prepared_fem_sha256, source.prepared_sha256) || ...
        ~strcmpi(identity.mesh_sha256, source.mesh_sha256) || ...
        ~isstruct(identity.sources) || ~all(isfield(identity.sources, {'path', 'sha256'}))
    error('trkg4:surfacePreparationLegacy', 'model_identity does not bind this prepared MAT and mesh.');
end
producer_path = 'src/run_trkg4_arm_conductivity_matrix.m';
match = strcmp({identity.sources.path}, producer_path);
if nnz(match) ~= 1
    error('trkg4:surfacePreparationLegacy', 'model_identity must name the legacy 40.05 producer exactly once.');
end
producer_hash = identity.sources(match).sha256;
local_sha(producer_hash);
project = fileparts(fileparts(mfilename('fullpath')));
producer_file = fullfile(project, producer_path);
local_check_hash(producer_file, producer_hash);
mm = p.fmdl_mm;
p.fmdl_m = mm;
p.fmdl_m.nodes = mm.nodes * 0.001;
p.coordinate_units = 'm'; % In-memory normalisation only; source MAT stays immutable.
p = rmfield(p, 'fmdl_mm');
units = struct('origin', 'explicit_legacy_format_and_verified_model_identity', ...
    'prepared_format', legacy_format, 'source_variable', 'fmdl_mm', ...
    'source_coordinate_units', 'mm', 'fem_coordinate_units', 'm', ...
    'volume_block_units', 'SI', 'node_scale_to_m', 0.001, ...
    'prepared_sha256', lower(source.prepared_sha256), ...
    'model_identity_sha256', lower(source.model_identity_sha256), ...
    'producer', struct('path', producer_path, 'sha256', lower(producer_hash)));
end

function local_validate_prepared(p)
required = {'fmdl_m', 'blocks', 'tissue_id', 'tissue_names'};
if ~all(isfield(p, required)) || ~isstruct(p.fmdl_m) || ~isscalar(p.fmdl_m)
    error('trkg4:surfacePreparationModel', 'Incomplete prepared_fem.mat.');
end
f = p.fmdl_m;
if ~all(isfield(f, {'nodes', 'elems', 'boundary', 'gnd_node', 'electrode', 'system_mat'})) || ...
        size(f.nodes, 2) ~= 3 || ~isreal(f.nodes) || any(~isfinite(f.nodes), 'all') || ...
        size(f.elems, 2) ~= 4 || numel(f.electrode) ~= 4 || ...
        ~iscell(p.blocks) || numel(p.blocks) ~= 5
    error('trkg4:surfacePreparationModel', ...
        'Expected a linear tetrahedral model with exactly four electrodes and five blocks.');
end
if isfield(p, 'coordinate_units') && ~strcmp(p.coordinate_units, 'm')
    error('trkg4:surfacePreparationModel', 'prepared_fem coordinates must be in metres.');
end
if isnumeric(f.system_mat)
    error('trkg4:surfacePreparationModel', ...
        'A cached numeric system_mat is not an independent EIDORS assembly.');
end
n = size(f.nodes, 1);
if ~isnumeric(f.gnd_node) || ~isscalar(f.gnd_node) || ~isfinite(f.gnd_node) || ...
        f.gnd_node ~= fix(f.gnd_node) || f.gnd_node < 1 || f.gnd_node > n
    error('trkg4:surfacePreparationModel', 'Invalid one-based FEM ground node.');
end
if ~isnumeric(p.tissue_id) || numel(p.tissue_id) ~= size(f.elems, 1) || ...
        any(~isfinite(p.tissue_id(:))) || any(p.tissue_id(:) ~= fix(p.tissue_id(:))) || ...
        any(p.tissue_id(:) < 1 | p.tissue_id(:) > numel(p.tissue_names)) || ...
        isempty(p.tissue_names)
    error('trkg4:surfacePreparationModel', 'Invalid tissue labels or names.');
end
for k = 1:5
    b = p.blocks{k};
    if ~issparse(b) || ~isreal(b) || ~isequal(size(b), [n+4, n+4]) || ...
            any(~isfinite(nonzeros(b))) || nnz(b(n+(1:4), :)) || nnz(b(:, n+(1:4)))
        error('trkg4:surfacePreparationModel', ...
            'Each existing block must be finite, sparse, nn+4 square and volume-only.');
    end
end
end

function local_validate_manifest(m)
local_fields(m, {'schema', 'source', 'contact', 'qc', 'inputs'});
if ~strcmp(local_text(m.schema), 'surface_sensitivity_preparation_v1')
    error('trkg4:surfacePreparationManifest', 'Unsupported manifest schema.');
end
local_fields(m.source, {'prepared_fem', 'prepared_sha256', 'mesh_sha256', 'surface_basename'});
local_fields(m.contact, {'area_mm2', 'z_contact_ohm_m2'});
local_fields(m.qc, {'max_surface_distance_mm', 'max_centroid_offset_mm', ...
    'min_area_fraction', 'max_area_ratio'});
local_sha(m.source.prepared_sha256); local_sha(m.source.mesh_sha256);
name = local_text(m.source.surface_basename);
if ~isempty(regexp(name, '[/\\:]', 'once')) || ismember(name, {'.', '..'})
    error('trkg4:surfacePreparationManifest', 'surface_basename must be a basename.');
end
local_positive(m.contact.area_mm2); local_positive(m.contact.z_contact_ohm_m2);
local_positive(m.qc.max_surface_distance_mm); local_positive(m.qc.max_centroid_offset_mm);
local_positive(m.qc.min_area_fraction); local_positive(m.qc.max_area_ratio);
if m.qc.min_area_fraction > 1 || m.qc.max_area_ratio < 1
    error('trkg4:surfacePreparationManifest', 'Invalid contact area QC interval.');
end
if ~isstruct(m.inputs) || isempty(m.inputs)
    error('trkg4:surfacePreparationManifest', 'At least one montage input is required.');
end
input_count = numel(m.inputs);
ids = cell(input_count, 1);
for k = 1:input_count
    local_fields(m.inputs(k), {'id', 'json', 'sha256'});
    ids{k} = local_text(m.inputs(k).id);
    if isempty(regexp(ids{k}, '^[a-z][a-z0-9_-]{0,63}$', 'once'))
        error('trkg4:surfacePreparationManifest', 'Invalid montage id.');
    end
    local_sha(m.inputs(k).sha256);
end
if numel(unique(ids)) ~= input_count
    error('trkg4:surfacePreparationManifest', 'Montage ids must be unique.');
end
end

function local_fields(s, names)
if ~isstruct(s) || ~isscalar(s) || ~all(isfield(s, names))
    error('trkg4:surfacePreparationManifest', 'Missing required manifest fields.');
end
end

function local_positive(value)
if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || value <= 0
    error('trkg4:surfacePreparationManifest', 'Expected a finite positive numeric scalar.');
end
end

function value = local_text(value)
if ~(ischar(value) && isrow(value) && ~isempty(value)) && ...
        ~(isstring(value) && isscalar(value) && ~ismissing(value) && strlength(value) > 0)
    error('trkg4:surfacePreparationManifest', 'Expected a nonempty text scalar.');
end
value = char(value);
end

function local_sha(value)
if isempty(regexp(local_text(value), '^[a-fA-F0-9]{64}$', 'once'))
    error('trkg4:surfacePreparationManifest', 'Expected a SHA-256 hexadecimal digest.');
end
end

function path = local_relative_file(base, value)
value = local_text(value);
if ~isempty(regexp(value, '(^[/\\])|:', 'once'))
    error('trkg4:surfacePreparationManifest', 'Data paths must be relative to the manifest.');
end
path = local_canonical(fullfile(base, strrep(value, '/', filesep)));
end

function path = local_canonical(path)
path = char(java.io.File(path).getCanonicalPath());
end

function yes = local_path_starts_with(path, root)
if ispc, yes = startsWith(lower(path), lower(root));
else, yes = startsWith(path, root); end
end

function local_check_hash(file, expected)
if ~strcmpi(trkg4_file_sha256(file), expected)
    error('trkg4:surfacePreparationHash', 'SHA-256 mismatch for %s.', file);
end
end

function [files, hashes, names] = local_source_files(base, source)
files = {}; hashes = {}; names = {};
optional = {'mesh', 'surface', 'mesh_acceptance', 'model_identity'};
for k = 1:numel(optional)
    name = optional{k};
    if ~isfield(source, name), continue; end
    hash_name = [name, '_sha256'];
    local_fields(source, {hash_name});
    local_sha(source.(hash_name));
    path = local_relative_file(base, source.(name));
    if strcmp(name, 'surface')
        [~, stem, extension] = fileparts(path);
        if ~strcmp([stem, extension], source.surface_basename)
            error('trkg4:surfacePreparationManifest', ...
                'source.surface filename differs from source.surface_basename.');
        end
    end
    files{end+1} = path; %#ok<AGROW>
    hashes{end+1} = source.(hash_name); %#ok<AGROW>
    names{end+1} = name; %#ok<AGROW>
end
end

function local_write_json(file, value)
if isfile(file)
    error('trkg4:outputWouldOverwrite', 'Refusing to overwrite %s.', file);
end
fid = fopen(file, 'w', 'n', 'UTF-8');
if fid < 0, error('trkg4:surfacePreparationWrite', 'Cannot write %s.', file); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
