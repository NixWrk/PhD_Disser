function report = trkg4_export_refined_surface_model(manifest_file, refined_mesh_file, output_directory)
%TRKG4_EXPORT_REFINED_SURFACE_MODEL Export SI blocks on a verified refinement.
%
% report = trkg4_export_refined_surface_model(manifest_file, refined_mesh_file, out)
% Reads only fmdl_mm and tissue_names from the parent's legacy 40.05 MAT.
% refined_mesh_file must contain nodes_mm (N-by-3), elems (T-by-4), boundary
% (B-by-3), tissue_id and group_id (one label per tet, groups 1..5).
% All mesh indices are MATLAB one-based. Original nodes must be an unchanged
% prefix, preserving the original ground node and its physical position.
%
% Required sibling refinement.json: status='passed_geometry_preserving_local_refinement',
% parent_prepared_sha256, refined_mesh_sha256, counts.nodes/tetrahedra/boundary,
% and all five named geometric checks true (see local_refinement_checks). Optional
% manifest_sha256 and group_counts are verified when present. This sidecar
% attests geometry preservation/conformity; the MATLAB checks do not replace
% that geometric verification or establish mesh convergence.
%
% First manifest montage supplies four v3 explicit_points centres and the
% contact area/z_contact. Canonical patch construction and strict diagnostics
% run before trkg4_export_sensitivity_model creates five NEW SI volume blocks.
% No parent volume blocks are loaded, reused, or modified. No factorisation.
%
% Output must be new/empty. Files: prepared_fem.mat, contact.mat,
% first_montage_diagnostics.csv and (written LAST, only after success)
% refinement_export.json. Failure throws; partial outputs must not be reused.
% The new prepared file uses source.prepared_format='si_geometry_si_blocks'.

manifest_file = local_path(manifest_file);
refined_mesh_file = local_path(refined_mesh_file);
output_directory = local_path(output_directory);
local_empty_output(output_directory);
base = fileparts(manifest_file);
manifest = jsondecode(fileread(manifest_file));
local_require(manifest, {'schema', 'source', 'inputs', 'contact', 'qc'});
assert(strcmp(manifest.schema, 'surface_sensitivity_preparation_v1'), ...
    'trkg4:refinedExportManifest', 'Unsupported manifest schema.');
source = manifest.source;
local_require(source, {'prepared_fem', 'prepared_sha256', 'prepared_format', 'surface_basename'});
assert(strcmp(source.prepared_format, 'legacy_40_05_mm_geometry_si_blocks'), ...
    'trkg4:refinedExportManifest', 'Parent must explicitly declare legacy 40.05 mm geometry / SI blocks.');
assert(isstruct(manifest.inputs) && ~isempty(manifest.inputs), ...
    'trkg4:refinedExportManifest', 'At least one montage is required.');
input = manifest.inputs(1);
local_require(input, {'id', 'json', 'sha256'});
local_require(manifest.contact, {'area_mm2', 'z_contact_ohm_m2'});
local_require(manifest.qc, {'max_surface_distance_mm', 'max_centroid_offset_mm', ...
    'min_area_fraction', 'max_area_ratio'});
values = [manifest.contact.area_mm2, manifest.contact.z_contact_ohm_m2, ...
    manifest.qc.max_surface_distance_mm, manifest.qc.max_centroid_offset_mm, ...
    manifest.qc.min_area_fraction, manifest.qc.max_area_ratio];
assert(isnumeric(values) && isreal(values) && numel(values) == 6 && ...
    all(isfinite(values)) && all(values > 0) && ...
    manifest.qc.min_area_fraction <= 1 && manifest.qc.max_area_ratio >= 1, ...
    'trkg4:refinedExportManifest', 'Invalid contact parameters or geometry QC thresholds.');
parent_file = local_relative(base, source.prepared_fem);
json_file = local_relative(base, input.json);
qc_file = fullfile(fileparts(refined_mesh_file), 'refinement.json');
manifest_hash = trkg4_file_sha256(manifest_file);
parent_hash = local_hash(parent_file, source.prepared_sha256);
input_hash = local_hash(json_file, input.sha256);
refined_hash = trkg4_file_sha256(refined_mesh_file);
if isfield(source, 'refined_mesh_sha256')
    local_hash(refined_mesh_file, source.refined_mesh_sha256);
end
qc_hash = trkg4_file_sha256(qc_file);
refinement = jsondecode(fileread(qc_file));
local_require(refinement, {'status', 'parent_prepared_sha256', 'refined_mesh_sha256', 'counts', 'checks'});
local_refinement_checks(refinement.checks);
assert(strcmp(refinement.status, 'passed_geometry_preserving_local_refinement') && ...
    strcmpi(refinement.parent_prepared_sha256, parent_hash) && ...
    strcmpi(refinement.refined_mesh_sha256, refined_hash), ...
    'trkg4:refinedExportQC', 'Refinement QC failed or is not bound to these parent/refined files.');
if isfield(refinement, 'manifest_sha256')
    assert(strcmpi(refinement.manifest_sha256, manifest_hash), ...
        'trkg4:refinedExportQC', 'Refinement QC belongs to another manifest.');
end
local_require(refinement.counts, {'nodes', 'tetrahedra', 'boundary'});

project = fileparts(fileparts(mfilename('fullpath')));
assert(strcmp(local_path(which('electrode_faces_by_area')), ...
    local_path(fullfile(project, 'vendor_stl_eidors', 'electrode_faces_by_area.m'))), ...
    'trkg4:refinedExportDependency', 'The canonical patch builder must be on the MATLAB path.');
code_names = {'trkg4_export_refined_surface_model', 'trkg4_export_sensitivity_model', ...
    'trkg4_load_surface_landmarks', 'electrode_faces_by_area', 'electrode_patch_context', ...
    'trkg4_electrode_diagnostics', 'trkg4_boundary_normals', ...
    'trkg4_cem_contact_matrix', 'trkg4_file_sha256', 'system_mat_fields', ...
    'system_mat_1st_order', 'mk_image', 'calc_system_mat'};
code_files = cellfun(@which, code_names, 'UniformOutput', false);
assert(all(~cellfun(@isempty, code_files)), ...
    'trkg4:refinedExportDependency', 'Required MATLAB/EIDORS functions are missing.');
code_hashes = cellfun(@trkg4_file_sha256, code_files, 'UniformOutput', false);

% Selective load: never materialise the parent's large stiffness blocks.
parent = load(parent_file, 'fmdl_mm', 'tissue_names');
local_require(parent, {'fmdl_mm', 'tissue_names'});
old = parent.fmdl_mm;
local_require(old, {'nodes', 'elems', 'boundary', 'gnd_node', 'system_mat'});
assert(~isnumeric(old.system_mat), 'trkg4:refinedExportModel', ...
    'A cached numeric system matrix cannot be inherited.');
tissue_names = parent.tissue_names;
old_counts = struct('nodes', size(old.nodes, 1), 'elements', size(old.elems, 1), ...
    'boundary_faces', size(old.boundary, 1));
new = load(refined_mesh_file, 'nodes_mm', 'elems', 'boundary', 'tissue_id', 'group_id');
local_require(new, {'nodes_mm', 'elems', 'boundary', 'tissue_id', 'group_id'});
assert(isnumeric(new.nodes_mm) && isreal(new.nodes_mm) && ...
    size(new.nodes_mm, 2) == 3 && all(isfinite(new.nodes_mm), 'all'), ...
    'trkg4:refinedExportMesh', 'nodes_mm must be a finite N-by-3 array.');
counts = struct('nodes', size(new.nodes_mm, 1), 'elements', size(new.elems, 1), ...
    'boundary_faces', size(new.boundary, 1));
assert(counts.nodes >= old_counts.nodes && counts.elements >= old_counts.elements && ...
    isequal(new.nodes_mm(1:old_counts.nodes, :), old.nodes), ...
    'trkg4:refinedExportMesh', 'Original nodes must remain unchanged, in their original order.');
names = {'nodes', 'elements', 'boundary_faces'};
qc_names = {'nodes', 'tetrahedra', 'boundary'};
for k = 1:numel(names)
    assert(isequal(double(refinement.counts.(qc_names{k})), double(counts.(names{k}))), ...
        'trkg4:refinedExportQC', 'Refinement QC counts do not match the refined MAT.');
end
local_indices(new.elems, counts.nodes, 4);
local_indices(new.boundary, counts.nodes, 3);
tissue_id = double(new.tissue_id(:));
group_id = double(new.group_id(:));
assert(isnumeric(new.tissue_id) && isreal(new.tissue_id) && ...
    numel(tissue_id) == counts.elements && all(isfinite(tissue_id)) && ...
    all(tissue_id == fix(tissue_id)) && all(tissue_id >= 1 & tissue_id <= numel(tissue_names)), ...
    'trkg4:refinedExportGroups', 'Invalid inherited tissue labels.');
assert(isnumeric(new.group_id) && isreal(new.group_id) && ...
    numel(group_id) == counts.elements && all(isfinite(group_id)) && ...
    all(group_id == fix(group_id)) && isequal(unique(group_id), (1:5)'), ...
    'trkg4:refinedExportGroups', 'group_id must partition all tetrahedra into five nonempty groups.');
groups = arrayfun(@(k) group_id == k, (1:5)', 'UniformOutput', false);
group_counts = cellfun(@nnz, groups);
if isfield(refinement, 'group_counts')
    assert(isequal(double(refinement.group_counts(:)), group_counts), ...
        'trkg4:refinedExportQC', 'Refinement QC group counts do not match the refined MAT.');
end

% Carry solver/measurement metadata only. All element/node-dependent fields
% (including unknown cached mappings) are discarded rather than copied.
keep = {'type', 'name', 'solve', 'jacobian', 'system_mat', ...
    'normalize_measurements', 'stimulation', 'meas_select', 'gnd_node'};
keep = intersect(keep, fieldnames(old), 'stable');
removed_fields = setdiff(fieldnames(old), [keep(:); {'nodes'; 'elems'; 'boundary'; 'electrode'}]);
fmdl_mm = struct();
for k = 1:numel(keep), fmdl_mm.(keep{k}) = old.(keep{k}); end
fmdl_mm.nodes = double(new.nodes_mm);
fmdl_mm.elems = double(new.elems);
fmdl_mm.boundary = double(new.boundary);
ground = old.gnd_node;
assert(isnumeric(ground) && isscalar(ground) && isfinite(ground) && ...
    ground == fix(ground) && ground >= 1 && ground <= old_counts.nodes, ...
    'trkg4:refinedExportMesh', 'The original ground must be a valid unchanged tissue node.');
clear old parent new group_id;

% The sidecar checks geometry preservation and conformity. Independently
% reject internal/duplicated/nonmanifold boundary faces before any export.
trkg4_boundary_normals(fmdl_mm);
landmarks = trkg4_load_surface_landmarks(json_file);
roles = {'I_plus', 'V_plus', 'V_minus', 'I_minus'};
assert(isfield(landmarks, 'schema') && strcmp(landmarks.schema, 'trkg4_ttrkg_surface_v3') && ...
    strcmp(landmarks.placement_mode, 'explicit_points') && ...
    strcmp(landmarks.source_basename, source.surface_basename) && ...
    isequal(cellstr(landmarks.electrode_order(:)), roles(:)), ...
    'trkg4:refinedExportLandmarks', 'First montage must be v3 explicit points on the declared surface.');
requested = landmarks.requested_centres_xyz_mm;
[fmdl_mm.electrode, induced] = electrode_faces_by_area(fmdl_mm, requested, ...
    manifest.contact.area_mm2, manifest.contact.z_contact_ohm_m2);
assert(numel(fmdl_mm.electrode) == 4, 'trkg4:refinedExportContacts', 'Expected exactly four contacts.');
for a = 1:4
    for b = a+1:4
        assert(isempty(intersect(fmdl_mm.electrode(a).nodes, fmdl_mm.electrode(b).nodes)), ...
            'trkg4:refinedExportContacts', 'Refined contacts share FEM nodes.');
    end
end
cfg = struct('length_unit', 'mm', 'electrode_area', manifest.contact.area_mm2, ...
    'max_electrode_surface_distance_mm', manifest.qc.max_surface_distance_mm, ...
    'max_electrode_patch_centroid_offset_mm', manifest.qc.max_centroid_offset_mm, ...
    'min_electrode_patch_area_fraction', manifest.qc.min_area_fraction, ...
    'max_electrode_patch_area_ratio', manifest.qc.max_area_ratio, ...
    'fail_on_electrode_diagnostics', true, 'electrode_diagnostics_verbose', true);
diagnostics = trkg4_electrode_diagnostics(fmdl_mm, requested, roles, cfg, requested, induced);
fmdl_m = fmdl_mm;
fmdl_m.nodes = fmdl_mm.nodes * 0.001;
clear fmdl_mm induced;

% Canonical exporter computes NEW five blocks, checks against actual EIDORS,
% and binds its first contact matrix to the newly written prepared MAT.
local_empty_output(output_directory);
exported = trkg4_export_sensitivity_model(fmdl_m, tissue_id, tissue_names, groups, output_directory);
clear fmdl_m groups tissue_id;
assert(strcmp(exported.status, 'exported_algebraic_check_passed') && exported.matrix_error < 1e-12, ...
    'trkg4:refinedExportAssembly', 'The canonical EIDORS assembly check did not pass.');
diagnostics_file = fullfile(output_directory, 'first_montage_diagnostics.csv');
writetable(diagnostics, diagnostics_file);

% Check drift again after the potentially long assembly and before the seal.
local_hash(manifest_file, manifest_hash);
local_hash(parent_file, parent_hash);
local_hash(refined_mesh_file, refined_hash);
local_hash(qc_file, qc_hash);
local_hash(json_file, input_hash);
for k = 1:numel(code_files), local_hash(code_files{k}, code_hashes{k}); end
report = struct('schema', 'refined_surface_model_export_v1', 'status', 'passed', ...
    'manifest_sha256', manifest_hash, 'parent_prepared_sha256', parent_hash, ...
    'refined_input_sha256', refined_hash, 'refined_mesh_sha256', refined_hash, ...
    'refinement_qc_sha256', qc_hash, 'prepared_sha256', exported.prepared_sha256, ...
    'new_prepared_sha256', exported.prepared_sha256, 'contact_sha256', exported.contact_sha256, ...
    'prepared_format', 'si_geometry_si_blocks', 'prepared_fem', 'prepared_fem.mat', ...
    'contact_matrix', 'contact.mat', 'first_montage_id', input.id, 'source_json_sha256', input_hash, ...
    'parent_counts', old_counts, 'counts', counts, 'group_counts', group_counts, ...
    'ground', ground, 'matrix_error', exported.matrix_error, ...
    'geometry_qc', 'passed_refinement_sidecar_and_boundary_and_first_contacts', ...
    'algebraic_qc', 'passed', 'mesh_convergence', 'untested', ...
    'factorisation_performed', false, 'fem_sweep_performed', false, ...
    'diagnostics_file', 'first_montage_diagnostics.csv', ...
    'diagnostics_sha256', trkg4_file_sha256(diagnostics_file));
report.units = struct('input_coordinates', 'mm', 'prepared_coordinates', 'm', ...
    'node_scale_to_m', 0.001, 'contact_impedance', 'ohm*m^2', 'volume_blocks', 'SI');
report.implementation = struct('function_name', code_names, 'sha256', code_hashes);
report.removed_parent_fields = removed_fields;
report.diagnostics = table2struct(diagnostics);
report.warnings = {'Only the first montage was checked here; all montages require contact preparation.', ...
    'Conforming geometry refinement does not establish FEM convergence or experimental validity.', ...
    'CT/STL voxel spacing is not FEM element size. No tissue resegmentation was performed.'};
local_json(fullfile(output_directory, 'refinement_export.json'), report);
end

function local_refinement_checks(checks)
required = {'original_vertices_unchanged', 'all_positive_tetrahedra', ...
    'group_volumes_preserved', 'conforming_boundary_verified', ...
    'contact_neighbourhood_edges_le_1mm'};
local_require(checks, required);
for k = 1:numel(required)
    value = checks.(required{k});
    assert(islogical(value) && isscalar(value) && value, ...
        'trkg4:refinedExportQC', 'Required refinement check did not pass: %s.', required{k});
end
end

function local_require(s, fields)
assert(isstruct(s) && isscalar(s) && all(isfield(s, fields)), ...
    'trkg4:refinedExportContract', 'Required fields are missing from the manifest, QC or MAT.');
end

function local_indices(indices, n, width)
assert(isnumeric(indices) && isreal(indices) && ~isempty(indices) && size(indices, 2) == width && ...
    all(isfinite(indices), 'all') && all(indices == fix(indices), 'all') && ...
    all(indices >= 1 & indices <= n, 'all') && ...
    all(diff(sort(indices, 2), 1, 2) ~= 0, 'all'), ...
    'trkg4:refinedExportMesh', 'Invalid or repeated one-based node indices.');
end

function path = local_path(path)
assert((ischar(path) && isrow(path) && ~isempty(path)) || ...
    (isstring(path) && isscalar(path) && ~ismissing(path) && strlength(path) > 0), ...
    'trkg4:refinedExportPath', 'A nonempty scalar path is required.');
path = char(java.io.File(char(path)).getCanonicalPath());
end

function path = local_relative(base, relative)
relative = char(relative);
assert(isrow(relative) && ~isempty(relative) && isempty(regexp(relative, '(^[/\\])|:', 'once')), ...
    'trkg4:refinedExportPath', 'Manifest data paths must be relative.');
path = local_path(fullfile(base, strrep(relative, '/', filesep)));
end

function hash = local_hash(file, expected)
assert(ischar(expected) && ~isempty(regexp(expected, '^[a-fA-F0-9]{64}$', 'once')), ...
    'trkg4:refinedExportHash', 'Expected a SHA-256 digest.');
hash = trkg4_file_sha256(file);
assert(strcmpi(hash, expected), 'trkg4:refinedExportHash', 'SHA-256 mismatch: %s.', file);
end

function local_empty_output(folder)
assert(~isfile(folder), 'trkg4:outputWouldOverwrite', 'Output path exists as a file.');
if isfolder(folder)
    entries = dir(folder);
    assert(all(ismember({entries.name}, {'.', '..'})), ...
        'trkg4:outputWouldOverwrite', 'Output directory must be new or empty.');
end
end

function local_json(file, value)
assert(~isfile(file), 'trkg4:outputWouldOverwrite', 'Refusing to overwrite the export seal.');
fid = fopen(file, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'trkg4:refinedExportWrite', 'Cannot create export seal.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
