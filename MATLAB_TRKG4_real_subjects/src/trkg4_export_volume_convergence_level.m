function report = trkg4_export_volume_convergence_level( ...
    manifest_file, refined_mesh_file, refinement_report_file, output_directory, level_id)
%TRKG4_EXPORT_VOLUME_CONVERGENCE_LEVEL Build new SI blocks on an independent mesh.
%
% The function uses a previously prepared TEPC model only as a source of
% solver metadata and tissue names. Nodes, elements, boundary, tissue labels,
% volume stiffness blocks and contacts are rebuilt for this mesh level.

manifest_file = local_path(manifest_file);
refined_mesh_file = local_path(refined_mesh_file);
refinement_report_file = local_path(refinement_report_file);
output_directory = local_path(output_directory);
level_id = char(string(level_id));
assert(~isempty(level_id), 'trkg4:volumeExportInput', 'A level ID is required.');
local_empty_output(output_directory);

base = fileparts(manifest_file);
manifest = jsondecode(fileread(manifest_file));
assert(strcmp(manifest.schema, 'surface_sensitivity_preparation_v1') && ...
    numel(manifest.inputs) >= 1, 'trkg4:volumeExportManifest', ...
    'Expected a TEPC surface sensitivity manifest.');
manifest_hash = trkg4_file_sha256(manifest_file);
parent_file = local_relative(base, manifest.source.prepared_fem);
assert(strcmpi(trkg4_file_sha256(parent_file), manifest.source.prepared_sha256), ...
    'trkg4:volumeExportHash', 'Parent prepared FEM hash mismatch.');
json_file = local_relative(base, manifest.inputs(1).json);
assert(strcmpi(trkg4_file_sha256(json_file), manifest.inputs(1).sha256), ...
    'trkg4:volumeExportHash', 'First montage JSON hash mismatch.');

mesh_hash = trkg4_file_sha256(refined_mesh_file);
refinement_hash = trkg4_file_sha256(refinement_report_file);
refinement = jsondecode(fileread(refinement_report_file));
assert(strcmp(refinement.status, 'passed_contact_refinement_for_volume_convergence') && ...
    strcmpi(refinement.refined_mesh_sha256, mesh_hash) && ...
    all(structfun(@(x) logical(x), refinement.checks)), ...
    'trkg4:volumeExportQC', 'Independent mesh refinement QC failed.');

parent = load(parent_file, 'fmdl_m', 'tissue_names');
assert(isfield(parent, 'fmdl_m') && isfield(parent, 'tissue_names'), ...
    'trkg4:volumeExportParent', 'Parent prepared FEM lacks solver metadata.');
old = parent.fmdl_m;
new = load(refined_mesh_file, 'nodes_mm', 'elems', 'boundary', 'tissue_id', 'group_id');
required = {'nodes_mm', 'elems', 'boundary', 'tissue_id', 'group_id'};
assert(all(isfield(new, required)), 'trkg4:volumeExportMesh', ...
    'Refined mesh is incomplete.');

nodes_mm = double(new.nodes_mm);
elems = double(new.elems);
boundary = double(new.boundary);
tissue_id = double(new.tissue_id(:));
group_id = double(new.group_id(:));
assert(size(nodes_mm, 2) == 3 && size(elems, 2) == 4 && size(boundary, 2) == 3 && ...
    numel(tissue_id) == size(elems, 1) && numel(group_id) == size(elems, 1), ...
    'trkg4:volumeExportMesh', 'Unexpected mesh array dimensions.');
assert(all(isfinite(nodes_mm), 'all') && all(isfinite(elems), 'all') && ...
    all(isfinite(boundary), 'all') && all(isfinite(tissue_id)) && all(isfinite(group_id)), ...
    'trkg4:volumeExportMesh', 'Mesh arrays must be finite.');
local_indices(elems, size(nodes_mm, 1), 4);
local_indices(boundary, size(nodes_mm, 1), 3);
assert(isequal(unique(tissue_id), (1:numel(parent.tissue_names))') && ...
    isequal(unique(group_id), (1:5)'), 'trkg4:volumeExportGroups', ...
    'Tissue and five-group labels must be consecutive and nonempty.');
groups = arrayfun(@(k) group_id == k, (1:5)', 'UniformOutput', false);

keep = {'type', 'solve', 'jacobian', 'system_mat', 'normalize_measurements'};
keep = intersect(keep, fieldnames(old), 'stable');
fmdl_mm = struct();
for k = 1:numel(keep), fmdl_mm.(keep{k}) = old.(keep{k}); end
fmdl_mm.name = sprintf('Nix TEPC independent volume mesh %s', level_id);
fmdl_mm.nodes = nodes_mm;
fmdl_mm.elems = elems;
fmdl_mm.boundary = boundary;
old_ground_mm = old.nodes(old.gnd_node, :) * 1000;
[ground_distance_mm, new_ground] = min(vecnorm(nodes_mm - old_ground_mm, 2, 2));
fmdl_mm.gnd_node = new_ground;
clear old new group_id;
trkg4_boundary_normals(fmdl_mm);

landmarks = trkg4_load_surface_landmarks(json_file);
roles = {'I_plus', 'V_plus', 'V_minus', 'I_minus'};
assert(strcmp(landmarks.schema, 'trkg4_ttrkg_surface_v3') && ...
    strcmp(landmarks.placement_mode, 'explicit_points') && ...
    isequal(cellstr(landmarks.electrode_order(:)), roles(:)), ...
    'trkg4:volumeExportLandmarks', 'Unexpected first montage landmarks.');
requested = landmarks.requested_centres_xyz_mm;
[fmdl_mm.electrode, induced] = electrode_faces_by_area(fmdl_mm, requested, ...
    manifest.contact.area_mm2, manifest.contact.z_contact_ohm_m2);
for a = 1:4
    for b = a+1:4
        assert(isempty(intersect(fmdl_mm.electrode(a).nodes, fmdl_mm.electrode(b).nodes)), ...
            'trkg4:volumeExportContacts', 'Contacts share FEM nodes.');
    end
end
qc_cfg = struct('length_unit', 'mm', 'electrode_area', manifest.contact.area_mm2, ...
    'max_electrode_surface_distance_mm', manifest.qc.max_surface_distance_mm, ...
    'max_electrode_patch_centroid_offset_mm', manifest.qc.max_centroid_offset_mm, ...
    'min_electrode_patch_area_fraction', manifest.qc.min_area_fraction, ...
    'max_electrode_patch_area_ratio', manifest.qc.max_area_ratio, ...
    'fail_on_electrode_diagnostics', true, 'electrode_diagnostics_verbose', true);
diagnostics = trkg4_electrode_diagnostics( ...
    fmdl_mm, requested, roles, qc_cfg, requested, induced);

scale_cfg = struct('scale_nodes_to_si', true, 'length_unit', 'mm');
fmdl_m = trkg4_scale_fmdl_to_si(fmdl_mm, scale_cfg);
tissue_names = parent.tissue_names;
clear fmdl_mm parent induced nodes_mm elems boundary;

local_empty_output(output_directory);
exported = trkg4_export_sensitivity_model( ...
    fmdl_m, tissue_id, tissue_names, groups, output_directory);
assert(strcmp(exported.status, 'exported_algebraic_check_passed') && ...
    exported.matrix_error < 1e-12, 'trkg4:volumeExportAssembly', ...
    'EIDORS algebraic assembly check failed.');
clear fmdl_m tissue_id groups;

diagnostics_file = fullfile(output_directory, 'first_montage_diagnostics.csv');
writetable(diagnostics, diagnostics_file);
code_names = {'trkg4_export_volume_convergence_level', ...
    'trkg4_export_sensitivity_model', 'trkg4_load_surface_landmarks', ...
    'electrode_faces_by_area', 'electrode_patch_context', ...
    'trkg4_electrode_diagnostics', 'trkg4_boundary_normals', ...
    'trkg4_cem_contact_matrix', 'trkg4_scale_fmdl_to_si', ...
    'system_mat_fields', 'mk_image', 'calc_system_mat'};
implementation = repmat(struct('function_name', '', 'sha256', ''), numel(code_names), 1);
for k = 1:numel(code_names)
    implementation(k).function_name = code_names{k};
    implementation(k).sha256 = trkg4_file_sha256(which(code_names{k}));
end

report = struct();
report.schema = 'tepc_volume_convergence_export_v1';
report.status = 'passed_independent_volume_level_export';
report.level_id = level_id;
report.manifest_sha256 = manifest_hash;
report.refined_mesh_sha256 = mesh_hash;
report.refinement_report_sha256 = refinement_hash;
report.prepared_fem = 'prepared_fem.mat';
report.prepared_fem_sha256 = exported.prepared_sha256;
report.first_contact = 'contact.mat';
report.first_contact_sha256 = exported.contact_sha256;
report.matrix_error = exported.matrix_error;
report.counts = exported.counts;
report.group_element_counts = exported.group_element_counts(:)';
report.ground = struct('node', new_ground, 'distance_from_baseline_ground_mm', ground_distance_mm);
report.first_montage_id = manifest.inputs(1).id;
report.diagnostics_file = 'first_montage_diagnostics.csv';
report.diagnostics_sha256 = trkg4_file_sha256(diagnostics_file);
report.diagnostics = table2struct(diagnostics);
report.implementation = implementation;
report.geometry_qc = 'passed_independent_mesh_refinement_and_contact_diagnostics';
report.algebraic_qc = 'passed';
report.mesh_convergence = 'pending_cross_level_analysis';
report.physical_validation = false;
local_json(fullfile(output_directory, 'volume_export.json'), report);
end

function path = local_path(value)
assert((ischar(value) && isrow(value) && ~isempty(value)) || ...
    (isstring(value) && isscalar(value) && strlength(value) > 0), ...
    'trkg4:volumeExportPath', 'A nonempty scalar path is required.');
path = char(java.io.File(char(value)).getCanonicalPath());
end

function path = local_relative(base, relative)
relative = char(relative);
assert(isrow(relative) && ~isempty(relative) && ...
    isempty(regexp(relative, '(^[/\\])|:', 'once')), ...
    'trkg4:volumeExportPath', 'Manifest paths must be relative.');
path = local_path(fullfile(base, strrep(relative, '/', filesep)));
end

function local_indices(indices, n, width)
assert(isnumeric(indices) && isreal(indices) && ~isempty(indices) && ...
    size(indices, 2) == width && all(indices == fix(indices), 'all') && ...
    all(indices >= 1 & indices <= n, 'all') && ...
    all(diff(sort(indices, 2), 1, 2) ~= 0, 'all'), ...
    'trkg4:volumeExportMesh', 'Invalid mesh indices.');
end

function local_empty_output(folder)
assert(~isfile(folder), 'trkg4:outputWouldOverwrite', 'Output path is a file.');
if isfolder(folder)
    entries = dir(folder);
    entries = entries(~ismember({entries.name}, {'.', '..'}));
    assert(isempty(entries), 'trkg4:outputWouldOverwrite', ...
        'Output directory must be new or empty.');
end
end

function local_json(file, value)
assert(~isfile(file), 'trkg4:outputWouldOverwrite', 'Refusing to overwrite export report.');
fid = fopen(file, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'trkg4:volumeExportWrite', 'Cannot write export report.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
