function report = trkg4_classify_volume_convergence_mesh(mesh_file, output_directory, level_id, trunk_size_mm)
%TRKG4_CLASSIFY_VOLUME_CONVERGENCE_MESH Import and label one independent mesh.
%
% The outer surface and CT-derived tissue STL files remain fixed. Only the
% Gmsh volume size changes between levels. The function writes a labelled
% mesh for subsequent identical contact refinement and a hash-bound QC report.

mesh_file = local_path(mesh_file);
output_directory = local_path(output_directory);
level_id = char(string(level_id));
assert(~isempty(level_id) && isfinite(trunk_size_mm) && trunk_size_mm > 0, ...
    'trkg4:volumeConvergenceInput', 'A level ID and positive trunk size are required.');
local_empty_output(output_directory);

cfg = trkg4_config('nik');
trkg4_prepare_runtime(cfg);
cfg.use_parallel_assignment = true;

mesh_hash = trkg4_file_sha256(mesh_file);
fmdl_mm = gmsh_mk_fwd_model(mesh_file, [], [], []);
fmdl_mm = remove_unused_nodes(fmdl_mm);
fmdl_mm.name = sprintf('Nix TEPC volume convergence %s, trunk %.6g mm', level_id, trunk_size_mm);
TR = triangulation(fmdl_mm.elems, fmdl_mm.nodes);
fmdl_mm.boundary = freeBoundary(TR);

nodes_mm = double(fmdl_mm.nodes);
elems = double(fmdl_mm.elems);
boundary = double(fmdl_mm.boundary);
clear TR fmdl_mm;
assert(size(nodes_mm, 2) == 3 && size(elems, 2) == 4 && size(boundary, 2) == 3, ...
    'trkg4:volumeConvergenceMesh', 'Unexpected mesh dimensions.');
assert(all(isfinite(nodes_mm), 'all') && all(isfinite(elems), 'all') && ...
    all(isfinite(boundary), 'all'), 'trkg4:volumeConvergenceMesh', ...
    'Mesh arrays must be finite.');

p0 = nodes_mm(elems(:, 1), :);
e1 = nodes_mm(elems(:, 2), :) - p0;
e2 = nodes_mm(elems(:, 3), :) - p0;
e3 = nodes_mm(elems(:, 4), :) - p0;
signed_volume_mm3 = dot(cross(e1, e2, 2), e3, 2) / 6;
edge_scale = vecnorm(e1, 2, 2) .* vecnorm(e2, 2, 2) .* vecnorm(e3, 2, 2);
assert(all(isfinite(signed_volume_mm3)) && ...
    all(signed_volume_mm3 > 128 * eps .* edge_scale), ...
    'trkg4:volumeConvergenceMesh', 'Nonpositive or numerically degenerate tetrahedra.');

centres = (nodes_mm(elems(:, 1), :) + nodes_mm(elems(:, 2), :) + ...
    nodes_mm(elems(:, 3), :) + nodes_mm(elems(:, 4), :)) / 4;
[~, tissue_id, tissue_names, tissue_counts] = assign_tissues_from_stl(centres, cfg);
expected_names = {'soft_tissue', 'lungs', 'heart', 'bones'};
assert(isequal(tissue_names(:), expected_names(:)), ...
    'trkg4:volumeConvergenceTissues', ...
    'Expected soft tissue, lungs, whole heart and bones; no separate blood mask.');

spec = trkg4_arm_montage_spec(cfg, 'point_disc_5mm', 20, 80);
arms = spec.arm_geometry;
distal = centres(:, 1) > arms.right.cut_x | centres(:, 1) < arms.left.cut_x;
soft_id = find(strcmp(tissue_names, 'soft_tissue'));
lung_id = find(strcmp(tissue_names, 'lungs'));
heart_id = find(strcmp(tissue_names, 'heart'));
bone_id = find(strcmp(tissue_names, 'bones'));
groups = {tissue_id == soft_id & ~distal, tissue_id == soft_id & distal, ...
    tissue_id == heart_id, tissue_id == lung_id, tissue_id == bone_id};
coverage = sum(cell2mat(groups), 2);
assert(all(coverage == 1) && all(cellfun(@any, groups)), ...
    'trkg4:volumeConvergenceGroups', 'The five groups must partition all tetrahedra.');
group_id = zeros(size(tissue_id), 'uint8');
for k = 1:numel(groups), group_id(groups{k}) = k; end

group_counts = cellfun(@nnz, groups);
group_volumes_mm3 = cellfun(@(mask) sum(signed_volume_mm3(mask)), groups);
tissue_volumes_mm3 = accumarray(tissue_id, signed_volume_mm3, ...
    [numel(tissue_names), 1], @sum, 0);
clear centres groups coverage p0 e1 e2 e3 edge_scale;

if ~isfolder(output_directory), mkdir(output_directory); end
classified_file = fullfile(output_directory, 'classified_mesh.mat');
save(classified_file, 'nodes_mm', 'elems', 'boundary', 'tissue_id', ...
    'group_id', 'tissue_names', '-v7');

source_files = {'trkg4_classify_volume_convergence_mesh', 'trkg4_config', ...
    'trkg4_subject_registry', 'trkg4_arm_montage_spec', ...
    'assign_tissues_from_stl', 'points_in_closed_surface', 'gmsh_mk_fwd_model'};
implementation = repmat(struct('function_name', '', 'sha256', ''), numel(source_files), 1);
for k = 1:numel(source_files)
    implementation(k).function_name = source_files{k};
    implementation(k).sha256 = trkg4_file_sha256(which(source_files{k}));
end
input_files = {cfg.body_stl, cfg.tissues(strcmp({cfg.tissues.name}, 'lungs')).file, ...
    cfg.tissues(strcmp({cfg.tissues.name}, 'heart')).file, ...
    cfg.tissues(strcmp({cfg.tissues.name}, 'bones')).file};
inputs = repmat(struct('path', '', 'sha256', ''), numel(input_files), 1);
for k = 1:numel(input_files)
    inputs(k).path = input_files{k};
    inputs(k).sha256 = trkg4_file_sha256(input_files{k});
end

report = struct();
report.schema = 'tepc_volume_mesh_classification_v1';
report.status = 'passed_independent_mesh_geometry_and_tissue_classification';
report.level_id = level_id;
report.trunk_size_mm = trunk_size_mm;
report.mesh_file = mesh_file;
report.mesh_sha256 = mesh_hash;
report.classified_mesh = 'classified_mesh.mat';
report.classified_mesh_sha256 = trkg4_file_sha256(classified_file);
report.counts = struct('nodes', size(nodes_mm, 1), 'tetrahedra', size(elems, 1), ...
    'boundary_faces', size(boundary, 1));
report.tissue_names = tissue_names;
report.tissue_counts = tissue_counts(:)';
report.tissue_volumes_mm3 = tissue_volumes_mm3(:)';
report.group_counts = group_counts(:)';
report.group_volumes_mm3 = group_volumes_mm3(:)';
report.total_volume_mm3 = sum(signed_volume_mm3);
report.min_tetra_volume_mm3 = min(signed_volume_mm3);
report.max_tetra_volume_mm3 = max(signed_volume_mm3);
report.checks = struct('finite_arrays', true, 'positive_tetrahedra', true, ...
    'complete_boundary_constructed', true, 'expected_tissues_present', true, ...
    'groups_partition_elements', true);
report.inputs = inputs;
report.implementation = implementation;
report.physical_validation = false;
report.mesh_solution_convergence = false;
local_json(fullfile(output_directory, 'classification.json'), report);
end

function path = local_path(value)
assert((ischar(value) && isrow(value) && ~isempty(value)) || ...
    (isstring(value) && isscalar(value) && strlength(value) > 0), ...
    'trkg4:volumeConvergencePath', 'A nonempty scalar path is required.');
path = char(java.io.File(char(value)).getCanonicalPath());
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
fid = fopen(file, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'trkg4:volumeConvergenceWrite', 'Cannot write classification report.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
