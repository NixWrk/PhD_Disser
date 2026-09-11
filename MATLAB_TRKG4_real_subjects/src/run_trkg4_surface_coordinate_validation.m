function summary = run_trkg4_surface_coordinate_validation(varargin)
%RUN_TRKG4_SURFACE_COORDINATE_VALIDATION Compare native and axial centres.
%
% This validation resolves all requested point-electrode centres along two
% proximal-to-distal surface paths and compares them with the current axial
% construction on the accepted artificial-arm mesh. It does not run an FEM
% forward solve and does not validate experimental electrode placement.

p = inputParser;
addParameter(p, 'SubjectId', 'nik', @(x) ischar(x) || isstring(x));
addParameter(p, 'InnerDistancesMm', [20 40], @local_positive_vector);
addParameter(p, 'OuterDistancesMm', [80 100], @local_positive_vector);
addParameter(p, 'PrebuiltMeshFile', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'MeshTargetSizeMm', 1, @local_positive_scalar);
parse(p, varargin{:});
opt = p.Results;

cfg = trkg4_config(char(opt.SubjectId));
root = cfg.project_root;
if strlength(string(opt.PrebuiltMeshFile)) == 0
    cfg.prebuilt_mesh_file = fullfile(root, 'output', ...
        'accepted_arm_parameter_mesh.msh');
else
    candidate = char(opt.PrebuiltMeshFile);
    if ~isfile(candidate)
        candidate = fullfile(root, candidate);
    end
    cfg.prebuilt_mesh_file = candidate;
end
cfg.mesh_target_size_mm = opt.MeshTargetSizeMm;

inner_values = double(opt.InnerDistancesMm(:)');
outer_values = double(opt.OuterDistancesMm(:)');
first_spec = trkg4_arm_montage_spec(cfg, 'point_disc_5mm', ...
    inner_values(1), outer_values(1));

% The existing mesh loader requires a valid bootstrap montage. These
% centres are replaced by the native-coordinate comparison after loading.
cfg.electrode_centres_xyz = first_spec.centres_xyz_mm;
cfg.electrode_centres_file = '';
cfg.electrode_area_mm2 = first_spec.nominal_area_mm2(1);
cfg.electrode_area = cfg.electrode_area_mm2;
cfg.electrode_patch_radius_mm = sqrt(cfg.electrode_area_mm2 / pi);
cfg.max_electrode_patch_centroid_offset_mm = ...
    cfg.electrode_patch_radius_mm;
cfg.fail_on_electrode_diagnostics = false;
cfg.electrode_diagnostics_file = '';
validate_stl_inputs(cfg);

[fmdl_mm, ~, mesh_timings] = trkg4_build_eidors_model_from_stl(cfg);
arms = first_spec.arm_geometry;
right_guides = local_arm_guides(arms.right, true, ...
    cfg.mesh_target_size_mm);
left_guides = local_arm_guides(arms.left, false, ...
    cfg.mesh_target_size_mm);

sample_distances_mm = local_required_distances(inner_values, outer_values);
[right_samples, right_path] = trkg4_surface_path( ...
    fmdl_mm.nodes, fmdl_mm.boundary, right_guides, sample_distances_mm);
[left_samples, left_path] = trkg4_surface_path( ...
    fmdl_mm.nodes, fmdl_mm.boundary, left_guides, sample_distances_mm);

labels = ["I_plus"; "V_plus"; "V_minus"; "I_minus"];
sides = ["right"; "right"; "left"; "left"];
n_rows = numel(inner_values) * numel(outer_values) * numel(labels);
rows = repmat(local_empty_row(), n_rows, 1);
row_index = 0;

for inner_mm = inner_values
    for outer_mm = outer_values
        legacy = trkg4_arm_montage_spec(cfg, 'point_disc_5mm', ...
            inner_mm, outer_mm);
        right_outer = local_sample_at(right_samples, ...
            sample_distances_mm, inner_mm + outer_mm);
        right_inner = local_sample_at(right_samples, ...
            sample_distances_mm, inner_mm);
        left_inner = local_sample_at(left_samples, ...
            sample_distances_mm, inner_mm);
        left_outer = local_sample_at(left_samples, ...
            sample_distances_mm, inner_mm + outer_mm);
        native = [right_outer; right_inner; left_inner; left_outer];
        requested_distance = [inner_mm + outer_mm; inner_mm; ...
            inner_mm; inner_mm + outer_mm];

        for electrode_index = 1:numel(labels)
            row_index = row_index + 1;
            difference = native(electrode_index, :) - ...
                legacy.centres_xyz_mm(electrode_index, :);
            row = local_empty_row();
            row.scenario_id = string(sprintf('in%g_out%g', ...
                inner_mm, outer_mm));
            row.label = labels(electrode_index);
            row.side = sides(electrode_index);
            row.inner_from_reference_mm = inner_mm;
            row.outer_from_inner_mm = outer_mm;
            row.distance_from_reference_mm = ...
                requested_distance(electrode_index);
            row.legacy_x_mm = legacy.centres_xyz_mm(electrode_index, 1);
            row.legacy_y_mm = legacy.centres_xyz_mm(electrode_index, 2);
            row.legacy_z_mm = legacy.centres_xyz_mm(electrode_index, 3);
            row.surface_x_mm = native(electrode_index, 1);
            row.surface_y_mm = native(electrode_index, 2);
            row.surface_z_mm = native(electrode_index, 3);
            row.centre_difference_mm = norm(difference);
            rows(row_index) = row;
        end
    end
end

summary = struct2table(rows);
out_dir = fullfile(root, 'output');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
summary_file = fullfile(out_dir, ...
    sprintf('%s_surface_coordinate_validation.csv', cfg.subject.id));
contract_file = fullfile(out_dir, ...
    sprintf('%s_surface_coordinate_validation.json', cfg.subject.id));
writetable(summary, summary_file);

contract = struct();
contract.status = 'technical_geometry_check_not_experimental_validation';
contract.subject_id = cfg.subject.id;
contract.coordinate_system = 'surface_path_v1';
contract.reference = ...
    'first resolved proximal guide point on each artificial arm';
contract.prebuilt_mesh_file = cfg.prebuilt_mesh_file;
contract.mesh_target_size_mm = cfg.mesh_target_size_mm;
contract.inner_distances_mm = inner_values;
contract.outer_distances_mm = outer_values;
contract.sample_distances_mm = sample_distances_mm;
contract.right_guide_points_xyz_mm = right_guides;
contract.left_guide_points_xyz_mm = left_guides;
contract.right_guide_offsets_mm = right_path.guide_offsets_mm;
contract.left_guide_offsets_mm = left_path.guide_offsets_mm;
contract.right_path_length_mm = right_path.path_length_mm;
contract.left_path_length_mm = left_path.path_length_mm;
contract.maximum_centre_difference_mm = max(summary.centre_difference_mm);
contract.mesh_timings = mesh_timings;
contract.fem_forward_solve_performed = false;
contract.ring_and_cuff_generalisation_tested = false;
local_write_json(contract_file, contract);

fprintf('Saved surface-coordinate summary: %s\n', summary_file);
fprintf('Saved surface-coordinate contract: %s\n', contract_file);
fprintf('Maximum centre difference: %.6g mm\n', ...
    contract.maximum_centre_difference_mm);
end

function guides = local_arm_guides(arm, is_right, mesh_step_mm)
if is_right
    first_x = arm.cut_x;
    last_x = arm.cylinder_x_max - mesh_step_mm;
else
    first_x = arm.cut_x;
    last_x = arm.cylinder_x_min + mesh_step_mm;
end
x_values = linspace(first_x, last_x, 5).';
guides = [x_values, repmat(arm.center_y, 5, 1), ...
    repmat(arm.center_z + arm.radius, 5, 1)];
end

function distances = local_required_distances(inner_values, outer_values)
distances = inner_values;
for inner_mm = inner_values
    distances = [distances, inner_mm + outer_values]; %#ok<AGROW>
end
distances = unique(distances(:));
end

function point = local_sample_at(samples, distances, requested)
index = find(abs(distances - requested) <= ...
    32 * eps(max(1, abs(requested))), 1);
if isempty(index)
    error('trkg4:missingSurfaceCoordinateSample', ...
        'No surface-path sample was computed at %.9g mm.', requested);
end
point = samples(index, :);
end

function row = local_empty_row()
row = struct('scenario_id', "", 'label', "", 'side', "", ...
    'inner_from_reference_mm', NaN, 'outer_from_inner_mm', NaN, ...
    'distance_from_reference_mm', NaN, ...
    'legacy_x_mm', NaN, 'legacy_y_mm', NaN, 'legacy_z_mm', NaN, ...
    'surface_x_mm', NaN, 'surface_y_mm', NaN, 'surface_z_mm', NaN, ...
    'centre_difference_mm', NaN);
end

function local_write_json(filename, value)
fid = fopen(filename, 'w');
if fid < 0
    error('trkg4:surfaceCoordinateContractWriteFailed', ...
        'Cannot open contract file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(value, 'PrettyPrint', true), 'char');
end

function ok = local_positive_vector(x)
ok = isnumeric(x) && isvector(x) && ~isempty(x) && ...
    all(isfinite(x)) && all(x > 0);
end

function ok = local_positive_scalar(x)
ok = isnumeric(x) && isscalar(x) && isfinite(x) && x > 0;
end
