function preflight = check_trkg4_ttrkg_electrode_geometry(subject_id, electrode_file)
%CHECK_TRKG4_TTRKG_ELECTRODE_GEOMETRY Preflight for the 3-D arms montage.
%
% This check does not optimise or silently move electrode coordinates. It
% builds the electrode patches with the same code as the forward solver and
% reports whether they are geometrically suitable as a baseline for a TTRKG
% sensitivity calculation.
%
% Usage:
%   preflight = check_trkg4_ttrkg_electrode_geometry('nik');
%   preflight = check_trkg4_ttrkg_electrode_geometry('nik', candidate_csv);

if nargin < 1 || isempty(subject_id)
    subject_id = 'nik';
end

cfg = trkg4_config(subject_id);
if nargin < 2 || isempty(electrode_file)
    electrode_file = cfg.subject.electrodes_csv;
end

cfg.electrode_centres_file = electrode_file;
cfg.show_figures = false;
% The standard diagnostics are still collected, but a failed old montage
% must remain inspectable so that the reason for rejection can be written.
cfg.fail_on_electrode_diagnostics = false;
cfg.electrode_diagnostics_file = '';

[fmdl_mm, ~, ~, diagnostics] = trkg4_build_eidors_model_from_stl(cfg);

n = height(diagnostics);
normal_coherence = nan(n, 1);
max_normal_deviation_deg = nan(n, 1);
for k = 1:n
    electrode_nodes = fmdl_mm.electrode(k).nodes(:);
    is_patch_face = all(ismember(fmdl_mm.boundary, electrode_nodes), 2);
    patch_faces = fmdl_mm.boundary(is_patch_face, :);
    [normal_coherence(k), max_normal_deviation_deg(k)] = ...
        local_patch_normal_diagnostics(fmdl_mm.nodes, patch_faces);
end

preflight = diagnostics;
preflight.patch_normal_coherence = normal_coherence;
preflight.patch_max_normal_deviation_deg = max_normal_deviation_deg;

target_area = cfg.electrode_area;
preflight.area_ratio = preflight.patch_area_mm2 / target_area;
preflight.pass_area = ...
    preflight.area_ratio >= cfg.min_electrode_patch_area_fraction & ...
    preflight.area_ratio <= cfg.max_electrode_patch_area_ratio;

% Sixty degrees is a deliberately permissive technical guard. It is not a
% physiological constant and should be refined after accepted montages have
% been rerun on the same mesh.
max_allowed_normal_deviation_deg = 60;
preflight.pass_surface_normal = ...
    preflight.patch_max_normal_deviation_deg <= ...
    max_allowed_normal_deviation_deg;

[preflight, arm_geometry_available] = ...
    local_add_artificial_arm_clearance(preflight, cfg);

preflight.pass_standard_geometry = ...
    preflight.nearest_boundary_distance_mm <= ...
        cfg.max_electrode_surface_distance_mm & ...
    preflight.patch_centroid_offset_mm <= ...
        cfg.max_electrode_patch_centroid_offset_mm;

if arm_geometry_available
    preflight.pass_preflight = preflight.pass_standard_geometry & ...
        preflight.pass_area & preflight.pass_surface_normal & ...
        preflight.pass_artificial_arm_clearance;
else
    preflight.pass_preflight = preflight.pass_standard_geometry & ...
        preflight.pass_area & preflight.pass_surface_normal;
end

[~, stem] = fileparts(electrode_file);
out_file = fullfile(cfg.project_root, 'output', sprintf( ...
    '%s_%s_geometry_preflight.csv', cfg.subject.id, stem));
writetable(preflight, out_file);

fprintf('\nTTRKG electrode geometry preflight: %s\n', electrode_file);
disp(preflight(:, {'label', 'patch_area_mm2', 'area_ratio', ...
    'patch_normal_coherence', 'patch_max_normal_deviation_deg', ...
    'pass_preflight'}));
fprintf('Saved preflight table: %s\n', out_file);

if all(preflight.pass_preflight)
    fprintf('PASS: the montage passed the implemented geometric checks.\n');
else
    fprintf(['FAIL: the montage is not an accepted sensitivity baseline. ', ...
        'Inspect the failed columns and change a separate candidate CSV.\n']);
end
end

function [table_out, available] = local_add_artificial_arm_clearance(table_in, cfg)
table_out = table_in;
available = false;

report_file = fullfile(cfg.project_root, 'output', ...
    sprintf('%s_stl_preparation_report.json', cfg.subject.id));
if ~isfile(report_file)
    warning('trkg4:missingArmReport', ...
        'No STL preparation report: artificial-arm clearance was not checked.');
    return;
end

records = jsondecode(fileread(report_file));
if isstruct(records) && isscalar(records) && isfield(records, 'arm_extension')
    body_record = records;
elseif iscell(records)
    body_index = find(cellfun(@(record) ...
        isstruct(record) && isfield(record, 'name') && ...
        strcmp(record.name, 'body'), records), 1);
    if isempty(body_index)
        body_record = struct();
    else
        body_record = records{body_index};
    end
elseif isstruct(records) && isfield(records, 'name')
    body_index = find(strcmp({records.name}, 'body'), 1);
    if isempty(body_index)
        body_record = struct();
    else
        body_record = records(body_index);
    end
else
    body_record = struct();
end
if ~isfield(body_record, 'arm_extension')
    warning('trkg4:missingArmGeometry', ...
        'No arm-extension geometry in the STL preparation report.');
    return;
end

arms = body_record.arm_extension;
margin_mm = cfg.electrode_patch_radius_mm + cfg.mesh_target_size_mm;
n = height(table_in);
distance_to_outer_cap_mm = nan(n, 1);
distance_to_ct_join_mm = nan(n, 1);
pass_clearance = false(n, 1);

for k = 1:n
    x = table_in.requested_x_mm(k);
    if x >= 0
        outer_cap_x = arms.right.cylinder_x_max;
        join_x = arms.right.cut_x;
        distance_to_outer_cap_mm(k) = abs(outer_cap_x - x);
        distance_to_ct_join_mm(k) = abs(x - join_x);
        pass_clearance(k) = ...
            x <= outer_cap_x - margin_mm && x >= join_x + margin_mm;
    else
        outer_cap_x = arms.left.cylinder_x_min;
        join_x = arms.left.cut_x;
        distance_to_outer_cap_mm(k) = abs(x - outer_cap_x);
        distance_to_ct_join_mm(k) = abs(join_x - x);
        pass_clearance(k) = ...
            x >= outer_cap_x + margin_mm && x <= join_x - margin_mm;
    end
end

table_out.distance_to_artificial_outer_cap_mm = distance_to_outer_cap_mm;
table_out.distance_to_ct_join_mm = distance_to_ct_join_mm;
table_out.required_artificial_boundary_margin_mm = repmat(margin_mm, n, 1);
table_out.pass_artificial_arm_clearance = pass_clearance;
available = true;
end

function [coherence, max_deviation_deg] = local_patch_normal_diagnostics(nodes, faces)
% Orientation-invariant normal analysis: n and -n represent one axis.
if isempty(faces)
    coherence = NaN;
    max_deviation_deg = NaN;
    return;
end

p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
normal_vectors = cross(p2 - p1, p3 - p1, 2);
twice_area = vecnorm(normal_vectors, 2, 2);
valid = twice_area > eps;
if ~any(valid)
    coherence = NaN;
    max_deviation_deg = NaN;
    return;
end

normal_vectors = normal_vectors(valid, :);
twice_area = twice_area(valid);
unit_normals = normal_vectors ./ twice_area;
weights = 0.5 * twice_area;

second_moment = zeros(3, 3);
for j = 1:size(unit_normals, 1)
    n_j = unit_normals(j, :)';
    second_moment = second_moment + weights(j) * (n_j * n_j');
end

[axes, eigenvalues] = eig(second_moment);
[largest_eigenvalue, principal_index] = max(real(diag(eigenvalues)));
principal_axis = real(axes(:, principal_index));
coherence = largest_eigenvalue / sum(weights);

cos_angle = abs(unit_normals * principal_axis);
cos_angle = min(1, max(0, cos_angle));
max_deviation_deg = max(acosd(cos_angle));
end
