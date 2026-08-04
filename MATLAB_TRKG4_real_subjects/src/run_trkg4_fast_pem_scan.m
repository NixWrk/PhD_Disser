function result = run_trkg4_fast_pem_scan(coordinate_step_mm, phi_step_deg, top_count)
%RUN_TRKG4_FAST_PEM_SCAN Dense point-electrode screening of centre and angle.
% The expensive bulk FEM matrix is assembled and factorized once. Transfer
% impedances between all candidate boundary nodes are then reused for a
% 1-mm (u,v) grid and an explicit phi sweep. Final candidates must be
% verified with the complete electrode model; PEM is only a screening model.

if nargin < 1 || isempty(coordinate_step_mm), coordinate_step_mm = 1; end
if nargin < 2 || isempty(phi_step_deg), phi_step_deg = 1; end
if nargin < 3 || isempty(top_count), top_count = 250; end

cfg = trkg4_config('nik');
root = cfg.project_root;
shared_file = fullfile(root, 'output', 'nik_trkg4_arms_full_v5_1mm_local2mm.mat');
shared = load(shared_file, 'fmdl_mesh_units', 'elem_sigma', ...
    'tissue_id', 'tissue_names');
fmdl_mm = shared.fmdl_mesh_units;
fmdl_si = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);

fit = readtable(fullfile(root, 'output', ...
    'nik_trkg4_inverse_inhale_fit_summary.csv'));
rho_soft = fit.rho_soft_ohm_m(1);
rho_lung = fit.rho_lungs_ohm_m(1);
elem_sigma = shared.elem_sigma;
soft_id = find(strcmp(shared.tissue_names, 'soft_tissue'), 1);
lung_id = find(strcmp(shared.tissue_names, 'lungs'), 1);
elem_sigma(shared.tissue_id == soft_id) = 1 / rho_soft;
elem_sigma(shared.tissue_id == lung_id) = 1 / rho_lung;

[sizes_mm, measured_z] = local_load_inhale_data(root);
experimental_slope = robustfit(sizes_mm, measured_z);
experimental_slope = experimental_slope(2);

boundary_nodes = unique(fmdl_mm.boundary(:));
boundary_xyz = fmdl_mm.nodes(boundary_nodes, :);
node_normals = local_boundary_node_normals(fmdl_mm);
[c0, n0, u0, v0] = local_reference_frame(root, fmdl_mm, ...
    boundary_nodes, boundary_xyz, node_normals);

% All requested 1-mm tangent-plane coordinates are retained, but raw grid
% points landing on the same discrete FEM node are evaluated only once.
centre_values = -120:coordinate_step_mm:120;
[centre_u_grid, centre_v_grid] = ndgrid(centre_values, centre_values);
centre_raw = c0 + centre_u_grid(:) .* u0 + centre_v_grid(:) .* v0;
centre_boundary_index = knnsearch(boundary_xyz, centre_raw);
centre_global_all = boundary_nodes(centre_boundary_index);
[centre_global, ~, centre_group] = unique(centre_global_all, 'stable');
centre_xyz = fmdl_mm.nodes(centre_global, :);

% Pick the raw (u,v) sample closest to each snapped surface node.
representative_index = zeros(numel(centre_global), 1);
snap_distance_sq = sum((centre_raw - boundary_xyz(centre_boundary_index, :)).^2, 2);
for group = 1:numel(centre_global)
    members = find(centre_group == group);
    [~, local_min] = min(snap_distance_sq(members));
    representative_index(group) = members(local_min);
end
centre_u = centre_u_grid(representative_index);
centre_v = centre_v_grid(representative_index);

lung_mesh = stlread(cfg.subject.stl.lungs);
[~, h_centre_mm] = knnsearch(lung_mesh.Points, centre_xyz);
anatomical = h_centre_mm >= 7 & h_centre_mm <= 30;
centre_global = centre_global(anatomical);
centre_xyz = centre_xyz(anatomical, :);
centre_u = centre_u(anatomical);
centre_v = centre_v(anatomical);
h_centre_mm = h_centre_mm(anatomical);

phi_values = -20:phi_step_deg:20;
n_centres = numel(centre_global);
n_phi = numel(phi_values);
n_pose = n_centres * n_phi;
n_size = numel(sizes_mm);

pose_u = zeros(n_pose, 1);
pose_v = zeros(n_pose, 1);
pose_phi = zeros(n_pose, 1);
pose_h = zeros(n_pose, 1);
pose_centres = zeros(n_pose, 3);
pose_axes = zeros(n_pose, 3);

fprintf('Fast PEM scan: %d raw centres -> %d anatomical FEM centres, %d phi values.\n', ...
    numel(centre_u_grid), n_centres, n_phi);
tic_scan = tic;
for centre_index = 1:n_centres
    rows = (centre_index - 1) * n_phi + (1:n_phi);
    centre = centre_xyz(centre_index, :);
    normal = node_normals(centre_global(centre_index), :);
    base_axis = u0 - dot(u0, normal) * normal;
    base_axis = base_axis / norm(base_axis);
    side_axis = cross(normal, base_axis);
    side_axis = side_axis / norm(side_axis);
    pose_u(rows) = centre_u(centre_index);
    pose_v(rows) = centre_v(centre_index);
    pose_phi(rows) = phi_values;
    pose_h(rows) = h_centre_mm(centre_index);
    pose_centres(rows, :) = repmat(centre, n_phi, 1);
    pose_axes(rows, :) = cosd(phi_values(:)) .* base_axis + ...
        sind(phi_values(:)) .* side_axis;
end

offsets_by_size = zeros(4, n_size);
for size_index = 1:n_size
    L = sizes_mm(size_index);
    offsets_by_size(:, size_index) = [-L/2; -L/4; L/4; L/2];
end
all_offsets = offsets_by_size(:)';
n_electrode_position = numel(all_offsets);
all_requested = zeros(n_pose * n_electrode_position, 3);
for electrode_index = 1:n_electrode_position
    rows = (electrode_index - 1) * n_pose + (1:n_pose);
    all_requested(rows, :) = pose_centres + ...
        all_offsets(electrode_index) .* pose_axes;
end
all_nearest = knnsearch(boundary_xyz, all_requested);
electrode_global_by_pose = reshape( ...
    uint32(boundary_nodes(all_nearest)), n_pose, n_electrode_position);

% Affine PEM->CEM correction from previously evaluated poses having exactly
% the same material parameters. This corrects the systematic point-vs-area
% electrode scale, while the validation table exposes the remaining error.
history = readtable(fullfile(root, 'output', 'nik_trkg4_inverse_inhale_history.csv'));
same_rho = abs(history.rho_soft_ohm_m - rho_soft) < 1e-9 & ...
    abs(history.rho_lungs_ohm_m - rho_lung) < 1e-9;
validation_history = history(same_rho, :);
z_columns = startsWith(validation_history.Properties.VariableNames, 'Z_FEM_L');
z_cem_validation = validation_history{:, z_columns};

validation_electrode_global = zeros(height(validation_history), 4 * n_size, 'uint32');
for k = 1:height(validation_history)
    validation_electrode_global(k, :) = local_pose_electrode_nodes( ...
        validation_history.centre_u_mm(k), validation_history.centre_v_mm(k), ...
        validation_history.phi_deg(k), sizes_mm, c0, u0, v0, ...
        boundary_nodes, boundary_xyz, node_normals);
end

candidate_global_nodes = unique(double([electrode_global_by_pose(:); ...
    validation_electrode_global(:)]));
candidate_global_nodes(candidate_global_nodes == fmdl_si.gnd_node) = [];
cache_file = fullfile(root, 'output', 'nik_trkg4_fast_pem_green_cache_v5_1mm_local2mm.mat');
cache_key = sprintf('n%d_e%d_rho%.9g_%.9g_step%.6g_phi%.6g_phi20', ...
    size(fmdl_mm.nodes, 1), size(fmdl_mm.elems, 1), ...
    rho_soft, rho_lung, coordinate_step_mm, phi_step_deg);
[green, cache_reused, factor_seconds, solve_seconds] = local_green_matrix( ...
    fmdl_si, elem_sigma, candidate_global_nodes, cache_file, cache_key);
global_to_candidate = zeros(size(fmdl_mm.nodes, 1), 1, 'uint32');
global_to_candidate(candidate_global_nodes) = uint32(1:numel(candidate_global_nodes));

electrode_local = double(global_to_candidate(double(electrode_global_by_pose)));
z_pem = zeros(n_pose, n_size);
green_size = size(green);
for size_index = 1:n_size
    columns = (size_index - 1) * 4 + (1:4);
    A = electrode_local(:, columns(1));
    M = electrode_local(:, columns(2));
    N = electrode_local(:, columns(3));
    B = electrode_local(:, columns(4));
    z_pem(:, size_index) = double( ...
        green(sub2ind(green_size, M, A)) - green(sub2ind(green_size, M, B)) ...
        - green(sub2ind(green_size, N, A)) + green(sub2ind(green_size, N, B)));
end
scan_seconds = toc(tic_scan);

z_pem_validation = zeros(size(z_cem_validation));
for k = 1:height(validation_history)
    z_pem_validation(k, :) = local_evaluate_pose( ...
        validation_history.centre_u_mm(k), validation_history.centre_v_mm(k), ...
        validation_history.phi_deg(k), sizes_mm, c0, u0, v0, ...
        boundary_nodes, boundary_xyz, node_normals, global_to_candidate, green);
end
calibration = [ones(numel(z_pem_validation), 1), z_pem_validation(:)] ...
    \ z_cem_validation(:);
z_fast = calibration(1) + calibration(2) * z_pem;
z_validation_corrected = calibration(1) + calibration(2) * z_pem_validation;

validation_rmse = sqrt(mean((z_validation_corrected(:) - z_cem_validation(:)).^2));
validation_correlation = corr(z_validation_corrected(:), z_cem_validation(:));

abs_residual = abs(z_fast - measured_z(:)');
huber_delta = 5;
huber = 0.5 * min(abs_residual, huber_delta).^2 + ...
    huber_delta * max(abs_residual - huber_delta, 0);
L_centered = sizes_mm - mean(sizes_mm);
slope = (z_fast * L_centered) / sum(L_centered.^2);
span_mm = max(sizes_mm) - min(sizes_mm);
slope_penalty = 0.20 * ((slope - experimental_slope) * span_mm).^2;
depth_prior = ((pose_h - 15) / 7).^2;
objective = mean(huber, 2) + slope_penalty + depth_prior;

all_scan = table(pose_u, pose_v, pose_phi, pose_h, objective, slope, ...
    'VariableNames', {'centre_u_mm', 'centre_v_mm', 'phi_deg', ...
    'h_centre_mm', 'fast_objective', 'slope_ohm_per_mm'});
for k = 1:n_size
    all_scan.(sprintf('Z_fast_L%03d_ohm', sizes_mm(k))) = z_fast(:, k);
end
all_file = fullfile(root, 'output', 'nik_trkg4_fast_pem_scan_all.csv');
writetable(all_scan, all_file);

[objective_sorted, order] = sort(objective);
keep = order(1:min(top_count, numel(order)));
top = table(pose_u(keep), pose_v(keep), pose_phi(keep), pose_h(keep), ...
    objective_sorted(1:numel(keep)), slope(keep), ...
    'VariableNames', {'centre_u_mm', 'centre_v_mm', 'phi_deg', ...
    'h_centre_mm', 'fast_objective', 'slope_ohm_per_mm'});
for k = 1:n_size
    top.(sprintf('Z_fast_L%03d_ohm', sizes_mm(k))) = z_fast(keep, k);
end

top_file = fullfile(root, 'output', 'nik_trkg4_fast_pem_scan_top.csv');
writetable(top, top_file);

validation = table(validation_history.evaluation, ...
    validation_history.centre_u_mm, validation_history.centre_v_mm, ...
    validation_history.phi_deg, ...
    sqrt(mean((z_validation_corrected - z_cem_validation).^2, 2)), ...
    'VariableNames', {'evaluation', 'centre_u_mm', 'centre_v_mm', ...
    'phi_deg', 'curve_RMSE_PEM_to_CEM_ohm'});
validation_file = fullfile(root, 'output', ...
    'nik_trkg4_fast_pem_validation.csv');
writetable(validation, validation_file);

summary = table(coordinate_step_mm, phi_step_deg, -20, 20, ...
    numel(centre_u_grid), n_centres, n_phi, n_pose, ...
    numel(candidate_global_nodes), rho_soft, rho_lung, ...
    calibration(1), calibration(2), validation_rmse, ...
    validation_correlation, factor_seconds, solve_seconds, scan_seconds, ...
    cache_reused, ...
    'VariableNames', {'coordinate_step_mm', 'phi_step_deg', ...
    'phi_min_deg', 'phi_max_deg', 'raw_centre_count', ...
    'unique_anatomical_centre_count', 'phi_count', 'pose_count', ...
    'candidate_node_count', 'rho_soft_ohm_m', 'rho_lungs_ohm_m', ...
    'PEM_to_CEM_intercept', 'PEM_to_CEM_slope', ...
    'validation_RMSE_ohm', 'validation_correlation', ...
    'factor_seconds', 'green_solve_seconds', 'scan_seconds', 'cache_reused'});
summary_file = fullfile(root, 'output', 'nik_trkg4_fast_pem_scan_summary.csv');
writetable(summary, summary_file);

result = struct('top', top, 'validation', validation, 'summary', summary, ...
    'all_file', all_file, 'top_file', top_file, 'validation_file', validation_file, ...
    'summary_file', summary_file, 'cache_file', cache_file);
fprintf('Fast PEM scan saved: %s\n', top_file);
fprintf('PEM/CEM validation RMSE %.4f Ohm, correlation %.5f.\n', ...
    validation_rmse, validation_correlation);
end

function [green, reused, factor_seconds, solve_seconds] = local_green_matrix( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_file, cache_key)
reused = false;
factor_seconds = 0;
solve_seconds = 0;
if isfile(cache_file)
    cached = load(cache_file, 'green', 'candidate_nodes', 'cache_key');
    if isfield(cached, 'cache_key') && strcmp(cached.cache_key, cache_key) && ...
            isequal(cached.candidate_nodes, candidate_nodes)
        green = cached.green;
        reused = true;
        return;
    end
end

model = fmdl_si;
model.electrode = struct([]);
model.stimulation = struct([]);
img = mk_image(model, elem_sigma);
system = calc_system_mat(img);
ground = model.gnd_node;
keep_nodes = (1:size(system.E, 1))';
keep_nodes(ground) = [];
reduced_index = zeros(size(system.E, 1), 1);
reduced_index(keep_nodes) = 1:numel(keep_nodes);
candidate_reduced = reduced_index(candidate_nodes);

tic_factor = tic;
factor = decomposition(system.E(keep_nodes, keep_nodes), 'chol');
factor_seconds = toc(tic_factor);

n_candidate = numel(candidate_nodes);
green = zeros(n_candidate, n_candidate, 'single');
block_size = 64;
tic_solve = tic;
for first = 1:block_size:n_candidate
    last = min(first + block_size - 1, n_candidate);
    columns = first:last;
    rhs = sparse(candidate_reduced(columns), 1:numel(columns), 1, ...
        numel(keep_nodes), numel(columns));
    voltage = factor \ rhs;
    green(:, columns) = single(full(voltage(candidate_reduced, :)));
end
solve_seconds = toc(tic_solve);
candidate_nodes = candidate_nodes; %#ok<NASGU>
save(cache_file, 'green', 'candidate_nodes', 'cache_key', '-v7.3');
end

function z = local_evaluate_pose(u, v, phi, sizes, c0, u0, v0, ...
    boundary_nodes, boundary_xyz, node_normals, node_map, green)
electrode_global = local_pose_electrode_nodes(u, v, phi, sizes, ...
    c0, u0, v0, boundary_nodes, boundary_xyz, node_normals);
local = double(node_map(double(electrode_global)));
z = zeros(1, numel(sizes));
for k = 1:numel(sizes)
    columns = (k - 1) * 4 + (1:4);
    A = local(columns(1)); M = local(columns(2));
    N = local(columns(3)); B = local(columns(4));
    z(k) = double(green(M,A) - green(M,B) - green(N,A) + green(N,B));
end
end

function electrode_global = local_pose_electrode_nodes(u, v, phi, sizes, ...
    c0, u0, v0, boundary_nodes, boundary_xyz, node_normals)
raw_centre = c0 + u * u0 + v * v0;
nearest_centre = knnsearch(boundary_xyz, raw_centre);
centre_global = boundary_nodes(nearest_centre);
centre = boundary_xyz(nearest_centre, :);
normal = node_normals(centre_global, :);
base_axis = u0 - dot(u0, normal) * normal;
base_axis = base_axis / norm(base_axis);
side_axis = cross(normal, base_axis);
side_axis = side_axis / norm(side_axis);
axis = cosd(phi) * base_axis + sind(phi) * side_axis;
electrode_global = zeros(1, 4 * numel(sizes), 'uint32');
for k = 1:numel(sizes)
    offsets = [-sizes(k)/2; -sizes(k)/4; sizes(k)/4; sizes(k)/2];
    nearest = knnsearch(boundary_xyz, centre + offsets .* axis);
    columns = (k - 1) * 4 + (1:4);
    electrode_global(columns) = uint32(boundary_nodes(nearest));
end
end

function [c0, n0, u0, v0] = local_reference_frame(root, fmdl, ...
    boundary_nodes, boundary_xyz, node_normals)
reference = readtable(fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_140mm_xyz_mm.csv'));
reference_xyz = [reference.x_mm, reference.y_mm, reference.z_mm];
reference_centre = mean(reference_xyz, 1);
reference_axis = reference_xyz(4, :) - reference_xyz(1, :);
reference_axis = reference_axis / norm(reference_axis);
nearest = knnsearch(boundary_xyz, reference_centre);
c0 = fmdl.nodes(boundary_nodes(nearest), :);
n0 = node_normals(boundary_nodes(nearest), :);
u0 = reference_axis - dot(reference_axis, n0) * n0;
u0 = u0 / norm(u0);
v0 = cross(n0, u0);
v0 = v0 / norm(v0);
end

function normals = local_boundary_node_normals(fmdl)
faces = fmdl.boundary;
nodes = fmdl.nodes;
p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
face_normals = cross(p2 - p1, p3 - p1, 2);
face_centres = (p1 + p2 + p3) / 3;
body_centre = mean(nodes, 1);
flip = dot(face_normals, face_centres - body_centre, 2) < 0;
face_normals(flip, :) = -face_normals(flip, :);
normals = zeros(size(nodes));
for d = 1:3
    normals(:, d) = accumarray(faces(:), repmat(face_normals(:, d), 3, 1), ...
        [size(nodes, 1), 1], @sum, 0);
end
lengths = vecnorm(normals, 2, 2);
valid = lengths > 0;
normals(valid, :) = normals(valid, :) ./ lengths(valid);
end

function [sizes, z_inhale] = local_load_inhale_data(root)
timestamp_dir = fullfile(root, '..', 'Colab Notebooks', 'timestamps');
files = dir(fullfile(timestamp_dir, '*nik.json'));
sizes = zeros(0, 1);
z_inhale = zeros(0, 1);
for k = 1:numel(files)
    record = jsondecode(fileread(fullfile(files(k).folder, files(k).name)));
    if record.size_mm == 90, continue; end
    hold_values = struct2cell(record.hold_levels);
    sizes(end + 1, 1) = record.size_mm; %#ok<AGROW>
    z_inhale(end + 1, 1) = hold_values{1}; %#ok<AGROW>
end
[sizes, order] = sort(sizes);
z_inhale = z_inhale(order);
end
