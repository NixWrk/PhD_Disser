function result = run_trkg4_fast_cem_refine(top_count, local_radius_mm, local_phi_step_deg)
%RUN_TRKG4_FAST_CEM_REFINE Exact low-rank CEM evaluation of PEM candidates.
% Reuses one grounded bulk FEM inverse on all nodes touched by finite-area
% electrodes, then solves only small local systems for each four-electrode
% configuration.

if nargin < 1 || isempty(top_count), top_count = 100; end
if nargin < 2 || isempty(local_radius_mm), local_radius_mm = 0; end
if nargin < 3 || isempty(local_phi_step_deg), local_phi_step_deg = 1; end
cfg = trkg4_config('nik');
trkg4_prepare_runtime(cfg);
root = cfg.project_root;
result_tag = 'v5_1mm_local2mm';
shared = load(fullfile(root, 'output', ...
    'nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units', 'elem_sigma', 'tissue_id', 'tissue_names');
fmdl_mm = shared.fmdl_mesh_units;
fmdl_si = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);

fit = readtable(fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_fit_summary_%s.csv', result_tag)));
rho_soft = fit.rho_soft_ohm_m(1);
rho_lung = fit.rho_lungs_ohm_m(1);
elem_sigma = shared.elem_sigma;
soft_id = find(strcmp(shared.tissue_names, 'soft_tissue'), 1);
lung_id = find(strcmp(shared.tissue_names, 'lungs'), 1);
elem_sigma(shared.tissue_id == soft_id) = 1 / rho_soft;
elem_sigma(shared.tissue_id == lung_id) = 1 / rho_lung;

pem_top = readtable(fullfile(root, 'output', 'nik_trkg4_fast_pem_scan_top.csv'));
pem_top = pem_top(1:min(top_count, height(pem_top)), :);
candidates = table(pem_top.centre_u_mm, pem_top.centre_v_mm, pem_top.phi_deg, ...
    false(height(pem_top), 1), 'VariableNames', ...
    {'centre_u_mm', 'centre_v_mm', 'phi_deg', 'is_validation'});
if local_radius_mm > 0
    offsets = -local_radius_mm:1:local_radius_mm;
    phi_local = -20:local_phi_step_deg:20;
    [du_local, dv_local, angle_local] = ndgrid(offsets, offsets, phi_local);
    local_candidates = table(fit.centre_u_mm(1) + du_local(:), ...
        fit.centre_v_mm(1) + dv_local(:), angle_local(:), ...
        false(numel(du_local), 1), 'VariableNames', ...
        {'centre_u_mm', 'centre_v_mm', 'phi_deg', 'is_validation'});
    candidates = [candidates; local_candidates];
end
candidates(end + 1, :) = {fit.centre_u_mm(1), fit.centre_v_mm(1), ...
    fit.phi_deg(1), true};
[~, unique_rows] = unique(candidates(:, 1:3), 'rows', 'stable');
candidates = candidates(unique_rows, :);

[sizes_mm, measured_z] = trkg4_load_nik_inhale_data(root);
experimental_line = robustfit(sizes_mm, measured_z);
experimental_slope = experimental_line(2);
pose = local_build_pose_context(fmdl_mm, root, cfg.subject.stl.lungs);
n_pose = height(candidates);
requested_pose_count = n_pose;
valid_pose = true(n_pose, 1);
n_size = numel(sizes_mm);
patch_faces = cell(n_pose, n_size, 4);
h_centre = zeros(n_pose, 1);
patch_context = [];
qc_cfg = cfg;
qc_cfg.electrode_diagnostics_verbose = false;
qc_cfg.electrode_diagnostics_file = '';

tic_geometry = tic;
for pose_index = 1:n_pose
    state = local_pose_state(pose, candidates.centre_u_mm(pose_index), ...
        candidates.centre_v_mm(pose_index), candidates.phi_deg(pose_index));
    h_centre(pose_index) = state.h_centre_mm;
    for size_index = 1:n_size
        [centres, ~] = local_grid_centres(pose, state, sizes_mm(size_index));
        % Same builder as the forward and inverse models. It used to be
        % electrode_faces_by_area_fast, which picked faces by a different
        % rule and sized the patch on the picked faces instead of the induced
        % ones - so this scan optimised a slightly different electrode
        % geometry from the one being refined. The induced faces now come
        % straight out of the builder instead of being rebuilt here.
        try
            [electrodes, induced_faces, patch_context] = electrode_faces_by_area( ...
                fmdl_mm, centres, cfg.electrode_area, cfg.z_contact, patch_context);
        catch exception
            expected_geometry_failure = ...
                any(strcmp(exception.identifier, { ...
                'electrode_faces_by_area:noSeed', ...
                'electrode_faces_by_area:patchTooSmall'})) || ...
                strcmp(exception.identifier, 'trkg4:electrodeDiagnostics');
            if expected_geometry_failure
                valid_pose(pose_index) = false;
                fprintf(['CEM geometry QC rejected candidate %d (u=%.3f, ', ...
                    'v=%.3f, phi=%.3f): %s\n'], pose_index, ...
                    candidates.centre_u_mm(pose_index), ...
                    candidates.centre_v_mm(pose_index), ...
                    candidates.phi_deg(pose_index), exception.message);
                break
            end
            rethrow(exception);
        end
        try
            fmdl_qc = fmdl_mm;
            fmdl_qc.electrode = electrodes;
            trkg4_electrode_diagnostics(fmdl_qc, centres, ...
                cfg.electrode_order, qc_cfg, centres, induced_faces);
        catch exception
            if strcmp(exception.identifier, 'trkg4:electrodeDiagnostics')
                valid_pose(pose_index) = false;
                fprintf(['CEM geometry QC rejected candidate %d (u=%.3f, ', ...
                    'v=%.3f, phi=%.3f): %s\n'], pose_index, ...
                    candidates.centre_u_mm(pose_index), ...
                    candidates.centre_v_mm(pose_index), ...
                    candidates.phi_deg(pose_index), exception.message);
                break
            end
            rethrow(exception);
        end
        for electrode_index = 1:4
            patch_faces{pose_index, size_index, electrode_index} = ...
                induced_faces{electrode_index};
        end
    end
end
rejected_geometry_count = sum(~valid_pose);
if ~all(valid_pose)
    candidates = candidates(valid_pose, :);
    patch_faces = patch_faces(valid_pose, :, :);
    h_centre = h_centre(valid_pose);
    n_pose = height(candidates);
end
if ~any(candidates.is_validation)
    error('trkg4:invalidCemValidationGeometry', ...
        'The full-EIDORS validation pose was rejected by electrode geometry QC.');
end
fprintf('CEM geometry QC: accepted %d of %d poses; rejected %d.\n', ...
    n_pose, requested_pose_count, rejected_geometry_count);
geometry_seconds = toc(tic_geometry);

all_patch_faces = unique(vertcat(patch_faces{:}));
candidate_nodes = unique(reshape(fmdl_mm.boundary(all_patch_faces, :), [], 1));
candidate_nodes(candidate_nodes == fmdl_si.gnd_node) = [];
cache_file = fullfile(root, 'output', ...
    'nik_trkg4_fast_cem_green_cache_v5_1mm_local2mm.mat');
cache_settings = struct( ...
    'mode', 'cem_lowrank_bulk_green', ...
    'result_tag', result_tag, ...
    'decomposition', 'chol', ...
    'green_storage_class', 'single', ...
    'block_size', 64, ...
    'contact_impedance_ohm_m2', cfg.z_contact);
[green, factor_seconds, solve_seconds, cache_reused] = local_bulk_green( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_file, cache_settings);
global_to_green = zeros(size(fmdl_mm.nodes, 1), 1);
global_to_green(candidate_nodes) = 1:numel(candidate_nodes);

z_cem_fast = zeros(n_pose, n_size);
tic_local = tic;
for pose_index = 1:n_pose
    for size_index = 1:n_size
        faces_for_grid = patch_faces(pose_index, size_index, :);
        z_cem_fast(pose_index, size_index) = local_cem_transfer( ...
            fmdl_si, faces_for_grid, global_to_green, green, cfg.z_contact);
    end
end
local_seconds = toc(tic_local);

abs_residual = abs(z_cem_fast - measured_z(:)');
huber_delta = 5;
huber = 0.5 * min(abs_residual, huber_delta).^2 + ...
    huber_delta * max(abs_residual - huber_delta, 0);
slope = zeros(n_pose, 1);
for pose_index = 1:n_pose
    fitted_line = robustfit(sizes_mm, z_cem_fast(pose_index, :)');
    slope(pose_index) = fitted_line(2);
end
span_mm = max(sizes_mm) - min(sizes_mm);
objective = mean(huber, 2) + ...
    0.20 * ((slope - experimental_slope) * span_mm).^2 + ...
    ((h_centre - 15) / 7).^2 + ...
    2 * max(h_centre - 30, 0).^2 + 2 * max(7 - h_centre, 0).^2 + ...
    0.25 * ((rho_lung - cfg.rho_cloud.lungs_ohm_m) / 6)^2;

comparison = readtable(fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_comparison_%s.csv', result_tag)));
validation_row = find(candidates.is_validation, 1);
validation_curve_rmse = sqrt(mean((z_cem_fast(validation_row, :)' ...
    - comparison.Z_FEM_ohm).^2));

output = candidates;
output.h_centre_mm = h_centre;
output.fast_CEM_objective = objective;
output.slope_ohm_per_mm = slope;
for size_index = 1:n_size
    output.(sprintf('Z_fast_CEM_L%03d_ohm', sizes_mm(size_index))) = ...
        z_cem_fast(:, size_index);
end
output = sortrows(output, 'fast_CEM_objective');
output_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_fast_cem_refine_%s.csv', result_tag));
writetable(output, output_file);

summary = table(top_count, local_radius_mm, local_phi_step_deg, ...
    requested_pose_count, rejected_geometry_count, n_pose, numel(candidate_nodes), ...
    geometry_seconds, factor_seconds, solve_seconds, local_seconds, ...
    validation_curve_rmse, cache_reused, ...
    'VariableNames', {'PEM_top_count', 'local_radius_mm', ...
    'local_phi_step_deg', 'requested_pose_count', ...
    'rejected_geometry_count', 'evaluated_pose_count', ...
    'patch_node_count', 'geometry_seconds', 'factor_seconds', ...
    'green_solve_seconds', 'local_CEM_seconds', ...
    'validation_curve_RMSE_ohm', 'cache_reused'});
summary_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_fast_cem_refine_summary_%s.csv', result_tag));
writetable(summary, summary_file);

result = struct('candidates', output, 'summary', summary, ...
    'output_file', output_file, 'summary_file', summary_file, 'cache_file', cache_file);
fprintf('Fast low-rank CEM refinement saved: %s\n', output_file);
fprintf('Validation against full EIDORS curve: RMSE %.9g Ohm.\n', ...
    validation_curve_rmse);
end

function z = local_cem_transfer(fmdl_si, patch_faces, ...
    global_to_green, green_all, z_contact)
% candidate_nodes used to be passed in but was never read: the node set is
% recovered from patch_faces below, and global_to_green already encodes the
% mapping into the precomputed Green block.
all_faces = unique(vertcat(patch_faces{:}));
patch_nodes = unique(reshape(fmdl_si.boundary(all_faces, :), [], 1));
local_green_index = global_to_green(patch_nodes);
if any(local_green_index == 0)
    error('Missing patch node in bulk Green matrix.');
end
G = double(green_all(local_green_index, local_green_index));
n_patch = numel(patch_nodes);
D = zeros(n_patch);
B = zeros(n_patch, 4);
C = zeros(4);
mass_template = [2, 1, 1; 1, 2, 1; 1, 1, 2];

for electrode_index = 1:4
    faces = patch_faces{electrode_index};
    for face_index = reshape(faces, 1, [])
        nodes = fmdl_si.boundary(face_index, :);
        points = fmdl_si.nodes(nodes, :);
        area = 0.5 * norm(cross(points(2, :) - points(1, :), ...
            points(3, :) - points(1, :)));
        [present, local] = ismember(nodes, patch_nodes);
        if ~all(present)
            error('Face node missing from local patch-node list.');
        end
        D(local, local) = D(local, local) + ...
            area / (12 * z_contact) * mass_template;
        B(local, electrode_index) = B(local, electrode_index) ...
            - area / (3 * z_contact);
        C(electrode_index, electrode_index) = ...
            C(electrode_index, electrode_index) + area / z_contact;
    end
end

Q = (eye(n_patch) + G * D) \ (G * B);
schur = C - B' * Q;
electrode_voltage = schur \ [1; 0; 0; -1];
z = electrode_voltage(2) - electrode_voltage(3);
end

function [green, factor_seconds, solve_seconds, reused] = local_bulk_green( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_file, cache_settings)
factor_seconds = 0;
solve_seconds = 0;
reused = false;
cache_schema = 'trkg4_grounded_green_cache_v2';
cache_fingerprint = local_green_cache_fingerprint( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_settings);

if isfile(cache_file)
    variables = whos('-file', cache_file);
    variable_names = {variables.name};
    required = {'green', 'candidate_nodes', ...
        'cache_fingerprint', 'cache_schema'};
    if ~all(ismember(required, variable_names))
        fprintf(['Green cache ignored (legacy cache has no complete ', ...
            'fingerprint); rebuilding: %s\n'], cache_file);
    else
        try
            cached = load(cache_file, required{:});
            cache_matches = isequal(cached.cache_schema, cache_schema) && ...
                isequal(cached.cache_fingerprint, cache_fingerprint) && ...
                isequal(double(cached.candidate_nodes(:)), ...
                double(candidate_nodes(:))) && ...
                isnumeric(cached.green) && ...
                isequal(size(cached.green), ...
                [numel(candidate_nodes), numel(candidate_nodes)]);
            if cache_matches
                green = cached.green;
                reused = true;
                return;
            end
            fprintf(['Green cache ignored (physical-input fingerprint ', ...
                'mismatch); rebuilding: %s\n'], cache_file);
        catch exception
            fprintf(['Green cache ignored (could not validate it: %s); ', ...
                'rebuilding: %s\n'], exception.message, cache_file);
        end
    end
end

model = fmdl_si;
model.electrode = struct([]);
model.stimulation = struct([]);
img = mk_image(model, elem_sigma);
system = calc_system_mat(img);
ground = model.gnd_node;
keep = (1:size(system.E, 1))';
keep(ground) = [];
reduced_index = zeros(size(system.E, 1), 1);
reduced_index(keep) = 1:numel(keep);
candidate_reduced = reduced_index(candidate_nodes);

tic_factor = tic;
factor = decomposition(system.E(keep, keep), 'chol');
factor_seconds = toc(tic_factor);

n = numel(candidate_nodes);
green = zeros(n, n, 'single');
block_size = 64;
tic_solve = tic;
for first = 1:block_size:n
    last = min(first + block_size - 1, n);
    columns = first:last;
    rhs = sparse(candidate_reduced(columns), 1:numel(columns), 1, ...
        numel(keep), numel(columns));
    voltage = factor \ rhs;
    green(:, columns) = single(full(voltage(candidate_reduced, :)));
end
solve_seconds = toc(tic_solve);

cache_schema = 'trkg4_grounded_green_cache_v2';
save(cache_file, 'green', 'candidate_nodes', ...
    'cache_fingerprint', 'cache_schema', '-v7.3');
end

function fingerprint = local_green_cache_fingerprint( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_settings)
fingerprint = trkg4_green_fingerprint( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_settings);
end
function pose = local_build_pose_context(fmdl, root, lung_file)
reference = readtable(fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_140mm_xyz_mm.csv'));
reference_centres = [reference.x_mm, reference.y_mm, reference.z_mm];
boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);
[node_normals, ~] = trkg4_boundary_normals(fmdl);
face_ids = repmat((1:size(fmdl.boundary, 1))', 3, 1);
vertex_faces = accumarray(fmdl.boundary(:), face_ids, ...
    [size(fmdl.nodes, 1), 1], @(x) {x}, {[]});
lung = stlread(lung_file);
pose = struct('fmdl', fmdl, 'boundary_nodes', boundary_nodes, ...
    'boundary_xyz', boundary_xyz, 'node_normals', node_normals, ...
    'vertex_faces', {vertex_faces}, ...
    'reference_centre', mean(reference_centres, 1), ...
    'reference_axis', (reference_centres(4, :) - reference_centres(1, :)) ...
    / norm(reference_centres(4, :) - reference_centres(1, :)), ...
    'lung_vertices', lung.Points);
end

function state = local_pose_state(pose, du, dv, phi_deg)
[c0, n0] = local_project_to_skin(pose, pose.reference_centre);
u0 = pose.reference_axis - dot(pose.reference_axis, n0) * n0;
u0 = u0 / norm(u0);
v0 = cross(n0, u0); v0 = v0 / norm(v0);
[centre, normal] = local_project_to_skin(pose, c0 + du * u0 + dv * v0);
base_axis = u0 - dot(u0, normal) * normal;
base_axis = base_axis / norm(base_axis);
side_axis = cross(normal, base_axis); side_axis = side_axis / norm(side_axis);
axis = cosd(phi_deg) * base_axis + sind(phi_deg) * side_axis;
[~, h] = knnsearch(pose.lung_vertices, centre);
state = struct('centre_xyz_mm', centre, 'axis_unit_xyz', axis / norm(axis), ...
    'h_centre_mm', h);
end

function [centres, requested] = local_grid_centres(pose, state, L)
offset = [-L/2; -L/4; L/4; L/2];
requested = state.centre_xyz_mm + offset .* state.axis_unit_xyz;
centres = zeros(4, 3);
for k = 1:4
    centres(k, :) = local_project_to_skin(pose, requested(k, :));
end
end

function [point, normal] = local_project_to_skin(pose, raw)
[idx, ~] = knnsearch(pose.boundary_xyz, raw, ...
    'K', min(16, size(pose.boundary_xyz, 1)));
global_nodes = pose.boundary_nodes(idx(:));
candidate_cells = pose.vertex_faces(global_nodes);
candidate_faces = unique(vertcat(candidate_cells{:}));
best = Inf; point = raw; normal = [0, 0, 0];
for face = reshape(candidate_faces, 1, [])
    nodes = pose.fmdl.boundary(face, :);
    tri = pose.fmdl.nodes(nodes, :);
    [candidate, bary] = local_closest_point_triangle(raw, ...
        tri(1, :), tri(2, :), tri(3, :));
    distance = sum((candidate - raw).^2);
    if distance < best
        best = distance; point = candidate;
        normal = bary * pose.node_normals(nodes, :);
    end
end
normal = normal / norm(normal);
end

function [q, bary] = local_closest_point_triangle(p, a, b, c)
ab = b-a; ac = c-a; ap = p-a; d1=dot(ab,ap); d2=dot(ac,ap);
if d1<=0 && d2<=0, q=a; bary=[1,0,0]; return; end
bp=p-b; d3=dot(ab,bp); d4=dot(ac,bp);
if d3>=0 && d4<=d3, q=b; bary=[0,1,0]; return; end
vc=d1*d4-d3*d2;
if vc<=0 && d1>=0 && d3<=0
    v=d1/(d1-d3); q=a+v*ab; bary=[1-v,v,0]; return;
end
cp=p-c; d5=dot(ab,cp); d6=dot(ac,cp);
if d6>=0 && d5<=d6, q=c; bary=[0,0,1]; return; end
vb=d5*d2-d1*d6;
if vb<=0 && d2>=0 && d6<=0
    w=d2/(d2-d6); q=a+w*ac; bary=[1-w,0,w]; return;
end
va=d3*d6-d5*d4;
if va<=0 && (d4-d3)>=0 && (d5-d6)>=0
    w=(d4-d3)/((d4-d3)+(d5-d6)); q=b+w*(c-b); bary=[0,1-w,w]; return;
end
denom=1/(va+vb+vc); v=vb*denom; w=vc*denom;
bary=[1-v-w,v,w]; q=bary(1)*a+bary(2)*b+bary(3)*c;
end
