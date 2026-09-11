function result = run_trkg4_inverse_inhale(max_evaluations, initial_points, ...
    compute_jacobian, data_selection, result_tag, patch_builder_mode, ...
    evaluation_only)
%RUN_TRKG4_INVERSE_INHALE Fit CT-FEM to the multi-grid inhale experiment.
% All rigid grids share one skin centre and one nominal surface axis. The
% fitted pose determines the non-constant skin-to-lung depth in the CT
% geometry; h is therefore a derived curve, not a second flat-layer input.
% Optional data_selection/result_tag/patch_builder_mode arguments support
% non-destructive ablations. The production patch builder is "disjoint";
% "legacy" reproduces the earliest independent-face-growth semantics;
% "legacy_fast" reconstructs the later node-disjoint fast implementation.
% evaluation_only returns after one objective evaluation and is used by the
% restartable batch driver to bound MATLAB memory use on the full v5 mesh.

if nargin < 1 || isempty(max_evaluations)
    max_evaluations = 30;
end
if nargin < 2
    initial_points = [];
end
if nargin < 3 || isempty(compute_jacobian)
    compute_jacobian = true;
end
if nargin < 4 || strlength(string(data_selection)) == 0
    data_selection = "manifest";
end
if nargin < 5 || strlength(string(result_tag)) == 0
    result_tag = "v5_1mm_local2mm";
end
if nargin < 6 || strlength(string(patch_builder_mode)) == 0
    patch_builder_mode = "disjoint";
end
if nargin < 7 || isempty(evaluation_only)
    evaluation_only = false;
end
data_selection = string(data_selection);
result_tag = string(result_tag);
patch_builder_mode = lower(string(patch_builder_mode));
if ~any(patch_builder_mode == ["disjoint", "legacy", "legacy_fast"])
    error('trkg4:invalidPatchBuilderMode', ...
        ['Patch builder mode must be "disjoint", "legacy", or ', ...
         '"legacy_fast".']);
end
if patch_builder_mode == "legacy" && result_tag == "v5_1mm_local2mm"
    result_tag = "v5_1mm_local2mm_legacy_patch";
end
if patch_builder_mode == "legacy_fast" && result_tag == "v5_1mm_local2mm"
    result_tag = "v5_1mm_local2mm_legacy_fast_patch";
end
if isempty(regexp(result_tag, '^[A-Za-z0-9_-]+$', 'once'))
    error('trkg4:invalidInverseResultTag', ...
        'Result tag must contain only letters, digits, underscores, or hyphens.');
end

cfg = trkg4_config('nik');
trkg4_prepare_runtime(cfg);
root = cfg.project_root;
shared_file = fullfile(root, 'output', ...
    'nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
if ~isfile(shared_file)
    error('Shared FEM result is missing: %s', shared_file);
end

shared = load(shared_file, 'fmdl_mesh_units', 'elem_sigma', ...
    'tissue_id', 'tissue_names');
fmdl_mm = shared.fmdl_mesh_units;
fmdl_si = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);
[fmdl_mm.stimulation, fmdl_mm.meas_select] = ...
    trkg4_make_4electrode_stimulation(cfg);
[fmdl_si.stimulation, fmdl_si.meas_select] = ...
    trkg4_make_4electrode_stimulation(cfg);

[sizes_mm, measured_z_ohm, source_files] = ...
    trkg4_load_nik_inhale_data(root, data_selection);
reference_centres = local_load_reference_grid(root);
pose = local_build_pose_context(fmdl_mm, reference_centres, ...
    cfg.subject.stl.lungs);

soft_id = find(strcmp(shared.tissue_names, 'soft_tissue'), 1);
lung_id = find(strcmp(shared.tissue_names, 'lungs'), 1);
if isempty(soft_id) || isempty(lung_id)
    error('Expected soft_tissue and lungs in the shared tissue assignment.');
end

experimental_line = robustfit(sizes_mm, measured_z_ohm);
experimental_slope = experimental_line(2);

% x = [rho_soft, rho_lung, centre_u_mm, centre_v_mm, phi_deg]
x0 = [cfg.rho_cloud.soft_ohm_m, cfg.rho_cloud.lungs_ohm_m, 0, 0, 0];
lb = [2.0, 10.0, -120, -120, -20];
ub = [10.0, 32.0, 120, 120, 20];
if ~isempty(initial_points)
    x0 = initial_points(1, :);
end

history_matrix = zeros(0, 9 + numel(sizes_mm));
history_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_history_%s.csv', result_tag));
eval_count = 0;
best_sampled_objective = inf;
best_sampled_x = x0;
best_sampled_evaluation = NaN;

    function objective_value = objective(x)
        evaluation_cache_cleanup = ...
            onCleanup(@() eidors_cache('clear_all'));
        eval_count = eval_count + 1;
        [predicted_z, pose_state] = local_forward_curve(x, sizes_mm, ...
            fmdl_mm, fmdl_si, pose, cfg, shared.elem_sigma, ...
            shared.tissue_id, soft_id, lung_id, patch_builder_mode);

        residual = predicted_z - measured_z_ohm;
        huber_delta_ohm = 5;
        abs_r = abs(residual);
        huber = 0.5 * min(abs_r, huber_delta_ohm).^2 + ...
            huber_delta_ohm * max(abs_r - huber_delta_ohm, 0);

        predicted_line = robustfit(sizes_mm, predicted_z);
        predicted_slope = predicted_line(2);
        span_mm = max(sizes_mm) - min(sizes_mm);
        slope_span_error = (predicted_slope - experimental_slope) * span_mm;

        % The experiment constrains the common axis to the lung field and
        % the CT visual estimate was about 15 mm. Keep that estimate soft,
        % but reject the anatomically different 40--60 mm deep solutions
        % that an unconstrained h<->rho_lung trade-off otherwise creates.
        depth_prior = ((pose_state.h_centre_mm - 15) / 7)^2;
        depth_range_penalty = 2 * max(pose_state.h_centre_mm - 30, 0)^2 + ...
            2 * max(7 - pose_state.h_centre_mm, 0)^2;
        rho_lung_prior = ((x(2) - cfg.rho_cloud.lungs_ohm_m) / 6)^2;
        objective_value = mean(huber) + 0.20 * slope_span_error^2 + ...
            depth_prior + depth_range_penalty + 0.25 * rho_lung_prior;
        if isfinite(objective_value) && objective_value < best_sampled_objective
            best_sampled_objective = objective_value;
            best_sampled_x = x;
            best_sampled_evaluation = eval_count;
        end

        history_matrix(end + 1, :) = [eval_count, x, objective_value, ...
            predicted_slope, pose_state.h_centre_mm, predicted_z(:)'];
        local_write_history(history_matrix, sizes_mm, history_file);
        fprintf(['Inverse eval %03d: J=%9.4f, rho=(%.3f, %.3f), ', ...
            'du/dv=(%+.1f,%+.1f) mm, phi=%+.1f deg, ', ...
            'h_c=%.1f mm, slope=%+.5f Ohm/mm\n'], ...
            eval_count, objective_value, x(1), x(2), x(3), x(4), ...
            x(5), pose_state.h_centre_mm, predicted_slope);
    end

initial_objective = objective(x0);
if evaluation_only
    first_history = local_history_table(history_matrix, sizes_mm);
    result = struct();
    result.parameters = struct('rho_soft_ohm_m', x0(1), ...
        'rho_lungs_ohm_m', x0(2), 'centre_u_mm', x0(3), ...
        'centre_v_mm', x0(4), 'phi_deg', x0(5));
    result.objective = initial_objective;
    result.experimental_robust_slope_ohm_per_mm = experimental_slope;
    result.fem_robust_slope_ohm_per_mm = first_history.slope_ohm_per_mm(1);
    result.h_centre_mm = first_history.h_centre_mm(1);
    result.history = first_history;
    result.data_selection = data_selection;
    result.patch_builder_mode = patch_builder_mode;
    result.result_tag = result_tag;
    return;
end

if max_evaluations > 0
    if exist('surrogateopt', 'file') == 2
        solver_name = "surrogateopt";
        opts = optimoptions('surrogateopt', ...
            'MaxFunctionEvaluations', max_evaluations, ...
            'Display', 'iter', ...
            'UseParallel', false, ...
            'PlotFcn', []);
        if ~isempty(initial_points)
            opts.InitialPoints = initial_points;
            opts.MinSurrogatePoints = max(12, size(initial_points, 1) + 5);
        end
        [x_best, best_objective, exitflag, solver_output] = ...
            surrogateopt(@objective, lb, ub, opts);
    else
        warning('trkg4:boundedPatternFallback', [ ...
            'surrogateopt is unavailable. Using a deterministic bounded ', ...
            'pattern search with millimetre-scale pose steps. This avoids ', ...
            'finite-difference gradients on discrete electrode patches.']);
        solver_name = "bounded_pattern_search";
        [x_best, best_objective, exitflag, solver_output] = ...
            local_bounded_pattern_search(@objective, x0, initial_objective, ...
            initial_points, lb, ub, max_evaluations);
    end
    if best_sampled_objective < best_objective
        x_best = best_sampled_x;
        best_objective = best_sampled_objective;
        solver_output.message = sprintf(['%s The reported point is evaluation %d, ', ...
            'the lowest finite objective actually computed before termination.'], ...
            solver_output.message, best_sampled_evaluation);
    end
else
    x_best = x0;
    best_objective = initial_objective;
    exitflag = 0;
    solver_name = "initial_point_only";
    solver_output = struct('message', 'Initial-point evaluation only.');
end

solver_output.best_sampled_evaluation = best_sampled_evaluation;
solver_output.best_sampled_objective = best_sampled_objective;
[predicted_z_ohm, pose_state] = local_forward_curve(x_best, sizes_mm, ...
    fmdl_mm, fmdl_si, pose, cfg, shared.elem_sigma, shared.tissue_id, ...
    soft_id, lung_id, patch_builder_mode);
[h_axis_mm, h_nearest_mm, skin_axis_xyz_mm] = ...
    local_depth_curve(pose, pose_state, -70:5:70);
[electrode_geometry, grid_geometry] = local_electrode_geometry( ...
    sizes_mm, pose, pose_state, fmdl_mm, cfg, patch_builder_mode);

fit_line = robustfit(sizes_mm, predicted_z_ohm);
residual_ohm = predicted_z_ohm - measured_z_ohm;
if compute_jacobian
    [jacobian, sensitivity, jacobian_svd] = local_parameter_jacobian( ...
        x_best, predicted_z_ohm, sizes_mm, fmdl_mm, fmdl_si, pose, cfg, shared.elem_sigma, ...
        shared.tissue_id, soft_id, lung_id, patch_builder_mode, lb, ub);
else
    jacobian = nan(numel(sizes_mm), 5);
    sensitivity = table();
    jacobian_svd = struct('condition', NaN, ...
        'singular_values', nan(5, 1));
end
comparison = table(sizes_mm, measured_z_ohm, predicted_z_ohm, ...
    residual_ohm, source_files, 'VariableNames', { ...
    'L_mm', 'Z_experiment_ohm', 'Z_FEM_ohm', 'residual_ohm', 'source_file'});
comparison_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_comparison_%s.csv', result_tag));
writetable(comparison, comparison_file);

depth_curve = table((-70:5:70)', h_axis_mm, h_nearest_mm, ...
    skin_axis_xyz_mm(:, 1), skin_axis_xyz_mm(:, 2), skin_axis_xyz_mm(:, 3), ...
    'VariableNames', {'axis_s_mm', 'h_inward_ray_mm', ...
    'h_nearest_lung_mm', 'skin_x_mm', 'skin_y_mm', 'skin_z_mm'});
depth_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_depth_curve_%s.csv', result_tag));
writetable(depth_curve, depth_file);

jacobian_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_jacobian_%s.csv', result_tag));
jacobian_table = array2table([sizes_mm, jacobian], 'VariableNames', { ...
    'L_mm', 'dZ_drho_soft', 'dZ_drho_lungs', 'dZ_dcentre_u_per_mm', ...
    'dZ_dcentre_v_per_mm', 'dZ_dphi_per_deg'});
writetable(jacobian_table, jacobian_file);
sensitivity_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_parameter_sensitivity_%s.csv', result_tag));
if ~isempty(sensitivity)
    writetable(sensitivity, sensitivity_file);
end
fit_summary = table(x_best(1), x_best(2), x_best(3), x_best(4), x_best(5), ...
    pose_state.centre_xyz_mm(1), pose_state.centre_xyz_mm(2), ...
    pose_state.centre_xyz_mm(3), pose_state.axis_unit_xyz(1), ...
    pose_state.axis_unit_xyz(2), pose_state.axis_unit_xyz(3), ...
    pose_state.h_centre_mm, experimental_slope, fit_line(2), ...
    sqrt(mean(residual_ohm.^2)), mean(abs(residual_ohm)), ...
    jacobian_svd.condition, size(fmdl_mm.nodes, 1), ...
    size(fmdl_mm.elems, 1), solver_name, exitflag, eval_count, ...
    data_selection, patch_builder_mode, string(shared_file), ...
    'VariableNames', {'rho_soft_ohm_m', 'rho_lungs_ohm_m', ...
    'centre_u_mm', 'centre_v_mm', 'phi_deg', 'centre_x_mm', ...
    'centre_y_mm', 'centre_z_mm', 'axis_x', 'axis_y', 'axis_z', ...
    'h_centre_mm', 'experimental_slope_ohm_per_mm', ...
    'fem_slope_ohm_per_mm', 'rms_residual_ohm', 'mae_residual_ohm', ...
    'jacobian_condition_scaled', 'mesh_nodes', 'mesh_tetrahedra', ...
    'solver_name', 'solver_exitflag', 'function_evaluations', ...
    'data_selection', 'patch_builder_mode', ...
    'source_mesh_result'});
fit_summary_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_fit_summary_%s.csv', result_tag));
writetable(fit_summary, fit_summary_file);
electrode_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_electrodes_%s.csv', result_tag));
writetable(electrode_geometry, electrode_file);
grid_geometry_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_grid_geometry_%s.csv', result_tag));
writetable(grid_geometry, grid_geometry_file);

result = struct();
result.parameters = struct('rho_soft_ohm_m', x_best(1), ...
    'rho_lungs_ohm_m', x_best(2), 'centre_u_mm', x_best(3), ...
    'centre_v_mm', x_best(4), 'phi_deg', x_best(5));
result.centre_xyz_mm = pose_state.centre_xyz_mm;
result.axis_unit_xyz = pose_state.axis_unit_xyz;
result.h_centre_mm = pose_state.h_centre_mm;
result.depth_curve = depth_curve;
result.electrode_geometry = electrode_geometry;
result.grid_geometry = grid_geometry;
result.comparison = comparison;
result.jacobian = jacobian_table;
result.parameter_sensitivity = sensitivity;
result.fit_summary = fit_summary;
result.jacobian_singular_values_scaled = jacobian_svd.singular_values;
result.jacobian_condition_scaled = jacobian_svd.condition;
result.experimental_robust_slope_ohm_per_mm = experimental_slope;
result.fem_robust_slope_ohm_per_mm = fit_line(2);
result.objective = best_objective;
result.exitflag = exitflag;
result.solver_name = solver_name;
result.solver_output = solver_output;
result.data_selection = data_selection;
result.patch_builder_mode = patch_builder_mode;
result.result_tag = result_tag;
result.history = local_history_table(history_matrix, sizes_mm);
result.assumptions = { ...
    'All rigid grids have one common centre on the skin surface.', ...
    'Nominal electrode offsets are [-L/2,-L/4,+L/4,+L/2].', ...
    'Heart and bone conductivities stay at their shared literature values.', ...
    ['CT geometry supplies the non-constant h(s); the approximate ', ...
    '15 mm measurement is a soft prior and h outside 7--30 mm is penalized.']};

result_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_%s.mat', result_tag));
save(result_file, '-struct', 'result', '-v7.3');
fprintf('\nSaved inverse result: %s\n', result_file);
fprintf('Saved FEM/experiment comparison: %s\n', comparison_file);
fprintf('Saved CT depth curve: %s\n', depth_file);
fprintf('Saved inverse Jacobian: %s\n', jacobian_file);
fprintf('Saved inverse fit summary: %s\n', fit_summary_file);
fprintf('Saved fitted electrode coordinates: %s\n', electrode_file);
fprintf('Saved fitted grid geometry: %s\n', grid_geometry_file);
fprintf('Scaled Jacobian condition number: %.6g\n', jacobian_svd.condition);
end

function [x_best, f_best, exitflag, output] = ...
        local_bounded_pattern_search(fun, x0, f0, initial_points, lb, ub, budget)
%LOCAL_BOUNDED_PATTERN_SEARCH Direct search for the discontinuous pose terms.
% The rho coordinates are continuous, but a small change of pose changes the
% model only when a boundary face enters or leaves an electrode.  SQP finite
% differences therefore see false zero derivatives.  This compact fallback
% evaluates meaningful coordinate steps, respects bounds, and uses every
% supplied initial point as an explicit start candidate.

x_best = x0;
f_best = f0;
evaluations = 0;
evaluated = x0;
if isempty(initial_points)
    initial_points = x0;
end
initial_points = min(max(initial_points, lb), ub);
initial_points = unique(initial_points, 'rows', 'stable');
for row = 1:size(initial_points, 1)
    candidate = initial_points(row, :);
    if any(all(abs(evaluated - candidate) < 1e-12, 2))
        continue;
    end
    [value, ok] = local_safe_objective(fun, candidate);
    evaluations = evaluations + 1;
    evaluated(end + 1, :) = candidate; %#ok<AGROW>
    if ok && value < f_best
        x_best = candidate;
        f_best = value;
    end
    if evaluations >= budget
        break;
    end
end

step = [0.75, 3.0, 10.0, 10.0, 5.0];
minimum_step = [0.05, 0.25, 0.5, 0.5, 0.5];
sweep = 0;
while evaluations < budget && any(step >= minimum_step)
    sweep = sweep + 1;
    sweep_start_value = f_best;
    sweep_best_x = x_best;
    sweep_best_value = f_best;
    for parameter = 1:numel(x_best)
        for direction = [-1, 1]
            if evaluations >= budget
                break;
            end
            candidate = x_best;
            candidate(parameter) = min(max( ...
                candidate(parameter) + direction * step(parameter), ...
                lb(parameter)), ub(parameter));
            if any(all(abs(evaluated - candidate) < 1e-12, 2))
                continue;
            end
            [value, ok] = local_safe_objective(fun, candidate);
            evaluations = evaluations + 1;
            evaluated(end + 1, :) = candidate; %#ok<AGROW>
            if ok && value < sweep_best_value
                sweep_best_x = candidate;
                sweep_best_value = value;
            end
        end
    end
    if sweep_best_value < f_best
        x_best = sweep_best_x;
        f_best = sweep_best_value;
    else
        step = step / 2;
    end
    fprintf(['Pattern sweep %02d: best J=%.6g, ', ...
        'step=(%.3g, %.3g, %.3g, %.3g, %.3g)\n'], ...
        sweep, f_best, step);
    if f_best >= sweep_start_value && all(step < minimum_step)
        break;
    end
end

exitflag = double(all(step < minimum_step));
if exitflag
    message = 'Pattern steps fell below their predefined resolution.';
else
    message = 'Function-evaluation budget reached.';
end
output = struct('message', message, 'iterations', sweep, ...
    'function_evaluations_after_initial', evaluations, ...
    'final_step', step, 'algorithm', 'bounded coordinate pattern search');
end

function [value, ok] = local_safe_objective(fun, candidate)
try
    value = fun(candidate);
    ok = isfinite(value);
catch exception
    invalid_patch = startsWith(exception.identifier, ...
        'electrode_faces_by_area:') || startsWith(exception.identifier, ...
        'electrode_faces_by_area_legacy:') || startsWith( ...
        exception.identifier, 'electrode_faces_by_area_legacy_fast:');
    if ~invalid_patch
        rethrow(exception);
    end
    value = Inf;
    ok = false;
end
end


function centres = local_load_reference_grid(root)
file = fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_140mm_xyz_mm.csv');
T = readtable(file, 'TextType', 'string');
centres = [T.x_mm, T.y_mm, T.z_mm];
end

function pose = local_build_pose_context(fmdl, reference_centres, lung_file)
boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);
node_normals = local_boundary_node_normals(fmdl);
face_ids = repmat((1:size(fmdl.boundary, 1))', 3, 1);
vertex_faces = accumarray(fmdl.boundary(:), face_ids, ...
    [size(fmdl.nodes, 1), 1], @(x) {x}, {[]});

reference_centre = mean(reference_centres, 1);
reference_axis = reference_centres(4, :) - reference_centres(1, :);
reference_axis = reference_axis / norm(reference_axis);

lung_tr = stlread(lung_file);
lung_vertices = lung_tr.Points;

pose = struct('fmdl', fmdl, 'boundary_nodes', boundary_nodes, ...
    'boundary_xyz', boundary_xyz, 'node_normals', node_normals, ...
    'vertex_faces', {vertex_faces}, ...
    'reference_centre', reference_centre, ...
    'reference_axis', reference_axis, 'lung_vertices', lung_vertices, ...
    'lung_faces', lung_tr.ConnectivityList);

% Patch precomputation depends only on the mesh, and the mesh is fixed for
% the whole fit. Building it here turns the per-evaluation patch rebuild from
% "reindex the whole boundary" into "walk the patch perimeter".
pose.patch_context = electrode_patch_context(fmdl);
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
    contribution = repmat(face_normals(:, d), 3, 1);
    normals(:, d) = accumarray(faces(:), contribution, ...
        [size(nodes, 1), 1], @sum, 0);
end
lengths = vecnorm(normals, 2, 2);
valid = lengths > 0;
normals(valid, :) = normals(valid, :) ./ lengths(valid);
end

function [z_curve, state] = local_forward_curve(x, sizes, fmdl_mm, ...
    fmdl_si, pose, cfg, base_sigma, tissue_id, soft_id, lung_id, patch_builder_mode)
elem_sigma = base_sigma;
elem_sigma(tissue_id == soft_id) = 1 / x(1);
elem_sigma(tissue_id == lung_id) = 1 / x(2);

state = local_pose_state(pose, x(3), x(4), x(5));
z_curve = zeros(numel(sizes), 1);
for k = 1:numel(sizes)
    % Centres are projected onto the skin surface exactly, and deliberately
    % NOT snapped to boundary nodes the way the forward runner does. The pose
    % offsets x(3:5) are continuous fit parameters; snapping would quantise
    % them to the node spacing and turn the objective into a staircase.
    centres = local_grid_centres(pose, state, sizes(k));
    model_mm = fmdl_mm;
    [model_mm.electrode, ~] = local_build_electrode_patches(model_mm, ...
        centres, cfg, pose.patch_context, patch_builder_mode);
    model_si = fmdl_si;
    model_si.electrode = model_mm.electrode;
    img = mk_image(model_si, elem_sigma);
    voltage = fwd_solve(img);
    z_curve(k) = real(voltage.meas(1) / cfg.current_ampere);

    % Every electrode size creates a different CEM system matrix. Keeping all
    % nine matrices in the EIDORS cache can exhaust memory during a multi-start
    % inverse search even though no later size reuses them. Release each one
    % immediately after extracting its scalar impedance.
    clear voltage img model_si model_mm
    eidors_cache('clear_all');
end
end

function state = local_pose_state(pose, du_mm, dv_mm, phi_deg)
% Build the reference tangent basis, move the centre, then rebuild the
% tangent plane at the new skin point.
[c0, n0] = local_project_to_skin(pose, pose.reference_centre);
u0 = pose.reference_axis - dot(pose.reference_axis, n0) * n0;
u0 = u0 / norm(u0);
v0 = cross(n0, u0);
v0 = v0 / norm(v0);

raw_centre = c0 + du_mm * u0 + dv_mm * v0;
[centre, normal] = local_project_to_skin(pose, raw_centre);
base_axis = u0 - dot(u0, normal) * normal;
base_axis = base_axis / norm(base_axis);
side_axis = cross(normal, base_axis);
side_axis = side_axis / norm(side_axis);
phi = deg2rad(phi_deg);
axis = cos(phi) * base_axis + sin(phi) * side_axis;
axis = axis / norm(axis);

[~, h_centre_mm] = knnsearch(pose.lung_vertices, centre);
state = struct('centre_xyz_mm', centre, 'normal_unit_xyz', normal, ...
    'axis_unit_xyz', axis, 'side_unit_xyz', side_axis, ...
    'h_centre_mm', h_centre_mm);
end

function centres = local_grid_centres(pose, state, L_mm)
offset = [-L_mm / 2; -L_mm / 4; L_mm / 4; L_mm / 2];
raw = state.centre_xyz_mm + offset .* state.axis_unit_xyz;
centres = zeros(4, 3);
for k = 1:4
    centres(k, :) = local_project_to_skin(pose, raw(k, :));
end
end

function [electrodes_table, grids_table] = local_electrode_geometry( ...
    sizes, pose, state, fmdl_mm, cfg, patch_builder_mode)
% Report both the requested skin points and the area-weighted centroids of
% the electrode patches selected on the discrete FEM boundary.
labels = ["I_plus"; "V_plus"; "V_minus"; "I_minus"];
n_grids = numel(sizes);
n_electrodes = 4 * n_grids;
L_column = zeros(n_electrodes, 1);
label_column = strings(n_electrodes, 1);
requested_xyz = zeros(n_electrodes, 3);
centroid_xyz = zeros(n_electrodes, 3);
area_mm2 = zeros(n_electrodes, 1);

actual_current_span = zeros(n_grids, 1);
actual_voltage_span = zeros(n_grids, 1);
actual_centre_xyz = zeros(n_grids, 3);
row = 0;
for k = 1:n_grids
    requested = local_grid_centres(pose, state, sizes(k));
    model = fmdl_mm;
    [model.electrode, induced_faces] = local_build_electrode_patches( ...
        model, requested, cfg, pose.patch_context, patch_builder_mode);
    patch_centres = zeros(4, 3);
    for e = 1:4
        row = row + 1;
        % The builder already returns the induced faces, i.e. the surface the
        % CEM integrates over; recomputing them with ismember over the whole
        % boundary would only repeat that work.
        patch_faces = model.boundary(induced_faces{e}, :);
        [patch_centres(e, :), area_mm2(row)] = ...
            local_area_weighted_face_centre(model.nodes, patch_faces);
        L_column(row) = sizes(k);
        label_column(row) = labels(e);
        requested_xyz(row, :) = requested(e, :);
        centroid_xyz(row, :) = patch_centres(e, :);
    end
    actual_current_span(k) = norm(patch_centres(4, :) - patch_centres(1, :));
    actual_voltage_span(k) = norm(patch_centres(3, :) - patch_centres(2, :));
    actual_centre_xyz(k, :) = mean(patch_centres, 1);
end

electrodes_table = table(L_column, label_column, ...
    requested_xyz(:, 1), requested_xyz(:, 2), requested_xyz(:, 3), ...
    centroid_xyz(:, 1), centroid_xyz(:, 2), centroid_xyz(:, 3), area_mm2, ...
    'VariableNames', {'L_mm', 'electrode', 'requested_x_mm', ...
    'requested_y_mm', 'requested_z_mm', 'patch_centroid_x_mm', ...
    'patch_centroid_y_mm', 'patch_centroid_z_mm', 'patch_area_mm2'});
grids_table = table(sizes(:), actual_current_span, actual_voltage_span, ...
    actual_centre_xyz(:, 1), actual_centre_xyz(:, 2), ...
    actual_centre_xyz(:, 3), 'VariableNames', {'L_nominal_mm', ...
    'I_to_I_actual_mm', 'V_to_V_actual_mm', 'patch_mean_x_mm', ...
    'patch_mean_y_mm', 'patch_mean_z_mm'});
end

function [centre, area] = local_area_weighted_face_centre(nodes, faces)
if isempty(faces)
    centre = [NaN, NaN, NaN];
    area = 0;
    return;
end
p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
face_area = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);
face_centre = (p1 + p2 + p3) / 3;
area = sum(face_area);
centre = sum(face_centre .* face_area, 1) / area;
end

function [point, normal] = local_project_to_skin(pose, raw_point)
[idx, ~] = knnsearch(pose.boundary_xyz, raw_point, ...
    'K', min(16, size(pose.boundary_xyz, 1)));
global_nodes = pose.boundary_nodes(idx(:));
candidate_cells = pose.vertex_faces(global_nodes);
candidate_faces = unique(vertcat(candidate_cells{:}));

best_distance_sq = Inf;
point = raw_point;
normal = [0, 0, 0];
for j = reshape(candidate_faces, 1, [])
    face_nodes = pose.fmdl.boundary(j, :);
    triangle = pose.fmdl.nodes(face_nodes, :);
    [candidate, barycentric] = local_closest_point_triangle( ...
        raw_point, triangle(1, :), triangle(2, :), triangle(3, :));
    distance_sq = sum((candidate - raw_point).^2);
    if distance_sq < best_distance_sq
        best_distance_sq = distance_sq;
        point = candidate;
        normal = barycentric * pose.node_normals(face_nodes, :);
    end
end
if norm(normal) == 0
    normal = point - mean(pose.fmdl.nodes, 1);
end
normal = normal / norm(normal);
end

function [q, bary] = local_closest_point_triangle(p, a, b, c)
% Closest point and barycentric weights; region tests from Ericson's
% Real-Time Collision Detection.
ab = b - a;
ac = c - a;
ap = p - a;
d1 = dot(ab, ap);
d2 = dot(ac, ap);
if d1 <= 0 && d2 <= 0
    q = a; bary = [1, 0, 0]; return;
end

bp = p - b;
d3 = dot(ab, bp);
d4 = dot(ac, bp);
if d3 >= 0 && d4 <= d3
    q = b; bary = [0, 1, 0]; return;
end

vc = d1 * d4 - d3 * d2;
if vc <= 0 && d1 >= 0 && d3 <= 0
    v = d1 / (d1 - d3);
    q = a + v * ab; bary = [1 - v, v, 0]; return;
end

cp = p - c;
d5 = dot(ab, cp);
d6 = dot(ac, cp);
if d6 >= 0 && d5 <= d6
    q = c; bary = [0, 0, 1]; return;
end

vb = d5 * d2 - d1 * d6;
if vb <= 0 && d2 >= 0 && d6 <= 0
    w = d2 / (d2 - d6);
    q = a + w * ac; bary = [1 - w, 0, w]; return;
end

va = d3 * d6 - d5 * d4;
if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    q = b + w * (c - b); bary = [0, 1 - w, w]; return;
end

denom = 1 / (va + vb + vc);
v = vb * denom;
w = vc * denom;
bary = [1 - v - w, v, w];
q = bary(1) * a + bary(2) * b + bary(3) * c;
end

function [h_ray, h_nearest, skin_xyz] = local_depth_curve(pose, state, s_mm)
skin_xyz = zeros(numel(s_mm), 3);
h_ray = NaN(numel(s_mm), 1);
h_nearest = zeros(numel(s_mm), 1);
for k = 1:numel(s_mm)
    raw = state.centre_xyz_mm + s_mm(k) * state.axis_unit_xyz;
    [skin_xyz(k, :), normal] = local_project_to_skin(pose, raw);
    [~, h_nearest(k)] = knnsearch(pose.lung_vertices, skin_xyz(k, :));
    h_ray(k) = local_ray_mesh_distance(skin_xyz(k, :), -normal, ...
        pose.lung_vertices, pose.lung_faces);
end
end

function distance = local_ray_mesh_distance(origin, direction, vertices, faces)
v0 = vertices(faces(:, 1), :);
edge1 = vertices(faces(:, 2), :) - v0;
edge2 = vertices(faces(:, 3), :) - v0;
direction = direction / norm(direction);
h = cross(repmat(direction, size(edge2, 1), 1), edge2, 2);
a = dot(edge1, h, 2);
valid = abs(a) > 1e-10;
f = zeros(size(a));
f(valid) = 1 ./ a(valid);
s = origin - v0;
u = f .* dot(s, h, 2);
q = cross(s, edge1, 2);
v = f .* (q * direction');
t = f .* dot(q, edge2, 2);
hit = valid & u >= -1e-9 & v >= -1e-9 & (u + v) <= 1 + 1e-9 & t > 1e-9;
if any(hit)
    distance = min(t(hit));
else
    distance = NaN;
end
end

function [J, sensitivity, diagnostics] = local_parameter_jacobian(x, z0, sizes, ...
    fmdl_mm, fmdl_si, pose, cfg, base_sigma, tissue_id, soft_id, lung_id, ...
    patch_builder_mode, lb, ub)
parameter_names = ["rho_soft_ohm_m"; "rho_lungs_ohm_m"; ...
    "centre_u_mm"; "centre_v_mm"; "phi_deg"];
step = [0.20, 0.50, 4.0, 4.0, 2.0];
plausible_scale = [1.0, 5.0, 10.0, 10.0, 10.0];
J = zeros(numel(sizes), numel(x));
effective_step = zeros(numel(x), 1);
difference_scheme = strings(numel(x), 1);
for j = 1:numel(x)
    trial_step = step(j);
    success = false;
    for attempt = 1:4
        xp = x;
        xm = x;
        xp(j) = min(x(j) + trial_step, ub(j));
        xm(j) = max(x(j) - trial_step, lb(j));
        [zp, plus_ok] = local_try_curve(xp);
        [zm, minus_ok] = local_try_curve(xm);
        if plus_ok && minus_ok && xp(j) > xm(j)
            J(:, j) = (zp - zm) / (xp(j) - xm(j));
            effective_step(j) = xp(j) - xm(j);
            difference_scheme(j) = "central";
            success = true;
            break
        end
        if attempt == 4 && plus_ok && xp(j) ~= x(j)
            J(:, j) = (zp - z0) / (xp(j) - x(j));
            effective_step(j) = xp(j) - x(j);
            difference_scheme(j) = "forward";
            success = true;
            break
        end
        if attempt == 4 && minus_ok && xm(j) ~= x(j)
            J(:, j) = (z0 - zm) / (x(j) - xm(j));
            effective_step(j) = x(j) - xm(j);
            difference_scheme(j) = "backward";
            success = true;
            break
        end
        trial_step = trial_step / 2;
    end
    if ~success
        error('trkg4:jacobianGeometryInvalid', ...
            'No valid finite-difference geometry for parameter %s.', ...
            parameter_names(j));
    end
end

J_scaled = J .* plausible_scale;
singular_values = svd(J_scaled, 'econ');
if singular_values(end) > 0
    condition = singular_values(1) / singular_values(end);
else
    condition = Inf;
end

assumed_sigma_ohm = 5;
covariance = assumed_sigma_ohm^2 * pinv(J' * J);
crlb_std = sqrt(max(diag(covariance), 0));
sensitivity = table(parameter_names, step(:), effective_step, ...
    difference_scheme, plausible_scale(:), vecnorm(J, 2, 1)', crlb_std, ...
    'VariableNames', {'parameter', 'finite_difference_step_requested', ...
    'finite_difference_step_effective', 'difference_scheme', ...
    'plausible_scale', 'jacobian_column_norm', ...
    'local_crlb_std_at_sigma5ohm'});
diagnostics = struct('singular_values', singular_values, ...
    'condition', condition, 'assumed_sigma_ohm', assumed_sigma_ohm);

    function [z, ok] = local_try_curve(x_trial)
        cache_cleanup = onCleanup(@() eidors_cache('clear_all'));
        try
            z = local_forward_curve(x_trial, sizes, fmdl_mm, fmdl_si, ...
                pose, cfg, base_sigma, tissue_id, soft_id, lung_id, patch_builder_mode);
            ok = true;
        catch exception
            invalid_patch = startsWith(exception.identifier, ...
                'electrode_faces_by_area:') || startsWith(exception.identifier, ...
                'electrode_faces_by_area_legacy:') || startsWith( ...
                exception.identifier, 'electrode_faces_by_area_legacy_fast:');
            if invalid_patch
                z = [];
                ok = false;
            else
                rethrow(exception);
            end
        end
    end
end

function [electrodes, induced_faces] = local_build_electrode_patches( ...
    fmdl, centres, cfg, patch_context, patch_builder_mode)
switch patch_builder_mode
    case "disjoint"
        [electrodes, induced_faces] = electrode_faces_by_area( ...
            fmdl, centres, cfg.electrode_area, cfg.z_contact, patch_context);
    case "legacy"
        [electrodes, induced_faces] = electrode_faces_by_area_legacy( ...
            fmdl, centres, cfg.electrode_area, cfg.z_contact);
    case "legacy_fast"
        [electrodes, induced_faces] = electrode_faces_by_area_legacy_fast( ...
            fmdl, centres, cfg.electrode_area, cfg.z_contact);
    otherwise
        error('trkg4:invalidPatchBuilderMode', ...
            'Unsupported patch builder mode: %s', patch_builder_mode);
end
end

function local_write_history(matrix, sizes, filename)
T = local_history_table(matrix, sizes);
writetable(T, filename);
end

function T = local_history_table(matrix, sizes)
names = {'evaluation', 'rho_soft_ohm_m', 'rho_lungs_ohm_m', ...
    'centre_u_mm', 'centre_v_mm', 'phi_deg', 'objective', ...
    'slope_ohm_per_mm', 'h_centre_mm'};
for k = 1:numel(sizes)
    names{end + 1} = sprintf('Z_FEM_L%03d_ohm', sizes(k)); %#ok<AGROW>
end
T = array2table(matrix, 'VariableNames', names);
end
