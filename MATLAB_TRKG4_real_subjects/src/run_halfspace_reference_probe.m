function report = run_halfspace_reference_probe(varargin)
%RUN_HALFSPACE_REFERENCE_PROBE
% Additional CTFEM reference probes for the half-space/BEM study.
%
% The calculation reads the already prepared CTFEM stiffness blocks from
% transverse_consistency_20260914. It does not build a mesh, search pairs,
% or alter the source/prepared files.
%
% Geometry scenarios:
%   baseline, s_plus_2mm, s_minus_2mm, t_plus_2mm, t_minus_2mm
% Each scenario moves all four saved PEM nodes and snaps each target to the
% nearest boundary node, retaining requested and actual displacement.
%
% Pulse states are mathematical linearization tests around rho=[4,16] Ohm*m.
% They use the unshifted baseline PEM RHS so that material nonlinearity is
% isolated from the node-shift probe.

cfg0 = trkg4_config('nik');
default_out = fullfile(cfg0.project_root, 'output', 'exploratory', ...
    'halfspace_bem_20260914');

p = inputParser;
addParameter(p, 'OutputDir', default_out, ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'BenchmarkRepeats', 3, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 3 && x <= 5 && mod(x,1) == 0);
parse(p, varargin{:});
opt = p.Results;

cfg = trkg4_config('nik');
root = cfg.project_root;
trkg4_prepare_runtime(cfg);
maxNumCompThreads(6);
try
    blas_threads_reported = maxNumCompThreads;
catch
    blas_threads_reported = NaN;
end

out = char(opt.OutputDir);
if ~local_is_absolute(out)
    out = fullfile(root, out);
end
if ~isfolder(out)
    [ok,msg] = mkdir(out);
    assert(ok, 'Cannot create output directory: %s', msg);
end

source_dir = fullfile(root, 'output', 'exploratory', ...
    'transverse_consistency_20260914');
prepared_path = fullfile(source_dir, 'prepared_fem.mat');
geometry_path = fullfile(source_dir, 'geometry_input.mat');
assert(isfile(prepared_path), 'Missing prepared FEM file: %s', prepared_path);
assert(isfile(geometry_path), 'Missing geometry input file: %s', geometry_path);

% The geometry input is the same export consumed by the parent study.
geometry = load(geometry_path, ...
    'nodes_mm', 'boundary_faces', 'centre', 'frame', 'sizes');
fem = load(prepared_path, 'Kall', 'Klung', 'P', 'free', 'sizes');

nodes_mm = geometry.nodes_mm;
boundary_faces = geometry.boundary_faces;
centre = geometry.centre(:).';
frame = geometry.frame;
sizes = geometry.sizes(:);
Kall = fem.Kall;
Klung = fem.Klung;
P = fem.P;
free = fem.free(:);

assert(iscell(Klung) && numel(Klung) >= 1, ...
    'prepared_fem.mat has no Klung{1}.');
Klung1 = Klung{1};
n = size(Kall, 1);
ns = numel(sizes);
assert(issparse(Kall) && issparse(Klung1) && issparse(P), ...
    'Kall, Klung{1}, and P must be sparse matrices.');
assert(size(Kall,1) == size(Kall,2) && size(Klung1,1) == n && ...
    size(Klung1,2) == n, 'Stiffness block dimensions disagree.');
assert(size(P,1) == n && size(P,2) == 2*ns, ...
    'P must contain one direct and one reciprocal RHS per size.');
assert(size(nodes_mm,1) == n && size(nodes_mm,2) == 3, ...
    'geometry_input nodes_mm does not match Kall.');
assert(size(frame,1) == 3 && size(frame,2) == 3, ...
    'geometry_input frame must be 3-by-3.');
assert(norm(frame' * frame - eye(3), 'fro') < 1e-8, ...
    'geometry_input frame is not orthonormal.');
assert(numel(unique(boundary_faces(:))) >= 4, ...
    'geometry_input has too few boundary nodes.');
boundary_nodes = unique(boundary_faces(:));
boundary_nodes = boundary_nodes(:);
assert(all(boundary_nodes >= 1 & boundary_nodes <= n), ...
    'geometry_input boundary node IDs are out of range.');
assert(isequal(sizes, geometry.sizes(:)), ...
    'Prepared sizes and geometry sizes differ.');
assert(ns == 9, 'Expected nine assembly sizes, got %d.', ns);

rho_base = [4, 16];
Ksoft = Kall - Klung1;
Kbase = Ksoft / rho_base(1) + Klung1 / rho_base(2);

% Recover the four saved PEM node points from the parent P columns.
% Electrode order is I_plus, V_plus, V_minus, I_minus.
electrode_names = ["I_plus"; "V_plus"; "V_minus"; "I_minus"];
original_node_ids = zeros(ns, 4);
for k = 1:ns
    direct = P(:, k);
    reciprocal = P(:, ns + k);
    i_plus = find(direct == 1);
    i_minus = find(direct == -1);
    v_plus = find(reciprocal == 1);
    v_minus = find(reciprocal == -1);
    assert(numel(i_plus) == 1 && numel(i_minus) == 1 && ...
        numel(v_plus) == 1 && numel(v_minus) == 1, ...
        'P does not contain exactly four point nodes for size %g.', sizes(k));
    original_node_ids(k,:) = [i_plus, v_plus, v_minus, i_minus];
end
assert(nnz(make_rhs(original_node_ids, n, ns) - P) == 0, ...
    'Recovered point nodes do not reproduce the saved P matrix.');

% Requested shifts are expressed in mm in the local frame columns s and t.
scenario_names = ["baseline"; "s_plus_2mm"; "s_minus_2mm"; ...
    "t_plus_2mm"; "t_minus_2mm"];
scenario_shift_st = [0, 0; 2, 0; -2, 0; 0, 2; 0, -2];
n_scenarios = numel(scenario_names);
shifted_node_ids = zeros(ns, 4, n_scenarios);
original_xyz = zeros(ns, 4, 3);
target_xyz = zeros(ns, 4, 3, n_scenarios);
snapped_xyz = zeros(ns, 4, 3, n_scenarios);
actual_disp_xyz = zeros(ns, 4, 3, n_scenarios);
actual_disp_std = zeros(ns, 4, 3, n_scenarios);
snap_distance = zeros(ns, 4, n_scenarios);

for k = 1:ns
    original_xyz(k,:,:) = reshape(nodes_mm(original_node_ids(k,:),:), 1, 4, 3);
    for q = 1:n_scenarios
        shift_xyz = scenario_shift_st(q,1) * frame(:,1).' + ...
            scenario_shift_st(q,2) * frame(:,2).';
        for e = 1:4
            node0 = original_node_ids(k,e);
            xyz0 = nodes_mm(node0,:);
            requested = xyz0 + shift_xyz;
            distances = vecnorm(nodes_mm(boundary_nodes,:) - requested, 2, 2);
            [distance, nearest] = min(distances);
            node1 = boundary_nodes(nearest);
            xyz1 = nodes_mm(node1,:);
            actual = xyz1 - xyz0;
            shifted_node_ids(k,e,q) = node1;
            target_xyz(k,e,:,q) = reshape(requested, 1, 1, 3);
            snapped_xyz(k,e,:,q) = reshape(xyz1, 1, 1, 3);
            actual_disp_xyz(k,e,:,q) = reshape(actual, 1, 1, 3);
            actual_disp_std(k,e,:,q) = reshape(actual * frame, 1, 1, 3);
            snap_distance(k,e,q) = distance;
        end
    end
end
assert(all(shifted_node_ids(:,:,1) == original_node_ids, 'all'), ...
    'Baseline node projection changed a saved PEM node.');

% The baseline and four projected point-RHS scenarios share the same
% material matrix, so one factorization serves all five scenarios.
base_factor = decomposition(Kbase(free, free), 'chol');
reference_count = n_scenarios * ns;
ref_model = strings(reference_count,1);
ref_scenario = strings(reference_count,1);
ref_L = zeros(reference_count,1);
ref_rho1 = zeros(reference_count,1);
ref_rho2 = zeros(reference_count,1);
ref_Z = zeros(reference_count,1);
ref_Zr = zeros(reference_count,1);
ref_J1 = zeros(reference_count,1);
ref_J2 = zeros(reference_count,1);
ref_residual = zeros(reference_count,1);
ref_recip = zeros(reference_count,1);
row_index = 0;
baseline_Z = zeros(ns,1);
baseline_J = zeros(ns,2);
baseline_residual = 0;
baseline_recip = zeros(ns,1);

for q = 1:n_scenarios
    Pq = make_rhs(shifted_node_ids(:,:,q), n, ns);
    U = zeros(n, 2*ns);
    U(free,:) = base_factor \ Pq(free,:);
    residual = norm(Kbase(free,:) * U - Pq(free,:), 'fro') / ...
        norm(Pq(free,:), 'fro');
    Wsoft = Ksoft * U(:,1:ns);
    Wlung = Klung1 * U(:,1:ns);
    for k = 1:ns
        z = real(Pq(:,ns+k)' * U(:,k));
        zr = real(Pq(:,k)' * U(:,ns+k));
        j1 = real(U(:,ns+k)' * Wsoft(:,k)) / rho_base(1)^2;
        j2 = real(U(:,ns+k)' * Wlung(:,k)) / rho_base(2)^2;
        row_index = row_index + 1;
        ref_model(row_index) = "CTFEM";
        ref_scenario(row_index) = scenario_names(q);
        ref_L(row_index) = sizes(k);
        ref_rho1(row_index) = rho_base(1);
        ref_rho2(row_index) = rho_base(2);
        ref_Z(row_index) = z;
        ref_Zr(row_index) = zr;
        ref_J1(row_index) = j1;
        ref_J2(row_index) = j2;
        ref_residual(row_index) = residual;
        ref_recip(row_index) = abs(z-zr);
        if q == 1
            baseline_Z(k) = z;
            baseline_J(k,:) = [j1, j2];
            baseline_residual = residual;
            baseline_recip(k) = abs(z-zr);
        end
    end
end
reference = table(ref_model, ref_scenario, ref_L, ref_rho1, ref_rho2, ...
    ref_Z, ref_Zr, ref_J1, ref_J2, ref_residual, ref_recip, ...
    'VariableNames', {'model','scenario','L_mm','rho1_known_ohm_m', ...
    'rho2_known_ohm_m','Z','Z_reciprocal','dZ_drho1','dZ_drho2', ...
    'relative_residual','reciprocity_abs'});
writetable(reference, fullfile(out, 'fem_reference.csv'));

% Save xyz and node IDs in separate files. A zero actual displacement is
% retained as a measured result of node quantization, never replaced by NaN.
node_count = n_scenarios * ns * 4;
xyz_model = strings(node_count,1);
xyz_scenario = strings(node_count,1);
xyz_L = zeros(node_count,1);
xyz_electrode = strings(node_count,1);
xyz_requested_s = zeros(node_count,1);
xyz_requested_t = zeros(node_count,1);
xyz_original = zeros(node_count,3);
xyz_target = zeros(node_count,3);
xyz_snapped = zeros(node_count,3);
xyz_actual = zeros(node_count,3);
xyz_actual_std = zeros(node_count,3);
xyz_snap_distance = zeros(node_count,1);
id_model = strings(node_count,1);
id_scenario = strings(node_count,1);
id_L = zeros(node_count,1);
id_electrode = strings(node_count,1);
id_original = zeros(node_count,1);
id_snapped = zeros(node_count,1);
id_same = false(node_count,1);
row_index = 0;
for q = 1:n_scenarios
    for k = 1:ns
        for e = 1:4
            row_index = row_index + 1;
            xyz_model(row_index) = "CTFEM";
            xyz_scenario(row_index) = scenario_names(q);
            xyz_L(row_index) = sizes(k);
            xyz_electrode(row_index) = electrode_names(e);
            xyz_requested_s(row_index) = scenario_shift_st(q,1);
            xyz_requested_t(row_index) = scenario_shift_st(q,2);
            xyz_original(row_index,:) = reshape(original_xyz(k,e,:),1,3);
            xyz_target(row_index,:) = reshape(target_xyz(k,e,:,q),1,3);
            xyz_snapped(row_index,:) = reshape(snapped_xyz(k,e,:,q),1,3);
            xyz_actual(row_index,:) = reshape(actual_disp_xyz(k,e,:,q),1,3);
            xyz_actual_std(row_index,:) = reshape(actual_disp_std(k,e,:,q),1,3);
            xyz_snap_distance(row_index) = snap_distance(k,e,q);
            id_model(row_index) = "CTFEM";
            id_scenario(row_index) = scenario_names(q);
            id_L(row_index) = sizes(k);
            id_electrode(row_index) = electrode_names(e);
            id_original(row_index) = original_node_ids(k,e);
            id_snapped(row_index) = shifted_node_ids(k,e,q);
            id_same(row_index) = original_node_ids(k,e) == shifted_node_ids(k,e,q);
        end
    end
end
xyz_table = table(xyz_model, xyz_scenario, xyz_L, xyz_electrode, ...
    xyz_requested_s, xyz_requested_t, ...
    xyz_original(:,1), xyz_original(:,2), xyz_original(:,3), ...
    xyz_target(:,1), xyz_target(:,2), xyz_target(:,3), ...
    xyz_snapped(:,1), xyz_snapped(:,2), xyz_snapped(:,3), ...
    xyz_actual(:,1), xyz_actual(:,2), xyz_actual(:,3), ...
    xyz_actual_std(:,1), xyz_actual_std(:,2), xyz_actual_std(:,3), ...
    xyz_snap_distance, ...
    'VariableNames', {'model','scenario','L_mm','electrode', ...
    'requested_shift_s_mm','requested_shift_t_mm', ...
    'original_x_mm','original_y_mm','original_z_mm', ...
    'target_x_mm','target_y_mm','target_z_mm', ...
    'snapped_x_mm','snapped_y_mm','snapped_z_mm', ...
    'actual_disp_x_mm','actual_disp_y_mm','actual_disp_z_mm', ...
    'actual_disp_s_mm','actual_disp_t_mm','actual_disp_d_mm', ...
    'snap_distance_mm'});
ids_table = table(id_model, id_scenario, id_L, id_electrode, ...
    id_original, id_snapped, id_same, ...
    'VariableNames', {'model','scenario','L_mm','electrode', ...
    'original_node_id','snapped_node_id','same_node'});
writetable(xyz_table, fullfile(out, 'fem_node_xyz.csv'));
writetable(ids_table, fullfile(out, 'fem_node_ids.csv'));

% Direct nonlinear material states for the pulse linearization check.
% These are assigned mathematical test amplitudes, not physiological norms.
pulse_names = ["delta_rho1_plus_0p0004"; "delta_rho2_plus_0p016"; ...
    "both_plus"; "both_minus"; "enlarged_control_10x"];
pulse_delta = [0.0004, 0; 0, 0.016; 0.0004, 0.016; ...
    -0.0004, -0.016; 0.004, 0.16];
n_pulses = size(pulse_delta,1);
pulse_count = n_pulses * ns;
pulse_model = strings(pulse_count,1);
pulse_scenario = strings(pulse_count,1);
pulse_L = zeros(pulse_count,1);
pulse_rho1_base = zeros(pulse_count,1);
pulse_rho2_base = zeros(pulse_count,1);
pulse_delta1 = zeros(pulse_count,1);
pulse_delta2 = zeros(pulse_count,1);
pulse_rho1_full = zeros(pulse_count,1);
pulse_rho2_full = zeros(pulse_count,1);
pulse_Zbase = zeros(pulse_count,1);
pulse_Zfull = zeros(pulse_count,1);
pulse_Jdelta = zeros(pulse_count,1);
pulse_response = zeros(pulse_count,1);
pulse_difference = zeros(pulse_count,1);
pulse_residual = zeros(pulse_count,1);
pulse_recip = zeros(pulse_count,1);
pulse_status = strings(pulse_count,1);
row_index = 0;
for ps = 1:n_pulses
    delta = pulse_delta(ps,:);
    rho_full = rho_base + delta;
    Kfull = Ksoft / rho_full(1) + Klung1 / rho_full(2);
    factor = decomposition(Kfull(free,free), 'chol');
    U = zeros(n,2*ns);
    U(free,:) = factor \ P(free,:);
    residual = norm(Kfull(free,:) * U - P(free,:), 'fro') / ...
        norm(P(free,:), 'fro');
    for k = 1:ns
        zfull = real(P(:,ns+k)' * U(:,k));
        zr = real(P(:,k)' * U(:,ns+k));
        zresponse = zfull - baseline_Z(k);
        jdelta = baseline_J(k,1) * delta(1) + ...
            baseline_J(k,2) * delta(2);
        row_index = row_index + 1;
        pulse_model(row_index) = "CTFEM";
        pulse_scenario(row_index) = pulse_names(ps);
        pulse_L(row_index) = sizes(k);
        pulse_rho1_base(row_index) = rho_base(1);
        pulse_rho2_base(row_index) = rho_base(2);
        pulse_delta1(row_index) = delta(1);
        pulse_delta2(row_index) = delta(2);
        pulse_rho1_full(row_index) = rho_full(1);
        pulse_rho2_full(row_index) = rho_full(2);
        pulse_Zbase(row_index) = baseline_Z(k);
        pulse_Zfull(row_index) = zfull;
        pulse_Jdelta(row_index) = jdelta;
        pulse_response(row_index) = zresponse;
        pulse_difference(row_index) = zresponse - jdelta;
        pulse_residual(row_index) = residual;
        pulse_recip(row_index) = abs(zfull-zr);
        pulse_status(row_index) = ...
            "mathematical_test_input_not_physiological_norm";
    end
end
pulse_table = table(pulse_model, pulse_scenario, pulse_L, ...
    pulse_rho1_base, pulse_rho2_base, pulse_delta1, pulse_delta2, ...
    pulse_rho1_full, pulse_rho2_full, pulse_Zbase, pulse_Zfull, ...
    pulse_Jdelta, pulse_response, pulse_difference, pulse_residual, ...
    pulse_recip, pulse_status, ...
    'VariableNames', {'model','scenario','L_mm', ...
    'rho1_base_known_ohm_m','rho2_base_known_ohm_m', ...
    'delta_rho1_ohm_m','delta_rho2_ohm_m', ...
    'rho1_full_known_ohm_m','rho2_full_known_ohm_m', ...
    'Zbase','Zfull','Jbase_delta','nonlinear_response', ...
    'nonlinear_minus_linear','relative_residual','reciprocity_abs', ...
    'amplitude_status'});
writetable(pulse_table, fullfile(out, 'fem_pulse_nonlinearity.csv'));

% Fresh direct-FEM benchmark. Each repeat reads prepared blocks from disk,
% assembles only the rho-specific K, then times factor, all 18 solves, and
% reciprocity-based derivatives separately. Prepared matrix build time is
% intentionally unknown and is never labelled as new CTFEM meshing.
n_repeats = opt.BenchmarkRepeats;
bench_model = strings(n_repeats,1);
bench_scenario = strings(n_repeats,1);
bench_repeat = zeros(n_repeats,1);
bench_threads_requested = repmat(6,n_repeats,1);
bench_threads_reported = repmat(blas_threads_reported,n_repeats,1);
bench_n_sizes = zeros(n_repeats,1);
bench_n_rhs = zeros(n_repeats,1);
bench_load = zeros(n_repeats,1);
bench_prepare = zeros(n_repeats,1);
bench_factor = zeros(n_repeats,1);
bench_solve = zeros(n_repeats,1);
bench_derivatives = zeros(n_repeats,1);
bench_fsd = zeros(n_repeats,1);
bench_total = zeros(n_repeats,1);
bench_max_residual = zeros(n_repeats,1);
bench_max_recip = zeros(n_repeats,1);
bench_result_count = n_repeats * ns;
br_model = strings(bench_result_count,1);
br_scenario = strings(bench_result_count,1);
br_repeat = zeros(bench_result_count,1);
br_L = zeros(bench_result_count,1);
br_Z = zeros(bench_result_count,1);
br_J1 = zeros(bench_result_count,1);
br_J2 = zeros(bench_result_count,1);
br_residual = zeros(bench_result_count,1);
br_recip = zeros(bench_result_count,1);
br_row = 0;

for rep = 1:n_repeats
    total_timer = tic;
    timer = tic;
    loaded = load(prepared_path, 'Kall', 'Klung', 'P', 'free', 'sizes');
    load_seconds = toc(timer);

    timer = tic;
    Kall_r = loaded.Kall;
    Klung_r = loaded.Klung{1};
    P_r = loaded.P;
    free_r = loaded.free(:);
    sizes_r = loaded.sizes(:);
    n_r = size(Kall_r,1);
    ns_r = numel(sizes_r);
    assert(n_r == n && ns_r == ns && size(P_r,2) == 2*ns, ...
        'Benchmark prepared dimensions changed.');
    Ksoft_r = Kall_r - Klung_r;
    K_r = Ksoft_r / rho_base(1) + Klung_r / rho_base(2);
    prepare_seconds = toc(timer);

    timer = tic;
    factor_r = decomposition(K_r(free_r,free_r), 'chol');
    factor_seconds = toc(timer);

    timer = tic;
    U_r = zeros(n_r,2*ns_r);
    U_r(free_r,:) = factor_r \ P_r(free_r,:);
    solve_seconds = toc(timer);

    timer = tic;
    residual_r = norm(K_r(free_r,:) * U_r - P_r(free_r,:), 'fro') / ...
        norm(P_r(free_r,:), 'fro');
    Wsoft_r = Ksoft_r * U_r(:,1:ns_r);
    Wlung_r = Klung_r * U_r(:,1:ns_r);
    max_recip_r = 0;
    for k = 1:ns_r
        z = real(P_r(:,ns_r+k)' * U_r(:,k));
        zr = real(P_r(:,k)' * U_r(:,ns_r+k));
        j1 = real(U_r(:,ns_r+k)' * Wsoft_r(:,k)) / rho_base(1)^2;
        j2 = real(U_r(:,ns_r+k)' * Wlung_r(:,k)) / rho_base(2)^2;
        max_recip_r = max(max_recip_r, abs(z-zr));
        br_row = br_row + 1;
        br_model(br_row) = "CTFEM";
        br_scenario(br_row) = "baseline_benchmark";
        br_repeat(br_row) = rep;
        br_L(br_row) = sizes_r(k);
        br_Z(br_row) = z;
        br_J1(br_row) = j1;
        br_J2(br_row) = j2;
        br_residual(br_row) = residual_r;
        br_recip(br_row) = abs(z-zr);
    end
    derivatives_seconds = toc(timer);
    fsd_seconds = factor_seconds + solve_seconds + derivatives_seconds;
    total_seconds = toc(total_timer);

    bench_model(rep) = "CTFEM";
    bench_scenario(rep) = "baseline_benchmark";
    bench_repeat(rep) = rep;
    bench_n_sizes(rep) = ns_r;
    bench_n_rhs(rep) = 2*ns_r;
    bench_load(rep) = load_seconds;
    bench_prepare(rep) = prepare_seconds;
    bench_factor(rep) = factor_seconds;
    bench_solve(rep) = solve_seconds;
    bench_derivatives(rep) = derivatives_seconds;
    bench_fsd(rep) = fsd_seconds;
    bench_total(rep) = total_seconds;
    bench_max_residual(rep) = residual_r;
    bench_max_recip(rep) = max_recip_r;
    clear loaded Kall_r Klung_r P_r free_r sizes_r Ksoft_r K_r factor_r U_r ...
        Wsoft_r Wlung_r;
end
benchmark = table(bench_model, bench_scenario, bench_repeat, ...
    bench_threads_requested, bench_threads_reported, bench_n_sizes, ...
    bench_n_rhs, bench_load, bench_prepare, bench_factor, bench_solve, ...
    bench_derivatives, bench_fsd, bench_total, bench_max_residual, ...
    bench_max_recip, ...
    repmat("unknown_prepared_matrix_build_time", n_repeats, 1), ...
    'VariableNames', {'model','scenario','repeat', ...
    'blas_threads_requested','blas_threads_reported','n_sizes', ...
    'n_forward_and_reciprocal_rhs','load_seconds','preparation_seconds', ...
    'factor_seconds','solve_seconds','derivatives_seconds', ...
    'factor_solve_derivatives_seconds','total_seconds', ...
    'max_relative_residual','max_reciprocity_abs', ...
    'prepared_matrix_build_time_status'});
writetable(benchmark, fullfile(out, 'fem_benchmark.csv'));

benchmark_results = table(br_model, br_scenario, br_repeat, br_L, br_Z, ...
    br_J1, br_J2, br_residual, br_recip, ...
    'VariableNames', {'model','scenario','repeat','L_mm','Z', ...
    'dZ_drho1','dZ_drho2','relative_residual','reciprocity_abs'});
writetable(benchmark_results, fullfile(out, 'fem_benchmark_results.csv'));

phase_names = ["load"; "preparation"; "factor"; "solve"; ...
    "derivatives"; "factor_solve_derivatives"; "total"];
phase_values = [bench_load, bench_prepare, bench_factor, bench_solve, ...
    bench_derivatives, bench_fsd, bench_total];
summary_mean = mean(phase_values,1).';
summary_std = std(phase_values,0,1).';
summary_min = min(phase_values,[],1).';
summary_max = max(phase_values,[],1).';
benchmark_summary = table(phase_names, summary_mean, summary_std, ...
    summary_min, summary_max, ...
    'VariableNames', {'phase','mean_seconds','std_seconds', ...
    'min_seconds','max_seconds'});
writetable(benchmark_summary, fullfile(out, 'fem_benchmark_summary.csv'));

% The manifest is intentionally explicit about what was and was not timed.
manifest = struct();
manifest.completed = true;
manifest.model = 'CTFEM';
manifest.rho_base_ohm_m = rho_base;
manifest.source_prepared_fem = prepared_path;
manifest.source_geometry_input = geometry_path;
manifest.source_prepared_fem_sha256 = trkg4_file_sha256(prepared_path);
manifest.source_geometry_input_sha256 = trkg4_file_sha256(geometry_path);
manifest.generator = [mfilename('fullpath') '.m'];
manifest.generator_sha256 = trkg4_file_sha256(manifest.generator);
manifest.runtime = struct('project_root', root, 'matlab_version', version, ...
    'blas_threads_requested', 6, 'blas_threads_reported', ...
    blas_threads_reported);
manifest.geometry = struct('centre_mm', centre, 'frame_columns_s_t_d', frame, ...
    'boundary_node_count', numel(boundary_nodes), ...
    'sizes_mm', sizes.');
manifest.point_projection = struct( ...
    'method', 'nearest boundary node in nodes_mm after local s/t shift', ...
    'scenarios', {cellstr(scenario_names)}, ...
    'requested_shift_s_t_mm', {scenario_shift_st}, ...
    'actual_displacement_saved', true, ...
    'node_ids_saved_separately', true, ...
    'quantized_zero_displacements_retained', true);
manifest.reference = struct('rows', height(reference), ...
    'relative_residual_max', max(reference.relative_residual), ...
    'reciprocity_abs_max', max(reference.reciprocity_abs));
manifest.pulse_linearization = struct( ...
    'rows', height(pulse_table), ...
    'scenarios', {cellstr(pulse_names)}, ...
    'delta_rho_ohm_m', {pulse_delta}, ...
    'amplitude_status', ...
    'mathematical_test_input_not_physiological_norm', ...
    'baseline_rhs', 'unshifted_saved_P');
manifest.benchmark = struct('repeats', n_repeats, ...
    'same_sizes', ns == 9, 'reciprocal_rhs_included', true, ...
    'timed_phases', {{'load','preparation','factor','solve', ...
    'derivatives','factor_solve_derivatives','total'}}, ...
    'prepared_matrix_build_time_status', 'unknown', ...
    'meshing_performed', false);
manifest.restrictions = struct('pair_search', false, ...
    'multi_ct', false, 'cem_full_16_cases', false, ...
    'prepared_source_modified', false, 'old_outputs_modified', false);
manifest.output_files = { ...
    'fem_reference.csv', 'fem_node_xyz.csv', 'fem_node_ids.csv', ...
    'fem_pulse_nonlinearity.csv', 'fem_benchmark.csv', ...
    'fem_benchmark_results.csv', 'fem_benchmark_summary.csv', ...
    'fem_probe_manifest.json'};
local_json(fullfile(out, 'fem_probe_manifest.json'), manifest);

report = manifest;
report.output_dir = out;
report.reference = reference;
report.pulse_nonlinearity = pulse_table;
report.benchmark = benchmark;
report.benchmark_summary = benchmark_summary;
fprintf('Halfspace CTFEM probe complete: %d reference rows, %d pulse rows, %d benchmark repeats.\n', ...
    height(reference), height(pulse_table), n_repeats);
end

function rhs = make_rhs(node_ids, n, ns)
rhs = sparse(n, 2*ns);
for k = 1:ns
    rhs(node_ids(k,1), k) = 1;
    rhs(node_ids(k,4), k) = -1;
    rhs(node_ids(k,2), ns+k) = 1;
    rhs(node_ids(k,3), ns+k) = -1;
end
end

function tf = local_is_absolute(path)
if ispc
    tf = (numel(path) >= 2 && path(2) == ':') || ...
        (numel(path) >= 2 && path(1) == '\' && path(2) == '\');
else
    tf = ~isempty(path) && path(1) == filesep;
end
end

function local_json(path, value)
fid = fopen(path, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'Cannot open JSON output: %s', path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(value, PrettyPrint=true));
end
