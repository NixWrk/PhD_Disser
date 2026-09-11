function results = run_trkg4_inverse_evaluation_batch( ...
    points_file, results_file, data_selection, patch_builder_mode)
%RUN_TRKG4_INVERSE_EVALUATION_BATCH Evaluate a small restartable FEM batch.
%
% A full v5 forward curve allocates large sparse system matrices.  MATLAB
% eventually fragments memory during a long inverse run even when EIDORS
% caches are cleared.  This helper evaluates a deliberately small point batch;
% the Python direct-search driver then starts a fresh MATLAB process for the
% next batch.  The mathematical model and objective are unchanged.

if nargin < 3 || strlength(string(data_selection)) == 0
    data_selection = "manifest";
end
if nargin < 4 || strlength(string(patch_builder_mode)) == 0
    patch_builder_mode = "disjoint";
end
points = readmatrix(points_file);
if size(points, 2) ~= 5 || isempty(points)
    error('trkg4:invalidEvaluationBatch', ...
        'The point file must contain a non-empty numeric N-by-5 matrix.');
end
if size(points, 1) > 6
    error('trkg4:evaluationBatchTooLarge', ...
        'At most six points are allowed per MATLAB process to bound memory.');
end

n = size(points, 1);
objective = nan(n, 1);
slope = nan(n, 1);
h_centre = nan(n, 1);
status = strings(n, 1); failure_code = strings(n, 1);
version_id = strings(n, 1); fingerprint = strings(n, 1);
for row = 1:n
    probe_tag = sprintf('restartable_probe_%s', ...
        char(lower(string(patch_builder_mode))));
    probe = run_trkg4_inverse_inhale(0, points(row, :), false, ...
        data_selection, probe_tag, patch_builder_mode, true);
    objective(row) = probe.objective;
    slope(row) = probe.fem_robust_slope_ohm_per_mm;
    h_centre(row) = probe.h_centre_mm;
    status(row) = probe.status;
    failure_code(row) = probe.failure_code;
    version_id(row) = string(probe.version);
    fingerprint(row) = string(probe.fingerprint);
end

results = table((1:n)', points(:, 1), points(:, 2), points(:, 3), ...
    points(:, 4), points(:, 5), objective, slope, h_centre, ...
    status, failure_code, version_id, fingerprint, ...
    'VariableNames', {'batch_row', 'rho_soft_ohm_m', ...
    'rho_lungs_ohm_m', 'centre_u_mm', 'centre_v_mm', 'phi_deg', ...
    'objective', 'fem_slope_ohm_per_mm', 'h_centre_mm', ...
    'status', 'failure_code', 'version', 'fingerprint'});
writetable(results, results_file);
fprintf('Saved restartable inverse batch: %s\n', results_file);
end
