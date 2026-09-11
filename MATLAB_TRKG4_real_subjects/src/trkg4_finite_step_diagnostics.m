function [J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, x, z0, lb, ub)
%TRKG4_FINITE_STEP_DIAGNOSTICS Inspect finite secants of a TRKG4 curve.
%
%   [J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
%       curve, x, z0, lb, ub)
%
% CURVE is a function handle accepting the parameter vector and returning
% one impedance vector.  J contains the secants from the first requested
% step group [0.20 0.50 4 4 2].  A second group at half those steps is
% evaluated independently and is reported in SENSITIVITY as a step check.
%
% The function deliberately reports finite-step secants.  It does not
% interpret them as derivatives of a continuous pose, estimate a noise
% covariance, or calculate a Fisher-information CRLB.  Only failures which
% identify a known geometry rejection are converted to an unavailable
% column.  Other failures are programming or technical failures and are
% rethrown.

if nargin ~= 5
    error('trkg4:finiteStepInput', ...
        'Expected curve, x, z0, lb and ub.');
end
if ~isa(curve, 'function_handle')
    error('trkg4:finiteStepInput', 'curve must be a function handle.');
end

[x, x_was_row] = local_vector_input(x, 'x');
z0 = local_vector_input(z0, 'z0');
lb = local_vector_input(lb, 'lb');
ub = local_vector_input(ub, 'ub');
n_parameter = numel(x);

if n_parameter < 1 || n_parameter > 5
    error('trkg4:finiteStepInput', ...
        'The fixed TRKG4 step contract supports one to five parameters.');
end
if numel(lb) ~= n_parameter || numel(ub) ~= n_parameter
    error('trkg4:finiteStepInput', ...
        'x, lb and ub must have the same number of elements.');
end
if any(~isfinite(x)) || any(isnan(lb)) || any(isnan(ub)) || ...
        any(lb > ub) || any(x < lb) || any(x > ub)
    error('trkg4:finiteStepInput', ...
        'Bounds must satisfy lb <= x <= ub with no NaN values.');
end
if any(~isfinite(z0))
    error('trkg4:finiteStepInput', ...
        'z0 must be a finite numeric vector.');
end

n_observation = numel(z0);
if n_observation < 1
    error('trkg4:finiteStepInput', 'z0 must not be empty.');
end

requested_step = [0.20; 0.50; 4.0; 4.0; 2.0];
requested_step = requested_step(1:n_parameter);
half_requested_step = requested_step / 2;
max_attempts = 4;

parameter = local_parameter_names(n_parameter);
plausible_scale = local_plausible_scales(n_parameter);
J = NaN(n_observation, n_parameter);
first = repmat(local_empty_result(), n_parameter, 1);
half = repmat(local_empty_result(), n_parameter, 1);

% z0 is part of the API contract, so seed the cache with it.  This also
% prevents an exact repeated x from causing an unnecessary curve call.
cache_points = x;
cache_values = {z0};
cache_ok = true;
cache_invalid_identifier = "";


for j = 1:n_parameter
    first(j) = local_step_group(j, requested_step(j));
    half(j) = local_step_group(j, half_requested_step(j));
    if first(j).ok
        J(:, j) = first(j).derivative;
    end
end

first_status = strings(n_parameter, 1);
half_status = strings(n_parameter, 1);
step_check_status = strings(n_parameter, 1);
first_scheme = strings(n_parameter, 1);
half_scheme = strings(n_parameter, 1);
first_effective = NaN(n_parameter, 1);
half_effective = NaN(n_parameter, 1);
first_error_id = strings(n_parameter, 1);
half_error_id = strings(n_parameter, 1);
relative_half_difference = NaN(n_parameter, 1);
column_norm = NaN(n_parameter, 1);

for j = 1:n_parameter
    first_status(j) = first(j).status;
    half_status(j) = half(j).status;
    first_scheme(j) = first(j).scheme;
    half_scheme(j) = half(j).scheme;
    first_effective(j) = first(j).effective_step;
    half_effective(j) = half(j).effective_step;
    first_error_id(j) = first(j).invalid_error_identifier;
    half_error_id(j) = half(j).invalid_error_identifier;
    if first(j).ok
        column_norm(j) = norm(first(j).derivative, 2);
    end
    if ~first(j).ok || ~half(j).ok
        step_check_status(j) = "unavailable";
    elseif isequal(first(j).sample_points, half(j).sample_points)
        step_check_status(j) = "same_effective_samples";
    else
        step_check_status(j) = "distinct_effective_samples";
    end
    if first(j).ok && half(j).ok && ...
            step_check_status(j) == "distinct_effective_samples"
        denominator = norm(first(j).derivative, 2);
        numerator = norm(first(j).derivative - half(j).derivative, 2);
        if denominator == 0
            if numerator == 0
                relative_half_difference(j) = 0;
            else
                relative_half_difference(j) = Inf;
            end
        else
            relative_half_difference(j) = numerator / denominator;
        end
    end
end

% Keep the first seven variables compatible with the former sensitivity
% table.  The last CRLB column is deliberately NaN: there is no established
% noise model, and a finite-step secant matrix is not a Fisher matrix.
local_crlb_std = NaN(n_parameter, 1);
sensitivity = table( ...
    parameter, requested_step, first_effective, ...
    first_scheme, plausible_scale, column_norm, local_crlb_std, ...
    first_status, half_requested_step, half_effective, half_scheme, ...
    half_status, step_check_status, relative_half_difference, ...
    first_error_id, half_error_id, ...
    'VariableNames', { ...
    'parameter', 'finite_difference_step_requested', ...
    'finite_difference_step_effective', 'difference_scheme', ...
    'plausible_scale', 'jacobian_column_norm', ...
    'local_crlb_std_at_sigma5ohm', 'status', ...
    'finite_difference_half_step_requested', ...
    'finite_difference_half_step_effective', 'half_difference_scheme', ...
    'half_step_status', 'step_check_status', ...
    'relative_difference_at_half_step', ...
    'invalid_error_identifier', 'half_invalid_error_identifier'});

all_columns_available = all(isfinite(J), 'all');
if all_columns_available
    scaled_J = J .* reshape(plausible_scale, 1, []);
    singular_values = svd(scaled_J, 'econ');
    numerical_rank = rank(scaled_J);
    column_rank_deficient = numerical_rank < n_parameter;
    rank_deficiency = max(n_parameter - numerical_rank, 0);
    if column_rank_deficient || isempty(singular_values) || ...
            singular_values(1) == 0
        condition = Inf;
    else
        rank_tolerance = eps(max(size(scaled_J))) * singular_values(1);
        if singular_values(end) <= rank_tolerance
            condition = Inf;
        else
            condition = singular_values(1) / singular_values(end);
        end
    end
else
    singular_values = NaN(min(size(J)), 1);
    numerical_rank = NaN;
    column_rank_deficient = NaN;
    rank_deficiency = NaN;
    condition = NaN;
end

diagnostics = struct( ...
    'semantic_status', ...
    'finite_step_secants_not_pose_fisher_information', ...
    'singular_values', singular_values, ...
    'condition', condition, ...
    'numerical_rank', numerical_rank, ...
    'column_rank_deficient', column_rank_deficient, ...
    'rank_deficiency', rank_deficiency, ...
    'noise_model', 'not_established', ...
    'spectrum_uses_plausible_scale', true, ...
    'all_columns_available', all_columns_available, ...
    'step_check_status', step_check_status, ...
    'step_check_details', sensitivity, ...
    'step_check', struct( ...
        'first_requested_steps', requested_step, ...
        'half_requested_steps', half_requested_step, ...
        'max_attempts_per_group', max_attempts, ...
        'details', sensitivity), ...
    'unique_curve_evaluations', numel(cache_ok), ...
    'cached_baseline_used', true);

    function result = local_step_group(parameter_index, requested)
        result = local_empty_result();
        result.requested_step = requested;
        trial_step = requested;
        had_invalid_geometry = false;
        last_invalid_identifier = "";

        for attempt = 1:max_attempts
            x_plus = x;
            x_minus = x;
            x_plus(parameter_index) = min(x(parameter_index) + trial_step, ...
                ub(parameter_index));
            x_minus(parameter_index) = max(x(parameter_index) - trial_step, ...
                lb(parameter_index));
            plus_delta = x_plus(parameter_index) - x(parameter_index);
            minus_delta = x(parameter_index) - x_minus(parameter_index);
            plus_available = plus_delta > 0;
            minus_available = minus_delta > 0;

            plus_ok = false;
            minus_ok = false;
            plus_z = [];
            minus_z = [];
            plus_id = "";
            minus_id = "";
            if plus_available
                [plus_z, plus_ok, plus_id] = local_evaluate(x_plus);
            end
            if minus_available
                [minus_z, minus_ok, minus_id] = local_evaluate(x_minus);
            end

            if plus_ok && minus_ok
                denominator = x_plus(parameter_index) - ...
                    x_minus(parameter_index);
                if denominator > 0
                    result.derivative = (plus_z - minus_z) / denominator;
                    result.effective_step = denominator;
                    result.scheme = "central";
                    result.ok = true;
                    result.status = local_valid_status(attempt, ...
                        plus_delta < requested || minus_delta < requested);
                    result.sample_points = [x_plus, x_minus];
                    result.attempts = attempt;
                    return;
                end
            end

            % If only one side is valid, retain the largest valid secant.
            % This is explicit one-sided information, not an imputed zero.
            if plus_ok
                result.derivative = (plus_z - z0) / plus_delta;
                result.effective_step = plus_delta;
                result.scheme = "forward";
                result.ok = true;
                result.status = local_valid_status(attempt, ...
                    plus_delta < requested);
                result.sample_points = x_plus;
                result.attempts = attempt;
                result.invalid_error_identifier = minus_id;
                return;
            end
            if minus_ok
                result.derivative = (z0 - minus_z) / minus_delta;
                result.effective_step = minus_delta;
                result.scheme = "backward";
                result.ok = true;
                result.status = local_valid_status(attempt, ...
                    minus_delta < requested);
                result.sample_points = x_minus;
                result.attempts = attempt;
                result.invalid_error_identifier = plus_id;
                return;
            end

            if strlength(plus_id) > 0
                had_invalid_geometry = true;
                last_invalid_identifier = plus_id;
            end
            if strlength(minus_id) > 0
                had_invalid_geometry = true;
                last_invalid_identifier = minus_id;
            end
            if attempt < max_attempts
                trial_step = trial_step / 2;
            end
        end

        result.derivative = NaN(n_observation, 1);
        result.effective_step = NaN;
        result.scheme = "unavailable";
        result.ok = false;
        if had_invalid_geometry
            result.status = "unavailable_geometry";
            result.invalid_error_identifier = last_invalid_identifier;
        else
            result.status = "unavailable_bounds";
        end
        result.attempts = max_attempts;
    end

    function [z, ok, invalid_identifier] = local_evaluate(x_trial)
        cached = all(cache_points == x_trial, 1);
        cache_index = find(cached, 1);
        if ~isempty(cache_index)
            z = cache_values{cache_index};
            ok = cache_ok(cache_index);
            invalid_identifier = cache_invalid_identifier(cache_index);
            return;
        end

        try
            if x_was_row
                raw = curve(x_trial.');
            else
                raw = curve(x_trial);
            end
            if ~isnumeric(raw) || ~isreal(raw) || ~isvector(raw) || ...
                    any(~isfinite(raw(:))) || numel(raw) ~= n_observation
                error('trkg4:invalidCurveOutput', ...
                    'curve must return the same finite real vector length as z0.');
            end
            z = raw(:);
            ok = true;
            invalid_identifier = "";
        catch exception
            if ~local_is_allowed_geometry_failure(exception)
                rethrow(exception);
            end
            z = [];
            ok = false;
            invalid_identifier = string(exception.identifier);
        end

        cache_points(:, end + 1) = x_trial;
        cache_values{end + 1} = z;
        cache_ok(end + 1) = ok;
        cache_invalid_identifier(end + 1) = invalid_identifier;

    end

    function status = local_valid_status(attempt, clipped)
        if attempt > 1
            status = "valid_after_step_reduction";
        elseif clipped
            status = "valid_with_bound_clipping";
        else
            status = "valid";
        end
    end
end

function [value, was_row] = local_vector_input(value, name)
if ~isnumeric(value) || ~isreal(value) || ...
        (~isvector(value) && ~isscalar(value))
    error('trkg4:finiteStepInput', '%s must be a real numeric vector.', name);
end
was_row = isrow(value);
value = value(:);
end

function result = local_empty_result()
result = struct( ...
    'derivative', [], ...
    'effective_step', NaN, ...
    'scheme', "unavailable", ...
    'status', "unavailable", ...
    'ok', false, ...
    'requested_step', NaN, ...
    'attempts', 0, ...
    'sample_points', [], ...
    'invalid_error_identifier', "");
end

function names = local_parameter_names(n_parameter)
if n_parameter == 5
    names = ["rho_soft_ohm_m"; "rho_lungs_ohm_m"; ...
        "centre_u_mm"; "centre_v_mm"; "phi_deg"];
else
    names = "parameter_" + string((1:n_parameter).');
end
end

function scales = local_plausible_scales(n_parameter)
if n_parameter == 5
    scales = [1; 5; 10; 10; 10];
else
    scales = ones(n_parameter, 1);
end
end

function tf = local_is_allowed_geometry_failure(exception)
identifier = string(exception.identifier);
allowed = [ ...
    "trkg4:electrodeDiagnostics"; ...
    "trkg4:patchOverlap"; ...
    "electrode_faces_by_area:noSeed"; ...
    "electrode_faces_by_area:patchTooSmall"; ...
    "electrode_faces_by_area_legacy:patchTooSmall"; ...
    "electrode_faces_by_area_legacy_fast:patchTooSmall"; ...
    "electrode_faces_by_area_fast:patchTooSmall"];
tf = any(identifier == allowed);
end
