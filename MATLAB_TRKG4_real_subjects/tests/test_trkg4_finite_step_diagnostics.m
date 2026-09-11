function tests = test_trkg4_finite_step_diagnostics
%TEST_TRKG4_FINITE_STEP_DIAGNOSTICS Synthetic contract tests.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
testCase.TestData.root = root;
addpath(fullfile(root, 'src'));
end

function testLinearCurveReturnsKnownSecantsAndNoCrLB(testCase)
A = [1, 2; -3, 4; 0.5, -1];
b = [7; -2; 1];
x = [2; -1];
curve = @(q) b + A * q;
z0 = curve(x);

[J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, x, z0, [-Inf; -Inf], [Inf; Inf]);

verifyEqual(testCase, J, A, 'AbsTol', 1e-12);
verifyEqual(testCase, sensitivity.difference_scheme, ["central"; "central"]);
verifyEqual(testCase, sensitivity.half_difference_scheme, ...
    ["central"; "central"]);
verifyEqual(testCase, sensitivity.step_check_status, ...
    ["distinct_effective_samples"; "distinct_effective_samples"]);
verifyEqual(testCase, sensitivity.relative_difference_at_half_step, ...
    [0; 0], 'AbsTol', 1e-12);
verifyTrue(testCase, all(isnan(sensitivity.local_crlb_std_at_sigma5ohm)));
verifyEqual(testCase, diagnostics.semantic_status, ...
    'finite_step_secants_not_pose_fisher_information');
verifyEqual(testCase, diagnostics.noise_model, 'not_established');
verifyEqual(testCase, diagnostics.condition, ...
    diagnostics.singular_values(1) / diagnostics.singular_values(end), ...
    'AbsTol', 1e-12);
json_text = jsonencode(diagnostics);
verifyTrue(testCase, ischar(json_text) || isstring(json_text));
verifyTrue(testCase, contains(string(json_text), '"step_check_details"'));
end

function testStepDependentCurveIsExposedByHalfStepCheck(testCase)
% A threshold models the discrete change of a pose/patch selection.  The
% first step crosses it and the half step does not.
x = 0;
curve = @(q) double(q >= 0.15);
z0 = curve(x);

[J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, x, z0, -1, 1);

verifyEqual(testCase, J, 2.5, 'AbsTol', 1e-12);
verifyEqual(testCase, sensitivity.difference_scheme, "central");
verifyEqual(testCase, sensitivity.half_difference_scheme, "central");
verifyEqual(testCase, sensitivity.step_check_status, "distinct_effective_samples");
verifyEqual(testCase, sensitivity.relative_difference_at_half_step, 1, ...
    'AbsTol', 1e-12);
verifyEqual(testCase, diagnostics.numerical_rank, 1);
end

function testInvalidPoseGetsLimitedStepReduction(testCase)
x = 0;
curve = @local_reject_large_pose;
z0 = 3;

[J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, x, z0, -1, 1);

% The requested 0.20 and 0.10 steps are invalid; 0.05 is accepted for
% both groups.  A central secant then has a denominator of 0.10.
verifyEqual(testCase, J, 2, 'AbsTol', 1e-12);
verifyEqual(testCase, sensitivity.status, "valid_after_step_reduction");
verifyEqual(testCase, sensitivity.finite_difference_step_effective, ...
    0.10, 'AbsTol', 1e-12);
verifyEqual(testCase, sensitivity.half_step_status, ...
    "valid_after_step_reduction");
verifyEqual(testCase, sensitivity.finite_difference_half_step_effective, ...
    0.10, 'AbsTol', 1e-12);
verifyEqual(testCase, sensitivity.step_check_status, "same_effective_samples");
verifyTrue(testCase, isnan(sensitivity.relative_difference_at_half_step));
verifyEqual(testCase, diagnostics.unique_curve_evaluations, 7);
end

function testCollapsedBoundsAreUnavailableBounds(testCase)
curve = @(q) 1;
[J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, 0, 1, 0, 0);

verifyTrue(testCase, isnan(J));
verifyEqual(testCase, sensitivity.status, "unavailable_bounds");
verifyEqual(testCase, sensitivity.invalid_error_identifier, "");
verifyEqual(testCase, sensitivity.step_check_status, "unavailable");
verifyFalse(testCase, diagnostics.all_columns_available);
end

function testRankDefectIsReportedWithoutFisherCalculation(testCase)
A = [1, 2; 2, 4; -1, -2];
curve = @(q) A * q;
x = [0; 0];
z0 = curve(x);

[J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, x, z0, [-Inf; -Inf], [Inf; Inf]);

verifyEqual(testCase, J, A, 'AbsTol', 1e-12);
verifyEqual(testCase, diagnostics.numerical_rank, 1);
verifyTrue(testCase, diagnostics.column_rank_deficient);
verifyEqual(testCase, diagnostics.rank_deficiency, 1);
verifyTrue(testCase, isinf(diagnostics.condition));
verifyTrue(testCase, all(isnan(sensitivity.local_crlb_std_at_sigma5ohm)));
end

function testEconomySvdExposesColumnRankDeficiency(testCase)
curve = @(q) q(1) + 2 * q(2);
x = [0; 0];
z0 = curve(x);

[J, ~, diagnostics] = trkg4_finite_step_diagnostics( ...
    curve, x, z0, [-Inf; -Inf], [Inf; Inf]);

verifyEqual(testCase, J, [1, 2], 'AbsTol', 1e-12);
verifyEqual(testCase, numel(diagnostics.singular_values), 1);
verifyEqual(testCase, diagnostics.numerical_rank, 1);
verifyTrue(testCase, diagnostics.column_rank_deficient);
verifyEqual(testCase, diagnostics.rank_deficiency, 1);
verifyTrue(testCase, isinf(diagnostics.condition));
end

function testKnownGeometryFailuresBecomeUnavailableColumns(testCase)
identifiers = { ...
    'trkg4:electrodeDiagnostics', ...
    'trkg4:patchOverlap', ...
    'electrode_faces_by_area:patchTooSmall', ...
    'electrode_faces_by_area:noSeed', ...
    'electrode_faces_by_area_legacy:patchTooSmall', ...
    'electrode_faces_by_area_legacy_fast:patchTooSmall'};

for k = 1:numel(identifiers)
    curve = @(q) local_throw_geometry(identifiers{k}); %#ok<NASGU>
    [J, sensitivity, diagnostics] = trkg4_finite_step_diagnostics( ...
        curve, 0, 1, -1, 1);
    verifyTrue(testCase, all(isnan(J), 'all'));
    verifyEqual(testCase, sensitivity.status, "unavailable_geometry");
    verifyEqual(testCase, sensitivity.finite_difference_step_requested, 0.20);
    verifyEqual(testCase, sensitivity.difference_scheme, "unavailable");
    verifyFalse(testCase, diagnostics.all_columns_available);
end
end

function testTechnicalFailureIsRethrown(testCase)
curve = @(q) local_throw_technical();
verifyError(testCase, @() trkg4_finite_step_diagnostics( ...
    curve, 0, 1, -1, 1), 'test:technicalFailure');
end

function z = local_reject_large_pose(q)
if abs(q) > 0.08
    error('trkg4:electrodeDiagnostics', 'Synthetic invalid pose.');
end
z = 3 + 2 * q;
end

function z = local_throw_geometry(identifier)
error(identifier, 'Synthetic geometry rejection.');
z = []; %#ok<UNRCH>
end

function z = local_throw_technical()
error('test:technicalFailure', 'Synthetic technical failure.');
z = []; %#ok<UNRCH>
end
