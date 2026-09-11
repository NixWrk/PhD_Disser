function tests = test_trkg4_arm_surface_centres
%TEST_TRKG4_ARM_SURFACE_CENTRES Tests for native arm montage coordinates.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
test_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(test_dir);
addpath(fullfile(project_root, 'src'));
end

function testIndependentRightAndLeftDistances(test_case)
[nodes, boundary, right_guides, left_guides] = local_two_strips();
fmdl = struct('nodes', nodes, 'boundary', boundary);

[centres, diagnostics] = trkg4_arm_surface_centres( ...
    fmdl, right_guides, left_guides, [2 3], [5 4]);

expected = [7 0 0; 2 0 0; 3 10 0; 7 10 0];
verifyEqual(test_case, centres, expected, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.centre_table.label, ...
    ["I_plus"; "V_plus"; "V_minus"; "I_minus"]);
verifyEqual(test_case, diagnostics.centre_table.side, ...
    ["right"; "right"; "left"; "left"]);
verifyEqual(test_case, diagnostics.right_realised_separation_mm, 5);
verifyEqual(test_case, diagnostics.left_realised_separation_mm, 4);
verifyEqual(test_case, diagnostics.coordinate_system, "surface_path_v1");
verifyFalse(test_case, diagnostics.symmetric_distance_request);
verifyEqual(test_case, diagnostics.placement_mode, ...
    "independent_path_distances");
end

function testScalarDistancesApplyToBothArms(test_case)
[nodes, boundary, right_guides, left_guides] = local_two_strips();
fmdl = struct('nodes', nodes, 'boundary', boundary);

[centres, diagnostics] = trkg4_arm_surface_centres( ...
    fmdl, right_guides, left_guides, 2, 5);

verifyEqual(test_case, centres, ...
    [7 0 0; 2 0 0; 2 10 0; 7 10 0], 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.inner_from_reference_mm, [2; 2]);
verifyEqual(test_case, diagnostics.outer_from_inner_mm, [5; 5]);
verifyTrue(test_case, diagnostics.symmetric_distance_request);
verifyEqual(test_case, diagnostics.placement_mode, ...
    "symmetric_equal_path_distances");
end

function testRigidTransformPreservesMontage(test_case)
[nodes, boundary, right_guides, left_guides] = local_two_strips();
fmdl = struct('nodes', nodes, 'boundary', boundary);
[base_centres, ~] = trkg4_arm_surface_centres( ...
    fmdl, right_guides, left_guides, 2, 5);

angle = 0.61;
rotation = [cos(angle) -sin(angle) 0; ...
    sin(angle) cos(angle) 0; 0 0 1];
translation = [13 -4 8];
transformed = struct();
transformed.nodes = (rotation * nodes.').'+ translation;
transformed.boundary = boundary;
right_transformed = (rotation * right_guides.').'+ translation;
left_transformed = (rotation * left_guides.').'+ translation;

[transformed_centres, ~] = trkg4_arm_surface_centres( ...
    transformed, right_transformed, left_transformed, 2, 5);
expected = (rotation * base_centres.').'+ translation;
verifyEqual(test_case, transformed_centres, expected, 'AbsTol', 1e-11);
end

function testGenericSurfaceWrapperUsesPositiveAndNegativePaths(test_case)
[nodes, boundary, positive_guides, negative_guides] = local_two_strips();
fmdl = struct('nodes', nodes, 'boundary', boundary);

[centres, diagnostics] = trkg4_surface_montage_centres( ...
    fmdl, positive_guides, negative_guides, 2, 5);

verifyEqual(test_case, centres, ...
    [7 0 0; 2 0 0; 2 10 0; 7 10 0], 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.centre_table.side, ...
    ["positive"; "positive"; "negative"; "negative"]);
verifyTrue(test_case, isfield(diagnostics, 'positive_path'));
verifyTrue(test_case, isfield(diagnostics, 'negative_path'));
verifyFalse(test_case, isfield(diagnostics, 'right_path'));
verifyEqual(test_case, diagnostics.positive_realised_separation_mm, 5);
verifyEqual(test_case, diagnostics.negative_realised_separation_mm, 5);
end

function testZeroOuterDistanceIsRejected(test_case)
[nodes, boundary, right_guides, left_guides] = local_two_strips();
fmdl = struct('nodes', nodes, 'boundary', boundary);
verifyError(test_case, @() trkg4_arm_surface_centres( ...
    fmdl, right_guides, left_guides, 2, 0), ...
    'trkg4:invalidArmSurfaceDistance');
end

function testPathMustReachOuterElectrode(test_case)
[nodes, boundary, right_guides, left_guides] = local_two_strips();
fmdl = struct('nodes', nodes, 'boundary', boundary);
verifyError(test_case, @() trkg4_arm_surface_centres( ...
    fmdl, right_guides, left_guides, 6, 6), ...
    'trkg4:distanceOutOfRange');
end

function [nodes, boundary, right_guides, left_guides] = local_two_strips()
[right_nodes, strip_faces] = local_strip(0);
[left_nodes, left_faces] = local_strip(10);
left_faces = left_faces + size(right_nodes, 1);
nodes = [right_nodes; left_nodes];
boundary = [strip_faces; left_faces];
right_guides = [0 0 0; 10 0 0];
left_guides = [0 10 0; 10 10 0];
end

function [nodes, faces] = local_strip(y_offset)
x = (0:10).';
nodes = [x, repmat(y_offset, numel(x), 1), zeros(numel(x), 1); ...
    x, repmat(y_offset + 1, numel(x), 1), zeros(numel(x), 1)];
faces = zeros(20, 3);
for k = 1:10
    lower_left = k;
    lower_right = k + 1;
    upper_left = 11 + k;
    upper_right = 11 + k + 1;
    faces(2 * k - 1, :) = [lower_left lower_right upper_left];
    faces(2 * k, :) = [lower_right upper_right upper_left];
end
end
