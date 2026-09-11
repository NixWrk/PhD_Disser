function tests = test_trkg4_surface_path
%TEST_TRKG4_SURFACE_PATH Tests for graph-based surface path construction.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
test_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(test_dir);
addpath(fullfile(project_root, 'src'));
test_case.TestData.project_root = project_root;
end

function testFlatGridPathAndNearestGuideDiagnostics(test_case)
[vertices, faces] = local_flat_grid(5, 3);
guides = [0 1 0; 4 1 0];
distances = (0:4).';

[sample_points, diagnostics] = trkg4_surface_path( ...
    vertices, faces, guides, distances);

expected_points = [distances, ones(size(distances)), zeros(size(distances))];
verifyEqual(test_case, sample_points, expected_points, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.path_length_mm, 4, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.original_guide_points_xyz_mm, guides, ...
    'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.resolved_guide_points_xyz_mm, guides, ...
    'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.guide_offsets_mm, ...
    zeros(2, 1), 'AbsTol', 1e-12);
verifyTrue(test_case, diagnostics.path_is_continuous);
verifyEqual(test_case, diagnostics.number_of_surface_vertices, 15);
verifyEqual(test_case, diagnostics.path_nodes(1), ...
    sub2ind([3 5], 2, 1));
verifyEqual(test_case, diagnostics.path_nodes(end), ...
    sub2ind([3 5], 2, 5));
verifyFalse(test_case, isfield(diagnostics, 'surface_graph_edges'));
verifyFalse(test_case, isfield(diagnostics, 'surface_graph_edge_lengths'));
end

function testUnusedVolumeNodeCannotCaptureGuide(test_case)
[vertices, faces] = local_flat_grid(5, 3);
surface_node = sub2ind([3 5], 2, 1);
unused_node = size(vertices, 1) + 1;
vertices = [vertices; 0 1 0.001];
guides = [0 1 0.0009; 4 1 0];

verifyLessThan(test_case, ...
    norm(vertices(unused_node, :) - guides(1, :)), ...
    norm(vertices(surface_node, :) - guides(1, :)));
[sample_points, diagnostics] = trkg4_surface_path( ...
    vertices, faces, guides, [0; 4]);

verifyEqual(test_case, diagnostics.guide_vertex_indices(1), surface_node);
verifyEqual(test_case, diagnostics.resolved_guide_points_xyz_mm(1, :), ...
    vertices(surface_node, :), 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.guide_offsets_mm(1), 0.0009, ...
    'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.number_of_surface_vertices, 15);
verifyFalse(test_case, any(diagnostics.path_nodes == unused_node));
verifyEqual(test_case, sample_points, [0 1 0; 4 1 0], ...
    'AbsTol', 1e-12);
end

function testRigidRotationAndTranslationPreservePath(test_case)
[vertices, faces] = local_flat_grid(5, 3);
guides = [0 1 0; 4 1 0];
distances = [0; 1.5; 4];
[base_samples, base_diagnostics] = trkg4_surface_path( ...
    vertices, faces, guides, distances);

angle_z = 0.37;
angle_y = -0.51;
rotation_z = [cos(angle_z) -sin(angle_z) 0; ...
    sin(angle_z) cos(angle_z) 0; 0 0 1];
rotation_y = [cos(angle_y) 0 sin(angle_y); ...
    0 1 0; -sin(angle_y) 0 cos(angle_y)];
rotation = rotation_z * rotation_y;
translation = [11.2 -3.4 7.8];
transformed_vertices = (rotation * vertices.').';
transformed_vertices = transformed_vertices + translation;
transformed_guides = (rotation * guides.').';
transformed_guides = transformed_guides + translation;

[transformed_samples, transformed_diagnostics] = trkg4_surface_path( ...
    transformed_vertices, faces, transformed_guides, distances);
expected_samples = (rotation * base_samples.').';
expected_samples = expected_samples + translation;

verifyEqual(test_case, transformed_samples, expected_samples, ...
    'AbsTol', 1e-11);
verifyEqual(test_case, transformed_diagnostics.path_length_mm, ...
    base_diagnostics.path_length_mm, 'AbsTol', 1e-11);
verifyEqual(test_case, transformed_diagnostics.path_nodes, ...
    base_diagnostics.path_nodes);
end

function testPathThroughThreeGuidePoints(test_case)
[vertices, faces] = local_flat_grid(7, 3);
guides = [0 1 0; 3 1 0; 6 1 0];

[sample_points, diagnostics] = trkg4_surface_path( ...
    vertices, faces, guides, [0; 3; 6]);
guide_nodes = diagnostics.guide_vertex_indices;

verifyEqual(test_case, diagnostics.path_length_mm, 6, 'AbsTol', 1e-12);
verifyEqual(test_case, sample_points, guides, 'AbsTol', 1e-12);
for guide_index = 1:numel(guide_nodes)
    occurrences = find(diagnostics.path_nodes == guide_nodes(guide_index));
    verifyNotEmpty(test_case, occurrences);
    verifyEqual(test_case, diagnostics.path_points_xyz_mm(occurrences(1), :), ...
        guides(guide_index, :), 'AbsTol', 1e-12);
end
verifyFalse(test_case, any(diff(diagnostics.path_nodes) == 0));
end

function testSequentialDuplicateGuidesDoNotDuplicatePathNodes(test_case)
[vertices, faces] = local_flat_grid(5, 3);
guides = [0 1 0; 0 1 0; 4 1 0];
[~, diagnostics] = trkg4_surface_path(vertices, faces, guides, 2);

verifyFalse(test_case, any(diff(diagnostics.path_nodes) == 0));
verifyEqual(test_case, diagnostics.path_length_mm, 4, 'AbsTol', 1e-12);
end

function testPolylineInterpolation(test_case)
path_points = [0 0 0; 3 0 0; 3 4 0];
distances = [0; 1.5; 3; 5];
[sample_points, diagnostics] = trkg4_sample_surface_path( ...
    path_points, distances);

expected_points = [0 0 0; 1.5 0 0; 3 0 0; 3 2 0];
verifyEqual(test_case, sample_points, expected_points, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.path_length_mm, 7, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.segment_lengths_mm, [3; 4], ...
    'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.cumulative_lengths_mm, [0; 3; 7], ...
    'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.distances_mm, distances, ...
    'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.segment_indices, [1; 1; 1; 2]);
verifyEqual(test_case, diagnostics.interpolation_fractions, ...
    [0; 0.5; 1; 0.5], 'AbsTol', 1e-12);
end

function testLargeGridGraphPath(test_case)
[vertices, faces] = local_flat_grid(101, 51);
guides = [0 25 0; 100 25 0];
distances = [0; 50; 100];

[sample_points, diagnostics] = trkg4_surface_path( ...
    vertices, faces, guides, distances);

verifyEqual(test_case, diagnostics.number_of_surface_vertices, 101 * 51);
verifyEqual(test_case, diagnostics.number_of_surface_edges, 15150);
verifyEqual(test_case, diagnostics.path_length_mm, 100, 'AbsTol', 1e-12);
verifyEqual(test_case, sample_points, ...
    [0 25 0; 50 25 0; 100 25 0], 'AbsTol', 1e-12);
end

function testDistanceRangeErrors(test_case)
[vertices, faces] = local_flat_grid(5, 3);
guides = [0 1 0; 4 1 0];

verifyError(test_case, @() trkg4_surface_path( ...
    vertices, faces, guides, -0.01), 'trkg4:distanceOutOfRange');
verifyError(test_case, @() trkg4_surface_path( ...
    vertices, faces, guides, 4.01), 'trkg4:distanceOutOfRange');
verifyError(test_case, @() trkg4_sample_surface_path( ...
    [0 0 0; 1 0 0], 1.01), 'trkg4:distanceOutOfRange');
end

function testInputAndConnectivityErrors(test_case)
[vertices, faces] = local_flat_grid(5, 3);
guides = [0 1 0; 4 1 0];

verifyError(test_case, @() trkg4_surface_path( ...
    vertices, faces(:, 1:2), guides, 0), 'trkg4:invalidNumericInput');
verifyError(test_case, @() trkg4_surface_path( ...
    vertices, faces, [0 1 0], 0), 'trkg4:invalidNumericInput');

disconnected_vertices = [0 0 0; 1 0 0; 0 1 0; ...
    10 0 0; 11 0 0; 10 1 0];
disconnected_faces = [1 2 3; 4 5 6];
verifyError(test_case, @() trkg4_surface_path( ...
    disconnected_vertices, disconnected_faces, ...
    [0 0 0; 10 0 0], 0), 'trkg4:noSurfacePath');
end

function [vertices, faces] = local_flat_grid(number_of_columns, number_of_rows)
[x_grid, y_grid] = meshgrid(0:(number_of_columns - 1), ...
    0:(number_of_rows - 1));
vertices = [x_grid(:), y_grid(:), zeros(numel(x_grid), 1)];
faces = zeros(2 * (number_of_columns - 1) * (number_of_rows - 1), 3);
face_index = 0;
for column = 1:(number_of_columns - 1)
    for row = 1:(number_of_rows - 1)
        lower_left = sub2ind([number_of_rows, number_of_columns], ...
            row, column);
        lower_right = sub2ind([number_of_rows, number_of_columns], ...
            row, column + 1);
        upper_left = sub2ind([number_of_rows, number_of_columns], ...
            row + 1, column);
        upper_right = sub2ind([number_of_rows, number_of_columns], ...
            row + 1, column + 1);

        face_index = face_index + 1;
        faces(face_index, :) = [lower_left, lower_right, upper_left];
        face_index = face_index + 1;
        faces(face_index, :) = [lower_right, upper_right, upper_left];
    end
end
end
