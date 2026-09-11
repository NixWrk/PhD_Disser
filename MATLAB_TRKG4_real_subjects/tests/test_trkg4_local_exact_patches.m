function tests = test_trkg4_local_exact_patches
% Contract tests for the local exact electrode-patch wrapper.
tests = functiontests(localfunctions);
end

function testLargePlaneUsesLocalAndMatchesCanonical(testCase)
add_project_paths();

fmdl = make_grid_surface(0:1:24, 0:1:24);
centres = [5.3, 12.2, 0; 18.7, 12.2, 0];
electrode_area = 2.25;
z_contact = 0.35;
options = struct('initial_radius', 6, 'max_local_attempts', 3);

[expected_el, expected_faces] = canonical_output( ...
    fmdl, centres, electrode_area, z_contact);
ctx = trkg4_local_patch_context(fmdl);
ctx_before = ctx;

[actual_el, actual_faces, info] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, options);
assert_same_outputs(testCase, actual_el, actual_faces, ...
    expected_el, expected_faces);

testCase.verifyFalse(info.full_fallback);
testCase.verifyGreaterThan(info.local_attempts, 0);
testCase.verifyLessThan(info.subset_face_count, info.total_face_count);
testCase.verifyEqual(info.total_face_count, size(fmdl.boundary, 1));
testCase.verifyEqual(ctx, ctx_before);

% The context owns a mesh snapshot and the hot call does not accept fmdl.
fmdl.nodes(1, :) = fmdl.nodes(1, :) + [100, 100, 100];
fmdl.boundary = fmdl.boundary(end:-1:1, :);
[repeat_el, repeat_faces] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, options);
assert_same_outputs(testCase, repeat_el, repeat_faces, ...
    expected_el, expected_faces);
end

function testTiesPreserveOrderingOnLocalPath(testCase)
add_project_paths();
fmdl=make_grid_surface(0:24,0:24);
saved_rng=rng; restore=onCleanup(@() rng(saved_rng));
rng(204);
fmdl.boundary=fmdl.boundary(randperm(size(fmdl.boundary,1)),:);
centres=[6 12 0;18 12 0];
ctx=trkg4_local_patch_context(fmdl);
[ref,rf]=canonical_output(fmdl,centres,2.25,0.1);
[el,faces,info]=trkg4_local_exact_patches(ctx,centres,2.25,0.1, ...
    struct('initial_radius',5,'max_local_attempts',3));
testCase.verifyFalse(info.full_fallback);
assert_same_outputs(testCase,el,faces,ref,rf);
end

function testSingleLocalTriangleUsesSafeFallback(testCase)
add_project_paths();
fmdl=struct('nodes',[0 0 0;1 0 0;0 1 0;1000 0 0;1001 0 0;1000 1 0], ...
    'boundary',[1 2 3;4 5 6]);
ctx=trkg4_local_patch_context(fmdl); centres=[0.25 0.25 0];
[ref,rf]=canonical_output(fmdl,centres,0.1,0.1);
[el,faces,info]=trkg4_local_exact_patches(ctx,centres,0.1,0.1);
testCase.verifyTrue(info.full_fallback);
assert_same_outputs(testCase,el,faces,ref,rf);
end

function testTiesAndFacePermutationsMatchCanonical(testCase)
add_project_paths();

fmdl = make_tie_surface();
fmdl.boundary = fmdl.boundary([3; 1; 2], :);
centres = [6, 1, 0; 10, 1, 0];
electrode_area = 0.75;
z_contact = 0.2;
options = struct('initial_radius', 4.5, 'max_local_attempts', 3);

distances = max_vertex_distances(fmdl, centres);
testCase.verifyTrue(any(abs(distances(:, 1) - distances(:, 2)) < 1e-12));

[expected_el, expected_faces] = canonical_output( ...
    fmdl, centres, electrode_area, z_contact);
ctx = trkg4_local_patch_context(fmdl);
[actual_el, actual_faces] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, options);
assert_same_outputs(testCase, actual_el, actual_faces, ...
    expected_el, expected_faces);
end

function testIrregularMeshUsesInducedAreaStopping(testCase)
add_project_paths();

fmdl = make_grid_surface( ...
    [0, 0.8, 2.4, 5.5, 9, 14], [0, 0.6, 1.7, 4.5, 8, 12]);
centres = [5.1, 4.3, 0];
electrode_area = 2.34567;
z_contact = 0.8;
options = struct('initial_radius', 5.5, 'max_local_attempts', 3);

[expected_el, expected_faces] = canonical_output( ...
    fmdl, centres, electrode_area, z_contact);
ctx = trkg4_local_patch_context(fmdl);
[actual_el, actual_faces] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, options);
assert_same_outputs(testCase, actual_el, actual_faces, ...
    expected_el, expected_faces);

induced_area = sum(face_area_values(fmdl, actual_faces{1}));
testCase.verifyGreaterThanOrEqual(induced_area, electrode_area);
testCase.verifyGreaterThan(induced_area, electrode_area);
end

function testNearbyCompetingElectrodesRemainDisjoint(testCase)
add_project_paths();

fmdl = make_grid_surface(0:1:16, 0:1:12);
centres = [6, 6, 0; 9, 6, 0];
electrode_area = 0.4;
z_contact = 0.15;
options = struct('initial_radius', 3.5, 'max_local_attempts', 3);

[expected_el, expected_faces] = canonical_output( ...
    fmdl, centres, electrode_area, z_contact);
ctx = trkg4_local_patch_context(fmdl);
[actual_el, actual_faces] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, options);
assert_same_outputs(testCase, actual_el, actual_faces, ...
    expected_el, expected_faces);

testCase.verifyTrue(isempty(intersect(actual_el(1).nodes, actual_el(2).nodes)));
end

function testNoSeedFailureMatchesCanonical(testCase)
add_project_paths();

fmdl = make_single_triangle_surface();
centres = [0.25, 0.25, 0; 0.25, 0.25, 0];
electrode_area = 0.1;
z_contact = 0.2;

[expected_error, actual_error] = canonical_and_local_errors( ...
    fmdl, centres, electrode_area, z_contact, []);
assert_same_errors(testCase, actual_error, expected_error);
testCase.verifyEqual(actual_error.identifier, ...
    'electrode_faces_by_area:noSeed');
end

function testPatchTooSmallFailureMatchesCanonical(testCase)
add_project_paths();

fmdl = make_single_triangle_surface();
centres = [0.25, 0.25, 0];
electrode_area = 1.0;
z_contact = 0.2;
options = struct('initial_radius', 1, 'max_local_attempts', 3);

[expected_error, actual_error] = canonical_and_local_errors( ...
    fmdl, centres, electrode_area, z_contact, options);
assert_same_errors(testCase, actual_error, expected_error);
testCase.verifyEqual(actual_error.identifier, ...
    'electrode_faces_by_area:patchTooSmall');
end

function testTinyRadiusGrowthAndForcedFallback(testCase)
add_project_paths();

fmdl = make_grid_surface(0:1:18, 0:1:18);
centres = [5.2, 9.2, 0; 12.8, 9.2, 0];
electrode_area = 1.75;
z_contact = 0.5;

[expected_el, expected_faces] = canonical_output( ...
    fmdl, centres, electrode_area, z_contact);
ctx = trkg4_local_patch_context(fmdl);

growth_options = struct('initial_radius', 1e-6, ...
    'max_local_attempts', 3);
[grown_el, grown_faces, grown_info] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, growth_options);
assert_same_outputs(testCase, grown_el, grown_faces, ...
    expected_el, expected_faces);
testCase.verifyGreaterThan(grown_info.local_attempts, 0);
testCase.verifyLessThanOrEqual(grown_info.local_attempts, 3);

forced_options = struct('initial_radius', 1e-6, ...
    'max_local_attempts', 0);
[fallback_el, fallback_faces, fallback_info] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, forced_options);
assert_same_outputs(testCase, fallback_el, fallback_faces, ...
    expected_el, expected_faces);
testCase.verifyTrue(fallback_info.full_fallback);
testCase.verifyEqual(fallback_info.local_attempts, 0);
testCase.verifyEqual(fallback_info.subset_face_count, ...
    fallback_info.total_face_count);
end

function add_project_paths()
test_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(test_dir);
addpath(fullfile(project_dir, 'tools'));
addpath(fullfile(project_dir, 'vendor_stl_eidors'));
end

function [el, induced_faces] = canonical_output( ...
        fmdl, centres, electrode_area, z_contact)
[el, induced_faces] = electrode_faces_by_area( ...
    fmdl, centres, electrode_area, z_contact);
end

function [expected_error, actual_error] = canonical_and_local_errors( ...
        fmdl, centres, electrode_area, z_contact, options)
expected_error = capture_error(@() invoke_canonical( ...
    fmdl, centres, electrode_area, z_contact));
ctx = trkg4_local_patch_context(fmdl);
if isempty(options)
    actual_error = capture_error(@() invoke_local( ...
        ctx, centres, electrode_area, z_contact));
else
    actual_error = capture_error(@() invoke_local( ...
        ctx, centres, electrode_area, z_contact, options));
end
end

function invoke_canonical(fmdl, centres, electrode_area, z_contact)
electrode_faces_by_area(fmdl, centres, electrode_area, z_contact);
end

function invoke_local(ctx, centres, electrode_area, z_contact, options)
if nargin < 5
    trkg4_local_exact_patches(ctx, centres, electrode_area, z_contact);
else
    trkg4_local_exact_patches( ...
        ctx, centres, electrode_area, z_contact, options);
end
end

function caught = capture_error(callable)
caught = [];
try
    callable();
catch caught_exception
    caught = caught_exception;
end
end

function assert_same_errors(testCase, actual_error, expected_error)
testCase.verifyFalse(isempty(expected_error));
testCase.verifyFalse(isempty(actual_error));
testCase.verifyEqual(actual_error.identifier, expected_error.identifier);
testCase.verifyEqual(actual_error.message, expected_error.message);
end

function assert_same_outputs( ...
        testCase, actual_el, actual_faces, expected_el, expected_faces)
testCase.verifyEqual(size(actual_el), size(expected_el));
testCase.verifyEqual(size(actual_faces), size(expected_faces));
testCase.verifyEqual(numel(actual_el), numel(expected_el));

for k = 1:numel(expected_el)
    testCase.verifyEqual(size(actual_el(k).nodes), ...
        size(expected_el(k).nodes));
    testCase.verifyEqual(actual_el(k).nodes, expected_el(k).nodes);
    testCase.verifyEqual(actual_el(k).z_contact, ...
        expected_el(k).z_contact);
    testCase.verifyEqual(size(actual_faces{k}), size(expected_faces{k}));
    testCase.verifyEqual(actual_faces{k}, expected_faces{k});
end
end

function fmdl = make_grid_surface(x_coordinates, y_coordinates)
n_x = numel(x_coordinates);
n_y = numel(y_coordinates);
nodes = zeros(n_x * n_y, 3);

for iy = 1:n_y
    for ix = 1:n_x
        node = ix + (iy - 1) * n_x;
        nodes(node, :) = [x_coordinates(ix), y_coordinates(iy), 0];
    end
end

faces = zeros(2 * (n_x - 1) * (n_y - 1), 3);
face = 0;
for iy = 1:(n_y - 1)
    for ix = 1:(n_x - 1)
        a = ix + (iy - 1) * n_x;
        b = a + 1;
        c = a + n_x;
        d = c + 1;

        face = face + 1;
        faces(face, :) = [a, b, d];
        face = face + 1;
        faces(face, :) = [a, d, c];
    end
end

fmdl = struct('nodes', nodes, 'boundary', faces);
end

function fmdl = make_single_triangle_surface()
fmdl = struct( ...
    'nodes', [0, 0, 0; 1, 0, 0; 0, 1, 0], ...
    'boundary', [1, 2, 3]);
end

function fmdl = make_tie_surface()
fmdl = struct( ...
    'nodes', [ ...
        7, 0, 0; 9, 0, 0; 8, 2, 0; ...
        0, 0, 0; 2, 0, 0; 1, 2, 0; ...
        14, 0, 0; 16, 0, 0; 15, 2, 0], ...
    'boundary', [1, 2, 3; 4, 5, 6; 7, 8, 9]);
end

function distances = max_vertex_distances(fmdl, centres)
n_faces = size(fmdl.boundary, 1);
distances = zeros(n_faces, size(centres, 1));
for k = 1:n_faces
    vertices = fmdl.nodes(fmdl.boundary(k, :), :);
    for j = 1:size(centres, 1)
        distances(k, j) = max(vecnorm(vertices - centres(j, :), 2, 2));
    end
end
end

function values = face_area_values(fmdl, face_ids)
if isempty(face_ids)
    values = zeros(0, 1);
    return;
end

face_nodes = fmdl.boundary(face_ids, :);
p1 = fmdl.nodes(face_nodes(:, 1), :);
p2 = fmdl.nodes(face_nodes(:, 2), :);
p3 = fmdl.nodes(face_nodes(:, 3), :);
values = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);
end
