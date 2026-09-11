function tests = test_trkg4_boundary_normals
%TEST_TRKG4_BOUNDARY_NORMALS Targeted mathematical tests for boundary normals.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src'));
test_case.TestData.project_root = project_root;
end

function testBoundaryRowOrderAndVertexFlips(test_case)
nodes = [ ...
    0 0 0;
    1 0 0;
    0 1 0;
    0 0 1;
    3 0 0;
    4 0 0;
    3 1 0;
    3 0 1];
elems = [1 2 3 4; 5 6 7 8];
canonical_faces = [ ...
    1 2 3;
    1 2 4;
    1 3 4;
    2 3 4;
    5 6 7;
    5 6 8;
    5 7 8;
    6 7 8];
canonical_owner = [1; 1; 1; 1; 2; 2; 2; 2];

rng(17);
permutation = randperm(size(canonical_faces, 1));
boundary = canonical_faces(permutation, :);
owner = canonical_owner(permutation);
flip = rand(size(boundary, 1), 1) > 0.5;
boundary(flip, [2 3]) = boundary(flip, [3 2]);

fmdl = struct('nodes', nodes, 'elems', elems, 'boundary', boundary);
[node_normals, face_normals] = trkg4_boundary_normals(fmdl);

verifySize(test_case, face_normals, [8 3]);
verifySize(test_case, node_normals, [8 3]);
for face_index = 1:size(boundary, 1)
    face_nodes = boundary(face_index, :);
    owner_nodes = elems(owner(face_index), :);
    opposite_node = setdiff(owner_nodes, face_nodes);
    expected = cross( ...
        nodes(face_nodes(2), :) - nodes(face_nodes(1), :), ...
        nodes(face_nodes(3), :) - nodes(face_nodes(1), :));
    if dot(expected, nodes(opposite_node, :) - nodes(face_nodes(1), :)) > 0
        expected = -expected;
    end
    expected = expected / norm(expected);
    verifyEqual(test_case, face_normals(face_index, :), expected, ...
        'AbsTol', 1e-12);
end
verifyEqual(test_case, vecnorm(face_normals, 2, 2), ...
    ones(8, 1), 'AbsTol', 1e-12);
verifyEqual(test_case, vecnorm(node_normals, 2, 2), ...
    ones(8, 1), 'AbsTol', 1e-12);
end

function testRigidTranslationPreservesNormals(test_case)
fmdl = local_tetrahedron();
[normal_before, face_before] = trkg4_boundary_normals(fmdl);
translation = [1000, -700, 250];
translated = fmdl;
translated.nodes = fmdl.nodes + translation;
[normal_after, face_after] = trkg4_boundary_normals(translated);

verifyEqual(test_case, face_after, face_before, 'AbsTol', 1e-12);
verifyEqual(test_case, normal_after, normal_before, 'AbsTol', 1e-12);
end

function testThinTranslatedTetraUsesLocalSign(test_case)
% A thin translated tetra is accepted because its local determinant is
% decisively above local round-off, despite its small height.
fmdl = struct( ...
    'nodes', [ ...
        100 -40 10;
        101 -40 10;
        100 -39 10;
        100 -40 10 + 1e-8], ...
    'elems', [1 2 3 4], ...
    'boundary', [ ...
        1 2 3;
        1 2 4;
        1 3 4;
        2 3 4]);
[~, face_normals] = trkg4_boundary_normals(fmdl);

verifyEqual(test_case, face_normals(1, :), [0 0 -1], ...
    'AbsTol', 1e-12);
end

function testMissingOwnerIsReported(test_case)
fmdl = local_tetrahedron();
fmdl.boundary(end, :) = [1 2 5];
fmdl.nodes(5, :) = [2 2 2];

verifyError(test_case, @() trkg4_boundary_normals(fmdl), ...
    'trkg4:boundaryNormalsTopology');
end

function fmdl = local_tetrahedron
fmdl = struct( ...
    'nodes', [0 0 0; 1 0 0; 0 1 0; 0 0 1], ...
    'elems', [1 2 3 4], ...
    'boundary', [1 2 3; 1 2 4; 1 3 4; 2 3 4]);
end
