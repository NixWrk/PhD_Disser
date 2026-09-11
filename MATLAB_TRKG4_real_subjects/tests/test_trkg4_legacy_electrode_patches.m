function tests = test_trkg4_legacy_electrode_patches
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
testCase.TestData.root = root;
addpath(fullfile(root, 'vendor_stl_eidors'));
end

function testMatchesPreC8fbca8NodeSelection(testCase)
fmdl = local_tetrahedron_model();
centres = [1/3, 1/3, -0.1; 1/3, 1/3, -0.1];
target_area = 0.45;
z_contact = 0.01;

[actual, induced_faces, diagnostics] = ...
    electrode_faces_by_area_legacy( ...
        fmdl, centres, target_area, z_contact);
expected = local_original_implementation( ...
    fmdl, centres, target_area, z_contact);

verifyEqual(testCase, actual, expected);
verifyEqual(testCase, induced_faces{1}, 1);
verifyEqual(testCase, induced_faces{2}, 1);
verifyEqual(testCase, diagnostics.selected_area, [0.5; 0.5], ...
    'AbsTol', 1e-12);
verifyEqual(testCase, diagnostics.induced_area, [0.5; 0.5], ...
    'AbsTol', 1e-12);
end

function testHistoricalOverlapIsPreserved(testCase)
fmdl = local_tetrahedron_model();
centres = repmat([1/3, 1/3, -0.1], 2, 1);
electrodes = electrode_faces_by_area_legacy( ...
    fmdl, centres, 0.45, 0.01);

verifyEqual(testCase, electrodes(1).nodes, electrodes(2).nodes);
verifyGreaterThan(testCase, ...
    numel(intersect(electrodes(1).nodes, electrodes(2).nodes)), 0);
end

function fmdl = local_tetrahedron_model()
fmdl.nodes = [0 0 0; 1 0 0; 0 1 0; 0 0 1];
fmdl.boundary = [1 2 3; 1 2 4; 1 3 4; 2 3 4];
end

function el = local_original_implementation( ...
    fmdl, electrode_centres, electrode_area, z_contact)
n_electrodes = size(electrode_centres, 1);
el = struct('nodes', cell(n_electrodes, 1), ...
    'z_contact', cell(n_electrodes, 1));
for j = 1:n_electrodes
    node_dist = sqrt(sum((fmdl.nodes - electrode_centres(j, :)).^2, 2));
    face_dist = max(node_dist(fmdl.boundary), [], 2);
    [~, face_order] = sort(face_dist);
    face_area = zeros(numel(face_order), 1);
    for k = 1:numel(face_order)
        points = fmdl.nodes(fmdl.boundary(face_order(k), :), :);
        face_area(k) = 0.5 * norm(cross( ...
            points(2, :) - points(1, :), points(3, :) - points(1, :)));
    end
    stop_index = find(cumsum(face_area) >= electrode_area, 1);
    el(j).nodes = unique(fmdl.boundary(face_order(1:stop_index), :));
    el(j).z_contact = z_contact;
end
end
