function tests = test_trkg4_legacy_fast_electrode_patches
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
testCase.TestData.root = root;
addpath(fullfile(root, 'vendor_stl_eidors'));
end

function testSkipsNodesUsedByPreviousElectrode(testCase)
fmdl.nodes = [ ...
    0, 0, 0; ...
    1, 0, 0; ...
    0, 1, 0; ...
    2, 0, 0; ...
    3, 0, 0; ...
    2, 1, 0];
fmdl.boundary = [1, 2, 3; 4, 5, 6];
centres = [0.25, 0.25, -0.1; 0.25, 0.25, -0.1];

[electrodes, induced_faces, diagnostics] = ...
    electrode_faces_by_area_legacy_fast( ...
    fmdl, centres, 0.45, 0.01);

verifyEqual(testCase, electrodes(1).nodes, [1, 2, 3]);
verifyEqual(testCase, electrodes(2).nodes, [4, 5, 6]);
verifyEmpty(testCase, intersect( ...
    electrodes(1).nodes, electrodes(2).nodes));
verifyEqual(testCase, induced_faces{1}, 1);
verifyEqual(testCase, induced_faces{2}, 2);
verifyEqual(testCase, diagnostics.selected_area, [0.5; 0.5], ...
    'AbsTol', 1e-12);
end

function testUsesSelectedRatherThanInducedAreaForStopping(testCase)
fmdl.nodes = [ ...
    0, 0, 0; ...
    1, 0, 0; ...
    0, 1, 0; ...
    1, 1, 0];
fmdl.boundary = [1, 2, 3; 2, 4, 3];
centre = [0.5, 0.5, -0.1];

[~, ~, diagnostics] = electrode_faces_by_area_legacy_fast( ...
    fmdl, centre, 0.45, 0.01);

verifyEqual(testCase, diagnostics.selected_face_count, 1);
verifyEqual(testCase, diagnostics.selected_area, 0.5, 'AbsTol', 1e-12);
verifyEqual(testCase, diagnostics.induced_area, 0.5, 'AbsTol', 1e-12);
end
