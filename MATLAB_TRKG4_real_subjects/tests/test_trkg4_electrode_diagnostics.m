function tests = test_trkg4_electrode_diagnostics
%TEST_TRKG4_ELECTRODE_DIAGNOSTICS Targeted surface-distance and centroid tests.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src'));
test_case.TestData.project_root = project_root;
end

function testTriangleSurfaceDistanceAndAreaWeightedCentroid(test_case)
fmdl = struct( ...
    'nodes', [ ...
        0 0 0;
        6 0 0;
        0 2 0;
        0 1 0], ...
    'boundary', [1 2 3; 1 2 4]);
fmdl.electrode = struct('nodes', [1; 2; 3; 4], 'z_contact', 0.1);
cfg = local_cfg(9);
area_centroid = [2, 5 / 9, 0];
requested = [1, 1, 3];
diagnostics = trkg4_electrode_diagnostics( ...
    fmdl, area_centroid, {'E1'}, cfg, requested);

verifyEqual(test_case, diagnostics.nearest_boundary_distance_mm, ...
    3, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.nearest_x_mm, 1, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.nearest_y_mm, 1, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.nearest_z_mm, 0, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.surface_face_index, 1);
verifyEqual(test_case, diagnostics.patch_area_mm2, 9, 'AbsTol', 1e-12);
verifyEqual(test_case, ...
    [diagnostics.patch_centroid_x_mm, diagnostics.patch_centroid_y_mm, ...
     diagnostics.patch_centroid_z_mm], ...
    area_centroid, 'AbsTol', 1e-12);
verifyEqual(test_case, diagnostics.patch_centroid_offset_mm, ...
    0, 'AbsTol', 1e-12);

node_distances = vecnorm(fmdl.nodes - requested, 2, 2);
verifyLessThan(test_case, diagnostics.nearest_boundary_distance_mm, ...
    min(node_distances));
end

function testOptionalInducedFacesMustMatchElectrodeNodes(test_case)
fmdl = local_flat_model();
cfg = local_cfg(9);
centres = [2, 5 / 9, 0];

diagnostics = trkg4_electrode_diagnostics( ...
    fmdl, centres, {'E1'}, cfg, centres, {[1; 2]});
verifyEqual(test_case, diagnostics.patch_face_count, 2);
verifyEqual(test_case, diagnostics.patch_area_mm2, 9, 'AbsTol', 1e-12);

verifyError(test_case, @() trkg4_electrode_diagnostics( ...
    fmdl, centres, {'E1'}, cfg, centres, {[1]}), ...
    'trkg4:electrodeDiagnosticsInput');
end

function testInclinedTriangleDistance(test_case)
nodes = [0 0 0; 2 0 0; 0 2 2];
boundary = [1 2 3];
fmdl = struct('nodes', nodes, 'boundary', boundary);
fmdl.electrode = struct('nodes', [1; 2; 3], 'z_contact', 0.1);
face_normal = cross(nodes(2, :) - nodes(1, :), ...
    nodes(3, :) - nodes(1, :));
face_normal = face_normal / norm(face_normal);
surface_point = 0.2 * nodes(1, :) + 0.3 * nodes(2, :) + ...
    0.5 * nodes(3, :);
requested = surface_point + 4 * face_normal;
cfg = local_cfg(0.5 * norm( ...
    cross(nodes(2, :) - nodes(1, :), nodes(3, :) - nodes(1, :))));

diagnostics = trkg4_electrode_diagnostics( ...
    fmdl, surface_point, {'inclined'}, cfg, requested);
verifyEqual(test_case, diagnostics.nearest_boundary_distance_mm, ...
    4, 'AbsTol', 1e-12);
verifyEqual(test_case, ...
    [diagnostics.nearest_x_mm, diagnostics.nearest_y_mm, ...
     diagnostics.nearest_z_mm], ...
    surface_point, 'AbsTol', 1e-12);
end

function testThresholdAndTechnicalErrorsHaveDifferentIds(test_case)
fmdl = local_flat_model();
centres = [2, 5 / 9, 0];
cfg = local_cfg(9);
cfg.max_electrode_surface_distance_mm = 0.1;

verifyError(test_case, @() trkg4_electrode_diagnostics( ...
    fmdl, centres, {'E1'}, cfg, [1, 1, 3]), ...
    'trkg4:electrodeDiagnostics');

cfg = local_cfg(9);
cfg.length_unit = 'm';
verifyError(test_case, @() trkg4_electrode_diagnostics( ...
    fmdl, centres, {'E1'}, cfg, centres), ...
    'trkg4:electrodeDiagnosticsInput');
end

function fmdl = local_flat_model
fmdl = struct( ...
    'nodes', [0 0 0; 6 0 0; 0 2 0; 0 1 0], ...
    'boundary', [1 2 3; 1 2 4]);
fmdl.electrode = struct('nodes', [1; 2; 3; 4], 'z_contact', 0.1);
end

function cfg = local_cfg(electrode_area)
cfg = struct( ...
    'length_unit', 'mm', ...
    'electrode_area', electrode_area, ...
    'max_electrode_surface_distance_mm', 5, ...
    'max_electrode_patch_centroid_offset_mm', 5, ...
    'min_electrode_patch_area_fraction', 0.95, ...
    'max_electrode_patch_area_ratio', 1.5, ...
    'fail_on_electrode_diagnostics', true, ...
    'electrode_diagnostics_verbose', false, ...
    'electrode_diagnostics_file', '');
end

