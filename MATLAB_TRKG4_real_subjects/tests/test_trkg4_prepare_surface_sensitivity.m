function tests = test_trkg4_prepare_surface_sensitivity
% Small synthetic EIDORS assembly tests only; no real geometry or solves.
tests = functiontests(localfunctions);
end

function setupOnce(t)
original_folder = pwd;
folder_cleanup = onCleanup(@() cd(original_folder)); %#ok<NASGU>
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'), fullfile(root, 'vendor_stl_eidors'));
if exist('system_mat_fields', 'file') == 0
    startup_file = getenv('EIDORS_STARTUP');
    assert(isfile(startup_file), 'Set EIDORS_STARTUP to the installed startup.m.');
    run(startup_file);
end
[t.TestData.frozen, t.TestData.requested] = local_model();
end

function setup(t)
fixture = t.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
t.TestData.folder = fixture.Folder;
mkdir(fullfile(fixture.Folder, 'raw_json'));
frozen = t.TestData.frozen;
save(fullfile(fixture.Folder, 'prepared_fem.mat'), '-struct', 'frozen', '-v7');
t.TestData.manifest = local_manifest(fixture.Folder, t.TestData.requested);
t.TestData.manifest_file = fullfile(fixture.Folder, 'manifest.json');
local_json(t.TestData.manifest_file, t.TestData.manifest);
end

function testSixMontagesMatchRealEidorsAndFrozenSiCoordinates(t)
original_sha = trkg4_file_sha256(fullfile(t.TestData.folder, 'prepared_fem.mat'));
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'passed', r.error_message);
verifyEqual(t, numel(r.montages), 6);
verifyEqual(t, {r.montages.status}, repmat({'passed'}, 1, 6));
verifyEqual(t, r.prepared_sha256, original_sha);
verifyEqual(t, r.manifest_sha256, trkg4_file_sha256(t.TestData.manifest_file));
verifyEqual(t, r.mesh_convergence, 'untested');
verifyFalse(t, r.factorisation_performed);
verifyFalse(t, r.fem_sweep_performed);
verifyEqual(t, r.mesh_hash_verification, 'manifest_assertion_only_no_mesh_file_in_contract');
f = t.TestData.frozen.fmdl_m;
n = size(f.nodes, 1);
volume = sparse(n+4, n+4);
for j = 1:5, volume = volume + t.TestData.frozen.blocks{j}; end
for k = 1:6
    item = r.montages(k);
    verifyLessThanOrEqual(t, item.matrix_error, 1e-12);
    verifyTrue(t, item.geometry_qc.passed);
    file = fullfile(t.TestData.folder, item.contact_matrix);
    c = load(file);
    verifyEqual(t, sort(fieldnames(c)), sort({'C'; 'ground'; 'prepared_sha256'; ...
        'electrode_ids'; 'matrix_error'}));
    verifyEqual(t, c.prepared_sha256, original_sha);
    verifyEqual(t, c.electrode_ids, {'E1', 'E2', 'E3', 'E4'});
    verifyEqual(t, c.ground, f.gnd_node);
    verifyEqual(t, trkg4_file_sha256(file), item.contact_sha256);
    geometry = jsondecode(fileread(fullfile(t.TestData.folder, item.geometry_file)));
    verifyEqual(t, geometry.status, 'passed');
    verifyEqual(t, geometry.prepared_sha256, original_sha);
    verifyEqual(t, geometry.manifest_sha256, r.manifest_sha256);
    verifyEqual(t, geometry.index_base, 1);
    verifyEqual(t, numel(geometry.electrodes), 4);
    verifyEqual(t, geometry.requested_centres_xyz_mm, t.TestData.requested, 'AbsTol', 1e-12);
    e = struct('nodes', cell(4, 1), 'z_contact', cell(4, 1));
    all_nodes = [];
    for j = 1:4
        patch = geometry.electrodes(j).patch;
        nodes = patch.nodes_global_1based;
        verifyEqual(t, patch.coords_mm, f.nodes(nodes, :)*1000, 'AbsTol', 1e-12);
        faces = reshape(nodes(patch.faces_1based), size(patch.faces_1based));
        verifyEqual(t, faces, f.boundary(patch.boundary_face_ids_1based, :));
        verifyEmpty(t, intersect(all_nodes, nodes));
        all_nodes = [all_nodes; nodes(:)]; %#ok<AGROW>
        e(j).nodes = nodes;
        e(j).z_contact = t.TestData.manifest.contact.z_contact_ohm_m2;
        verifyEqual(t, geometry.electrodes(j).z_contact_ohm_m2, e(j).z_contact);
    end
    f.electrode = e;
    expected_C = trkg4_cem_contact_matrix(f);
    verifyEqual(t, c.C, expected_C); % Exact original SI coordinates, no round trip.
    reference = calc_system_mat(mk_image(f, 1));
    verifyLessThanOrEqual(t, norm(volume+c.C-reference.E, 'fro')/norm(reference.E, 'fro'), 1e-12);
    d = readtable(fullfile(t.TestData.folder, item.diagnostics_file));
    verifyEqual(t, height(d), 4);
    verifyTrue(t, all(d.pass_geometry));
    verifyGreaterThan(t, d.patch_face_count, zeros(4, 1));
    verifyGreaterThanOrEqual(t, d.area_fraction, repmat(.95, 4, 1));
    verifyLessThanOrEqual(t, d.area_fraction, repmat(1.5, 4, 1));
    verifyEqual(t, [d.patch_centroid_x_mm, d.patch_centroid_y_mm, ...
        d.patch_centroid_z_mm], geometry.actual_centres_xyz_mm, 'AbsTol', 1e-10);
end
saved = jsondecode(fileread(fullfile(t.TestData.folder, 'contacts', 'preparation.json')));
verifyEqual(t, saved.status, 'passed');
verifyFalse(t, isfile(fullfile(t.TestData.folder, 'contacts', 'preparation.lock')));
verifyEqual(t, trkg4_file_sha256(fullfile(t.TestData.folder, 'prepared_fem.mat')), original_sha);
end

function testSingleMontageArchiveIsSupported(t)
m = t.TestData.manifest;
m.inputs = m.inputs(1);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'passed');
verifyEqual(t, numel(r.montages), 1);
verifyEqual(t, r.counts.montages, 1);
verifyEqual(t, r.montages.id, 'tepc_2');
verifyEqual(t, numel(dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat'))), 1);
end

function testReadyArtifactsCannotBeOverwritten(t)
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'passed');
seal = fullfile(t.TestData.folder, 'contacts', 'preparation.json');
hash = trkg4_file_sha256(seal);
verifyError(t, @() trkg4_prepare_surface_sensitivity(t.TestData.manifest_file), ...
    'trkg4:outputWouldOverwrite');
verifyEqual(t, trkg4_file_sha256(seal), hash);
end

function testCoarseFacesFailAreaQcAndStillExportMeasuredGeometry(t)
% One 2-mm mesh triangle has area 2 mm2; this target is deliberately much
% smaller. A nearest-face patch must NOT be declared acceptable just because
% the canonical builder returns successfully.
m = t.TestData.manifest;
m.contact.area_mm2 = .1;
m.qc.max_centroid_offset_mm = sqrt(.1/pi);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
for k = 1:6
    verifyEqual(t, r.montages(k).status, 'failed');
    verifyEqual(t, r.montages(k).error_identifier, 'trkg4:surfacePreparationGeometry');
    d = readtable(fullfile(t.TestData.folder, r.montages(k).diagnostics_file));
    verifyFalse(t, all(d.pass_area));
    verifyGreaterThan(t, d.patch_face_count, zeros(4, 1));
    g = jsondecode(fileread(fullfile(t.TestData.folder, r.montages(k).geometry_file)));
    verifyEqual(t, numel(g.electrodes), 4);
    verifyNotEmpty(t, g.electrodes(1).patch.coords_mm);
end
end

function testFarRequestedCentreFailsWithoutLosingDiagnostics(t)
m = t.TestData.manifest;
file = fullfile(t.TestData.folder, m.inputs(1).json);
p = jsondecode(fileread(file));
roles = fieldnames(p.preliminary_electrode_centers_xyz_mm);
for k = 1:4
    p.preliminary_electrode_centers_xyz_mm.(roles{k})(3) = ...
        p.preliminary_electrode_centers_xyz_mm.(roles{k})(3) + 8;
end
local_json(file, p);
m.inputs(1).sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.montages(1).status, 'failed');
verifyEmpty(t, r.montages(1).contact_matrix);
d = readtable(fullfile(t.TestData.folder, r.montages(1).diagnostics_file));
verifyGreaterThan(t, d.nearest_boundary_distance_mm(1), 5);
verifyEqual(t, d.pass_surface_distance(1), 0);
verifyTrue(t, all(strcmp({r.montages(2:6).status}, 'passed')));
end

function testTamperedJsonBlocksReadiness(t)
file = fullfile(t.TestData.folder, t.TestData.manifest.inputs(1).json);
fid = fopen(file, 'a'); fprintf(fid, '\n'); fclose(fid);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.montages(1).error_identifier, 'trkg4:surfacePreparationHash');
verifyEmpty(t, r.montages(1).contact_matrix);
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationHash');
end

function testPreparedHashMismatchStopsBeforeGeometry(t)
m = t.TestData.manifest;
m.source.prepared_sha256 = repmat('0', 1, 64);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationHash');
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
end

function testOptionalMeshAndSurfaceHashesAreVerified(t)
m = t.TestData.manifest;
% Small provenance fixtures, not medical data or a second mesh pipeline.
local_json(fullfile(t.TestData.folder, 'mesh_source.json'), struct('synthetic', true));
local_json(fullfile(t.TestData.folder, 'synthetic_cube.stl'), struct('synthetic', true));
m.source.mesh = 'mesh_source.json';
m.source.mesh_sha256 = trkg4_file_sha256(fullfile(t.TestData.folder, m.source.mesh));
m.source.surface = 'synthetic_cube.stl';
m.source.surface_sha256 = trkg4_file_sha256(fullfile(t.TestData.folder, m.source.surface));
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'passed');
verifyEqual(t, r.mesh_hash_verification, 'file_sha256_verified');
verifyEqual(t, r.verified_source_files, {'mesh', 'surface'});
verifyNotEmpty(t, r.implementation);
end

function testTamperedOptionalMeshStopsBeforeGeometry(t)
m = t.TestData.manifest;
local_json(fullfile(t.TestData.folder, 'mesh_source.json'), struct('synthetic', true));
m.source.mesh = 'mesh_source.json';
m.source.mesh_sha256 = repmat('0', 1, 64);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationHash');
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
end

function testRoleOrderAndSurfaceMismatchCannotBeSilentlyReinterpreted(t)
m = t.TestData.manifest;
file = fullfile(t.TestData.folder, m.inputs(1).json);
p = jsondecode(fileread(file));
p.electrode_order = {'I_minus', 'V_plus', 'V_minus', 'I_plus'};
local_json(file, p);
m.inputs(1).sha256 = trkg4_file_sha256(file);
file = fullfile(t.TestData.folder, m.inputs(2).json);
p = jsondecode(fileread(file)); p.source_basename = 'different_surface.stl';
local_json(file, p);
m.inputs(2).sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.montages(1).error_identifier, 'trkg4:load_surface_landmarks:InvalidTtrkgContract');
verifyEqual(t, r.montages(2).error_identifier, 'trkg4:surfacePreparationLandmarks');
verifyTrue(t, all(strcmp({r.montages(3:6).status}, 'passed')));
end

function testNonExplicitModeRejected(t)
m = t.TestData.manifest;
file = fullfile(t.TestData.folder, m.inputs(1).json);
p = jsondecode(fileread(file)); p.schema = 'trkg4_ttrkg_arm_surface_v2';
local_json(file, p); m.inputs(1).sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.montages(1).status, 'failed');
verifyEmpty(t, r.montages(1).contact_matrix);
end

function testExistingVolumeBlocksMustMatchActualEidors(t)
p = t.TestData.frozen;
p.blocks{1} = p.blocks{1} * 1.1;
local_replace_prepared(t, p);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyTrue(t, all(strcmp({r.montages.error_identifier}, 'trkg4:surfacePreparationAssembly')));
verifyGreaterThan(t, [r.montages.matrix_error], repmat(1e-12, 1, 6));
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
end

function testVolumeBlocksCannotContainOldContactRows(t)
p = t.TestData.frozen;
p.blocks{1}(end, end) = 1;
local_replace_prepared(t, p);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationModel');
end

function testMillimetrePreparedModelIsNotRescaledByGuess(t)
p = t.TestData.frozen; p.coordinate_units = 'mm';
local_replace_prepared(t, p);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationModel');
end

function testExplicitLegacyMmGeometryUsesOriginalPatchesAndExactSiScale(t)
[m, p] = local_legacy_fixture(t);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'passed', r.error_message);
verifyEqual(t, r.prepared_units.origin, 'explicit_legacy_format_and_verified_model_identity');
verifyEqual(t, r.prepared_units.source_variable, 'fmdl_mm');
verifyEqual(t, r.prepared_units.source_coordinate_units, 'mm');
verifyEqual(t, r.prepared_units.fem_coordinate_units, 'm');
verifyEqual(t, r.prepared_units.node_scale_to_m, .001);
verifyEqual(t, r.prepared_units.producer.sha256, ...
    trkg4_file_sha256(which('run_trkg4_arm_conductivity_matrix')));
verifyTrue(t, any(strcmp({r.implementation.function_name}, 'run_trkg4_arm_conductivity_matrix')));
g = jsondecode(fileread(fullfile(t.TestData.folder, r.montages.geometry_file)));
f = p.fmdl_mm;
f.nodes = p.fmdl_mm.nodes * .001;
for j = 1:4
    patch = g.electrodes(j).patch;
    verifyEqual(t, patch.coords_mm, p.fmdl_mm.nodes(patch.nodes_global_1based, :), 'AbsTol', 1e-12);
    f.electrode(j).nodes = patch.nodes_global_1based;
    f.electrode(j).z_contact = m.contact.z_contact_ohm_m2;
end
c = load(fullfile(t.TestData.folder, r.montages.contact_matrix));
verifyEqual(t, c.C, trkg4_cem_contact_matrix(f));
verifyLessThanOrEqual(t, c.matrix_error, 1e-12);
verifyEqual(t, c.prepared_sha256, m.source.prepared_sha256);
verifyEqual(t, trkg4_file_sha256(fullfile(t.TestData.folder, 'prepared_fem.mat')), m.source.prepared_sha256);
end

function testLegacyMmCannotBeDetectedWithoutExplicitFormat(t)
[m, ~] = local_legacy_fixture(t);
m.source = rmfield(m.source, 'prepared_format');
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationModel');
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
end

function testLegacyMmCannotBeDeclaredAsMetres(t)
[m, ~] = local_legacy_fixture(t);
m.source.prepared_coordinate_units = 'm';
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationModel');
end

function testLegacyRequiresIdentityBindingToExactPreparedMat(t)
[m, ~] = local_legacy_fixture(t);
file = fullfile(t.TestData.folder, m.source.model_identity);
identity = jsondecode(fileread(file));
identity.prepared_fem_sha256 = repmat('0', 1, 64);
local_json(file, identity);
m.source.model_identity_sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationLegacy');
end

function testLegacyProducerCodeHashMustMatch(t)
[m, ~] = local_legacy_fixture(t);
file = fullfile(t.TestData.folder, m.source.model_identity);
identity = jsondecode(fileread(file));
identity.sources.sha256 = repmat('0', 1, 64);
local_json(file, identity);
m.source.model_identity_sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationHash');
end

function testLegacyBlocksInWrongUnitsFailEidorsComparison(t)
local_legacy_fixture(t, 1000);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.montages.error_identifier, 'trkg4:surfacePreparationAssembly');
verifyGreaterThan(t, r.montages.matrix_error, 1e-12);
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
end

function testSiGeometryNeedsItsOwnExplicitUnits(t)
p = rmfield(t.TestData.frozen, 'coordinate_units');
local_replace_prepared(t, p);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:surfacePreparationModel');
end

function testDuplicateBoundaryFailsExternalTopologyCheck(t)
p = t.TestData.frozen;
p.fmdl_m.boundary(end+1, :) = p.fmdl_m.boundary(1, :);
local_replace_prepared(t, p);
r = trkg4_prepare_surface_sensitivity(t.TestData.manifest_file);
verifyEqual(t, r.status, 'failed');
verifyEqual(t, r.error_identifier, 'trkg4:boundaryNormalsTopology');
verifyEmpty(t, dir(fullfile(t.TestData.folder, 'contacts', '*_contact.mat')));
end

function testInvalidManifestRejectedBeforeCreatingOutputs(t)
m = t.TestData.manifest;
m.inputs(6).id = m.inputs(1).id;
local_json(t.TestData.manifest_file, m);
verifyError(t, @() trkg4_prepare_surface_sensitivity(t.TestData.manifest_file), ...
    'trkg4:surfacePreparationManifest');
verifyFalse(t, isfolder(fullfile(t.TestData.folder, 'contacts')));
m = t.TestData.manifest; m.inputs = m.inputs([]);
local_json(t.TestData.manifest_file, m);
verifyError(t, @() trkg4_prepare_surface_sensitivity(t.TestData.manifest_file), ...
    'trkg4:surfacePreparationManifest');
m = t.TestData.manifest; m.contact.z_contact_ohm_m2 = 0;
local_json(t.TestData.manifest_file, m);
verifyError(t, @() trkg4_prepare_surface_sensitivity(t.TestData.manifest_file), ...
    'trkg4:surfacePreparationManifest');
end

function testPathsMustBeRelativeAndJsonMustStayInsideRawFolder(t)
m = t.TestData.manifest; m.source.prepared_fem = fullfile(t.TestData.folder, 'prepared_fem.mat');
local_json(t.TestData.manifest_file, m);
verifyError(t, @() trkg4_prepare_surface_sensitivity(t.TestData.manifest_file), ...
    'trkg4:surfacePreparationManifest');
m = t.TestData.manifest; m.inputs(1).json = 'raw_json/../outside.json';
local_json(t.TestData.manifest_file, m);
verifyError(t, @() trkg4_prepare_surface_sensitivity(t.TestData.manifest_file), ...
    'trkg4:surfacePreparationManifest');
verifyFalse(t, isfolder(fullfile(t.TestData.folder, 'contacts')));
end

function [m, p] = local_legacy_fixture(t, block_scale)
if nargin < 2, block_scale = 1; end
p = t.TestData.frozen;
p.fmdl_mm = p.fmdl_m;
p.fmdl_mm.nodes = p.fmdl_m.nodes * 1000;
f_si = p.fmdl_mm;
f_si.nodes = p.fmdl_mm.nodes * .001;
F = system_mat_fields(f_si);
for k = 1:5
    rows = reshape((3*(find(p.tissue_id == k)-1)+(1:3)).', [], 1);
    v = F(rows, :);
    p.blocks{k} = (v'*v) * block_scale;
end
p = rmfield(p, {'fmdl_m', 'coordinate_units'});
file = fullfile(t.TestData.folder, 'prepared_fem.mat');
save(file, '-struct', 'p', '-v7');
m = t.TestData.manifest;
m.inputs = m.inputs(1);
m.source.prepared_sha256 = trkg4_file_sha256(file);
m.source.prepared_format = 'legacy_40_05_mm_geometry_si_blocks';
identity = struct('prepared_fem_sha256', m.source.prepared_sha256, ...
    'mesh_sha256', m.source.mesh_sha256, ...
    'sources', struct('path', 'src/run_trkg4_arm_conductivity_matrix.m', ...
    'sha256', trkg4_file_sha256(which('run_trkg4_arm_conductivity_matrix'))));
m.source.model_identity = 'legacy_identity.json';
file = fullfile(t.TestData.folder, m.source.model_identity);
local_json(file, identity);
m.source.model_identity_sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
end

function local_replace_prepared(t, p)
file = fullfile(t.TestData.folder, 'prepared_fem.mat');
save(file, '-struct', 'p', '-v7');
m = t.TestData.manifest;
m.source.prepared_sha256 = trkg4_file_sha256(file);
local_json(t.TestData.manifest_file, m);
end

function m = local_manifest(folder, requested)
m = struct('schema', 'surface_sensitivity_preparation_v1', ...
    'source', struct('prepared_fem', 'prepared_fem.mat', ...
        'prepared_sha256', trkg4_file_sha256(fullfile(folder, 'prepared_fem.mat')), ...
        'mesh_sha256', repmat('a', 1, 64), 'surface_basename', 'synthetic_cube.stl'), ...
    'contact', struct('area_mm2', pi*2.5^2, ...
        'z_contact_ohm_m2', 2.5e-7*.05/(pi*.01^2/4)), ...
    'qc', struct('max_surface_distance_mm', 5, 'max_centroid_offset_mm', 2.5, ...
        'min_area_fraction', .95, 'max_area_ratio', 1.5));
for k = 1:6
    id = sprintf('tepc_%d', k+1);
    p = struct('schema', 'trkg4_ttrkg_surface_v3', 'modality', 'TTRKG', ...
        'montage', 'four_electrode_surface', 'placement_mode', 'explicit_points', ...
        'coordinate_system', 'surface_path_v1', 'units', 'mm', ...
        'source_basename', 'synthetic_cube.stl', 'symmetry', [], ...
        'electrode_order', {{'I_plus', 'V_plus', 'V_minus', 'I_minus'}}, ...
        'reference_definition', 'user_selected_centres_on_external_body_surface');
    % Object insertion order deliberately differs from the physical role order.
    p.preliminary_electrode_centers_xyz_mm = struct( ...
        'I_minus', requested(4, :), 'V_plus', requested(2, :), ...
        'I_plus', requested(1, :), 'V_minus', requested(3, :));
    rel = ['raw_json/', id, '.json'];
    local_json(fullfile(folder, rel), p);
    m.inputs(k) = struct('id', id, 'json', rel, ...
        'sha256', trkg4_file_sha256(fullfile(folder, rel)));
end
end

function [p, requested] = local_model()
[x, y, z] = ndgrid(0:2:12, 0:2:12, 0:2:12);
nodes_mm = [x(:), y(:), z(:)];
dt = delaunayTriangulation(nodes_mm);
f = struct('type', 'fwd_model', 'name', 'small surface preparation test', ...
    'nodes', nodes_mm/1000 + [0.123456789321, -.234567891239, .345678912347], ...
    'elems', dt.ConnectivityList, 'boundary', freeBoundary(dt), 'gnd_node', 2, ...
    'system_mat', @system_mat_1st_order, 'solve', @fwd_solve_1st_order, ...
    'jacobian', @jacobian_adjoint, 'normalize_measurements', 0);
requested = [3 3 12;9 3 12;3 9 12;9 9 12] + ...
    [0.123456789321, -.234567891239, .345678912347]*1000;
mm = f; mm.nodes = f.nodes*1000;
f.electrode = electrode_faces_by_area(mm, requested, pi*2.5^2, 1e-4);
fields = system_mat_fields(f);
ne = size(f.elems, 1);
tissue_id = mod((1:ne)'-1, 5)+1;
blocks = cell(5, 1);
for k = 1:5
    rows = reshape((3*(find(tissue_id == k)-1)+(1:3)).', [], 1);
    v = fields(rows, :);
    blocks{k} = v'*v;
end
p = struct('fmdl_m', f, 'blocks', {blocks}, 'tissue_id', tissue_id, ...
    'tissue_names', {{'one', 'two', 'three', 'four', 'five'}}, 'coordinate_units', 'm');
end

function local_json(file, value)
fid = fopen(file, 'w', 'n', 'UTF-8');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
