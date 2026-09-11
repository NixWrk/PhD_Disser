function tests = test_trkg4_green_cache_contract
%TEST_TRKG4_GREEN_CACHE_CONTRACT Behavioural tests for Green fingerprints.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(project_root, 'src'));
test_case.TestData.project_root = project_root;
end

function testSamePayloadHasSameFingerprint(test_case)
fmdl = local_fixture();
sigma = [0.2; 0.4];
candidate_nodes = [2; 5];
settings = local_settings();

first = trkg4_green_fingerprint(fmdl, sigma, candidate_nodes, settings);
second = trkg4_green_fingerprint(fmdl, sigma, candidate_nodes, settings);

verifyClass(test_case, first, 'char');
verifyEqual(test_case, first, second);
verifyLength(test_case, first, 64);
end

function testPhysicalAndOperatorChangesInvalidateFingerprint(test_case)
fmdl = local_fixture();
sigma = [0.2; 0.4];
candidate_nodes = [2; 5];
settings = local_settings();
base = trkg4_green_fingerprint(fmdl, sigma, candidate_nodes, settings);

% Every mutation below preserves the array sizes/counts of the fixture.
changed = fmdl;
changed.nodes(1, 1) = changed.nodes(1, 1) + 0.125;
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);

changed = fmdl;
changed.elems(1, [3 4]) = changed.elems(1, [4 3]);
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);

changed = fmdl;
changed.boundary(1, 3) = 4;
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);

changed_sigma = sigma;
changed_sigma(2) = changed_sigma(2) + 0.05;
local_verify_changed(test_case, base, fmdl, changed_sigma, candidate_nodes, settings);

changed = fmdl;
changed.gnd_node = 2;
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);

changed_candidates = [3; 5];
local_verify_changed(test_case, base, fmdl, sigma, changed_candidates, settings);

changed_settings = settings;
changed_settings.block_size = settings.block_size + 1;
local_verify_changed(test_case, base, fmdl, sigma, candidate_nodes, changed_settings);

% These fields are deliberately outside the explicit arrays. They affect
% calc_system_mat and therefore must remain in the residual operator context.
changed = fmdl;
changed.system_mat = 'changed-system-mat';
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);

changed = fmdl;
changed.approx_type = 'changed-approximation';
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);

changed = fmdl;
changed.coarse2fine(1, 1) = changed.coarse2fine(1, 1) + 0.25;
local_verify_changed(test_case, base, changed, sigma, candidate_nodes, settings);
end

function testLegacyRecordIsStructurallyInvalid(test_case)
% A legacy record has the matrix and node list but no fingerprint metadata.
% The production cache loaders have an explicit required-field branch; this
% small fixture documents the condition without assembling a FEM operator.
legacy = struct('green', zeros(2), 'candidate_nodes', [2; 5]);
required = {'green', 'candidate_nodes', 'cache_fingerprint', 'cache_schema'};
valid = all(isfield(legacy, required));
verifyFalse(test_case, valid);
end

function local_verify_changed(test_case, base, fmdl, sigma, candidate_nodes, settings)
changed = trkg4_green_fingerprint(fmdl, sigma, candidate_nodes, settings);
verifyFalse(test_case, strcmp(base, changed));
end

function fmdl = local_fixture
fmdl = struct( ...
    'nodes', [ ...
        0 0 0;
        1 0 0;
        0 1 0;
        0 0 1;
        1 1 1], ...
    'elems', [1 2 3 4; 2 3 4 5], ...
    'boundary', [1 2 3; 1 2 4; 1 3 4; 2 3 4], ...
    'gnd_node', 1, ...
    'system_mat', 'fixture-system-mat', ...
    'approx_type', 'fixture-approximation', ...
    'coarse2fine', speye(5), ...
    'electrode', struct([]), ...
    'stimulation', struct([]), ...
    'meas_select', []);
end

function settings = local_settings
settings = struct( ...
    'mode', 'behavioral-test', ...
    'decomposition', 'chol', ...
    'block_size', 64, ...
    'green_storage_class', 'single', ...
    'contact_impedance_ohm_m2', 1e-4);
end