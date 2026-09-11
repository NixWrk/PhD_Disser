function tests = test_trkg4_nik_inhale_data
tests = functiontests(localfunctions);
end

function testAcceptedIndependentSizes(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
[sizes, z, sources] = trkg4_load_nik_inhale_data(root);

verifyEqual(testCase, sizes(:)', [50 60 70 80 90 110 120 130 140]);
verifyEqual(testCase, numel(z), 9);
verifyEqual(testCase, numel(sources), 9);
verifyTrue(testCase, any(sources == "90nik.csv"));
verifyFalse(testCase, any(sources == "100nik.csv"));
end

function testLegacyDuplicateAblation(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
[sizes, z, sources] = trkg4_load_nik_inhale_data(root, "legacy_100");

verifyEqual(testCase, sizes(:)', [50 60 70 80 100 110 120 130 140]);
verifyEqual(testCase, numel(z), 9);
verifyEqual(testCase, numel(sources), 9);
verifyFalse(testCase, any(sources == "90nik.csv"));
verifyTrue(testCase, any(sources == "100nik.csv"));
end
