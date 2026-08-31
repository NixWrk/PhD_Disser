function tests = test_trkg4_arm_electrode_model
%TEST_TRKG4_ARM_ELECTRODE_MODEL Pure geometry and pattern tests.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
test_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(test_dir);
addpath(fullfile(project_root, 'src'));
addpath(fullfile(project_root, 'vendor_stl_eidors'));
test_case.TestData.cfg = trkg4_config('nik');
end

function testDistanceConvention(test_case)
cfg = test_case.TestData.cfg;
spec = trkg4_arm_montage_spec(cfg, 'point_disc_5mm', 20, 80);
arms = spec.arm_geometry;

verifyEqual(test_case, spec.centres_xyz_mm(2, 1), ...
    arms.right.cut_x + 20, 'AbsTol', 1e-12);
verifyEqual(test_case, spec.centres_xyz_mm(1, 1), ...
    arms.right.cut_x + 20 + 80, 'AbsTol', 1e-12);
verifyEqual(test_case, spec.centres_xyz_mm(3, 1), ...
    arms.left.cut_x - 20, 'AbsTol', 1e-12);
verifyEqual(test_case, spec.centres_xyz_mm(4, 1), ...
    arms.left.cut_x - 20 - 80, 'AbsTol', 1e-12);
end

function testPointDiscsAreOnTop(test_case)
cfg = test_case.TestData.cfg;
spec = trkg4_arm_montage_spec(cfg, 'point_disc_5mm', 20, 80);
arms = spec.arm_geometry;

verifyEqual(test_case, spec.centres_xyz_mm(1:2, 2), ...
    repmat(arms.right.center_y, 2, 1), 'AbsTol', 1e-12);
verifyEqual(test_case, spec.centres_xyz_mm(1:2, 3), ...
    repmat(arms.right.center_z + arms.right.radius, 2, 1), ...
    'AbsTol', 1e-12);
verifyEqual(test_case, spec.centres_xyz_mm(3:4, 2), ...
    repmat(arms.left.center_y, 2, 1), 'AbsTol', 1e-12);
verifyEqual(test_case, spec.centres_xyz_mm(3:4, 3), ...
    repmat(arms.left.center_z + arms.left.radius, 2, 1), ...
    'AbsTol', 1e-12);
end

function testDefaultGridClearsCapsAndJoins(test_case)
cfg = test_case.TestData.cfg;
kinds = ["point_disc_5mm", "circumferential_ring", ...
    "cross_section_equivalent"];
for inner_mm = [20 40]
    for outer_mm = [80 100]
        for kind = kinds
            spec = trkg4_arm_montage_spec( ...
                cfg, kind, inner_mm, outer_mm);
            verifyTrue(test_case, spec.pass_clearance, ...
                sprintf('Failed %s, inner=%g, outer=%g', ...
                kind, inner_mm, outer_mm));
        end
    end
end
end

function testCrossSectionEquivalentArea(test_case)
cfg = test_case.TestData.cfg;
spec = trkg4_arm_montage_spec( ...
    cfg, 'cross_section_equivalent', 20, 80);

verifyEqual(test_case, spec.nominal_area_mm2, ...
    pi * spec.arm_radius_mm.^2, 'RelTol', 1e-12);
verifyEqual(test_case, spec.axial_width_mm, ...
    spec.arm_radius_mm / 2, 'RelTol', 1e-12);
end

function testRingMatchesArmDiameterAndHasFiveMillimetreWidth(test_case)
cfg = test_case.TestData.cfg;
spec = trkg4_arm_montage_spec(cfg, 'circumferential_ring', 20, 80);

verifyEqual(test_case, spec.contact_inner_diameter_mm, ...
    2 * spec.arm_radius_mm, 'RelTol', 1e-12);
verifyEqual(test_case, spec.axial_width_mm, ...
    repmat(5, 4, 1), 'AbsTol', 1e-12);
verifyEqual(test_case, spec.nominal_area_mm2, ...
    pi .* spec.contact_inner_diameter_mm .* spec.axial_width_mm, ...
    'RelTol', 1e-12);
end

function testReciprocalPatterns(test_case)
cfg = test_case.TestData.cfg;
[stim, meas_select] = trkg4_make_reciprocity_stimulation(cfg);

verifyEqual(test_case, stim(1).stim_pattern, ...
    cfg.current_ampere * [1; 0; 0; -1]);
verifyEqual(test_case, stim(1).meas_pattern, [0 1 -1 0]);
verifyEqual(test_case, stim(2).stim_pattern, ...
    cfg.current_ampere * [0; 1; -1; 0]);
verifyEqual(test_case, stim(2).meas_pattern, [1 0 0 -1]);
verifyEqual(test_case, meas_select, true(2, 1));
end
