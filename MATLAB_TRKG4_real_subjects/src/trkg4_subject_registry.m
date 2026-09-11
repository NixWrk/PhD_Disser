function subjects = trkg4_subject_registry(project_root)
%TRKG4_SUBJECT_REGISTRY Local registry of real-subject inputs.

if nargin < 1 || isempty(project_root)
    project_root = fileparts(fileparts(mfilename('fullpath')));
end

nik = struct();
nik.id = 'nik';
nik.display_name = 'Nik';
nik.stl = struct();
% Keep the original CT-derived surfaces for traceability, but solve on the
% cavity-filled solids. The bulk lung resistivity represents tissue + air.
nik.stl.body_raw = fullfile(project_root, 'data', 'nik', 'stl', 'body.stl');
nik.stl.lungs_legacy = fullfile(project_root, 'data', 'nik', 'stl', 'lungs.stl');
nik.stl.lungs_ct_full_clean = fullfile(project_root, 'data', 'nik', 'stl', ...
    'lungs_ct_full_clean.stl');
nik.stl.lungs_candidate_v3 = fullfile(project_root, 'data', 'nik', 'stl', ...
    'lungs_solid_v3_volume_fill.stl');
nik.stl.body = fullfile(project_root, 'data', 'nik', 'stl', ...
    'body_solid_v3_1mm_r10_volume_fill.stl');
% V3 approved for the full forward calculations.
nik.stl.lungs = nik.stl.lungs_candidate_v3;
nik.stl.heart = fullfile(project_root, 'data', 'nik', 'stl', 'heart.stl');
nik.stl.bones = fullfile(project_root, 'data', 'nik', 'stl', 'bones.stl');
% Not segmented for Nik yet. The tissue stays declared and enabled so that it
% is picked up automatically once the mask exists; until then the run reports
% it as absent instead of pretending blood was assigned.
nik.stl.blood = fullfile(project_root, 'data', 'nik', 'stl', 'blood.stl');

% Default montage. There is no generic electrodes_4_xyz_mm.csv on disk: every
% montage is a named file, and the arms system is the one the shared mesh and
% the published forward result were built on. Pointing the default at a file
% that does not exist made run_trkg4_forward_subject('nik') fail immediately.
% Other montages are selected through cfg.electrode_centres_file, as
% run_trkg4_electrode_systems does.
nik.electrodes_csv = fullfile(project_root, 'data', 'nik', ...
    'electrodes', 'electrodes_4_arms_xyz_mm.csv');

% The fixed-heart-geometry model does not use systolic signal fitting.
% Experimental signals can be added later only for separate validation.
nik.signal = struct();
nik.prebuilt_mesh_file = fullfile(project_root, 'output', ...
    'nik_body_solid_v3_1mm_local2mm_alg1.msh');
nik.mesh_cache_file = fullfile(project_root, 'output', ...
    'nik_body_mesh_v5_1mm_local2mm_cache.mat');
nik.mesh_target_size_mm = 2;
nik.rho_cloud = struct( ...
    'soft_ohm_m', 4.728174786010649, ...
    'lungs_ohm_m', 17.4067, ...
    'lungs_inhale_ohm_m', 17.73503700425868, ...
    'source', ['Nik solution clouds / historical MATLAB-Python ', ...
        'model calculations']);
nik.heart_mask_representation = 'whole_heart';
nik.required_tissues = {'lungs', 'heart', 'bones'};

georg = struct();
georg.id = 'georg';
georg.display_name = 'Georg';
georg.stl = struct();
georg.stl.body_source_ct = fullfile(project_root, 'data', 'georg', ...
    'stl', 'body_source_ct.stl');
% The accepted calculation must use the exact surface from which its
% subject-specific Gmsh mesh was built, never the Nik body or Nik mesh.
georg.stl.body_raw = fullfile(project_root, 'data', 'georg', ...
    'stl', 'body_with_arm_extensions.stl');
georg.stl.body = fullfile(project_root, 'data', 'georg', ...
    'stl', 'body_arm_parameter_surface_1mm.stl');
georg.stl.lungs = fullfile(project_root, 'data', 'georg', 'stl', ...
    'lungs.stl');
georg.stl.heart = fullfile(project_root, 'data', 'georg', 'stl', ...
    'heart.stl');
georg.stl.bones = fullfile(project_root, 'data', 'georg', 'stl', ...
    'bones.stl');
% No separate blood mask was supplied for Georg.
georg.stl.blood = fullfile(project_root, 'data', 'georg', 'stl', ...
    'blood.stl');
georg.electrodes_csv = fullfile(project_root, 'data', 'georg', ...
    'electrodes', 'electrodes_4_arms_xyz_mm.csv');
georg.signal = struct();
georg.prebuilt_mesh_file = fullfile(project_root, 'output', ...
    'georg_body_arm_parameter_mesh_1mm_candidate.msh');
georg.mesh_cache_file = fullfile(project_root, 'output', ...
    'georg_body_arm_parameter_mesh_1mm_cache.mat');
georg.mesh_target_size_mm = 1;
% These are deliberately transferred model parameters. They make the run a
% geometry/solver portability test, not an estimate of Georg tissue values.
georg.rho_cloud = struct( ...
    'soft_ohm_m', 4.728174786010649, ...
    'lungs_ohm_m', 17.4067, ...
    'lungs_inhale_ohm_m', 17.73503700425868, ...
    'source', ['Transferred from Nik for the Georg geometry portability ', ...
        'test; not subject-specific tissue estimates']);
georg.heart_mask_representation = 'whole_heart';
georg.required_tissues = {'lungs', 'heart', 'bones'};

yaroslav = struct();
yaroslav.id = 'yaroslav';
yaroslav.display_name = 'Yaroslav';
yaroslav.stl = struct();
yaroslav.stl.body_source_ct = fullfile(project_root, 'data', 'yaroslav', ...
    'stl', 'body_source_ct.stl');
yaroslav.stl.body_raw = fullfile(project_root, 'data', 'yaroslav', ...
    'stl', 'body_with_arm_extensions.stl');
yaroslav.stl.body = fullfile(project_root, 'data', 'yaroslav', ...
    'stl', 'body_arm_parameter_surface_2mm.stl');
yaroslav.stl.lungs = fullfile(project_root, 'data', 'yaroslav', 'stl', ...
    'lungs.stl');
% No manually accepted myocardium/whole-heart mask is mixed into this test.
yaroslav.stl.heart = fullfile(project_root, 'data', 'yaroslav', 'stl', ...
    'heart.stl');
yaroslav.stl.bones = fullfile(project_root, 'data', 'yaroslav', 'stl', ...
    'bones.stl');
yaroslav.stl.blood = fullfile(project_root, 'data', 'yaroslav', 'stl', ...
    'blood.stl');
yaroslav.electrodes_csv = fullfile(project_root, 'data', 'yaroslav', ...
    'electrodes', 'electrodes_4_arms_xyz_mm.csv');
yaroslav.signal = struct();
yaroslav.prebuilt_mesh_file = fullfile(project_root, 'output', ...
    'yaroslav_body_arm_parameter_mesh_electrode1mm_surface2mm_clean.msh');
yaroslav.mesh_cache_file = fullfile(project_root, 'output', ...
    'yaroslav_body_arm_parameter_mesh_electrode1mm_surface2mm_clean_cache.mat');
% The artificial-arm surface is resolved at 2 mm. Four local volume slabs
% around the electrode planes use a 1 mm target; the remaining artificial-arm
% volume is allowed to grow to 5 mm.
yaroslav.mesh_target_size_mm = 2;
yaroslav.rho_cloud = struct( ...
    'soft_ohm_m', 4.728174786010649, ...
    'lungs_ohm_m', 17.4067, ...
    'lungs_inhale_ohm_m', 17.73503700425868, ...
    'source', ['Transferred from Nik for the Yaroslav geometry portability ', ...
        'test; not subject-specific tissue estimates']);
yaroslav.heart_mask_representation = 'blood_pool_only_no_myocardium';
yaroslav.required_tissues = {'lungs', 'bones', 'blood'};

subjects = [nik, georg, yaroslav];
end
