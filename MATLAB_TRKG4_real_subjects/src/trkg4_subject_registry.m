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

subjects = nik;
end
