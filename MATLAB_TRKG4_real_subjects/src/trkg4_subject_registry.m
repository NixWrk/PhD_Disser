function subjects = trkg4_subject_registry(project_root)
%TRKG4_SUBJECT_REGISTRY Local registry of real-subject inputs.

if nargin < 1 || isempty(project_root)
    project_root = fileparts(fileparts(mfilename('fullpath')));
end

nik = struct();
nik.id = 'nik';
nik.display_name = 'Nik';
nik.stl = struct();
nik.stl.body = fullfile(project_root, 'data', 'nik', 'stl', 'body.stl');
nik.stl.lungs = fullfile(project_root, 'data', 'nik', 'stl', 'lungs.stl');
nik.stl.heart = fullfile(project_root, 'data', 'nik', 'stl', 'heart.stl');
nik.stl.bones = fullfile(project_root, 'data', 'nik', 'stl', 'bones.stl');
nik.stl.blood = fullfile(project_root, 'data', 'nik', 'stl', 'blood.stl');
nik.electrodes_csv = fullfile(project_root, 'data', 'nik', ...
    'electrodes', 'electrodes_4_xyz_mm.csv');

% The fixed-heart-geometry model does not use systolic signal fitting.
% Experimental signals can be added later only for separate validation.
nik.signal = struct();

subjects = nik;
end
