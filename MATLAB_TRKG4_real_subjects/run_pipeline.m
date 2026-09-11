function report = run_pipeline(varargin)
%RUN_PIPELINE Canonical entry point for MATLAB results consumed by notebooks.
%
% Examples:
%   report = run_pipeline('Profile', 'inverse_report');
%   report = run_pipeline('Profile', 'preflight');
%   report = run_pipeline('Profile', 'arm_electrode_report', ...
%       'ArmPrebuiltMeshFile', 'output/accepted_arm_parameter_mesh.msh', ...
%       'ArmMeshTargetSizeMm', 1);
%   report = run_pipeline('Profile', 'arm_conductivity_matrix');
%   report = run_pipeline('Profile', 'electrode_sensitivity', ...
%       'StudyFile', study_file, 'OutputDirectory', output_directory);

project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'src'));
addpath(fullfile(project_root, 'vendor_stl_eidors'));

profile_index = find(cellfun(@(x) (ischar(x) || isstring(x)) && ...
    strcmpi(string(x), "Profile"), varargin), 1);
if ~isempty(profile_index) && profile_index < numel(varargin)
    profile = string(varargin{profile_index + 1});
else
    profile = "";
end

if strcmpi(profile, "electrode_sensitivity")
    args = varargin;
    args(profile_index:profile_index + 1) = [];
    report = run_trkg4_electrode_sensitivity(args{:});
elseif strcmpi(profile, "arm_conductivity_matrix")
    args = varargin;
    args(profile_index:profile_index + 1) = [];
    report = run_trkg4_arm_conductivity_matrix(args{:});
else
    report = run_trkg4_notebook_pipeline(varargin{:});
end
end
