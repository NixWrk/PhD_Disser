function report = run_pipeline(varargin)
%RUN_PIPELINE Canonical entry point for MATLAB results consumed by notebooks.
%
% Examples:
%   report = run_pipeline('Profile', 'inverse_report');
%   report = run_pipeline('Profile', 'preflight');
%   report = run_pipeline('Profile', 'arm_electrode_report', ...
%       'ArmPrebuiltMeshFile', 'output/accepted_arm_parameter_mesh.msh', ...
%       'ArmMeshTargetSizeMm', 1);

project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'src'));
addpath(fullfile(project_root, 'vendor_stl_eidors'));

report = run_trkg4_notebook_pipeline(varargin{:});
end
