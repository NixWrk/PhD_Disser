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

profile_index=find(cellfun(@(x)(ischar(x)||isstring(x)) && strcmpi(string(x),"Profile"),varargin),1);
if ~isempty(profile_index) && strcmpi(string(varargin{profile_index+1}),"two_tissue_comparison")
    report=run_trkg4_two_tissue_comparison(varargin{:});
    return;
end
if ~isempty(profile_index) && strcmpi(string(varargin{profile_index+1}),"arm_conductivity_matrix")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_arm_conductivity_matrix(args{:});
else
    % The exhaustive scan has its own stage/checkpoint contract. Route it through
% the same public entrypoint while leaving historical profile semantics intact.
profile_index=find(strcmpi(string(varargin(1:2:end)),"Profile"),1);
if ~isempty(profile_index) && strcmpi(string(varargin{2*profile_index}),"full_pose_scan")
    report = run_trkg4_full_pose_scan(varargin{:});
else
    report = run_trkg4_notebook_pipeline(varargin{:});
end
end
end
