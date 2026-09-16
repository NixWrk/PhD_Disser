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
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"lateral_resistivity_range")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_lateral_resistivity_range(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"lateral_array_design")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_lateral_array_design(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"lateral_real_application")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_lateral_real_application(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"c01_baseline_models")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_c01_baseline_models(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"halfspace_reference_probe")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_halfspace_reference_probe(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"transverse_depth")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_transverse_depth(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"transverse_inverse_cem")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_transverse_cem_inverse(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && strcmpi(string(varargin{profile_index+1}),"transverse_geometry")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_transverse_geometry(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && ...
        strcmpi(string(varargin{profile_index+1}),"electrode_sensitivity")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_electrode_sensitivity(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && ...
        strcmpi(string(varargin{profile_index+1}),"geometry_validation_derivative_qc")
    report=run_trkg4_geometry_validation_derivative_qc();
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && ...
        strcmpi(string(varargin{profile_index+1}),"geometry_validation_cem")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_geometry_validation_cem(args{:});
    return;
end
if ~isempty(profile_index) && profile_index < numel(varargin) && ...
        strcmpi(string(varargin{profile_index+1}),"geometry_validation")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_geometry_validation(args{:});
    return;
end
if ~isempty(profile_index) && strcmpi(string(varargin{profile_index+1}),"geometry_reduction")
    args=varargin;args(profile_index:profile_index+1)=[];
    report=run_trkg4_geometry_reduction(args{:});
    return;
end

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
