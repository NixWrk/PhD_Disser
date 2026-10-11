function contract=trkg4_georg_current_model(project_root,require_converged)
% Resolve the common Georg reference; never fall back to a legacy mesh.
if nargin<1,project_root=fileparts(fileparts(mfilename('fullpath')));end
if nargin<2,require_converged=false;end
path=fullfile(project_root,'output','georg_current_model.local.json');
assert(isfile(path),'Shared Georg model is not selected; prepare and verify it first');
contract=jsondecode(fileread(path));
assert(any(strcmp(contract.status,{'numerical_reference_checked','geometry_reference_checked'})),...
    'Shared Georg geometry has not passed checks');
if require_converged
    assert(strcmp(contract.status,'numerical_reference_checked'),'FEM precision is unresolved; quantitative inversion is not permitted');
end
for j=1:numel(contract.files)
    file=fullfile(project_root,contract.files(j).path);
    assert(isfile(file)&&strcmp(trkg4_file_sha256(file),contract.files(j).sha256),...
        'Shared Georg model input changed: %s',contract.files(j).path);
end
end
