function contract=trkg4_georg_current_model(project_root,require_converged,pointer_file)
% Resolve the common Georg reference; never fall back to a legacy mesh.
if nargin<1,project_root=fileparts(fileparts(mfilename('fullpath')));end
if nargin<2,require_converged=false;end
if nargin<3||isempty(pointer_file),pointer_file=fullfile('output','georg_current_model.local.json');end
path=char(pointer_file);
if isempty(regexp(path,'^([A-Za-z]:[\\/]|[\\/])','once')),path=fullfile(project_root,path);end
assert(isfile(path),'Shared Georg model is not selected; prepare and verify it first');
contract=jsondecode(fileread(path));
assert(any(strcmp(contract.status,{'numerical_reference_checked','geometry_reference_checked'})),...
    'Shared Georg geometry has not passed checks');
if require_converged
    permitted=strcmp(contract.status,'numerical_reference_checked');
    flags={'mesh_convergence_passed','integration_comparison_passed','quantitative_inversion_allowed'};
    for k=1:numel(flags)
        permitted=permitted && isfield(contract,flags{k}) && ...
            islogical(contract.(flags{k})) && isscalar(contract.(flags{k})) && contract.(flags{k});
    end
    assert(permitted,'trkg4:georgPrecisionUnresolved',...
        'FEM precision is unresolved; quantitative inversion is not permitted');
end
for j=1:numel(contract.files)
    file=char(contract.files(j).path);
    if isempty(regexp(file,'^([A-Za-z]:[\\/]|[\\/])','once')),file=fullfile(project_root,file);end
    assert(isfile(file)&&strcmp(trkg4_file_sha256(file),contract.files(j).sha256),...
        'Shared Georg model input changed: %s',contract.files(j).path);
end
end
