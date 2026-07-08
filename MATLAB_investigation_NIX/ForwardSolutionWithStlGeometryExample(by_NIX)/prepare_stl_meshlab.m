function output_file = prepare_stl_meshlab(input_file, output_file, varargin)
%PREPARE_STL_MESHLAB Repair/decimate STL with PyMeshLab for Netgen.
%   OUTPUT_FILE = PREPARE_STL_MESHLAB(INPUT_FILE, OUTPUT_FILE) runs the
%   Python PyMeshLab helper and writes an ASCII STL.

if nargin < 2 || isempty(output_file)
    [folder, name] = fileparts(input_file);
    output_file = fullfile(folder, [name '_meshlab_ascii.stl']);
end

params = inputParser;
params.addParameter('TargetFaces', 200000, @(x) isnumeric(x) && isscalar(x));
params.addParameter('CloseHoles', 100, @(x) isnumeric(x) && isscalar(x));
params.addParameter('MinComponentFaces', 100, @(x) isnumeric(x) && isscalar(x));
params.addParameter('MergeThreshold', 0, @(x) isnumeric(x) && isscalar(x));
params.parse(varargin{:});

script_file = fullfile(fileparts(mfilename('fullpath')), 'prepare_stl_meshlab.py');

cmd = sprintf(['python "%s" "%s" "%s" --target-faces %d --close-holes %d ', ...
    '--min-component-faces %d --merge-threshold %.12g'], ...
    script_file, input_file, output_file, ...
    params.Results.TargetFaces, params.Results.CloseHoles, ...
    params.Results.MinComponentFaces, params.Results.MergeThreshold);

status = system(cmd);
if status ~= 0
    error('PyMeshLab STL preparation failed for %s.', input_file);
end
end
