function trkg4_prepare_runtime(cfg)
%TRKG4_PREPARE_RUNTIME Load EIDORS and expose an optional Gmsh executable.
%
% Machine-specific locations come from environment variables and are not
% stored in the repository:
%   EIDORS_STARTUP  full path to EIDORS startup.m
%   GMSH_EXE        full path to gmsh.exe

% A caller may also provide the same values through cfg.eidors_startup and
% cfg.gmsh_executable. Already loaded EIDORS sessions are left unchanged.

if nargin < 1 || isempty(cfg)
    cfg = struct();
end

if exist('fwd_solve', 'file') == 0 || exist('mk_image', 'file') == 0
    startup_file = local_config_value(cfg, 'eidors_startup', ...
        local_environment_value('EIDORS_STARTUP'));
    if isempty(startup_file) || ~isfile(startup_file)
        error('trkg4:missingEidors', [ ...
            'EIDORS is not loaded. Set EIDORS_STARTUP to the full path ', ...
            'of EIDORS startup.m or pass cfg.eidors_startup.']);
    end
    run(startup_file);
end

required = {'fwd_solve', 'mk_image', 'gmsh_mk_fwd_model'};
missing = required(cellfun(@(name) exist(name, 'file') == 0, required));
if ~isempty(missing)
    error('trkg4:incompleteEidors', ...
        'EIDORS is incomplete; missing functions: %s.', ...
        strjoin(missing, ', '));
end

gmsh_executable = local_config_value(cfg, 'gmsh_executable', ...
    local_environment_value('GMSH_EXE'));
if isempty(gmsh_executable)
    return;
end
if ~isfile(gmsh_executable)
    error('trkg4:missingGmshExecutable', ...
        'Configured Gmsh executable is missing: %s', gmsh_executable);
end

gmsh_dir = fileparts(gmsh_executable);
path_entries = strsplit(getenv('PATH'), pathsep);
if ~any(strcmpi(path_entries, gmsh_dir))
    setenv('PATH', [gmsh_dir, pathsep, getenv('PATH')]);
end
end

function value = local_config_value(cfg, field_name, fallback)
value = fallback;
if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
    value = char(cfg.(field_name));
end
end

function value = local_environment_value(name)
value = getenv(name);
if ~isempty(value) || ~ispc
    return;
end

% A long-running parent application may have started before setx or the
% Windows environment dialog changed the user variable. In that case a new
% MATLAB child process inherits the stale process environment. Read the same
% per-user value from the registry as a fallback, without storing a path in
% the repository.
try
    value = winqueryreg('HKEY_CURRENT_USER', 'Environment', name);
catch
    value = '';
end
end
