function report = run_trkg4_electrode_sensitivity(varargin)
%RUN_TRKG4_ELECTRODE_SENSITIVITY Run the Python sensitivity orchestrator.
%
% Required name-value inputs: StudyFile and OutputDirectory.
% Optional inputs: PythonExecutable ("python"), DependenciesDirectory (""),
% and Threads (4).

args = local_normalize_arguments(varargin);
p = inputParser;
p.FunctionName = mfilename;
addParameter(p, 'StudyFile', '', @local_optional_text);
addParameter(p, 'OutputDirectory', '', @local_optional_text);
addParameter(p, 'PythonExecutable', 'python', @local_text);
addParameter(p, 'DependenciesDirectory', '', @local_optional_text);
addParameter(p, 'Threads', 4, @local_positive_integer);
parse(p, args{:});
opt = p.Results;
report = local_launch(opt);
end

function args = local_normalize_arguments(args)
if isempty(args)
    return;
end

parameter_names = ["StudyFile", "OutputDirectory", "PythonExecutable", ...
    "DependenciesDirectory", "Threads"];
first_is_name = (ischar(args{1}) || isstring(args{1})) && ...
    any(strcmpi(string(args{1}), parameter_names));
if first_is_name
    return;
end
if numel(args) < 2
    error('trkg4:missingRequiredArgument', ...
        'StudyFile and OutputDirectory are required.');
end
args = [{'StudyFile', args{1}, 'OutputDirectory', args{2}}, args(3:end)];
end

function ok = local_text(value)
ok = (ischar(value) && isrow(value)) || ...
    (isstring(value) && isscalar(value) && ~ismissing(value));
end

function ok = local_optional_text(value)
ok = isempty(value) || local_text(value);
end

function ok = local_positive_integer(value)
ok = isnumeric(value) && isreal(value) && isscalar(value) && ...
    isfinite(value) && value >= 1 && value == fix(value);
end

function report = local_launch(opt)
study_file = char(opt.StudyFile);
output_directory = char(opt.OutputDirectory);
python_executable = char(opt.PythonExecutable);
dependencies_directory = char(opt.DependenciesDirectory);

local_require_nonempty_path(study_file, 'StudyFile');
local_require_nonempty_path(output_directory, 'OutputDirectory');
local_require_nonempty_path(python_executable, 'PythonExecutable');
local_require_existing_file(study_file, 'StudyFile');

project_root = fileparts(fileparts(mfilename('fullpath')));
python_script = fullfile(project_root, 'tools', ...
    'run_electrode_sensitivity.py');
local_require_existing_file(python_script, 'orchestrator Python script');

if ~isempty(dependencies_directory)
    local_require_existing_directory(dependencies_directory, ...
        'DependenciesDirectory');
end
if local_is_explicit_path(python_executable)
    local_require_existing_file(python_executable, 'PythonExecutable');
end
if isfile(output_directory)
    error('trkg4:outputIsFile', ...
        'OutputDirectory points to a file: %s', output_directory);
end
if isfolder(output_directory)
    local_validate_shell_text(output_directory, 'OutputDirectory');
end

command_arguments = {python_executable, python_script, ...
    '--study', study_file, '--output', output_directory};
if ~isempty(dependencies_directory)
    command_arguments = [command_arguments, ...
        {'--deps', dependencies_directory}]; %#ok<AGROW>
end
command_arguments = [command_arguments, ...
    {'--threads', sprintf('%d', opt.Threads)}]; %#ok<AGROW>
quoted_arguments = cellfun(@local_shell_quote, command_arguments, ...
    'UniformOutput', false);
command = strjoin(quoted_arguments, ' ');

[exit_code, console_output] = system(command);
if exit_code ~= 0
    if isempty(strtrim(console_output))
        console_output = 'Python process returned no diagnostic output.';
    end
    error('trkg4:electrodeSensitivityPythonFailed', ...
        'Python orchestrator failed with exit code %d:\n%s', ...
        exit_code, console_output);
end

completion_file = fullfile(output_directory, 'completion.json');
if ~isfile(completion_file)
    error('trkg4:missingElectrodeSensitivityCompletion', ...
        ['Python exited successfully but did not create completion.json ', ...
         'in %s.'], output_directory);
end

try
    report = jsondecode(fileread(completion_file));
catch exception
    error('trkg4:invalidElectrodeSensitivityCompletion', ...
        'Cannot decode %s: %s', completion_file, exception.message);
end
if ~isstruct(report) || ~isscalar(report)
    error('trkg4:invalidElectrodeSensitivityCompletion', ...
        'completion.json must contain one JSON object.');
end

if ~isfield(report, 'status') || ...
        ~(ischar(report.status) || ...
        (isstring(report.status) && isscalar(report.status))) || ...
        ~strcmp(string(report.status), 'completed_numerical_checks_passed')
    error('trkg4:electrodeSensitivityCompletionNotSuccessful', ...
        ['completion.json status must be ', ...
         '''completed_numerical_checks_passed''.']);
end

report.matlab_wrapper = struct( ...
    'python_exit_code', exit_code, ...
    'python_executable', python_executable, ...
    'study_file', study_file, ...
    'output_directory', output_directory, ...
    'completion_file', completion_file);
end

function local_require_nonempty_path(value, name)
if isempty(value)
    error('trkg4:missingRequiredArgument', '%s is required.', name);
end
local_validate_shell_text(value, name);
end

function local_require_existing_file(value, name)
if ~isfile(value)
    error('trkg4:missingInputFile', '%s does not exist: %s', name, value);
end
local_validate_shell_text(value, name);
end

function local_require_existing_directory(value, name)
if ~isfolder(value)
    error('trkg4:missingInputDirectory', ...
        '%s does not exist: %s', name, value);
end
local_validate_shell_text(value, name);
end

function local_validate_shell_text(value, name)
if any(value == char(0)) || any(value == char(10)) || any(value == char(13))
    error('trkg4:unsafeCommandArgument', ...
        '%s contains a NUL or line break.', name);
end
end

function ok = local_is_explicit_path(value)
ok = ~isempty(strfind(value, filesep)) || ... %#ok<STREMP>
    ~isempty(strfind(value, '/')) || ... %#ok<STREMP>
    (ispc && ~isempty(regexp(value, '^[A-Za-z]:', 'once')));
end

function quoted = local_shell_quote(value)
value = char(value);
local_validate_shell_text(value, 'command argument');
if ispc
    % Paths cannot contain a double quote on Windows. Reject characters
    % expanded by cmd.exe instead of allowing them to alter the command.
    if any(value == char(34)) || any(value == '%') || any(value == '!')
        error('trkg4:unsafeCommandArgument', ...
            'A Windows command argument contains a shell-expanded character.');
    end
    trailing_backslashes = 0;
    for index = numel(value):-1:1
        if value(index) == char(92)
            trailing_backslashes = trailing_backslashes + 1;
        else
            break;
        end
    end
    if trailing_backslashes > 0
        value = [value, repmat(char(92), 1, trailing_backslashes)];
    end
    quoted = [char(34), value, char(34)];
    return;
end

single_quote = char(39);
backslash = char(92);
value = strrep(value, single_quote, ...
    [single_quote, backslash, single_quote, single_quote]);
quoted = [single_quote, value, single_quote];
end
