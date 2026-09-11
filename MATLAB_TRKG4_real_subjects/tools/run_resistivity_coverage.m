% Independent rho-grid checks through the canonical pipeline.
% The caller provides rho_python_executable.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(project_root);
base = fullfile(project_root,'output','exploratory','resistivity_sensitivity_20260910');
plan = jsondecode(fileread(fullfile(base,'coverage_plan.json')));
for i = 1:numel(plan.montages)
    montage = plan.montages{i};
    fprintf('START %d/%d %s %s\n',i,numel(plan.montages),montage,datestr(now,31));
    run_pipeline('Profile','electrode_sensitivity', ...
        'StudyFile',fullfile(base,'studies',[montage '.json']), ...
        'OutputDirectory',fullfile(base,'fem',montage), ...
        'PythonExecutable',rho_python_executable, ...
        'DependenciesDirectory',fullfile(project_root,'output','exploratory','arm_sigma_20260908','python_solver_deps'), ...
        'Threads',4);
    fprintf('DONE %d/%d %s %s\n',i,numel(plan.montages),montage,datestr(now,31));
end
