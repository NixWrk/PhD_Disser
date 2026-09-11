function run_resistivity_refinement(rho_python_executable)
%RUN_RESISTIVITY_REFINEMENT Execute sealed batches via the canonical pipeline.
% Completed batches are verified/reused. An interrupted batch gets a new attempt.
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(project_root);
addpath(fullfile(project_root,'src'));
base = fullfile(project_root,'output','exploratory','resistivity_refinement_20260910');
plan_file = fullfile(base,'refinement_plan.json');
assert(strcmp(trkg4_file_sha256(plan_file),strtrim(fileread(fullfile(base,'refinement_plan.sha256')))), 'Refinement plan integrity mismatch');
plan = jsondecode(fileread(plan_file));
deps = fullfile(project_root,'output','exploratory','arm_sigma_20260908','python_solver_deps');
progress_file = fullfile(base,'progress.json');
progress = struct('status','running','completed_batches',0,'total_batches',numel(plan.batches), ...
    'completed_new_states',0,'new_states_required',plan.new_fem_states,'current_batch','');
for i = 1:numel(plan.batches)
    b = plan.batches(i);
    study_file = fullfile(base,b.study);
    assert(strcmp(trkg4_file_sha256(study_file),b.study_sha256),'Study changed after preregistration');
    output = fullfile(base,b.output);
    % Reject ambiguous completions before choosing a restart folder.
    completed = {};
    if isfile(fullfile(output,'completion.json')), completed{end+1} = output; end
    attempts = dir([output '_attempt_*']);
    for ai = 1:numel(attempts)
        candidate = fullfile(attempts(ai).folder,attempts(ai).name);
        if attempts(ai).isdir && isfile(fullfile(candidate,'completion.json'))
            completed{end+1} = candidate; %#ok<AGROW>
        end
    end
    assert(numel(completed)<=1,'More than one completed batch attempt; explicit review required');
    if numel(completed)==1, output=completed{1}; end
    % Prior unfinished artifacts remain untouched and visible for investigation.
    attempt = 0;
    while isfolder(output) && ~isfile(fullfile(output,'completion.json'))
        listing = dir(output);
        if numel(listing) <= 2, break; end
        attempt = attempt+1;
        output = fullfile(base,sprintf('%s_attempt_%02d',b.output,attempt));
    end
    progress.current_batch = b.id;
    progress.updated_local = datestr(now,31);
    local_write_json(progress_file,progress);
    fprintf('START %d/%d %s %s\n',i,numel(plan.batches),b.id,datestr(now,31));
    try
        run_pipeline('Profile','electrode_sensitivity','StudyFile',study_file, ...
            'OutputDirectory',output,'PythonExecutable',rho_python_executable, ...
            'DependenciesDirectory',deps,'Threads',4);
    catch exception
        progress.status = 'failed';
        progress.message = exception.message;
        local_write_json(progress_file,progress);
        rethrow(exception);
    end
    progress.completed_batches = i;
    progress.completed_new_states = progress.completed_new_states+b.states;
    local_write_json(progress_file,progress);
    fprintf('DONE %d/%d %s %s\n',i,numel(plan.batches),b.id,datestr(now,31));
end
progress.status = 'fem_complete_analysis_pending';
progress.current_batch = '';
progress.updated_local = datestr(now,31);
local_write_json(progress_file,progress);
end

function local_write_json(path,value)
temporary = [path '.tmp'];
f = fopen(temporary,'w','n','UTF-8');
assert(f>=0,'Cannot write progress file');
cleaner = onCleanup(@() fclose(f));
fprintf(f,'%s\n',jsonencode(value));
clear cleaner;
movefile(temporary,path,'f');
end
