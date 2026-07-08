function results = run_trkg4_batch(subject_ids)
%RUN_TRKG4_BATCH Run all configured subjects or a selected id list.

root = fileparts(fileparts(mfilename('fullpath')));
subjects = trkg4_subject_registry(root);

if nargin < 1 || isempty(subject_ids)
    subject_ids = {subjects.id};
end
subject_ids = cellstr(string(subject_ids));

results = cell(numel(subject_ids), 1);
for k = 1:numel(subject_ids)
    fprintf('\n=== %s ===\n', subject_ids{k});
    results{k} = run_trkg4_forward_subject(subject_ids{k});
    trkg4_result_summary(results{k});
end
end
