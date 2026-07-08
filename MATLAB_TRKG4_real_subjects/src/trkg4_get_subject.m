function subject = trkg4_get_subject(subjects, subject_id)
%TRKG4_GET_SUBJECT Return one subject struct by id.

ids = string({subjects.id});
idx = find(strcmpi(ids, string(subject_id)), 1);
if isempty(idx)
    error('Unknown subject id: %s', subject_id);
end
subject = subjects(idx);
end
