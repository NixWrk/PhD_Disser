function run_families(manifestfile)
% Verify the explicit private adapter, then dispatch one canonical FEM batch.
m=jsondecode(fileread(manifestfile));
addpath(fullfile(m.root,'MATLAB_TRKG4_real_subjects','src'));
assert(strcmp(trkg4_file_sha256(which('run_families')),m.wrapper_matlab_sha256));
assert(strcmp(trkg4_file_sha256(m.template_path),m.template_sha256));
assert(strcmp(trkg4_file_sha256(m.generated_driver),m.matlab_sha256));
addpath(fileparts(m.generated_driver));
solve_families(manifestfile);
end
