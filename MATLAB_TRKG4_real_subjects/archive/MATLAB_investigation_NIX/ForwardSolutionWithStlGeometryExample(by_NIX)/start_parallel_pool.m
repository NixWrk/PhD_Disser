function pool = start_parallel_pool(workers)
%START_PARALLEL_POOL Start a MATLAB parallel pool for tissue assignment.

pool = [];
if ~license('test', 'Distrib_Computing_Toolbox')
    warning('Parallel Computing Toolbox is not available; using serial assignment.');
    return;
end

pool = gcp('nocreate');
if ~isempty(pool)
    fprintf('Using existing parallel pool: %s with %d workers.\n', class(pool), pool.NumWorkers);
    return;
end

if nargin < 1 || isempty(workers)
    workers = 'auto';
end

try
    if ischar(workers) || isstring(workers)
        pool = parpool('threads');
    else
        pool = parpool('threads', workers);
    end
catch
    if ischar(workers) || isstring(workers)
        pool = parpool('local');
    else
        pool = parpool('local', workers);
    end
end

fprintf('Started parallel pool: %s with %d workers.\n', class(pool), pool.NumWorkers);
end
