function summary = summarize_body_resolution_results()
%SUMMARIZE_BODY_RESOLUTION_RESULTS Compare saved BODY resolution sweeps.

result_files = { ...
    fullfile('output', 'body_resolution_sweep', 'forward_body_6mm.mat'), ...
    fullfile('output', 'body_resolution_sweep', 'forward_body_4mm.mat'), ...
    fullfile('output', 'body_resolution_sweep', 'forward_body_3mm.mat'), ...
    fullfile('output', 'body_resolution_sweep_highres', 'forward_body_2mm.mat'), ...
    fullfile('output', 'body_resolution_sweep_highres', 'forward_body_1.5mm.mat'), ...
    fullfile('output', 'body_resolution_sweep_highres', 'forward_body_1mm.mat')};

existing = result_files(cellfun(@isfile, result_files));
if isempty(existing)
    error('No BODY resolution result files found.');
end

n = numel(existing);
surface_mm = zeros(n, 1);
surface_faces = zeros(n, 1);
nodes = zeros(n, 1);
elems = zeros(n, 1);
mesh_seconds = zeros(n, 1);
assignment_seconds = zeros(n, 1);
solve_seconds = zeros(n, 1);
total_seconds = zeros(n, 1);
nan_count = zeros(n, 1);
inf_count = zeros(n, 1);
measurements = cell(n, 1);

for i = 1:n
    data = load(existing{i}, 'cfg', 'fmdl', 'v', 'timings');
    surface_mm(i) = parse_surface_mm(data.cfg.body_stl);
    surface_faces(i) = size(data.fmdl.boundary, 1);
    nodes(i) = size(data.fmdl.nodes, 1);
    elems(i) = size(data.fmdl.elems, 1);
    mesh_seconds(i) = data.timings.mesh_seconds;
    assignment_seconds(i) = data.timings.assignment_seconds;
    solve_seconds(i) = data.timings.solve_seconds;
    total_seconds(i) = data.timings.total_seconds;
    nan_count(i) = sum(isnan(data.v.meas));
    inf_count(i) = sum(isinf(data.v.meas));
    measurements{i} = data.v.meas;
end

[surface_mm, order] = sort(surface_mm, 'descend');
existing = existing(order);
surface_faces = surface_faces(order);
nodes = nodes(order);
elems = elems(order);
mesh_seconds = mesh_seconds(order);
assignment_seconds = assignment_seconds(order);
solve_seconds = solve_seconds(order);
total_seconds = total_seconds(order);
nan_count = nan_count(order);
inf_count = inf_count(order);
measurements = measurements(order);

ref = measurements{end};
rel_l2_vs_finest = zeros(n, 1);
max_abs_vs_finest = zeros(n, 1);
max_rel_vs_finest = zeros(n, 1);
for i = 1:n
    err = measurements{i} - ref;
    rel_l2_vs_finest(i) = norm(err) / max(norm(ref), eps);
    max_abs_vs_finest(i) = max(abs(err));
    max_rel_vs_finest(i) = max(abs(err) ./ max(abs(ref), eps));
end

summary = table(surface_mm, surface_faces, nodes, elems, mesh_seconds, ...
    assignment_seconds, solve_seconds, total_seconds, nan_count, inf_count, ...
    rel_l2_vs_finest, max_abs_vs_finest, max_rel_vs_finest, existing(:), ...
    'VariableNames', {'surface_mm', 'surface_faces', 'nodes', 'elems', ...
    'mesh_seconds', 'assignment_seconds', 'solve_seconds', 'total_seconds', ...
    'nan_count', 'inf_count', 'rel_l2_vs_finest', 'max_abs_vs_finest', ...
    'max_rel_vs_finest', 'result_file'});

out_dir = fullfile('output', 'body_resolution_sweep_combined');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
save(fullfile(out_dir, 'body_resolution_summary.mat'), 'summary');
writetable(summary, fullfile(out_dir, 'body_resolution_summary.csv'));
disp(summary);
end

function surface_mm = parse_surface_mm(filename)
token = regexp(filename, 'resampled_(\d+(?:\.\d+)?)mm', 'tokens', 'once');
if isempty(token)
    surface_mm = NaN;
else
    surface_mm = str2double(token{1});
end
end
