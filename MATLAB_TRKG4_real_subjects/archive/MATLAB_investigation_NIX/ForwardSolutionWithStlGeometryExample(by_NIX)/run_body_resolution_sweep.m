%% Run a BODY surface-resolution sweep for the YAROSLAV STL-only model.
% This is the meaningful sweep for STL anatomy simplification: the external
% surface resolution changes, while tissue masks and conductivities stay fixed.

if ~exist('body_surface_files', 'var') || isempty(body_surface_files)
    body_surface_files = { ...
        fullfile('output', 'prepared', 'YAROSLAV_BODY_resampled_6mm_meshfix.stl'), ...
        fullfile('output', 'prepared', 'YAROSLAV_BODY_resampled_4mm_meshfix.stl'), ...
        fullfile('output', 'prepared', 'YAROSLAV_BODY_resampled_3mm_meshfix.stl')};
end

if ~exist('sweep_dir', 'var') || isempty(sweep_dir)
    sweep_dir = fullfile('output', 'body_resolution_sweep');
end

if ~exist(sweep_dir, 'dir')
    mkdir(sweep_dir);
end

summary = table('Size', [0, 11], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double', ...
                      'double', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'surface_mm', 'surface_faces', 'nodes', 'elems', ...
                      'measurements', 'mesh_seconds', 'assignment_seconds', ...
                      'solve_seconds', 'total_seconds', 'nan_count', 'inf_count'});

for i = 1:numel(body_surface_files)
    body_file = body_surface_files{i};
    surface_mm = parse_surface_mm(body_file);
    fprintf('\n=== BODY surface %.6g mm (%d/%d) ===\n', surface_mm, i, numel(body_surface_files));

    cfg_override = tissue_config();
    cfg_override.body_stl = body_file;
    cfg_override.gmsh_mesh_size = [];
    cfg_override.show_figures = false;
    cfg_override.reuse_mesh_cache = true;
    cfg_override.mesh_cache_file = fullfile(sweep_dir, sprintf('body_mesh_%gmm.mat', surface_mm));
    cfg_override.output_file = fullfile(sweep_dir, sprintf('forward_body_%gmm.mat', surface_mm));

    validate_stl_inputs(cfg_override);
    run_forward_multitissue_from_stl;

    row = array2table([surface_mm, size(fmdl.boundary, 1), size(fmdl.nodes, 1), ...
           size(fmdl.elems, 1), numel(v.meas), timings.mesh_seconds, ...
           timings.assignment_seconds, timings.solve_seconds, timings.total_seconds, ...
           sum(isnan(v.meas)), sum(isinf(v.meas))], ...
           'VariableNames', summary.Properties.VariableNames);
    summary = [summary; row]; %#ok<AGROW>
    save(fullfile(sweep_dir, 'body_resolution_sweep_summary.mat'), 'summary', 'body_surface_files');
    writetable(summary, fullfile(sweep_dir, 'body_resolution_sweep_summary.csv'));
end

disp(summary);

function surface_mm = parse_surface_mm(filename)
token = regexp(filename, 'resampled_(\d+(?:\.\d+)?)mm', 'tokens', 'once');
if isempty(token)
    surface_mm = NaN;
else
    surface_mm = str2double(token{1});
end
end
