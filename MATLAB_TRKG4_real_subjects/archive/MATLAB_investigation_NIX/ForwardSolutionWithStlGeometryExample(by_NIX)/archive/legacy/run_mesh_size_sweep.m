%% Run a mesh-size sweep for the YAROSLAV STL-only forward model.
% Override `mesh_sizes` before running this script to customize the sweep.

if ~exist('mesh_sizes', 'var') || isempty(mesh_sizes)
    mesh_sizes = [40, 35, 30, 25];
end

if ~exist('sweep_dir', 'var') || isempty(sweep_dir)
    sweep_dir = fullfile('output', 'mesh_sweep');
end

if ~exist(sweep_dir, 'dir')
    mkdir(sweep_dir);
end

summary = table('Size', [0, 10], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double', ...
                      'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'mesh_size', 'nodes', 'elems', 'measurements', ...
                      'mesh_seconds', 'assignment_seconds', 'solve_seconds', ...
                      'total_seconds', 'nan_count', 'inf_count'});

for i = 1:numel(mesh_sizes)
    mesh_size = mesh_sizes(i);
    fprintf('\n=== Mesh size %.6g (%d/%d) ===\n', mesh_size, i, numel(mesh_sizes));

    cfg_override = tissue_config();
    cfg_override.gmsh_mesh_size = mesh_size;
    cfg_override.show_figures = false;
    cfg_override.reuse_mesh_cache = true;
    cfg_override.mesh_cache_file = fullfile(sweep_dir, sprintf('body_mesh_gmsh_%g.mat', mesh_size));
    cfg_override.output_file = fullfile(sweep_dir, sprintf('forward_gmsh_%g.mat', mesh_size));

    if mesh_size == 40 && ~isfile(cfg_override.mesh_cache_file)
        default_cache = fullfile('output', 'stl_body_mesh_cache.mat');
        if isfile(default_cache)
            copyfile(default_cache, cfg_override.mesh_cache_file);
        end
    end

    validate_stl_inputs(cfg_override);
    run_forward_multitissue_from_stl;

    row = array2table([mesh_size, size(fmdl.nodes, 1), size(fmdl.elems, 1), numel(v.meas), ...
           timings.mesh_seconds, timings.assignment_seconds, timings.solve_seconds, ...
           timings.total_seconds, sum(isnan(v.meas)), sum(isinf(v.meas))], ...
           'VariableNames', summary.Properties.VariableNames);
    summary = [summary; row]; %#ok<AGROW>
    save(fullfile(sweep_dir, 'mesh_sweep_summary.mat'), 'summary', 'mesh_sizes');
    writetable(summary, fullfile(sweep_dir, 'mesh_sweep_summary.csv'));
end

disp(summary);
