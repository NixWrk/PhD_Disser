%% Run a matched-resolution sweep for BODY, lungs, bones, and blood.
% This sweep changes all available STL anatomy surfaces together.

if ~exist('surface_mm_values', 'var') || isempty(surface_mm_values)
    surface_mm_values = [6, 4, 3, 2];
end

if ~exist('sweep_dir', 'var') || isempty(sweep_dir)
    sweep_dir = fullfile('output', 'all_anatomy_resolution_sweep');
end

if ~exist(sweep_dir, 'dir')
    mkdir(sweep_dir);
end

summary = table('Size', [0, 14], ...
    'VariableTypes', repmat({'double'}, 1, 14), ...
    'VariableNames', {'surface_mm', 'body_faces', 'lungs_faces', 'bones_faces', ...
                      'blood_faces', 'nodes', 'elems', 'measurements', ...
                      'mesh_seconds', 'assignment_seconds', 'solve_seconds', ...
                      'total_seconds', 'nan_count', 'inf_count'});

for i = 1:numel(surface_mm_values)
    surface_mm = surface_mm_values(i);
    fprintf('\n=== ALL anatomy surface %.6g mm (%d/%d) ===\n', ...
        surface_mm, i, numel(surface_mm_values));

    files = anatomy_files_for_resolution(surface_mm);
    assert_required_files(files);

    cfg_override = tissue_config();
    cfg_override.body_stl = files.body;
    cfg_override.tissues = override_tissue_file(cfg_override.tissues, 'lungs', files.lungs);
    cfg_override.tissues = override_tissue_file(cfg_override.tissues, 'bones', files.bones);
    cfg_override.tissues = override_tissue_file(cfg_override.tissues, 'blood', files.blood);
    cfg_override.gmsh_mesh_size = [];
    cfg_override.show_figures = false;
    cfg_override.reuse_mesh_cache = true;
    cfg_override.mesh_cache_file = fullfile(sweep_dir, sprintf('body_mesh_%gmm.mat', surface_mm));
    cfg_override.output_file = fullfile(sweep_dir, sprintf('forward_all_anatomy_%gmm.mat', surface_mm));

    validate_stl_inputs(cfg_override);
    run_forward_multitissue_from_stl;

    row = array2table([surface_mm, count_stl_faces(files.body), ...
           count_stl_faces(files.lungs), count_stl_faces(files.bones), ...
           count_stl_faces(files.blood), size(fmdl.nodes, 1), size(fmdl.elems, 1), ...
           numel(v.meas), timings.mesh_seconds, timings.assignment_seconds, ...
           timings.solve_seconds, timings.total_seconds, sum(isnan(v.meas)), ...
           sum(isinf(v.meas))], ...
           'VariableNames', summary.Properties.VariableNames);
    summary = [summary; row]; %#ok<AGROW>
    save(fullfile(sweep_dir, 'all_anatomy_resolution_sweep_summary.mat'), ...
        'summary', 'surface_mm_values');
    writetable(summary, fullfile(sweep_dir, 'all_anatomy_resolution_sweep_summary.csv'));
end

disp(summary);

function files = anatomy_files_for_resolution(surface_mm)
tag = sprintf('%gmm', surface_mm);
root = fullfile('output', 'prepared');
files = struct();
files.body = fullfile(root, ['YAROSLAV_BODY_resampled_', tag, '_meshfix.stl']);
files.lungs = fullfile(root, ['YAROSLAV_LUNGS_resampled_', tag, '_meshfix.stl']);
files.bones = fullfile(root, ['YAROSLAV_BONES_resampled_', tag, '_meshfix.stl']);
files.blood = fullfile(root, ['YAROSLAV_BLOOD_resampled_', tag, '_meshfix.stl']);
end

function assert_required_files(files)
names = fieldnames(files);
for i = 1:numel(names)
    if ~isfile(files.(names{i}))
        error('Missing %s STL for matched anatomy sweep: %s', names{i}, files.(names{i}));
    end
end
end

function tissues = override_tissue_file(tissues, name, filename)
for i = 1:numel(tissues)
    if strcmp(tissues(i).name, name)
        tissues(i).file = filename;
        return;
    end
end
error('Tissue entry not found: %s', name);
end

function n_faces = count_stl_faces(filename)
surface = stl_read(filename);
n_faces = size(surface.boundary, 1);
end
