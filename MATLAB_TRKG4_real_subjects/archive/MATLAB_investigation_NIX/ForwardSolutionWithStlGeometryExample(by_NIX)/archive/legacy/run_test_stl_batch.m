function run_test_stl_batch()
%RUN_TEST_STL_BATCH Smoke test for the multi-tissue STL forward pipeline.

cfg_override = tissue_config();
cfg_override.body_stl = create_smoke_body_stl(fullfile('output', 'smoke_body_ascii.stl'));
cfg_override.mesh_fineness = 'very_coarse';
cfg_override.show_figures = false;
cfg_override.electrode_height_fraction = 0.5;

% Keep this smoke test focused on STL meshing, electrode placement, and the
% forward solve. Full tissue masking can be expensive for raw CT surfaces.
for i = 1:numel(cfg_override.tissues)
    cfg_override.tissues(i).enabled = false;
end

if ~exist('output', 'dir')
    mkdir('output');
end

diary(fullfile('output', 'test_stl_run_diary.txt'));
cleanup = onCleanup(@() diary('off'));

fprintf('STL smoke test started: %s\n', char(datetime('now')));
fprintf('Body STL: %s\n', cfg_override.body_stl);
fprintf('Mesh fineness: %s\n', cfg_override.mesh_fineness);
fprintf('Electrode height fraction: %.3f\n', cfg_override.electrode_height_fraction);

run('run_forward_multitissue_from_stl.m');

fprintf('STL smoke test finished: %s\n', char(datetime('now')));
end
