function result = run_trkg4_forward_subject(subject_id, varargin)
%RUN_TRKG4_FORWARD_SUBJECT Run one 4-electrode forward TRKG calculation.

if nargin < 1 || isempty(subject_id)
    subject_id = 'nik';
end

cfg = trkg4_config(subject_id);
cfg = trkg4_apply_overrides(cfg, varargin{:});

addpath(cfg.src_dir);
addpath(cfg.vendor_dir);
if ~isempty(cfg.eidors_startup)
    run(cfg.eidors_startup);
end

validate_stl_inputs(cfg);

total_timer = tic;
[fmdl_mm, electrode_centres_xyz_mm, timings, electrode_diagnostics] = ...
    trkg4_build_eidors_model_from_stl(cfg);

assignment_timer = tic;
if cfg.use_parallel_assignment
    start_parallel_pool(cfg.parallel_workers);
end
centroids_mm = element_centroids(fmdl_mm);
[elem_sigma, tissue_id, tissue_names, tissue_counts] = assign_tissues_from_stl(centroids_mm, cfg);
timings.assignment_seconds = toc(assignment_timer);

fmdl = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);
img = mk_image(fmdl, 1);
img.elem_data = elem_sigma;
img.name = sprintf('%s TRKG4 conductivity', cfg.subject.id);

solve_timer = tic;
v = fwd_solve(img);
timings.solve_seconds = toc(solve_timer);
timings.total_seconds = toc(total_timer);

Z_model_ohm = v.meas(1) / cfg.current_ampere;

result = struct();
result.cfg = cfg;
result.fmdl = fmdl;
result.fmdl_mesh_units = fmdl_mm;
result.img = img;
result.v = v;
result.Z_model_ohm = Z_model_ohm;
result.elem_sigma = elem_sigma;
result.tissue_id = tissue_id;
result.tissue_names = tissue_names;
result.tissue_counts = tissue_counts;
result.electrode_centres_xyz_mm = electrode_centres_xyz_mm;
result.electrode_diagnostics = electrode_diagnostics;
result.timings = timings;

out_dir = fileparts(cfg.output_file);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
save(cfg.output_file, '-struct', 'result', '-v7.3');

if cfg.show_figures
    figure('Name', 'TRKG4 model');
    show_fem(fmdl);
    title(sprintf('%s TRKG4 real CT/STL model', cfg.subject.id));

    figure('Name', 'TRKG4 forward measurement');
    plot(v.meas, 'o', 'LineWidth', 1.2);
    grid on;
    ylabel('V for configured stimulation');
    title('4-electrode transfer measurement');
end
end
