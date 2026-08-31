%% Multi-tissue forward EIT calculation from STL surfaces
% The outer body STL is meshed into tetrahedra. Internal tissue STLs are
% used to classify tetrahedral element centroids and assign conductivities.

if exist('cfg_override', 'var')
    cfg = cfg_override;
    clear cfg_override;
else
    clearvars;
    cfg = tissue_config();
end
close all; clc;

if ~isfield(cfg, 'show_figures')
    cfg.show_figures = true;
end

if ~isfield(cfg, 'electrode_height_fraction')
    cfg.electrode_height_fraction = 0.5;
end
if ~isfield(cfg, 'mesher')
    cfg.mesher = 'netgen';
end
if ~isfield(cfg, 'gmsh_mesh_size')
    cfg.gmsh_mesh_size = [];
end
if ~isfield(cfg, 'reuse_mesh_cache')
    cfg.reuse_mesh_cache = false;
end
if ~isfield(cfg, 'mesh_cache_file')
    cfg.mesh_cache_file = fullfile('output', 'stl_body_mesh_cache.mat');
end
if ~isfield(cfg, 'parallel_workers')
    cfg.parallel_workers = 'auto';
end
if ~isfield(cfg, 'output_file')
    cfg.output_file = fullfile('output', 'forward_multitissue_result.mat');
end

if ~isfile(cfg.body_stl)
    error(['Outer body STL not found: %s\n', ...
           'Put the closed external surface at stl/body.stl or edit tissue_config.m.'], cfg.body_stl);
end

fprintf('Conductivities: %s at %.6g Hz\n', cfg.itis_database, cfg.frequency_hz);
timings = struct();
total_timer = tic;

%% Build volume mesh from outer body STL
body_stl = absolute_path(cfg.body_stl);
mesh_timer = tic;
mesh_cache_key = stl_mesh_cache_key(cfg, body_stl);
if cfg.reuse_mesh_cache
    [fmdl, cache_hit] = load_cached_mesh(cfg.mesh_cache_file, mesh_cache_key);
else
    fmdl = [];
    cache_hit = false;
end

if cache_hit
    fprintf('Loaded cached body mesh: %s\n', cfg.mesh_cache_file);
else
    switch lower(cfg.mesher)
        case 'netgen'
            fprintf('Meshing outer body STL with Netgen: %s\n', body_stl);
            fmdl = ng_stl2tet(body_stl, cfg.mesh_fineness);
        case 'gmsh'
            configure_project_gmsh();
            fprintf('Meshing outer body STL with Gmsh: %s\n', body_stl);
            fmdl = gmsh_stl2tet(body_stl, cfg.gmsh_mesh_size);
        otherwise
            error('Unsupported cfg.mesher: %s', cfg.mesher);
    end
    fmdl = remove_unused_nodes(fmdl);
    save_cached_mesh(cfg.mesh_cache_file, fmdl, mesh_cache_key, cfg.reuse_mesh_cache);
end
fmdl.name = 'Multi-STL forward model';
timings.mesh_seconds = toc(mesh_timer);

boundary_timer = tic;
TR = triangulation(fmdl.elems, fmdl.nodes);
fmdl.boundary = freeBoundary(TR);
timings.boundary_seconds = toc(boundary_timer);
fprintf('Mesh: %d nodes, %d tetrahedra, %d boundary faces\n', ...
    size(fmdl.nodes, 1), size(fmdl.elems, 1), size(fmdl.boundary, 1));
fprintf('Meshing required %.3f s; boundary extraction %.3f s\n', ...
    timings.mesh_seconds, timings.boundary_seconds);

%% Electrodes
electrode_timer = tic;
if isempty(cfg.electrode_centres_xyz)
    electrode_centres_xyz = default_ring_electrodes(fmdl.nodes, cfg.n_electrodes, cfg.electrode_height_fraction);
    fprintf('Using default ring electrode centres at %.1f%% of model height.\n', ...
        100 * cfg.electrode_height_fraction);
else
    electrode_centres_xyz = cfg.electrode_centres_xyz;
end

if size(electrode_centres_xyz, 1) ~= cfg.n_electrodes || size(electrode_centres_xyz, 2) ~= 3
    error('cfg.electrode_centres_xyz must be %d x 3.', cfg.n_electrodes);
end

fmdl.electrode = electrode_faces_by_area(fmdl, electrode_centres_xyz, cfg.electrode_area, cfg.z_contact);
[fmdl.stimulation, fmdl.meas_select] = mk_stim_patterns( ...
    cfg.n_electrodes, 1, [0, 1], [0, 1], {'rotate_meas'}, 1);
timings.electrode_seconds = toc(electrode_timer);

%% Tissue assignment
assignment_timer = tic;
if isfield(cfg, 'use_parallel_assignment') && cfg.use_parallel_assignment
    start_parallel_pool(cfg.parallel_workers);
end
centroids = element_centroids(fmdl);
[elem_sigma, tissue_id, tissue_names, tissue_counts] = assign_tissues_from_stl(centroids, cfg);
timings.assignment_seconds = toc(assignment_timer);

img = mk_image(fmdl, 1);
img.elem_data = elem_sigma;
img.name = 'Multi-tissue conductivity';

fprintf('\nTissue assignment:\n');
for i = 1:numel(tissue_names)
    sigma_i = unique(elem_sigma(tissue_id == i));
    if isempty(sigma_i)
        fprintf('  %-14s %8d elems, sigma = n/a\n', tissue_names{i}, tissue_counts(i));
    else
        fprintf('  %-14s %8d elems, sigma = %.4g S/m\n', tissue_names{i}, tissue_counts(i), sigma_i(1));
    end
end
fprintf('Tissue assignment required %.3f s\n', timings.assignment_seconds);

%% Solve forward problem
tic;
v = fwd_solve(img);
fwd_duration = toc;
timings.solve_seconds = fwd_duration;
timings.total_seconds = toc(total_timer);

fprintf('\nForward solution required %.3f s\n', fwd_duration);
fprintf('Measurements: %d\n', numel(v.meas));
fprintf('Total run time %.3f s\n', timings.total_seconds);

%% Visualize
if cfg.show_figures
    figure('Name', 'Multi-STL EIDORS model');
    show_fem(fmdl);
    title('Multi-STL forward model');

    figure('Name', 'Conductivity slices');
    show_slices(img, [inf, inf, mean(fmdl.nodes(:,3))]);
    eidors_colourbar(img);
    title('Assigned conductivity');

    figure('Name', 'Forward measurements');
    plot(v.meas, '-o', 'LineWidth', 1.1);
    grid on; grid minor;
    xlabel('Measurement number');
    ylabel('Transfer impedance');
    title('Forward transfer impedance');
end

%% Save result
if ~exist('output', 'dir')
    mkdir('output');
end
save(cfg.output_file, ...
    'cfg', 'fmdl', 'img', 'v', 'elem_sigma', 'tissue_id', 'tissue_names', ...
    'tissue_counts', 'electrode_centres_xyz', 'fwd_duration', 'timings');
fprintf('Saved result: %s\n', cfg.output_file);

function filename = absolute_path(filename)
if isfile(filename)
    info = dir(filename);
    filename = fullfile(info.folder, info.name);
end
end

function configure_project_gmsh()
cache_dir = fullfile(pwd, 'output', 'eidors_cache');
if ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end
eidors_cache('cache_path', cache_dir);
end

function [fmdl, cache_hit] = load_cached_mesh(cache_file, expected_key)
fmdl = [];
cache_hit = false;
if ~isfile(cache_file)
    return;
end

cache = load(cache_file, 'fmdl', 'mesh_cache_key');
if isfield(cache, 'mesh_cache_key') && isequaln(cache.mesh_cache_key, expected_key)
    fmdl = cache.fmdl;
    cache_hit = true;
end
end

function save_cached_mesh(cache_file, fmdl, mesh_cache_key, reuse_mesh_cache)
if ~reuse_mesh_cache
    return;
end

cache_dir = fileparts(cache_file);
if ~isempty(cache_dir) && ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end
save(cache_file, 'fmdl', 'mesh_cache_key', '-v7.3');
fprintf('Saved body mesh cache: %s\n', cache_file);
end
