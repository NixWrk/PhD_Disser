%% Build and cache only the BODY volume mesh.
% Override `body_stl` and `mesh_cache_file` before running this script.

if ~exist('body_stl', 'var') || isempty(body_stl)
    body_stl = fullfile('output', 'prepared', 'YAROSLAV_BODY_resampled_0.5mm_meshfix.stl');
end

if ~exist('mesh_cache_file', 'var') || isempty(mesh_cache_file)
    mesh_cache_file = fullfile('output', 'body_resolution_sweep_highres', 'body_mesh_0.5mm.mat');
end

cfg = tissue_config();
cfg.body_stl = body_stl;
cfg.gmsh_mesh_size = [];
cfg.mesher = 'gmsh';

if ~isfile(cfg.body_stl)
    error('Body STL not found: %s', cfg.body_stl);
end

body_stl_abs = absolute_path(cfg.body_stl);
mesh_cache_key = stl_mesh_cache_key(cfg, body_stl_abs);

if isfile(mesh_cache_file)
    cache = load(mesh_cache_file, 'mesh_cache_key');
    if isfield(cache, 'mesh_cache_key') && isequaln(cache.mesh_cache_key, mesh_cache_key)
        fprintf('Mesh cache already exists and matches: %s\n', mesh_cache_file);
        return;
    end
end

configure_project_gmsh();
fprintf('Meshing BODY only with Gmsh: %s\n', body_stl_abs);
started = tic;
fmdl = gmsh_stl2tet(body_stl_abs, cfg.gmsh_mesh_size);
fmdl = remove_unused_nodes(fmdl);
mesh_seconds = toc(started);

cache_dir = fileparts(mesh_cache_file);
if ~isempty(cache_dir) && ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end
save(mesh_cache_file, 'fmdl', 'mesh_cache_key', 'mesh_seconds', '-v7.3');
fprintf('Saved mesh cache: %s\n', mesh_cache_file);
fprintf('Mesh: %d nodes, %d tetrahedra; %.3f s\n', ...
    size(fmdl.nodes, 1), size(fmdl.elems, 1), mesh_seconds);

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
