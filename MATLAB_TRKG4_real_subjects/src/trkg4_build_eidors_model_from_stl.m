function [fmdl, electrode_centres_xyz, timings, electrode_diagnostics] = trkg4_build_eidors_model_from_stl(cfg)
%TRKG4_BUILD_EIDORS_MODEL_FROM_STL Mesh body STL and attach 4 electrodes.

timings = struct();
electrode_diagnostics = table();
body_stl = trkg4_absolute_path(cfg.body_stl);

mesh_timer = tic;
mesh_cache_key = stl_mesh_cache_key(cfg, body_stl);
if cfg.reuse_mesh_cache
    [fmdl, cache_hit] = local_load_cached_mesh(cfg.mesh_cache_file, mesh_cache_key);
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
            local_configure_project_gmsh(cfg);
            fprintf('Meshing outer body STL with Gmsh: %s\n', body_stl);
            fmdl = gmsh_stl2tet(body_stl, cfg.gmsh_mesh_size);
        otherwise
            error('Unsupported cfg.mesher: %s', cfg.mesher);
    end
    fmdl = remove_unused_nodes(fmdl);
    local_save_cached_mesh(cfg.mesh_cache_file, fmdl, mesh_cache_key, cfg.reuse_mesh_cache);
end
fmdl.name = sprintf('%s body mesh in %s', cfg.subject.id, cfg.length_unit);
timings.mesh_seconds = toc(mesh_timer);

boundary_timer = tic;
TR = triangulation(fmdl.elems, fmdl.nodes);
fmdl.boundary = freeBoundary(TR);
timings.boundary_seconds = toc(boundary_timer);

electrode_timer = tic;
[electrode_centres_xyz, electrode_labels] = trkg4_load_electrode_centres(cfg);
fmdl.electrode = electrode_faces_by_area( ...
    fmdl, electrode_centres_xyz, cfg.electrode_area, cfg.z_contact);
electrode_diagnostics = trkg4_electrode_diagnostics( ...
    fmdl, electrode_centres_xyz, electrode_labels, cfg);
[fmdl.stimulation, fmdl.meas_select] = trkg4_make_4electrode_stimulation(cfg);
timings.electrode_seconds = toc(electrode_timer);

fprintf('Mesh: %d nodes, %d tetrahedra, %d boundary faces\n', ...
    size(fmdl.nodes, 1), size(fmdl.elems, 1), size(fmdl.boundary, 1));
end

function local_configure_project_gmsh(cfg)
cache_dir = fullfile(cfg.project_root, 'output', 'eidors_cache');
if ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end
eidors_cache('cache_path', cache_dir);
end

function [fmdl, cache_hit] = local_load_cached_mesh(cache_file, expected_key)
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

function local_save_cached_mesh(cache_file, fmdl, mesh_cache_key, reuse_mesh_cache)
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
