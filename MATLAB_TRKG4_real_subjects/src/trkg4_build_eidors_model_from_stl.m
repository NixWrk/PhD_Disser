function [fmdl, electrode_centres_xyz, timings, electrode_diagnostics] = trkg4_build_eidors_model_from_stl(cfg)
%TRKG4_BUILD_EIDORS_MODEL_FROM_STL Mesh body STL and attach 4 electrodes.

timings = struct();
% Declared up front so the output has a defined type if the function is
% interrupted before the electrodes are attached.
electrode_diagnostics = table(); %#ok<NASGU>
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
    % A configured prebuilt mesh is a hard requirement, not a hint. Falling
    % back to on-the-fly meshing would silently replace the verified 1 mm
    % surface / 2 mm local mesh by a cfg.gmsh_mesh_size one (25 mm by
    % default) and change every reported impedance without a single warning.
    use_prebuilt = isfield(cfg, 'prebuilt_mesh_file') && ...
        ~isempty(cfg.prebuilt_mesh_file);
    if use_prebuilt && ~isfile(cfg.prebuilt_mesh_file)
        error('trkg4:missingPrebuiltMesh', ...
            ['cfg.prebuilt_mesh_file is set but does not exist:\n  %s\n', ...
             'Rebuild it with tools/gmsh_stl_to_msh_robust.py, or clear ', ...
             'cfg.prebuilt_mesh_file to mesh from the STL at ', ...
             'cfg.gmsh_mesh_size = %g mm.'], ...
            cfg.prebuilt_mesh_file, cfg.gmsh_mesh_size);
    end
    if use_prebuilt
        fprintf('Loading verified prebuilt Gmsh mesh: %s\n', ...
            cfg.prebuilt_mesh_file);
        fmdl = gmsh_mk_fwd_model(cfg.prebuilt_mesh_file, [], [], []);
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
[requested_centres_xyz, electrode_labels] = trkg4_load_electrode_centres(cfg);

% Snapping moves the centre onto a boundary node before the patch is grown.
% The requested coordinates are kept separately: they are the only input that
% can still reveal a marker placed off the real skin, in a cavity or in a
% leftover hole, and the diagnostics measure that distance on them. Measuring
% it on the snapped point would return zero by construction.
if cfg.snap_electrode_centres_to_nodes
    electrode_centres_xyz = trkg4_snap_electrode_centres( ...
        fmdl, requested_centres_xyz);
else
    electrode_centres_xyz = requested_centres_xyz;
end
fmdl.electrode = electrode_faces_by_area( ...
    fmdl, electrode_centres_xyz, cfg.electrode_area, cfg.z_contact);
electrode_diagnostics = trkg4_electrode_diagnostics( ...
    fmdl, electrode_centres_xyz, electrode_labels, cfg, requested_centres_xyz);
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
