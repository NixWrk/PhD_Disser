function results = run_trkg4_electrode_systems()
%RUN_TRKG4_ELECTRODE_SYSTEMS Compare accepted right-rib electrode systems.
% The volume mesh and tissue assignment are built once and shared by every
% electrode configuration, so differences in Z are caused by placement only.

base_cfg = trkg4_config('nik');
root = base_cfg.project_root;

systems = struct('id', {}, 'display_name', {}, 'electrodes_file', {});
systems(end + 1) = local_system('right_rib_050mm', 'Right rib, 50 mm span', ...
    fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_050mm_xyz_mm.csv'));
systems(end + 1) = local_system('right_rib_140mm', 'Right rib, 140 mm span', ...
    fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_140mm_xyz_mm.csv'));

base_cfg.electrode_centres_file = systems(1).electrodes_file;
base_cfg.output_file = local_output_file(root, systems(1).id, '.mat');
base_cfg.electrode_diagnostics_file = local_output_file(root, systems(1).id, ...
    '_electrode_diagnostics.csv');
validate_stl_inputs(base_cfg);

total_timer = tic;
[fmdl_template_mm, first_centres, shared_timings, first_diagnostics] = ...
    trkg4_build_eidors_model_from_stl(base_cfg);

assignment_timer = tic;
if base_cfg.use_parallel_assignment
    start_parallel_pool(base_cfg.parallel_workers);
end
centroids_mm = element_centroids(fmdl_template_mm);
[elem_sigma, tissue_id, tissue_names, tissue_counts] = ...
    assign_tissues_from_stl(centroids_mm, base_cfg);
tissue_volumes_l = local_tissue_volumes_l( ...
    fmdl_template_mm, tissue_id, numel(tissue_names));
shared_timings.assignment_seconds = toc(assignment_timer);

% Reused by every system after the first; see the patch build below.
patch_context = [];

% Index of the lungs volume in tissue_volumes_l. Resolved by name rather than
% hardcoded to 2: assign_tissues_from_stl appends only the tissues whose STL
% was actually found, so a missing mask shifts every later index.
lungs_volume_index = find(strcmp(tissue_names, 'lungs'), 1);
if isempty(lungs_volume_index)
    error('trkg4:noLungTissue', ...
        'Lungs were not assigned; the electrode-system comparison is meaningless.');
end

results = cell(numel(systems), 1);
summary_rows = cell(numel(systems), 14);
for k = 1:numel(systems)
    cfg = trkg4_config('nik');
    cfg.electrode_system_id = systems(k).id;
    cfg.electrode_system_name = systems(k).display_name;
    cfg.electrode_centres_file = systems(k).electrodes_file;
    cfg.output_file = local_output_file(root, systems(k).id, '.mat');
    cfg.electrode_diagnostics_file = local_output_file(root, systems(k).id, ...
        '_electrode_diagnostics.csv');

    fmdl_mm = fmdl_template_mm;
    electrode_timer = tic;
    if k == 1
        electrode_centres_xyz_mm = first_centres;
        electrode_diagnostics = first_diagnostics;
    else
        [requested_centres_mm, electrode_labels] = ...
            trkg4_load_electrode_centres(cfg);
        if cfg.snap_electrode_centres_to_nodes
            electrode_centres_xyz_mm = trkg4_snap_electrode_centres( ...
                fmdl_mm, requested_centres_mm);
        else
            electrode_centres_xyz_mm = requested_centres_mm;
        end
        % The mesh is identical for every system, so the patch context is
        % built once and reused; only the centres change between systems.
        [fmdl_mm.electrode, ~, patch_context] = electrode_faces_by_area( ...
            fmdl_mm, electrode_centres_xyz_mm, cfg.electrode_area, ...
            cfg.z_contact, patch_context);
        electrode_diagnostics = trkg4_electrode_diagnostics(fmdl_mm, ...
            electrode_centres_xyz_mm, electrode_labels, cfg, ...
            requested_centres_mm);
    end
    [fmdl_mm.stimulation, fmdl_mm.meas_select] = ...
        trkg4_make_4electrode_stimulation(cfg);
    electrode_seconds = toc(electrode_timer);

    fmdl = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);
    img = mk_image(fmdl, 1);
    img.elem_data = elem_sigma;
    img.name = sprintf('%s TRKG4 %s conductivity', ...
        cfg.subject.id, systems(k).id);

    solve_timer = tic;
    v = fwd_solve(img);
    solve_seconds = toc(solve_timer);
    Z_model_ohm = v.meas(1) / cfg.current_ampere;

    timings = shared_timings;
    timings.electrode_seconds = electrode_seconds;
    timings.solve_seconds = solve_seconds;
    timings.total_shared_run_seconds = toc(total_timer);

    result = struct();
    result.cfg = cfg;
    result.electrode_system = systems(k);
    result.fmdl = fmdl;
    result.fmdl_mesh_units = fmdl_mm;
    result.img = img;
    result.v = v;
    result.Z_model_ohm = Z_model_ohm;
    result.elem_sigma = elem_sigma;
    result.tissue_id = tissue_id;
    result.tissue_names = tissue_names;
    result.tissue_counts = tissue_counts;
    result.tissue_volumes_l = tissue_volumes_l;
    result.electrode_centres_xyz_mm = electrode_centres_xyz_mm;
    result.electrode_diagnostics = electrode_diagnostics;
    result.timings = timings;
    save(cfg.output_file, '-struct', 'result', '-v7.3');

    results{k} = result;
    summary_rows(k, :) = {string(systems(k).id), string(systems(k).display_name), ...
        real(Z_model_ohm), imag(Z_model_ohm), abs(Z_model_ohm), ...
        angle(Z_model_ohm) * 180 / pi, cfg.frequency_hz, ...
        cfg.rho_cloud.soft_ohm_m, cfg.rho_cloud.lungs_ohm_m, ...
        size(fmdl.nodes, 1), size(fmdl.elems, 1), ...
        solve_seconds, tissue_volumes_l(lungs_volume_index), ...
        string(fullfile('output', sprintf( ...
        'nik_trkg4_%s_full_v5_1mm_local2mm.mat', systems(k).id)))};
    fprintf('\n%s: Z = %.9g %+.9gi Ohm\n', systems(k).display_name, ...
        real(Z_model_ohm), imag(Z_model_ohm));
end

summary = cell2table(summary_rows, 'VariableNames', { ...
    'system_id', 'system_name', 'Z_real_ohm', 'Z_imag_ohm', ...
    'Z_abs_ohm', 'Z_phase_deg', 'frequency_hz', ...
    'rho_soft_ohm_m', 'rho_lungs_ohm_m', 'nodes', 'tetrahedra', ...
    'solve_seconds', 'lungs_volume_l', 'output_mat'});
summary_file = fullfile(root, 'output', ...
    'nik_trkg4_electrode_systems_v5_1mm_local2mm_summary.csv');
writetable(summary, summary_file);
fprintf('\nSaved comparison summary: %s\n', summary_file);
end

function system = local_system(id, display_name, electrodes_file)
system = struct('id', id, 'display_name', display_name, ...
    'electrodes_file', electrodes_file);
end

function filename = local_output_file(root, system_id, suffix)
filename = fullfile(root, 'output', ...
    sprintf('nik_trkg4_%s_full_v5_1mm_local2mm%s', system_id, suffix));
end

function volumes_l = local_tissue_volumes_l(fmdl_mm, tissue_id, n_tissues)
nodes = fmdl_mm.nodes;
elems = fmdl_mm.elems;
a = nodes(elems(:, 2), :) - nodes(elems(:, 1), :);
b = nodes(elems(:, 3), :) - nodes(elems(:, 1), :);
c = nodes(elems(:, 4), :) - nodes(elems(:, 1), :);
elem_volume_mm3 = abs(dot(a, cross(b, c, 2), 2)) / 6;
volumes_l = accumarray(tissue_id(:), elem_volume_mm3, ...
    [n_tissues, 1], @sum, 0) / 1e6;
end
