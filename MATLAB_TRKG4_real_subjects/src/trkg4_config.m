function cfg = trkg4_config(subject_id)
%TRKG4_CONFIG Configuration for 4-electrode real-subject TRKG modeling.

if nargin < 1 || isempty(subject_id)
    subject_id = 'nik';
end

cfg = struct();
cfg.project_root = fileparts(fileparts(mfilename('fullpath')));
cfg.src_dir = fullfile(cfg.project_root, 'src');
cfg.vendor_dir = fullfile(cfg.project_root, 'vendor_stl_eidors');
addpath(cfg.src_dir);
addpath(cfg.vendor_dir);

subjects = trkg4_subject_registry(cfg.project_root);
cfg.subject = trkg4_get_subject(subjects, subject_id);

cfg.input_mode = 'real_ct_stl_trkg4';
cfg.frequency_hz = 50e3;
cfg.itis_database = 'ITIS Tissue Properties Database V5.0';
cfg.eidors_startup = '';

% CT STL files are expected in millimetres. Tissue classification and
% electrode patch selection use mm; the FEM solve is scaled to metres.
cfg.length_unit = 'mm';
cfg.scale_nodes_to_si = true;

cfg.mesher = 'gmsh';       % gmsh or netgen
cfg.gmsh_mesh_size = 25;   % in STL units, normally mm
cfg.mesh_fineness = 'moderate';
cfg.show_figures = false;
cfg.use_parallel_assignment = true;
cfg.parallel_workers = 'auto';
cfg.reuse_mesh_cache = true;

cfg.n_electrodes = 4;
cfg.electrode_order = {'I_plus', 'V_plus', 'V_minus', 'I_minus'};
cfg.current_ampere = 1;
cfg.electrode_area_mm2 = pi * 10^2 / 4 * 0.9;
cfg.electrode_area = cfg.electrode_area_mm2;
cfg.z_contact_ohm_m2 = 2.5e-7 * 0.05 / (pi * 0.01^2 / 4);
cfg.z_contact = cfg.z_contact_ohm_m2;
cfg.electrode_centres_xyz = [];
cfg.electrode_centres_file = cfg.subject.electrodes_csv;
cfg.max_electrode_surface_distance_mm = 2 * cfg.gmsh_mesh_size;
cfg.max_electrode_patch_centroid_offset_mm = 2 * cfg.gmsh_mesh_size;
cfg.fail_on_electrode_diagnostics = false;
cfg.electrode_diagnostics_file = fullfile(cfg.project_root, 'output', ...
    sprintf('%s_electrode_diagnostics.csv', cfg.subject.id));

cfg.body_stl = cfg.subject.stl.body;
cfg.output_file = fullfile(cfg.project_root, 'output', ...
    sprintf('%s_trkg4_forward.mat', cfg.subject.id));
cfg.mesh_cache_file = fullfile(cfg.project_root, 'output', ...
    sprintf('%s_body_mesh_cache.mat', cfg.subject.id));

% Replace these by selected points from the solution clouds for each
% subject/state. Defaults are the current Nik investigation values.
cfg.rho_cloud.soft_ohm_m = 4.728174786010649;
cfg.rho_cloud.lungs_ohm_m = 17.4067;
cfg.rho_cloud.lungs_inhale_ohm_m = 17.73503700425868;
cfg.rho_cloud.source = 'Nik solution clouds / COMSOL investigation notes';

sigma_soft = 1 / cfg.rho_cloud.soft_ohm_m;
sigma_lung = 1 / cfg.rho_cloud.lungs_ohm_m;
sigma_heart = itis_conductivity('Heart Muscle', cfg.frequency_hz);
sigma_bone = itis_conductivity('Bone (Cortical)', cfg.frequency_hz);
sigma_blood = itis_conductivity('Blood', cfg.frequency_hz);

cfg.background.name = 'soft_tissue';
cfg.background.itis_name = 'own_solution_cloud';
cfg.background.sigma = sigma_soft;

cfg.required_tissues = {'lungs', 'heart', 'bones'};
cfg.tissues = struct('name', {}, 'file', {}, 'itis_name', {}, 'sigma', {}, 'enabled', {});
cfg.tissues(end + 1) = local_tissue_entry('lungs', cfg.subject.stl.lungs, ...
    'own_solution_cloud', sigma_lung, true);
cfg.tissues(end + 1) = local_tissue_entry('heart', cfg.subject.stl.heart, ...
    'Heart Muscle', sigma_heart, true);
cfg.tissues(end + 1) = local_tissue_entry('bones', cfg.subject.stl.bones, ...
    'Bone (Cortical)', sigma_bone, true);
cfg.tissues(end + 1) = local_tissue_entry('blood', cfg.subject.stl.blood, ...
    'Blood', sigma_blood, true);
end

function tissue = local_tissue_entry(name, file, itis_name, sigma, enabled)
tissue = struct( ...
    'name', name, ...
    'file', file, ...
    'itis_name', itis_name, ...
    'sigma', sigma, ...
    'enabled', enabled);
end
