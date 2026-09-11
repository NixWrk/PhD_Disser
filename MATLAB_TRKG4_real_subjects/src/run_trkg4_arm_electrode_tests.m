function summary = run_trkg4_arm_electrode_tests(varargin)
%RUN_TRKG4_ARM_ELECTRODE_TESTS Compare arm electrode geometry and reciprocity.
%
% The mesh and tissue assignment are built once. Every scenario changes only
% the electrode position/type. Two transfer measurements are solved together:
% direct I+/I- -> V+/V- and reciprocal V+/V- -> I+/I-.
%
% Example:
%   run_trkg4_arm_electrode_tests( ...
%       'InnerDistancesMm', [20 40], ...
%       'OuterDistancesMm', [80 100]);

p = inputParser;
addParameter(p, 'SubjectId', 'nik', @(x) ischar(x) || isstring(x));
addParameter(p, 'InnerDistancesMm', [20 40], @local_positive_vector);
addParameter(p, 'OuterDistancesMm', [80 100], @local_positive_vector);
addParameter(p, 'ElectrodeKinds', ["point_disc_5mm", ...
    "circumferential_ring", "wide_cuff_equivalent_area", ...
    "outer_planes_inner_rings"]);
addParameter(p, 'DiscDiameterMm', 5, @local_positive_scalar);
addParameter(p, 'RingWidthMm', 5, @local_positive_scalar);
addParameter(p, 'PrebuiltMeshFile', '', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'OutputSubdirectory', 'output', ...
    @(x) (ischar(x) || isstring(x)) && strlength(string(x)) > 0);
addParameter(p, 'MeshTargetSizeMm', [], ...
    @(x) isempty(x) || local_positive_scalar(x));
addParameter(p, 'ReciprocityRelativeTolerance', 1e-8, @local_positive_scalar);
addParameter(p, 'ReciprocityAbsoluteToleranceOhm', 1e-10, @local_positive_scalar);
addParameter(p, 'SimilarityFactorLimit', 2, ...
    @(x) local_positive_scalar(x) && x > 1);
parse(p, varargin{:});
opt = p.Results;

cfg = trkg4_config(char(opt.SubjectId));
if strlength(string(opt.PrebuiltMeshFile)) > 0
    cfg.prebuilt_mesh_file = char(opt.PrebuiltMeshFile);
end
if ~isempty(opt.MeshTargetSizeMm)
    cfg.mesh_target_size_mm = opt.MeshTargetSizeMm;
end
root = cfg.project_root;
inner_values = opt.InnerDistancesMm(:)';
outer_values = opt.OuterDistancesMm(:)';
kinds = string(opt.ElectrodeKinds(:)');

first_spec = trkg4_arm_montage_spec(cfg, "point_disc_5mm", ...
    inner_values(1), outer_values(1), ...
    'DiscDiameterMm', opt.DiscDiameterMm, ...
    'RingWidthMm', opt.RingWidthMm);

% Bootstrap the verified mesh loader with a safe generated montage instead
% of the rejected historical absolute-coordinate CSV.
cfg.electrode_centres_xyz = first_spec.centres_xyz_mm;
cfg.electrode_centres_file = '';
cfg.electrode_area_mm2 = first_spec.nominal_area_mm2(1);
cfg.electrode_area = cfg.electrode_area_mm2;
cfg.electrode_patch_radius_mm = sqrt(cfg.electrode_area_mm2 / pi);
cfg.max_electrode_patch_centroid_offset_mm = ...
    cfg.electrode_patch_radius_mm;
cfg.fail_on_electrode_diagnostics = false;
cfg.electrode_diagnostics_file = '';
validate_stl_inputs(cfg);

total_timer = tic;
[fmdl_template_mm, ~, mesh_timings] = ...
    trkg4_build_eidors_model_from_stl(cfg);

assignment_timer = tic;
if cfg.use_parallel_assignment
    start_parallel_pool(cfg.parallel_workers);
end
centroids_mm = element_centroids(fmdl_template_mm);
[elem_sigma, tissue_id, tissue_names, tissue_counts] = ...
    assign_tissues_from_stl(centroids_mm, cfg);
assignment_seconds = toc(assignment_timer);

n_cases = numel(inner_values) * numel(outer_values) * numel(kinds);
rows = repmat(local_empty_row(), n_cases, 1);
diagnostics_all = table();
case_index = 0;

for inner_mm = inner_values
    for outer_mm = outer_values
        for kind = kinds
            case_index = case_index + 1;
            spec = trkg4_arm_montage_spec(cfg, kind, inner_mm, outer_mm, ...
                'DiscDiameterMm', opt.DiscDiameterMm, ...
                'RingWidthMm', opt.RingWidthMm);

            fmdl_mm = fmdl_template_mm;
            build_timer = tic;
            elem_sigma_case = elem_sigma;
            if spec.kind == "outer_planes_inner_rings"
                [fmdl_mm, patch_diagnostics, keep_elements] = ...
                    trkg4_build_mixed_arm_electrodes( ...
                    fmdl_mm, spec, cfg.z_contact, cfg);
                elem_sigma_case = elem_sigma(keep_elements);
            else
                [fmdl_mm.electrode, patch_diagnostics] = ...
                    trkg4_build_arm_electrodes( ...
                    fmdl_mm, spec, cfg.z_contact);
            end
            if ~ismember('contact_geometry', patch_diagnostics.Properties.VariableNames)
                patch_diagnostics.contact_geometry = repmat(spec.kind, height(patch_diagnostics), 1);
            end
            [fmdl_mm.stimulation, fmdl_mm.meas_select] = ...
                trkg4_make_reciprocity_stimulation(cfg);
            build_seconds = toc(build_timer);

            fmdl = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);
            img = mk_image(fmdl, 1);
            img.elem_data = elem_sigma_case;
            img.name = sprintf('%s arm-electrode test %s', ...
                cfg.subject.id, spec.kind);

            solve_timer = tic;
            lastwarn('');
            voltage = fwd_solve(img);
            [solver_warning_message, solver_warning_id] = lastwarn;
            solver_warning_observed = ~isempty(solver_warning_message);
            solve_seconds = toc(solve_timer);
            transfer = voltage.meas(:) / cfg.current_ampere;
            if numel(transfer) ~= 2
                error('trkg4:unexpectedReciprocityOutput', ...
                    'Expected two transfer values, received %d.', ...
                    numel(transfer));
            end

            z_direct = transfer(1);
            z_reciprocal = transfer(2);
            absolute_error = abs(z_direct - z_reciprocal);
            comparison_scale = max([abs(z_direct), abs(z_reciprocal), eps]);
            relative_error = absolute_error / comparison_scale;
            tolerance = opt.ReciprocityAbsoluteToleranceOhm + ...
                opt.ReciprocityRelativeTolerance * comparison_scale;

            scenario_id = sprintf('in%g_out%g_%s', ...
                inner_mm, outer_mm, spec.kind);
            rows(case_index) = local_result_row(scenario_id, spec, ...
                z_direct, z_reciprocal, absolute_error, relative_error, ...
                absolute_error <= tolerance, solver_warning_observed, ...
                string(solver_warning_id), build_seconds, solve_seconds, cfg);

            patch_diagnostics.scenario_id = repmat(string(scenario_id), ...
                height(patch_diagnostics), 1);
            patch_diagnostics.inner_from_trunk_mm = repmat(inner_mm, ...
                height(patch_diagnostics), 1);
            patch_diagnostics.outer_from_inner_mm = repmat(outer_mm, ...
                height(patch_diagnostics), 1);
            diagnostics_all = [diagnostics_all; patch_diagnostics]; %#ok<AGROW>

            fprintf(['%s: direct %.9g %+.9gi Ohm, reciprocal ', ...
                '%.9g %+.9gi Ohm, relative error %.3g\n'], ...
                scenario_id, real(z_direct), imag(z_direct), ...
                real(z_reciprocal), imag(z_reciprocal), relative_error);
        end
    end
end

summary = struct2table(rows);
summary = local_add_geometry_comparison(summary, opt.SimilarityFactorLimit);

out_dir = fullfile(root, char(opt.OutputSubdirectory));
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
summary_file = fullfile(out_dir, ...
    sprintf('%s_trkg4_arm_electrode_model_tests_summary.csv', cfg.subject.id));
diagnostics_file = fullfile(out_dir, ...
    sprintf('%s_trkg4_arm_electrode_model_tests_patches.csv', cfg.subject.id));
mat_file = fullfile(out_dir, ...
    sprintf('%s_trkg4_arm_electrode_model_tests.mat', cfg.subject.id));
figure_file = fullfile(out_dir, ...
    sprintf('%s_trkg4_arm_electrode_model_tests.png', cfg.subject.id));
contract_file = fullfile(out_dir, ...
    sprintf('%s_trkg4_arm_electrode_model_tests_contract.json', cfg.subject.id));

writetable(summary, summary_file);
writetable(diagnostics_all, diagnostics_file);

timings = struct('mesh', mesh_timings, ...
    'assignment_seconds', assignment_seconds, ...
    'total_seconds', toc(total_timer));
save(mat_file, 'summary', 'diagnostics_all', 'timings', ...
    'tissue_id', 'tissue_names', 'tissue_counts', '-v7.3');
local_write_contract(contract_file, cfg, opt, summary, timings, ...
    tissue_names, tissue_counts, diagnostics_all);
local_save_figure(summary, figure_file, opt);

fprintf('\nSaved summary: %s\n', summary_file);
fprintf('Saved patch diagnostics: %s\n', diagnostics_file);
fprintf('Saved contract: %s\n', contract_file);
fprintf('Saved figure: %s\n', figure_file);
end

function row = local_empty_row()
row = struct( ...
    'scenario_id', "", ...
    'subject_id', "", ...
    'electrode_kind', "", ...
    'inner_from_trunk_mm', NaN, ...
    'outer_from_inner_mm', NaN, ...
    'Z_direct_real_ohm', NaN, ...
    'Z_direct_imag_ohm', NaN, ...
    'Z_direct_abs_ohm', NaN, ...
    'Z_reciprocal_real_ohm', NaN, ...
    'Z_reciprocal_imag_ohm', NaN, ...
    'Z_reciprocal_abs_ohm', NaN, ...
    'reciprocity_absolute_error_ohm', NaN, ...
    'reciprocity_relative_error', NaN, ...
    'reciprocity_pass', false, ...
    'solver_warning_observed', false, ...
    'solver_warning_id', "", ...
    'position_clearance_pass', false, ...
    'nominal_contact_area_mean_mm2', NaN, ...
    'frequency_hz', NaN, ...
    'contact_impedance_ohm_m2', NaN, ...
    'build_seconds', NaN, ...
    'solve_seconds', NaN);
end

function row = local_result_row(scenario_id, spec, z_direct, z_reciprocal, ...
    absolute_error, relative_error, reciprocity_pass, ...
    solver_warning_observed, solver_warning_id, build_seconds, ...
    solve_seconds, cfg)
row = local_empty_row();
row.scenario_id = string(scenario_id);
row.subject_id = string(cfg.subject.id);
row.electrode_kind = spec.kind;
row.inner_from_trunk_mm = spec.inner_from_trunk_mm;
row.outer_from_inner_mm = spec.outer_from_inner_mm;
row.Z_direct_real_ohm = real(z_direct);
row.Z_direct_imag_ohm = imag(z_direct);
row.Z_direct_abs_ohm = abs(z_direct);
row.Z_reciprocal_real_ohm = real(z_reciprocal);
row.Z_reciprocal_imag_ohm = imag(z_reciprocal);
row.Z_reciprocal_abs_ohm = abs(z_reciprocal);
row.reciprocity_absolute_error_ohm = absolute_error;
row.reciprocity_relative_error = relative_error;
row.reciprocity_pass = reciprocity_pass;
row.solver_warning_observed = solver_warning_observed;
row.solver_warning_id = solver_warning_id;
row.position_clearance_pass = spec.pass_clearance;
row.nominal_contact_area_mean_mm2 = mean(spec.nominal_area_mm2);
row.frequency_hz = cfg.frequency_hz;
row.contact_impedance_ohm_m2 = cfg.z_contact;
row.build_seconds = build_seconds;
row.solve_seconds = solve_seconds;
end

function summary = local_add_geometry_comparison(summary, factor_limit)
n = height(summary);
ratio = nan(n, 1);
relative_difference = nan(n, 1);
same_sign = false(n, 1);
within_factor = false(n, 1);

for k = 1:n
    same_position = summary.inner_from_trunk_mm == ...
        summary.inner_from_trunk_mm(k) & ...
        summary.outer_from_inner_mm == summary.outer_from_inner_mm(k);
    reference = find(same_position & ...
        summary.electrode_kind == "point_disc_5mm", 1);
    if isempty(reference) || summary.Z_direct_abs_ohm(reference) <= eps
        continue;
    end

    ratio(k) = summary.Z_direct_abs_ohm(k) / ...
        summary.Z_direct_abs_ohm(reference);
    z_k = complex(summary.Z_direct_real_ohm(k), ...
        summary.Z_direct_imag_ohm(k));
    z_ref = complex(summary.Z_direct_real_ohm(reference), ...
        summary.Z_direct_imag_ohm(reference));
    relative_difference(k) = abs(z_k - z_ref) / abs(z_ref);
    same_sign(k) = sign(real(z_k)) == sign(real(z_ref));
    within_factor(k) = ratio(k) >= 1 / factor_limit && ...
        ratio(k) <= factor_limit;
end

summary.magnitude_ratio_to_point_disc = ratio;
summary.relative_difference_to_point_disc = relative_difference;
summary.same_real_sign_as_point_disc = same_sign;
summary.within_screening_factor_of_point_disc = within_factor;
summary.similarity_factor_limit = repmat(factor_limit, n, 1);
end

function local_write_contract(filename, cfg, opt, summary, timings, ...
    tissue_names, tissue_counts, diagnostics_all)
contract = struct();
contract.status = 'model_test_not_experimental_validation';
contract.subject_id = cfg.subject.id;
contract.geometry_reference = 'CT/artificial-arm join plane';
contract.electrode_kinds = cellstr(unique(summary.electrode_kind));
contract.inner_distances_mm = opt.InnerDistancesMm;
contract.outer_distances_mm = opt.OuterDistancesMm;
contract.reciprocity_relative_tolerance = ...
    opt.ReciprocityRelativeTolerance;
contract.reciprocity_absolute_tolerance_ohm = ...
    opt.ReciprocityAbsoluteToleranceOhm;
contract.similarity_factor_limit = opt.SimilarityFactorLimit;
contract.similarity_is_screening_not_theorem = true;
contract.point_model = sprintf('surface disc, diameter %.6g mm', ...
    opt.DiscDiameterMm);
contract.ring_model = sprintf([ ...
    'circumferential cuff, inner diameter equals local arm diameter, ', ...
    'axial width %.6g mm'], opt.RingWidthMm);
wide_spec = trkg4_arm_montage_spec(cfg, "wide_cuff_equivalent_area", ...
    opt.InnerDistancesMm(1), opt.OuterDistancesMm(1), ...
    'DiscDiameterMm', opt.DiscDiameterMm, ...
    'RingWidthMm', opt.RingWidthMm);
contract.wide_cuff_model = [ ...
    'circumferential cuff whose surface contact area equals ', ...
    'pi*R^2; not an internal cross-section electrode'];
contract.wide_cuff_nominal_width_mm = ...
    unique(wide_spec.axial_width_mm)';
contract.mixed_montage_model = 'outer planes I+/I-; inner circumferential 5 mm cuffs V+/V-';
contract.cross_section_plane_model = [ ...
    'ideal zero-thickness internal cross-sectional surface; ', ...
    'represented by shared mesh faces without removing tissue elements; ', ...
    'mesh-conforming surface is slightly jagged and is not attachable'];
contract.cross_section_plane_nominal_thickness_mm = 0;
plane_rows = diagnostics_all.electrode_kind == "outer_planes_inner_rings" & ...
    ismember(diagnostics_all.label, ["I_plus", "I_minus"]);
if any(plane_rows)
    plane_span_mm = diagnostics_all.patch_x_max_mm(plane_rows) - ...
        diagnostics_all.patch_x_min_mm(plane_rows);
    contract.cross_section_plane_realised_x_span_mm_range = ...
        [min(plane_span_mm), max(plane_span_mm)];
    plane_area_ratio = diagnostics_all.area_ratio(plane_rows);
    contract.cross_section_plane_area_ratio_range = ...
        [min(plane_area_ratio), max(plane_area_ratio)];
    plane_projected_ratio = ...
        diagnostics_all.projected_cross_section_area_ratio(plane_rows);
    contract.cross_section_plane_projected_area_ratio_range = ...
        [min(plane_projected_ratio), max(plane_projected_ratio)];
end
contract.mesh_file = cfg.prebuilt_mesh_file;
contract.frequency_hz = cfg.frequency_hz;
contract.contact_impedance_ohm_m2 = cfg.z_contact;
contract.current_ampere = cfg.current_ampere;
contract.complex_permittivity_used = false;
available_tissues = strings(0, 1);
for tissue_index = 1:numel(cfg.tissues)
    tissue = cfg.tissues(tissue_index);
    if tissue.enabled && isfile(tissue.file)
        available_tissues(end + 1, 1) = string(tissue.name); %#ok<AGROW>
    end
end
if isempty(available_tissues)
    contract.model_domain = [ ...
        'full body volume with soft-tissue background and no separately ', ...
        'assigned tissue STL masks'];
else
    contract.model_domain = sprintf([ ...
        'full body volume with soft-tissue background and separately ', ...
        'assigned available STL masks: %s'], ...
        strjoin(available_tissues, ', '));
end
contract.tissue_assignment_method = [ ...
    'tetrahedron-centroid classification by closed STL masks; ', ...
    'a missing optional mask retains the previous/background class'];
contract.tissues = local_tissue_contract(cfg, tissue_names, tissue_counts);
contract.all_reciprocity_pass = all(summary.reciprocity_pass);
contract.solver_warning_observed = any(summary.solver_warning_observed);
contract.solver_warning_ids = unique( ...
    summary.solver_warning_id(summary.solver_warning_id ~= ""));
contract.solver_warning_requires_numerical_review = ...
    contract.solver_warning_observed;
contract.timings = timings;

fid = fopen(filename, 'w');
if fid < 0
    error('trkg4:contractWriteFailed', ...
        'Cannot open contract file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(contract, 'PrettyPrint', true), 'char');
end

function records = local_tissue_contract(cfg, tissue_names, tissue_counts)
template = struct('name', '', 'sigma_s_per_m', NaN, ...
    'rho_ohm_m', NaN, 'source', '', 'mask_available', false, ...
    'assigned_elements', 0);
records = repmat(template, numel(cfg.tissues) + 1, 1);
records(1).name = cfg.background.name;
records(1).sigma_s_per_m = cfg.background.sigma;
records(1).rho_ohm_m = 1 / cfg.background.sigma;
records(1).source = cfg.rho_cloud.source;
records(1).mask_available = true;
records(1).assigned_elements = local_assigned_count( ...
    cfg.background.name, tissue_names, tissue_counts);
for k = 1:numel(cfg.tissues)
    tissue = cfg.tissues(k);
    records(k + 1).name = tissue.name;
    records(k + 1).sigma_s_per_m = tissue.sigma;
    records(k + 1).rho_ohm_m = 1 / tissue.sigma;
    records(k + 1).source = tissue.itis_name;
    records(k + 1).mask_available = isfile(tissue.file);
    records(k + 1).assigned_elements = local_assigned_count( ...
        tissue.name, tissue_names, tissue_counts);
end
end

function count = local_assigned_count(name, tissue_names, tissue_counts)
index = find(strcmp(tissue_names, name), 1);
if isempty(index)
    count = 0;
else
    count = tissue_counts(index);
end
end

function local_save_figure(summary, filename, opt)
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1500 800]);
cleanup = onCleanup(@() close(fig));

tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(1:height(summary), summary.Z_direct_abs_ohm, 'o-', ...
    'LineWidth', 1.2, 'MarkerSize', 5);
grid on;
ylabel('|Z|, Ohm');
title('Arm-electrode geometry comparison');
xticks(1:height(summary));
xticklabels(summary.scenario_id);
xtickangle(45);

nexttile;
semilogy(1:height(summary), ...
    max(summary.reciprocity_relative_error, realmin), 'o-', ...
    'LineWidth', 1.2, 'MarkerSize', 5);
hold on;
yline(opt.ReciprocityRelativeTolerance, '--r', ...
    'Configured relative tolerance');
grid on;
ylabel('Reciprocity relative error');
xlabel('Scenario');
xticks(1:height(summary));
xticklabels(summary.scenario_id);
xtickangle(45);

exportgraphics(fig, filename, 'Resolution', 180);
end

function ok = local_positive_vector(x)
ok = isnumeric(x) && isvector(x) && ~isempty(x) && ...
    all(isfinite(x)) && all(x > 0);
end

function ok = local_positive_scalar(x)
ok = isnumeric(x) && isscalar(x) && isfinite(x) && x > 0;
end
