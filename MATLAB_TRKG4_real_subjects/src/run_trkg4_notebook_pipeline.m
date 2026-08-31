function report = run_trkg4_notebook_pipeline(varargin)
%RUN_TRKG4_NOTEBOOK_PIPELINE Produce the MATLAB artefacts used by ipynb reports.
%
% The function is deliberately explicit about dependencies between stages.
% It does not silently accept stale files unless ReuseExistingResults=true.

p = inputParser;
addParameter(p, 'Profile', 'inverse_report', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'SubjectId', 'nik', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'ReuseExistingResults', false, ...
    @(x) islogical(x) && isscalar(x));
addParameter(p, 'MaxInverseEvaluations', 30, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'ComputeJacobian', true, ...
    @(x) islogical(x) && isscalar(x));
addParameter(p, 'ArmPrebuiltMeshFile', '', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'ArmMeshTargetSizeMm', [], ...
    @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
parse(p, varargin{:});
opt = p.Results;

profile = lower(string(opt.Profile));
subject_id = lower(string(opt.SubjectId));
if subject_id ~= "nik"
    error('trkg4:pipelineSubjectNotImplemented', ...
        ['The inverse/PEM/CEM notebook pipeline is currently implemented ', ...
         'only for subject "nik". Add subject-specific experimental and ', ...
         'electrode inputs before selecting another subject.']);
end

cfg = trkg4_config(char(subject_id));
root = cfg.project_root;
out_dir = fullfile(root, 'output');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

stages = local_profile_stages(profile);
rows = repmat(local_empty_stage(), numel(stages), 1);
pipeline_timer = tic;
pipeline_started = local_utc_now();

for k = 1:numel(stages)
    stage = stages(k);
    expected = local_expected_outputs(root, char(subject_id), stage, ...
        opt.ComputeJacobian);
    rows(k).stage = stage;
    rows(k).producer = local_stage_producer(stage);
    rows(k).expected_outputs = strjoin(local_relative_paths(expected, root), '; ');
    rows(k).started_utc = local_utc_now();
    stage_timer = tic;

    try
        if opt.ReuseExistingResults && ~isempty(expected) && all(isfile(expected))
            rows(k).status = "reused_existing";
            rows(k).message = "All declared outputs already exist.";
        else
            local_run_stage(stage, root, cfg, opt);
            missing = expected(~isfile(expected));
            if ~isempty(missing)
                error('trkg4:pipelineMissingOutput', ...
                    'Stage %s finished but did not produce: %s', ...
                    stage, strjoin(missing, ', '));
            end
            rows(k).status = "completed";
            rows(k).message = "";
        end
    catch exception
        rows(k).status = "failed";
        rows(k).message = string(exception.message);
        rows(k).duration_seconds = toc(stage_timer);
        report = local_report(profile, subject_id, pipeline_started, ...
            toc(pipeline_timer), rows(1:k));
        local_write_report(report, out_dir);
        rethrow(exception);
    end
    rows(k).duration_seconds = toc(stage_timer);
end

report = local_report(profile, subject_id, pipeline_started, ...
    toc(pipeline_timer), rows);
local_write_report(report, out_dir);
fprintf('Pipeline %s completed in %.1f s.\n', profile, report.total_seconds);
end

function stages = local_profile_stages(profile)
switch profile
    case "preflight"
        stages = ["preflight", "tests"];
    case "inverse_report"
        stages = ["preflight", "tests", "electrode_systems", ...
            "inverse", "pem_scan", "cem_refine", "patch_overlap"];
    case "arm_electrode_report"
        stages = ["preflight", "tests", "arm_electrodes"];
    case "all"
        stages = ["preflight", "tests", "electrode_systems", ...
            "inverse", "pem_scan", "cem_refine", "patch_overlap", ...
            "arm_electrodes"];
    otherwise
        error('trkg4:unknownPipelineProfile', ...
            'Unknown profile "%s". Use preflight, inverse_report, arm_electrode_report, or all.', ...
            profile);
end
end

function local_run_stage(stage, root, cfg, opt)
switch stage
    case "preflight"
        trkg4_prepare_runtime(cfg);
        validate_stl_inputs(cfg);
        if ~isfile(cfg.prebuilt_mesh_file)
            error('trkg4:missingAcceptedMesh', ...
                'Accepted prebuilt mesh is missing: %s', cfg.prebuilt_mesh_file);
        end
    case "tests"
        test_results = runtests(fullfile(root, 'tests'));
        assertSuccess(test_results);
    case "electrode_systems"
        run_trkg4_electrode_systems();
    case "inverse"
        run_trkg4_inverse_inhale(opt.MaxInverseEvaluations, [], ...
            opt.ComputeJacobian);
    case "pem_scan"
        run_trkg4_fast_pem_scan();
    case "cem_refine"
        run_trkg4_fast_cem_refine();
    case "patch_overlap"
        check_trkg4_patch_overlap_v5();
    case "arm_electrodes"
        mesh_file = string(opt.ArmPrebuiltMeshFile);
        if strlength(mesh_file) == 0
            error('trkg4:armMeshMustBeExplicit', ...
                ['The historical accepted mesh does not support arbitrary ', ...
                 'longitudinal arm-electrode positions. Supply an independently ', ...
                 'QC-accepted mesh through ArmPrebuiltMeshFile.']);
        end
        if ~isfile(mesh_file)
            project_relative = fullfile(root, char(mesh_file));
            if isfile(project_relative)
                mesh_file = string(project_relative);
            end
        end
        if ~isfile(mesh_file)
            error('trkg4:missingArmMesh', 'Arm mesh is missing: %s', mesh_file);
        end
        args = {'PrebuiltMeshFile', char(mesh_file)};
        if ~isempty(opt.ArmMeshTargetSizeMm)
            args = [args, {'MeshTargetSizeMm', opt.ArmMeshTargetSizeMm}];
        end
        run_trkg4_arm_electrode_tests(args{:});
    otherwise
        error('trkg4:unknownPipelineStage', 'Unknown stage: %s', stage);
end
end

function files = local_expected_outputs(root, subject_id, stage, compute_jacobian)
out = fullfile(root, 'output');
switch stage
    case {"preflight", "tests"}
        files = strings(0, 1);
    case "electrode_systems"
        files = string(fullfile(out, { ...
            sprintf('%s_trkg4_arms_full_v5_1mm_local2mm.mat', subject_id), ...
            sprintf('%s_trkg4_electrode_systems_v5_1mm_local2mm_summary.csv', subject_id)}));
    case "inverse"
        stem = sprintf('%s_trkg4_inverse_inhale_', subject_id);
        tag = 'v5_1mm_local2mm';
        names = { ...
            [stem tag '.mat'], [stem 'fit_summary_' tag '.csv'], ...
            [stem 'comparison_' tag '.csv'], [stem 'depth_curve_' tag '.csv'], ...
            [stem 'electrodes_' tag '.csv'], [stem 'grid_geometry_' tag '.csv'], ...
            [stem 'jacobian_' tag '.csv'], ...
            [stem 'history_' tag '.csv']};
        if compute_jacobian
            names{end + 1} = [stem 'parameter_sensitivity_' tag '.csv'];
        end
        files = string(fullfile(out, names));
    case "pem_scan"
        files = string(fullfile(out, { ...
            sprintf('%s_trkg4_fast_pem_scan_all.csv', subject_id), ...
            sprintf('%s_trkg4_fast_pem_scan_top.csv', subject_id), ...
            sprintf('%s_trkg4_fast_pem_validation.csv', subject_id), ...
            sprintf('%s_trkg4_fast_pem_scan_summary.csv', subject_id)}));
    case "cem_refine"
        files = string(fullfile(out, { ...
            sprintf('%s_trkg4_fast_cem_refine_v5_1mm_local2mm.csv', subject_id), ...
            sprintf('%s_trkg4_fast_cem_refine_summary_v5_1mm_local2mm.csv', subject_id)}));
    case "patch_overlap"
        files = string(fullfile(out, ...
            sprintf('%s_trkg4_patch_overlap_v5_1mm_local2mm.csv', subject_id)));
    case "arm_electrodes"
        files = string(fullfile(out, { ...
            sprintf('%s_trkg4_arm_electrode_model_tests_summary.csv', subject_id), ...
            sprintf('%s_trkg4_arm_electrode_model_tests_patches.csv', subject_id), ...
            sprintf('%s_trkg4_arm_electrode_model_tests_contract.json', subject_id), ...
            sprintf('%s_trkg4_arm_electrode_model_tests.mat', subject_id), ...
            sprintf('%s_trkg4_arm_electrode_model_tests.png', subject_id)}));
    otherwise
        files = strings(0, 1);
end
files = files(:);
end

function producer = local_stage_producer(stage)
switch stage
    case "preflight", producer = "trkg4_prepare_runtime + validate_stl_inputs";
    case "tests", producer = "runtests(tests)";
    case "electrode_systems", producer = "run_trkg4_electrode_systems";
    case "inverse", producer = "run_trkg4_inverse_inhale";
    case "pem_scan", producer = "run_trkg4_fast_pem_scan";
    case "cem_refine", producer = "run_trkg4_fast_cem_refine";
    case "patch_overlap", producer = "check_trkg4_patch_overlap_v5";
    case "arm_electrodes", producer = "run_trkg4_arm_electrode_tests";
end
end

function relative = local_relative_paths(files, root)
relative = string(files);
prefix = string([root filesep]);
for k = 1:numel(relative)
    relative(k) = erase(relative(k), prefix);
end
end

function row = local_empty_stage()
row = struct('stage', "", 'producer', "", 'status', "pending", ...
    'started_utc', "", 'duration_seconds', NaN, ...
    'expected_outputs', "", 'message', "");
end

function report = local_report(profile, subject_id, started, seconds, rows)
report = struct();
report.contract_version = 1;
report.status = "method_development_not_validated";
report.profile = profile;
report.subject_id = subject_id;
report.started_utc = started;
report.finished_utc = local_utc_now();
report.total_seconds = seconds;
report.stages = struct2table(rows);
end

function local_write_report(report, out_dir)
stem = sprintf('%s_matlab_pipeline_%s_manifest', ...
    report.subject_id, report.profile);
csv_file = fullfile(out_dir, [stem '.csv']);
json_file = fullfile(out_dir, [stem '.json']);
writetable(report.stages, csv_file);
payload = report;
payload.stages = table2struct(report.stages);
fid = fopen(json_file, 'w');
if fid < 0
    error('trkg4:pipelineManifestWrite', ...
        'Cannot write pipeline manifest: %s', json_file);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(payload), 'char');
end

function value = local_utc_now()
value = string(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end
