
function verify_geometry_probe()
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
cd(root); addpath('src','vendor_stl_eidors');
cfg=trkg4_config('nik'); trkg4_prepare_runtime(cfg);
cfg.electrode_diagnostics_file=''; cfg.electrode_diagnostics_verbose=false;
cfg.fail_on_electrode_diagnostics=true;
shared=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
fmdl=shared.fmdl_mesh_units;
pose=local_build_pose_context(fmdl,local_load_reference_grid(root),cfg.subject.stl.lungs);
fit=readtable(fullfile(root,'output','nik_trkg4_inverse_inhale_fit_summary_disjoint_restartable_manifest90.csv'));
x0=[fit.rho_soft_ohm_m fit.rho_lungs_ohm_m fit.centre_u_mm fit.centre_v_mm fit.phi_deg];
[sizes,~,~]=trkg4_load_nik_inhale_data(root);
% Deterministic geometry screening around the historical candidate. These
% are starting positions, not estimates obtained from experimental fitting.
offsets=[0,-4,4,-8,8,-16,16]; angles=[0,-4,4,-8,8];
rows=cell(0,9); valid_points=zeros(0,5); count=0; timer=tic;
for ai=1:numel(angles)
for ui=1:numel(offsets)
for vi=1:numel(offsets)
    x=x0+[0,0,offsets(ui),offsets(vi),angles(ai)];
    if x(5)<-20 || x(5)>20, continue; end
    state=local_pose_state(pose,x(3),x(4),x(5));
    failed=0; reason=''; max_offset=0; max_ratio=0;
    for k=1:numel(sizes)
        centres=local_grid_centres(pose,state,sizes(k)); model=fmdl;
        try
            [model.electrode, faces]=local_build_electrode_patches( ...
                model,centres,cfg,pose.patch_context,"disjoint");
            diagnostic=trkg4_electrode_diagnostics(model,centres, ...
                string(cfg.electrode_order),cfg,centres,faces);
            max_offset=max(max_offset,max(diagnostic.patch_centroid_offset_mm));
            max_ratio=max(max_ratio,max(diagnostic.patch_area_mm2)/cfg.electrode_area);
        catch exception
            if ~any(string(exception.identifier)==["trkg4:electrodeDiagnostics", ...
                    "electrode_faces_by_area:patchTooSmall","electrode_faces_by_area:noSeed"])
                rethrow(exception);
            end
            failed=failed+1; reason=exception.identifier;
        end
    end
    count=count+1;
    rows(end+1,:)={count,x(3),x(4),x(5),failed,max_offset,max_ratio,state.h_centre_mm,reason};
    if failed==0, valid_points(end+1,:)=x; end
    fprintf('GEOMETRY %03d u=%+.3f v=%+.3f phi=%+.3f failed=%d/%d accepted=%d seconds=%.1f\n', ...
        count,x(3),x(4),x(5),failed,numel(sizes),size(valid_points,1),toc(timer));
    T=cell2table(rows,'VariableNames',{'evaluation','u_mm','v_mm','phi_deg', ...
        'failed_grid_count','max_offset_among_passed_mm','max_area_ratio_among_passed', ...
        'skin_lung_depth_mm','last_failure'});
    writetable(T,fullfile(here,'geometry_probe.csv'));
    if ~isempty(valid_points), writematrix(valid_points,fullfile(here,'admissible_starts.csv')); end
    if size(valid_points,1)>=3, return; end
end
end
end
if isempty(valid_points)
    error('trkg4:noAdmissibleProbePoint','No all-size admissible pose in the declared geometry probe.');
end
end

function centres = local_load_reference_grid(root)
file = fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_140mm_xyz_mm.csv');
T = readtable(file, 'TextType', 'string');
centres = [T.x_mm, T.y_mm, T.z_mm];
end

function pose = local_build_pose_context(fmdl, reference_centres, lung_file)
boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);
node_normals = local_boundary_node_normals(fmdl);
face_ids = repmat((1:size(fmdl.boundary, 1))', 3, 1);
vertex_faces = accumarray(fmdl.boundary(:), face_ids, ...
    [size(fmdl.nodes, 1), 1], @(x) {x}, {[]});

reference_centre = mean(reference_centres, 1);
reference_axis = reference_centres(4, :) - reference_centres(1, :);
reference_axis = reference_axis / norm(reference_axis);

lung_tr = stlread(lung_file);
lung_vertices = lung_tr.Points;

pose = struct('fmdl', fmdl, 'boundary_nodes', boundary_nodes, ...
    'boundary_xyz', boundary_xyz, 'node_normals', node_normals, ...
    'vertex_faces', {vertex_faces}, ...
    'reference_centre', reference_centre, ...
    'reference_axis', reference_axis, 'lung_vertices', lung_vertices, ...
    'lung_faces', lung_tr.ConnectivityList);

% Patch precomputation depends only on the mesh, and the mesh is fixed for
% the whole fit. Building it here turns the per-evaluation patch rebuild from
% "reindex the whole boundary" into "walk the patch perimeter".
pose.patch_context = electrode_patch_context(fmdl);
end

function normals = local_boundary_node_normals(fmdl)
normals = trkg4_boundary_normals(fmdl);
end

function [z_curve, state] = local_forward_curve(x, sizes, fmdl_mm, ...
    fmdl_si, pose, cfg, base_sigma, tissue_id, soft_id, lung_id, patch_builder_mode)
elem_sigma = base_sigma;
elem_sigma(tissue_id == soft_id) = 1 / x(1);
elem_sigma(tissue_id == lung_id) = 1 / x(2);

state = local_pose_state(pose, x(3), x(4), x(5));
% Check every grid before spending time on any FEM solve.
patches = cell(numel(sizes), 1);
qc_cfg = cfg;
qc_cfg.electrode_diagnostics_file = '';
qc_cfg.electrode_diagnostics_verbose = false;
qc_cfg.fail_on_electrode_diagnostics = true;
for k = 1:numel(sizes)
    % Continuous projected centres still select discrete boundary faces.
    centres = local_grid_centres(pose, state, sizes(k));
    model_mm = fmdl_mm;
    [model_mm.electrode, induced_faces] = local_build_electrode_patches( ...
        model_mm, centres, cfg, pose.patch_context, patch_builder_mode);
    trkg4_electrode_diagnostics(model_mm, centres, ...
        string(cfg.electrode_order), qc_cfg, centres, induced_faces);
    if patch_builder_mode == "disjoint"
        all_nodes = vertcat(model_mm.electrode.nodes);
        if numel(unique(all_nodes)) ~= numel(all_nodes)
            error('trkg4:patchOverlap', 'Electrode patches share CEM nodes.');
        end
    end
    patches{k} = model_mm.electrode;
end
z_curve = zeros(numel(sizes), 1);
for k = 1:numel(sizes)
    model_mm = fmdl_mm;
    model_mm.electrode = patches{k};
    model_si = fmdl_si;
    model_si.electrode = model_mm.electrode;
    img = mk_image(model_si, elem_sigma);
    voltage = fwd_solve(img);
    z_curve(k) = real(voltage.meas(1) / cfg.current_ampere);
    if ~isfinite(z_curve(k))
        error('trkg4:nonfiniteForward', 'FEM returned a non-finite impedance.');
    end

    % Every electrode size creates a different CEM system matrix. Keeping all
    % nine matrices in the EIDORS cache can exhaust memory during a multi-start
    % inverse search even though no later size reuses them. Release each one
    % immediately after extracting its scalar impedance.
    clear voltage img model_si model_mm
    eidors_cache('clear_all');
end
end

function state = local_pose_state(pose, du_mm, dv_mm, phi_deg)
% Build the reference tangent basis, move the centre, then rebuild the
% tangent plane at the new skin point.
[c0, n0] = local_project_to_skin(pose, pose.reference_centre);
u0 = pose.reference_axis - dot(pose.reference_axis, n0) * n0;
u0 = u0 / norm(u0);
v0 = cross(n0, u0);
v0 = v0 / norm(v0);

raw_centre = c0 + du_mm * u0 + dv_mm * v0;
[centre, normal] = local_project_to_skin(pose, raw_centre);
base_axis = u0 - dot(u0, normal) * normal;
base_axis = base_axis / norm(base_axis);
side_axis = cross(normal, base_axis);
side_axis = side_axis / norm(side_axis);
phi = deg2rad(phi_deg);
axis = cos(phi) * base_axis + sin(phi) * side_axis;
axis = axis / norm(axis);

[~, h_centre_mm] = knnsearch(pose.lung_vertices, centre);
state = struct('centre_xyz_mm', centre, 'normal_unit_xyz', normal, ...
    'axis_unit_xyz', axis, 'side_unit_xyz', side_axis, ...
    'h_centre_mm', h_centre_mm);
end

function centres = local_grid_centres(pose, state, L_mm)
offset = [-L_mm / 2; -L_mm / 4; L_mm / 4; L_mm / 2];
raw = state.centre_xyz_mm + offset .* state.axis_unit_xyz;
centres = zeros(4, 3);
for k = 1:4
    centres(k, :) = local_project_to_skin(pose, raw(k, :));
end
end

function [electrodes_table, grids_table] = local_electrode_geometry( ...
    sizes, pose, state, fmdl_mm, cfg, patch_builder_mode)
% Report both the requested skin points and the area-weighted centroids of
% the electrode patches selected on the discrete FEM boundary.
labels = ["I_plus"; "V_plus"; "V_minus"; "I_minus"];
n_grids = numel(sizes);
n_electrodes = 4 * n_grids;
L_column = zeros(n_electrodes, 1);
label_column = strings(n_electrodes, 1);
requested_xyz = zeros(n_electrodes, 3);
centroid_xyz = zeros(n_electrodes, 3);
area_mm2 = zeros(n_electrodes, 1);

actual_current_span = zeros(n_grids, 1);
actual_voltage_span = zeros(n_grids, 1);
actual_centre_xyz = zeros(n_grids, 3);
row = 0;
for k = 1:n_grids
    requested = local_grid_centres(pose, state, sizes(k));
    model = fmdl_mm;
    [model.electrode, induced_faces] = local_build_electrode_patches( ...
        model, requested, cfg, pose.patch_context, patch_builder_mode);
    patch_centres = zeros(4, 3);
    for e = 1:4
        row = row + 1;
        % The builder already returns the induced faces, i.e. the surface the
        % CEM integrates over; recomputing them with ismember over the whole
        % boundary would only repeat that work.
        patch_faces = model.boundary(induced_faces{e}, :);
        [patch_centres(e, :), area_mm2(row)] = ...
            local_area_weighted_face_centre(model.nodes, patch_faces);
        L_column(row) = sizes(k);
        label_column(row) = labels(e);
        requested_xyz(row, :) = requested(e, :);
        centroid_xyz(row, :) = patch_centres(e, :);
    end
    actual_current_span(k) = norm(patch_centres(4, :) - patch_centres(1, :));
    actual_voltage_span(k) = norm(patch_centres(3, :) - patch_centres(2, :));
    actual_centre_xyz(k, :) = mean(patch_centres, 1);
end

electrodes_table = table(L_column, label_column, ...
    requested_xyz(:, 1), requested_xyz(:, 2), requested_xyz(:, 3), ...
    centroid_xyz(:, 1), centroid_xyz(:, 2), centroid_xyz(:, 3), area_mm2, ...
    'VariableNames', {'L_mm', 'electrode', 'requested_x_mm', ...
    'requested_y_mm', 'requested_z_mm', 'patch_centroid_x_mm', ...
    'patch_centroid_y_mm', 'patch_centroid_z_mm', 'patch_area_mm2'});
grids_table = table(sizes(:), actual_current_span, actual_voltage_span, ...
    actual_centre_xyz(:, 1), actual_centre_xyz(:, 2), ...
    actual_centre_xyz(:, 3), 'VariableNames', {'L_nominal_mm', ...
    'I_to_I_actual_mm', 'V_to_V_actual_mm', 'patch_mean_x_mm', ...
    'patch_mean_y_mm', 'patch_mean_z_mm'});
end

function [centre, area] = local_area_weighted_face_centre(nodes, faces)
if isempty(faces)
    centre = [NaN, NaN, NaN];
    area = 0;
    return;
end
p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
face_area = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);
face_centre = (p1 + p2 + p3) / 3;
area = sum(face_area);
centre = sum(face_centre .* face_area, 1) / area;
end

function [point, normal] = local_project_to_skin(pose, raw_point)
[idx, ~] = knnsearch(pose.boundary_xyz, raw_point, ...
    'K', min(16, size(pose.boundary_xyz, 1)));
global_nodes = pose.boundary_nodes(idx(:));
candidate_cells = pose.vertex_faces(global_nodes);
candidate_faces = unique(vertcat(candidate_cells{:}));

best_distance_sq = Inf;
point = raw_point;
normal = [0, 0, 0];
for j = reshape(candidate_faces, 1, [])
    face_nodes = pose.fmdl.boundary(j, :);
    triangle = pose.fmdl.nodes(face_nodes, :);
    [candidate, barycentric] = local_closest_point_triangle( ...
        raw_point, triangle(1, :), triangle(2, :), triangle(3, :));
    distance_sq = sum((candidate - raw_point).^2);
    if distance_sq < best_distance_sq
        best_distance_sq = distance_sq;
        point = candidate;
        normal = barycentric * pose.node_normals(face_nodes, :);
    end
end
if norm(normal) == 0
    error('trkg4:invalidSurfaceNormal', ...
        'Projected point has no valid outward normal from the volume mesh.');
end
normal = normal / norm(normal);
end

function [q, bary] = local_closest_point_triangle(p, a, b, c)
% Closest point and barycentric weights; region tests from Ericson's
% Real-Time Collision Detection.
ab = b - a;
ac = c - a;
ap = p - a;
d1 = dot(ab, ap);
d2 = dot(ac, ap);
if d1 <= 0 && d2 <= 0
    q = a; bary = [1, 0, 0]; return;
end

bp = p - b;
d3 = dot(ab, bp);
d4 = dot(ac, bp);
if d3 >= 0 && d4 <= d3
    q = b; bary = [0, 1, 0]; return;
end

vc = d1 * d4 - d3 * d2;
if vc <= 0 && d1 >= 0 && d3 <= 0
    v = d1 / (d1 - d3);
    q = a + v * ab; bary = [1 - v, v, 0]; return;
end

cp = p - c;
d5 = dot(ab, cp);
d6 = dot(ac, cp);
if d6 >= 0 && d5 <= d6
    q = c; bary = [0, 0, 1]; return;
end

vb = d5 * d2 - d1 * d6;
if vb <= 0 && d2 >= 0 && d6 <= 0
    w = d2 / (d2 - d6);
    q = a + w * ac; bary = [1 - w, 0, w]; return;
end

va = d3 * d6 - d5 * d4;
if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    q = b + w * (c - b); bary = [0, 1 - w, w]; return;
end

denom = 1 / (va + vb + vc);
v = vb * denom;
w = vc * denom;
bary = [1 - v - w, v, w];
q = bary(1) * a + bary(2) * b + bary(3) * c;
end

function [h_ray, h_nearest, skin_xyz] = local_depth_curve(pose, state, s_mm)
skin_xyz = zeros(numel(s_mm), 3);
h_ray = NaN(numel(s_mm), 1);
h_nearest = zeros(numel(s_mm), 1);
for k = 1:numel(s_mm)
    raw = state.centre_xyz_mm + s_mm(k) * state.axis_unit_xyz;
    [skin_xyz(k, :), normal] = local_project_to_skin(pose, raw);
    [~, h_nearest(k)] = knnsearch(pose.lung_vertices, skin_xyz(k, :));
    h_ray(k) = local_ray_mesh_distance(skin_xyz(k, :), -normal, ...
        pose.lung_vertices, pose.lung_faces);
end
end

function distance = local_ray_mesh_distance(origin, direction, vertices, faces)
v0 = vertices(faces(:, 1), :);
edge1 = vertices(faces(:, 2), :) - v0;
edge2 = vertices(faces(:, 3), :) - v0;
direction = direction / norm(direction);
h = cross(repmat(direction, size(edge2, 1), 1), edge2, 2);
a = dot(edge1, h, 2);
valid = abs(a) > 1e-10;
f = zeros(size(a));
f(valid) = 1 ./ a(valid);
s = origin - v0;
u = f .* dot(s, h, 2);
q = cross(s, edge1, 2);
v = f .* (q * direction');
t = f .* dot(q, edge2, 2);
hit = valid & u >= -1e-9 & v >= -1e-9 & (u + v) <= 1 + 1e-9 & t > 1e-9;
if any(hit)
    distance = min(t(hit));
else
    distance = NaN;
end
end


function [electrodes, induced_faces] = local_build_electrode_patches( ...
    fmdl, centres, cfg, patch_context, patch_builder_mode)
switch patch_builder_mode
    case "disjoint"
        [electrodes, induced_faces] = electrode_faces_by_area( ...
            fmdl, centres, cfg.electrode_area, cfg.z_contact, patch_context);
    case "legacy"
        [electrodes, induced_faces] = electrode_faces_by_area_legacy( ...
            fmdl, centres, cfg.electrode_area, cfg.z_contact);
    case "legacy_fast"
        [electrodes, induced_faces] = electrode_faces_by_area_legacy_fast( ...
            fmdl, centres, cfg.electrode_area, cfg.z_contact);
    otherwise
        error('trkg4:invalidPatchBuilderMode', ...
            'Unsupported patch builder mode: %s', patch_builder_mode);
end
end

