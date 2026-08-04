function result = run_trkg4_fast_cem_refine(top_count, local_radius_mm, local_phi_step_deg)
%RUN_TRKG4_FAST_CEM_REFINE Exact low-rank CEM evaluation of PEM candidates.
% Reuses one grounded bulk FEM inverse on all nodes touched by finite-area
% electrodes, then solves only small local systems for each four-electrode
% configuration.

if nargin < 1 || isempty(top_count), top_count = 100; end
if nargin < 2 || isempty(local_radius_mm), local_radius_mm = 0; end
if nargin < 3 || isempty(local_phi_step_deg), local_phi_step_deg = 1; end
cfg = trkg4_config('nik');
root = cfg.project_root;
result_tag = 'v5_1mm_local2mm';
shared = load(fullfile(root, 'output', 'nik_trkg4_arms_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units', 'elem_sigma', 'tissue_id', 'tissue_names');
fmdl_mm = shared.fmdl_mesh_units;
fmdl_si = trkg4_scale_fmdl_to_si(fmdl_mm, cfg);

fit = readtable(fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_fit_summary_%s.csv', result_tag)));
rho_soft = fit.rho_soft_ohm_m(1);
rho_lung = fit.rho_lungs_ohm_m(1);
elem_sigma = shared.elem_sigma;
soft_id = find(strcmp(shared.tissue_names, 'soft_tissue'), 1);
lung_id = find(strcmp(shared.tissue_names, 'lungs'), 1);
elem_sigma(shared.tissue_id == soft_id) = 1 / rho_soft;
elem_sigma(shared.tissue_id == lung_id) = 1 / rho_lung;

pem_top = readtable(fullfile(root, 'output', 'nik_trkg4_fast_pem_scan_top.csv'));
pem_top = pem_top(1:min(top_count, height(pem_top)), :);
candidates = table(pem_top.centre_u_mm, pem_top.centre_v_mm, pem_top.phi_deg, ...
    false(height(pem_top), 1), 'VariableNames', ...
    {'centre_u_mm', 'centre_v_mm', 'phi_deg', 'is_validation'});
if local_radius_mm > 0
    offsets = -local_radius_mm:1:local_radius_mm;
    phi_local = -20:local_phi_step_deg:20;
    [du_local, dv_local, angle_local] = ndgrid(offsets, offsets, phi_local);
    local_candidates = table(fit.centre_u_mm(1) + du_local(:), ...
        fit.centre_v_mm(1) + dv_local(:), angle_local(:), ...
        false(numel(du_local), 1), 'VariableNames', ...
        {'centre_u_mm', 'centre_v_mm', 'phi_deg', 'is_validation'});
    candidates = [candidates; local_candidates];
end
candidates(end + 1, :) = {fit.centre_u_mm(1), fit.centre_v_mm(1), ...
    fit.phi_deg(1), true};
[~, unique_rows] = unique(candidates(:, 1:3), 'rows', 'stable');
candidates = candidates(unique_rows, :);

[sizes_mm, measured_z] = local_load_inhale_data(root);
experimental_line = robustfit(sizes_mm, measured_z);
experimental_slope = experimental_line(2);
pose = local_build_pose_context(fmdl_mm, root, cfg.subject.stl.lungs);
n_pose = height(candidates);
n_size = numel(sizes_mm);
patch_faces = cell(n_pose, n_size, 4);
h_centre = zeros(n_pose, 1);
patch_context = [];

tic_geometry = tic;
for pose_index = 1:n_pose
    state = local_pose_state(pose, candidates.centre_u_mm(pose_index), ...
        candidates.centre_v_mm(pose_index), candidates.phi_deg(pose_index));
    h_centre(pose_index) = state.h_centre_mm;
    for size_index = 1:n_size
        centres = local_grid_centres(pose, state, sizes_mm(size_index));
        % Same builder as the forward and inverse models. It used to be
        % electrode_faces_by_area_fast, which picked faces by a different
        % rule and sized the patch on the picked faces instead of the induced
        % ones - so this scan optimised a slightly different electrode
        % geometry from the one being refined. The induced faces now come
        % straight out of the builder instead of being rebuilt here.
        [~, induced_faces, patch_context] = electrode_faces_by_area( ...
            fmdl_mm, centres, cfg.electrode_area, cfg.z_contact, patch_context);
        for electrode_index = 1:4
            patch_faces{pose_index, size_index, electrode_index} = ...
                induced_faces{electrode_index};
        end
    end
end
geometry_seconds = toc(tic_geometry);

all_patch_faces = unique(vertcat(patch_faces{:}));
candidate_nodes = unique(reshape(fmdl_mm.boundary(all_patch_faces, :), [], 1));
candidate_nodes(candidate_nodes == fmdl_si.gnd_node) = [];
[green, factor_seconds, solve_seconds] = local_bulk_green( ...
    fmdl_si, elem_sigma, candidate_nodes);
global_to_green = zeros(size(fmdl_mm.nodes, 1), 1);
global_to_green(candidate_nodes) = 1:numel(candidate_nodes);

z_cem_fast = zeros(n_pose, n_size);
tic_local = tic;
for pose_index = 1:n_pose
    for size_index = 1:n_size
        faces_for_grid = patch_faces(pose_index, size_index, :);
        z_cem_fast(pose_index, size_index) = local_cem_transfer( ...
            fmdl_si, faces_for_grid, global_to_green, green, cfg.z_contact);
    end
end
local_seconds = toc(tic_local);

abs_residual = abs(z_cem_fast - measured_z(:)');
huber_delta = 5;
huber = 0.5 * min(abs_residual, huber_delta).^2 + ...
    huber_delta * max(abs_residual - huber_delta, 0);
slope = zeros(n_pose, 1);
for pose_index = 1:n_pose
    fitted_line = robustfit(sizes_mm, z_cem_fast(pose_index, :)');
    slope(pose_index) = fitted_line(2);
end
span_mm = max(sizes_mm) - min(sizes_mm);
objective = mean(huber, 2) + ...
    0.20 * ((slope - experimental_slope) * span_mm).^2 + ...
    ((h_centre - 15) / 7).^2 + ...
    2 * max(h_centre - 30, 0).^2 + 2 * max(7 - h_centre, 0).^2 + ...
    0.25 * ((rho_lung - cfg.rho_cloud.lungs_ohm_m) / 6)^2;

comparison = readtable(fullfile(root, 'output', ...
    sprintf('nik_trkg4_inverse_inhale_comparison_%s.csv', result_tag)));
validation_row = find(candidates.is_validation, 1);
validation_curve_rmse = sqrt(mean((z_cem_fast(validation_row, :)' ...
    - comparison.Z_FEM_ohm).^2));

output = candidates;
output.h_centre_mm = h_centre;
output.fast_CEM_objective = objective;
output.slope_ohm_per_mm = slope;
for size_index = 1:n_size
    output.(sprintf('Z_fast_CEM_L%03d_ohm', sizes_mm(size_index))) = ...
        z_cem_fast(:, size_index);
end
output = sortrows(output, 'fast_CEM_objective');
output_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_fast_cem_refine_%s.csv', result_tag));
writetable(output, output_file);

summary = table(top_count, local_radius_mm, local_phi_step_deg, ...
    n_pose, numel(candidate_nodes), ...
    geometry_seconds, factor_seconds, solve_seconds, local_seconds, ...
    validation_curve_rmse, ...
    'VariableNames', {'PEM_top_count', 'local_radius_mm', ...
    'local_phi_step_deg', 'evaluated_pose_count', ...
    'patch_node_count', 'geometry_seconds', 'factor_seconds', ...
    'green_solve_seconds', 'local_CEM_seconds', ...
    'validation_curve_RMSE_ohm'});
summary_file = fullfile(root, 'output', ...
    sprintf('nik_trkg4_fast_cem_refine_summary_%s.csv', result_tag));
writetable(summary, summary_file);

result = struct('candidates', output, 'summary', summary, ...
    'output_file', output_file, 'summary_file', summary_file);
fprintf('Fast low-rank CEM refinement saved: %s\n', output_file);
fprintf('Validation against full EIDORS curve: RMSE %.9g Ohm.\n', ...
    validation_curve_rmse);
end

function z = local_cem_transfer(fmdl_si, patch_faces, ...
    global_to_green, green_all, z_contact)
% candidate_nodes used to be passed in but was never read: the node set is
% recovered from patch_faces below, and global_to_green already encodes the
% mapping into the precomputed Green block.
all_faces = unique(vertcat(patch_faces{:}));
patch_nodes = unique(reshape(fmdl_si.boundary(all_faces, :), [], 1));
local_green_index = global_to_green(patch_nodes);
if any(local_green_index == 0)
    error('Missing patch node in bulk Green matrix.');
end
G = double(green_all(local_green_index, local_green_index));
n_patch = numel(patch_nodes);
D = zeros(n_patch);
B = zeros(n_patch, 4);
C = zeros(4);
mass_template = [2, 1, 1; 1, 2, 1; 1, 1, 2];

for electrode_index = 1:4
    faces = patch_faces{electrode_index};
    for face_index = reshape(faces, 1, [])
        nodes = fmdl_si.boundary(face_index, :);
        points = fmdl_si.nodes(nodes, :);
        area = 0.5 * norm(cross(points(2, :) - points(1, :), ...
            points(3, :) - points(1, :)));
        [present, local] = ismember(nodes, patch_nodes);
        if ~all(present)
            error('Face node missing from local patch-node list.');
        end
        D(local, local) = D(local, local) + ...
            area / (12 * z_contact) * mass_template;
        B(local, electrode_index) = B(local, electrode_index) ...
            - area / (3 * z_contact);
        C(electrode_index, electrode_index) = ...
            C(electrode_index, electrode_index) + area / z_contact;
    end
end

Q = (eye(n_patch) + G * D) \ (G * B);
schur = C - B' * Q;
electrode_voltage = schur \ [1; 0; 0; -1];
z = electrode_voltage(2) - electrode_voltage(3);
end

function [green, factor_seconds, solve_seconds] = local_bulk_green( ...
    fmdl_si, elem_sigma, candidate_nodes)
model = fmdl_si;
model.electrode = struct([]);
model.stimulation = struct([]);
img = mk_image(model, elem_sigma);
system = calc_system_mat(img);
ground = model.gnd_node;
keep = (1:size(system.E, 1))';
keep(ground) = [];
reduced_index = zeros(size(system.E, 1), 1);
reduced_index(keep) = 1:numel(keep);
candidate_reduced = reduced_index(candidate_nodes);
tic_factor = tic;
factor = decomposition(system.E(keep, keep), 'chol');
factor_seconds = toc(tic_factor);
n = numel(candidate_nodes);
green = zeros(n, n, 'single');
tic_solve = tic;
for first = 1:64:n
    last = min(first + 63, n);
    columns = first:last;
    rhs = sparse(candidate_reduced(columns), 1:numel(columns), 1, ...
        numel(keep), numel(columns));
    voltage = factor \ rhs;
    green(:, columns) = single(full(voltage(candidate_reduced, :)));
end
solve_seconds = toc(tic_solve);
end

function pose = local_build_pose_context(fmdl, root, lung_file)
reference = readtable(fullfile(root, 'data', 'nik', 'electrodes', ...
    'electrodes_4_right_rib_140mm_xyz_mm.csv'));
reference_centres = [reference.x_mm, reference.y_mm, reference.z_mm];
boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);
node_normals = local_boundary_node_normals(fmdl);
face_ids = repmat((1:size(fmdl.boundary, 1))', 3, 1);
vertex_faces = accumarray(fmdl.boundary(:), face_ids, ...
    [size(fmdl.nodes, 1), 1], @(x) {x}, {[]});
lung = stlread(lung_file);
pose = struct('fmdl', fmdl, 'boundary_nodes', boundary_nodes, ...
    'boundary_xyz', boundary_xyz, 'node_normals', node_normals, ...
    'vertex_faces', {vertex_faces}, ...
    'reference_centre', mean(reference_centres, 1), ...
    'reference_axis', (reference_centres(4, :) - reference_centres(1, :)) ...
    / norm(reference_centres(4, :) - reference_centres(1, :)), ...
    'lung_vertices', lung.Points);
end

function state = local_pose_state(pose, du, dv, phi_deg)
[c0, n0] = local_project_to_skin(pose, pose.reference_centre);
u0 = pose.reference_axis - dot(pose.reference_axis, n0) * n0;
u0 = u0 / norm(u0);
v0 = cross(n0, u0); v0 = v0 / norm(v0);
[centre, normal] = local_project_to_skin(pose, c0 + du * u0 + dv * v0);
base_axis = u0 - dot(u0, normal) * normal;
base_axis = base_axis / norm(base_axis);
side_axis = cross(normal, base_axis); side_axis = side_axis / norm(side_axis);
axis = cosd(phi_deg) * base_axis + sind(phi_deg) * side_axis;
[~, h] = knnsearch(pose.lung_vertices, centre);
state = struct('centre_xyz_mm', centre, 'axis_unit_xyz', axis / norm(axis), ...
    'h_centre_mm', h);
end

function centres = local_grid_centres(pose, state, L)
offset = [-L/2; -L/4; L/4; L/2];
raw = state.centre_xyz_mm + offset .* state.axis_unit_xyz;
centres = zeros(4, 3);
for k = 1:4
    centres(k, :) = local_project_to_skin(pose, raw(k, :));
end
end

function [point, normal] = local_project_to_skin(pose, raw)
[idx, ~] = knnsearch(pose.boundary_xyz, raw, ...
    'K', min(16, size(pose.boundary_xyz, 1)));
global_nodes = pose.boundary_nodes(idx(:));
candidate_cells = pose.vertex_faces(global_nodes);
candidate_faces = unique(vertcat(candidate_cells{:}));
best = Inf; point = raw; normal = [0, 0, 0];
for face = reshape(candidate_faces, 1, [])
    nodes = pose.fmdl.boundary(face, :);
    tri = pose.fmdl.nodes(nodes, :);
    [candidate, bary] = local_closest_point_triangle(raw, ...
        tri(1, :), tri(2, :), tri(3, :));
    distance = sum((candidate - raw).^2);
    if distance < best
        best = distance; point = candidate;
        normal = bary * pose.node_normals(nodes, :);
    end
end
normal = normal / norm(normal);
end

function [q, bary] = local_closest_point_triangle(p, a, b, c)
ab = b-a; ac = c-a; ap = p-a; d1=dot(ab,ap); d2=dot(ac,ap);
if d1<=0 && d2<=0, q=a; bary=[1,0,0]; return; end
bp=p-b; d3=dot(ab,bp); d4=dot(ac,bp);
if d3>=0 && d4<=d3, q=b; bary=[0,1,0]; return; end
vc=d1*d4-d3*d2;
if vc<=0 && d1>=0 && d3<=0
    v=d1/(d1-d3); q=a+v*ab; bary=[1-v,v,0]; return;
end
cp=p-c; d5=dot(ab,cp); d6=dot(ac,cp);
if d6>=0 && d5<=d6, q=c; bary=[0,0,1]; return; end
vb=d5*d2-d1*d6;
if vb<=0 && d2>=0 && d6<=0
    w=d2/(d2-d6); q=a+w*ac; bary=[1-w,0,w]; return;
end
va=d3*d6-d5*d4;
if va<=0 && (d4-d3)>=0 && (d5-d6)>=0
    w=(d4-d3)/((d4-d3)+(d5-d6)); q=b+w*(c-b); bary=[0,1-w,w]; return;
end
denom=1/(va+vb+vc); v=vb*denom; w=vc*denom;
bary=[1-v-w,v,w]; q=bary(1)*a+bary(2)*b+bary(3)*c;
end

function normals = local_boundary_node_normals(fmdl)
faces=fmdl.boundary; nodes=fmdl.nodes;
p1=nodes(faces(:,1),:); p2=nodes(faces(:,2),:); p3=nodes(faces(:,3),:);
fn=cross(p2-p1,p3-p1,2); fc=(p1+p2+p3)/3; centre=mean(nodes,1);
flip=dot(fn,fc-centre,2)<0; fn(flip,:)=-fn(flip,:);
normals=zeros(size(nodes));
for d=1:3
    normals(:,d)=accumarray(faces(:),repmat(fn(:,d),3,1), ...
        [size(nodes,1),1],@sum,0);
end
lengths=vecnorm(normals,2,2); valid=lengths>0;
normals(valid,:)=normals(valid,:)./lengths(valid);
end

function [sizes, z] = local_load_inhale_data(root)
files=dir(fullfile(root,'..','Colab Notebooks','timestamps','*nik.json'));
sizes=[]; z=[];
for k=1:numel(files)
    record=jsondecode(fileread(fullfile(files(k).folder,files(k).name)));
    if record.size_mm==90, continue; end
    holds=struct2cell(record.hold_levels);
    sizes(end+1,1)=record.size_mm; z(end+1,1)=holds{1}; %#ok<AGROW>
end
[sizes,order]=sort(sizes); z=z(order);
end
