function export_search_geometry(tag)
% Geometry-only reconstruction using exact copies of the published pose helpers.
% No impedance calculation, optimization, or rewriting of existing results.
if nargin<1, tag='auditfix_20260908'; end
assert(~isempty(regexp(tag,'^[A-Za-z0-9_-]+$','once')));
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(here));cd(root);addpath('src','vendor_stl_eidors');
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
opt=struct('ResultTag',tag,'DataSelection','manifest','PatchBuilderMode','disjoint','ComputeJacobian',true);
trkg4_validate_inverse_contract(root,opt);
S=load('output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat','fmdl_mesh_units');f=S.fmdl_mesh_units;
pose=local_build_pose_context(f,local_load_reference_grid(root),cfg.subject.stl.lungs);
T=readtable(fullfile('output',['nik_trkg4_inverse_direct_search_history_' tag '.csv']),'TextType','string');
P=zeros(height(T),3);N=P;A=P;H=zeros(height(T),1);
for k=1:height(T)
 state=local_pose_state(pose,T.centre_u_mm(k),T.centre_v_mm(k),T.phi_deg(k));
 P(k,:)=state.centre_xyz_mm;N(k,:)=state.normal_unit_xyz;A(k,:)=state.axis_unit_xyz;H(k)=state.h_centre_mm;
end
T.centre_x_mm=P(:,1);T.centre_y_mm=P(:,2);T.centre_z_mm=P(:,3);
T.normal_x=N(:,1);T.normal_y=N(:,2);T.normal_z=N(:,3);
T.axis_x=A(:,1);T.axis_y=A(:,2);T.axis_z=A(:,3);T.reconstructed_h_vertex_mm=H;
good=T.status=="evaluated"; h_error=max(abs(T.h_centre_mm(good)-H(good)));assert(h_error<1e-8);
fit=readtable(fullfile('output',['nik_trkg4_inverse_inhale_fit_summary_' tag '.csv']));
valid_ids=find(good);[~,b]=min(T.objective(good));best=valid_ids(b);
xyz_error=max(abs(P(best,:)-[fit.centre_x_mm fit.centre_y_mm fit.centre_z_mm]));assert(xyz_error<1e-8);
axis_error=max(abs(A(best,:)-[fit.axis_x fit.axis_y fit.axis_z]));assert(axis_error<1e-8);
writetable(T,fullfile(here,['search_points_' tag '.csv']));
boundary_ids=unique(f.boundary(:));nodes_mm=f.nodes(boundary_ids,:);[~,boundary_faces]=ismember(f.boundary,boundary_ids);
[reference_centre,reference_normal]=local_project_to_skin(pose,pose.reference_centre);
reference_u=pose.reference_axis-dot(pose.reference_axis,reference_normal)*reference_normal;reference_u=reference_u/norm(reference_u);
reference_v=cross(reference_normal,reference_u);reference_v=reference_v/norm(reference_v);
save(fullfile(here,['search_surface_' tag '.mat']),'nodes_mm','boundary_faces','reference_centre','reference_normal','reference_u','reference_v','-v7');
R=struct('result_tag',tag,'contract_validated',true,'fem_recomputed',false,'rows',height(T),'h_max_error_mm',h_error,'best_xyz_max_error_mm',xyz_error,'best_axis_max_error',axis_error,'boundary_faces',size(boundary_faces,1));
fid=fopen(fullfile(here,['geometry_checks_' tag '.json']),'w');cleaner=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));disp(R);
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

