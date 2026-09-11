function report=verify_trkg4_full_scan_best(result_tag)
% Independent reference: original MATLAB projection + global canonical patches
% + full EIDORS CEM (including reciprocal measurement), no selected Green.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'src'));cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
assert(~isempty(regexp(result_tag,'^[A-Za-z0-9_-]+$','once')),'Invalid tag');
out=fullfile(root,'output',['full_pose_scan_' result_tag]);
manifest=jsondecode(fileread(fullfile(out,'manifest.json')));
assert(manifest.complete,'A complete published scan is required');
records=[manifest.inputs(:);manifest.outputs(:)];
for k=1:numel(records)
    assert(strcmp(trkg4_file_sha256(fullfile(out,records(k).path)),records(k).sha256),'Changed source/result');
end
summary=jsondecode(fileread(fullfile(out,'summary.json')));best=summary.bestrecord;
assert(isstruct(best),'No valid best pose');
run=jsondecode(fileread(fullfile(out,'run_manifest.json')));
chunk=run.physics.centre_chunk;bi=ceil(best.centre_index/chunk);
checkpoint=fullfile(out,'geometry',sprintf('block_%05d.mat',bi));
check=jsondecode(fileread([checkpoint '.json']));
assert(strcmp(check.fingerprint,summary.fingerprint)&&strcmp(check.sha256,trkg4_file_sha256(checkpoint)));
b=load(checkpoint);rr=find(b.centre_ids==best.centre_index & b.phi_deg==best.phi_deg);
vr=find(b.valid_rows==rr);assert(isscalar(vr),'Best checkpoint row is missing');
shared=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
f=shared.fmdl_mesh_units;si=trkg4_scale_fmdl_to_si(f,cfg);
[sizes,measured]=trkg4_load_nik_inhale_data(root,'manifest');
assert(isequal(sizes(:),summary.experiment.L_mm(:))&&max(abs(measured(:)-summary.experiment.Z_ohm(:)))<1e-12);
sigma=shared.elem_sigma;
sigma(shared.tissue_id==find(strcmp(shared.tissue_names,'soft_tissue'),1))=1/summary.rho1;
sigma(shared.tissue_id==find(strcmp(shared.tissue_names,'lungs'),1))=1/summary.rho2;
ref=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
pose=local_build_pose_context(f,[ref.x_mm ref.y_mm ref.z_mm],cfg.subject.stl.lungs);
state=local_pose_state(pose,best.u_mm,best.v_mm,best.phi_deg);
assert(max(abs(state.centre_xyz_mm-[best.x_mm best.y_mm best.z_mm]))<1e-9);
qc=cfg;qc.electrode_diagnostics_file='';qc.electrode_diagnostics_verbose=false;qc.fail_on_electrode_diagnostics=true;
[stim,select]=trkg4_make_4electrode_stimulation(cfg);
reciprocal=stim;reciprocal.stim_pattern=cfg.current_ampere*[0;1;-1;0];reciprocal.meas_pattern=[1 0 0 -1];
si.stimulation=[stim reciprocal];si.meas_select=[select;select];
z=zeros(numel(sizes),1);zr=z;coords=zeros(4,numel(sizes),3);count=z;
for j=1:numel(sizes)
    centres=local_grid_centres(pose,state,sizes(j));coords(:,j,:)=reshape(centres,4,1,3);
    [el,faces]=electrode_faces_by_area(f,centres,cfg.electrode_area,cfg.z_contact,pose.patch_context);
    model=f;model.electrode=el;
    trkg4_electrode_diagnostics(model,centres,string(cfg.electrode_order),qc,centres,faces);
    for e=1:4,assert(isequal(double(b.patch_faces{vr,j,e}(:)),double(faces{e}(:))),'Native/global reference patches differ');end
    si.electrode=el;sol=fwd_solve(mk_image(si,sigma));
    assert(numel(sol.meas)==2&&isreal(sol.meas),'Unexpected full CEM data');
    z(j)=sol.meas(1)/cfg.current_ampere;zr(j)=sol.meas(2)/cfg.current_ampere;count(j)=sum(arrayfun(@(e)numel(e.nodes),el));
    clear sol;eidors_cache('clear_all');
    fprintf('Independent full CEM L=%g: %.12g Ohm; stored difference %.3g; reciprocity %.3g Ohm\n', ...
        sizes(j),z(j),abs(z(j)-best.z_ohm(j)),abs(z(j)-zr(j)));
end
err=abs(z-best.z_ohm(:));tolerance=1e-7+1e-9*abs(z);
assert(all(err<tolerance),'Full CEM differs from stored scan');
assert(all(abs(z-zr)<tolerance),'Full CEM reciprocity failed');
report=struct('passed',true,'kind','independent_numerical_CEM_check_not_physical_validation', ...
    'result_tag',result_tag,'fingerprint',summary.fingerprint,'centre_index',best.centre_index,'phi_deg',best.phi_deg, ...
    'sizes_mm',sizes,'full_eidors_ohm',z,'stored_ohm',best.z_ohm(:),'reciprocal_ohm',zr,'contact_nodes',count, ...
    'max_difference_ohm',max(err),'max_reciprocity_difference_ohm',max(abs(z-zr)), ...
    'independent_rmse_ohm',sqrt(mean((z-measured(:)).^2)), ...
    'reference_projection','original_MATLAB_triangle_functions_copied_verbatim', ...
    'reference_patch_builder','global_canonical_electrode_faces_by_area', ...
    'verifier_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
for k=1:numel(records)
    assert(strcmp(trkg4_file_sha256(fullfile(out,records(k).path)),records(k).sha256),'Changed source/result during verification');
end
fid=fopen(fullfile(out,'independent_best_qc.json'),'w');assert(fid>0);cleaner=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));clear cleaner;
save(fullfile(out,'independent_best_electrodes.mat'),'coords','sizes','state','-v7');
disp(report);
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

function centres = local_grid_centres(pose, state, L_mm)
offset = [-L_mm / 2; -L_mm / 4; L_mm / 4; L_mm / 2];
raw = state.centre_xyz_mm + offset .* state.axis_unit_xyz;
centres = zeros(4, 3);
for k = 1:4
    centres(k, :) = local_project_to_skin(pose, raw(k, :));
end
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
