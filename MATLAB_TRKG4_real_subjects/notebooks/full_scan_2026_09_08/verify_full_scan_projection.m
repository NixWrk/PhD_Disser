function verify_full_scan_projection()
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));cd(root);addpath('src','vendor_stl_eidors','tools/full_pose_scan');
S=load('output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat','fmdl_mesh_units');f=S.fmdl_mesh_units;
c=full_scan_context(f);nf=size(f.boundary,1);facelist=repmat((1:nf)',3,1);
pose=struct('fmdl',f,'boundary_nodes',c.boundary_nodes,'boundary_xyz',c.nodes(c.boundary_nodes,:), ...
'node_normals',c.normals,'vertex_faces',{accumarray(f.boundary(:),facelist,[size(f.nodes,1),1],@(x){x},{[]})});
ref=readtable('data/nik/electrodes/electrodes_4_right_rib_140mm_xyz_mm.csv');r=mean([ref.x_mm ref.y_mm ref.z_mm],1);
[c0,n0]=full_scan_project(c,r);axis=[ref.x_mm(4)-ref.x_mm(1),ref.y_mm(4)-ref.y_mm(1),ref.z_mm(4)-ref.z_mm(1)];axis=axis/norm(axis);
u0=axis-dot(axis,n0)*n0;u0=u0/norm(u0);v0=cross(n0,u0);v0=v0/norm(v0);
[a,b]=ndgrid(-120:20:120);raw=c0+a(:)*u0+b(:)*v0;
rng(9812);sample=randi(numel(c.boundary_nodes),100,1);raw=[raw;c.nodes(c.boundary_nodes(sample),:)+randn(100,3)*5;r];
tic;[P,N,F]=full_scan_project(c,raw);seconds_mex=toc;
oldP=zeros(size(P));oldN=oldP;tic;
for k=1:size(raw,1),[oldP(k,:),oldN(k,:)]=local_project_to_skin(pose,raw(k,:));end
seconds_matlab=toc;max_xyz=max(abs(P-oldP),[],'all');max_normal=max(abs(N-oldN),[],'all');
assert(max_xyz<1e-9);assert(max_normal<1e-8);assert(all(isfinite(P),'all'));
T=readtable('output/nik_trkg4_inverse_inhale_fit_summary_auditfix_20260908.csv');[best,nn]=full_scan_project(c,c0+T.centre_u_mm*u0+T.centre_v_mm*v0);
assert(max(abs(best-[T.centre_x_mm T.centre_y_mm T.centre_z_mm]))<1e-9);
base=u0-dot(u0,nn)*nn;base=base/norm(base);side=cross(nn,base);side=side/norm(side);angle=T.phi_deg*pi/180;aa=cos(angle)*base+sin(angle)*side;aa=aa/norm(aa);
assert(max(abs(aa-[T.axis_x T.axis_y T.axis_z]))<1e-9);
[gu,gv]=ndgrid(-120:120);tic;[allP,allN,allF]=full_scan_project(c,c0+gu(:)*u0+gv(:)*v0);full_centres_seconds=toc;
assert(size(allP,1)==58081 && all(isfinite(allP),'all'));
R=struct('status','passed','cases',size(raw,1),'max_xyz_difference_mm',max_xyz,'max_normal_difference',max_normal,'seconds_mex',seconds_mex,'seconds_matlab',seconds_matlab,'full_centres_seconds',full_centres_seconds,'centre_count',58081);
fid=fopen('notebooks/full_scan_2026_09_08/projection_checks.json','w');cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));disp(R);
% Cache geometry only; the scan manifest will bind its exact source/input hashes.
centres=table((1:numel(gu))',gu(:),gv(:),allP(:,1),allP(:,2),allP(:,3),allN(:,1),allN(:,2),allN(:,3),allF, ...
'VariableNames',{'centre_index','u_mm','v_mm','x_mm','y_mm','z_mm','nx','ny','nz','surface_face'});
save('notebooks/full_scan_2026_09_08/centres_geometry.mat','centres','c0','u0','v0','-v7');
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

