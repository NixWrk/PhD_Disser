function inside = points_in_closed_surface(points, vertices, faces, use_parallel)
%POINTS_IN_CLOSED_SURFACE Point-in-polyhedron test for closed triangle meshes.
% Uses +X ray casting and parity of triangle intersections. The mesh should
% be closed and non-self-intersecting. Points exactly on a surface may be
% numerically ambiguous. Triangles are indexed in a YZ grid so large CT
% surfaces can be used without testing every triangle against every point.

if nargin < 4 || isempty(use_parallel)
    use_parallel = false;
end

n_points = size(points, 1);
inside = false(n_points, 1);
chunk_size = 5000;

direction = [1, 0, 0];
tri_a = vertices(faces(:,1), :);
tri_b = vertices(faces(:,2), :);
tri_c = vertices(faces(:,3), :);
[bins, y_edges, z_edges] = triangle_yz_bins(vertices, faces);

% A tiny deterministic Y/Z tilt reduces degeneracy when a ray hits an edge.
direction = direction / norm(direction);
ray_eps = 1e-7 * max(range(vertices, 1));
if ray_eps == 0
    ray_eps = 1e-7;
end

starts = 1:chunk_size:n_points;
stops = min(starts + chunk_size - 1, n_points);
n_chunks = numel(starts);

if use_parallel && n_chunks > 1
    inside_chunks = cell(n_chunks, 1);
    parfor c = 1:n_chunks
        p = points(starts(c):stops(c), :); %#ok<PFBNS>
        inside_chunks{c} = classify_chunk(p, ray_eps, y_edges, z_edges, bins, ...
            direction, tri_a, tri_b, tri_c);
    end
    inside = vertcat(inside_chunks{:});
else
    for c = 1:n_chunks
        inside(starts(c):stops(c)) = classify_chunk( ...
            points(starts(c):stops(c), :), ray_eps, y_edges, z_edges, bins, ...
            direction, tri_a, tri_b, tri_c);
    end
end
end

function inside = classify_chunk(points, ray_eps, y_edges, z_edges, bins, direction, tri_a, tri_b, tri_c)
points(:,2) = points(:,2) + ray_eps;
points(:,3) = points(:,3) + 0.37 * ray_eps;

counts = zeros(size(points, 1), 1);
point_bins = point_yz_bins(points, y_edges, z_edges);
valid = point_bins > 0;
unique_bins = unique(point_bins(valid));

for b = reshape(unique_bins, 1, [])
    point_idx = find(point_bins == b);
    tri_idx = bins{b};
    if isempty(tri_idx)
        continue;
    end

    pb = points(point_idx, :);
    local_counts = zeros(numel(point_idx), 1);
    for j = 1:numel(tri_idx)
        f = tri_idx(j);
        hit = ray_triangle_intersect(pb, direction, tri_a(f,:), tri_b(f,:), tri_c(f,:));
        local_counts = local_counts + hit;
    end
    counts(point_idx) = local_counts;
end

inside = mod(counts, 2) == 1;
end

function [bins, y_edges, z_edges] = triangle_yz_bins(vertices, faces)
n_faces = size(faces, 1);
n_bins_axis = max(24, min(160, ceil(sqrt(n_faces / 12))));

mins = min(vertices, [], 1);
maxs = max(vertices, [], 1);
pad = 10 * eps(max(maxs - mins));
if pad == 0
    pad = 1e-10;
end
y_edges = linspace(mins(2) - pad, maxs(2) + pad, n_bins_axis + 1);
z_edges = linspace(mins(3) - pad, maxs(3) + pad, n_bins_axis + 1);

bins = cell(n_bins_axis, n_bins_axis);
yv = vertices(:,2);
zv = vertices(:,3);

for f = 1:n_faces
    idx = faces(f,:);
    iy0 = edge_bin(min(yv(idx)), y_edges);
    iy1 = edge_bin(max(yv(idx)), y_edges);
    iz0 = edge_bin(min(zv(idx)), z_edges);
    iz1 = edge_bin(max(zv(idx)), z_edges);

    for iy = iy0:iy1
        for iz = iz0:iz1
            bins{iy, iz}(end + 1) = f;
        end
    end
end

bins = bins(:);
end

function point_bins = point_yz_bins(points, y_edges, z_edges)
n_bins_axis = numel(y_edges) - 1;
iy = discretize(points(:,2), y_edges);
iz = discretize(points(:,3), z_edges);
valid = ~isnan(iy) & ~isnan(iz);

point_bins = zeros(size(points, 1), 1);
point_bins(valid) = sub2ind([n_bins_axis, n_bins_axis], iy(valid), iz(valid));
end

function idx = edge_bin(value, edges)
n_bins_axis = numel(edges) - 1;
idx = find(value <= edges(2:end), 1, 'first');
if isempty(idx)
    idx = n_bins_axis;
end
idx = max(1, min(n_bins_axis, idx));
end

function hit = ray_triangle_intersect(origins, direction, v0, v1, v2)
% Vectorized Moller-Trumbore intersection for many ray origins and one tri.

tol = 1e-10;
edge1 = v1 - v0;
edge2 = v2 - v0;
h = cross(direction, edge2);
a = dot(edge1, h);

if abs(a) < tol
    hit = false(size(origins, 1), 1);
    return;
end

f = 1 / a;
s = origins - v0;
u = f * (s * h');
q = cross(s, repmat(edge1, size(s, 1), 1), 2);
v = f * (q * direction');
t = f * (q * edge2');

hit = (u >= -tol) & (u <= 1 + tol) & ...
      (v >= -tol) & ((u + v) <= 1 + tol) & ...
      (t > tol);
end
