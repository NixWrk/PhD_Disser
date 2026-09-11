function [node_normals, face_normals] = trkg4_boundary_normals(fmdl_mm)
%TRKG4_BOUNDARY_NORMALS Orient boundary normals from adjacent tetrahedra.
%
%   [node_normals, face_normals] = trkg4_boundary_normals(fmdl_mm)
%
% `face_normals` has one unit outward normal for every row of
% `fmdl_mm.boundary`, in exactly that row order.  The orientation is decided
% from the centroid of the tetrahedron adjacent to each boundary face: the
% normal points away from that tetrahedron.  No global body centre or
% convexity assumption is used.
%
% `node_normals` has one area-weighted, unit normal for every mesh node.
% Interior nodes are returned as [0 0 0].  Boundary faces must be triangles
% with exactly one adjacent, non-degenerate tetrahedron.  Ambiguous,
% missing, duplicated, or degenerate topology is rejected explicitly.
%
% The function does not rescale coordinates.  The `_mm` suffix documents the
% current TRKG4 convention: coordinates are expected in millimetres and the
% returned normals are dimensionless.

if nargin ~= 1 || ~isstruct(fmdl_mm)
    error('trkg4:boundaryNormalsInput', ...
        'Expected one FEM model struct with nodes, elems, and boundary.');
end
required_fields = {'nodes', 'elems', 'boundary'};
if ~all(isfield(fmdl_mm, required_fields))
    error('trkg4:boundaryNormalsInput', ...
        'FEM model must contain nodes, elems, and boundary fields.');
end

nodes = double(fmdl_mm.nodes);
boundary = double(fmdl_mm.boundary);
elems = double(fmdl_mm.elems);
if size(nodes, 2) ~= 3 || isempty(nodes) || any(~isfinite(nodes), 'all')
    error('trkg4:boundaryNormalsInput', ...
        'fmdl.nodes must be a finite N-by-3 array.');
end
if size(boundary, 2) ~= 3 || isempty(boundary) || ...
        any(~isfinite(boundary), 'all') || any(boundary(:) ~= fix(boundary(:)))
    error('trkg4:boundaryNormalsInput', ...
        'fmdl.boundary must be a non-empty integer N-by-3 array.');
end
if size(elems, 2) < 4 || isempty(elems) || ...
        any(~isfinite(elems(:, 1:4)), 'all') || ...
        any(elems(:, 1:4) ~= fix(elems(:, 1:4)), 'all')
    error('trkg4:boundaryNormalsInput', ...
        'fmdl.elems must contain a non-empty integer tetrahedral N-by-4 array.');
end

n_nodes = size(nodes, 1);
boundary = boundary(:, 1:3);
elems = elems(:, 1:4);
if any(boundary(:) < 1 | boundary(:) > n_nodes) || ...
        any(elems(:) < 1 | elems(:) > n_nodes)
    error('trkg4:boundaryNormalsInput', ...
        'Boundary and tetrahedron indices must refer to fmdl.nodes.');
end
if any(elems(:, 1) == elems(:, 2) | elems(:, 1) == elems(:, 3) | ...
        elems(:, 1) == elems(:, 4) | elems(:, 2) == elems(:, 3) | ...
        elems(:, 2) == elems(:, 4) | elems(:, 3) == elems(:, 4))
    error('trkg4:boundaryNormalsDegenerate', ...
        'A tetrahedron contains repeated node indices.');
end

n_faces = size(boundary, 1);
boundary_keys = sort(boundary, 2);
[boundary_keys_unique, ~, boundary_group] = ...
    unique(boundary_keys, 'rows');
if size(boundary_keys_unique, 1) ~= n_faces
    duplicate_rows = accumarray(boundary_group, 1);
    error('trkg4:boundaryNormalsTopology', ...
        'fmdl.boundary contains %d duplicated triangle row(s).', ...
        sum(duplicate_rows > 1));
end

% The face order used below is independent of the orientation stored in
% fmdl.boundary.  It lets us find the adjacent tetrahedron by exact
% node-index matching while retaining the original boundary row order for
% the output normal.
tet_face_patterns = [1 2 3; 1 2 4; 1 3 4; 2 3 4];
adjacent_tet = zeros(n_faces, 1);
adjacent_count = zeros(n_faces, 1);
tet_ids = (1:size(elems, 1)).';
for pattern_index = 1:size(tet_face_patterns, 1)
    tet_faces = sort(elems(:, tet_face_patterns(pattern_index, :)), 2);
    [is_boundary, boundary_index] = ismember( ...
        tet_faces, boundary_keys_unique, 'rows');
    if ~any(is_boundary)
        continue;
    end
    hit_boundary = boundary_index(is_boundary);
    hit_tet = tet_ids(is_boundary);
    adjacent_count = adjacent_count + accumarray( ...
        hit_boundary, 1, [n_faces, 1], @sum, 0);
    first_hit = adjacent_tet(hit_boundary) == 0;
    adjacent_tet(hit_boundary(first_hit)) = hit_tet(first_hit);
end

if any(adjacent_count == 0)
    error('trkg4:boundaryNormalsTopology', ...
        '%d boundary face(s) have no adjacent tetrahedron.', ...
        sum(adjacent_count == 0));
end
if any(adjacent_count > 1)
    error('trkg4:boundaryNormalsNonManifold', ...
        '%d boundary face(s) have more than one adjacent tetrahedron.', ...
        sum(adjacent_count > 1));
end

% Convert the owner map from unique sorted face-key order back to the exact
% original fmdl.boundary row order before pairing it with face geometry.
adjacent_tet = adjacent_tet(boundary_group);

p1 = nodes(boundary(:, 1), :);
p2 = nodes(boundary(:, 2), :);
p3 = nodes(boundary(:, 3), :);
raw_area_vectors = cross(p2 - p1, p3 - p1, 2);
bbox_span = max(nodes, [], 1) - min(nodes, [], 1);
global_scale = max(1, max(bbox_span));
face_edge_scale = max([ ...
    vecnorm(p2 - p1, 2, 2), ...
    vecnorm(p3 - p1, 2, 2), ...
    vecnorm(p3 - p2, 2, 2)], [], 2);
area_vector_tolerance = 100 * eps(max(1, face_edge_scale .^ 2));
raw_area_vector_norm = vecnorm(raw_area_vectors, 2, 2);
if any(raw_area_vector_norm <= area_vector_tolerance)
    error('trkg4:boundaryNormalsDegenerate', ...
        '%d boundary face(s) have zero or numerically negligible area.', ...
        sum(raw_area_vector_norm <= area_vector_tolerance));
end

owner_nodes = elems(adjacent_tet, :);
is_face_node = bsxfun(@eq, owner_nodes, boundary(:, 1)) | ...
    bsxfun(@eq, owner_nodes, boundary(:, 2)) | ...
    bsxfun(@eq, owner_nodes, boundary(:, 3));
opposite_node = max(owner_nodes .* double(~is_face_node), [], 2);
if any(opposite_node == 0)
    error('trkg4:boundaryNormalsTopology', ...
        'At least one boundary face has no unique opposite tetrahedron node.');
end
opposite_vector = nodes(opposite_node, :) - p1;
orientation_dot = sum(raw_area_vectors .* opposite_vector, 2);
local_height_scale = abs(orientation_dot) ./ raw_area_vector_norm;
% The orientation determinant is a dot product of the area vector and
% the opposite-vertex vector. Scale its round-off test by the actual
% operands of that operation, rather than by the size of the whole body.
% This remains invariant under rigid translation and accepts measured
% sliver faces whose sign is many orders above local round-off.
opposite_vector_norm = vecnorm(opposite_vector, 2, 2);
operation_scale = raw_area_vector_norm .* opposite_vector_norm;
orientation_tolerance = 100 * eps(max(1, operation_scale));
ambiguous = abs(orientation_dot) <= orientation_tolerance;
if any(ambiguous)
    bad_faces = find(ambiguous);
    details = cell(min(numel(bad_faces), 8), 1);
    for detail_index = 1:numel(details)
        face_index = bad_faces(detail_index);
        details{detail_index} = sprintf( ...
            ['face %d, owner tetra %d, opposite node %d, dot %.17g, ', ...
             'tol %.17g, operation_scale %.17g, edge_scale %.17g, ', ...
             'height %.17g'], ...
            face_index, adjacent_tet(face_index), ...
            opposite_node(face_index), orientation_dot(face_index), ...
            orientation_tolerance(face_index), ...
            operation_scale(face_index), face_edge_scale(face_index), ...
            local_height_scale(face_index));
    end
    error('trkg4:boundaryNormalsDegenerate', ...
        ['Outward orientation is numerically indeterminate for %d ', ...
         'boundary face(s): %s'], numel(bad_faces), ...
        strjoin(details, '; '));
end

flip = orientation_dot > 0;
oriented_area_vectors = raw_area_vectors;
oriented_area_vectors(flip, :) = -oriented_area_vectors(flip, :);
face_normals = oriented_area_vectors ./ ...
    vecnorm(oriented_area_vectors, 2, 2);

% Sum oriented face area vectors at each node.  The factor 1/2 is common to
% every contribution and is retained to make the area weighting explicit.
node_area_vectors = zeros(n_nodes, 3);
for dimension = 1:3
    node_area_vectors(:, dimension) = accumarray( ...
        boundary(:), repmat(0.5 * oriented_area_vectors(:, dimension), 3, 1), ...
        [n_nodes, 1], @sum, 0);
end
node_lengths = vecnorm(node_area_vectors, 2, 2);
boundary_node = accumarray(boundary(:), 1, [n_nodes, 1], @sum, 0) > 0;
node_area_tolerance = 100 * eps(global_scale ^ 2);
undefined_node = boundary_node & node_lengths <= node_area_tolerance;
if any(undefined_node)
    error('trkg4:boundaryNormalsDegenerate', ...
        ['Cannot determine a node normal for %d boundary node(s): ', ...
         'oriented incident face areas cancel.'], sum(undefined_node));
end
node_normals = zeros(n_nodes, 3);
valid_node = node_lengths > 0;
node_normals(valid_node, :) = node_area_vectors(valid_node, :) ./ ...
    node_lengths(valid_node);
end
