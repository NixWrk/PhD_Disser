function [sample_points, diagnostics] = trkg4_surface_path( ...
    vertices, faces, guide_points, sample_distances)
%TRKG4_SURFACE_PATH Build and sample a shortest path on a triangular surface.
%
%   [SAMPLE_POINTS, DIAGNOSTICS] = TRKG4_SURFACE_PATH(VERTICES, FACES,
%   GUIDE_POINTS, SAMPLE_DISTANCES) constructs a weighted graph from the
%   unique edges of the triangular surface, snaps every guide point to its
%   nearest surface vertex, connects consecutive snapped vertices by
%   shortest graph paths, and samples the resulting polyline at the
%   requested cumulative distances.
%
%   VERTICES is an N-by-3 numeric array. FACES is an M-by-3 array of
%   positive integer vertex indices. GUIDE_POINTS is a K-by-3 array with
%   K >= 2. SAMPLE_DISTANCES is a finite numeric vector expressed in the
%   same units as VERTICES.
%
%   The path uses Euclidean lengths of mesh edges. Only consecutive
%   duplicate vertex indices are removed when graph segments are joined;
%   non-consecutive repetitions are retained. The diagnostic structure
%   contains the input and snapped guide points, their displacements, the
%   complete path, and interpolation diagnostics.

if nargin ~= 4
    error('trkg4:invalidInputCount', ...
        'Expected vertices, faces, guide_points, and sample_distances.');
end

local_validate_matrix(vertices, 'vertices', 3, 3);
local_validate_matrix(faces, 'faces', 3, 1);
local_validate_matrix(guide_points, 'guide_points', 3, 2);
local_validate_distances(sample_distances);

vertices = double(vertices);
faces = double(faces);
guide_points = double(guide_points);
sample_distances = double(sample_distances(:));

if any(faces(:) ~= fix(faces(:))) || any(faces(:) < 1) || ...
        any(faces(:) > size(vertices, 1))
    error('trkg4:invalidFaces', ...
        'Faces must contain positive integer indices into vertices.');
end

if any(faces(:, 1) == faces(:, 2)) || ...
        any(faces(:, 2) == faces(:, 3)) || ...
        any(faces(:, 1) == faces(:, 3))
    error('trkg4:degenerateFace', ...
        'Every triangular face must contain three distinct vertex indices.');
end

triangle_vectors_1 = vertices(faces(:, 2), :) - vertices(faces(:, 1), :);
triangle_vectors_2 = vertices(faces(:, 3), :) - vertices(faces(:, 1), :);
triangle_double_area = sqrt(sum(cross( ...
    triangle_vectors_1, triangle_vectors_2, 2).^2, 2));
if any(triangle_double_area <= 0)
    error('trkg4:zeroAreaFace', ...
        'Every triangular face must have non-zero geometric area.');
end

[surface_graph, surface_vertex_indices, number_of_surface_edges] = ...
    local_build_surface_graph(vertices, faces);

[guide_vertex_indices, resolved_guide_points] = ...
    local_snap_guides(vertices, surface_vertex_indices, guide_points);

path_nodes = zeros(0, 1);
for guide_index = 1:(numel(guide_vertex_indices) - 1)
    segment_nodes = local_shortest_path( ...
        surface_graph, guide_vertex_indices(guide_index), ...
        guide_vertex_indices(guide_index + 1));
    path_nodes = local_append_segment(path_nodes, segment_nodes);
end

% Remove only consecutive duplicate nodes at segment joins. A repeated
% node occurring later in the path remains part of the path history.
path_nodes = local_remove_consecutive_duplicates(path_nodes);
local_validate_path_continuity(path_nodes, surface_graph);
path_points = vertices(path_nodes, :);

[sample_points, sample_diagnostics] = ...
    trkg4_sample_surface_path(path_points, sample_distances);

guide_displacements = resolved_guide_points - guide_points;
guide_offsets = sqrt(sum(guide_displacements.^2, 2));

diagnostics = struct();
diagnostics.original_guide_points_xyz_mm = guide_points;
diagnostics.resolved_guide_points_xyz_mm = resolved_guide_points;
diagnostics.guide_vertex_indices = guide_vertex_indices;
diagnostics.guide_displacements_xyz_mm = guide_displacements;
diagnostics.guide_offsets_mm = guide_offsets;
diagnostics.number_of_surface_vertices = numel(surface_vertex_indices);
diagnostics.number_of_surface_edges = number_of_surface_edges;
diagnostics.path_nodes = path_nodes;
diagnostics.path_points_xyz_mm = path_points;
diagnostics.path_cumulative_lengths_mm = ...
    sample_diagnostics.cumulative_lengths_mm;
diagnostics.path_length_mm = sample_diagnostics.path_length_mm;
diagnostics.sample_distances_mm = sample_diagnostics.distances_mm;
diagnostics.sample_segment_indices = ...
    sample_diagnostics.segment_indices;
diagnostics.sample_interpolation_fractions = ...
    sample_diagnostics.interpolation_fractions;
diagnostics.path_is_continuous = true;
end

function local_validate_matrix(value, name, number_of_columns, minimum_rows)
if ~isnumeric(value) || ~isreal(value) || ~ismatrix(value) || ...
        size(value, 2) ~= number_of_columns || ...
        size(value, 1) < minimum_rows || any(~isfinite(value(:)))
    error('trkg4:invalidNumericInput', ...
        '%s must be a finite real numeric array with %d columns and at least %d rows.', ...
        name, number_of_columns, minimum_rows);
end
end

function local_validate_distances(distances)
if ~isnumeric(distances) || ~isreal(distances) || ...
        (~isvector(distances) && ~isscalar(distances)) || ...
        isempty(distances) || any(~isfinite(distances(:)))
    error('trkg4:invalidDistances', ...
        'sample_distances must be a non-empty finite real numeric vector.');
end
end

function [surface_graph, surface_vertex_indices, number_of_surface_edges] = ...
    local_build_surface_graph(vertices, faces)
raw_edges = [faces(:, [1 2]); faces(:, [2 3]); faces(:, [3 1])];
unique_edges = unique(sort(raw_edges, 2), 'rows');
surface_vertex_indices = unique(faces(:));

edge_vectors = vertices(unique_edges(:, 2), :) - ...
    vertices(unique_edges(:, 1), :);
edge_lengths = sqrt(sum(edge_vectors.^2, 2));
if any(~isfinite(edge_lengths)) || any(edge_lengths <= 0)
    error('trkg4:zeroLengthEdge', ...
        'Every unique surface edge must have a positive finite length.');
end

number_of_vertices = size(vertices, 1);
surface_graph = graph( ...
    unique_edges(:, 1), unique_edges(:, 2), edge_lengths, ...
    number_of_vertices);
number_of_surface_edges = numedges(surface_graph);
end

function [indices, resolved_points] = ...
    local_snap_guides(vertices, surface_vertex_indices, guide_points)
number_of_guides = size(guide_points, 1);
indices = zeros(number_of_guides, 1);
resolved_points = zeros(number_of_guides, 3);
surface_points = vertices(surface_vertex_indices, :);
for guide_index = 1:number_of_guides
    displacement = surface_points - guide_points(guide_index, :);
    squared_distance = sum(displacement.^2, 2);
    [~, surface_index] = min(squared_distance);
    indices(guide_index) = surface_vertex_indices(surface_index);
    resolved_points(guide_index, :) = vertices(indices(guide_index), :);
end
end

function node_path = local_shortest_path(surface_graph, start_node, end_node)
if start_node == end_node
    node_path = start_node;
    return;
end

% MATLAB computes the shortest path using the graph weights. The same
% graph object is reused for every consecutive guide-point pair.
node_path = shortestpath(surface_graph, start_node, end_node);
if isempty(node_path)
    error('trkg4:noSurfacePath', ...
        ['No connected surface path exists between guide vertices %d ', ...
        'and %d.'], start_node, end_node);
end
node_path = node_path(:);
end

function combined_path = local_append_segment(existing_path, segment_path)
if isempty(existing_path)
    combined_path = segment_path(:);
    return;
end

if existing_path(end) ~= segment_path(1)
    error('trkg4:brokenSurfacePath', ...
        'Neighbouring shortest-path segments do not share an endpoint.');
end
combined_path = [existing_path; segment_path(2:end)];
end

function path_nodes = local_remove_consecutive_duplicates(path_nodes)
if numel(path_nodes) < 2
    return;
end
keep = [true; path_nodes(2:end) ~= path_nodes(1:end - 1)];
path_nodes = path_nodes(keep);
end

function local_validate_path_continuity(path_nodes, surface_graph)
if isempty(path_nodes)
    error('trkg4:emptySurfacePath', 'The resolved surface path is empty.');
end
if numel(path_nodes) < 2
    return;
end

edge_indices = findedge(surface_graph, path_nodes(1:end - 1), ...
    path_nodes(2:end));
if any(edge_indices == 0)
    error('trkg4:discontinuousSurfacePath', ...
        'The resolved path contains a pair of non-adjacent surface nodes.');
end
end
