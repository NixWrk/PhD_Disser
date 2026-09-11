function diagnostics = trkg4_electrode_diagnostics( ...
    fmdl, centres, labels, cfg, requested, induced_faces)
%TRKG4_ELECTRODE_DIAGNOSTICS Check the realised electrode geometry.
%
%   diagnostics = trkg4_electrode_diagnostics( ...
%       fmdl_mm, centres_mm, labels, cfg)
%   diagnostics = trkg4_electrode_diagnostics( ...
%       fmdl_mm, centres_mm, labels, cfg, requested_mm, induced_faces)
%
% centres_mm are the effective centres passed to the patch builder.
% requested_mm are the original marker/pose points before snapping or
% projection; when omitted they repeat centres_mm. All coordinates and
% the boundary mesh are expected in millimetres. The diagnostic does not
% rescale them.
%
% induced_faces is an optional n-electrode cell array of boundary face
% indices. When supplied, it must be the exact face set induced by each
% fmdl.electrode(k).nodes, as returned by electrode_faces_by_area. Passing
% it avoids re-deriving the same face set in hot loops, while the consistency
% check prevents a stale or mismatched face list from changing the metric.
%
% nearest_boundary_distance_mm is the Euclidean distance from the
% requested point to the closest TRIANGULAR boundary surface. It is not a
% distance to a nearest mesh node. patch_centroid_*_mm is the geometric
% centroid of the induced contact surface, weighted by triangle area.
%
% The established geometry-failure identifier is intentionally stable:
% trkg4:electrodeDiagnostics.

if nargin < 5 || isempty(requested)
    requested = centres;
end
if nargin < 6
    induced_faces = [];
end

local_validate_inputs(fmdl, centres, labels, requested, cfg, induced_faces);
if isfield(cfg, 'length_unit') && ~isempty(cfg.length_unit) && ...
        ~ismember(lower(char(cfg.length_unit)), ...
        {'mm', 'millimeter', 'millimetre'})
    error('trkg4:electrodeDiagnosticsInput', ...
        'Electrode diagnostics expects fmdl coordinates in millimetres.');
end

nodes = double(fmdl.nodes);
boundary = double(fmdl.boundary);
n = size(centres, 1);
n_boundary_faces = size(boundary, 1);
labels = local_labels_as_cellstr(labels, n);

nearest_distance_mm = zeros(n, 1);
snap_distance_mm = zeros(n, 1);
patch_centroid_offset_mm = nan(n, 1);
patch_area_mm2 = zeros(n, 1);
patch_face_count = zeros(n, 1);
nearest_xyz = nan(n, 3);
surface_face_index = zeros(n, 1);
patch_centroid_xyz = nan(n, 3);

for k = 1:n
    [nearest_xyz(k, :), nearest_distance_mm(k), surface_face_index(k)] = ...
        local_closest_point_on_surface(requested(k, :), nodes, boundary);
    snap_distance_mm(k) = norm(centres(k, :) - requested(k, :));

    electrode_nodes = double(fmdl.electrode(k).nodes(:));
    if any(~isfinite(electrode_nodes)) || ...
            any(electrode_nodes ~= fix(electrode_nodes)) || ...
            any(electrode_nodes < 1 | electrode_nodes > size(nodes, 1))
        error('trkg4:electrodeDiagnosticsInput', ...
            'Electrode %d contains invalid node indices.', k);
    end
    electrode_nodes = unique(electrode_nodes);
    node_in_electrode = false(size(nodes, 1), 1);
    node_in_electrode(electrode_nodes) = true;

    boundary_node_flags = reshape(node_in_electrode(boundary(:)), size(boundary));
    induced_from_nodes = find(all(boundary_node_flags, 2));
    if isempty(induced_faces)
        patch_faces = induced_from_nodes;
    else
        patch_faces = double(induced_faces{k}(:));
        if any(~isfinite(patch_faces)) || ...
                any(patch_faces ~= fix(patch_faces)) || ...
                any(patch_faces < 1 | patch_faces > n_boundary_faces)
            error('trkg4:electrodeDiagnosticsInput', ...
                'Electrode %d contains invalid boundary face indices.', k);
        end
        patch_faces = unique(patch_faces);
        if ~isequal(patch_faces, induced_from_nodes)
            error('trkg4:electrodeDiagnosticsInput', ...
                ['The supplied induced face set for electrode %d does not ', ...
                 'match fmdl.electrode(%d).nodes.'], k, k);
        end
    end

    patch_face_count(k) = numel(patch_faces);
    if isempty(patch_faces)
        patch_area_mm2(k) = 0;
        continue;
    end
    patch_nodes = boundary(patch_faces, :);
    p1 = nodes(patch_nodes(:, 1), :);
    p2 = nodes(patch_nodes(:, 2), :);
    p3 = nodes(patch_nodes(:, 3), :);
    face_area = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);
    if any(~isfinite(face_area)) || any(face_area <= 0)
        error('trkg4:electrodeDiagnosticsInput', ...
            'Electrode %d contains a degenerate boundary triangle.', k);
    end
    face_centroid = (p1 + p2 + p3) / 3;
    patch_area_mm2(k) = sum(face_area);
    patch_centroid_xyz(k, :) = (face_area.' * face_centroid) / ...
        patch_area_mm2(k);
    patch_centroid_offset_mm(k) = norm( ...
        patch_centroid_xyz(k, :) - centres(k, :));
end

diagnostics = table( ...
    labels(:), ...
    requested(:, 1), requested(:, 2), requested(:, 3), ...
    centres(:, 1), centres(:, 2), centres(:, 3), ...
    snap_distance_mm, ...
    nearest_distance_mm, ...
    nearest_xyz(:, 1), nearest_xyz(:, 2), nearest_xyz(:, 3), ...
    surface_face_index, ...
    patch_centroid_offset_mm, ...
    patch_centroid_xyz(:, 1), patch_centroid_xyz(:, 2), ...
    patch_centroid_xyz(:, 3), ...
    patch_area_mm2, patch_face_count, ...
    'VariableNames', { ...
    'label', ...
    'requested_x_mm', 'requested_y_mm', 'requested_z_mm', ...
    'effective_x_mm', 'effective_y_mm', 'effective_z_mm', ...
    'snap_distance_mm', ...
    'nearest_boundary_distance_mm', ...
    'nearest_x_mm', 'nearest_y_mm', 'nearest_z_mm', ...
    'surface_face_index', ...
    'patch_centroid_offset_mm', ...
    'patch_centroid_x_mm', 'patch_centroid_y_mm', ...
    'patch_centroid_z_mm', ...
    'patch_area_mm2', 'patch_face_count'});

if local_diagnostics_verbose(cfg)
    fprintf('\nElectrode placement diagnostics:\n');
    for k = 1:n
        fprintf(['  %-8s requested-to-surface %.3f mm, snap %.3f mm, ', ...
            'patch centroid offset %.3f mm, area %.3f mm^2 (%d faces)\n'], ...
            labels{k}, nearest_distance_mm(k), snap_distance_mm(k), ...
            patch_centroid_offset_mm(k), patch_area_mm2(k), ...
            patch_face_count(k));
    end
end

local_check_thresholds(diagnostics, cfg);
local_write_diagnostics(diagnostics, cfg);
end

function local_validate_inputs(fmdl, centres, labels, requested, cfg, induced_faces)
if ~isstruct(fmdl) || ~all(isfield(fmdl, {'nodes', 'boundary', 'electrode'}))
    error('trkg4:electrodeDiagnosticsInput', ...
        'FEM model must contain nodes, boundary, and electrode fields.');
end
if size(fmdl.nodes, 2) ~= 3 || size(fmdl.boundary, 2) ~= 3
    error('trkg4:electrodeDiagnosticsInput', ...
        'FEM nodes must be N-by-3 and boundary faces M-by-3.');
end
if size(centres, 2) ~= 3 || size(requested, 2) ~= 3 || ...
        size(centres, 1) ~= size(requested, 1) || ...
        any(~isfinite(centres), 'all') || any(~isfinite(requested), 'all')
    error('trkg4:electrodeDiagnosticsInput', ...
        'Effective and requested centres must be finite N-by-3 arrays.');
end
if numel(fmdl.electrode) ~= size(centres, 1)
    error('trkg4:electrodeDiagnosticsInput', ...
        'The number of electrodes does not match the centre array.');
end
if numel(labels) ~= size(centres, 1)
    error('trkg4:electrodeDiagnosticsInput', ...
        'The number of labels does not match the electrode count.');
end
if ~isstruct(cfg)
    error('trkg4:electrodeDiagnosticsInput', 'cfg must be a configuration struct.');
end
required_cfg = {'max_electrode_surface_distance_mm', ...
    'max_electrode_patch_centroid_offset_mm', ...
    'min_electrode_patch_area_fraction', ...
    'max_electrode_patch_area_ratio', 'electrode_area'};
if ~all(isfield(cfg, required_cfg))
    error('trkg4:electrodeDiagnosticsInput', ...
        'cfg is missing one or more electrode QC thresholds.');
end
if ~isempty(induced_faces) && ...
        (~iscell(induced_faces) || numel(induced_faces) ~= size(centres, 1))
    error('trkg4:electrodeDiagnosticsInput', ...
        'induced_faces must be an n-electrode cell array.');
end
end

function labels = local_labels_as_cellstr(labels, n)
if ischar(labels)
    if size(labels, 1) == n
        labels = cellstr(labels);
    else
        labels = {labels};
    end
elseif isstring(labels)
    labels = cellstr(labels(:));
else
    labels = labels(:);
end
end

function verbose = local_diagnostics_verbose(cfg)
verbose = true;
if isfield(cfg, 'electrode_diagnostics_verbose') && ...
        ~isempty(cfg.electrode_diagnostics_verbose)
    verbose = logical(cfg.electrode_diagnostics_verbose);
end
end

function [closest_point, distance, face_index] = ...
    local_closest_point_on_surface(point, nodes, faces)
%LOCAL_CLOSEST_POINT_ON_SURFACE Exact distance to a triangle mesh.
% The plane candidate is used only when its barycentric coordinates are
% inside the triangle; otherwise the closest point on one of the three
% segments is used. The calculation is vectorized over all boundary faces.

if isempty(faces)
    error('trkg4:electrodeDiagnosticsInput', ...
        'The FEM model has no boundary triangles.');
end
p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
ab = p2 - p1;
ac = p3 - p1;
bc = p3 - p2;
normal = cross(ab, ac, 2);
bbox_span = max(nodes, [], 1) - min(nodes, [], 1);
global_scale = max(1, max(bbox_span));
face_edge_scale = max([ ...
    vecnorm(ab, 2, 2), ...
    vecnorm(ac, 2, 2), ...
    vecnorm(bc, 2, 2)], [], 2);
area_tolerance = 100 * eps(max(1, face_edge_scale .^ 2));
length_tolerance = 100 * eps(max(1, face_edge_scale));
normal_squared = sum(normal.^2, 2);
if any(sqrt(normal_squared) <= area_tolerance)
    error('trkg4:electrodeDiagnosticsInput', ...
        'The boundary contains a degenerate triangle.');
end

% Orthogonal projection to each triangle plane.
signed_distance = sum((point - p1) .* normal, 2) ./ normal_squared;
plane_point = point - signed_distance .* normal;
d00 = sum(ab .* ab, 2);
d01 = sum(ab .* ac, 2);
d11 = sum(ac .* ac, 2);
v2 = plane_point - p1;
d20 = sum(v2 .* ab, 2);
d21 = sum(v2 .* ac, 2);
denominator = normal_squared;
v = (d11 .* d20 - d01 .* d21) ./ denominator;
w = (d00 .* d21 - d01 .* d20) ./ denominator;
u = 1 - v - w;
inside = u >= -1e-12 & v >= -1e-12 & w >= -1e-12;
plane_distance_squared = inf(size(inside));
plane_distance_squared(inside) = sum( ...
    (plane_point(inside, :) - point).^2, 2);

[ab_point, ab_distance_squared] = ...
    local_closest_point_on_segments(point, p1, p2, ab, length_tolerance);
[bc_point, bc_distance_squared] = ...
    local_closest_point_on_segments(point, p2, p3, bc, length_tolerance);
[ca_point, ca_distance_squared] = ...
    local_closest_point_on_segments(point, p3, p1, -ac, length_tolerance);

candidate_distance_squared = [plane_distance_squared, ...
    ab_distance_squared, bc_distance_squared, ca_distance_squared];
[best_distance_squared, candidate_kind] = ...
    min(candidate_distance_squared, [], 2);
[distance_squared, face_index] = min(best_distance_squared);
candidate_points = {plane_point, ab_point, bc_point, ca_point};
point_rows = candidate_kind(face_index);
closest_point = candidate_points{point_rows}(face_index, :);
distance = sqrt(max(0, distance_squared));
end

function [closest_point, distance_squared] = ...
    local_closest_point_on_segments(point, start_point, end_point, edge, tolerance)
edge_squared = sum(edge .* edge, 2);
if any(edge_squared <= tolerance .^ 2)
    error('trkg4:electrodeDiagnosticsInput', ...
        'The boundary contains a zero-length triangle edge.');
end
parameter = sum((point - start_point) .* edge, 2) ./ edge_squared;
parameter = max(0, min(1, parameter));
closest_point = start_point + parameter .* edge;
distance_squared = sum((closest_point - point).^2, 2);
end

function local_check_thresholds(diagnostics, cfg)
bad_surface = ~isfinite(diagnostics.nearest_boundary_distance_mm) | ...
    diagnostics.nearest_boundary_distance_mm > ...
    cfg.max_electrode_surface_distance_mm;
bad_offset = ~isfinite(diagnostics.patch_centroid_offset_mm) | ...
    diagnostics.patch_centroid_offset_mm > ...
    cfg.max_electrode_patch_centroid_offset_mm;
bad_area_low = ~isfinite(diagnostics.patch_area_mm2) | ...
    diagnostics.patch_area_mm2 < ...
    cfg.min_electrode_patch_area_fraction * cfg.electrode_area;
bad_area_high = isfinite(diagnostics.patch_area_mm2) & ...
    diagnostics.patch_area_mm2 > cfg.max_electrode_patch_area_ratio * ...
    cfg.electrode_area;

if ~any(bad_surface | bad_offset | bad_area_low | bad_area_high)
    return;
end

reasons = {};
if any(bad_surface)
    reasons{end + 1} = sprintf( ...
        ['%d electrode(s) are more than %.3g mm from the triangular ', ...
         'body surface.'], sum(bad_surface), ...
        cfg.max_electrode_surface_distance_mm);
end
if any(bad_offset)
    reasons{end + 1} = sprintf( ...
        ['%d area-weighted patch centroid(s) drifted more than %.3g mm ', ...
         'from the requested centre.'], sum(bad_offset), ...
        cfg.max_electrode_patch_centroid_offset_mm);
end
if any(bad_area_low)
    reasons{end + 1} = sprintf( ...
        ['%d patch(es) have less than %.0f%% of the requested %.4g ', ...
         'mm^2 contact area.'], sum(bad_area_low), ...
        100 * cfg.min_electrode_patch_area_fraction, cfg.electrode_area);
end
if any(bad_area_high)
    reasons{end + 1} = sprintf( ...
        ['%d patch(es) exceed the requested %.4g mm^2 area by more ', ...
         'than a factor of %.3g.'], sum(bad_area_high), ...
        cfg.electrode_area, cfg.max_electrode_patch_area_ratio);
end
message = strjoin(reasons, ' ');

if local_diagnostics_verbose(cfg)
    fprintf('Electrode geometry QC violation: %s\n', message);
end
fail = isfield(cfg, 'fail_on_electrode_diagnostics') && ...
    logical(cfg.fail_on_electrode_diagnostics);
if fail
    error('trkg4:electrodeDiagnostics', '%s', message);
else
    warning('trkg4:electrodeDiagnostics', '%s', message);
end
end

function local_write_diagnostics(diagnostics, cfg)
if ~isfield(cfg, 'electrode_diagnostics_file') || ...
        isempty(cfg.electrode_diagnostics_file)
    return;
end
out_dir = fileparts(cfg.electrode_diagnostics_file);
if ~isempty(out_dir) && ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
writetable(diagnostics, cfg.electrode_diagnostics_file);
fprintf('Saved electrode diagnostics: %s\n', cfg.electrode_diagnostics_file);
end
