function diagnostics = trkg4_electrode_diagnostics(fmdl, centres, labels, cfg, requested)
%TRKG4_ELECTRODE_DIAGNOSTICS Check where requested electrode centres landed.
%
% `centres` are the effective centres the patch builder was given. `requested`
% are the original coordinates from the electrode CSV, before any snapping.
% They are reported separately on purpose:
%
%   * nearest_boundary_distance_mm is measured from the REQUESTED point. It
%     answers the only question this diagnostic exists for - was the marker
%     placed on the real skin, or on a cavity wall, a leftover hole, or a
%     point floating inside the body? Measured from a snapped centre it is
%     identically zero and detects nothing.
%
%   * snap_distance_mm is how far snapping had to move the centre. It is
%     bounded by the local node spacing and is a mesh-quality number, not a
%     placement-quality one.
%
% Older callers may omit `requested`; the requested columns then repeat the
% effective centres and snap_distance_mm is zero.

if nargin < 5 || isempty(requested)
    requested = centres;
end

n = numel(fmdl.electrode);
boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);

nearest_distance_mm = zeros(n, 1);
snap_distance_mm = zeros(n, 1);
patch_centroid_offset_mm = zeros(n, 1);
patch_area_mm2 = zeros(n, 1);
patch_face_count = zeros(n, 1);
nearest_xyz = zeros(n, 3);
patch_centroid_xyz = zeros(n, 3);

for k = 1:n
    d = sqrt(sum((boundary_xyz - requested(k, :)).^2, 2));
    [nearest_distance_mm(k), idx] = min(d);
    nearest_xyz(k, :) = boundary_xyz(idx, :);
    snap_distance_mm(k) = norm(centres(k, :) - requested(k, :));

    electrode_nodes = fmdl.electrode(k).nodes(:);
    patch_centroid_xyz(k, :) = mean(fmdl.nodes(electrode_nodes, :), 1);
    patch_centroid_offset_mm(k) = norm(patch_centroid_xyz(k, :) - centres(k, :));

    % Induced faces: every boundary face whose three nodes belong to the
    % electrode. This is the surface the complete electrode model integrates
    % over, and it is the same measure electrode_faces_by_area targets.
    is_patch_face = all(ismember(fmdl.boundary, electrode_nodes), 2);
    patch_faces = fmdl.boundary(is_patch_face, :);
    patch_face_count(k) = size(patch_faces, 1);
    patch_area_mm2(k) = local_boundary_area(fmdl.nodes, patch_faces);
end

diagnostics = table( ...
    labels(:), ...
    requested(:, 1), requested(:, 2), requested(:, 3), ...
    centres(:, 1), centres(:, 2), centres(:, 3), ...
    snap_distance_mm, ...
    nearest_distance_mm, ...
    nearest_xyz(:, 1), nearest_xyz(:, 2), nearest_xyz(:, 3), ...
    patch_centroid_offset_mm, ...
    patch_centroid_xyz(:, 1), patch_centroid_xyz(:, 2), patch_centroid_xyz(:, 3), ...
    patch_area_mm2, patch_face_count, ...
    'VariableNames', { ...
    'label', ...
    'requested_x_mm', 'requested_y_mm', 'requested_z_mm', ...
    'effective_x_mm', 'effective_y_mm', 'effective_z_mm', ...
    'snap_distance_mm', ...
    'nearest_boundary_distance_mm', ...
    'nearest_x_mm', 'nearest_y_mm', 'nearest_z_mm', ...
    'patch_centroid_offset_mm', ...
    'patch_centroid_x_mm', 'patch_centroid_y_mm', 'patch_centroid_z_mm', ...
    'patch_area_mm2', 'patch_face_count'});

fprintf('\nElectrode placement diagnostics:\n');
for k = 1:n
    fprintf(['  %-8s requested-to-skin %.3f mm, snap %.3f mm, patch ', ...
        'centroid offset %.3f mm, area %.3f mm^2 (%d faces)\n'], ...
        labels(k), nearest_distance_mm(k), snap_distance_mm(k), ...
        patch_centroid_offset_mm(k), patch_area_mm2(k), patch_face_count(k));
end

local_check_thresholds(diagnostics, cfg);
local_write_diagnostics(diagnostics, cfg);
end

function area = local_boundary_area(nodes, faces)
if isempty(faces)
    area = 0;
    return;
end
p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
area = sum(0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2));
end

function local_check_thresholds(diagnostics, cfg)
bad_surface = diagnostics.nearest_boundary_distance_mm > ...
    cfg.max_electrode_surface_distance_mm;
bad_offset = diagnostics.patch_centroid_offset_mm > ...
    cfg.max_electrode_patch_centroid_offset_mm;

% Patch area, both directions.
%
% The lower bound is an invariant guard rather than a live test: when the
% electrodes come from electrode_faces_by_area the induced area is >= the
% target by construction. It still matters here, because this function is
% also run on models whose `fmdl.electrode` came from elsewhere - a reloaded
% .mat, an older result, a hand-assigned set - and there the invariant is a
% real assumption that can be violated.
%
% The upper bound is a live test. Growth stops on the first step that reaches
% the target, but one added face can induce several faces at once, so a patch
% can overshoot. A large overshoot means the electrode covers noticeably more
% skin than the physical one and changes the current density under it.
bad_area = diagnostics.patch_area_mm2 < ...
    cfg.min_electrode_patch_area_fraction * cfg.electrode_area;
bad_area_high = diagnostics.patch_area_mm2 > ...
    cfg.max_electrode_patch_area_ratio * cfg.electrode_area;

if ~any(bad_surface | bad_offset | bad_area | bad_area_high)
    return;
end

reasons = {};
if any(bad_surface)
    reasons{end + 1} = sprintf( ...
        ['%d electrode(s) sit more than %.3g mm from the body boundary. ', ...
         'Check that the coordinates are on the intended real skin ', ...
         'surface, not on a cavity or a leftover hole.'], ...
        sum(bad_surface), cfg.max_electrode_surface_distance_mm);
end
if any(bad_offset)
    reasons{end + 1} = sprintf( ...
        ['%d patch centroid(s) drifted more than %.3g mm from the ', ...
         'requested centre, so the patch grew asymmetrically.'], ...
        sum(bad_offset), cfg.max_electrode_patch_centroid_offset_mm);
end
if any(bad_area)
    reasons{end + 1} = sprintf( ...
        ['%d patch(es) reached less than %.0f%% of the requested %.4g ', ...
         'mm^2 contact area.'], sum(bad_area), ...
        100 * cfg.min_electrode_patch_area_fraction, cfg.electrode_area);
end
if any(bad_area_high)
    reasons{end + 1} = sprintf( ...
        ['%d patch(es) overshot the requested %.4g mm^2 contact area by ', ...
         'more than a factor of %.3g; the mesh is probably too coarse ', ...
         'under those electrodes.'], sum(bad_area_high), ...
        cfg.electrode_area, cfg.max_electrode_patch_area_ratio);
end
message = strjoin(reasons, ' ');

if cfg.fail_on_electrode_diagnostics
    error('trkg4:electrodeDiagnostics', '%s', message);
else
    warning('trkg4:electrodeDiagnostics', '%s', message);
end
end

function local_write_diagnostics(diagnostics, cfg)
if isempty(cfg.electrode_diagnostics_file)
    return;
end

out_dir = fileparts(cfg.electrode_diagnostics_file);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
writetable(diagnostics, cfg.electrode_diagnostics_file);
fprintf('Saved electrode diagnostics: %s\n', cfg.electrode_diagnostics_file);
end
