function diagnostics = trkg4_electrode_diagnostics(fmdl, centres, labels, cfg)
%TRKG4_ELECTRODE_DIAGNOSTICS Check where requested electrode centres landed.

n = numel(fmdl.electrode);
boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);

nearest_distance_mm = zeros(n, 1);
patch_centroid_offset_mm = zeros(n, 1);
patch_area_mm2 = zeros(n, 1);
nearest_xyz = zeros(n, 3);
patch_centroid_xyz = zeros(n, 3);

for k = 1:n
    d = sqrt(sum((boundary_xyz - centres(k, :)).^2, 2));
    [nearest_distance_mm(k), idx] = min(d);
    nearest_xyz(k, :) = boundary_xyz(idx, :);

    electrode_nodes = fmdl.electrode(k).nodes(:);
    patch_centroid_xyz(k, :) = mean(fmdl.nodes(electrode_nodes, :), 1);
    patch_centroid_offset_mm(k) = norm(patch_centroid_xyz(k, :) - centres(k, :));

    is_patch_face = all(ismember(fmdl.boundary, electrode_nodes), 2);
    patch_area_mm2(k) = local_boundary_area(fmdl.nodes, fmdl.boundary(is_patch_face, :));
end

diagnostics = table( ...
    labels(:), ...
    centres(:, 1), centres(:, 2), centres(:, 3), ...
    nearest_distance_mm, ...
    nearest_xyz(:, 1), nearest_xyz(:, 2), nearest_xyz(:, 3), ...
    patch_centroid_offset_mm, ...
    patch_centroid_xyz(:, 1), patch_centroid_xyz(:, 2), patch_centroid_xyz(:, 3), ...
    patch_area_mm2, ...
    'VariableNames', { ...
    'label', ...
    'requested_x_mm', 'requested_y_mm', 'requested_z_mm', ...
    'nearest_boundary_distance_mm', ...
    'nearest_x_mm', 'nearest_y_mm', 'nearest_z_mm', ...
    'patch_centroid_offset_mm', ...
    'patch_centroid_x_mm', 'patch_centroid_y_mm', 'patch_centroid_z_mm', ...
    'patch_area_mm2'});

fprintf('\nElectrode placement diagnostics:\n');
for k = 1:n
    fprintf('  %-8s nearest %.3f mm, patch centroid offset %.3f mm, area %.3f mm^2\n', ...
        labels(k), nearest_distance_mm(k), patch_centroid_offset_mm(k), patch_area_mm2(k));
end

local_check_thresholds(diagnostics, cfg);
local_write_diagnostics(diagnostics, cfg);
end

function area = local_boundary_area(nodes, faces)
area = 0;
for i = 1:size(faces, 1)
    points = nodes(faces(i, :), :);
    area = area + 0.5 * norm(cross(points(2, :) - points(1, :), points(3, :) - points(1, :)));
end
end

function local_check_thresholds(diagnostics, cfg)
bad_surface = diagnostics.nearest_boundary_distance_mm > cfg.max_electrode_surface_distance_mm;
bad_offset = diagnostics.patch_centroid_offset_mm > cfg.max_electrode_patch_centroid_offset_mm;
if ~any(bad_surface | bad_offset)
    return;
end

message = ['At least one electrode is far from its selected body boundary patch. ', ...
    'Check that electrode coordinates are on the intended real skin surface, ', ...
    'not on a cavity or a leftover hole.'];
if cfg.fail_on_electrode_diagnostics
    error(message);
else
    warning(message);
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
