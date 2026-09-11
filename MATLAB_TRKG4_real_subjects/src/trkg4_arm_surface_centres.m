function [centres_xyz_mm, diagnostics] = trkg4_arm_surface_centres( ...
    fmdl, right_guide_points_xyz_mm, left_guide_points_xyz_mm, ...
    inner_from_reference_mm, outer_from_inner_mm)
%TRKG4_ARM_SURFACE_CENTRES Resolve a four-electrode arm montage on skin.
%
% The right and left guide polylines must run from a proximal reference
% towards the distal arm. Electrode distances are measured along the
% resolved surface paths. Scalar distances are applied to both arms; a
% two-element vector specifies [right, left] values independently.
%
% Returned centres follow the project electrode order:
%   I_plus, V_plus, V_minus, I_minus.

if nargin ~= 5
    error('trkg4:invalidInputCount', ...
        ['Expected fmdl, right and left guide points, inner distance, ', ...
         'and outer distance.']);
end
if ~isstruct(fmdl) || ~isfield(fmdl, 'nodes') || ...
        ~isfield(fmdl, 'boundary')
    error('trkg4:invalidSurfaceModel', ...
        'fmdl must contain nodes and triangular boundary fields.');
end

symmetric_distance_request = isscalar(inner_from_reference_mm) && ...
    isscalar(outer_from_inner_mm);
inner_mm = local_side_values(inner_from_reference_mm, ...
    'inner_from_reference_mm');
outer_mm = local_side_values(outer_from_inner_mm, ...
    'outer_from_inner_mm');
if any(outer_mm <= 0)
    error('trkg4:invalidArmSurfaceDistance', ...
        'outer_from_inner_mm must contain positive distances.');
end

right_distances_mm = [inner_mm(1) + outer_mm(1); inner_mm(1)];
left_distances_mm = [inner_mm(2); inner_mm(2) + outer_mm(2)];

[right_centres, right_path] = trkg4_surface_path( ...
    fmdl.nodes, fmdl.boundary, right_guide_points_xyz_mm, ...
    right_distances_mm);
[left_centres, left_path] = trkg4_surface_path( ...
    fmdl.nodes, fmdl.boundary, left_guide_points_xyz_mm, ...
    left_distances_mm);

centres_xyz_mm = [right_centres; left_centres];
labels = ["I_plus"; "V_plus"; "V_minus"; "I_minus"];
sides = ["right"; "right"; "left"; "left"];
roles = ["outer_current"; "inner_voltage"; ...
    "inner_voltage"; "outer_current"];
distance_from_reference_mm = [right_distances_mm; left_distances_mm];

centre_table = table(labels, sides, roles, ...
    distance_from_reference_mm, centres_xyz_mm(:, 1), ...
    centres_xyz_mm(:, 2), centres_xyz_mm(:, 3), ...
    'VariableNames', {'label', 'side', 'role', ...
    'distance_from_reference_mm', 'x_mm', 'y_mm', 'z_mm'});

diagnostics = struct();
diagnostics.coordinate_system = "surface_path_v1";
diagnostics.reference_definition = ...
    "first resolved guide point on each proximal-to-distal surface path";
diagnostics.inner_from_reference_mm = inner_mm;
diagnostics.outer_from_inner_mm = outer_mm;
diagnostics.right_path = right_path;
diagnostics.left_path = left_path;
diagnostics.centre_table = centre_table;
diagnostics.right_realised_separation_mm = ...
    abs(right_distances_mm(1) - right_distances_mm(2));
diagnostics.left_realised_separation_mm = ...
    abs(left_distances_mm(2) - left_distances_mm(1));
diagnostics.symmetric_distance_request = symmetric_distance_request;
if symmetric_distance_request
    diagnostics.placement_mode = "symmetric_equal_path_distances";
else
    diagnostics.placement_mode = "independent_path_distances";
end
end

function values = local_side_values(value, name)
if ~isnumeric(value) || ~isreal(value) || isempty(value) || ...
        ~isvector(value) || ~ismember(numel(value), [1 2]) || ...
        any(~isfinite(value(:))) || any(value(:) < 0)
    error('trkg4:invalidArmSurfaceDistance', ...
        '%s must contain one or two finite non-negative distances.', name);
end
values = double(value(:));
if isscalar(values)
    values = repmat(values, 2, 1);
end
end
