function spec = trkg4_arm_montage_spec( ...
    cfg, electrode_kind, inner_from_trunk_mm, outer_from_inner_mm, varargin)
%TRKG4_ARM_MONTAGE_SPEC Parameterised four-electrode montage on both arms.
%
% The current electrodes are distal and the voltage electrodes are proximal:
%
%   right: trunk -- V_plus -- I_plus
%   left:  trunk -- V_minus -- I_minus
%
% `inner_from_trunk_mm` is measured from the CT/artificial-arm join plane.
% It is a reproducible model reference, not yet a validated anatomical
% landmark of the experimental electrode placement. `outer_from_inner_mm`
% is measured distally from the inner electrode on each arm.

p = inputParser;
addParameter(p, 'DiscDiameterMm', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'RingWidthMm', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
parse(p, varargin{:});

validateattributes(inner_from_trunk_mm, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
validateattributes(outer_from_inner_mm, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});

arms = local_load_arm_geometry(cfg);
kind = lower(string(electrode_kind));

x_right_inner = arms.right.cut_x + inner_from_trunk_mm;
x_right_outer = x_right_inner + outer_from_inner_mm;
x_left_inner = arms.left.cut_x - inner_from_trunk_mm;
x_left_outer = x_left_inner - outer_from_inner_mm;

right_top = [arms.right.center_y, ...
    arms.right.center_z + arms.right.radius];
left_top = [arms.left.center_y, ...
    arms.left.center_z + arms.left.radius];

centres = [ ...
    x_right_outer, right_top; ... % I_plus
    x_right_inner, right_top; ... % V_plus
    x_left_inner,  left_top;  ... % V_minus
    x_left_outer,  left_top];     % I_minus

radii = [arms.right.radius; arms.right.radius; ...
    arms.left.radius; arms.left.radius];

switch kind
    case {"point_disc_5mm", "disc_5mm", "point_like"}
        kind = "point_disc_5mm";
        axial_width_mm = repmat(p.Results.DiscDiameterMm, 4, 1);
        contact_inner_diameter_mm = nan(4, 1);
        nominal_area_mm2 = repmat( ...
            pi * p.Results.DiscDiameterMm^2 / 4, 4, 1);
        description = "Top-mounted 5 mm circular surface electrode";
    case {"circumferential_ring", "ring"}
        kind = "circumferential_ring";
        axial_width_mm = repmat(p.Results.RingWidthMm, 4, 1);
        contact_inner_diameter_mm = 2 * radii;
        nominal_area_mm2 = 2 * pi .* radii .* axial_width_mm;
        description = strjoin([ ...
            "Circumferential cuff whose inner diameter equals ", ...
            "the model arm diameter; axial width is 5 mm"], "");
    case {"wide_cuff_equivalent_area", "cross_section_equivalent", ...
            "area_equivalent_cuff", "large"}
        kind = "wide_cuff_equivalent_area";
        % Surface area 2*pi*R*w equals cross-section area pi*R^2.
        axial_width_mm = radii / 2;
        contact_inner_diameter_mm = 2 * radii;
        nominal_area_mm2 = pi .* radii.^2;
        description = strjoin(["Wide circumferential cuff with contact area equal ", ...
            "to the arm cross-section area; not an internal cut plane"], "");
    case "outer_planes_inner_rings"
        axial_width_mm = repmat(p.Results.RingWidthMm, 4, 1);
        axial_width_mm([1 4]) = 0;
        contact_inner_diameter_mm = 2 * radii;
        nominal_area_mm2 = 2 * pi .* radii .* axial_width_mm;
        nominal_area_mm2([1 4]) = pi * radii([1 4]).^2;
        description = "Outer internal current planes; inner circumferential voltage cuffs";
        centres(1, 2:3) = [arms.right.center_y, arms.right.center_z];
        centres(4, 2:3) = [arms.left.center_y, arms.left.center_z];
    case {"cross_section_plane", "ideal_cross_section_plane", "plane"}
        error('trkg4:retiredFourPlaneMontage', ...
            ['The four-plane montage was rejected by the author on 2026-09-08. ', ...
             'Use outer_planes_inner_rings: outer planes and inner 5 mm cuffs.']);
    otherwise
        error('trkg4:unknownArmElectrodeKind', ...
            'Unknown arm electrode kind: %s', electrode_kind);
end

labels = string(cfg.electrode_order(:));
side = ["right"; "right"; "left"; "left"];
role = ["outer_current"; "inner_voltage"; ...
    "inner_voltage"; "outer_current"];
distance_to_join_mm = [ ...
    x_right_outer - arms.right.cut_x; ...
    x_right_inner - arms.right.cut_x; ...
    arms.left.cut_x - x_left_inner; ...
    arms.left.cut_x - x_left_outer];
distance_to_outer_cap_mm = [ ...
    arms.right.cylinder_x_max - x_right_outer; ...
    arms.right.cylinder_x_max - x_right_inner; ...
    x_left_inner - arms.left.cylinder_x_min; ...
    x_left_outer - arms.left.cylinder_x_min];

% The patch must remain at least one local mesh step away from a cap or join
% after accounting for half its axial size.
required_clearance_mm = axial_width_mm / 2 + cfg.mesh_target_size_mm;
pass_clearance = distance_to_join_mm >= required_clearance_mm & ...
    distance_to_outer_cap_mm >= required_clearance_mm;

position_table = table(labels, side, role, centres(:, 1), centres(:, 2), ...
    centres(:, 3), axial_width_mm, contact_inner_diameter_mm, ...
    nominal_area_mm2, ...
    distance_to_join_mm, distance_to_outer_cap_mm, ...
    required_clearance_mm, pass_clearance, ...
    'VariableNames', {'label', 'side', 'role', 'x_mm', 'y_mm', 'z_mm', ...
    'axial_width_mm', 'contact_inner_diameter_mm', 'nominal_area_mm2', ...
    'distance_to_join_mm', ...
    'distance_to_outer_cap_mm', 'required_clearance_mm', ...
    'pass_clearance'});

spec = struct();
spec.subject_id = string(cfg.subject.id);
spec.kind = kind;
spec.description = description;
spec.inner_reference = "CT/artificial-arm join plane";
spec.inner_from_trunk_mm = inner_from_trunk_mm;
spec.outer_from_inner_mm = outer_from_inner_mm;
spec.centres_xyz_mm = centres;
spec.labels = labels;
spec.side = side;
spec.arm_radius_mm = radii;
spec.axial_width_mm = axial_width_mm;
spec.contact_inner_diameter_mm = contact_inner_diameter_mm;
spec.nominal_area_mm2 = nominal_area_mm2;
spec.cross_section_area_per_face_mm2 = pi .* radii.^2;
spec.is_internal_plane = kind == "outer_planes_inner_rings";
spec.mesh_target_size_mm = cfg.mesh_target_size_mm;
spec.position_table = position_table;
spec.pass_clearance = all(pass_clearance);
spec.arm_geometry = arms;
end

function arms = local_load_arm_geometry(cfg)
report_file = fullfile(cfg.project_root, 'output', ...
    sprintf('%s_stl_preparation_report.json', cfg.subject.id));
if ~isfile(report_file)
    error('trkg4:missingArmGeometryReport', ...
        'Missing STL preparation report: %s', report_file);
end

records = jsondecode(fileread(report_file));
if isstruct(records) && isscalar(records) && isfield(records, 'arm_extension')
    body_record = records;
elseif iscell(records)
    body_index = find(cellfun(@(record) ...
        isstruct(record) && isfield(record, 'name') && ...
        strcmp(record.name, 'body'), records), 1);
    if isempty(body_index)
        body_record = struct();
    else
        body_record = records{body_index};
    end
elseif isstruct(records) && isfield(records, 'name')
    body_index = find(strcmp({records.name}, 'body'), 1);
    if isempty(body_index)
        body_record = struct();
    else
        body_record = records(body_index);
    end
else
    body_record = struct();
end
if ~isfield(body_record, 'arm_extension')
    error('trkg4:missingArmGeometry', ...
        'The STL preparation report has no arm_extension geometry.');
end
arms = body_record.arm_extension;
end
