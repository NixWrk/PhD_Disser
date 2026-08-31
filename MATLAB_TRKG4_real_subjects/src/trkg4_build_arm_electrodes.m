function [electrodes, diagnostics] = trkg4_build_arm_electrodes( ...
    fmdl, spec, z_contact)
%TRKG4_BUILD_ARM_ELECTRODES Build discs, rings or area-equivalent cuffs.
%
% Geometry is built in mesh units (millimetres in the current project).
% The returned EIDORS electrode node sets are attached before the model is
% scaled to SI units.

if ~spec.pass_clearance
    failed = spec.position_table(~spec.position_table.pass_clearance, :);
    error('trkg4:unsafeArmElectrodePosition', ...
        ['Electrode coordinates intersect the protected cap/join margin. ', ...
         'Failed labels: %s'], strjoin(failed.label, ', '));
end

switch spec.kind
    case "point_disc_5mm"
        if any(abs(spec.nominal_area_mm2 - spec.nominal_area_mm2(1)) > eps)
            error('trkg4:unequalDiscAreas', ...
                'The canonical area builder requires equal disc areas.');
        end
        [electrodes, induced_faces] = electrode_faces_by_area( ...
            fmdl, spec.centres_xyz_mm, spec.nominal_area_mm2(1), z_contact);
    case {"circumferential_ring", "cross_section_equivalent"}
        [electrodes, induced_faces] = ...
            local_build_circumferential_patches(fmdl, spec, z_contact);
    otherwise
        error('trkg4:unknownArmElectrodeKind', ...
            'Unsupported arm electrode kind: %s', spec.kind);
end

diagnostics = local_patch_diagnostics(fmdl, electrodes, induced_faces, spec);

all_nodes = cell(numel(electrodes), 1);
for k = 1:numel(electrodes)
    all_nodes{k} = electrodes(k).nodes(:);
end
for first = 1:numel(electrodes) - 1
    for second = first + 1:numel(electrodes)
        if ~isempty(intersect(all_nodes{first}, all_nodes{second}))
            error('trkg4:armElectrodeOverlap', ...
                'Electrodes %s and %s share FEM nodes.', ...
                spec.labels(first), spec.labels(second));
        end
    end
end
end

function [electrodes, induced_faces] = ...
    local_build_circumferential_patches(fmdl, spec, z_contact)
faces = fmdl.boundary;
p1 = fmdl.nodes(faces(:, 1), :);
p2 = fmdl.nodes(faces(:, 2), :);
p3 = fmdl.nodes(faces(:, 3), :);
face_centres = (p1 + p2 + p3) / 3;

n = numel(spec.labels);
electrodes = struct('nodes', cell(n, 1), 'z_contact', cell(n, 1));
induced_faces = cell(n, 1);
used_nodes = false(size(fmdl.nodes, 1), 1);

for k = 1:n
    x0 = spec.centres_xyz_mm(k, 1);
    width = spec.axial_width_mm(k);
    arm = local_arm_for_electrode(spec, k);
    radial = hypot(face_centres(:, 2) - arm.center_y, ...
        face_centres(:, 3) - arm.center_z);

    % Centroid selection is followed by induced-face reconstruction, matching
    % the surface EIDORS actually integrates over. The radial tolerance keeps
    % only the artificial cylindrical wall and excludes unrelated anatomy.
    radial_tolerance_mm = max(2 * spec.mesh_target_size_mm, ...
        0.05 * arm.radius);
    selected = abs(face_centres(:, 1) - x0) <= width / 2 & ...
        abs(radial - arm.radius) <= radial_tolerance_mm;
    selected_faces = find(selected);
    if isempty(selected_faces)
        error('trkg4:noArmRingFaces', ...
            'No boundary faces selected for electrode %s.', spec.labels(k));
    end

    nodes = unique(faces(selected_faces, :));
    if any(used_nodes(nodes))
        error('trkg4:armElectrodeOverlap', ...
            'Electrode %s overlaps a previous electrode.', spec.labels(k));
    end
    induced = find(all(ismember(faces, nodes), 2));

    electrodes(k).nodes = nodes;
    electrodes(k).z_contact = z_contact;
    induced_faces{k} = induced;
    used_nodes(nodes) = true;
end
end

function arm = local_arm_for_electrode(spec, index)
if spec.side(index) == "right"
    arm = spec.arm_geometry.right;
else
    arm = spec.arm_geometry.left;
end
end

function diagnostics = local_patch_diagnostics( ...
    fmdl, electrodes, induced_faces, spec)
n = numel(electrodes);
realised_area_mm2 = zeros(n, 1);
node_count = zeros(n, 1);
face_count = zeros(n, 1);
x_min_mm = zeros(n, 1);
x_max_mm = zeros(n, 1);
x_centroid_mm = zeros(n, 1);

for k = 1:n
    patch_faces = fmdl.boundary(induced_faces{k}, :);
    p1 = fmdl.nodes(patch_faces(:, 1), :);
    p2 = fmdl.nodes(patch_faces(:, 2), :);
    p3 = fmdl.nodes(patch_faces(:, 3), :);
    realised_area_mm2(k) = sum(0.5 * ...
        vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2));
    node_count(k) = numel(electrodes(k).nodes);
    face_count(k) = numel(induced_faces{k});
    xyz = fmdl.nodes(electrodes(k).nodes, :);
    x_min_mm(k) = min(xyz(:, 1));
    x_max_mm(k) = max(xyz(:, 1));
    x_centroid_mm(k) = mean(xyz(:, 1));
end

diagnostics = spec.position_table;
diagnostics.electrode_kind = repmat(spec.kind, n, 1);
diagnostics.realised_area_mm2 = realised_area_mm2;
diagnostics.area_ratio = realised_area_mm2 ./ spec.nominal_area_mm2;
diagnostics.node_count = node_count;
diagnostics.face_count = face_count;
diagnostics.patch_x_min_mm = x_min_mm;
diagnostics.patch_x_max_mm = x_max_mm;
diagnostics.patch_x_centroid_mm = x_centroid_mm;
end
