function [fmdl, diagnostics, keep_elements] = ...
    trkg4_build_arm_cross_section_electrodes(fmdl, spec, z_contact)
%TRKG4_BUILD_ARM_CROSS_SECTION_ELECTRODES Build ideal internal arm planes.
%
% This is a numerical comparison geometry, not a physically attachable
% electrode. Each plane is represented by existing internal mesh faces that
% separate tetrahedron centroids on opposite sides of the requested arm
% cross-section. Tissue elements are not removed. The resulting surface is
% mesh-conforming and therefore slightly jagged rather than exactly planar.

if spec.kind ~= "cross_section_plane"
    error('trkg4:notCrossSectionPlane', ...
        'Expected cross_section_plane, received %s.', spec.kind);
end

nodes = double(fmdl.nodes);
elems = double(fmdl.elems);
n_elements = size(elems, 1);
element_points = reshape(nodes(elems', :), 4, n_elements, 3);
element_centres = squeeze(mean(element_points, 1));
keep_elements = true(n_elements, 1);

if isfield(fmdl, 'electrode')
    fmdl = rmfield(fmdl, 'electrode');
end
if isfield(fmdl, 'stimulation')
    fmdl = rmfield(fmdl, 'stimulation');
end
if isfield(fmdl, 'meas_select')
    fmdl = rmfield(fmdl, 'meas_select');
end

n_electrodes = numel(spec.labels);
electrode_template = struct('nodes', [], 'faces', [], ...
    'z_contact', z_contact);
fmdl.electrode = repmat(electrode_template, n_electrodes, 1);
all_electrode_faces = cell(n_electrodes, 1);

for k = 1:n_electrodes
    arm = local_arm_for_electrode(spec, k);
    x0 = spec.centres_xyz_mm(k, 1);
    radial = hypot(element_centres(:, 2) - arm.center_y, ...
        element_centres(:, 3) - arm.center_z);

    % Ten local mesh steps give enough neighbouring tetrahedra to recover
    % every shared face of the centroid-defined interface without building
    % the global face-adjacency table for the 4.3-million-element mesh.
    half_window_mm = max(10 * spec.mesh_target_size_mm, 10);
    candidate = abs(element_centres(:, 1) - x0) <= half_window_mm & ...
        radial <= arm.radius + 2 * spec.mesh_target_size_mm;
    candidate_ids = find(candidate);
    if isempty(candidate_ids)
        error('trkg4:noCrossSectionCandidates', ...
            'No tetrahedra found near plane %s.', spec.labels(k));
    end

    local_elems = elems(candidate_ids, :);
    faces = [local_elems(:, [1 2 3]); local_elems(:, [1 2 4]); ...
        local_elems(:, [1 3 4]); local_elems(:, [2 3 4])];
    owners = repmat(candidate_ids, 4, 1);
    [sorted_faces, order] = sortrows(sort(faces, 2));
    duplicate_start = find(all( ...
        sorted_faces(1:end-1, :) == sorted_faces(2:end, :), 2));
    if isempty(duplicate_start)
        error('trkg4:noCrossSectionAdjacency', ...
            'No shared faces found near plane %s.', spec.labels(k));
    end

    owner_a = owners(order(duplicate_start));
    owner_b = owners(order(duplicate_start + 1));
    side_a = element_centres(owner_a, 1) >= x0;
    side_b = element_centres(owner_b, 1) >= x0;
    interface_faces = sorted_faces(duplicate_start(side_a ~= side_b), :);

    face_centres = (nodes(interface_faces(:, 1), :) + ...
        nodes(interface_faces(:, 2), :) + ...
        nodes(interface_faces(:, 3), :)) / 3;
    face_radial = hypot(face_centres(:, 2) - arm.center_y, ...
        face_centres(:, 3) - arm.center_z);
    interface_faces = interface_faces( ...
        face_radial <= arm.radius + spec.mesh_target_size_mm, :);
    interface_faces = unique(interface_faces, 'rows');
    if isempty(interface_faces)
        error('trkg4:noCrossSectionFaces', ...
            'No internal faces remain for electrode %s.', spec.labels(k));
    end

    for previous = 1:k-1
        if ~isempty(intersect(interface_faces, ...
                all_electrode_faces{previous}, 'rows'))
            error('trkg4:crossSectionPlaneOverlap', ...
                'Cross-section electrode %s overlaps another electrode.', ...
                spec.labels(k));
        end
    end
    all_electrode_faces{k} = interface_faces;
    fmdl.electrode(k).faces = interface_faces;
end

diagnostics = spec.position_table;
realised_area_mm2 = zeros(n_electrodes, 1);
projected_area_mm2 = zeros(n_electrodes, 1);
node_count = zeros(n_electrodes, 1);
face_count = zeros(n_electrodes, 1);
x_min_mm = zeros(n_electrodes, 1);
x_max_mm = zeros(n_electrodes, 1);
x_centroid_mm = zeros(n_electrodes, 1);

for k = 1:n_electrodes
    faces = fmdl.electrode(k).faces;
    p1 = nodes(faces(:, 1), :);
    p2 = nodes(faces(:, 2), :);
    p3 = nodes(faces(:, 3), :);
    normal_vectors = cross(p2 - p1, p3 - p1, 2);
    normal_lengths = vecnorm(normal_vectors, 2, 2);
    face_area_mm2 = 0.5 * normal_lengths;
    face_centroid_x_mm = (p1(:, 1) + p2(:, 1) + p3(:, 1)) / 3;
    realised_area_mm2(k) = sum(face_area_mm2);
    projected_area_mm2(k) = sum(face_area_mm2 .* ...
        abs(normal_vectors(:, 1)) ./ max(normal_lengths, eps));
    node_count(k) = numel(unique(faces));
    face_count(k) = size(faces, 1);
    all_x = nodes(unique(faces), 1);
    x_min_mm(k) = min(all_x);
    x_max_mm(k) = max(all_x);
    x_centroid_mm(k) = sum(face_area_mm2 .* face_centroid_x_mm) / ...
        realised_area_mm2(k);
end

diagnostics.electrode_kind = repmat(spec.kind, n_electrodes, 1);
diagnostics.realised_area_mm2 = realised_area_mm2;
diagnostics.area_ratio = realised_area_mm2 ./ diagnostics.nominal_area_mm2;
diagnostics.projected_cross_section_area_mm2 = projected_area_mm2;
diagnostics.projected_cross_section_area_ratio = ...
    projected_area_mm2 ./ diagnostics.nominal_area_mm2;
diagnostics.node_count = node_count;
diagnostics.face_count = face_count;
diagnostics.patch_x_min_mm = x_min_mm;
diagnostics.patch_x_max_mm = x_max_mm;
diagnostics.patch_x_centroid_mm = x_centroid_mm;
end

function arm = local_arm_for_electrode(spec, index)
if spec.side(index) == "right"
    arm = spec.arm_geometry.right;
else
    arm = spec.arm_geometry.left;
end
end
