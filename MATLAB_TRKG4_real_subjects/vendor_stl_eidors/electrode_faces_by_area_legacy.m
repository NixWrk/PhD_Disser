function [el, induced_faces, diagnostics] = electrode_faces_by_area_legacy( ...
    fmdl, electrode_centres, electrode_area, z_contact)
%ELECTRODE_FACES_BY_AREA_LEGACY Reproduce the pre-c8fbca8 patch semantics.
%
% This function intentionally preserves the historical behaviour used by
% archived TRKG4 calculations:
%
%   * every electrode independently takes globally nearest boundary faces;
%   * the stopping area is the sum of explicitly selected faces;
%   * shared nodes and faces between electrodes are allowed;
%   * EIDORS still integrates every boundary face induced by the resulting
%     node set, so the effective CEM area can exceed the stopping area.
%
% It is a reproducibility control, not the production contact model. Use
% electrode_faces_by_area for disjoint physical patches.

faces = fmdl.boundary;
nodes = fmdl.nodes;
n_electrodes = size(electrode_centres, 1);
n_faces = size(faces, 1);
n_nodes = size(nodes, 1);
if n_faces == 0
    error('electrode_faces_by_area_legacy:emptyBoundary', ...
        'The forward model has no boundary faces.');
end

p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
face_area = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);

el = struct('nodes', cell(n_electrodes, 1), ...
    'z_contact', cell(n_electrodes, 1));
induced_faces = cell(n_electrodes, 1);
selected_area = zeros(n_electrodes, 1);
induced_area = zeros(n_electrodes, 1);
selected_face_count = zeros(n_electrodes, 1);
induced_face_count = zeros(n_electrodes, 1);

for electrode_index = 1:n_electrodes
    centre = electrode_centres(electrode_index, :);
    node_distance = sqrt(sum((nodes - centre).^2, 2));
    face_distance = max(node_distance(faces), [], 2);
    [~, face_order] = sort(face_distance);

    cumulative_area = cumsum(face_area(face_order));
    stop_index = find(cumulative_area >= electrode_area, 1);
    if isempty(stop_index)
        error('electrode_faces_by_area_legacy:patchTooSmall', ...
            ['The complete boundary area %.6g is below the requested ', ...
             'electrode area %.6g.'], sum(face_area), electrode_area);
    end

    selected = face_order(1:stop_index);
    electrode_nodes = unique(faces(selected, :));
    node_mask = false(n_nodes, 1);
    node_mask(electrode_nodes) = true;
    induced = find(all(node_mask(faces), 2));

    el(electrode_index).nodes = electrode_nodes;
    el(electrode_index).z_contact = z_contact;
    induced_faces{electrode_index} = induced;
    selected_area(electrode_index) = sum(face_area(selected));
    induced_area(electrode_index) = sum(face_area(induced));
    selected_face_count(electrode_index) = numel(selected);
    induced_face_count(electrode_index) = numel(induced);
end

diagnostics = table((1:n_electrodes)', selected_face_count, ...
    induced_face_count, selected_area, induced_area, ...
    induced_area ./ selected_area, ...
    'VariableNames', {'electrode_index', 'selected_face_count', ...
    'induced_face_count', 'selected_area', 'induced_area', ...
    'induced_to_selected_area_ratio'});
end
