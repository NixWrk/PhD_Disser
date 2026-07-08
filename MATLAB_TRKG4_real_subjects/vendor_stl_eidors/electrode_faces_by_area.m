function el = electrode_faces_by_area(fmdl, electrode_centres, electrode_area, z_contact)
%ELECTRODE_FACES_BY_AREA Assign EIDORS electrodes from nearest boundary faces.
% Each electrode is grown around the nearest boundary patch until the sum of
% triangle areas reaches electrode_area.

n_electrodes = size(electrode_centres, 1);
el = struct('nodes', cell(n_electrodes, 1), 'z_contact', cell(n_electrodes, 1));

for j = 1:n_electrodes
    centre = electrode_centres(j, :);
    node_dist = sqrt(sum((fmdl.nodes - centre).^2, 2));
    face_dist = max(node_dist(fmdl.boundary), [], 2);
    [~, face_order] = sort(face_dist);

    selected_faces = [];
    selected_area = 0;

    for k = 1:numel(face_order)
        face_idx = face_order(k);
        face_nodes = fmdl.boundary(face_idx, :);
        tri_points = fmdl.nodes(face_nodes, :);
        selected_area = selected_area + triangle_area_3d(tri_points);
        selected_faces(end + 1) = face_idx; %#ok<AGROW>

        if selected_area >= electrode_area
            break;
        end
    end

    el(j).nodes = unique(fmdl.boundary(selected_faces, :));
    el(j).z_contact = z_contact;
end
end

function area = triangle_area_3d(points)
a = points(2, :) - points(1, :);
b = points(3, :) - points(1, :);
area = 0.5 * norm(cross(a, b));
end
