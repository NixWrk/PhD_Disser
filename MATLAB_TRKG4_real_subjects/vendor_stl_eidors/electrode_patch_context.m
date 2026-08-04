function context = electrode_patch_context(fmdl)
%ELECTRODE_PATCH_CONTEXT Mesh-only precomputation for ELECTRODE_FACES_BY_AREA.
%
% Holds the boundary triangles, their areas and a compressed node -> incident
% faces index. Nothing here depends on the electrode positions, so a single
% context serves an entire optimisation or pose scan over the same mesh.
%
% Build it once and pass it as the fifth argument of electrode_faces_by_area.
% On the v5 mesh this removes the dominant per-call cost from loops that
% rebuild the patches for every grid size at every optimiser evaluation.

faces = fmdl.boundary;
n_faces = size(faces, 1);
n_nodes = size(fmdl.nodes, 1);

p1 = fmdl.nodes(faces(:, 1), :);
p2 = fmdl.nodes(faces(:, 2), :);
p3 = fmdl.nodes(faces(:, 3), :);

% Compressed adjacency: the incident faces of node n are stored contiguously
% in node_face_list(node_face_start(n) : node_face_start(n+1)-1). A cell array
% would work too, but this form avoids per-node cell dereferencing in the
% inner growth loop.
entry_node = faces(:);
entry_face = repmat((1:n_faces)', 3, 1);
[sorted_node, order] = sort(entry_node);
node_face_list = entry_face(order);
node_face_start = cumsum([1; accumarray(sorted_node, 1, [n_nodes, 1])]);

context = struct( ...
    'faces', faces, ...
    'n_nodes', n_nodes, ...
    'p1', p1, 'p2', p2, 'p3', p3, ...
    'face_area', 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2), ...
    'node_face_list', node_face_list, ...
    'node_face_start', node_face_start);
end
