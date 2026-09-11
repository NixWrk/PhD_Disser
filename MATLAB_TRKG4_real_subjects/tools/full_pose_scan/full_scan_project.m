function [points,normals,face_ids] = full_scan_project(context,raw)
% Same neighbourhood and triangle projection as the published inverse model.
near=knnsearch(context.tree,raw,'K',min(16,numel(context.boundary_nodes)));
near=reshape(context.boundary_nodes(near),size(near));
[points,normals,face_ids]=full_scan_project_mex(double(raw), ...
    context.nodes,context.boundary,context.normals,double(near), ...
    context.node_face_list,context.node_face_start);
end
