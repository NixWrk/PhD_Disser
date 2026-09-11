function context = full_scan_context(fmdl)
% Immutable geometry shared by all centres and rotation angles.
context=struct('nodes',double(fmdl.nodes),'boundary',double(fmdl.boundary));
context.normals=trkg4_boundary_normals(fmdl);
context.boundary_nodes=unique(fmdl.boundary(:));
context.tree=KDTreeSearcher(context.nodes(context.boundary_nodes,:));
pc=electrode_patch_context(fmdl);
context.node_face_list=double(pc.node_face_list);
context.node_face_start=double(pc.node_face_start);
end
