function snapped = trkg4_snap_electrode_centres(fmdl, requested)
%TRKG4_SNAP_ELECTRODE_CENTRES Project requested centres to boundary nodes.
% The fine v5 surface has a median edge scale of about 1.5 mm, so node
% snapping is submillimetric on average and prevents a patch centred several
% millimetres inside a thin limb from growing around both sides of the limb.

boundary_nodes = unique(fmdl.boundary(:));
boundary_xyz = fmdl.nodes(boundary_nodes, :);
nearest = knnsearch(boundary_xyz, requested);
snapped = boundary_xyz(nearest, :);
end
