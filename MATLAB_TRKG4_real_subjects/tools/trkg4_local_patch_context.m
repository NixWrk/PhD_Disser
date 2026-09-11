function ctx = trkg4_local_patch_context(fmdl)
%TRKG4_LOCAL_PATCH_CONTEXT Immutable snapshot for exact local patch/QC calls.
% ctx = trkg4_local_patch_context(fmdl_mm)
% Rebuild when nodes or boundary changes; do not edit the returned context.
% No source/output files, persistent globals, FEM solves or toolboxes used.
if ~isstruct(fmdl) || ~all(isfield(fmdl, {'nodes','boundary'}))
    error('trkg4:localPatchInput','Expected nodes and boundary.');
end
nodes = double(fmdl.nodes); faces = double(fmdl.boundary);
if size(nodes,2) ~= 3 || size(faces,2) ~= 3 || isempty(faces) || ...
        any(~isfinite(nodes),'all') || any(~isfinite(faces),'all') || ...
        any(faces(:) ~= fix(faces(:))) || ...
        any(faces(:)<1 | faces(:)>size(nodes,1))
    error('trkg4:localPatchInput','Expected finite nodes and valid triangle indices.');
end
ctx.fmdl = struct('nodes',nodes,'boundary',faces);
ctx.canonical = electrode_patch_context(ctx.fmdl);
ctx.version = 1;
% Three sorted coordinate indexes give a conservative cube query. A face's
% max-vertex distance is at least its first vertex's coordinate distance.
ctx.axis_value=cell(1,3); ctx.axis_face=cell(1,3);
for dimension=1:3
    [ctx.axis_value{dimension},ctx.axis_face{dimension}]= ...
        sort(ctx.canonical.p1(:,dimension));
end
ctx.coordinate_scale = max(1,max(abs(nodes),[],'all'));
ctx.face_centroid = (ctx.canonical.p1 + ctx.canonical.p2 + ctx.canonical.p3)/3;
% Preserve canonical QC's global degeneracy check, but perform it ONCE.
p = ctx.canonical;
ab=p.p2-p.p1; ac=p.p3-p.p1; bc=p.p3-p.p2;
edge_scale=max([vecnorm(ab,2,2),vecnorm(ac,2,2),vecnorm(bc,2,2)],[],2);
area_tol=100*eps(max(1,edge_scale.^2));
length_tol=100*eps(max(1,edge_scale));
if any(2*p.face_area <= area_tol) || ...
        any(sum(ab.^2,2)<=length_tol.^2 | sum(ac.^2,2)<=length_tol.^2 | ...
        sum(bc.^2,2)<=length_tol.^2)
    error('trkg4:electrodeDiagnosticsInput','The boundary contains a degenerate triangle.');
end
end
