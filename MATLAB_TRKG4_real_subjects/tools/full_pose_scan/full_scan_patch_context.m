function context = full_scan_patch_context(fmdl)
%FULL_SCAN_PATCH_CONTEXT Build once per IMMUTABLE mesh, per worker.
% Requires tools/ and vendor_stl_eidors/ on MATLAB path.
context=trkg4_local_patch_context(fmdl);
context.cache_identity=full_scan_mesh_identity(fmdl);
end
