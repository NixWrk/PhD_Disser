function [electrodes,induced_faces,stats,cache] = full_scan_patches( ...
    fmdl,projected_centres,cfg,patch_context,projection_face_ids,cache)
%FULL_SCAN_PATCHES Exact canonical patch selection and projected-centre QC.
% ctx=full_scan_patch_context(fmdl); call repeatedly on the SAME frozen mesh.
% fmdl remains in this API for main integration; the actual geometry is the
% immutable context snapshot. Rebuild context if ANY coordinate/boundary
% index changes; counts below are a cheap misuse check, not a mesh hash.
% cfg thresholds and electrode_area/z_contact retain canonical semantics.
% Optional cfg.local_patch_options controls radius/retries only (not physics).
% Known geometry rejections are canonical noSeed/patchTooSmall (builder) or
% trkg4:electrodeDiagnostics (QC). Technical failures propagate unchanged.
% This wrapper never writes output and never runs FEM.
if nargin<6,cache=[];end
if size(fmdl.nodes,1)~=size(patch_context.fmdl.nodes,1) || ...
        ~isequal(size(fmdl.boundary),size(patch_context.fmdl.boundary))
    error('trkg4:localPatchInput','Rebuild patch context for this mesh.');
end
options=struct();
if isfield(cfg,'local_patch_options'), options=cfg.local_patch_options; end
[electrodes,induced_faces,info,cache]=trkg4_local_exact_patches( ...
    patch_context,projected_centres,cfg.electrode_area,cfg.z_contact,options,cache);
q=trkg4_local_projected_qc(patch_context,electrodes,projected_centres, ...
    projection_face_ids,cfg,induced_faces);
quiet=isfield(cfg,'return_geometry_status')&&cfg.return_geometry_status;
if ~q.passed && ~quiet
    error('trkg4:electrodeDiagnostics', ...
        ['Electrode QC: %d surface, %d centroid, %d low-area, %d high-area ', ...
         'violations of unchanged cfg thresholds.'], ...
        sum(q.bad_surface),sum(q.bad_offset),sum(q.bad_area_low),sum(q.bad_area_high));
end
stats=info;
stats.qc=q;
stats.passed=q.passed;
stats.max_centroid_offset_mm=max(q.patch_centroid_offset_mm);
stats.min_area_mm2=min(q.patch_area_mm2);
stats.max_area_mm2=max(q.patch_area_mm2);
stats.global_surface_evaluations=q.global_surface_evaluations;
end
