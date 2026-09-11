function tests=test_trkg4_local_projected_qc
tests=functiontests(localfunctions);
end
function setupOnce(~)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'tools'),fullfile(root,'tools','full_pose_scan'), ...
    fullfile(root,'src'),fullfile(root,'vendor_stl_eidors'));
end
function testAreaCentroidAndProjectedProofMatchCanonical(t)
[fmdl,cfg]=fixture(); ctx=full_scan_patch_context(fmdl);
el=struct('nodes',(1:4)','z_contact',cfg.z_contact); fmdl.electrode=el;
c=[0 0 0]; faces={[1;2]};
q=trkg4_local_projected_qc(ctx,el,c,1,cfg,faces);
d=trkg4_electrode_diagnostics(fmdl,c,{'E'},cfg,c,faces);
verifyTrue(t,q.passed); verifyEqual(t,q.global_surface_evaluations,0);
verifyEqual(t,q.surface_distance_upper_bound_mm,0);
verifyEqual(t,q.patch_area_mm2,d.patch_area_mm2);
verifyEqual(t,q.patch_centroid_offset_mm,d.patch_centroid_offset_mm);
verifyEqual(t,q.patch_centroid_xyz_mm, ...
    [d.patch_centroid_x_mm,d.patch_centroid_y_mm,d.patch_centroid_z_mm]);
end
function testBadProjectionHintFallsBackInsteadOfFalseRejection(t)
[fmdl,cfg]=fixture();
fmdl.nodes=[fmdl.nodes;50 0 0;54 0 0;50 2 0];
fmdl.boundary=[fmdl.boundary;5 6 7]; ctx=full_scan_patch_context(fmdl);
el=struct('nodes',(1:4)','z_contact',cfg.z_contact);
q=trkg4_local_projected_qc(ctx,el,[0 0 0],3,cfg,{[1;2]});
verifyTrue(t,q.passed); verifyEqual(t,q.global_surface_evaluations,1);
verifyTrue(t,q.surface_distance_is_exact); verifyEqual(t,q.surface_distance_upper_bound_mm,0);
end
function testThresholdEqualityAndCrossing(t)
[fmdl,cfg]=fixture(); ctx=full_scan_patch_context(fmdl);
el=struct('nodes',(1:4)','z_contact',cfg.z_contact); c=[0 0 0]; faces={[1;2]};
q=trkg4_local_projected_qc(ctx,el,c,1,cfg,faces);
cfg.max_electrode_patch_centroid_offset_mm=q.patch_centroid_offset_mm;
equal=trkg4_local_projected_qc(ctx,el,c,1,cfg,faces); verifyTrue(t,equal.passed);
cfg.max_electrode_patch_centroid_offset_mm=cfg.max_electrode_patch_centroid_offset_mm-1e-6;
failed=trkg4_local_projected_qc(ctx,el,c,1,cfg,faces);
verifyFalse(t,failed.passed); verifyTrue(t,failed.bad_offset);
verifyEqual(t,failed.identifier,'trkg4:electrodeDiagnostics');
fmdl.electrode=el;
verifyError(t,@() trkg4_electrode_diagnostics(fmdl,c,{'E'},cfg,c,faces), ...
    'trkg4:electrodeDiagnostics');
end
function testInducedMismatchAndIndexErrorRemainTechnical(t)
[fmdl,cfg]=fixture(); ctx=full_scan_patch_context(fmdl);
el=struct('nodes',(1:4)','z_contact',cfg.z_contact);
verifyError(t,@() trkg4_local_projected_qc(ctx,el,[0 0 0],1,cfg,{1}), ...
    'trkg4:electrodeDiagnosticsInput');
verifyError(t,@() trkg4_local_projected_qc(ctx,el,[0 0 0],99,cfg,{[1;2]}), ...
    'trkg4:electrodeDiagnosticsInput');
end
function testStandardWrapperPreservesStatusesAndStats(t)
[fmdl,cfg]=fixture(); ctx=full_scan_patch_context(fmdl); c=[0 0 0];
[el,faces,stats]=full_scan_patches(fmdl,c,cfg,ctx,1);
[ref,rf]=electrode_faces_by_area(fmdl,c,cfg.electrode_area,cfg.z_contact);
verifyEqual(t,el,ref); verifyEqual(t,faces,rf);
verifyGreaterThanOrEqual(t,stats.min_area_mm2,cfg.electrode_area);
cfg.max_electrode_patch_centroid_offset_mm=0;
verifyError(t,@() full_scan_patches(fmdl,c,cfg,ctx,1),'trkg4:electrodeDiagnostics');
end
function [fmdl,cfg]=fixture
fmdl=struct('nodes',[0 0 0;4 0 0;0 2 0;0 -1 1], ...
    'boundary',[1 2 3;2 1 4]);
cfg=struct('electrode_area',3,'z_contact',0.01,'length_unit','mm', ...
    'max_electrode_surface_distance_mm',0.1,'max_electrode_patch_centroid_offset_mm',10, ...
    'min_electrode_patch_area_fraction',0.5,'max_electrode_patch_area_ratio',3, ...
    'fail_on_electrode_diagnostics',true,'electrode_diagnostics_verbose',false, ...
    'electrode_diagnostics_file','');
end
