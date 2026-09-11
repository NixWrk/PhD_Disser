"""Build a geometry-only probe from the current inverse helpers."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
s=(ROOT/'src/run_trkg4_inverse_inhale.m').read_text(encoding='utf-8-sig')
geometry=s[s.index('function centres = local_load_reference_grid'):s.index('function [J, sensitivity, diagnostics]')]
builder=s[s.index('function [electrodes, induced_faces] = local_build_electrode_patches'):s.index('function local_write_history')]
main=r"""
function verify_geometry_probe()
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
cd(root); addpath('src','vendor_stl_eidors');
cfg=trkg4_config('nik'); trkg4_prepare_runtime(cfg);
cfg.electrode_diagnostics_file=''; cfg.electrode_diagnostics_verbose=false;
cfg.fail_on_electrode_diagnostics=true;
shared=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
fmdl=shared.fmdl_mesh_units;
pose=local_build_pose_context(fmdl,local_load_reference_grid(root),cfg.subject.stl.lungs);
fit=readtable(fullfile(root,'output','nik_trkg4_inverse_inhale_fit_summary_disjoint_restartable_manifest90.csv'));
x0=[fit.rho_soft_ohm_m fit.rho_lungs_ohm_m fit.centre_u_mm fit.centre_v_mm fit.phi_deg];
[sizes,~,~]=trkg4_load_nik_inhale_data(root);
% Deterministic geometry screening around the historical candidate. These
% are starting positions, not estimates obtained from experimental fitting.
offsets=[0,-4,4,-8,8,-16,16]; angles=[0,-4,4,-8,8];
rows=cell(0,9); valid_points=zeros(0,5); count=0; timer=tic;
for ai=1:numel(angles)
for ui=1:numel(offsets)
for vi=1:numel(offsets)
    x=x0+[0,0,offsets(ui),offsets(vi),angles(ai)];
    if x(5)<-20 || x(5)>20, continue; end
    state=local_pose_state(pose,x(3),x(4),x(5));
    failed=0; reason=''; max_offset=0; max_ratio=0;
    for k=1:numel(sizes)
        centres=local_grid_centres(pose,state,sizes(k)); model=fmdl;
        try
            [model.electrode, faces]=local_build_electrode_patches( ...
                model,centres,cfg,pose.patch_context,"disjoint");
            diagnostic=trkg4_electrode_diagnostics(model,centres, ...
                string(cfg.electrode_order),cfg,centres,faces);
            max_offset=max(max_offset,max(diagnostic.patch_centroid_offset_mm));
            max_ratio=max(max_ratio,max(diagnostic.patch_area_mm2)/cfg.electrode_area);
        catch exception
            if ~any(string(exception.identifier)==["trkg4:electrodeDiagnostics", ...
                    "electrode_faces_by_area:patchTooSmall","electrode_faces_by_area:noSeed"])
                rethrow(exception);
            end
            failed=failed+1; reason=exception.identifier;
        end
    end
    count=count+1;
    rows(end+1,:)={count,x(3),x(4),x(5),failed,max_offset,max_ratio,state.h_centre_mm,reason};
    if failed==0, valid_points(end+1,:)=x; end
    fprintf('GEOMETRY %03d u=%+.3f v=%+.3f phi=%+.3f failed=%d/%d accepted=%d seconds=%.1f\n', ...
        count,x(3),x(4),x(5),failed,numel(sizes),size(valid_points,1),toc(timer));
    T=cell2table(rows,'VariableNames',{'evaluation','u_mm','v_mm','phi_deg', ...
        'failed_grid_count','max_offset_among_passed_mm','max_area_ratio_among_passed', ...
        'skin_lung_depth_mm','last_failure'});
    writetable(T,fullfile(here,'geometry_probe.csv'));
    if ~isempty(valid_points), writematrix(valid_points,fullfile(here,'admissible_starts.csv')); end
    if size(valid_points,1)>=3, return; end
end
end
end
if isempty(valid_points)
    error('trkg4:noAdmissibleProbePoint','No all-size admissible pose in the declared geometry probe.');
end
end
"""
(HERE/'verify_geometry_probe.m').write_text(main+'\n'+geometry+'\n'+builder,encoding='utf-8')
print('Prepared isolated current-code geometry probe; it will not write canonical FEM results')
