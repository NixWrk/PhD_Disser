function export_arm_solver_benchmark
addpath('src');cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);maxNumCompThreads(4);
out=fullfile(cfg.project_root,'output','exploratory','arm_sigma_20260908');
s=load(fullfile(out,'prepared_fem.mat'),'fmdl_mm');
spec=trkg4_arm_montage_spec(cfg,'outer_planes_inner_rings',20,80);
[model,~]=trkg4_build_mixed_arm_electrodes(s.fmdl_mm,spec,cfg.z_contact,cfg);
fmdl=trkg4_scale_fmdl_to_si(model,cfg);C=trkg4_cem_contact_matrix(fmdl);
ground=fmdl.gnd_node;
save(fullfile(out,'solver_contact_benchmark.mat'),'C','ground','-v7');
end
