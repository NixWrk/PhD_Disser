"""Prepare independent final FEM reproduction and reciprocity verification."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
source=(ROOT/'src/run_trkg4_inverse_inhale.m').read_text(encoding='utf-8')
geometry=source[source.index('function centres = local_load_reference_grid'):source.index('function [J, sensitivity, diagnostics]')]
builder=source[source.index('function [electrodes, induced_faces] = local_build_electrode_patches'):source.index('function local_write_history')]
main=r"""
function verify_final_fem(tag)
if nargin<1,tag='auditfix_20260908';end
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(here));cd(root);addpath('src','vendor_stl_eidors');
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
S=load('output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat', ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
f=S.fmdl_mesh_units;si=trkg4_scale_fmdl_to_si(f,cfg);
[f.stimulation,f.meas_select]=trkg4_make_4electrode_stimulation(cfg);
[si.stimulation,si.meas_select]=trkg4_make_4electrode_stimulation(cfg);
fit=readtable(fullfile('output',['nik_trkg4_inverse_inhale_fit_summary_' tag '.csv']));
curve=readtable(fullfile('output',['nik_trkg4_inverse_inhale_comparison_' tag '.csv']));
x=[fit.rho_soft_ohm_m fit.rho_lungs_ohm_m fit.centre_u_mm fit.centre_v_mm fit.phi_deg];
pose=local_build_pose_context(f,local_load_reference_grid(root),cfg.subject.stl.lungs);
soft=find(strcmp(S.tissue_names,'soft_tissue'),1);lung=find(strcmp(S.tissue_names,'lungs'),1);
[z,state]=local_forward_curve(x,curve.L_mm,f,si,pose,cfg,S.elem_sigma,S.tissue_id,soft,lung,"disjoint");
R=struct('result_tag',tag,'status','numerical_checks_not_physical_validation', ...
    'forward_max_abs_difference_ohm',max(abs(z-curve.Z_FEM_ohm)));
assert(R.forward_max_abs_difference_ohm<1e-9,'Forward reproduction failed');
sigma=S.elem_sigma;sigma(S.tissue_id==soft)=1/x(1);sigma(S.tissue_id==lung)=1/x(2);
reciprocity=cell(3,1);recip_sizes=[50 90 140];
for k=1:3
    L=recip_sizes(k);centres=local_grid_centres(pose,state,L);
    el=local_build_electrode_patches(f,centres,cfg,pose.patch_context,"disjoint");
    model=si;model.electrode=el;
    model.stimulation.stim_pattern=sparse([0;1;-1;0])*cfg.current_ampere;
    model.stimulation.meas_pattern=sparse([1 0 0 -1]);
    image=mk_image(model,sigma);value=fwd_solve(image);recip=real(value.meas(1)/cfg.current_ampere);
    direct=z(curve.L_mm==L);delta=abs(recip-direct);
    reciprocity{k}=struct('L_mm',L,'direct_ohm',direct,'reciprocal_ohm',recip,'abs_difference_ohm',delta);
    assert(delta<1e-9,'Reciprocity failed');clear image value model;eidors_cache('clear_all');
end
R.reciprocity=reciprocity;
% Independent owner matching: one concatenated face array, separate from
% the production helper's four pattern-wise accumulations.
p=f.nodes;t=f.elems;b=f.boundary;
all_faces=sort([t(:,[1 2 3]);t(:,[1 2 4]);t(:,[1 3 4]);t(:,[2 3 4])],2);
[found,ids]=ismember(sort(b,2),all_faces,'rows');assert(all(found));owner=mod(ids-1,size(t,1))+1;
[~,normals]=trkg4_boundary_normals(f);wrong=0;
for k=1:size(b,1)
    opposite=setdiff(t(owner(k),:),b(k,:));
    wrong=wrong+(dot(normals(k,:),p(opposite,:)-p(b(k,1),:))>=0);
end
R.outward_normal_failures=wrong;assert(wrong==0,'Outward normal verification failed');
sixv=dot(p(t(:,2),:)-p(t(:,1),:),cross(p(t(:,3),:)-p(t(:,1),:),p(t(:,4),:)-p(t(:,1),:),2),2);
R.mesh=struct('nodes',size(p,1),'tetrahedra',size(t,1),'negative_or_zero_signed_volume',sum(sixv<=0), ...
    'min_tetra_volume_mm3',min(abs(sixv))/6,'total_volume_l',sum(abs(sixv))/6e6, ...
    'mesh_convergence_tested',false);
assert(all(sixv>0),'Signed tetrahedral volumes failed');
fid=fopen(fullfile(here,['final_fem_checks_' tag '.json']),'w');cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));
fprintf('FINAL FEM: all 9 curves reproduced, reciprocity 50/90/140, outward normals and signed volumes passed\n');disp(R);
end
"""
(HERE/'verify_final_fem.m').write_text(main+'\n'+geometry+'\n'+builder,encoding='utf-8')
