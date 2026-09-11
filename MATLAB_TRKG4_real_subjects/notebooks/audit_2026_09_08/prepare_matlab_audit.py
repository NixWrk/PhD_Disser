"""Build an isolated harness using verbatim geometry helpers from audited code."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = (ROOT/'src/run_trkg4_inverse_inhale.m').read_text(encoding='utf-8-sig')
geometry = source[source.index('function centres = local_load_reference_grid'):source.index('function [J, sensitivity, diagnostics]')]
builder = source[source.index('function [electrodes, induced_faces] = local_build_electrode_patches'):source.index('function local_write_history')]
main = r'''
function audit_matlab()
here=fileparts(mfilename('fullpath'));
root=fileparts(fileparts(here));
cd(root); addpath('src','vendor_stl_eidors');
cfg=trkg4_config('nik'); trkg4_prepare_runtime(cfg);
fprintf('AUDIT: MATLAB %s\n',version);
test_files={'tests/test_trkg4_nik_inhale_data.m', ...
    'tests/test_trkg4_legacy_electrode_patches.m', ...
    'tests/test_trkg4_legacy_fast_electrode_patches.m'};
tr=runtests(test_files);
audit.tests=table2struct(table(tr));
disp(table(tr));
tag='disjoint_restartable_manifest90';
fit=readtable(fullfile(root,'output',['nik_trkg4_inverse_inhale_fit_summary_' tag '.csv']));
expected=readtable(fullfile(root,'output',['nik_trkg4_inverse_inhale_comparison_' tag '.csv']));
shared=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
fmdl=shared.fmdl_mesh_units;
pose=local_build_pose_context(fmdl,local_load_reference_grid(root),cfg.subject.stl.lungs);
x=[fit.rho_soft_ohm_m fit.rho_lungs_ohm_m fit.centre_u_mm fit.centre_v_mm fit.phi_deg];
sizes=expected.L_mm;
state=local_pose_state(pose,x(3),x(4),x(5));
base_sets=cell(numel(sizes),4);
audit.patch_diagnostics={};
cfg.electrode_diagnostics_file='';
for k=1:numel(sizes)
    centres=local_grid_centres(pose,state,sizes(k));
    [el,faces]=electrode_faces_by_area(fmdl,centres,cfg.electrode_area,cfg.z_contact,pose.patch_context);
    model=fmdl; model.electrode=el;
    row=struct('L_mm',sizes(k),'gate','passed','message','');
    for e=1:4, base_sets{k,e}=el(e).nodes; end
    try
        trkg4_electrode_diagnostics(model,centres,string(cfg.electrode_order),cfg,centres);
    catch err
        if ~strcmp(err.identifier,'trkg4:electrodeDiagnostics'),rethrow(err);end
        row.gate='failed'; row.message=err.message;
    end
    audit.patch_diagnostics{k}=row;
    fprintf('AUDIT PATCH L=%g: %s\n',sizes(k),row.gate);
end
% No FEM solve is needed to prove exact invariance: fixed sigma, nodes,
% contact impedance and electrode node sets define exactly the same system.
audit.plateau={}; n=0;
for j=3:5
    for step=[1e-6,0.01,0.1,0.5]
        for direction=[-1,1]
            trial=x; trial(j)=trial(j)+direction*step;
            st=local_pose_state(pose,trial(3),trial(4),trial(5));
            changed=0; invalid=false;
            for k=1:numel(sizes)
                centres=local_grid_centres(pose,st,sizes(k));
                try
                    el=electrode_faces_by_area(fmdl,centres,cfg.electrode_area,cfg.z_contact,pose.patch_context);
                    for e=1:4,changed=changed+~isequal(el(e).nodes,base_sets{k,e});end
                catch err
                    if ~startsWith(err.identifier,'electrode_faces_by_area:'),rethrow(err);end
                    invalid=true;
                end
            end
            n=n+1;
            audit.plateau{n}=struct('parameter_index',j,'signed_step',direction*step, ...
                'changed_electrodes',changed,'invalid_geometry',invalid);
            fprintf('AUDIT PLATEAU j=%d step=%g changed=%d invalid=%d\n',j,direction*step,changed,invalid);
        end
    end
end
% Actual tetrahedral geometry QC, independent of previous summary claims.
p=fmdl.nodes; t=fmdl.elems;
sixv=dot(p(t(:,2),:)-p(t(:,1),:),cross(p(t(:,3),:)-p(t(:,1),:),p(t(:,4),:)-p(t(:,1),:),2),2);
audit.mesh=struct('nodes',size(p,1),'tetrahedra',size(t,1),'zero_abs_volume_lt1e_10',sum(abs(sixv)<6e-10), ...
 'positive_orientation',sum(sixv>0),'negative_orientation',sum(sixv<0),'volume_l',sum(abs(sixv))/6e6);
% Orientation by global centre is not valid on arbitrary non-convex surfaces.
b=fmdl.boundary; sorted_tet_faces=sort([t(:,[1 2 3]);t(:,[1 2 4]);t(:,[1 3 4]);t(:,[2 3 4])],2);
[found,ind]=ismember(sort(b,2),sorted_tet_faces,'rows');
owner=mod(ind-1,size(t,1))+1; tetcent=(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:)+p(t(:,4),:))/4;
fc=(p(b(:,1),:)+p(b(:,2),:)+p(b(:,3),:))/3;
fn=cross(p(b(:,2),:)-p(b(:,1),:),p(b(:,3),:)-p(b(:,1),:),2);
radial=dot(fn,fc-mean(p,1),2); topology=dot(fn,fc-tetcent(owner,:),2);
audit.mesh.radial_normal_wrong_faces=sum((radial.*topology)<0 & found);
audit.mesh.boundary_faces=size(b,1);
clear sorted_tet_faces tetcent owner ind sixv;
fid=fopen(fullfile(here,'matlab_checks.json'),'w');fwrite(fid,jsonencode(audit),'char');fclose(fid);
% Recompute three actual transfer impedances; add reciprocal stimulation.
model_si=trkg4_scale_fmdl_to_si(fmdl,cfg);
[stim,sel]=trkg4_make_4electrode_stimulation(cfg);
recip=stim;recip.stim_pattern=[0;1;-1;0];recip.meas_pattern=[1 0 0 -1];
model_si.stimulation=[stim;recip];model_si.meas_select=true(2,1);
sigma=shared.elem_sigma;
soft=find(strcmp(shared.tissue_names,'soft_tissue'),1);lung=find(strcmp(shared.tissue_names,'lungs'),1);
sigma(shared.tissue_id==soft)=1/x(1);sigma(shared.tissue_id==lung)=1/x(2);
audit.forward={};
for L=[50 90 140]
    k=find(sizes==L);centres=local_grid_centres(pose,state,L);
    model_si.electrode=electrode_faces_by_area(fmdl,centres,cfg.electrode_area,cfg.z_contact,pose.patch_context);
    tic;v=fwd_solve(mk_image(model_si,sigma));z=real(v.meas);
    row=struct('L_mm',L,'Z_ohm',z(1),'saved_Z_ohm',expected.Z_FEM_ohm(k), ...
        'difference_ohm',z(1)-expected.Z_FEM_ohm(k),'reciprocity_abs_error_ohm',abs(z(1)-z(2)),'seconds',toc);
    audit.forward{end+1}=row;
    fprintf('AUDIT FORWARD %s\n',jsonencode(row));eidors_cache('clear_all');
end
fid=fopen(fullfile(here,'matlab_checks.json'),'w');fwrite(fid,jsonencode(audit),'char');fclose(fid);
fprintf('AUDIT COMPLETE\n');
end
'''
sha=hashlib.sha256(source.encode()).hexdigest()
(HERE/'audit_matlab.m').write_text(f'% Source geometry sha256: {sha}\n'+main+'\n'+geometry+'\n'+builder,encoding='utf-8')
print(HERE/'audit_matlab.m')
