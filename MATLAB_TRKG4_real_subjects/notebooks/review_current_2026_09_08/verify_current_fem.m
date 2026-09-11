function verify_current_fem(run_tests)
% Rebuild contacts from exported requested points, independently of pose helpers.
% All new outputs stay beside this script. Existing fit artifacts are read only.
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
if nargin<1,run_tests=true;end
cd(root); addpath('src','vendor_stl_eidors');
cfg=trkg4_config('nik'); trkg4_prepare_runtime(cfg);
cfg.electrode_diagnostics_file=''; cfg.electrode_diagnostics_verbose=false;
cfg.fail_on_electrode_diagnostics=true;
testnames={'test_trkg4_boundary_normals','test_trkg4_electrode_diagnostics', ...
    'test_trkg4_finite_step_diagnostics','test_trkg4_inverse_contract', ...
    'test_trkg4_green_fingerprint','test_trkg4_green_cache_contract', ...
    'test_trkg4_nik_inhale_data'};
files=cellfun(@(s)fullfile(root,'tests',[s '.m']),testnames,'UniformOutput',false);
if run_tests
tests=runtests(files);
R=struct('status','numerical_checks_not_physical_validation', ...
    'tests_passed',sum([tests.Passed]),'tests_failed',sum([tests.Failed]), ...
    'tests_incomplete',sum([tests.Incomplete]));
local_save(here,R); assert(all([tests.Passed]),'Selected MATLAB tests failed');
else
R=jsondecode(fileread(fullfile(here,'current_fem_checks.json')));
assert(R.tests_passed==30 && R.tests_failed==0 && R.tests_incomplete==0);
end
tag='auditfix_20260908';
fit=readtable(fullfile('output',['nik_trkg4_inverse_inhale_fit_summary_' tag '.csv']));
curve=readtable(fullfile('output',['nik_trkg4_inverse_inhale_comparison_' tag '.csv']));
electrodes=readtable(fullfile('output',['nik_trkg4_inverse_inhale_electrodes_' tag '.csv']));
S=load('output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat', ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
f=S.fmdl_mesh_units; si=trkg4_scale_fmdl_to_si(f,cfg);
[si.stimulation,si.meas_select]=trkg4_make_4electrode_stimulation(cfg);
sigma=S.elem_sigma;
sigma(S.tissue_id==find(strcmp(S.tissue_names,'soft_tissue'),1))=1/fit.rho_soft_ohm_m;
sigma(S.tissue_id==find(strcmp(S.tissue_names,'lungs'),1))=1/fit.rho_lungs_ohm_m;
context=electrode_patch_context(f);
R.tag=tag;R.contact_z_ohm_m2=cfg.z_contact;R.tissue_names=S.tissue_names;
R.grid_checks=cell(height(curve),1);R.homogeneous_checks={};R.contact_sweep={};
R.max_patch_centroid_rebuild_error_mm=0;
R.max_patch_centroid_offset_mm=0; R.max_patch_area_rebuild_error_mm2=0;
for k=1:height(curve)
    L=curve.L_mm(k); E=electrodes(electrodes.L_mm==L,:);
    assert(height(E)==4);
    requested=[E.requested_x_mm E.requested_y_mm E.requested_z_mm];
    [el,induced]=electrode_faces_by_area(f,requested,cfg.electrode_area,cfg.z_contact,context);
    modelmm=f;modelmm.electrode=el;
    trkg4_electrode_diagnostics(modelmm,requested,string(cfg.electrode_order),cfg,requested,induced);
    nodes=vertcat(el.nodes);assert(numel(unique(nodes))==numel(nodes),'Shared CEM nodes');
    for e=1:4
        faces=f.boundary(induced{e},:);
        a=f.nodes(faces(:,1),:);b=f.nodes(faces(:,2),:);c=f.nodes(faces(:,3),:);
        areas=vecnorm(cross(b-a,c-a,2),2,2)/2;
        centre=sum(((a+b+c)/3).*areas,1)/sum(areas);
        expected=[E.patch_centroid_x_mm(e) E.patch_centroid_y_mm(e) E.patch_centroid_z_mm(e)];
        R.max_patch_centroid_rebuild_error_mm=max(R.max_patch_centroid_rebuild_error_mm,norm(centre-expected));
        R.max_patch_centroid_offset_mm=max(R.max_patch_centroid_offset_mm,norm(centre-requested(e,:)));
        R.max_patch_area_rebuild_error_mm2=max(R.max_patch_area_rebuild_error_mm2,abs(sum(areas)-E.patch_area_mm2(e)));
    end
    model=si;model.electrode=el;
    value=local_solve(model,sigma,cfg.current_ampere);
    R.grid_checks{k}=struct('L_mm',L,'Z_ohm',value, ...
        'saved_Z_ohm',curve.Z_FEM_ohm(k),'abs_difference_ohm',abs(value-curve.Z_FEM_ohm(k)));
    assert(abs(value-curve.Z_FEM_ohm(k))<1e-8,'FEM does not reproduce saved result');
    fprintf('CURRENT FEM L=%g: %.12f Ohm, delta %.3g\n',L,value,abs(value-curve.Z_FEM_ohm(k)));
    if any(L==[50 60 90 140])
        uniform=local_solve(model,ones(size(sigma))/fit.rho_soft_ohm_m,cfg.current_ampere);
        planar=8*fit.rho_soft_ohm_m/(3*pi*L*1e-3);
        R.homogeneous_checks{end+1}=struct('L_mm',L,'rho_ohm_m',fit.rho_soft_ohm_m, ...
            'finite_body_CEM_ohm',uniform,'nominal_planar_point_ohm',planar, ...
            'difference_ohm',uniform-planar,'difference_percent',100*(uniform/planar-1));
    end
    if L==90
        reciprocal=model;
        reciprocal.stimulation.stim_pattern=[0;1;-1;0]*cfg.current_ampere;
        reciprocal.stimulation.meas_pattern=[1 0 0 -1];
        rv=local_solve(reciprocal,sigma,cfg.current_ampere);
        R.reciprocity_90mm_difference_ohm=abs(rv-value);
        assert(abs(rv-value)<1e-8,'Reciprocity failed');
        for factor=[.1 10]
            probe=model;
            for e=1:4,probe.electrode(e).z_contact=cfg.z_contact*factor;end
            zv=local_solve(probe,sigma,cfg.current_ampere);
            R.contact_sweep{end+1}=struct('factor',factor,'Z_ohm',zv,'difference_ohm',zv-value);
        end
    end
    local_save(here,R);
end
assert(R.max_patch_centroid_rebuild_error_mm<1e-8);
R.complete=true;R.forward_max_difference_ohm=max(cellfun(@(a)a.abs_difference_ohm,R.grid_checks));
R.homogeneous_interpretation='Finite curved body, CEM patches and discretisation remain; difference from planar point electrodes is NOT pure curvature or FEM error.';
R.mesh_convergence_tested=false;R.contact_sweep_interpretation='Synthetic factor changes; no measured contact uncertainty.';
local_save(here,R);fprintf('CURRENT FEM REVIEW COMPLETE\n');
end

function z=local_solve(model,sigma,current)
cleanup=onCleanup(@()eidors_cache('clear_all'));
img=mk_image(model,sigma);answer=fwd_solve(img);z=real(answer.meas(1)/current);
assert(isfinite(z),'Nonfinite FEM result');
end

function local_save(here,result)
fid=fopen(fullfile(here,'current_fem_checks.json'),'w');assert(fid>0);
cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true));
end

