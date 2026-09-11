function report=run_trkg4_two_tissue_comparison(varargin)
% Same body/contacts, two resistivities: closed lung versus its depth extension.
p=inputParser;
addParameter(p,'Profile','two_tissue_comparison');
addParameter(p,'ResultTag','auditfix_20260908',@(s)~isempty(regexp(char(s),'^[A-Za-z0-9_-]+$','once')));
addParameter(p,'Stage','solve',@(s)any(string(s)==["export","solve"]));
parse(p,varargin{:});opt=p.Results;
cfg=trkg4_config('nik');root=cfg.project_root;tag=char(opt.ResultTag);
out=fullfile(root,'output',['two_tissue_' tag]);if ~isfolder(out),mkdir(out);end
shared=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
S=load(shared,'fmdl_mesh_units','tissue_id','tissue_names');f=S.fmdl_mesh_units;
mesh_sha256=trkg4_file_sha256(shared);
if string(opt.Stage)=="export"
    centres_mm=element_centroids(f);
    old_lung_mask=S.tissue_id==find(strcmp(S.tissue_names,'lungs'),1);
    a=f.nodes(f.elems(:,2),:)-f.nodes(f.elems(:,1),:);
    b=f.nodes(f.elems(:,3),:)-f.nodes(f.elems(:,1),:);
    c=f.nodes(f.elems(:,4),:)-f.nodes(f.elems(:,1),:);
    element_volume_mm3=abs(dot(a,cross(b,c,2),2))/6;
    assert(all(element_volume_mm3>0));
    save(fullfile(out,'mesh_samples.mat'),'centres_mm','old_lung_mask', ...
        'element_volume_mm3','mesh_sha256','-v7');
    report=struct('stage','export','elements',numel(old_lung_mask),'mesh_sha256',mesh_sha256);
    disp(report);return;
end
trkg4_prepare_runtime(cfg);
mask_file=fullfile(out,'lung_masks.mat');M=load(mask_file);
assert(strcmp(strtrim(M.mesh_sha256),mesh_sha256),'Two-tissue masks belong to another mesh');
closed=logical(M.closed_lung_mask(:));extended=logical(M.extended_lung_mask(:));
assert(numel(closed)==size(f.elems,1) && numel(extended)==numel(closed));
assert(all(extended(closed)) && any(extended & ~closed));
parameters_file=fullfile(out,'geometry.json');G=jsondecode(fileread(parameters_file));
rho=[G.rho_soft_ohm_m,G.rho_lung_ohm_m];assert(all(isfinite(rho)&rho>0));
electrode_file=fullfile(root,'output',['nik_trkg4_inverse_inhale_electrodes_' tag '.csv']);
Eall=readtable(electrode_file,'TextType','string');
assert(strcmp(G.electrodes_sha256,trkg4_file_sha256(electrode_file)));
si=trkg4_scale_fmdl_to_si(f,cfg);
[si.stimulation,si.meas_select]=trkg4_make_4electrode_stimulation(cfg);
cfg.electrode_diagnostics_file='';cfg.electrode_diagnostics_verbose=false;
cfg.fail_on_electrode_diagnostics=true;context=electrode_patch_context(f);
sizes=G.L_mm(:);assert(isequal(sizes,[50;60;70;80;90;110;120;130;140]));
Z_closed=nan(size(sizes));Z_extended=Z_closed;reciprocity=[];
for k=1:numel(sizes)
    E=Eall(Eall.L_mm==sizes(k),:);
    assert(isequal(string(E.electrode),["I_plus";"V_plus";"V_minus";"I_minus"]));
    xyz=[E.requested_x_mm,E.requested_y_mm,E.requested_z_mm];
    [el,induced]=electrode_faces_by_area(f,xyz,cfg.electrode_area,cfg.z_contact,context);
    mm=f;mm.electrode=el;
    trkg4_electrode_diagnostics(mm,xyz,string(cfg.electrode_order),cfg,xyz,induced);
    nodes=vertcat(el.nodes);assert(numel(nodes)==numel(unique(nodes)));
    model=si;model.electrode=el;
    for mode=1:2
        mask=closed;if mode==2,mask=extended;end
        sigma=ones(numel(mask),1)/rho(1);sigma(mask)=1/rho(2);
        z=local_solve(model,sigma,cfg.current_ampere);
        if mode==1,Z_closed(k)=z;else,Z_extended(k)=z;end
        if (mode==1 && any(sizes(k)==[50 90 140])) || (mode==2 && sizes(k)==90)
            inverse=model;
            inverse.stimulation.stim_pattern=[0;1;-1;0]*cfg.current_ampere;
            inverse.stimulation.meas_pattern=[1 0 0 -1];
            zr=local_solve(inverse,sigma,cfg.current_ampere);
            reciprocity(end+1,:)=[sizes(k),mode,abs(z-zr)]; %#ok<AGROW>
            assert(abs(z-zr)<1e-8,'Reciprocity failed');
        end
    end
    fprintf('TWO TISSUES L=%g: closed=%.9f extended=%.9f Ohm\n',sizes(k),Z_closed(k),Z_extended(k));
    table_out=table(sizes,Z_closed,Z_extended,'VariableNames', ...
        {'L_mm','Z_closed_lung_CEM_ohm','Z_extended_lung_CEM_ohm'});
    writetable(table_out,fullfile(out,'fem_progress.csv'));
end
assert(all(isfinite([Z_closed;Z_extended])));
file=fullfile(out,'fem.csv');writetable(table_out,file);
report=struct('schema_version',1,'status','two_tissue_model_comparison_not_validation', ...
    'result_tag',tag,'rho_soft_ohm_m',rho(1),'rho_lung_ohm_m',rho(2), ...
    'soft_definition','Every body element outside the selected lung region, including below the diaphragm, heart and bones', ...
    'body_and_contacts_identical',true,'geometry_qc_passed',true, ...
    'mesh_sha256',mesh_sha256,'mask_sha256',trkg4_file_sha256(mask_file), ...
    'geometry_sha256',trkg4_file_sha256(parameters_file), ...
    'electrodes_sha256',trkg4_file_sha256(electrode_file), ...
    'fem_csv_sha256',trkg4_file_sha256(file), ...
    'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']), ...
    'matlab_version',version,'reciprocity_checks',reciprocity, ...
    'mesh_convergence_tested',false,'units',struct('length','mm','rho','ohm_m','Z','ohm'));
fid=fopen(fullfile(out,'fem_contract.json'),'w');assert(fid>0);
cleaner=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));
fprintf('Two-tissue FEM comparison complete\n');
end

function z=local_solve(model,sigma,current)
cleanup=onCleanup(@()eidors_cache('clear_all'));
img=mk_image(model,sigma);v=fwd_solve(img);z=real(v.meas(1)/current);
assert(isfinite(z));
end
