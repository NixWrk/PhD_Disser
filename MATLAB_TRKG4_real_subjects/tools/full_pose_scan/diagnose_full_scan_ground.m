function results=diagnose_full_scan_ground()
% Bounded diagnostic of numerical grounding, not a pose scan.
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'src')); addpath(folder);
cfg=trkg4_config('nik'); trkg4_prepare_runtime(cfg);
shared=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
f=shared.fmdl_mesh_units; si=trkg4_scale_fmdl_to_si(f,cfg);
[si.stimulation,si.meas_select]=trkg4_make_4electrode_stimulation(cfg);
fit=readtable(fullfile(root,'output','nik_trkg4_inverse_inhale_fit_summary_auditfix_20260908.csv'));
rho=[fit.rho_soft_ohm_m,fit.rho_lungs_ohm_m]; sigma=shared.elem_sigma;
sigma(shared.tissue_id==find(strcmp(shared.tissue_names,'soft_tissue'),1))=1/rho(1);
sigma(shared.tissue_id==find(strcmp(shared.tissue_names,'lungs'),1))=1/rho(2);
T=readtable(fullfile(root,'output','nik_trkg4_inverse_inhale_electrodes_auditfix_20260908.csv'));
T=T(T.L_mm==140,:); centres=[T.requested_x_mm,T.requested_y_mm,T.requested_z_mm]+[0,0,1];
[el,faces]=electrode_faces_by_area(f,centres,cfg.electrode_area,cfg.z_contact);
qc=cfg; qc.electrode_diagnostics_file='';qc.electrode_diagnostics_verbose=false;
f.electrode=el; trkg4_electrode_diagnostics(f,centres,string(cfg.electrode_order),qc,centres,faces);
results=struct(); results.rho=rho; results.cases=struct([]);
for e=0:2
    grounded=si;
    if e>0, grounded.gnd_node=el(e).nodes(1); end
    nodes=unique([vertcat(el.nodes);grounded.gnd_node]);
    map=zeros(size(si.nodes,1),1);map(nodes)=1:numel(nodes);
    bulk=grounded;bulk.electrode=struct([]);bulk.stimulation=struct([]);
    sys=calc_system_mat(mk_image(bulk,sigma)); n=size(si.nodes,1);
    keep=(1:n)';keep(grounded.gnd_node)=[];red=zeros(n,1);red(keep)=1:numel(keep);
    A=sys.E(keep,keep);A=(A+A')/2;factor=decomposition(A,'chol');
    active=find(nodes~=grounded.gnd_node); rhs=sparse(red(nodes(active)),1:numel(active),1,n-1,numel(active));
    V=factor\rhs; G0=zeros(numel(nodes));G0(active,active)=V(red(nodes(active)),:);G0=(G0+G0')/2;
    unscaled=full_scan_cem_transfer(grounded,faces,map,G0,cfg.z_contact);
    scale=1./sqrt(diag(A)); S=spdiags(scale,0,n-1,n-1); H=S*A*S;H=(H+H')/2;
    scaled_factor=decomposition(H,'chol'); Vs=scale.*(scaled_factor\(S*rhs));
    G=zeros(numel(nodes));G(active,active)=Vs(red(nodes(active)),:);G=(G+G')/2;
    scaled=full_scan_cem_transfer(grounded,faces,map,G,cfg.z_contact);
    backward=norm(A*Vs-rhs,'fro')/(norm(A,1)*norm(Vs,'fro')+norm(rhs,'fro'));
    full_model=grounded;full_model.electrode=el;
    voltage=fwd_solve(mk_image(full_model,sigma));reference=real(voltage.meas(1));
    entry=struct('ground_electrode',e,'ground_node',grounded.gnd_node,'scaled',scaled,'unscaled',unscaled, ...
        'full',reference,'scaled_error',scaled-reference,'unscaled_error',unscaled-reference,'bulk_backward',backward);
    if e==0,results.cases=entry;else,results.cases(end+1)=entry;end
    fprintf('GROUND electrode %d node %d: full %.15g scaled %.15g (err %.3g) unscaled %.15g (err %.3g)\n', ...
        e,grounded.gnd_node,reference,scaled,scaled-reference,unscaled,unscaled-reference);
    fid=fopen(fullfile(folder,'ground_diagnostic.json'),'w');fprintf(fid,'%s\n',jsonencode(results,'PrettyPrint',true));fclose(fid);
    clear G G0 V Vs A S H scaled_factor rhs factor sys voltage
    eidors_cache('clear_all');
end
end
