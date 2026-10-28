function report = pulmovista_patient_check(configFile)
%PULMOVISTA_PATIENT_CHECK Exploratory real-CT/real-EIT physical inverse check.
% Reuses supplied PrepareSTLModel and CompartmentReconstruction; no AI training.
% Existing CT mesh is not promoted to a converged or validated patient model.
cfg=jsondecode(fileread(configFile));
run(fullfile(cfg.pipeline_root,'setup_pipeline.m'));
out=cfg.patient_output_dir; if ~isfolder(out), mkdir(out); end
set(0,'DefaultFigureVisible','off');
report=struct('status','running','clinical_validation',false,'AI_training',false, ...
    'absolute_voltage_calibration_verified',false,'mesh_convergence_proven',false, ...
    'belt_position_measured',false,'contact_size_measured',false, ...
    'heart_mask_coverage_unverified',true,'frequency_Hz',cfg.frequency_Hz, ...
    'current_A',cfg.current_A,'config_sha256',trkg4_file_sha256(configFile));
try
    model_sources=struct('mesh_sha256',trkg4_file_sha256(cfg.mesh_path), ...
        'materials_sha256',trkg4_file_sha256(fullfile(out,'materials_centroid.mat')), ...
        'builder_sha256',trkg4_file_sha256([mfilename('fullpath'),'.m']));
    modelFile=fullfile(out,'eit_model.mat');
    if isfile(modelFile)
        d=load(modelFile,'fmdl','electrode_qc','model_config_sha256','model_sources');
        assert(strcmp(d.model_config_sha256,report.config_sha256),'Configuration changed; use fresh output');
        assert(isfield(d,'model_sources') && isequal(d.model_sources,model_sources), ...
            'Mesh/materials/builder changed; use a fresh output directory');
        fmdl=d.fmdl; electrode_qc=d.electrode_qc;
    else
        m=load(cfg.mesh_path); mat=load(fullfile(out,'materials_centroid.mat'));
        fmdl=struct('type','fwd_model','name','Individual CT EIT exploratory', ...
            'nodes',double(m.nodes_mm)*1e-3,'elems',double(m.elems), ...
            'boundary',double(m.boundary),'gnd_node',1, ...
            'solve','fwd_solve_1st_order','jacobian','jacobian_adjoint', ...
            'system_mat','system_mat_1st_order','normalize_measurements',0);
        fmdl.mat_idx_labels={'soft_background','lung','bone','heart','fat'};
        fmdl.mat_idx=arrayfun(@(k)find(mat.tissue_id==k),1:5,'UniformOutput',false);
        torso=struct('vertices',fmdl.nodes,'faces',fmdl.boundary);
        elec=pstl_electrodes_auto(torso,struct('height',cfg.belt_height_m, ...
            'origin_xy',cfg.belt_origin_xy_m,'Oy',[0 -1 0],'n_electrodes',16));
        [fmdl,contours]=pstl_attach_electrodes(fmdl,elec, ...
            struct('shape',cfg.electrode_shape,'width',cfg.electrode_width_m, ...
            'height',cfg.electrode_height_m,'z_contact',cfg.z_contact_ohm_m2));
        electrode_qc=contact_qc(fmdl,elec,cfg);
        stim=cr_default_stim(16,struct('current_ma',cfg.current_A*1000));
        assert(stim.n_meas==208); fmdl.stimulation=stim.stimulation;
        fmdl.meas_select=stim.meas_select;
        model_config_sha256=report.config_sha256;
        save(modelFile,'fmdl','elec','contours','electrode_qc','model_config_sha256','model_sources','-v7.3');
        writematrix(elec,fullfile(out,'electrode_centres_m.csv'));
    end
    report.electrodes=electrode_qc;
    report.model_sources=model_sources;
    if isfield(cfg,'build_model_only')
        assert(islogical(cfg.build_model_only) && isscalar(cfg.build_model_only), ...
            'build_model_only must be a scalar JSON boolean');
        if cfg.build_model_only
            report.status='contact_geometry_checked_no_electrical_solve';
            write_report(out,report);
            fprintf('PATIENT_GEOMETRY_ONLY height_m=%.6f contacts=%d\n', ...
                electrode_qc.actual_height_m,electrode_qc.finite_patches);
            return;
        end
    end
    write_report(out,report);
    cm=cr_compartments_from_mat_idx(fmdl); sigma0=cfg.sigma0_S_m(:);
    d=load(fullfile(cfg.output_dir,[cfg.reference_record,'_reader.mat']));
    nominal=(0:numel(d.times_s)-1)/d.fs_hz;
    assert(max(abs(d.times_s-d.times_s(1)-nominal))<1e-5,'Timestamp grid differs from frame grid');
    selected=nominal>=cfg.reference_interval_s(1) & nominal<=cfg.reference_interval_s(2);
    assert(nnz(selected)>=40 && size(d.vv,1)==208);
    Vref=mean(d.vv(:,selected),2);
    report.reference_interval_s=cfg.reference_interval_s;
    report.reference_frames=nnz(selected); report.reference_status=cfg.reference_interval_status;
    report.reference_frame_indices_1based=find(selected);
    report.reference_selection='inclusive nominal frame grid after measured timestamp consistency check';
    report.model_sources=model_sources;
    report.names=cm.names; report.compartment_volume_ml=cm.volume_m3*1e6;
    fprintf('PATIENT_FORWARD_START nodes=%d elements=%d\n',size(fmdl.nodes,1),size(fmdl.elems,1));
    t=tic; [S0,V0]=cr_compartment_sensitivity(fmdl,cm.C,sigma0);
    report.first_forward_seconds=toc(t);
    report.initial_sigma_S_m=sigma0;
    report.initial_residual_pct=100*norm(Vref-V0)/norm(Vref);
    report.initial_sign_agreement=mean(sign(Vref)==sign(V0));
    report.reciprocity_relative_error=reciprocity(fmdl,V0);
    assert(report.reciprocity_relative_error<1e-7,'Reciprocity failed');
    report.status='forward_and_reciprocity_checked_inverse_pending';
    save(fullfile(out,'initial_forward.mat'),'S0','V0','Vref','sigma0','selected');
    write_report(out,report);
    fprintf('PATIENT_FORWARD seconds=%.2f residual=%.3f%% reciprocity=%.3g\n', ...
        report.first_forward_seconds,report.initial_residual_pct,report.reciprocity_relative_error);
    % Bone and fat are fixed reference scenarios, not measured parameters.
    t=tic; r=cr_compartment_solve(fmdl,Vref,sigma0, ...
        struct('method','gn','fixed',{{'bone','fat'}}, ...
        'fit_gain',false,'max_iterations',cfg.max_iterations));
    report.inverse_seconds=toc(t);
    report.conditional_fit=small_result(r);
    report.conditional_fit.fixed_parameter_basis=cfg.initial_values_basis;
    report.status='conditional_inverse_exploratory_not_validated';
    save(fullfile(out,'conditional_fit.mat'),'r','Vref','selected','-v7.3');
    write_report(out,report);
    fprintf('PATIENT_INVERSE residual=%.3f%% converged=%d sigma=%s\n', ...
        r.residual_pct,r.converged,mat2str(r.sigma',6));
catch err
    report.status='failed'; report.identifier=err.identifier; report.message=err.message;
    write_report(out,report); rethrow(err);
end
end

function q=contact_qc(f,e,c)
areas=zeros(16,1); nnode=zeros(16,1); components=zeros(16,1);
centre_error=zeros(16,1); seen=[];
for k=1:16
    ids=f.electrode(k).nodes(:); assert(~isempty(ids));
    assert(isempty(intersect(seen,ids)),'Overlapping electrode nodes'); seen=[seen;ids]; %#ok<AGROW>
    faces=f.boundary(all(ismember(f.boundary,ids),2),:); assert(~isempty(faces));
    a=f.nodes(faces(:,2),:)-f.nodes(faces(:,1),:);
    b=f.nodes(faces(:,3),:)-f.nodes(faces(:,1),:);
    ar=vecnorm(cross(a,b,2),2,2)/2; areas(k)=sum(ar);
    cen=(f.nodes(faces(:,1),:)+f.nodes(faces(:,2),:)+f.nodes(faces(:,3),:))/3;
    centre_error(k)=norm(sum(cen.*ar,1)/sum(ar)-e(k,:));
    [~,~,id]=unique(faces); compact=reshape(id,size(faces));
    g=graph([compact(:,1);compact(:,2);compact(:,3)], ...
        [compact(:,2);compact(:,3);compact(:,1)]);
    components(k)=max(conncomp(g)); nnode(k)=numel(ids);
end
ratio=areas/(c.electrode_width_m*c.electrode_height_m);
assert(all(components==1) && all(ratio>.75 & ratio<1.25),'Invalid electrode patch geometry');
q=struct('finite_patches',16,'disjoint_nodes',true,'connected_components',components, ...
    'node_counts',nnode,'actual_area_mm2',areas*1e6,'area_ratio_to_assumed',ratio, ...
    'area_centroid_offset_mm',centre_error*1e3,'z_contact_ohm_m2',c.z_contact_ohm_m2, ...
    'actual_height_m',mean(e(:,3)),'basis',c.contact_basis);
end

function err=reciprocity(f,v)
n=numel(f.stimulation); I=full([f.stimulation.stim_pattern]);
amp=max(abs(I),[],1); Q=I./amp;
R=nan(n); row=0;
for k=1:n
    M=full(f.stimulation(k).meas_pattern);
    for j=1:size(M,1)
        plus=find(max(abs(Q-M(j,:)'),[],1)<1e-12);
        minus=find(max(abs(Q+M(j,:)'),[],1)<1e-12);
        if numel(plus)==1, R(k,plus)=v(row+j)/amp(k); end
        if numel(minus)==1, R(k,minus)=-v(row+j)/amp(k); end
    end
    row=row+size(M,1);
end
valid=isfinite(R)&isfinite(R'); assert(nnz(valid)>100);
delta=R-R'; err=norm(delta(valid))/norm(R(valid));
end

function s=small_result(r)
s=struct('sigma_S_m',r.sigma,'rho_ohm_m',1./r.sigma, ...
    'fixed',r.fixed,'residual_pct',r.residual_pct,'converged',r.converged, ...
    'history',r.history,'gain',r.gain,'cond_S',r.cond_S);
J=r.S(:,~r.fixed).*r.sigma(~r.fixed)';
sv=svd(J); s.free_log_singular_values=sv; s.free_log_condition=sv(1)/sv(end);
s.free_column_correlation=(J./vecnorm(J))'*(J./vecnorm(J));
end

function write_report(out,r)
fid=fopen(fullfile(out,'patient_check.local.json'),'w','n','UTF-8'); assert(fid>=0);
cl=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(r,'PrettyPrint',true));
end
