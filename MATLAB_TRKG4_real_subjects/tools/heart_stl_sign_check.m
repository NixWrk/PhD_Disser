function report=heart_stl_sign_check(configFile)
% Decompose geometry change into signed tissue-replacement directions.
% Extra Jacobian columns are conductivity directions, not new compartments.
cfg=jsondecode(fileread(configFile));run(fullfile(cfg.pipeline_root,'setup_pipeline.m'));
maxNumCompThreads(4);
d=load(cfg.ttrkg_model_file,'f');f=d.f;I=cfg.current_A;
out=cfg.output_dir;if ~isfolder(out),mkdir(out);end
report=struct('status','running','source_config_sha256',trkg4_file_sha256(configFile), ...
    'source_code_sha256',trkg4_file_sha256([mfilename('fullpath'),'.m']), ...
    'definition','dSigma_k = dFraction_k * (sigma_k - sigma_heart), k=soft,lung,bone,fat', ...
    'interpretation','local derivative of actual finite replacement; finite forward checked separately', ...
    'clinical_validation',false,'rows',{{}});
for j=1:numel(cfg.jobs)
    job=cfg.jobs(j);
    d=load(job.base_material,'fractions');C=sparse(double(d.fractions));
    d=load(job.plus_material,'fractions');Cp=sparse(double(d.fractions));
    sigma=1./job.rho_ohm_m(:);others=[1 2 3 5];
    D=(Cp(:,others)-C(:,others))*spdiags(sigma(others)-sigma(4),0,4,4);
    ds=full((Cp-C)*sigma);
    assert(norm(full(sum(D,2))-ds)<1e-10*max(norm(ds),1));
    extended=[C,D,sparse(max(ds,0)),sparse(min(ds,0))];
    [S,V]=cr_compartment_sensitivity(f,extended,[sigma;zeros(6,1)]);
    z0=V/I;K=S/I;
    img=mk_image(f,1);img.elem_data=full(Cp*sigma);vp=fwd_solve(img);
    finite=(vp.meas-V)/I;
    h=.01;
    img.elem_data=full(C*sigma)+h*ds;vplus=fwd_solve(img);
    img.elem_data=full(C*sigma)-h*ds;vminus=fwd_solve(img);
    fd=(vplus.meas-vminus.meas)/(2*h*I);
    linear=sum(K(6:9));
    rel=abs(fd-linear)/max(abs(fd),1e-14);assert(rel<1e-4,'Directional derivative mismatch');
    self=f;self.stimulation.meas_pattern=f.stimulation.stim_pattern'/I;
    [SS,VS]=cr_compartment_sensitivity(self,extended,[sigma;zeros(6,1)]);
    row=struct('id',job.id,'rho_ohm_m',job.rho_ohm_m,'Z0_ohm',z0, ...
        'finite_deltaZ_mohm',finite*1000,'linear_total_mohm',linear*1000, ...
        'linear_by_replaced_tissue_mohm',K(6:9)*1000, ...
        'direction_names',{{'soft','lung','bone','fat'}}, ...
        'positive_conductivity_direction_mohm',K(10)*1000, ...
        'negative_conductivity_direction_mohm',K(11)*1000, ...
        'directional_derivative_relative_error',rel, ...
        'drive_impedance_ohm',VS/I,'drive_positive_conductivity_direction_mohm',SS(10)/I*1000, ...
        'drive_negative_conductivity_direction_mohm',SS(11)/I*1000, ...
        'base_material_sha256',trkg4_file_sha256(job.base_material), ...
        'plus_material_sha256',trkg4_file_sha256(job.plus_material));
    assert(row.drive_positive_conductivity_direction_mohm<=1e-8 && row.drive_negative_conductivity_direction_mohm>=-1e-8, ...
        'Two-terminal monotonicity sign check failed');
    report.rows{end+1}=row;
    save(fullfile(out,[job.id,'.mat']),'row','S','V','SS','VS');write_report(out,report);
    fprintf('SIGN_CHECK %s finite=%.6g mOhm soft/lung/bone/fat=%s\n', ...
        job.id,finite*1000,mat2str(K(6:9)*1000,5));
    clear C Cp D extended d img ds
    eidors_cache('clear_all');
end
report.status='directional_replacement_and_monotonicity_checks_passed';write_report(out,report);
end

function write_report(out,report)
fid=fopen(fullfile(out,'sign_check.local.json'),'w','n','UTF-8');assert(fid>=0);
cl=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report,'PrettyPrint',true));
end
