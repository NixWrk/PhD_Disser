function report=heart_stl_electrical_check(configFile)
%HEART_STL_ELECTRICAL_CHECK Real STL replacements on the existing CT/CEM mesh.
% Only the established EIDORS/CompartmentReconstruction operator is used.
% Tissue scenarios and geometric probes are not measured cardiac volumes.
cfg=jsondecode(fileread(configFile));
run(fullfile(cfg.pipeline_root,'setup_pipeline.m'));
maxNumCompThreads(cfg.threads);
assert(isempty(gcp('nocreate')),'Run without a MATLAB parallel pool');
out=cfg.output_dir; if ~isfolder(out), mkdir(out); end
plan=jsondecode(fileread(cfg.material_plan));
assert(strcmp(plan.schema,'heart_stl_materials_v1'));
assert(strcmp(trkg4_file_sha256(cfg.mesh_path),plan.mesh_sha256),'Mesh changed');
assert(strcmp(trkg4_file_sha256(cfg.stl_path),plan.stl_sha256),'STL changed');
de=load(cfg.eit_model_file,'fmdl'); fe=de.fmdl;
dt=load(cfg.ttrkg_model_file,'f'); ft=dt.f;
assert(isequal(fe.nodes,ft.nodes)&&isequal(fe.elems,ft.elems),'Different EIT/TT meshes');
ne=size(fe.elems,1);
identity=struct('config_sha256',trkg4_file_sha256(configFile), ...
    'plan_sha256',trkg4_file_sha256(cfg.material_plan), ...
    'implementation_sha256',trkg4_file_sha256([mfilename('fullpath'),'.m']), ...
    'eit_model_sha256',trkg4_file_sha256(cfg.eit_model_file), ...
    'ttrkg_model_sha256',trkg4_file_sha256(cfg.ttrkg_model_file), ...
    'sensitivity_sha256',trkg4_file_sha256(which('cr_compartment_sensitivity')));
report=struct('schema','heart_stl_electrical_check_v1','status','running', ...
    'identity',identity,'names',{{'soft_background','lung','bone','heart','fat'}}, ...
    'frequency_Hz',cfg.frequency_Hz,'current_A',cfg.current_A, ...
    'no_AI_training',true,'clinical_validation',false, ...
    'mesh_convergence_proven',false,'source_STL_anatomically_validated',false, ...
    'breath_interval_status',cfg.breath_interval_status, ...
    'montage_basis',cfg.montage_basis,'rows',{{}});
rows={};
for k=1:numel(cfg.jobs)
    job=cfg.jobs(k);
    idx=find(strcmp({plan.variants.id},job.variant)); assert(numel(idx)==1);
    v=plan.variants(idx);
    file=fullfile(fileparts(cfg.material_plan),v.material_file);
    assert(strcmp(trkg4_file_sha256(file),v.material_sha256),'Materials changed');
    resultFile=fullfile(out,[job.id,'.mat']);
    if isfile(resultFile)
        saved=load(resultFile,'row','identity');
        assert(isequal(saved.identity,identity),'Cached result identity differs; use new output');
        row=saved.row;
    else
        d=load(file,'fractions','tissue_id');
        if isfield(d,'fractions')
            C=sparse(double(d.fractions));
        else
            labels=double(d.tissue_id(:));
            assert(numel(labels)==ne && all(labels>=1 & labels<=5 & labels==fix(labels)));
            C=sparse((1:ne)',labels,1,ne,5);
        end
        assert(isequal(size(C),[ne 5]) && all(nonzeros(C)>=0));
        assert(max(abs(full(sum(C,2))-1))<1e-6,'Invalid material partition');
        sigma=1./job.rho_ohm_m(:);
        assert(numel(sigma)==5 && all(isfinite(sigma)&sigma>0));
        fprintf('STL_JOB_START %d/%d %s\n',k,numel(cfg.jobs),job.id);
        t=tic;
        if job.eit
            [S_eit,V_eit]=cr_compartment_sensitivity(fe,C,sigma);
        else
            S_eit=[]; V_eit=[];
        end
        [Jtt,Vtt]=cr_compartment_sensitivity(ft,C,sigma);
        Ktt=Jtt/cfg.current_A; Ztt=Vtt/cfg.current_A;
        row=struct('id',job.id,'variant',job.variant,'scenario',job.scenario, ...
            'rho_ohm_m',job.rho_ohm_m,'volume_ml',v.volume_ml, ...
            'target_volume_ml',v.target_volume_ml,'quadrature',v.quadrature, ...
            'Ztt_ohm',Ztt,'Ktt_ohm_per_S_m',Ktt,'seconds',toc(t), ...
            'material_sha256',v.material_sha256,'eit_computed',job.eit);
        if k==1
            img=mk_image(ft,1); img.elem_data=full(C*sigma);
            native=fwd_solve(img);
            row.native_forward_relative_error=norm(native.meas-Vtt)/max(norm(Vtt),eps);
            reverse=ft;
            reverse.stimulation.stim_pattern=ft.stimulation.meas_pattern'*cfg.current_A;
            reverse.stimulation.meas_pattern=ft.stimulation.stim_pattern'/cfg.current_A;
            ri=mk_image(reverse,1); ri.elem_data=img.elem_data;
            vr=fwd_solve(ri);
            row.reciprocity_relative_error=norm(vr.meas-Vtt)/max(norm(Vtt),eps);
            assert(row.native_forward_relative_error<1e-8 && row.reciprocity_relative_error<1e-8, ...
                'STL material native forward / reciprocity check failed');
            clear img native ri vr reverse
        end
        if job.eit
            dd=load(cfg.reference_prediction_file,'Vref');
            row.eit_relative_residual_pct=100*norm(dd.Vref-V_eit)/norm(dd.Vref);
        end
        save(resultFile,'row','identity','sigma','S_eit','V_eit','Ktt','Ztt');
        fprintf('STL_JOB_DONE %s Ztt=%.8g seconds=%.2f\n',job.id,Ztt,row.seconds);
        clear C d
        % Keep cache growth bounded; each geometry uses its own material matrix.
        eidors_cache('clear_all');
    end
    rows{end+1}=row; report.rows=rows; %#ok<AGROW>
    write_json(fullfile(out,'electrical_check.local.json'),report);
end
report.status='conditional_STL_geometry_electrical_calculations_completed';
write_json(fullfile(out,'electrical_check.local.json'),report);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
