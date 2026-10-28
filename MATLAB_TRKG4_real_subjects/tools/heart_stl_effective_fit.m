function report=heart_stl_effective_fit(configFile)
%HEART_STL_EFFECTIVE_FIT Conditional effective properties for each heart shape.
% Fractional-material adapter for the log-GN loop in cr_compartment_solve.
% The physical operator and exact relative residual remain the upstream ones.
% These fitted parameters are not independently measured tissue properties.
cfg=jsondecode(fileread(configFile));
run(fullfile(cfg.pipeline_root,'setup_pipeline.m'));
maxNumCompThreads(cfg.threads);
assert(isempty(gcp('nocreate')),'Run without a parallel pool');
out=cfg.output_dir; if ~isfolder(out), mkdir(out); end
plan=jsondecode(fileread(cfg.material_plan));
assert(strcmp(plan.schema,'heart_stl_materials_v1'));
assert(strcmp(trkg4_file_sha256(cfg.mesh_path),plan.mesh_sha256));
assert(strcmp(trkg4_file_sha256(cfg.stl_path),plan.stl_sha256));
d=load(cfg.eit_model_file,'fmdl'); fe=d.fmdl;
d=load(cfg.ttrkg_model_file,'f'); ft=d.f;
assert(isequal(fe.nodes,ft.nodes)&&isequal(fe.elems,ft.elems));
d=load(cfg.reference_prediction_file,'Vref'); y=double(d.Vref(:));
free=cfg.free_indices(:);
assert(isequal(free,[1;2;4]) || isequal(free,[1;2]), ...
    'Допускается оценка фона/лёгкого/сердца либо фона/лёгкого при фиксированном сердце');
fixed=setdiff((1:5)',free);
sigma0=1./cfg.initial_rho_ohm_m(:);
identity=struct('config_sha256',trkg4_file_sha256(configFile), ...
    'plan_sha256',trkg4_file_sha256(cfg.material_plan), ...
    'implementation_sha256',trkg4_file_sha256([mfilename('fullpath'),'.m']), ...
    'eit_model_sha256',trkg4_file_sha256(cfg.eit_model_file), ...
    'ttrkg_model_sha256',trkg4_file_sha256(cfg.ttrkg_model_file), ...
    'reference_sha256',trkg4_file_sha256(cfg.reference_prediction_file), ...
    'upstream_solve_sha256',trkg4_file_sha256(which('cr_compartment_solve')), ...
    'sensitivity_sha256',trkg4_file_sha256(which('cr_compartment_sensitivity')), ...
    'residual_sha256',trkg4_file_sha256(which('cr_relative_residual')));
report=struct('schema','heart_stl_effective_fit_v1','status','running', ...
    'identity',identity,'names',{{'soft_background','lung','bone','heart','fat'}}, ...
    'frequency_Hz',cfg.frequency_Hz,'current_A',cfg.current_A, ...
    'free_indices',free,'fixed_indices',fixed,'gain',1, ...
    'breath_interval_status',cfg.breath_interval_status, ...
    'montage_basis',cfg.montage_basis, ...
    'validation_status','exploratory_hypothesis_not_validated','rows',{{}});
rows={};
for k=1:numel(cfg.shapes)
    job=cfg.shapes(k); file=fullfile(out,[job.id,'.mat']);
    if isfile(file)
        saved=load(file,'row','identity');
        assert(isequal(saved.identity,identity),'Cached identity differs');
        row=saved.row;
    else
        [C,v]=materials(plan,cfg.material_plan,job.base,size(fe.elems,1));
        fprintf('EFFECTIVE_FIT_START %s\n',job.id); t=tic;
        [sigma,S_eit,V_eit,history,converged,reason]=fit_properties(fe,C,y,sigma0,free,cfg.max_iterations);
        [Ktt,Ztt]=cr_compartment_sensitivity(ft,C,sigma);
        Ktt=Ktt/cfg.current_A; Ztt=Ztt/cfg.current_A;
        [r,Jlog]=cr_relative_residual(y,V_eit,S_eit(:,free).*sigma(free)');
        ns=vecnorm(Jlog); sn=svd(Jlog./max(ns,eps));
        row=struct('id',job.id,'base_variant',job.base,'rho_ohm_m',1./sigma, ...
            'converged',converged,'stop_reason',reason,'iterations',size(history.sigma,1)-1, ...
            'relative_objective',sum(r.^2),'eit_relative_residual_pct',100*norm(y-V_eit)/norm(y), ...
            'normalized_jacobian_singular_values',sn,'normalized_jacobian_condition',sn(1)/sn(end), ...
            'Ztt_ohm',Ztt,'volume_ml',v.volume_ml,'target_volume_ml',v.target_volume_ml, ...
            'material_sha256',v.material_sha256,'history',history);
        if k==1
            % Check the adapter's actual residual derivative in log parameters.
            direction=[.3;-.4;.5]; direction=direction(1:numel(free)); h=1e-4;
            sp=sigma; sm=sigma;
            sp(free)=sp(free).*exp(h*direction); sm(free)=sm(free).*exp(-h*direction);
            [~,vp]=cr_compartment_sensitivity(fe,C,sp);
            [~,vm]=cr_compartment_sensitivity(fe,C,sm);
            rp=cr_relative_residual(y,vp,zeros(size(vp)));
            rm=cr_relative_residual(y,vm,zeros(size(vm)));
            analytic=-Jlog*direction; numeric=(rp-rm)/(2*h);
            row.residual_derivative_relative_error=norm(numeric-analytic)/max(norm(analytic),eps);
            assert(row.residual_derivative_relative_error<1e-3,'Residual derivative check failed');
        end
        clear C
        eidors_cache('clear_all');
        probes=cell(1,2); ids={job.minus,job.plus};
        for j=1:2
            [C,p]=materials(plan,cfg.material_plan,ids{j},size(fe.elems,1));
            [S,V]=cr_compartment_sensitivity(fe,C,sigma);
            [Kt,Z]=cr_compartment_sensitivity(ft,C,sigma);
            probes{j}=struct('variant',ids{j},'volume_ml',p.volume_ml, ...
                'target_volume_ml',p.target_volume_ml,'material_sha256',p.material_sha256, ...
                'S_eit',S,'V_eit',V,'Ktt',Kt/cfg.current_A,'Ztt',Z/cfg.current_A);
            clear C
            eidors_cache('clear_all');
        end
        dvol=probes{2}.volume_ml(4)-probes{1}.volume_ml(4);
        JV_eit=(probes{2}.V_eit-probes{1}.V_eit)/dvol;
        KV_tt=(probes{2}.Ztt-probes{1}.Ztt)/dvol;
        row.volume_sensitivity_mohm_per_ml=1000*KV_tt;
        row.minus_deltaZ_mohm=1000*(probes{1}.Ztt-Ztt);
        row.plus_deltaZ_mohm=1000*(probes{2}.Ztt-Ztt);
        row.seconds=toc(t);
        save(file,'row','identity','sigma','S_eit','V_eit','Ktt','Ztt','JV_eit','KV_tt','probes');
        fprintf('EFFECTIVE_FIT_DONE %s rho=%s residual=%.4f%% converged=%d\n', ...
            job.id,mat2str(row.rho_ohm_m',6),row.eit_relative_residual_pct,converged);
    end
    rows{end+1}=row; report.rows=rows; %#ok<AGROW>
    write_json(fullfile(out,'effective_fit.local.json'),report);
end
report.status='conditional_effective_parameter_fits_completed';
write_json(fullfile(out,'effective_fit.local.json'),report);
end

function [C,v]=materials(plan,planFile,id,ne)
idx=find(strcmp({plan.variants.id},id)); assert(numel(idx)==1);
v=plan.variants(idx); file=fullfile(fileparts(planFile),v.material_file);
assert(strcmp(trkg4_file_sha256(file),v.material_sha256));
d=load(file,'fractions'); C=sparse(double(d.fractions));
assert(isequal(size(C),[ne,5]) && all(nonzeros(C)>=0));
assert(max(abs(full(sum(C,2))-1))<1e-6);
end

function [sigma,S,V,history,converged,reason]=fit_properties(fmdl,C,y,sigma,free,maxIterations)
% Same positive log parameters, damping and exact residual as upstream GN;
% gain is fixed to one and only explicit free compartments are optimized.
[S,V]=cr_compartment_sensitivity(fmdl,C,sigma);
r=cr_relative_residual(y,V,zeros(size(V)));
history=struct('sigma',sigma','objective',sum(r.^2), ...
    'residual_pct',100*norm(y-V)/norm(y));
mu=1e-3; converged=false; reason='iteration_limit';
for it=1:maxIterations
    [r,J]=cr_relative_residual(y,V,S(:,free).*sigma(free)');
    H=J'*J; g=J'*r; accepted=false;
    for tries=1:8
        step=(H+mu*diag(max(diag(H),eps)))\g;
        if any(~isfinite(step))||max(abs(step))>2, mu=mu*5; continue; end
        candidate=sigma; candidate(free)=sigma(free).*exp(step);
        if any(~isfinite(candidate)|candidate<=0), mu=mu*5; continue; end
        [Sn,Vn]=cr_compartment_sensitivity(fmdl,C,candidate);
        rn=cr_relative_residual(y,Vn,zeros(size(Vn)));
        if all(isfinite(rn))&&all(isfinite(Sn(:)))&&norm(rn)<norm(r)
            accepted=true; mu=max(mu/3,1e-9); break;
        end
        mu=mu*5;
    end
    if ~accepted, reason='no_accepted_step'; break; end
    sigma=candidate; S=Sn; V=Vn;
    history.sigma(end+1,:)=sigma'; %#ok<AGROW>
    history.objective(end+1)=sum(rn.^2); %#ok<AGROW>
    history.residual_pct(end+1)=100*norm(y-V)/norm(y); %#ok<AGROW>
    fprintf('EFFECTIVE_ITER %d rho_free=%s residual=%.4f%%\n',it,mat2str(1./sigma(free)',6),history.residual_pct(end));
    eidors_cache('clear_all');
    if max(abs(step))<1e-4, converged=true; reason='small_log_step'; break; end
end
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
