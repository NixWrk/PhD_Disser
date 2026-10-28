function report=pulmovista_operator_audit(configFile)
% Сверка сохранённого быстрого оператора с EIDORS и контактные сценарии.
cfg=jsondecode(fileread(configFile));
run(fullfile(cfg.pipeline_root,'setup_pipeline.m'));
maxNumCompThreads(cfg.threads);
out=cfg.output_dir; if ~isfolder(out), mkdir(out); end
plan=jsondecode(fileread(cfg.material_plan));
idx=find(strcmp({plan.variants.id},cfg.material_variant)); assert(numel(idx)==1);
v=plan.variants(idx); materialFile=fullfile(fileparts(cfg.material_plan),v.material_file);
assert(strcmp(trkg4_file_sha256(materialFile),v.material_sha256));
d=load(materialFile,'fractions'); C=sparse(double(d.fractions)); clear d
d=load(cfg.reference_file,'Vref'); y=d.Vref(:);
identity=struct('config',trkg4_file_sha256(configFile), ...
    'implementation',trkg4_file_sha256([mfilename('fullpath'),'.m']), ...
    'material',v.material_sha256,'reference',trkg4_file_sha256(cfg.reference_file), ...
    'fwd_solve',trkg4_file_sha256(which('fwd_solve_1st_order')), ...
    'system_fields',trkg4_file_sha256(which('system_mat_fields')));
report=struct('status','running','identity',identity,'rows',{{}}, ...
    'scientific_status','exploratory_hypothesis_not_validated');
rows={};
for k=1:numel(cfg.models)
    job=cfg.models(k); s=load(job.fit_file,'sigma','V_eit','S_eit','identity');
    assert(strcmp(trkg4_file_sha256(job.model_file),s.identity.eit_model_sha256));
    d=load(job.model_file,'fmdl'); f=d.fmdl; clear d
    assert(size(C,1)==size(f.elems,1));
    assert(all([f.electrode.z_contact]==.01));
    img=mk_image(f,1); img.elem_data=full(C*s.sigma(:));
    solved=fwd_solve(img); actual=double(solved.meas(:));
    check=norm(actual-s.V_eit)/norm(actual);
    assert(check<1e-9,'Сохранённый быстрый оператор расходится с EIDORS');
    row=struct('id',job.id,'model_sha256',trkg4_file_sha256(job.model_file), ...
        'fit_sha256',trkg4_file_sha256(job.fit_file),'fast_vs_standard_relative_error',check, ...
        'rho_ohm_m',1./s.sigma(:),'baseline_voltage_residual_pct',100*norm(y-actual)/norm(y));
    fprintf('OPERATOR_BASE %s fast_standard=%.3g\n',job.id,check);
    eidors_cache('clear_all');
    if k==numel(cfg.models)
        doubled=img;
        for j=1:numel(doubled.fwd_model.stimulation)
            doubled.fwd_model.stimulation(j).stim_pattern=2*f.stimulation(j).stim_pattern;
        end
        solved=fwd_solve(doubled); vd=double(solved.meas(:));
        row.double_current_relative_error=norm(vd-2*actual)/norm(2*actual);
        assert(row.double_current_relative_error<1e-9,'Нарушена линейность по току');
        clear doubled
        eidors_cache('clear_all');
    end
    contacts=cell(1,numel(cfg.contact_scenarios_ohm_m2));
    for j=1:numel(contacts)
        z=cfg.contact_scenarios_ohm_m2(j); candidate=img;
        [candidate.fwd_model.electrode.z_contact]=deal(z);
        solved=fwd_solve(candidate); vc=double(solved.meas(:));
        contacts{j}=struct('z_contact_ohm_m2',z, ...
            'voltage_change_pct_of_baseline',100*norm(vc-actual)/norm(actual), ...
            'residual_pct',100*norm(y-vc)/norm(y), 'V',vc);
        fprintf('OPERATOR_CONTACT %s z=%.5g delta=%.5f%% residual=%.5f%%\n', ...
            job.id,z,contacts{j}.voltage_change_pct_of_baseline,contacts{j}.residual_pct);
        clear candidate
        eidors_cache('clear_all');
    end
    row.contacts=contacts;
    save(fullfile(out,[job.id,'.mat']),'row','actual','identity');
    rows{end+1}=row; report.rows=rows; write_json(fullfile(out,'operator_audit.local.json'),report); %#ok<AGROW>
    clear f img
end
report.status='operator_and_uniform_contact_scenarios_checked';
report.limitations={'Сценарии контакта не являются измеренными сопротивлениями кожи', ...
    'Не проверены индивидуальные различия 16 контактов и комплексная контактная проводимость', ...
    'Численная согласованность не подтверждает абсолютную приборную калибровку и анатомию', ...
    'Тканевые параметры фиксированы в контактных сценариях; совместная инверсия не выполнялась'};
write_json(fullfile(out,'operator_audit.local.json'),report);
end

function write_json(path,value)
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
