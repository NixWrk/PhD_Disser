function identity = trkg4_inverse_identity(cfg, data_selection, patch_builder_mode)
%TRKG4_INVERSE_IDENTITY Identify fixed inputs of one inverse-model run.
% The identity deliberately excludes fitted rho/pose parameters. It changes
% when fixed data, mesh, operator code or contact settings change.
root=cfg.project_root;
files={fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    cfg.subject.stl.lungs, ...
    fullfile(root,'data','nik','experimental','nik_breath_hold_levels.csv'), ...
    fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv')};
code=[dir(fullfile(root,'src','*.m'));dir(fullfile(root,'vendor_stl_eidors','*.m'))];
for k=1:numel(code), files{end+1}=fullfile(code(k).folder,code(k).name); end
records=repmat(struct('name','','path','','sha256',''),numel(files),1);
for k=1:numel(files)
    file=char(files{k}); [~,name,extension]=fileparts(file);
    records(k).name=[name extension];
    if startsWith(file,[root filesep],'IgnoreCase',true)
        records(k).path=strrep(file(numel(root)+2:end),'\','/');
    end
    records(k).sha256=trkg4_file_sha256(file);
end
runtime=struct('matlab',version);
operator_files={'fwd_solve','system_mat_fields'};
for k=1:numel(operator_files)
    file=which(operator_files{k});
    if isempty(file)
        error('trkg4:inverseOperatorMissing','Missing EIDORS operator %s.',operator_files{k});
    end
    runtime.([operator_files{k} '_sha256'])=trkg4_file_sha256(file);
end
payload=struct('version','trkg4-inverse-auditfix-20260908-v1', ...
    'data_selection',char(data_selection),'patch_builder_mode',char(patch_builder_mode), ...
    'current_ampere',cfg.current_ampere,'contact_area_mm2',cfg.electrode_area, ...
    'z_contact_ohm_m2',cfg.z_contact,'runtime',runtime,'inputs',records);
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(unicode2native(jsonencode(payload),'UTF-8'));
bytes=typecast(digest.digest(),'uint8');
identity=payload;
identity.fingerprint=lower(reshape(dec2hex(bytes,2).',1,[]));
end
