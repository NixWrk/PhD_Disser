function report = run_heart_replacement_electrical_pilot(plan_file, candidate)
% Fresh EIDORS volume blocks for one unchanged-mesh heart material map.
% Called by tools/heart_replacement_electrical_pilot.py; no pool or solve.
% A completed export is immutable; the Python coordinator selects attempts.
plan = jsondecode(fileread(plan_file));
assert(strcmp(plan.schema, 'heart_replacement_electrical_pilot_v1'));
assert(ismember(candidate, {'nominal','sphere','ellipsoid'}));
base = fileparts(plan_file);
item = plan.candidates.(candidate);
assert(strcmp(trkg4_file_sha256(plan.source_prepared), plan.source_prepared_sha256));
assert(strcmp(trkg4_file_sha256(fullfile(base,item.labels_mat)), item.labels_mat_sha256));
for k=1:numel(plan.implementation)
    v=plan.implementation(k);
    assert(strcmp(trkg4_file_sha256(v.path),v.sha256),'Pilot implementation changed');
end
maxNumCompThreads(plan.threads);
assert(isempty(gcp('nocreate')), 'No parallel pool allowed');
fprintf('Loading unchanged SI geometry only; old blocks are NOT loaded.\n');
frozen = load(plan.source_prepared, 'fmdl_m','coordinate_units','tissue_names','tissue_id');
assert(strcmp(frozen.coordinate_units,'m'));
fmdl_m = frozen.fmdl_m;
labels = load(fullfile(base,item.labels_mat),'tissue_id');
tissue_id = double(labels.tissue_id(:));
assert(numel(tissue_id)==size(fmdl_m.elems,1));
assert(isequal(unique(tissue_id),(1:4)'));
assert(isequal(cellstr(frozen.tissue_names(:)), {'soft_tissue';'lungs';'heart';'bones'}));
if strcmp(candidate,'nominal')
    assert(isequal(tissue_id,double(frozen.tissue_id(:))), 'Nominal labels changed');
end
groups=arrayfun(@(k) tissue_id==k,1:4,'UniformOutput',false);
tissue_names={'soft_tissue','lungs','heart','bones'};
out=fullfile(base,item.export_directory);
assert(~isfolder(out),'Export attempt already exists');
fprintf('Fresh EIDORS assembly: %s, %d elements.\n',candidate,numel(tissue_id));
tic;
report=trkg4_export_sensitivity_model(fmdl_m,tissue_id,tissue_names,groups,out);
% The exporter verifies homogeneous assembly; also verify ACTUAL materials.
blocks=load(report.prepared_fem_file,'blocks');
contact=load(report.contact_file,'C');
sigma=double(plan.conductivity_by_tissue_id(:));
K=contact.C;
for k=1:4, K=K+sigma(k)*blocks.blocks{k}; end
img=mk_image(fmdl_m,1); img.elem_data=sigma(tissue_id);
reference=calc_system_mat(img);
err=norm(K-reference.E,'fro')/norm(reference.E,'fro');
assert(isfinite(err)&&err<1e-12,'Heterogeneous EIDORS assembly failed');
report.heterogeneous_matrix_error=err;
report.candidate=candidate;
report.assembly_seconds=toc;
report.source_prepared_sha256=plan.source_prepared_sha256;
report.labels_mat_sha256=item.labels_mat_sha256;
report.plan_sha256=trkg4_file_sha256(plan_file);
report.matlab_version=version;
report.eidors_system_mat_fields=which('system_mat_fields');
report.eidors_system_mat_fields_sha256=trkg4_file_sha256(which('system_mat_fields'));
report.eidors_calc_system_mat_sha256=trkg4_file_sha256(which('calc_system_mat'));
report.geometry_policy='Original SI fmdl_m nodes/elems unchanged; only tissue_id changes';
report.old_material_blocks_loaded=false;
report.no_4D_registration=true;
assert(strcmp(trkg4_file_sha256(plan.source_prepared),plan.source_prepared_sha256));
assert(strcmp(trkg4_file_sha256(fullfile(base,item.labels_mat)),item.labels_mat_sha256));
for k=1:numel(plan.implementation)
    v=plan.implementation(k);
    assert(strcmp(trkg4_file_sha256(v.path),v.sha256),'Pilot implementation changed during export');
end
fid=fopen(fullfile(out,'export.json.tmp'),'w','n','UTF-8');
assert(fid>=0); fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true)); fclose(fid);
movefile(fullfile(out,'export.json.tmp'),fullfile(out,'export.json'));
fprintf('EXPORT COMPLETE %s: heterogeneous error %.3g\n',candidate,err);
end
