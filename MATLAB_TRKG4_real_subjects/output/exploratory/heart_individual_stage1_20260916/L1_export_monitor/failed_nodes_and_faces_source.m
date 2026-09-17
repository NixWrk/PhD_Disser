function report = trkg4_export_heart_refinement(config_file)
% Preserve exact outer/contact geometry while rebuilding a refined P1 model.
cfg=jsondecode(fileread(config_file));
maxNumCompThreads(cfg.threads);
assert(isempty(gcp('nocreate')));
local_hash(cfg.source_prepared,cfg.source_prepared_sha256);
local_hash(cfg.refined_mesh,cfg.refined_mesh_sha256);
local_hash(cfg.refinement_report,cfg.refinement_report_sha256);
local_hash(cfg.contact_geometry,cfg.contact_geometry_sha256);
local_hash(cfg.original_contact,cfg.original_contact_sha256);
qc=jsondecode(fileread(cfg.refinement_report));
assert(strcmp(qc.status,'passed_heart_volume_refinement'));
assert(all(structfun(@(v) logical(v),qc.checks)));
assert(strcmp(qc.source_prepared_sha256,cfg.source_prepared_sha256));
assert(strcmp(qc.refined_mesh_sha256,cfg.refined_mesh_sha256));
assert(~isfolder(cfg.output_directory),'Use a new export directory');
parent=load(cfg.source_prepared,'fmdl_m','tissue_names','tissue_id','coordinate_units');
assert(strcmp(parent.coordinate_units,'m'));
new=load(cfg.refined_mesh,'nodes_m','elems','boundary','tissue_id','parent_tet_id');
old=parent.fmdl_m;oldnn=size(old.nodes,1);
assert(isequal(new.nodes_m(1:oldnn,:),old.nodes));
assert(isequal(double(new.boundary),double(old.boundary)));
assert(isequal(double(new.tissue_id(:)),double(parent.tissue_id(double(new.parent_tet_id(:))))));
keep={'type','solve','jacobian','system_mat','normalize_measurements'};
keep=intersect(keep,fieldnames(old),'stable');
fmdl_m=struct();
for k=1:numel(keep),fmdl_m.(keep{k})=old.(keep{k});end
fmdl_m.name='Cardiac neighbourhood refinement with unchanged CEM contacts';
fmdl_m.nodes=double(new.nodes_m);fmdl_m.elems=double(new.elems);
fmdl_m.boundary=double(new.boundary);fmdl_m.gnd_node=old.gnd_node;
geometry=jsondecode(fileread(cfg.contact_geometry));
assert(strcmp(geometry.prepared_sha256,cfg.source_prepared_sha256));
assert(strcmp(geometry.status,'passed'));
assert(strcmp(geometry.schema,'surface_sensitivity_contact_geometry_v1'));
assert(geometry.index_base==1);
assert(strcmp(geometry.id,'tepc_2')&&numel(geometry.electrodes)==4);
for j=1:4
    e=geometry.electrodes(j);ids=double(e.patch.nodes_global_1based(:));
    faces=ids(double(e.patch.faces_1based));
    assert(max(abs(fmdl_m.nodes(ids,:)*1000-double(e.patch.coords_mm)),[],'all')<1e-9);
    assert(all(ismember(sort(faces,2),sort(fmdl_m.boundary,2),'rows')));
    fmdl_m.electrode(j).nodes=ids;
    fmdl_m.electrode(j).faces=faces;
    fmdl_m.electrode(j).z_contact=e.z_contact_ohm_m2;
end
C=trkg4_cem_contact_matrix(fmdl_m);
original=load(cfg.original_contact,'C','ground','prepared_sha256','electrode_ids','matrix_error');
assert(strcmp(original.prepared_sha256,cfg.source_prepared_sha256));
assert(isequal(original.electrode_ids(:),{'E1';'E2';'E3';'E4'}));
assert(isfinite(original.matrix_error)&&original.matrix_error<1e-12);
newnn=size(fmdl_m.nodes,1);
subset=[1:oldnn,newnn+(1:4)];
cem_error=norm(C(subset,subset)-original.C,'fro')/norm(original.C,'fro');
assert(cem_error<1e-12);
assert(nnz(C(oldnn+1:newnn,:))==0&&nnz(C(:,oldnn+1:newnn))==0);
assert(fmdl_m.gnd_node==original.ground);
tissue_id=double(new.tissue_id(:));
groups=arrayfun(@(j) tissue_id==j,(1:4)','UniformOutput',false);
export=trkg4_export_sensitivity_model(fmdl_m,tissue_id,parent.tissue_names,groups,cfg.output_directory);
assert(export.matrix_error<1e-12);
report=struct('status','passed_heart_refinement_export','source_prepared_sha256',cfg.source_prepared_sha256, ...
    'refined_mesh_sha256',cfg.refined_mesh_sha256,'refinement_report_sha256',cfg.refinement_report_sha256, ...
    'prepared_file','prepared_fem.mat','prepared_sha256',export.prepared_sha256, ...
    'contact_file','contact.mat','contact_sha256',export.contact_sha256, ...
    'nodes',newnn,'elements',size(fmdl_m.elems,1),'ground_zero_based',fmdl_m.gnd_node-1, ...
    'contact_embedding_relative_error',cem_error,'matrix_relative_error',export.matrix_error, ...
    'original_nodes_preserved',true,'outer_boundary_preserved',true,'contact_faces_preserved',true, ...
    'contact_geometry_sha256',cfg.contact_geometry_sha256,'original_contact_sha256',cfg.original_contact_sha256, ...
    'sensitivity_exporter_sha256',trkg4_file_sha256(which('trkg4_export_sensitivity_model')), ...
    'contact_assembler_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')), ...
    'boundary_validation','Parent boundary preserved; conforming midpoint subdivision checked by mesh QC', ...
    'config_sha256',trkg4_file_sha256(config_file),'implementation_sha256',trkg4_file_sha256(which(mfilename)), ...
    'spatial_convergence',false,'physical_validation',false);
fid=fopen(fullfile(cfg.output_directory,'heart_refinement_export.json'),'w','n','UTF-8');assert(fid>=0);
fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));fclose(fid);
end
function local_hash(path,expected)
assert(strcmp(trkg4_file_sha256(path),expected),'Changed refinement input: %s',path);
end
