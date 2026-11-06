function report=run_bmstu_apparent_resistivity(varargin)
% BMSTU BASE2, own CT, four finite pads and one homogeneous rho per record.
p=inputParser;addParameter(p,'InputFile','');addParameter(p,'OutputDir','');
parse(p,varargin{:});a=p.Results;g=jsondecode(fileread(a.InputFile));
assert(strcmp(g.schema,'bmstu_uniform_rho_geometry_v1')&&g.mesh_QC.passed);
assert(g.four_lateral_contacts_only&&~g.TTRKG_channel_excitation_or_passive_pads_modelled);
out=char(a.OutputDir);assert(~isfolder(out),'Use a fresh output directory');mkdir(out);
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'tools'));
cfg=struct('scale_nodes_to_si',true,'length_unit','mm');maxNumCompThreads(6);
assert(strcmp(trkg4_file_sha256(g.mesh_path),g.mesh_sha256));
assert(strcmp(trkg4_file_sha256(g.observations_plan),g.observations_plan_sha256));
assert(strcmp(trkg4_file_sha256(g.parent_mesh),g.parent_mesh_sha256));
assert(strcmp(trkg4_file_sha256(g.native_body),g.native_body_sha256));
assert(strcmp(trkg4_file_sha256(g.geometry_source_path),g.geometry_source_sha256));
assert(strcmp(trkg4_file_sha256(g.parent_prepared_materials_path),g.parent_prepared_materials_sha256));
M=load(g.mesh_path);mm=struct('name','BMSTU individual homogeneous CT');
mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
mm.normalize_measurements=0;clear M;
cfg.scale_nodes_to_si=true;cfg.length_unit='mm';
fprintf('BMSTU %s: homogeneous sum of own immutable EIDORS bulk matrices\n',g.subject_alias);
S=load(g.parent_prepared_materials_path,'parts','identity','mm');
assert(strcmp(S.identity.subject_alias,g.subject_alias)&&strcmp(S.identity.mesh_sha256,g.parent_mesh_sha256));
assert(strcmp(S.identity.bulk_matrix_units,'S for sigma=1 S/m'));
assert(isequal(size(S.parts),[2,5]));field_kernel_hash=S.identity.field_kernel_sha256;
same_mesh=isequal(S.mm.nodes,mm.nodes)&&isequal(double(S.mm.elems),mm.elems)&&isequal(double(S.mm.boundary),mm.boundary);
if same_mesh
    K0=sparse(size(mm.nodes,1),size(mm.nodes,1));coarse=K0;
    for k=1:5,K0=K0+S.parts{2,k};coarse=coarse+S.parts{1,k};end
    partition_error=norm(K0-coarse,'fro')/norm(K0,'fro');assert(partition_error<1e-12);
    clear S coarse;bulk_origin='own_immutable_prepared_EIDORS_material_matrices';
else
    clear S;cfg=trkg4_config(g.subject_alias);trkg4_prepare_runtime(cfg);
    f=eidors_obj('fwd_model','BMSTU own refined homogeneous CT');
    f.nodes=mm.nodes;f.elems=mm.elems;f.boundary=mm.boundary;f.gnd_node=1;
    f.electrode=struct([]);f.stimulation=struct([]);f.normalize_measurements=0;
    f.solve=@fwd_solve_1st_order;f.system_mat=@system_mat_1st_order;f.jacobian=@jacobian_adjoint;
    F=system_mat_fields(trkg4_scale_fmdl_to_si(f,cfg));K0=F'*F;clear F f;
    partition_error=[];field_kernel_hash=trkg4_file_sha256(which('system_mat_fields'));
    bulk_origin='fresh_canonical_EIDORS_on_geometry_preserving_refinement';
end
assert(norm(K0-K0','fro')/norm(K0,'fro')<1e-12);
ctx=trkg4_local_patch_context(mm);nc=numel(g.contacts);C=cell(nc,1);contact_rows=struct([]);
for ci=1:nc
    contact=g.contacts(ci);xyz=contact.centres_lps_mm;
    assert(isequal(size(xyz),[4,3]));
    fprintf('BMSTU %s: finite contact QC L=%g mm\n',g.subject_alias,contact.size_mm);
    [el,faces,certificate]=trkg4_local_exact_patches(ctx,xyz,g.nominal_area_mm2,g.z_contact_ohm_m2);
    allnodes=[];
    for k=1:4
        ff=mm.boundary(faces{k},:);el(k).faces=ff;
        induced=find(all(ismember(mm.boundary,el(k).nodes),2));assert(isequal(induced(:),faces{k}(:)));
        area=ctx.canonical.face_area(faces{k});actual=sum(area);
        centroid=sum(ctx.face_centroid(faces{k},:).*area,1)/actual;
        offset=norm(centroid-xyz(k,:));extent=max(vecnorm(mm.nodes(el(k).nodes,:)-xyz(k,:),2,2));
        row=struct('size_mm',contact.size_mm,'electrode',k,'nominal_area_mm2',g.nominal_area_mm2,...
            'actual_area_mm2',actual,'area_relative_error',abs(actual/g.nominal_area_mm2-1),...
            'centroid_error_mm',offset,'extent_mm',extent,'centre_lps_mm',xyz(k,:),...
            'actual_centroid_lps_mm',centroid,'boundary_faces',numel(faces{k}));
        contact_rows=append(contact_rows,row);allnodes=[allnodes;el(k).nodes(:)]; %#ok<AGROW>
    end
    assert(numel(allnodes)==numel(unique(allnodes)),'Overlapping finite pads');
    current=contact_rows([contact_rows.size_mm]==contact.size_mm);
    write_json(fullfile(out,sprintf('contacts_L%d.local.json',contact.size_mm)),struct('contacts',current,'certificate',certificate));
    assert(all([current.area_relative_error]<=g.contact_area_relative_error_max),'Finite contact area QC failed');
    assert(all([current.centroid_error_mm]<=g.contact_centroid_error_mm_max),'Finite contact centroid QC failed');
    assert(all([current.extent_mm]<=g.contact_extent_mm_max),'Finite contact extent QC failed');
    model=mm;model.electrode=el;
    C{ci}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(model,cfg));
    assert(norm(C{ci}-C{ci}',1)/max(1,norm(C{ci},1))<1e-12);
    assert(norm(C{ci}*ones(size(C{ci},1),1),inf)/max(1,norm(C{ci},inf))<1e-12);
end
identity=struct('geometry_plan_sha256',trkg4_file_sha256(a.InputFile),'mesh_sha256',g.mesh_sha256,...
    'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
    'fit_sha256',trkg4_file_sha256(which('trkg4_uniform_rho_fit')),...
    'response_kernel_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
    'field_kernel_sha256',field_kernel_hash,'bulk_origin',bulk_origin,...
    'prepared_materials_sha256',g.parent_prepared_materials_sha256,...
    'patch_kernel_sha256',trkg4_file_sha256(which('electrode_faces_by_area')),...
    'contact_kernel_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')));
answers=struct([]);clock=tic;
for oi=1:numel(g.observations)
    o=g.observations(oi);ci=find([g.contacts.size_mm]==o.size_mm);assert(isscalar(ci));
    assert(strcmp(trkg4_file_sha256(o.source_path),o.source_sha256));
    assert(strcmp(trkg4_file_sha256(o.annotation_path),o.annotation_sha256));
    fprintf('BMSTU %s %s L=%g mm Z=%.8g Ohm\n',g.subject_alias,o.record_id,o.size_mm,o.Z_ohm);
    answer=trkg4_uniform_rho_fit(K0,C{ci},o.Z_ohm);
    answer.record_id=o.record_id;answer.subject_alias=g.subject_alias;answer.size_mm=o.size_mm;
    answer.annotation_status=o.annotation_status;answer.status='exploratory_hypothesis_not_validated';
    answer.geometry_is_individual=true;answer.all_materials_homogeneous=true;
    answers=append(answers,answer);write_json(fullfile(out,[o.record_id '.local.json']),answer);
end
report=struct('schema','bmstu_uniform_rho_results_v1','subject_alias',g.subject_alias,...
    'identity',identity,'answers',answers,'contact_qc',contact_rows,'placement',g.placement,...
    'mesh_nodes',size(mm.nodes,1),'mesh_elements',size(mm.elems,1),'mesh_QC',g.mesh_QC,...
    'partition_q1024_q4096_relative_error',partition_error,...
    'contact_diameter_mm',g.nominal_diameter_mm,'z_contact_ohm_m2',g.z_contact_ohm_m2,...
    'four_lateral_contacts_only',true,'TTRKG_coupling_modelled',false,...
    'acquisition_frequency_verified',false,'elapsed_solve_seconds',toc(clock),...
    'global_mesh_convergence_proven',false,'quantitative_result_accepted',false,...
    'status','exploratory_hypothesis_not_validated');
write_json(fullfile(out,'results.local.json'),report);
fprintf('BMSTU COMPLETE %s %d records\n',g.subject_alias,numel(answers));
end
function a=append(a,b)
if isempty(a),a=b;else,a(end+1)=b;end
end
function write_json(path,value)
assert(~isfile(path));f=fopen(path,'w','n','UTF-8');assert(f>0);guard=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value,PrettyPrint=true));
end
