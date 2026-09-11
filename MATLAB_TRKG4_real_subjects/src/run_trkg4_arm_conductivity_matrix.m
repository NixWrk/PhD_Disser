function report = run_trkg4_arm_conductivity_matrix(varargin)
% Full 3^3 conductivity design on 16 montages, exact affine FEM assembly.
% EIDORS supplies volume stiffness. The CEM triangle integral is checked
% against EIDORS on every montage. Every matrix entry is a new direct solve.
p=inputParser;
addParameter(p,'MontageIndices',1:16,@isnumeric);
addParameter(p,'CaseIndices',1:27,@isnumeric);
addParameter(p,'OutputSubdirectory','output/exploratory/arm_sigma_20260908');
addParameter(p,'PrepareOnly',false,@islogical);
addParameter(p,'Solver','pardiso',@(x)ismember(string(x),["pardiso","matlab_direct"]));
addParameter(p,'ExportOnly',false,@islogical);
addParameter(p,'ValidateDerivatives',true,@islogical);
parse(p,varargin{:}); opt=p.Results;
cfg=trkg4_config('nik'); root=cfg.project_root;
cd(root); trkg4_prepare_runtime(cfg);
eidors_cache('cache_size',128*1024*1024);
maxNumCompThreads(8);
out=fullfile(root,opt.OutputSubdirectory); if ~isfolder(out),mkdir(out);end
study=trkg4_arm_conductivity_design(cfg);
mesh_file=fullfile(root,'output','accepted_arm_parameter_mesh.msh');
acceptance=jsondecode(fileread(fullfile(root,'output','accepted_arm_parameter_mesh_acceptance.json')));
assert(strcmp(acceptance.status,'accepted_for_parameterized_arm_electrode_method_development'));
assert(all(structfun(@(x)logical(x),acceptance.checks)));
mesh_hash=trkg4_file_sha256(mesh_file);
assert(strcmp(mesh_hash,acceptance.accepted_mesh_sha256),'Accepted mesh hash changed');
source_files={'src/run_trkg4_arm_conductivity_matrix.m','src/trkg4_arm_conductivity_design.m', ...
 'src/trkg4_arm_montage_spec.m','src/trkg4_build_arm_electrodes.m', ...
 'src/trkg4_build_arm_cross_section_electrodes.m','src/trkg4_build_mixed_arm_electrodes.m', ...
 'src/trkg4_cem_contact_matrix.m','src/trkg4_config.m','src/trkg4_subject_registry.m', ...
 'vendor_stl_eidors/electrode_faces_by_area.m','vendor_stl_eidors/assign_tissues_from_stl.m', ...
 'vendor_stl_eidors/points_in_closed_surface.m','vendor_stl_eidors/itis_conductivity.m', ...
 'tools/solve_arm_conductivity_matrix.py'};
input_files={mesh_file,cfg.body_stl,fullfile(root,'output','nik_stl_preparation_report.json')};
for t=1:numel(cfg.tissues)
 if cfg.tissues(t).enabled && isfile(cfg.tissues(t).file),input_files{end+1}=cfg.tissues(t).file;end
end
identity=struct('study',study,'mesh_sha256',mesh_hash,'matlab_version',version);
preparation_file=fullfile(out,'preparation_identity.json');
if isfile(preparation_file)
 identity.preparation_identity_sha256=trkg4_file_sha256(preparation_file);
 identity.prepared_fem_sha256=trkg4_file_sha256(fullfile(out,'prepared_fem.mat'));
end
for k=1:numel(source_files)
 identity.sources(k)=struct('path',source_files{k},'sha256',trkg4_file_sha256(fullfile(root,source_files{k})));
end
for k=1:numel(input_files)
 [~,nm,ex]=fileparts(input_files{k});
 identity.inputs(k)=struct('name',[nm ex],'sha256',trkg4_file_sha256(input_files{k}));
end
for k=1:3
 eidors_functions={'system_mat_fields','fwd_solve_1st_order','left_divide'};
 file=which(eidors_functions{k});
 identity.eidors(k)=struct('name',eidors_functions{k},'sha256',trkg4_file_sha256(file));
end
identity_file=fullfile(out,'identity.json');
if isfile(identity_file)
 old=jsondecode(fileread(identity_file));
 assert(isequaln(old,jsondecode(jsonencode(identity))),'Study inputs/source changed: use a new output directory');
else
 local_json(identity_file,identity);
end
prepared_file=fullfile(out,'prepared_fem.mat');
if ~isfile(prepared_file)
 cfg.prebuilt_mesh_file=mesh_file; cfg.mesh_target_size_mm=1;
 cfg.use_parallel_assignment=false;
 spec=trkg4_arm_montage_spec(cfg,'point_disc_5mm',20,80);
 cfg.electrode_centres_xyz=spec.centres_xyz_mm;cfg.electrode_centres_file='';
 cfg.electrode_area=spec.nominal_area_mm2(1);cfg.electrode_area_mm2=cfg.electrode_area;
 cfg.fail_on_electrode_diagnostics=false;cfg.electrode_diagnostics_file='';
 [fmdl_mm,~]=trkg4_build_eidors_model_from_stl(cfg);
 centres=element_centroids(fmdl_mm);
 [sigma0,tissue_id,tissue_names,tissue_counts]=assign_tissues_from_stl(centres,cfg);
 arms=spec.arm_geometry;
 distal=centres(:,1)>arms.right.cut_x | centres(:,1)<arms.left.cut_x;
 soft_id=find(strcmp(tissue_names,'soft_tissue'));lung_id=find(strcmp(tissue_names,'lungs'));
 heart_id=find(strcmp(tissue_names,'heart'));bone_id=find(strcmp(tissue_names,'bones'));
 assert(all([~isempty(soft_id),~isempty(lung_id),~isempty(heart_id),~isempty(bone_id)]));
 groups={tissue_id==soft_id & ~distal,tissue_id==soft_id & distal, ...
     tissue_id==heart_id,tissue_id==lung_id,tissue_id==bone_id};
 assert(all(sum(cell2mat(groups),2)==1),'Unexpected tissue class');
 fmdl=trkg4_scale_fmdl_to_si(fmdl_mm,cfg);
 fprintf('Preparing EIDORS volume stiffness blocks...\n');
 F=system_mat_fields(fmdl); ne=size(fmdl.elems,1); nn=size(fmdl.nodes,1);
 blocks=cell(5,1);
 for g=1:5
   rr=reshape((3*(find(groups{g})-1)+(1:3))',[],1);
   G=F(rr,:); blocks{g}=G'*G;
   fprintf('Block %d: %d elements, %d nonzeros\n',g,sum(groups{g}),nnz(blocks{g}));
 end
 clear F G rr centres;
 bone_sigma=cfg.tissues(strcmp({cfg.tissues.name},'bones')).sigma;
 save(prepared_file,'fmdl_mm','blocks','sigma0','tissue_id','tissue_names','tissue_counts','bone_sigma','-v7.3');
 fprintf('Prepared FEM saved.\n');
end
if opt.PrepareOnly,report=identity;return;end
load(prepared_file,'fmdl_mm','blocks','sigma0','tissue_id','tissue_names','tissue_counts','bone_sigma');
cfg.mesh_target_size_mm=1;
% Derivative groups: all soft tissue, whole heart, lungs, distal soft, torso soft.
Ksoft=blocks{1}+blocks{2}; Kheart=blocks{3}; Klung=blocks{4}; Kbone=blocks{5};
index=0;
for inner=study.inner_distances_mm
 for outer=study.outer_distances_mm
  for ki=1:numel(study.montage_kinds)
   index=index+1;if ~ismember(index,opt.MontageIndices),continue;end
   kind=study.montage_kinds{ki}; montage=sprintf('in%d_out%d_%s',inner,outer,kind);
   done_file=fullfile(out,[montage '_results.csv']);
   if isfile(done_file)
     prior=readtable(done_file);
     fd_needed=opt.ValidateDerivatives && inner==20 && outer==80 && ~isfile(fullfile(out,[montage '_derivative_qc.csv']));
     if all(ismember(opt.CaseIndices,prior.case_index)) && ~fd_needed,fprintf('Reusing %s\n',montage);continue;end
   end
   spec=trkg4_arm_montage_spec(cfg,kind,inner,outer);
   model=fmdl_mm;
   if strcmp(kind,'outer_planes_inner_rings')
    [model,patches]=trkg4_build_mixed_arm_electrodes(model,spec,cfg.z_contact,cfg);
   else
    [model.electrode,patches]=trkg4_build_arm_electrodes(model,spec,cfg.z_contact);
    patches.contact_geometry=repmat(string(kind),4,1);
   end
   [model.stimulation,model.meas_select]=trkg4_make_reciprocity_stimulation(cfg);
   fmdl=trkg4_scale_fmdl_to_si(model,cfg);
   C=trkg4_cem_contact_matrix(fmdl);
   n=size(fmdl.nodes,1); N=n+4; q=zeros(N,2);
   q(n+[1 4],1)=[1;-1];q(n+[2 3],2)=[1;-1];
   free=setdiff(1:N,fmdl.gnd_node);
   % Full EIDORS assembly at the reference proves equivalence of the
   % affine volume split and both boundary/internal contact formulations.
   img=mk_image(fmdl,1);img.elem_data=sigma0;
   exact=calc_system_mat(img);
   reference=local_matrix(study.baseline);
   matrix_error=norm(reference-exact.E,'fro')/norm(exact.E,'fro');
   assert(matrix_error<1e-12,'Affine matrix differs from EIDORS');
   clear exact reference;
   writetable(patches,fullfile(out,[montage '_patches.csv']));
   if strcmp(opt.Solver,'pardiso') || opt.ExportOnly
    ground=fmdl.gnd_node; montage_index=index; inner_mm=inner; outer_mm=outer; electrode_kind=kind;
    save(fullfile(out,[montage '_contact.mat']),'C','ground','montage_index','inner_mm','outer_mm','electrode_kind','matrix_error','-v7');
    fprintf('Exported checked CEM matrix: %s\n',montage);
    continue;
   end
   rows=struct([]);
   if isfile(done_file), rows=table2struct(readtable(done_file));end
   for ci=opt.CaseIndices
    if ~isempty(rows)&&any([rows.case_index]==ci),continue;end
    sig=study.parameter_matrix(ci,:); started=tic;
    K=local_matrix(sig); lastwarn('');
    v=zeros(N,2);v(free,:)=left_divide(K(free,free),q(free,:),1e-10);
    [wm,wi]=lastwarn;
    z=[q(:,2)'*v(:,1),q(:,1)'*v(:,2)];
    residual=norm(K(free,:)*v-q(free,:),'fro')/norm(q(free,:),'fro');
    reciprocity=abs(diff(z))/max(abs(z));
    d=[-v(:,2)'*(Ksoft*v(:,1)),-v(:,2)'*(Kheart*v(:,1)),-v(:,2)'*(Klung*v(:,1))];
    darm=-v(:,2)'*(blocks{2}*v(:,1));dtorso=-v(:,2)'*(blocks{1}*v(:,1));
    row=struct('montage_index',index,'montage',string(montage),'electrode_kind',string(kind), ...
      'inner_mm',inner,'outer_mm',outer,'case_index',ci, ...
      'sigma_soft',sig(1),'sigma_heart',sig(2),'sigma_lung',sig(3), ...
      'Z_direct_ohm',z(1),'Z_reciprocal_ohm',z(2),'reciprocity_relative_error',reciprocity, ...
      'relative_residual',residual,'matrix_relative_error',matrix_error, ...
      'dZ_dsigma_soft',d(1),'dZ_dsigma_heart',d(2),'dZ_dsigma_lung',d(3), ...
      'elasticity_soft',d(1)*sig(1)/z(1),'elasticity_heart',d(2)*sig(2)/z(1), ...
      'elasticity_lung',d(3)*sig(3)/z(1), ...
      'dZ_dsigma_soft_distal',darm,'dZ_dsigma_soft_torso',dtorso, ...
      'solver_warning',string(wi),'solver_warning_message',string(wm),'elapsed_seconds',toc(started));
    assert(all(isfinite([z,d,residual,reciprocity])),'Non-finite FEM result');
    assert(residual<1e-7 && reciprocity<1e-8,'Numerical QC failed');
    if isempty(rows),rows=row;else,rows(end+1)=row;end %#ok<AGROW>
    writetable(struct2table(rows),done_file);
    fprintf('%s case %d/27: Z=%.9g, H=%.4g, L=%.4g, residual %.2g, %.1fs\n', ...
      montage,ci,z(1),row.elasticity_heart,row.elasticity_lung,residual,row.elapsed_seconds);
   end
   % Centered finite differences at two step sizes, once per contact kind.
   if opt.ValidateDerivatives && inner==20 && outer==80
    fdfile=fullfile(out,[montage '_derivative_qc.csv']);
    if ~isfile(fdfile)
     K=local_matrix(study.baseline);v=zeros(N,2);v(free,:)=left_divide(K(free,free),q(free,:),1e-10);
     derivative=[-v(:,2)'*(Ksoft*v(:,1)),-v(:,2)'*(Kheart*v(:,1)),-v(:,2)'*(Klung*v(:,1))];
     fdrows=struct([]);
     for t=1:3
      for step=study.finite_difference_relative_steps
       h=step*study.baseline(t); zfd=zeros(1,2);
       for direction=1:2
        sig=study.baseline;sig(t)=sig(t)+(2*direction-3)*h;
        K=local_matrix(sig);w=left_divide(K(free,free),q(free,1),1e-10);
        zfd(direction)=q(free,2)'*w;
       end
       fd=diff(zfd)/(2*h);err=abs(fd-derivative(t))/max(abs(derivative(t)),1e-10);
       fdrow=struct('tissue_index',t,'relative_step',step,'adjoint',derivative(t),'finite_difference',fd,'relative_error',err);
       if isempty(fdrows),fdrows=fdrow;else,fdrows(end+1)=fdrow;end %#ok<AGROW>
       assert(err<0.002,'Derivative QC failed');
      end
     end
     writetable(struct2table(fdrows),fdfile);
    end
   end
  end
 end
end
if strcmp(opt.Solver,'pardiso') && ~opt.ExportOnly
 python=getenv('TRKG4_PYTHON');if isempty(python),python='python';end
 command=sprintf('"%s" "%s" --output "%s" --montages %s --cases %s', ...
  python,fullfile(root,'tools','solve_arm_conductivity_matrix.py'),out, ...
  strjoin(string(opt.MontageIndices),','),strjoin(string(opt.CaseIndices),','));
 [status,message]=system(command,'-echo');assert(status==0,'%s',message);
end
report=struct('status','requested_batch_completed','output_subdirectory',opt.OutputSubdirectory,'study',study);

 function K=local_matrix(sig)
   K=sig(1)*Ksoft+sig(2)*Kheart+sig(3)*Klung+bone_sigma*Kbone+C;
 end
end

function local_json(file,value)
f=fopen(file,'w','n','UTF-8');assert(f>=0);close=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value,'PrettyPrint',true));
end
