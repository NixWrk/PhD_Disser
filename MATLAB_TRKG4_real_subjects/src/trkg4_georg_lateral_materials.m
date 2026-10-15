function [cache,identity]=trkg4_georg_lateral_materials(cfg,base)
% Full EIDORS assembly with explicit rule/units provenance, including legacy
% q256/q1024 base plans. Historical caches are never relabelled or overwritten.
planfile=fullfile(base,'plan.local.json');plan=jsondecode(fileread(planfile));
if isfield(plan,'numerical_quadrature_points')
 points=reshape(plan.numerical_quadrature_points,1,[]);
 if isfield(plan,'integration_stability')
  declaration=plan.integration_stability;
  assert(strcmp(declaration.schema,'georg_integration_stability_v1')&&...
   isnumeric(points)&&numel(points)==2&&all(isfinite(points))&&...
   all(points>0)&&all(points==fix(points))&&points(2)>points(1)&&...
   all(points<=2^30)&&all(log2(points)==fix(log2(points)))&&...
   isequal(points,reshape(declaration.points,1,[]))&&...
   isequal(points,reshape(plan.integration_rules,1,[])),...
   'trkg4:invalidIntegrationDeclaration','Invalid declared nested integration rules');
  assert(isnumeric(declaration.seed)&&isscalar(declaration.seed)&&...
   isfinite(declaration.seed)&&declaration.seed==fix(declaration.seed)&&...
   declaration.seed>=0&&declaration.seed<2^32&&...
   isequal(declaration.seed,plan.integration_seed)&&...
   isequal(declaration.all_mixed_tetrahedra_resampled,true)&&...
   isequal(declaration.fractions_are_nested,true)&&...
   isequal(declaration.unchanged_parent_elements_reused,0)&&...
   strcmp(declaration.coordinate_system,'LPS')&&strcmp(declaration.coordinate_units,'mm')&&...
   ~isfield(plan,'parent_directory'),...
   'trkg4:invalidIntegrationDeclaration','Invalid integration seed/sampling/reuse contract');
 else
  assert(isequal(points,[1024,4096]));
 end
else
 points=reshape(plan.integration_rules,1,[]);assert(isequal(points,[256,1024]));
end
for j=1:numel(plan.sources)
 assert(strcmp(trkg4_file_sha256(plan.sources(j).path),plan.sources(j).sha256),'Refined integration input changed');
end
fraction_files=arrayfun(@(q)fullfile(base,sprintf('fractions_q%d.mat',q)),points,'UniformOutput',false);
fraction_hashes=cellfun(@trkg4_file_sha256,fraction_files,'UniformOutput',false);
if isfield(plan,'integration_stability')
 declared=plan.integration_stability.fraction_sources;
 assert(numel(declared)==2,'trkg4:invalidIntegrationDeclaration','Expected two fraction identities');
 for j=1:2
  assert(declared(j).points==points(j)&&strcmp(declared(j).sha256,fraction_hashes{j}),...
   'trkg4:invalidIntegrationDeclaration','Declared integration fraction identity changed');
 end
end
signature=trkg4_georg_material_rules(cfg,points,fraction_hashes,...
 trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')));
parentcache='';parentid=[];parent_hash='';
if isfield(plan,'parent_directory')
 parent=char(plan.parent_directory);[parentcache,parentid]=trkg4_georg_lateral_materials(cfg,parent);
 parent_hash=trkg4_file_sha256(parentcache);
end
identity=struct('plan_sha256',trkg4_file_sha256(planfile),...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'parent_prepared_sha256',parent_hash,...
 'mesh_sha256',trkg4_file_sha256(fullfile(base,'mesh.mat')),...
 'field_builder_sha256',trkg4_file_sha256(which('system_mat_fields')),...
 'rule_helper_sha256',trkg4_file_sha256(which('trkg4_georg_material_rules')),...
 'quadrature_points',points,'material_rules',signature);
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(uint8(unicode2native(jsonencode(identity),'UTF-8')));
key=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
cache=fullfile(base,['prepared_material_' key '.mat']);
if isfile(cache)
 saved=load(cache,'identity');assert(isequal(saved.identity,identity),'Refined material cache changed');return;
end
parts=cell(2,5);parent_rows=zeros(1,2);
if isempty(parentcache)
 mm=eidors_obj('fwd_model','Georg shared CT-derived model');
 mm.gnd_node=1;mm.electrode=struct([]);mm.stimulation=struct([]);
 mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;
 mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;
else
 S=load(parentcache);mm=S.mm;
 if strcmp(identity.mesh_sha256,parentid.mesh_sha256)&&...
   strcmp(identity.field_builder_sha256,parentid.field_builder_sha256)
  [~,parent_rows]=trkg4_georg_material_rules(cfg,points,fraction_hashes,...
   signature.scale_helper_sha256,parentid.material_rules);
  for rule=find(parent_rows),parts(rule,:)=S.parts(parent_rows(rule),:);end
 end
 clear S;
end
M=load(fullfile(base,'mesh.mat'));mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);clear M;
rules=find(parent_rows==0);
if ~isempty(rules),F=system_mat_fields(trkg4_scale_fmdl_to_si(mm,cfg));end
for rule=rules
 fraction=load(fraction_files{rule});
 assert(isequal(size(fraction.fractions),[size(mm.elems,1),5])&&...
  all(sum(fraction.fractions,2)==1)&&all(fraction.fractions(:)>=0),'Invalid material fractions');
 for j=1:5
  w=repelem(fraction.fractions(:,j),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
  parts{rule,j}=B'*B;fprintf('NIX material q%d tissue=%d ready\n',points(rule),j);
 end
end
clear F B w fraction;
coarse=sparse(size(parts{1,1},1),size(parts{1,1},2));fine=coarse;
for j=1:5,coarse=coarse+parts{1,j};fine=fine+parts{2,j};end
homogeneous_bulk_relative_difference=norm(coarse-fine,'fro')/norm(coarse,'fro');
assert(homogeneous_bulk_relative_difference<1e-12,'Integration changed homogeneous conductivity operator');
fprintf('NIX homogeneous operator difference %.8g\n',homogeneous_bulk_relative_difference);
clear coarse fine;save(cache,'parts','mm','identity','homogeneous_bulk_relative_difference','-v7.3');
end
