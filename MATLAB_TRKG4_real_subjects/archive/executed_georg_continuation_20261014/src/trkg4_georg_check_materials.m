function [cache,identity]=trkg4_georg_check_materials(cfg,base)
% Material operators for fixed-parameter numerical checks, never inversion.
% The same P1 integral-mean assembly as trkg4_georg_lateral_materials, with
% explicitly named point counts; allows a stronger rule without renaming it.
planfile=fullfile(base,'plan.local.json');plan=jsondecode(fileread(planfile));
points=reshape(plan.numerical_quadrature_points,1,[]);
fraction_files=arrayfun(@(q)fullfile(base,sprintf('fractions_q%d.mat',q)),points,'UniformOutput',false);
fraction_hashes=cellfun(@trkg4_file_sha256,fraction_files,'UniformOutput',false);
signature=trkg4_georg_material_rules(cfg,points,fraction_hashes,...
 trkg4_file_sha256(which('trkg4_scale_fmdl_to_si')));
for j=1:numel(plan.sources)
 assert(strcmp(trkg4_file_sha256(plan.sources(j).path),plan.sources(j).sha256),'Numerical-check input changed');
end
parent=char(plan.parent_directory);
parentplan=jsondecode(fileread(fullfile(parent,'plan.local.json')));
if isfield(parentplan,'validation_scope')
 [parentcache,~]=trkg4_georg_check_materials(cfg,parent);
else
 [parentcache,~]=trkg4_georg_lateral_materials(cfg,parent);
end
identity=struct('plan_sha256',trkg4_file_sha256(planfile),...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'parent_prepared_sha256',trkg4_file_sha256(parentcache),...
 'mesh_sha256',trkg4_file_sha256(fullfile(base,'mesh.mat')),...
 'field_builder_sha256',trkg4_file_sha256(which('system_mat_fields')),...
 'rule_helper_sha256',trkg4_file_sha256(which('trkg4_georg_material_rules')),...
 'quadrature_points',points,'material_rules',signature);
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(uint8(unicode2native(jsonencode(identity),'UTF-8')));
key=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2).',1,[]));
cache=fullfile(base,['prepared_check_' key '.mat']);
if isfile(cache)
 saved=load(cache,'identity');assert(isequal(saved.identity,identity),'Stale numerical-check operators');return;
end
P=load(parentcache,'mm');mm=P.mm;clear P;
M=load(fullfile(base,'mesh.mat'));mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);clear M;
F=system_mat_fields(trkg4_scale_fmdl_to_si(mm,cfg));parts=cell(2,5);
for rule=1:2
 fraction=load(fraction_files{rule});
 assert(isequal(size(fraction.fractions),[size(mm.elems,1),5])&&...
  all(sum(fraction.fractions,2)==1)&&all(fraction.fractions(:)>=0),'Invalid material fractions');
 for j=1:5
  w=repelem(fraction.fractions(:,j),3);B=spdiags(sqrt(w),0,numel(w),numel(w))*F;
  parts{rule,j}=B'*B;fprintf('CHECK material q%d tissue=%d ready\n',points(rule),j);
 end
end
clear F B w fraction;
coarse=sparse(size(parts{1,1},1),size(parts{1,1},2));fine=coarse;
for j=1:5,coarse=coarse+parts{1,j};fine=fine+parts{2,j};end
homogeneous_bulk_relative_difference=norm(coarse-fine,'fro')/norm(coarse,'fro');
assert(homogeneous_bulk_relative_difference<1e-12,'Integration changed homogeneous operator');
clear coarse fine;
save(cache,'parts','mm','identity','homogeneous_bulk_relative_difference','-v7.3');
end
