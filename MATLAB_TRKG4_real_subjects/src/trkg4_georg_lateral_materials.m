function [cache,identity]=trkg4_georg_lateral_materials(cfg,base)
% Unchanged full EIDORS assembly, optionally using nested q1024/q4096.
planfile=fullfile(base,'plan.local.json');plan=jsondecode(fileread(planfile));
if ~isfield(plan,'numerical_quadrature_points')
 identity=run_georg_ct_model('OutputDir',base,'Stage','prepare');
 cache=fullfile(base,['prepared_' identity.plan_sha256(1:8) '_' identity.worker_sha256(1:8) '.mat']);return;
end
assert(isequal(plan.numerical_quadrature_points(:)',[1024,4096]));
for j=1:numel(plan.sources)
 assert(strcmp(trkg4_file_sha256(plan.sources(j).path),plan.sources(j).sha256),'Refined integration input changed');
end
parent=char(plan.parent_directory);[parentcache,parentid]=trkg4_georg_lateral_materials(cfg,parent);
identity=struct('plan_sha256',trkg4_file_sha256(planfile),...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'parent_prepared_sha256',trkg4_file_sha256(parentcache),...
 'field_builder_sha256',trkg4_file_sha256(which('system_mat_fields')),...
 'quadrature_points',[1024,4096]);
cache=fullfile(base,['prepared_nix_' identity.plan_sha256(1:8) '_' identity.worker_sha256(1:8) '.mat']);
if isfile(cache)
 saved=load(cache,'identity');assert(isequal(saved.identity,identity),'Refined material cache changed');return;
end
same_mesh=strcmp(trkg4_file_sha256(fullfile(base,'mesh.mat')),trkg4_file_sha256(fullfile(parent,'mesh.mat')));
S=load(parentcache);mm=S.mm;parts=cell(2,5);
if same_mesh
 parts(1,:)=S.parts(2,:);rules=2;
else
 M=load(fullfile(base,'mesh.mat'));mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
 rules=1:2;clear M;
end
clear S;
F=system_mat_fields(trkg4_scale_fmdl_to_si(mm,cfg));
points=[1024,4096];
for rule=rules
 fraction=load(fullfile(base,sprintf('fractions_q%d.mat',points(rule))));assert(all(sum(fraction.fractions,2)==1));
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
