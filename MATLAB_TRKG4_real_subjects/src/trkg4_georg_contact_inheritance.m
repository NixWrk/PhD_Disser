function [inherited,identity]=trkg4_georg_contact_inheritance(mm,plan,current_mesh_file)
% The pose reference and physical-contact master are independent contracts.
% Prove that child boundary triangles reproduce the contact master's boundary.
master=char(plan.exact_contact_parent_mesh);mapfile=char(plan.boundary_parent_map);
identity=struct('contact_parent_mesh_sha256',trkg4_file_sha256(master),...
 'current_mesh_sha256',trkg4_file_sha256(current_mesh_file),...
 'boundary_parent_map_sha256','','mapping_helper_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
inherited=[];
if isempty(mapfile)
 assert(strcmp(identity.contact_parent_mesh_sha256,identity.current_mesh_sha256),...
  'trkg4:wrongContactMaster','Direct contacts require the current contact-master mesh');return;
end
identity.boundary_parent_map_sha256=trkg4_file_sha256(mapfile);
M=load(master);parent_mm=mm;parent_mm.nodes=M.nodes_mm;
parent_mm.elems=double(M.elems);parent_mm.boundary=double(M.boundary);clear M;
maps=load(mapfile);parent=double(maps.boundary_parent(:));nf=size(parent_mm.boundary,1);
assert(numel(parent)==size(mm.boundary,1)&&all(parent>=1&parent<=nf&parent==fix(parent)),...
 'trkg4:wrongContactMap','Contact inheritance must address every child boundary face with one-based master IDs');
counts=accumarray(parent,1,[nf,1]);assert(all(counts>=1),'trkg4:wrongContactMap','A contact-master boundary face is missing');
if isfield(maps,'boundary_child_counts')
 assert(isequal(counts,double(maps.boundary_child_counts(:))),'trkg4:wrongContactMap','Declared child counts changed');
end
area=zeros(nf,1);moments=zeros(nf,3);maxplane=0;
for first=1:50000:numel(parent)
 ids=first:min(first+49999,numel(parent));p=parent(ids);
 a=parent_mm.nodes(parent_mm.boundary(p,1),:);
 ab=parent_mm.nodes(parent_mm.boundary(p,2),:)-a;
 ac=parent_mm.nodes(parent_mm.boundary(p,3),:)-a;
 normal=cross(ab,ac,2);normal_norm=vecnorm(normal,2,2);
 v1=mm.nodes(mm.boundary(ids,1),:);v2=mm.nodes(mm.boundary(ids,2),:);v3=mm.nodes(mm.boundary(ids,3),:);
 for v={v1,v2,v3}
  delta=v{1}-a;dist=abs(sum(delta.*normal,2))./normal_norm;maxplane=max(maxplane,max(dist));
  d00=sum(ab.^2,2);d01=sum(ab.*ac,2);d11=sum(ac.^2,2);
  d20=sum(delta.*ab,2);d21=sum(delta.*ac,2);den=d00.*d11-d01.^2;
  u=(d11.*d20-d01.*d21)./den;w=(d00.*d21-d01.*d20)./den;
  assert(all(u>=-1e-8&w>=-1e-8&u+w<=1+1e-8),'trkg4:wrongContactMap','A child exits its declared master triangle');
 end
 child_normal=cross(v2-v1,v3-v1,2);assert(all(sum(child_normal.*normal,2)>0),...
  'trkg4:wrongContactMap','Inherited boundary orientation changed');
 child_area=vecnorm(child_normal,2,2)/2;area=area+accumarray(p,child_area,[nf,1]);
 centre=(v1+v2+v3)/3;
 for d=1:3,moments(:,d)=moments(:,d)+accumarray(p,child_area.*centre(:,d),[nf,1]);end
end
f=parent_mm.boundary;n=parent_mm.nodes;
master_area=vecnorm(cross(n(f(:,2),:)-n(f(:,1),:),n(f(:,3),:)-n(f(:,1),:),2),2,2)/2;
area_error=max(abs(area./master_area-1));
centroid_error=max(vecnorm(moments./area-(n(f(:,1),:)+n(f(:,2),:)+n(f(:,3),:))/3,2,2));
assert(maxplane<1e-9&&area_error<1e-10&&centroid_error<1e-9,...
 'trkg4:wrongContactMap','Contact-master polyhedron or boundary moments changed');
identity.master_boundary_identity=struct('maximum_plane_distance_mm',maxplane,...
 'maximum_relative_area_difference',area_error,'maximum_centroid_difference_mm',centroid_error,...
 'one_based_mapping_verified',true);
inherited=struct('coarse_mm',parent_mm,'coarse_context',trkg4_local_patch_context(parent_mm),...
 'child_faces',{accumarray(parent,(1:numel(parent))',[nf,1],@(v){v})});
end
