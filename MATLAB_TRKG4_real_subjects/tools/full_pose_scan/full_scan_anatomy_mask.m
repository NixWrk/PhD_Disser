function mask=full_scan_anatomy_mask(fmdl,preparation_report)
%FULL_SCAN_ANATOMY_MASK Explicit search-domain mask; bulk mesh is unchanged.
% Artificial arm cylinders come from the subject's preparation provenance.
% Faces touching those volumes are excluded conservatively. Both planar
% extreme-z closure caps are excluded; neighbouring skin is tested by patch.
records=jsondecode(fileread(preparation_report));
if iscell(records)
    body=[];for k=1:numel(records),if isfield(records{k},'arm_extension'),body=records{k};break;end;end
else
    body=records(find(arrayfun(@(x)isfield(x,'arm_extension')&&~isempty(x.arm_extension),records),1));
end
assert(~isempty(body)&&isfield(body,'arm_extension'),'full_scan:anatomyInput','Missing recorded arm geometry');
x=fmdl.nodes;f=fmdl.boundary;n=size(f,1);
zlo=min(x(f(:),3));zhi=max(x(f(:),3));tol=1e-6;arm_tolerance=2;
z=reshape(x(f(:),3),size(f));codes=zeros(n,1,'uint8');
codes(all(abs(z-zlo)<=tol,2))=uint8(1);
codes(all(abs(z-zhi)<=tol,2))=uint8(2);
for side={'left','right'}
    arm=body.arm_extension.(side{1});
    radial=hypot(x(:,2)-arm.center_y,x(:,3)-arm.center_z);
    inside=x(:,1)>=arm.cylinder_x_min-arm_tolerance & ...
        x(:,1)<=arm.cylinder_x_max+arm_tolerance & radial<=arm.radius+arm_tolerance;
    faces_touch=any(reshape(inside(f(:)),size(f)),2);
    codes(faces_touch)=uint8(3);
end
assert(any(codes==1)&&any(codes==2)&&any(codes==3),'full_scan:anatomyInput','Expected caps and artificial arms were not found');
mask=struct('schema','trkg4_search_skin_mask_v1','face_code',codes,'allowed_faces',codes==0, ...
    'mesh_identity',full_scan_mesh_identity(fmdl), ...
    'preparation_sha256',trkg4_file_sha256(preparation_report), ...
    'z_caps_mm',[zlo,zhi],'cap_tolerance_mm',tol,'arm_tolerance_mm',arm_tolerance, ...
    'arms',body.arm_extension,'counts',[sum(codes==0),sum(codes==1),sum(codes==2),sum(codes==3)], ...
    'labels',{{'allowed_skin','lower_artificial_cap','upper_artificial_cap','artificial_arm'}}, ...
    'status','explicit_geometric_search_domain_not_manual_anatomical_segmentation', ...
    'rationale','Author excludes arms and lower STL cut; upper closure cap is also not skin. Contact triangles touching the recorded arm volumes have a conservative 2 mm mesh tolerance.');
end
