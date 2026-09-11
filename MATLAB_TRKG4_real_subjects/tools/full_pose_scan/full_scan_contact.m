function contact = full_scan_contact(fmdl_si, faces, z_contact)
%FULL_SCAN_CONTACT Compile one immutable finite-area contact, in SI units.
validateattributes(z_contact,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(faces,{'numeric'},{'vector','nonempty','integer','>=',1,'<=',size(fmdl_si.boundary,1)});
faces=double(faces(:));
assert(numel(unique(faces))==numel(faces),'full_scan:patchInput','Duplicate contact faces');
tri=fmdl_si.boundary(faces,:); nodes=unique(tri(:)); xyz=fmdl_si.nodes;
areas=0.5*vecnorm(cross(xyz(tri(:,2),:)-xyz(tri(:,1),:), ...
    xyz(tri(:,3),:)-xyz(tri(:,1),:),2),2,2);
assert(all(isfinite(areas)&areas>0),'full_scan:patchArea','Degenerate contact');
[~,local]=ismember(tri,nodes); mass=zeros(numel(nodes));
for a=1:3
    for b=1:3
        mass=mass+full(sparse(local(:,a),local(:,b), ...
            areas*((1+(a==b))/(12*z_contact)),numel(nodes),numel(nodes)));
    end
end
[R,flag]=chol(mass);
assert(flag==0,'full_scan:contactSPD','Contact mass is not SPD');
contact=struct('faces',faces,'triangles',tri,'nodes',nodes, ...
    'xyz',xyz(nodes,:),'z_contact',z_contact,'R',R);
end
