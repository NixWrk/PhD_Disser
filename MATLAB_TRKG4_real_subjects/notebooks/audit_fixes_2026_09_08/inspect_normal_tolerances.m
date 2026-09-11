function inspect_normal_tolerances()
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));cd(root);addpath('src');
S=load('output/nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat','fmdl_mesh_units');
f=S.fmdl_mesh_units;p=f.nodes;t=f.elems;b=f.boundary;
all_faces=sort([t(:,[1 2 3]);t(:,[1 2 4]);t(:,[1 3 4]);t(:,[2 3 4])],2);
[found,ids]=ismember(sort(b,2),all_faces,'rows');assert(all(found));owner=mod(ids-1,size(t,1))+1;
p1=p(b(:,1),:);ab=p(b(:,2),:)-p1;ac=p(b(:,3),:)-p1;
normal=cross(ab,ac,2);fc=(p1+p(b(:,2),:)+p(b(:,3),:))/3;
tc=(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:)+p(t(:,4),:))/4;
dot_old=dot(normal,tc(owner,:)-fc,2);
scale=max(max(p,[],1)-min(p,[],1));tol=100*eps(scale^3);
indices=find(abs(dot_old)<=tol);R=cell(numel(indices),1);
for j=1:numel(indices)
 k=indices(j); other=setdiff(t(owner(k),:),b(k,:));assert(numel(other)==1);
 d=p(other,:)-p1(k,:);detval=dot(normal(k,:),d);
 op_scale=sum(abs(normal(k,:).*d));op_tol=100*eps(max(1,op_scale));
 R{j}=struct('boundary_face',k,'owner_tetra',owner(k),'centroid_dot_mm3',dot_old(k), ...
 'global_tolerance_mm3',tol,'opposite_vertex_determinant_mm3',detval, ...
 'local_operation_scale_mm3',op_scale,'local_operation_tolerance_mm3',op_tol, ...
 'tetra_volume_mm3',abs(detval)/6,'longest_face_edge_mm',max([norm(ab(k,:)),norm(ac(k,:)),norm(ab(k,:)-ac(k,:))]));
 disp(R{j});
end
fid=fopen('notebooks/audit_fixes_2026_09_08/normal_tolerance_probe.json','w');c=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));
end
