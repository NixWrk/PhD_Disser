function export_lateral_fat_contacts()
% Export exactly the saved finite contact triangles, without rebuilding them.
cfg=trkg4_config('nik');root=cfg.project_root;
base=fullfile(root,'output/exploratory/lateral_array_design_20260916');
out=fullfile(root,'output/exploratory/lateral_fat_inverse_20260916');
if ~isfolder(out),mkdir(out);end
S=load(fullfile(base,'prepared_fem.mat'),'electrodes','sizes');
M=load(fullfile(base,'refined_mesh.mat'),'nodes_mm');items=struct([]);
for k=1:numel(S.sizes)
 for e=1:4
  faces=S.electrodes{k}(e).faces;[ids,~,ix]=unique(faces(:));tri=reshape(ix,size(faces));
  row=struct('L_mm',S.sizes(k),'electrode',e,'nodes_mm',M.nodes_mm(ids,:),...
   'triangles_zero_based',tri-1);
  if isempty(items),items=row;else,items(end+1)=row;end
 end
end
result=struct('contacts',items,'prepared_sha256',trkg4_file_sha256(fullfile(base,'prepared_fem.mat')),...
 'mesh_sha256',trkg4_file_sha256(fullfile(base,'refined_mesh.mat')),...
 'exporter_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
fid=fopen(fullfile(out,'contacts.local.json'),'w','n','UTF-8');assert(fid>=0);
c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(result));
end
