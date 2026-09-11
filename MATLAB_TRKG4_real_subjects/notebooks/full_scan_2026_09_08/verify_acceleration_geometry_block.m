function report=verify_acceleration_geometry_block()
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'src'),fullfile(root,'tools'),fullfile(root,'tools','full_pose_scan'));
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
cfg.local_patch_options=struct('cache_tile_mm',8,'cache_limit',64);
s=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'),'fmdl_mesh_units');f=s.fmdl_mesh_units;
projection=full_scan_context(f);pc=full_scan_patch_context(f);
mask=full_scan_anatomy_mask(f,fullfile(root,'output','nik_stl_preparation_report.json'));
ref=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
xyz=[ref.x_mm ref.y_mm ref.z_mm];ax=xyz(4,:)-xyz(1,:);ax=ax/norm(ax);
[c0,n0]=full_scan_project(projection,mean(xyz,1));u0=ax-dot(ax,n0)*n0;u0=u0/norm(u0);v0=cross(n0,u0);v0=v0/norm(v0);
uv=[0,-30;1,-30;-80,-120;80,-120];[P,N]=full_scan_project(projection,c0+uv(:,1)*u0+uv(:,2)*v0);
centres=table((1:4)',uv(:,1),uv(:,2),P(:,1),P(:,2),P(:,3), ...
    'VariableNames',{'centre_index','u_mm','v_mm','x_mm','y_mm','z_mm'});
old=jsondecode(fileread(fullfile(root,'output','full_pose_scan_fullscan_20260908_v2','run_manifest.json')));
[sizes,~]=trkg4_load_nik_inhale_data(root,'manifest');
d=struct('root',root,'cfg',cfg,'f',f,'projection',projection,'patch_context',pc, ...
    'P',P,'N',N,'u0',u0,'sizes',sizes,'centres',centres,'grid',struct('phi',-20:20), ...
    'physics',old.physics,'fingerprint','bounded_geometry_check', ...
    'opt',struct('Workers',2,'ReuseGeometryTag','fullscan_20260908_v2'));
pool=gcp('nocreate');if isempty(pool),parpool('Threads',2);end
fresh=full_scan_geometry_block(d,1:4,mask,[]);
[reused,~]=full_scan_reuse_geometry(d,1:4,[]);imported=full_scan_geometry_block(d,1:4,mask,reused);
save(fullfile(root,'output','exploratory','fullscan_acceleration_20260909','geometry_block_debug.mat'),'fresh','imported','reused','-v7');
assert(isequal(fresh.valid_rows,imported.valid_rows));
different=fresh.status_code~=imported.status_code;
% Historical rejected rows have no saved contacts: their precise cause is unknown.
assert(all(fresh.status_code(different)==3 & imported.status_code(different)==2 & reused.status_code(different)==2));
assert(isequal(fresh.patch_faces,imported.patch_faces),'Fresh/imported contact sets differ');
assert(any(fresh.status_code==3)&&any(fresh.status_code==1)&&fresh.geometry_stats.subset_cache_hits>0);
single=full_scan_geometry_block(d,3,mask,[]);assert(isequal(size(single.centre_ids),[41,1]));
assert(all(single.status_code>0));
report=struct('passed',true,'poses',164,'exact_valid_rows_and_contact_faces',true,'historical_unspecified_rejections',sum(different), ...
    'valid',sum(fresh.status_code==1),'QC_rejected',sum(fresh.status_code==2),'outside',sum(fresh.status_code==3), ...
    'fresh_stats',fresh.geometry_stats,'imported_stats',imported.geometry_stats,'single_centre_shape_correct',true);
full_scan_write_json(fullfile(root,'output','exploratory','fullscan_acceleration_20260909','geometry_block_checks.json'),report);disp(report);
end
