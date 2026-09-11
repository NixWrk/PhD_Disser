function report=verify_acceleration_real_geometry(coherent,tile,capacity)
if nargin<1,coherent=false;end
if nargin<2,tile=8;end
if nargin<3,capacity=64;end
suffix='';repetitions=3;
if coherent,suffix=sprintf('_coherent_t%g_c%d',tile,capacity);repetitions=1;end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'src'),fullfile(root,'tools'),fullfile(root,'tools','full_pose_scan'));
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);cfg.return_geometry_status=true;
s=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'),'fmdl_mesh_units');f=s.fmdl_mesh_units;
pc=full_scan_patch_context(f);projection=full_scan_context(f);
mask=full_scan_anatomy_mask(f,fullfile(root,'output','nik_stl_preparation_report.json'));
out=fullfile(root,'output','exploratory','fullscan_acceleration_20260909');
save(fullfile(out,'anatomy_mask.mat'),'mask','-v7');
full_scan_write_json(fullfile(out,'anatomy_mask.json'),rmfield(mask,{'face_code','allowed_faces'}));
ref=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
refxyz=[ref.x_mm,ref.y_mm,ref.z_mm];refaxis=refxyz(4,:)-refxyz(1,:);refaxis=refaxis/norm(refaxis);
[c0,n0]=full_scan_project(projection,mean(refxyz,1));u0=refaxis-dot(refaxis,n0)*n0;u0=u0/norm(u0);v0=cross(n0,u0);v0=v0/norm(v0);
poses=[0,-30,0;0,-30,1;0,-30,2;1,-30,0;1,-30,1;1,-30,2; ...
    -80,-120,-20;80,-120,20;0,-80,0;0,0,0;0,-30,-20;0,-30,20];
if coherent
    poses=[zeros(41,1),-30*ones(41,1),(-20:20)';ones(41,1),-30*ones(41,1),(-20:20)'];
end
[sizes,~]=trkg4_load_nik_inhale_data(root,'manifest');cache=[];rows=[];
for repeat=1:repetitions
    for k=1:size(poses,1)
        [c,n]=full_scan_project(projection,c0+poses(k,1)*u0+poses(k,2)*v0);
        base=u0-dot(u0,n)*n;base=base/norm(base);side=cross(n,base);side=side/norm(side);
        ax=cos(poses(k,3)*pi/180)*base+sin(poses(k,3)*pi/180)*side;ax=ax/norm(ax);
        for j=numel(sizes):-1:1
            [p,~,hint]=full_scan_project(projection,c+[-sizes(j)/2;-sizes(j)/4;sizes(j)/4;sizes(j)/2].*ax);
            reference_cfg=cfg;reference_cfg.local_patch_options=struct('initial_radius',6*sqrt(cfg.electrode_area/pi));
            cached_cfg=reference_cfg;cached_cfg.local_patch_options.cache_tile_mm=tile;cached_cfg.local_patch_options.cache_limit=capacity;
            tic;[e1,f1,q1,id1]=evaluate(f,p,reference_cfg,pc,hint,[]);t1=toc;
            tic;[e2,f2,q2,id2,cache]=evaluate(f,p,cached_cfg,pc,hint,cache);t2=toc;
            assert(strcmp(id1,id2),'Builder outcomes differ');
            if isempty(id1)
                assert(isequal(e1,e2)&&isequal(f1,f2),'Electrode nodes/faces differ');
                assert(q1.passed==q2.passed && isequal(q1.qc.bad_surface,q2.qc.bad_surface) && ...
                    isequal(q1.qc.bad_offset,q2.qc.bad_offset)&&isequal(q1.qc.bad_area_low,q2.qc.bad_area_low)&& ...
                    isequal(q1.qc.bad_area_high,q2.qc.bad_area_high),'QC differs');
                passed=q1.passed;hits=q2.subset_cache_hits;misses=q2.subset_cache_misses;
            else,passed=false;hits=0;misses=0;end
            rows(end+1,:)=[repeat,k,sizes(j),t1,t2,passed,hits,misses,any(~mask.allowed_faces(hint))]; %#ok<AGROW>
        end
    end
    fprintf('Cached geometry verification repeat %d: %d assemblies agree exactly.\n',repeat,size(rows,1));
end
T=array2table(rows,'VariableNames',{'repeat','pose','L_mm','reference_seconds','cached_seconds','passed','cache_hits','cache_misses','early_anatomy_excluded'});
writetable(T,fullfile(out,['real_geometry_checks' suffix '.csv']));
warm=rows(:,1)>1;if repetitions==1,warm(:)=true;end
report=struct('passed',true,'assemblies',size(rows,1),'distinct_assemblies',size(poses,1)*numel(sizes), ...
    'exact_nodes_faces_and_QC',true,'reference_total_warm_seconds',sum(rows(warm,4)), ...
    'cached_total_warm_seconds',sum(rows(warm,5)),'cache_hits',sum(rows(:,7)),'cache_misses',sum(rows(:,8)), ...
    'mask_counts',mask.counts,'scope','bounded numerical equivalence; no FEM; not whole-grid throughput');
full_scan_write_json(fullfile(out,['real_geometry_checks' suffix '.json']),report);disp(report);
end
function [el,faces,q,id,cache]=evaluate(f,p,cfg,pc,hint,cache)
el=[];faces=[];q=[];id='';
try,[el,faces,q,cache]=full_scan_patches(f,p,cfg,pc,hint,cache);
catch ex
    if ~any(strcmp(ex.identifier,{'electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall','trkg4:electrodeDiagnostics'})),rethrow(ex);end
    id=ex.identifier;
end
end
