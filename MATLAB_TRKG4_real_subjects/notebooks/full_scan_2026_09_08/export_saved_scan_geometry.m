function result=export_saved_scan_geometry(stage,out_relative)
% Geometry-only postprocessing of committed run_pipeline CEM results.
% No new FEM solution, search, electrode rebuilding, or parameter estimation.
if nargin<1,stage='centres';end
if nargin<2,out_relative='output/exploratory/scan_localization_20260910';end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));cd(root);
addpath('src','tools','tools/full_pose_scan','vendor_stl_eidors');
out=fullfile(root,out_relative);source=fullfile(root,'output','full_pose_scan_accel_pilot_20260909_01');
run=jsondecode(fileread(fullfile(source,'run_manifest.json')));
% Only read the unchanged geometric operators; unrelated new pipeline profiles
% do not affect geometry postprocessing and are not executed here.
required={'tools/full_pose_scan/full_scan_project.m','tools/full_pose_scan/full_scan_context.m', ...
    'tools/full_pose_scan/full_scan_geometry_block.m','tools/full_pose_scan/full_scan_project_mex.mexw64'};
sources=run.physics.producer_sources;
for i=1:numel(required)
    ix=find(strcmp(strrep({sources.path},'\','/'),required{i}));assert(isscalar(ix),'Missing geometry source');
    assert(strcmp(trkg4_file_sha256(fullfile(root,sources(ix).path)),sources(ix).sha256),'Geometry source changed');
end
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
mesh_path=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
s=load(mesh_path,'fmdl_mesh_units');f=s.fmdl_mesh_units;ctx=full_scan_context(f);
c=readtable(fullfile(source,'centres.csv'));
ref=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
xyz=[ref.x_mm ref.y_mm ref.z_mm];ax=xyz(4,:)-xyz(1,:);ax=ax/norm(ax);
[c0,n0]=full_scan_project(ctx,mean(xyz,1));u0=ax-dot(ax,n0)*n0;u0=u0/norm(u0);v0=cross(n0,u0);v0=v0/norm(v0);
[P,N,F]=full_scan_project(ctx,c0+c.u_mm*u0+c.v_mm*v0);
assert(max(abs(P-c{:,{'x_mm','y_mm','z_mm'}}),[],'all')<1e-8,'Centre projection mismatch');
base=u0-sum(u0.*N,2).*N;base=base./vecnorm(base,2,2);side=cross(N,base,2);side=side./vecnorm(side,2,2);
if strcmp(stage,'centres')
    t=c;t.nx=N(:,1);t.ny=N(:,2);t.nz=N(:,3);t.face_index=F;
    t.ux=base(:,1);t.uy=base(:,2);t.uz=base(:,3);t.vx=side(:,1);t.vy=side(:,2);t.vz=side(:,3);
    writetable(t,fullfile(out,'centres_geometry.csv'));
    inputs=struct();for key={'lungs','bones','body'},inputs.(key{1})=struct('path',cfg.subject.stl.(key{1}),'sha256',trkg4_file_sha256(cfg.subject.stl.(key{1})));end
    result=struct('stage','geometry_postprocessing_only','fem_executed',false,'centres',height(c), ...
        'source_fingerprint',run.fingerprint,'mesh_sha256',trkg4_file_sha256(mesh_path),'surfaces',inputs, ...
        'centre_coordinates_match_saved_scan',true,'source_script_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
    full_scan_write_json(fullfile(out,'geometry_sources.json'),result);
elseif strcmp(stage,'candidates')
    q=readtable(fullfile(out,'candidate_requests.csv'),'TextType','string');sizes=run.physics.rho_ohm_m; %#ok<NASGU>
    summary=jsondecode(fileread(fullfile(source,'live','summary.json')));sizes=summary.experiment.L_mm(:)';
    roles={'I+','V+','V-','I-'};rows={};profiles={};cached_bi=0;
    for k=1:height(q)
        ci=q.centre_index(k);phi=q.phi_deg(k);direction=cosd(phi)*base(ci,:)+sind(phi)*side(ci,:);direction=direction/norm(direction);
        bi=ceil(ci/run.physics.centre_chunk);
        if bi~=cached_bi
            path=fullfile(source,'geometry',sprintf('block_%05d.mat',bi));check=jsondecode(fileread([path '.json']));
            assert(strcmp(check.sha256,trkg4_file_sha256(path))&&strcmp(check.fingerprint,run.fingerprint));
            b=load(path);cached_bi=bi;
        end
        rr=find(b.centre_ids==ci & b.phi_deg==phi);vr=find(b.valid_rows==rr);assert(isscalar(vr),'Candidate not valid');
        for j=1:numel(sizes)
            L=sizes(j);raw=P(ci,:)+[-L/2;-L/4;L/4;L/2]*direction;
            [projected,normals]=full_scan_project(ctx,raw);
            for e=1:4
                tri=f.boundary(b.patch_faces{vr,j,e},:);areas=.5*vecnorm(cross(f.nodes(tri(:,2),:)-f.nodes(tri(:,1),:),f.nodes(tri(:,3),:)-f.nodes(tri(:,1),:),2),2,2);
                centres=(f.nodes(tri(:,1),:)+f.nodes(tri(:,2),:)+f.nodes(tri(:,3),:))/3;centroid=sum(areas.*centres,1)/sum(areas);
                rows(end+1,:)={char(q.candidate_id(k)),ci,q.u_mm(k),q.v_mm(k),phi,L,roles{e}, ...
                    projected(e,1),projected(e,2),projected(e,3),centroid(1),centroid(2),centroid(3), ...
                    normals(e,1),normals(e,2),normals(e,3),sum(areas),size(tri,1)}; %#ok<AGROW>
            end
        end
        for offset=-70:2:70
            [point,normal]=full_scan_project(ctx,P(ci,:)+offset*direction);
            profiles(end+1,:)={char(q.candidate_id(k)),ci,phi,offset,point(1),point(2),point(3),normal(1),normal(2),normal(3),N(ci,1),N(ci,2),N(ci,3)}; %#ok<AGROW>
        end
    end
    names={'candidate_id','centre_index','u_mm','v_mm','phi_deg','L_mm','electrode','x_mm','y_mm','z_mm', ...
        'patch_centroid_x_mm','patch_centroid_y_mm','patch_centroid_z_mm','nx','ny','nz','contact_area_mm2','contact_faces'};
    writetable(cell2table(rows,'VariableNames',names),fullfile(out,'candidate_electrodes.csv'));
    writetable(cell2table(profiles,'VariableNames',{'candidate_id','centre_index','phi_deg','axis_s_mm','x_mm','y_mm','z_mm','nx','ny','nz','centre_nx','centre_ny','centre_nz'}),fullfile(out,'candidate_axis_geometry.csv'));
    result=struct('fem_executed',false,'candidates',height(q),'electrodes',size(rows,1),'axis_points',size(profiles,1), ...
        'coordinates','mm in original CT/FEM frame; projected requested centres and actual finite-contact area centroids', ...
        'measured_installation_coordinates',false,'source_fingerprint',run.fingerprint);
    full_scan_write_json(fullfile(out,'candidate_geometry_verification.json'),result);
else,error('Unknown stage');end
disp(result);
end
