function report=verify_full_scan_geometry_mex(requested_poses)
% Bounded real/synthetic equivalence, not the full sweep; no FEM.
if nargin<1, requested_poses=76; end
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
addpath(fullfile(root,'tools'),fullfile(root,'tools','full_pose_scan'), ...
    fullfile(root,'src'),fullfile(root,'vendor_stl_eidors'));
synthetic=synthetic_checks();
S=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'),'fmdl_mesh_units');
fmdl=S.fmdl_mesh_units; ctx=full_scan_patch_context(fmdl); projection=full_scan_context(fmdl);
cfg=trkg4_config('nik'); cfg.electrode_diagnostics_file='';
cfg.native_geometry_debug=true;
cfg.electrode_diagnostics_verbose=false; cfg.fail_on_electrode_diagnostics=true;
sizes=[50 60 70 80 90 110 120 130 140]; ns=numel(sizes);
T=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
ref=[T.x_mm,T.y_mm,T.z_mm]; axis=ref(4,:)-ref(1,:); axis=axis/norm(axis);
[c0,n0]=full_scan_project(projection,mean(ref,1));
u=axis-dot(axis,n0)*n0; u=u/norm(u); v=cross(n0,u); v=v/norm(v);
[uu,vv]=ndgrid(-120:40:120,-120:40:120);
poses=[uu(:),vv(:),mod((0:numel(uu)-1)'*13,41)-20];
saved_rng=rng; restore=onCleanup(@()rng(saved_rng)); rng(84061);
poses=[poses;240*rand(23,2)-120,41*rand(23,1)-20.5]; clear restore
count=size(poses,1)+4;
P=zeros(count,4,ns,3); F=zeros(count,4,ns,'uint32');
for pi=1:size(poses,1)
    [c,n]=full_scan_project(projection,c0+poses(pi,1)*u+poses(pi,2)*v);
    ax=u-dot(u,n)*n; ax=ax/norm(ax); side=cross(n,ax); side=side/norm(side);
    ax=cosd(poses(pi,3))*ax+sind(poses(pi,3))*side; ax=ax/norm(ax);
    for si=1:ns
        L=sizes(si);
        [points,~,faces]=full_scan_project(projection,c+[-L/2;-L/4;L/4;L/2].*ax);
        P(pi,:,si,:)=reshape(points,[1,4,1,3]); F(pi,:,si)=uint32(faces);
    end
end
tags={'disjoint_restartable_manifest90','auditfix_20260908'};
for ti=1:2
    T=readtable(fullfile(root,'output',['nik_trkg4_inverse_inhale_electrodes_' tags{ti} '.csv']));
    for si=1:ns
        q=T.L_mm==sizes(si); raw=[T.requested_x_mm(q),T.requested_y_mm(q),T.requested_z_mm(q)];
        [points,~,faces]=full_scan_project(projection,raw);
        P(count-4+ti,:,si,:)=reshape(points,[1,4,1,3]); F(count-4+ti,:,si)=uint32(faces);
    end
end
P(count-1,:,:,:)=repmat(P(count-2,1,:,:),[1 4 1 1]); F(count-1,:,:)=repmat(F(count-2,1,:),[1 4 1]);
P(count,:,:,:)=P(count-2,:,:,:); F(count,:,:)=uint32(1); % deliberately poor but valid hint
selection=unique(round(linspace(1,count,min(count,requested_poses))));
P=P(selection,:,:,:); F=F(selection,:,:); n=size(P,1);
% Warm both native schedules. MEX is called exclusively on the client.
[~,~,~]=full_scan_geometry_mex(ctx,P(1:min(n,2),:,:,:),F(1:min(n,2),:,:),cfg,1);
t1=zeros(3,1); t8=t1;
for r=1:3
    t=tic; [s1,p1,f1]=full_scan_geometry_mex(ctx,P,F,cfg,1); t1(r)=toc(t);
    t=tic; [status,patches,fail]=full_scan_geometry_mex(ctx,P,F,cfg,8); t8(r)=toc(t);
    assert(isequal(s1,status)&&isequal(p1,patches)&&isequal(f1,fail),'OpenMP changed outputs.');
end
fprintf('NATIVE_EARLY poses=%d assemblies=%d valid=%d invalid=%d fallback=%d serial=%.4fs omp8=%.4fs\n', ...
    n,n*ns,nnz(status==1),nnz(status==2),nnz(status==3),median(t1),median(t8));
disp([double(unique(fail(:))),arrayfun(@(x)nnz(fail==x),unique(fail(:)))]);
native_status=status; wrapper_time=zeros(n,ns); fallback_time=zeros(n,ns);
canonical_time=zeros(n,ns); geometry_ids={'electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall','trkg4:electrodeDiagnostics'};
for pi=1:n
    for si=1:ns
        c=reshape(P(pi,:,si,:),[4,3]); pf=reshape(F(pi,:,si),[4,1]);
        t=tic; id=''; canonical_faces={}; canonical_el=[];
        try
            [canonical_el,canonical_faces]=electrode_faces_by_area(ctx.fmdl,c,cfg.electrode_area,cfg.z_contact,ctx.canonical);
            model=ctx.fmdl; model.electrode=canonical_el;
            trkg4_electrode_diagnostics(model,c,cfg.electrode_order,cfg,c,canonical_faces);
        catch ex
            id=ex.identifier;
            if ~any(strcmp(id,geometry_ids)), rethrow(ex); end
        end
        canonical_time(pi,si)=toc(t);
        t=tic; wid=''; wf={}; we=[];
        try
            [we,wf]=full_scan_patches(fmdl,c,cfg,ctx,double(pf));
        catch ex
            wid=ex.identifier;
            if ~any(strcmp(wid,geometry_ids)), rethrow(ex); end
        end
        wrapper_time(pi,si)=toc(t);
        assert(strcmp(id,wid),'Canonical/wrapper geometry status mismatch.');
        if isempty(id), assert(isequal(canonical_el,we)&&isequal(canonical_faces,wf)); end
        if status(pi,si)==3
            fallback_time(pi,si)=wrapper_time(pi,si);
            % This is exactly the integration main must perform.
            status(pi,si)=uint8(1+~isempty(wid));
            if isempty(wid)
                for j=1:4, patches{pi,si,j}=uint32(wf{j}); end
            end
        elseif status(pi,si)==1
            assert(isempty(id),'Native accepted a canonical rejection.');
            for j=1:4
                assert(isequal(patches{pi,si,j},uint32(canonical_faces{j})),'Native global induced face IDs differ.');
            end
        elseif status(pi,si)==2
            assert(~isempty(id),'Native rejected a canonical success.');
            assert(strcmp(geometry_ids{fail(pi,si)},id),'Native failure identifier differs.');
        else
            error('Unexpected native status.');
        end
        assert((status(pi,si)==1)==isempty(id),'Resolved fallback outcome mismatch.');
    end
    if mod(pi,4)==0||pi==n
        fprintf('VERIFY_NATIVE %d/%d poses exact; resolved valid=%d\n',pi,n,nnz(status(1:pi,:)==1));
    end
end
report=struct('synthetic_cases',synthetic,'poses',n,'assemblies',n*ns, ...
    'native_valid',nnz(native_status==1),'native_invalid',nnz(native_status==2), ...
    'native_fallback',nnz(native_status==3),'resolved_valid',nnz(status==1), ...
    'native_serial_seconds',median(t1),'native_openmp8_seconds',median(t8), ...
    'canonical_seconds',sum(canonical_time,'all'),'wrapper_seconds',sum(wrapper_time,'all'), ...
    'matlab_fallback_seconds',sum(fallback_time,'all'), ...
    'all_exact',true,'openmp_deterministic',true,'full_sweep_run',false,'fem_run',false, ...
    'failure_values',unique(fail(:))','failure_counts',arrayfun(@(x)nnz(fail==x),unique(fail(:))'));
name=sprintf('native_geometry_verification_%d',n);
fid=fopen(fullfile(here,[name '.json']),'w'); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); disp(report);
end

function count=synthetic_checks()
[x,y]=ndgrid(0:60,0:60); nodes=[x(:),y(:),zeros(numel(x),1)];
n=size(x,1); [i,j]=ndgrid(1:n-1,1:n-1); a=i(:)+(j(:)-1)*n;
faces=[a,a+1,a+n; a+1,a+n+1,a+n];
model=struct('nodes',nodes,'boundary',faces);
ctx=full_scan_patch_context(model);
cfg=struct('electrode_area',0.71,'z_contact',0.1,'length_unit','mm', ...
    'max_electrode_surface_distance_mm',0.1,'max_electrode_patch_centroid_offset_mm',10, ...
    'min_electrode_patch_area_fraction',0.1,'max_electrode_patch_area_ratio',5, ...
    'electrode_diagnostics_verbose',false,'electrode_diagnostics_file','', ...
    'fail_on_electrode_diagnostics',true);
c=[10.31 30.11 0;23.62 30.14 0;36.73 30.17 0;50.94 30.19 0];
pf=zeros(4,1);
for j=1:4, [~,~,pf(j)]=trkg4_local_triangle_distance(ctx,c(j,:),(1:size(faces,1))'); end
[el,ff]=full_scan_patches(model,c,cfg,ctx,pf);
[s,p,f]=native_one(ctx,c,pf,cfg); assert(s==1,'Synthetic nonambiguous plane must run natively.');
for j=1:4, assert(isequal(p{j},uint32(ff{j}))); end
% Permutation retains actual global tie order, including common furthest vertices.
order=[2:2:size(faces,1),1:2:size(faces,1)];
model.boundary=faces(order,:); ctx2=full_scan_patch_context(model);
[~,pf2]=ismember(pf,order); [~,ff2]=full_scan_patches(model,c,cfg,ctx2,pf2);
[s,p,~]=native_one(ctx2,c,pf2,cfg); assert(s==1);
for j=1:4, assert(isequal(p{j},uint32(ff2{j}))); end
% Sign-reflected distance inputs are EXACT ties, not numerical ambiguity.
signed_c=c; signed_c(:,2)=30; signed_pf=pf;
for j=1:4, [~,~,signed_pf(j)]=trkg4_local_triangle_distance(ctx,signed_c(j,:),(1:size(faces,1))'); end
[~,signed_faces]=full_scan_patches(model_with_faces(nodes,faces),signed_c,cfg,ctx,signed_pf);
[s,p,~]=native_one(ctx,signed_c,signed_pf,cfg); assert(s==1);
for j=1:4, assert(isequal(p{j},uint32(signed_faces{j}))); end
signed_c(1,2)=signed_c(1,2)+1e-14;
[s,~,f]=native_one(ctx,signed_c,signed_pf,cfg); assert(s==3&&ismember(f,[10 16 17]));
% Other exact/near bisectors deliberately delegate numerical decisions.
tie=[10 30 0;23 30 0;37 30 0;50 30 0];
[s,~,f]=native_one(ctx,tie,pf,cfg); assert(s==3&&ismember(f,[10 16 17]));
tie(2,1)=tie(2,1)+1e-13; [s,~,f]=native_one(ctx,tie,pf,cfg); assert(s==3&&ismember(f,[10 16 17]));
% Guard a target-area equality at an intermediate stopping comparison.
equal_cfg=cfg; equal_cfg.electrode_area=1;
[s,~,f]=native_one(ctx,c,pf,equal_cfg); assert(s==3&&f==11);
% Canonical weighted-centroid threshold equality must be delegated.
q=trkg4_local_projected_qc(ctx,el,c,pf,cfg,ff);
equal_cfg=cfg; equal_cfg.max_electrode_patch_centroid_offset_mm=max(q.patch_centroid_offset_mm);
[s,~,f]=native_one(ctx,c,pf,equal_cfg); assert(s==3&&f==14);
bad_cfg=cfg; bad_cfg.max_electrode_patch_centroid_offset_mm=0;
[s,~,f]=native_one(ctx,c,pf,bad_cfg); assert(s==2&&f==3);
% Projection hints are checked; no false physical rejection on a bad hint.
[s,~,f]=native_one(ctx,c,ones(4,1),cfg); assert(s==3&&f==13);
% Real noSeed on a complete two-face domain, distinct from uncertified local failure.
tiny=struct('nodes',[0 0 0;1 0 0;0 1 0;2 0 0;3 0 0;2 1 0], ...
    'boundary',[1 2 3;4 5 6]); ct=full_scan_patch_context(tiny);
tc=cfg; tc.electrode_area=0.1;
[s,~,f]=native_one(ct,repmat([0.12 0.23 0],4,1),ones(4,1),tc);
assert(s==2&&f==1);
try
    electrode_faces_by_area(tiny,repmat([0.12 0.23 0],4,1),tc.electrode_area,tc.z_contact,ct.canonical);
    error('Missing canonical noSeed.');
catch ex
    assert(strcmp(ex.identifier,'electrode_faces_by_area:noSeed'));
end
tc.electrode_area=100;
[s,~,f]=native_one(ct,[0.12 .23 0;2.17 .26 0;10.2 .13 0;20.3 .42 0],ones(4,1),tc);
assert(s==2&&f==2);
% Malformed global indices remain technical exceptions.
try
    [~,~,~]=native_one(ctx,c,[0;pf(2:end)],cfg); error('Missing input error.');
catch ex
    assert(strcmp(ex.identifier,'trkg4:fullScanGeometryInput'));
end
count=13; fprintf('NATIVE_SYNTHETIC %d cases passed\n',count);
end
function m=model_with_faces(nodes,faces)
m=struct('nodes',nodes,'boundary',faces);
end
function [s,p,f]=native_one(ctx,c,pf,cfg)
[s,p,f]=full_scan_geometry_mex(ctx,reshape(c,[1,4,1,3]),reshape(uint32(pf),[1,4]),cfg,1);
end
