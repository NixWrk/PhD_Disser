function report=benchmark_exact_local_patches(repetitions,local_options)
% Small real-mesh equivalence/benchmark. NO sweep, FEM, or canonical writes.
if nargin<1, repetitions=3; end
if nargin<2, local_options=struct(); end
artifact='local_patches_benchmark';
if isfield(local_options,'initial_radius')
    artifact=sprintf('%s_radius_%0.4f',artifact,local_options.initial_radius);
end
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(here));
addpath(fullfile(root,'tools'),fullfile(root,'tools','full_pose_scan'), ...
    fullfile(root,'src'),fullfile(root,'vendor_stl_eidors'));
s=load(fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat'), ...
    'fmdl_mesh_units'); fmdl=s.fmdl_mesh_units;
cfg=trkg4_config('nik'); cfg.electrode_diagnostics_file='';
cfg.electrode_diagnostics_verbose=false; cfg.fail_on_electrode_diagnostics=true;
t=tic; ctx=full_scan_patch_context(fmdl); prepare_seconds=toc(t);
projection=full_scan_context(fmdl);
cases=struct('name',{},'L',{},'centres',{},'projection_faces',{});
% Saved old rejected-by-QC pose and newly corrected pose. Reproject their
% rounded CSV coordinates through main's EXACT projection implementation.
tags={'disjoint_restartable_manifest90','auditfix_20260908'};
for ti=1:numel(tags)
    file=fullfile(root,'output',['nik_trkg4_inverse_inhale_electrodes_' tags{ti} '.csv']);
    T=readtable(file); sizes=unique(T.L_mm,'stable');
    for L=reshape(sizes,1,[])
        rows=T.L_mm==L;
        raw=[T.requested_x_mm(rows),T.requested_y_mm(rows),T.requested_z_mm(rows)];
        [c,~,f]=full_scan_project(projection,raw);
        cases(end+1)=struct('name',tags{ti},'L',L,'centres',c,'projection_faces',f); %#ok<AGROW>
    end
end
% Sparse broad poses are verification samples, NOT scan pruning or ROI rules.
T=readtable(fullfile(root,'data','nik','electrodes','electrodes_4_right_rib_140mm_xyz_mm.csv'));
ref=[T.x_mm,T.y_mm,T.z_mm]; axis=ref(4,:)-ref(1,:); axis=axis/norm(axis);
[c0,n0]=full_scan_project(projection,mean(ref,1));
u0=axis-dot(axis,n0)*n0; u0=u0/norm(u0); v0=cross(n0,u0); v0=v0/norm(v0);
poses=[-120 -120 -20; -120 120 20; 120 -120 0; 120 120 -20; ...
    0 -120 20; 0 120 -20; -120 0 0; 120 0 20; 0 0 0];
for pi=1:size(poses,1)
    [c,n]=full_scan_project(projection,c0+poses(pi,1)*u0+poses(pi,2)*v0);
    ax=u0-dot(u0,n)*n; ax=ax/norm(ax); side=cross(n,ax); side=side/norm(side);
    ax=cos(deg2rad(poses(pi,3)))*ax+sin(deg2rad(poses(pi,3)))*side; ax=ax/norm(ax);
    for L=[50 90 140]
        [points,~,faces]=full_scan_project(projection,c+[-L/2;-L/4;L/4;L/2].*ax);
        cases(end+1)=struct('name',sprintf('broad_%d',pi),'L',L, ...
            'centres',points,'projection_faces',faces); %#ok<AGROW>
    end
end
% Identical centres MUST reproduce the canonical noSeed rejection.
rejected=cases(1); rejected.name='identical_centres';
rejected.centres(:,:)=repmat(rejected.centres(1,:),4,1);
rejected.projection_faces(:)=rejected.projection_faces(1);
cases(end+1)=rejected;
rows=cell(numel(cases),13);
for ci=1:numel(cases)
    c=cases(ci).centres; pf=cases(ci).projection_faces;
    full_time=zeros(repetitions,1); local_time=full_time;
    qc_full_time=nan(repetitions,1); qc_local_time=qc_full_time;
    info=struct('full_fallback',true,'subset_face_count',NaN,'local_attempts',NaN); qc_id='';
    for ri=1:repetitions
        t=tic; [a,af,aid]=local_build(@() electrode_faces_by_area( ...
            ctx.fmdl,c,cfg.electrode_area,cfg.z_contact,ctx.canonical)); full_time(ri)=toc(t);
        t=tic;
        try
            [b,bf,info]=trkg4_local_exact_patches(ctx,c,cfg.electrode_area,cfg.z_contact,local_options);
            bid='';
        catch ex
            b=[]; bf=[]; bid=ex.identifier;
            if ~local_geometry_id(bid), rethrow(ex); end
        end
        local_time(ri)=toc(t);
        assert(strcmp(aid,bid),'Canonical/local builder rejection mismatch.');
        if ~isempty(aid), continue; end
        assert(isequal(a,b) && isequal(af,bf),'Node/face arrays differ.');
        model=ctx.fmdl; model.electrode=a;
        t=tic;
        try
            trkg4_electrode_diagnostics(model,c,cfg.electrode_order,cfg,c,af); qc_id='';
        catch ex
            qc_id=ex.identifier;
            if ~strcmp(qc_id,'trkg4:electrodeDiagnostics'), rethrow(ex); end
        end
        qc_full_time(ri)=toc(t);
        t=tic; q=trkg4_local_projected_qc(ctx,b,c,pf,cfg,bf); qc_local_time(ri)=toc(t);
        assert(strcmp(qc_id,q.identifier),'Canonical/local QC acceptance differs.');
        assert(q.global_surface_evaluations==0,'Projected-centre certificate unexpectedly failed.');
        % Compare exact area and weighted centroid against canonical QC output.
        if ri==1
            quiet=cfg; quiet.fail_on_electrode_diagnostics=false;
            state=warning('off','trkg4:electrodeDiagnostics'); cleanup=onCleanup(@() warning(state));
            d=trkg4_electrode_diagnostics(model,c,cfg.electrode_order,quiet,c,af);
            clear cleanup
            assert(isequal(d.patch_area_mm2,q.patch_area_mm2));
            assert(isequal(d.patch_centroid_offset_mm,q.patch_centroid_offset_mm));
        end
    end
    rows(ci,:)={cases(ci).name,cases(ci).L,aid,qc_id,median(full_time), ...
        median(local_time),median(qc_full_time,'omitnan'),median(qc_local_time,'omitnan'), ...
        info.full_fallback,info.subset_face_count,repetitions,isempty(aid)&&isempty(qc_id),info.local_attempts};
    fprintf('VERIFY %d/%d %s L%d builder=%s QC=%s local %.4fs full %.4fs\n', ...
        ci,numel(cases),cases(ci).name,cases(ci).L,aid,qc_id,median(local_time),median(full_time));
end
table_out=cell2table(rows,'VariableNames',{'case_name','L_mm','builder_error','qc_error', ...
    'canonical_builder_seconds','local_builder_seconds','canonical_qc_seconds','local_qc_seconds', ...
    'full_fallback','subset_face_count','repetitions','accepted','local_attempts'});
writetable(table_out,fullfile(here,[artifact '.csv']));
info_memory=whos('ctx');
report=struct('case_count',numel(cases),'repetitions',repetitions, ...
    'mesh_faces',size(ctx.canonical.faces,1),'context_seconds',prepare_seconds, ...
    'context_bytes',info_memory.bytes,'all_node_face_and_qc_checks_passed',true, ...
    'median_canonical_builder_seconds',median(table_out.canonical_builder_seconds), ...
    'median_local_builder_seconds',median(table_out.local_builder_seconds), ...
    'median_canonical_qc_seconds',median(table_out.canonical_qc_seconds,'omitnan'), ...
    'median_local_qc_seconds',median(table_out.local_qc_seconds,'omitnan'), ...
    'successful_builder_fallbacks',sum(table_out.full_fallback & cellfun(@isempty,table_out.builder_error)), ...
    'accepted_cases',sum(table_out.accepted),'pose_count',241*241*41, ...
    'local_options',local_options, ...
    'first_attempt_successes',sum(~table_out.full_fallback & table_out.local_attempts==1), ...
    'median_successful_local_attempts',median(table_out.local_attempts,'omitnan'), ...
    'mean_local_builder_seconds',mean(table_out.local_builder_seconds));
fid=fopen(fullfile(here,[artifact '.json']),'w');
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s',jsonencode(report,PrettyPrint=true));
disp(report);
end

function [el,f,id]=local_build(callback)
try
    [el,f]=callback(); id='';
catch ex
    el=[]; f=[]; id=ex.identifier;
    if ~local_geometry_id(id), rethrow(ex); end
end
end

function yes=local_geometry_id(id)
yes=any(strcmp(id,{'electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall'}));
end
