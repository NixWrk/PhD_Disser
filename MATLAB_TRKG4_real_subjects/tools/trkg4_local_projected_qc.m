function q = trkg4_local_projected_qc(ctx,el,centres,projection_faces,cfg,induced_faces)
%TRKG4_LOCAL_PROJECTED_QC Exact canonical acceptance with local certificates.
% q=trkg4_local_projected_qc(ctx,el,centres,projection_faces,cfg,induced_faces)
% centres serve as BOTH requested and effective points (inverse convention).
% projection_faces contains one GLOBAL triangle id per centre from the main
% projection routine. It is a hint that is checked, not blindly trusted.
% A distance to that triangle <= the canonical surface threshold certifies
% global surface acceptance: min(all faces) <= min(specified face).
% Otherwise a full canonical-arithmetic surface search determines the test.
% q.surface_distance_upper_bound_mm is NOT claimed to be the global minimum
% except when q.surface_distance_is_exact is true (zero or global fallback).
% The predicate is exact even if the supplied projection face is a poor hint.
%
% Area/centroid use ascending GLOBAL induced face IDs and canonical arithmetic.
% All globally incident faces of el(k).nodes are enumerated by the context's
% CSR adjacency. Every induced face is in this set; therefore the induced
% consistency check omits no possible face. No full-boundary ismember/table.
%
% Physical failures: q.passed=false, q.identifier='trkg4:electrodeDiagnostics',
% with per-electrode bad_* flags. Technical errors THROW
% trkg4:electrodeDiagnosticsInput. cfg thresholds are never altered.
% This returns a quiet struct; it does not write files or emit warnings and
% does not use cfg.fail_on_electrode_diagnostics to suppress failure status.
n=size(centres,1); p=ctx.canonical; nf=size(p.faces,1);
if size(centres,2)~=3 || any(~isfinite(centres),'all') || ...
        numel(el)~=n || numel(projection_faces)~=n || ...
        ~iscell(induced_faces) || numel(induced_faces)~=n
    error('trkg4:electrodeDiagnosticsInput','Inconsistent centres/electrodes/projection faces.');
end
fields={'max_electrode_surface_distance_mm','max_electrode_patch_centroid_offset_mm', ...
    'min_electrode_patch_area_fraction','max_electrode_patch_area_ratio','electrode_area'};
if ~isstruct(cfg) || ~all(isfield(cfg,fields))
    error('trkg4:electrodeDiagnosticsInput','Missing canonical QC thresholds.');
end
if isfield(cfg,'length_unit') && ~isempty(cfg.length_unit) && ...
        ~ismember(lower(char(cfg.length_unit)),{'mm','millimeter','millimetre'})
    error('trkg4:electrodeDiagnosticsInput','Expected mesh coordinates in millimetres.');
end
for k=1:numel(fields)
    value=cfg.(fields{k});
    if ~isnumeric(value) || ~isscalar(value) || isnan(value) || value<0
        error('trkg4:electrodeDiagnosticsInput','Invalid QC threshold %s.',fields{k});
    end
end
local_indices(projection_faces,nf);
q=struct('passed',false,'identifier','', ...
    'surface_distance_upper_bound_mm',zeros(n,1), ...
    'surface_distance_is_exact',false(n,1), ...
    'global_surface_evaluations',0, 'patch_area_mm2',zeros(n,1), ...
    'patch_face_count',zeros(n,1),'patch_centroid_xyz_mm',nan(n,3), ...
    'patch_centroid_offset_mm',nan(n,1));
for k=1:n
    distance=trkg4_local_triangle_distance(ctx,centres(k,:),projection_faces(k));
    exact=(distance==0);
    if ~isfinite(distance) || distance>cfg.max_electrode_surface_distance_mm
        distance=trkg4_local_triangle_distance(ctx,centres(k,:),(1:nf)');
        exact=true; q.global_surface_evaluations=q.global_surface_evaluations+1;
    end
    q.surface_distance_upper_bound_mm(k)=distance;
    q.surface_distance_is_exact(k)=exact;
    nodes=double(el(k).nodes(:)); local_indices(nodes,p.n_nodes); nodes=unique(nodes);
    incident=cell(numel(nodes),1);
    for ni=1:numel(nodes)
        incident{ni}=p.node_face_list(p.node_face_start(nodes(ni)): ...
            p.node_face_start(nodes(ni)+1)-1);
    end
    all_incident=unique(vertcat(incident{:}));
    induced=all_incident(all(ismember(p.faces(all_incident,:),nodes),2));
    supplied=double(induced_faces{k}(:)); local_indices(supplied,nf);
    supplied=unique(supplied);
    if ~isequal(supplied(:),induced(:))
        error('trkg4:electrodeDiagnosticsInput','Induced face mismatch for electrode %d.',k);
    end
    q.patch_face_count(k)=numel(induced);
    if isempty(induced), continue; end
    area=p.face_area(induced);
    q.patch_area_mm2(k)=sum(area);
    q.patch_centroid_xyz_mm(k,:)=(area.'*ctx.face_centroid(induced,:))/sum(area);
    q.patch_centroid_offset_mm(k)=norm(q.patch_centroid_xyz_mm(k,:)-centres(k,:));
end
q.bad_surface=~isfinite(q.surface_distance_upper_bound_mm) | ...
    q.surface_distance_upper_bound_mm>cfg.max_electrode_surface_distance_mm;
q.bad_offset=~isfinite(q.patch_centroid_offset_mm) | ...
    q.patch_centroid_offset_mm>cfg.max_electrode_patch_centroid_offset_mm;
q.bad_area_low=~isfinite(q.patch_area_mm2) | ...
    q.patch_area_mm2<cfg.min_electrode_patch_area_fraction*cfg.electrode_area;
q.bad_area_high=isfinite(q.patch_area_mm2) & ...
    q.patch_area_mm2>cfg.max_electrode_patch_area_ratio*cfg.electrode_area;
q.passed=~any(q.bad_surface | q.bad_offset | q.bad_area_low | q.bad_area_high);
if ~q.passed, q.identifier='trkg4:electrodeDiagnostics'; end
end

function local_indices(ids,n)
if any(~isfinite(ids(:)) | ids(:)~=fix(ids(:)) | ids(:)<1 | ids(:)>n)
    error('trkg4:electrodeDiagnosticsInput','Invalid global node or triangle indices.');
end
end
