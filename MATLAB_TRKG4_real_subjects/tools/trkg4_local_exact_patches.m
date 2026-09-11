function [el, induced_faces, info, cache] = trkg4_local_exact_patches( ...
    ctx, centres, electrode_area, z_contact, options, cache)
%TRKG4_LOCAL_EXACT_PATCHES Certified restriction of the canonical builder.
% [el, faces, info] = trkg4_local_exact_patches(ctx, centres, area, zc, options)
% ctx = trkg4_local_patch_context(fmdl); output node/face IDs are GLOBAL.
% options.initial_radius: initial cube half-width in mesh units (default
% 6*sqrt(area/pi), at least 1). options.max_local_attempts: default 3; 0
% explicitly calls the full canonical builder. Radius doubles on retries.
%
% Calls UNMODIFIED electrode_faces_by_area on compact local submeshes.
% A successful local result is accepted only when:
% (1) its seed for each electrode is within R, whereas every omitted face
% has max-vertex distance > R to EVERY electrode;
% (2) all global faces incident to every final patch node are present.
% Ownership is face-local and uses ALL centres. (1) fixes the global seed;
% (2) preserves every frontier and induced-area update through growth.
% Ascending global face/node maps preserve ties, vertex order and floating
% point summation order. Thus the whole canonical growth trace is identical.
% Uncertified successes/failures expand, then fall back to the full builder;
% local noSeed/patchTooSmall is never presented as a global rejection.
if nargin < 5, options=struct(); end
if nargin < 6, cache=[]; end
tile=0;cache_limit=64;
if isfield(options,'cache_tile_mm'),tile=options.cache_tile_mm;end
if isfield(options,'cache_limit'),cache_limit=options.cache_limit;end
if size(centres,2)~=3 || isempty(centres) || any(~isfinite(centres),'all') || ...
        ~isscalar(electrode_area) || ~isfinite(electrode_area) || electrode_area<=0
    error('trkg4:localPatchInput','Expected finite centres and positive area.');
end
radius=max(1,6*sqrt(electrode_area/pi)); attempts=3;
if isfield(options,'initial_radius'), radius=options.initial_radius; end
if isfield(options,'max_local_attempts'), attempts=options.max_local_attempts; end
if ~isscalar(radius) || ~isfinite(radius) || radius<=0 || ...
        ~isscalar(attempts) || ~isfinite(attempts) || attempts<0 || attempts~=fix(attempts)
    error('trkg4:localPatchInput','Invalid radius or max_local_attempts.');
end
p=ctx.canonical; nf=size(p.faces,1);
info=struct('full_fallback',true,'local_attempts',0, ...
    'subset_face_count',nf,'total_face_count',nf,'radius',radius, ...
    'subset_cache_hits',0,'subset_cache_misses',0);
for attempt=1:attempts
    info.local_attempts=attempt;
    % Inflate only the broad-phase box, never any physical threshold.
    pad=128*eps(max([ctx.coordinate_scale; abs(centres(:)); radius]));
    if tile>0
        [entry,cache,hit]=trkg4_local_cached_subset(ctx,centres,radius,tile,cache_limit,cache);
        ids=entry.ids;
        info.subset_cache_hits=info.subset_cache_hits+hit;
        info.subset_cache_misses=info.subset_cache_misses+~hit;
    else
        ids=local_cube_faces(ctx,centres,radius+pad);
    end
    if numel(ids)==nf, break; end
    % The vendor routine indexes a logical vector by the face matrix. For a
    % one-row boundary MATLAB applies vector-shape rules differently; do not
    % introduce that special case by restricting an otherwise larger mesh.
    if numel(ids)>=2
        if tile>0
            global_nodes=entry.global_nodes;local_model=entry.model;local_context=entry.context;
        else
            [global_nodes,~,mapped]=unique(p.faces(ids,:));
            local_model=struct('nodes',ctx.fmdl.nodes(global_nodes,:), ...
                'boundary',reshape(mapped,numel(ids),3));
            local_context=electrode_patch_context(local_model);
        end
        try
            [trial,trial_faces]=electrode_faces_by_area(local_model, ...
                centres,electrode_area,z_contact,local_context);
            succeeded=true;
        catch exception
            if ~any(strcmp(exception.identifier, ...
                    {'electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall'}))
                rethrow(exception);
            end
            succeeded=false;
        end
        if succeeded && local_certified(ctx,ids,global_nodes,local_context,trial,centres,radius)
            el=trial; induced_faces=trial_faces;
            for j=1:numel(el)
                el(j).nodes=global_nodes(trial(j).nodes);
                induced_faces{j}=ids(trial_faces{j});
            end
            info.full_fallback=false;
            info.subset_face_count=numel(ids); info.radius=radius;
            return;
        end
    end
    radius=2*radius;
end
% Full canonical path preserves exact IDs AND messages of real failures.
[el,induced_faces]=electrode_faces_by_area(ctx.fmdl,centres, ...
    electrode_area,z_contact,p);
info.radius=radius;
end

function ids=local_cube_faces(ctx,centres,radius)
pieces=cell(size(centres,1),1); n=numel(ctx.axis_value{1});
for j=1:size(centres,1)
    first=zeros(1,3); last=zeros(1,3);
    for d=1:3
        first(d)=local_bound(ctx.axis_value{d},centres(j,d)-radius,false);
        last(d)=local_bound(ctx.axis_value{d},centres(j,d)+radius,true)-1;
    end
    [count,d]=min(last-first+1);
    if count<=0, pieces{j}=zeros(0,1); continue; end
    candidate=ctx.axis_face{d}(first(d):min(n,last(d)));
    delta=abs(ctx.canonical.p1(candidate,:)-centres(j,:));
    pieces{j}=candidate(all(delta<=radius,2));
end
ids=unique(vertcat(pieces{:}));
end

function index=local_bound(values,value,strict)
lo=1; hi=numel(values)+1;
while lo<hi
    mid=floor((lo+hi)/2);
    if values(mid)<value || (strict && values(mid)==value)
        lo=mid+1;
    else
        hi=mid;
    end
end
index=lo;
end

function ok=local_certified(ctx,ids,global_nodes,lc,trial,centres,radius)
ok=false; nf=numel(ids); ne=numel(trial);
dist=zeros(nf,ne);
for j=1:ne
    dist(:,j)=max([vecnorm(lc.p1-centres(j,:),2,2), ...
        vecnorm(lc.p2-centres(j,:),2,2), ...
        vecnorm(lc.p3-centres(j,:),2,2)],[],2);
end
[~,owner]=min(dist,[],2); used=false(lc.n_nodes,1);
for j=1:ne
    used_faces=reshape(used(lc.faces(:)),size(lc.faces));
    legal=owner==j & ~any(used_faces,2);
    if ~any(legal) || min(dist(legal,j))>=radius, return; end
    used(trial(j).nodes)=true;
end
nodes=global_nodes(used); p=ctx.canonical;
incident=cell(numel(nodes),1);
for k=1:numel(nodes)
    a=p.node_face_start(nodes(k)); b=p.node_face_start(nodes(k)+1)-1;
    incident{k}=p.node_face_list(a:b);
end
ok=all(ismember(vertcat(incident{:}),ids));
end
