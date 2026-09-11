function [entry,cache,hit] = trkg4_local_cached_subset(ctx,centres,radius,tile,limit,cache)
%TRKG4_LOCAL_CACHED_SUBSET Conservative tiled cubes; exact global face order.
% Explicit bounded cache is owned by one serial worker task, never shared.
assert(isfield(ctx,'cache_identity'),'trkg4:localPatchInput','A frozen mesh identity is required for subset caching');
validateattributes(tile,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(limit,{'numeric'},{'scalar','integer','positive'});
if isempty(cache)
    cache=struct('identity',ctx.cache_identity,'keys',{{}},'entries',{{}},'next',1);
end
assert(isequal(cache.identity,ctx.cache_identity),'trkg4:localPatchInput','Subset cache belongs to a different mesh');
anchors=unique(floor(centres/tile),'rows');
key=sprintf('%.17g,',[radius;tile;anchors(:)]);
where=find(strcmp(cache.keys,key),1); hit=~isempty(where);
if hit,entry=cache.entries{where};return;end
mid=anchors*tile+tile/2;
pad=128*eps(max([ctx.coordinate_scale;abs(centres(:));radius;abs(mid(:))]));
half=radius+tile/2+pad; pieces=cell(size(mid,1),1);
for j=1:size(mid,1)
    starts=zeros(1,3); stops=zeros(1,3);
    for d=1:3
        starts(d)=bound(ctx.axis_value{d},mid(j,d)-half,false);
        stops(d)=bound(ctx.axis_value{d},mid(j,d)+half,true)-1;
    end
    [count,d]=min(stops-starts+1);
    if count<=0,pieces{j}=zeros(0,1);continue;end
    candidate=ctx.axis_face{d}(starts(d):stops(d));
    pieces{j}=candidate(all(abs(ctx.canonical.p1(candidate,:)-mid(j,:))<=half,2));
end
ids=unique(vertcat(pieces{:}));
entry=struct('ids',ids,'global_nodes',[],'model',[],'context',[]);
if numel(ids)>=2 && numel(ids)<size(ctx.fmdl.boundary,1)
    [nodes,~,mapped]=unique(ctx.fmdl.boundary(ids,:));
    entry.global_nodes=nodes;
    entry.model=struct('nodes',ctx.fmdl.nodes(nodes,:),'boundary',reshape(mapped,numel(ids),3));
    entry.context=electrode_patch_context(entry.model);
end
if numel(cache.keys)<limit
    cache.keys{end+1}=key;cache.entries{end+1}=entry;
else
    cache.keys{cache.next}=key;cache.entries{cache.next}=entry;
    cache.next=mod(cache.next,limit)+1;
end
end

function i=bound(x,value,strict)
lo=1;hi=numel(x)+1;
while lo<hi
    mid=floor((lo+hi)/2);
    if x(mid)<value || (strict&&x(mid)==value),lo=mid+1;else,hi=mid;end
end
i=lo;
end
