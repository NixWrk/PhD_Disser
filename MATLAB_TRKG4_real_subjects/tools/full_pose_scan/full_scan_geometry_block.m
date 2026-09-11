function block=full_scan_geometry_block(d,ids,mask,reused)
%FULL_SCAN_GEOMETRY_BLOCK Preserve every centre/angle and explicit exclusions.
% Geometry status: 1 valid, 2 canonical QC failure, 3 outside search domain.
if nargin<4,reused=[];end
na=numel(d.grid.phi);ns=numel(d.sizes);nc=numel(ids);
centre_ids=reshape(repelem(ids(:),na),[],1);phi_deg=repmat(d.grid.phi(:),nc,1);nr=numel(centre_ids);
normals=d.N(centre_ids,:);base=d.u0-sum(d.u0.*normals,2).*normals;base=base./vecnorm(base,2,2);
side=cross(normals,base,2);side=side./vecnorm(side,2,2);
axis=cos(phi_deg*pi/180).*base+sin(phi_deg*pi/180).*side;axis=axis./vecnorm(axis,2,2);
offsets=[-d.sizes(:)'/2;-d.sizes(:)'/4;d.sizes(:)'/4;d.sizes(:)'/2];
raw=zeros(nr*numel(offsets),3);
for j=1:numel(offsets),rr=(j-1)*nr+(1:nr);raw(rr,:)=d.P(centre_ids,:)+offsets(j)*axis;end
[projected,~,face]=full_scan_project(d.projection,raw);
projected=reshape(projected,nr,4,ns,3);face=reshape(face,nr,4,ns);
outside=any(reshape(~mask.allowed_faces(face),nr,[]),2);
status=zeros(nr,1,'uint8');failure=repmat({''},nr,1);details=cell(nr,1);parts=cell(nr,ns,4);
status(outside)=3;failure(outside)={'trkg4:outsideSearchRegion'};
for k=find(outside)'
    [e,j]=find(reshape(~mask.allowed_faces(face(k,:,:)),4,ns),1);
    code=mask.face_code(face(k,e,j));
    details{k}=struct('stage','projected_centre','L_mm',d.sizes(j),'electrode',e,'mask_code',double(code));
end
reuse_count=0;
if ~isempty(reused)
    assert(isequal(reused.centre_ids,centre_ids)&&isequal(reused.phi_deg,phi_deg),'full_scan:reuseIndex','Wrong reused rows');
    take=status==0 & reused.status_code>0;status(take)=reused.status_code(take);
    failure(take)=reused.failure_code(take);details(take)=reused.failure_detail(take);
    for k=find(take & status==1)'
        patches=reshape(reused.patch_rows(k,:,:),ns,4);
        allfaces=vertcat(patches{:});
        if any(~mask.allowed_faces(allfaces))
            status(k)=3;failure{k}='trkg4:outsideSearchRegion';details{k}=struct('stage','reused_contact','mask_codes',unique(double(mask.face_code(allfaces(~mask.allowed_faces(allfaces))))));
        else,parts(k,:,:)=reused.patch_rows(k,:,:);end
    end
    reuse_count=sum(take);
end
todo=find(status==0);task_size=4*na;tasks=ceil(numel(todo)/task_size);output=cell(tasks,1);
limit=d.opt.Workers;if limit==1,limit=0;end
cfg=d.cfg;cfg.return_geometry_status=true;
parfor (ti=1:tasks,limit)
    rows=todo((ti-1)*task_size+1:min(ti*task_size,numel(todo)));
    cache=[];ss=zeros(numel(rows),1,'uint8');ff=cell(numel(rows),1);dd=ff;pp=cell(numel(rows),ns,4);
    hits=0;misses=0;fallbacks=0;built=0;
    for k=1:numel(rows)
        ri=rows(k);ss(k)=1;ff{k}='';
        [~,order]=sort(d.sizes,'descend');
        for j=order(:)'
            try
                [~,faces,stats,cache]=full_scan_patches(d.f,reshape(projected(ri,:,j,:),4,3),cfg, ...
                    d.patch_context,reshape(face(ri,:,j),4,1),cache);
                hits=hits+stats.subset_cache_hits;misses=misses+stats.subset_cache_misses;
                fallbacks=fallbacks+stats.full_fallback;built=built+1;
                if ~stats.passed
                    ss(k)=2;ff{k}='trkg4:electrodeDiagnostics';
                    dd{k}=struct('stage','QC','L_mm',d.sizes(j),'bad_surface',stats.qc.bad_surface, ...
                        'bad_offset',stats.qc.bad_offset,'bad_area_low',stats.qc.bad_area_low,'bad_area_high',stats.qc.bad_area_high);
                    break;
                end
                bad=find(cellfun(@(v)any(~mask.allowed_faces(v)),faces),1);
                if ~isempty(bad)
                    ss(k)=3;ff{k}='trkg4:outsideSearchRegion';
                    dd{k}=struct('stage','contact','L_mm',d.sizes(j),'electrode',bad, ...
                        'mask_codes',unique(double(mask.face_code(faces{bad}(~mask.allowed_faces(faces{bad}))))));break;
                end
                for e=1:4,pp{k,j,e}=uint32(faces{e}(:));end
            catch exception
                expected={'trkg4:electrodeDiagnostics','electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall','trkg4:patchOverlap'};
                if ~any(strcmp(exception.identifier,expected)),rethrow(exception);end
                ss(k)=2;ff{k}=exception.identifier;dd{k}=struct('stage','builder','L_mm',d.sizes(j),'message',exception.message);break;
            end
        end
        if ss(k)~=1,pp(k,:,:)={[]};end
    end
    output{ti}=struct('rows',rows,'status',ss,'failure',{ff},'detail',{dd},'parts',{pp}, ...
        'hits',hits,'misses',misses,'fallbacks',fallbacks,'built',built);
end
stats=struct('subset_cache_hits',0,'subset_cache_misses',0,'full_fallbacks',0,'assemblies_built',0, ...
    'reused_geometry_rows',reuse_count,'early_anatomy_exclusions',sum(outside));
for ti=1:tasks
    o=output{ti};status(o.rows)=o.status;failure(o.rows)=o.failure;details(o.rows)=o.detail;parts(o.rows,:,:)=o.parts;
    stats.subset_cache_hits=stats.subset_cache_hits+o.hits;stats.subset_cache_misses=stats.subset_cache_misses+o.misses;
    stats.full_fallbacks=stats.full_fallbacks+o.fallbacks;stats.assemblies_built=stats.assemblies_built+o.built;
end
valid_rows=find(status==1);patch_faces=parts(valid_rows,:,:);
all_faces=unique(vertcat(patch_faces{:}));candidate_nodes=unique(reshape(d.f.boundary(all_faces,:),[],1));
block=struct('centre_ids',centre_ids,'phi_deg',phi_deg,'status_code',status,'failure_code',{failure}, ...
    'failure_detail',{details},'valid_rows',valid_rows,'patch_faces',{patch_faces},'candidate_nodes',candidate_nodes, ...
    'fingerprint',d.fingerprint,'geometry_stats',stats);
end
