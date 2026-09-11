function [reuse,cache]=full_scan_reuse_geometry(d,ids,cache)
%FULL_SCAN_REUSE_GEOMETRY Import only committed, compatible old geometry.
% Old centre IDs are mapped by (u,v); the old result directory is read-only.
reuse=[];
tag=char(d.opt.ReuseGeometryTag);if isempty(tag),return;end
source=fullfile(d.root,'output',['full_pose_scan_' tag]);
if isempty(cache)
    old=jsondecode(fileread(fullfile(source,'run_manifest.json')));
    keys={'source_tag','source_contract_sha256','rho_ohm_m','geometry_thresholds','centre_projection','contact_model'};
    for k=1:numel(keys)
        a=old.physics.(keys{k});b=d.physics.(keys{k});
        if isnumeric(a)&&isvector(a)&&isnumeric(b)&&isvector(b),a=a(:);b=b(:);end
        assert(isequal(a,b), ...
            'full_scan:reusePhysics','Old geometry does not match %s',keys{k});
    end
    oldcentres=readtable(fullfile(source,'centres.csv'));
    [found,oldids]=ismember(d.centres{:,{'u_mm','v_mm'}},oldcentres{:,{'u_mm','v_mm'}},'rows');
    assert(all(found),'full_scan:reuseIndex','Old grid does not contain requested centres');
    assert(max(abs(d.centres{:,{'x_mm','y_mm','z_mm'}}-oldcentres{oldids,{'x_mm','y_mm','z_mm'}}),[],'all')<1e-8, ...
        'full_scan:reuseIndex','Projected centres differ');
    assert(isequal(old.physics.phi_deg(:),d.grid.phi(:)),'full_scan:reuseIndex','Angle grid differs');
    cache=struct('source',source,'old',old,'oldids',oldids,'numbers',[],'blocks',{{}},'records',{{}});
end
assert(strcmp(cache.source,source),'full_scan:reuseInput','Source cache changed');
na=numel(d.grid.phi);ns=numel(d.sizes);nr=numel(ids)*na;
reuse=struct('centre_ids',reshape(repelem(ids(:),na),[],1),'phi_deg',repmat(d.grid.phi(:),numel(ids),1), ...
    'status_code',zeros(nr,1,'uint8'),'failure_code',{repmat({''},nr,1)}, ...
    'failure_detail',{cell(nr,1)},'patch_rows',{cell(nr,ns,4)},'source_checkpoints',struct([]));
oldids=cache.oldids(ids);blockids=ceil(oldids/cache.old.physics.centre_chunk);
for bi=unique(blockids)'
    where=find(cache.numbers==bi,1);
    if isempty(where)
        path=fullfile(source,'geometry',sprintf('block_%05d.mat',bi));
        if ~isfile(path)||~isfile([path '.json']),continue;end
        check=jsondecode(fileread([path '.json']));
        assert(strcmp(check.fingerprint,cache.old.fingerprint)&&strcmp(check.sha256,trkg4_file_sha256(path)), ...
            'full_scan:reuseChecksum','Old checkpoint identity/hash mismatch');
        b=load(path);
        assert(strcmp(b.fingerprint,cache.old.fingerprint)&&all(ismember(b.status_code,[1,2])), ...
            'full_scan:reuseChecksum','Unexpected old checkpoint');
        assert(isequal(b.valid_rows(:),find(b.status_code==1)),'full_scan:reuseIndex','Invalid old row map');
        if numel(cache.numbers)>=3,cache.numbers(1)=[];cache.blocks(1)=[];cache.records(1)=[];end
        cache.numbers(end+1)=bi;cache.blocks{end+1}=b;
        cache.records{end+1}=struct('source_tag',tag,'block',bi,'sha256',check.sha256,'fingerprint',check.fingerprint);
        where=numel(cache.numbers);
    end
    b=cache.blocks{where};record=cache.records{where};
    if isempty(reuse.source_checkpoints),reuse.source_checkpoints=record;else,reuse.source_checkpoints(end+1)=record;end
    for ci=find(blockids==bi)'
        rr=find(b.centre_ids==oldids(ci));target=(ci-1)*na+(1:na);
        assert(numel(rr)==na&&isequal(b.phi_deg(rr),d.grid.phi(:)),'full_scan:reuseIndex','Missing/duplicate old angle');
        reuse.status_code(target)=b.status_code(rr);reuse.failure_code(target)=b.failure_code(rr);
        if isfield(b,'failure_detail'),reuse.failure_detail(target)=b.failure_detail(rr);end
        [present,vrows]=ismember(rr,b.valid_rows);target=target(present);
        reuse.patch_rows(target,:,:)=b.patch_faces(vrows(present),:,:);
    end
end
end
