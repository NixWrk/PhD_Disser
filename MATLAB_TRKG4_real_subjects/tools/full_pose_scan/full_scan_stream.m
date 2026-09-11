function report=full_scan_stream(d)
%FULL_SCAN_STREAM Geometry -> CEM -> report in bounded spatial groups.
geo=fullfile(d.out,'geometry');sol=fullfile(d.out,'solved');live=fullfile(d.out,'live','parts');
for folder={geo,sol,live},if ~isfolder(folder{1}),mkdir(folder{1});end;end
mask=full_scan_anatomy_mask(d.f,fullfile(d.root,'output','nik_stl_preparation_report.json'));
save(fullfile(d.out,'anatomy_mask.mat'),'mask','-v7');
full_scan_write_json(fullfile(d.out,'anatomy_mask.json'),rmfield(mask,{'face_code','allowed_faces'}));
pc=full_scan_patch_context(d.f);d.patch_context=pc;
nc=height(d.centres);nb=ceil(nc/d.opt.CentreChunk);cap=min(nb,d.opt.MaxBlocks);
pending={};pending_indices=[];pending_nodes=[];parts={};previous=[];cem_cache=[];reuse_cache=[];
stats=struct('geometry_seconds',0,'CEM_seconds',0,'reused_geometry_rows',0, ...
    'new_CEM_solves',0,'reused_CEM_results',0,'reused_green_columns',0,'new_green_columns',0, ...
    'subset_cache_hits',0,'subset_cache_misses',0,'early_anatomy_exclusions',0);
for bi=1:cap
    ids=(bi-1)*d.opt.CentreChunk+1:min(bi*d.opt.CentreChunk,nc);
    gpath=fullfile(geo,sprintf('block_%05d.mat',bi));spath=fullfile(sol,sprintf('block_%05d.mat',bi));
    b=full_scan_checkpoint('read',gpath,d.fingerprint);
    if isempty(b)
        timer=tic;[reused,reuse_cache]=full_scan_reuse_geometry(d,ids,reuse_cache);
        b=full_scan_geometry_block(d,ids,mask,reused);
        if ~isempty(reused),b.source_checkpoints=reused.source_checkpoints;end
        b.geometry_seconds=toc(timer);full_scan_checkpoint('write',gpath,d.fingerprint,b);
        stats.geometry_seconds=stats.geometry_seconds+b.geometry_seconds;
        if ~isempty(reuse_cache) && ~isfile(fullfile(d.out,'source_centre_map.csv'))
            mapping=d.centres(:,{'centre_index','u_mm','v_mm'});mapping.source_centre_index=reuse_cache.oldids;
            writetable(mapping,fullfile(d.out,'source_centre_map.csv'));
        end
    end
    for field={'reused_geometry_rows','subset_cache_hits','subset_cache_misses','early_anatomy_exclusions'}
        stats.(field{1})=stats.(field{1})+b.geometry_stats.(field{1});
    end
    fprintf('STREAM geometry %d/%d centres: valid %d, QC rejected %d, outside %d; reused %d rows.\n', ...
        ids(end),nc,numel(b.valid_rows),sum(b.status_code==2),sum(b.status_code==3),b.geometry_stats.reused_geometry_rows);
    solved=full_scan_checkpoint('read',spath,d.fingerprint);
    if ~isempty(solved)
        parts{end+1}=write_part(d,bi,solved); %#ok<AGROW>
    else
        proposed=union(pending_nodes,b.candidate_nodes);
        if ~isempty(pending) && (numel(proposed)>d.opt.GreenNodeLimit || numel(pending)>=d.opt.StreamGroupBlocks)
            [new,previous,cem_cache,s]=solve_group(d,pending,pending_indices,previous,cem_cache);
            parts=[parts,new];stats=add_stats(stats,s);pending={};pending_indices=[];pending_nodes=[];
            full_scan_publish(d,parts,true);
        end
        pending{end+1}=b;pending_indices(end+1)=bi;pending_nodes=union(pending_nodes,b.candidate_nodes); %#ok<AGROW>
        if numel(pending)>=d.opt.StreamGroupBlocks || bi==cap || numel(pending_nodes)>=d.opt.GreenNodeLimit
            [new,previous,cem_cache,s]=solve_group(d,pending,pending_indices,previous,cem_cache);
            parts=[parts,new];stats=add_stats(stats,s);pending={};pending_indices=[];pending_nodes=[];
            summary=full_scan_publish(d,parts,bi<cap);
            fprintf('STREAM CEM: %d evaluated poses, %d outside domain, %d remain.\n', ...
                summary.counts.evaluated,summary.counts.outside_search_region,summary.counts.missing);
        end
    end
    full_scan_write_json(fullfile(d.out,'progress.json'),struct('stage','streaming','complete',false, ...
        'blocks_done',bi,'blocks_total',nb,'centres_done',ids(end),'centres_total',nc, ...
        'live_report','live/report/full_scan.html','acceleration',stats));
end
if ~isempty(pending)
    [new,previous,cem_cache,s]=solve_group(d,pending,pending_indices,previous,cem_cache); %#ok<ASGLU>
    parts=[parts,new];stats=add_stats(stats,s);
end
summary=full_scan_publish(d,parts,false);
report=struct('complete',cap==nb,'stage','streaming_complete','counts',summary.counts, ...
    'bestrecord',summary.bestrecord,'acceleration',stats,'out_dir',d.out);
if cap<nb,report.stage='paused_after_block_limit';end
full_scan_write_json(fullfile(d.out,'acceleration_summary.json'),report);
full_scan_write_json(fullfile(d.out,'progress.json'),report);
end

function [parts,previous,cache,stats]=solve_group(d,blocks,indices,previous,cache)
stats=struct('CEM_seconds',0,'new_CEM_solves',0,'reused_CEM_results',0,'reused_green_columns',0,'new_green_columns',0);
parts={};ns=numel(d.sizes);timer=tic;
for ib=1:numel(blocks)
    b=blocks{ib};nv=numel(b.valid_rows);zv=zeros(nv,ns);first=1;
    % Whole-pose partitions enforce the Green memory limit exactly.
    while first<=nv
        nodes=[];last=first-1;
        for k=first:nv
            f=reshape(b.patch_faces(k,:,:),[],1);allfaces=unique(vertcat(f{:}));
            proposed=union(nodes,reshape(d.si.boundary(allfaces,:),[],1));
            if numel(proposed)>d.opt.GreenNodeLimit
                assert(k>first,'full_scan:greenMemoryGuard','One pose exceeds Green limit');break;
            end
            nodes=proposed;last=k;
        end
        [G,gm]=full_scan_green(d.si,d.sigma,nodes,'',struct('scan_fingerprint',d.fingerprint),previous);
        map=zeros(size(d.si.nodes,1),1);map(nodes)=1:numel(nodes);
        [zv(first:last,:),cache,cs]=full_scan_cem_batch(d.si,b.patch_faces(first:last,:,:),map,G,d.cfg.z_contact, ...
            d.fingerprint,d.opt.Workers,cache);
        previous=struct('green',G,'meta',gm);
        stats.new_CEM_solves=stats.new_CEM_solves+cs.new_CEM_solves;
        stats.reused_CEM_results=stats.reused_CEM_results+cs.reused_CEM_results;
        stats.reused_green_columns=stats.reused_green_columns+gm.reused_green_columns;
        stats.new_green_columns=stats.new_green_columns+gm.new_green_columns;
        first=last+1;
    end
    z=nan(numel(b.centre_ids),ns);z(b.valid_rows,:)=zv;
    rmse=sqrt(mean((z-d.measured(:)').^2,2));mae=mean(abs(z-d.measured(:)'),2);
    solved=struct('centre_ids',b.centre_ids,'phi_deg',b.phi_deg,'status_code',b.status_code, ...
        'failure_code',{b.failure_code},'failure_detail',{b.failure_detail},'z',z,'rmse',rmse,'mae',mae, ...
        'valid_count',nv,'fingerprint',d.fingerprint);
    full_scan_checkpoint('write',fullfile(d.out,'solved',sprintf('block_%05d.mat',indices(ib))),d.fingerprint,solved);
    parts{end+1}=write_part(d,indices(ib),solved); %#ok<AGROW>
end
stats.CEM_seconds=toc(timer);
end

function part=write_part(d,bi,b)
name=sprintf('parts/poses_%05d.csv',bi);path=fullfile(d.out,'live',name);
status=repmat("invalid_geometry",numel(b.centre_ids),1);status(b.status_code==1)="evaluated";status(b.status_code==3)="outside_search_region";
t=table(b.centre_ids,b.phi_deg,status,b.rmse,b.mae,string(b.failure_code), ...
    'VariableNames',{'centre_index','phi_deg','status','rmse_ohm','mae_ohm','failure_code'});
for j=1:numel(d.sizes),t.(sprintf('Z_L%03d_ohm',d.sizes(j)))=b.z(:,j);end
tmp=[path '.tmp.csv'];writetable(t,tmp);movefile(tmp,path,'f');
counts=struct('evaluated',sum(b.status_code==1),'invalid_geometry',sum(b.status_code==2), ...
    'outside_search_region',sum(b.status_code==3),'processed',numel(b.centre_ids));
valid=find(b.status_code==1);best=NaN;
if ~isempty(valid)
    [~,ii]=min(b.rmse(valid));ii=valid(ii);best=table2struct(d.centres(b.centre_ids(ii),:));
    best.phi_deg=b.phi_deg(ii);best.rmse_ohm=b.rmse(ii);best.mae_ohm=b.mae(ii);best.z_ohm=b.z(ii,:);
end
part=struct('path',name,'sha256',trkg4_file_sha256(path),'counts',counts,'best',best);
end

function a=add_stats(a,b)
for field=fieldnames(b)',a.(field{1})=a.(field{1})+b.(field{1});end
end
