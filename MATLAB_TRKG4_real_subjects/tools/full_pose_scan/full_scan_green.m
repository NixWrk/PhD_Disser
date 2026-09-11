function [green, meta] = full_scan_green(fmdl_si, sigma, candidate_nodes, cache_file, identity, previous)
%FULL_SCAN_GREEN Exact selected inverse of the grounded bulk FEM, in double.
% candidate_nodes order is preserved, including gnd_node (a zero row/column).
% identity is the caller's scalar struct or fingerprint string; geometry/sigma/order/operator source
% hashes and MATLAB runtime are additionally fingerprinted here. An existing
% cache is read-only: incompatible/corrupt caches raise an error, never rebuild.
% At most 15000 candidates per call (1.8 GB). Partition larger unions in the
% caller, retaining every node of each contact. One persistent global bulk
% factor is reused between groups; clear full_scan_green releases it.
% Pass cache_file='' for an in-memory solve. Create each cache from one writer.

if nargin<6,previous=[];end
validateattributes(fmdl_si.nodes, {'double'}, {'2d','real','finite','size',[NaN,3]});
validateattributes(fmdl_si.elems, {'numeric'}, {'2d','real','finite','size',[NaN,4]});
n_nodes = size(fmdl_si.nodes,1);
validateattributes(fmdl_si.gnd_node, {'numeric'}, {'scalar','integer','>=',1,'<=',n_nodes});
validateattributes(sigma, {'numeric'}, {'vector','real','finite','positive'});
if ~ismember(numel(sigma), [1,size(fmdl_si.elems,1)])
    error('full_scan:conductivityInput','Expected scalar or one conductivity per tetrahedron.');
end
validateattributes(candidate_nodes, {'numeric'}, {'vector','nonempty','integer','>=',1,'<=',n_nodes});
candidate_nodes = double(candidate_nodes(:));
if numel(unique(candidate_nodes)) ~= numel(candidate_nodes)
    error('full_scan:candidateInput','Candidate nodes must be unique; their order defines Green indexing.');
end
if (ischar(identity) && isrow(identity) && ~isempty(identity)) || (isstring(identity) && isscalar(identity) && strlength(identity)>0)
    identity=struct('fingerprint',char(identity));
end
if ~isstruct(identity) || ~isscalar(identity)
    error('full_scan:identityInput','identity must be a scalar struct or nonempty fingerprint string.');
end
max_green_nodes=15000;
if numel(candidate_nodes)>max_green_nodes
    error('full_scan:greenMemoryGuard', ...
        'Requested %d Green nodes (%.2f GB); cap is %d (1.8 GB). Split exact contact unions into spatial groups.', ...
        numel(candidate_nodes),8*double(numel(candidate_nodes))^2/1e9,max_green_nodes);
end
cache_file = char(cache_file);
schema = 'full_scan_grounded_green_double_v1';
runtime = struct('matlab',version);
operator_names = {'calc_system_mat','system_mat_1st_order','system_mat_fields', ...
    'fwd_model_parameters','find_electrode_bdy','mk_image','trkg4_green_fingerprint', ...
    'full_scan_green','full_scan_cem_transfer'};
for k=1:numel(operator_names)
    file=which(operator_names{k});
    if isempty(file), error('full_scan:runtimeMissing','Missing operator %s.',operator_names{k}); end
    runtime.([operator_names{k} '_sha256'])=trkg4_file_sha256(file);
end
settings=struct('schema',schema,'identity',identity,'runtime',runtime, ...
    'storage_class','double','block_size',64,'solve','chol');
fingerprint=trkg4_green_fingerprint(fmdl_si,sigma,candidate_nodes,settings);
n=numel(candidate_nodes);
if ~isempty(cache_file) && isfile(cache_file)
    timer=tic;
    info=whos('-file',cache_file);
    gi=find(strcmp({info.name},'green'),1);
    if isempty(gi) || ~ismember('meta',{info.name}) || ...
            ~strcmp(info(gi).class,'double') || ~isequal(info(gi).size,[n,n]) || ...
            info(gi).complex || info(gi).sparse
        error('full_scan:cacheMismatch','Cache has an incompatible payload: %s',cache_file);
    end
    saved=load(cache_file,'meta'); meta=saved.meta;
    required={'schema','fingerprint','candidate_nodes','ground_node','green_sha256'};
    if ~isstruct(meta) || ~isscalar(meta) || ~all(isfield(meta,required)) || ...
            ~strcmp(meta.schema,schema) || ~strcmp(meta.fingerprint,fingerprint) || ...
            ~isequal(meta.candidate_nodes,candidate_nodes) || meta.ground_node~=fmdl_si.gnd_node
        error('full_scan:cacheMismatch','Cache identity mismatch; file left untouched: %s',cache_file);
    end
    saved=load(cache_file,'green'); green=saved.green; clear saved
    if ~strcmp(local_green_hash(green),meta.green_sha256)
        error('full_scan:cacheCorrupt','Green payload checksum mismatch: %s',cache_file);
    end
    meta.cache_reused=true;
    meta.cache_read_seconds=toc(timer);
    return
end

persistent bulk
bulk_fingerprint=trkg4_green_fingerprint(fmdl_si,sigma,zeros(0,1),settings);
reused_factor=~isempty(bulk) && strcmp(bulk.fingerprint,bulk_fingerprint);
if ~reused_factor
    bulk=[]; % release the previous model's factor before allocating another
    model=fmdl_si; model.electrode=struct([]); model.stimulation=struct([]);
    timer=tic;
    system=calc_system_mat(mk_image(model,double(sigma(:))));
    K=system.E; clear system
    if ~isequal(size(K),[n_nodes,n_nodes]) || ~isreal(K) || any(~isfinite(nonzeros(K)))
        error('full_scan:bulkOperator','Expected a finite real first-order nodal bulk matrix.');
    end
    skew=norm(K-K',1)/max(norm(K,1),realmin);
    if skew>1e-12, error('full_scan:bulkSymmetry','Bulk operator relative skew %.3g.',skew); end
    K=(K+K')/2;
    keep=(1:n_nodes)'; keep(model.gnd_node)=[];
    reduced_index=zeros(n_nodes,1); reduced_index(keep)=1:numel(keep);
    K=K(keep,keep);
    assembly_seconds=toc(timer);
    timer=tic;
    % Use the same assembled operator as full EIDORS. On the QC mesh,
    % diagonal rescaling worsened forward error; refinement broke symmetry.
    factor=decomposition(K,'chol');
    factor_seconds=toc(timer);
    bulk=struct('fingerprint',bulk_fingerprint,'K',K,'factor',factor, ...
        'keep',keep,'reduced_index',reduced_index,'K_norm',norm(K,1));
else
    K=bulk.K; factor=bulk.factor; keep=bulk.keep;
    reduced_index=bulk.reduced_index; assembly_seconds=0; factor_seconds=0;
end
K_norm=bulk.K_norm;
candidate_reduced=reduced_index(candidate_nodes);
green=zeros(n,n,'double');
reused_columns=false(n,1);
if ~isempty(previous)
    assert(isstruct(previous)&&all(isfield(previous,{'green','meta'})), ...
        'full_scan:cacheMismatch','Expected previous Green and metadata');
    pm=previous.meta;pg=previous.green;
    assert(strcmp(pm.bulk_fingerprint,bulk_fingerprint)&&pm.ground_node==fmdl_si.gnd_node, ...
        'full_scan:cacheMismatch','Previous Green uses another bulk operator');
    assert(isa(pg,'double')&&isequal(size(pg),[numel(pm.candidate_nodes),numel(pm.candidate_nodes)])&& ...
        strcmp(local_green_hash(pg),pm.green_sha256),'full_scan:cacheCorrupt','Previous Green is corrupt');
    [reused_columns,old_index]=ismember(candidate_nodes,pm.candidate_nodes);
    ii=find(reused_columns);green(ii,ii)=pg(old_index(ii),old_index(ii));
    clear pg
end
active=find(candidate_reduced>0);
solve_active=active(~reused_columns(active));
max_residual=0; max_backward_error=0; block_size=settings.block_size;
timer=tic;
for first=1:block_size:numel(solve_active)
    columns=solve_active(first:min(first+block_size-1,numel(solve_active)));
    rhs=sparse(candidate_reduced(columns),1:numel(columns),1,numel(keep),numel(columns));
    voltage=factor\rhs;
    defect=rhs-K*voltage;
    residual=norm(defect,'fro')/sqrt(numel(columns));
    backward_error=norm(defect,'fro')/(K_norm*norm(voltage,'fro')+sqrt(numel(columns)));
    max_residual=max(max_residual,residual);
    max_backward_error=max(max_backward_error,backward_error);
    if ~isfinite(residual) || ~isfinite(backward_error) || backward_error>1e-12
        error('full_scan:bulkResidual','Bulk normwise backward error %.3g (RHS residual %.3g).',backward_error,residual);
    end
    green(active,columns)=full(voltage(candidate_reduced(active),:));
    % Cross entries with reused columns follow reciprocity of the same SPD bulk.
    green(columns,reused_columns)=green(reused_columns,columns)';
    if first==1 || mod(first-1,10*block_size)==0 || columns(end)==solve_active(end)
        fprintf('Green: %d/%d non-ground columns, %.2fs, residual %.3g.\n', ...
            min(first+block_size-1,numel(solve_active)),numel(solve_active),toc(timer),residual);
    end
end
solve_seconds=toc(timer);
clear voltage defect rhs factor K
% Symmetrise in small tiles: never allocate a second n-by-n Green matrix.
max_skew=0; max_entry=0;
for first=1:256:n
    rows=first:min(first+255,n);
    for last=first:256:n
        columns=last:min(last+255,n);
        a=green(rows,columns); b=green(columns,rows)';
        max_skew=max(max_skew,max(abs(a-b),[],'all'));
        max_entry=max(max_entry,max(abs(a),[],'all'));
        symmetric=(a+b)/2;
        green(rows,columns)=symmetric; green(columns,rows)=symmetric';
    end
end
relative_skew=max_skew/max(max_entry,realmin);
fprintf('Green raw symmetry: relative %.3g, max entry %.3g, max defect %.3g.\n',relative_skew,max_entry,max_skew);
if relative_skew>1e-9
    error('full_scan:greenSymmetry','Selected inverse relative skew %.3g.',relative_skew);
end
meta=struct('schema',schema,'fingerprint',fingerprint,'identity',identity, ...
    'runtime',runtime,'candidate_nodes',candidate_nodes,'ground_node',fmdl_si.gnd_node, ...
    'bulk_fingerprint',bulk_fingerprint,'bulk_factor_reused',reused_factor,'max_green_nodes',max_green_nodes, ...
    'storage_class','double','green_bytes',8*double(n)^2,'assembly_seconds',assembly_seconds, ...
    'factor_seconds',factor_seconds,'solve_seconds',solve_seconds, ...
    'reused_green_columns',sum(reused_columns),'new_green_columns',numel(solve_active), ...
    'max_bulk_residual',max_residual,'max_bulk_backward_error',max_backward_error, ...
    'raw_green_relative_skew',relative_skew, ...
    'green_sha256',local_green_hash(green),'cache_reused',false,'cache_read_seconds',0);
if ~isempty(cache_file)
    folder=fileparts(cache_file); if isempty(folder), folder=pwd; end
    if ~isfolder(folder), error('full_scan:cacheDirectory','Create the cache directory before solving.'); end
    temporary=[tempname(folder) '.mat'];
    cleanup=onCleanup(@() local_remove_temporary(temporary));
    save(temporary,'green','meta','-v7.3','-nocompression');
    if isfile(cache_file), error('full_scan:cacheRace','Another writer created this cache; refusing replacement.'); end
    [ok,message]=movefile(temporary,cache_file);
    if ~ok, error('full_scan:cachePublish','Cannot publish cache: %s',message); end
end
end

function hash=local_green_hash(green)
digest=java.security.MessageDigest.getInstance('SHA-256');
for first=1:64:size(green,2)
    block=green(:,first:min(first+63,size(green,2)));
    if any(~isfinite(block),'all')
        error('full_scan:cacheCorrupt','Green contains non-finite entries.');
    end
    digest.update(typecast(block(:),'uint8'));
end
bytes=typecast(digest.digest(),'uint8');
hash=lower(reshape(dec2hex(bytes,2).',1,[]));
end

function local_remove_temporary(file)
if isfile(file), delete(file); end
end
