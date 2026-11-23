function electrical_shift_20261119(metadatafile)
% Bounded wrapper around the unchanged canonical CEM driver; no new meshes.
meta=jsondecode(fileread(metadatafile)); p=jsondecode(fileread(meta.all_plan));
folder=fileparts(metadatafile); addpath(fullfile(p.root,'MATLAB_TRKG4_real_subjects','src'));
assert(strcmp(trkg4_file_sha256(meta.original_extended_plan),meta.original_extended_plan_sha256));
assert(strcmp(trkg4_file_sha256(p.shared_file),p.shared_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_back_comparison')),meta.driver_sha256));
d=load(p.shared_file,'identity');
assert(strcmp(d.identity.mesh_sha256,p.mesh_sha256));
assert(strcmp(d.identity.plan_sha256,meta.original_extended_plan_sha256)); clear d;
try
    trkg4_back_comparison(meta.all_plan,'contacts');
catch exception
    qcfile=fullfile(p.work,'contact_qc.local.json');
    if ~isfile(qcfile), rethrow(exception); end
    q=jsondecode(fileread(qcfile));
    assert(strcmp(q.plan_sha256,trkg4_file_sha256(meta.all_plan)));
    if q.passed, rethrow(exception); end
    fprintf('Expected contact-QC rejection retained; no refinement.\n');
end
q=jsondecode(fileread(fullfile(p.work,'contact_qc.local.json')));
names={'scorpion_lateral','back_right','back_left'};
nominals=struct([]); eligible=false(numel(p.contacts),1);
for j=1:numel(names)
    indices=find(strcmp({p.contacts.side},names{j}));
    n=indices(strcmp({p.contacts(indices).direction},'none'));
    assert(numel(n)==1);
    entry=struct('placement',names{j},'qc_passed',q.rows(n).passed,'operator_reproduced',false);
    nominals=[nominals;entry]; %#ok<AGROW>
    if q.rows(n).passed, eligible(indices)=[q.rows(indices).passed]; end
end
candidate=p; candidate.work=fullfile(folder,'accepted');candidate.contacts=p.contacts(eligible);
candidatefile=fullfile(candidate.work,'plan.local.json');
if any(eligible)
    writejson(candidatefile,candidate);
    trkg4_back_comparison(candidatefile,'contacts');
    current=load(fullfile(candidate.work,'contacts.mat'),'C','plan_sha256');
    assert(strcmp(current.plan_sha256,trkg4_file_sha256(candidatefile)));
    assert(strcmp(trkg4_file_sha256(meta.original_contact_cache),meta.original_contact_cache_sha256));
    previous=load(meta.original_contact_cache,'C','plan_sha256');
    assert(strcmp(previous.plan_sha256,meta.original_extended_plan_sha256));
    for j=1:numel(names)
        if ~nominals(j).qc_passed, continue; end
        n=find(strcmp({candidate.contacts.side},names{j}) & strcmp({candidate.contacts.direction},'none'));
        if j==1
            % No historical lateral baseline exists on this extended mesh.
            % Its new canonical, QC-passing nominal is the same-mesh reference.
            nominals(j).operator_reproduced=true;
        else
            side=erase(names{j},'back_');
            k=find(strcmp({meta.original_plan_contacts.side},side) & [meta.original_plan_contacts.size_mm]==140);
            assert(numel(k)==1);
            nominals(j).operator_reproduced=nnz(current.C{n}-previous.C{k})==0;
            fprintf('NOMINAL %s exact historical contact operator %d\n',names{j},nominals(j).operator_reproduced);
        end
    end
    clear current previous;
    % A changed zero-shift contact operator is not silently accepted as a shift.
    keep=ismember({candidate.contacts.side},{nominals([nominals.operator_reproduced]).placement});
    if ~all(keep)
        candidate.work=fullfile(folder,'accepted_nominal_checked');
        if ~isfolder(candidate.work),mkdir(candidate.work);end
        candidate.contacts=candidate.contacts(keep);
        candidatefile=fullfile(candidate.work,'plan.local.json');
        writejson(candidatefile,candidate);
    end
end
selection=struct('nominals',nominals,'accepted_ids',{{candidate.contacts.id}},'result_work',candidate.work);
writejson(fullfile(folder,'selection.local.json'),selection);
if ~isempty(candidate.contacts)
    trkg4_back_comparison(candidatefile,'all');
end
writejson(fullfile(folder,'complete.local.json'),struct('completed',true,'accepted_cases',numel(candidate.contacts)));
end

function writejson(path,value)
f=fopen(path,'w','n','UTF-8');assert(f>=0);cleanup=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value));
end
