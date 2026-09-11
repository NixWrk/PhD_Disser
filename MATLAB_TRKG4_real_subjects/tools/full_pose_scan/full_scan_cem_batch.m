function [z,cache,stats] = full_scan_cem_batch(fmdl,patches,map,G,zc,identity,workers,cache)
%FULL_SCAN_CEM_BATCH Exact deduplication; every requested row is restored.
% Cache belongs to one frozen mesh/material/contact/operator identity.
% Its maps are accessed on the client only; worker data are immutable cells.
if nargin<8 || isempty(cache)
    cache=struct('identity',identity,'z_contact',zc,'limit',50000, ...
        'voltages',containers.Map('KeyType','char','ValueType','double'), ...
        'contacts',containers.Map('KeyType','char','ValueType','any'));
end
assert(cache.z_contact==zc,'full_scan:cacheMismatch','Contact impedance changed');
assert(isequal(cache.identity,identity),'full_scan:cacheMismatch','CEM cache identity changed');
assert(iscell(patches)&&size(patches,3)==4,'full_scan:patchInput','Expected poses by sizes by four electrodes');
np=size(patches,1); ns=size(patches,2); requests=reshape(patches,[],4); nr=size(requests,1);
keys=cell(nr,1); contact_keys=cell(nr,4);
for i=1:nr
    for e=1:4
        faces=requests{i,e};
        validateattributes(faces,{'numeric'},{'vector','nonempty','integer','positive'});
        contact_keys{i,e}=sprintf('%u,',faces(:));
    end
    keys{i}=strjoin(contact_keys(i,:),'|');
end
[unique_keys,first,group]=unique(keys,'stable'); nu=numel(first);
values=zeros(nu,1); missing=false(nu,1);
for k=1:nu
    if isKey(cache.voltages,unique_keys{k}), values(k)=cache.voltages(unique_keys{k});
    else, missing(k)=true; end
end
to_solve=find(missing); prepared=cell(numel(to_solve),4); faces_to_solve=cell(numel(to_solve),4);
new_contacts=0; reused_contacts=0;
for k=1:numel(to_solve)
    i=first(to_solve(k)); faces_to_solve(k,:)=requests(i,:);
    for e=1:4
        key=contact_keys{i,e};
        if isKey(cache.contacts,key)
            prepared{k,e}=cache.contacts(key); reused_contacts=reused_contacts+1;
        else
            prepared{k,e}=full_scan_contact(fmdl,requests{i,e},zc); new_contacts=new_contacts+1;
            if cache.contacts.Count>=cache.limit, remove(cache.contacts,cache.contacts.keys); end
            cache.contacts(key)=prepared{k,e};
        end
    end
end
fresh=zeros(numel(to_solve),1); limit=workers;if workers==1,limit=0;end
timer=tic;
parfor (k=1:numel(to_solve),limit)
    fresh(k)=full_scan_cem_transfer(fmdl,faces_to_solve(k,:),map,G,zc,prepared(k,:));
end
seconds=toc(timer); values(to_solve)=fresh;
for k=1:numel(to_solve)
    if cache.voltages.Count>=cache.limit, remove(cache.voltages,cache.voltages.keys); end
    cache.voltages(unique_keys{to_solve(k)})=fresh(k);
end
z=reshape(values(group),np,ns);
stats=struct('requests',nr,'unique_four_contact_sets',nu,'new_CEM_solves',numel(to_solve), ...
    'reused_CEM_results',nr-numel(to_solve),'new_contact_factors',new_contacts, ...
    'reused_contact_factors',reused_contacts,'local_solve_seconds',seconds);
end
