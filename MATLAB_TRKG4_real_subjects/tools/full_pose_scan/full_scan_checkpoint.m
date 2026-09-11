function value=full_scan_checkpoint(action,path,fingerprint,value)
%FULL_SCAN_CHECKPOINT Atomic payload followed by SHA-bound commit sidecar.
if strcmp(action,'read')
    value=[];if ~isfile(path)||~isfile([path '.json']),return;end
    check=jsondecode(fileread([path '.json']));
    assert(strcmp(check.fingerprint,fingerprint)&&strcmp(check.sha256,trkg4_file_sha256(path)), ...
        'full_scan:checkpoint','Checkpoint identity/hash mismatch');
    value=load(path);
    assert(strcmp(value.fingerprint,fingerprint),'full_scan:checkpoint','Payload identity mismatch');
elseif strcmp(action,'write')
    temp=[path '.tmp.mat'];save(temp,'-struct','value','-v7');movefile(temp,path,'f');
    full_scan_write_json([path '.json'],struct('fingerprint',fingerprint,'sha256',trkg4_file_sha256(path)));
else,error('full_scan:checkpoint','Unknown checkpoint action');end
end
