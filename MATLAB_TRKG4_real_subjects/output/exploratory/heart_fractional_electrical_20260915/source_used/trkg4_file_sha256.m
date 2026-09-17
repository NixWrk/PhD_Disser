function hash = trkg4_file_sha256(filename)
%TRKG4_FILE_SHA256 Stream a file into SHA-256 without loading it into RAM.
fid = fopen(filename, 'rb');
if fid < 0
    error('trkg4:hashFileMissing', 'Cannot read file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
digest = java.security.MessageDigest.getInstance('SHA-256');
while true
    chunk = fread(fid, 1024 * 1024, '*uint8');
    if isempty(chunk), break; end
    digest.update(chunk);
end
bytes = typecast(digest.digest(), 'uint8');
hash = lower(reshape(dec2hex(bytes, 2).', 1, []));
end
