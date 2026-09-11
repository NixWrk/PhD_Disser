function key=full_scan_mesh_identity(fmdl)
%FULL_SCAN_MESH_IDENTITY Compute once on client, before immutable worker use.
md=java.security.MessageDigest.getInstance('SHA-256');
for field={'nodes','boundary'}
    x=double(fmdl.(field{1}));
    md.update(typecast(uint64(size(x)),'uint8'));
    md.update(typecast(x(:),'uint8'));
end
bytes=typecast(md.digest(),'uint8');key=lower(reshape(dec2hex(bytes,2)',1,[]));
end
