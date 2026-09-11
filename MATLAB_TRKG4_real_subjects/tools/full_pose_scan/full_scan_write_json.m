function full_scan_write_json(path,value)
temp=[path '.tmp'];fid=fopen(temp,'w');assert(fid>0,'full_scan:output','Cannot create JSON');
cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(value,'PrettyPrint',true));clear cleanup;
movefile(temp,path,'f');
end
