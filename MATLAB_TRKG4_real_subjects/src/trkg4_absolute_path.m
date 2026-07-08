function filename = trkg4_absolute_path(filename)
%TRKG4_ABSOLUTE_PATH Return absolute path when the file exists.

if isfile(filename)
    info = dir(filename);
    filename = fullfile(info.folder, info.name);
end
end
