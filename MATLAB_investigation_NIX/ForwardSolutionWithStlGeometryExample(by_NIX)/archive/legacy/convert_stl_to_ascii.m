function ascii_file = convert_stl_to_ascii(input_file, output_file)
%CONVERT_STL_TO_ASCII Convert an STL readable by EIDORS into ASCII STL.

if nargin < 2 || isempty(output_file)
    [folder, name] = fileparts(input_file);
    output_file = fullfile(folder, [name '_ascii.stl']);
end

fprintf('Reading STL: %s\n', input_file);
fv = stl_read(input_file);
fprintf('STL triangles: %d, vertices: %d\n', size(fv.boundary, 1), size(fv.nodes, 1));

patch_data.faces = fv.boundary;
patch_data.vertices = fv.nodes;

fprintf('Writing ASCII STL: %s\n', output_file);
stl_write(patch_data, output_file, 'txt');
ascii_file = output_file;
end
