function stl_file = create_smoke_body_stl(stl_file)
%CREATE_SMOKE_BODY_STL Create a small valid ASCII box STL for tests.

if nargin < 1 || isempty(stl_file)
    stl_file = fullfile('output', 'smoke_body_ascii.stl');
end

if ~exist(fileparts(stl_file), 'dir')
    mkdir(fileparts(stl_file));
end

z_levels = [-120, -10, 10, 120];
vertices = zeros(numel(z_levels) * 4, 3);
for k = 1:numel(z_levels)
    idx = (k - 1) * 4 + (1:4);
    vertices(idx, :) = [ ...
        -80, -55, z_levels(k); ...
         80, -55, z_levels(k); ...
         80,  55, z_levels(k); ...
        -80,  55, z_levels(k)];
end

faces = zeros(2 + (numel(z_levels) - 1) * 8 + 2, 3);
faces(1:2, :) = [1, 3, 2; 1, 4, 3];
face_idx = 3;
for k = 1:(numel(z_levels) - 1)
    a = (k - 1) * 4 + (1:4);
    b = k * 4 + (1:4);
    next_faces = [ ...
        a(1), a(2), b(2); a(1), b(2), b(1); ...
        a(2), a(3), b(3); a(2), b(3), b(2); ...
        a(3), a(4), b(4); a(3), b(4), b(3); ...
        a(4), a(1), b(1); a(4), b(1), b(4)];
    faces(face_idx:(face_idx + 7), :) = next_faces;
    face_idx = face_idx + 8;
end
top = (numel(z_levels) - 1) * 4 + (1:4);
faces(face_idx:(face_idx + 1), :) = [top(1), top(2), top(3); top(1), top(3), top(4)];

fv.faces = faces;
fv.vertices = vertices;
stl_write(fv, stl_file, 'txt');
end
