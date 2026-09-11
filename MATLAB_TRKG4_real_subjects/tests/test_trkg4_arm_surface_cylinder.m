function tests = test_trkg4_arm_surface_cylinder
%TEST_TRKG4_ARM_SURFACE_CYLINDER Test the legacy-equivalent cylinder case.
tests = functiontests(localfunctions);
end

function setupOnce(test_case)
test_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(test_dir);
addpath(fullfile(project_root, 'src'));
end

function testCylindricalArmReproducesAxialPlacement(test_case)
[right_nodes, cylinder_faces] = local_cylinder(0, 0, 3);
[left_nodes, left_faces] = local_cylinder(0, 20, 3);
left_faces = left_faces + size(right_nodes, 1);
fmdl = struct('nodes', [right_nodes; left_nodes], ...
    'boundary', [cylinder_faces; left_faces]);
right_guides = [0 0 3; 10 0 3];
left_guides = [0 20 3; 10 20 3];

[centres, ~] = trkg4_arm_surface_centres( ...
    fmdl, right_guides, left_guides, 2, 5);

verifyEqual(test_case, centres, ...
    [7 0 3; 2 0 3; 2 20 3; 7 20 3], 'AbsTol', 1e-12);
end

function [nodes, faces] = local_cylinder(x_start, y_centre, radius)
x_values = (x_start:(x_start + 10)).';
angles = (0:7).' * (2 * pi / 8);
nodes = zeros(numel(x_values) * numel(angles), 3);
for axial_index = 1:numel(x_values)
    rows = (axial_index - 1) * numel(angles) + (1:numel(angles));
    nodes(rows, :) = [ ...
        repmat(x_values(axial_index), numel(angles), 1), ...
        y_centre + radius * sin(angles), ...
        radius * cos(angles)];
end

faces = zeros(2 * (numel(x_values) - 1) * numel(angles), 3);
face_index = 0;
for axial_index = 1:(numel(x_values) - 1)
    for angle_index = 1:numel(angles)
        next_angle = mod(angle_index, numel(angles)) + 1;
        first = (axial_index - 1) * numel(angles) + angle_index;
        second = (axial_index - 1) * numel(angles) + next_angle;
        third = axial_index * numel(angles) + angle_index;
        fourth = axial_index * numel(angles) + next_angle;
        face_index = face_index + 1;
        faces(face_index, :) = [first third second];
        face_index = face_index + 1;
        faces(face_index, :) = [second third fourth];
    end
end
end
