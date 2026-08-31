%% Forward EIT calculation from STL geometry
% This script builds an EIDORS forward model from an STL file and computes
% simulated EIT measurements. It does not need COMSOL reference data.

clearvars; close all; clc;

%% Inputs
stl_file = fullfile('stl', 'model.stl');
if ~isfile(stl_file)
    stl_files = dir(fullfile('stl', '*.stl'));
    if isempty(stl_files)
        error('No STL file found. Put model.stl or any *.stl file into the stl folder.');
    end
    stl_file = fullfile(stl_files(1).folder, stl_files(1).name);
end

n_electrodes = 16;
mesh_fineness = 'moderate'; % very_coarse, coarse, moderate, fine, very_fine

rho_background = 2.5;       % Ohm*m, edit for your material
sigma_background = 1 / rho_background;

% Area is in the same squared units as the STL coordinates.
% The inherited COMSOL example used roughly a 10 mm diameter electrode.
electrode_area = pi * 10^2 / 4 * 0.9;

% Contact impedance inherited from the COMSOL comparison example.
z_contact = 2.5e-7 * 0.05 / (pi * 0.01^2 / 4);

% Replace [] with real Nx3 electrode centres when available.
% Example:
% electrode_centres_xyz = [x1 y1 z1; x2 y2 z2; ...];
electrode_centres_xyz = [];

%% Build tetrahedral model from STL
fprintf('Meshing STL with Netgen: %s\n', stl_file);
fmdl = ng_stl2tet(stl_file, mesh_fineness);
fmdl.name = 'STL forward model';

TR = triangulation(fmdl.elems, fmdl.nodes);
fmdl.boundary = freeBoundary(TR);

if isempty(electrode_centres_xyz)
    electrode_centres_xyz = default_ring_electrodes(fmdl.nodes, n_electrodes);
    fprintf('Using default ring electrode centres from model bounding box.\n');
end

if size(electrode_centres_xyz, 1) ~= n_electrodes || size(electrode_centres_xyz, 2) ~= 3
    error('electrode_centres_xyz must be %d x 3.', n_electrodes);
end

fmdl.electrode = electrode_faces_by_area(fmdl, electrode_centres_xyz, electrode_area, z_contact);
[fmdl.stimulation, fmdl.meas_select] = mk_stim_patterns( ...
    n_electrodes, 1, [0, 1], [0, 1], {'rotate_meas'}, 1);

%% Solve forward problem
img = mk_image(fmdl, sigma_background);

tic;
v = fwd_solve(img);
fwd_duration = toc;

fprintf('Forward solution required %.3f s\n', fwd_duration);
fprintf('Measurements: %d\n', numel(v.meas));

%% Visualize
figure('Name', 'STL EIDORS model');
show_fem(fmdl);
title('STL forward model');

figure('Name', 'Forward measurements');
plot(v.meas, '-o', 'LineWidth', 1.1);
grid on; grid minor;
xlabel('Measurement number');
ylabel('Voltage / impedance response');
title('Forward solution');

%% Save result
if ~exist('output', 'dir')
    mkdir('output');
end
save(fullfile('output', 'forward_result.mat'), ...
    'fmdl', 'img', 'v', 'rho_background', 'sigma_background', ...
    'electrode_centres_xyz', 'electrode_area', 'z_contact', ...
    'mesh_fineness', 'fwd_duration', 'stl_file');
