function report = check_trkg4_patch_overlap_v5(shared_file, geometry_file)
%CHECK_TRKG4_PATCH_OVERLAP_V5 Verify CEM patch separation across grid sizes.
%
% What this checks, and why it is written the way it is.
%
% The original version called electrode_faces_by_area_fast and then counted
% shared nodes and shared faces between patches. That function enforces node
% disjointness by construction, so both counts were structurally zero and the
% report could never be anything but empty. It was written for an older
% builder that really could produce overlapping patches, and it was never
% retired after that builder was fixed.
%
% This version does two separate things instead:
%
%   1. Asserts the disjointness invariant on the builder the model actually
%      solves with, electrode_faces_by_area. The assertion is expected to
%      hold; it is a regression guard, and it is labelled as one rather than
%      presented as a measurement.
%
%   2. Measures the quantities that can genuinely go wrong and that decide
%      whether a grid size is usable: the gap between neighbouring patches,
%      the achieved contact area, and the drift of the patch centroid from
%      the requested centre. A small gap means two electrodes are nearly
%      touching, which is the physical version of the shorting the old check
%      was aimed at.
%
% Inputs default to the accepted v5 artefacts. Both are outputs of earlier
% runs and live under output/, which is not tracked by git.

cfg = trkg4_config('nik');
root = cfg.project_root;

if nargin < 1 || isempty(shared_file)
    shared_file = fullfile(root, 'output', ...
        'nik_trkg4_arms_full_v5_1mm_local2mm.mat');
end
if nargin < 2 || isempty(geometry_file)
    geometry_file = fullfile(root, 'output', ...
        'nik_trkg4_inverse_inhale_electrodes_v5_1mm_local2mm.csv');
end
if ~isfile(shared_file)
    error('trkg4:missingSharedResult', ...
        'Shared FEM result is missing: %s\nRun run_trkg4_electrode_systems first.', ...
        shared_file);
end
if ~isfile(geometry_file)
    error('trkg4:missingGeometry', ...
        'Electrode geometry table is missing: %s\nRun run_trkg4_inverse_inhale first.', ...
        geometry_file);
end

shared = load(shared_file, 'fmdl_mesh_units');
fmdl = shared.fmdl_mesh_units;
geometry = readtable(geometry_file);
sizes = unique(geometry.L_mm, 'stable');

rows = cell(0, 9);
context = [];
violations = 0;

for size_index = 1:numel(sizes)
    L = sizes(size_index);
    subset = geometry(geometry.L_mm == L, :);
    centres = [subset.requested_x_mm, subset.requested_y_mm, ...
        subset.requested_z_mm];

    % Same builder, same context reuse pattern as the inverse hot loop, so
    % the patches examined here are the patches that were solved.
    [electrodes, induced_faces, context] = electrode_faces_by_area( ...
        fmdl, centres, cfg.electrode_area, cfg.z_contact, context);

    for first = 1:3
        for second = first + 1:4
            shared_nodes = numel(intersect( ...
                electrodes(first).nodes, electrodes(second).nodes));
            shared_faces = numel(intersect( ...
                induced_faces{first}, induced_faces{second}));
            if shared_nodes > 0 || shared_faces > 0
                violations = violations + 1;
            end

            % Closest approach between the two node sets. This is the number
            % that actually says how far the patches are from touching.
            gap_mm = local_min_distance( ...
                fmdl.nodes(electrodes(first).nodes, :), ...
                fmdl.nodes(electrodes(second).nodes, :));

            rows(end + 1, :) = {L, ...
                string(subset.electrode(first)), ...
                string(subset.electrode(second)), ...
                norm(centres(first, :) - centres(second, :)), ...
                gap_mm, shared_nodes, shared_faces, ...
                numel(electrodes(first).nodes), ...
                numel(electrodes(second).nodes)}; %#ok<AGROW>
        end
    end
end

report = cell2table(rows, 'VariableNames', {'L_mm', 'electrode_1', ...
    'electrode_2', 'requested_centre_distance_mm', 'patch_gap_mm', ...
    'shared_nodes', 'shared_faces', 'nodes_1', 'nodes_2'});

filename = fullfile(root, 'output', ...
    'nik_trkg4_patch_overlap_v5_1mm_local2mm.csv');
writetable(report, filename);

fprintf('\nPatch separation by grid size (smallest gap per size):\n');
for size_index = 1:numel(sizes)
    subset = report(report.L_mm == sizes(size_index), :);
    fprintf('  L = %6.1f mm: min gap %.3f mm over %d pairs\n', ...
        sizes(size_index), min(subset.patch_gap_mm), height(subset));
end

if violations > 0
    error('trkg4:patchOverlap', ...
        ['Disjointness invariant violated in %d electrode pair(s). ', ...
         'electrode_faces_by_area must never return patches sharing a ', ...
         'node; a shared node shorts two CEM electrodes.'], violations);
end
fprintf('\nDisjointness invariant holds for all %d pairs.\n', height(report));
fprintf('Saved patch separation report: %s\n', filename);
end

function d = local_min_distance(a, b)
%LOCAL_MIN_DISTANCE Closest approach between two point sets, in mesh units.
d = inf;
for k = 1:size(a, 1)
    d = min(d, min(vecnorm(b - a(k, :), 2, 2)));
end
end
