function [centres, labels] = trkg4_load_electrode_centres(cfg)
%TRKG4_LOAD_ELECTRODE_CENTRES Load 4 electrode centres in STL coordinates.

if ~isempty(cfg.electrode_centres_xyz)
    centres = cfg.electrode_centres_xyz;
    labels = string(cfg.electrode_order(:));
else
    if isempty(cfg.electrode_centres_file) || ~isfile(cfg.electrode_centres_file)
        error(['Missing 4-electrode coordinate file:\n  %s\n', ...
               'Create it from data/nik/electrodes/electrodes_4_xyz_mm.csv.example.'], ...
               cfg.electrode_centres_file);
    end
    T = readtable(cfg.electrode_centres_file, 'VariableNamingRule', 'preserve');
    names = lower(string(T.Properties.VariableNames));
    ix = local_find_column(names, {'x_mm', 'x'});
    iy = local_find_column(names, {'y_mm', 'y'});
    iz = local_find_column(names, {'z_mm', 'z'});
    centres = [T{:, ix}, T{:, iy}, T{:, iz}];
    labels = local_read_labels(T, names, cfg.electrode_order);
end

if ~isequal(size(centres), [4, 3]) || any(~isfinite(centres), 'all')
    error('Electrode centres must be a finite 4 x 3 array in mm.');
end
end

function idx = local_find_column(names, candidates)
idx = [];
for k = 1:numel(candidates)
    idx = find(strcmp(names, lower(candidates{k})), 1);
    if ~isempty(idx)
        return;
    end
end
error('Missing electrode coordinate column: %s', strjoin(candidates, ' or '));
end

function labels = local_read_labels(T, names, expected_order)
idx = find(strcmp(names, 'label'), 1);
if isempty(idx)
    labels = string(expected_order(:));
    return;
end

labels = string(T{:, idx});
expected = string(expected_order(:));
if ~isequal(labels, expected)
    error(['Electrode labels must be exactly:\n  %s\n', ...
           'Current file order is:\n  %s'], ...
           strjoin(expected, ', '), strjoin(labels, ', '));
end
end
