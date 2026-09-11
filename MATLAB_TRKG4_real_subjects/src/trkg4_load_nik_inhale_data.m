function [sizes_mm, z_inhale_ohm, source_files] = ...
    trkg4_load_nik_inhale_data(root, selection_mode)
%TRKG4_LOAD_NIK_INHALE_DATA Load the accepted independent inhale levels.
% The subject manifest is the single authority for duplicate exclusion.
% selection_mode="manifest" uses the accepted subject-specific decision.
% selection_mode="legacy_100" is a diagnostic ablation that reproduces the
% historical 100-mm-in/90-mm-out attribution without editing the manifest.

if nargin < 2 || strlength(string(selection_mode)) == 0
    selection_mode = "manifest";
end

input_file = fullfile(root, 'data', 'nik', 'experimental', ...
    'nik_breath_hold_levels.csv');
if ~isfile(input_file)
    error('trkg4:missingNikInhaleManifest', ...
        'Missing Nik inhale manifest: %s', input_file);
end

T = readtable(input_file, 'TextType', 'string');
required = ["L_mm", "Z_inhale_hold_ohm", "source_file", "is_independent"];
if ~all(ismember(required, string(T.Properties.VariableNames)))
    error('trkg4:invalidNikInhaleManifest', ...
        'Nik inhale manifest is missing one or more required columns.');
end

selection_mode = string(selection_mode);
switch selection_mode
    case "manifest"
        accepted = T.is_independent == 1;
    case "legacy_100"
        row90 = T.L_mm == 90;
        row100 = T.L_mm == 100;
        if sum(row90) ~= 1 || sum(row100) ~= 1 || ...
                T.Z_inhale_hold_ohm(row90) ~= T.Z_inhale_hold_ohm(row100)
            error('trkg4:invalidNikLegacyDuplicate', ...
                'Legacy ablation requires one equal-valued 90/100-mm pair.');
        end
        accepted = T.is_independent == 1;
        accepted(row90) = false;
        accepted(row100) = true;
    otherwise
        error('trkg4:unknownNikInhaleSelection', ...
            'Unknown Nik inhale selection mode: %s', selection_mode);
end
T = T(accepted, :);
if height(T) ~= 9 || numel(unique(T.L_mm)) ~= height(T)
    error('trkg4:invalidNikIndependentSeries', ...
        'Expected exactly nine unique independent Nik grid sizes.');
end
if any(~isfinite(T.L_mm)) || any(~isfinite(T.Z_inhale_hold_ohm))
    error('trkg4:nonfiniteNikInhaleData', ...
        'Nik inhale sizes and impedance values must be finite.');
end

[sizes_mm, order] = sort(T.L_mm);
z_inhale_ohm = T.Z_inhale_hold_ohm(order);
source_files = T.source_file(order);
end
