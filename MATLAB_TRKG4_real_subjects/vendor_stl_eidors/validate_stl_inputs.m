function validate_stl_inputs(cfg)
%VALIDATE_STL_INPUTS Print and validate the STL surfaces used by the model.

if nargin < 1 || isempty(cfg)
    cfg = trkg4_config('nik');
end

fprintf('STL input mode: %s\n', cfg.input_mode);

if isempty(cfg.body_stl) || ~isfile(cfg.body_stl)
    error(['External body STL is required.\n', ...
           'Expected data/<subject>/stl/body.stl or edit trkg4_subject_registry.m.']);
end

fprintf('Body surface:\n');
print_file_info(cfg.body_stl);

fprintf('\nTissue surfaces:\n');
missing_required = {};
missing_optional = {};
for k = 1:numel(cfg.tissues)
    tissue = cfg.tissues(k);
    is_required = any(strcmp(tissue.name, cfg.required_tissues));
    if isempty(tissue.file) || ~isfile(tissue.file)
        if is_required
            fprintf('  %-12s MISSING REQUIRED\n', tissue.name);
            missing_required{end+1} = tissue.name; %#ok<AGROW>
        else
            fprintf('  %-12s missing optional -> no separate assignment\n', tissue.name);
            missing_optional{end+1} = tissue.name; %#ok<AGROW>
        end
        continue;
    end

    if is_required
        label = 'required';
    else
        label = 'optional';
    end
    fprintf('  %-12s %s\n', tissue.name, label);
    print_file_info(tissue.file);
end

if ~isempty(missing_required)
    error('Missing required tissue STL files: %s', strjoin(missing_required, ', '));
end

% Optional does not mean harmless. An absent mask is not assigned as a
% separate class: an element retains its previous tissue-mask assignment or,
% when no earlier mask contains it, the background. Say so explicitly so the
% result is never read as if every declared tissue had been resolved.
if ~isempty(missing_optional)
    fprintf(['\nNOTE: %s is not assigned as a separate tissue class. ', ...
        'Affected elements retain an earlier tissue-mask assignment or ', ...
        'the %s background. Do not claim that this class was resolved ', ...
        'separately in this run.\n'], strjoin(missing_optional, ', '), ...
        cfg.background.name);
end

fprintf('\nSTL input check passed.\n');
end

function print_file_info(filename)
info = dir(filename);
fprintf('    %s (%.1f MB)\n', filename, info.bytes / 1024^2);
end
