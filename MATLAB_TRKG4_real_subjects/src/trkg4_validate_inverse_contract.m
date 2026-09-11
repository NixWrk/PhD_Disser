function trkg4_validate_inverse_contract(root, opt)
filename = fullfile(root, 'output', sprintf( ...
    'nik_trkg4_inverse_inhale_contract_%s.json', opt.ResultTag));
contract = jsondecode(fileread(filename));
if contract.schema_version ~= 1 || ...
        string(contract.result_tag) ~= string(opt.ResultTag) || ...
        string(contract.data_selection) ~= string(opt.DataSelection) || ...
        string(contract.patch_builder_mode) ~= string(opt.PatchBuilderMode) || ...
        logical(contract.compute_jacobian) ~= opt.ComputeJacobian || ...
        ~logical(contract.geometry_qc_passed)
    error('trkg4:inverseContractMismatch', ...
        'The saved inverse result does not match the requested run.');
end
expected_units = struct('coordinates', 'mm', 'resistivity', 'ohm_m', ...
    'transfer_impedance', 'ohm', 'contact_impedance', 'ohm_m2');
unit_fields = fieldnames(expected_units);
if ~isfield(contract, 'units') || ~all(isfield(contract.units, unit_fields))
    error('trkg4:inverseContractMismatch', 'Missing physical units in inverse contract.');
end
for k = 1:numel(unit_fields)
    key = unit_fields{k};
    if ~strcmp(contract.units.(key), expected_units.(key))
        error('trkg4:inverseContractMismatch', 'Wrong physical unit for %s.', key);
    end
end
if isfield(contract, 'runtime')
    if ~strcmp(contract.runtime.matlab, version)
        error('trkg4:inverseContractMismatch', 'MATLAB runtime changed.');
    end
    names = {'fwd_solve', 'system_mat_fields'};
    for k = 1:numel(names)
        key = [names{k} '_sha256'];
        if isfield(contract.runtime, key)
            file = which(names{k});
            if isempty(file) || ~strcmp( ...
                    trkg4_file_sha256(file), contract.runtime.(key))
                error('trkg4:inverseContractMismatch', ...
                    'EIDORS operator changed or unavailable: %s.', names{k});
            end
        end
    end
end
records = [contract.artifacts(:); contract.inputs(:)];
for k = 1:numel(records)
    if isempty(records(k).path)
        error('trkg4:externalInputNotVerified', ...
            'Cannot reuse without verifying external input %s.', records(k).name);
    end
    relative = strrep(char(records(k).path), '/', filesep);
    candidate = char(java.io.File(fullfile(root, relative)).getCanonicalPath());
    canonical_root = char(java.io.File(root).getCanonicalPath());
    if ~startsWith(candidate, [canonical_root filesep], 'IgnoreCase', true) || ...
            ~isfile(candidate) || ...
            ~strcmp(trkg4_file_sha256(candidate), records(k).sha256)
        error('trkg4:inverseContractMismatch', ...
            'Changed or missing inverse artifact/input: %s', records(k).name);
    end
end
end
