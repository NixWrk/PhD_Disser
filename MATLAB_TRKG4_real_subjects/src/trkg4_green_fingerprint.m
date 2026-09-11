function fingerprint = trkg4_green_fingerprint( ...
    fmdl_si, elem_sigma, candidate_nodes, cache_settings)
%TRKG4_GREEN_FINGERPRINT Hash the physical payload of a grounded Green solve.
%
%   fingerprint = trkg4_green_fingerprint( ...
%       fmdl_si, elem_sigma, candidate_nodes, cache_settings)
%
% The fingerprint explicitly covers coordinates, connectivity, boundary
% connectivity, element conductivity, ground, candidate-node order, and
% settings. It also retains the complete residual fmdl operator context
% after removing only fields represented explicitly here or supplied by the
% electrode/stimulation solve: nodes, elems, boundary, gnd_node, gnd_nodes,
% electrode, stimulation, meas_select, and elem_sigma. Thus fields such as
% system_mat, approx_type, coarse2fine, and other operator context cannot
% silently change without invalidating the cache.
%
% Numeric classes in the explicit payload and settings are canonicalised to
% double; field order in scalar settings structs is canonicalised. Array
% ordering remains significant because it determines matrix indexing.
%
% This helper does not inspect or use a cache. A cache record is valid only
% when it contains both cache_fingerprint and cache_schema and its
% fingerprint equals this value. A legacy record without those fields must
% be treated as invalid by the caller.

if nargin ~= 4 || ~isstruct(fmdl_si) || ~isscalar(fmdl_si)
    error('trkg4:greenFingerprintInput', ...
        'Expected fmdl_si, element sigma, candidate nodes, and settings.');
end
required_fields = {'nodes', 'elems', 'gnd_node'};
if ~all(isfield(fmdl_si, required_fields))
    error('trkg4:greenFingerprintInput', ...
        'fmdl_si must contain nodes, elems, and gnd_node.');
end
if ~isstruct(cache_settings) || ~isscalar(cache_settings)
    error('trkg4:greenFingerprintInput', ...
        'cache_settings must be a scalar struct.');
end

nodes = double(fmdl_si.nodes);
elems = double(fmdl_si.elems);
elem_sigma = double(elem_sigma);
gnd_node = double(fmdl_si.gnd_node);
candidate_nodes = double(candidate_nodes(:));
if isfield(fmdl_si, 'boundary')
    boundary = double(fmdl_si.boundary);
else
    boundary = [];
end
if isfield(fmdl_si, 'gnd_nodes')
    ground_context = struct( ...
        'gnd_node', gnd_node, ...
        'gnd_nodes', double(fmdl_si.gnd_nodes));
else
    ground_context = struct('gnd_node', gnd_node);
end

operator_context = fmdl_si;
explicit_fields = {'nodes', 'elems', 'boundary', 'elem_sigma', ...
    'gnd_node', 'gnd_nodes', 'electrode', 'stimulation', 'meas_select'};
present_fields = explicit_fields(isfield(operator_context, explicit_fields));
operator_context = rmfield(operator_context, present_fields);

payload = struct( ...
    'schema', 'trkg4_grounded_green_fingerprint_v4', ...
    'nodes', nodes, ...
    'elems', elems, ...
    'boundary', boundary, ...
    'elem_sigma', elem_sigma, ...
    'ground', ground_context, ...
    'candidate_nodes', candidate_nodes, ...
    'settings', local_canonicalise(cache_settings), ...
    'operator_context', operator_context);

if any(~isfinite(payload.nodes), 'all') || ...
        any(~isfinite(payload.elems), 'all') || ...
        any(~isfinite(payload.boundary), 'all') || ...
        any(~isfinite(payload.elem_sigma), 'all') || ...
        any(~isfinite(payload.ground.gnd_node), 'all') || ...
        any(~isfinite(payload.candidate_nodes), 'all')
    error('trkg4:greenFingerprintInput', ...
        'Fingerprint payload contains non-finite numeric values.');
end

bytes = getByteStreamFromArray(payload);
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(bytes);
digest_bytes = typecast(digest.digest(), 'uint8');
fingerprint = lower(reshape(dec2hex(digest_bytes, 2).', 1, []));
end

function value = local_canonicalise(value)
% Canonicalise common settings values without changing their meaning.
if isnumeric(value)
    value = double(value);
elseif islogical(value) || ischar(value)
    % Keep logical and character arrays in their native stable form.
elseif isstring(value)
    if isscalar(value)
        value = char(value);
    else
        value = cellstr(value);
    end
elseif iscell(value)
    for index = 1:numel(value)
        value{index} = local_canonicalise(value{index});
    end
elseif isstruct(value)
    field_names = sort(fieldnames(value));
    value = orderfields(value, field_names);
    for element_index = 1:numel(value)
        for field_index = 1:numel(field_names)
            field_name = field_names{field_index};
            value(element_index).(field_name) = ...
                local_canonicalise(value(element_index).(field_name));
        end
    end
else
    error('trkg4:greenFingerprintInput', ...
        'Unsupported cache setting type: %s.', class(value));
end
end
