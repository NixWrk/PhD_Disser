function report = trkg4_export_sensitivity_model( ...
    fmdl_m, tissue_id, tissue_names, groups, output_directory)
%TRKG4_EXPORT_SENSITIVITY_MODEL Export a checked SI FEM/CEM algebraic model.
%
% fmdl_m is already in metres and is never rescaled or edited. The output
% records only the algebraic assembly check, not anatomy or convergence QC.

if nargin ~= 5
    error('trkg4:exportArgumentCount', ...
        'Expected fmdl_m, tissue_id, tissue_names, groups, output_directory.');
end

[node_count, element_count, electrode_count, ground] = ...
    local_validate_model(fmdl_m);
[tissue_id, tissue_names] = local_validate_tissues( ...
    tissue_id, tissue_names, element_count);
groups = local_validate_groups(groups, element_count);
output_directory = local_validate_output_directory(output_directory);

required_functions = {'system_mat_fields', 'mk_image', ...
    'calc_system_mat', 'trkg4_cem_contact_matrix', 'trkg4_file_sha256'};
missing = required_functions(cellfun( ...
    @(name) exist(name, 'file') == 0, required_functions));
if ~isempty(missing)
    error('trkg4:missingEidorsForSensitivityExport', ...
        'Required MATLAB/EIDORS functions are missing: %s.', ...
        strjoin(missing, ', '));
end

fields = system_mat_fields(fmdl_m);
expected_field_rows = 3 * element_count;
expected_columns = node_count + electrode_count;
if size(fields, 1) < expected_field_rows || ...
        size(fields, 2) ~= expected_columns
    error('trkg4:unexpectedSystemMatFieldsSize', ...
        ['system_mat_fields returned %d-by-%d; expected at least ', ...
         '%d volume rows and %d columns for a linear tetrahedral model.'], ...
        size(fields, 1), size(fields, 2), ...
        expected_field_rows, expected_columns);
end
% EIDORS appends CEM-contact rows after the 3*ne volume rows. Only the
% volume rows contribute to the exported F_g' * F_g blocks.
volume_fields = fields(1:expected_field_rows, :);
if ~isreal(volume_fields) || any(~isfinite(nonzeros(volume_fields)))
    error('trkg4:nonfiniteVolumeFields', ...
        'The volume rows of system_mat_fields must be finite and real.');
end

blocks = cell(numel(groups), 1);
assembled = sparse(expected_columns, expected_columns);
for group_index = 1:numel(groups)
    rows = reshape( ...
        (3 * (find(groups{group_index}) - 1) + (1:3)).', [], 1);
    group_fields = volume_fields(rows, :);
    if ~issparse(group_fields)
        group_fields = sparse(group_fields);
    end
    blocks{group_index} = group_fields' * group_fields;
    assembled = assembled + blocks{group_index};
end

C = trkg4_cem_contact_matrix(fmdl_m);
if ~isequal(size(C), [expected_columns, expected_columns])
    error('trkg4:unexpectedContactMatrixSize', ...
        'C has size %d-by-%d; expected %d-by-%d.', ...
        size(C, 1), size(C, 2), expected_columns, expected_columns);
end
if any(~isfinite(nonzeros(C)))
    error('trkg4:nonfiniteContactMatrix', ...
        'C contains non-finite entries.');
end
electrode_diagonal = full(diag(C(node_count + (1:electrode_count), ...
    node_count + (1:electrode_count))));
if any(electrode_diagonal <= 0)
    error('trkg4:emptyElectrodeContact', ...
        'Every electrode must have a positive CEM contact contribution.');
end

% Actual EIDORS comparison required by the export contract.
reference_system = calc_system_mat(mk_image(fmdl_m, 1));
if ~isfield(reference_system, 'E')
    error('trkg4:missingEidorsSystemMatrix', ...
        'calc_system_mat did not return the E matrix.');
end
reference_matrix = reference_system.E;
if ~isequal(size(reference_matrix), [expected_columns, expected_columns])
    error('trkg4:unexpectedEidorsSystemMatrixSize', ...
        'E has an unexpected system-matrix size.');
end
assembled_with_contact = assembled + C;
reference_norm = norm(reference_matrix, 'fro');
matrix_error = norm(assembled_with_contact - reference_matrix, 'fro') / ...
    max(reference_norm, eps);
if ~isfinite(matrix_error)
    error('trkg4:nonfiniteMatrixError', ...
        'The EIDORS assembly comparison is non-finite.');
end
if matrix_error >= 1e-12
    error('trkg4:eidorsAssemblyMismatch', ...
        'sum(blocks)+C differs from E with relative error %.6g.', ...
        matrix_error);
end

local_create_output_directory(output_directory);
prepared_file = fullfile(output_directory, 'prepared_fem.mat');
contact_file = fullfile(output_directory, 'contact.mat');
if isfile(prepared_file) || isfile(contact_file)
    error('trkg4:outputWouldOverwrite', ...
        'A prepared output already exists in %s.', output_directory);
end

coordinate_units = 'm';
save(prepared_file, 'fmdl_m', 'blocks', 'tissue_id', ...
    'tissue_names', 'coordinate_units', '-v7.3');
% Bind contact.mat to the exact prepared file written above.
prepared_sha256 = trkg4_file_sha256(prepared_file);
electrode_ids = arrayfun(@(index) sprintf('E%d', index), ...
    1:electrode_count, 'UniformOutput', false);
save(contact_file, 'C', 'ground', 'matrix_error', ...
    'prepared_sha256', 'electrode_ids', '-v7');

prepared_hash = prepared_sha256;
contact_hash = trkg4_file_sha256(contact_file);
counts = struct( ...
    'nodes', node_count, ...
    'elements', element_count, ...
    'electrodes', electrode_count, ...
    'groups', numel(groups), ...
    'tissues', numel(tissue_names));
hashes = struct( ...
    'prepared_fem', prepared_hash, ...
    'contact', contact_hash, ...
    'prepared_fem_sha256', prepared_hash, ...
    'contact_sha256', contact_hash);
report = struct();
report.status = 'exported_algebraic_check_passed';
report.paths = struct('prepared_fem', prepared_file, 'contact', contact_file);
report.hashes = hashes;
report.counts = counts;
report.units = struct( ...
    'coordinates', 'm', ...
    'coordinate_units', 'm', ...
    'conductivity', 'S/m', ...
    'contact_impedance', 'ohm*m^2');
report.coordinate_units = 'm';
report.matrix_error = matrix_error;
report.electrode_ids = electrode_ids;
report.prepared_sha256 = prepared_sha256;
report.prepared_fem_file = prepared_file;
report.contact_file = contact_file;
report.prepared_fem_sha256 = prepared_hash;
report.contact_sha256 = contact_hash;
report.validation = struct( ...
    'algebraic_matrix_check', 'passed', ...
    'anatomical_qc', 'not_evaluated', ...
    'mesh_convergence', 'not_evaluated', ...
    'contact_terms_in_volume_blocks', false);
report.group_element_counts = cellfun(@nnz, groups);
end

function [node_count, element_count, electrode_count, ground] = ...
    local_validate_model(fmdl_m)
if ~isstruct(fmdl_m) || ~isscalar(fmdl_m)
    error('trkg4:invalidForwardModel', ...
        'fmdl_m must be one scalar EIDORS forward-model struct.');
end
if ~isfield(fmdl_m, 'nodes') || ~isfield(fmdl_m, 'elems')
    error('trkg4:invalidForwardModel', ...
        'fmdl_m must contain nodes and elems.');
end

nodes = fmdl_m.nodes;
if ~isnumeric(nodes) || ~isreal(nodes) || ndims(nodes) ~= 2 || ...
        size(nodes, 2) ~= 3 || isempty(nodes) || ...
        any(~isfinite(nodes(:)))
    error('trkg4:invalidCoordinates', ...
        'fmdl_m.nodes must be a finite N-by-3 real array.');
end
node_count = size(nodes, 1);

elems = fmdl_m.elems;
if ~isnumeric(elems) || ~isreal(elems) || ndims(elems) ~= 2 || ...
        size(elems, 2) ~= 4 || isempty(elems)
    error('trkg4:invalidElements', ...
        'fmdl_m.elems must be a non-empty linear tetrahedral array.');
end
element_count = size(elems, 1);
elem_values = double(elems(:));
if any(~isfinite(elem_values)) || any(elem_values ~= fix(elem_values)) || ...
        any(elem_values < 1) || any(elem_values > node_count)
    error('trkg4:invalidElementIndices', ...
        'fmdl_m.elems contains a non-integer or out-of-range node index.');
end
sorted_elems = sort(double(elems), 2);
if any(any(diff(sorted_elems, 1, 2) == 0))
    error('trkg4:repeatedElementNode', ...
        'A tetrahedral element repeats a node index.');
end
node_values = double(nodes);
element_indices = double(elems);
p0 = node_values(element_indices(:, 1), :);
edge_1 = node_values(element_indices(:, 2), :) - p0;
edge_2 = node_values(element_indices(:, 3), :) - p0;
edge_3 = node_values(element_indices(:, 4), :) - p0;
determinants = dot(edge_1, cross(edge_2, edge_3, 2), 2);
edge_scale = vecnorm(edge_1, 2, 2) .* vecnorm(edge_2, 2, 2) .* ...
    vecnorm(edge_3, 2, 2);
if any(~isfinite(determinants)) || ...
        any(abs(determinants) <= 128 * eps * edge_scale)
    error('trkg4:degenerateElements', ...
        'fmdl_m.elems contains a degenerate tetrahedron.');
end

if ~isfield(fmdl_m, 'gnd_node')
    error('trkg4:missingGroundNode', ...
        'fmdl_m.gnd_node is required.');
end
ground_value = fmdl_m.gnd_node;
if ~isnumeric(ground_value) || ~isreal(ground_value) || ...
        ~isscalar(ground_value) || ~isfinite(ground_value) || ...
        ground_value ~= fix(ground_value) || ...
        ground_value < 1 || ground_value > node_count
    error('trkg4:groundMustBeTissueNode', ...
        'gnd_node must be a 1-based tissue-node index.');
end
ground = double(ground_value);
if ~any(double(elems(:)) == ground)
    error('trkg4:groundMustBeTissueNode', ...
        'gnd_node must occur in at least one tissue element.');
end

if ~isfield(fmdl_m, 'electrode') || ...
        ~isstruct(fmdl_m.electrode) || isempty(fmdl_m.electrode)
    error('trkg4:missingElectrodes', ...
        'fmdl_m.electrode must contain at least two electrodes.');
end
electrode_count = numel(fmdl_m.electrode);
if electrode_count < 2
    error('trkg4:tooFewElectrodes', ...
        'At least two electrodes are required.');
end
for electrode_index = 1:electrode_count
    electrode = fmdl_m.electrode(electrode_index);
    if ~isfield(electrode, 'z_contact')
        error('trkg4:invalidElectrodeContact', ...
            'Electrode %d has no z_contact.', electrode_index);
    end
    z_contact = electrode.z_contact;
    if ~isnumeric(z_contact) || ~isreal(z_contact) || ...
            ~isscalar(z_contact) || ~isfinite(z_contact) || z_contact <= 0
        error('trkg4:invalidElectrodeContact', ...
            'Electrode %d has an invalid z_contact.', electrode_index);
    end

    has_faces = isfield(electrode, 'faces') && ...
        ~isempty(electrode.faces);
    has_nodes = isfield(electrode, 'nodes') && ...
        ~isempty(electrode.nodes);
    if ~has_faces && ~has_nodes
        error('trkg4:emptyElectrodeContact', ...
            'Electrode %d has neither faces nor nodes.', electrode_index);
    end
    if has_faces
        local_validate_faces(electrode.faces, node_count, electrode_index, nodes);
    else
        if ~isfield(fmdl_m, 'boundary') || isempty(fmdl_m.boundary)
            error('trkg4:missingBoundaryForElectrode', ...
                ['Electrode %d has no faces and fmdl_m.boundary is ', ...
                 'missing.'], electrode_index);
        end
        local_validate_faces(fmdl_m.boundary, node_count, electrode_index, nodes);
    end
    if has_nodes
        local_validate_nodes(electrode.nodes, node_count, electrode_index);
    end
end
end

function local_validate_faces(faces, node_count, electrode_index, nodes)
if ~isnumeric(faces) || ~isreal(faces) || ndims(faces) ~= 2 || ...
        size(faces, 2) ~= 3 || isempty(faces)
    error('trkg4:invalidElectrodeFaces', ...
        'Electrode %d faces must be an N-by-3 array.', electrode_index);
end
face_indices = double(faces);
if any(~isfinite(face_indices(:))) || ...
        any(face_indices(:) ~= fix(face_indices(:))) || ...
        any(face_indices(:) < 1) || any(face_indices(:) > node_count)
    error('trkg4:invalidElectrodeFaces', ...
        'Electrode %d faces contain an invalid node index.', electrode_index);
end
if any(any(diff(sort(face_indices, 2), 1, 2) == 0))
    error('trkg4:repeatedElectrodeNode', ...
        'Electrode %d repeats a node within a face.', electrode_index);
end

if nargin >= 4 && ~isempty(nodes)
    node_values = double(nodes);
    p0 = node_values(face_indices(:, 1), :);
    edge_1 = node_values(face_indices(:, 2), :) - p0;
    edge_2 = node_values(face_indices(:, 3), :) - p0;
    cross_norm = vecnorm(cross(edge_1, edge_2, 2), 2, 2);
    edge_scale = vecnorm(edge_1, 2, 2) .* vecnorm(edge_2, 2, 2);
    if any(~isfinite(cross_norm)) || ...
            any(cross_norm <= 128 * eps * edge_scale)
        error('trkg4:degenerateElectrodeFaces', ...
            'Electrode %d contains a degenerate face.', electrode_index);
    end
end
end

function local_validate_nodes(nodes, node_count, electrode_index)
if ~isnumeric(nodes) || ~isreal(nodes) || ~isvector(nodes) || ...
        isempty(nodes)
    error('trkg4:invalidElectrodeNodes', ...
        'Electrode %d nodes must be a non-empty index vector.', ...
        electrode_index);
end
values = double(nodes(:));
if any(~isfinite(values)) || any(values ~= fix(values)) || ...
        any(values < 1) || any(values > node_count)
    error('trkg4:invalidElectrodeNodes', ...
        'Electrode %d nodes contain an invalid index.', electrode_index);
end
end
function [ids, names] = local_validate_tissues(ids, names, ne)
if ~isnumeric(ids) || ~isreal(ids) || ~isvector(ids) || numel(ids) ~= ne || ...
        any(~isfinite(ids(:))) || any(ids(:) ~= fix(ids(:)))
    error('trkg4:invalidTissueLabels', 'One finite integer tissue label per element is required.');
end
ids = double(ids(:));
if isstring(names), names = cellstr(names(:)); end
if ~iscellstr(names) || isempty(names) || any(cellfun(@(s) isempty(strtrim(s)), names)) || ...
        numel(unique(names)) ~= numel(names) || ...
        ~isequal(unique(ids), (1:numel(names))')
    error('trkg4:invalidTissueNames', 'Nonempty unique names must match consecutive labels 1..N.');
end
names = names(:);
end

function groups = local_validate_groups(groups, ne)
if ~iscell(groups) || isempty(groups)
    error('trkg4:invalidGroups', 'Groups must be a nonempty cell array of logical element masks.');
end
coverage = zeros(ne, 1);
for k = 1:numel(groups)
    mask = groups{k};
    if ~islogical(mask) || ~isvector(mask) || numel(mask) ~= ne || ~any(mask)
        error('trkg4:invalidGroups', 'Every group must be a nonempty logical mask of all elements.');
    end
    groups{k} = mask(:);
    coverage = coverage + double(mask(:));
end
if any(coverage ~= 1)
    error('trkg4:invalidGroups', 'Groups must partition all elements exactly once.');
end
groups = groups(:);
end

function directory = local_validate_output_directory(directory)
if isstring(directory) && isscalar(directory), directory = char(directory); end
if ~ischar(directory) || ~isrow(directory) || isempty(strtrim(directory)) || any(directory == char(0))
    error('trkg4:invalidOutputDirectory', 'Output directory must be a nonempty scalar path.');
end
if isfile(directory)
    error('trkg4:outputWouldOverwrite', 'Output path is an existing file.');
end
if isfolder(directory)
    entries = dir(directory);
    entries = entries(~ismember({entries.name}, {'.', '..'}));
    if ~isempty(entries)
        error('trkg4:outputWouldOverwrite', 'Output directory must be new or empty.');
    end
end
end

function local_create_output_directory(directory)
local_validate_output_directory(directory);
if ~isfolder(directory)
    [ok, message] = mkdir(directory);
    if ~ok, error('trkg4:cannotCreateOutput', 'Cannot create output directory: %s', message); end
end
end
