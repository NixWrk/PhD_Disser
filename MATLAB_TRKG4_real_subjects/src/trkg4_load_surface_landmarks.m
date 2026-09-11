function landmarks = trkg4_load_surface_landmarks(json_file)
%TRKG4_LOAD_SURFACE_LANDMARKS Load validated surface-path landmarks from JSON.
%
% The current contract uses positive and negative guide paths on arbitrary
% parts of the external body surface. The legacy arm-specific v2 contract
% and the original schema-free right/left contract remain readable.
% Unknown fields and browser-derived preliminary centres are not trusted or
% propagated to consumers.

    if ~(ischar(json_file) && isrow(json_file)) && ...
            ~(isstring(json_file) && isscalar(json_file))
        error("trkg4:load_surface_landmarks:InvalidInputPath", ...
            "Путь к JSON должен быть текстовым скаляром.");
    end

    json_file = char(json_file);
    if isempty(json_file)
        error("trkg4:load_surface_landmarks:InvalidInputPath", ...
            "Путь к JSON не должен быть пустым.");
    end
    if ~isfile(json_file)
        error("trkg4:load_surface_landmarks:FileNotFound", ...
            "JSON-файл поверхностных ориентиров не найден.");
    end

    try
        json_text = fileread(json_file);
    catch exception
        error("trkg4:load_surface_landmarks:FileRead", ...
            "Не удалось прочитать JSON-файл: %s", exception.message);
    end

    try
        payload = jsondecode(json_text);
    catch exception
        error("trkg4:load_surface_landmarks:InvalidJson", ...
            "JSON-файл имеет недопустимый синтаксис: %s", exception.message);
    end

    if ~isstruct(payload) || ~isscalar(payload)
        error("trkg4:load_surface_landmarks:InvalidContract", ...
            "Корень JSON должен быть одним объектом.");
    end

    localRequireFields(payload, ...
        {'coordinate_system', 'units', 'source_basename'});

    coordinate_system = localTextScalar( ...
        payload.coordinate_system, ...
        "trkg4:load_surface_landmarks:InvalidCoordinateSystem", ...
        "coordinate_system");
    if ~strcmp(coordinate_system, 'surface_path_v1')
        error("trkg4:load_surface_landmarks:InvalidCoordinateSystem", ...
            "Поле coordinate_system должно иметь значение «surface_path_v1».");
    end

    units = localTextScalar( ...
        payload.units, ...
        "trkg4:load_surface_landmarks:InvalidUnits", ...
        "units");
    if ~strcmp(units, 'mm')
        error("trkg4:load_surface_landmarks:InvalidUnits", ...
            "Поле units должно иметь значение «mm».");
    end

    source_basename = localTextScalar( ...
        payload.source_basename, ...
        "trkg4:load_surface_landmarks:InvalidSourceBasename", ...
        "source_basename");
    if ~localIsBasename(source_basename)
        error("trkg4:load_surface_landmarks:InvalidSourceBasename", ...
            ['Поле source_basename должно содержать только имя файла ' ...
             'без абсолютного или вложенного пути.']);
    end

    schema = '';
    if isfield(payload, 'schema')
        schema = localTextScalar(payload.schema, ...
            "trkg4:load_surface_landmarks:InvalidTtrkgContract", "schema");
    end

    landmarks = struct( ...
        'coordinate_system', coordinate_system, ...
        'units', units, ...
        'source_basename', source_basename);

    if strcmp(schema, 'trkg4_ttrkg_surface_v3')
        localRequireFields(payload, {'placement_mode'});
        placement_mode = localTextScalar(payload.placement_mode, ...
            "trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
            "placement_mode");
        if strcmp(placement_mode, 'explicit_points')
            localRequireFields(payload, ...
                {'preliminary_electrode_centers_xyz_mm'});
            landmarks.requested_centres_xyz_mm = localElectrodeCentres( ...
                payload.preliminary_electrode_centers_xyz_mm);
        elseif strcmp(placement_mode, 'symmetric_paths')
            localRequireFields(payload, ...
                {'positive_guide_points_xyz_mm', ...
                 'negative_guide_points_xyz_mm'});
            landmarks.positive_guide_points_xyz_mm = localGuidePoints( ...
                payload.positive_guide_points_xyz_mm, ...
                'positive_guide_points_xyz_mm');
            landmarks.negative_guide_points_xyz_mm = localGuidePoints( ...
                payload.negative_guide_points_xyz_mm, ...
                'negative_guide_points_xyz_mm');
        else
            error("trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
                "Поле placement_mode имеет недопустимое значение.");
        end
    else
        localRequireFields(payload, ...
            {'right_guide_points_xyz_mm', 'left_guide_points_xyz_mm'});
        landmarks.right_guide_points_xyz_mm = localGuidePoints( ...
            payload.right_guide_points_xyz_mm, ...
            'right_guide_points_xyz_mm');
        landmarks.left_guide_points_xyz_mm = localGuidePoints( ...
            payload.left_guide_points_xyz_mm, ...
            'left_guide_points_xyz_mm');
    end

    if ~isempty(schema)
        ttrkg = localTtrkgContract(payload);
        landmarks.schema = ttrkg.schema;
        landmarks.modality = ttrkg.modality;
        landmarks.montage = ttrkg.montage;
        landmarks.placement_mode = ttrkg.placement_mode;
        landmarks.reference_definition = ttrkg.reference_definition;
        landmarks.electrode_order = ttrkg.electrode_order;
        landmarks.symmetry = ttrkg.symmetry;
    end
end

function localRequireFields(payload, required_fields)
    for field_index = 1:numel(required_fields)
        field_name = required_fields{field_index};
        if ~isfield(payload, field_name)
            error("trkg4:load_surface_landmarks:MissingField", ...
                "В JSON отсутствует обязательное поле «%s».", field_name);
        end
    end
end

function text_value = localTextScalar(value, error_identifier, field_name)
    is_character_scalar = ischar(value) && isrow(value);
    is_string_scalar = isstring(value) && isscalar(value) && ~ismissing(value);
    if ~(is_character_scalar || is_string_scalar)
        error(error_identifier, ...
            "Поле %s должно быть непустой текстовой строкой.", field_name);
    end

    text_value = char(value);
    if isempty(text_value)
        error(error_identifier, ...
            "Поле %s не должно быть пустым.", field_name);
    end
end

function is_basename = localIsBasename(value)
    has_separator = contains(value, '/') || contains(value, '\');
    has_drive_prefix = ~isempty(regexp(value, '^[A-Za-z]:', 'once'));
    is_dot_component = strcmp(value, '.') || strcmp(value, '..');
    is_basename = ~has_separator && ~has_drive_prefix && ~is_dot_component;
end

function contract = localTtrkgContract(payload)
    localRequireFields(payload, ...
        {'schema', 'modality', 'montage', ...
         'reference_definition', 'electrode_order'});

    schema = localTextScalar(payload.schema, ...
        "trkg4:load_surface_landmarks:InvalidTtrkgContract", "schema");
    modality = localTextScalar(payload.modality, ...
        "trkg4:load_surface_landmarks:InvalidTtrkgContract", "modality");
    montage = localTextScalar(payload.montage, ...
        "trkg4:load_surface_landmarks:InvalidTtrkgContract", "montage");
    reference_definition = localTextScalar(payload.reference_definition, ...
        "trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
        "reference_definition");

    if strcmp(schema, 'trkg4_ttrkg_surface_v3')
        localRequireFields(payload, {'placement_mode', 'symmetry'});
        placement_mode = localTextScalar(payload.placement_mode, ...
            "trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
            "placement_mode");
        is_explicit = strcmp(placement_mode, 'explicit_points') && ...
            strcmp(reference_definition, ...
            'user_selected_centres_on_external_body_surface');
        is_symmetric = strcmp(placement_mode, 'symmetric_paths') && ...
            strcmp(reference_definition, ...
            'first_resolved_guide_point_at_user_selected_anatomical_reference');
        if ~strcmp(montage, 'four_electrode_surface') || ...
                ~(is_explicit || is_symmetric)
            error("trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
                "Поля монтажа, режима или опорной точки ТТРКГ недопустимы.");
        end
    elseif strcmp(schema, 'trkg4_ttrkg_arm_surface_v2')
        placement_mode = 'symmetric_paths';
        is_explicit = false;
        is_symmetric = strcmp(montage, 'four_electrode_arm_surface') && ...
            strcmp(reference_definition, ...
            'first_resolved_guide_point_at_arm_trunk_boundary_proximal_to_distal');
        if ~is_symmetric
            error("trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
                "Поля устаревшего arm-контракта ТТРКГ недопустимы.");
        end
    else
        error("trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
            "Схема контракта ТТРКГ не поддерживается.");
    end
    if ~strcmp(modality, 'TTRKG')
        error("trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
            "Поле modality должно иметь значение TTRKG.");
    end

    if is_explicit
        if ~isempty(payload.symmetry)
            error("trkg4:load_surface_landmarks:InvalidTtrkgSymmetry", ...
                "В режиме explicit_points поле symmetry должно быть пустым.");
        end
        symmetry = [];
    else
        if ~isstruct(payload.symmetry) || ~isscalar(payload.symmetry) || ...
                ~isfield(payload.symmetry, 'mode') || ...
                ~isfield(payload.symmetry, 'inner_from_reference_mm') || ...
                ~isfield(payload.symmetry, 'outer_from_inner_mm')
            error("trkg4:load_surface_landmarks:InvalidTtrkgSymmetry", ...
                "Раздел symmetry должен задавать режим и два расстояния.");
        end
        mode = localTextScalar(payload.symmetry.mode, ...
            "trkg4:load_surface_landmarks:InvalidTtrkgSymmetry", ...
            "symmetry.mode");
        if ~strcmp(mode, 'equal_path_distances')
            error("trkg4:load_surface_landmarks:InvalidTtrkgSymmetry", ...
                "Поддерживается режим symmetry.mode=equal_path_distances.");
        end
        inner_mm = localFiniteNonnegativeScalar( ...
            payload.symmetry.inner_from_reference_mm, ...
            'symmetry.inner_from_reference_mm');
        outer_mm = localFinitePositiveScalar( ...
            payload.symmetry.outer_from_inner_mm, ...
            'symmetry.outer_from_inner_mm');
        symmetry = struct('mode', mode, ...
            'inner_from_reference_mm', inner_mm, ...
            'outer_from_inner_mm', outer_mm);
    end

    electrode_order = string(payload.electrode_order(:));
    expected_order = ["I_plus"; "V_plus"; "V_minus"; "I_minus"];
    if ~isequal(electrode_order, expected_order)
        error("trkg4:load_surface_landmarks:InvalidTtrkgContract", ...
            "Порядок электродов должен быть I_plus, V_plus, V_minus, I_minus.");
    end

    contract = struct('schema', schema, 'modality', modality, ...
        'montage', montage, 'placement_mode', placement_mode, ...
        'reference_definition', reference_definition, ...
        'electrode_order', electrode_order, 'symmetry', symmetry);
end

function scalar = localFiniteNonnegativeScalar(value, field_name)
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value < 0
        error("trkg4:load_surface_landmarks:InvalidTtrkgSymmetry", ...
            "Поле %s должно быть конечным неотрицательным числом.", field_name);
    end
    scalar = double(value);
end

function scalar = localFinitePositiveScalar(value, field_name)
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value <= 0
        error("trkg4:load_surface_landmarks:InvalidTtrkgSymmetry", ...
            "Поле %s должно быть конечным положительным числом.", field_name);
    end
    scalar = double(value);
end

function centres = localElectrodeCentres(value)
    expected = {'I_plus', 'V_plus', 'V_minus', 'I_minus'};
    if ~isstruct(value) || ~isscalar(value)
        error("trkg4:load_surface_landmarks:InvalidElectrodeCentres", ...
            "Центры электродов должны быть объектом с четырьмя именованными точками.");
    end
    centres = zeros(4, 3);
    for index = 1:numel(expected)
        name = expected{index};
        if ~isfield(value, name)
            error("trkg4:load_surface_landmarks:InvalidElectrodeCentres", ...
                "Отсутствует центр электрода %s.", name);
        end
        point = double(value.(name));
        if ~isnumeric(point) || ~isreal(point) || ...
                numel(point) ~= 3 || any(~isfinite(point(:)))
            error("trkg4:load_surface_landmarks:InvalidElectrodeCentres", ...
                "Центр электрода %s должен содержать три конечные координаты.", name);
        end
        centres(index, :) = reshape(point, 1, 3);
    end
end

function points = localGuidePoints(value, field_name)
    if ~(isnumeric(value) && isreal(value) && ismatrix(value))
        error("trkg4:load_surface_landmarks:InvalidGuidePoints", ...
            "Поле %s должно быть числовой матрицей.", field_name);
    end

    points = double(value);
    if size(points, 2) ~= 3 || size(points, 1) < 2
        error("trkg4:load_surface_landmarks:InvalidGuidePoints", ...
            ['Поле %s должно иметь размер N×3 при N не менее 2 ' ...
             '(по одной XYZ-строке на точку).'], field_name);
    end
    if any(~isfinite(points), 'all')
        error("trkg4:load_surface_landmarks:InvalidGuidePoints", ...
            "Поле %s содержит NaN, Inf или -Inf.", field_name);
    end
end
