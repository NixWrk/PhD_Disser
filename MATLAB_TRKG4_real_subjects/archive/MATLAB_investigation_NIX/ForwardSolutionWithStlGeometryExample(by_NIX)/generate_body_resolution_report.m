function report = generate_body_resolution_report()
%GENERATE_BODY_RESOLUTION_REPORT Save tables, plots, and a Markdown report.

out_dir = fullfile('output', 'body_resolution_sweep_combined');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

[report, measurements, metadata] = collect_body_resolution_results();
writetable(report, fullfile(out_dir, 'body_resolution_report_table.csv'));
save(fullfile(out_dir, 'body_resolution_report.mat'), 'report', 'measurements', 'metadata');

plot_measurements(report, measurements, out_dir);
plot_errors(report, out_dir);
plot_costs(report, out_dir);
plot_complexity(report, out_dir);
write_markdown_report(report, metadata, out_dir);

disp(report);
fprintf('Saved report to %s\n', fullfile(out_dir, 'body_resolution_report.md'));
end

function [report, measurements, metadata] = collect_body_resolution_results()
result_files = { ...
    fullfile('output', 'body_resolution_sweep', 'forward_body_6mm.mat'), ...
    fullfile('output', 'body_resolution_sweep', 'forward_body_4mm.mat'), ...
    fullfile('output', 'body_resolution_sweep', 'forward_body_3mm.mat'), ...
    fullfile('output', 'body_resolution_sweep_highres', 'forward_body_2mm.mat'), ...
    fullfile('output', 'body_resolution_sweep_highres', 'forward_body_1.5mm.mat'), ...
    fullfile('output', 'body_resolution_sweep_highres', 'forward_body_1mm.mat')};

existing = result_files(cellfun(@isfile, result_files));
if isempty(existing)
    error('No BODY resolution result files found.');
end

n = numel(existing);
surface_mm = zeros(n, 1);
surface_faces = zeros(n, 1);
nodes = zeros(n, 1);
elems = zeros(n, 1);
measurements_count = zeros(n, 1);
mesh_seconds = zeros(n, 1);
assignment_seconds = zeros(n, 1);
solve_seconds = zeros(n, 1);
total_seconds = zeros(n, 1);
nan_count = zeros(n, 1);
inf_count = zeros(n, 1);
soft_tissue = zeros(n, 1);
lungs = zeros(n, 1);
bones = zeros(n, 1);
blood = zeros(n, 1);
measurements = cell(n, 1);

metadata = struct();
for i = 1:n
    data = load(existing{i}, 'cfg', 'fmdl', 'v', 'timings', 'tissue_names', 'tissue_counts');
    surface_mm(i) = parse_surface_mm(data.cfg.body_stl);
    surface_faces(i) = size(data.fmdl.boundary, 1);
    nodes(i) = size(data.fmdl.nodes, 1);
    elems(i) = size(data.fmdl.elems, 1);
    measurements_count(i) = numel(data.v.meas);
    mesh_seconds(i) = data.timings.mesh_seconds;
    assignment_seconds(i) = data.timings.assignment_seconds;
    solve_seconds(i) = data.timings.solve_seconds;
    total_seconds(i) = data.timings.total_seconds;
    nan_count(i) = sum(isnan(data.v.meas));
    inf_count(i) = sum(isinf(data.v.meas));
    measurements{i} = data.v.meas;

    counts = tissue_count_map(data.tissue_names, data.tissue_counts);
    soft_tissue(i) = get_count(counts, 'soft_tissue');
    lungs(i) = get_count(counts, 'lungs');
    bones(i) = get_count(counts, 'bones');
    blood(i) = get_count(counts, 'blood');

    if i == n
        metadata.cfg = data.cfg;
    end
end

[surface_mm, order] = sort(surface_mm, 'descend');
existing = existing(order);
surface_faces = surface_faces(order);
nodes = nodes(order);
elems = elems(order);
measurements_count = measurements_count(order);
mesh_seconds = mesh_seconds(order);
assignment_seconds = assignment_seconds(order);
solve_seconds = solve_seconds(order);
total_seconds = total_seconds(order);
nan_count = nan_count(order);
inf_count = inf_count(order);
soft_tissue = soft_tissue(order);
lungs = lungs(order);
bones = bones(order);
blood = blood(order);
measurements = measurements(order);

ref = measurements{end};
rel_l2_vs_finest = zeros(n, 1);
max_abs_vs_finest = zeros(n, 1);
max_rel_vs_finest = zeros(n, 1);
rel_l2_vs_next = nan(n, 1);
for i = 1:n
    err = measurements{i} - ref;
    rel_l2_vs_finest(i) = norm(err) / max(norm(ref), eps);
    max_abs_vs_finest(i) = max(abs(err));
    max_rel_vs_finest(i) = max(abs(err) ./ max(abs(ref), eps));
    if i < n
        next = measurements{i + 1};
        rel_l2_vs_next(i) = norm(measurements{i} - next) / max(norm(next), eps);
    end
end

report = table(surface_mm, surface_faces, nodes, elems, measurements_count, ...
    mesh_seconds, assignment_seconds, solve_seconds, total_seconds, ...
    soft_tissue, lungs, bones, blood, nan_count, inf_count, ...
    rel_l2_vs_finest, max_abs_vs_finest, max_rel_vs_finest, rel_l2_vs_next, existing(:), ...
    'VariableNames', {'surface_mm', 'surface_faces', 'nodes', 'elems', ...
    'measurements', 'mesh_seconds', 'assignment_seconds', 'solve_seconds', ...
    'total_seconds', 'soft_tissue_elems', 'lungs_elems', 'bones_elems', ...
    'blood_elems', 'nan_count', 'inf_count', 'rel_l2_vs_1mm', ...
    'max_abs_vs_1mm', 'max_rel_vs_1mm', 'rel_l2_vs_next_finer', 'result_file'});

metadata.reference_surface_mm = report.surface_mm(end);
metadata.reference_file = report.result_file{end};
metadata.generated_at = char(datetime('now'));
metadata.prepared_half_mm_surface = fullfile('output', 'prepared', ...
    'YAROSLAV_BODY_resampled_0.5mm_meshfix.stl');
end

function plot_measurements(report, measurements, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
hold on;
for i = 1:height(report)
    plot(measurements{i}, 'LineWidth', 1.1, ...
        'DisplayName', sprintf('%g mm', report.surface_mm(i)));
end
grid on;
xlabel('Measurement index');
ylabel('Transfer impedance, Ohm-equivalent');
title('Forward EIT transfer impedance by BODY surface resolution');
legend('Location', 'best');
exportgraphics(fig, fullfile(out_dir, 'measurements_by_resolution.png'), 'Resolution', 180);
close(fig);
end

function plot_errors(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
semilogy(report.surface_mm, max(report.rel_l2_vs_1mm, eps), '-o', 'LineWidth', 1.3);
set(gca, 'XDir', 'reverse');
grid on;
xlabel('BODY surface resolution, mm');
ylabel('Relative L2 error vs 1 mm');
title('Transfer-impedance difference relative to 1 mm BODY model');
exportgraphics(fig, fullfile(out_dir, 'relative_l2_error_vs_1mm.png'), 'Resolution', 180);
close(fig);

fig = figure('Visible', 'off', 'Color', 'w');
plot(report.surface_mm, report.max_abs_vs_1mm, '-o', 'LineWidth', 1.3);
set(gca, 'XDir', 'reverse');
grid on;
xlabel('BODY surface resolution, mm');
ylabel('Max absolute impedance difference');
title('Maximum absolute transfer-impedance difference vs 1 mm');
exportgraphics(fig, fullfile(out_dir, 'max_abs_error_vs_1mm.png'), 'Resolution', 180);
close(fig);
end

function plot_costs(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
bar(report.surface_mm, [report.mesh_seconds, report.assignment_seconds, report.solve_seconds] / 60, 'stacked');
set(gca, 'XDir', 'reverse');
grid on;
xlabel('BODY surface resolution, mm');
ylabel('Runtime, minutes');
title('Runtime by calculation stage');
legend({'Meshing', 'Tissue assignment', 'Forward solve'}, 'Location', 'northwest');
exportgraphics(fig, fullfile(out_dir, 'runtime_by_stage.png'), 'Resolution', 180);
close(fig);
end

function plot_complexity(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
yyaxis left;
plot(report.surface_mm, report.elems, '-o', 'LineWidth', 1.3);
ylabel('Tetrahedra');
yyaxis right;
plot(report.surface_mm, report.surface_faces, '-s', 'LineWidth', 1.3);
ylabel('Boundary faces');
set(gca, 'XDir', 'reverse');
grid on;
xlabel('BODY surface resolution, mm');
title('Mesh complexity by BODY surface resolution');
exportgraphics(fig, fullfile(out_dir, 'mesh_complexity.png'), 'Resolution', 180);
close(fig);
end

function write_markdown_report(report, metadata, out_dir)
filename = fullfile(out_dir, 'body_resolution_report.md');
fid = fopen(filename, 'w', 'n', 'UTF-8');
if fid < 0
    error('Cannot write report: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));

cfg = metadata.cfg;
fprintf(fid, '# Отчёт по влиянию разрешения BODY STL на расчёт импеданса EIT\n\n');
fprintf(fid, 'Сформировано: %s\n\n', metadata.generated_at);
fprintf(fid, '## Что именно считалось\n\n');
fprintf(fid, ['Сравнение выполнено на одной и той же КТ-геометрии с несколькими тканями. ', ...
    'Менялось только разрешение внешней поверхности тела `BODY`. ', ...
    'Внутренние STL-маски тканей во всех расчётах были одинаковыми: лёгкие, кости и кровь. ', ...
    'STL сердца в текущем наборе отсутствовал и считался опциональным.\n\n']);
fprintf(fid, '- Решатель: прямой EIT-расчёт импеданса в MATLAB R2022b + EIDORS 3.12.\n');
fprintf(fid, '- Построение объёмной сетки: Gmsh через EIDORS `gmsh_stl2tet`.\n');
fprintf(fid, '- Назначение тканей: по попаданию центров тетраэдров внутрь STL-масок тканей.\n');
fprintf(fid, '- Источник проводимостей: %s.\n', cfg.itis_database);
fprintf(fid, '- Частота: %.6g Гц.\n', cfg.frequency_hz);
fprintf(fid, '- Электроды: %d электродов, автоматическое кольцо на %.0f%% высоты модели.\n', ...
    cfg.n_electrodes, 100 * cfg.electrode_height_fraction);
fprintf(fid, ['- Схема стимуляции/измерений: соседняя подача тока и вращающиеся соседние измерения через `mk_stim_patterns`.\n', ...
    '- Сравниваемая величина: transfer impedance, то есть напряжение отклика, нормированное на ток возбуждения. При стандартном токе 1 А `v.meas` численно соответствует импедансу.\n']);
fprintf(fid, '- Текущий эталон для численного сравнения: BODY %.3g мм.\n\n', metadata.reference_surface_mm);

fprintf(fid, '## Связь с первым COMSOL-репозиторием\n\n');
fprintf(fid, ['В первом репозитории `ForwardSolutionWithComsoleGeometryExample` считался однородный COMSOL/MATLAB случай на готовой COMSOL-тетра-сетке ', ...
    '`ComsolData/Mesh_simp_Yar_cut.mphtxt`. Скрипт `RecontructUsingComsolGeometry.m` сравнивал MATLAB `v.meas*1e3` с COMSOL-таблицей `Z` и явно подписывал график как `Impedance, Ohm`.\n\n']);
fprintf(fid, 'Параметры исходной COMSOL-сетки:\n\n');
fprintf(fid, '- узлы: 60 543;\n');
fprintf(fid, '- тетраэдры: 342 235;\n');
fprintf(fid, '- граничные треугольники после `freeBoundary`: 12 962;\n');
fprintf(fid, '- габариты модели: 380 x 204 x 135 мм;\n');
fprintf(fid, '- медианное ребро тетраэдров: 2.24 мм;\n');
fprintf(fid, '- среднее ребро тетраэдров: 3.74 мм;\n');
fprintf(fid, '- 90-й процентиль ребра: 8.43 мм;\n');
fprintf(fid, '- медианный эквивалентный размер тетраэдра по объёму `(6V)^(1/3)`: 1.78 мм.\n\n');
fprintf(fid, ['Поэтому исходный COMSOL-расчёт был существенно грубее текущего 1 мм STL/FEM-прогона по числу тетраэдров ', ...
    'и ближе к начальному coarse-диапазону. Это не означает полное геометрическое равенство, потому что COMSOL использовал уже готовую тетра-сетку, ', ...
    'а текущая серия строит тетра-сетку из STL-поверхностей.\n\n']);

fprintf(fid, 'Поверхность BODY 0.5 мм подготовлена и проверена, но полный FEM-расчёт для неё в этот отчёт пока не входит.\n\n');
if isfile(metadata.prepared_half_mm_surface)
    info = dir(metadata.prepared_half_mm_surface);
    fprintf(fid, '- Подготовленная BODY STL 0.5 мм: `%s` (%.2f ГБ).\n\n', ...
        metadata.prepared_half_mm_surface, info.bytes / 1024^3);
end

fprintf(fid, '## Итоговая таблица\n\n');
fprintf(fid, ['| BODY, мм | Граничные треугольники | Узлы | Тетраэдры | Полное время, мин | Решение, мин | ', ...
    'Отн. L2 ошибка к 1 мм | Max abs к 1 мм | NaN | Inf |\n']);
fprintf(fid, '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n');
for i = 1:height(report)
    fprintf(fid, '| %.3g | %d | %d | %d | %.2f | %.2f | %.5g | %.5g | %d | %d |\n', ...
        report.surface_mm(i), report.surface_faces(i), report.nodes(i), report.elems(i), ...
        report.total_seconds(i) / 60, report.solve_seconds(i) / 60, ...
        report.rel_l2_vs_1mm(i), report.max_abs_vs_1mm(i), ...
        report.nan_count(i), report.inf_count(i));
end

fprintf(fid, '\n## Картинки\n\n');
fprintf(fid, '![Расчётный импеданс](measurements_by_resolution.png)\n\n');
fprintf(fid, '![Относительная L2 ошибка](relative_l2_error_vs_1mm.png)\n\n');
fprintf(fid, '![Максимальная абсолютная ошибка](max_abs_error_vs_1mm.png)\n\n');
fprintf(fid, '![Время по этапам](runtime_by_stage.png)\n\n');
fprintf(fid, '![Сложность сетки](mesh_complexity.png)\n\n');

fprintf(fid, '## Интерпретация\n\n');
fprintf(fid, ['Модель BODY 1 мм является самым подробным завершённым прямым расчётом и поэтому используется как текущий численный эталон. ', ...
    'Ошибка монотонно уменьшается при повышении детализации внешней поверхности тела: 6 мм явно грубо, 3 мм всё ещё существенно отличается, ', ...
    '2 мм лучше, но остаётся на уровне нескольких процентов от 1 мм, а 1.5 мм уже заметно ближе к 1 мм.\n\n']);
fprintf(fid, ['Важно: этот отчёт изолирует только влияние разрешения внешней поверхности BODY. ', ...
    'Для полной постановки задачи обоснования упрощения анатомии нужно следующим шагом прогнать такую же серию для всех STL-элементов: ', ...
    'BODY, лёгких, костей и крови. Текущий отчёт не проверяет упрощение внутренних STL-масок и не использует конформные границы тканей внутри объёмной сетки.\n']);
end

function surface_mm = parse_surface_mm(filename)
token = regexp(filename, 'resampled_(\d+(?:\.\d+)?)mm', 'tokens', 'once');
if isempty(token)
    surface_mm = NaN;
else
    surface_mm = str2double(token{1});
end
end

function counts = tissue_count_map(names, values)
counts = containers.Map();
for i = 1:numel(names)
    counts(names{i}) = values(i);
end
end

function value = get_count(counts, name)
if isKey(counts, name)
    value = counts(name);
else
    value = 0;
end
end
