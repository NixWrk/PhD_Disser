function report = generate_all_anatomy_resolution_report()
%GENERATE_ALL_ANATOMY_RESOLUTION_REPORT Save tables, plots, and Markdown.

out_dir = fullfile('output', 'all_anatomy_resolution_sweep');
if ~exist(out_dir, 'dir')
    error('All-anatomy sweep directory not found: %s', out_dir);
end

[report, measurements, metadata] = collect_all_anatomy_results(out_dir);
writetable(report, fullfile(out_dir, 'all_anatomy_resolution_report_table.csv'));
save(fullfile(out_dir, 'all_anatomy_resolution_report.mat'), ...
    'report', 'measurements', 'metadata');

plot_measurements(report, measurements, out_dir);
plot_errors(report, out_dir);
plot_costs(report, out_dir);
plot_complexity(report, out_dir);
plot_tissue_counts(report, out_dir);
write_markdown_report(report, metadata, out_dir);

disp(report);
fprintf('Saved report to %s\n', fullfile(out_dir, 'all_anatomy_resolution_report.md'));
end

function [report, measurements, metadata] = collect_all_anatomy_results(out_dir)
files = dir(fullfile(out_dir, 'forward_all_anatomy_*mm.mat'));
if isempty(files)
    error('No all-anatomy result files found in %s.', out_dir);
end

n = numel(files);
surface_mm = zeros(n, 1);
body_faces = zeros(n, 1);
lungs_faces = zeros(n, 1);
bones_faces = zeros(n, 1);
blood_faces = zeros(n, 1);
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
result_file = cell(n, 1);

metadata = struct();
for i = 1:n
    result_file{i} = fullfile(files(i).folder, files(i).name);
    data = load(result_file{i}, 'cfg', 'fmdl', 'v', 'timings', ...
        'tissue_names', 'tissue_counts');

    surface_mm(i) = parse_surface_mm(files(i).name);
    body_faces(i) = count_stl_faces(data.cfg.body_stl);
    lungs_faces(i) = count_tissue_faces(data.cfg.tissues, 'lungs');
    bones_faces(i) = count_tissue_faces(data.cfg.tissues, 'bones');
    blood_faces(i) = count_tissue_faces(data.cfg.tissues, 'blood');
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

    if ~isfield(metadata, 'cfg')
        metadata.cfg = data.cfg;
    end
end

[surface_mm, order] = sort(surface_mm, 'descend');
body_faces = body_faces(order);
lungs_faces = lungs_faces(order);
bones_faces = bones_faces(order);
blood_faces = blood_faces(order);
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
result_file = result_file(order);

ref = measurements{end};
rel_l2_vs_2mm = zeros(n, 1);
max_abs_vs_2mm = zeros(n, 1);
max_rel_vs_2mm = zeros(n, 1);
rel_l2_vs_next_finer = nan(n, 1);
for i = 1:n
    err = measurements{i} - ref;
    rel_l2_vs_2mm(i) = norm(err) / max(norm(ref), eps);
    max_abs_vs_2mm(i) = max(abs(err));
    max_rel_vs_2mm(i) = max(abs(err) ./ max(abs(ref), eps));
    if i < n
        next = measurements{i + 1};
        rel_l2_vs_next_finer(i) = norm(measurements{i} - next) / max(norm(next), eps);
    end
end

report = table(surface_mm, body_faces, lungs_faces, bones_faces, blood_faces, ...
    nodes, elems, measurements_count, mesh_seconds, assignment_seconds, ...
    solve_seconds, total_seconds, soft_tissue, lungs, bones, blood, ...
    nan_count, inf_count, rel_l2_vs_2mm, max_abs_vs_2mm, ...
    max_rel_vs_2mm, rel_l2_vs_next_finer, result_file(:), ...
    'VariableNames', {'surface_mm', 'body_faces', 'lungs_faces', ...
    'bones_faces', 'blood_faces', 'nodes', 'elems', 'measurements', ...
    'mesh_seconds', 'assignment_seconds', 'solve_seconds', ...
    'total_seconds', 'soft_tissue_elems', 'lungs_elems', ...
    'bones_elems', 'blood_elems', 'nan_count', 'inf_count', ...
    'rel_l2_vs_2mm', 'max_abs_vs_2mm', 'max_rel_vs_2mm', ...
    'rel_l2_vs_next_finer', 'result_file'});

metadata.reference_surface_mm = report.surface_mm(end);
metadata.reference_file = report.result_file{end};
metadata.generated_at = char(datetime('now'));
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
title('Forward EIT transfer impedance by full-anatomy STL resolution');
legend('Location', 'best');
exportgraphics(fig, fullfile(out_dir, 'measurements_by_resolution.png'), 'Resolution', 180);
close(fig);
end

function plot_errors(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
semilogy(report.surface_mm, max(report.rel_l2_vs_2mm, eps), '-o', 'LineWidth', 1.3);
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Matched STL surface resolution, mm');
ylabel('Relative L2 error vs 2 mm');
title('Transfer-impedance difference relative to 2 mm full-anatomy model');
exportgraphics(fig, fullfile(out_dir, 'relative_l2_error_vs_2mm.png'), 'Resolution', 180);
close(fig);

fig = figure('Visible', 'off', 'Color', 'w');
plot(report.surface_mm, report.max_abs_vs_2mm, '-o', 'LineWidth', 1.3);
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Matched STL surface resolution, mm');
ylabel('Max absolute impedance difference');
title('Maximum absolute transfer-impedance difference vs 2 mm');
exportgraphics(fig, fullfile(out_dir, 'max_abs_error_vs_2mm.png'), 'Resolution', 180);
close(fig);
end

function plot_costs(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
bar(report.surface_mm, [report.mesh_seconds, report.assignment_seconds, report.solve_seconds] / 60, 'stacked');
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Matched STL surface resolution, mm');
ylabel('Runtime, minutes');
title('Runtime by calculation stage');
legend({'Mesh load/build', 'Tissue assignment', 'Forward solve'}, 'Location', 'northwest');
exportgraphics(fig, fullfile(out_dir, 'runtime_by_stage.png'), 'Resolution', 180);
close(fig);
end

function plot_complexity(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
yyaxis left;
plot(report.surface_mm, report.elems, '-o', 'LineWidth', 1.3);
ylabel('Tetrahedra');
yyaxis right;
plot(report.surface_mm, report.body_faces + report.lungs_faces + ...
    report.bones_faces + report.blood_faces, '-s', 'LineWidth', 1.3);
ylabel('Total STL faces');
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Matched STL surface resolution, mm');
title('Volume mesh and input STL complexity');
exportgraphics(fig, fullfile(out_dir, 'mesh_and_surface_complexity.png'), 'Resolution', 180);
close(fig);
end

function plot_tissue_counts(report, out_dir)
fig = figure('Visible', 'off', 'Color', 'w');
bar(report.surface_mm, [report.soft_tissue_elems, report.lungs_elems, ...
    report.bones_elems, report.blood_elems], 'stacked');
set(gca, 'XDir', 'reverse');
grid on;
xlabel('Matched STL surface resolution, mm');
ylabel('Tetrahedra assigned to tissue');
title('Assigned tissue element counts');
legend({'Soft tissue', 'Lungs', 'Bones', 'Blood'}, 'Location', 'northwest');
exportgraphics(fig, fullfile(out_dir, 'tissue_element_counts.png'), 'Resolution', 180);
close(fig);
end

function write_markdown_report(report, metadata, out_dir)
filename = fullfile(out_dir, 'all_anatomy_resolution_report.md');
fid = fopen(filename, 'w', 'n', 'UTF-8');
if fid < 0
    error('Cannot write report: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

cfg = metadata.cfg;
fprintf(fid, '# Отчет по влиянию разрешения всех STL-тканей на расчет импеданса EIT\n\n');
fprintf(fid, 'Сформировано: %s\n\n', metadata.generated_at);

fprintf(fid, '## Что считалось\n\n');
fprintf(fid, ['Считалась прямая задача EIT в MATLAB R2022b + EIDORS 3.12. ', ...
    'На каждом уровне разрешения одновременно менялись все доступные STL-поверхности анатомии: ', ...
    'внешнее тело BODY, легкие, кости и кровь. Сердца в текущем наборе STL пока нет, поэтому оно не участвовало в разметке тканей.\n\n']);
fprintf(fid, '- Построение/загрузка объемной сетки: Gmsh через EIDORS `gmsh_stl2tet`.\n');
fprintf(fid, '- Назначение тканей: по попаданию центров тетраэдров внутрь соответствующих STL-масок.\n');
fprintf(fid, '- Проводимости: фиксированные значения из %s на %.6g Гц.\n', cfg.itis_database, cfg.frequency_hz);
fprintf(fid, '- Электроды: %d электродов, кольцо примерно на %.0f%% высоты модели.\n', ...
    cfg.n_electrodes, 100 * cfg.electrode_height_fraction);
fprintf(fid, '- Стимуляция/измерения: соседняя подача тока и соседние измерения через `mk_stim_patterns`.\n');
fprintf(fid, ['- Сравниваемая величина: transfer impedance. При стандартном токе 1 А массив `v.meas` ', ...
    'численно соответствует импедансному отклику.\n']);
fprintf(fid, '- Текущий численный эталон в этой серии: %.3g мм, то есть самый подробный завершенный полный анатомический прогон.\n\n', ...
    metadata.reference_surface_mm);

fprintf(fid, '## Важное ограничение\n\n');
fprintf(fid, ['Этот отчет уже проверяет упрощение всей текущей анатомии, а не только внешнего BODY. ', ...
    'Но эталон пока 2 мм, потому что полная серия 1 мм или 0.5 мм для всех тканей еще не завершена. ', ...
    'Значит выводы 6/4/3 мм относительно 2 мм полезны как первая численная оценка, но финальное обоснование ', ...
    'для медицински значимой точности должно опираться на более подробный полный анатомический прогон.\n\n']);

fprintf(fid, '## Итоговая таблица\n\n');
fprintf(fid, ['| STL, мм | BODY faces | LUNGS faces | BONES faces | BLOOD faces | Узлы | Тетраэдры | ', ...
    'Время, мин | Решение, мин | Отн. L2 к 2 мм | Max abs к 2 мм | NaN | Inf |\n']);
fprintf(fid, '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n');
for i = 1:height(report)
    fprintf(fid, '| %.3g | %d | %d | %d | %d | %d | %d | %.2f | %.2f | %.5g | %.5g | %d | %d |\n', ...
        report.surface_mm(i), report.body_faces(i), report.lungs_faces(i), ...
        report.bones_faces(i), report.blood_faces(i), report.nodes(i), ...
        report.elems(i), report.total_seconds(i) / 60, ...
        report.solve_seconds(i) / 60, report.rel_l2_vs_2mm(i), ...
        report.max_abs_vs_2mm(i), report.nan_count(i), report.inf_count(i));
end

fprintf(fid, '\n## Разметка тканей\n\n');
fprintf(fid, '| STL, мм | Soft tissue elems | Lungs elems | Bones elems | Blood elems |\n');
fprintf(fid, '|---:|---:|---:|---:|---:|\n');
for i = 1:height(report)
    fprintf(fid, '| %.3g | %d | %d | %d | %d |\n', report.surface_mm(i), ...
        report.soft_tissue_elems(i), report.lungs_elems(i), ...
        report.bones_elems(i), report.blood_elems(i));
end

fprintf(fid, '\n## Картинки\n\n');
fprintf(fid, '![Расчетный импеданс](measurements_by_resolution.png)\n\n');
fprintf(fid, '![Относительная L2 ошибка](relative_l2_error_vs_2mm.png)\n\n');
fprintf(fid, '![Максимальная абсолютная ошибка](max_abs_error_vs_2mm.png)\n\n');
fprintf(fid, '![Время по этапам](runtime_by_stage.png)\n\n');
fprintf(fid, '![Сложность сетки и STL](mesh_and_surface_complexity.png)\n\n');
fprintf(fid, '![Число элементов тканей](tissue_element_counts.png)\n\n');

fprintf(fid, '## Интерпретация\n\n');
fprintf(fid, ['На этой серии видно именно влияние согласованного упрощения всех имеющихся STL-масок. ', ...
    'Грубые уровни 6 и 4 мм сравниваются с 2 мм по 208 импедансным измерениям одной и той же схемы электродов. ', ...
    'Если относительная L2 ошибка и максимальная абсолютная ошибка окажутся недостаточными для выбранного допуска, ', ...
    'следующий шаг - запуск 1.5/1 мм полной анатомии и перенос эталона с 2 мм на самый подробный завершенный расчет.\n']);
end

function surface_mm = parse_surface_mm(filename)
token = regexp(filename, '_(\d+(?:\.\d+)?)mm\.mat$', 'tokens', 'once');
if isempty(token)
    surface_mm = NaN;
else
    surface_mm = str2double(token{1});
end
end

function n_faces = count_tissue_faces(tissues, name)
n_faces = 0;
for i = 1:numel(tissues)
    if strcmp(tissues(i).name, name) && isfield(tissues(i), 'file') && isfile(tissues(i).file)
        n_faces = count_stl_faces(tissues(i).file);
        return;
    end
end
end

function n_faces = count_stl_faces(filename)
surface = stl_read(filename);
n_faces = size(surface.boundary, 1);
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
