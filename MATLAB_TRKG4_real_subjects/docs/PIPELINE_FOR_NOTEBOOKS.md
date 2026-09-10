# Единый MATLAB/Python-pipeline для notebook-отчётов

## Назначение

Папка `MATLAB_TRKG4_real_subjects/` является единственным активным источником
MATLAB-расчётов для notebook-файлов проекта. Notebook-файлы не должны
воспроизводить FEM-решатель внутри Python: они читают явно именованные
артефакты из `output/`, проверяют контракт и оформляют расчёт как отчёт.

Текущий статус всей цепочки — разработка методики, а не завершённая физическая
или клиническая валидация.

## Точка входа

Из корня папки MATLAB:

```matlab
report = run_pipeline('Profile', 'preflight');
report = run_pipeline('Profile', 'inverse_report');
```

Профиль `inverse_report` последовательно выполняет:

1. проверку MATLAB/EIDORS/Gmsh и обязательных входов;
2. модульные тесты;
3. расчёт трёх фиксированных электродных систем на одной сетке;
4. обратную задачу по многоразмерной серии на вдохе;
5. быстрый PEM-поиск;
6. CEM-уточнение;
7. контроль пересечения электродных патчей.

По умолчанию результаты пересчитываются. Использование уже существующих
файлов разрешается только явно:

```matlab
report = run_pipeline('Profile', 'inverse_report', ...
    'ReuseExistingResults', true);
```

При каждом запуске создаются CSV- и JSON-манифесты стадий. Они фиксируют
исполнителя расчёта, статус, длительность и ожидаемые артефакты, но не заменяют
provenance входных медицинских данных и commit SHA.

## Карта MATLAB → ipynb

| Потребитель | Производитель | Основные артефакты |
|---|---|---|
| `notebooks/TRKG4_inverse_inhale_results.ipynb` | `run_trkg4_electrode_systems` | `nik_trkg4_arms_full_v5_1mm_local2mm.mat`, сводка систем |
| тот же отчёт | `run_trkg4_inverse_inhale` | comparison, fit summary, depth curve, electrode/grid geometry, Jacobian, parameter sensitivity, history |
| тот же отчёт | `run_trkg4_fast_pem_scan` | PEM all/top/validation/summary |
| тот же отчёт | `run_trkg4_fast_cem_refine` | CEM candidates/summary |
| тот же отчёт | `check_trkg4_patch_overlap_v5` | `nik_trkg4_patch_overlap_v5_1mm_local2mm.csv` |
| `Colab Notebooks/40.02_...ipynb` | `run_trkg4_arm_electrode_tests` | arm-electrode summary, patches, contract, MAT и PNG |

Генератор `notebooks/build_trkg4_inverse_report.py` создаёт структуру
отчётного ipynb, но не выполняет MATLAB-расчёт. Текущие файлы v5 из таблицы
выше обязательны. Два нетегированных CSV старой сетки v4 необязательны: при
их отсутствии пропускается только историческое сравнение сеток.

DICOM не является входом MATLAB/FEM-конвейера. Для наложения HU в отчёте
локально задаётся `KALMYKOV_DICOM_DIR`; без этой переменной отчёт продолжает
работу по STL и расчётным CSV. Абсолютный путь к медицинским данным в коде
не хранится.

## Параметрический монтаж на руках

Профиль запускается отдельно:

```matlab
report = run_pipeline('Profile', 'arm_electrode_report', ...
    'ArmPrebuiltMeshFile', 'output/accepted_arm_parameter_mesh.msh', ...
    'ArmMeshTargetSizeMm', 1);
```

Файл сетки обязателен. Историческая принятая сетка поддерживает фиксированный
монтаж, но не произвольное продольное перемещение малых электродов по рукам.
Поэтому pipeline не подменяет сетку молча и не запускает этот профиль без
явно принятого файла.

Диагностический равномерный 1-мм прогон от 31.08.2026 построил сетку на
5 601 998 узлов и 33 866 614 тетраэдров с объёмом 20,1663 л. Геометрический
объём сохранился с расхождением около 0,022%, но были найдены вырожденные и
инвертированный элементы, а размер сетки примерно в 47 раз превысил прежний.
Эта сетка не является принятой расчётной сеткой.

## Подготовка и QC геометрии

Историческая операция равномерного remesh перенесена в общий скрипт:

```powershell
python tools/remesh_stl_uniform.py input.stl output.stl `
  --cell-size-mm 1 --report output/remesh_report.json
```

MeshFix отключён по умолчанию. Он включается только явно и его результат
отклоняется при превышении заданного изменения объёма:

```powershell
python tools/remesh_stl_uniform.py input.stl output.stl `
  --cell-size-mm 2 --repair meshfix --max-volume-change-percent 0.5
```

Проверка объёмной сетки:

```powershell
python tools/check_msh_quality.py output/model.msh `
  --report output/model_mesh_qc.json
```

Без `--allow-defects` команда возвращает ошибку при наличии инвертированных
или вырожденных тетраэдров.

## Активный код и история

- `src/` — точки входа и логика текущей TRKG4-постановки;
- `vendor_stl_eidors/` — используемые проверенные EIDORS/STL-helper-функции;
- `tools/` — активная подготовка и QC STL/MSH;
- `tests/` — модульные тесты;
- `archive/MATLAB_investigation_NIX/` — ранний прототип и старые sweep-скрипты.

Архив не является вторым способом получить актуальные числа. Его старые
16-электродные кольцевые постановки, единицы и sweep-отчёты сохраняются только
для происхождения решений и критического аудита.

## Научное оформление сохранённого перебора в разделе 20

Канонический результат находится в ноутбуке 20.10; отдельный конкурирующий
научный ноутбук не создаётся. `notebooks/TRKG4_saved_scan_scientific.md`
содержит текст раздела с определениями, допущениями, ограничениями и подписями.
`notebooks/build_saved_scan_chapter.py` проверяет контрольные суммы существующих
массивов, повторяет вычисление RMSE и формирует шесть статических научных
рисунков, таблицы и Markdown-ячейки с вложенными изображениями.

Из корня `MATLAB_TRKG4_real_subjects`:

```powershell
python -X utf8 notebooks/build_saved_scan_chapter.py
python -X utf8 notebooks/update_scientific_report.py --output-dir output/exploratory/scan_localization_20260910/scientific_notebook/report_revision
```

Порядок обязателен: сборщик отчёта отказывается включать отсутствующую или
устаревшую главу. Проверяются контрольные суммы её текста, производителя и
ячеек. Существующие 25 вычислительных ячеек и 63 численных выхода сохраняются;
новые решения FEM и исполнение ячеек не выполняются. HTML экспортируется без
входного кода в разметке, изображения включаются непосредственно в документ.

Каталог `scientific_notebook` содержит `chapter_manifest.json`, исходные PNG,
`chapter_cells.json`, полный текст главы для редакционного просмотра и отчёт
проверки сохранности канонического ноутбука. Рисунки используют сохранённые
массивы: текущий четырёхматериальный CEM-перебор не подменяется отдельным
двухтканным опытом. Численные статусы не объявляют экспериментальную валидацию.
