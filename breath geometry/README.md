# Breath Geometry

Исследовательский конвейер для оценки изменения грудной клетки и лёгких между вдохом и
выдохом и для построения ансамбля возможных геометрий выдоха по одной КТ вдоха.

Проект связан с четырёхэлектродными TRKG-моделями из соседнего
`MATLAB_TRKG4_real_subjects`. Итоговая цель — оценивать не только геометрическую ошибку в
миллиметрах, но и её влияние на вычисленный импеданс.

## Текущий статус

Сейчас реализованы воспроизводимый реестр пар, FOV-aware landmark-gated registration
benchmark и полнообъёмные skin-to-lung профили без электродного фильтра. Классический
elastix baseline не проходит gate. Locked ConvexAdam улучшил expert TRE у 13/13 случаев,
но также прошёл полный gate у 0/13, поэтому парные карты толщины остаются заблокированы.
Исторический S1.0 batch на phantom v2.0 superseded из-за осевой сингулярности ground truth.
На исправленном axis-safe v2.1 S1.0 прошёл 2/3: shallow и nominal PASS, deep anisotropic
провален по dense lung field и slip. Новый frozen S1.1 с low-rank tangential model прошёл
multipattern synthetic gate 4/4, включая заранее не запускавшийся longitudinal case.
Следующий этап — development real pairs и затем независимый Gate 1L/1B; до них
подтверждённой оценки изменения мышцы/жира и модели перехода вдох→выдох нет.

Начинать чтение следует здесь:

1. [Текущий подтверждённый статус](docs/PROJECT_STATUS.md)
2. [Определения измеряемых величин](docs/MEASUREMENT_MODEL.md)
3. [Порядок реализации и gate-критерии](docs/ROADMAP.md)
4. [Результат парного registration benchmark](docs/REGISTRATION_BENCHMARK.md)
5. [Выбор и locked benchmark ConvexAdam](docs/CONVEXADAM_BENCHMARK.md)
6. [Журнал решений, проблем и переходов](docs/DECISION_LOG.md)
7. [Спецификация sliding registration S1](docs/SLIDING_REGISTRATION_S1.md)
8. [Аудит текущего notebook](docs/NOTEBOOK_AUDIT.md)
9. [Правила данных и приватности](docs/DATA_PRIVACY.md)

Большие документы [RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md),
[research_and_implementation_plan.md](research_and_implementation_plan.md) и
[benchmark_protocol_inspiration_expiration.md](benchmark_protocol_inspiration_expiration.md)
сохраняются как журнал исследования. Они содержат как полезные наблюдения, так и устаревшие
или отозванные результаты; текущие решения берутся из документов выше.

## Быстрый старт на Windows

~~~powershell
powershell -ExecutionPolicy Bypass -File tools/bootstrap.ps1 -WithGeometry
powershell -ExecutionPolicy Bypass -File tools/install_dcm2niix.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[registration]"
powershell -ExecutionPolicy Bypass -File tools/setup_registration_env.ps1
Copy-Item configs/paths.local.example.yaml configs/paths.local.yaml
.\.venv\Scripts\breathgeom.exe tools status
.\.venv\Scripts\breathgeom.exe project validate --config configs/paths.local.yaml
~~~

`configs/paths.local.yaml` содержит реальные пути, исключён из Git и не должен включать
персональные данные в публикуемые outputs.

## Доступные команды

~~~powershell
.\.venv\Scripts\breathgeom.exe --help
.\.venv\Scripts\breathgeom.exe data list
.\.venv\Scripts\breathgeom.exe data dirlab-inventory "E:\КТ папка\dirlab_copdgene"
.\.venv\Scripts\breathgeom.exe data pairs-manifest `
  --dirlab-root "E:\КТ папка\dirlab_copdgene" `
  --lungct-root "E:\КТ папка\learn2reg_lungct" --checksums
.\.venv\Scripts\breathgeom.exe manifest scan --config configs/paths.local.yaml --output data/interim/manifest.local.csv
.\.venv\Scripts\breathgeom.exe registration benchmark `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset dirlab_copdgene --subject copd1
.\.venv\Scripts\breathgeom.exe registration benchmark `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset learn2reg_lungct --subject LungCT_0001 `
  --method convexadam --params configs/convexadam_locked.json `
  --registration-python .venv-registration/Scripts/python.exe
.\.venv\Scripts\breathgeom.exe profiles extract-pair `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset dirlab_copdgene --subject copd1
.\.venv\Scripts\breathgeom.exe measure wall data/interim/ct.nii.gz --side right --csv data/interim/wall.csv
~~~

PyTorch/ConvexAdam устанавливаются отдельно в `.venv-registration`: основная среда
тестов и анализа не зависит от CUDA. `registration benchmark` сохраняет поле только для
случая, прошедшего все QC-gates.
Флаг `--save-failed-fields` сохраняет проваленное поле только в локальный
`quarantine_NOT_FOR_MEASUREMENT`; profile/measurement pipeline такие поля не принимает.
`profiles extract-pair` всегда может сохранить раздельные однофазные профили, но создаёт
paired deltas только при наличии прошедшего gate поля. Разность независимых однофазных
медиан не считается дыхательным эффектом.

`measure wall` — разведочная skin-to-lung метрика, а не финальная оценка ткани под
электродами. Ограничения и точные определения описаны в
[MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md).

## Notebook

`01_inhale_exhale_inspection.ipynb` служит историческим визуальным QC одного случая.
Выполненные batch-отчёты находятся в `02_registration_benchmark.ipynb` и
`03_whole_body_profiles.ipynb`; locked ConvexAdam разобран в
`04_convexadam_registration_benchmark.ipynb`. Multi-region representation и synthetic
контроли показаны в `05_sliding_phantom.ipynb`; frozen S1.0 FAIL — в
`06_sliding_s1_synthetic_benchmark.ipynb`. Числа batch-анализов всё равно берутся из
`results/`.
Подробная оценка первого notebook — в [NOTEBOOK_AUDIT.md](docs/NOTEBOOK_AUDIT.md).

## Проверки

~~~powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
~~~

Зелёные unit-тесты подтверждают программные инварианты, но не заменяют landmark TRE,
проверку регистрации, сегментации и внешнюю физиологическую валидацию.

## Ключевые правила

- исходные DICOM неизменяемы и не попадают в Git;
- основное анатомическое представление — маски/поля в физическом пространстве КТ, STL
  является производным;
- source/target space, LPS/RAS и единицы всегда записываются явно;
- respiratory, 4DCT и longitudinal данные не смешиваются;
- импедансная дыхательная кривая не считается кривой объёма без калибровки;
- одна КТ вдоха даёт ансамбль возможных выдохов, а не единственное доказанное состояние;
- научный результат должен воспроизводиться batch-командой и иметь provenance/QC.

Полные инструкции для разработчиков и автоматизированных агентов находятся в
[AGENTS.md](AGENTS.md).
