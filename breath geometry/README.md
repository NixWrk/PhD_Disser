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
Но real-development S1.1 дал `0/6`. Следующий frozen screen показал, что Gaussian repair
убирает folding до contact, а одношаговый normal coupling снова его создаёт (`0/5` для
обоих coupled-вариантов). Численная основа следующего joint piecewise-SVF прошла
замороженный J1.0 gate: 4/4 аналитических cases, 8/8 региональных строк, folding 0,
худший endpoint p95 0.000305 мм. J1.1 curved representation также прошёл: positive 2/2
и controls 2/2. Он показал, что почти нулевая fixed-normal ошибка может сопровождаться
39.8% collision, а валидный конечный contact — ненулевой fixed-normal разностью.
Это всё ещё не регистрация КТ. Текущий этап — J1.2 joint optimizer только на synthetic
images. CUDA foundation прошла identity smoke, но старый synthetic generator был
остановлен exact-truth preflight: 0/3 по advected contact, optimizer и challenge не
запускались. Contact-valid v2 доказал непрерывный общий contact с худшим analytic p95
`0.000643 мм`, но его сетки `1.10–1.25 мм` провалили frozen raster gate 0/3:
p95 `0.753–0.848 мм`, coverage `92.2–94.9%`. Optimizer и challenge снова не
запускались. Это исправлено только разрешением grid в v3: exact truth прошла 3/3,
raster p95 `0.640–0.719 мм`, coverage `97.3–98.3%`. Теперь разрешён только frozen
development search 3×3; held-out challenge и Gate 1L/1B остаются закрыты. До них
нет подтверждённой оценки изменения мышцы/жира и модели перехода вдох→выдох.
Frozen search выполнен и дал `0/9 PASS`: contact/topology сохранены, но estimated
tangential slip только `0.003–0.007 мм` при truth `1.327–2.998 мм`. Candidate не
выбран, held-out остаётся закрытым.

Начинать чтение следует здесь:

1. [Текущий подтверждённый статус](docs/PROJECT_STATUS.md)
2. [Определения измеряемых величин](docs/MEASUREMENT_MODEL.md)
3. [Порядок реализации и gate-критерии](docs/ROADMAP.md)
4. [Результат парного registration benchmark](docs/REGISTRATION_BENCHMARK.md)
5. [Выбор и locked benchmark ConvexAdam](docs/CONVEXADAM_BENCHMARK.md)
6. [Журнал решений, проблем и переходов](docs/DECISION_LOG.md)
7. [Спецификация sliding registration S1](docs/SLIDING_REGISTRATION_S1.md)
8. [Результат S1.2 heuristic screen](docs/SLIDING_S12_HEURISTIC_SCREEN.md)
9. [Design joint piecewise-SVF S1.2](docs/SLIDING_S12_JOINT_DESIGN.md)
10. [Численный gate piecewise-SVF J1.0](docs/PIECEWISE_SVF_J10_NUMERIC_GATE.md)
11. [Curved representation gate J1.1](docs/PIECEWISE_SVF_J11_REPRESENTATION_GATE.md)
12. [Протокол J1.2 synthetic development](docs/PIECEWISE_SVF_J12_DEVELOPMENT_PROTOCOL.md)
13. [Контактно-согласованный протокол J1.2 v2](docs/PIECEWISE_SVF_J12_CONTACT_PROTOCOL_V2.md)
14. [Finer-grid протокол J1.2 v3](docs/PIECEWISE_SVF_J12_FINE_GRID_PROTOCOL_V3.md)
15. [Identifiability screen J1.2](docs/PIECEWISE_SVF_J12_IDENTIFIABILITY_SCREEN.md)
16. [Аудит текущего notebook](docs/NOTEBOOK_AUDIT.md)
17. [Правила данных и приватности](docs/DATA_PRIVACY.md)

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
.\.venv\Scripts\breathgeom.exe registration sliding-real-development
.\.venv\Scripts\breathgeom.exe registration sliding-real-diagnose
.\.venv\Scripts\breathgeom.exe registration sliding-s12-heuristic-screen
.\.venv\Scripts\breathgeom.exe registration piecewise-svf-j10-numeric
.\.venv\Scripts\breathgeom.exe registration piecewise-svf-j11-representation
.\.venv\Scripts\breathgeom.exe registration piecewise-svf-j12-preflight
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
`06_sliding_s1_synthetic_benchmark.ipynb`, synthetic S1.1 PASS — в
`07_sliding_s11_synthetic_benchmark.ipynb`, а real-development S1.1 FAIL —
в `08_sliding_s11_real_development.ipynb`. Отказ простого S1.2 repair/coupling класса
показан в `09_sliding_s12_heuristic_screen.ipynb`; численный J1.0 piecewise-SVF —
в `10_piecewise_svf_j10_numeric_gate.ipynb`, curved representation J1.1 —
в `11_piecewise_svf_j11_representation_gate.ipynb`, блокирующий J1.2 truth preflight —
в `12_piecewise_svf_j12_truth_preflight.ipynb`, а отдельный continuous/raster preflight
contact-valid v2 — в `13_piecewise_svf_j12_contact_preflight_v2.ipynb`. Finer-grid v3
PASS показан в `14_piecewise_svf_j12_contact_preflight_v3.ipynb`, development FAIL —
в `15_piecewise_svf_j12_contact_development_v3.ipynb`. Числа batch-анализов всё равно
берутся из `results/`.
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
