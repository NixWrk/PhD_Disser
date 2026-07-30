# Breath Geometry

Исследовательский конвейер для оценки изменения грудной клетки и лёгких между вдохом и
выдохом и для построения ансамбля возможных геометрий выдоха по одной КТ вдоха.

Проект связан с четырёхэлектродными TRKG-моделями из соседнего
`MATLAB_TRKG4_real_subjects`. Итоговая цель — оценивать не только геометрическую ошибку в
миллиметрах, но и её влияние на вычисленный импеданс.

## Текущий статус

Сейчас реализованы воспроизводимый реестр пар, landmark-gated registration benchmark и
полнообъёмные skin-to-lung профили без электродного фильтра. Классический elastix baseline
на `copd1` улучшает соответствие, но не проходит gate; подтверждённой оценки изменения
мышцы/жира и модели перехода вдох→выдох пока нет.

Начинать чтение следует здесь:

1. [Текущий подтверждённый статус](docs/PROJECT_STATUS.md)
2. [Определения измеряемых величин](docs/MEASUREMENT_MODEL.md)
3. [Порядок реализации и gate-критерии](docs/ROADMAP.md)
4. [Аудит текущего notebook](docs/NOTEBOOK_AUDIT.md)
5. [Правила данных и приватности](docs/DATA_PRIVACY.md)

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
.\.venv\Scripts\breathgeom.exe profiles extract-pair `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset dirlab_copdgene --subject copd1
.\.venv\Scripts\breathgeom.exe measure wall data/interim/ct.nii.gz --side right --csv data/interim/wall.csv
~~~

`registration benchmark` сохраняет поле только для случая, прошедшего все QC-gates.
`profiles extract-pair` всегда может сохранить раздельные однофазные профили, но создаёт
paired deltas только при наличии прошедшего gate поля. Разность независимых однофазных
медиан не считается дыхательным эффектом.

`measure wall` — разведочная skin-to-lung метрика, а не финальная оценка ткани под
электродами. Ограничения и точные определения описаны в
[MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md).

## Notebook

[01_inhale_exhale_inspection.ipynb](notebooks/01_inhale_exhale_inspection.ipynb) служит
визуальным QC одного случая DIR-Lab. Он не является воспроизводимым benchmark и не должен
быть источником итоговых чисел. Подробная оценка — в
[NOTEBOOK_AUDIT.md](docs/NOTEBOOK_AUDIT.md).

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
