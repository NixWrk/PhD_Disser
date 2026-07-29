# Breath Geometry

Исследовательский конвейер для построения индивидуальной анатомии максимального выдоха по КТ глубокого вдоха.

## Статус

Проект находится на этапе Gate 0–1:

- исследовательский протокол зафиксирован;
- создан независимый Git-репозиторий;
- готовится воспроизводимое Python-окружение;
- архивы старых КТ извлекаются во внешнем read-only каталоге;
- медицинские данные и идентификаторы людей не коммитятся.

## Документация

- [Научный отчёт: цели, обзор, открытые данные, осуществимость](docs/RESEARCH_REPORT.md)
- [Литературный обзор](literature_review_inspiration_to_expiration.md)
- [Протокол benchmark](benchmark_protocol_inspiration_expiration.md)
- [Мастер-план исследования и разработки](research_and_implementation_plan.md)
- [Окружение и внешние инструменты](docs/ENVIRONMENT.md)
- [Политика данных и псевдонимизации](docs/DATA_PRIVACY.md)
- [Предварительный статус инвентаризации КТ](docs/DATA_INVENTORY_STATUS.md)
- [План ближайшей реализации](docs/NEXT_STEPS.md)
- [Открытые КТ-датасеты](docs/OPEN_DATASETS.md)

## Быстрый старт на Windows

~~~powershell
powershell -ExecutionPolicy Bypass -File tools/bootstrap.ps1 -WithGeometry
powershell -ExecutionPolicy Bypass -File tools/install_dcm2niix.ps1
.\.venv\Scripts\breathgeom.exe tools status
.\.venv\Scripts\breathgeom.exe project validate --config configs/paths.local.yaml
~~~

Локальный файл configs/paths.local.yaml создаётся из примера и исключён из Git.

## Первый безопасный DICOM-скан

Scanner читает только заголовки DICOM, не загружает pixel data и не экспортирует PatientName, PatientID, дату рождения или accession number.
Поддерживаются как классические однофайловые срезы, так и Enhanced Multi-frame CT с геометрией в Shared Functional Groups.

~~~powershell
.\.venv\Scripts\breathgeom.exe manifest scan --config configs/paths.local.yaml --output data/interim/manifest.local.csv --max-files 10000
~~~

Пока распаковка продолжается, такой результат считается предварительным. Финальный manifest строится только после стабилизации состава файлов.

## Осмотр пар вдох/выдох

Три ошибки подряд прошли ruff, mypy и весь набор тестов и были видны только глазами, поэтому
осмотр — обязательный шаг перед любым числом, а не иллюстрация.

~~~powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/01_inhale_exhale_inspection.ipynb
~~~

[notebooks/01_inhale_exhale_inspection.ipynb](notebooks/01_inhale_exhale_inspection.ipynb)
проверяет оси по позвонку, гладкость сечений, компоненты маски и совмещение, а затем
показывает смещение границ и компартментов. Вращаемые модели дополнительно сохраняются в
`notebooks/figures/*.html` и открываются любым браузером: JupyterLab и VS Code не исполняют
скрипты в HTML-выводе, поэтому одного встроенного рисунка недостаточно.

## Толщина стенки под электродной решёткой

Кратчайший путь кожа → лёгкое в боковом секторе — это `h` двуслойной импедансной модели. Том приводится к RAS+ перед измерением, поэтому зеркальный переворот не может пройти незамеченным.

~~~powershell
.\.venv\Scripts\breathgeom.exe measure wall data/interim/ct.nii.gz --side right --csv data/interim/wall.csv
~~~

Разделение жир/мышца по HU достоверно только на нативных сериях: контраст сдвигает мышечное окно.

## Главные правила

1. Обучение моделей в проекте не проводится: только inference готовых моделей и их валидация. Если обучение необходимо, оно оформляется отдельной задачей и выполняется на открытых данных, а не на этих шести людях.
2. Исходные DICOM неизменяемы.
3. В Git нет DICOM, STL, NIfTI и архивов. Идентифицирующие сведения в тексте допускаются, пока репозиторий однопользовательский и без remote — условия и порядок обратного перехода в [DATA_PRIVACY.md](docs/DATA_PRIVACY.md).
4. Маски в пространстве КТ являются основным анатомическим представлением; STL — производный формат.
5. Все преобразования имеют явно заданные source/target space и единицы миллиметров.
6. Выдох тестового человека недоступен P0/P0V prediction-коду.
