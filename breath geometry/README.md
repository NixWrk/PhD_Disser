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

- [Литературный обзор](literature_review_inspiration_to_expiration.md)
- [Протокол benchmark](benchmark_protocol_inspiration_expiration.md)
- [Мастер-план исследования и разработки](research_and_implementation_plan.md)
- [Окружение и внешние инструменты](docs/ENVIRONMENT.md)
- [Политика данных и псевдонимизации](docs/DATA_PRIVACY.md)
- [Предварительный статус инвентаризации КТ](docs/DATA_INVENTORY_STATUS.md)
- [План ближайшей реализации](docs/NEXT_STEPS.md)

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

## Главные правила

1. Обучение моделей в проекте не проводится: только inference готовых моделей и их валидация. Если обучение необходимо, оно оформляется отдельной задачей и выполняется на открытых данных, а не на этих шести людях.
2. Исходные DICOM неизменяемы.
3. В Git нет DICOM, STL, NIfTI, архивов и персональных идентификаторов.
4. Маски в пространстве КТ являются основным анатомическим представлением; STL — производный формат.
5. Все преобразования имеют явно заданные source/target space и единицы миллиметров.
6. Выдох тестового человека недоступен P0/P0V prediction-коду.
