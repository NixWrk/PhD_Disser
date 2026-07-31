# Notebooks

Notebook в этом каталоге — интерактивный QC/отчёт, а не единственное место реализации
алгоритма или хранения результата.

Правила:

- переиспользуемая логика находится в `src/breathgeom` и покрывается тестами;
- входные пути и case IDs поступают из конфига/manifest;
- количественные таблицы строятся из batch-артефактов в `results/`;
- Plotly HTML и другие тяжёлые outputs генерируются локально и не коммитятся;
- после изменения библиотечного кода старые cell outputs считаются устаревшими;
- notebook не показывает локальные phase differences, если registration gate не пройден;
- финальная ячейка сообщает QC verdict и перечень допустимых выводов.

Текущий notebook `01_inhale_exhale_inspection.ipynb` — исторический прототип для `copd1`.
Его подробный аудит находится в `../docs/NOTEBOOK_AUDIT.md`.

## Отчётные notebooks

- `02_registration_benchmark.ipynb` читает subject-level batch-таблицы регистрации,
  показывает expert TRE, геометрические метрики, причины провала gate и provenance;
- `03_whole_body_profiles.ipynb` показывает QC лучей по всей поверхности лёгких,
  независимые распределения полной толщины, жира, мышц и кости, региональные таблицы
  и пространственную карту;
- `04_convexadam_registration_benchmark.ipynb` показывает blind development selection,
  locked сравнение с Elastix, матрицу gate, региональный TRE и provenance;
- `05_sliding_phantom.ipynb` проверяет multi-region представление плеврального скольжения,
  региональный Jacobian и два отрицательных контроля; это representation gate, а не
  benchmark алгоритма регистрации;
- outputs этих четырёх notebooks коммитятся как удобное представление результатов, но
  источником чисел остаются воспроизводимые артефакты в `results/`;
- после обновления batch-артефактов все четыре notebooks нужно выполнить заново и проверить,
  что в них нет локальных путей и медицинских идентификаторов.

Bootstrap регистрирует ядро `breathgeom` из `.venv`. Выполнение одного отчёта из корня
репозитория:

~~~powershell
powershell -ExecutionPolicy Bypass -File tools/execute_notebook.ps1 `
  -Notebook notebooks/05_sliding_phantom.ipynb
~~~

Скрипт намеренно вызывает `nbconvert` через Python из `.venv` и явно выбирает kernel.
Запуск `jupyter.exe` без проверки kernel запрещён: на машине разработчика он может найти
команду и ядро из другой установки Python.
