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
- `06_sliding_s1_synthetic_benchmark.ipynb` читает axis-safe v2.1 frozen
  batch-артефакты S1.0,
  проверяет checksums, показывает gate matrix и spatial error maps; текущий verdict
  `FAIL (2/3)`;
- `07_sliding_s11_synthetic_benchmark.ipynb` читает multipattern v3 frozen
  batch-артефакты S1.1, показывает modal coefficients, comparison с исходным ConvexAdam,
  spatial QC и synthetic verdict `PASS (4/4)`;
- `08_sliding_s11_real_development.ipynb` читает frozen real-development и отдельный
  post-hoc failure-diagnosis batch, показывает gate `FAIL (0/6)`, topology/keypoint
  отказ, modal coefficients и spatial QC; expert landmarks не использует;
- `09_sliding_s12_heuristic_screen.ipynb` читает frozen post-hoc model-class screen,
  раздельно показывает correspondence/topology/contact и обосновывает отказ от
  `Gaussian repair + algebraic normal coupling`; expert landmarks не использует;
- `10_piecewise_svf_j10_numeric_gate.ipynb` проверяет checksums frozen analytic batch,
  показывает endpoint/round-trip/Jacobian относительно заранее заданных порогов,
  planar normal/slip invariant и влияние численной границы FOV; КТ не читает;
- `11_piecewise_svf_j11_representation_gate.ipynb` показывает curved-interface
  contact/slip, signed gap/collision maps и различие fixed-normal screen с конечным
  advected contact; КТ и expert landmarks не читает;
- `12_piecewise_svf_j12_truth_preflight.ipynb` показывает, почему exact fields старого
  phantom не проходят новый advected-contact gate 0/3 и блокируют optimizer до замены
  generator; challenge не загружает;
- outputs этих одиннадцати notebooks коммитятся как удобное представление результатов, но
  источником чисел остаются воспроизводимые артефакты в `results/`;
- после обновления batch-артефактов все одиннадцать notebooks нужно выполнить заново и
  проверить,
  что в них нет локальных путей и медицинских идентификаторов.

Bootstrap регистрирует ядро `breathgeom` из `.venv`. Выполнение одного отчёта из корня
репозитория:

~~~powershell
powershell -ExecutionPolicy Bypass -File tools/execute_notebook.ps1 `
  -Notebook notebooks/06_sliding_s1_synthetic_benchmark.ipynb
~~~

Скрипт намеренно вызывает `nbconvert` через Python из `.venv` и явно выбирает kernel.
Запуск `jupyter.exe` без проверки kernel запрещён: на машине разработчика он может найти
команду и ядро из другой установки Python.
