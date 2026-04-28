# cardio-stl-repair

Python-порт модуля
[`Project/RFFI.m`](../WolframMath-master/Kernel/Project/RFFI.m) из
репозитория `WolframMath-master`. Назначение: автоматическая починка
триангулированных STL-сеток сердца (закрытие дыр через морфологическое
замыкание + повторное извлечение поверхности через marching cubes/squares),
взвешенное прореживание точек, экспорт в CSV.

## Почему отдельный репозиторий

Модуль `RFFI.m` функционально независим от остального
`WolframMath-master`: единственная внешняя зависимость —
`GetFilesNames` из `utils/MySystem.m` (тривиальный обход директории,
портируется одной строкой). Тематически это другая задача:
3D-обработка сеток, а не радиоимпедансная кардиография. Поэтому порт
живёт отдельно от
[`cardio-model-py/`](../cardio-model-py/).

## Статус

**В разработке.** Реализация ведётся по плану, см.
[`docs/PORT_PLAN.md`](docs/PORT_PLAN.md).

## Установка (когда будет готово)

```bash
pip install -e .[dev]
```

Все зависимости (`trimesh`, `scikit-image`, `scipy`, `numpy`) — основные;
опциональные (`pyvista` для visualization, `pymeshfix` для fallback) —
в extra `[full]`.

## Соседние проекты

- [`cardio-model-py/`](../cardio-model-py/) — порт остального
  Wolfram-кода (физические модели, объёмы, систола, импеданс).
- [`../3D_Slicer/`](../../3D_Slicer/) — параллельный pipeline DICOM →
  сегментация → COMSOL. По смыслу `cardio-stl-repair` встаёт между
  «сегментацией» и «comsol_ready»: чинит дыры и сглаживает mesh
  перед FEM-симуляцией.
