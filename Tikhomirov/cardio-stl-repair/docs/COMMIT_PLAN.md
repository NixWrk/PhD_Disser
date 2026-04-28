# План коммитов: `cardio-stl-repair`

Привязка к этапам из [PORT_PLAN.md](PORT_PLAN.md). Каждый коммит —
атомарный, оставляет репо в зелёном состоянии (`pytest -xvs` проходит
на том, что уже портировано). Стиль сообщений: префикс типа на
латинице, тело — на русском. Тип: `chore` (инфра), `feat` (новый код),
`test`, `docs`, `fix`, `refactor`.

---

## Фаза 0 — инфраструктура

| #   | Тип   | Сообщение                                          | Содержимое                                                                                                                       |
| --- | ----- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 01  | chore | init: pyproject, README, .gitignore                | `pyproject.toml` (numpy, trimesh, scikit-image, scipy; extra `full` и `dev`), `README.md`, `.gitignore`                          |
| 02  | docs  | docs: PORT_PLAN и COMMIT_PLAN                      | Файлы плана (уже есть)                                                                                                           |
| 03  | feat  | feat: каркас пакета cardio_stl_repair              | `src/cardio_stl_repair/__init__.py` с `__version__`; пустой `repair.py` с docstring                                              |
| 04  | test  | test: smoke-тест импорта                           | `tests/test_import.py` — `import cardio_stl_repair` проходит                                                                     |

---

## Фаза 1 — Вокселизация и обратная сборка mesh (этап 1 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 05  | feat  | feat(repair): _voxelize (mesh → bool mask + pitch + origin)     | `mesh.voxelized(pitch).fill().matrix.astype(bool)` с авто-pitch по `target_resolution=256`                                       |
| 06  | test  | test: voxelize куба → объём mask·pitch³ ≈ объёму куба           | Tolerance ±5%                                                                                                                    |
| 07  | feat  | feat(repair): _mask_to_mesh (gaussian → marching_cubes → Trimesh) | `gaussian_filter(sigma=1)` + `skimage.measure.marching_cubes(level=0.5)` + `trimesh.Trimesh(...)`. Применить scale (pitch) и origin |
| 08  | test  | test: round-trip куб → vox → mesh → объём ≈ оригиналу           | Tolerance ±5%; bbox проверяется отдельно                                                                                         |
| 09  | feat  | feat(repair): _fit_to_bbox (scale + translate в bbox оригинала) | Прямое повторение Mathematica `ScalingTransform`+`TranslationTransform` с `np.round(scale, 3)`                                   |
| 10  | test  | test: bbox preservation (icosphere → vox → fit_to_bbox)         | `mesh.bounds` совпадает с целевыми до 0.001                                                                                      |

---

## Фаза 2 — Морфология (этап 2 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 11  | feat  | feat(repair): _close_per_slice (2D closing по axis)             | Итерация по слоям, `binary_closing(slice, footprint=disk(r))`                                                                    |
| 12  | feat  | feat(repair): _close_3d (3D closing с ball footprint)           | `binary_closing(mask, footprint=ball(r))` или эквивалент через `scipy.ndimage`                                                   |
| 13  | test  | test: «бублик с дырой» → close_3d → дыра закрывается            | Чек: после closing количество True-вокселей выросло (контролируемо)                                                              |
| 14  | test  | test: per_slice vs 3d на одном входе                            | Оба возвращают замкнутые маски, объёмы близки (±15%)                                                                             |

---

## Фаза 3 — Починка mesh (этап 3 плана)

| #   | Тип   | Сообщение                                                                    | Содержимое                                                                                                                       |
| --- | ----- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 15  | feat  | feat(repair): @dataclass HeartRepairResult                                   | `src`, `repaired`, `closed_mask`, `scale`                                                                                        |
| 16  | feat  | feat(repair): heart_repair_test(mesh, closure_radius, mode='3d'/'per_slice') | Композиция: `_voxelize` → pad → close → `_mask_to_mesh` → `_fit_to_bbox`                                                         |
| 17  | feat  | feat(repair): heart_repair (упрощённая обёртка с radius=20)                  | Возвращает только Trimesh, не intermediate                                                                                        |
| 18  | test  | test: «сфера с дырой» → heart_repair → объём ≈ объёму целой сферы            | Tolerance ±5%; синтетика через `trimesh.boolean.difference(icosphere, cylinder)`                                                 |
| 19  | test  | test: round-trip (load STL → repair → save → load) сохраняет объём           | Tolerance ±0.1%                                                                                                                  |
| 20  | test  | test: после repair `mesh.is_watertight is True`                              | Если фейлит — добавить `mesh.fill_holes()` в реализацию                                                                          |

---

## Фаза 4 — `Task4` верхний уровень (этап 4 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 21  | feat  | feat(repair): @dataclass Task4Result + task4(stl_path, dest_dir, export) | Composes heart_repair_test(r=1) и r=20; export трёх файлов с суффиксами `_src/_fix/_repair`                                      |
| 22  | feat  | feat(repair): task4_export (обёртка над task4)                  | Тонкая обёртка                                                                                                                    |
| 23  | test  | test(task4): синтетика → 3 файла на диске                       | tmp dir + проверка наличия файлов и валидности STL                                                                               |
| 24  | test  | test(task4): repair имеет больше изменений объёма, чем fix      | abs(vol_repair − vol_src) > abs(vol_fix − vol_src)                                                                                |

---

## Фаза 5 — Экспорт точек (этап 5 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 25  | feat  | feat(points): export_points_to_file (STL/Trimesh → CSV)         | `mesh.merge_vertices()` → vertices → `np.unique` (axis=0) → `np.round(decimals=2)` → CSV (`%.2f`, comma, без header)             |
| 26  | test  | test(points): round-trip CSV → np.loadtxt → equal до 0.01       | tmp dir; синтетика — куб с известными вершинами                                                                                  |
| 27  | test  | test(points): дедупликация работает                             | mesh с дубликатами вершин → CSV не содержит повторов                                                                              |

---

## Фаза 6 — Weighted resampling (этап 6 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 28  | feat  | feat(points): w_point(points, first, n, axis)                   | Прямой перенос Wolfram-логики: интервалы → выборка → веса `[n, n-1, ..., 1]` → конкатенация → sort                               |
| 29  | feat  | feat(points): w_point_packet (1..ndiap)                         | Список результатов w_point для разных n                                                                                          |
| 30  | feat  | feat(points): w_point_packet_export                             | Загрузить STL → w_point_packet → CSV-файлы по каждому n                                                                          |
| 31  | test  | test(w_point): размер выхода = sum(1..n) · N_per_bin            | На равномерном распределении проверка количества                                                                                  |
| 32  | test  | test(w_point): точки внутри своего интервала                    | Проверка `Between` для каждого bin                                                                                                |

---

## Фаза 7 — Парсер имён файлов (этап 7 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 33  | feat  | feat(metadata): @dataclass FilenameMetadata + parse_filename    | Подражает `FileNameToList`: split по `_`, признаки «Имеется»/«Не имеется», «Вдох»/«Выдох» по числу <30, «После систолой»/«Перед» |
| 34  | test  | test(metadata): фиксированные имена → ожидаемые поля            | 3-4 примера имён, в т.ч. с `repair`/без и с `sis`/без                                                                            |

---

## Фаза 8 — Пакетные операции (этап 8 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 35  | feat  | feat(batch): get_files_names (из MySystem.m, тривиально)        | `pathlib.Path.glob` — обёртка для совместимости названий                                                                          |
| 36  | feat  | feat(batch): heart_repair_thread, packet_heart_repair           | Single-file и пачкой                                                                                                              |
| 37  | feat  | feat(batch): prepare_stl_batch, stl_to_points_batch             | Папочные операции                                                                                                                  |
| 38  | test  | test(batch): tmp папка с двумя синтетическими STL → batch → выход | Полный smoke                                                                                                                     |

---

## Фаза 9 — Опциональные доработки (этап 9 плана)

| #   | Тип   | Сообщение                                                          | Содержимое                                                                                                                       |
| --- | ----- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| 39  | feat  | feat(repair): pymeshfix-fallback                                   | Soft import, активируется при `mode="pymeshfix"` или авто при сильном изменении объёма                                            |
| 40  | feat  | feat(viz): pyvista preview для task4                               | `verbose=True` показывает src/fix/repair в одном окне; soft import                                                                |
| 41  | test  | test: pymeshfix-fallback на «сфере с большой дырой»                | Без него closing не справился бы; pymeshfix даёт замкнутый mesh                                                                   |

---

## Фаза 10 — Документация и пример (этап 10 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                       |
| --- | ----- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 42  | docs  | docs: USAGE.md с примерами                                      | `docs/USAGE.md` — таблица Wolfram→Python, примеры вызова из Python и CLI                                                         |
| 43  | feat  | feat(cli): __main__.py (минимальный CLI)                        | `python -m cardio_stl_repair task4 path/to/file.stl --dest dest/`                                                                |
| 44  | docs  | docs(notebook): 01_smoke.ipynb                                  | «Сфера с дырой» — vox → close → mesh → repair end-to-end                                                                         |
| 45  | docs  | docs: обновить README с быстрым стартом                         | Раздел «Установка», «Quick start», ссылки на cardio-model-py и 3D_Slicer                                                          |
| 46  | chore | tag: v0.1.0                                                     | После того как все synthetic-тесты зелёные                                                                                       |

---

## Принципы

- **Каждый коммит — зелёный тест.** Реализация и её тест — в одном
  коммите или в двух соседних.
- **Синтетика — единственный источник тестов в фазах 1–9.**
  Реальных STL нет; integration-тесты появятся, когда появятся файлы.
- **Soft-imports.** `pyvista` и `pymeshfix` — внутри функций, не на
  уровне модуля. Это позволяет ставить пакет без них.
- **Размер коммита.** Один dataclass или одна публичная функция + её
  тест = один коммит. Большие функции (`heart_repair_test`) могут
  идти в два коммита: «реализация» и «watertight + edge cases».

## Оценка по объёму

- Фаза 0: 0.5 дня (инфра).
- Фаза 1: 1–2 дня (vox + mesh + bbox fit).
- Фаза 2: 1 день (морфология).
- Фаза 3: 2 дня (починка + watertight).
- Фаза 4: 1 день (Task4).
- Фаза 5: 0.5 дня (export points).
- Фаза 6: 0.5–1 день (w_point).
- Фаза 7: 0.3 дня (filename parser).
- Фаза 8: 0.5 дня (batch).
- Фаза 9: 1 день (опциональные).
- Фаза 10: 0.5 дня (docs).

**Итого ~9–11 рабочих дней.**
