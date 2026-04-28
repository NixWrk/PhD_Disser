# План: порт `Project/RFFI.m` → пакет `cardio_stl_repair`

## Контекст

Этот репозиторий (`cardio-stl-repair/`) — Python-порт модуля
[`Project/RFFI.m`](../../WolframMath-master/Kernel/Project/RFFI.m) из
проекта `WolframMath-master`. Назначение: автоматическая обработка
триангулированных STL-сеток сердца (закрытие дыр через морфологическое
замыкание + повторное извлечение поверхности через marching squares/
cubes) с последующим экспортом точек в CSV.

### Почему отдельный репозиторий

`RFFI.m` функционально независим от остального `WolframMath-master`:
- ни один другой модуль `Kernel/**/*.m` не импортирует символы из
  `RFFI.m` (проверено `grep` по символам);
- сам `RFFI.m` не использует `Needs[...]` ни для какого другого модуля;
- единственная внешняя зависимость — функция `GetFilesNames` из
  `utils/MySystem.m` (обход директории, тривиально портируется одной
  строкой через `pathlib.Path.glob`).

Тематически это другая задача — 3D-обработка сеток, а не
радиоимпедансная кардиография. Поэтому порт живёт отдельно от
[`cardio-model-py/`](../../cardio-model-py/) и имеет свой `pyproject.toml`,
свою цепочку тестов, свои зависимости (`trimesh`, `scikit-image`).

### Соседние проекты

- [`cardio-model-py/`](../../cardio-model-py/) — порт остального
  Wolfram-кода (физические модели, finders, объёмы, систола, импеданс).
- [`../../3D_Slicer/`](../../../3D_Slicer/) — параллельный Python-проект
  Калмыкова: pipeline DICOM → сегментация → COMSOL с venv на
  torch/torchvision/dicom2nifti/gdcm. Реализационных `.py`-файлов в
  дереве нет (видимо, перенесены или ещё не написаны), но папочная
  структура (`raw_dicom/`, `preprocessed/`, `segmentations/{auto,
  corrected, final_labelmap}/`, `comsol_ready/`) показывает, что
  сосед закрывает «вход» (DICOM→сегментация→воксели) и «выход»
  (готовые модели для COMSOL). `cardio-stl-repair` по смыслу встаёт
  между этими двумя стадиями — «доводка» triangle mesh перед FEM.
  Это не значит, что порт обязан интегрироваться с `3D_Slicer` прямо
  сейчас, но имена/форматы лучше не вступать с ним в конфликт.

Тестовых STL-файлов в обоих деревьях нет. Валидация пойдёт на
синтетике + (когда появятся) реальные данные.

## Полный анализ RFFI.m

### Иерархия и точки входа

```
Точки входа:
  PrepareStl(srcPath, targetPath)            ← пакетная обработка папки
  StlToPointsFile(srcPath, targetPath)       ← пакетный экспорт точек
  PacketHeartRepair(listOfPathAndNames, dest) ← альтернативный пакетный режим
  Task4Export(pairPathName, targetpath)      ← одиночная: «всё включено»
  WPointPacketExport(pathstl, ...)           ← одиночная: weighted-resample

Главные функции (single-mesh):
  Task4(pairPathName, pathDest, export)
    └── HeartRepairTest(stl, repk=1)[[4]]    ← возвращает sslice (закрытые слои)
        ├── RegionImage[stl]                  ← solid воксель
        ├── ImagePad[..., 20, 0]              ← padding нулями
        ├── Image3D[..., "Bit", ViewPoint→Front]
        ├── Image3DSlices                     ← список 2D-картинок
        ├── Map[Closing[#, DiskMatrix[repk]]&, slice]
        └── ImageMesh[Image3D[sslice]]        ← обратная сборка mesh
    └── HeartRepairTest(stl, repk=20)[[4]]    ← то же с большим радиусом
    ├── ImageMesh[Blur[Image3D[sslice]],
    │             Method→"MarchingSquares",
    │             Boxed→True, ViewPoint→Back] ← дополнительный Blur+MS
    ├── RegionBounds + ScalingTransform       ← scale до bbox оригинала
    └── TranslationTransform                  ← сдвиг до bbox оригинала

  HeartRepair(stl)  ← упрощённая версия (без Task4-обвязки)
    └── (та же логика, что HeartRepairTest, но фиксированный disk=20)

  HeartRepairThread(path, dest, namepattern, isView)
    ← обёртка над HeartRepair: load STL → repair → export

Утилиты:
  ExportPointsToFile(pathstl, target, namesuffix)
    ← Import[..., "GraphicsComplex"][[1]] (вершины) → DeleteDuplicates
      → Round[..., 0.01] → Export как CSV

  WPoint(array, first, n, axe)                  ← weighted resampling
  WPointPacket(array, first, ndiap, axe)        ← набор WPoint на 1..ndiap
  WPointPacketExport(stlPath, target, suffix, {first, ndiap, axe})

  FileNameToList(file, scale, volume)
    ← парсер имени файла в список метаданных:
      [имя, list[1], "Имеется"/"Не имеется", "Вдох"/"Выдох",
       "После систолой"/"Перед систолой", list[3], scale_x, scale_y,
       scale_z, volume]
    ← признаки берёт из частей `_`-разделённого имени и числа list[3]<30
```

### Что именно делает `Task4` (по строкам)

```mathematica
stlsrc = Import[pairPathName[[1]]];                    // load STL

// "FIX" версия (лёгкое замыкание, repk=1):
fixslice = HeartRepairTest[stlsrc, 1][[4]];            // [[4]] = sslice
meshfix  = ImageMesh[Blur[Image3D[fixslice]],
                     Method->"MarchingSquares",
                     Boxed->True, ViewPoint->Back];

bound      = RegionBounds[stlsrc];                     // bbox оригинала
boundfix   = RegionBounds[meshfix];                    // bbox перестроенного
box        = Last[#] - First[#] & /@ bound;            // размеры по осям
boxfix     = Last[#] - First[#] & /@ boundfix;
scalefix   = Round[box/boxfix, 0.001];                 // масштабные коэффициенты

s1            = ScalingTransform[scalefix];
stlfixscale   = TransformedRegion[meshfix, s1];        // масштабированный
boundfixscale = RegionBounds[stlfixscale];
d             = First /@ (boundfixscale - bound);      // смещение
s2            = TranslationTransform[-d];
stlfix        = TransformedRegion[stlfixscale, s2];    // FIT в bbox оригинала

// "REPAIR" версия (тяжёлое замыкание, repk=20):
// (то же самое для repk=20 → stlrepair)

if export:
  Export[<dest>+name+"src.stl", stlsrc];
  Export[<dest>+name+"fix.stl", stlfix];
  Export[<dest>+name+"repair.stl", stlrepair];

return {stlsrc, stlfix, stlrepair}
```

### Поведение Mathematica-операций (тонкие места)

| Операция                                  | Что делает в Mathematica                                                                                | Эквивалент в Python                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `RegionImage[stl]`                        | Solid вокселизация (заполненная) с автоподобранным разрешением. Возвращает `Image3D` бит-глубины 1.    | `mesh.voxelized(pitch).fill().matrix.astype(bool)` (trimesh), pitch вручную.   |
| `ImagePad[img, 20, 0]`                    | Окружить каждую грань 20 пикселями нулей.                                                              | `np.pad(mask, 20, mode='constant', constant_values=0)`                         |
| `Image3D[..., "Bit", ViewPoint->Front]`   | Тип бит-1; `ViewPoint` влияет только на отображение, не на данные.                                      | `mask.astype(bool)`. ViewPoint игнорируется.                                   |
| `Image3DSlices[img]`                      | Список 2D-срезов вдоль оси «глубины» (по умолчанию Z, but axis index = `ImageDimensions`'s last).      | `[mask[:, :, k] for k in range(mask.shape[2])]` (или другой axis — см. ниже).  |
| `Closing[img, DiskMatrix[r]]`             | 2D морф. замыкание 2D-диском радиуса r (для `Image`).                                                   | `skimage.morphology.binary_closing(slice, footprint=disk(r))`                  |
| `Image3D[stack_of_slices]`                | Собрать обратно в 3D.                                                                                  | `np.stack(slices, axis=2)`                                                     |
| `ImageMesh[Image3D[mask]]`                | По умолчанию использует **Dual marching cubes** для бинарного 3D-image.                                | `skimage.measure.marching_cubes(mask.astype(float), level=0.5)`                |
| `ImageMesh[Blur[img], "MarchingSquares"]` | `Blur` без аргументов = Gaussian с радиусом 2; `"MarchingSquares"` для 3D неоднозначно (см. ниже).      | `gaussian_filter(mask.astype(float), sigma=1)` затем marching_cubes.           |
| `RegionBounds[mesh]`                      | `[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`                                                            | `mesh.bounds.T` (trimesh даёт `[[min], [max]]`)                                |
| `ScalingTransform[{sx, sy, sz}]`          | Масштабирование от **начала координат**, не от центра bbox.                                            | `mesh.apply_transform(np.diag([sx, sy, sz, 1]))`                               |
| `TranslationTransform[v]`                 | Сдвиг.                                                                                                  | `mesh.apply_translation(-v)` (после ScalingTransform!)                         |
| `Volume[mesh]`                            | Объём signed (для замкнутых mesh).                                                                      | `mesh.volume` (trimesh)                                                        |
| `Import[..., "GraphicsComplex"][[1]]`     | Список координат вершин mesh.                                                                          | `mesh.vertices`                                                                |
| `Round[res, 0.01]`                        | Round to 0.01.                                                                                          | `np.round(arr, 2)`                                                             |
| `Export[..., "CSV"]`                      | Запись массива как CSV.                                                                                | `np.savetxt(path, arr, delimiter=',', fmt='%.2f')`                             |

### Неоднозначность `Method->"MarchingSquares"` в `ImageMesh[Image3D[...]]`

В документации Mathematica `ImageMesh` для 3D-image обычно использует
`"DualMarchingCubes"`. Опция `"MarchingSquares"` строго говоря —
2D-метод. Скорее всего, автор кода либо:

- (a) хотел поточечный 2D MS на каждом срезе и сборку «лестницей»
  (тогда mesh выйдет с явными ступеньками — характерный вид
  старых STL-починок), или
- (b) Mathematica молча игнорирует невалидную опцию и применяет
  `DualMarchingCubes`.

Это — главная неопределённость порта. План валидации:
сделать **обе** реализации (`mode="3d"` и `mode="per_slice"`) и
сравнить визуально и численно. По умолчанию использовать 3D —
он даёт более гладкую, физически правильную поверхность.

### `RegionImage` — авто-разрешение

В Mathematica `RegionImage[stl]` подбирает разрешение так, чтобы
максимальная сторона была примерно 200–300 вокселей. Эквивалент:
```python
target_resolution = 256
pitch = max(mesh.extents) / target_resolution
```
Если pitch выйдет слишком грубым — добавим параметр.

## Соответствия Wolfram → Python (специфичные для RFFI)

```
RFFI.m функция                  →  Python (cardio_stl_repair)
---------------------------------------------------------------
HeartRepair[stl]                →  heart_repair(mesh, closure_radius=20)
HeartRepairTest[stl, repk]      →  heart_repair_test(mesh, closure_radius=repk)
                                   возвращает dataclass с .src, .repaired,
                                   .closed_slices, .scale
Task4[pair, dest, export]       →  task4(stl_path, dest_dir, *, export=True)
                                   возвращает (src, fix, repair) три Trimesh
Task4Export[pair, target]       →  task4_export(stl_path, target_dir)
HeartRepairThread[...]          →  heart_repair_thread(path, dest, namepattern,
                                   *, view=False)
PacketHeartRepair[list, dest]   →  packet_heart_repair(pairs, dest)
PrepareStl[src, target]         →  prepare_stl_batch(src_folder, target_folder)
StlToPointsFile[src, target]    →  stl_to_points_batch(src_folder, target_folder)
ExportPointsToFile[stl, t, sfx] →  export_points_to_file(mesh_or_path, target,
                                   suffix, *, decimals=2, dedup=True)
WPoint[arr, first, n, axe]      →  w_point(points, first, n, axis)
WPointPacket[arr, first, n, axe]→  w_point_packet(points, first, ndiap, axis)
WPointPacketExport[...]         →  w_point_packet_export(...)
FileNameToList[file, sc, vol]   →  parse_filename(path, scale, volume)
                                   возвращает dataclass FilenameMetadata
```

## Структура целевого пакета

Пакет `cardio_stl_repair` — один-два модуля, ~400–500 строк суммарно.
Если файл разрастается выше 600 строк — разбивать по тематикам
(`mesh_io.py`, `repair.py`, `points.py`, `batch.py`).

```
src/cardio_stl_repair/__init__.py
src/cardio_stl_repair/repair.py
  ├── (private) _voxelize(mesh, pitch=None, target_resolution=256)
  │              → returns (mask: np.ndarray[bool], pitch: float, origin: np.ndarray[3])
  ├── (private) _close_per_slice(mask, radius, axis=2)
  ├── (private) _close_3d(mask, radius)
  ├── (private) _mask_to_mesh(mask, pitch, origin, smooth_sigma=1.0)
  │              → trimesh.Trimesh
  ├── (private) _fit_to_bbox(mesh, target_bounds, decimals_scale=3)
  │              → mesh со scale+translate, чтобы bbox совпал с target
  ├── @dataclass HeartRepairResult
  │       src: trimesh.Trimesh
  │       repaired: trimesh.Trimesh
  │       closed_mask: np.ndarray
  │       scale: np.ndarray[3]
  ├── heart_repair_test(mesh, closure_radius, *, mode="3d") → HeartRepairResult
  ├── heart_repair(mesh, *, closure_radius=20) → trimesh.Trimesh
  ├── @dataclass Task4Result
  │       src: trimesh.Trimesh; fix: trimesh.Trimesh; repair: trimesh.Trimesh
  ├── task4(stl_path, dest_dir, *, export=True) → Task4Result
  ├── task4_export(stl_path, target_dir) → None
  ├── heart_repair_thread(path, dest, namepattern, *, view=False)
  ├── packet_heart_repair(pairs, dest)
  ├── export_points_to_file(mesh_or_path, target_dir, suffix, *, decimals=2)
  ├── w_point(points, first, n, axis)
  ├── w_point_packet(points, first, ndiap, axis) → list[np.ndarray]
  ├── w_point_packet_export(stl_path, target, suffix, *, first, ndiap, axis)
  ├── @dataclass FilenameMetadata
  │       full_name, patient, has_sth, breath_phase ('Вдох'/'Выдох'),
  │       systole_phase, num, scale_x/y/z (Optional), volume (Optional)
  ├── parse_filename(path, scale=None, volume=None) → FilenameMetadata
  ├── prepare_stl_batch(src_folder, target_folder, *, repair=True)
  └── stl_to_points_batch(src_folder, target_folder)
```

## Зависимости

В основном `[project.dependencies]` (это пакет — не дополнение, всё ядро):

```toml
dependencies = [
    "numpy>=1.24",
    "trimesh>=4.0",
    "scikit-image>=0.20",
    "scipy>=1.10",
]

[project.optional-dependencies]
full = [
    "pyvista>=0.43",     # визуализация (verbose mode)
    "pymeshfix>=0.16",   # альтернативная починка дыр
]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
    "mypy>=1.10",
]
```

`pyvista`/`pymeshfix` — мягкие
импорты внутри функций, чтобы основной `stl` extra ставился быстро
и без MSVC-сборки.

## Этапы порта

Каждый этап заканчивается зелёным тестом и одним коммитом.

### Этап 1 — Вокселизация и обратная сборка mesh (1–2 дня)

- Реализовать `_voxelize`, `_mask_to_mesh`, `_fit_to_bbox`.
- Тест: куб 10×10×10 → vox → mesh → объём ≈ 1000 ±5%, bbox сохраняется.
- Тест: икосфера → vox → mesh → объём ≈ объём сферы ±5%.
- Коммит: `feat(stl_repair): voxelize + marching_cubes + bbox fit`.

### Этап 2 — Морфология (1 день)

- Реализовать `_close_per_slice`, `_close_3d`.
- Тест: «бублик с дырой» (cube minus cylinder) → close_3d с
  достаточным радиусом → mask замыкается.
- Тест: per_slice vs 3d — оба прогоняют, сравниваем числа.
- Коммит: `feat(stl_repair): morphological closing (per-slice and 3D)`.

### Этап 3 — Починка одного mesh (2 дня)

- Реализовать `heart_repair_test` и `heart_repair`.
- Тест: «сфера с дырой» (icosphere − cylinder через `trimesh.boolean`)
  → repair → объём после ≈ объём целой сферы ±5%.
- Тест: round-trip (load STL → repair → save → load) сохраняет объём
  с точностью ±0.1%.
- Тест: bbox preservation после `_fit_to_bbox` — точное совпадение
  до 3-го знака.
- Коммит: `feat(stl_repair): heart_repair / heart_repair_test`.

### Этап 4 — `Task4` верхний уровень (1 день)

- Реализовать `task4` и `task4_export`.
- Тест: дать синтетический STL → проверить, что генерируются три
  файла с правильными суффиксами `_src/_fix/_repair.stl`.
- Тест: `Task4Result.fix` имеет малый объём изменений vs src,
  `Task4Result.repair` — больший (по разнице volume).
- Коммит: `feat(stl_repair): task4 pipeline`.

### Этап 5 — Экспорт точек (0.5 дня)

- Реализовать `export_points_to_file`.
- Тест: round-trip CSV → загрузка `np.loadtxt` → совпадение точек
  с округлением 0.01.
- Тест: дедупликация — два совпадающих vertex объединяются.
- Коммит: `feat(stl_repair): export_points_to_file`.

### Этап 6 — Weighted resampling (0.5–1 день)

- Реализовать `w_point`, `w_point_packet`, `w_point_packet_export`.
- Тест: проверка распределения весов: для `n=3, axis=2`
  результирующий массив должен иметь количество `(3+2+1)*N_per_bin`.
- Тест: точки внутри каждого интервала по `axis` действительно
  попадают в этот интервал (Between-проверка).
- Коммит: `feat(stl_repair): w_point weighted resampling utilities`.

### Этап 7 — Парсер имён файлов (0.3 дня)

- Реализовать `parse_filename` и dataclass `FilenameMetadata`.
- Тест: фиксированные тестовые строки имён → ожидаемые поля.
- Тест: робастность к разным разделителям и неполным именам.
- Коммит: `feat(stl_repair): parse_filename / FilenameMetadata`.

### Этап 8 — Пакетные операции (0.5 дня)

- `prepare_stl_batch`, `stl_to_points_batch`, `packet_heart_repair`,
  `heart_repair_thread`.
- Тест: сделать tmp-папку с двумя синтетическими STL, прогнать batch,
  проверить наличие нужных файлов в выходе.
- Коммит: `feat(stl_repair): batch operations`.

### Этап 9 — Опциональные доработки (1 день)

- Pymeshfix-fallback в `heart_repair`: если морф. замыкание слишком
  «съедает» геометрию (объём изменился >25%), переключаемся на
  `pymeshfix.MeshFix`. Soft import.
- pyvista-визуализация в `verbose` режиме `task4` (показ src/fix/repair
  в одном окне).
- Коммит: `feat(stl_repair): pymeshfix fallback + pyvista preview`.

### Этап 10 — Документация и пример (0.5 дня)

- `docs/STL_REPAIR.md` с примерами использования и таблицей
  Wolfram→Python.
- `notebooks/02_stl_repair_smoke.ipynb` — синтетика «сфера с дырой».
- README обновить (добавить раздел про `[stl]` extra).
- Коммит: `docs(stl): usage and notebook`.

**Итого: ~8–9 рабочих дней.**

## Тонкие места и риски

1. **Метод mesh-extraction** — `Method->"MarchingSquares"` неоднозначен.
   План: реализовать обе ветки (`mode="3d"` по умолчанию, `mode="per_slice"`
   как опция) и сравнить визуально на тесте «сфера с дырой».
2. **`Blur` без параметров** — Mathematica использует Gaussian σ≈1 на
   image. Python: `gaussian_filter(mask.astype(float), sigma=1.0)`. Точное
   значение проверить эмпирически (объёмная разница после repair).
3. **Порядок осей** — Mathematica row/column convention vs NumPy:
   `Image3DSlices` идёт по последней оси `ImageDimensions`. Тест-бенчмарк
   на синтетическом анизотропном кубе (например, 10×20×30) покажет, не
   перепутаны ли оси.
4. **`ScalingTransform` от начала координат** — это важно. Если делать
   `mesh.apply_scale(s)` в trimesh, оно тоже от центра координат, не
   от центра mesh — совпадает. Но порядок `s1` (scale) затем `s2`
   (translate) обязателен.
5. **Округление `Round[box/boxfix, 0.001]`** — даст наблюдаемый сдвиг,
   повторить (`np.round(scale, 3)`).
6. **Топология после repair** — marching_cubes на closed-mask может
   дать non-watertight mesh при артефактах. План: после извлечения
   `mesh.fill_holes()` (trimesh) и `mesh.process()` в обязательном
   порядке.
7. **Объём** — `trimesh.Trimesh.volume` возвращает signed volume; для
   non-watertight даёт ерунду. Проверять `mesh.is_watertight` и при
   необходимости `mesh.volume = mesh.bounding_box.volume * fill_ratio`.
8. **CSV формат точек** — `Round[..., 0.01]` + `Export[..., "CSV"]`
   даёт CSV с колонками x,y,z и форматом по 2 знаков. Сохранить тот
   же формат, чтобы downstream-код (если он читает эти файлы) не сломался.
9. **Реальных STL нет** — все тесты синтетические. Когда появятся
   реальные данные, добавим integration-тест с эталонным объёмом.
10. **Соседний проект `Калмыков/3D_Slicer/`** — он будет производить
    STL «на входе» к нашему `stl_repair`. Чтобы они стыковались, в
    интерфейсе принимаем как `Path`, так и `trimesh.Trimesh`. Проверить
    позже совместно.

## Валидация

### Synthetic suite (всегда выполняется)

| Тест                      | Вход                              | Ожидание                                                |
| ------------------------- | --------------------------------- | ------------------------------------------------------- |
| Cube voxelize round-trip  | Куб 10×10×10                      | объём после mask_to_mesh ≈ 1000 ±5%, bbox сохраняется   |
| Sphere voxelize           | icosphere(r=10)                   | объём ≈ 4/3·π·r³ ±5%                                    |
| Sphere with hole — repair | icosphere − cylinder (вдоль Z)    | объём после repair ≈ объём целой icosphere ±5%          |
| Per-slice vs 3D mode      | sphere with hole                  | оба дают замкнутый mesh; объёмы близки (±10%)           |
| Round-trip                | sphere → repair → save → load     | объём не меняется (±0.1%)                               |
| Task4 file outputs        | sphere                            | в `dest/` есть `_src.stl`, `_fix.stl`, `_repair.stl`    |
| BBox preservation         | sphere → repair → fit_to_bbox     | bbox совпадает с оригинальным до 0.001                  |
| WPoint distribution       | random points в [0,10] по axis=0  | `len(w_point(p, 0, 3, 0)) == sum(1..3) * len_per_bin`   |
| Filename parser           | "ArtN_X_25_..._repair_..."        | dataclass с правильными полями                          |

### Integration suite (когда появятся реальные STL)

- Реальный сердечный STL → repair → визуальная проверка в 3D Slicer
  (нужен файл .mrml открытый бок-о-бок: src и repair).
- Сравнение с историческими CSV-выходами Mathematica (если найдутся).
- Сравнение объёма с эталоном из лаборатории (если запишут).

## Открытые вопросы (вынести наверх перед реализацией)

1. **Нужны ли FIX и REPAIR обе версии?** Task4 генерирует и r=1 (fix),
   и r=20 (repair). Для группы — обе нужны или достаточно одной?
   Если только `repair` — упростим API.
2. **Бит-в-бит совместимость CSV-точек.** Есть ли уже потребители
   CSV-файлов от Mathematica? Если да — сохраняем тот же формат
   (`%.2f`, comma, без header). Если нет — можно перейти на стандарт
   (header `x,y,z`).
3. **Интеграция с 3D_Slicer (Калмыков).** Должен ли наш `stl_repair`
   потреблять выход их pipeline `comsol_ready/`? Это влияет только на
   тип API (Path vs Trimesh), не на ядро.
4. **Источник тестовых STL** для будущей валидации: синтетика
   достаточна на этапе 1–9, но к этапу 10 хорошо иметь хотя бы один
   реальный файл (даже один — для smoke).

## Верификация (как понять, что порт удался)

1. `pytest tests/test_stl_repair.py -xvs` — все synthetic-тесты зелёные.
2. `python -m cardio_stl_repair --help` — CLI работает (если
   делаем CLI; задача-минимум — Python API).
3. `notebooks/02_stl_repair_smoke.ipynb` — работает end-to-end на
   синтетической сфере с дырой; визуальные графики до/после.
4. Когда появится реальный STL — открыть src и repair в 3D Slicer
   рядом, убедиться, что геометрия закрыта и узнаваемо «сердечная».
