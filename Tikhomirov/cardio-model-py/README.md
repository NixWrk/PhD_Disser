# cardio-model-py

Python-порт репозитория [`WolframMath-master`](../WolframMath-master) --
**радиоимпедансная кардиография (РЕО-32) совмещённая с МРТ**.

Рассчитывает ударный объём левого желудочка по данным 5-канального
торакального реографа и 7-точечным контурам сердца из МРТ.

> Модуль `Project/RFFI.m` (STL-починка) функционально независим и
> портируется отдельно -- см. [`../cardio-stl-repair/`](../cardio-stl-repair/).

## Статус реализации

| Фаза | Модуль | Статус | Тесты |
|------|--------|--------|-------|
| 0 | Инфраструктура (`pyproject.toml`, каркас) | Готово | 2 |
| 1 | Данные пациентов (`data/`) | Готово | 35 |
| 2 | Физическое ядро (`models`, `finders`, `radial`) | Готово | 61 |
| 3 | Геометрия и объёмы (`geometry`, `volume`) | Готово | 24 |
| 4 | Подгонка сферы и систола (`sphere_fit`, `sistole`) | Готово | 70 |
| 5 | Верхний пайплайн (`pipeline`) | Готово | 19 |
| 6 | DSP и импортёры (`dsp`, `importers/`) | **Заглушки** | -- |

**Всего: 222 теста, все зелёные.** Фазы 1-5 полностью реализованы
и проверены против эталонных значений из Wolfram Engine 14.3.

### Заглушки (только docstring, без кода)

| Файл | Wolfram-источник | Причина |
|------|-----------------|---------|
| `dsp.py` | `DspLib.m` + `MyFunction.m` | Фаза 6 (#43-#44), нет приоритета |
| `plotting.py` | -- | verbose-визуализация (#40) |
| `importers/reo32.py` | `ReoImporter.m` | Нет реальных файлов РЕО-32 (#45) |
| `importers/comsol.py` | `Comsol.m` | Одна функция, низкий приоритет (#46) |

## Установка

```bash
# Из каталога проекта:
pip install -e .[dev]

# Проверка:
pytest -q
```

Требует Python >= 3.10. Зависимости: `numpy`, `scipy`, `matplotlib`, `shapely`.

## Структура проекта

```
cardio-model-py/
├── pyproject.toml
├── docs/
│   ├── PORT_PLAN.md              полный план переноса
│   ├── COMMIT_PLAN.md            план коммитов по фазам
│   └── REFERENCE_VALUES.md       откуда взяты эталоны
│
├── src/cardio_model/
│   ├── __init__.py
│   │
│   │   ── ФИЗИЧЕСКОЕ ЯДРО ──
│   ├── models.py                 OneLayerModel, TwoLayerModel, SphereModel
│   ├── finders.py                FindRoOne, FindDRSphere, FindDRSphereFl, ...
│   ├── radial.py                 RadEvalMethod1..4, RadEval
│   │
│   │   ── ГЕОМЕТРИЯ И ОБЪЁМ ──
│   ├── geometry.py               KubicInterpol, GetCoordAfterMove, сечения
│   ├── volume.py                 VolumeByContour, SVbyContourAnd5Move
│   │
│   │   ── ПОДГОНКА СФЕРЫ ──
│   ├── sphere_fit.py             EqualSphere (NM), MassCenterContour
│   │
│   │   ── СИСТОЛА ──
│   ├── sistole.py                SistoleContour, ValvePoint, ValveConturMove
│   │
│   │   ── ПАЙПЛАЙН ──
│   ├── pipeline.py               EqualSphereMoveModelling, RadialEvaluation,
│   │                             TotalModelling, DxDyFind
│   │
│   │   ── ЗАГЛУШКИ (фаза 6) ──
│   ├── dsp.py                    (пусто) FirFilter, GetR, MyDeriv
│   ├── plotting.py               (пусто) обёртки matplotlib
│   ├── importers/
│   │   ├── reo32.py              (пусто) импорт РЕО-32
│   │   └── comsol.py             (пусто) ExtractComsolTable
│   │
│   │   ── ДАННЫЕ ПАЦИЕНТОВ ──
│   └── data/
│       ├── patients.py           контуры (7 точек), zBase, dZRad, параметры
│       ├── mri.py                контуры с предсердием (11 точек), ValveMove
│       ├── imp.py                временной ряд Artem (25 фреймов)
│       └── moves/
│           ├── __init__.py       load_moves, edge_move_5pt, edge_move_7pt
│           ├── alex.txt
│           ├── artem.txt
│           └── ivan.txt
│
└── tests/
    ├── reference_values.json     эталон из Wolfram Engine 14.3
    ├── test_import.py            smoke-тесты (2)
    ├── test_data_loaded.py       данные пациентов (35)
    ├── test_models.py            one/two/sphere модели (24)
    ├── test_finders.py           табличный поиск (15)
    ├── test_radial.py            RadEvalMethod1/2 (22)
    ├── test_geometry_volume.py   интерполяция, объём (24)
    ├── test_sphere_fit.py        NM-подгонка сферы (34)
    ├── test_sistole.py           систолический контур (36)
    └── test_pipeline.py          сквозной пайплайн (19)
```

## Архитектура

### Вычислительный конвейер (снизу вверх)

```
 models.py          Прямые модели: Z(rho, геометрия) → импеданс (Ом)
     ↓
 finders.py         Обратные задачи: Z_measured → rho / dR (табличный поиск)
     ↓
 radial.py          5 каналов x 4 метода → dR[5] (метры)
     ↓
 geometry.py        7 точек → KubicInterpol → 186 точек → сечения
     ↓
 volume.py          сечения → усечённые конусы → VolumeByContour (мл)
     ↓
 sphere_fit.py      контур → Nelder-Mead → центр + радиус экв. сферы
     ↓
 sistole.py         диастолический контур + dR → систолический контур
     ↓
 pipeline.py        полная сборка: dR → систола → объём → SV (мл)
```

### Поток данных (`radial_evaluation`)

```
    Импедансные данные              МРТ-данные
    (dZ, zBase — 5 каналов)       (контур 7 точек, valveMove)
           |                              |
    [RadEvalMethod 1/2/4]          [EqualSphere]
           |                              |
      dR[5] (метры)            center_move[2] (мм)
           |                              |
           +---> SistoleContour <---------+
                      |
              систолический контур
                      |
         VolumeByContour(д) − VolumeByContour(с)
                      |
                SV = delta / 2  (мл)
```

## Основные функции

### Быстрый старт: ударный объём одного пациента

```python
from cardio_model.pipeline import radial_evaluation

# Метод 1: без поправки первого слоя, без движения клапана
sv = radial_evaluation("Ivan", "None", "Use_Contour_MassCenter", "None")
print(f"SV (Ivan) = {sv:.1f} мл")

# Метод 2: с поправкой первого слоя + движение клапана
sv2 = radial_evaluation("Ivan", "ValveMove", "Use_Contour_MassCenter", "FirstLayer")
print(f"SV (Ivan, method 2) = {sv2:.1f} мл")
```

### Перебор всех комбинаций методов

```python
from cardio_model.pipeline import total_modelling

# 2 sistole_methods x 2 eq_sphere_methods x 3 rad_eval_methods = 12
results = total_modelling("Artem")
# results[i][j][k] = SV для комбинации (sistole, sphere, radial)
```

### Модель импеданса

```python
from cardio_model.models import sphere_model

# Импеданс для сферы радиуса R=42мм на глубине h=22мм
Z = sphere_model(rho1=1.35, rho2=1.35, a=0.05, b=0.025, R=0.042, h=0.022, x=0, y=0.035)
```

### Объём по контуру

```python
from cardio_model.data.patients import get_contour
from cardio_model.volume import volume_by_contour

contour = get_contour("Ivan")       # (7, 2) float
vol = volume_by_contour(contour)    # ≈ 485 мл
```

## Соответствие Wolfram → Python

| Wolfram (WolframMath-master) | Python (cardio-model-py) |
|------------------------------|--------------------------|
| `Models.m` → `OneLayerModel(rho1, a, b)` | `models.one_layer_model(rho1, a, b)` |
| `Models.m` → `SphereModel(rho1, rho2, a, b, R, h, x, y)` | `models.sphere_model(rho1, rho2, a, b, R, h, x, y)` |
| `ModelFinders.m` → `FindDRSphere(Zb, dZ, ro1, ...)` | `finders.find_dr_sphere(Zb, dZ, ro1, ...)` |
| `RadialEvaluation.m` → `RadEvalMethod1(obj, param)` | `radial.rad_eval_method1(obj, param)` |
| `VolumeCalc.m` → `KubicInterpol(points, 30, fringe)` | `geometry.kubic_interpol(points, n=30, fringe=None)` |
| `VolumeCalc.m` → `VolumeByContour(points, 3)` | `volume.volume_by_contour(points, h_step=3.0)` |
| `SphereMovingFunction.m` → `EqualSphere(c, method, r)` | `sphere_fit.equal_sphere(c, method, radius_if_need=r)` |
| `SistoleFunction.m` → `SistoleContour(d, move, vm, m)` | `sistole.sistole_contour(d, move, vm, m)` |
| `EqualSphereMoveModellingLib.m` → `RadialEvaluation(...)` | `pipeline.radial_evaluation(...)` |
| `EqualSphereMoveModellingLib.m` → `TotalModelling(name)` | `pipeline.total_modelling(name)` |
| `Sum[..., {n, 0, 100}]` + `LegendreP[n, x]` | `for n in range(101)` + рекуррентность |
| `Partition[Table[...], 2, 1]` + `Select` | `_tabular_crossing()` — намеренный табличный поиск |
| `NMinimize[..., "NelderMead"]` | `scipy.optimize.minimize(method='Nelder-Mead')` |
| `Round[x, 0.0001]` | `np.round(x, 4)` |
| `RegionCentroid[Polygon[c]]` | `shapely.geometry.Polygon(c).centroid` |

## Верификация

Эталонные значения сгенерированы скриптом `gen_reference.m` на
**Wolfram Engine 14.3** и хранятся в `tests/reference_values.json`.

Точные совпадения проверяются для:
- `RadEvalMethod1` / `RadEvalMethod2` — `np.testing.assert_array_equal`
- `FindDRSphere` / `FindDRSphereFl` — `abs(got - expected) < 1e-8`
- `VolumeByContour` — `abs(got - expected) <= 0.5` мл
- `SVbyContourAnd5Move` — `abs(got - expected) <= 0.5` мл
- `EqSphNMCenterRadius` — `abs(got - expected) <= 0.5` (стохастичен)

## Известные решения

1. **Табличный поиск вместо root-finding.** `ModelFinders.m` намеренно
   использует `Partition[Table[...]]` + `Select` вместо `FindRoot`,
   потому что `SphereModel` немонотонна при малых R. Python-порт
   сохраняет этот подход (`_tabular_crossing`).

2. **Граничные производные в `KubicInterpol`.** Mathematica решает
   СЛАУ с заглушками {1,1} в крайних строках правой части, затем
   *перезаписывает* dp[0] и dp[-1] значениями `fringe`. Python
   делает то же — это не баг, а особенность порта.

3. **`Round[x, 0.0001]`** обязателен в `RadEvalMethod1/2` — без него
   результаты расходятся с эталоном из-за накопления ошибок в
   табличном поиске.

4. **`_RO2 = 1.35`** (удельное сопротивление крови) зафиксировано
   во всех RadEvalMethod.

## Документация

- [`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) — карта Wolfram-кода,
  целевая структура, риски, соответствия
- [`docs/COMMIT_PLAN.md`](docs/COMMIT_PLAN.md) — план коммитов по фазам
- [`docs/REFERENCE_VALUES.md`](docs/REFERENCE_VALUES.md) — протокол
  сборки эталонных значений

## Связь с исходным репозиторием

Wolfram-код ([`WolframMath-master/`](../WolframMath-master/)) остаётся
нетронутым и служит эталоном. Все правки идут только в этот пакет.
