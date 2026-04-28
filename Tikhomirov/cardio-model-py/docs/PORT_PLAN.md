# План: Перенос репозитория WolframMath-master на Python

## Контекст

Репозиторий — научный код на Wolfram Mathematica (целевая версия 11.3, см. шапки `Help.m`,
`HelpInit.m`; даты файлов 2017–2021), часть исследовательского проекта по
радиоимпедансной кардиографии (РЕО-32) совмещённой с МРТ. Авторы: Aleksey/Alex.
Цель портирования — снять зависимость от платной проприетарной Mathematica
и перейти на Python (NumPy/SciPy/matplotlib), чтобы код можно было запускать
на любом окружении (в т. ч. на машинах остальных участников группы)
и интегрировать с остальным кодом группы Kardio_respiratory_dudes.

Какая «версия математики использовалась». В шапках `Kernel/help/Help.m` и
`Kernel/help/HelpInit.m` явно стоит `Mathematica Version: 11.3` (2018 г.).
Проект пользуется функциями, доступными с 11.x (`RegionCentroid`, `ImageMesh`,
`Image3D`, `Association`, `OptionsPattern`), и должен работать на 11.3–13.x
без переписывания. Никаких точных «требований» в файлах не указано — это
типовой проект на Wolfram Language, написанный с плагином IntelliJ.

## Что в репозитории есть (карта кода)

```
WolframMath-master/
├── README.md                          (почти пустой — «# Описание»)
├── MathematicaIDEA.m / .nb            (мини-обёртка plugin)
├── NBTest.nb                          (нерелевантная заметка)
├── Kernel/
│   ├── init.m                         (точка входа: подгружает все .m)
│   ├── README.md                      (краткая карта пакета)
│   ├── core/                          ← ЯДРО ФИЗИЧЕСКОЙ МОДЕЛИ
│   │   ├── Models.m                   OneLayerModel, TwoLayerModel,
│   │   │                              SphereModel, SphereInfluence
│   │   │                              (формулы импеданса на основе
│   │   │                              разложения по полиномам Лежандра, n=0..100)
│   │   ├── ModelFinders.m             FindRoOne, FindHTwo, FindRSphere,
│   │   │                              FindHSphere, FindRoSphere,
│   │   │                              FindRoSphereNMead, FindDRSphere,
│   │   │                              FindDRSphereDxDy, FindDRSphereFl,
│   │   │                              FindDRSphereFlDxDy
│   │   ├── RadialEvaluation.m         GetIvanRadial/AlexRadial/ArtemRadial,
│   │   │                              GetIvanParam/AlexParam/ArtemParam,
│   │   │                              RadEvalMethod1..4, RadEval,
│   │   │                              HCoutourDxDyForEachChannel, DxDyFind
│   │   ├── VolumeCalc.m               GetCoordAfterMove, GetContursInTime,
│   │   │                              HeartContours, HeartEdgeMoving,
│   │   │                              KubicInterpol (кубич. эрмитова),
│   │   │                              ListOfParallelSectionForContour,
│   │   │                              VolumeBySection (Симпсон-усечённый конус),
│   │   │                              VolumeByContour, SVbyContourAnd5Move,
│   │   │                              SVfinder
│   │   ├── SphereModel.m              (дубль формул из Models.m, тестовый)
│   │   ├── DspLib.m                   FirFilter, FirFilterSZA
│   │   ├── alex.txt / artem.txt /     5 строк по 10 чисел —
│   │   └── ivan.txt                   движения краёв сердца по 5 каналам
│   ├── part1/
│   │   ├── SistoleFunction.m          SistoleContour, SistoleNone,
│   │   │                              SistoleValveMove, ValvePoint,
│   │   │                              ValveConturMove
│   │   ├── SphereMovingFunction.m     EqualSphere (NelderMead),
│   │   │                              EqSphNMCenterRadius, EqSphNMRadius,
│   │   │                              EqSphNMCenter, EqSphMassCenter,
│   │   │                              MassCenterContour, ReqForHeart,
│   │   │                              CircleRadiusByContourSqr,
│   │   │                              AbsolutSphereXYinLocalChannel
│   │   ├── SphereMovingOptimization.m (пустой)
│   │   └── EqualSphereMoveModellingLib.m
│   │                                  EqualSphereMoveModelling,
│   │                                  RadialEvaluation, TotalModelling
│   │                                  ← ВЕРХНЕУРОВНЕВЫЙ КОНВЕЙЕР
│   ├── history/
│   │   ├── MRIdata.m                  ValveMove, RbyMRIbyVolume,
│   │   │                              HeartContoursWithAtrial,
│   │   │                              AtrialPoints, NewYCalculation
│   │   ├── IMPdata.m                  RadialExpData (временной ряд для Artem),
│   │   │                              artemRadialArray/Fix, artemSphereMoving
│   │   └── ReoImporter.m              Forma0Reo32, Forma1Reo32,
│   │                                  Reo32 импорт из .txt (хардкод путей
│   │                                  в C:\Users\Alexey\Box Sync\...)
│   ├── help/                          (HelpInit.m, Help.m — загрузка
│   │                                  doc-строк из help.txt; help.txt отсутствует)
│   ├── utils/
│   │   ├── MyFunction.m               GetR (R-пик ЭКГ), Reo32Import,
│   │   │                              Reo32Cutter, MyDeriv, MyIntegrate,
│   │   │                              MyFilter, zeroCrossings и т.д.
│   │   ├── MySystem.m                 GetBoxPath, GetFilesNames
│   │   └── Comsol.m                   ExtractComsolTable
│   ├── simple/SimpleFunction.m        (xx[y]:=y*4 — демо)
│   └── Project/RFFI.m                 Task4, HeartRepair, ExportPointsToFile,
│                                      WPoint*, PrepareStl, StlToPointsFile
│                                      ← STL-починка; портируется в
│                                        ОТДЕЛЬНЫЙ репозиторий cardio-stl-repair/
└── implem/
    ├── fat/Modelling0.m               DeltaZPulse, DZMatrix (демо)
    └── sphereMove/SphereMoveModelling.nb  (пример вызова RadialEvaluation)
```

## Стратегия порта (рекомендуемая)

Идея — сохранить ту же декомпозицию по модулям, что и в Mathematica
(меньше шансов потерять вычислительный смысл), и переписать поверх
NumPy/SciPy. Файлы данных (`alex.txt`, `artem.txt`, `ivan.txt`) переносятся
без изменений и читаются `numpy.loadtxt`.

### Структура целевого проекта

Целевой код пишется **в отдельный репозиторий** (имя/путь — на усмотрение
пользователя; дальше по тексту он называется условно `cardio-model-py/`).
Текущий `WolframMath-master/` остаётся нетронутым и служит эталоном.

```
cardio-model-py/                 (отдельный репозиторий)
├── pyproject.toml               (numpy, scipy, matplotlib, shapely, pytest)
├── src/cardio_model/
│   ├── __init__.py
│   ├── models.py                ← Kernel/core/Models.m
│   ├── finders.py               ← Kernel/core/ModelFinders.m
│   ├── radial.py                ← Kernel/core/RadialEvaluation.m
│   ├── geometry.py              ← KubicInterpol, GetCoordAfterMove,
│   │                              ListOfParallelSectionForContour
│   ├── volume.py                ← VolumeBySection, VolumeByContour, SVfinder
│   ├── sphere_fit.py            ← Kernel/part1/SphereMovingFunction.m
│   ├── sistole.py               ← Kernel/part1/SistoleFunction.m
│   ├── pipeline.py              ← Kernel/part1/EqualSphereMoveModellingLib.m
│   ├── dsp.py                   ← DspLib.m + GetR/MyDeriv/MyIntegrate
│   ├── importers/
│   │   ├── reo32.py             ← ReoImporter.m + Reo32Import/Cutter
│   │   └── comsol.py            ← utils/Comsol.m
│   ├── data/
│   │   ├── patients.py          ← данные Ivan/Alex/Artem (контуры,
│   │   │                          dZRad, zBase, a/b/R/h/x/y/hFat/flSize)
│   │   └── moves/{alex,ivan,artem}.txt   (копии без изменений)
│   └── plotting.py              ← обёртки matplotlib над ListLinePlot/Show
├── tests/
│   ├── reference_values.json    (эталонные значения из Mathematica)
│   ├── test_models.py
│   ├── test_finders.py
│   ├── test_radial.py
│   ├── test_geometry_volume.py
│   ├── test_sphere_fit.py
│   └── test_pipeline.py
└── notebooks/
    └── 01_smoke.ipynb           (повторяет implem/sphereMove/SphereMoveModelling.nb)
```

### Ключевые соответствия Wolfram → Python

| Wolfram                                         | Python                                                       |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `Sum[..., {n, 0, 100}]` с `LegendreP[n, x]`     | `numpy.polynomial.legendre.legval` или явное `for n in range(101)` через рекуррентность |
| `FindRoot[f==0, {x, x0}]`                       | `scipy.optimize.brentq` (с заданием бракета) или `fsolve`    |
| `NMinimize[..., Method->"NelderMead"]`          | `scipy.optimize.minimize(method='Nelder-Mead', bounds=...)`  |
| `Table[..., {i, n}]`                            | списочное включение / `np.fromiter`                          |
| `Partition[lst, 2, 1]`                          | `list(zip(lst, lst[1:]))` или `np.lib.stride_tricks.sliding_window_view` |
| `Association[...]`                              | `dict` (с insertion order) или `dataclass`                   |
| `Round[x, 0.0001]`                              | `np.round(x, 4)`                                             |
| `Switch[name, "Ivan", ..., "Alex", ...]`        | `dict.get` lookup или `match`/`if`                           |
| `ListConvolve[k, s]`                            | `scipy.signal.fftconvolve(s, k, mode='same')` (с проверкой ориентации) |
| `CoordinateTransform["Cartesian"->"Polar", p]`  | `r, phi = np.hypot(...), np.arctan2(...)` (помнить: Wolfram возвращает `{r, phi}`) |
| `Polygon[c]` + `RegionCentroid` / `Area`        | `shapely.geometry.Polygon(c).centroid` / `.area`             |
| `EuclideanDistance[a, b]`                       | `np.linalg.norm(a-b)`                                        |
| `VectorAngle[a, b]`                             | `np.arccos(np.clip(np.dot(â, b̂), -1, 1))`                   |
| `Closing[img, DiskMatrix[r]]`                   | `scipy.ndimage.binary_closing` или `skimage.morphology.closing` (используется только в `cardio-stl-repair`) |
| `Import[..., "Table"]`                          | `np.loadtxt` / `pandas.read_csv`                             |
| `OptionsPattern[]` + `Options[F]={...}`         | kwargs со значениями по умолчанию                            |
| `Module[{...}, body]`                           | обычная функция с локальными переменными                     |

### Этапы порта (выполнять в этом порядке)

1. **Собрать эталон из .nb и комментариев** (`tests/reference_values.json`).
   Mathematica/Wolfram Engine для запуска нет, поэтому действуем так:
   - Распарсить `implem/sphereMove/SphereMoveModelling.nb` (≈15 700 строк),
     `MathematicaIDEA.nb`, `NBTest.nb` — извлечь сохранённые `Output`-ячейки
     (числа, массивы, графика как BoxData). `.nb` — это plain-text
     S-expressions, читается обычным `re`/парсером (или модулем
     `mathematica` на PyPI). Отсюда берём фактические значения, которые
     Mathematica посчитала в последний раз.
   - Из закомментированных таблиц в `IMPdata.m` (`artemRadialArray`,
     `artemSphereMoving`) — это уже численные «ground truth», их можно
     перенести 1-в-1 в JSON.
   - Из usage-строк (`::usage`) и комментариев `(* ... *)` забрать любые
     явно указанные тестовые значения / пары «вход → выход».
   - Чего извлечь не удалось — пометить как «эталон отсутствует» и
     заменить тестом самосогласованности (например: `FindRoSphere` →
     прогнать `SphereModel` с найденным `R` и убедиться, что результат
     совпадает с `Zb` с допуском 1e-4).

   Минимальный набор величин, которые желательно закрепить:
   - `OneLayerModel[1, 0.04, 0.02]`,
     `TwoLayerModel[1, 1.35, 0.03, 0.04, 0.02]`,
     `SphereModel[1, 1.35, 0.04, 0.02, 0.05, 0.03, 0, 0.025]`
   - `RadEvalMethod1[GetArtemRadial[], GetArtemParam[]]` → массив 5 чисел
   - `VolumeByContour[ivan7pointContour, 3]`
   - `EqSphNMCenterRadius[ivan7pointContour]`
   - `RadialEvaluation["Artem", "None", "Use_Contour", "FirstLayer", False]`

2. **Перенести физическое ядро** (`models.py` → `finders.py` → `radial.py`).
   Это даёт ~80% научной ценности репозитория. Особое внимание:
   - В `Models.m` функция `Zep` использует свободный символ `n` внутри
     `Sum[...{n, 0, 100}]` — n — индекс суммирования; в Python это явный цикл/векторизация.
   - `FindRSphere`/`FindHSphere` сделаны не через `FindRoot`, а через
     табулированный линейный поиск с `Partition[..., 2, 1]`. Это намеренно —
     модель немонотонна на малых R, и `FindRoot` падает. Сохранить ту же
     стратегию: построить таблицу значений и взять интервал, в котором
     меняется знак.

3. **Перенести геометрию и объёмы** (`geometry.py`, `volume.py`).
   - `KubicInterpol` — это эрмитова кубическая сплайн-интерполяция с явным
     решением СЛАУ через `MatrixM`/`MatrixR` (`np.linalg.solve`). Можно
     оставить «как есть», либо заменить на `scipy.interpolate.CubicHermiteSpline`
     с теми же концевыми касательными — проверить эталоном.
   - `VolumeBySection` — Simpson по усечённым конусам (`CutKonusCircleVolume`).
   - `GetCoordAfterMove` — поворот контура: декарт → полярные → +Δr → декарт.
     В Wolfram `CoordinateTransform` возвращает `{r, φ}`, в Python это
     обычная пара `np.hypot/np.arctan2`. Внимание к порядку аргументов!

4. **Перенести подгонку сферы и систолу** (`sphere_fit.py`, `sistole.py`).
   `EqualSphere` использует `NMinimize` с боксовыми границами →
   `scipy.optimize.minimize(method='Nelder-Mead', bounds=Bounds(...))`.

5. **Собрать пайплайн** (`pipeline.py`). Дублирует
   `EqualSphereMoveModellingLib.m`. Ничего нового — только вызовы предыдущих
   модулей и matplotlib для verbose-режима.

6. **DSP и импортёры** (`dsp.py`, `importers/reo32.py`).
   `Reo32Import` читает табличные `.txt` без заголовков; формат каналов
   (нечётные = пульс, чётные = база, ЭКГ — третий с конца) явно описан в коде.
   Хардкод-пути из `ReoImporter.m` нужно вынести в config / параметр.
   Реальных `.txt` от REO-32 у нас нет — модуль портируется «всухую»
   по логике Wolfram-кода. Тесты для него ограничены:
   - синтетические сигналы (пилот-данные `np.linspace`/синус) — проверка,
     что `GetR` возвращает локальные максимумы;
   - проверка формата на тестовом CSV, сгенерированном самим Python-кодом
     по тому же layout, что и REO-32 (round-trip).
   Этот модуль помечается в README как «требует валидации на реальных
   данных, когда они появятся».

7. **Пациентские данные** (`data/patients.py`). Все ассоциации
   `GetIvanRadial`/`GetAlexParam`/контуры/`AtrialPoints`/`ValveMove`
   переносятся как обычные dict-литералы (или `@dataclass`).

8. **RFFI / STL-починка** — вынесена в отдельный репозиторий
   [`cardio-stl-repair/`](../../cardio-stl-repair/) (см.
   `cardio-stl-repair/docs/PORT_PLAN.md`). Этот модуль функционально
   независим от научного ядра (`RFFI.m` не использует `Needs[]` ни на
   что из остального `Kernel/`, и сам он импортируется только из
   `init.m` для удобства загрузки). Там — свой `pyproject.toml`,
   свои зависимости (`trimesh`, `scikit-image`), свой план коммитов.

### Ключевые риски / тонкие места

- **Полиномы Лежандра, n=0..100.** `Zep` суммирует ряд до 100. На границе
  сходимости (большие `R/rp` или `R/re`) ряд может расходиться/терять
  точность. Стоит сравнить численно с Mathematica на нескольких точках
  и при необходимости перейти на `mpmath` с повышенной точностью.
- **`Round[x, 0.0001]` в `RadEvalMethod*`.** Округление до 4-го знака
  встроено в результат — порт должен это сохранять, иначе поломаются
  эталонные значения.
- **Юникод в именах символов** (`\[Rho]1`, `\[Pi]`). В Python переименовать
  в `rho1`, `pi=np.pi` — на корректность это не влияет, но облегчает чтение.
- **Порядок индексации `dxdy[[i,2]], dxdy[[i,1]]`** — в `RadEvalMethod4`
  явный комментарий «сначала 2 потом 1, так как в модели x и y переставлены».
  Нужно сохранить ту же перестановку.
- **`HelpInit`/`Help`** требует `help.txt`, которого в репо нет → в Python
  заменить на обычные docstring + `help()`.
- **Mathematica 1-индексирована.** `points[[i+1]]`, `Part[..., 1]` —
  при переписывании внимательно сдвигать индексы.

## Файлы, которые нужно создавать/редактировать (для фазы реализации)

Все правки идут в **отдельный новый репозиторий** `cardio-model-py/`
(путь — на усмотрение пользователя). Текущий `WolframMath-master/`
**не трогается** и остаётся эталоном. Единственная возможная правка
здесь — добавить в корневой `README.md` ссылку на новый репозиторий.

## Верификация

End-to-end smoke (выполняется после фазы реализации):

1. `pip install -e cardio-model-py[dev]`; `pytest -xvs` — должны пройти
   все эталонные тесты (модели, finders, объём, сфера, пайплайн
   `radial_evaluation("Artem", ...)`). Эталон собран на этапе 1
   (из `.nb`-файлов и комментариев), пометка «эталон отсутствует»
   допускается, но в этом случае тест проверяет самосогласованность
   (см. этап 1).
2. Запустить `notebooks/01_smoke.ipynb` — он должен повторить вызов из
   `implem/sphereMove/SphereMoveModelling.nb` и выдать численный SV
   и контурные графики.
3. Прогнать `RadEvalMethod1`/`Method2`/`Method4` на трёх пациентах
   (Ivan, Alex, Artem) — массив 3×5 значений; сверить с числами,
   извлечёнными из Output-ячеек `.nb` (где удалось).
4. Порт `RFFI.m` (STL-починка) живёт в отдельном репозитории
   [`cardio-stl-repair/`](../../cardio-stl-repair/) — у него свои
   тесты и своя цепочка верификации.

## Открытые моменты, которые проявятся при реализации

- `help.txt`, на который ссылается `HelpInit.m`, отсутствует в репо.
  В Python заменяется на стандартные docstrings — никаких потерь.
- Реальных REO-32 `.txt` нет — `importers/reo32.py` валидируется только
  синтетикой; пометить как «требует валидации».
- Из `.nb`-файлов (особенно `SphereMoveModelling.nb`, ~15 700 строк)
  нужно вытащить максимум численного output для эталона. Если парсер
  плагин-формата окажется хрупким — извлекать вручную точечно, по
  ключевым ячейкам.
