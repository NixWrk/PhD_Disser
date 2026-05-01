# WolframMath-master

Mathematica-библиотека для **радиоимпедансной кардиографии (РЕО-32)
совмещённой с МРТ**. Рассчитывает ударный объём левого желудочка по
данным 5-канального торакального реографа и контурам сердца из МРТ.

Целевая версия: Mathematica 11.3 (IntelliJ IDEA plugin); совместим с
Wolfram Engine 14.x.

## Структура проекта

```
WolframMath-master/
├── Kernel/
│   ├── init.m                          точка входа — загружает все .m
│   │
│   ├── core/                           ЯДРО ФИЗИЧЕСКОЙ МОДЕЛИ
│   │   ├── Models.m                    OneLayerModel, TwoLayerModel,
│   │   │                               SphereModel (Лежандр n=0..100)
│   │   ├── ModelFinders.m              FindRoOne/HTwo/RSphere/HSphere/
│   │   │                               RoSphere/DRSphere/DRSphereFl/
│   │   │                               DRSphereFlDxDy (табличный поиск)
│   │   ├── RadialEvaluation.m          RadEvalMethod1..4, RadEval,
│   │   │                               DxDyFind, пациентские данные
│   │   ├── VolumeCalc.m                KubicInterpol, GetCoordAfterMove,
│   │   │                               VolumeByContour, SVbyContourAnd5Move
│   │   ├── SphereModel.m              тестовый дубль Models.m (SphereModelTest)
│   │   ├── DspLib.m                    FirFilter, FirFilterSZA
│   │   ├── alex.txt / artem.txt /      данные движений границ сердца
│   │   └── ivan.txt                    (5 каналов x N точек)
│   │
│   ├── part1/                          ПАЙПЛАЙН ВЕРХНЕГО УРОВНЯ
│   │   ├── SphereMovingFunction.m      EqualSphere (Nelder-Mead подгонка),
│   │   │                               MassCenterContour, ReqForHeart
│   │   ├── SistoleFunction.m           SistoleContour, ValvePoint,
│   │   │                               ValveConturMove
│   │   ├── EqualSphereMoveModellingLib.m  EqualSphereMoveModelling,
│   │   │                               RadialEvaluation, TotalModelling
│   │   └── SphereMovingOptimization.m  ЗАГЛУШКА (пустой файл)
│   │
│   ├── history/                        ПАЦИЕНТСКИЕ / ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ
│   │   ├── MRIdata.m                   ValveMove, RbyMRIbyVolume,
│   │   │                               HeartContoursWithAtrial, AtrialPoints
│   │   ├── IMPdata.m                   artemRadialArray (25x6),
│   │   │                               artemSphereMoving (25x4)
│   │   └── ReoImporter.m              Forma0/1Reo32, EcgReo32
│   │
│   ├── help/                           ДОКУМЕНТАЦИЯ
│   │   ├── HelpInit.m                  DocImport (загрузчик help.txt)
│   │   └── Help.m                      Help[] — пользовательская справка
│   │
│   ├── utils/                          УТИЛИТЫ
│   │   ├── MyFunction.m                GetR (R-пик ЭКГ), MyDeriv,
│   │   │                               MyIntegrate, MyFilter, Reo32Import
│   │   ├── MySystem.m                  GetBoxPath, GetFilesNames
│   │   └── Comsol.m                    ExtractComsolTable
│   │
│   └── Project/
│       └── RFFI.m                      STL-починка: HeartRepair, Task4,
│                                       ExportPointsToFile, WPoint
│
├── implem/
│   ├── fat/Modelling0.m               DeltaZPulse, DZMatrix (эксперимент)
│   └── sphereMove/SphereMoveModelling.nb  интерактивный пример
│
├── gen_reference.m                     скрипт генерации эталонных значений
├── MathematicaIDEA.m / .nb             обёртка IntelliJ plugin
└── NBTest.nb                           тестовый notebook
```

## Архитектура

### Вычислительный конвейер (снизу вверх)

```
Уровень 1 — Прямые модели (Models.m):
  OneLayerModel(rho1, a, b)               гомогенный цилиндр
  TwoLayerModel(rho1, rho2, h, a, b)      двухслойный (ткань + жир)
  SphereModel(rho1, rho2, a, b, R, h, x, y)   цилиндр + сфера (сердце),
                                              ряд Лежандра n=0..100

Уровень 2 — Обратные задачи (ModelFinders.m):
  FindRoOne         аналитическая инверсия OneLayerModel
  FindRoSphere      табличный поиск rho1 при известном Zb
  FindDRSphere      табличный поиск dR при известном dZ
  FindDRSphereFl    то же + поправка первого слоя (жир)
  FindDRSphereFlDxDy  + латеральное смещение центра

Уровень 3 — Радиальная оценка (RadialEvaluation.m):
  RadEvalMethod1    dR без поправок
  RadEvalMethod2    dR + первый слой
  RadEvalMethod4    dR + первый слой + смещение центра сферы

Уровень 4 — Геометрия контура (VolumeCalc.m):
  KubicInterpol               7 точек → 186 точек (эрмитов сплайн)
  GetCoordAfterMove           полярный сдвиг контура
  VolumeByContour             параллельные сечения → усечённые конусы → мл

Уровень 5 — Систола + подгонка сферы (part1/):
  SistoleContour              диастолический контур + dR → систолический
  EqualSphere                 Nelder-Mead подгонка эквивалентной сферы

Уровень 6 — Полный пайплайн (EqualSphereMoveModellingLib.m):
  EqualSphereMoveModelling    смещение центра сферы (диастола → систола)
  RadialEvaluation            dR → систолический контур → объём → SV
  TotalModelling              перебор 2 x 2 x 3 = 12 комбинаций методов
```

### Поток данных для одного пациента

```
      Экспериментальные данные                МРТ-данные
      (dZ по 5 каналам, zBase)            (7-точечный контур, valveMove)
              |                                     |
     [RadEvalMethod1..4]                    [EqualSphere]
              |                                     |
        dR[5] (метры)                   center_move[2] (мм)
              |                                     |
              +---->  SistoleContour  <------+------+
                           |
                   систолический контур (7 точек)
                           |
              VolumeByContour(диастол) - VolumeByContour(систол)
                           |
                     SV = разница / 2  (мл)
```

## Зависимости

- **Mathematica** >=11.3 (или Wolfram Engine >=13.0 для headless)
- Нет внешних пакетов — только стандартная библиотека Wolfram Language
- Для `RFFI.m`: Mathematica с поддержкой `Image3D`, `ImageMesh`, `Closing`

## Данные пациентов

Три пациента с полными наборами данных: **Ivan**, **Alex**, **Artem**.

| Источник | Файл | Формат |
|----------|------|--------|
| Контуры (7 точек) | `VolumeCalc.m` | [[x,y], ...] мм |
| Контуры с предсердием (11 точек) | `MRIdata.m` | [[x,y], ...] мм |
| Импедансные данные (zBase, dZRad) | `RadialEvaluation.m` | Association |
| Параметры модели (a,b,R,h,x,y) | `RadialEvaluation.m` | Association |
| Движения границ (5 каналов) | `alex.txt`, `ivan.txt`, `artem.txt` | TSV, 5xN |
| Временной ряд (Artem, 25 фреймов) | `IMPdata.m` | массив 25x6 |

## Известные проблемы

1. **`SphereMovingOptimization.m` — пустой файл.** Полностью заглушка,
   Private секция пуста.

2. **`EqualSphereMoveModellingLib.m`** — объявлены `TestString` и
   `ConvertToCyr`, но реализации нет.

3. **`RadialEvaluation.m`** — объявлен `GetAlexParamV1`, реализации нет.

4. **`IMPdata.m`** — `artemSphereMoving02mm` — символ без присвоения
   (строка 163). Дублирование `RadialExpData[name_]` (строки 28, 31).

5. **`Help.m`** — содержит ошибочную копию `SVbyContourAnd5Move[]`
   с багом: `Verb = "no"` вместо `Verb == "no"` (присвоение вместо
   сравнения).

6. **`ReoImporter.m`** — хардкод путей к `C:\Users\Alexey\Box Sync\...`.

7. **`MathematicaIDEA.m`** — ссылается на `Needs["Test`"]`, которого нет.

## Python-порт

Параллельный репозиторий [`cardio-model-py/`](../cardio-model-py/)
содержит Python-порт ядра (фазы 1–5 завершены). Модуль STL-починки
(`RFFI.m`) портируется отдельно — см. [`cardio-stl-repair/`](../cardio-stl-repair/).
