# План коммитов: порт WolframMath-master → cardio-model-py

Каждый коммит — атомарный, верифицируемый, оставляет репозиторий в
рабочем состоянии (`pytest -xvs` зелёный на том, что уже портировано).
План предполагает работу одного автора в линейной истории на ветке
`master` без PR (для удобства синхронизации с командой). Для крупных
этапов можно открывать feature-ветки `feat/<этап>` и сливать
`fast-forward`.

Сокращения по типам коммитов: `chore` (инфра), `feat` (новый
функционал/модуль), `data` (перенос данных), `test` (тесты/эталон),
`docs` (документация), `fix` (исправление), `refactor` (рефакторинг
без поведения).

Привязка к этапам из [PORT_PLAN.md](PORT_PLAN.md) указана в скобках
(например, «этап 2» = пункт 2 в разделе «Этапы порта»).

---

## Фаза 0 — инфраструктура (1 день)

| #   | Тип    | Сообщение                                             | Содержимое                                                                                              |
| --- | ------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 01  | chore  | init: pyproject, README, .gitignore, лицензия         | `pyproject.toml` (numpy/scipy/matplotlib/shapely/pytest), `README.md`, `.gitignore` (Python+venv+IDE), `LICENSE` если нужна |
| 02  | docs   | docs: добавить PORT_PLAN.md и COMMIT_PLAN.md          | Файлы плана уже есть — фиксируем их в первом docs-коммите                                               |
| 03  | chore  | ci: pytest workflow + ruff/mypy                       | `.github/workflows/test.yml` (если есть GitHub) + `pyproject.toml` секции `[tool.ruff]` / `[tool.mypy]` |
| 04  | feat   | пакет cardio_model: пустой каркас модулей             | Все `__init__.py` + пустые модули с docstring, чтобы импорт `cardio_model` уже работал                  |

После фазы 0: `pip install -e .[dev]` работает, `pytest` запускается (тестов нет), `python -c "import cardio_model"` проходит.

---

## Фаза 1 — эталон и пациентские данные (этапы 1, 7 плана)

Сначала фиксируем «контракт» — потом по нему пишем код. Так каждый
следующий коммит можно мгновенно проверить.

| #   | Тип   | Сообщение                                                  | Содержимое                                                                                                                                                  |
| --- | ----- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 05  | data  | data(patients): контуры и параметры Ivan/Alex/Artem        | `data/patients.py` — словари из `RadialEvaluation.m` (`GetIvanRadial/Param`, `GetAlexRadial/Param`, `GetArtemRadial/Param`) и `VolumeCalc.m` (контуры из 7 точек) |
| 06  | data  | data(mri): контуры с предсердием, valve move, R по объёму  | Перенос из `MRIdata.m` (`HeartContoursWithAtrial`, `AtrialPoints`, `ValveMove`, `RbyMRIbyVolume`)                                                           |
| 07  | data  | data(imp): временной ряд radial/sphere для Artem           | `artemRadialArray`, `artemSphereMoving` из `IMPdata.m` (раскомментировать живые версии)                                                                     |
| 08  | data  | data(moves): копии alex.txt, ivan.txt, artem.txt           | Файлы из `Kernel/core/*.txt` копируются 1-в-1 в `data/moves/`                                                                                               |
| 09  | test  | test(reference): эталон из .nb и комментариев              | `tests/reference_values.json` + парсер `tests/_extract_nb_outputs.py` (необязательный) для извлечения Output-ячеек из `SphereMoveModelling.nb`              |
| 10  | docs  | docs(reference): протокол сборки эталона                   | `docs/REFERENCE_VALUES.md` — какие значения откуда взяты, какие пометки «отсутствует»                                                                       |

После фазы 1: эталонный JSON загружается, `tests/test_data_loaded.py` проверяет, что все пациенты имеют 5-канальные параметры.

---

## Фаза 2 — физическое ядро (этап 2 плана)

| #   | Тип   | Сообщение                                                              | Содержимое                                                                                                                                          |
| --- | ----- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 11  | feat  | feat(models): OneLayerModel, TwoLayerModel                             | `models.py`: `one_layer_model(rho1, a, b)`, `two_layer_model(rho1, rho2, h, a, b)` — прямой перенос формул из `Models.m`                            |
| 12  | test  | test(models): one/two layer против эталона                             | Сверка с `reference_values.json` и с `Round[..., 0.0001]`                                                                                           |
| 13  | feat  | feat(models): SphereModel + SphereInfluence (Лежандр n=0..100)         | `sphere_model(rho1, rho2, a, b, R, h, x, y)` — `Zep`/`SubZ`/`SphereInfluence` через явный цикл по n или `numpy.polynomial.legendre.legval`          |
| 14  | test  | test(models): SphereModel против эталона + сходимость ряда             | Тест на 3 точки + sanity-чек что `n=100` достаточно (residual между n=80 и n=100 < 1e-8)                                                            |
| 15  | feat  | feat(finders): FindRoOne (brentq), FindHTwo (табличный поиск)          | `finders.py` — `FindRoOne` через `scipy.optimize.brentq`, `FindHTwo` через `np.linspace` + знакоперемена (как в `Partition[..., 2, 1]`)             |
| 16  | feat  | feat(finders): FindRSphere/HSphere/RoSphere — табличный поиск          | Сохранить намеренный табличный подход (модель немонотонна — см. PORT_PLAN.md / «Тонкие места»)                                                      |
| 17  | feat  | feat(finders): FindRoSphereNMead (Nelder-Mead)                         | `scipy.optimize.minimize(method='Nelder-Mead')`                                                                                                     |
| 18  | feat  | feat(finders): FindDRSphere + DxDy + Fl + FlDxDy                       | 4 близких варианта одной функции — общий приватный helper `_dr_search`                                                                              |
| 19  | test  | test(finders): self-consistency + эталон                               | После `r = FindRSphere(Zb,...)` проверка `SphereModel(...,r,...) ≈ Zb` с допуском 1e-4                                                              |
| 20  | feat  | feat(radial): RadEvalMethod1..4 + RadEval диспетчер                    | `radial.py` — 4 метода + общий вход `rad_eval(method, ...)`. Сохранить `np.round(..., 4)` (см. «Тонкие места»)                                      |
| 21  | feat  | feat(radial): HCoutourDxDyForEachChannel + DxDyFind                    | Геометрические утилиты для перевода смещения центра сферы в локальные координаты канала                                                              |
| 22  | test  | test(radial): RadEvalMethod1 на Artem против эталона                   | Массив 5 чисел — сверка с числами из `.nb` или с `Round[..., 0.0001]`                                                                              |

После фазы 2: основная физика воспроизведена и протестирована. Это — самый ценный кусок порта.

---

## Фаза 3 — геометрия и объёмы (этап 3 плана)

| #   | Тип   | Сообщение                                                       | Содержимое                                                                                                                                          |
| --- | ----- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 23  | feat  | feat(geometry): GetCoordAfterMove (полярный сдвиг)              | `geometry.py` — `coord_after_move(contour, move)`; перевод декарт↔полярные через `np.hypot/np.arctan2`                                              |
| 24  | feat  | feat(geometry): KubicInterpol (эрмитова кубич. сплайн)          | Прямой перенос `MatrixM`/`MatrixR`/`getP` из `VolumeCalc.m` через `np.linalg.solve`. Опционально — параллельная реализация на `CubicHermiteSpline`  |
| 25  | feat  | feat(geometry): GetLPoint, ListOfParallelSectionForContour      | Параллельные сечения интерполированного контура                                                                                                     |
| 26  | feat  | feat(volume): VolumeBySection (Simpson/конус), VolumeByContour  | `volume.py` — `cut_konus_circle_volume`, `volume_by_section`, `volume_by_contour`                                                                    |
| 27  | feat  | feat(volume): SVbyContourAnd5Move, SVfinder                     | Полный пайплайн стрик-volume → SV                                                                                                                   |
| 28  | test  | test(geometry,volume): VolumeByContour для Ivan против эталона  | Сверка с числом из `.nb` или, если эталона нет — с самосогласованностью (объём ↔ масштаб)                                                           |
| 29  | feat  | feat(geometry): GetContursInTime                                | Эволюция контура во времени по 5-канальной траектории                                                                                              |

После фазы 3: можно посчитать SV для трёх пациентов из контурных данных — без импеданса.

---

## Фаза 4 — подгонка сферы и систола (этап 4 плана)

| #   | Тип   | Сообщение                                                  | Содержимое                                                                                                                                          |
| --- | ----- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 30  | feat  | feat(sphere_fit): MassCenterContour, CircleRadiusByContourSqr | `sphere_fit.py` — центроид через `shapely.geometry.Polygon(...).centroid` и площадной радиус                                                      |
| 31  | feat  | feat(sphere_fit): EqSphNMCenterRadius/Radius/Center        | `scipy.optimize.minimize(method='Nelder-Mead', bounds=...)` — три варианта подгонки                                                                 |
| 32  | feat  | feat(sphere_fit): EqualSphere диспетчер + EqSphMassCenter  | Унифицированный вход `equal_sphere(contour, method, radius_if_need=None)`                                                                            |
| 33  | feat  | feat(sphere_fit): MassCenterTriangle, TrianglesFromContour, AbsolutSphereXYinLocalChannel | Вспомогательные                                                                                                                  |
| 34  | feat  | feat(sistole): SistoleNone, SistoleValveMove, ValvePoint   | `sistole.py` — построение систолического контура из диастолического + смещения                                                                       |
| 35  | feat  | feat(sistole): SistoleContour диспетчер + ValveConturMove  | Учёт движения клапанного кольца параллельно                                                                                                         |
| 36  | test  | test(sphere_fit): эквивалентная сфера для Ivan/Alex/Artem  | Проверяем, что центр и радиус близки к ожидаемым (из `.nb` если есть)                                                                              |

---

## Фаза 5 — верхний пайплайн (этап 5 плана)

| #   | Тип   | Сообщение                                                 | Содержимое                                                                                                                                          |
| --- | ----- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 37  | feat  | feat(pipeline): EqualSphereMoveModelling                  | `pipeline.py` — собирает sistole+sphere fit                                                                                                         |
| 38  | feat  | feat(pipeline): RadialEvaluation (сквозной + verbose)     | Полный конвейер, включая matplotlib-визуализацию при `verbose=True`                                                                                 |
| 39  | feat  | feat(pipeline): TotalModelling (перебор методов)          | Заглавная функция, которая обходит все комбинации методов                                                                                            |
| 40  | feat  | feat(plotting): обёртки matplotlib над ListLinePlot/Show  | `plotting.py` — мини-DSL поверх matplotlib для совместимости с verbose-режимом                                                                      |
| 41  | test  | test(pipeline): RadialEvaluation("Artem", ...) end-to-end | Сравнение полного SV с эталоном из `.nb`                                                                                                            |
| 42  | docs  | docs(notebook): 01_smoke.ipynb                            | Повторение `implem/sphereMove/SphereMoveModelling.nb` на Python                                                                                      |

После фазы 5: основная задача порта решена. Это первая «релизная» точка — можно тегировать `v0.1.0`.

---

## Фаза 6 — DSP и импортёры (этап 6 плана)

| #   | Тип   | Сообщение                                          | Содержимое                                                                                                                                                |
| --- | ----- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 43  | feat  | feat(dsp): FirFilter, FirFilterSZA                 | `dsp.py` — `scipy.signal.fftconvolve` с подгонкой границ как в `DspLib.m`                                                                                  |
| 44  | feat  | feat(dsp): GetR, MyDeriv, MyIntegrate, zeroCrossings | Сигнальные утилиты из `MyFunction.m`                                                                                                                    |
| 45  | feat  | feat(importers): reo32 (Reo32Import, Reo32Cutter, Forma1Reo32) | `importers/reo32.py` — формат каналов (нечёт=пульс, чёт=база, ЭКГ — третий с конца). Хардкод-пути убираются в параметр.                          |
| 46  | feat  | feat(importers): comsol ExtractComsolTable         | `importers/comsol.py`                                                                                                                                     |
| 47  | test  | test(dsp,reo32): синтетика (R-пик на синусе, FIR round-trip) | Реальных REO-32 нет → только синтетика. README помечает «требует валидации».                                                                    |

После фазы 6: тег `v0.2.0`. Импорт сторонних форматов есть, но валидация ждёт реальных данных.

> **Примечание.** Бывшая «фаза 7 — STL-починка (RFFI)» вынесена в
> отдельный репозиторий [`cardio-stl-repair/`](../../cardio-stl-repair/),
> со своим [PORT_PLAN](../../cardio-stl-repair/docs/PORT_PLAN.md) и
> [COMMIT_PLAN](../../cardio-stl-repair/docs/COMMIT_PLAN.md). Здесь её нет.

---

## Фаза 7 — финал

| #   | Тип   | Сообщение                                                  | Содержимое                                                                                                                                          |
| --- | ----- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 48  | docs  | docs: README с быстрым стартом и таблицей соответствий     | Расширить `README.md`: установка, smoke-вызов `radial_evaluation`, таблица «функция Wolfram → функция Python»                                       |
| 49  | docs  | docs: ссылка из родительского WolframMath-master/README.md | Один-единственный апдейт в исходном репозитории — указатель на `cardio-model-py/` и `cardio-stl-repair/`                                            |
| 50  | chore | chore: tag v1.0.0                                          | После того как все пациенты прогнаны и числа сходятся                                                                                              |

---

## Принципы

- **Каждый коммит — зелёный тест.** Если коммит вводит модуль, его тест
  идёт в том же коммите или в следующем (но не позднее).
- **Эталон — раньше реализации.** Фаза 1 даёт `reference_values.json`,
  по которому проверяются все последующие коммиты.
- **Размер коммита.** Один модуль / одна функция верхнего уровня + её
  тест = один коммит. Большие пайплайны (`RadialEvaluation`) разбиваются
  на «скелет + тест-заглушка», потом «реализация», потом «verbose».
- **Сообщения коммитов.** На русском, в стиле существующего репозитория
  (`844b06e Убрать EXE-артефакты ...`). Префикс типа (`feat:`, `test:`, ...)
  на латинице, тело — на русском.
- **Отсутствие эталона — не блокер.** Если для функции нет эталона из
  `.nb`, тест проверяет самосогласованность (round-trip, инверсия,
  размерность) и помечается как `@pytest.mark.no_reference`.

## Оценка по объёму

- Фазы 0–1: ~1–2 дня (инфра + перенос данных).
- Фаза 2 (физическое ядро): ~3–5 дней. Самая аккуратная.
- Фаза 3 (геометрия/объём): ~2–3 дня.
- Фаза 4 (подгонка/систола): ~2 дня.
- Фаза 5 (пайплайн): ~1–2 дня.
- Фаза 6 (DSP+импортёры): ~2 дня.
- Фаза 7 (финал): ~0.5 дня.

Итого ~11–17 дней одиночной работы. STL-починка (~9–11 дней) — в
[`cardio-stl-repair/`](../../cardio-stl-repair/) отдельной задачей.
