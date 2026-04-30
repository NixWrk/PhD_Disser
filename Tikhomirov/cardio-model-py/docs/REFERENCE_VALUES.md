# Эталонные значения для верификации порта

## Статус

Файл `tests/reference_values.json` — источник истины для тестов порта.
Числа заполнены на основании:

1. **Константы из кода** — значения `dZRad`, `zBase`, `param_R/y/h` и пр.
   взяты прямо из `RadialEvaluation.m`, `VolumeCalc.m`, `MRIdata.m`.
   Статус: ✅ заполнено.

2. **Результаты функций** — вывод `RadEvalMethod1`, `VolumeByContour` и пр.
   требует запуска `implem/sphereMove/SphereMoveModelling.nb` в Mathematica 11.3.
   Статус: ⏳ `null` — ожидают заполнения.

## Как дополнить эталон

### Если есть доступ к Mathematica

```mathematica
(* Запустить в SphereMoveModelling.nb *)
<< RadialEvaluation`
<< ModelFinders`

RadEvalMethod1[GetArtemRadial[], GetArtemParam[]]
(* → вставить результат в reference_values.json :: rad_eval_method1.Artem *)
```

### Если Mathematica недоступна

Принять результат порта как временный эталон после самосогласованности:
```python
import pytest
# Пометить как @pytest.mark.no_reference
# После валидации вручную — занести в JSON
```

## Источники данных

| Ключ JSON                         | Источник Mathematica                              |
|-----------------------------------|--------------------------------------------------|
| `patients.*.radial_dZRad`         | `GetIvanRadial[]` / `GetAlexRadial[]` etc.       |
| `patients.*.param_R`              | `GetIvanParam[]` etc.                            |
| `patients.*.contour_first`        | `ivan7pointContour` etc. в `VolumeCalc.m`        |
| `mri.*.valve_move`                | `ValveMove[person]` в `MRIdata.m`                |
| `mri.*.r_by_volume`               | `RbyMRIbyVolume[person]` в `MRIdata.m`           |
| `imp.artem_sphere_moving_row0`    | `artemSphereMoving[[1]]` в `IMPdata.m`           |
| `rad_eval_method1.*`              | `RadEvalMethod1[GetXRadial[], GetXParam[]]`      |

## Тонкие места при проверке

- **`Round[..., 0.0001]`** в оригинале: допуск сравнения `atol=1e-4`.
- Контур МРТ с предсердием замкнут: `c[0] == c[-1]` (11 точек, 10 уникальных).
- `artemRadialArrayFix` рассчитывается через деление на максимум пика (строка 9).
- `.txt` файлы движений: `alex.txt`/`ivan.txt` — 5×10, `artem.txt` — 5×9 (разное число точек времени).

## История изменений

| Дата       | Изменение                                        |
|------------|--------------------------------------------------|
| 2026-04-29 | Первая версия: константы из кода, числа радиала ждут .nb |
