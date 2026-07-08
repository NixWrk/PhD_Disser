# Runbook: COMSOL 6.0

Рабочая установка COMSOL 6.0:

```powershell
C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\bin\win64
```

Рабочий prefsdir:

```powershell
C:\tmp\comsol_prefs_allfiles
```

## Компиляция Java

Из корня репозитория:

```powershell
Copy-Item -LiteralPath 'D:\Git_Code\Kardio_respiratory_dudes\Калмыков\COMSOL_investigation_NIX\java\comsol_clean_TTrans_NIX_radius_sweep_6_0.java' -Destination 'C:\tmp\comsol_clean_TTrans_NIX_radius_sweep_6_0.java' -Force

& 'C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\bin\win64\comsolcompile.exe' `
  -jdkroot 'C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\java\win64\jre' `
  'C:\tmp\comsol_clean_TTrans_NIX_radius_sweep_6_0.java'
```

Важно: перед запуском основной Java-файл ожидает STEP в:

```powershell
C:\tmp\TTrans_for_NIX.step
```

Если нужно восстановить из репозитория:

```powershell
Copy-Item -LiteralPath 'D:\Git_Code\Kardio_respiratory_dudes\Калмыков\COMSOL_investigation_NIX\geometry\TTrans_for_NIX.step' -Destination 'C:\tmp\TTrans_for_NIX.step' -Force
```

## Запуск batch

Общий шаблон:

```powershell
& 'C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\bin\win64\comsolbatch.exe' `
  -prefsdir 'C:\tmp\comsol_prefs_allfiles' `
  -inputfile 'C:\tmp\comsol_clean_TTrans_NIX_radius_sweep_6_0.class' `
  -batchlog 'C:\tmp\run.log' `
  -autosave off `
  -recoverydir 'C:\tmp' `
  MODE
```

Важно: запуск без `MODE` раньше давал NPE/неудобное поведение. Всегда указывать режим.

## Экспорт картинок

Текущее правило: каждый новый расчет должен сохранять не только числа и `.mph`,
но и проверочные PNG. Минимальный набор:

- geometry view;
- material selections;
- Terminal/Ground или point source/sink;
- voltage averaging selections;
- отдельный кадр, где видны электроды или выбранные электродные сущности.

Для текущей постановки без нового расчета используется:

```text
java/comsol_export_visualization_6_0.java
```

Готовые PNG лежат в:

```text
visualization/
```

## Однородная серия с электродами

Файл:

```text
java/comsol_homogeneous_electrode_study_6_0.java
```

Смысл: все домены модели задаются как мягкая ткань `rho_soft = 4.728[ohm*m]`,
кроме явных металлических электродов/пэдов. Серия считает несколько вариантов
электродов и для каждого варианта экспортирует `.mph`, CSV-строку и PNG.

Компиляция:

```powershell
Copy-Item -LiteralPath 'D:\Git_Code\Kardio_respiratory_dudes\Калмыков\COMSOL_investigation_NIX\java\comsol_homogeneous_electrode_study_6_0.java' -Destination 'C:\tmp\comsol_homogeneous_electrode_study_6_0.java' -Force

& 'C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\bin\win64\comsolcompile.exe' `
  -jdkroot 'C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\java\win64\jre' `
  'C:\tmp\comsol_homogeneous_electrode_study_6_0.java'
```

Запуск:

```powershell
& 'C:\PC\COMSOL.Multiphysics.6.0.318.Win.Linux.macOS-SSQ\Installed_comsol\COMSOL60\Multiphysics\bin\win64\comsolbatch.exe' `
  -prefsdir 'C:\tmp\comsol_prefs_allfiles' `
  -inputfile 'C:\tmp\comsol_homogeneous_electrode_study_6_0.class' `
  -batchlog 'C:\tmp\comsol_homogeneous_electrode_study_6_0.log' `
  -autosave off `
  -recoverydir 'C:\tmp'
```

Итоговая папка:

```text
homogeneous_electrode_study/
```

Начать просмотр удобно с:

```text
homogeneous_electrode_study/electrode_layout_contact_sheet.png
homogeneous_electrode_study/homogeneous_electrode_summary.csv
```

## Основные режимы

- `point` - точечные источники/измерения; последний вариант перенесен на верх цилиндра руки (`y=-50 мм`, `z=208 мм`).
- `window` - Terminal/Ground на выбранных body-top faces.
- `smallpads` - маленькие сферические pads.
- `delrhquick` - быстрый sweep `delRh = 0..3 мм`.
- `r0only` - грубая калибровка `Rh0`.
- `electrodes` - перебор старых электродных доменов.
- `point_homogeneous`, `window_homogeneous` - однородная диагностика.
- `window_rho_div10`, `window_rho_match_z1`, `window_soft_match_lung_ct` - диагностические подгонки масштаба `rho`.
- `skin_gaussian` - экспериментальный вариант с гауссовыми токовыми пятнами по наружной поверхности; зависал на линейном решателе из-за Neumann-only постановки.

## Probe-файлы

- `comsol_probe_import_TTrans_NIX_6_0.java` - проверка импорта STEP и доменных bbox.
- `comsol_probe_body_surface_TTrans_NIX_6_0.java` - поиск поверхностей soft-body рядом с прежними электродными x.
- `comsol_probe_body_boundary_candidates_6_0.java` - ранжирование candidate-boundaries по координатам около электродов.
- `comsol_probe_point_current_6_0.java` - свойства `PointCurrentSource`.
- `comsol_probe_boundary_current_6_0.java` - свойства `BoundaryCurrentSource`.
- `comsol_probe_normal_current_6_0.java` - проверка `NormalCurrentDensity`; рабочее имя свойства `nJ`.

## Контроль зависших процессов

```powershell
Get-Process | Where-Object { $_.ProcessName -like '*comsol*' -or $_.ProcessName -like '*java*' } | Select-Object Id,ProcessName,CPU,StartTime
```

Остановить конкретный зависший batch:

```powershell
Stop-Process -Id <PID>
```

## Что открыть в GUI

Диагностический файл:

```text
models/TTrans_NIX_point_radius_6_0_diagnostic.mph
```

Он сохранен из COMSOL 6.0 и должен открываться в GUI 6.0. Использовать как материал для инспекции текущего состояния, не как финальную научную модель.

## MATLAB LiveLink черновик

Файл:

```text
matlab/comsol_matlab_fixed_radius_heart_impedance_check.m
```

Смысл: держать геометрию и радиус сердца фиксированными, а `Z1/Z2` пытаться объяснить изменением эффективного сопротивления сердца-сферы. В этой сессии этот MATLAB-сценарий не запускался; перед использованием надо поправить пути/кодировку имени субъекта и указать GUI-cleaned MPH.
