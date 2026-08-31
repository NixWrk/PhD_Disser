# Набор файлов для запуска STL-расчетов в MATLAB

Этот список отвечает на практический вопрос: какие файлы нужны, чтобы запустить
подобный расчет импеданса EIT при загрузке STL-анатомии.

## Минимальный MATLAB-набор

Эти файлы должны лежать в одной папке MATLAB-проекта:

```text
run_stl_impedance_calculation.m
run_forward_multitissue_from_stl.m
tissue_config.m
validate_stl_inputs.m
assign_tissues_from_stl.m
points_in_closed_surface.m
element_centroids.m
default_ring_electrodes.m
electrode_faces_by_area.m
itis_conductivity.m
print_itis_conductivities.m
start_parallel_pool.m
stl_mesh_cache_key.m
```

Для текущей анатомической задачи основной файл запуска:

```matlab
run_stl_impedance_calculation
```

## Файлы для sweep/обоснования упрощения

Чтобы сравнивать влияние разрешения STL на импедансные измерения:

```text
run_all_anatomy_resolution_sweep.m
generate_all_anatomy_resolution_report.m
run_body_resolution_sweep.m
generate_body_resolution_report.m
build_body_mesh_cache.m
```

Главная серия для текущей задачи:

```matlab
surface_mm_values = [6 4 3 2];
run_all_anatomy_resolution_sweep
generate_all_anatomy_resolution_report
```

## Файлы для подготовки STL

Для подготовки/ремеша поверхностей нужны Python-скрипты:

```text
prepare_anatomy_resolutions.py
prepare_body_resolutions.py
prepare_stl_meshlab.py
```

И Python-пакеты:

```powershell
python -m pip install pymeshlab pymeshfix
```

Пример подготовки согласованных разрешений:

```powershell
python prepare_anatomy_resolutions.py --sizes 6 4 3 2
```

## STL-данные

Для нового пациента лучше использовать универсальные имена:

```text
stl/body.stl
stl/lungs.stl
stl/bones.stl
stl/blood.stl
stl/heart.stl      optional
```

Для текущего набора исходники лежат в:

```text
stl/raw/YAROSLAV_BODY.stl
stl/raw/YAROSLAV_LUNGS.stl
stl/raw/YAROSLAV_BONES.stl
stl/raw/YAROSLAV_BLOOD.stl
stl/raw/YAROSLAV_SOFT.stl
```

Подготовленные поверхности и результаты создаются в `output/`.

## Условия запуска

- MATLAB R2022b.
- EIDORS 3.12 подключен к MATLAB path.
- Gmsh доступен для EIDORS.
- Parallel Computing Toolbox желателен, но не обязателен.
- STL-поверхности должны быть замкнутыми и желательно проверенными/починенными.
- Координаты STL должны быть в миллиметрах, как в текущей модели.
- Электроды по умолчанию ставятся кольцом на 50% высоты; для реального
  эксперимента лучше задать измеренные `cfg.electrode_centres_xyz`.

## Что считается

Считается прямая задача EIT. `v.meas` - это transfer impedance response: при
стандартном токе 1 А значения численно соответствуют импедансному отклику.

Ткани сейчас задаются через IT'IS/Gabriel коэффициенты в `itis_conductivity.m`:

- soft tissue: Muscle;
- lungs: Lung (Inflated);
- bones: Bone (Cortical);
- blood: Blood;
- heart: Heart Muscle, если есть STL.
