# Итоговый handoff: библиотечные М1–М9

Работа остановлена после fullsurface 32768. Новые вычисления не запускаются.
Созданы три собственных файла кода; общий runner, реестр, notebook и report не редактировались.
Коммитов от исполнителя нет. Все файлы разделены ниже на код, third-party и результаты.
`OWNED_FILES.json` содержит те же точные пути, размеры и SHA-256; у самой описи нет самохэша.
`OWNED_FILES.txt` — плоский список с категорией, пригодный для выборочной передачи.

## Условия _02 относительно _01

`ct_fullsurface_32768_01` — НЕ завершённый расчёт другого метода. Подготовка остановилась
при сериализации вложенных NumPy-массивов из MAT source_metadata в JSON. MATLAB не запускался.
`ct_fullsurface_32768_02` — повтор в новом каталоге после исправления только JSON-сериализации
в `tools/heart_library_fits.py`. Физические входы и параметры подгонки не изменены.
Проверены точное равенство массивов выбранных 32768 точек, совпадение всех метаданных случаев,
хэшей полной квадратуры/весов, версии геометрического адаптера и общих файлов кода в _01 и _02.
Хэш контейнера MAT может различаться из-за временной метки заголовка; отдельно сравнивались массивы.
_01 сохранён без перезаписи; завершённые результаты брать только из _02.

Полные квадратуры: Adam 1963950, Nix 914292, Georg 2116200 точек.
Из каждой заново выбраны 32768 уникальных точек детерминированно по полным площадочным весам.
Пилотные 8192 координаты не служили источником новой подгонки.
М1–М6: hyperellipsoidfit 1.6.0.0, lambda=0, normalize=true, HES eta=1;
М7–М8: Ellipsoid fit 1.3.0.0; М9: Sphere Fit 1.1.0.0. Upstream без исправлений.
Один новый MATLAB-батч 3×9, maxNumCompThreads(1), без FEM/parpool; 27/27 допустимы.
Идентификаторы M*/названия библиотек/kind=library_M*/sampling_family=ellipsoid сохранены раздельно.

## Различие двух проверок для source_ru

Проверка 4096→8192 происходила ВНУТРИ уже сокращённой пилотной меры 8192 точек с агрегатными весами.
Она не могла выявить потери исходной полной поверхности. Обе геометрии оценивались на пилотных 8192.

Проверка 8192→32768: новая подгонка происходит из ПОЛНОЙ заново построенной surface_quadrature
исходной маски. Ограничение исходной меры 8192 снято. Обе геометрии, включая прежнюю 8192,
повторно оценены одной текущей версией numeric_metrics на одинаковых свежих 32768 точках.
Это всё ещё оценочная выборка, совпадающая с новой выборкой подгонки, а не интегрирование
по всем 0.91–2.12 млн точек квадратуры и не независимая анатомическая валидация.
Общие версии метрик изменились относительно ПИЛОТА (но не между fullsurface _01/_02);
текущий и прежний хэши сохранены, использованные файлы находятся в _02/code_snapshot.

## Максимальные изменения 8192→32768

| Показатель | Максимум модуля | Случай |
|---|---:|---|
| Центр | 0.365683 мм | Nix, М6 |
| Объём относительно модели 8192 | 0.655886% | Georg, М7 |
| Полуось относительно 8192 | 0.897227% | Nix, М6 |
| Относительная норма разности тензора формы | 0.017619 | Adam, М2 |
| Dice | 0.002752505 | Georg, М2 |
| Ошибка объёма относительно маски | 0.807357 п.п. | Georg, М7 |
| RMS поверхности | 0.046640 мм | Adam, М5 |

Одна смена попарного порядка: Nix, М2 (TAUB) против М6 (2-NORM), RMS.
Разность М2−М6 на единой оценочной выборке составляет +0.008733396 мм для параметров 8192
и −0.009317849 мм для параметров 32768. Это малое преимущество неустойчиво.
Сравнение 4096→8192 ранее давало три смены порядка. Из-за разных оценочных выборок и исходных
мер нельзя использовать числа 3→1 как строгий показатель порядка сходимости.
Максимум изменения центра и объёма снизился, но |ΔDice| вырос с 0.00172845 до 0.00275251.
Достаточность 32768 и окончательный рейтинг НЕ объявлены.

## Главные результаты для включения

- `_02/geometry_comparison/results.json`: 54 повторные оценки, 27 разностей, одна смена порядка.
- `_02/stability_summary.json` и `_02/stability_report.md`: компактные выводы без окончательного рейтинга.
- `_02/full_surface_provenance.json`, `_02/full_source_checks.json`: происхождение и проверка полной выборки.
- `_02/fits_32768/results.json`: исходный MATLAB-ответ, настройки, версии и предупреждения.
- `ct_examples_01/geometry_comparison/results.json`: прежняя проверка 4096/8192, не перезаписана.

Ниже полный перечень owned files; пути относительно корня репозитория. Results сохраняются
локально: среди них есть медицинские геометрические массивы и конфигурационные абсолютные пути;
автоматически включать весь results-каталог в Git нельзя. Third-party содержит оригинальные
лицензии и точные архивы, а также наши dependency manifest/README. Thumbs.db относится к upstream
архиву, не является исходником benchmark и не требуется для запуска.

## Код

- `MATLAB_TRKG4_real_subjects/src/run_heart_library_fits.m`
- `MATLAB_TRKG4_real_subjects/tools/heart_library_fits.py`
- `MATLAB_TRKG4_real_subjects/tools/heart_library_geometry_comparison.py`

## Third-party и паспорт зависимостей

- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/ellipsoid_fit/ellipsoid_fit/ellipsoid_fit.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/ellipsoid_fit/ellipsoid_fit/test_ellipsoid_fit.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/ellipsoid_fit/ellipsoid_fit/test_hyperboloid_fit.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/ellipsoid_fit/license.txt`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/ellipsoid_fit/upstream.zip`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/hyperellipsoidfit/Demo.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/hyperellipsoidfit/hyperellipsoidfit.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/hyperellipsoidfit/license.txt`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/hyperellipsoidfit/upstream.zip`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/method_mapping.json`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/provenance.json`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/README.md`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/license.txt`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/sphereFit/html/sphereFit_Example.html`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/sphereFit/html/sphereFit_Example.png`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/sphereFit/html/sphereFit_Example_01.png`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/sphereFit/html/Thumbs.db`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/sphereFit/sphereFit.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/sphereFit/sphereFit_Example.m`
- `MATLAB_TRKG4_real_subjects/vendor_heart_fits/sphereFit/upstream.zip`

## Результаты и технические артефакты

- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/check_4096/input.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/check_4096/results.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/check_4096/RUN_AFTER_COORDINATOR_APPROVAL.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/check_4096/runtime_filename_alias/ellipsoid_fit_new.m`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/check_4096/shared_points.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/check_4096/source_snapshot.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M1__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M1__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M2__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M2__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M3__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M3__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M4__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M4__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M5__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M5__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M6__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M6__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M7__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M7__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M8__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M8__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M9__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/adam_phase_03_whole_heart__M9__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M1__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M1__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M2__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M2__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M3__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M3__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M4__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M4__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M5__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M5__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M6__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M6__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M7__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M7__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M8__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M8__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M9__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/georg_phase_05_whole_heart__M9__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M1__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M1__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M2__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M2__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M3__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M3__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M4__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M4__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M5__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M5__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M6__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M6__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M7__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M7__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M8__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M8__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M9__4096.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/nix_phase_02_whole_heart__M9__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/geometry_comparison/results.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/handoff_report.md`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/handoff_summary.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/input.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/matlab.log`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/results.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/RUN_AFTER_COORDINATOR_APPROVAL.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/runtime_filename_alias/ellipsoid_fit_new.m`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/shared_points.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_examples_01/source_snapshot.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_01/fits_32768/shared_points.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_01/full_surface_provenance.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_01/full_surface_source.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_additional_ellipsoid_methods.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_additional_sphere_methods.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_affine_geometry.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_construction_variants.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_library_fits.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_library_geometry_comparison.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/heart_reduced_geometry.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/run_heart_library_fits.m`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/code_snapshot/run_heart_method_comparison.py`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/evaluation_clouds_32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/fits_32768/input.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/fits_32768/results.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/fits_32768/RUN_AFTER_COORDINATOR_APPROVAL.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/fits_32768/runtime_filename_alias/ellipsoid_fit_new.m`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/fits_32768/shared_points.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/fits_32768/source_snapshot.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/full_source_checks.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/full_surface_provenance.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/full_surface_source.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M1__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M1__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M2__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M2__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M3__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M3__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M4__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M4__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M5__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M5__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M6__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M6__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M7__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M7__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M8__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M8__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M9__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/adam_phase_03_whole_heart__M9__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M1__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M1__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M2__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M2__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M3__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M3__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M4__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M4__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M5__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M5__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M6__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M6__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M7__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M7__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M8__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M8__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M9__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/georg_phase_05_whole_heart__M9__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M1__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M1__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M2__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M2__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M3__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M3__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M4__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M4__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M5__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M5__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M6__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M6__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M7__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M7__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M8__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M8__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M9__32768.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/nix_phase_02_whole_heart__M9__8192.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/geometry_comparison/results.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/matlab.log`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/REPRODUCE.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/stability_report.md`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/ct_fullsurface_32768_02/stability_summary.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/handoff.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/HANDOFF_FINAL.md`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/handoff_integrity.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/json_ingestion/input.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/json_ingestion/RUN_AFTER_COORDINATOR_APPROVAL.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/json_ingestion/source_snapshot.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/mat_ingestion/input.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/mat_ingestion/RUN_AFTER_COORDINATOR_APPROVAL.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/mat_ingestion/shared_points.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/mat_ingestion/source_snapshot.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/OWNED_FILES.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/OWNED_FILES.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/python_self_test.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/audit.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/input.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/matlab.log`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/matlab_retry.log`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/results.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/RUN_AFTER_COORDINATOR_APPROVAL.txt`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/runtime_filename_alias/ellipsoid_fit_new.m`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/shared_points.mat`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_01/source_snapshot.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_source.json`
- `MATLAB_TRKG4_real_subjects/output/exploratory/heart_library_fits_20260915/synthetic_source.mat`
