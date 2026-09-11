# Критический аудит КТ, 4D-сегментации и перехода к УО/ФВ

Дата: 11.09.2026. Репозиторий: D:/Аспа/Kalmykov_PhD.

Аудит выполнен только чтением. Файлы репозитория не изменялись, Git mutations не выполнялись, ноутбуки, тяжёлые модели и тестовые наборы не запускались. Приватные бинарные массивы не копировались. Единственный созданный файл — этот отчёт во временном каталоге вне репозитория. Understand Anything не использовался.

Основной агент сообщил о проверке JSON/AST всех 62 текущих ipynb, 34 passed в pytest Colab и DOM-проверке корневых HTML. Эти результаты не выдаются за проверки данного аудита и после сообщения не повторялись. Коммиты e864a4a, 2c4e0ef, 99634bb сообщены основным агентом как фиксация того же содержания без научных правок.

Работа объединяет собственное чтение научного ядра 21.03 и два разрешённых параллельных аудита: Carver — root20, история и upstream timing; Schrodinger — конверсия, сегментация, QC, validation и pilot. Старые аудиты и сохранённые отметки passed использованы только как исторические артефакты. Ключевые locators и несколько дефектов исполнителей дополнительно сопоставлены с текущим кодом.

## Основной научный вывод

Текущий корпус поддерживает исследовательское описание объёмов автоматических масок, их геометрии, межфазных разностей и условных периодических оценок. Он не устанавливает независимую точность эндокардиальных объёмов, правильность индивидуальных ED/ES и валидированность УО/ФВ. Это ограничение исследовательской постановки, а не доказанная ошибочность всех чисел.

Последнее решение автора учтено: новые ручные маски не ожидаются из-за недостаточной различимости границ; работа продолжается на существующих TotalSegmentator. Остаточный контраст Георга остаётся фактом, но не доказательством достаточного качества границ. R–R Георга 491 и 1159 мс должны оставаться раздельными. Статическая ручная сегментация Inobitec не является автоматически покадровым независимым референсом этой ветки.

Отмена нового ручного входа не превращает геометрию масок, согласие алгоритмов, нормирование на BSA или близость к МРТ-нормам в независимую валидацию. В нынешнем 21.03 это в основном сформулировано корректно.

## 1. Ведомость охвата и уровень чтения

«Полностью» означает последовательное статическое чтение текста/исходника, а не запуск и не анатомическую проверку каждого среза. Для ноутбуков отдельно прочитаны Markdown и code source; крупные графические outputs рассматриваются по указанному ниже уровню.

### Правила и корневые документы

- [AGENTS.md](<../../../AGENTS.md>) — полностью, 603 строки.
- [AGENTS.md](<../../AGENTS.md>) — полностью, 321 строка; более глубоких AGENTS в cardiac-участке не найдено.
- [MATLAB AGENTS](<../../../MATLAB_TRKG4_real_subjects/AGENTS.md>) — полностью, 81 строка.

| Файл | Назначение | Охват |
|---|---|---|
| [20.01_DICOM_QC_и_геометрия_тома.ipynb](<../../20.01_DICOM_QC_и_геометрия_тома.ipynb>) | Выбор DICOM-серии, геометрический QC, манифест принятия | Все 12 ячеек: 7 Markdown, 5 code; outputs отсутствуют. Ядро непосредственно в ноутбуке: read_series, selector_matches, select_one_series, geometry_manifest. |
| [20.02_Регистрация_Inobitec_и_электродов.ipynb](<../../20.02_Регистрация_Inobitec_и_электродов.ipynb>) | Передача КТ, ручной анатомии и электродов через регистрацию | Все 12 ячеек: 7 Markdown, 5 code; outputs отсутствуют. validate_transform, read_electrodes, build_registration_manifest прочитаны. |
| [20.03_HU_и_доля_воздуха_в_лёгком.ipynb](<../../20.03_HU_и_доля_воздуха_в_лёгком.ipynb>) | Контракт будущего HU/воздушности | Все 8 Markdown-ячеек. Исполняемого расчёта нет; не считать готовым результатом. |
| [20.04_Сценарии_электрических_свойств_лёгкого.ipynb](<../../20.04_Сценарии_электрических_свойств_лёгкого.ipynb>) | Контракт электрических сценариев | Все 10 Markdown-ячеек. Исполняемого расчёта нет. |

Связанные [config/ct_paths.example.json](<../../config/ct_paths.example.json>); [config/README.md](<../../config/README.md>); [params/ct.json](<../../params/ct.json>); [params/README.md](<../../params/README.md>); [34.03_КТ_как_референс_разработки.md](<../../34.03_КТ_как_референс_разработки.md>); [comsol_matlab_fixed_radius_heart_impedance_check.m](<../../comsol_matlab_fixed_radius_heart_impedance_check.m>) прочитаны полностью. Последний файл — исторический потребитель rho2_ct, не действующее ядро серии. [trkg4_subject_registry.m](<../../../MATLAB_TRKG4_real_subjects/src/trkg4_subject_registry.m>); [trkg4_load_electrode_centres.m](<../../../MATLAB_TRKG4_real_subjects/src/trkg4_load_electrode_centres.m>) прочитаны полностью как смежные интерфейсы анатомии и электродов.

Отдельные генераторы root20 не найдены поиском имён, функций и версий алгоритмов в доступных неархивных исходниках. Это результат ограниченного поиска, не утверждение об отсутствии генераторов во всех private snapshots. Локальный ct_paths.local.json отсутствует, KALMYKOV_CT_CONFIG не задана; внешние принятые манифесты 20.01–20.02 поэтому не проверены.

### Документы cardiac_4d_segmentation

| Файл/группа | Назначение и уровень чтения |
|---|---|
| [README.md](<../../cardiac_4d_segmentation/README.md>); [21.00_Карта_4D_сердца_и_RR.md](<../../cardiac_4d_segmentation/21.00_Карта_4D_сердца_и_RR.md>) | Карта происхождения, запуска и статусов; полностью. |
| [20.05_4D_КТ_сегментация_сердца.md](<../../cardiac_4d_segmentation/20.05_4D_КТ_сегментация_сердца.md>); [20.06_4D_КТ_сегментация_v2_полное_сердце.md](<../../cardiac_4d_segmentation/20.06_4D_КТ_сегментация_v2_полное_сердце.md>); [20.07_Статическая_КТ_полная_сегментация_TotalSegmentator.md](<../../cardiac_4d_segmentation/20.07_Статическая_КТ_полная_сегментация_TotalSegmentator.md>) | Протоколы v1, v2 и статической анатомии; полностью. |
| [21.01_4D_сердце_интерактивно.ipynb](<../../cardiac_4d_segmentation/21.01_4D_сердце_интерактивно.ipynb>) | Все 7 ячеек source. В code-ячейках 2/4/6 — комментарии и сохранённые Plotly outputs; вычисления находятся в генераторах. Графические payload не изучались побайтно и все ракурсы интерактивно не просматривались. |
| [21.02_Интерактивная_4D_визуализация_и_анатомия_TotalSegmentator.md](<../../cardiac_4d_segmentation/21.02_Интерактивная_4D_визуализация_и_анатомия_TotalSegmentator.md>) | Методология визуализации и её ограничения; полностью. |
| [21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb](<../../cardiac_4d_segmentation/21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb>) | Все 36 ячеек source, существенные текстовые и табличные outputs, связи с генераторами. Code-ячейки являются контейнерами сохранённых результатов, а не самодостаточным расчётом. |
| [21.04_heart_subtraction_and_RR_report.md](<../../cardiac_4d_segmentation/21.04_heart_subtraction_and_RR_report.md>); [21.05_Валидация_анатомической_сегментации_сердца_4D_КТ.md](<../../cardiac_4d_segmentation/21.05_Валидация_анатомической_сегментации_сердца_4D_КТ.md>); [21.06_Точная_сегментация_крови_четырёх_камер_4D_КТ.md](<../../cardiac_4d_segmentation/21.06_Точная_сегментация_крови_четырёх_камер_4D_КТ.md>); [21.07_Бенчмарк_автоматизированной_сегментации_крови_4D_КТ.md](<../../cardiac_4d_segmentation/21.07_Бенчмарк_автоматизированной_сегментации_крови_4D_КТ.md>) | Исторические протоколы анализа, validation, точной сегментации и benchmark; полностью. Их ранние планы не автоматически текущие. |
| [ОБСУЖДЕНИЕ_GPT_2026-09-09.md](<../../cardiac_4d_segmentation/ОБСУЖДЕНИЕ_GPT_2026-09-09.md>); [heart_rr_analysis/README.md](<../../cardiac_4d_segmentation/heart_rr_analysis/README.md>); [4D_сердце_интерактивно/README.md](<../../cardiac_4d_segmentation/4D_сердце_интерактивно/README.md>) | Полностью. В обсуждении прочитаны 193 строки локального конспекта, включая позднее уточнение; исходный внешний 1622-строчный диалог не читался. |
| [21.01_4D_сердце_интерактивно.html](<../../cardiac_4d_segmentation/21.01_4D_сердце_интерактивно.html>); [21.03_Вычитание_крови_и_объёмы_камер_RR.html](<../../cardiac_4d_segmentation/21.03_Вычитание_крови_и_объёмы_камер_RR.html>) | Производные exports: проверены происхождение/связи и существенный контент через notebook/generator; полного визуального просмотра всех HTML нет. |

### Авторский код cardiac: поимённый охват

Все перечисленные далее группы прочитаны полностью статически, кроме явно указанных ограничений визуального кода.

- Конверсия и регистрация: [cardiac4d.py](<../../cardiac_4d_segmentation/cardiac4d.py>); [spj_static_series_to_nifti.py](<../../cardiac_4d_segmentation/spj_static_series_to_nifti.py>); [build_v2_cardiac_products.py](<../../cardiac_4d_segmentation/build_v2_cardiac_products.py>); [temporal_registration.py](<../../cardiac_4d_segmentation/temporal_registration.py>). Проверялись HU, сетки/affine, фазы, единицы, перенос масок и происхождение.
- Сегментация: [segment_totalseg_v1.py](<../../cardiac_4d_segmentation/segment_totalseg_v1.py>); [segment_totalseg_v2.py](<../../cardiac_4d_segmentation/segment_totalseg_v2.py>); [segment_totalseg_anatomy.py](<../../cardiac_4d_segmentation/segment_totalseg_anatomy.py>).
- Первичный QC и STL: [render_qc.py](<../../cardiac_4d_segmentation/render_qc.py>); [render_static_anatomy_qc.py](<../../cardiac_4d_segmentation/render_static_anatomy_qc.py>); [inventory_reference_stl.py](<../../cardiac_4d_segmentation/inventory_reference_stl.py>); [summarize_cohort_qc.py](<../../cardiac_4d_segmentation/summarize_cohort_qc.py>); [summarize_v2_cohort_qc.py](<../../cardiac_4d_segmentation/summarize_v2_cohort_qc.py>); [summarize_anatomy_qc.py](<../../cardiac_4d_segmentation/summarize_anatomy_qc.py>); [summarize_static_segmentation_qc.py](<../../cardiac_4d_segmentation/summarize_static_segmentation_qc.py>).
- Инспекция и timing: [inspect_cardiac_dicom_metadata.py](<../../cardiac_4d_segmentation/inspect_cardiac_dicom_metadata.py>); [inspect_spj_embedded_dicoms.py](<../../cardiac_4d_segmentation/inspect_spj_embedded_dicoms.py>); [diagnose_spj_pixel.py](<../../cardiac_4d_segmentation/diagnose_spj_pixel.py>); [scan_dicom_series_catalog.py](<../../cardiac_4d_segmentation/scan_dicom_series_catalog.py>); [summarize_cardiac_phase_tags.py](<../../cardiac_4d_segmentation/summarize_cardiac_phase_tags.py>); [extract_ecg_report_images.py](<../../cardiac_4d_segmentation/extract_ecg_report_images.py>); [extract_deidentified_ecg_trace.py](<../../cardiac_4d_segmentation/extract_deidentified_ecg_trace.py>); [summarize_ecg_report_ocr.py](<../../cardiac_4d_segmentation/summarize_ecg_report_ocr.py>).
- Научное ядро и главный генератор: [analyze_subtraction_and_rr.py](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py>); [cardiac_function_analysis.py](<../../cardiac_4d_segmentation/cardiac_function_analysis.py>); [cardiac_joint_dynamics.py](<../../cardiac_4d_segmentation/cardiac_joint_dynamics.py>); [cardiac_reference_indexing.py](<../../cardiac_4d_segmentation/cardiac_reference_indexing.py>); [cardiac_notebook_text.py](<../../cardiac_4d_segmentation/cardiac_notebook_text.py>); [build_cardiac_function_report.py](<../../cardiac_4d_segmentation/build_cardiac_function_report.py>). Константы научного текста дополнительно сопоставлены с соответствующими уже прочитанными ячейками 21.03; совпадение проверено. Активные вычислительные пути прочитаны углублённо.
- Отчёт и визуализация: [build_interactive_cardiac4d.py](<../../cardiac_4d_segmentation/build_interactive_cardiac4d.py>); [build_cardiac_ct_review.py](<../../cardiac_4d_segmentation/build_cardiac_ct_review.py>) — полностью; [build_cardiac_joint_report.py](<../../cardiac_4d_segmentation/build_cardiac_joint_report.py>); [cardiac_synchronized_view.py](<../../cardiac_4d_segmentation/cardiac_synchronized_view.py>); [cardiac_volume_time_plots.py](<../../cardiac_4d_segmentation/cardiac_volume_time_plots.py>) — полностью научные/временные пути, тексты и связи; реализация некоторых графических фрагментов проверялась выборочно. Полного визуального тестирования браузера нет.
- Validation и сравнение кандидатов: [validate_manual_cardiac_masks.py](<../../cardiac_4d_segmentation/validate_manual_cardiac_masks.py>); [build_cardiac_validation_candidates.py](<../../cardiac_4d_segmentation/build_cardiac_validation_candidates.py>); [benchmark_blood_segmentation_candidates.py](<../../cardiac_4d_segmentation/benchmark_blood_segmentation_candidates.py>); [pilot_blood_segmentation_candidates.py](<../../cardiac_4d_segmentation/pilot_blood_segmentation_candidates.py>); [prepare_blood_chamber_review.py](<../../cardiac_4d_segmentation/prepare_blood_chamber_review.py>). Углублённо: независимость, метрики, bootstrap, no-reference режимы и provenance.
- Пилот Георга: [georg_ventricular_mask_pilot.py](<../../cardiac_4d_segmentation/georg_ventricular_mask_pilot.py>); [prepare_georg_prompt_pilot.py](<../../cardiac_4d_segmentation/prepare_georg_prompt_pilot.py>); [build_georg_pilot_review.py](<../../cardiac_4d_segmentation/build_georg_pilot_review.py>); [summarize_georg_mask_pilot.py](<../../cardiac_4d_segmentation/summarize_georg_mask_pilot.py>).

Это покрывает все 42 Python-файла верхнего уровня cardiac по указанному уровню, а не только совпадения поиска.

### Конфигурации, тесты и сохранённые outputs

Полностью прочитаны [.gitignore](<../../cardiac_4d_segmentation/.gitignore>); [config.example.json](<../../cardiac_4d_segmentation/config.example.json>); [requirements-base.txt](<../../cardiac_4d_segmentation/requirements-base.txt>); [requirements-totalsegmentator-v1.txt](<../../cardiac_4d_segmentation/requirements-totalsegmentator-v1.txt>); [requirements-totalsegmentator-v2.txt](<../../cardiac_4d_segmentation/requirements-totalsegmentator-v2.txt>); [requirements-visualization.txt](<../../cardiac_4d_segmentation/requirements-visualization.txt>); [manual_validation_manifest.example.json](<../../cardiac_4d_segmentation/manual_validation_manifest.example.json>); [benchmark_blood_segmentation_manifest.example.json](<../../cardiac_4d_segmentation/benchmark_blood_segmentation_manifest.example.json>); [blood_chamber_review_manifest.example.json](<../../cardiac_4d_segmentation/blood_chamber_review_manifest.example.json>); [pilot_blood_segmentation_manifest.example.json](<../../cardiac_4d_segmentation/pilot_blood_segmentation_manifest.example.json>); [pilot_blood_segmentation_manifest.local.json](<../../cardiac_4d_segmentation/pilot_blood_segmentation_manifest.local.json>). Локальный manifest прочитан как конфигурация; медицинские массивы по его ссылкам не копировались.

Все 11 файлов в tests прочитаны без запуска: [test_cardiac4d.py](<../../cardiac_4d_segmentation/tests/test_cardiac4d.py>); [test_spj_static_series_to_nifti.py](<../../cardiac_4d_segmentation/tests/test_spj_static_series_to_nifti.py>); [test_cardiac_function_analysis.py](<../../cardiac_4d_segmentation/tests/test_cardiac_function_analysis.py>); [test_cardiac_joint_dynamics.py](<../../cardiac_4d_segmentation/tests/test_cardiac_joint_dynamics.py>); [test_cardiac_reference_indexing.py](<../../cardiac_4d_segmentation/tests/test_cardiac_reference_indexing.py>); [test_validate_manual_cardiac_masks.py](<../../cardiac_4d_segmentation/tests/test_validate_manual_cardiac_masks.py>); [test_build_cardiac_validation_candidates.py](<../../cardiac_4d_segmentation/tests/test_build_cardiac_validation_candidates.py>); [test_benchmark_blood_segmentation_candidates.py](<../../cardiac_4d_segmentation/tests/test_benchmark_blood_segmentation_candidates.py>); [test_pilot_blood_segmentation_candidates.py](<../../cardiac_4d_segmentation/tests/test_pilot_blood_segmentation_candidates.py>); [test_prepare_blood_chamber_review.py](<../../cardiac_4d_segmentation/tests/test_prepare_blood_chamber_review.py>); [test_georg_mask_pilot.py](<../../cardiac_4d_segmentation/tests/test_georg_mask_pilot.py>).

Три QA-скрипта отчётов прочитаны полностью без запуска: [joint_dynamics/qa/verify_reader.py](<../../cardiac_4d_segmentation/heart_rr_analysis/function_analysis/joint_dynamics/qa/verify_reader.py>); [joint_dynamics/qa/scientific_layout/verify_scientific_layout.py](<../../cardiac_4d_segmentation/heart_rr_analysis/function_analysis/joint_dynamics/qa/scientific_layout/verify_scientific_layout.py>); [reference_indexing/qa/verify_reference_indexing.py](<../../cardiac_4d_segmentation/heart_rr_analysis/function_analysis/reference_indexing/qa/verify_reference_indexing.py>). Их старые ожидаемые числа ячеек и passed-результаты не приняты за текущую проверку.

Прочитаны существенные CSV/JSON текущего function/joint/reference-indexing: базовые объёмы и изменения, геометрия, оценки функций, профили, пространственные разности, таблицы BSA/норм и манифесты. Полностью прочитаны cardiac_gating_source_data.csv, cardiac_dicom_metadata_by_subject.csv и ecg_traces/manifest.json. Проверены арифметические тождества и согласованность выходов, перечисленные ниже; исходные DICOM/NIfTI не пересчитывались.

В Georg pilot выборочно прочитаны final_reader.json, pilot_comparison.json, visual_review_decisions.json: происхождение, статусы, интерпретация и показанные существенные числа. Полностью прочитаны guided/phase_04_heart_ventricle_left/review.json и qa/final_checks.json. Остальные индивидуальные reviews и два review_summary — инвентаризация.

MIGRATION_4D_RR_2026-09-09.json проверен как карта переносов/происхождения; полного повторного чтения всех снимков нет. Четыре language-review JSON рассмотрены как исторические артефакты; текущий function_analysis language-review прочитан полностью, остальные выборочно. Их оценки не заменяют аудит текста.

Визуально просмотрены два реальных сохранённых изображения: CT-review Nix phase_03 overview.png и карта изменений Georg phase_04→phase_05 ЛЖ changes.png. Это ограниченный просмотр иллюстраций, не экспертная анатомическая проверка. Изображение Adam phase_09 успешно не отобразилось и не считается просмотренным.

Инвентаризация производных деревьев: heart_rr_analysis — 21 CSV, 52 HTML, 65 JSON, 12325 PNG, 3 QA .py, 2 log, 7 txt и README; 4D_сердце_интерактивно — 3 HTML, 2 CSV, 2 JSON и README; локальный derived — 4 JSON и 2 PNG. Эти числа не означают полного чтения всех артефактов. Вспомогательные logs/txt, неуказанные manifests и все PNG не проверялись целиком.

Vendor, .venv/.venv-v2/.venv-nninteractive, веса, model cache и приватные анатомические массивы — только инвентаризация либо не читались. Внешние manifests запусков не сверялись с фактическими медицинскими файлами; библиотеки не устанавливались. Литература по ссылкам заново не проверялась.

### Исторический КТ-корпус

- [20.90_Скетч_пороговой_КТ_сегментации.ipynb](<../../archive/legacy/20.90_Скетч_пороговой_КТ_сегментации.ipynb>); [README.md](<../../archive/legacy/README.md>) — полностью; в 20.90 одна provenance-pointer ячейка.
- Через read-only git show прочитан исходный Colab Notebooks/10_КТ_оценка_HU.ipynb в 1ac7813654b59861e359cffe3ad40941baeab2d9: все 23 ячейки и единственный stream-output конфигурации. SHA-256 совпал с миграционным манифестом.
- Исторический объединённый 20.01 в fff4d3a: покрыты все 24 ячейки, изменённые прочитаны, неизменённые точно сопоставлены с прочитанным источником; outputs отсутствуют.
- Старые потребители 05/09/11 целиком не читались: проверены связи из КТ-источника/манифеста.

В [3D_Slicer](<../../../3D_Slicer>) полностью прочитаны [README.md](<../../../3D_Slicer/README.md>); [main.py](<../../../3D_Slicer/main.py>); [01_preprocess.py](<../../../3D_Slicer/01_preprocess.py>); [02_segment_batch.py](<../../../3D_Slicer/02_segment_batch.py>); [03_postprocess.py](<../../../3D_Slicer/03_postprocess.py>); [04_qc.py](<../../../3D_Slicer/04_qc.py>); [05_export_comsol.py](<../../../3D_Slicer/05_export_comsol.py>); [06_visualize_qc.py](<../../../3D_Slicer/06_visualize_qc.py>); [09_cohort_summary.py](<../../../3D_Slicer/09_cohort_summary.py>); [tests/test_pipeline.py](<../../../3D_Slicer/tests/test_pipeline.py>); [config.json](<../../../3D_Slicer/config.json>); [requirements.txt](<../../../3D_Slicer/requirements.txt>); [.gitignore](<../../../3D_Slicer/.gitignore>); [setup_env.bat](<../../../3D_Slicer/setup_env.bat>). В [gui.py](<../../../3D_Slicer/gui.py>) прочитаны только строки 48–105 и 408–512. Каталог data отсутствует; численные результаты этой ветки не проверены.

Private snapshots 21.03_before_function_analysis_20260909, 21.03_before_georg_pilot_20260910, 21.03_before_joint_dynamics_20260910, 21.03_before_reference_indexing_20260910, 21.03_before_scientific_layout_20260910, cardiac_docs_before_20260910 и commit_cardiac_20260910 только инвентаризированы. Их содержание не выдаётся за проверенное.

## 2. Двенадцать приоритетных проблем

P1 — существенный риск для происхождения/геометрии/научной интерпретации; P2 — важный дефект инструмента, QC или актуальности. «Доказанный дефект» ниже означает подтверждённое поведение кода или противоречие сохранённого текста. Реальное повреждение медицинских данных не подразумевается без отдельного указания.

### 1. P1. Сегментации переиспользуются без проверки исходной КТ и запуска модели

**Тип:** доказанный дефект воспроизводимости; текущая подмена масок не установлена.

**Locator:** [segment_totalseg_v2.py:129](<../../cardiac_4d_segmentation/segment_totalseg_v2.py#L129>), manifest [строка 183](<../../cardiac_4d_segmentation/segment_totalseg_v2.py#L183>); [segment_totalseg_v1.py:107](<../../cardiac_4d_segmentation/segment_totalseg_v1.py#L107>).

В v2 наличие файлов и совпадение сетки достаточно для reused_existing_output. Хэши CT, модели и исходного запуска не сверяются. Затем manifest содержит текущую среду/версию, а для reused-фазы — только её id и статус. В v1 при reuse нет даже проверки сетки.

**Последствие:** CT той же формы и affine либо изменённая модель могут получить старые маски с новой атрибуцией окружения. Поздняя проверка geometry cache этого не исправляет: она хеширует уже выбранную маску.

**Предложение:** переиспользовать только результат с сохранённым исходным manifest и совпавшими CT/model/parameters/output hashes; происхождение reused-запуска сохранять отдельно от текущего вызова.

### 2. P1. При DICOM/SPJ→NIfTI теряется смещение срезов внутри плоскости

**Тип:** доказанный дефект для соответствующего класса геометрии; наличие таких серий в текущем корпусе не проверено.

**Locators:** [cardiac4d.py:364](<../../cardiac_4d_segmentation/cardiac4d.py#L364>), [affine:394](<../../cardiac_4d_segmentation/cardiac4d.py#L394>), [spj_static_series_to_nifti.py:146](<../../cardiac_4d_segmentation/spj_static_series_to_nifti.py#L146>). Связанный неполный QC: [20.01:354](<../../20.01_DICOM_QC_и_геометрия_тома.ipynb#L354>), cell id geometry-code-2001.

Проверяются проекции ImagePositionPatient на нормаль, а третья ось affine задаётся normal × slice_spacing. Позиции (0,0,0), (1,0,1), (2,0,2) при аксиальной ориентации проходят такую проверку, но сдвиг по первой координате теряется. Root20 также сохраняет сокращённую геометрию без полных позиций/origin.

**Последствие:** искажение формы и взаимного положения срезов, расстояний и регистрации при незамеченной сдвиговой геометрии. Не следует из этого автоматически объявлять объёмы нынешнего корпуса неверными.

**Предложение:** проверять полный вектор межсрезового смещения и совместимость с выбранным affine; неподдерживаемую геометрию явно блокировать либо корректно преобразовывать.

### 3. P1. Передача данных между этапами не фиксирует однозначную серию/сессию

**Тип:** доказанные пробелы проверок происхождения; смешение текущих сессий не доказано.

**Root20:** [20.02:269](<../../20.02_Регистрация_Inobitec_и_электродов.ipynb#L269>), cell id manifest-logic-2002: проверяются версия, статус и subject_id. Текущие selected_series и DICOM-набор не сверяются; [на строке 488](<../../20.02_Регистрация_Inobitec_и_электродов.ipynb#L488>) хеш копируется из принятого манифеста. Другую серию того же человека этот handoff самостоятельно не обнаружит.

**4D timing:** [summarize_cardiac_phase_tags.py:122](<../../cardiac_4d_segmentation/summarize_cardiac_phase_tags.py#L122>) агрегирует все DICOM каталога; [строка 243](<../../cardiac_4d_segmentation/summarize_cardiac_phase_tags.py#L243>) сохраняет хеш первого UID без проверки единственности серии. Private GE-теги читаются по фиксированным адресам без проверки подходящего private creator. [summarize_ecg_report_ocr.py:55](<../../cardiac_4d_segmentation/summarize_ecg_report_ocr.py#L55>) связывает OCR/растровую кривую по subject; [extract_deidentified_ecg_trace.py:77](<../../cardiac_4d_segmentation/extract_deidentified_ecg_trace.py#L77>) использует жёсткое series307 и имя PNG без её хеша.

**Downstream:** [analyze_subtraction_and_rr.py:1247](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1247>) задаёт независимые default-пути timing, которые не меняются при одном --derived-root; [строка 1278](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1278>) и [строка 1343](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1343>) связывают сведения по субъекту/номинальному проценту фазы, без доказательства общей исходной серии.

**Последствие:** одноимённые фазы того же человека могут получить чужие RR, миллисекунды или КТ-привязку. Это затрагивает смысл ED/ES и межфазных изменений раньше любых формул УО.

**Предложение:** единый идентификатор acquisition/series с проверяемой цепочкой hashes и явной семантикой GE-тегов во всех manifests; принадлежность OCR-страницы и изображения кривой — часть этой цепочки.

**Связанный нюанс приватности:** [summarize_cardiac_phase_tags.py:138](<../../cardiac_4d_segmentation/summarize_cardiac_phase_tags.py#L138>) сохраняет исходный SeriesInstanceUID в общем словаре, экспортируемом [на строке 245](<../../cardiac_4d_segmentation/summarize_cardiac_phase_tags.py#L245>), одновременно с хешированным UID. Поэтому весь JSON нельзя считать обезличенным только по наличию series_uid_sha256. Публикация/утечка не установлена.

### 4. P1. Пространственный QC выбранного кандидата ED теряется в таблице УО/ФВ

**Тип:** доказанный дефект передачи QC; анатомическое усечение и величина его ошибки не доказаны.

**Locators:** [cardiac_function_analysis.py:87](<../../cardiac_4d_segmentation/cardiac_function_analysis.py#L87>), [estimates:122](<../../cardiac_4d_segmentation/cardiac_function_analysis.py#L122>), [таблица:243](<../../cardiac_4d_segmentation/build_cardiac_function_report.py#L243>); [chamber_geometry.csv:39](<../../cardiac_4d_segmentation/heart_rr_analysis/function_analysis/chamber_geometry.csv#L39>), [ventricular_function_estimates.csv:27](<../../cardiac_4d_segmentation/heart_rr_analysis/function_analysis/ventricular_function_estimates.csv#L27>).

Adam, ПЖ, phase_09, 98%: volume=217.297410 мл, touches_image_boundary=True. Этот же объём выбран максимумом/кандидатом ED; выводятся SV=88.561106 мл и EF=40.755712%. Признак касания границы не переносится в строку оценки и читательскую таблицу. Общий статус available_mask_extrema_not_confirmed_ed_es присутствует, но не объясняет конкретную проблему выбранной фазы.

**Последствие:** пользователь видит условную оценку без важного установленного пространственного предупреждения. Замкнутая поверхность после padding не доказывает полноту анатомической полости.

**Предложение:** передавать QC обеих выбранных фаз, показывать его рядом с оценкой; отдельно разбирать влияние пограничной маски на максимум и разность. Не объявлять маску усечённой до просмотра границы.

### 5. P2. Экспортированные фазовые подписи противоречат направлению собственной кривой

**Тип:** доказанный дефект текущего результата.

**Locators:** [analyze_subtraction_and_rr.py:34](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L34>), [phase_band_label:313](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L313>), [chamber_volume_changes_closed_cycle.csv:143](<../../cardiac_4d_segmentation/heart_rr_analysis/chamber_volume_changes_closed_cycle.csv#L143>) и [строка 147](<../../cardiac_4d_segmentation/heart_rr_analysis/chamber_volume_changes_closed_cycle.csv#L147>).

Интервалы 0–45%, 45–80%, 80–100% жёстко названы уменьшением объёма желудочков, ранним/средним наполнением и поздней диастолой. У Adam ЛЖ при 40% модельный объём увеличивается на +0.607309 мл за шаг, при 44% на +1.279055 мл, но оба имеют метку «Уменьшение объёма желудочков».

**Последствие:** метаданные экспортируют неверное направление изменения и незаверенную физиологическую фазу. Это не исправляется оговоркой, что клапанные события не определялись.

**Предложение:** отделить координату R–R и вычисленное направление изменения от гипотезы механической фазы. Индивидуальную клапанную/диастолическую разметку не получать из фиксированных процентных полос.

### 6. P2. STAPLE может представить повторения одного эксперта как независимый консенсус

**Тип:** доказанный дефект проверки сохранённого validation-инструмента; текущий независимый референс не заявлен существующим.

**Locator:** [validate_manual_cardiac_masks.py:827](<../../cardiac_4d_segmentation/validate_manual_cardiac_masks.py#L827>); проверки independence [строка 724](<../../cardiac_4d_segmentation/validate_manual_cardiac_masks.py#L724>); описание консенсуса [строка 1181](<../../cardiac_4d_segmentation/validate_manual_cardiac_masks.py#L1181>).

Требуются три различных annotation_id, но не три независимых эксперта/источника. Три сеанса одного эксперта проходят условие. Проверка независимости аннотации от кандидата не доказывает независимость аннотаций между собой.

**Последствие:** внутриэкспертная повторяемость может стать «консенсусом независимых аннотаций». Дублирование источника также не исключается данным условием.

**Предложение:** различать эксперт, сеанс, исходная маска и зависимости; отдельные статусы intra-rater/inter-rater/adjudicated. Эта находка не является предложением вернуть отменённую ручную разметку как обязательный следующий шаг.

### 7. P2. Отдельный Georg reader повторяет отменённое решение автора

**Тип:** доказанное противоречие текущего генератора/standalone-артефакта; основной 21.03 имеет историческую оговорку.

**Locators:** [summarize_georg_mask_pilot.py:96](<../../cardiac_4d_segmentation/summarize_georg_mask_pilot.py#L96>) и [строка 134](<../../cardiac_4d_segmentation/summarize_georg_mask_pilot.py#L134>); [final_reader.json](<../../cardiac_4d_segmentation/heart_rr_analysis/function_analysis/georg_pilot_20260910/final_reader.json>); актуальное правило [Colab AGENTS.md:250](<../../AGENTS.md#L250>).

Reader сообщает как подтверждённый автором факт нормальную различимость крови/миокарда у Георга и ставит новые исправленные маски следующим шагом перед УО/ФВ. Позднее автор отменил новые ручные маски из-за недостаточного контраста.

**Граница находки:** [build_cardiac_function_report.py:394](<../../cardiac_4d_segmentation/build_cardiac_function_report.py#L394>) уже добавляет перед приложением Б явную историческую оговорку; поэтому неверно утверждать, что весь нынешний 21.03 требует ручную разметку. Проблема остаётся в самостоятельном reader и его генераторе.

**Предложение:** переносить дату/статус отмены в сам первичный reader, чтобы он сохранял корректный смысл при отдельном открытии и повторной генерации.

### 8. P2. Разрешённый запуск только для Георга падает на корректном отказе от склейки циклов

**Тип:** доказанный по потоку управления дефект; запуск не выполнялся.

**Locators:** [--subjects:1258](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1258>), [отказ:1408](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1408>), [continue:1470](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1470>), [запись CSV:1564](<../../cardiac_4d_segmentation/analyze_subtraction_and_rr.py#L1564>).

При --subjects georg merge rejected приводит к profiles=None. Для всех камер выполняется continue, cycle_change_rows остаётся пустым. Затем используется cycle_change_rows[0]. Файл closed-cycle CSV уже открыт в режиме w; более ранние таблицы тоже записаны.

**Последствие:** IndexError вместо полного отчёта об отказе; возможно обнуление прежнего closed-cycle CSV и частично обновлённый набор результатов.

**Предложение:** отказ от модели считать штатным результатом, сохранять схему пустой таблицы и статус; публиковать согласованный набор артефактов только после успешного завершения.

### 9. P2. Benchmark отвергает JSON provenance как не-NIfTI

**Тип:** доказанный дефект интерфейса.

**Locators:** [benchmark_blood_segmentation_candidates.py:397](<../../cardiac_4d_segmentation/benchmark_blood_segmentation_candidates.py#L397>), [_resolve_path:116](<../../cardiac_4d_segmentation/benchmark_blood_segmentation_candidates.py#L116>), [example manifest:31](<../../cardiac_4d_segmentation/benchmark_blood_segmentation_manifest.example.json#L31>).

provenance_manifest_path передаётся в функцию, допускающую только .nii/.nii.gz. Пример правильно использует .json, но после заполнения существующими путями такой manifest будет отвергнут.

**Последствие:** штатный путь сохранения происхождения не работает. Удаление поля обходит ошибку ценой потери связи с запуском; тесты этот случай не покрывают.

**Предложение:** разделить проверку путей изображения и произвольного provenance-файла.

### 10. P2. Первичные QC-панели путают плоскости и физические пропорции

**Тип:** доказанный дефект для стандартной аксиальной геометрии собственного конвертера; не переносится автоматически на все поздние viewers.

**Locators:** [render_qc.py:94](<../../cardiac_4d_segmentation/render_qc.py#L94>), [imshow:19](<../../cardiac_4d_segmentation/render_qc.py#L19>), [render_static_anatomy_qc.py:69](<../../cardiac_4d_segmentation/render_static_anatomy_qc.py#L69>).

ct[:,j,:] подписан coronal, ct[i,:,:] — sagittal, тогда как при данной конвенции осей подписи обратные. imshow не учитывает voxel spacing.

**Последствие:** неверные названия сечений и растяжение анизотропных данных ухудшают анатомический QC. Совпадение контура и CT на одинаково растянутом изображении не проверяет физическую геометрию.

**Предложение:** выводить плоскости по affine/orientation и учитывать физический aspect; выводы об анатомии на старых панелях переоценивать с учётом этого ограничения.

### 11. P2. Проверка geometry cache не проверяет целостность его содержимого

**Тип:** доказанный пробел воспроизводимости; текущая порча cache не обнаружена.

**Locators:** [build_cardiac_function_report.py:185](<../../cardiac_4d_segmentation/build_cardiac_function_report.py#L185>), [reuse:200](<../../cardiac_4d_segmentation/build_cardiac_function_report.py#L200>), [presentation-only:334](<../../cardiac_4d_segmentation/build_cardiac_function_report.py#L334>).

Fingerprint содержит входные файлы и cardiac_function_analysis.py, но не код загрузки collect_geometry. При совпавшем input_fingerprint строки JSON принимаются без проверки своего хеша. В presentation-only проверяются хеши CSV, затем геометрические rows берутся из geometry_cache.json, не сверяя их с проверенным chamber_geometry.csv.

**Последствие:** изменённый cache при неизменной метке происхождения может попасть в изображения и не согласоваться с проверенными таблицами. Изменение логики загрузчика также не обязательно инвалидирует cache.

**Текущее состояние проверено:** все 156 строк cache совпали с CSV; это потенциальный дефект защиты, не обнаруженное расхождение.

**Предложение:** проверять хеш содержимого cache и полный набор вычислительных зависимостей либо использовать единственный проверяемый источник геометрических строк.

### 12. P2, историческая ветка. «Покрытие» 3D_Slicer тождественно равно 100%

**Тип:** доказанный дефект старого QC; не доказательство ошибки нынешнего cardiac-пайплайна.

**Locators:** [3D_Slicer/04_qc.py:119](<../../../3D_Slicer/04_qc.py#L119>), [test_pipeline.py:217](<../../../3D_Slicer/tests/test_pipeline.py#L217>).

Для целочисленного labelmap и body_threshold=1 числитель labelmap>0 и знаменатель labelmap>=1 совпадают. Любая непустая маска получает полное покрытие, включая маску с пропусками. Тест проверяет положительный случай.

**Последствие:** старые отчёты о coverage не подтверждают полноту сегментации. Историческую неудачу автоматизации следует сохранять, не реабилитировать через этот QC.

**Связанный технический факт:** [main.py:101](<../../../3D_Slicer/main.py#L101>) не передаёт preprocess обязательный --all, требуемый [01_preprocess.py:327](<../../../3D_Slicer/01_preprocess.py#L327>). Полный старый launcher не является готовой воспроизводимой альтернативой Inobitec.

## 3. Что выполнено хорошо и что проверено этим аудитом

1. Root20 разделяет DICOM-QC, регистрацию, HU и электрические сценарии. 20.03–20.04 не выдают контракты за завершённые расчёты и отказываются от прежнего необоснованного «независимого подтверждения» двуслойной модели.
2. Основной 21.03 различает объём автоматической маски, гипотетическую коррекцию, глобальные экстремумы, неполный цикл и периодическую модель. Поля independent_reference_available=False и exploratory_hypothesis_not_validated сохранены. ED/ES не объявлены независимо принятыми.
3. R–R Георга 491/1159 мс не склеиваются; модели отказано явно. Времена и циклы не подменены единой нормированной кривой.
4. Объёмы/геометрия считаются на исходной сетке; упрощённая поверхность для интерактивного показа не используется как основание численного объёма. Геометрический остаток whole-heart minus chambers не назван автоматически миокардом.
5. Исходные маски и экспериментальные кандидаты разделены. Нет молчаливого разрешения overlap; сохранены необрезанное объединение камер, no_accuracy_inference/agreement_only, ограничения модельных подсказок и ноль принятых кандидатов пилота.
6. Для разностей различаются добавленная и удалённая области; малая итоговая разность не автоматически изоволюмическая фаза. Временное разрешение 140 мс и зависимость соседних реконструкций уже описаны.
7. BSA рассматривается как масштабирование. Указаны происхождение антропометрии, ограничения определения полостей и то, что МРТ-нормы не калибруют маски. Уравнивание ЛЖ/ПЖ или подгонка масок к нормам не используется как автоматическое исправление.
8. Есть manifests, hashes и разделение presentation-only. Это полезная основа воспроизводимости несмотря на перечисленные пробелы.

### Собственные лёгкие проверки сохранённых результатов

- Проверены совпадения девяти code hashes в function_analysis_manifest, четырёх function CSV и хешей notebook/HTML; текущие совпали.
- В joint manifest совпали три code hashes и восемь CSV hashes. В reference indexing совпали hashes двух итоговых таблиц; полный внешне-медицинский provenance этим не проверен.
- Все 156 строк geometry_cache согласованы с chamber_geometry.csv.
- Для 118 численных строк оценок проверены SV=EDV−ESV и EF=100×SV/EDV; расхождений не найдено.
- Для 18 пространственных случаев совпало added−removed=net.
- Для 16 BSA-строк обратное масштабирование совпало до машинной точности; EF от одного скалярного BSA не меняется.
- В проверенном наборе нет заявленного независимого референса.

Эти результаты подтверждают внутреннюю согласованность сохранённых таблиц. Они не проверяют анатомические границы, правильность времени или физическую/экспериментальную точность.

### Существенные исследовательские ограничения

Все 33 выбранных межфазных интервала имеют шаг примерно 49.1–127.49 мс, меньше указанного разрешения 140 мс; для двух желудочков это 66 интервальных оценок. Такие разности и секущие зависимых реконструкций нельзя превращать в независимую высокочастотную динамику. Сетка модели 1% не повышает разрешение CT.

Выбор наблюдаемого max/min не определяет анатомические ED/ES. В частности, у Георга глобальные экстремумы ПЖ относятся к разным R–R; у Nix глобальные экстремумы также могут происходить из разных циклов. Соответствующие статусы в таблицах присутствуют, поэтому это ограничение допустимости интерпретации, а не тайно заявленный валидированный УО.

Проверенная чувствительность к числу гармоник и ridge даёт: Adam ЛЖ SV 43.89–56.11 мл, EF 28.27–35.81%; Adam ПЖ SV 71.99–93.98 мл, EF 33.19–42.79%; Nix ЛЖ SV 51.94–65.41 мл, EF 34.78–42.41%; Nix ПЖ SV 86.33–112.93 мл, EF 36.94–47.70%. Это диапазоны моделей, не доверительные интервалы и не оценка всей погрешности. Чувствительность к cycle-contrast penalty, удалению фаз, полноте покрытия и temporal reconstruction kernel этой сеткой не исчерпана. Locator: 21.03, cell id bc9a403c (output чувствительности), ядро [cardiac_function_analysis.py:175](<../../cardiac_4d_segmentation/cardiac_function_analysis.py#L175>).

Согласие whole-heart/chamber-моделей, гладкость объёмов, нормированные кривые, размеры и площадь поверхности происходят от тех же изображений/масок. Они полезны для QC, но не независимы как доказательство точности эндокардия. Индексирование по BSA сохраняет ошибку сегментации в масштабе и вообще не меняет EF.

Ограничение переносимости: локально проверенные ссылки текущего HTML существуют, но часть targets не отслеживается Git. В главном 21.03 найдено 83 локальных ссылки, 0 отсутствующих в этой рабочей копии, 35 targets вне Git; в joint standalone — 24/0/5. В CT-viewer Nix phase_03 два из трёх статически указанных файлов не tracked, динамические PNG также внешние. Это не основание коммитить медицинские изображения. Нужен ясный договор восстановления локальных артефактов. [heart_rr_analysis/README.md:37](<../../cardiac_4d_segmentation/heart_rr_analysis/README.md#L37>) говорит об исключении каталога, хотя часть HTML/QA фактически tracked.

## 4. Идеи и отрицательные результаты, которые следует сохранить

- Неудачу исторической 3D_Slicer-автоматизации и происхождение итоговой ручной Inobitec-анатомии. Исправление launcher/QC не сделает старую ветку научно успешной задним числом.
- HU, воздушность и модели смешения как отдельные исследовательские гипотезы; чувствительность к сопротивлению матрицы и ограничения межмодального переноса. Значения params/ct.json не переобозначать измеренными электрическими свойствами.
- Сравнение v1/v2, whole-heart, камер и миокарда как разбор согласованности определений и геометрии. «Независимая» whole-organ сеть в смысле конструкции маски не равна независимому измерительному референсу.
- Раздельные два R–R Георга и отказ от единого периодического профиля. Причину различия интервалов не выводить из одного факта 491/1159 мс.
- Пилот 16 nnInteractive-кандидатов с нулём принятых. Изменение разности ЛЖ 14−10 с 49.34 до 25.32 мл без точек и до 50.80 мл с подсказками показывает чувствительность результата к способу маскирования; близость к исходному числу не доказывает правильности кандидата. Наблюдения старого визуального пилота сохранять с указанием, что просмотр был ограниченным.
- Георг 04→05: добавление около 1.59 мл и удаление около 1.79 мл могут дать net около −0.20 мл. Сохранить идею раздельной пространственной динамики; не превращать компенсацию изменений в доказанное IVC/IVR.
- Чувствительность условных SV/EF к объёмным сценариям, способу склейки и регуляризации. Сценарии не заменяют метрологически установленное распределение ошибки.
- Отдельные координаты: gating CT, механическая фаза, экспериментальная ЭКГ/ТТРКГ. Их объединение — самостоятельная задача, а не следствие общей метки R.
- Решение автора не готовить новые ручные маски на недостаточном контрасте. Сохранить его дату и мотив, не теряя конечной потребности независимо проверить применимость методики.

## 5. Предложения структуры без переименований и перемещений

1. Оставить текущие имена. В 21.00 закрепить таблицу ролей: 20.01–20.04 — контракты CT-входов; 20.05–20.07 — протоколы конверсии/сегментации; 21.01–21.02 — визуальные представления; 21.03 — единственный текущий научный отчёт; 21.04–21.07 и Georg pilot — датированные протоколы/история. Для каждого: статус, upstream, генератор, outputs, ограничения и последнее авторское решение.
2. Для цепочки input→conversion→segmentation→timing→geometry→function→reader определить один контракт происхождения. Subject не должен заменять acquisition/series identity. Все вычисляемые стадии должны либо подтвердить совпадение upstream hashes, либо явно остановиться; reused результат сохраняет свой исходный запуск.
3. Разделить в существующих manifests три независимых статуса: техническая пригодность, анатомическая/временная интерпретируемость и независимая валидация. Касание границы, неуверенная фаза, неполный цикл и отказ от склейки должны доходить до конкретной итоговой строки.
4. Зафиксировать владельца каждого notebook/HTML: какие script и narrative constants его строят, какие inputs обязательны, что означает запуск comment-only ячеек. Текст последнего решения автора должен храниться в источнике standalone reader, а не только добавляться внешним генератором.
5. Существующие outputs организовать логически через manifests: пересоздаваемые публичные таблицы/описания, локальные производные медицинских данных, визуальные страницы с локальными зависимостями, исторические результаты. Не добавлять медицинские изображения в Git ради исправления ссылок. Описать восстановление внешних targets и понятное сообщение об их отсутствии.
6. Историю сохранять через текущие pointers, migration manifests и Git; не создавать конкурирующую каноническую копию. Старые планы и неудачные варианты маркировать датой/отменой, сохраняя их отрицательные результаты.
7. Исправления кода выполнять отдельным последующим этапом. Приоритет: происхождение масок и timing, геометрия конверсии, передача QC; далее ошибки отказных путей и benchmark; затем читательские подписи и актуальность текста. Для будущей проверки нужны небольшие регрессионные случаи на конкретные найденные дефекты, а не повтор полного тяжёлого расчёта.

На этом аудит назначенного участка завершён. Охват не расширялся за перечисленные связи; предложения не применялись к репозиторию.
