# Результаты 21.01 и анатомический QC 21.02

[Общая точка входа серии 21](../21.00_Карта_4D_сердца_и_RR.md) ·
[Ноутбук 21.01](../21.01_4D_сердце_интерактивно.ipynb) ·
[HTML 21.01](../21.01_4D_сердце_интерактивно.html) ·
[Описание 21.02](../21.02_Интерактивная_4D_визуализация_и_анатомия_TotalSegmentator.md)

`*_cardiac_4d.html` и `manifest.json` создаёт
[build_interactive_cardiac4d.py](../build_interactive_cardiac4d.py)
из внешних `<subject>/products_v2`. Тот же скрипт создаёт общий notebook и
HTML `21.01` в родительской папке.

`anatomy_products_qc.csv`, `anatomy_subjects_qc.csv` и
`anatomy_qc_summary.json` создаёт отдельно
[summarize_anatomy_qc.py](../summarize_anatomy_qc.py).
Они относятся к окружающей анатомии на `phase_00`, описанной в `21.02`,
и не пересоздаются генератором 4D-графиков.

Статус масок — `automatic_segmentation_pending_manual_review`.
Это папка производных представлений и QC; исходные NIfTI находятся во внешнем
хранилище. Расчёт объёмов камер и времени R–R находится в
[соседней ветке 21.03](../heart_rr_analysis/README.md).
