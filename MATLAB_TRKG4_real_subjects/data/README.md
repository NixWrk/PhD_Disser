# data

Эта папка предназначена для локальных данных испытуемых. STL и CSV сигналов игнорируются git.

Структура:

```text
data/<subject>/
  stl/
    body_solid_v3_1mm_r10_volume_fill.stl   решаемая поверхность тела
    lungs_solid_v3_volume_fill.stl          решаемый объём лёгких
    heart.stl
    bones.stl
    blood.stl                               optional
    body.stl, lungs_ct_full_clean.stl       исходники CT, для прослеживаемости
  electrodes/
    electrodes_4_arms_xyz_mm.csv            монтаж по умолчанию
    electrodes_4_right_rib_050mm_xyz_mm.csv
    electrodes_4_right_rib_140mm_xyz_mm.csv
  experimental/
    nik_breath_hold_levels.csv              уровни задержек вдоха и выдоха
  signal/
    optional validation signals
```

Точные имена, которые читает код, заданы в `src/trkg4_subject_registry.m`.
Единого `electrodes_4_xyz_mm.csv` нет: каждый монтаж — отдельный именованный
файл, см. `docs/ELECTRODES_4.md`.

Координаты STL и электродов должны быть в одной системе координат.

Перед использованием CT/STL обязательна процедура из
`docs/CT_STL_MANDATORY_PREPROCESSING.md`. Для Nik рабочий полный исходник
лёгких — `lungs_ct_full_clean.stl`; ранний `lungs.stl` обрезан по трахее и не
должен использоваться как источник новой версии.

## Состояние локальных данных Nik

Файлы этой папки в git не хранятся, поэтому наличие их на конкретной машине
нужно проверять отдельно. На момент правки в `data/nik/stl` **отсутствуют**:

- `lungs_ct_full_clean.stl` — исходник, из которого получен принятый
  `lungs_solid_v3_volume_fill.stl`. Локальная копия
  `Z:/02 Big_data/3D/NIX/CT_10_12_25_RNCH_NIX_STL/NIX_CT_lungs_clear_smooth.stl`.
  Без него обязательную процедуру предобработки нельзя воспроизвести заново,
  хотя уже построенная solid-версия на месте и расчёт идёт;
- `lungs.stl` — ранняя обрезанная поверхность, упомянутая выше как
  непригодная. Её отсутствие ни на что не влияет;
- `blood.stl` — сегментация крови и сосудов. Ткань объявлена и включена, но
  без маски соответствующий объём остаётся с проводимостью мягкой ткани, и
  расчётный `Z` смещён на неизвестную величину. Расчёт сообщает об этом
  предупреждением; см. `docs/INVERSE_IDENTIFIABILITY_AND_BREATHING_STATES.md`,
  раздел 1.1.
