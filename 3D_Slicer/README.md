# Пайплайн сегментации КТ → labelmap для COMSOL

Автоматическая сегментация КТ (DICOM) с получением 7-классного labelmap
для импорта в COMSOL Multiphysics в качестве многодоменной геометрической модели.

## Ткани (классы)

| Метка | Ткань             |
|-------|-------------------|
| 1     | bone              |
| 2     | lung              |
| 3     | heart             |
| 4     | muscle            |
| 5     | fat_subcutaneous  |
| 6     | skin              |
| 7     | other             |

---

## Установка окружения

### Windows (рекомендуется)

```bat
setup_env.bat
```

Скрипт создаёт `.venv`, устанавливает PyTorch с CUDA 12.1 и все зависимости.

### Ручная установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Требования

- Python 3.10+
- CUDA 12.x (опционально, без GPU работает медленнее)
- ~15 ГБ места: модели TotalSegmentator скачиваются при первом запуске (~3 ГБ)

---

## Структура папок

```
3D_Slicer/
├── 01_preprocess.py        # DICOM → NIfTI
├── 02_segment_batch.py     # TotalSegmentator (104 класса)
├── 03_postprocess.py       # 104 класса → 7-класс labelmap
├── 04_qc.py                # Контроль качества
├── 05_export_comsol.py     # Экспорт → NRRD/MHD для COMSOL
├── 06_visualize_qc.py      # PNG-визуализация QC
├── 09_cohort_summary.py    # Сводный отчёт по когорте
├── main.py                 # Оркестратор пайплайна
├── config.json             # Все параметры
├── requirements.txt
├── setup_env.bat
├── tests/
│   └── test_pipeline.py    # Smoke-тесты
└── data/
    ├── preprocessed/       # .nii.gz после ресэмплинга
    ├── segmentations/
    │   ├── auto/           # TotalSegmentator маски
    │   ├── corrected/      # Ручные коррекции (опционально)
    │   └── final_labelmap/ # 7-класс labelmap
    ├── comsol_ready/       # NRRD/MHD для COMSOL
    └── logs/               # JSON/CSV отчёты, PNG визуализации
```

---

## Конфигурация (`config.json`)

Ключевые параметры:

```json
{
  "paths": {
    "raw_dicom": "Z:/path/to/DICOM"   // ← указать свой путь
  },
  "preprocess": {
    "target_spacing_mm": [1.5, 1.5, 1.5]
  },
  "segmentation": {
    "device": "gpu",                  // "gpu" или "cpu"
    "max_retries": 1,                 // повторы при сбое
    "timeout_seconds": 3600,          // таймаут на пациента
    "fallback_to_cpu_on_oom": true    // авто-fallback при OOM
  },
  "hu_presets": {
    "body_mask_threshold": -500       // порог маски тела
  },
  "skin_generation": {
    "thickness_mm": 3.0
  },
  "qc": {
    "min_coverage_ratio": 0.98,
    "max_islands_per_tissue": 3,
    "boundary_ratio_threshold": 0.6
  }
}
```

---

## Использование

### Полный пайплайн

```bash
python main.py --all
```

### Один пациент

```bash
python main.py --all --patient Series_0301
```

### Отдельные шаги

```bash
# 1. Показать доступные серии DICOM
python 01_preprocess.py --list

# 2. Предобработка одной серии
python 01_preprocess.py --series 301

# 3. Предобработка всех серий
python main.py --step preprocess

# 4. Сегментация (быстрый режим, CPU)
python main.py --step segment --fast --cpu

# 5. Постобработка → labelmap
python main.py --step postprocess

# 6. Контроль качества
python main.py --step qc

# 7. Экспорт для COMSOL (NRRD по умолчанию, или MHD)
python main.py --step export
python main.py --step export --mhd

# 8. Визуализация QC (PNG-срезы)
python main.py --step visualize

# 9. Сводный отчёт по когорте
python 09_cohort_summary.py
python 09_cohort_summary.py --only-fail
```

---

## Ручные коррекции

Если TotalSegmentator ошибся в конкретной маске (например, сердце):

1. Открыть `data/segmentations/auto/<patient_id>/` в 3D Slicer
2. Отредактировать нужную маску (например `heart.nii.gz`)
3. Сохранить в `data/segmentations/corrected/<patient_id>/heart.nii.gz`
4. Перезапустить постобработку: `python main.py --step postprocess --patient <ID>`

Скрипт автоматически подставит корректированную маску вместо авто.
Список подставленных масок записывается в `data/logs/mask_sources_<patient>.json`.

---

## Импорт в COMSOL

После шага `export` в `data/comsol_ready/<patient_id>/` лежат:

- `labelmap.nrrd` — полный 7-классный объём
- `bone.nrrd`, `lung.nrrd`, ... — отдельные бинарные маски
- `metadata.json` — spacing, размеры, объёмы тканей

**Импорт в COMSOL Multiphysics:**
1. `File → Import → Image Data`
2. Указать `labelmap.nrrd`
3. Каждый целочисленный класс (1–7) автоматически становится доменом
4. Назначить материальные свойства по доменам

---

## Тесты

```bash
# Без pytest
python tests/test_pipeline.py

# Через pytest
python -m pytest tests/ -v
```

Тесты работают без DICOM и GPU — на синтетических numpy-фантомах.

---

## Логирование

Каждый запуск скрипта создаёт файл `data/logs/<script>_YYYYMMDD_HHMMSS.log`.
QC-отчёты: `data/logs/qc_<patient>.json`.
Сводный отчёт: `data/logs/cohort_summary.csv` + `cohort_summary.json`.
