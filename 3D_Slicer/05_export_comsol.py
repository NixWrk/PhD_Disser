"""
05_export_comsol.py — Экспорт финального labelmap в форматы, пригодные для COMSOL.

Форматы вывода:
  - .nrrd (основной): labelmap целиком + отдельные бинарные маски по тканям
  - .mhd/.raw: альтернатива, если COMSOL не читает .nrrd напрямую

COMSOL Multiphysics читает .nrrd через Image Data → Interpolated на mesh.
Каждая ткань экспортируется как отдельный бинарный файл для назначения
материальных свойств по доменам.

Вход:  data/segmentations/final_labelmap/<patient_id>_labelmap.nii.gz
Выход: data/comsol_ready/<patient_id>/
         labelmap.nrrd          — полный многоклассовый labelmap
         bone.nrrd              — бинарная маска кости
         lung.nrrd              — ...и т.д. по каждой ткани
         metadata.json          — параметры сетки, метки, spacing

Запуск:
    python 05_export_comsol.py                  # все пациенты
    python 05_export_comsol.py --patient P001   # один
    python 05_export_comsol.py --mhd            # формат MHD/RAW вместо NRRD
"""

import argparse
import json
import logging
import struct
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent


def setup_logging(script_name: str) -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{ts}.log"

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info("Лог записывается в: %s", log_file)
    return logger


log = setup_logging("05_export")


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def nifti_to_nrrd(
    data: np.ndarray,
    affine: np.ndarray,
    spacing: tuple,
    out_path: Path,
    dtype=np.uint8,
):
    """
    Сохраняет массив как NRRD-файл (текстовый заголовок + бинарные данные).
    COMSOL читает NRRD версии 4 с little-endian raw-данными.
    """
    data = data.astype(dtype)

    # NRRD ожидает порядок осей (x, y, z) — NIfTI хранит (z, y, x) после get_fdata()
    # Меняем на (x, y, z) для корректного импорта в COMSOL
    data_nrrd = np.transpose(data, (2, 1, 0))

    # Направления осей из affine (первые 3 столбца, нормированные)
    directions = []
    for col in range(3):
        vec = affine[:3, col]
        norm = np.linalg.norm(vec)
        directions.append(vec / norm if norm > 0 else vec)

    raw_data = data_nrrd.tobytes(order="C")

    header_lines = [
        "NRRD0004",
        f"type: {'unsigned char' if dtype == np.uint8 else 'short'}",
        f"dimension: 3",
        f"sizes: {data_nrrd.shape[0]} {data_nrrd.shape[1]} {data_nrrd.shape[2]}",
        f"spacings: {spacing[0]:.6f} {spacing[1]:.6f} {spacing[2]:.6f}",
        f"space directions: ({directions[0][0]:.6f},{directions[0][1]:.6f},{directions[0][2]:.6f})"
        f" ({directions[1][0]:.6f},{directions[1][1]:.6f},{directions[1][2]:.6f})"
        f" ({directions[2][0]:.6f},{directions[2][1]:.6f},{directions[2][2]:.6f})",
        "space: left-posterior-superior",
        "endian: little",
        "encoding: raw",
        "",  # пустая строка отделяет заголовок от данных
    ]

    with open(out_path, "wb") as f:
        f.write("\n".join(header_lines).encode("ascii"))
        f.write(raw_data)

    log.info("  NRRD: %s (%.1f МБ)", out_path.name, out_path.stat().st_size / 1e6)


def nifti_to_mhd(
    data: np.ndarray,
    affine: np.ndarray,
    spacing: tuple,
    out_path: Path,
    dtype=np.uint8,
):
    """
    Сохраняет как MHD+RAW — альтернатива для COMSOL если NRRD не читается.
    """
    data = data.astype(dtype)
    raw_path = out_path.with_suffix(".raw")

    with open(raw_path, "wb") as f:
        f.write(data.tobytes(order="C"))

    origin = affine[:3, 3]
    type_str = "MET_UCHAR" if dtype == np.uint8 else "MET_SHORT"

    mhd_lines = [
        "ObjectType = Image",
        "NDims = 3",
        f"DimSize = {data.shape[2]} {data.shape[1]} {data.shape[0]}",
        f"ElementSpacing = {spacing[0]:.6f} {spacing[1]:.6f} {spacing[2]:.6f}",
        f"Offset = {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}",
        f"ElementType = {type_str}",
        "ElementByteOrderMSB = False",
        f"ElementDataFile = {raw_path.name}",
    ]

    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(mhd_lines))

    log.info("  MHD: %s", out_path.name)


def export_patient(patient_id: str, cfg: dict, use_mhd: bool = False) -> Path:
    """Экспортирует labelmap и отдельные тканевые маски для одного пациента."""
    final_dir = ROOT / cfg["paths"]["seg_final"]
    labelmap_path = final_dir / f"{patient_id}_labelmap.nii.gz"

    if not labelmap_path.exists():
        raise FileNotFoundError(f"Labelmap не найден: {labelmap_path}")

    out_dir = ROOT / cfg["paths"]["comsol_ready"] / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(labelmap_path))
    labelmap = img.get_fdata().astype(np.uint8)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    affine = img.affine

    ext = ".mhd" if use_mhd else ".nrrd"
    save_fn = nifti_to_mhd if use_mhd else nifti_to_nrrd

    log.info("Экспорт %s → %s", patient_id, out_dir)

    # 1. Полный labelmap
    save_fn(labelmap, affine, spacing, out_dir / f"labelmap{ext}")

    # 2. Отдельные бинарные маски по тканям
    tissue_labels = cfg["tissue_labels"]
    tissue_files = {}
    for label_str, tissue_name in tissue_labels.items():
        label_val = int(label_str)
        binary = (labelmap == label_val).astype(np.uint8)
        out_file = out_dir / f"{tissue_name}{ext}"
        save_fn(binary, affine, spacing, out_file)
        tissue_files[tissue_name] = {
            "file": str(out_file.relative_to(ROOT)),
            "label_value": label_val,
            "voxel_count": int(binary.sum()),
            "volume_ml": round(float(binary.sum()) * float(np.prod(spacing)) / 1000.0, 1),
        }

    # 3. Метаданные для COMSOL
    metadata = {
        "patient": patient_id,
        "exported_at": datetime.now().isoformat(),
        "format": "mhd" if use_mhd else "nrrd",
        "spacing_mm": list(spacing),
        "shape_xyz": [int(labelmap.shape[2]), int(labelmap.shape[1]), int(labelmap.shape[0])],
        "affine": affine.tolist(),
        "labelmap_file": f"labelmap{ext}",
        "tissues": tissue_files,
        "comsol_import_note": (
            "В COMSOL: File → Import → Image Data → указать labelmap.nrrd. "
            "Каждый класс (1–7) соответствует одному домену материала."
        ),
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("Метаданные: %s", meta_path)
    log.info("Готово: %s", out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Экспорт labelmap → COMSOL")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--mhd", action="store_true", help="Формат MHD/RAW вместо NRRD")
    args = parser.parse_args()

    cfg = load_config()
    final_dir = ROOT / cfg["paths"]["seg_final"]

    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted([
            f.name.replace("_labelmap.nii.gz", "")
            for f in final_dir.glob("*_labelmap.nii.gz")
        ])

    if not patients:
        log.error("Нет labelmaps в %s", final_dir)
        sys.exit(1)

    log.info("Экспорт %d пациентов, формат: %s", len(patients), "MHD" if args.mhd else "NRRD")

    ok, fail = 0, 0
    for pid in patients:
        try:
            export_patient(pid, cfg, use_mhd=args.mhd)
            ok += 1
        except Exception as e:
            log.error("ОШИБКА %s: %s", pid, e)
            fail += 1

    log.info("=== Итого: OK=%d FAIL=%d ===", ok, fail)


if __name__ == "__main__":
    main()
