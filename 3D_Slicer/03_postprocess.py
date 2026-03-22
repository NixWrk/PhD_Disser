"""
03_postprocess.py — Объединение масок TotalSegmentator в 7-классный labelmap для COMSOL.

Логика:
1. Загрузить multilabel-маску из TotalSegmentator (104 класса)
2. Сгруппировать в 5 тканевых классов (кости, лёгкие, сердце, мышцы, подкожный жир)
3. Сгенерировать маску тела → вычислить кожу (эрозия)
4. «Other» = тело минус все остальные
5. Применить правило приоритета → единый labelmap без зазоров

Вход:  data/segmentations/auto/<patient_id>/
Выход: data/segmentations/final_labelmap/<patient_id>_labelmap.nii.gz

Запуск:
    python 03_postprocess.py                  # все
    python 03_postprocess.py --patient P001   # один
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent


def setup_logging(script_name: str) -> logging.Logger:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
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


log = setup_logging("03_postprocess")

# TotalSegmentator v2 label indices (multilabel.nii.gz)
# Полный справочник: https://github.com/wasserth/TotalSegmentator#class-details
TOTALSEG_GROUPS = {
    "bone": list(range(25, 50)) + list(range(69, 91)),  # позвонки + рёбра + кости конечностей
    "lung": [11, 12, 13, 14, 15],  # 5 долей
    "heart": [
        16, 17, 18, 19, 20, 21, 22,  # камеры сердца, миокард, аорта
        23, 24,  # полые вены
    ],
    "muscle": list(range(91, 104)),  # мышечные группы
    "fat_subcutaneous": [104] if True else [],  # subcutaneous fat (если есть в модели)
}


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_individual_masks(seg_dir: Path) -> dict[str, np.ndarray]:
    """
    Загружает отдельные маски TotalSegmentator из папки (если multilabel нет).
    Возвращает dict: имя_файла → бинарный массив.
    """
    masks = {}
    for f in sorted(seg_dir.glob("*.nii.gz")):
        if f.name == "multilabel.nii.gz":
            continue
        img = nib.load(str(f))
        masks[f.stem] = img.get_fdata().astype(np.uint8)
    return masks


def group_by_name(masks: dict[str, np.ndarray], config: dict) -> dict[str, np.ndarray]:
    """
    Группирует маски TotalSegmentator по тканевым классам через имена файлов.
    Использует wildcard-маппинг из config.json.
    """
    import fnmatch

    tissue_map = config["totalseg_to_tissue_map"]
    shape = next(iter(masks.values())).shape
    grouped = {}

    for tissue, patterns in tissue_map.items():
        combined = np.zeros(shape, dtype=np.uint8)
        matched = 0
        for mask_name, mask_data in masks.items():
            for pattern in patterns:
                if fnmatch.fnmatch(mask_name, pattern):
                    combined = np.maximum(combined, mask_data)
                    matched += 1
                    break
        grouped[tissue] = combined
        log.info("  %s: %d масок объединено", tissue, matched)

    return grouped


def generate_body_mask(ct_data: np.ndarray, threshold: float = -500) -> np.ndarray:
    """Маска тела: порог + заполнение дыр + keep largest."""
    body = (ct_data > threshold).astype(np.uint8)

    # Заполнить дыры послойно (аксиальные срезы)
    for z in range(body.shape[0]):
        body[z] = ndimage.binary_fill_holes(body[z]).astype(np.uint8)

    # Keep largest connected component
    labeled, n_comp = ndimage.label(body)
    if n_comp > 1:
        sizes = ndimage.sum(body, labeled, range(1, n_comp + 1))
        largest_label = np.argmax(sizes) + 1
        body = (labeled == largest_label).astype(np.uint8)

    log.info("Маска тела: %d вокселей", body.sum())
    return body


def generate_skin_mask(
    body_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    thickness_mm: float = 3.0,
) -> np.ndarray:
    """Кожа = тело минус эрозия тела на thickness_mm."""
    # Радиус эрозии в вокселях по каждой оси
    radii = [max(1, int(round(thickness_mm / s))) for s in spacing_mm]
    struct = ndimage.generate_binary_structure(3, 1)

    eroded = ndimage.binary_erosion(body_mask, structure=struct, iterations=min(radii))
    skin = body_mask.astype(np.uint8) - eroded.astype(np.uint8)
    skin = np.clip(skin, 0, 1).astype(np.uint8)

    log.info("Кожа: толщина ~%.1f мм, %d вокселей", thickness_mm, skin.sum())
    return skin


def apply_priority_rule(
    tissue_masks: dict[str, np.ndarray],
    priority_order: list[str],
    body_mask: np.ndarray,
    label_map: dict[str, int],
) -> np.ndarray:
    """
    Применяет правило приоритета: каждый воксель получает метку
    ткани с наивысшим приоритетом (первые в списке = выше).

    Гарантирует: каждый воксель тела имеет ровно одну метку ≥ 1.
    """
    shape = body_mask.shape
    labelmap = np.zeros(shape, dtype=np.uint8)

    # Заполняем в обратном порядке — последний записанный имеет приоритет
    for tissue_name in reversed(priority_order):
        label_value = label_map[tissue_name]
        if tissue_name in tissue_masks:
            mask = tissue_masks[tissue_name]
            labelmap[mask > 0] = label_value

    # «Other» заполняет все воксели тела, которые остались без метки
    other_label = label_map["other"]
    unassigned = (labelmap == 0) & (body_mask > 0)
    labelmap[unassigned] = other_label

    # Обнулить всё за пределами тела
    labelmap[body_mask == 0] = 0

    # Статистика
    for tissue_name in priority_order:
        lv = label_map[tissue_name]
        count = np.sum(labelmap == lv)
        log.info("  label %d (%s): %d вокселей", lv, tissue_name, count)

    return labelmap


def process_patient(patient_id: str, cfg: dict) -> Path:
    """Полный постпроцессинг одного пациента."""
    seg_dir = ROOT / cfg["paths"]["seg_auto"] / patient_id
    prep_path = ROOT / cfg["paths"]["preprocessed"] / f"{patient_id}.nii.gz"
    out_dir = ROOT / cfg["paths"]["seg_final"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загрузить КТ (для маски тела)
    ct_img = nib.load(str(prep_path))
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]

    # Загрузить маски TotalSegmentator
    log.info("Загружаю маски из %s", seg_dir)
    individual_masks = load_individual_masks(seg_dir)

    if not individual_masks:
        raise FileNotFoundError(f"Нет масок в {seg_dir}")

    # Группировать по тканям
    log.info("Группировка по тканям:")
    tissue_masks = group_by_name(individual_masks, cfg)

    # Маска тела
    body_threshold = cfg["hu_presets"]["body_mask_threshold"]
    body_mask = generate_body_mask(ct_data, threshold=body_threshold)

    # Кожа
    skin_cfg = cfg["skin_generation"]
    skin_mask = generate_skin_mask(body_mask, spacing, skin_cfg["thickness_mm"])
    tissue_masks["skin"] = skin_mask

    # Собрать label map
    label_map = {v: int(k) for k, v in cfg["tissue_labels"].items()}
    priority = cfg["priority_order"]

    log.info("Применяю правило приоритета:")
    labelmap = apply_priority_rule(tissue_masks, priority, body_mask, label_map)

    # Проверка: все воксели тела покрыты?
    body_voxels = body_mask.sum()
    assigned_voxels = (labelmap > 0).sum()
    coverage = assigned_voxels / body_voxels if body_voxels > 0 else 0
    log.info("Покрытие: %.4f (%d / %d)", coverage, assigned_voxels, body_voxels)

    if coverage < 0.999:
        log.warning("ВНИМАНИЕ: покрытие < 99.9%%!")

    # Сохранить
    out_path = out_dir / f"{patient_id}_labelmap.nii.gz"
    labelmap_img = nib.Nifti1Image(labelmap, ct_img.affine, ct_img.header)
    labelmap_img.header.set_data_dtype(np.uint8)
    nib.save(labelmap_img, str(out_path))
    log.info("Сохранено: %s", out_path)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Постобработка → labelmap для COMSOL")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    auto_dir = ROOT / cfg["paths"]["seg_auto"]

    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted([d.name for d in auto_dir.iterdir() if d.is_dir()])

    if not patients:
        log.error("Нет сегментированных пациентов в %s", auto_dir)
        sys.exit(1)

    results = []
    for pid in patients:
        try:
            out = process_patient(pid, cfg)
            results.append(("OK", pid, str(out)))
        except Exception as e:
            log.error("ОШИБКА %s: %s", pid, e)
            results.append(("FAIL", pid, str(e)))

    log.info("=== Итого ===")
    ok = sum(1 for r in results if r[0] == "OK")
    log.info("Успешно: %d / %d", ok, len(results))


if __name__ == "__main__":
    main()
