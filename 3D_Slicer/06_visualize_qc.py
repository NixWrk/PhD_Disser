"""
06_visualize_qc.py — Визуализация QC: PNG-срезы с наложением labelmap на КТ.

Для каждого пациента сохраняет один PNG с тремя ортогональными срезами
(аксиальный, корональный, сагиттальный), на которых цветом показаны ткани.

Цвета тканей:
  1 bone             — белый
  2 lung             — голубой
  3 heart            — красный
  4 muscle           — оранжевый
  5 fat_subcutaneous — жёлтый
  6 skin             — розовый
  7 other            — серый

Вход:
  data/preprocessed/<patient_id>.nii.gz     (КТ для фона)
  data/segmentations/final_labelmap/<patient_id>_labelmap.nii.gz

Выход:
  data/logs/qc_viz_<patient_id>.png

Запуск:
    python 06_visualize_qc.py                  # все пациенты
    python 06_visualize_qc.py --patient P001   # один
    python 06_visualize_qc.py --no-ct          # только маска без CT-фона
"""

import argparse
import json
import logging
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


log = setup_logging("06_visualize")

# Цвета тканей: label_value → (R, G, B) в диапазоне 0–1
TISSUE_COLORS = {
    0: None,              # фон — прозрачный
    1: (1.0, 1.0, 1.0),  # bone — белый
    2: (0.3, 0.8, 1.0),  # lung — голубой
    3: (0.9, 0.1, 0.1),  # heart — красный
    4: (1.0, 0.5, 0.1),  # muscle — оранжевый
    5: (1.0, 0.95, 0.2), # fat — жёлтый
    6: (1.0, 0.6, 0.7),  # skin — розовый
    7: (0.5, 0.5, 0.5),  # other — серый
}


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def normalize_ct(ct_data: np.ndarray, wl: float = 40, ww: float = 400) -> np.ndarray:
    """Нормализует CT в 0–1 по окну window level/width (мягкие ткани по умолчанию)."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    ct_clipped = np.clip(ct_data, lo, hi)
    return (ct_clipped - lo) / (hi - lo)


def labelmap_to_rgb(
    labelmap_slice: np.ndarray,
    ct_slice: np.ndarray | None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Конвертирует 2D-срез labelmap в RGB-изображение.
    Если ct_slice задан — накладывает маску поверх КТ с прозрачностью alpha.
    """
    h, w = labelmap_slice.shape

    if ct_slice is not None:
        rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)  # grayscale КТ
    else:
        rgb = np.zeros((h, w, 3), dtype=np.float32)

    for label_val, color in TISSUE_COLORS.items():
        if color is None:
            continue
        mask = labelmap_slice == label_val
        if not mask.any():
            continue
        for c, val in enumerate(color):
            if ct_slice is not None:
                rgb[mask, c] = rgb[mask, c] * (1 - alpha) + val * alpha
            else:
                rgb[mask, c] = val

    return np.clip(rgb, 0, 1)


def get_center_slices(volume: np.ndarray) -> tuple[int, int, int]:
    """Возвращает индексы центральных срезов по каждой оси."""
    return volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2


def visualize_patient(
    patient_id: str,
    cfg: dict,
    show_ct: bool = True,
) -> Path:
    """Создаёт PNG с тремя срезами labelmap для одного пациента."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # без GUI
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.error("matplotlib не установлен. Запустите: pip install matplotlib")
        raise

    final_dir = ROOT / cfg["paths"]["seg_final"]
    prep_dir = ROOT / cfg["paths"]["preprocessed"]
    log_dir = ROOT / cfg["paths"]["logs"]

    labelmap_path = final_dir / f"{patient_id}_labelmap.nii.gz"
    ct_path = prep_dir / f"{patient_id}.nii.gz"

    if not labelmap_path.exists():
        raise FileNotFoundError(f"Labelmap не найден: {labelmap_path}")

    lm_img = nib.load(str(labelmap_path))
    labelmap = lm_img.get_fdata().astype(np.uint8)

    ct_data = None
    if show_ct and ct_path.exists():
        ct_img = nib.load(str(ct_path))
        ct_data = normalize_ct(ct_img.get_fdata().astype(np.float32))
    elif show_ct:
        log.warning("КТ не найден (%s) — визуализация без фона", ct_path)

    ax_idx, cor_idx, sag_idx = get_center_slices(labelmap)

    # Срезы labelmap: (z, y, x) → аксиальный=z, корональный=y, сагиттальный=x
    slices_lm = {
        "Аксиальный": labelmap[ax_idx, :, :],
        "Корональный": labelmap[:, cor_idx, :],
        "Сагиттальный": labelmap[:, :, sag_idx],
    }
    slices_ct = {}
    if ct_data is not None:
        slices_ct = {
            "Аксиальный": ct_data[ax_idx, :, :],
            "Корональный": ct_data[:, cor_idx, :],
            "Сагиттальный": ct_data[:, :, sag_idx],
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a1a")

    tissue_labels = cfg["tissue_labels"]
    legend_patches = []
    for label_str, tissue_name in tissue_labels.items():
        lv = int(label_str)
        color = TISSUE_COLORS.get(lv)
        if color:
            legend_patches.append(
                mpatches.Patch(color=color, label=f"{lv}: {tissue_name}")
            )

    for ax, (title, lm_slice) in zip(axes, slices_lm.items()):
        ct_slice = slices_ct.get(title)
        rgb = labelmap_to_rgb(lm_slice, ct_slice)
        ax.imshow(rgb, origin="lower", aspect="equal")
        ax.set_title(title, color="white", fontsize=12)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    fig.suptitle(
        f"QC Визуализация: {patient_id}",
        color="white", fontsize=14, fontweight="bold",
    )
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=9,
        facecolor="#2a2a2a",
        labelcolor="white",
        framealpha=0.8,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = log_dir / f"qc_viz_{patient_id}.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info("Визуализация сохранена: %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Визуализация QC: PNG-срезы с labelmap")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--no-ct", action="store_true", help="Без КТ-фона")
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

    ok, fail = 0, 0
    for pid in patients:
        try:
            visualize_patient(pid, cfg, show_ct=not args.no_ct)
            ok += 1
        except Exception as e:
            log.error("ОШИБКА %s: %s", pid, e)
            fail += 1

    log.info("=== Итого: OK=%d FAIL=%d ===", ok, fail)


if __name__ == "__main__":
    main()
