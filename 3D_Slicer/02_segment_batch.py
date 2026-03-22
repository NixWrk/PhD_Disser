"""
02_segment_batch.py — Батч-сегментация через TotalSegmentator.

Вход:  .nii.gz файлы из data/preprocessed/
Выход: папки масок в data/segmentations/auto/<patient_id>/

Запуск:
    python 02_segment_batch.py                  # все пациенты
    python 02_segment_batch.py --patient P001   # один
    python 02_segment_batch.py --fast           # быстрый режим (ниже качество)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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


log = setup_logging("02_segment")


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def run_totalsegmentator(
    input_path: Path,
    output_dir: Path,
    task: str = "total",
    fast: bool = False,
    device: str = "gpu",
    multilabel: bool = True,
) -> dict:
    """Запускает TotalSegmentator CLI и возвращает метаданные запуска."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "totalsegmentator",
        "-i", str(input_path),
        "-o", str(output_dir),
        "--task", task,
    ]

    if fast:
        cmd.append("--fast")

    if device == "gpu":
        cmd.extend(["--device", "gpu"])
    else:
        cmd.extend(["--device", "cpu"])

    if multilabel:
        ml_path = output_dir / "multilabel.nii.gz"
        cmd.extend(["--ml", "--statistics"])

    start = time.time()
    log.info("Запуск: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    elapsed = time.time() - start

    run_info = {
        "patient": input_path.stem.replace(".nii", ""),
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "fast_mode": fast,
        "device": device,
        "elapsed_seconds": round(elapsed, 1),
        "return_code": result.returncode,
        "totalseg_version": get_totalseg_version(),
    }

    if result.returncode != 0:
        log.error("TotalSegmentator ОШИБКА (код %d):\n%s", result.returncode, result.stderr[-500:])
        run_info["error"] = result.stderr[-500:]
    else:
        log.info("Готово за %.1f сек: %s", elapsed, input_path.stem)

    return run_info


def get_totalseg_version() -> str:
    """Получает версию TotalSegmentator."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "totalsegmentator", "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_run_log(log_dir: Path, run_info: dict):
    """Сохраняет лог запуска в JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    patient = run_info["patient"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"seg_{patient}_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    log.info("Лог: %s", log_path)


def main():
    parser = argparse.ArgumentParser(description="Батч-сегментация TotalSegmentator")
    parser.add_argument("--patient", type=str, default=None, help="ID одного пациента")
    parser.add_argument("--fast", action="store_true", help="Быстрый режим (ниже качество)")
    parser.add_argument("--cpu", action="store_true", help="Принудительно CPU")
    args = parser.parse_args()

    cfg = load_config()
    seg_cfg = cfg["segmentation"]
    prep_dir = ROOT / cfg["paths"]["preprocessed"]
    auto_dir = ROOT / cfg["paths"]["seg_auto"]
    log_dir = ROOT / cfg["paths"]["logs"]

    device = "cpu" if args.cpu else seg_cfg["device"]
    fast = args.fast or seg_cfg["use_fast_mode"]
    task = seg_cfg["totalsegmentator_task"]
    multilabel = seg_cfg["multilabel_output"]

    if args.patient:
        nii_files = list(prep_dir.glob(f"{args.patient}*"))
    else:
        nii_files = sorted(prep_dir.glob("*.nii.gz"))

    if not nii_files:
        log.error("Нет файлов .nii.gz в %s", prep_dir)
        sys.exit(1)

    log.info("Найдено %d файлов для сегментации", len(nii_files))

    results = []
    for nii_path in nii_files:
        patient_id = nii_path.name.replace(".nii.gz", "")
        out_dir = auto_dir / patient_id

        # Пропустить, если уже сегментировано
        if (out_dir / "multilabel.nii.gz").exists():
            log.info("Пропуск %s — уже есть результат", patient_id)
            results.append(("SKIP", patient_id))
            continue

        run_info = run_totalsegmentator(
            nii_path, out_dir, task=task, fast=fast, device=device, multilabel=multilabel,
        )
        save_run_log(log_dir, run_info)

        status = "OK" if run_info["return_code"] == 0 else "FAIL"
        results.append((status, patient_id))

    log.info("=== Итого ===")
    for status, pid in results:
        log.info("  %s: %s", status, pid)

    ok = sum(1 for s, _ in results if s == "OK")
    skip = sum(1 for s, _ in results if s == "SKIP")
    fail = sum(1 for s, _ in results if s == "FAIL")
    log.info("OK=%d  SKIP=%d  FAIL=%d", ok, skip, fail)


if __name__ == "__main__":
    main()
