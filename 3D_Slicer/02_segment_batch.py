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


OOM_SIGNATURES = [
    "out of memory",
    "cuda out of memory",
    "runtimeerror: cuda",
    "outofmemoryerror",
    "cudamemoryerror",
]


def is_oom_error(stderr: str) -> bool:
    """Определяет, является ли ошибка нехваткой видеопамяти."""
    text = stderr.lower()
    return any(sig in text for sig in OOM_SIGNATURES)


def build_totalseg_cmd(
    input_path: Path,
    output_dir: Path,
    task: str,
    fast: bool,
    device: str,
    multilabel: bool,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "totalsegmentator",
        "-i", str(input_path),
        "-o", str(output_dir),
        "--task", task,
    ]
    if fast:
        cmd.append("--fast")
    cmd.extend(["--device", device])
    if multilabel:
        cmd.extend(["--ml", "--statistics"])
    return cmd


def run_totalsegmentator(
    input_path: Path,
    output_dir: Path,
    task: str = "total",
    fast: bool = False,
    device: str = "gpu",
    multilabel: bool = True,
    max_retries: int = 1,
    timeout_seconds: int = 3600,
    fallback_to_cpu: bool = True,
) -> dict:
    """
    Запускает TotalSegmentator CLI с поддержкой retry и fallback на CPU при OOM.

    При OOM-ошибке на GPU автоматически повторяет на CPU (если fallback_to_cpu=True).
    При других ошибках — повторяет max_retries раз.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = input_path.stem.replace(".nii", "")

    attempts = []
    current_device = device

    # Попытки: оригинальный device + fallback на CPU
    devices_to_try = [current_device]
    if fallback_to_cpu and current_device == "gpu":
        devices_to_try.append("cpu")

    final_result = None
    for attempt_device in devices_to_try:
        for retry in range(max_retries + 1):
            attempt_num = len(attempts) + 1
            cmd = build_totalseg_cmd(
                input_path, output_dir, task, fast, attempt_device, multilabel
            )

            label = f"попытка {attempt_num}, device={attempt_device}"
            if retry > 0:
                label += f", retry {retry}"
            log.info("TotalSegmentator: %s — %s", patient_id, label)
            log.info("CMD: %s", " ".join(cmd))

            start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                log.error(
                    "ТАЙМАУТ после %.0f сек (лимит: %d сек)",
                    elapsed, timeout_seconds,
                )
                attempts.append({
                    "attempt": attempt_num, "device": attempt_device,
                    "error": "timeout", "elapsed_seconds": round(elapsed, 1),
                })
                continue

            elapsed = time.time() - start
            rc = result.returncode

            attempt_info = {
                "attempt": attempt_num,
                "device": attempt_device,
                "return_code": rc,
                "elapsed_seconds": round(elapsed, 1),
            }

            if rc == 0:
                log.info("Успешно за %.1f сек (device=%s)", elapsed, attempt_device)
                attempts.append(attempt_info)
                final_result = result
                current_device = attempt_device
                break
            else:
                stderr_tail = result.stderr[-800:]
                log.error("ОШИБКА (код %d):\n%s", rc, stderr_tail)
                attempt_info["error"] = stderr_tail
                attempts.append(attempt_info)

                if is_oom_error(result.stderr) and attempt_device == "gpu" and fallback_to_cpu:
                    log.warning(
                        "Обнаружена нехватка GPU-памяти — переключаю на CPU"
                    )
                    break  # выйти из retry-цикла, перейти к CPU в devices_to_try
        else:
            # retry-цикл завершился без break → все попытки провалились на этом device
            continue

        if final_result is not None:
            break  # успех — выйти из devices_to_try

    run_info = {
        "patient": patient_id,
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "fast_mode": fast,
        "device_used": current_device,
        "elapsed_seconds": attempts[-1]["elapsed_seconds"] if attempts else 0,
        "return_code": final_result.returncode if final_result else -1,
        "totalseg_version": get_totalseg_version(),
        "attempts": attempts,
    }

    if final_result is None or final_result.returncode != 0:
        log.error(
            "TotalSegmentator не завершился успешно после %d попыток",
            len(attempts),
        )
        run_info["success"] = False
    else:
        run_info["success"] = True

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
            nii_path, out_dir,
            task=task,
            fast=fast,
            device=device,
            multilabel=multilabel,
            max_retries=seg_cfg.get("max_retries", 1),
            timeout_seconds=seg_cfg.get("timeout_seconds", 3600),
            fallback_to_cpu=seg_cfg.get("fallback_to_cpu_on_oom", True),
        )
        save_run_log(log_dir, run_info)

        status = "OK" if run_info.get("success") else "FAIL"
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
