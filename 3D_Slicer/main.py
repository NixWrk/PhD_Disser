"""
main.py — Точка входа для запуска пайплайна сегментации из IDE.

Запуск всего пайплайна:
    python main.py --all

Запуск отдельных шагов:
    python main.py --step preprocess
    python main.py --step segment
    python main.py --step postprocess
    python main.py --step qc

Один пациент:
    python main.py --all --patient P001

Быстрый режим TotalSegmentator (ниже качество):
    python main.py --all --fast

CPU-only:
    python main.py --all --cpu
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

STEPS = {
    "preprocess":  "01_preprocess.py",
    "segment":     "02_segment_batch.py",
    "postprocess": "03_postprocess.py",
    "qc":          "04_qc.py",
    "export":      "05_export_comsol.py",
}


def run_step(step_name: str, script: str, extra_args: list[str]) -> bool:
    """Запускает один шаг пайплайна как подпроцесс."""
    script_path = ROOT / script
    if not script_path.exists():
        log.error("Скрипт не найден: %s", script_path)
        return False

    cmd = [sys.executable, str(script_path)] + extra_args
    log.info("=" * 60)
    log.info("ШАГ: %s (%s)", step_name, script)
    log.info("CMD: %s", " ".join(cmd))
    log.info("=" * 60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        log.error("ОШИБКА на шаге '%s' (код %d)", step_name, result.returncode)
        return False

    log.info("Шаг '%s' завершён успешно.", step_name)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Пайплайн сегментации КТ → labelmap для COMSOL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --all                       # полный пайплайн
  python main.py --step preprocess           # только предобработка
  python main.py --step segment --fast       # сегментация (быстрая)
  python main.py --all --patient P001        # один пациент
  python main.py --step qc                   # только QC
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Запустить все шаги последовательно")
    group.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        help="Запустить один шаг: preprocess | segment | postprocess | qc | export",
    )

    parser.add_argument("--patient", type=str, default=None, help="ID одного пациента")
    parser.add_argument("--fast", action="store_true", help="Быстрый режим TotalSegmentator")
    parser.add_argument("--cpu", action="store_true", help="Принудительно CPU")
    parser.add_argument("--mhd", action="store_true", help="Экспорт в MHD/RAW вместо NRRD")

    args = parser.parse_args()

    # Собираем аргументы для проброса
    def build_extra_args(step_name: str) -> list[str]:
        extra = []
        if args.patient:
            extra += ["--patient", args.patient]
        if step_name == "segment":
            if args.fast:
                extra.append("--fast")
            if args.cpu:
                extra.append("--cpu")
        if step_name == "export" and args.mhd:
            extra.append("--mhd")
        return extra

    if args.all:
        steps_to_run = list(STEPS.items())
    else:
        steps_to_run = [(args.step, STEPS[args.step])]

    log.info("Пайплайн: %s", " → ".join(name for name, _ in steps_to_run))

    for step_name, script in steps_to_run:
        extra = build_extra_args(step_name)
        ok = run_step(step_name, script, extra)
        if not ok:
            log.error("Пайплайн остановлен из-за ошибки на шаге '%s'", step_name)
            sys.exit(1)

    log.info("=" * 60)
    log.info("ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
