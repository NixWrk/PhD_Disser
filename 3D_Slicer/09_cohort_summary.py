"""
09_cohort_summary.py — Сводный отчёт QC по всей когорте.

Собирает все qc_<patient>.json из data/logs/ и формирует:
  - Таблицу в консоли (статус, объёмы, ключевые проблемы)
  - cohort_summary.csv (для анализа в Excel/pandas)
  - cohort_summary.json (для программной обработки)

Запуск:
    python 09_cohort_summary.py
    python 09_cohort_summary.py --log-dir path/to/logs
    python 09_cohort_summary.py --only-fail     # только проблемные
"""

import argparse
import csv
import json
import logging
import sys
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


log = setup_logging("09_summary")

STATUS_ICON = {
    "PASS": "OK ",
    "WARN": "!! ",
    "FAIL": "ERR",
    "MISSING": "---",
}


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_qc_reports(log_dir: Path) -> list[dict]:
    """Загружает все qc_*.json из папки логов."""
    reports = []
    for f in sorted(log_dir.glob("qc_*.json")):
        # Исключить qc_report.csv и другие не-JSON файлы
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            reports.append(data)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Не удалось прочитать %s: %s", f.name, e)
    return reports


def format_volume(vol_ml: float | str) -> str:
    """Форматирует объём для вывода."""
    if isinstance(vol_ml, (int, float)):
        return f"{vol_ml:6.0f}"
    return "  n/a"


def print_table(reports: list[dict], tissue_names: list[str]):
    """Выводит сводную таблицу в консоль."""
    # Заголовок
    tissue_cols = "  ".join(f"{t[:5]:>5}" for t in tissue_names)
    header = f"{'Пациент':<20} {'Статус':^6}  {'Issues':>6}  {'Покр.':>5}  {tissue_cols}"
    sep = "-" * len(header)

    log.info("\n%s\n%s\n%s", sep, header, sep)

    for r in reports:
        status = r.get("status", "?")
        icon = STATUS_ICON.get(status, "?  ")
        n_issues = r.get("n_issues", 0)
        coverage = r.get("coverage", {}).get("coverage_ratio", "")
        cov_str = f"{coverage:.3f}" if isinstance(coverage, float) else "  n/a"

        vols = r.get("volumes", {})
        vol_cols = "  ".join(
            format_volume(vols.get(t, {}).get("volume_ml", "n/a"))
            for t in tissue_names
        )

        patient = r.get("patient", "?")[:20]
        log.info(
            "%-20s [%s] %6d  %5s  %s",
            patient, icon, n_issues, cov_str, vol_cols,
        )

    log.info(sep)


def save_csv(reports: list[dict], tissue_names: list[str], out_path: Path):
    """Сохраняет сводный CSV."""
    fieldnames = ["patient", "status", "n_issues", "coverage_ratio"]
    fieldnames += [f"vol_{t}_ml" for t in tissue_names]
    fieldnames += ["issues_summary"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in reports:
            vols = r.get("volumes", {})
            issues = r.get("issues", [])
            issues_str = "; ".join(
                f"{i.get('type','?')}({i.get('tissue', '')})"
                for i in issues
            )

            row = {
                "patient": r.get("patient", ""),
                "status": r.get("status", ""),
                "n_issues": r.get("n_issues", 0),
                "coverage_ratio": r.get("coverage", {}).get("coverage_ratio", ""),
                "issues_summary": issues_str,
            }
            for t in tissue_names:
                row[f"vol_{t}_ml"] = vols.get(t, {}).get("volume_ml", "")
            writer.writerow(row)

    log.info("CSV сохранён: %s", out_path)


def print_stats(reports: list[dict]):
    """Выводит агрегированную статистику."""
    total = len(reports)
    if total == 0:
        log.info("Нет данных для статистики")
        return

    by_status: dict[str, int] = {}
    for r in reports:
        s = r.get("status", "UNKNOWN")
        by_status[s] = by_status.get(s, 0) + 1

    log.info("\n=== Итоговая статистика по когорте ===")
    log.info("Всего пациентов: %d", total)
    for status, count in sorted(by_status.items()):
        pct = 100 * count / total
        bar = "#" * int(pct / 5)
        log.info("  %s: %3d  (%5.1f%%)  %s", status, count, pct, bar)

    # Пациенты с ошибками — вывести списком
    fail_patients = [r["patient"] for r in reports if r.get("status") in ("FAIL", "MISSING")]
    if fail_patients:
        log.info("\nТребуют внимания (FAIL/MISSING):")
        for p in fail_patients:
            log.info("  - %s", p)

    warn_patients = [r["patient"] for r in reports if r.get("status") == "WARN"]
    if warn_patients:
        log.info("\nПредупреждения (WARN) — %d пациентов:", len(warn_patients))
        for p in warn_patients:
            log.info("  - %s", p)

    # Топ-3 самые частые проблемы
    issue_counts: dict[str, int] = {}
    for r in reports:
        for issue in r.get("issues", []):
            key = issue.get("type", "unknown")
            issue_counts[key] = issue_counts.get(key, 0) + 1

    if issue_counts:
        log.info("\nТоп проблем:")
        for issue_type, cnt in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]:
            log.info("  %-25s: %d", issue_type, cnt)


def main():
    parser = argparse.ArgumentParser(description="Сводный QC-отчёт по когорте")
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        help="Папка с qc_*.json (по умолчанию data/logs/)",
    )
    parser.add_argument(
        "--only-fail", action="store_true",
        help="Показать только пациентов со статусом FAIL/WARN",
    )
    args = parser.parse_args()

    cfg = load_config()
    log_dir = args.log_dir or (ROOT / cfg["paths"]["logs"])

    if not log_dir.exists():
        log.error("Папка логов не существует: %s", log_dir)
        sys.exit(1)

    reports = load_qc_reports(log_dir)
    if not reports:
        log.error("Нет QC-отчётов в %s (ожидаются файлы qc_*.json)", log_dir)
        sys.exit(1)

    if args.only_fail:
        reports = [r for r in reports if r.get("status") in ("FAIL", "WARN", "MISSING")]
        log.info("Фильтр: только FAIL/WARN/MISSING (%d пациентов)", len(reports))

    # Список тканей из конфига
    tissue_names = list(cfg["tissue_labels"].values())

    print_table(reports, tissue_names)
    print_stats(reports)

    # Сохранить CSV и JSON
    csv_path = log_dir / "cohort_summary.csv"
    save_csv(reports, tissue_names, csv_path)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_patients": len(reports),
        "by_status": {},
        "reports": reports,
    }
    for r in reports:
        s = r.get("status", "UNKNOWN")
        summary["by_status"][s] = summary["by_status"].get(s, 0) + 1

    json_path = log_dir / "cohort_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("JSON сохранён: %s", json_path)


if __name__ == "__main__":
    main()
