"""
04_qc.py — Контроль качества labelmaps: объёмы, связность, покрытие, аномалии.

Вход:  data/segmentations/final_labelmap/<patient_id>_labelmap.nii.gz
Выход: data/logs/qc_report.csv + per-patient JSON

Запуск:
    python 04_qc.py                  # все пациенты
    python 04_qc.py --patient P001   # один
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def compute_volumes_ml(labelmap: np.ndarray, spacing_mm: tuple, labels: dict) -> dict:
    """Вычисляет объём каждого сегмента в мл."""
    voxel_vol_ml = float(np.prod(spacing_mm)) / 1000.0  # мм³ → мл
    volumes = {}
    for label_str, tissue_name in labels.items():
        label_val = int(label_str)
        count = int(np.sum(labelmap == label_val))
        volumes[tissue_name] = {
            "voxels": count,
            "volume_ml": round(count * voxel_vol_ml, 1),
        }
    return volumes


def check_volume_ranges(volumes: dict, expected: dict) -> list[dict]:
    """Проверяет объёмы на попадание в допустимые диапазоны."""
    issues = []
    for tissue, vol_info in volumes.items():
        vol = vol_info["volume_ml"]
        if tissue in expected:
            exp = expected[tissue]
            if vol < exp["min"]:
                issues.append({
                    "type": "volume_low",
                    "tissue": tissue,
                    "value_ml": vol,
                    "expected_min": exp["min"],
                    "severity": "warning",
                })
            elif vol > exp["max"]:
                issues.append({
                    "type": "volume_high",
                    "tissue": tissue,
                    "value_ml": vol,
                    "expected_max": exp["max"],
                    "severity": "warning",
                })
    return issues


def count_islands(labelmap: np.ndarray, label_val: int) -> int:
    """Считает количество связных компонент для одной метки."""
    binary = (labelmap == label_val).astype(np.uint8)
    if binary.sum() == 0:
        return 0
    _, n_components = ndimage.label(binary)
    return n_components


def check_connectivity(labelmap: np.ndarray, labels: dict, max_islands: int) -> list[dict]:
    """Проверяет что каждая ткань не фрагментирована."""
    issues = []
    for label_str, tissue_name in labels.items():
        label_val = int(label_str)
        n_islands = count_islands(labelmap, label_val)
        if n_islands > max_islands:
            issues.append({
                "type": "fragmented",
                "tissue": tissue_name,
                "islands": n_islands,
                "max_allowed": max_islands,
                "severity": "warning",
            })
    return issues


def check_coverage(labelmap: np.ndarray, body_threshold: int = 1) -> dict:
    """Проверяет что нет пустых вокселей внутри тела."""
    body_voxels = int(np.sum(labelmap >= body_threshold))
    # Общее количество непустых
    total_nonzero = int(np.sum(labelmap > 0))

    return {
        "body_voxels": body_voxels,
        "labeled_voxels": total_nonzero,
        "coverage_ratio": round(total_nonzero / body_voxels, 6) if body_voxels > 0 else 0,
    }


def check_boundary_smoothness(labelmap: np.ndarray, labels: dict) -> list[dict]:
    """
    Простая проверка: считает долю граничных вокселей для каждой ткани.
    Высокая доля может указывать на «рваные» границы.
    """
    issues = []
    struct = ndimage.generate_binary_structure(3, 1)

    for label_str, tissue_name in labels.items():
        label_val = int(label_str)
        binary = (labelmap == label_val).astype(np.uint8)
        total = binary.sum()
        if total == 0:
            continue

        eroded = ndimage.binary_erosion(binary, structure=struct)
        boundary = binary.astype(int) - eroded.astype(int)
        boundary_ratio = float(boundary.sum()) / float(total)

        # Если > 60% вокселей — граничные, ткань слишком «тонкая» или фрагментированная
        if boundary_ratio > 0.6 and tissue_name != "skin":
            issues.append({
                "type": "thin_or_rough",
                "tissue": tissue_name,
                "boundary_ratio": round(boundary_ratio, 3),
                "severity": "info",
            })

    return issues


def qc_patient(patient_id: str, cfg: dict) -> dict:
    """Полный QC одного пациента."""
    final_dir = ROOT / cfg["paths"]["seg_final"]
    labelmap_path = final_dir / f"{patient_id}_labelmap.nii.gz"

    if not labelmap_path.exists():
        return {"patient": patient_id, "status": "MISSING", "issues": []}

    img = nib.load(str(labelmap_path))
    labelmap = img.get_fdata().astype(np.uint8)
    spacing = img.header.get_zooms()[:3]
    labels = cfg["tissue_labels"]
    qc_cfg = cfg["qc"]

    log.info("QC: %s, shape=%s", patient_id, labelmap.shape)

    # 1. Объёмы
    volumes = compute_volumes_ml(labelmap, spacing, labels)
    for tissue, vi in volumes.items():
        log.info("  %s: %.1f мл (%d вокс)", tissue, vi["volume_ml"], vi["voxels"])

    # 2. Проверка диапазонов
    issues = check_volume_ranges(volumes, qc_cfg["expected_volumes_ml"])

    # 3. Связность
    issues += check_connectivity(labelmap, labels, qc_cfg["max_islands_per_tissue"])

    # 4. Покрытие
    coverage = check_coverage(labelmap)
    log.info("  Покрытие: %.4f", coverage["coverage_ratio"])
    if coverage["coverage_ratio"] < qc_cfg["min_coverage_ratio"]:
        issues.append({
            "type": "low_coverage",
            "coverage_ratio": coverage["coverage_ratio"],
            "min_required": qc_cfg["min_coverage_ratio"],
            "severity": "error",
        })

    # 5. Границы
    issues += check_boundary_smoothness(labelmap, labels)

    # Итоговый статус
    has_errors = any(i["severity"] == "error" for i in issues)
    has_warnings = any(i["severity"] == "warning" for i in issues)

    if has_errors:
        status = "FAIL"
    elif has_warnings:
        status = "WARN"
    else:
        status = "PASS"

    for issue in issues:
        level = "⛔" if issue["severity"] == "error" else "⚠️" if issue["severity"] == "warning" else "ℹ️"
        log.info("  %s %s: %s", level, issue["type"], issue.get("tissue", ""))

    report = {
        "patient": patient_id,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "volumes": volumes,
        "coverage": coverage,
        "issues": issues,
        "n_issues": len(issues),
    }

    return report


def save_csv_report(reports: list[dict], csv_path: Path, labels: dict):
    """Сводный CSV-отчёт по всем пациентам."""
    tissue_names = list(labels.values())

    fieldnames = ["patient", "status", "n_issues", "coverage"]
    fieldnames += [f"vol_{t}_ml" for t in tissue_names]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            row = {
                "patient": r["patient"],
                "status": r["status"],
                "n_issues": r["n_issues"],
                "coverage": r.get("coverage", {}).get("coverage_ratio", ""),
            }
            for t in tissue_names:
                vol = r.get("volumes", {}).get(t, {}).get("volume_ml", "")
                row[f"vol_{t}_ml"] = vol
            writer.writerow(row)

    log.info("CSV-отчёт: %s", csv_path)


def main():
    parser = argparse.ArgumentParser(description="QC labelmaps")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    final_dir = ROOT / cfg["paths"]["seg_final"]
    log_dir = ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)

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

    reports = []
    for pid in patients:
        report = qc_patient(pid, cfg)
        reports.append(report)

        # Сохранить per-patient JSON
        json_path = log_dir / f"qc_{pid}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    # Сводный CSV
    csv_path = log_dir / "qc_report.csv"
    save_csv_report(reports, csv_path, cfg["tissue_labels"])

    # Итого
    log.info("=== Итого QC ===")
    for r in reports:
        log.info("  %s: %s (%d замечаний)", r["patient"], r["status"], r["n_issues"])

    passed = sum(1 for r in reports if r["status"] == "PASS")
    log.info("PASS: %d / %d", passed, len(reports))


if __name__ == "__main__":
    main()
