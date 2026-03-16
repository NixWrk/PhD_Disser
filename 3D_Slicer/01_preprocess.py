"""
01_preprocess.py — Конвертация DICOM → NIfTI с ресэмплингом и нормализацией ориентации.

Поддерживает плоскую структуру DICOM (все .dcm в одной папке, серии по именам файлов).
Группировка серий — через pydicom (быстро на сетевых дисках).

Выход: файлы .nii.gz в data/preprocessed/

Запуск:
    python 01_preprocess.py --list              # показать все серии
    python 01_preprocess.py --series 301        # конвертировать серию 301
    python 01_preprocess.py --series 301 4      # несколько серий
    python 01_preprocess.py --all               # все серии с >= min-slices
"""

import argparse
import collections
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

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


# ── Быстрое сканирование через имена файлов ──


def group_files_by_filename(dicom_dir: Path) -> dict[str, list[str]]:
    """
    Группирует .dcm файлы по номеру серии, извлечённому из имени файла.
    Паттерн: PATIENT.CT.STUDY.{series_num}.{instance}.{...}.dcm
    """
    series_files = collections.defaultdict(list)
    for fn in os.listdir(str(dicom_dir)):
        if not fn.lower().endswith(".dcm"):
            continue
        parts = fn.split(".")
        if len(parts) > 3:
            series_files[parts[3]].append(os.path.join(str(dicom_dir), fn))
    return dict(series_files)


def group_files_by_dicom_header(dicom_dir: Path) -> dict[str, list[str]]:
    """
    Запасной метод: группирует файлы по SeriesNumber из DICOM-заголовка.
    Медленнее, но работает для любых имён файлов.
    """
    series_files = collections.defaultdict(list)
    folder = str(dicom_dir)
    files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
    log.info("Сканирование заголовков %d файлов...", len(files))

    for i, fn in enumerate(files):
        if i % 1000 == 0 and i > 0:
            log.info("  %d / %d", i, len(files))
        fp = os.path.join(folder, fn)
        ds = pydicom.dcmread(fp, stop_before_pixels=True, specific_tags=[
            "SeriesNumber",
        ])
        snum = str(getattr(ds, "SeriesNumber", "unknown"))
        series_files[snum].append(fp)

    return dict(series_files)


def get_series_info(file_paths: list[str]) -> dict:
    """Читает метаданные из первого файла серии."""
    ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True)
    return {
        "series_number": str(getattr(ds, "SeriesNumber", "?")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "modality": str(getattr(ds, "Modality", "")),
        "patient_name": str(getattr(ds, "PatientName", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "slice_thickness": str(getattr(ds, "SliceThickness", "?")),
        "n_slices": len(file_paths),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
    }


# ── Конвертация и обработка ──


def read_series_from_files(file_paths: list[str]) -> sitk.Image:
    """Загружает DICOM-серию из списка файлов через SimpleITK."""
    reader = sitk.ImageSeriesReader()

    # SimpleITK нужны отсортированные файлы — сортируем по SeriesInstanceUID
    # Используем GDCM для правильной сортировки
    sorted_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(Path(file_paths[0]).parent),
    )

    # Фильтруем только наши файлы
    our_files_set = set(os.path.normpath(f) for f in file_paths)
    sorted_filtered = [f for f in sorted_files if os.path.normpath(f) in our_files_set]

    if not sorted_filtered:
        # Fallback: используем файлы как есть
        sorted_filtered = sorted(file_paths)

    reader.SetFileNames(sorted_filtered)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    return image


def resample_volume(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
) -> sitk.Image:
    """Ресэмплинг тома до заданного изотропного spacing."""
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    new_size = (original_size * original_spacing / np.array(target_spacing)).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)

    result = resampler.Execute(image)
    log.info(
        "Ресэмплинг: %s → %s, spacing %.2f×%.2f×%.2f",
        image.GetSize(), result.GetSize(), *target_spacing,
    )
    return result


def reorient_volume(image: sitk.Image, target_orientation: str = "LPS") -> sitk.Image:
    """Приведение к стандартной ориентации."""
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(target_orientation)
    return orienter.Execute(image)


def verify_hu_range(image: sitk.Image) -> bool:
    """Проверяет что значения похожи на HU."""
    arr = sitk.GetArrayViewFromImage(image)
    vmin, vmax = float(arr.min()), float(arr.max())
    is_hu = vmin < -500 and vmax > 100
    if not is_hu:
        log.warning("Подозрительный диапазон: [%.0f, %.0f] — не HU?", vmin, vmax)
    return is_hu


def make_safe_filename(info: dict) -> str:
    """Генерирует безопасное имя файла из метаданных."""
    parts = []
    patient = info.get("patient_name", "").replace("^", "_").replace(" ", "_")
    if patient:
        parts.append(patient)
    snum = info.get("series_number", "")
    if snum:
        parts.append(f"s{snum}")
    desc = info.get("series_description", "").replace(" ", "_").replace("/", "_")
    if desc:
        parts.append(desc)
    if not parts:
        parts.append("unknown")
    name = "_".join(parts)
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name


def process_series(
    file_paths: list[str],
    info: dict,
    output_dir: Path,
    cfg: dict,
) -> Path:
    """Полный пайплайн: DICOM → reorient → resample → save .nii.gz"""
    prep_cfg = cfg["preprocess"]
    spacing = tuple(prep_cfg["target_spacing_mm"])
    orientation = prep_cfg["target_orientation"]

    log.info("Загрузка серии %s (%s, %d слайсов)...",
             info["series_number"], info["series_description"], info["n_slices"])

    image = read_series_from_files(file_paths)
    log.info("Загружено: size=%s, spacing=%.2f×%.2f×%.2f",
             image.GetSize(), *image.GetSpacing())

    verify_hu_range(image)
    image = reorient_volume(image, orientation)
    image = resample_volume(image, spacing)

    filename = make_safe_filename(info)
    output_path = output_dir / f"{filename}{prep_cfg['output_format']}"
    sitk.WriteImage(image, str(output_path), useCompression=True)
    log.info("Сохранено: %s", output_path)
    return output_path


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="DICOM → NIfTI предобработка")
    parser.add_argument("--list", action="store_true",
                        help="Показать доступные серии и выйти")
    parser.add_argument("--series", nargs="+", type=str, default=None,
                        help="Номера серий для конвертации (через пробел)")
    parser.add_argument("--all", action="store_true",
                        help="Конвертировать все подходящие серии")
    parser.add_argument("--min-slices", type=int, default=50,
                        help="Минимум слайсов (по умолчанию 50)")
    parser.add_argument("--use-headers", action="store_true",
                        help="Группировать по DICOM-заголовкам (медленно)")
    args = parser.parse_args()

    cfg = load_config()
    dicom_path = cfg["paths"]["raw_dicom"]
    dicom_dir = Path(dicom_path)
    if not dicom_dir.is_absolute():
        dicom_dir = ROOT / dicom_dir

    output_dir = ROOT / cfg["paths"]["preprocessed"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("DICOM: %s", dicom_dir)

    # Группировка файлов
    if args.use_headers:
        series_map = group_files_by_dicom_header(dicom_dir)
    else:
        series_map = group_files_by_filename(dicom_dir)

    log.info("Найдено серий: %d", len(series_map))

    # Метаданные
    all_info = {}
    for snum, files in series_map.items():
        all_info[snum] = get_series_info(files)

    # --list
    if args.list:
        hdr = "{:>6}  {:>7}  {:<40}  {:<12}  {:<8}  {}".format(
            "Series", "Slices", "Description", "Size", "SlThk", "Mod")
        print(hdr)
        print("-" * 95)
        for snum in sorted(all_info.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
            info = all_info[snum]
            size = "{}x{}".format(info.get("rows", "?"), info.get("columns", "?"))
            print("{:>6}  {:>7}  {:<40}  {:<12}  {:<8}  {}".format(
                snum, info["n_slices"], info["series_description"][:40],
                size, info["slice_thickness"][:8], info["modality"]))
        return

    # Выбор серий
    if args.series:
        selected = {s: series_map[s] for s in args.series if s in series_map}
        missing = [s for s in args.series if s not in series_map]
        if missing:
            log.warning("Серии не найдены: %s", missing)
    elif args.all:
        selected = {}
        for snum, files in series_map.items():
            info = all_info[snum]
            if info["n_slices"] < args.min_slices:
                continue
            if info["modality"] != "CT":
                continue
            # Пропускаем скауты и скриншоты
            desc = info["series_description"].lower()
            if any(skip in desc for skip in ["scout", "screen save", "dose", "ecg", "smart prep"]):
                continue
            selected[snum] = files
    else:
        parser.error("Укажи --list, --series или --all")
        return

    if not selected:
        log.error("Нет серий для обработки")
        sys.exit(1)

    log.info("К обработке: %d серий", len(selected))

    results = []
    for snum, files in sorted(selected.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99999):
        info = all_info[snum]
        try:
            out = process_series(files, info, output_dir, cfg)
            results.append(("OK", snum, str(out)))
        except Exception as e:
            log.error("ОШИБКА серия %s: %s", snum, e)
            results.append(("FAIL", snum, str(e)))

    log.info("=== Итого ===")
    for status, snum, detail in results:
        log.info("  %s: серия %s — %s", status, snum, detail)
    ok = sum(1 for r in results if r[0] == "OK")
    log.info("Успешно: %d / %d", ok, len(results))


if __name__ == "__main__":
    main()
