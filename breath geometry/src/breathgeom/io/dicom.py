from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError

SAFE_TAGS = [
    "SOPClassUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "SeriesNumber",
    "SeriesDescription",
    "ProtocolName",
    "PatientPosition",
    "Rows",
    "Columns",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "NominalScannedPixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "NumberOfFrames",
    "SharedFunctionalGroupsSequence",
    "Manufacturer",
    "ManufacturerModelName",
    "ConvolutionKernel",
    "KVP",
]

FORBIDDEN_IDENTITY_FIELDS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "AccessionNumber",
    "InstitutionName",
}

ARCHIVE_SUFFIXES = {".rar", ".zip", ".7z", ".tar", ".gz"}


@dataclass(frozen=True)
class DicomHeader:
    sop_class_uid: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
    series_number: int | None
    series_description: str | None
    protocol_name: str | None
    patient_position: str | None
    rows: int | None
    columns: int | None
    pixel_spacing_x_mm: float | None
    pixel_spacing_y_mm: float | None
    slice_thickness_mm: float | None
    manufacturer: str | None
    scanner_model: str | None
    convolution_kernel: str | None
    kvp: float | None
    number_of_frames: int
    pixel_spacing_source: str | None
    slice_thickness_source: str | None


@dataclass(frozen=True)
class SeriesManifestRow:
    series_instance_uid: str
    study_instance_uid: str
    modality: str
    sop_class_uid: str
    series_number: int | None
    series_description: str | None
    protocol_name: str | None
    patient_position: str | None
    file_count: int
    frame_count: int
    enhanced_multiframe: bool
    duplicate_sop_count: int
    geometry_variant_count: int
    rows: int | None
    columns: int | None
    pixel_spacing_x_mm: float | None
    pixel_spacing_y_mm: float | None
    pixel_spacing_source: str | None
    slice_thickness_mm: float | None
    slice_thickness_source: str | None
    manufacturer: str | None
    scanner_model: str | None
    convolution_kernel: str | None
    kvp: float | None
    source_root: str
    scan_status: str


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _functional_pixel_measures(dataset: pydicom.dataset.Dataset) -> Any:
    shared = dataset.get("SharedFunctionalGroupsSequence")
    if not shared:
        return None
    pixel_measures = shared[0].get("PixelMeasuresSequence")
    if not pixel_measures:
        return None
    return pixel_measures[0]


def _pixel_spacing(dataset: pydicom.dataset.Dataset) -> tuple[Any, str | None]:
    for keyword in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        value = dataset.get(keyword)
        if value is not None:
            return value, keyword
    measures = _functional_pixel_measures(dataset)
    if measures is not None and measures.get("PixelSpacing") is not None:
        return measures.get("PixelSpacing"), "SharedFunctionalGroupsSequence.PixelMeasuresSequence"
    return None, None


def _slice_thickness(dataset: pydicom.dataset.Dataset) -> tuple[float | None, str | None]:
    for keyword in ("SliceThickness", "SpacingBetweenSlices"):
        value = _optional_float(dataset.get(keyword))
        if value is not None:
            return value, keyword
    measures = _functional_pixel_measures(dataset)
    if measures is not None:
        value = _optional_float(measures.get("SliceThickness"))
        if value is not None:
            return value, "SharedFunctionalGroupsSequence.PixelMeasuresSequence"
    return None, None


def read_dicom_header(path: Path) -> DicomHeader | None:
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=False,
            specific_tags=SAFE_TAGS,
        )
    except (InvalidDicomError, OSError, PermissionError):
        return None

    study_uid = _optional_text(dataset.get("StudyInstanceUID"))
    series_uid = _optional_text(dataset.get("SeriesInstanceUID"))
    sop_uid = _optional_text(dataset.get("SOPInstanceUID"))
    if not study_uid or not series_uid or not sop_uid:
        return None

    pixel_spacing, pixel_spacing_source = _pixel_spacing(dataset)
    spacing_x: float | None = None
    spacing_y: float | None = None
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        spacing_y = _optional_float(pixel_spacing[0])
        spacing_x = _optional_float(pixel_spacing[1])
    slice_thickness, slice_thickness_source = _slice_thickness(dataset)

    return DicomHeader(
        sop_class_uid=_optional_text(dataset.get("SOPClassUID")) or "",
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        sop_instance_uid=sop_uid,
        modality=_optional_text(dataset.get("Modality")) or "",
        series_number=_optional_int(dataset.get("SeriesNumber")),
        series_description=_optional_text(dataset.get("SeriesDescription")),
        protocol_name=_optional_text(dataset.get("ProtocolName")),
        patient_position=_optional_text(dataset.get("PatientPosition")),
        rows=_optional_int(dataset.get("Rows")),
        columns=_optional_int(dataset.get("Columns")),
        pixel_spacing_x_mm=spacing_x,
        pixel_spacing_y_mm=spacing_y,
        slice_thickness_mm=slice_thickness,
        manufacturer=_optional_text(dataset.get("Manufacturer")),
        scanner_model=_optional_text(dataset.get("ManufacturerModelName")),
        convolution_kernel=_optional_text(dataset.get("ConvolutionKernel")),
        kvp=_optional_float(dataset.get("KVP")),
        number_of_frames=_optional_int(dataset.get("NumberOfFrames")) or 1,
        pixel_spacing_source=pixel_spacing_source,
        slice_thickness_source=slice_thickness_source,
    )


def iter_candidate_files(root: Path, max_files: int | None = None) -> Iterable[Path]:
    seen = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in ARCHIVE_SUFFIXES:
            continue
        yield path
        seen += 1
        if max_files is not None and seen >= max_files:
            return


def scan_dicom_series(root: Path, max_files: int | None = None) -> list[SeriesManifestRow]:
    grouped: dict[str, list[DicomHeader]] = defaultdict(list)
    for path in iter_candidate_files(root, max_files=max_files):
        header = read_dicom_header(path)
        if header is not None:
            grouped[header.series_instance_uid].append(header)

    rows: list[SeriesManifestRow] = []
    for series_uid, headers in sorted(grouped.items()):
        first = headers[0]
        sop_uids = [header.sop_instance_uid for header in headers]
        duplicate_count = len(sop_uids) - len(set(sop_uids))
        geometry_variants = {
            (
                header.rows,
                header.columns,
                header.pixel_spacing_x_mm,
                header.pixel_spacing_y_mm,
                header.slice_thickness_mm,
            )
            for header in headers
        }
        rows.append(
            SeriesManifestRow(
                series_instance_uid=series_uid,
                study_instance_uid=first.study_instance_uid,
                modality=first.modality,
                sop_class_uid=first.sop_class_uid,
                series_number=first.series_number,
                series_description=first.series_description,
                protocol_name=first.protocol_name,
                patient_position=first.patient_position,
                file_count=len(headers),
                frame_count=sum(header.number_of_frames for header in headers),
                enhanced_multiframe=any(header.number_of_frames > 1 for header in headers),
                duplicate_sop_count=duplicate_count,
                geometry_variant_count=len(geometry_variants),
                rows=first.rows,
                columns=first.columns,
                pixel_spacing_x_mm=first.pixel_spacing_x_mm,
                pixel_spacing_y_mm=first.pixel_spacing_y_mm,
                pixel_spacing_source=first.pixel_spacing_source,
                slice_thickness_mm=first.slice_thickness_mm,
                slice_thickness_source=first.slice_thickness_source,
                manufacturer=first.manufacturer,
                scanner_model=first.scanner_model,
                convolution_kernel=first.convolution_kernel,
                kvp=first.kvp,
                source_root=str(root),
                scan_status="preliminary" if max_files is not None else "complete_scan",
            )
        )
    return rows


def write_manifest_csv(rows: list[SeriesManifestRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SeriesManifestRow.__dataclass_fields__)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
