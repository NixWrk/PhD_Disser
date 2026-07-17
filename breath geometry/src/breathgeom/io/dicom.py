from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SliceThickness",
    "SpacingBetweenSlices",
    "NumberOfFrames",
    "SharedFunctionalGroupsSequence",
    "PerFrameFunctionalGroupsSequence",
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

SHARED_GROUPS = "SharedFunctionalGroupsSequence"
PER_FRAME_GROUPS = "PerFrameFunctionalGroupsSequence"

# Two frames closer than this along the slice normal are the same physical location.
POSITION_TOLERANCE_MM = 1e-3
# Slice spacing is reported as uniform when every step stays within this of the median.
SPACING_TOLERANCE_MM = 0.01
# A step wider than this multiple of the median step is treated as a missing slice.
GAP_FACTOR = 1.5

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrameGeometry:
    """Per-frame physical geometry collected from a single DICOM container."""

    positions: tuple[tuple[float, float, float], ...]
    orientations: tuple[tuple[float, ...], ...]
    position_source: str | None
    orientation_source: str | None
    stack_ids: tuple[str, ...]
    temporal_indices: tuple[int, ...]


@dataclass(frozen=True)
class SeriesGeometry:
    """Slice ordering derived from physical positions, never from file names."""

    slice_count: int | None
    spacing_z_mm: float | None
    spacing_z_source: str | None
    spacing_z_uniform: bool | None
    spacing_z_max_deviation_mm: float | None
    coverage_z_mm: float | None
    duplicate_position_count: int | None
    gap_count: int | None
    image_orientation_patient: str | None
    orientation_source: str | None
    orientation_variant_count: int | None
    stack_count: int | None
    temporal_position_count: int | None


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
    geometry: FrameGeometry


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
    slice_count: int | None
    spacing_z_mm: float | None
    spacing_z_source: str | None
    spacing_z_uniform: bool | None
    spacing_z_max_deviation_mm: float | None
    coverage_z_mm: float | None
    duplicate_position_count: int | None
    gap_count: int | None
    image_orientation_patient: str | None
    orientation_source: str | None
    orientation_variant_count: int | None
    stack_count: int | None
    temporal_position_count: int | None
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


def _first_item(dataset: Any, keyword: str) -> Any:
    sequence = dataset.get(keyword) if dataset is not None else None
    if not sequence:
        return None
    return sequence[0]


def _functional_pixel_measures(dataset: pydicom.dataset.Dataset) -> Any:
    return _first_item(_first_item(dataset, SHARED_GROUPS), "PixelMeasuresSequence")


def _pixel_spacing(dataset: pydicom.dataset.Dataset) -> tuple[Any, str | None]:
    for keyword in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        value = dataset.get(keyword)
        if value is not None:
            return value, keyword
    measures = _functional_pixel_measures(dataset)
    if measures is not None and measures.get("PixelSpacing") is not None:
        return measures.get("PixelSpacing"), f"{SHARED_GROUPS}.PixelMeasuresSequence"
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
            return value, f"{SHARED_GROUPS}.PixelMeasuresSequence"
    return None, None


def _position_from_group(group: Any) -> tuple[float, float, float] | None:
    plane = _first_item(group, "PlanePositionSequence")
    value = plane.get("ImagePositionPatient") if plane is not None else None
    if value is None or len(value) < 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _orientation_from_group(group: Any) -> tuple[float, ...] | None:
    plane = _first_item(group, "PlaneOrientationSequence")
    value = plane.get("ImageOrientationPatient") if plane is not None else None
    if value is None or len(value) < 6:
        return None
    return tuple(float(component) for component in value[:6])


def _frame_content(group: Any) -> tuple[str | None, int | None]:
    content = _first_item(group, "FrameContentSequence")
    if content is None:
        return None, None
    stack_id = _optional_text(content.get("StackID"))
    temporal_index = _optional_int(content.get("TemporalPositionIndex"))
    return stack_id, temporal_index


def _enhanced_geometry(dataset: pydicom.dataset.Dataset) -> FrameGeometry:
    per_frame = dataset.get(PER_FRAME_GROUPS)
    shared_item = _first_item(dataset, SHARED_GROUPS)
    shared_orientation = _orientation_from_group(shared_item)

    positions: list[tuple[float, float, float]] = []
    orientations: set[tuple[float, ...]] = set()
    stack_ids: set[str] = set()
    temporal_indices: set[int] = set()
    orientation_source = f"{SHARED_GROUPS}.PlaneOrientationSequence" if shared_orientation else None

    for group in per_frame or []:
        position = _position_from_group(group)
        if position is not None:
            positions.append(position)
        orientation = shared_orientation or _orientation_from_group(group)
        if orientation is not None:
            orientations.add(orientation)
            if orientation_source is None:
                orientation_source = f"{PER_FRAME_GROUPS}.PlaneOrientationSequence"
        stack_id, temporal_index = _frame_content(group)
        if stack_id is not None:
            stack_ids.add(stack_id)
        if temporal_index is not None:
            temporal_indices.add(temporal_index)

    return FrameGeometry(
        positions=tuple(positions),
        orientations=tuple(sorted(orientations)),
        position_source=f"{PER_FRAME_GROUPS}.PlanePositionSequence" if positions else None,
        orientation_source=orientation_source,
        stack_ids=tuple(sorted(stack_ids)),
        temporal_indices=tuple(sorted(temporal_indices)),
    )


def _classic_geometry(dataset: pydicom.dataset.Dataset) -> FrameGeometry:
    position_value = dataset.get("ImagePositionPatient")
    positions: list[tuple[float, float, float]] = []
    if position_value is not None and len(position_value) >= 3:
        positions.append(
            (float(position_value[0]), float(position_value[1]), float(position_value[2]))
        )

    orientation_value = dataset.get("ImageOrientationPatient")
    orientations: list[tuple[float, ...]] = []
    orientation_source: str | None = None
    if orientation_value is not None and len(orientation_value) >= 6:
        orientations.append(tuple(float(component) for component in orientation_value[:6]))
        orientation_source = "ImageOrientationPatient"

    return FrameGeometry(
        positions=tuple(positions),
        orientations=tuple(orientations),
        position_source="ImagePositionPatient" if positions else None,
        orientation_source=orientation_source,
        stack_ids=(),
        temporal_indices=(),
    )


def _frame_geometry(dataset: pydicom.dataset.Dataset) -> FrameGeometry:
    if dataset.get(PER_FRAME_GROUPS):
        return _enhanced_geometry(dataset)
    return _classic_geometry(dataset)


def slice_normal(orientation: tuple[float, ...]) -> FloatArray:
    """Unit normal of the slice plane from the row and column direction cosines."""
    row = np.asarray(orientation[0:3], dtype=np.float64)
    column = np.asarray(orientation[3:6], dtype=np.float64)
    normal: FloatArray = np.cross(row, column)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("Degenerate ImageOrientationPatient: row and column are parallel")
    return normal / norm


def slice_positions_mm(
    positions: Iterable[tuple[float, float, float]],
    orientation: tuple[float, ...],
) -> FloatArray:
    """Project frame origins onto the slice normal and sort them physically.

    Ordering never uses file names or InstanceNumber, per Этап 2 of the master plan.
    """
    points = np.asarray(list(positions), dtype=np.float64)
    if points.size == 0:
        return np.empty(0, dtype=np.float64)
    projected: FloatArray = points @ slice_normal(orientation)
    return np.sort(projected)


def summarize_geometry(headers: list[DicomHeader]) -> SeriesGeometry:
    """Derive slice ordering, spacing uniformity, gaps and coverage for one series."""
    positions: list[tuple[float, float, float]] = []
    orientations: set[tuple[float, ...]] = set()
    stack_ids: set[str] = set()
    temporal_indices: set[int] = set()
    position_source: str | None = None
    orientation_source: str | None = None

    for header in headers:
        geometry = header.geometry
        positions.extend(geometry.positions)
        orientations.update(geometry.orientations)
        stack_ids.update(geometry.stack_ids)
        temporal_indices.update(geometry.temporal_indices)
        position_source = position_source or geometry.position_source
        orientation_source = orientation_source or geometry.orientation_source

    empty = SeriesGeometry(
        slice_count=None,
        spacing_z_mm=None,
        spacing_z_source=None,
        spacing_z_uniform=None,
        spacing_z_max_deviation_mm=None,
        coverage_z_mm=None,
        duplicate_position_count=None,
        gap_count=None,
        image_orientation_patient=None,
        orientation_source=None,
        orientation_variant_count=None,
        stack_count=len(stack_ids) or None,
        temporal_position_count=len(temporal_indices) or None,
    )
    if not positions or not orientations:
        return empty

    orientation = sorted(orientations)[0]
    orientation_text = "\\".join(f"{component:g}" for component in orientation)
    try:
        projected = slice_positions_mm(positions, orientation)
    except ValueError:
        return empty

    steps = np.diff(projected)
    duplicate_count = int(np.count_nonzero(steps < POSITION_TOLERANCE_MM))
    distinct = steps[steps >= POSITION_TOLERANCE_MM]
    coverage = float(projected[-1] - projected[0])

    if distinct.size == 0:
        return SeriesGeometry(
            slice_count=len(projected) - duplicate_count,
            spacing_z_mm=None,
            spacing_z_source=position_source,
            spacing_z_uniform=None,
            spacing_z_max_deviation_mm=None,
            coverage_z_mm=coverage,
            duplicate_position_count=duplicate_count,
            gap_count=None,
            image_orientation_patient=orientation_text,
            orientation_source=orientation_source,
            orientation_variant_count=len(orientations),
            stack_count=len(stack_ids) or None,
            temporal_position_count=len(temporal_indices) or None,
        )

    spacing = float(np.median(distinct))
    deviation = float(np.max(np.abs(distinct - spacing)))
    return SeriesGeometry(
        slice_count=len(projected) - duplicate_count,
        spacing_z_mm=spacing,
        spacing_z_source=position_source,
        spacing_z_uniform=bool(deviation <= SPACING_TOLERANCE_MM),
        spacing_z_max_deviation_mm=deviation,
        coverage_z_mm=coverage,
        duplicate_position_count=duplicate_count,
        gap_count=int(np.count_nonzero(distinct > GAP_FACTOR * spacing)),
        image_orientation_patient=orientation_text,
        orientation_source=orientation_source,
        orientation_variant_count=len(orientations),
        stack_count=len(stack_ids) or None,
        temporal_position_count=len(temporal_indices) or None,
    )


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
        geometry=_frame_geometry(dataset),
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
        geometry = summarize_geometry(headers)
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
                slice_count=geometry.slice_count,
                spacing_z_mm=geometry.spacing_z_mm,
                spacing_z_source=geometry.spacing_z_source,
                spacing_z_uniform=geometry.spacing_z_uniform,
                spacing_z_max_deviation_mm=geometry.spacing_z_max_deviation_mm,
                coverage_z_mm=geometry.coverage_z_mm,
                duplicate_position_count=geometry.duplicate_position_count,
                gap_count=geometry.gap_count,
                image_orientation_patient=geometry.image_orientation_patient,
                orientation_source=geometry.orientation_source,
                orientation_variant_count=geometry.orientation_variant_count,
                stack_count=geometry.stack_count,
                temporal_position_count=geometry.temporal_position_count,
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
