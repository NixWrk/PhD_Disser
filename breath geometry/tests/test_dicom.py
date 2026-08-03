import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    CTImageStorage,
    EnhancedCTImageStorage,
    ExplicitVRLittleEndian,
    generate_uid,
)

from breathgeom.io.dicom import (
    FORBIDDEN_IDENTITY_FIELDS,
    SeriesManifestRow,
    breath_hold_index,
    file_sha256,
    scan_dicom_series,
    series_checksum,
    slice_normal,
)

ENHANCED_CT_STORAGE = EnhancedCTImageStorage


def _write_enhanced_ct(
    path: Path,
    series_uid: str,
    z_positions: list[float],
    slice_thickness: float = 0.5,
) -> None:
    """Enhanced multi-frame CT whose geometry lives only in functional groups."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ENHANCED_CT_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = ENHANCED_CT_STORAGE
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.10"
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.PatientName = "PRIVATE^PERSON"
    dataset.PatientID = "DO-NOT-EXPORT"
    dataset.Rows = 512
    dataset.Columns = 512
    dataset.NumberOfFrames = len(z_positions)

    measures = Dataset()
    measures.PixelSpacing = [0.473, 0.473]
    measures.SliceThickness = slice_thickness
    shared = Dataset()
    shared.PixelMeasuresSequence = Sequence([measures])
    dataset.SharedFunctionalGroupsSequence = Sequence([shared])

    frames = []
    for index, z in enumerate(z_positions, start=1):
        position = Dataset()
        position.ImagePositionPatient = [-84.1381, -112.8881, z]
        orientation = Dataset()
        orientation.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        content = Dataset()
        content.StackID = "1"
        content.InStackPositionNumber = index
        content.TemporalPositionIndex = 1
        frame = Dataset()
        frame.PlanePositionSequence = Sequence([position])
        frame.PlaneOrientationSequence = Sequence([orientation])
        frame.FrameContentSequence = Sequence([content])
        frames.append(frame)
    dataset.PerFrameFunctionalGroupsSequence = Sequence(frames)
    dataset.save_as(path, enforce_file_format=True)


def _write_test_dicom(path: Path, series_uid: str, instance_number: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.SeriesNumber = 7
    dataset.InstanceNumber = instance_number
    dataset.PatientName = "PRIVATE^PERSON"
    dataset.PatientID = "DO-NOT-EXPORT"
    dataset.PatientPosition = "HFS"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.PixelSpacing = [0.8, 0.9]
    dataset.SliceThickness = 1.5
    dataset.save_as(path, enforce_file_format=True)


def test_scanner_groups_series_and_excludes_identity(tmp_path: Path) -> None:
    series_uid = "1.2.826.0.1.3680043.8.498.2"
    _write_test_dicom(tmp_path / "a", series_uid, 1)
    _write_test_dicom(tmp_path / "b.dcm", series_uid, 2)

    rows = scan_dicom_series(tmp_path)

    assert len(rows) == 1
    assert rows[0].file_count == 2
    assert rows[0].frame_count == 2
    assert not rows[0].enhanced_multiframe
    assert rows[0].pixel_spacing_x_mm == 0.9
    assert rows[0].pixel_spacing_y_mm == 0.8
    manifest_fields = set(SeriesManifestRow.__dataclass_fields__)
    assert manifest_fields.isdisjoint(FORBIDDEN_IDENTITY_FIELDS)


def test_scanner_reads_enhanced_ct_pixel_measures(tmp_path: Path) -> None:
    path = tmp_path / "enhanced_ct"
    series_uid = "1.2.826.0.1.3680043.8.498.3"
    _write_test_dicom(path, series_uid, 1)

    dataset = pydicom.dcmread(path)
    del dataset.PixelSpacing
    del dataset.SliceThickness
    dataset.NumberOfFrames = 1049
    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = [0.473, 0.473]
    pixel_measures.SliceThickness = 0.5
    shared = Dataset()
    shared.PixelMeasuresSequence = Sequence([pixel_measures])
    dataset.SharedFunctionalGroupsSequence = Sequence([shared])
    dataset.save_as(path, enforce_file_format=True)

    rows = scan_dicom_series(tmp_path)

    assert len(rows) == 1
    assert rows[0].enhanced_multiframe
    assert rows[0].frame_count == 1049
    assert rows[0].pixel_spacing_x_mm == 0.473
    assert rows[0].pixel_spacing_y_mm == 0.473
    assert rows[0].slice_thickness_mm == 0.5
    assert rows[0].pixel_spacing_source == ("SharedFunctionalGroupsSequence.PixelMeasuresSequence")


def test_schema_matches_manifest_row() -> None:
    """The published schema must not drift from the row actually written to CSV."""
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert set(schema["properties"]) == set(SeriesManifestRow.__dataclass_fields__)
    assert set(schema["required"]) <= set(SeriesManifestRow.__dataclass_fields__)


def test_slice_normal_of_axial_orientation() -> None:
    normal = slice_normal((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    assert list(normal) == [0.0, 0.0, 1.0]


def test_spacing_comes_from_positions_not_slice_thickness(tmp_path: Path) -> None:
    """Overlapping reconstruction: 0.5 mm slices reconstructed every 0.25 mm."""
    z_positions = [-707.50, -707.75, -708.00, -708.25, -708.50]
    _write_enhanced_ct(tmp_path / "overlapped", "1.2.826.0.1.3680043.8.498.11", z_positions)

    rows = scan_dicom_series(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row.slice_thickness_mm == 0.5
    assert row.spacing_z_mm == pytest.approx(0.25)
    assert row.spacing_z_uniform is True
    assert row.slice_count == 5
    assert row.coverage_z_mm == pytest.approx(1.0)
    assert row.gap_count == 0
    assert row.spacing_z_source == "PerFrameFunctionalGroupsSequence.PlanePositionSequence"
    assert row.image_orientation_patient == "1\\0\\0\\0\\1\\0"


def test_frames_are_ordered_by_physical_position(tmp_path: Path) -> None:
    """Storage order must not drive slice ordering; only physical position may."""
    shuffled = [-708.25, -707.50, -708.50, -708.00, -707.75]
    _write_enhanced_ct(tmp_path / "shuffled", "1.2.826.0.1.3680043.8.498.12", shuffled)

    rows = scan_dicom_series(tmp_path)

    assert rows[0].spacing_z_mm == pytest.approx(0.25)
    assert rows[0].spacing_z_uniform is True
    assert rows[0].coverage_z_mm == pytest.approx(1.0)


def test_missing_slice_is_reported_as_gap(tmp_path: Path) -> None:
    z_positions = [0.0, -0.25, -0.5, -1.0, -1.25]
    _write_enhanced_ct(tmp_path / "gapped", "1.2.826.0.1.3680043.8.498.13", z_positions)

    rows = scan_dicom_series(tmp_path)

    assert rows[0].spacing_z_mm == pytest.approx(0.25)
    assert rows[0].gap_count == 1
    assert rows[0].spacing_z_uniform is False
    assert rows[0].spacing_z_max_deviation_mm == pytest.approx(0.25)


def test_duplicate_position_is_counted(tmp_path: Path) -> None:
    z_positions = [0.0, -0.25, -0.25, -0.5]
    _write_enhanced_ct(tmp_path / "duplicated", "1.2.826.0.1.3680043.8.498.14", z_positions)

    rows = scan_dicom_series(tmp_path)

    assert rows[0].duplicate_position_count == 1
    assert rows[0].slice_count == 3
    assert rows[0].spacing_z_mm == pytest.approx(0.25)


def test_series_geometry_spans_several_files(tmp_path: Path) -> None:
    """A series is split across containers; geometry must aggregate over all of them."""
    series_uid = "1.2.826.0.1.3680043.8.498.15"
    _write_enhanced_ct(tmp_path / "part_a", series_uid, [0.0, -0.25, -0.5])
    _write_enhanced_ct(tmp_path / "part_b", series_uid, [-0.75, -1.0, -1.25])

    rows = scan_dicom_series(tmp_path)

    assert len(rows) == 1
    assert rows[0].file_count == 2
    assert rows[0].frame_count == 6
    assert rows[0].slice_count == 6
    assert rows[0].spacing_z_mm == pytest.approx(0.25)
    assert rows[0].spacing_z_uniform is True
    assert rows[0].gap_count == 0
    assert rows[0].coverage_z_mm == pytest.approx(1.25)


def test_breath_holds_split_by_time_gap() -> None:
    """Reconstructions of one held breath share an index; a later scan gets the next."""
    base = datetime(2020, 3, 17, 9, 40, 0)
    acquisitions = [base, base + timedelta(seconds=5), base + timedelta(seconds=208)]

    assert breath_hold_index(acquisitions, base) == (0, 0.0)
    assert breath_hold_index(acquisitions, base + timedelta(seconds=5)) == (0, 5.0)
    assert breath_hold_index(acquisitions, base + timedelta(seconds=208)) == (1, 208.0)


def test_breath_hold_index_without_timing() -> None:
    assert breath_hold_index([], None) == (None, None)


def test_manifest_never_exports_an_absolute_timestamp(tmp_path: Path) -> None:
    """Dates tied to a person are identifying; only offsets may leave the scanner."""
    _write_enhanced_ct(tmp_path / "timed", "1.2.826.0.1.3680043.8.498.20", [0.0, -0.25])

    rows = scan_dicom_series(tmp_path)
    exported = " ".join(str(value) for value in asdict(rows[0]).values())

    assert "2020" not in exported
    assert "acquisition_datetime" not in SeriesManifestRow.__dataclass_fields__


def test_checksums_are_opt_in_and_stable(tmp_path: Path) -> None:
    _write_enhanced_ct(tmp_path / "hashed", "1.2.826.0.1.3680043.8.498.21", [0.0, -0.25])

    without = scan_dicom_series(tmp_path)
    with_sums = scan_dicom_series(tmp_path, checksums=True)
    again = scan_dicom_series(tmp_path, checksums=True)

    assert without[0].source_checksum is None
    assert with_sums[0].source_checksum is not None
    assert with_sums[0].source_checksum == again[0].source_checksum


def test_series_checksum_ignores_file_order() -> None:
    assert series_checksum(["bb", "aa"]) == series_checksum(["aa", "bb"])
    assert series_checksum(["aa"]) != series_checksum(["aa", "bb"])


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"enhanced ct" * 200_000
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)

    assert file_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_contrast_and_fov_reach_the_manifest(tmp_path: Path) -> None:
    """Contrast and reconstruction diameter decide whether a series is usable."""
    path = tmp_path / "annotated"
    _write_enhanced_ct(path, "1.2.826.0.1.3680043.8.498.22", [0.0, -0.25])

    dataset = pydicom.dcmread(path)
    shared = dataset.SharedFunctionalGroupsSequence[0]
    usage = Dataset()
    usage.ContrastBolusAgentAdministered = "YES"
    shared.ContrastBolusUsageSequence = Sequence([usage])
    recon = Dataset()
    recon.ReconstructionDiameter = 220.312
    shared.CTReconstructionSequence = Sequence([recon])
    frame_type = Dataset()
    frame_type.FrameType = ["ORIGINAL", "PRIMARY", "CARDIAC_CTA"]
    shared.CTImageFrameTypeSequence = Sequence([frame_type])
    dataset.save_as(path, enforce_file_format=True)

    rows = scan_dicom_series(tmp_path)

    assert rows[0].contrast_administered == "YES"
    assert rows[0].reconstruction_diameter_mm == pytest.approx(220.312)
    assert rows[0].frame_type is not None
    assert "CARDIAC_CTA" in rows[0].frame_type


def test_slice_without_position_has_no_slice_geometry(tmp_path: Path) -> None:
    path = tmp_path / "no_position"
    _write_test_dicom(path, "1.2.826.0.1.3680043.8.498.16", 1)

    rows = scan_dicom_series(tmp_path)

    assert rows[0].spacing_z_mm is None
    assert rows[0].gap_count is None
