from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from breathgeom.io.dicom import (
    FORBIDDEN_IDENTITY_FIELDS,
    SeriesManifestRow,
    scan_dicom_series,
)


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
    assert rows[0].pixel_spacing_x_mm == 0.9
    assert rows[0].pixel_spacing_y_mm == 0.8
    manifest_fields = set(SeriesManifestRow.__dataclass_fields__)
    assert manifest_fields.isdisjoint(FORBIDDEN_IDENTITY_FIELDS)
