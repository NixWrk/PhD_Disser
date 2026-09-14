"""Synthetic, uncompressed Enhanced CT checks; no external data or codecs."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

MODULE_PATH = Path(__file__).parents[1] / "enhanced_cardiac_ct.py"
SPEC = importlib.util.spec_from_file_location("enhanced_cardiac_ct", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def macro(**fields):
    item = Dataset()
    for name, value in fields.items():
        setattr(item, name, value)
    return Sequence([item])


def synthetic_phase(*, reversed_order=True, per_frame_hu=True):
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = module.ENHANCED_CT_UID
    dataset.SOPInstanceUID = generate_uid()
    dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.99"
    dataset.PatientName = "SYNTHETIC^PRIVATE_MARKER"
    dataset.PatientID = "PRIVATE_TEST_IDENTIFIER"
    dataset.Modality = "CT"
    dataset.Rows, dataset.Columns, dataset.NumberOfFrames = 2, 3, 4
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    root_half = np.sqrt(0.5)
    u = np.array([root_half, root_half, 0.0])
    v = np.array([-0.5, 0.5, root_half])
    normal = np.cross(u, v)
    origin = np.array([12.0, -24.0, 7.0])
    shared = Dataset()
    shared.PlaneOrientationSequence = macro(ImageOrientationPatient=[*u, *v])
    shared.PixelMeasuresSequence = macro(PixelSpacing=[0.7, 1.1], SliceThickness=5.0)
    if not per_frame_hu:
        shared.PixelValueTransformationSequence = macro(
            RescaleSlope=1.25, RescaleIntercept=-999.5, RescaleType="HU")
    dataset.SharedFunctionalGroupsSequence = Sequence([shared])
    frames, arrays, hu = [], [], []
    spatial_indices = [3, 2, 1, 0] if reversed_order else [0, 1, 2, 3]
    for frame_index, spatial_index in enumerate(spatial_indices):
        frame = Dataset()
        position = origin + spatial_index * 2.5 * normal
        frame.PlanePositionSequence = macro(ImagePositionPatient=position.tolist())
        slope, intercept = ((1.5 + frame_index / 2, -1000.0 + frame_index / 4)
                            if per_frame_hu else (1.25, -999.5))
        if per_frame_hu:
            frame.PixelValueTransformationSequence = macro(
                RescaleSlope=slope, RescaleIntercept=intercept, RescaleType="HU")
        pixels = np.arange(6, dtype=np.int16).reshape(2, 3) + spatial_index * 100
        arrays.append(pixels)
        hu.append(pixels * slope + intercept)
        frames.append(frame)
    dataset.PerFrameFunctionalGroupsSequence = Sequence(frames)
    dataset.PixelData = np.stack(arrays).astype("<i2").tobytes()
    return dataset, np.stack(hu), (origin, u, v, normal)


def write_dicom(path, dataset):
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)


class EnhancedCTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_oblique_reverse_frames_hu_and_physical_step(self):
        dataset, hu, axes = synthetic_phase()
        source = self.root / "private_source_name.dcm"
        write_dicom(source, dataset)
        image, info = module.read_phase(source)
        self.assertEqual(image.shape, (2, 3, 4))
        self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
        np.testing.assert_array_equal(
            np.asarray(image.dataobj), hu[::-1].transpose(1, 2, 0).astype(np.float32))
        origin, u, v, normal = axes
        # Independently evaluate voxel locations in patient coordinates.
        for row, col, slice_index in ((0, 0, 0), (1, 2, 3), (1, 1, 1)):
            expected_lps = origin + row * 0.7 * v + col * 1.1 * u + slice_index * 2.5 * normal
            expected_ras = expected_lps * [-1, -1, 1]
            np.testing.assert_allclose(
                (image.affine @ [row, col, slice_index, 1])[:3], expected_ras, atol=1e-6)
        self.assertEqual(info["source_order"], [3, 2, 1, 0])
        self.assertEqual(info["source_frame_numbers_1based"], [4, 3, 2, 1])
        self.assertAlmostEqual(info["voxel_spacing_mm"][2], 2.5)
        self.assertEqual(info["slice_thickness_mm"], 5.0)
        self.assertAlmostEqual(abs(np.linalg.det(image.affine[:3, :3])), 0.7 * 1.1 * 2.5)
        self.assertEqual(image.header.get_xyzt_units(), ("mm", "unknown"))
        destination = self.root / "roundtrip.nii.gz"
        nib.save(image, destination)
        restored = nib.load(destination)
        np.testing.assert_array_equal(restored.get_fdata(), np.asarray(image.dataobj))
        np.testing.assert_allclose(restored.get_sform(), image.affine, atol=2e-6)
        np.testing.assert_allclose(restored.get_qform(), image.affine, atol=2e-6)

    def test_per_frame_geometry_and_shared_rescale(self):
        dataset, hu, _ = synthetic_phase(reversed_order=False, per_frame_hu=False)
        shared = dataset.SharedFunctionalGroupsSequence[0]
        for frame in dataset.PerFrameFunctionalGroupsSequence:
            frame.PlaneOrientationSequence = copy.deepcopy(shared.PlaneOrientationSequence)
            frame.PixelMeasuresSequence = copy.deepcopy(shared.PixelMeasuresSequence)
        del shared.PlaneOrientationSequence
        del shared.PixelMeasuresSequence
        path = self.root / "per_frame_geometry.dcm"
        write_dicom(path, dataset)
        image, info = module.read_phase(path)
        np.testing.assert_array_equal(
            np.asarray(image.dataobj), hu.transpose(1, 2, 0).astype(np.float32))
        self.assertEqual(info["functional_group_sources_by_slice"]["PixelMeasuresSequence"],
                         ["per_frame"] * 4)
        self.assertEqual(info["functional_group_sources_by_slice"]["PixelValueTransformationSequence"],
                         ["shared"] * 4)

    def test_rejects_invalid_geometry_and_rescale(self):
        cases = ("duplicate", "nonuniform", "mixed_orientation", "nonfinite_position",
                 "nonfinite_rescale", "inplane_shift", "nonpositive_spacing", "mixed_thickness",
                 "frame_count", "missing_rescale", "nonorthogonal", "non_hu")
        for problem in cases:
            with self.subTest(problem=problem):
                dataset, _, axes = synthetic_phase(reversed_order=False)
                frames = dataset.PerFrameFunctionalGroupsSequence
                shared = dataset.SharedFunctionalGroupsSequence[0]
                if problem == "duplicate":
                    frames[1].PlanePositionSequence = copy.deepcopy(frames[0].PlanePositionSequence)
                elif problem == "nonuniform":
                    point = np.array(frames[1].PlanePositionSequence[0].ImagePositionPatient) + axes[3] * 0.2
                    frames[1].PlanePositionSequence[0].ImagePositionPatient = point.tolist()
                elif problem == "mixed_orientation":
                    frames[2].PlaneOrientationSequence = macro(ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
                elif problem == "nonfinite_position":
                    frames[1].PlanePositionSequence[0].ImagePositionPatient = [float("nan"), 0, 0]
                elif problem == "nonfinite_rescale":
                    frames[1].PixelValueTransformationSequence[0].RescaleSlope = float("inf")
                elif problem == "inplane_shift":
                    point = np.array(frames[1].PlanePositionSequence[0].ImagePositionPatient) + axes[1] * 0.2
                    frames[1].PlanePositionSequence[0].ImagePositionPatient = point.tolist()
                elif problem == "nonpositive_spacing":
                    shared.PixelMeasuresSequence[0].PixelSpacing = [0, 1]
                elif problem == "mixed_thickness":
                    frames[1].PixelMeasuresSequence = macro(PixelSpacing=[0.7, 1.1], SliceThickness=0.5)
                elif problem == "frame_count":
                    frames.pop()
                elif problem == "missing_rescale":
                    del frames[1].PixelValueTransformationSequence
                elif problem == "nonorthogonal":
                    shared.PlaneOrientationSequence[0].ImageOrientationPatient = [1, 0, 0, 1, 0, 0]
                elif problem == "non_hu":
                    frames[0].PixelValueTransformationSequence[0].RescaleType = "US"
                path = self.root / f"{problem}.dcm"
                write_dicom(path, dataset)
                with self.assertRaises(ValueError):
                    module.read_phase(path)

    def make_manifest(self, *, shifted_second=False):
        entries = []
        for percent in (95, 0):
            dataset, _, axes = synthetic_phase()
            if shifted_second and percent == 95:
                for frame in dataset.PerFrameFunctionalGroupsSequence:
                    position = np.asarray(frame.PlanePositionSequence[0].ImagePositionPatient) + axes[1] * 0.2
                    frame.PlanePositionSequence[0].ImagePositionPatient = position.tolist()
            source = self.root / f"private-{percent}.dcm"
            write_dicom(source, dataset)
            entries.append({"source_path": source.name, "phase_percent": percent})
        manifest = self.root / "selected.local.json"
        manifest.write_text(json.dumps({"case_id": "case01", "sequence_id": "seq01",
                                       "selection_basis": "PRIVATE_SELECTION_MARKER",
                                       "phases": entries}), encoding="utf-8")
        return manifest

    def test_cli_contract_sorting_hashes_and_privacy(self):
        manifest = self.make_manifest()
        output = self.root / "case01" / "seq01" / "phases"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(["--manifest", str(manifest), "--output-dir", str(output),
                                         "--case-id", "case01", "--sequence-id", "seq01"]), 0)
        text = (output / "conversion_manifest.json").read_text(encoding="utf-8")
        result = json.loads(text)
        self.assertEqual(result["phase_count"], 2)
        self.assertEqual(result["source_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual([p["phase_percent"] for p in result["phases"]], [0, 95])
        self.assertEqual([p["source_manifest_phase_index"] for p in result["phases"]], [1, 0])
        for index, phase in enumerate(result["phases"]):
            self.assertEqual(phase["phase_id"], f"phase_{index:02d}")
            nifti = output / phase["nifti"]
            self.assertEqual(phase["sha256"], hashlib.sha256(nifti.read_bytes()).hexdigest())
            source = self.root / f"private-{int(phase['phase_percent'])}.dcm"
            self.assertEqual(phase["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            original = pydicom.dcmread(source, stop_before_pixels=True)
            self.assertNotIn(str(original.SOPInstanceUID), text)
            self.assertNotIn(str(original.FrameOfReferenceUID), text)
            self.assertNotIn("rr_interval_ms", phase)
            self.assertNotIn("derived_time_from_R0_ms", phase)
        for private_text in ("SYNTHETIC", "PRIVATE_TEST_IDENTIFIER", "private-", "PRIVATE_SELECTION_MARKER", str(self.root)):
            self.assertNotIn(private_text, text)
        with self.assertRaises(FileExistsError):
            module.convert_manifest(manifest, output, "case01", "seq01")

    def test_invalid_later_phase_publishes_no_partial_output(self):
        manifest = self.make_manifest(shifted_second=True)
        output = self.root / "sequence" / "phases"
        with self.assertRaisesRegex(ValueError, "Mixed geometry across phases"):
            module.convert_manifest(manifest, output, "case01", "seq01")
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_manifest_identity_and_phase_validation(self):
        manifest = self.make_manifest()
        original = json.loads(manifest.read_text(encoding="utf-8"))
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "case_id differs"):
            module.convert_manifest(manifest, output, "other_case", "seq01")
        for invalid in (float("nan"), float("inf"), -5, 100, True, 0):
            with self.subTest(phase_percent=invalid):
                bad = copy.deepcopy(original)
                bad["phases"][0]["phase_percent"] = invalid
                manifest.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    module.convert_manifest(manifest, output, "case01", "seq01")
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
