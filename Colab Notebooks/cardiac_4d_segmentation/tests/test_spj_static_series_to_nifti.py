from __future__ import annotations

import importlib.util
import struct
import unittest
from pathlib import Path

import numpy as np
import pydicom


MODULE_PATH = Path(__file__).parents[1] / "spj_static_series_to_nifti.py"
SPEC = importlib.util.spec_from_file_location("spj_static_series_to_nifti", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def header(transfer_syntax: str = module.EXPLICIT_VR_LITTLE_ENDIAN) -> pydicom.Dataset:
    dataset = pydicom.Dataset()
    dataset.file_meta = pydicom.Dataset()
    dataset.file_meta.TransferSyntaxUID = transfer_syntax
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.PixelRepresentation = 1
    return dataset


class NativePixelDataTests(unittest.TestCase):
    def test_reads_explicit_vr_pixels_and_ignores_container_trailer(self):
        expected = np.array([[-1000, -250, 0], [125, 800, 3071]], dtype="<i2")
        value = expected.tobytes()
        pixel_element = (
            module.PIXEL_DATA_TAG_LE
            + b"OW"
            + b"\x00\x00"
            + struct.pack("<I", len(value))
            + value
        )
        raw = b"container-prefix" + pixel_element + b"proprietary-trailer"
        actual = module.uncompressed_pixel_array(raw, header())
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_compressed_transfer_syntax(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported embedded Pixel Data"):
            module.uncompressed_pixel_array(b"", header("1.2.840.10008.1.2.4.90"))


if __name__ == "__main__":
    unittest.main()
