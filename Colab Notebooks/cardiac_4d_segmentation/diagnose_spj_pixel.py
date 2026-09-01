#!/usr/bin/env python
"""Temporary read-only diagnostic for one embedded SPJ DICOM pixel object."""

from __future__ import annotations

import argparse
import io
import mmap
from pathlib import Path

import pydicom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spj", required=True)
    parser.add_argument("--series-number", required=True)
    args = parser.parse_args()
    source = Path(args.spj)
    with source.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            offsets = []
            position = 0
            while True:
                found = mapped.find(b"DICM", position)
                if found < 0:
                    break
                if found >= 128:
                    offsets.append(found - 128)
                position = found + 4
            for index, start in enumerate(offsets):
                end = offsets[index + 1] if index + 1 < len(offsets) else len(mapped)
                raw = mapped[start:end]
                try:
                    header = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
                except Exception:
                    continue
                if str(getattr(header, "SeriesNumber", "")) != args.series_number:
                    continue
                print("chunk", index, "bytes", len(raw))
                print("header tags", len(header), "orientation", getattr(header, "ImageOrientationPatient", None))
                full = pydicom.dcmread(io.BytesIO(raw))
                print("full tags", len(full), "orientation", getattr(full, "ImageOrientationPatient", None))
                print("transfer syntax", getattr(full.file_meta, "TransferSyntaxUID", None))
                print("pixel present", "PixelData" in full, "pixel bytes", len(getattr(full, "PixelData", b"")))
                print("rows cols", getattr(full, "Rows", None), getattr(full, "Columns", None))
                try:
                    pixels = full.pixel_array
                    print("pixel array", pixels.shape, pixels.dtype, pixels.min(), pixels.max())
                except Exception as error:
                    print("pixel error", type(error).__name__, str(error))
                return
    raise RuntimeError("No selected embedded object")


if __name__ == "__main__":
    main()

