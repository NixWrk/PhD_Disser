#!/usr/bin/env python
"""Render ECG report Secondary Capture DICOM objects to local PNG for review."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    return alias.strip().lower(), Path(raw_path).resolve()


def normalized_pixels(dataset: pydicom.Dataset) -> np.ndarray:
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        values = pixels.astype(np.float32)
        low, high = np.percentile(values, [0.2, 99.8])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        values = np.clip((values - low) / max(high - low, 1.0), 0.0, 1.0)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            values = 1.0 - values
        return np.rint(values * 255).astype(np.uint8)
    if pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        if pixels.dtype == np.uint8:
            return pixels
        maximum = float(np.max(pixels))
        return np.rint(np.clip(pixels / max(maximum, 1.0), 0, 1) * 255).astype(np.uint8)
    raise ValueError(f"Unsupported Secondary Capture pixel shape: {pixels.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", required=True, type=parse_mapping)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "local_medical_secondary_capture_pending_burned_in_text_review",
        "subjects": {},
        "warning": "PNG files may contain burned-in identifiers and must not be committed or published before visual redaction.",
    }
    for alias, folder in args.series:
        alias_dir = output_root / alias
        alias_dir.mkdir(parents=True, exist_ok=True)
        items = []
        datasets = []
        for path in folder.glob("*.dcm"):
            dataset = pydicom.dcmread(str(path))
            instance = int(getattr(dataset, "InstanceNumber", len(datasets) + 1) or len(datasets) + 1)
            datasets.append((instance, path, dataset))
        for index, (instance, path, dataset) in enumerate(sorted(datasets), start=1):
            pixels = normalized_pixels(dataset)
            output_name = f"ecg_report_{index:02d}_instance_{instance}.png"
            output_path = alias_dir / output_name
            plt.imsave(output_path, pixels, cmap="gray" if pixels.ndim == 2 else None, vmin=0, vmax=255)
            items.append(
                {
                    "instance_number": instance,
                    "output": output_name,
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": int(getattr(dataset, "Rows", pixels.shape[0])),
                    "columns": int(getattr(dataset, "Columns", pixels.shape[1])),
                    "photometric_interpretation": str(getattr(dataset, "PhotometricInterpretation", "")),
                    "burned_in_annotation": str(getattr(dataset, "BurnedInAnnotation", "")),
                    "acquisition_time": str(getattr(dataset, "AcquisitionTime", "")),
                    "content_time": str(getattr(dataset, "ContentTime", "")),
                }
            )
        report["subjects"][alias] = {"series_description": "ECG Report", "images": items}
    (output_root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({alias: len(value["images"]) for alias, value in report["subjects"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
