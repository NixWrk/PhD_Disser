#!/usr/bin/env python
"""Create a de-identified one-header-per-folder DICOM series catalog."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pydicom


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    return alias.strip().lower(), Path(raw_path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=parse_mapping)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for alias, root in args.root:
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            files = sorted(folder.glob("*.dcm"))
            if not files:
                continue
            dataset = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True, defer_size="1 KB")
            sop = getattr(dataset, "SOPClassUID", None)
            records.append(
                {
                    "subject": alias,
                    "source_folder_name": folder.name,
                    "file_count": len(files),
                    "series_number": str(getattr(dataset, "SeriesNumber", "")),
                    "series_description": str(getattr(dataset, "SeriesDescription", "")),
                    "protocol_name": str(getattr(dataset, "ProtocolName", "")),
                    "modality": str(getattr(dataset, "Modality", "")),
                    "sop_class_name": str(getattr(sop, "name", "")),
                    "has_waveform_sequence": bool(getattr(dataset, "WaveformSequence", None)),
                    "rows": int(getattr(dataset, "Rows", 0) or 0),
                    "columns": int(getattr(dataset, "Columns", 0) or 0),
                    "burned_in_annotation": str(getattr(dataset, "BurnedInAnnotation", "")),
                }
            )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "deidentified_one_header_per_series_folder",
        "series": records,
        "waveform_series": [item for item in records if item["has_waveform_sequence"] or "waveform" in item["sop_class_name"].lower()],
        "ecg_named_series": [item for item in records if "ecg" in (item["series_description"] + " " + item["protocol_name"]).lower()],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"series": len(records), "waveform": len(report["waveform_series"]), "ecg_named": len(report["ecg_named_series"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
