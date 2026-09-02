#!/usr/bin/env python
"""Extract only non-identifying cardiac timing fields from local ECG Report OCR."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    return alias.strip().lower(), Path(raw_path).resolve()


def heart_rate_blocks(text: str) -> list[dict]:
    pattern = re.compile(
        r"(?P<label>Scan heart rate statistics|Breath-hold recording heart rate statistics):\s*"
        r"Min:\s*(?P<min>\d+)\s+Avg:\s*(?P<avg>\d+)\s+Max:\s*(?P<max>\d+)\s+Variability:\s*(?P<variability>\d+)",
        re.IGNORECASE,
    )
    result = []
    for match in pattern.finditer(text):
        result.append(
            {
                "label": match.group("label"),
                "minimum_bpm": int(match.group("min")),
                "average_bpm": int(match.group("avg")),
                "maximum_bpm": int(match.group("max")),
                "variability_bpm": int(match.group("variability")),
            }
        )
    return result


def required(pattern: str, text: str, field: str):
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse {field}")
    return match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", action="append", required=True, type=parse_mapping)
    parser.add_argument("--series-catalog", required=True)
    parser.add_argument("--trace-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog = json.loads(Path(args.series_catalog).read_text(encoding="utf-8"))
    trace_by_subject = {}
    if args.trace_manifest:
        trace_manifest = json.loads(Path(args.trace_manifest).read_text(encoding="utf-8"))
        trace_by_subject = {item["subject"]: item for item in trace_manifest.get("items", [])}

    subjects = []
    for alias, path in args.ocr:
        text = path.read_text(encoding="utf-8", errors="replace")
        phases = required(r"StartPhase:\s*(\d+)%\s*to\s*EndPhase:\s*(\d+)%", text, "phase range")
        series = required(r"Series Number:\s*(\d+)", text, "series number")
        scan_type = required(r"Scan Type:\s*([^\r\n]+)", text, "scan type")
        blocks = heart_rate_blocks(text)
        if not blocks:
            raise ValueError(f"Could not parse heart-rate statistics for {alias}")
        subjects.append(
            {
                "subject": alias,
                "source": "GE ECG Report Secondary Capture Image Storage",
                "series_number_referenced_by_report": int(series.group(1)),
                "start_phase_percent": int(phases.group(1)),
                "end_phase_percent": int(phases.group(2)),
                "scan_type": scan_type.group(1).strip(),
                "heart_rate_statistics": blocks,
                "report_warning": "Not for Diagnostic ECG use",
                "digital_waveform_sequence_available": False,
                "deidentified_trace": trace_by_subject.get(alias),
                "interpretation": "Actual scanner ECG/gating report raster; not calibrated voltage samples and not a diagnostic ECG.",
            }
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "scanner_ecg_report_evidence_available_for_adam_and_nix",
        "subjects": subjects,
        "series_catalog_result": {
            "series_count": len(catalog.get("series", [])),
            "waveform_series_count": len(catalog.get("waveform_series", [])),
            "ecg_named_series": [
                {key: item[key] for key in ("subject", "series_number", "series_description", "sop_class_name", "file_count")}
                for item in catalog.get("ecg_named_series", [])
            ],
        },
        "limitations": [
            "No standard DICOM WaveformSequence was found in the exported series catalog.",
            "The ECG curve is available only as a scanner report raster for Adam and Nix.",
            "No ECG Report series was exported or embedded for Georg.",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({item["subject"]: [item["start_phase_percent"], item["end_phase_percent"]] for item in subjects}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
