#!/usr/bin/env python
"""Inventory cardiac synchronization metadata without exporting DICOM pixels or PHI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pydicom
from pydicom.datadict import keyword_for_tag


CARDIAC_PATTERN = re.compile(
    r"cardiac|trigger|heart|ecg|electrocard|waveform|r[-_ ]?r|beat|cine|phase|"
    r"temporal|frame.*time|time.*frame|gating|gated",
    re.IGNORECASE,
)
EXPLICIT_TIME_KEYWORDS = {
    "AcquisitionDateTime",
    "AcquisitionTime",
    "ContentTime",
    "SeriesTime",
    "StudyTime",
    "InstanceCreationTime",
}
IDENTITY_PATTERN = re.compile(
    r"patient|physician|operator|institution|station|accession|birth|address|telephone|name",
    re.IGNORECASE,
)


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    return alias.strip().lower(), Path(raw_path).resolve()


def uid_hash(value: object) -> str | None:
    text = str(value or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def safe_value(element) -> str:
    if element.VR == "SQ":
        return f"sequence_items={len(element.value or [])}"
    value = element.value
    if isinstance(value, bytes):
        return f"bytes={len(value)}"
    if isinstance(value, (list, tuple)) and len(value) > 24:
        return f"items={len(value)}; first={str(value[:4])[:160]}"
    text = str(value)
    return text if len(text) <= 240 else text[:237] + "..."


def add(counter_map: dict[str, Counter], key: str, value: str) -> None:
    counter_map[key][value] += 1


def interesting_element(element) -> bool:
    keyword = element.keyword or keyword_for_tag(element.tag) or ""
    name = str(element.name or "")
    if IDENTITY_PATTERN.search(keyword) or IDENTITY_PATTERN.search(name):
        return False
    return keyword in EXPLICIT_TIME_KEYWORDS or bool(CARDIAC_PATTERN.search(keyword + " " + name))


def waveform_summary(dataset: pydicom.Dataset) -> list[dict]:
    result = []
    sequence = getattr(dataset, "WaveformSequence", None)
    if not sequence:
        return result
    for index, item in enumerate(sequence):
        channels = []
        for channel in getattr(item, "ChannelDefinitionSequence", []) or []:
            meanings = []
            for source in getattr(channel, "ChannelSourceSequence", []) or []:
                meaning = str(getattr(source, "CodeMeaning", "")).strip()
                if meaning:
                    meanings.append(meaning)
            channels.append(
                {
                    "label": str(getattr(channel, "ChannelLabel", "")),
                    "source_meanings": meanings,
                    "sensitivity": str(getattr(channel, "ChannelSensitivity", "")),
                    "sensitivity_units": [
                        str(getattr(unit, "CodeMeaning", ""))
                        for unit in getattr(channel, "ChannelSensitivityUnitsSequence", []) or []
                    ],
                }
            )
        data = getattr(item, "WaveformData", None)
        data_bytes = len(data) if isinstance(data, bytes) else None
        result.append(
            {
                "multiplex_group": index,
                "channels": int(getattr(item, "NumberOfWaveformChannels", 0) or 0),
                "samples": int(getattr(item, "NumberOfWaveformSamples", 0) or 0),
                "sampling_frequency_hz": float(getattr(item, "SamplingFrequency", 0) or 0),
                "bits_allocated": int(getattr(item, "WaveformBitsAllocated", 0) or 0),
                "waveform_data_bytes": data_bytes,
                "channel_definitions": channels,
            }
        )
    return result


def under(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def sorted_values(counter: Counter) -> list[dict]:
    def numeric_key(item):
        value = item[0]
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return [{"value": value, "count": count} for value, count in sorted(counter.items(), key=numeric_key)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", action="append", required=True, type=parse_mapping)
    parser.add_argument("--accepted-series", action="append", default=[], type=parse_mapping)
    parser.add_argument("--max-files", type=int, help="Optional diagnostic limit per subject root")
    parser.add_argument("--start-file", type=int, default=0, help="Zero-based offset before --max-files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    accepted = dict(args.accepted_series)
    series_records = {}
    unreadable = []
    subject_counts = {}
    for alias, root in args.subject:
        files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".dcm")
        files = files[args.start_file :]
        if args.max_files is not None:
            files = files[:args.max_files]
        subject_counts[alias] = {"dicom_files_seen": len(files), "dicom_headers_read": 0}
        for path in files:
            try:
                dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True, defer_size="1 KB")
            except Exception as exc:
                unreadable.append({"subject": alias, "relative_path": str(path.relative_to(root)), "error": type(exc).__name__})
                continue
            if not getattr(dataset, "SOPClassUID", None) and not getattr(dataset, "Modality", None):
                continue
            subject_counts[alias]["dicom_headers_read"] += 1
            series_uid = uid_hash(getattr(dataset, "SeriesInstanceUID", "")) or "missing-series-uid"
            key = (alias, series_uid)
            if key not in series_records:
                series_records[key] = {
                    "subject": alias,
                    "series_uid_sha256": series_uid,
                    "series_number": str(getattr(dataset, "SeriesNumber", "")),
                    "series_description": str(getattr(dataset, "SeriesDescription", "")),
                    "protocol_name": str(getattr(dataset, "ProtocolName", "")),
                    "modality": str(getattr(dataset, "Modality", "")),
                    "sop_class_name": str(getattr(getattr(dataset, "SOPClassUID", None), "name", "")),
                    "relative_directories": set(),
                    "file_count": 0,
                    "accepted_4d_series": False,
                    "metadata": defaultdict(Counter),
                    "private_cardiac_metadata": defaultdict(Counter),
                    "private_creators": set(),
                    "waveforms": [],
                }
            record = series_records[key]
            record["file_count"] += 1
            record["relative_directories"].add(str(path.parent.relative_to(root)))
            record["accepted_4d_series"] = record["accepted_4d_series"] or under(path, accepted.get(alias))
            for element in dataset.iterall():
                keyword = element.keyword or keyword_for_tag(element.tag) or ""
                name = str(element.name or "")
                if element.tag.is_private and element.tag.element < 0x0100 and element.VR in {"LO", "SH", "UT"}:
                    creator = safe_value(element)
                    if not IDENTITY_PATTERN.search(creator):
                        record["private_creators"].add(creator)
                if not interesting_element(element):
                    continue
                label = f"{element.tag} {keyword or name}"
                if element.tag.is_private:
                    add(record["private_cardiac_metadata"], label, safe_value(element))
                else:
                    add(record["metadata"], label, safe_value(element))
            waveforms = waveform_summary(dataset)
            if waveforms and waveforms not in record["waveforms"]:
                record["waveforms"].append(waveforms)

    output_series = []
    for record in series_records.values():
        metadata = {key: sorted_values(counter) for key, counter in sorted(record.pop("metadata").items())}
        private = {key: sorted_values(counter) for key, counter in sorted(record.pop("private_cardiac_metadata").items())}
        record["relative_directories"] = sorted(record["relative_directories"])
        record["private_creators"] = sorted(record["private_creators"])
        record["cardiac_metadata"] = metadata
        record["private_cardiac_metadata"] = private
        record["has_waveform"] = bool(record["waveforms"])
        text = " ".join(
            [record["series_description"], record["protocol_name"], record["modality"], record["sop_class_name"]]
        )
        record["potentially_relevant"] = bool(
            record["accepted_4d_series"]
            or metadata
            or private
            or record["has_waveform"]
            or CARDIAC_PATTERN.search(text)
        )
        output_series.append(record)

    def series_sort_key(record):
        try:
            number = int(record["series_number"])
        except ValueError:
            number = 10**9
        return record["subject"], number, record["series_description"]

    output_series.sort(key=series_sort_key)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "deidentified_dicom_header_inventory_not_physiological_validation",
        "subject_counts": subject_counts,
        "series_count": len(output_series),
        "potentially_relevant_series_count": sum(item["potentially_relevant"] for item in output_series),
        "waveform_series_count": sum(item["has_waveform"] for item in output_series),
        "series": output_series,
        "unreadable": unreadable,
        "limitations": [
            "Pixel Data and Waveform Data values are not exported.",
            "Patient and operator identity fields are intentionally excluded.",
            "Header metadata alone does not prove the physiological correctness of cardiac gating.",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "series_count", "potentially_relevant_series_count", "waveform_series_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
