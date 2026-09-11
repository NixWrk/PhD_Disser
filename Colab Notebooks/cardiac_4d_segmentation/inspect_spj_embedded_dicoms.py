#!/usr/bin/env python
"""Read-only inventory of DICOM objects embedded in an Inobitec SPJ container.

The proprietary IPFFS container is not reimplemented.  The script only finds
standard 128-byte-preamble + ``DICM`` objects, reads headers in memory, hashes
UIDs, and reports whether any embedded CT series forms a complete 4D partition.
It does not extract or rewrite medical images.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mmap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pydicom

from cardiac4d import PHASE_KEYWORDS, phase_candidate, sha256_text, slice_coordinate


HEADER_TAGS = [
    "Modality",
    "SOPClassUID",
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "NominalPercentageOfCardiacPhase",
    "TemporalPositionIdentifier",
    "TriggerTime",
    "AcquisitionNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dicm_offsets(path: Path):
    offsets = []
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = 0
            while True:
                found = mapped.find(b"DICM", position)
                if found < 0:
                    break
                if found >= 128:
                    offsets.append(found)
                position = found + 4
    return offsets


def inspect(path: Path):
    offsets = dicm_offsets(path)
    starts = [offset - 128 for offset in offsets]
    groups = defaultdict(list)
    rejected = []
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for index, start in enumerate(starts):
                end = starts[index + 1] if index + 1 < len(starts) else len(mapped)
                try:
                    dataset = pydicom.dcmread(
                        io.BytesIO(mapped[start:end]),
                        stop_before_pixels=True,
                        specific_tags=HEADER_TAGS,
                    )
                except Exception as error:
                    rejected.append({"index": index, "error": type(error).__name__})
                    continue
                uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
                if not uid:
                    rejected.append({"index": index, "error": "missing_series_uid"})
                    continue
                groups[uid].append((Path(f"embedded_{index:05d}"), dataset))

    series = []
    for uid, records in groups.items():
        first = records[0][1]
        coordinates = [slice_coordinate(dataset) for _, dataset in records]
        phase_candidates = [phase_candidate(records, key) for key in PHASE_KEYWORDS]
        series.append(
            {
                "modality": str(getattr(first, "Modality", "")),
                "series_number": str(getattr(first, "SeriesNumber", "")),
                "series_description": str(
                    getattr(first, "SeriesDescription", "")
                ).strip(),
                "series_uid_sha256": sha256_text(uid),
                "study_uid_sha256": sha256_text(
                    str(getattr(first, "StudyInstanceUID", ""))
                ),
                "instances": len(records),
                "unique_slice_positions_0_001mm": len(
                    {round(value, 3) for value in coordinates if value is not None}
                ),
                "matrix": [
                    int(getattr(first, "Rows", 0)),
                    int(getattr(first, "Columns", 0)),
                ],
                "pixel_spacing_mm": [
                    float(value) for value in getattr(first, "PixelSpacing", [])
                ],
                "phase_candidates": phase_candidates,
            }
        )
    series.sort(key=lambda item: (item["modality"], item["series_number"]))
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "container_format_observed": "IPFFS with embedded standard DICOM signatures",
        "source_file_name": path.name,
        "source_size_bytes": path.stat().st_size,
        "source_sha256": sha256_file(path),
        "dicm_signatures": len(offsets),
        "parsed_objects": sum(len(records) for records in groups.values()),
        "rejected_candidates": rejected,
        "series": series,
        "status": "read_only_embedded_dicom_inventory",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spj", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = inspect(Path(args.spj).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Parsed {report['parsed_objects']}/{report['dicm_signatures']} embedded DICOM objects"
    )
    for item in report["series"]:
        usable = [
            candidate["keyword"]
            for candidate in item["phase_candidates"]
            if candidate["usable_complete_partition"]
        ]
        print(
            item["modality"],
            item["series_number"],
            item["series_description"],
            item["instances"],
            "4D keys:",
            usable,
        )


if __name__ == "__main__":
    main()

