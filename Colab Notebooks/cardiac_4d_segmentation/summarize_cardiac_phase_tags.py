#!/usr/bin/env python
"""Quickly summarize standard and GE cardiac phase tags in accepted CT series."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from pydicom.datadict import DicomDictionary, keyword_for_tag
from pydicom.tag import Tag


PATTERN = re.compile(
    r"cardiac|trigger|heart|ecg|electrocard|waveform|r[-_ ]?r|beat|cine|phase|temporal|gating|gated",
    re.IGNORECASE,
)
IDENTITY_PATTERN = re.compile(r"patient|name|birth|address|operator|physician|institution", re.IGNORECASE)
GE_TAGS = {
    Tag(0x0019, 0x102C): "NumberOfTriggers",
    Tag(0x0019, 0x102F): "TriggerFrequency",
    Tag(0x0043, 0x1040): "TriggerOnPosition",
    Tag(0x0043, 0x1042): "DASTriggerSource",
    Tag(0x0045, 0x1030): "CardiacReconAlgorithm",
    Tag(0x0045, 0x1031): "AvgHeartRateForImage",
    Tag(0x0045, 0x1032): "GETemporalResolutionSeconds",
    Tag(0x0045, 0x1050): "TemporalCenterViewAngle",
    Tag(0x0049, 0x1001): "CTCardiacSequence",
    Tag(0x0049, 0x1002): "HeartRateAtConfirm",
    Tag(0x0049, 0x1003): "AvgHeartRatePriorToConfirm",
    Tag(0x0049, 0x1004): "MinHeartRatePriorToConfirm",
    Tag(0x0049, 0x1005): "MaxHeartRatePriorToConfirm",
    Tag(0x0049, 0x1006): "StdDevHeartRatePriorToConfirm",
    Tag(0x0049, 0x1007): "NumHeartRateSamplesPriorToConfirm",
    Tag(0x0049, 0x1008): "AutoHeartRateDetectPredict",
    Tag(0x0049, 0x1009): "SystemOptimizedHeartRate",
    Tag(0x0049, 0x100A): "EkgMonitorType",
    Tag(0x0049, 0x100B): "NumReconSectors",
    Tag(0x0049, 0x100C): "RpeakTimeStamps",
    Tag(0x0049, 0x1016): "EkgGatingType",
}
GE_CARDIAC_SEQUENCE_TAG = Tag(0x0049, 0x1001)
GE_CARDIAC_SEQUENCE_MEMBER_TAGS = {
    tag: label for tag, label in GE_TAGS.items()
    if tag.group == 0x0049 and 0x1002 <= tag.element <= 0x100B
}
TIME_KEYWORDS = {"AcquisitionTime", "ContentTime", "SeriesTime"}


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ALIAS=PATH")
    alias, raw_path = value.split("=", 1)
    return alias.strip().lower(), Path(raw_path).resolve()


def standard_tags() -> set[Tag]:
    result = set()
    for raw_tag, details in DicomDictionary.items():
        name = details[2]
        keyword = details[4]
        if IDENTITY_PATTERN.search(name + " " + keyword):
            continue
        if PATTERN.search(name + " " + keyword) or keyword in TIME_KEYWORDS:
            result.add(Tag(raw_tag))
    for keyword in ("SeriesNumber", "SeriesDescription", "ProtocolName", "Modality", "SOPClassUID", "SeriesInstanceUID", "InstanceNumber", "RevolutionTime"):
        tag = pydicom.datadict.tag_for_keyword(keyword)
        if tag is not None:
            result.add(Tag(tag))
    return result


def safe_value(value) -> str:
    if isinstance(value, bytes):
        return f"bytes={len(value)}"
    text = str(value)
    return text if len(text) <= 200 else text[:197] + "..."


def numeric_values(counter: Counter) -> list[float]:
    result = []
    for value, count in counter.items():
        try:
            result.extend([float(value)] * count)
        except ValueError:
            pass
    return result


def numeric_vectors(counter: Counter) -> list[dict]:
    result = []
    for value, count in counter.items():
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, (list, tuple)):
                parsed = [parsed]
            values = [float(item) for item in parsed]
        except (ValueError, SyntaxError, TypeError):
            continue
        result.append({"values": values, "image_count": count})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", required=True, type=parse_mapping)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    requested_tags = standard_tags() | set(GE_TAGS) | {
        Tag(group, 0x0010)
        for group in {tag.group for tag in GE_TAGS}
    }
    records = []
    for alias, folder in args.series:
        metadata = defaultdict(Counter)
        by_phase = defaultdict(lambda: defaultdict(Counter))
        read = 0
        for path in sorted(folder.glob("*.dcm")):
            dataset = pydicom.dcmread(
                str(path), stop_before_pixels=True, force=True, specific_tags=sorted(requested_tags)
            )
            read += 1
            phase = safe_value(getattr(dataset, "NominalPercentageOfCardiacPhase", "missing"))
            for tag in requested_tags:
                if tag == GE_CARDIAC_SEQUENCE_TAG:
                    continue
                element = dataset.get(tag)
                if element is None:
                    continue
                label = GE_TAGS.get(tag) or element.keyword or keyword_for_tag(tag) or str(tag)
                value = safe_value(element.value)
                metadata[label][value] += 1
                by_phase[phase][label][value] += 1
            sequence_element = dataset.get(GE_CARDIAC_SEQUENCE_TAG)
            if sequence_element is not None:
                for item in sequence_element.value:
                    for tag, label in GE_CARDIAC_SEQUENCE_MEMBER_TAGS.items():
                        element = item.get(tag)
                        if element is None:
                            continue
                        value = safe_value(element.value)
                        metadata[label][value] += 1
                        by_phase[phase][label][value] += 1

        phase_rows = []
        for phase, fields in by_phase.items():
            try:
                phase_number = float(phase)
            except ValueError:
                phase_number = None
            row = {"nominal_percentage": phase_number, "slice_count": sum(fields.get("NominalPercentageOfCardiacPhase", {}).values())}
            for label in ("AvgHeartRateForImage", "GETemporalResolutionSeconds", "TemporalCenterViewAngle"):
                values = numeric_values(fields.get(label, Counter()))
                if values:
                    row[label] = {
                        "minimum": float(np.min(values)),
                        "median": float(np.median(values)),
                        "maximum": float(np.max(values)),
                    }
            phase_rows.append(row)
        phase_rows.sort(key=lambda item: (item["nominal_percentage"] is None, item["nominal_percentage"] or 0))

        compact = {
            label: [{"value": value, "count": count} for value, count in counter.items()]
            for label, counter in sorted(metadata.items())
        }
        heart_rates = numeric_values(metadata.get("AvgHeartRateForImage", Counter()))
        nominal = [row["nominal_percentage"] for row in phase_rows if row["nominal_percentage"] is not None]

        rpeak_vectors = numeric_vectors(metadata.get("RpeakTimeStamps", Counter()))
        selected_rpeaks = rpeak_vectors[0]["values"] if len(rpeak_vectors) == 1 else []
        rpeak_rr_ms = (
            (1000.0 * np.diff(np.asarray(selected_rpeaks, dtype=float))).tolist()
            if len(selected_rpeaks) >= 2 else []
        )
        rpeak_heart_rates = [60000.0 / value for value in rpeak_rr_ms if value > 0]

        cycle_parameters = []
        cycle_starts_ms = {}
        elapsed_ms = 0.0
        cycle_indices = sorted({int(value // 100.0) for value in nominal})
        for cycle_index in cycle_indices:
            cycle_hr = [
                row["AvgHeartRateForImage"]["median"]
                for row in phase_rows
                if row["nominal_percentage"] is not None
                and int(row["nominal_percentage"] // 100.0) == cycle_index
                and "AvgHeartRateForImage" in row
            ]
            if not cycle_hr:
                continue
            heart_rate = float(np.median(cycle_hr))
            if cycle_index < len(rpeak_rr_ms) and rpeak_rr_ms[cycle_index] > 0:
                rr_ms = float(rpeak_rr_ms[cycle_index])
                elapsed_ms = 1000.0 * (selected_rpeaks[cycle_index] - selected_rpeaks[0])
                timing_source = "GE RpeakTimeStamps (0049,100C)"
            else:
                rr_ms = 60000.0 / heart_rate
                timing_source = "60000 / GE AvgHeartRateForImage (0045,1031)"
            cycle_starts_ms[cycle_index] = elapsed_ms
            parameters = {
                "cycle_index": cycle_index,
                "median_image_heart_rate_bpm": heart_rate,
                "rr_interval_ms": rr_ms,
                "cycle_start_from_R0_ms": elapsed_ms,
                "timing_source": timing_source,
            }
            if cycle_index < len(rpeak_heart_rates):
                parameters["heart_rate_from_rpeak_interval_bpm"] = rpeak_heart_rates[cycle_index]
                parameters["image_vs_rpeak_heart_rate_difference_bpm"] = heart_rate - rpeak_heart_rates[cycle_index]
                parameters["rpeak_start_s"] = selected_rpeaks[cycle_index]
                parameters["rpeak_end_s"] = selected_rpeaks[cycle_index + 1]
            cycle_parameters.append(parameters)
            elapsed_ms += rr_ms
        for row in phase_rows:
            value = row["nominal_percentage"]
            if value is None:
                continue
            cycle_index = int(value // 100.0)
            cycle = next((item for item in cycle_parameters if item["cycle_index"] == cycle_index), None)
            if cycle is None:
                continue
            within_cycle = value % 100.0
            row["cycle_index"] = cycle_index
            row["phase_percent_within_cycle"] = within_cycle
            row["derived_time_from_R0_ms"] = (
                cycle_starts_ms[cycle_index]
                + within_cycle / 100.0 * cycle["rr_interval_ms"]
            )

        summary = {
            "subject": alias,
            "source_folder_name": folder.name,
            "dicom_files_read": read,
            "series_uid_sha256": hashlib.sha256(str(next(iter(metadata.get("SeriesInstanceUID", {})), "")).encode()).hexdigest(),
            "phase_rows": phase_rows,
            "all_present_cardiac_metadata": compact,
        }
        if rpeak_vectors:
            summary["manufacturer_rpeak_timing"] = {
                "dicom_tag": "GE RpeakTimeStamps (0049,100C)",
                "unique_timestamp_sets": rpeak_vectors,
                "status": "manufacturer_trigger_timestamps_not_ecg_waveform_samples",
            }
            if selected_rpeaks:
                summary["manufacturer_rpeak_timing"].update(
                    {
                        "selected_rpeak_timestamps_s": selected_rpeaks,
                        "rr_intervals_ms": rpeak_rr_ms,
                        "heart_rates_from_rpeak_intervals_bpm": rpeak_heart_rates,
                    }
                )
        if cycle_parameters:
            used_rpeaks = any(
                item["timing_source"].startswith("GE RpeakTimeStamps")
                for item in cycle_parameters
            )
            summary["derived_multicycle_time_mapping"] = {
                "cycle_parameters": cycle_parameters,
                "definition": (
                    "Cumulative cycle time from NominalPercentageOfCardiacPhase and the GE RpeakTimeStamps interval for each cycle."
                    if used_rpeaks else
                    "Cumulative cycle time from NominalPercentageOfCardiacPhase and median GE AvgHeartRateForImage within each cycle index."
                ),
                "status": (
                    "derived_manufacturer_time_coordinate_verified_by_trigger_timestamps_not_waveform_samples"
                    if used_rpeaks else
                    "derived_manufacturer_time_coordinate_not_recorded_waveform_samples"
                ),
            }
            revolution_times = numeric_values(metadata.get("RevolutionTime", Counter()))
            aligned_rows = [
                row for row in phase_rows
                if "derived_time_from_R0_ms" in row and "TemporalCenterViewAngle" in row
            ]
            if revolution_times and aligned_rows:
                revolution_ms = 1000.0 * float(np.median(revolution_times))
                offsets = np.asarray([
                    row["TemporalCenterViewAngle"]["median"] / 360.0 * revolution_ms
                    - row["derived_time_from_R0_ms"]
                    for row in aligned_rows
                ])
                median_offset = float(np.median(offsets))
                residuals = offsets - median_offset
                summary["derived_multicycle_time_mapping"]["center_view_angle_cross_check"] = {
                    "revolution_time_ms": revolution_ms,
                    "median_constant_offset_ms": median_offset,
                    "maximum_absolute_residual_ms": float(np.max(np.abs(residuals))),
                    "interpretation": "A near-constant offset supports the phase/heart-rate time mapping but is not an independent ECG measurement.",
                }
        if heart_rates:
            median_hr = float(np.median(heart_rates))
            summary["derived_from_AvgHeartRateForImage"] = {
                "median_heart_rate_bpm": median_hr,
                "rr_interval_ms_from_60000_over_hr": 60000.0 / median_hr,
            }
        if len(nominal) > 1:
            summary["nominal_phase_summary"] = {
                "minimum_percent": min(nominal),
                "maximum_percent": max(nominal),
                "median_step_percentage_points": float(np.median(np.diff(nominal))),
                "modulo_100_percent": [value % 100.0 for value in nominal],
            }
        records.append(summary)

    report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "manufacturer_trigger_timestamps_support_phase_link_pending_waveform_validation",
        "subjects": records,
        "interpretation": {
            "NominalPercentageOfCardiacPhase": "Manufacturer reconstruction coordinate; values above 100 are retained and not silently wrapped.",
            "rr_interval_ms_from_60000_over_hr": "Derived estimate from image-level average heart rate, not a recorded beat-to-beat R-R series.",
            "RpeakTimeStamps": "Manufacturer trigger timestamps used to reconstruct R-R intervals; these are not stored ECG waveform samples and do not prove the physiological cause of an abnormal interval.",
            "GETemporalResolutionSeconds": "Temporal width/resolution of a reconstructed image, not phase sampling interval.",
            "derived_time_from_R0_ms": "Piecewise time coordinate using manufacturer R-peak timestamps when available, otherwise the image-level heart rate; it remains a reconstruction coordinate.",
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({item["subject"]: len(item["phase_rows"]) for item in records}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
