import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from record_qc import (
    EXP02_COLUMNS,
    _longest_true_duration,
    _validate_candidate_manifest,
    build_exp02_candidate_manifest,
)


class RecordQcTests(unittest.TestCase):
    def test_longest_true_duration_uses_contiguous_segment(self):
        time_s = np.arange(0.0, 10.0, 1.0)
        mask = np.array([False, True, True, False, True, True, True, True, False, False])
        self.assertEqual(_longest_true_duration(time_s, mask), 3.0)


    def test_exp02_classified_protocol_records_are_not_unclassified_or_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            derived_root = root / "derived"

            def write_record(path: Path, offset: float) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                time_s = np.arange(0.0, 4.01, 0.005)
                values = np.column_stack([
                    time_s,
                    np.sin(time_s) + offset,
                    np.full_like(time_s, 20.0 + offset),
                    np.sin(2.0 * time_s),
                    np.cos(time_s) + offset,
                    np.full_like(time_s, 30.0 + offset),
                    np.full_like(time_s, 4700.0),
                    np.full_like(time_s, 4700.0),
                ])
                np.savetxt(
                    path,
                    values,
                    delimiter=",",
                    header=",".join(EXP02_COLUMNS),
                    comments="",
                    encoding="utf-8",
                )

            subjects = []
            for index, subject_id in enumerate(("exp02_nik", "exp02_georg"), 1):
                directory = data_root / subject_id
                suffix = f"s{index}"
                write_record(directory / f"50{suffix}.csv", float(index))
                write_record(directory / "protocol.csv", float(index + 10))
                subjects.append({
                    "subject_id": subject_id,
                    "data_subdir": subject_id,
                    "filename_suffix": suffix,
                    "sizes_mm": [50],
                    "excluded_recordings": [],
                    "classified_additional_recordings": [{
                        "record_id": f"{subject_id}_protocol_breathing",
                        "relative_path": "protocol.csv",
                        "role": "full_breathing_protocol_side_size_unknown",
                        "expected_active_channels": [1, 2],
                        "configuration_id": "protocol_size_unknown",
                        "montage_id": "ttrkg_geometry_pending",
                        "side_montage_id": "side_size_unknown",
                        "side_size_mm": None,
                    }],
                })

            config = {
                "data_root": str(data_root),
                "derived_root": str(derived_root),
                "expected_independent_record_count": 2,
                "record_qc": {
                    "active_channel_threshold_ohm": 5.0,
                    "minimum_active_duration_s": 4.0,
                    "ttrkg_montage_id": "ttrkg_geometry_pending",
                    "side_montage_id_template": "side_{size_mm}",
                },
                "subjects": subjects,
            }
            config_path = root / "exp02.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            _, manifest = build_exp02_candidate_manifest(config_path)
            protocol_records = [
                item
                for item in manifest["records"]
                if item["role"] == "full_breathing_protocol_side_size_unknown"
            ]
            self.assertEqual(len(protocol_records), 2)
            self.assertTrue(all(item["include"] is False for item in protocol_records))
            self.assertTrue(all(item["side_size_mm"] is None for item in protocol_records))
            self.assertEqual(
                sum(item["unclassified_unique_records"] for item in manifest["inventory"].values()),
                0,
            )
            self.assertNotIn(
                "classify_unlisted_unique_exports",
                manifest["manual_review_requirements"],
            )

    def test_candidate_rejects_duplicate_included_content(self):
        record = {
            "record_id": "a",
            "input_sha256": "1" * 64,
            "include": True,
            "expected_active_channels": [1, 2],
            "actual_active_channels": [1, 2],
        }
        manifest = {
            "status": "pending_manual_review",
            "records": [record, {**record, "record_id": "b"}],
        }
        with self.assertRaises(ValueError):
            _validate_candidate_manifest(manifest, 2)

    def test_candidate_requires_expected_channel_state(self):
        manifest = {
            "status": "pending_manual_review",
            "records": [{
                "record_id": "a",
                "input_sha256": "1" * 64,
                "include": True,
                "expected_active_channels": [1],
                "actual_active_channels": [1, 2],
            }],
        }
        with self.assertRaises(ValueError):
            _validate_candidate_manifest(manifest, 1)


if __name__ == "__main__":
    unittest.main()
