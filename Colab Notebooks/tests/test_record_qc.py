import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from record_qc import _longest_true_duration, _validate_candidate_manifest


class RecordQcTests(unittest.TestCase):
    def test_longest_true_duration_uses_contiguous_segment(self):
        time_s = np.arange(0.0, 10.0, 1.0)
        mask = np.array([False, True, True, False, True, True, True, True, False, False])
        self.assertEqual(_longest_true_duration(time_s, mask), 3.0)

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
