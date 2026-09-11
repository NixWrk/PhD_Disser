from __future__ import annotations

import importlib.util
from copy import deepcopy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "electrode_sensitivity_contract.py"
SPEC = importlib.util.spec_from_file_location("electrode_sensitivity_contract", CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


def _study(electrodes: list[str], drive: list[float], measure: list[float]) -> dict:
    return {
        "schema_version": "electrode_sensitivity_v1",
        "status": "exploratory_hypothesis_not_validated",
        "source": {
            "prepared_fem": "../frozen_outputs/legacy_prepared_fem_5_blocks.mat",
            "prepared_sha256": "a" * 64,
            "frequency_hz": 100_000.0,
        },
        "parameters": [
            {
                "id": "soft",
                "block_indices": [0],
                "nominal_sigma": 0.4,
                "fixed": False,
            },
            {
                "id": "bone",
                "block_indices": [1],
                "nominal_sigma": 0.02,
                "fixed": True,
            },
        ],
        "states": [
            {
                "id": "baseline",
                "conductivity": {"soft": 0.4, "bone": 0.02},
            },
            {
                "id": "inhale",
                "conductivity": {"soft": 0.45, "bone": 0.02},
            },
        ],
        "montages": [
            {
                "id": "montage",
                "contact_matrix": "../frozen_outputs/contacts/montage.mat",
                "contact_sha256": "b" * 64,
                "electrodes": electrodes,
                "contact_geometry_provenance": "CEM contact matrix prepared by the geometry pipeline",
                "channels": [
                    {"id": "standard", "drive": drive, "measure": measure},
                ],
            },
        ],
        "analysis": {"map_states": [], "save_fields": False},
    }


class ElectrodeSensitivityContractTests(unittest.TestCase):
    def test_accepts_three_four_and_six_electrode_patterns(self) -> None:
        cases = {
            3: ([0.5, 0.5, -1.0], [1.0, -0.5, -0.5]),
            4: ([0.5, 0.5, -0.5, -0.5], [1.0, 0.0, -1.0, 0.0]),
            6: (
                [0.25, 0.25, 0.25, 0.25, -0.5, -0.5],
                [0.5, 0.5, -0.25, -0.25, -0.25, -0.25],
            ),
        }
        for count, (drive, measure) in cases.items():
            with self.subTest(electrodes=count):
                study = _study([f"electrode_{i}" for i in range(count)], drive, measure)
                self.assertEqual(CONTRACT.validate_study(study), study)

    def test_role_names_and_drive_measure_overlap_are_not_special(self) -> None:
        study = _study(
            ["measure_plus", "arbitrary_surface_contact", "drive_minus"],
            [0.5, 0.5, -1.0],
            [1.0, -0.5, -0.5],
        )
        study["montages"][0]["channels"][0]["id"] = "V_reference"
        CONTRACT.validate_study(study)

    def test_validate_returns_an_independent_copy(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        validated = CONTRACT.validate_study(study)
        self.assertEqual(validated, study)
        self.assertIsNot(validated, study)
        self.assertIsNot(validated["montages"], study["montages"])
        validated["montages"][0]["electrodes"][0] = "changed"
        self.assertEqual(study["montages"][0]["electrodes"][0], "e0")

    def test_example_loads_and_zero_hashes_remain_structurally_valid(self) -> None:
        example_path = ROOT / "configs" / "electrode_sensitivity_study.example.json"
        study = CONTRACT.load_study(example_path)
        self.assertEqual(study["schema_version"], "electrode_sensitivity_v1")
        self.assertEqual(len(study["montages"][0]["electrodes"]), 6)
        self.assertGreaterEqual(len(study["montages"][0]["channels"]), 2)
        self.assertEqual(study["source"]["prepared_sha256"], "0" * 64)
        self.assertEqual(study["montages"][0]["contact_sha256"], "0" * 64)

    def test_rejects_bad_normalization(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.75, 0.75, -1.5], [1.0, -0.5, -0.5])
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "positive entries must sum to 1"):
            CONTRACT.validate_study(study)

    def test_rejects_nan(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [float("nan"), -0.5, -0.5])
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "finite number"):
            CONTRACT.validate_study(study)

    def test_rejects_state_coverage_mismatch(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        del study["states"][1]["conductivity"]["bone"]
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "cover exactly all parameter ids"):
            CONTRACT.validate_study(study)

    def test_rejects_fixed_parameter_change(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        study["states"][1]["conductivity"]["bone"] = 0.03
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "fixed parameter"):
            CONTRACT.validate_study(study)

    def test_rejects_duplicate_blocks_and_empty_block_lists(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        study["parameters"][1]["block_indices"] = [0]
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "overlaps"):
            CONTRACT.validate_study(study)

        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        study["parameters"][0]["block_indices"] = []
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "must not be empty"):
            CONTRACT.validate_study(study)

    def test_rejects_absolute_and_url_paths(self) -> None:
        for path in ("C:\\models\\volume.mat", "/models/volume.mat", "https://example.test/volume.mat"):
            with self.subTest(path=path):
                study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
                study["source"]["prepared_fem"] = path
                with self.assertRaisesRegex(CONTRACT.StudyValidationError, "relative path"):
                    CONTRACT.validate_study(study)

    def test_rejects_duplicate_electrodes_and_ids(self) -> None:
        study = _study(["e0", "e0", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "duplicate"):
            CONTRACT.validate_study(study)

        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        study["montages"][0]["channels"].append(
            {"id": "standard", "drive": [1.0, -1.0, 0.0], "measure": [1.0, -1.0, 0.0]}
        )
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "duplicate id"):
            CONTRACT.validate_study(study)

    def test_allows_repeated_channel_id_in_different_montages(self) -> None:
        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        second = deepcopy(study["montages"][0])
        second["id"] = "other_montage"
        second["contact_matrix"] = "../frozen_outputs/contacts/other.mat"
        study["montages"].append(second)
        CONTRACT.validate_study(study)

    def test_rejects_nonportable_ids(self) -> None:
        mutations = (
            ("parameters", "id", "soft/skin"),
            ("states", "id", "baseline.state"),
            ("montages", "id", "montage one"),
        )
        for collection, key, value in mutations:
            with self.subTest(field=f"{collection}.{key}"):
                study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
                study[collection][0][key] = value
                with self.assertRaisesRegex(CONTRACT.StudyValidationError, "portable ID"):
                    CONTRACT.validate_study(study)

        study = _study(["e0", "e1/e2", "e3"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "portable ID"):
            CONTRACT.validate_study(study)

        study = _study(["e0", "e1", "e2"], [0.5, 0.5, -1.0], [1.0, -0.5, -0.5])
        study["montages"][0]["channels"][0]["id"] = "channel.one"
        with self.assertRaisesRegex(CONTRACT.StudyValidationError, "portable ID"):
            CONTRACT.validate_study(study)


if __name__ == "__main__":
    unittest.main()
