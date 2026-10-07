"""Regression checks against reversed/shifted acquisition phase inputs."""

import unittest

from mp10_respiratory_phases import check_fit_phase, phase_contract, phase_stage, validate_scenario


class RespiratoryPhaseTests(unittest.TestCase):
    def test_published_phase_labels(self):
        for percent in range(0, 100, 10):
            with self.subTest(percent=percent):
                self.assertEqual(phase_stage(f"T{percent:02}"), percent / 100)
                self.assertEqual(phase_stage(f"g{percent:03}"), percent / 100)

    def test_inhale_cannot_be_assigned_to_shape_reference(self):
        with self.assertRaises(ValueError):
            phase_contract("g000", 0.7, [0.0, 0.5, 0.7])

    def test_unknown_phase_requires_explicit_scenario_mode(self):
        with self.assertRaises(ValueError):
            phase_contract("deep_inspiration", 0.7, [0.0, 0.5, 0.7])
        result = phase_contract("deep_inspiration", 0.7, [0.0, 0.5, 0.7], allow_unverified=True)
        self.assertEqual(result["status"], "unverified_phase_scenario_only")
        self.assertIsNone(result["cycle_zero"])

    def test_source_anchor_is_derived(self):
        self.assertEqual(phase_contract("T50", None, [0.0, 0.5, 0.7])["source_stage"], 0.5)

    def test_saved_report_and_fit_mismatches_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_scenario({"source_phase": "T70", "stage_anchor": 0.5,
                               "stages": [{"stage": 0.0}, {"stage": 0.5}, {"stage": 0.7}]}, "dirlab")
        with self.assertRaises(ValueError):
            check_fit_phase({"source_phase": "g070"}, "g000")

    def test_undocumented_bins_and_dataset_mismatch(self):
        for phase in ("T05", "g100", "cardiac_00", "T70extra"):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                phase_stage(phase)
        with self.assertRaises(ValueError):
            phase_stage("g070", "dirlab")


if __name__ == "__main__":
    unittest.main()
