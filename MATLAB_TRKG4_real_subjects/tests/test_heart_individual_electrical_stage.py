"""Contract checks: analytic-volume denominator and translation units."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import heart_individual_electrical_stage as stage


class StageContractTests(unittest.TestCase):
    def test_volume_design_and_translation_are_separate(self):
        volume = stage.state_design("volume", [.005, .01])
        self.assertEqual(len(volume), 5)
        self.assertTrue(all(s["translation_m"] == [0, 0, 0] for s in volume))
        translation = stage.state_design("translation", [1.0, 2.0])
        self.assertEqual(len(translation), 13)
        self.assertTrue(all(s["volume_fraction"] == 0 for s in translation))
        self.assertEqual(next(s["translation_m"] for s in translation
                              if s["parameter"] == "z_mm" and s["signed_step"] == 2.0),
                         [0, 0, .002])

    def test_derivative_keeps_analytic_denominator_despite_volume_transfer_error(self):
        rows = [
            dict(parameter="baseline", signed_step=0, analytic_volume_ml=100,
                 material_volume_ml=100, Z_ohm=5),
            dict(parameter="volume_fraction", signed_step=-.01,
                 analytic_volume_ml=99, material_volume_ml=99.5, Z_ohm=5.02),
            dict(parameter="volume_fraction", signed_step=.01,
                 analytic_volume_ml=101, material_volume_ml=100.5, Z_ohm=4.98)]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "plan.json").write_text("{}", encoding="utf-8")
            p = dict(states=rows, source_prepared_sha256="synthetic")
            with patch.object(stage.fractional, "checked_result", side_effect=lambda out, p, s: s):
                result = stage.summarize(out, p)
            self.assertEqual(result["status"], "complete")
            d = result["derivatives"][0]
            self.assertAlmostEqual(d["derivative"], -.02, places=12)
            self.assertAlmostEqual(d["material_volume_derivative"], .5)
            self.assertFalse(result["spatial_convergence"])
            self.assertFalse(result["physical_validation"])

    def test_incomplete_plus_minus_pair_is_not_reported_as_derivative(self):
        rows = [dict(parameter="baseline", signed_step=0, analytic_volume_ml=100,
                     material_volume_ml=100, Z_ohm=5),
                dict(parameter="x_mm", signed_step=1, analytic_volume_ml=100,
                     material_volume_ml=100, Z_ohm=5.1)]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            (out / "plan.json").write_text("{}", encoding="utf-8")
            p=dict(states=rows+[dict(missing=True)],source_prepared_sha256="synthetic")
            with patch.object(stage.fractional,"checked_result",
                              side_effect=lambda out,p,s: None if s.get("missing") else s):
                result=stage.summarize(out,p)
            self.assertEqual(result["status"],"partial")
            self.assertEqual(result["derivatives"],[])

if __name__ == "__main__":
    unittest.main()
