import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_heart_volume_mesh_block import shape_errors


def reports(values):
    return {g: {"derivatives": {level: [dict(step=h,Sminus=v[0],derivative=v[1],Splus=v[2])
        for h in (.005,.01)] for level in ("coarse","fine")}}
        for g,v in values.items()}


class ShapeErrorTests(unittest.TestCase):
    def test_central_preference_can_hide_opposite_directional_preference(self):
        rows=shape_errors(reports({"individual":(-6.,-10.,-14.),"sphere":(-8.,-8.,-8.),"ellipsoid":(-9.,-9.,-9.)}))
        selected={r["metric"]:r for r in rows if r["mesh"]=="fine" and r["step"]==.005}
        self.assertEqual(selected["derivative"]["preferred"],"ellipsoid")
        self.assertEqual(selected["Sminus"]["preferred"],"sphere")
        self.assertEqual(selected["Splus"]["preferred"],"ellipsoid")
        self.assertAlmostEqual(selected["Sminus"]["sphere_absolute_relative_error"],1/3)

    def test_zero_reference_does_not_create_a_finite_relative_error(self):
        rows=shape_errors(reports({"individual":(-1.,0.,1.),"sphere":(-2.,0.,2.),"ellipsoid":(-3.,0.,3.)}))
        central=[r for r in rows if r["metric"]=="derivative"]
        self.assertTrue(all(r["preferred"]=="undefined_zero_reference" for r in central))
        self.assertTrue(all(r["sphere_absolute_relative_error"] is None for r in central))

    def test_incomplete_step_set_is_rejected(self):
        data=reports({"individual":(-1.,-1.,-1.),"sphere":(-2.,-2.,-2.),"ellipsoid":(-3.,-3.,-3.)})
        data["ellipsoid"]["derivatives"]["fine"].pop()
        with self.assertRaises(ValueError):shape_errors(data)


if __name__=="__main__":unittest.main()
