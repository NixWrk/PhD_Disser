import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from prepare_surface_sensitivity import make_states, validate_landmarks, ROLES

class SurfacePreparationTests(unittest.TestCase):
    def test_cartesian_coverage_and_independent_checks(self):
        profile={"bounds":{"soft":[2.,20.],"heart":[1.,5.],"lung":[3.,9.]},"bone_fixed_rho_ohm_m":48.}
        axes,states,pilot=make_states(profile)
        self.assertEqual([sum(r==role for r,_,_ in states) for role in ["grid3","grid5_added","check5"]],[27,98,64])
        coords={tuple(rho.values()) for _,_,rho in states}
        self.assertEqual(len(coords),189)
        self.assertEqual(len(pilot),13)
        for role,s,rho in states:
            for k in axes:
                self.assertAlmostEqual(s["conductivity"][k]*rho[k],1.)
                self.assertTrue(axes[k][0] <= rho[k] <= axes[k][-1])
            self.assertEqual(s["conductivity"]["bone"],1/48.)
        base=pilot[0]["conductivity"]
        for s in pilot[1:]:
            self.assertEqual(sum(s["conductivity"][k]!=base[k] for k in axes),1)
    def payload(self):
        return dict(schema="trkg4_ttrkg_surface_v3",modality="TTRKG",montage="four_electrode_surface",placement_mode="explicit_points",coordinate_system="surface_path_v1",units="mm",symmetry=None,reference_definition="user_selected_centres_on_external_body_surface",electrode_order=ROLES,source_basename="body.stl",preliminary_electrode_centers_xyz_mm={k:[i,0.,0.] for i,k in enumerate(ROLES)})
    def test_roles_named_not_insertion_order(self):
        v=self.payload();v["preliminary_electrode_centers_xyz_mm"]=dict(reversed(list(v["preliminary_electrode_centers_xyz_mm"].items())))
        self.assertEqual(validate_landmarks(v)["I_plus"],[0,0.,0.])
    def test_wrong_units_duplicate_and_nonfinite_rejected(self):
        for mutate in [lambda v:v.update(units="m"),lambda v:v["preliminary_electrode_centers_xyz_mm"].update(I_plus=[1,0,0]),lambda v:v["preliminary_electrode_centers_xyz_mm"].update(I_plus=[float("nan"),0,0]),lambda v:v.update(electrode_order=list(reversed(ROLES)))]:
            with self.subTest(mutate=mutate):
                v=self.payload();mutate(v)
                with self.assertRaises(ValueError):validate_landmarks(v)

if __name__=="__main__":unittest.main()
