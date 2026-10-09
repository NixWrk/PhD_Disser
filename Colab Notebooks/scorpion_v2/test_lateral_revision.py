"""Mathematical and geometry invariants for the 2026-10-09 revision."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from lateral_pilot import ROOT, DiskModel
from lateral_identifiability import KernelBank, vector
from lateral_admissible_sets import cloud, profile, single_assembly
from build_georg_fem_reports import require_current_geometry

sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
from scorpion_georg_geometry_revision import embedded_join, longitudinal_centres, RADIUS


class RevisionChecks(unittest.TestCase):
    def test_bounded_cloud_is_consistent_with_independent_forward_evaluation(self):
        bounds=np.array([[1.3,40.],[3.6,25.]])
        bank=KernelBank(h=[2.,10.,30.],q=np.geomspace(.1,15.,33))
        y=vector([4.,15.,20.])+[.1,-.05,.08,-.1,.03]
        minimum=float(profile(bank,y,bounds)['rss'].min())
        points=cloud(bank,y,bounds,minimum)
        self.assertGreater(len(points),0)
        for p in points[::max(1,len(points)//15)]:
            self.assertTrue(np.all(p[:2]>=bounds[:,0]-1e-10))
            self.assertTrue(np.all(p[:2]<=bounds[:,1]+1e-10))
            residual=vector(p[:3])-y
            np.testing.assert_allclose(residual@residual,p[3],rtol=1e-9,atol=1e-9)
            self.assertLessEqual(p[3],1.1*minimum+1e-9)

    def test_scalar_curves_reproduce_observation_without_claiming_unique_pair(self):
        bank=KernelBank(h=[5.,20.,40.],q=np.geomspace(.1,15.,49))
        bounds=np.array([[1.3,40.],[3.6,25.]])
        observation=vector([4.,15.,20.])[2]
        _,lines,error=single_assembly(bank,observation,bounds,2,[1.3,4.,40.],[3.6,15.,25.])
        self.assertLess(error,1e-7)
        self.assertGreater(np.isfinite(lines).sum(),5)
        solutions=[]
        for j,fixed in enumerate([3.6,15.,25.]):
            if np.isfinite(lines[0,j,1]):
                solutions.append([lines[0,j,1],fixed])
                predicted=DiskModel([100],20.,radial=2,angular=8,terms=2048).evaluate(solutions[-1])[0][0]
                self.assertAlmostEqual(predicted,observation,places=7)
        self.assertGreaterEqual(len(solutions),2)

    def test_whole_disk_join_is_embedded_and_longitudinal_array_keeps_common_centre(self):
        body=trimesh.creation.box(extents=[300.,200.,200.])
        original=dict(direction='negative_axis',cut_coordinate=-150.,center_y=0.,center_z=0.)
        join=embedded_join(body,original,np.array([.5,1.5,2.5]))
        self.assertAlmostEqual(join['cross_section_covered_fraction'],1.,places=10)
        self.assertLess(abs(join['polygon_area_mm2']/(np.pi*RADIUS**2)-1),.00011)
        centres=np.asarray(longitudinal_centres(body,0.,0.))
        np.testing.assert_allclose(centres[:,:,1],0.,atol=1e-10)
        np.testing.assert_allclose(centres[:,:,2].mean(axis=1),0.,atol=1e-10)
        np.testing.assert_allclose(centres[:,3,2]-centres[:,0,2],[60,80,100,120,140])

    def test_superseded_geometry_cannot_be_consumed_as_current(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'Experiments/scorpion_v2_georg/derived/exploratory') as temp:
            out=Path(temp);(out/'superseded_geometry.local.json').write_text('{}')
            with self.assertRaises(ValueError):require_current_geometry(out)
            require_current_geometry(out,allow_superseded=True)


if __name__=='__main__':unittest.main()
