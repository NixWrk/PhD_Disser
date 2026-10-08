"""Mathematical checks independent of the experimental fitting outcome."""
import unittest
import numpy as np
from lateral_identifiability import KernelBank, vector, log_jacobian


class IdentifiabilityChecks(unittest.TestCase):
    def test_homogeneous_medium_has_no_thickness_information(self):
        reference=vector([4.,4.,.5])
        for h in (3.,30.,100.):
            np.testing.assert_allclose(vector([4.,4.,h]),reference,rtol=1e-12)
        self.assertLess(np.linalg.norm(log_jacobian([4.,4.,30.])[:,2]),1e-10)

    def test_solution_cloud_obeys_actual_forward_residual_and_both_bounds(self):
        bank=KernelBank(h=[2.,15.,30.],q=np.geomspace(.001,1000.,25))
        truth=np.array([4.,16.,30.]);y=vector(truth)+np.array([.15,-.1,.1,-.05,.03])
        minimum=float(np.min(bank.profile(y)['sse']))
        cloud,_=bank.cloud(y,minimum)
        self.assertGreater(len(cloud),0)
        self.assertTrue(np.all((cloud[:,:2]>=.1*(1-1e-12))&(cloud[:,:2]<=100*(1+1e-12))))
        for point in cloud[::max(1,len(cloud)//20)]:
            residual=vector(point[:3])-y
            np.testing.assert_allclose(residual@residual,point[3],rtol=2e-10,atol=1e-10)
            self.assertLessEqual(point[3],1.10*minimum*(1+1e-10))

    def test_logarithmic_jacobian_matches_independent_perturbations(self):
        p=np.array([2.7,21.,17.]);step=2e-5;columns=[]
        for j in range(3):
            plus=p.copy();minus=p.copy();plus[j]*=np.exp(step);minus[j]*=np.exp(-step)
            columns.append((vector(plus)-vector(minus))/(2*step))
        np.testing.assert_allclose(log_jacobian(p),np.column_stack(columns),rtol=1e-6,atol=1e-8)


if __name__=='__main__':unittest.main()
