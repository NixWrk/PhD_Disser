import sys
from pathlib import Path
import numpy as np
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_construction_variants import fit_surface_spheres, surface_quadrature

class ConstructionCriteriaTests(unittest.TestCase):
    def test_recover_translated_sphere_with_unequal_weights(self):
        k=np.arange(160);z=1-2*(k+.5)/len(k);theta=k*np.pi*(3-np.sqrt(5))
        unit=np.column_stack((np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z))
        center=np.array([12.,-7.,21.]);radius=18.
        points=center+radius*unit
        fixed,free=fit_surface_spheres(points,np.linspace(.2,2,len(k)),center+[2,1,-1],
                                       np.stack((points.min(0),points.max(0))))
        np.testing.assert_allclose(free['center_mm'],center,atol=1e-7)
        self.assertAlmostEqual(free['radius_mm'],radius,places=7)
        self.assertLess(free['surface_RMS_mm'],1e-7)
        self.assertLess(free['surface_RMS_mm'],fixed['surface_RMS_mm'])
        self.assertFalse(any(free['optimization']['active_bounds']))

    def test_fixed_center_radius_uses_surface_area_weights(self):
        points=np.array([[2,0,0],[-4,0,0],[0,3,0],[0,-3,0],[0,0,2],[0,0,-2.]])
        weights=np.array([1,4,2,2,1,1.])
        fixed,_=fit_surface_spheres(points,weights,np.zeros(3),np.stack((points.min(0),points.max(0))))
        self.assertAlmostEqual(fixed['radius_mm'],34/11)

    def test_surface_quadrature_respects_physical_similarity(self):
        mask=np.zeros((8,9,10),bool);mask[1:6,2:8,2:7]=True
        affine=np.diag([1.,2.,3.,1.]);p,w,meta=surface_quadrature(mask,affine)
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        transform=np.eye(4);transform[:3,:3]=2*rotation;transform[:3,3]=[11,-23,40]
        q,v,other=surface_quadrature(mask,transform@affine)
        np.testing.assert_allclose(q,p@(2*rotation).T+transform[:3,3],atol=1e-12)
        np.testing.assert_allclose(v,4*w,atol=1e-12)
        self.assertAlmostEqual(other['area_mm2'],4*meta['area_mm2'])
        self.assertTrue(np.all(w>0))

if __name__=='__main__': unittest.main()
