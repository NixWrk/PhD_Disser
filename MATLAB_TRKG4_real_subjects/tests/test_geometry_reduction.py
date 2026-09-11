"""Physical and inverse-problem checks independent of the CT reference data."""
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from geometry_reduction_study import ellipsoid_interval
from geometry_reduction_analysis import recover


def test_ellipsoid_ray_and_depth_translation():
    centre=np.array([2.,3.,10.]);Q=np.diag([1/9,1/16,1/25])
    origin=np.array([2.,3.,0.]);direction=np.array([0.,0.,1.])
    np.testing.assert_allclose(ellipsoid_interval(origin,direction,centre,Q),[5,15])
    moved=centre+7*direction
    np.testing.assert_allclose(ellipsoid_interval(origin,direction,moved,Q),[12,22])
    assert all(np.isnan(x) for x in ellipsoid_interval(np.array([20.,3.,0.]),direction,centre,Q))


def test_joint_rotation_does_not_change_intersection():
    angle=.73
    R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    c=np.array([0.,0.,10.]);d=np.array([0.,0.,1.]);Q=np.diag([1/9,1/16,1/25])
    np.testing.assert_allclose(ellipsoid_interval(np.zeros(3),R@d,R@c,R@Q@R.T),[5,15])


def synthetic_spline():
    x=np.linspace(np.log(.5),np.log(32),61)
    # Different depth responses: conductivity contrast changes curve shape.
    a=np.array([1.,1.4,1.8,2.2,2.6,3.,3.5,4.,4.5])
    b=np.array([.1,.15,.23,.4,.7,1.,1.5,2.4,3.6])
    f=a[None,:]+b[None,:]*np.exp(x[:,None])
    return CubicHermiteSpline(x,f,b[None,:]*np.exp(x[:,None]),axis=0)


def test_recovery_on_independent_contrast_and_scale():
    spline=synthetic_spline();rho1=4.3;ratio=5.7
    y=rho1*spline(np.log(ratio))
    fit,pred,jac=recover(spline,y)
    np.testing.assert_allclose([fit['rho1'],fit['rho2']],[rho1,rho1*ratio],rtol=1e-7)
    np.testing.assert_allclose(pred,y,rtol=1e-8)
    eps=1e-5
    rho2=rho1*ratio
    def forward(r1,r2): return r1*spline(np.log(r2/r1))
    fd1=(forward(rho1*np.exp(eps),rho2)-forward(rho1*np.exp(-eps),rho2))/(2*eps)
    fd2=(forward(rho1,rho2*np.exp(eps))-forward(rho1,rho2*np.exp(-eps)))/(2*eps)
    np.testing.assert_allclose(jac,np.column_stack([fd1,fd2]),rtol=1e-6)


def test_pair_prediction_uses_unfitted_sizes():
    spline=synthetic_spline();y=6*spline(np.log(3.9))
    fit,pred,_=recover(spline,y,[0,8])
    np.testing.assert_allclose([fit['rho1'],fit['rho2']],[6,23.4],rtol=1e-7)
    np.testing.assert_allclose(pred[1:8],y[1:8],rtol=1e-8)


def test_collinear_responses_are_flagged_as_ill_conditioned():
    x=np.linspace(np.log(.5),np.log(32),21);a=np.arange(1,10,dtype=float)
    f=np.exp(x[:,None])*a
    spline=CubicHermiteSpline(x,f,f,axis=0)
    fit,_,_=recover(spline,3*spline(np.log(5)))
    assert fit['condition_log_parameters']>1e12
    assert fit['tied_candidates']>1


def test_duplicate_serialized_contrasts_keep_distinct_runs():
    import pandas as pd
    from geometry_reduction_analysis import curve_at
    data=pd.DataFrame({'ratio_index':np.repeat([5,6],9),'ratio':np.ones(18),
                       'L_mm':np.tile(np.arange(9),2),'f_ohm_per_ohm_m':np.tile(np.arange(9),2)})
    np.testing.assert_equal(curve_at(data,1.),np.arange(9))
