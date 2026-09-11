"""Independent box-constrained inverse checks and held-out data separation."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import lsq_linear

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from geometry_validation_tests import recover_bounded


def linear_tissue_model():
    # In this artificial conductor Z=A[:,0]*rho1+A[:,1]*rho2, so an
    # independent convex bounded-linear solver supplies the reference.
    A=np.array([[1.,.1],[.8,.2],[.6,.5],[.4,.8],[.2,1.]])
    x=np.linspace(np.log(.1),np.log(100),601)
    f=A[:,0]+np.exp(x[:,None])*A[:,1]
    return A,CubicHermiteSpline(x,f,np.exp(x[:,None])*A[:,1],axis=0,extrapolate=False)


@pytest.mark.parametrize('truth',[[3.,8.],[.1,16.],[12.,2.],[10.,25.]])
def test_profiled_absolute_bounds_agree_with_independent_convex_solution(truth):
    A,spline=linear_tissue_model();bounds=np.array([[1.,6.],[4.,12.]])
    y=A@truth;reference=lsq_linear(A,y,bounds=(bounds[:,0],bounds[:,1]),tol=1e-13)
    fit,pred,_=recover_bounded(spline,y,bounds)
    np.testing.assert_allclose([fit['rho1'],fit['rho2']],reference.x,rtol=3e-7,atol=1e-7)
    np.testing.assert_allclose(pred,A@reference.x,rtol=3e-7,atol=1e-7)
    assert np.all(np.array([fit['rho1'],fit['rho2']])>=bounds[:,0]-1e-8)
    assert np.all(np.array([fit['rho1'],fit['rho2']])<=bounds[:,1]+1e-8)


def test_heldout_measurement_cannot_influence_tissue_fit():
    A,spline=linear_tissue_model();y=A@np.array([3.,8.]);ids=[0,1,2,3]
    fit,pred,_=recover_bounded(spline,y,[[1,6],[4,12]],ids)
    changed=y.copy();changed[-1]=10000
    fit2,pred2,_=recover_bounded(spline,changed,[[1,6],[4,12]],ids)
    assert fit==fit2
    np.testing.assert_equal(pred,pred2)


def test_prior_and_library_without_common_contrasts_are_rejected():
    A,spline=linear_tissue_model()
    with pytest.raises(ValueError,match='no contrast interval'):
        recover_bounded(spline,A@np.array([3,8]),[[1,2],[500,1000]])


def test_invalid_inputs_do_not_silently_produce_an_estimate():
    A,spline=linear_tissue_model();y=A@np.array([3,8])
    with pytest.raises(ValueError):recover_bounded(spline,y,[[0,2],[5,10]])
    with pytest.raises(ValueError):recover_bounded(spline,y,[[1,2],[5,10]],[0,0])
    y[1]=np.nan
    with pytest.raises(ValueError):recover_bounded(spline,y,[[1,2],[5,10]])
