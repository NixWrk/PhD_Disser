import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from build_best_pose_review import ranking_radius

def test_exact_data_space_boundary_and_a_point_on_either_side():
    prediction=np.array([[0.,0.],[4.,0.],[0.,2.]])
    observed=np.zeros(2)
    rival,radius,delta=ranking_radius(prediction,observed,0)
    assert rival==2
    np.testing.assert_allclose(radius,1/np.sqrt(2),atol=1e-14)
    np.testing.assert_allclose(delta,[0,1],atol=1e-14)
    for fraction,winner in [(.99,0),(1.01,2)]:
        errors=np.sum((prediction-observed-fraction*delta)**2,axis=1)
        assert errors.argmin()==winner

def test_no_smaller_perturbation_beats_winner_in_any_direction():
    prediction=np.array([[1.,2.,3.],[3.,4.,3.],[0.,-1.,2.],[5.,1.,2.]])
    observed=np.array([1.1,1.9,3.2]);best=np.sum((prediction-observed)**2,axis=1).argmin()
    rival,radius,delta=ranking_radius(prediction,observed,best)
    rng=np.random.default_rng(20260910);directions=rng.normal(size=(1000,3))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    perturbed=observed+.999*radius*np.sqrt(3)*directions
    errors=np.sum((prediction[None,:,:]-perturbed[:,None,:])**2,axis=2)
    assert np.all(errors.argmin(axis=1)==best)
    tied=np.sum((prediction-observed-delta)**2,axis=1)
    np.testing.assert_allclose(tied[best],tied[rival],atol=1e-12)
