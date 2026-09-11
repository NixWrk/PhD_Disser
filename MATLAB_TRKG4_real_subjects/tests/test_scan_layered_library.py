import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scan_layered_library import planar,build_library,fit_batch

SIZES=np.array([50,60,70,80,90,110,120,130,140.])
R1,R2=5.5521664546649,30.7069468090815

def test_limits_and_known_h_recovery():
    lib=build_library(SIZES,R1,R2)
    h=np.array([0.,.25,24.7,75.,200.,np.inf]);z=planar(h,SIZES,R1,R2)
    np.testing.assert_allclose(z[0],8*R2/(3*np.pi*SIZES*.001),atol=1e-11)
    np.testing.assert_allclose(z[-1],8*R1/(3*np.pi*SIZES*.001),atol=1e-11)
    fit=fit_batch(z,lib)
    assert np.max(fit['rmse_ohm'])<2*lib['epsilon_ohm']
    np.testing.assert_allclose(fit['h_mm'][:-1],h[:-1],atol=.001)
    assert fit['infinite'][-1]

def test_chord_certificate_covers_actual_curve_and_tail():
    lib=build_library(SIZES,R1,R2)
    for fraction in [.211,.5,.789]:
        h=lib['h'][:-1]+fraction*np.diff(lib['h'])
        line=lib['z'][:-1]+fraction*np.diff(lib['z'],axis=0)
        assert np.max(abs(planar(h,SIZES,R1,R2)-line))<=lib['epsilon_ohm']+1e-9
    for h in [lib['tail_start_mm'],10*lib['tail_start_mm']]:
        assert np.max(abs(planar([h],SIZES,R1,R2)[0]-lib['asymptote']))<=lib['epsilon_ohm']

def test_global_fit_and_leave_one_out_match_independent_optimization():
    lib=build_library(SIZES,R1,R2)
    rng=np.random.default_rng(215)
    target=planar([4.,24.7,63.],SIZES,R1,R2)+rng.normal(0,8,(3,9))
    target=np.vstack([target,8*R1/(3*np.pi*SIZES*.001)-5])
    fit=fit_batch(target,lib)
    for i,y in enumerate(target):
        for omitted in [-1,0,4,8]:
            mask=np.ones(9,bool)
            if omitted>=0:mask[omitted]=False
            def loss(h):return np.mean((planar([h],SIZES,R1,R2)[0,mask]-y[mask])**2)
            grid=np.r_[np.linspace(0,200,801),np.geomspace(201,1e5,100)]
            vals=np.array([loss(h) for h in grid]);indices=np.flatnonzero((vals[1:-1]<=vals[:-2])&(vals[1:-1]<=vals[2:]))+1
            candidates=[loss(0),loss(np.inf)]
            candidates += [minimize_scalar(loss,bounds=(grid[k-1],grid[k+1]),method='bounded').fun for k in indices]
            h=fit['h_mm'][i] if omitted<0 else fit['loo_h_mm'][i,omitted]
            assert np.sqrt(loss(h))-np.sqrt(min(candidates))<=2*lib['epsilon_ohm']+1e-7

def test_held_out_value_does_not_change_its_own_h_fit():
    lib=build_library(SIZES,R1,R2);a=planar([27],SIZES,R1,R2)
    b=a.copy();b[0,3]+=50
    fa=fit_batch(a,lib);fb=fit_batch(b,lib)
    np.testing.assert_allclose(fa['loo_h_mm'][0,3],fb['loo_h_mm'][0,3],atol=1e-6)
    assert abs(fa['h_mm'][0]-fb['h_mm'][0])>1
