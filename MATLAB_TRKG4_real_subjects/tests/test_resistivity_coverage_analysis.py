"""Analytic checks of rho-grid interpolation and endpoint handling."""
import sys,itertools
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_resistivity_coverage import audit_interpolation,endpoint_excursions
from resistivity_sensitivity_metrics import range_scaled_metrics
P=['soft','heart','lung']
PROFILE={'bounds':{'soft':[1,4],'heart':[1,5],'lung':[2,6]},'original_rho_axes':{'soft':[1,2,4],'heart':[1,3,5],'lung':[2,6]}}
def frame(points):
    rows=[]
    for i,(s,h,l) in enumerate(points):
        derivatives=np.array([6*s+.1*h*l,.4*h+.1*s*l,1.4*l+.1*s*h])
        Q=range_scaled_metrics(derivatives,np.array([3,4,4]))['selectivity']
        row=dict(montage='analytic',state=f's{i}',rho_soft=s,rho_heart=h,rho_lung=l,Z_ohm=3*s*s+.2*h*h+.7*l*l+.1*s*h*l)
        row.update(zip(['S_'+k for k in P],derivatives));row.update(zip(['Q_'+k for k in P],Q));rows.append(row)
    return pd.DataFrame(rows)
def inputs():
    coarse=frame(itertools.product(*PROFILE['original_rho_axes'].values()))
    checks=frame(itertools.product([1.5,3],[2,4],[4]))
    return coarse,checks
def test_multilinear_derivatives_are_exact_even_when_Z_is_not_linear():
    coarse,checks=inputs();a=audit_interpolation(coarse,checks,PROFILE)
    assert np.all(a.absolute_error<1e-12)
    assert a.derivative_pass.all() and a.Q_pass.all()
    assert a.Z_relative_error.max()>0

def test_interior_change_fails_declared_rule():
    coarse,checks=inputs();checks['S_heart']*=2
    a=audit_interpolation(coarse,checks,PROFILE)
    assert not a.loc[a.compartment=='heart','derivative_pass'].any()

def test_endpoint_difference_matches_analytic_integral():
    coarse,_=inputs();e=endpoint_excursions(coarse,PROFILE)
    for _,r in e.iterrows():
        lo,hi=PROFILE['bounds'][r.compartment]
        if r.compartment=='soft': expected=3*(hi**2-lo**2)+.1*r.background_rho_heart*r.background_rho_lung*(hi-lo)
        elif r.compartment=='heart':expected=.2*(hi**2-lo**2)+.1*r.background_rho_soft*r.background_rho_lung*(hi-lo)
        else:expected=.7*(hi**2-lo**2)+.1*r.background_rho_soft*r.background_rho_heart*(hi-lo)
        assert np.isclose(r.signed_excursion_ohm,expected)
    assert list(e.groupby('compartment').size())==[6,9,6]

def test_endpoint_grouping_tolerates_roundtrip_noise():
    coarse,_=inputs();expected=endpoint_excursions(coarse,PROFILE)
    changed=(coarse.rho_heart==3)&(coarse.rho_lung==6)
    coarse.loc[changed,'rho_heart']=np.nextafter(3.0,4.0)
    actual=endpoint_excursions(coarse,PROFILE)
    pd.testing.assert_frame_equal(actual,expected)
