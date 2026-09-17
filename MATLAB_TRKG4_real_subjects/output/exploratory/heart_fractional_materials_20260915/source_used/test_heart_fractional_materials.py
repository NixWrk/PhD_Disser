"""Independent geometry and P1-limit checks for fractional material integration."""
import itertools
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from heart_fractional_materials import (uniform_tet_barycentric,integrate_fractions,
    material_fractions,p1_element_matrix,safe_box_overlap,expand_heart_fraction)

def sphere(radius=1.,centre=(0,0,0)):
    return {'centre_m':list(centre),'semiaxes_m':[radius]*3,'axes_columns':np.eye(3).tolist()}

def cube_tets(n=4):
    knots=np.linspace(-1.5,1.5,n+1);result=[]
    for i,j,k in itertools.product(range(n),repeat=3):
        lower=np.array([knots[i],knots[j],knots[k]])
        for axes in itertools.permutations(range(3)):
            v=[lower.copy()]
            for axis in axes:
                q=v[-1].copy();q[axis]+=3/n;v.append(q)
            result.append(v)
    return np.array(result)

def tet_volumes(v):
    return np.abs(np.linalg.det(v[:,1:]-v[:,:1]))/6

def test_nested_uniform_spacings():
    bary=uniform_tet_barycentric(4096)
    assert np.array_equal(bary[:64],uniform_tet_barycentric(64))
    assert np.array_equal(bary[:512],uniform_tet_barycentric(512))
    assert np.all(bary>=0) and np.max(abs(bary.sum(axis=1)-1))<1e-14
    assert np.max(abs(bary.mean(axis=0)-.25))<.0015
    assert np.max(abs((bary**2).mean(axis=0)-.1))<.0015
    with pytest.raises(ValueError):uniform_tet_barycentric(65)

def test_exact_inside_outside_and_containment_without_inside_vertices():
    small=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]])
    large=np.array([[3,0,-1],[-3,0,-1],[0,3,1],[0,-3,1]])
    tets=np.array([small,small+4,large])
    f,c=integrate_fractions(tets,sphere(.3))
    assert np.all(f[:,0]==1) and np.all(f[:,1]==0)
    assert c['inside'][0] and c['outside'][1] and c['quadrature'][2]
    assert f[-1,2]>0

def test_centroid_outside_does_not_drop_intersection():
    v=np.array([[[.9,0,0],[2,0,0],[2,1,0],[2,0,1]]])
    assert np.linalg.norm(v.mean(axis=1)[0])>1
    assert safe_box_overlap(v,np.array([-1,-1,-1]),np.ones(3))[0]
    f,c=integrate_fractions(v,sphere())
    assert c['quadrature'][0] and f[-1,0]>0

def test_known_sphere_and_rotated_ellipsoid_volumes():
    v=cube_tets();vol=tet_volumes(v)
    assert abs(vol.sum()-27)<1e-12
    th=.43;R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
    candidates=[sphere(),{'centre_m':[.1,-.1,.05],'semiaxes_m':[.8,1.,.6],'axes_columns':R.tolist()}]
    for p in candidates:
        f,_=integrate_fractions(v,p)
        expected=4*np.pi/3*np.prod(p['semiaxes_m'])
        errors=abs(f@vol/expected-1)
        assert errors[-1]<.003, errors
        assert errors[-1]<errors[0], errors

def test_affine_equivalence_of_sphere_and_ellipsoid():
    v=cube_tets(2);p=sphere();f,_=integrate_fractions(v,p)
    theta=.37;R=np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
    a=np.array([.5,1.2,2.]);c=np.array([4,2,3]);transformed=(v*a)@R.T+c
    g,_=integrate_fractions(transformed,{'centre_m':c,'semiaxes_m':a,'axes_columns':R})
    assert np.array_equal(f,g)

def test_immutable_background_and_sigma_limits():
    bg=np.array([1,2,4,1]);before=bg.copy();f=np.array([0,.2,.7,1.])
    fractions=material_fractions(f,bg)
    assert np.array_equal(bg,before)
    assert np.allclose(fractions.sum(axis=1),1)
    assert np.allclose(fractions[:,2],f)
    assert np.allclose(fractions@np.full(4,.3),.3)
    sigma=np.array([.1,.2,.6,.03])
    assert np.allclose(material_fractions(np.zeros(4),bg)@sigma,sigma[bg-1])
    assert np.allclose(material_fractions(np.ones(4),bg)@sigma,.6)
    with pytest.raises(ValueError):material_fractions(f,np.array([1,2,3,4]))
    with pytest.raises(ValueError):material_fractions(np.array([0,0,-.01,1]),bg)

def test_p1_matrix_integral_and_known_plane_cut():
    # Unit simplex z<=t has volume fraction 1-(1-t)^3 (exact integration).
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    gradients=np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])
    unit=(gradients@gradients.T)/6
    for t in (0,.1,.3,.5,1):
        f=1-(1-t)**3;fractions=np.array([1-f,0,f,0]);sigma=np.array([.1,.2,.6,.03])
        actual=p1_element_matrix(v,fractions,sigma)
        exact=((1-f)*.1+f*.6)*unit
        assert np.allclose(actual,exact,rtol=0,atol=1e-15)
        assert np.allclose(p1_element_matrix(v,fractions,np.full(4,.3)),.3*unit,rtol=0,atol=1e-15)
        # Material block linearity and current conservation.
        assert np.allclose(actual.sum(axis=1),0,atol=1e-15)
    with pytest.raises(ValueError):p1_element_matrix(v,np.array([.5,.6]),np.array([1,2]))

def test_expand_sparse_global_indices():
    indices=np.array([1,7]);values=np.array([.25,.5],dtype='float32')
    f=expand_heart_fraction(10,indices,values)
    assert f.shape==(10,) and np.array_equal(f[indices],values)
    assert np.count_nonzero(f)==2
    with pytest.raises(ValueError):expand_heart_fraction(10,np.array([1,1]),values)
