"""Independent physical limits and ray-extension geometry checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('two_tissue',
    Path(__file__).resolve().parents[1] / 'tools/compare_curved_two_tissue.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


@pytest.mark.parametrize('rho', [1., 5.55, 30.7])
def test_homogeneous_halfspace_tetrapolar(rho):
    L = 140.
    z = model.tetrapolar(L, lambda a,b: model.slab_green(abs(a-b), 30, 80, rho, rho))
    assert z == pytest.approx(8 * rho / (3 * np.pi * L / 1000), rel=2e-13)


def test_vanishing_lung_and_deep_exit_limits():
    rho1, rho2, h, L = 5.55, 30.7, 30., 140.
    homogeneous = 8 * rho1 / (3 * np.pi * L / 1000)
    zero = model.tetrapolar(L, lambda a,b: model.slab_green(abs(a-b),h,0,rho1,rho2))
    assert zero == pytest.approx(homogeneous, rel=2e-13)
    infinite = model.tetrapolar(L, lambda a,b:
        rho1 / (2*np.pi) * model.image_kernel(abs(a-b),h,rho1,rho2))
    errors = []
    for t in [40., 400., 4000.]:
        z = model.tetrapolar(L, lambda a,b: model.slab_green(abs(a-b),h,t,rho1,rho2))
        errors.append(abs(z-infinite))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.001


@pytest.mark.parametrize('r,h,t', [(12.5,20,11), (35,30,116), (105,54,200)])
def test_general_stack_matches_special_three_layer(r,h,t):
    a = model.slab_green(r,h,t,5.55,30.7)
    b = model.stack_green(r,[h,h+t],5.55,30.7)
    assert a == pytest.approx(b, abs=1e-8)


def test_invalid_crossing_sequences_rejected():
    with pytest.raises(ValueError):
        model.stack_green(35,[20,50,90],5,30)
    with pytest.raises(ValueError):
        model.stack_green(35,[20,10],5,30)


def test_closed_box_and_first_entry_extension(tmp_path):
    vtk = pytest.importorskip('vtk')
    cube = vtk.vtkCubeSource()
    cube.SetBounds(0,2,0,2,0,2)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(cube.GetOutputPort())
    writer = vtk.vtkSTLWriter()
    path = tmp_path / 'lung_box.stl'
    writer.SetFileName(str(path))
    writer.SetInputConnection(triangles.GetOutputPort())
    assert writer.Write() == 1
    points = np.array([[-1,1,1], [1,1,1], [3,1,1], [3,3,1], [1,1,3]], dtype=float)
    closed, extended = model.classify_lung(points,path,np.zeros(3),
                                           np.array([0.,1.,0.]),np.array([1.,0.,0.]))
    np.testing.assert_array_equal(closed, [False,True,False,False,False])
    np.testing.assert_array_equal(extended, [False,True,True,False,False])
