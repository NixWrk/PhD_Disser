import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from sensitivity_probes import select_volume_probe, local_stiffness


def test_equal_effective_volume_with_unequal_elements():
    centres = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], float)
    volume = np.array([2,5,3,8])*1e-6
    labels = np.array([1,1,2,2])
    a = select_volume_probe(centres, volume, labels, 1, 4, [0,0,0])
    b = select_volume_probe(centres, volume, labels, 2, 4, [2,0,0])
    assert a['effective_volume_ml'] == pytest.approx(4)
    assert b['effective_volume_ml'] == pytest.approx(4)
    assert a['support_volume_ml'] == pytest.approx(7)
    assert b['support_volume_ml'] == pytest.approx(11)
    assert a['weights'] == pytest.approx([1,.4])
    assert b['weights'] == pytest.approx([1,.125])


def test_local_stiffness_linear_field_and_orientation():
    nodes = np.array([[0,0,0],[2,0,0],[0,3,0],[0,0,4]], float)
    u = nodes @ [1,2,3]
    v = nodes @ [-2,4,1]
    for element in [[[0,1,2,3]], [[3,2,1,0]]]:
        B = local_stiffness(nodes, np.array(element), np.array([0]), np.array([.25]), 6)
        up = np.r_[u, 0, 0]; vp = np.r_[v, 0, 0]
        assert -vp @ (B @ up) == pytest.approx(-4*.25*9)
        assert np.asarray(B.sum(axis=1)).ravel() == pytest.approx(np.zeros(6))
        assert B.toarray() == pytest.approx(B.toarray().T)


def test_probe_invalid_inputs():
    with pytest.raises(ValueError):
        select_volume_probe([[0,0,0]], [1e-6], [1], 1, 2)
    with pytest.raises(ValueError):
        select_volume_probe([[0,0,0]], [-1], [1], 1, 1)
    with pytest.raises(ValueError):
        select_volume_probe([[0,0,0]], [1e-6], [1], 1, 0)


def test_numerically_singular_local_probe_is_rejected():
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1e-12]])
    with pytest.raises(ValueError,match='Degenerate'):
        local_stiffness(nodes,np.array([[0,1,2,3]]),np.array([0]),np.array([1.]),4)
