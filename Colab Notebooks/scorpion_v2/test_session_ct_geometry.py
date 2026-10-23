"""Analytical checks of the native-cell ray integration, independent of CT data."""
import numpy as np
import pytest

from session_ct_geometry import interval_material_lengths


def test_partial_voxels_and_material_interfaces():
    # Unit cells centred at 0..4; a 3 mm interval cuts both endpoint cells.
    result=interval_material_lengths(np.arange(5.),np.array([0,4,4,2,0]),.25,3.25,1.)
    np.testing.assert_allclose(result,[.25,0,.75,0,2.])
    assert result.sum()==3.


def test_world_origin_and_voxel_size():
    origin=127.5;scale=.765625
    x=origin+scale*np.arange(5.)
    result=interval_material_lengths(x,np.array([0,4,4,2,0]),origin+.25*scale,origin+3.25*scale,scale)
    np.testing.assert_allclose(result,scale*np.array([.25,0,.75,0,2.]),atol=1e-12)


def test_outside_field_of_view_cannot_be_silently_filled():
    with pytest.raises(ValueError,match='cover the full interval'):
        interval_material_lengths(np.arange(5.),np.zeros(5,dtype=int),-2.,2.,1.)
