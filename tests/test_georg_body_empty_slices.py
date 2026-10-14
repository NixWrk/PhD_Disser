"""Regression checks for empty CT planes becoming conducting plates."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MATLAB_TRKG4_real_subjects/tools'))
from georg_ct_model import body_mask


def test_air_planes_remain_empty():
    hu = np.full((32, 32, 3), -1000.0)
    hu[8:24, 8:24, 1] = 0
    mask, _ = body_mask(hu)
    assert not mask[:, :, 0].any()
    assert not mask[:, :, 2].any()
    assert mask[16, 16, 1]
    assert mask[:, :, 1].sum() == 244


def test_component_removed_by_opening_does_not_select_air():
    hu = np.full((32, 32, 3), -1000.0)
    hu[8:24, 8:24, 1] = 0
    hu[16, 16, 0] = 0
    mask, _ = body_mask(hu)
    assert not mask[:, :, 0].any()
    assert mask[:, :, 1].any()


def test_air_volume_rejected_before_surface_extraction():
    with pytest.raises(ValueError, match='no above-threshold body'):
        body_mask(np.full((16, 16, 3), -1000.0))
