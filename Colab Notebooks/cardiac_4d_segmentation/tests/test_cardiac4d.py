from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "cardiac4d.py"
SPEC = importlib.util.spec_from_file_location("cardiac4d", MODULE_PATH)
cardiac4d = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cardiac4d)


def test_mask_metrics_use_affine_determinant_for_volume():
    mask = np.zeros((4, 5, 6), dtype=bool)
    mask[1:3, 1:4, 2:5] = True
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    metrics = cardiac4d.mask_metrics(mask, affine)
    assert metrics["voxel_count"] == 18
    assert metrics["volume_ml"] == pytest.approx(18 * 24 / 1000)
    assert metrics["connected_components"] == 1


def test_export_mesh_keeps_world_coordinates_and_is_watertight(tmp_path):
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    affine = np.eye(4)
    affine[:3, 3] = [10.0, 20.0, 30.0]
    result = cardiac4d.export_mesh(mask, affine, tmp_path / "cube.stl")
    assert (tmp_path / "cube.stl").is_file()
    assert result["watertight"] is True
    assert result["units"].startswith("mm")


def test_clean_binary_removes_small_component():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[1, 1, 1] = True
    mask[5:10, 5:10, 5:10] = True
    cleaned = cardiac4d.clean_binary(mask, min_component_ml=0.01, voxel_ml=0.001)
    assert not cleaned[1, 1, 1]
    assert cleaned[7, 7, 7]

