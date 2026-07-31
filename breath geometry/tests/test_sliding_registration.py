import json
from pathlib import Path

import numpy as np
import pytest

from breathgeom.measure.sliding_registration import (
    couple_normal_components,
    load_sliding_s1_params,
    lung_interface_normal,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def spherical_mask(shape: tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64)
    centre = (np.asarray(shape, dtype=np.float64) - 1.0)[:, None, None, None] / 2.0
    radius = np.sqrt(np.sum(np.square(grid - centre), axis=0))
    return radius <= 12.0


def test_normal_coupling_preserves_tangential_slip() -> None:
    mask = spherical_mask()
    normal, _ = lung_interface_normal(mask, (1.0, 1.0, 1.0))
    tangent = np.stack(
        (-normal[..., 1], normal[..., 0], np.zeros(mask.shape)),
        axis=-1,
    )
    tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = np.divide(
        tangent,
        tangent_norm,
        out=np.zeros_like(tangent),
        where=tangent_norm > 1e-8,
    )
    lung = (2.0 * normal + 4.0 * tangent).astype(np.float32)
    body = (3.0 * normal).astype(np.float32)

    lung_coupled, body_coupled, returned_normal, weight = couple_normal_components(
        lung,
        body,
        mask,
        (1.0, 1.0, 1.0),
        normal_smoothing_mm=2.0,
        contact_core_mm=1.5,
        contact_taper_mm=3.0,
        shared_normal_source="body",
    )

    core = (weight == 1.0) & (tangent_norm[..., 0] > 1e-8)
    difference = lung_coupled - body_coupled
    normal_mismatch = np.abs(np.sum(difference * returned_normal, axis=-1))
    tangential = difference - np.sum(
        difference * returned_normal,
        axis=-1,
    )[..., None] * returned_normal
    assert float(np.max(normal_mismatch[core])) < 1e-5
    assert np.median(np.linalg.norm(tangential[core], axis=-1)) == pytest.approx(4.0)
    assert np.array_equal(body_coupled, body)


def test_normal_coupling_tapers_to_raw_fields() -> None:
    mask = spherical_mask()
    lung = np.zeros(mask.shape + (3,), dtype=np.float32)
    body = np.ones_like(lung)

    lung_coupled, body_coupled, _, weight = couple_normal_components(
        lung,
        body,
        mask,
        (1.0, 1.0, 1.0),
        normal_smoothing_mm=2.0,
        contact_core_mm=1.0,
        contact_taper_mm=2.0,
        shared_normal_source="mean",
    )

    outside = weight == 0.0
    assert np.array_equal(lung_coupled[outside], lung[outside])
    assert np.array_equal(body_coupled[outside], body[outside])


def test_versioned_s1_configuration_is_complete() -> None:
    params = load_sliding_s1_params(REPO_ROOT / "configs/sliding_s1_v0.json")

    assert params.version == "s1.0"
    assert params.lung.grid_sp == 3
    assert params.lung.disp_hw == 6
    assert params.lung.lambda_weight == 0.6
    assert params.body.level_spacing_mm == (6.0, 3.0)
    assert params.normal_smoothing_mm == 2.0
    assert params.shared_normal_source == "body"


def test_s1_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    source = REPO_ROOT / "configs/sliding_s1_v0.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["undeclared_parameter"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_sliding_s1_params(path)
