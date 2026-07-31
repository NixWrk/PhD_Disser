from pathlib import Path

import numpy as np
import pytest

from breathgeom.measure.joint_svf_registration import (
    RUNNER_INPUT_KEYS,
    build_runner_parameters,
    load_joint_svf_search,
    register_joint_piecewise_svf,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = REPO_ROOT / "configs/piecewise_svf_j12_development_search_v1.json"


def test_frozen_joint_search_is_finite_and_holds_challenge() -> None:
    search = load_joint_svf_search(SEARCH_PATH)

    assert search.search_version == "piecewise-svf-j1.2-development-search-v1"
    assert [variant.variant_id for variant in search.variants] == [
        "balanced",
        "contact_strong",
        "smooth_strong",
    ]
    assert search.challenge_suite_path.endswith("j12_challenge_suite_v1.json")
    assert "exact candidate config" in search.challenge_hold_condition
    assert {
        "fixed_image",
        "moving_image",
        "fixed_lung_mask",
        "moving_lung_mask",
        "fixed_body_mask",
        "moving_body_mask",
        "spacing_mm",
    } == RUNNER_INPUT_KEYS


def test_runner_parameters_contain_no_case_or_truth_metadata() -> None:
    search = load_joint_svf_search(SEARCH_PATH)
    payload = build_runner_parameters(search, search.variants[0])
    serialized = str(payload)

    assert "variant_id" not in serialized
    assert "case_id" not in serialized
    assert "truth" not in serialized
    assert "challenge" not in serialized
    assert payload["integrator"]["squaring_steps"] == 7
    assert payload["weights"]["advected_surface_weight"] == 2.0


def test_joint_adapter_rejects_mismatched_grids_before_external_run(
    tmp_path: Path,
) -> None:
    search = load_joint_svf_search(SEARCH_PATH)
    image = np.zeros((16, 16, 16), dtype=np.int16)
    mask = np.zeros_like(image, dtype=bool)
    mask[4:12, 4:12, 4:12] = True

    with pytest.raises(ValueError, match="share one 3D grid"):
        register_joint_piecewise_svf(
            image,
            image,
            mask,
            mask,
            mask,
            mask[:, :, :-1],
            (1.0, 1.0, 1.0),
            search=search,
            weights=search.variants[0],
            python_executable=tmp_path / "missing.exe",
            repo_root=REPO_ROOT,
        )
