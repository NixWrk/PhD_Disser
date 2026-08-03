from pathlib import Path

import numpy as np

from breathgeom.measure.contact_svf_phantom import (
    load_contact_svf_suite,
    make_contact_svf_phantom_pair,
)
from breathgeom.synthetic_j12_identifiability import (
    CANDIDATE_NAMES,
    DataTermRecord,
    IdentifiabilityThresholds,
    build_identifiability_candidates,
    decide_identifiability,
    load_identifiability_screen,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = REPO_ROOT / "configs/piecewise_svf_j12_identifiability_screen_v1.json"
SUITE_PATH = (
    REPO_ROOT / "configs/piecewise_svf_j12_contact_development_suite_v3.json"
)


def test_frozen_identifiability_screen_has_finite_candidates_and_thresholds() -> None:
    screen = load_identifiability_screen(SCREEN_PATH)

    assert screen.screen_version == "piecewise-svf-j1.2-identifiability-screen-v1"
    assert screen.candidates == CANDIDATE_NAMES
    assert screen.metrics == (
        "mind_mse_full",
        "mind_mse_interior",
        "intensity_mse_full",
        "intensity_mse_interior",
    )
    assert screen.thresholds.truth_to_glued_ratio_max == 0.8


def test_candidate_builder_keeps_truth_inverse_and_glued_distinct() -> None:
    case = load_contact_svf_suite(SUITE_PATH).cases[0]
    generated = make_contact_svf_phantom_pair(case.params)
    candidates = build_identifiability_candidates(
        case,
        generated,
        generated.pair.phantom.lung_displacement_mm,
        generated.pair.phantom.body_displacement_mm,
    )

    assert tuple(candidates) == CANDIDATE_NAMES
    assert np.count_nonzero(candidates["zero"][0]) == 0
    assert np.array_equal(
        candidates["glued_mean_velocity"][0],
        candidates["glued_mean_velocity"][1],
    )
    assert not np.allclose(
        candidates["truth"][0],
        candidates["inverse_truth"][0],
    )


def _records(
    *,
    truth_mind: float,
    truth_intensity: float,
    glued_mind: float,
    glued_intensity: float,
) -> tuple[DataTermRecord, ...]:
    values = {
        "truth": (truth_mind, truth_intensity),
        "estimated_balanced": (0.3, 0.3),
        "glued_mean_velocity": (glued_mind, glued_intensity),
        "common_scale_only": (0.7, 0.7),
        "zero": (1.0, 1.0),
        "inverse_truth": (1.0, 1.0),
    }
    return tuple(
        DataTermRecord(
            screen_version="test",
            suite_version="test",
            case_id="case",
            candidate=candidate,
            region=region,
            mind_mse_full=mind,
            mind_mse_interior=mind,
            intensity_mse_full=intensity,
            intensity_mse_interior=intensity,
        )
        for candidate, (mind, intensity) in values.items()
        for region in ("lung", "body")
    )


def test_decision_distinguishes_descriptor_from_optimizer_failure() -> None:
    thresholds = IdentifiabilityThresholds(0.8, 0.8, 0.8, 0.8)
    descriptor = decide_identifiability(
        _records(
            truth_mind=0.45,
            truth_intensity=0.1,
            glued_mind=0.5,
            glued_intensity=0.5,
        ),
        thresholds=thresholds,
        balanced_gate_reasons=("tangential_slip",),
    )
    optimizer = decide_identifiability(
        _records(
            truth_mind=0.1,
            truth_intensity=0.1,
            glued_mind=0.5,
            glued_intensity=0.5,
        ),
        thresholds=thresholds,
        balanced_gate_reasons=("tangential_slip",),
    )

    assert descriptor.classification == "descriptor_failure"
    assert optimizer.classification == "optimizer_or_parameterization_failure"
