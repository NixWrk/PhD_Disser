from pathlib import Path

from breathgeom.measure.sliding_phantom import make_sliding_phantom_pair
from breathgeom.synthetic_s1 import (
    evaluate_synthetic_s1_fields,
    load_sliding_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_sliding_suite_has_three_unique_variants() -> None:
    suite = load_sliding_suite(
        REPO_ROOT / "configs/sliding_phantom_suite_v1.json"
    )

    assert suite.suite_version == "sliding-phantom-v2.0"
    assert len(suite.cases) == 3
    assert len({case.case_id for case in suite.cases}) == 3
    assert suite.cases[-1].params.spacing_mm == (1.25, 1.25, 1.5)


def test_exact_hidden_fields_pass_algorithmic_synthetic_gate() -> None:
    suite = load_sliding_suite(
        REPO_ROOT / "configs/sliding_phantom_suite_v1.json"
    )
    case = suite.cases[0]
    pair = make_sliding_phantom_pair(case.params)

    record = evaluate_synthetic_s1_fields(
        suite_version=suite.suite_version,
        case=case,
        pair=pair,
        s1_version="exact-control",
        lung_displacement_mm=pair.phantom.lung_displacement_mm,
        body_displacement_mm=pair.phantom.body_displacement_mm,
    )

    assert record.gate_pass
    assert record.gate_reasons == ()
    assert record.lung_field_p95_mm == 0.0
    assert record.body_field_p95_mm == 0.0


def test_artificially_glued_fields_fail_slip_and_endpoint_gates() -> None:
    suite = load_sliding_suite(
        REPO_ROOT / "configs/sliding_phantom_suite_v1.json"
    )
    case = suite.cases[0]
    pair = make_sliding_phantom_pair(case.params)

    record = evaluate_synthetic_s1_fields(
        suite_version=suite.suite_version,
        case=case,
        pair=pair,
        s1_version="glued-control",
        lung_displacement_mm=pair.phantom.body_displacement_mm,
        body_displacement_mm=pair.phantom.body_displacement_mm,
    )

    assert not record.gate_pass
    assert "lung_field_p95" in record.gate_reasons
    assert "tangential_slip" in record.gate_reasons
