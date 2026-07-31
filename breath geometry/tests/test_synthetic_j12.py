import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from breathgeom.measure.joint_svf_registration import (
    JointSVFResult,
    JointSVFRuntime,
    load_joint_svf_search,
)
from breathgeom.measure.sliding_phantom import make_sliding_phantom_pair
from breathgeom.synthetic_j12 import (
    evaluate_advected_surface,
    evaluate_j12_development,
    evaluate_j12_truth_preflight,
    run_j12_development,
    validate_j12_suite_truth,
    write_j12_truth_preflight,
)
from breathgeom.synthetic_s1 import load_sliding_suite

REPO_ROOT = Path(__file__).resolve().parents[1]


def runtime() -> JointSVFRuntime:
    return JointSVFRuntime(
        runner_version="joint-piecewise-svf-torch-v0",
        torch_version="test",
        convexadam_version="test",
        mind_source_module="test",
        mind_source_file_name="test.py",
        scipy_version="test",
        device="test",
        peak_gpu_memory_bytes=1,
        elapsed_s=0.0,
        control_shape_dhw=(3, 3, 3),
        surface_point_count=1,
        objective_initial=1.0,
        objective_final=0.0,
        rejected_topology_updates=0,
        final_learning_rate=0.03,
        lung_jacobian_minimum_runner=1.0,
        body_jacobian_minimum_runner=1.0,
        translation_self_check_max_error_mm=0.0,
        translation_self_check_jacobian_max_error=0.0,
        objective_components_final={},
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_frozen_v1_suite_truth_is_rejected_before_optimizer() -> None:
    suite = load_sliding_suite(
        REPO_ROOT / "configs/piecewise_svf_j12_development_suite_v1.json"
    )
    search = load_joint_svf_search(
        REPO_ROOT / "configs/piecewise_svf_j12_development_search_v1.json"
    )
    case = suite.cases[0]
    pair = make_sliding_phantom_pair(case.params)
    result = JointSVFResult(
        lung_displacement_mm=pair.phantom.lung_displacement_mm,
        body_displacement_mm=pair.phantom.body_displacement_mm,
        lung_velocity_mm=np.zeros_like(pair.phantom.lung_displacement_mm),
        body_velocity_mm=np.zeros_like(pair.phantom.body_displacement_mm),
        transform_direction=pair.transform_direction,
        runtime=runtime(),
    )

    surface = evaluate_advected_surface(
        pair,
        result,
        tolerance_mm=search.gate.advected_target_surface_p95_max_mm,
    )
    record = evaluate_j12_development(
        suite_version=suite.suite_version,
        case_id=case.case_id,
        search=search,
        weights=search.variants[0],
        pair=pair,
        result=result,
    )

    failures = validate_j12_suite_truth(suite, search)

    assert surface.coverage_min < 0.95
    assert surface.abs_distance_p95_max_mm > 0.75
    assert not record.gate_pass
    assert "advected_surface_p95" in record.gate_reasons
    assert "advected_surface_coverage" in record.gate_reasons
    assert set(failures) == {
        "dev_nominal_azimuthal",
        "dev_deep_longitudinal",
        "dev_shallow_low_texture",
    }


def test_preflight_writes_blocking_provenance_without_challenge(
    tmp_path: Path,
) -> None:
    suite_path = REPO_ROOT / "configs/piecewise_svf_j12_development_suite_v1.json"
    search_path = (
        REPO_ROOT / "configs/piecewise_svf_j12_development_search_v1.json"
    )
    suite = load_sliding_suite(suite_path)
    search = load_joint_svf_search(search_path)
    records = evaluate_j12_truth_preflight(suite, search)
    manifest_path = write_j12_truth_preflight(
        records,
        tmp_path,
        suite_path=suite_path,
        search_path=search_path,
        search=search,
        repo_root=REPO_ROOT,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["truth_pass_count"] == 0
    assert not manifest["all_truth_pass"]
    assert not manifest["optimizer_started"]
    assert not manifest["development_search_allowed"]
    assert not manifest["challenge"]["loaded"]
    assert manifest["summary_sha256"] == sha256(tmp_path / "summary.csv")

    with pytest.raises(ValueError, match="suite truth fails"):
        run_j12_development(
            suite,
            search,
            registration_python=tmp_path / "missing.exe",
            repo_root=REPO_ROOT,
            temporary_root=tmp_path,
        )


def test_zero_fields_fail_motion_development_gate() -> None:
    suite = load_sliding_suite(
        REPO_ROOT / "configs/piecewise_svf_j12_development_suite_v1.json"
    )
    search = load_joint_svf_search(
        REPO_ROOT / "configs/piecewise_svf_j12_development_search_v1.json"
    )
    case = suite.cases[0]
    pair = make_sliding_phantom_pair(case.params)
    zeros = np.zeros_like(pair.phantom.lung_displacement_mm)
    result = JointSVFResult(
        lung_displacement_mm=zeros,
        body_displacement_mm=zeros,
        lung_velocity_mm=zeros,
        body_velocity_mm=zeros,
        transform_direction=pair.transform_direction,
        runtime=runtime(),
    )

    record = evaluate_j12_development(
        suite_version=suite.suite_version,
        case_id=case.case_id,
        search=search,
        weights=search.variants[0],
        pair=pair,
        result=result,
    )

    assert not record.gate_pass
    assert "lung_field_p95" in record.gate_reasons
    assert "tangential_slip" in record.gate_reasons
