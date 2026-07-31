import hashlib
import json
from pathlib import Path

import pytest

from breathgeom.synthetic_j11 import (
    J11Config,
    J11Run,
    load_j11_config,
    run_j11_gate,
    write_j11_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/piecewise_svf_j11_representation_gate_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@pytest.fixture(scope="module")
def frozen_gate() -> tuple[J11Config, tuple[J11Run, ...]]:
    config = load_j11_config(CONFIG_PATH)
    return config, run_j11_gate(config)


def test_frozen_j11_config_has_positive_and_signed_negative_cases() -> None:
    config = load_j11_config(CONFIG_PATH)

    assert config.protocol_version == "piecewise-svf-j1.1-representation-v1"
    assert [case.role for case in config.cases] == [
        "positive",
        "positive",
        "negative_global_glued",
        "negative_posthoc_fixed_normal",
    ]
    assert config.interface.radii_mm == (22.0, 22.0, 28.0)
    assert config.thresholds.target_surface_abs_p95_max_mm == 0.05


def test_frozen_j11_gate_passes_positives_and_recognizes_controls(
    frozen_gate: tuple[J11Config, tuple[J11Run, ...]],
) -> None:
    _, runs = frozen_gate
    by_id = {run.case.case_id: run.record for run in runs}

    positives = [run.record for run in runs if run.case.role == "positive"]
    assert len(positives) == 2
    assert all(record.candidate_full_gate_pass for record in positives)
    assert all(record.expected_signature_pass for record in positives)
    assert all(record.topology_pass for record in positives)
    assert all(record.contact_pass for record in positives)

    glued = by_id["global_glued_control"]
    assert not glued.candidate_full_gate_pass
    assert glued.expected_signature_pass
    assert glued.topology_pass
    assert glued.contact_pass
    assert not glued.slip_pass
    assert "tangential_slip" in glued.candidate_gate_reasons

    posthoc = by_id["posthoc_fixed_normal_euler_control"]
    assert not posthoc.candidate_full_gate_pass
    assert posthoc.expected_signature_pass
    assert posthoc.topology_pass
    assert not posthoc.contact_pass
    assert (
        posthoc.fixed_normal_mismatch_p95_mm
        <= 0.01
    )
    assert posthoc.collision_fraction > 0.01


def test_j11_writer_hashes_all_case_artifacts(
    frozen_gate: tuple[J11Config, tuple[J11Run, ...]],
    tmp_path: Path,
) -> None:
    config, runs = frozen_gate
    manifest_path = write_j11_gate(
        runs,
        tmp_path,
        config=config,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["all_expected_signatures_pass"]
    assert manifest["positive_pass_count"] == 2
    assert manifest["negative_control_recognized_count"] == 2
    assert manifest["config"]["sha256"] == sha256(CONFIG_PATH)
    assert manifest["summary_sha256"] == sha256(tmp_path / "summary.csv")
    assert len(manifest["field_sha256"]) == 4
    assert len(manifest["record_sha256"]) == 4
