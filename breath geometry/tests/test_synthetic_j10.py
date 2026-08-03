import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from breathgeom.synthetic_j10 import (
    load_j10_config,
    run_j10_gate,
    write_j10_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/piecewise_svf_j10_numeric_gate_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_frozen_j10_config_has_all_analytic_invariants() -> None:
    config = load_j10_config(CONFIG_PATH)

    assert config.protocol_version == "piecewise-svf-j1.0-numeric-v1"
    assert config.transform_direction == "fixed-expiration_to_moving-inspiration"
    assert [case.kind for case in config.cases] == [
        "zero",
        "planar_sliding_translation",
        "diagonal_affine",
        "opposed_axial_rotation",
    ]
    assert config.evaluation_margin_voxels == (8, 8, 8)


def test_frozen_j10_gate_preserves_analytic_fields_and_sliding() -> None:
    config = load_j10_config(CONFIG_PATH)

    runs = run_j10_gate(config)

    records = [record for run in runs for record in run.records]
    assert len(records) == 8
    assert all(record.gate_pass for record in records)
    sliding = next(
        record
        for record in records
        if record.case_id == "planar_sliding_translation"
        and record.region == "lung"
    )
    assert sliding.normal_mismatch_p95_mm == pytest.approx(0.0, abs=1e-6)
    assert sliding.tangential_slip_truth_mm == pytest.approx(4.0)
    assert sliding.tangential_slip_observed_mm == pytest.approx(4.0, abs=1e-6)
    assert all(record.nonpositive_jacobian_fraction == 0.0 for record in records)


def test_writer_records_hashes_and_rejects_tight_posthoc_gate(tmp_path: Path) -> None:
    config = load_j10_config(CONFIG_PATH)
    runs = run_j10_gate(config)
    manifest_path = write_j10_gate(
        runs,
        tmp_path,
        config=config,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["all_pass"]
    assert manifest["case_pass_count"] == 4
    assert manifest["region_pass_count"] == 8
    assert manifest["config"]["sha256"] == sha256(CONFIG_PATH)
    assert manifest["summary_sha256"] == sha256(tmp_path / "summary.csv")
    assert set(manifest["field_sha256"]) == {
        "zero.npz",
        "planar_sliding_translation.npz",
        "diagonal_affine.npz",
        "opposed_axial_rotation.npz",
    }

    impossible = replace(
        config,
        thresholds=replace(
            config.thresholds,
            endpoint_p95_max_mm=1e-12,
        ),
    )
    failed = [
        record
        for run in run_j10_gate(impossible)
        for record in run.records
        if not record.gate_pass
    ]
    assert failed
    assert any("endpoint_p95" in record.gate_reasons for record in failed)
