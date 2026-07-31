import hashlib
import json
from pathlib import Path

from breathgeom.measure.contact_svf_phantom import load_contact_svf_suite
from breathgeom.measure.joint_svf_registration import load_joint_svf_search
from breathgeom.synthetic_j12_contact import (
    evaluate_contact_truth_preflight,
    write_contact_truth_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = (
    REPO_ROOT / "configs/piecewise_svf_j12_contact_development_suite_v2.json"
)
SEARCH_PATH = (
    REPO_ROOT / "configs/piecewise_svf_j12_contact_development_search_v2.json"
)
FINE_SUITE_PATH = (
    REPO_ROOT / "configs/piecewise_svf_j12_contact_development_suite_v3.json"
)
FINE_SEARCH_PATH = (
    REPO_ROOT / "configs/piecewise_svf_j12_contact_development_search_v3.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_frozen_v2_contact_truth_is_analytic_but_raster_ineligible() -> None:
    suite = load_contact_svf_suite(SUITE_PATH)
    search = load_joint_svf_search(SEARCH_PATH)

    runs = evaluate_contact_truth_preflight(suite, search)

    assert len(runs) == 3
    assert not any(run.record.truth_gate_pass for run in runs)
    assert all(run.record.analytic_abs_distance_p95_max_mm <= 0.05 for run in runs)
    assert all(run.record.analytic_surface_coverage_min >= 0.99 for run in runs)
    assert all(run.record.raster_abs_distance_p95_max_mm > 0.75 for run in runs)
    assert all(run.record.raster_surface_coverage_min < 0.95 for run in runs)
    assert all(
        run.record.gate_reasons
        == ("raster_surface_p95", "raster_surface_coverage")
        for run in runs
    )


def test_contact_truth_writer_hashes_summary_and_diagnostics(tmp_path: Path) -> None:
    suite = load_contact_svf_suite(SUITE_PATH)
    search = load_joint_svf_search(SEARCH_PATH)
    runs = evaluate_contact_truth_preflight(suite, search)

    manifest_path = write_contact_truth_preflight(
        runs,
        tmp_path,
        suite_path=SUITE_PATH,
        search_path=SEARCH_PATH,
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
    assert len(manifest["diagnostic_sha256"]) == 3
    for name, expected in manifest["diagnostic_sha256"].items():
        assert sha256(tmp_path / name) == expected


def test_frozen_v3_fine_grid_truth_passes_without_gate_changes() -> None:
    suite = load_contact_svf_suite(FINE_SUITE_PATH)
    search = load_joint_svf_search(FINE_SEARCH_PATH)

    runs = evaluate_contact_truth_preflight(suite, search)

    assert len(runs) == 3
    assert all(run.record.truth_gate_pass for run in runs)
    assert all(run.record.analytic_abs_distance_p95_max_mm <= 0.05 for run in runs)
    assert all(run.record.analytic_surface_coverage_min >= 0.99 for run in runs)
    assert all(run.record.raster_abs_distance_p95_max_mm <= 0.75 for run in runs)
    assert all(run.record.raster_surface_coverage_min >= 0.95 for run in runs)
    assert all(not run.record.gate_reasons for run in runs)
