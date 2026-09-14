"""Regression checks for cross-level gating and S/B scale provenance."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


_PATH = Path(__file__).resolve().parents[1] / "tools" / "tepc_volume_convergence.py"
_SPEC = importlib.util.spec_from_file_location("tepc_volume_convergence", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
tepc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tepc)


def test_first_transition_failure_blocks_overall_convergence_pass():
    comparison = pd.DataFrame(
        {
            "transition": ["L12_to_L09", "L09_to_L06"],
            "passed": [False, True],
        }
    )
    volumes = pd.DataFrame(
        {
            "transition": ["L12_to_L09", "L09_to_L06"],
            "passed": [False, True],
        }
    )
    top = pd.DataFrame({"top_stable": [True], "fine_top_gap_resolved": [True]})
    pairs = pd.DataFrame({"acceptable_stable_or_unresolved": [True]})

    checks = tepc._convergence_checks(comparison, volumes, top, pairs)

    assert checks["all_final_metric_changes_within_tolerance"]
    assert checks["all_final_volume_changes_within_tolerance"]
    assert not checks["all_transition_metric_changes_within_tolerance"]
    assert not checks["all_transition_volume_changes_within_tolerance"]
    assert not all(checks.values())


def _result_frame(level, multiplier):
    rows = []
    for index, montage in enumerate(tepc.MONTAGES, 1):
        rows.append(
            {
                "level": level,
                "montage": montage,
                "state": "s0",
                "Z_ohm": 1.0,
                "S_soft": float(index) * multiplier,
                "S_heart": float(index) * multiplier,
                "S_lung": float(index) * multiplier,
                "B_soft": float(index) * multiplier,
                "B_heart": float(index) * multiplier,
                "B_lung": float(index) * multiplier,
                "Q_soft": 0.1 * index,
                "Q_heart": 0.1 * index,
                "Q_lung": 0.1 * index,
            }
        )
    return pd.DataFrame(rows)


def test_new_level_outliers_do_not_expand_l12_scale_floor(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    level09 = tmp_path / "level09"
    level06 = tmp_path / "level06"
    for path in (baseline / "analysis", level09, level06):
        path.mkdir(parents=True)
    (baseline / "analysis" / "summary.json").write_text("{}", encoding="utf-8")
    for path in (level09, level06):
        (path / "verification.json").write_text("{}", encoding="utf-8")
    selection = tmp_path / "state_selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "selected_before_new_mesh_results",
                "selected": [{"state": "s0"}],
            }
        ),
        encoding="utf-8",
    )

    baseline_frame = _result_frame("L12", 1.0)
    level_frames = {
        "level09": _result_frame("L09", 1000.0),
        "level06": _result_frame("L06", 1000.0),
    }
    monkeypatch.setattr(tepc, "_load_baseline_frame", lambda *args: baseline_frame)
    monkeypatch.setattr(
        tepc,
        "_load_level_frame",
        lambda path, deps: level_frames[path.name],
    )
    monkeypatch.setattr(
        tepc,
        "_verify_independent_mesh_builds",
        lambda baseline_path, levels: {"L09": {}, "L06": {}},
    )
    monkeypatch.setattr(
        tepc,
        "_mesh_volumes",
        lambda *args: {level: [1.0] * 5 for level in ("L12", "L09", "L06")},
    )

    output = tmp_path / "analysis"
    args = SimpleNamespace(
        baseline=baseline,
        selection=selection,
        level09=level09,
        level06=level06,
        deps=tmp_path / "deps",
        output=output,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        tepc.analyse(args)

    metrics = pd.read_csv(output / "metric_convergence.csv")
    soft_rows = metrics[(metrics["metric"] == "S") & (metrics["compartment"] == "soft")]
    assert set(soft_rows["global_scale"]) == {6.0}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    origin = summary["criteria"]["S_and_B_scale_origin"]
    assert origin["basis"] == "selected L12 rows only"
    assert origin["source_level"] == "L12"
    assert origin["scales"]["S_soft"] == 6.0
    assert origin["scales"]["B_soft"] == 6.0
