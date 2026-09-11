"""Contract tests for the restartable inverse driver using a mock MATLAB runner."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_restartable_inverse_search.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("restartable_inverse_search", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def _point() -> list[float]:
    return [5.0, 20.0, 1.0, 2.0, 3.0]


def _record(status: str = "evaluated", failure_code: str = "") -> dict:
    return driver._make_record(
        evaluation=1,
        point=_point(),
        objective=4.0 if status == "evaluated" else driver.INVALID_GEOMETRY_OBJECTIVE,
        slope=-0.5,
        h_centre=15.0,
        status=status,
        failure_code=failure_code,
    )


def _configure_tmp_driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(driver, "OUT", tmp_path)
    monkeypatch.setattr(driver, "ROOT", tmp_path)


def test_resume_rejects_changed_data_before_history_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)
    history_file = tmp_path / "history.csv"
    summary_file = tmp_path / "summary.json"
    identity_file = tmp_path / "identity.json"
    identity = driver.build_run_identity(
        tag="auditfix_20260908",
        mode="disjoint",
        data_selection="manifest",
        root=tmp_path,
    )
    driver.write_history(history_file, [_record()])
    identity_file.write_text(json.dumps(identity), encoding="utf-8")

    data_file = tmp_path / "data" / "nik" / "experimental" / "nik_breath_hold_levels.csv"
    data_file.parent.mkdir(parents=True)
    data_file.write_text("changed\n", encoding="utf-8")

    with pytest.raises(driver.ResumeIdentityError):
        driver.load_resume_state(
            history_file,
            summary_file,
            identity_file,
            tag="auditfix_20260908",
            mode="disjoint",
            data_selection="manifest",
            root=tmp_path,
        )


def test_arbitrary_matlab_failure_is_technical_and_never_a_penalty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)

    def runner(command, **kwargs):
        kwargs["stdout"].write("license server unavailable\n")
        return SimpleNamespace(returncode=17)

    with pytest.raises(driver.TechnicalEvaluationError) as error:
        driver.evaluate_points(
            [_point()],
            tag="technical_failure",
            mode="disjoint",
            data_selection="manifest",
            first_evaluation=1,
            runner=runner,
        )
    assert not (tmp_path / "history.csv").exists()
    assert "No physical penalty" in str(error.value)



def test_preexisting_batch_results_are_deleted_before_successful_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)
    tag = "stale_results"
    results_file = tmp_path / f".restartable_{tag}_001_results.csv"
    with results_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "batch_row",
                *driver.PARAMETER_NAMES,
                "objective",
                "fem_slope_ohm_per_mm",
                "h_centre_mm",
                "status",
                "failure_code",
                "version",
                "fingerprint",
            ]
        )
        writer.writerow(
            [
                "1",
                *_point(),
                "1.0",
                "-0.5",
                "15.0",
                "evaluated",
                "",
                "stale-version",
                "stale-fingerprint",
            ]
        )

    def noop_runner(command, **kwargs):
        return SimpleNamespace(returncode=0)

    history_file = tmp_path / "history.csv"
    with pytest.raises(driver.TechnicalEvaluationError, match="fresh"):
        driver.evaluate_points(
            [_point()],
            tag=tag,
            mode="disjoint",
            data_selection="manifest",
            first_evaluation=1,
            checkpoint_file=history_file,
            runner=noop_runner,
        )
    assert not results_file.exists()
    assert not history_file.exists()


def test_known_geometry_failure_is_recorded_but_cannot_be_finalised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command[-1])
        kwargs["stdout"].write("Error: trkg4:electrodeDiagnostics\n")
        return SimpleNamespace(returncode=1)

    rows = driver.evaluate_points(
        [_point()],
        tag="geometry_failure",
        mode="disjoint",
        data_selection="manifest",
        first_evaluation=1,
        runner=runner,
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "invalid_geometry"
    assert rows[0]["failure_code"] == "trkg4:electrodeDiagnostics"
    assert rows[0]["objective"] == driver.INVALID_GEOMETRY_OBJECTIVE
    with pytest.raises(driver.NoValidEvaluationError):
        driver.best_evaluated(rows)
    with pytest.raises(driver.NoValidEvaluationError):
        driver.finalise_result(
            rows[0],
            tag="geometry_failure",
            mode="disjoint",
            data_selection="manifest",
            runner=lambda *args, **kwargs: pytest.fail("finalise was called"),
        )
    assert len(calls) == 1



def test_geometry_allowlist_is_exact() -> None:
    assert driver.KNOWN_GEOMETRY_ERROR_CODES == {
        "trkg4:electrodeDiagnostics",
        "trkg4:patchOverlap",
        "electrode_faces_by_area:noSeed",
        "electrode_faces_by_area:patchTooSmall",
        "electrode_faces_by_area_legacy:patchTooSmall",
        "electrode_faces_by_area_legacy_fast:patchTooSmall",
    }
    assert not driver._is_known_geometry_code("trkg4:electrodeDiagnosticsInput")
    assert not driver._is_known_geometry_code("electrode_faces_by_area:emptyBoundary")


@pytest.mark.parametrize(
    "message",
    [
        "Error: trkg4:electrodeDiagnosticsInput\n",
        "Error: electrode_faces_by_area:emptyBoundary\n",
    ],
)
def test_unallowlisted_geometry_like_failures_are_technical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)

    def runner(command, **kwargs):
        kwargs["stdout"].write(message)
        return SimpleNamespace(returncode=1)

    with pytest.raises(driver.TechnicalEvaluationError):
        driver.evaluate_points(
            [_point()],
            tag="geometry_error_boundary",
            mode="disjoint",
            data_selection="manifest",
            first_evaluation=1,
            runner=runner,
        )



def test_initial_points_seed_is_used_and_hashed_in_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)
    seed_file = tmp_path / "admissible_starts.csv"
    seed_file.write_text(
        "5.0,20.0,1.0,2.0,3.0\n"
        "5.5,21.0,4.0,5.0,6.0\n",
        encoding="utf-8",
    )

    points = driver.initial_points(seed_file)
    assert points == [
        [5.0, 20.0, 1.0, 2.0, 3.0],
        [5.5, 21.0, 4.0, 5.0, 6.0],
    ]
    identity_before = driver.build_run_identity(
        tag="auditfix_20260908",
        mode="disjoint",
        data_selection="manifest",
        root=tmp_path,
        initial_points_path=seed_file,
    )
    assert identity_before["seed"]["path"] == "admissible_starts.csv"
    assert identity_before["seed"]["sha256"]

    seed_file.write_text(
        "5.0,20.0,1.0,2.0,3.0\n"
        "5.5,21.0,4.0,5.0,7.0\n",
        encoding="utf-8",
    )
    identity_after = driver.build_run_identity(
        tag="auditfix_20260908",
        mode="disjoint",
        data_selection="manifest",
        root=tmp_path,
        initial_points_path=seed_file,
    )
    assert identity_after["fingerprint"] != identity_before["fingerprint"]


def test_batch_contract_requires_explicit_status_and_engine_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)

    def runner(command, **kwargs):
        point_file = next(tmp_path.glob(".restartable_batch_contract_*_points.csv"))
        result_file = tmp_path / point_file.name.replace("_points.csv", "_results.csv")
        with point_file.open(newline="", encoding="utf-8") as stream:
            point = next(csv.reader(stream))
        fields = [
            "batch_row",
            *driver.PARAMETER_NAMES,
            "objective",
            "fem_slope_ohm_per_mm",
            "h_centre_mm",
        ]
        with result_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerow(["1", *point, "4", "-0.5", "15"])
        return SimpleNamespace(returncode=0)

    with pytest.raises(driver.EvaluationContractError):
        driver.evaluate_points(
            [_point()],
            tag="batch_contract",
            mode="disjoint",
            data_selection="manifest",
            first_evaluation=1,
            runner=runner,
        )


def test_successful_finalise_uses_jacobian_and_hashes_search_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tmp_driver(monkeypatch, tmp_path)
    tag = "auditfix_20260908"
    point = _point()
    best = _record()
    expressions: list[str] = []

    def runner(command, **kwargs):
        expressions.append(command[-1])
        for path in driver._final_artifact_paths(tag, tmp_path).values():
            path.write_text("fresh artifact\n", encoding="utf-8")
        contract = {
            "result_tag": tag,
            "patch_builder_mode": "disjoint",
            "data_selection": "manifest",
            "compute_jacobian": True,
            "version": "main-v2",
            "fingerprint": "engine-fingerprint",
            "parameters": dict(zip(driver.PARAMETER_NAMES, point)),
            "inputs": [
                {"name": "nik_breath_hold_levels.csv", "path": "", "sha256": "input"},
            ],
            "artifacts": [
                {
                    "name": path.name,
                    "path": driver._root_relative_path(path, tmp_path),
                    "sha256": driver._sha256_file(path),
                }
                for path in driver._final_artifact_paths(tag, tmp_path).values()
            ],
        }
        (tmp_path / f"nik_trkg4_inverse_inhale_contract_{tag}.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        (tmp_path / f"nik_trkg4_inverse_inhale_jacobian_diagnostics_{tag}.json").write_text(
            json.dumps({"condition": 1.0, "semantic_status": "computed"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    contract = driver.finalise_result(
        best,
        tag=tag,
        mode="disjoint",
        data_selection="manifest",
        runner=runner,
    )

    assert ",true," in expressions[0]
    assert isinstance(contract["artifacts"], list)
    assert isinstance(contract["inputs"], list)

    history_file = tmp_path / f"nik_trkg4_inverse_direct_search_history_{tag}.csv"
    summary_file = tmp_path / f"nik_trkg4_inverse_direct_search_summary_{tag}.json"
    history_file.write_text("history\n", encoding="utf-8")
    summary_file.write_text('{"status":"complete"}\n', encoding="utf-8")
    contract_file = tmp_path / f"nik_trkg4_inverse_inhale_contract_{tag}.json"
    original_artifact_count = len(contract["artifacts"])
    updated = driver._attach_driver_artifacts(
        contract_file,
        contract,
        {
            "inverse_direct_search_history": history_file,
            "inverse_direct_search_summary": summary_file,
        },
    )
    driver._validate_driver_artifacts(updated, root=tmp_path)
    assert len(updated["artifacts"]) == original_artifact_count + 2
    saved = json.loads(contract_file.read_text(encoding="utf-8"))
    saved_artifacts = {
        record["name"]: record for record in saved["artifacts"]
    }
    assert saved_artifacts[history_file.name]["sha256"]
    assert saved_artifacts[summary_file.name]["sha256"]
    assert saved_artifacts[history_file.name]["path"] == history_file.name
    assert saved_artifacts[summary_file.name]["path"] == summary_file.name


# These fixtures patch ROOT/defaults only in tests. SCRIPT remains the ordinary
# production import path above; the candidate is tested in a temporary layout.
import hashlib
import io
import math
import re
import sys
from contextlib import redirect_stdout


class SearchMock:
    def __init__(self, monkeypatch, root, source=SCRIPT, *, mode="disjoint", seeds=1):
        self.mp, self.root, self.mode = monkeypatch, root, mode
        root.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location("isolated_search", source)
        self.d = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.d)
        self.mp.setattr(self.d, "ROOT", root)
        self.mp.setattr(self.d, "OUT", root / "output")
        for name in ("_resolve_initial_points_path", "build_run_identity", "_validate_identity",
                     "load_resume_state", "_restore_search_state"):
            if hasattr(self.d, name):
                fn = getattr(self.d, name)
                if "root" in (fn.__kwdefaults__ or {}):
                    self.mp.setattr(fn, "__kwdefaults__", {**fn.__kwdefaults__, "root": root})
        for group in self.d._identity_file_groups(root).values():
            for path in group:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text("fixed mock\n", encoding="utf-8")
        (root / "tools/run_restartable_inverse_search.py").write_bytes(source.read_bytes())
        self.seed = root / "admissible_starts.csv"
        self.seed.write_text("".join(
            ",".join(map(str, [5, 20, 1 + i, 2, 3])) + "\n" for i in range(seeds)
        ), encoding="utf-8")
        self.tag, self.calls, self.finals = "mock_edges", [], []
        self.invalid = False
        self.invalid_metrics = (math.nan, math.nan)
        self.history = self.d.OUT / f"nik_trkg4_inverse_direct_search_history_{self.tag}.csv"
        self.summary = self.d.OUT / f"nik_trkg4_inverse_direct_search_summary_{self.tag}.json"
        self.identity = self.d._identity_path(self.tag)
        self.contract = self.d._contract_path(self.tag)

    def state(self):
        return json.loads(self.d._search_state_path(self.tag).read_text(encoding="utf-8"))

    def rows(self):
        return self.d.load_history(self.history)

    def resume(self):
        return self.d.load_resume_state(
            self.history, self.summary, self.identity, tag=self.tag, mode=self.mode,
            data_selection="manifest", initial_points_path=self.seed,
        )

    def run(self, budget, *, no_resume=False, stop_after=None):
        accepted = 0
        d = self.d

        def runner(command, **kwargs):
            nonlocal accepted
            assert Path(kwargs["cwd"]) == self.root
            expression = command[-1]
            if "run_trkg4_inverse_evaluation_batch(" in expression:
                match = re.search(
                    r"run_trkg4_inverse_evaluation_batch\('([^']*)','([^']*)',", expression
                )
                points_file, results_file = map(Path, match.groups())
                assert points_file.is_relative_to(self.root)
                assert results_file.is_relative_to(self.root)
                with points_file.open(newline="", encoding="utf-8") as stream:
                    points = [[float(x) for x in row] for row in csv.reader(stream)]
                self.next_point = points[0]
                if stop_after is not None and accepted >= stop_after:
                    raise KeyboardInterrupt("mock interruption")
                with results_file.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([
                        "batch_row", *d.PARAMETER_NAMES, "objective",
                        "fem_slope_ohm_per_mm", "h_centre_mm",
                        "status", "failure_code", "version", "fingerprint",
                    ])
                    for i, point in enumerate(points, 1):
                        objective = 0.5 if point == [5, 20, -0.25, 2, 3] else 1.0
                        writer.writerow([
                            i, *point, objective,
                            *(self.invalid_metrics if self.invalid else (-0.5, 15)),
                            "invalid_geometry" if self.invalid else "evaluated",
                            "trkg4:electrodeDiagnostics" if self.invalid else "",
                            "mock-v1", "mock-fingerprint",
                        ])
                self.calls.extend(points)
                accepted += len(points)
            else:
                assert "],true," in expression
                point = [float(x) for x in re.search(
                    r"run_trkg4_inverse_inhale\(0,\[([^]]+)\]", expression
                ).group(1).split()]
                assert not self.invalid
                self.finals.append(point)
                artifacts = list(d._final_artifact_paths(self.tag).values())
                for path in artifacts:
                    path.write_text(f"mock final {len(self.finals)}\n", encoding="utf-8")
                diagnostics = d._diagnostics_path(self.tag)
                diagnostics.write_text(json.dumps({
                    "condition": 1.0, "generation": len(self.finals),
                }), encoding="utf-8")
                artifacts.append(diagnostics)
                record = lambda path: {
                    "name": path.name, "path": path.relative_to(self.root).as_posix(),
                    "sha256": d._sha256_file(path),
                }
                self.contract.write_text(json.dumps({
                    "schema_version": 1, "model_revision": "auditfix_20260908",
                    "result_tag": self.tag, "parameters": dict(zip(d.PARAMETER_NAMES, point)),
                    "data_selection": "manifest", "patch_builder_mode": self.mode,
                    "compute_jacobian": True, "geometry_qc_passed": True,
                    "status": "numerical_candidate_not_validated",
                    "version": "mock-v1", "fingerprint": "mock-fingerprint",
                    "artifacts": list(map(record, artifacts)), "inputs": [record(self.seed)],
                }), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        argv = ["driver", "--mode", self.mode, "--tag", self.tag, "--budget", str(budget),
                "--initial-points", str(self.seed)]
        if no_resume:
            argv.append("--no-resume")
        with self.mp.context() as mp, redirect_stdout(io.StringIO()):
            mp.setattr(sys, "argv", argv)
            mp.setattr(d.subprocess, "run", runner)
            d.main()


def _trajectory(search):
    return [(search.d.record_point(row), row["objective"]) for row in search.rows()]


def test_completed_extension_restarts_after_new_rows(monkeypatch, tmp_path):
    full = SearchMock(monkeypatch, tmp_path / "full")
    full.run(5)
    full.run(20)
    cut = SearchMock(monkeypatch, tmp_path / "cut")
    cut.run(5)
    with pytest.raises(KeyboardInterrupt):
        cut.run(20, stop_after=3)
    assert len(cut.rows()) == 8
    assert not cut.contract.exists()
    assert json.loads(cut.summary.read_text())["finalization"]["status"] == "running"
    cut.run(20)
    assert _trajectory(cut) == _trajectory(full)
    assert cut.finals == full.finals
    assert cut.state()["phase"] == "complete"
    assert len(cut.resume()[0]) == 20


@pytest.mark.parametrize("stop", [0, 2])
def test_no_resume_replaces_completed_state_before_evaluations(monkeypatch, tmp_path, stop):
    fresh = SearchMock(monkeypatch, tmp_path / "fresh")
    fresh.run(20)
    cut = SearchMock(monkeypatch, tmp_path / "cut")
    cut.run(5)
    old_generation = cut.state()["run_id"]
    with pytest.raises(KeyboardInterrupt):
        cut.run(20, no_resume=True, stop_after=stop)
    assert len(cut.rows()) == stop
    assert cut.state()["run_id"] != old_generation
    assert not cut.contract.exists()
    cut.run(20)
    assert _trajectory(cut) == _trajectory(fresh)


@pytest.mark.parametrize("stop", [0, 8, 21, 36, 41])
def test_step_and_partial_pass_preserve_trajectory(monkeypatch, tmp_path, stop):
    full = SearchMock(monkeypatch, tmp_path / "full")
    full.run(50)
    cut = SearchMock(monkeypatch, tmp_path / "cut")
    with pytest.raises(KeyboardInterrupt):
        cut.run(50, stop_after=stop)
    if stop in (36, 41):
        assert cut.resume()[2] == [v / 8 for v in cut.d.INITIAL_STEP]
    if stop == 36:
        assert cut.state()["pending"]["next"] == 5  # improved best, unfinished sweep
    cut.run(50)
    assert _trajectory(cut) == _trajectory(full)
    assert cut.calls == full.calls
    assert cut.finals == full.finals
    assert cut.state()["step"] == full.state()["step"]


@pytest.mark.parametrize("window", ["before_commit", "after_commit", "before_csv", "after_csv"])
def test_interruption_between_checkpoint_and_csv_is_recoverable(monkeypatch, tmp_path, window):
    full = SearchMock(monkeypatch, tmp_path / "full")
    full.run(20)
    cut = SearchMock(monkeypatch, tmp_path / "cut")
    atomic = cut.d._atomic_bytes
    tripped = False

    def interrupted_write(path, content):
        nonlocal tripped
        checkpoint = path == cut.d._search_state_path(cut.tag)
        target_commit = checkpoint and len(json.loads(content)["history"]) == 1
        target_csv = path == cut.history and len(content.splitlines()) == 2
        target = target_commit if "commit" in window else target_csv
        if target and not tripped:
            tripped = True
            if window.startswith("after"):
                atomic(path, content)
            raise KeyboardInterrupt(window)
        return atomic(path, content)

    with monkeypatch.context() as mp:
        mp.setattr(cut.d, "_atomic_bytes", interrupted_write)
        with pytest.raises(KeyboardInterrupt):
            cut.run(20)
    assert tripped
    cut.run(20)
    assert _trajectory(cut) == _trajectory(full)
    assert cut.finals == full.finals


@pytest.mark.parametrize("mutation", ["history", "identity", "checkpoint", "data", "code", "seed"])
def test_checkpoint_never_silently_accepts_unrelated_state(monkeypatch, tmp_path, mutation):
    search = SearchMock(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        search.run(20, stop_after=3)
    target = {
        "history": search.history, "identity": search.identity,
        "checkpoint": search.d._search_state_path(search.tag),
        "data": tmp_path / "data/nik/experimental/nik_breath_hold_levels.csv",
        "code": tmp_path / "tools/run_restartable_inverse_search.py", "seed": search.seed,
    }[mutation]
    if mutation == "checkpoint":
        state = search.state()
        state["step"][0] *= 2  # valid JSON but no matching checksum
        target.write_text(json.dumps(state), encoding="utf-8")
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    before = search.history.read_bytes()
    with pytest.raises(search.d.ResumeIdentityError):
        search.run(20)
    assert search.history.read_bytes() == before


def test_completed_publication_guards_and_artifact_list(monkeypatch, tmp_path):
    search = SearchMock(monkeypatch, tmp_path)
    search.run(5)
    contract = json.loads(search.contract.read_text())
    assert isinstance(contract["artifacts"], list) and isinstance(contract["inputs"], list)
    assert {search.history.name, search.summary.name}.issubset(
        record["name"] for record in contract["artifacts"]
    )
    search.d._validate_driver_artifacts(contract, root=tmp_path)
    search.contract.write_bytes(search.contract.read_bytes() + b"\n")
    with pytest.raises(search.d.ResumeIdentityError, match="publication"):
        search.run(20)


def test_invalid_only_search_never_finalises_checkpoint(monkeypatch, tmp_path):
    search = SearchMock(monkeypatch, tmp_path)
    search.invalid = True
    with pytest.raises(search.d.NoValidEvaluationError):
        search.run(5)
    assert not search.finals and not search.contract.exists()
    assert all(row["status"] == "invalid_geometry" for row in search.rows())
    with pytest.raises(search.d.NoValidEvaluationError):
        search.run(20)


# Reference hashes obtained from the manifest-pinned pre-fix driver, same mock J.
@pytest.mark.parametrize("mode", ["disjoint", "legacy_fast"])
@pytest.mark.parametrize("seeds,budget,expected", [
    (1, 20, "c253990ffbe03314571733c4367e57cac320ac1b1aff9d7f346af9cc8fefa9ae"),
    (1, 100, "a8542430b00ef89ecfc27e68d5ab280464204560a92adb751f256d76d4ed55a1"),
    (3, 20, "6ca2a59b06152f270ea3670de7bd8fcc0b19ee313eaa4986f8d694f76c80add7"),
    (3, 100, "f7eb90f2a2a60c3e8a0004e78ec35913fb9de6f17363fb285b4d69843c02cf84"),
])
def test_uninterrupted_trajectory_matches_pre_fix_driver(monkeypatch, tmp_path, mode, seeds, budget, expected):
    search = SearchMock(monkeypatch, tmp_path, mode=mode, seeds=seeds)
    search.run(budget)
    digest = hashlib.sha256(json.dumps(_trajectory(search), separators=(",", ":")).encode()).hexdigest()
    assert digest == expected
    assert search.finals[-1] == search.d.record_point(search.d.best_evaluated(search.rows()))


def test_legacy_batch_partial_pass_resumes_without_recentering(monkeypatch, tmp_path):
    full = SearchMock(monkeypatch, tmp_path / "full", mode="legacy_fast", seeds=3)
    full.run(50)
    cut = SearchMock(monkeypatch, tmp_path / "cut", mode="legacy_fast", seeds=3)
    with pytest.raises(KeyboardInterrupt):
        cut.run(50, stop_after=38)
    assert cut.state()["pending"]["next"] == 5
    cut.run(50)
    assert cut.calls == full.calls
    assert _trajectory(cut) == _trajectory(full)


@pytest.mark.parametrize("no_resume", [False, True])
def test_transition_commit_recovers_before_old_outputs_are_withdrawn(monkeypatch, tmp_path, no_resume):
    search = SearchMock(monkeypatch, tmp_path)
    search.run(5)
    atomic = search.d._atomic_bytes

    def stop_after_transition(path, content):
        atomic(path, content)
        if path == search.d._search_state_path(search.tag) and json.loads(content)["phase"] == "running":
            raise KeyboardInterrupt("transition committed; projection not started")

    with monkeypatch.context() as mp:
        mp.setattr(search.d, "_atomic_bytes", stop_after_transition)
        with pytest.raises(KeyboardInterrupt):
            search.run(20, no_resume=no_resume)
    assert search.contract.exists()  # still the old coherent publication
    search.run(20)
    assert len(search.resume()[0]) == 20


def test_interrupted_final_checkpoint_republishes_only_the_valid_best(monkeypatch, tmp_path):
    search = SearchMock(monkeypatch, tmp_path)
    atomic = search.d._atomic_bytes

    def stop_before_complete_commit(path, content):
        if path == search.d._search_state_path(search.tag) and json.loads(content)["phase"] == "complete":
            raise KeyboardInterrupt("publication written; complete checkpoint not committed")
        atomic(path, content)

    with monkeypatch.context() as mp:
        mp.setattr(search.d, "_atomic_bytes", stop_before_complete_commit)
        with pytest.raises(KeyboardInterrupt):
            search.run(20)
    accepted = list(search.calls)
    assert search.state()["phase"] == "running"
    search.run(20)
    assert search.calls == accepted
    assert len(search.finals) == 2 and search.finals[0] == search.finals[1]
    assert search.state()["phase"] == "complete"
    assert len(search.resume()[0]) == 20


# The checkpoint must be consumable by a strict JSON parser, including refusal
# rows; its wire tokens must never leak into the public numeric CSV contract.
def _strict_state(search):
    def reject_constant(value):
        raise ValueError(f"Forbidden JSON constant: {value}")
    return json.loads(search.d._search_state_path(search.tag).read_text(encoding="utf-8"),
                      parse_constant=reject_constant)


def test_seed_checkpoint_is_strict_json_before_first_evaluation(monkeypatch, tmp_path):
    search = SearchMock(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        search.run(5, stop_after=0)
    wire = _strict_state(search)
    assert wire["pending"]["before"] == "inf"
    assert wire["pending"]["next"] == 0 and wire["history"] == []
    history, identity, step = search.resume()
    assert history == [] and identity == wire["identity"]
    assert step == search.d.INITIAL_STEP and search.calls == []
    search.run(5)
    assert search.calls[0] == [5, 20, 1, 2, 3]
    assert _strict_state(search)["phase"] == "complete"


@pytest.mark.parametrize("metrics", [(math.nan, math.nan), (math.inf, -math.inf)],
                         ids=["nan", "signed_infinities"])
def test_invalid_geometry_checkpoint_is_strict_json_and_resumes(monkeypatch, tmp_path, metrics):
    search = SearchMock(monkeypatch, tmp_path, seeds=3)
    search.invalid, search.invalid_metrics = True, metrics
    with pytest.raises(KeyboardInterrupt):
        search.run(5, stop_after=1)
    wire = _strict_state(search)
    fields = ("fem_slope_ohm_per_mm", "h_centre_mm")
    assert wire["pending"]["before"] == "inf" and wire["pending"]["next"] == 1
    assert [wire["history"][0][field] for field in fields] == list(map(str, metrics))
    before_csv = search.history.read_bytes()
    history, identity, _ = search.resume()
    assert identity == wire["identity"]
    assert history[0]["status"] == "invalid_geometry"
    assert history[0]["failure_code"] == "trkg4:electrodeDiagnostics"
    assert history[0]["objective"] == search.d.INVALID_GEOMETRY_OBJECTIVE
    for field, expected in zip(fields, metrics):
        actual = history[0][field]
        assert isinstance(actual, float)
        assert math.isnan(actual) if math.isnan(expected) else actual == expected
    assert search.history.read_bytes() == before_csv == search.d._history_bytes(history)
    with search.history.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
        assert [row[field] for field in fields] == list(map(str, metrics))
    search.invalid = False
    search.run(5)
    final_wire = _strict_state(search)
    assert final_wire["identity"] == identity
    assert [row["status"] for row in final_wire["history"]] == ["invalid_geometry"] + ["evaluated"] * 4
    assert len(search.calls) == 5  # committed refusal is not silently retried
    assert search.finals == [search.d.record_point(search.d.best_evaluated(search.rows()))]
    assert search.finals[0] != search.calls[0]
    assert search.resume()[1] == identity  # includes contract/artifact checks


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_resume_rejects_nonstandard_json_constants_before_any_writes(monkeypatch, tmp_path, literal):
    search = SearchMock(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        search.run(5, stop_after=0)
    path = search.d._search_state_path(search.tag)
    text = path.read_text(encoding="utf-8")
    assert '\"before\": \"inf\"' in text
    path.write_text(text.replace('\"before\": \"inf\"', '\"before\": ' + literal), encoding="utf-8")
    before = {p: p.read_bytes() for p in (path, search.history, search.identity, search.summary)}
    with pytest.raises(search.d.ResumeIdentityError, match="strict JSON"):
        search.resume()
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("objective", [math.nan, math.inf, -math.inf])
def test_encoded_nonfinite_evaluated_objective_is_still_rejected(monkeypatch, tmp_path, objective):
    search = SearchMock(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        search.run(5, stop_after=1)
    state = search.d._restore_search_state(
        search.history, search.summary, search.identity, tag=search.tag, mode=search.mode,
        data_selection="manifest", initial_points_path=search.seed,
    )
    state["history"][0]["objective"] = objective
    # Supply even a matching checksum and projection: semantic validation must
    # still reject a non-finite objective with status evaluated.
    search.d._save_search_state(state, search.history, search.identity, search.summary)
    assert _strict_state(search)["history"][0]["objective"] == str(objective)
    with pytest.raises(search.d.ResumeIdentityError, match="non-finite objective"):
        search.resume()


def test_state_writer_rejects_nonfinite_outside_explicit_numeric_fields(monkeypatch, tmp_path):
    search = SearchMock(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        search.run(5, stop_after=0)
    state = _strict_state(search)
    state["unexpected_nonfinite"] = math.nan
    path = search.d._search_state_path(search.tag)
    before = {p: p.read_bytes() for p in (path, search.history, search.identity, search.summary)}
    with pytest.raises(ValueError, match="Out of range float"):
        search.d._save_search_state(state, search.history, search.identity, search.summary)
    assert before == {p: p.read_bytes() for p in before}
