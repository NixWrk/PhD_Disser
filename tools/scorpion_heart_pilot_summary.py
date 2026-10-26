"""Summarize frozen heart pilots, normalizing MATLAB singleton structs in memory.

This reader never writes plans/reports or changes the frozen producer/worker.
Only schema scorpion_heart_forward_v1 with scalar ``answers`` is normalized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MATLAB_TRKG4_real_subjects/tools"))
import scorpion_fem_heart_pilot as producer

SCHEMA = "scorpion_heart_forward_v1"
GEOMETRIES = ("wholeheart_ct", "sphere_v095", "sphere_v100", "sphere_v105")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(value):
    """Return a shallow copy only for the explicitly supported singleton shape."""
    if isinstance(value, dict) and value.get("schema") == SCHEMA and isinstance(value.get("answers"), dict):
        return dict(value, answers=[value["answers"]])
    return value


@contextmanager
def normalized_reader(normalized_paths):
    # The frozen implementation remains the one source for summary arithmetic.
    # The CLI is single-process; always restore its reader, also after failure.
    original = producer.read
    def read(path):
        raw = original(path)
        answer = normalize(raw)
        if answer is not raw:
            normalized_paths.append(str(Path(path).resolve()))
        return answer
    producer.read = read
    try:
        yield
    finally:
        producer.read = original


def require_matrix(value, shape, name):
    array = producer.np.asarray(value, dtype=float)
    require(array.shape == shape and producer.np.isfinite(array).all(), f"Invalid {name}")


def summarize(directory):
    directory = Path(directory).resolve()
    plan_path = directory / "plan.local.json"
    plan = producer.read(plan_path)
    require(plan.get("schema") == "scorpion_heart_pilot_plan_v1", "Wrong plan schema")
    expected = {f"{g}_q{q}" for g in GEOMETRIES for q in (1024, 4096)}
    variants = plan["variants"]
    require(len(variants) == 8 and {v["id"] for v in variants} == expected, "Expected eight unique material variants")
    require(all(v["id"] == f"{v['geometry']}_q{v['q']}" for v in variants), "Variant geometry/q binding failed")
    plan_sha = producer.sha(plan_path)
    producer_path = Path(producer.__file__).resolve()
    producer_sha = producer.sha(producer_path)
    pinned = [s for s in plan["source_files"] if Path(s["path"]).resolve() == producer_path]
    require(len(pinned) == 1 and pinned[0]["sha256"] == producer_sha, "Frozen producer SHA differs from plan")
    baseline_path = Path(plan["baseline_forward"])
    baseline = producer.read(baseline_path)
    baseline_sha = producer.sha(baseline_path)
    require(baseline.get("subject_alias") == plan["subject_alias"] and baseline.get("FEM_was_run") is True,
            "Baseline subject/completion binding failed")
    require(baseline.get("frequency_hz") == plan["frequency_hz"] == 100000, "Frequency binding failed")
    reference_path = Path(plan["reference_materials"])
    reference_sha = producer.sha(reference_path)
    pinned_ref = [s for s in plan["source_files"] if Path(s["path"]).resolve() == reference_path.resolve()]
    require(len(pinned_ref) == 1 and pinned_ref[0]["sha256"] == reference_sha, "Reference SHA differs from plan")
    require(baseline["identity"]["reference_sha256"] == reference_sha and
            baseline["identity"]["mesh_sha256"] == plan["mesh_sha256"], "Baseline reference/mesh binding failed")
    worker = ROOT / "MATLAB_TRKG4_real_subjects/src/run_scorpion_heart_pilot.m"
    worker_sha = producer.sha(worker)
    sources = [("adapter", Path(__file__).resolve()), ("producer", producer_path), ("worker", worker),
               ("plan", plan_path), ("baseline", baseline_path), ("reference", reference_path)]
    cases_by_variant = []
    for variant in variants:
        path = directory / ("solve_" + variant["id"]) / "report.local.json"
        if not path.exists():
            failed = path.parent / "failed.local.json"
            if failed.exists():
                sources.append(("failure_report", failed))
            continue
        raw = producer.read(path)
        report = normalize(raw)
        require(report.get("schema") == SCHEMA and report.get("action") == "solve" and
                report.get("FEM_was_run") is True, "Not a completed heart solve report")
        require(report["subject_alias"] == plan["subject_alias"] and report["frequency_hz"] == 100000,
                "Solve subject/frequency binding failed")
        require(report["variant"] == variant["id"], "Solve variant binding failed")
        identity = report["identity"]
        require(identity["plan_sha256"] == plan_sha, "Solve plan SHA binding failed")
        require(identity["baseline_sha256"] == baseline_sha, "Solve baseline SHA binding failed")
        require(identity["worker_sha256"] == worker_sha, "Solve worker SHA binding failed")
        require(identity["mesh_sha256"] == plan["mesh_sha256"] and identity["reference_sha256"] == reference_sha,
                "Solve mesh/reference binding failed")
        require(identity["cem_sha256"] == baseline["identity"]["cem_sha256"], "Solve CEM binding failed")
        rows = report["answers"]
        require(isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows), "Malformed answers")
        cases = [r["case_id"] for r in rows]
        require(len(cases) == len(set(cases)), "Duplicate solve cases")
        if plan.get("case_ids"):
            require(set(cases) == set(plan["case_ids"]), "Solve case coverage binding failed")
        cases_by_variant.append(set(cases))
        for row in rows:
            require(row["rule"] == variant["q"] and row["state"] == "inhale", "Solve state/q binding failed")
            for key in ("transfer_matrix_ohm", "delta_from_wholeheart_inhale_ohm", "baseline_wholeheart_transfer_ohm"):
                require_matrix(row[key], (4, 4), key)
            qc = row["system_checks"]
            require(qc["system_relative_residual"] < 1e-7 and qc["reciprocity_ohm"] < 1e-7,
                    "Solve residual/reciprocity failed")
        sources.append(("solve_report", path))
    require(not cases_by_variant or all(c == cases_by_variant[0] for c in cases_by_variant),
            "Variant case sets differ")
    fingerprint = [dict(kind=kind, path=str(path.resolve()), sha256=producer.sha(path)) for kind, path in sources]
    normalized = []
    with normalized_reader(normalized):
        result = producer.summarize(directory)
    for item in fingerprint:
        require(producer.sha(item["path"]) == item["sha256"], "Summary source changed during reading")
    result["reader_normalization"] = dict(schema=SCHEMA,
        policy="scalar answers object becomes a length-one list in memory; existing lists unchanged",
        normalized_report_paths=normalized, sources_modified=False)
    result["sources"] = fingerprint
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer.write(args.output, summarize(args.directory))


if __name__ == "__main__":
    main()
