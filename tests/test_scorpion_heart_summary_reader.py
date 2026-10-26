"""MATLAB singleton/list compatibility, immutable inputs, and branch bindings."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import scorpion_heart_pilot_summary as reader


def write(path, data):
    reader.producer.write(path, data)


def fixture(tmp_path, scalar):
    pilot = tmp_path / "pilot"; pilot.mkdir()
    producer = Path(reader.producer.__file__).resolve()
    reference = tmp_path / "reference.json"; write(reference, {"synthetic": True})
    baseline = tmp_path / "forward.json"
    write(baseline, dict(subject_alias="synthetic", frequency_hz=100000, FEM_was_run=True,
        identity=dict(reference_sha256=reader.producer.sha(reference), mesh_sha256="mesh", cem_sha256="cem")))
    variants = [dict(id=f"{g}_q{q}", geometry=g, q=q) for g in reader.GEOMETRIES for q in (1024, 4096)]
    volumes = dict(wholeheart_ct=100, sphere_v095=95, sphere_v100=100, sphere_v105=105)
    plan = dict(schema="scorpion_heart_pilot_plan_v1", subject_alias="synthetic", frequency_hz=100000,
        mesh_sha256="mesh", reference_materials=str(reference), baseline_forward=str(baseline),
        case_ids=["L140_TT2x2"], source_files=[dict(path=str(p), sha256=reader.producer.sha(p)) for p in (producer, reference)],
        variants=variants, geometry=dict(sphere=dict(volume_ml=100),
        rows=[dict(id=v["id"], volume_ml_by_material=[100, 100, 100, volumes[v["geometry"]], 100]) for v in variants]))
    write(pilot / "plan.local.json", plan)
    identity = dict(plan_sha256=reader.producer.sha(pilot / "plan.local.json"),
        baseline_sha256=reader.producer.sha(baseline),
        worker_sha256=reader.producer.sha(ROOT / "MATLAB_TRKG4_real_subjects/src/run_scorpion_heart_pilot.m"),
        mesh_sha256="mesh", reference_sha256=reader.producer.sha(reference), cem_sha256="cem")
    for v in variants:
        # Signed transfer entries ensure the adapter preserves numerical meaning.
        delta = np.eye(4) * (volumes[v["geometry"]] - 100)
        delta[1, 0] = (volumes[v["geometry"]] - 100) * -.02
        row = dict(case_id="L140_TT2x2", rule=v["q"], state="inhale",
            transfer_matrix_ohm=(np.eye(4)+delta).tolist(),
            delta_from_wholeheart_inhale_ohm=delta.tolist(), baseline_wholeheart_transfer_ohm=np.eye(4).tolist(),
            system_checks=dict(system_relative_residual=1e-12, reciprocity_ohm=1e-12))
        report = dict(schema=reader.SCHEMA, action="solve", subject_alias="synthetic", frequency_hz=100000,
            FEM_was_run=True, variant=v["id"], identity=identity, answers=row if scalar else [row])
        folder = pilot / ("solve_"+v["id"]); folder.mkdir()
        write(folder / "report.local.json", report)
    return pilot


def semantic(result):
    return {k: v for k, v in result.items() if k not in ("sources", "reader_normalization")}


def test_scalar_and_list_identical_semantics_and_no_mutations(tmp_path):
    one = tmp_path / "scalar"; one.mkdir()
    many = tmp_path / "list"; many.mkdir()
    p1, p2 = fixture(one, True), fixture(many, False)
    files = [p for folder in (one, many) for p in folder.rglob("*.json")]
    before = {p: p.read_bytes() for p in files}
    original_reader = reader.producer.read
    scalar, listed = reader.summarize(p1), reader.summarize(p2)
    assert reader.producer.read is original_reader
    assert semantic(scalar) == semantic(listed)
    assert len(scalar["reader_normalization"]["normalized_report_paths"]) == 8
    assert not listed["reader_normalization"]["normalized_report_paths"]
    assert scalar["reader_normalization"]["sources_modified"] is False
    assert {p: p.read_bytes() for p in files} == before
    assert scalar["status"] == "exploratory_forward_complete"
    assert len(scalar["comparisons"]) == 4 and len(scalar["volume_secants"]) == 2
    assert {s["kind"] for s in scalar["sources"]} >= {"adapter", "producer", "plan", "solve_report"}
    assert scalar["quantitative_result_accepted"] is False


@pytest.mark.parametrize("key,value,match", [
    ("plan_sha256", "wrong", "plan SHA"),
    ("reference_sha256", "blood_branch", "mesh/reference"),
    ("cem_sha256", "different_wiring", "CEM"),
    ("baseline_sha256", "different_baseline", "baseline SHA"),
])
def test_refuses_wrong_plan_or_material_branch(tmp_path, key, value, match):
    pilot = fixture(tmp_path, True)
    path = pilot / "solve_sphere_v100_q4096/report.local.json"
    report = reader.producer.read(path); report["identity"][key] = value
    path.write_text(__import__("json").dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        reader.summarize(pilot)


def test_normalization_does_not_touch_other_schemas():
    value = dict(schema="another_schema", answers={"a": 1})
    assert reader.normalize(value) is value
    value = dict(schema=reader.SCHEMA, answers=[{"a": 1}])
    assert reader.normalize(value) is value


def test_missing_variant_stays_incomplete(tmp_path):
    pilot = fixture(tmp_path, True)
    path = pilot / "solve_sphere_v105_q4096/report.local.json"
    # A synthetic fixture only: rename one receipt to represent an incomplete run.
    path.rename(path.with_suffix(".pending"))
    result = reader.summarize(pilot)
    assert result["status"] == "incomplete"
    assert result["incomplete_or_failed"] == [dict(variant="sphere_v105_q4096", reason="not run")]
