from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

try:
    import h5py
except ModuleNotFoundError:
    h5py = None
import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import heart_cardiac_cycle_electrical as cycle
import run_electrode_sensitivity as electrode


def test_priority_and_montage_contract():
    states = [{"id": f"phase_{index:02d}"} for index in reversed(range(12))]
    ordered = [item["id"] for item in cycle._ordered_states({"states": states})]
    assert ordered[:3] == ["phase_00", "phase_03", "phase_09"]
    assert ordered[3:] == [
        "phase_11", "phase_10", "phase_08", "phase_07", "phase_06",
        "phase_05", "phase_04", "phase_02", "phase_01",
    ]
    specification = cycle.montage_spec()
    assert specification["electrodes"] == ["E1", "E2", "E3", "E4"]
    assert specification["channels"][0]["drive"] == [1, 0, 0, -1]
    assert specification["channels"][0]["measure"] == [0, 1, -1, 0]


def test_existing_load_contact_and_channel_rhs(tmp_path):
    contact_path = tmp_path / "contact.mat"
    prepared_sha256 = "a" * 64
    sio.savemat(contact_path, {
        "C": sp.eye(7, format="csr"),
        "ground": np.array([[1]]),
        "matrix_error": np.array([[0.0]]),
        "prepared_sha256": prepared_sha256,
        "electrode_ids": np.array(["E1", "E2", "E3", "E4"], dtype=object),
    })
    contact = electrode.load_contact(
        contact_path, 3, 4, 0, prepared_sha256, list(cycle.ELECTRODES)
    )
    rhs = electrode.channel_rhs(3, cycle.montage_spec())
    assert contact.shape == (7, 7)
    assert rhs.shape == (7, 2)
    np.testing.assert_array_equal(rhs[3:, 0], [1, 0, 0, -1])
    np.testing.assert_array_equal(rhs[3:, 1], [0, 1, -1, 0])


def _write_sparse_blocks(path: Path, bad: bool = False) -> None:
    with h5py.File(path, "w") as file:
        file.create_dataset("nn", data=np.array([[3.0]]))
        file.create_dataset("ground", data=np.array([[1.0]]))
        references = []
        for index in range(4):
            group = file.create_group(f"block_{index}")
            group.attrs["MATLAB_sparse"] = 7
            if bad and index == 0:
                data = np.array([1.0])
                ir = np.array([3], dtype=np.int64)
                jc = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
            else:
                data = np.ones(3)
                ir = np.array([0, 1, 2], dtype=np.int64)
                jc = np.array([0, 1, 2, 3, 3, 3, 3, 3], dtype=np.int64)
            group.create_dataset("data", data=data)
            group.create_dataset("ir", data=ir)
            group.create_dataset("jc", data=jc)
            references.append(group.ref)
        file.create_dataset("blocks", data=np.asarray(references, dtype=h5py.ref_dtype))


@pytest.mark.skipif(h5py is None, reason="h5py is unavailable")
def test_fractional_blocks_reject_electrode_terms(tmp_path):
    runner = SimpleNamespace(h5py=h5py, sp=sp, np=np)
    good = tmp_path / "blocks.mat"
    _write_sparse_blocks(good)
    blocks = cycle._load_fractional_blocks(runner, good, 3, 0)
    assert len(blocks) == 4
    assert all(block.shape == (7, 7) for block in blocks)
    bad = tmp_path / "bad_blocks.mat"
    _write_sparse_blocks(bad, bad=True)
    with pytest.raises(ValueError, match="electrode terms"):
        cycle._load_fractional_blocks(runner, bad, 3, 0)


def _prepare_fixture(tmp_path: Path):
    source = tmp_path / "prepared_fem.mat"
    with h5py.File(source, "w") as file:
        fmdl = file.create_group("fmdl_m")
        fmdl.create_dataset("nodes", data=np.zeros((3, 5)))
        fmdl.create_dataset("elems", data=np.ones((4, 2)))
        fmdl.create_dataset("gnd_node", data=np.array([[1.0]]))
    materials = tmp_path / "materials"
    materials.mkdir()
    background = materials / "background.mat"
    sio.savemat(background, {"background": np.array([[1, 2]])})
    states = []
    for index in range(12):
        fraction = materials / f"phase_{index:02d}_fraction.mat"
        sio.savemat(fraction, {"heart_fraction": np.array([[0.2, 0.3]])})
        states.append({
            "id": f"phase_{index:02d}",
            "fraction_mat": fraction.name,
            "fraction_mat_sha256": cycle.sha(fraction),
            "material_volume_ml": 10.0 + index,
            "mask_volume_ml": 11.0 + index,
            "rr_percent": float(index),
            "cycle_index": 0,
        })
    geometry_inputs = []
    for name in ("image.nii.gz", "config.json", "timing.json", "manifest.json"):
        item = tmp_path / name
        item.write_text(name, encoding="utf-8")
        geometry_inputs.append({"path": str(item), "sha256": cycle.sha(item)})
    implementation = tmp_path / "geometry_prep.py"
    implementation.write_text("def prepare(): pass\n", encoding="utf-8")
    geometry = {
        "source_prepared": str(source),
        "source_prepared_sha256": cycle.sha(source),
        "background_mat_sha256": cycle.sha(background),
        "node_count": 5,
        "element_count": 2,
        "inputs": geometry_inputs,
        "implementation": {"path": str(implementation), "sha256": cycle.sha(implementation)},
        "states": states,
    }
    (materials / "geometry.json").write_text(json.dumps(geometry), encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    contacts = []
    for montage in cycle.MONTAGES:
        contact = tmp_path / f"{montage}.mat"
        contact.write_bytes(montage.encode("ascii"))
        contacts.append({"id": montage, "path": str(contact), "sha256": cycle.sha(contact)})
    base_plan = {
        "schema": "heart_replacement_electrical_pilot_v1",
        "source_prepared": str(source),
        "source_prepared_sha256": cycle.sha(source),
        "node_count": 5, "element_count": 2, "ground_zero_based": 0,
        "threads": 1, "memory_limit_gib": 32.0,
        "runtime": {"deps": str(tmp_path / "deps"), "matlab": "matlab",
                    "eidors_startup": "startup.m", "python": "python"},
        "contacts": contacts, "inputs": [],
        "rho_ohm_m": {"soft_tissue": 1.0},
        "conductivity_by_tissue_id": [1.0, 2.0, 3.0, 4.0],
        "criteria": {}, "frequency_hz": 50000.0, "model": "test",
    }
    cycle.write(baseline / "plan.json", base_plan)
    return materials, baseline, base_plan, implementation


@pytest.mark.skipif(h5py is None, reason="h5py is unavailable")
def test_prepare_binds_geometry_provenance_and_old_schema(tmp_path, monkeypatch):
    materials, baseline, base_plan, implementation = _prepare_fixture(tmp_path)
    runner = SimpleNamespace(h5py=h5py, sio=sio, np=np)
    monkeypatch.setattr(cycle.core, "load_plan", lambda _: base_plan)
    monkeypatch.setattr(cycle, "configure", lambda *_: runner)
    output = tmp_path / "electrical"
    cycle.prepare(SimpleNamespace(materials=materials, baseline=baseline, output=output))
    plan = cycle.read(output / "plan.json")
    input_paths = {item["path"] for item in plan["inputs"]}
    assert all(item["path"] in input_paths for item in plan["materials_geometry"]["inputs"])
    assert any(item["path"] == str(implementation.resolve()) for item in plan["implementation"])
    assert plan["schema"] == cycle.SCHEMA
    assert plan["required_solutions"] == 72
    assert [item["id"] for item in plan["contacts"]] == list(cycle.MONTAGES)
    assert cycle.load_plan(output)["schema"] == cycle.SCHEMA


def test_run_refuses_incomplete_state_directory(tmp_path, monkeypatch):
    plan = {"states": [{"id": f"phase_{index:02d}"} for index in range(12)], "runtime": {"deps": "unused"}, "threads": 1}
    (tmp_path / "phase_00").mkdir()
    monkeypatch.setattr(cycle, "load_plan", lambda _: plan)
    monkeypatch.setattr(cycle, "configure", lambda *_: None)
    with pytest.raises(ValueError, match="Incomplete state export"):
        cycle.run(SimpleNamespace(output=tmp_path))
    assert not (tmp_path / "run.lock").exists()


def test_partial_summary_is_flat_and_bounded(tmp_path, monkeypatch):
    states = [{"id": f"phase_{index:02d}"} for index in reversed(range(12))]
    plan = {"states": states, "contacts": [{"id": x} for x in cycle.MONTAGES],
            "required_solutions": 72, "limitations": ["observed truncated region"]}
    available = {("phase_00", "tepc_2"), ("phase_03", "tepc_2")}
    def fake_completed(_output, _plan, state, montage):
        if (state["id"], montage) not in available:
            return None
        return {"state_id": state["id"], "montage": montage, "channel_id": cycle.CHANNEL_ID,
                "rr_percent": 0.0, "cycle_index": 0, "material_volume_ml": 1.0,
                "mask_volume_ml": 1.0, "Z_ohm": 1.0, "reciprocal_Z_ohm": 1.0,
                "max_relative_residual": 1e-8, "reciprocity_absolute_ohm": 1e-10,
                "max_current_error_A": 1e-10, "max_ground_residual_A": 1e-10,
                "solve_seconds": 0.1}
    monkeypatch.setattr(cycle, "_completed", fake_completed)
    assert cycle.summarize(tmp_path, plan) == 2
    summary = cycle.read(tmp_path / "summary.json")
    assert summary["status"] == "partial"
    assert summary["completed"] == 2 and summary["required"] == 72
    assert all("id" in row and "montage" in row for row in summary["states"])
    assert summary["numerical_qc"]["completed_rows"] == 2
    assert summary["phase_order"][:3] == ["phase_00", "phase_03", "phase_09"]
