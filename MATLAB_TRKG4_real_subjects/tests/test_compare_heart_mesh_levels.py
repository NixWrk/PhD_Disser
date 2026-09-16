from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import compare_heart_mesh_levels as compare_tool


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _state(parameter: str, step: float, q: int, analytic: float = 100.0) -> dict:
    return {
        "id": f"{parameter}_{step}_{q}".replace(".", "p"),
        "geometry": "individual",
        "parameter": parameter,
        "signed_step": step,
        "volume_fraction": step if parameter == "volume_fraction" else 0.0,
        "translation_m": [0.0, 0.0, 0.0],
        "points_per_tet": q,
        "analytic_volume_ml": analytic if parameter == "baseline" else analytic * (1.0 + step),
        "material_volume_ml": analytic if parameter == "baseline" else analytic * (1.0 + step),
    }


def _plan(prepared: Path, contact: Path, original_hash: str, states: list[dict], *, deformation: str = "uniform") -> dict:
    return {
        "source_prepared": str(prepared),
        "source_prepared_sha256": _sha(prepared),
        "original_geometric_source_sha256": original_hash,
        "geometry": "individual",
        "parameter": "volume",
        "deformation": deformation,
        "frequency_hz": 50000,
        "rho_ohm_m": 0.05,
        "conductivity_by_tissue_id": [1.0, 2.0, 3.0, 4.0],
        "ground_zero_based": 7,
        "electrode_model": {
            "diameter_mm": 5,
            "area_mm2": 19.634954084936208,
            "z_contact_ohm_m2": 0.00015915494309189532,
        },
        "contact": {
            "id": "tepc_2",
            "path": str(contact),
            "sha256": _sha(contact),
            "areas_mm2": [20.0, 20.1, 19.8, 19.9],
            "actual_centres_xyz_mm": [[float(i), 1.0, 2.0] for i in range(4)],
        },
        "states": states,
        "inputs": [],
    }


def _make_stages(tmp_path: Path, *, fine_q: int = 4096):
    coarse_dir = tmp_path / "coarse_stage"
    fine_dir = tmp_path / "fine_stage"
    coarse_dir.mkdir()
    fine_dir.mkdir()
    original = coarse_dir / "original_prepared.mat"
    fine_prepared = fine_dir / "prepared_fem.mat"
    coarse_contact = coarse_dir / "contact.mat"
    fine_contact = fine_dir / "contact.mat"
    original.write_bytes(b"coarse prepared mesh")
    fine_prepared.write_bytes(b"fine prepared mesh")
    coarse_contact.write_bytes(b"coarse contact")
    fine_contact.write_bytes(b"fine contact with dimension-dependent bytes")
    original_hash = _sha(original)
    designs = [
        _state("baseline", 0.0, 512),
        _state("volume_fraction", -0.01, 512),
        _state("volume_fraction", 0.01, 512),
        _state("volume_fraction", -0.005, 512),
        _state("volume_fraction", 0.005, 512),
    ]
    coarse_plan = _plan(original, coarse_contact, original_hash, designs)
    fine_states = [dict(s, id=s["id"] + "_fine", points_per_tet=fine_q) for s in designs]
    fine_plan = _plan(fine_prepared, fine_contact, original_hash, fine_states)
    refinement = {
        "status": "passed_heart_refinement_export",
        "source_prepared_sha256": _sha(original),
        "original_contact_sha256": _sha(coarse_contact),
        "prepared_file": "prepared_fem.mat",
        "prepared_sha256": _sha(fine_prepared),
        "contact_file": "contact.mat",
        "contact_sha256": _sha(fine_contact),
        "original_nodes_preserved": True,
        "outer_boundary_preserved": True,
        "contact_faces_preserved": True,
        "contact_embedding_relative_error": 0.0,
        "matrix_relative_error": 0.0,
    }
    refinement_path = fine_dir / "heart_refinement_export.json"
    _write(refinement_path, refinement)
    fine_plan["inputs"] = [{"path": str(refinement_path), "sha256": _sha(refinement_path)}]
    _write(coarse_dir / "plan.json", coarse_plan)
    _write(fine_dir / "plan.json", fine_plan)
    # The production loader is replaced by the fixture loader below, so the
    # tiny plans need no sealed background/fraction files.
    return coarse_dir, fine_dir, coarse_plan, fine_plan


def _install_fake_stage(monkeypatch, plans, rows_by_dir):
    def fake_load_plan(directory):
        return plans[Path(directory).resolve()]

    def fake_summarize(directory, plan):
        rows = rows_by_dir[Path(directory).resolve()]
        return {
            "status": "complete",
            "completed": len(rows),
            "required": len(plan["states"]),
            "rows": list(rows.values()),
            "step_checks": [],
            "spatial_convergence": False,
        }

    def fake_checked_result(directory, plan, state):
        return rows_by_dir[Path(directory).resolve()].get(state["id"])

    monkeypatch.setattr(compare_tool.fractional, "load_plan", fake_load_plan)
    monkeypatch.setattr(compare_tool.individual, "summarize", fake_summarize)
    monkeypatch.setattr(compare_tool.fractional, "checked_result", fake_checked_result)


def _rows(plan, z_by_design):
    rows = {}
    for state in plan["states"]:
        key = compare_tool._design_key(compare_tool._state_design(state, plan))
        rows[state["id"]] = {
            "id": state["id"],
            "geometry": state["geometry"],
            "parameter": state["parameter"],
            "signed_step": state["signed_step"],
            "volume_fraction": state["volume_fraction"],
            "translation_m": state["translation_m"],
            "points_per_tet": state["points_per_tet"],
            "analytic_volume_ml": state["analytic_volume_ml"],
            "material_volume_ml": state["material_volume_ml"],
            "Z_ohm": z_by_design[key],
            "max_relative_residual": 1e-9,
            "reciprocity_absolute_ohm": 1e-10,
            "max_current_error_A": 1e-10,
            "max_ground_residual_A": 1e-10,
        }
    return rows


def test_compare_allows_mesh_and_contact_hash_changes_and_reports_derivatives(monkeypatch, tmp_path):
    coarse_dir, fine_dir, coarse, fine = _make_stages(tmp_path, fine_q=4096 * 4)
    # The fine electrical mesh and its contact file are intentionally different;
    # the physical contact metadata remains identical.
    coarse_values = {}
    fine_values = {}
    for state in coarse["states"]:
        key = compare_tool._design_key(compare_tool._state_design(state, coarse))
        step = compare_tool._state_design(state, coarse)["signed_step"]
        coarse_values[key] = 10.0 + 2.0 * step
        fine_values[key] = 10.0 + 3.0 * step
    plans = {coarse_dir.resolve(): coarse, fine_dir.resolve(): fine}
    rows = {
        coarse_dir.resolve(): _rows(coarse, coarse_values),
        fine_dir.resolve(): _rows(fine, fine_values),
    }
    _install_fake_stage(monkeypatch, plans, rows)

    report = compare_tool.compare(coarse_dir, fine_dir, tmp_path / "report")

    assert report["status"] == "complete"
    assert report["spatial_convergence"] is False
    assert report["quadrature"] == {"coarse_points_per_tet": [512], "fine_points_per_tet": [16384]}
    assert report["baseline_Z_shift"]["Z_fine_minus_coarse_ohm"] == 0.0
    assert report["coarse_plan_sha256"] == _sha(coarse_dir / "plan.json")
    assert report["fine_plan_sha256"] == _sha(fine_dir / "plan.json")
    assert report["implementation_sha256"] == _sha(Path(compare_tool.__file__))
    assert report["fine_refinement_provenance"]["hashes_passed"] is True
    differences = {(d["parameter"], d["step"]): d for d in report["derivative_comparison"]}
    assert differences[("volume_fraction", 0.005)]["derivative_coarse"] == pytest.approx(2.0 / 100.0)
    assert differences[("volume_fraction", 0.005)]["derivative_fine"] == pytest.approx(3.0 / 100.0)
    assert (tmp_path / "report" / "comparison.json").is_file()
    assert (tmp_path / "report" / "comparison.csv").is_file()
    text = (tmp_path / "report" / "comparison.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text


def test_reports_one_sided_slopes_and_asymmetry(monkeypatch, tmp_path):
    coarse_dir, fine_dir, coarse, fine = _make_stages(tmp_path)
    coarse_values = {}
    fine_values = {}
    for state in coarse["states"]:
        design = compare_tool._state_design(state, coarse)
        key = compare_tool._design_key(design)
        if design["parameter"] == "baseline":
            coarse_values[key] = fine_values[key] = 10.0
        elif design["signed_step"] == -0.005:
            coarse_values[key] = fine_values[key] = 9.8
        elif design["signed_step"] == 0.005:
            coarse_values[key] = fine_values[key] = 10.1
        else:
            coarse_values[key] = fine_values[key] = 10.0 + design["signed_step"]
    plans = {coarse_dir.resolve(): coarse, fine_dir.resolve(): fine}
    rows = {
        coarse_dir.resolve(): _rows(coarse, coarse_values),
        fine_dir.resolve(): _rows(fine, fine_values),
    }
    _install_fake_stage(monkeypatch, plans, rows)

    report = compare_tool.compare(coarse_dir, fine_dir, tmp_path / "report")

    slope = next(d for d in report["derivatives"]["coarse"] if d["step"] == 0.005)
    # dV = 0.005 * 100 ml = 0.5 ml: Sminus=0.4, Splus=0.2.
    assert slope["Sminus"] == pytest.approx(0.4)
    assert slope["Splus"] == pytest.approx(0.2)
    assert slope["one_sided_normalized_difference"] == pytest.approx(-2.0 / 3.0)
    assert slope["derivative"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda p: p.update(original_geometric_source_sha256="0" * 64), "Original geometric source"),
        (lambda p: p["contact"].update(areas_mm2=[21.0, 20.1, 19.8, 19.9]), "Physical contact geometry"),
        (lambda p: p.update(deformation="different motion law"), "deformation"),
    ],
)
def test_rejects_changed_anatomy_contact_or_motion_law(monkeypatch, tmp_path, mutation, message):
    coarse_dir, fine_dir, coarse, fine = _make_stages(tmp_path)
    mutation(fine)
    plans = {coarse_dir.resolve(): coarse, fine_dir.resolve(): fine}
    rows = {
        coarse_dir.resolve(): _rows(coarse, {compare_tool._design_key(compare_tool._state_design(s, coarse)): 10.0 for s in coarse["states"]}),
        fine_dir.resolve(): _rows(fine, {compare_tool._design_key(compare_tool._state_design(s, fine)): 10.0 for s in fine["states"]}),
    }
    _install_fake_stage(monkeypatch, plans, rows)
    with pytest.raises(ValueError, match=message):
        compare_tool.compare(coarse_dir, fine_dir, tmp_path / "report")


def test_uses_only_common_completed_states(monkeypatch, tmp_path):
    coarse_dir, fine_dir, coarse, fine = _make_stages(tmp_path)
    plans = {coarse_dir.resolve(): coarse, fine_dir.resolve(): fine}
    coarse_rows = _rows(coarse, {compare_tool._design_key(compare_tool._state_design(s, coarse)): 10.0 for s in coarse["states"]})
    fine_rows = _rows(fine, {compare_tool._design_key(compare_tool._state_design(s, fine)): 10.0 for s in fine["states"]})
    # The last positive state is still planned, but has no completion in fine.
    fine_rows.pop(fine["states"][-1]["id"])
    rows = {coarse_dir.resolve(): coarse_rows, fine_dir.resolve(): fine_rows}
    _install_fake_stage(monkeypatch, plans, rows)

    report = compare_tool.compare(coarse_dir, fine_dir, tmp_path / "report")

    assert report["status"] == "partial"
    assert report["state_designs"]["compared_completed"] == 4
    assert report["state_designs"]["fine_only_completed"] == []
    assert report["baseline_Z_shift"] is not None
